"""Tests for the MiRA-style SubgoalPlanner."""

import json
import os
import re
from unittest.mock import MagicMock, patch

import pytest

from heros.planner import (
    APIKeyMissingError,
    Milestone,
    PlanParsingError,
    PlanValidationError,
    PlannerError,
    SubgoalPlan,
    SubgoalPlanner,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_openai_response():
    """Return a mocked OpenAI chat completion response object."""

    def make_response(content: str):
        mock_choice = MagicMock()
        mock_choice.message.content = content

        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        return mock_resp

    return make_response


@pytest.fixture
def valid_milestones_response():
    """Return a valid JSON string that the LLM would return."""
    return json.dumps({
        "milestones": [
            {
                "id": "m1",
                "description": "Understand the requirements and gather context.",
                "rubric": "Requirements are identified and listed.",
                "expected_output": "A list of requirements.",
            },
            {
                "id": "m2",
                "description": "Design the solution architecture.",
                "rubric": "Architecture diagram or design doc is produced.",
                "expected_output": "Architecture document.",
            },
            {
                "id": "m3",
                "description": "Implement the core functionality.",
                "rubric": "Code compiles and basic tests pass.",
                "expected_output": "Working code in src/.",
            },
        ]
    })


@pytest.fixture
def minimal_milestones_response():
    """Return a minimal valid JSON (1 milestone)."""
    return json.dumps({
        "milestones": [
            {
                "id": "m1",
                "description": "Do the task.",
                "rubric": "Task is done.",
                "expected_output": "Result.",
            },
        ]
    })


@pytest.fixture
def max_milestones_response():
    """Return a 10-milestone JSON."""
    return json.dumps({
        "milestones": [
            {
                "id": f"m{i}",
                "description": f"Step {i}: Complete sub-task {i}.",
                "rubric": f"Sub-task {i} is verified.",
                "expected_output": f"Artifact {i}.",
            }
            for i in range(1, 11)
        ]
    })


# ---------------------------------------------------------------------------
# API Key handling
# ---------------------------------------------------------------------------

class TestAPIKeyMissing:
    """Test that missing API key raises a helpful error."""

    def test_raises_when_no_key_and_no_env(self):
        """Should raise APIKeyMissingError when no key is set anywhere."""
        # Ensure no key in env
        env = {"OPENAI_API_KEY": ""}
        with patch.dict(os.environ, env, clear=False):
            with pytest.raises(APIKeyMissingError) as exc_info:
                SubgoalPlanner()

        assert "OPENAI_API_KEY" in str(exc_info.value)
        assert "environment variable" in str(exc_info.value) or "api_key" in str(
            exc_info.value
        )

    def test_raises_when_explicit_none(self):
        """Should raise when api_key=None and env has no key."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False):
            with pytest.raises(APIKeyMissingError):
                SubgoalPlanner(api_key=None)

    def test_accepts_explicit_api_key(self, mock_openai_response):
        """Should not raise when api_key is passed directly."""
        with patch(
            "heros.planner.OpenAI"
        ) as mock_openai_cls, patch.dict(
            os.environ, {"OPENAI_API_KEY": ""}, clear=False
        ):
            mock_openai_cls.return_value.chat.completions.create = MagicMock(
                return_value=mock_openai_response('{"milestones": []}')
            )
            # Should not raise
            planner = SubgoalPlanner(api_key="sk-test-key-123")
            assert planner._client is not None


# ---------------------------------------------------------------------------
# Planning depth validation
# ---------------------------------------------------------------------------

class TestPlanningDepthValidation:
    """Test that planning_depth parameter is validated correctly."""

    @pytest.mark.parametrize("depth", [0, -1, 11, 20, 100])
    def test_rejects_out_of_range_depth(self, depth):
        with pytest.raises(ValueError) as exc_info:
            SubgoalPlanner(planning_depth=depth)
        assert "planning_depth" in str(exc_info.value)

    @pytest.mark.parametrize("depth", [1, 3, 5, 7, 10])
    def test_accepts_valid_depths(self, depth, mock_openai_response):
        with patch(
            "heros.planner.OpenAI"
        ) as mock_openai_cls, patch.dict(
            os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False
        ):
            mock_openai_cls.return_value.chat.completions.create = MagicMock(
                return_value=mock_openai_response('{"milestones": []}')
            )
            planner = SubgoalPlanner(
                planning_depth=depth, api_key="sk-test"
            )
            assert planner.planning_depth == depth

    def test_rejects_non_integer_depth(self):
        with pytest.raises(TypeError):
            SubgoalPlanner(planning_depth="5")  # type: ignore

        with pytest.raises(TypeError):
            SubgoalPlanner(planning_depth=5.0)  # type: ignore


# ---------------------------------------------------------------------------
# Basic plan creation
# ---------------------------------------------------------------------------

class TestPlanCreation:
    """Test basic LLM-driven plan creation."""

    def test_plan_returns_subgoal_plan(
        self,
        valid_milestones_response,
        mock_openai_response,
    ):
        with patch(
            "heros.planner.OpenAI"
        ) as mock_openai_cls, patch.dict(
            os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False
        ):
            mock_openai_cls.return_value.chat.completions.create = MagicMock(
                return_value=mock_openai_response(valid_milestones_response)
            )
            planner = SubgoalPlanner(planning_depth=5, api_key="sk-test")
            plan = planner.plan("Build a web scraper")

            assert isinstance(plan, SubgoalPlan)
            assert plan.task == "Build a web scraper"
            assert len(plan.milestones) == 3

    def test_plan_milestones_have_all_required_fields(
        self,
        valid_milestones_response,
        mock_openai_response,
    ):
        with patch(
            "heros.planner.OpenAI"
        ) as mock_openai_cls, patch.dict(
            os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False
        ):
            mock_openai_cls.return_value.chat.completions.create = MagicMock(
                return_value=mock_openai_response(valid_milestones_response)
            )
            planner = SubgoalPlanner(api_key="sk-test")
            plan = planner.plan("Build a web scraper")

            for m in plan.milestones:
                assert isinstance(m, Milestone)
                assert m.id  # non-empty
                assert m.description  # non-empty
                assert m.rubric  # non-empty
                # expected_output can be empty string
                assert isinstance(m.expected_output, str)

    def test_plan_includes_task_in_response(
        self,
        valid_milestones_response,
        mock_openai_response,
    ):
        with patch(
            "heros.planner.OpenAI"
        ) as mock_openai_cls, patch.dict(
            os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False
        ):
            mock_openai_cls.return_value.chat.completions.create = MagicMock(
                return_value=mock_openai_response(valid_milestones_response)
            )
            planner = SubgoalPlanner(api_key="sk-test")
            task = "Train a model on MNIST"
            plan = planner.plan(task)
            assert plan.task == task

    def test_to_dict_returns_serializable_structure(
        self,
        valid_milestones_response,
        mock_openai_response,
    ):
        with patch(
            "heros.planner.OpenAI"
        ) as mock_openai_cls, patch.dict(
            os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False
        ):
            mock_openai_cls.return_value.chat.completions.create = MagicMock(
                return_value=mock_openai_response(valid_milestones_response)
            )
            planner = SubgoalPlanner(api_key="sk-test")
            plan = planner.plan("Build a web scraper")
            d = plan.to_dict()

            assert isinstance(d, dict)
            assert d["task"] == "Build a web scraper"
            assert isinstance(d["milestones"], list)
            assert len(d["milestones"]) == 3

            # Should be JSON-serializable
            json.dumps(d)


# ---------------------------------------------------------------------------
# Planning depth parameter (1-10 milestones)
# ---------------------------------------------------------------------------

class TestPlanningDepth:
    """Test that planning_depth influences the number of milestones."""

    def test_depth_1_returns_minimal_plan(
        self,
        minimal_milestones_response,
        mock_openai_response,
    ):
        with patch(
            "heros.planner.OpenAI"
        ) as mock_openai_cls, patch.dict(
            os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False
        ):
            mock_openai_cls.return_value.chat.completions.create = MagicMock(
                return_value=mock_openai_response(minimal_milestones_response)
            )
            planner = SubgoalPlanner(planning_depth=1, api_key="sk-test")
            plan = planner.plan("Do something")
            assert len(plan.milestones) == 1

    def test_depth_10_returns_max_plan(
        self,
        max_milestones_response,
        mock_openai_response,
    ):
        with patch(
            "heros.planner.OpenAI"
        ) as mock_openai_cls, patch.dict(
            os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False
        ):
            mock_openai_cls.return_value.chat.completions.create = MagicMock(
                return_value=mock_openai_response(max_milestones_response)
            )
            planner = SubgoalPlanner(planning_depth=10, api_key="sk-test")
            plan = planner.plan("Do something complex")
            assert len(plan.milestones) == 10

    @pytest.mark.parametrize(
        "depth,num_milestones",
        [(3, 3), (5, 3), (7, 3)],
    )
    def test_various_depths_return_correct_count(
        self,
        depth,
        num_milestones,
        valid_milestones_response,
        mock_openai_response,
    ):
        """When the mocked LLM returns 3 milestones, we get 3 milestones."""
        with patch(
            "heros.planner.OpenAI"
        ) as mock_openai_cls, patch.dict(
            os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False
        ):
            mock_openai_cls.return_value.chat.completions.create = MagicMock(
                return_value=mock_openai_response(valid_milestones_response)
            )
            planner = SubgoalPlanner(planning_depth=depth, api_key="sk-test")
            plan = planner.plan("Do something")
            assert len(plan.milestones) == num_milestones


# ---------------------------------------------------------------------------
# Replanning after failure
# ---------------------------------------------------------------------------

class TestReplanning:
    """Test replan() after a milestone failure."""

    def test_replan_returns_new_plan(
        self,
        mock_openai_response,
    ):
        revised_response = json.dumps({
            "milestones": [
                {
                    "id": "m1",
                    "description": "Simplified step 1.",
                    "rubric": "Step 1 complete.",
                    "expected_output": "Output 1.",
                },
                {
                    "id": "m2",
                    "description": "Step 2 after failure.",
                    "rubric": "Step 2 complete.",
                    "expected_output": "Output 2.",
                },
            ]
        })

        with patch(
            "heros.planner.OpenAI"
        ) as mock_openai_cls, patch.dict(
            os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False
        ):
            mock_openai_cls.return_value.chat.completions.create = MagicMock(
                return_value=mock_openai_response(revised_response)
            )
            planner = SubgoalPlanner(api_key="sk-test")

            original_plan = SubgoalPlan(
                task="Build a web scraper",
                milestones=[
                    Milestone(
                        id="m1",
                        description="Fetch the page",
                        rubric="Page fetched successfully",
                        expected_output="HTML content",
                    ),
                    Milestone(
                        id="m2",
                        description="Parse the data",
                        rubric="Data extracted successfully",
                        expected_output="Structured data",
                    ),
                ],
            )

            new_plan = planner.replan(original_plan, "m1")

            assert isinstance(new_plan, SubgoalPlan)
            assert new_plan.task == "Build a web scraper"
            assert len(new_plan.milestones) == 2

    def test_replan_raises_for_unknown_milestone_id(
        self,
        mock_openai_response,
    ):
        with patch(
            "heros.planner.OpenAI"
        ) as mock_openai_cls, patch.dict(
            os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False
        ):
            mock_openai_cls.return_value.chat.completions.create = MagicMock(
                return_value=mock_openai_response('{"milestones": []}')
            )
            planner = SubgoalPlanner(api_key="sk-test")

            plan = SubgoalPlan(
                task="Build a web scraper",
                milestones=[
                    Milestone(id="m1", description="Step 1", rubric="Done"),
                ],
            )

            with pytest.raises(ValueError) as exc_info:
                planner.replan(plan, "m99")  # Does not exist

            assert "m99" in str(exc_info.value)
            assert "not found" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestParseValidation:
    """Test that parse/validation errors are raised correctly."""

    def test_raises_on_invalid_json_response(
        self,
        mock_openai_response,
    ):
        with patch(
            "heros.planner.OpenAI"
        ) as mock_openai_cls, patch.dict(
            os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False
        ):
            mock_openai_cls.return_value.chat.completions.create = MagicMock(
                return_value=mock_openai_response("This is not JSON at all!")
            )
            planner = SubgoalPlanner(api_key="sk-test")

            with pytest.raises(PlanParsingError) as exc_info:
                planner.plan("Do something")
            assert "Could not parse" in str(exc_info.value)

    def test_raises_when_milestones_missing(
        self,
        mock_openai_response,
    ):
        with patch(
            "heros.planner.OpenAI"
        ) as mock_openai_cls, patch.dict(
            os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False
        ):
            mock_openai_cls.return_value.chat.completions.create = MagicMock(
                return_value=mock_openai_response('{"other_key": "value"}')
            )
            planner = SubgoalPlanner(api_key="sk-test")

            with pytest.raises(PlanValidationError) as exc_info:
                planner.plan("Do something")
            assert "milestones" in str(exc_info.value)

    def test_raises_when_milestone_missing_id(
        self,
        mock_openai_response,
    ):
        with patch(
            "heros.planner.OpenAI"
        ) as mock_openai_cls, patch.dict(
            os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False
        ):
            mock_openai_cls.return_value.chat.completions.create = MagicMock(
                return_value=mock_openai_response(
                    json.dumps({
                        "milestones": [
                            {
                                "description": "No id field",
                                "rubric": "OK",
                            }
                        ]
                    })
                )
            )
            planner = SubgoalPlanner(api_key="sk-test")

            with pytest.raises(PlanValidationError) as exc_info:
                planner.plan("Do something")
            assert "id" in str(exc_info.value)

    def test_raises_when_milestone_missing_rubric(
        self,
        mock_openai_response,
    ):
        with patch(
            "heros.planner.OpenAI"
        ) as mock_openai_cls, patch.dict(
            os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False
        ):
            mock_openai_cls.return_value.chat.completions.create = MagicMock(
                return_value=mock_openai_response(
                    json.dumps({
                        "milestones": [
                            {
                                "id": "m1",
                                "description": "Has no rubric",
                            }
                        ]
                    })
                )
            )
            planner = SubgoalPlanner(api_key="sk-test")

            with pytest.raises(PlanValidationError) as exc_info:
                planner.plan("Do something")
            assert "rubric" in str(exc_info.value)

    def test_raises_when_milestones_list_empty(
        self,
        mock_openai_response,
    ):
        with patch(
            "heros.planner.OpenAI"
        ) as mock_openai_cls, patch.dict(
            os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False
        ):
            mock_openai_cls.return_value.chat.completions.create = MagicMock(
                return_value=mock_openai_response('{"milestones": []}')
            )
            planner = SubgoalPlanner(api_key="sk-test")

            with pytest.raises(PlanValidationError) as exc_info:
                planner.plan("Do something")
            assert "empty" in str(exc_info.value)

    def test_raises_on_llm_api_error(
        self,
        mock_openai_response,
    ):
        with patch(
            "heros.planner.OpenAI"
        ) as mock_openai_cls, patch.dict(
            os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False
        ):
            mock_openai_cls.return_value.chat.completions.create = MagicMock(
                side_effect=Exception("Rate limit exceeded")
            )
            planner = SubgoalPlanner(api_key="sk-test")

            with pytest.raises(PlannerError) as exc_info:
                planner.plan("Do something")
            assert "Rate limit exceeded" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Test edge cases: very short/long tasks, JSON fences, etc."""

    def test_very_short_task(
        self,
        valid_milestones_response,
        mock_openai_response,
    ):
        with patch(
            "heros.planner.OpenAI"
        ) as mock_openai_cls, patch.dict(
            os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False
        ):
            mock_openai_cls.return_value.chat.completions.create = MagicMock(
                return_value=mock_openai_response(valid_milestones_response)
            )
            planner = SubgoalPlanner(api_key="sk-test")
            plan = planner.plan("Go")

            assert isinstance(plan, SubgoalPlan)
            assert len(plan.milestones) > 0

    def test_very_long_task(
        self,
        valid_milestones_response,
        mock_openai_response,
    ):
        with patch(
            "heros.planner.OpenAI"
        ) as mock_openai_cls, patch.dict(
            os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False
        ):
            mock_openai_cls.return_value.chat.completions.create = MagicMock(
                return_value=mock_openai_response(valid_milestones_response)
            )
            planner = SubgoalPlanner(api_key="sk-test")
            long_task = "Build a complete " + "X" * 5000 + " system"
            plan = planner.plan(long_task)

            assert isinstance(plan, SubgoalPlan)
            assert plan.task == long_task

    def test_json_fence_in_response(
        self,
        valid_milestones_response,
        mock_openai_response,
    ):
        """LLM sometimes wraps JSON in ```json ... ``` fences."""
        fenced = f"```json\n{valid_milestones_response}\n```"

        with patch(
            "heros.planner.OpenAI"
        ) as mock_openai_cls, patch.dict(
            os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False
        ):
            mock_openai_cls.return_value.chat.completions.create = MagicMock(
                return_value=mock_openai_response(fenced)
            )
            planner = SubgoalPlanner(api_key="sk-test")
            plan = planner.plan("Build a web scraper")

            assert len(plan.milestones) == 3

    def test_plain_json_without_fence(
        self,
        valid_milestones_response,
        mock_openai_response,
    ):
        with patch(
            "heros.planner.OpenAI"
        ) as mock_openai_cls, patch.dict(
            os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False
        ):
            mock_openai_cls.return_value.chat.completions.create = MagicMock(
                return_value=mock_openai_response(valid_milestones_response)
            )
            planner = SubgoalPlanner(api_key="sk-test")
            plan = planner.plan("Build a web scraper")

            assert len(plan.milestones) == 3

    def test_json_embedded_in_text(
        self,
        valid_milestones_response,
        mock_openai_response,
    ):
        """JSON buried inside regular text should still be parsed."""
        preamble = "Here is the plan I recommend:\n"
        postamble = "\n\nLet me know if you'd like me to elaborate."

        with patch(
            "heros.planner.OpenAI"
        ) as mock_openai_cls, patch.dict(
            os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False
        ):
            mock_openai_cls.return_value.chat.completions.create = MagicMock(
                return_value=mock_openai_response(
                    preamble + valid_milestones_response + postamble
                )
            )
            planner = SubgoalPlanner(api_key="sk-test")
            plan = planner.plan("Build a web scraper")

            assert len(plan.milestones) == 3

    def test_milestone_expected_output_optional(
        self,
        mock_openai_response,
    ):
        """Milestones without expected_output should default to empty string."""
        response = json.dumps({
            "milestones": [
                {
                    "id": "m1",
                    "description": "Do it",
                    "rubric": "Done",
                    # no expected_output
                },
            ]
        })

        with patch(
            "heros.planner.OpenAI"
        ) as mock_openai_cls, patch.dict(
            os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False
        ):
            mock_openai_cls.return_value.chat.completions.create = MagicMock(
                return_value=mock_openai_response(response)
            )
            planner = SubgoalPlanner(api_key="sk-test")
            plan = planner.plan("Do it")

            assert plan.milestones[0].expected_output == ""


# ---------------------------------------------------------------------------
# Integration-like tests (real API, skipped without key)
# ---------------------------------------------------------------------------

class TestRealAPIClient:
    """Integration tests that call the real API when key is available.

    These tests are only run when OPENAI_API_KEY is set.
    They are marked with pytest.mark.skipif so they won't fail in CI
    without a key.
    """

    @pytest.mark.skipif(
        os.environ.get("OPENAI_API_KEY") in (None, ""),
        reason="OPENAI_API_KEY not set",
    )
    def test_real_plan_call_returns_valid_plan(self):
        """End-to-end test with the real OpenAI API."""
        planner = SubgoalPlanner(planning_depth=5)
        plan = planner.plan("Write a simple HTTP server in Python")

        assert isinstance(plan, SubgoalPlan)
        assert plan.task == "Write a simple HTTP server in Python"
        assert 1 <= len(plan.milestones) <= 10

        for m in plan.milestones:
            assert m.id
            assert m.description
            assert m.rubric

    @pytest.mark.skipif(
        os.environ.get("OPENAI_API_KEY") in (None, ""),
        reason="OPENAI_API_KEY not set",
    )
    def test_real_replan_call(self):
        """End-to-end test for replan with real API."""
        planner = SubgoalPlanner(planning_depth=5)
        original = planner.plan("Write a simple HTTP server in Python")

        # Replan with the first milestone marked as failed
        revised = planner.replan(original, original.milestones[0].id)

        assert isinstance(revised, SubgoalPlan)
        assert revised.task == original.task
        assert len(revised.milestones) >= 1
