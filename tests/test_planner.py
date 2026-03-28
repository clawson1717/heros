"""
Tests for the HeRoS Subgoal Planner (Step 3).

Covers: Milestone dataclass, SubgoalPlanner initialization and planning,
rule-based decomposition, LLM client integration (mock), depth constraints,
ordered output, and error handling.
"""

import pytest
from heros.planner import (
    Milestone,
    SubgoalPlanner,
    LLMPlanner,
    detect_task_type,
    _build_rule_milestones,
    _RULE_BASED_TEMPLATES,
)


# ---------------------------------------------------------------------------
# Milestone Dataclass Tests
# ---------------------------------------------------------------------------

class TestMilestoneDataclass:
    def test_create_milestone(self):
        m = Milestone(
            id="m1",
            description="Write the function",
            rubric="Function exists with correct signature",
            expected_output="sort.py with sort function",
            order=0,
        )
        assert m.id == "m1"
        assert m.description == "Write the function"
        assert m.rubric == "Function exists with correct signature"
        assert m.expected_output == "sort.py with sort function"
        assert m.order == 0

    def test_to_dict(self):
        m = Milestone(id="m2", description="Test it", rubric="Tests pass", expected_output="test output", order=1)
        d = m.to_dict()
        assert d["id"] == "m2"
        assert d["description"] == "Test it"
        assert d["rubric"] == "Tests pass"
        assert d["expected_output"] == "test output"
        assert d["order"] == 1

    def test_from_dict(self):
        data = {"id": "m3", "description": "Deploy", "rubric": "Deployed", "expected_output": "URL", "order": 2}
        m = Milestone.from_dict(data)
        assert m.id == "m3"
        assert m.description == "Deploy"
        assert m.rubric == "Deployed"
        assert m.expected_output == "URL"
        assert m.order == 2

    def test_roundtrip(self):
        original = Milestone(id="m1", description="x", rubric="y", expected_output="z", order=0)
        restored = Milestone.from_dict(original.to_dict())
        assert restored.id == original.id
        assert restored.description == original.description
        assert restored.rubric == original.rubric
        assert restored.expected_output == original.expected_output
        assert restored.order == original.order


# ---------------------------------------------------------------------------
# SubgoalPlanner Initialization Tests
# ---------------------------------------------------------------------------

class TestSubgoalPlannerInit:
    def test_default_init(self):
        planner = SubgoalPlanner()
        assert planner.max_subgoals == 10
        assert planner.min_subgoals == 1

    def test_custom_depth(self):
        planner = SubgoalPlanner(min_subgoals=3, max_subgoals=7)
        assert planner.min_subgoals == 3
        assert planner.max_subgoals == 7

    def test_max_subgoals_out_of_range_raises(self):
        with pytest.raises(ValueError, match="max_subgoals must be between"):
            SubgoalPlanner(max_subgoals=0)
        with pytest.raises(ValueError, match="max_subgoals must be between"):
            SubgoalPlanner(max_subgoals=11)

    def test_min_subgoals_below_one_raises(self):
        with pytest.raises(ValueError, match="min_subgoals must be at least"):
            SubgoalPlanner(min_subgoals=0)

    def test_min_exceeds_max_raises(self):
        with pytest.raises(ValueError, match="min_subgoals must not exceed"):
            SubgoalPlanner(min_subgoals=5, max_subgoals=3)


# ---------------------------------------------------------------------------
# Rule-Based Planning Tests
# ---------------------------------------------------------------------------

class TestRuleBasedPlanning:
    def _check_milestones(self, milestones, task, max_sg, min_sg):
        assert isinstance(milestones, list)
        assert len(milestones) >= min_sg
        assert len(milestones) <= max_sg
        for m in milestones:
            assert isinstance(m, Milestone)
            assert m.id.startswith("m")
            assert m.description
            assert m.rubric
            assert m.expected_output is not None
            assert m.order >= 0
        # Check ordering
        orders = [m.order for m in milestones]
        assert orders == sorted(orders)
        # Check IDs are sequential
        for i, m in enumerate(milestones):
            assert m.id == f"m{i + 1}"

    def test_code_generation_task(self):
        planner = SubgoalPlanner(min_subgoals=1, max_subgoals=10)
        milestones = planner.plan("Write a Python function that sorts a list using quicksort")
        self._check_milestones(milestones, "code_generation", 10, 1)
        task_type = detect_task_type("Write a Python function that sorts a list using quicksort")
        assert task_type == "code_generation"

    def test_web_navigation_task(self):
        planner = SubgoalPlanner(min_subgoals=1, max_subgoals=10)
        milestones = planner.plan("Navigate to example.com, click the login button, and fill out the form")
        self._check_milestones(milestones, "web_navigation", 10, 1)
        task_type = detect_task_type("click the login button and fill out the form")
        assert task_type == "web_navigation"

    def test_data_analysis_task(self):
        planner = SubgoalPlanner(min_subgoals=1, max_subgoals=10)
        milestones = planner.plan("Analyze this CSV dataset and produce a visualization")
        self._check_milestones(milestones, "data_analysis", 10, 1)
        task_type = detect_task_type("Analyze this CSV dataset")
        assert task_type == "data_analysis"

    def test_reasoning_task(self):
        planner = SubgoalPlanner(min_subgoals=1, max_subgoals=10)
        milestones = planner.plan("Prove that the sum of two even numbers is even")
        self._check_milestones(milestones, "reasoning", 10, 1)
        task_type = detect_task_type("prove that the sum of two even numbers is even")
        assert task_type == "reasoning"

    def test_general_task(self):
        planner = SubgoalPlanner(min_subgoals=1, max_subgoals=10)
        milestones = planner.plan("Do something interesting with AI")
        self._check_milestones(milestones, "general", 10, 1)
        task_type = detect_task_type("Do something interesting with AI")
        assert task_type == "general"

    def test_max_subgoals_enforced(self):
        planner = SubgoalPlanner(min_subgoals=1, max_subgoals=3)
        milestones = planner.plan("Write a Python function that sorts a list using quicksort")
        assert len(milestones) <= 3

    def test_min_subgoals_enforced(self):
        planner = SubgoalPlanner(min_subgoals=5, max_subgoals=10)
        milestones = planner.plan("Analyze this dataset")
        assert len(milestones) >= 5


# ---------------------------------------------------------------------------
# LLM Client Integration Tests (Mock)
# ---------------------------------------------------------------------------

class MockLLMClient:
    def __init__(self, response: str):
        self.calls = []
        self._response = response

    def complete(self, prompt: str, **kwargs) -> str:
        self.calls.append(prompt)
        return self._response


class TestLLMPlanning:
    def test_llm_client_is_called(self):
        mock = MockLLMClient("""[
            {"id": "m1", "description": "Write code", "rubric": "Code exists", "expected_output": "file.py", "order": 0}
        ]""")
        planner = SubgoalPlanner(llm_client=mock, min_subgoals=1, max_subgoals=5)
        milestones = planner.plan("Write a hello world script")
        assert len(mock.calls) >= 1
        assert "subgoal" in mock.calls[0].lower() or "milestone" in mock.calls[0].lower()

    def test_llm_fallback_on_error(self):
        # LLM that raises an error
        def bad_complete(prompt: str, **kwargs) -> str:
            raise RuntimeError("LLM unavailable")

        planner = SubgoalPlanner(llm_client=bad_complete, min_subgoals=1, max_subgoals=10)
        # Should fall back to rule-based, not raise
        milestones = planner.plan("Write a Python function")
        assert len(milestones) >= 1

    def test_llm_result_parsed_and_sorted(self):
        mock = MockLLMClient("""[
            {"id": "m3", "description": "Third", "rubric": "OK", "expected_output": "out", "order": 2},
            {"id": "m1", "description": "First", "rubric": "OK", "expected_output": "out", "order": 0},
            {"id": "m2", "description": "Second", "rubric": "OK", "expected_output": "out", "order": 1}
        ]""")
        planner = SubgoalPlanner(llm_client=mock, min_subgoals=1, max_subgoals=10)
        milestones = planner.plan("Test task")
        # Should be sorted by order
        assert milestones[0].id == "m1"
        assert milestones[1].id == "m2"
        assert milestones[2].id == "m3"


class TestLLMPlannerSubclass:
    def test_llm_client_none_falls_back_to_rule_based(self):
        # LLMPlanner with None behaves like SubgoalPlanner (no-op LLM)
        planner = LLMPlanner(llm_client=None, min_subgoals=1, max_subgoals=5)  # type: ignore[arg-type]
        milestones = planner.plan("Write a function")
        assert len(milestones) >= 1  # falls back to rule-based

    def test_uses_llm_client(self):
        mock = MockLLMClient("""[{"id": "m1", "description": "x", "rubric": "y", "expected_output": "z", "order": 0}]""")
        planner = LLMPlanner(llm_client=mock, min_subgoals=1, max_subgoals=5)
        milestones = planner.plan("test")
        assert len(milestones) == 1


# ---------------------------------------------------------------------------
# plan_from_template Tests
# ---------------------------------------------------------------------------

class TestPlanFromTemplate:
    def test_template_used(self):
        template = [
            {"description": "Step 1", "rubric": "R1", "expected_output": "O1"},
            {"description": "Step 2", "rubric": "R2", "expected_output": "O2"},
        ]
        planner = SubgoalPlanner(min_subgoals=1, max_subgoals=10)
        milestones = planner.plan_from_template("irrelevant task", template)
        assert len(milestones) == 2
        assert milestones[0].description == "Step 1"
        assert milestones[1].description == "Step 2"

    def test_empty_template_falls_back(self):
        planner = SubgoalPlanner(min_subgoals=1, max_subgoals=10)
        milestones = planner.plan_from_template("Analyze data", [])
        assert len(milestones) >= 1

    def test_max_subgoals_limits_template(self):
        template = [{"description": f"Step {i}", "rubric": f"R{i}", "expected_output": f"O{i}"} for i in range(20)]
        planner = SubgoalPlanner(min_subgoals=1, max_subgoals=5)
        milestones = planner.plan_from_template("irrelevant", template)
        assert len(milestones) == 5

    def test_padding_to_min_subgoals(self):
        template = [{"description": "Only one", "rubric": "OK", "expected_output": "out"}]
        planner = SubgoalPlanner(min_subgoals=5, max_subgoals=10)
        milestones = planner.plan_from_template("irrelevant", template)
        assert len(milestones) >= 5


# ---------------------------------------------------------------------------
# detect_task_type Tests
# ---------------------------------------------------------------------------

class TestDetectTaskType:
    def test_code_generation(self):
        assert detect_task_type("Write a Python function") == "code_generation"
        assert detect_task_type("Implement an API endpoint") == "code_generation"
        assert detect_task_type("debug this script") == "code_generation"

    def test_web_navigation(self):
        assert detect_task_type("click the button and navigate") == "web_navigation"
        assert detect_task_type("login to the website") == "web_navigation"

    def test_data_analysis(self):
        assert detect_task_type("Analyze the dataset") == "data_analysis"
        assert detect_task_type("Plot the CSV data") == "data_analysis"

    def test_reasoning(self):
        assert detect_task_type("Prove the theorem") == "reasoning"
        assert detect_task_type("Logical deduction") == "reasoning"

    def test_unknown_falls_back_to_general(self):
        assert detect_task_type("do something random xyz123") == "general"


# ---------------------------------------------------------------------------
# _build_rule_milestones Tests
# ---------------------------------------------------------------------------

class TestBuildRuleMilestones:
    def test_returns_ordered_list(self):
        milestones = _build_rule_milestones("Analyze a CSV", min_subgoals=1, max_subgoals=10)
        assert milestones == sorted(milestones, key=lambda m: m.order)

    def test_ids_are_sequential(self):
        milestones = _build_rule_milestones("Write code", min_subgoals=1, max_subgoals=10)
        for i, m in enumerate(milestones):
            assert m.id == f"m{i + 1}"

    def test_respects_max(self):
        milestones = _build_rule_milestones("Write code", min_subgoals=1, max_subgoals=3)
        assert len(milestones) <= 3

    def test_respects_min(self):
        milestones = _build_rule_milestones("Simple task", min_subgoals=5, max_subgoals=10)
        assert len(milestones) >= 5
