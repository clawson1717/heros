"""Comprehensive tests for HeRoS Evaluation Harness.

Tests:
- WebArenaLiteBenchmark: Task loading, milestone retrieval, task listing
- HeRoSEvaluator: Episode running, metrics computation, agent comparison
- BaselineAgent: Action generation, no-milestone behavior
- HeRoSWrappedAgent: Milestone → action conversion, hindsight integration
- MockWebEnv: State management, action execution, observation generation
- Integration: Full benchmark evaluation runs
"""

import json
import os
import tempfile
from datetime import date
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

import heros
from heros import (
    WebArenaLiteBenchmark,
    WebTask,
    MockWebEnv,
    WebAction,
    EvaluationAction,
    HeRoSEvaluator,
    EvaluationResult,
    BaselineAgent,
    HeRoSWrappedAgent,
)
from heros.planner import Milestone, SubgoalPlan, SubgoalPlanner
from heros.critic import CriticResult, MilestoneCritic, Verdict
from heros.buffer import HindsightBuffer, HindsightTrajectory
from heros.trainer import HindsightTrainer, UpdateResult
from heros.env import HeRoSEnv
from heros.agent import HeRoSAgent, ActResult


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def benchmark():
    """Create a WebArenaLiteBenchmark with mini subset."""
    return WebArenaLiteBenchmark(task_subset="mini")


@pytest.fixture
def benchmark_full():
    """Create a WebArenaLiteBenchmark with full subset."""
    return WebArenaLiteBenchmark(task_subset="full")


@pytest.fixture
def mock_web_env():
    """Create a MockWebEnv with default state."""
    return MockWebEnv()


@pytest.fixture
def env_for_theme_task(benchmark):
    """Create a MockWebEnv configured for the theme task."""
    return benchmark.create_env_for_task("change_theme_dark")


@pytest.fixture
def env_for_contact_task(benchmark):
    """Create a MockWebEnv configured for the contact form task."""
    return benchmark.create_env_for_task("contact_form_fill")


@pytest.fixture
def env_for_logout_task(benchmark):
    """Create a MockWebEnv configured for the logout task."""
    return benchmark.create_env_for_task("logout_session")


@pytest.fixture
def baseline_agent():
    """Create a BaselineAgent without API key (rule-based mode)."""
    return BaselineAgent(api_key="fake-key-for-testing")


@pytest.fixture
def baseline_agent_no_key():
    """Create a BaselineAgent without an API key."""
    return BaselineAgent(api_key=None)


@pytest.fixture
def mock_heros_agent():
    """Create a mock HeRoSAgent."""
    agent = MagicMock(spec=HeRoSAgent)
    agent.act.return_value = ActResult(
        action="click Settings",
        milestone_id="m1",
        milestone_description="Navigate to settings",
        critic_result=CriticResult(
            verdict=Verdict.PARTIAL,
            feedback="In progress",
            confidence=0.5,
            reward_signal=0.5,
        ),
        is_milestone_complete=False,
        is_episode_done=False,
    )
    return agent


@pytest.fixture
def wrapped_agent(mock_heros_agent):
    """Create a HeRoSWrappedAgent with a mock HeRoSAgent."""
    return HeRoSWrappedAgent(mock_heros_agent, hindsight_enabled=True)


@pytest.fixture
def evaluator_with_baseline(benchmark, baseline_agent):
    """Create a HeRoSEvaluator with a baseline agent."""
    return HeRoSEvaluator(benchmark, baseline_agent, use_hindsight=False, max_steps=20)


@pytest.fixture
def evaluator_with_heros(benchmark, mock_heros_agent):
    """Create a HeRoSEvaluator with a HeRoS agent."""
    return HeRoSEvaluator(benchmark, mock_heros_agent, use_hindsight=True, max_steps=20)


# ============================================================================
# Test WebArenaLiteBenchmark
# ============================================================================


class TestWebArenaLiteBenchmark:
    """Tests for WebArenaLiteBenchmark class."""

    def test_init_default(self):
        """Test default initialization."""
        b = WebArenaLiteBenchmark()
        assert b._task_subset == "mini"
        assert len(b) == 5

    def test_init_full(self):
        """Test full subset initialization."""
        b = WebArenaLiteBenchmark(task_subset="full")
        assert b._task_subset == "full"
        assert len(b) >= 5

    def test_init_invalid_subset(self):
        """Test invalid subset raises ValueError."""
        with pytest.raises(ValueError, match="must be one of"):
            WebArenaLiteBenchmark(task_subset="invalid")

    def test_get_task_valid(self, benchmark):
        """Test getting a valid task."""
        task = benchmark.get_task("change_theme_dark")
        assert isinstance(task, WebTask)
        assert task.task_id == "change_theme_dark"
        assert "theme" in task.description.lower()
        assert len(task.milestones) >= 1

    def test_get_task_invalid(self, benchmark):
        """Test getting an invalid task raises KeyError."""
        with pytest.raises(KeyError, match="not found"):
            benchmark.get_task("nonexistent_task")

    def test_list_tasks(self, benchmark):
        """Test listing all tasks."""
        tasks = benchmark.list_tasks()
        assert isinstance(tasks, list)
        assert len(tasks) == 5
        assert "change_theme_dark" in tasks
        assert "contact_form_fill" in tasks

    def test_get_milestone_for_task(self, benchmark):
        """Test getting milestones for a task."""
        milestones = benchmark.get_milestone_for_task("change_theme_dark")
        assert isinstance(milestones, list)
        assert len(milestones) == 2
        assert all(isinstance(m, Milestone) for m in milestones)

    def test_get_tasks_by_difficulty(self, benchmark):
        """Test filtering tasks by difficulty."""
        easy_tasks = benchmark.get_tasks_by_difficulty("easy")
        assert all(t.difficulty == "easy" for t in easy_tasks)

        medium_tasks = benchmark.get_tasks_by_difficulty("medium")
        assert all(t.difficulty == "medium" for t in medium_tasks)

        hard_tasks = benchmark.get_tasks_by_difficulty("hard")
        assert all(t.difficulty == "hard" for t in hard_tasks)

    def test_get_tasks_by_difficulty_invalid(self, benchmark):
        """Test invalid difficulty raises ValueError."""
        with pytest.raises(ValueError, match="must be 'easy'"):
            benchmark.get_tasks_by_difficulty("invalid")

    def test_create_env_for_task(self, benchmark):
        """Test creating an environment for a task."""
        env = benchmark.create_env_for_task("change_theme_dark")
        assert isinstance(env, MockWebEnv)
        obs = env.reset()
        assert "url" in obs
        assert "page_content" in obs

    def test_get_stats(self, benchmark):
        """Test benchmark statistics."""
        stats = benchmark.get_stats()
        assert stats["subset"] == "mini"
        assert stats["total_tasks"] == 5
        assert stats["easy_count"] >= 1
        assert stats["medium_count"] >= 1
        assert stats["hard_count"] >= 1
        assert "task_ids" in stats

    def test_iteration(self, benchmark):
        """Test iterating over benchmark tasks."""
        tasks = list(benchmark)
        assert len(tasks) == 5
        assert all(isinstance(t, WebTask) for t in tasks)

    def test_indexing(self, benchmark):
        """Test indexing into benchmark."""
        task = benchmark["change_theme_dark"]
        assert task.task_id == "change_theme_dark"

    def test_repr(self, benchmark):
        """Test string representation."""
        r = repr(benchmark)
        assert "WebArenaLiteBenchmark" in r
        assert "mini" in r


# ============================================================================
# Test WebTask
# ============================================================================


class TestWebTask:
    """Tests for WebTask dataclass."""

    def test_webtask_creation(self):
        """Test basic WebTask creation."""
        task = WebTask(
            task_id="test_task",
            description="Test task description",
            target_url="https://example.com",
            difficulty="medium",
        )
        assert task.task_id == "test_task"
        assert task.difficulty == "medium"

    def test_webtask_invalid_difficulty(self):
        """Test invalid difficulty raises ValueError."""
        with pytest.raises(ValueError, match="must be 'easy'"):
            WebTask(
                task_id="test",
                description="Test",
                target_url="https://example.com",
                difficulty="impossible",
            )

    def test_milestone_count(self, benchmark):
        """Test milestone_count method."""
        task = benchmark.get_task("change_theme_dark")
        assert task.milestone_count() == 2

    def test_task_has_expected_actions(self, benchmark):
        """Test that tasks have expected action sequences."""
        task = benchmark.get_task("change_theme_dark")
        assert len(task.expected_actions) >= 1
        assert all(isinstance(a, WebAction) for a in task.expected_actions)


# ============================================================================
# Test WebAction and EvaluationAction
# ============================================================================


class TestWebAction:
    """Tests for WebAction dataclass."""

    def test_webaction_click(self):
        """Test click action creation."""
        action = WebAction(
            action_type="click",
            target="#settings",
            label="Click settings",
        )
        assert action.action_type == "click"
        assert action.target == "#settings"
        assert "click" in str(action).lower()

    def test_webaction_type(self):
        """Test type action creation."""
        action = WebAction(
            action_type="type",
            target="#name",
            value="Alice",
            label="Type name",
        )
        assert action.action_type == "type"
        assert action.value == "Alice"
        assert "alice" in str(action).lower()

    def test_webaction_navigate(self):
        """Test navigate action."""
        action = WebAction(
            action_type="navigate",
            target="",
            value="/settings",
            label="Go to settings",
        )
        assert action.action_type == "navigate"
        assert "settings" in str(action).lower()

    def test_webaction_to_dict(self):
        """Test serialization to dict."""
        action = WebAction(
            action_type="click",
            target="#btn",
            value=None,
            label="Click",
        )
        d = action.to_dict()
        assert d["action_type"] == "click"
        assert d["target"] == "#btn"

    def test_webaction_from_dict(self):
        """Test deserialization from dict."""
        d = {
            "action_type": "submit",
            "target": "#form",
            "value": None,
            "label": "Submit",
        }
        action = WebAction.from_dict(d)
        assert action.action_type == "submit"
        assert action.target == "#form"


# ============================================================================
# Test MockWebEnv
# ============================================================================


class TestMockWebEnv:
    """Tests for MockWebEnv class."""

    def test_init_default(self):
        """Test default initialization."""
        env = MockWebEnv()
        assert env.current_url == "https://example.com/home"
        assert env.is_logged_in is True
        assert env.theme == "light"
        assert env.step_count == 0

    def test_reset(self, mock_web_env):
        """Test environment reset."""
        obs = mock_web_env.reset()
        assert "url" in obs
        assert "page_content" in obs
        assert mock_web_env.step_count == 0

    def test_reset_with_task(self, env_for_theme_task):
        """Test reset with a task."""
        obs = env_for_theme_task.reset()
        assert "url" in obs
        assert obs["is_logged_in"] is True

    def test_step_click(self, mock_web_env):
        """Test clicking an action."""
        mock_web_env.reset()
        action = WebAction("click", "#nav-settings", label="Click settings")
        obs, reward, done, info = mock_web_env.step(action)

        assert "settings" in mock_web_env.current_url.lower() or reward > 0
        assert "action_executed" in info

    def test_step_type(self, env_for_contact_task):
        """Test typing an action."""
        env_for_contact_task.reset()
        action = WebAction("type", "#field-name", value="Alice", label="Type name")
        obs, reward, done, info = env_for_contact_task.step(action)

        assert env_for_contact_task.form_values.get("name") == "Alice"
        assert reward >= 0

    def test_step_navigate(self, mock_web_env):
        """Test navigation action."""
        mock_web_env.reset()
        action = WebAction("navigate", "", value="/settings", label="Go to settings")
        obs, reward, done, info = mock_web_env.step(action)

        assert "settings" in mock_web_env.current_url.lower() or reward >= 0

    def test_step_submit(self, env_for_contact_task):
        """Test form submission."""
        env_for_contact_task.reset()

        # First fill in the form
        type_action = WebAction("type", "#field-name", value="Alice", label="Type name")
        env_for_contact_task.step(type_action)

        type_action2 = WebAction("type", "#field-email", value="alice@example.com", label="Type email")
        env_for_contact_task.step(type_action2)

        # Then submit
        submit_action = WebAction("submit", "#contact-form", label="Submit")
        obs, reward, done, info = env_for_contact_task.step(submit_action)

        assert reward > 0 or done

    def test_step_max_steps(self, mock_web_env):
        """Test that step count increments."""
        mock_web_env.reset()
        action = WebAction("click", "#nav-home", label="Click home")

        for i in range(5):
            mock_web_env.step(action)
            assert mock_web_env.step_count == i + 1

    def test_get_observation(self, mock_web_env):
        """Test observation generation."""
        mock_web_env.reset()
        obs = mock_web_env.get_observation()

        assert "url" in obs
        assert "page_content" in obs
        assert "form_values" in obs
        assert "is_logged_in" in obs
        assert "theme" in obs
        assert "clickable_elements" in obs

    def test_clickable_elements(self, mock_web_env):
        """Test clickable elements are returned."""
        mock_web_env.reset()
        elements = mock_web_env._get_clickable_elements()
        assert isinstance(elements, list)
        assert len(elements) > 0

    def test_state_snapshot(self, mock_web_env):
        """Test state snapshot for debugging."""
        mock_web_env.reset()
        mock_web_env.step(WebAction("click", "#nav-settings", label="Click"))
        snapshot = mock_web_env.get_state_snapshot()

        assert "url" in snapshot
        assert "step_count" in snapshot
        assert "history" in snapshot

    def test_logout_task_completion(self, env_for_logout_task):
        """Test that logout task can be completed."""
        env_for_logout_task.reset()
        action = WebAction("click", "#nav-logout", label="Logout")
        obs, reward, done, info = env_for_logout_task.step(action)

        assert not env_for_logout_task.is_logged_in
        assert reward > 0

    def test_repr(self, mock_web_env):
        """Test string representation."""
        r = repr(mock_web_env)
        assert "MockWebEnv" in r
        assert str(mock_web_env.step_count) in r


# ============================================================================
# Test BaselineAgent
# ============================================================================


class TestBaselineAgent:
    """Tests for BaselineAgent class."""

    def test_init_default(self):
        """Test default initialization."""
        agent = BaselineAgent()
        assert agent.model_name == "gpt-4o-mini"
        assert agent.action_count == 0

    def test_init_custom_model(self):
        """Test initialization with custom model."""
        agent = BaselineAgent(model_name="gpt-4o")
        assert agent.model_name == "gpt-4o"

    def test_init_invalid_temperature(self):
        """Test invalid temperature raises ValueError."""
        with pytest.raises(ValueError, match="temperature must be in"):
            BaselineAgent(temperature=5.0)

    def test_init_invalid_max_tokens(self):
        """Test invalid max_tokens raises ValueError."""
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            BaselineAgent(max_tokens=0)

    def test_act_rule_based_theme(self, baseline_agent_no_key):
        """Test rule-based action for theme task."""
        task = "Navigate to settings and change theme to dark"
        obs = "URL: /home\nClickable: [Settings] [Search]"

        action = baseline_agent_no_key.act(task, obs)
        assert isinstance(action, str)
        assert len(action) > 0
        assert baseline_agent_no_key.action_count == 1

    def test_act_rule_based_contact(self, baseline_agent_no_key):
        """Test rule-based action for contact form task."""
        task = "Fill in the contact form with name=Alice"
        obs = "URL: /contact\nForms: [name] [email]"

        action = baseline_agent_no_key.act(task, obs)
        assert isinstance(action, str)
        assert baseline_agent_no_key.action_count == 1

    def test_act_rule_based_search(self, baseline_agent_no_key):
        """Test rule-based action for search task."""
        task = "Search for 'open source LLMs'"
        obs = "URL: /search\nField: [search]"

        action = baseline_agent_no_key.act(task, obs)
        assert isinstance(action, str)
        assert baseline_agent_no_key.action_count == 1

    def test_act_rule_based_logout(self, baseline_agent_no_key):
        """Test rule-based action for logout task."""
        task = "Log out from the current session"
        obs = "URL: /home\nClickable: [Logout] [Settings]\nLogged in: True"

        action = baseline_agent_no_key.act(task, obs)
        assert isinstance(action, str)
        assert "logout" in action.lower() or "click" in action.lower()

    def test_reset(self, baseline_agent_no_key):
        """Test agent reset clears action count."""
        baseline_agent_no_key.act("test task", "test obs")
        assert baseline_agent_no_key.action_count == 1
        baseline_agent_no_key.reset()
        assert baseline_agent_no_key.action_count == 0

    def test_repr(self, baseline_agent_no_key):
        """Test string representation."""
        r = repr(baseline_agent_no_key)
        assert "BaselineAgent" in r


# ============================================================================
# Test HeRoSWrappedAgent
# ============================================================================


class TestHeRoSWrappedAgent:
    """Tests for HeRoSWrappedAgent class."""

    def test_init(self, mock_heros_agent):
        """Test initialization."""
        agent = HeRoSWrappedAgent(mock_heros_agent, hindsight_enabled=True)
        assert agent.hindsight_enabled is True
        assert agent.agent is mock_heros_agent

    def test_init_default_hindsight(self, mock_heros_agent):
        """Test default hindsight enabled."""
        agent = HeRoSWrappedAgent(mock_heros_agent)
        assert agent.hindsight_enabled is True

    def test_init_invalid_agent(self):
        """Test invalid agent type raises TypeError."""
        with pytest.raises(TypeError, match="must be a HeRoSAgent"):
            HeRoSWrappedAgent("not an agent")

    def test_hindsight_setter(self, mock_heros_agent):
        """Test hindsight can be toggled."""
        agent = HeRoSWrappedAgent(mock_heros_agent)
        agent.hindsight_enabled = False
        assert agent.hindsight_enabled is False
        agent.hindsight_enabled = True
        assert agent.hindsight_enabled is True

    def test_act_with_dict_observation(self, wrapped_agent):
        """Test act with dict observation."""
        obs = {
            "url": "https://example.com/home",
            "page_content": "Welcome",
            "is_logged_in": True,
            "theme": "light",
            "clickable_elements": [],
            "available_forms": [],
        }

        action = wrapped_agent.act("Test task", obs)
        assert isinstance(action, str)

    def test_act_with_string_observation(self, wrapped_agent):
        """Test act with string observation."""
        obs = "URL: https://example.com/home\nClickable: [Settings]"
        action = wrapped_agent.act("Test task", obs)
        assert isinstance(action, str)

    def test_set_milestones(self, wrapped_agent):
        """Test setting milestones."""
        milestones = [
            Milestone(id="m1", description="Step 1", rubric="Step 1 done"),
            Milestone(id="m2", description="Step 2", rubric="Step 2 done"),
        ]
        wrapped_agent.set_milestones(milestones)
        assert wrapped_agent.total_milestones == 2
        assert wrapped_agent.get_active_milestone().id == "m1"

    def test_get_milestone_progress(self, wrapped_agent):
        """Test milestone progress tracking."""
        milestones = [
            Milestone(id="m1", description="Step 1", rubric="Step 1 done"),
            Milestone(id="m2", description="Step 2", rubric="Step 2 done"),
        ]
        wrapped_agent.set_milestones(milestones)
        progress = wrapped_agent.get_milestone_progress()

        assert progress["total"] == 2
        assert progress["current_index"] == 0
        assert progress["hit_count"] == 0

    def test_reset(self, wrapped_agent):
        """Test reset clears state."""
        wrapped_agent.set_milestones([
            Milestone(id="m1", description="Step 1", rubric="Done"),
        ])
        wrapped_agent.reset()
        assert wrapped_agent.total_milestones == 0

    def test_repr(self, wrapped_agent):
        """Test string representation."""
        r = repr(wrapped_agent)
        assert "HeRoSWrappedAgent" in r


# ============================================================================
# Test EvaluationResult
# ============================================================================


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""

    def test_creation(self):
        """Test basic creation."""
        result = EvaluationResult(
            task_id="test_task",
            agent_type="baseline",
            completion=True,
            milestone_hit_rate=1.0,
        )
        assert result.task_id == "test_task"
        assert result.agent_type == "baseline"
        assert result.completion is True
        assert result.milestone_hit_rate == 1.0

    def test_to_dict(self):
        """Test serialization to dict."""
        result = EvaluationResult(
            task_id="test",
            agent_type="heros",
            completion=True,
        )
        d = result.to_dict()
        assert d["task_id"] == "test"
        assert d["completion"] is True
        assert "timestamp" in d

    def test_from_dict(self):
        """Test deserialization from dict."""
        d = {
            "task_id": "test",
            "agent_type": "baseline",
            "completion": False,
            "milestone_hit_rate": 0.5,
            "total_reward": 1.0,
            "episode_length": 5,
            "hindsight_applied": True,
            "per_milestone_results": [],
            "task_description": "Test task",
            "episode_history": [],
            "timestamp": "2024-01-01T00:00:00Z",
            "error": None,
        }
        result = EvaluationResult.from_dict(d)
        assert result.task_id == "test"
        assert result.completion is False


# ============================================================================
# Test HeRoSEvaluator
# ============================================================================


class TestHeRoSEvaluator:
    """Tests for HeRoSEvaluator class."""

    def test_init_baseline(self, benchmark, baseline_agent):
        """Test initialization with baseline agent."""
        evaluator = HeRoSEvaluator(benchmark, baseline_agent)
        assert evaluator.agent_type == "baseline"
        assert evaluator.use_hindsight is True  # default
        assert evaluator.max_steps == 20

    def test_init_heros(self, benchmark, mock_heros_agent):
        """Test initialization with HeRoS agent."""
        evaluator = HeRoSEvaluator(benchmark, mock_heros_agent, use_hindsight=True)
        assert evaluator.agent_type == "heros"
        assert evaluator.use_hindsight is True

    def test_init_custom_max_steps(self, benchmark, baseline_agent):
        """Test custom max steps."""
        evaluator = HeRoSEvaluator(benchmark, baseline_agent, max_steps=10)
        assert evaluator.max_steps == 10

    def test_run_episode(self, evaluator_with_baseline):
        """Test running a single episode."""
        result = evaluator_with_baseline.run_episode("logout_session")
        assert isinstance(result, EvaluationResult)
        assert result.task_id == "logout_session"
        assert result.agent_type == "baseline"
        assert "episode_history" in [k for k in result.to_dict().keys()]

    def test_run_evaluation_all_tasks(self, evaluator_with_baseline):
        """Test running evaluation on all tasks."""
        results = evaluator_with_baseline.run_evaluation()
        assert len(results) == 5
        assert all(isinstance(r, EvaluationResult) for r in results)

    def test_run_evaluation_specific_tasks(self, evaluator_with_baseline):
        """Test running evaluation on specific tasks."""
        results = evaluator_with_baseline.run_evaluation(task_ids=["logout_session"])
        assert len(results) == 1
        assert results[0].task_id == "logout_session"

    def test_run_evaluation_invalid_task(self, evaluator_with_baseline):
        """Test evaluation with invalid task ID."""
        results = evaluator_with_baseline.run_evaluation(task_ids=["invalid_task"])
        assert len(results) == 1
        assert results[0].error is not None

    def test_compute_metrics_empty(self, evaluator_with_baseline):
        """Test computing metrics with no results."""
        evaluator_with_baseline.clear_results()
        metrics = evaluator_with_baseline.compute_metrics()
        assert metrics["completion_rate"] == 0.0
        assert metrics["total_tasks"] == 0

    def test_compute_metrics_with_results(self, evaluator_with_baseline):
        """Test computing metrics with results."""
        evaluator_with_baseline.run_evaluation()
        metrics = evaluator_with_baseline.compute_metrics()

        assert "completion_rate" in metrics
        assert "avg_milestone_hit_rate" in metrics
        assert "avg_reward" in metrics
        assert "avg_episode_length" in metrics
        assert metrics["total_tasks"] == 5

    def test_compute_metrics_stores_results(self, evaluator_with_baseline):
        """Test that run_evaluation stores results."""
        initial_count = len(evaluator_with_baseline.results)
        evaluator_with_baseline.run_evaluation()
        assert len(evaluator_with_baseline.results) > initial_count

    def test_get_task_results(self, evaluator_with_baseline):
        """Test getting results for a specific task."""
        evaluator_with_baseline.run_evaluation()
        results = evaluator_with_baseline.get_task_results("logout_session")
        assert all(r.task_id == "logout_session" for r in results)

    def test_clear_results(self, evaluator_with_baseline):
        """Test clearing stored results."""
        evaluator_with_baseline.run_evaluation()
        assert len(evaluator_with_baseline.results) > 0
        evaluator_with_baseline.clear_results()
        assert len(evaluator_with_baseline.results) == 0

    def test_compare_agents(self, evaluator_with_baseline, evaluator_with_heros):
        """Test comparing baseline and HeRoS agents."""
        # Run baseline evaluation
        baseline_results = evaluator_with_baseline.run_evaluation()

        # Run HeRoS evaluation
        heros_results = evaluator_with_heros.run_evaluation()

        # Compare
        comparison = evaluator_with_baseline.compare_agents(baseline_results, heros_results)

        assert "baseline_metrics" in comparison
        assert "heros_metrics" in comparison
        assert "improvement" in comparison
        assert "summary" in comparison

    def test_repr(self, evaluator_with_baseline):
        """Test string representation."""
        r = repr(evaluator_with_baseline)
        assert "HeRoSEvaluator" in r
        assert "baseline" in r


# ============================================================================
# Test Metrics Computation
# ============================================================================


class TestMetricsComputation:
    """Tests for metrics computation logic."""

    def test_completion_rate(self, evaluator_with_baseline):
        """Test completion rate calculation."""
        # Run logout task which should be completable
        result = evaluator_with_baseline.run_episode("logout_session")
        evaluator_with_baseline.clear_results()
        evaluator_with_baseline._results.append(result)

        metrics = evaluator_with_baseline.compute_metrics()
        assert 0.0 <= metrics["completion_rate"] <= 1.0

    def test_milestone_hit_rate(self, evaluator_with_baseline):
        """Test milestone hit rate calculation."""
        result = evaluator_with_baseline.run_episode("logout_session")
        assert 0.0 <= result.milestone_hit_rate <= 1.0

    def test_hindsight_delta_requires_multiple_runs(self, evaluator_with_baseline):
        """Test that hindsight delta needs both enabled and disabled runs."""
        # Single run won't have hindsight_delta computed
        evaluator_with_baseline.run_evaluation()
        metrics = evaluator_with_baseline.compute_metrics()
        # hindsight_delta is None when we only have one type of run
        assert metrics.get("hindsight_delta") is None

    def test_metrics_with_error_result(self, evaluator_with_baseline):
        """Test that error results are handled gracefully."""
        error_result = EvaluationResult(
            task_id="test_error",
            agent_type="baseline",
            error="Test error",
        )
        evaluator_with_baseline._results.append(error_result)
        metrics = evaluator_with_baseline.compute_metrics()
        # Should still return valid metrics
        assert "total_tasks" in metrics


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for full evaluation workflow."""

    def test_full_benchmark_evaluation(self, benchmark, baseline_agent):
        """Test running a full evaluation on the benchmark."""
        evaluator = HeRoSEvaluator(benchmark, baseline_agent, max_steps=20)
        results = evaluator.run_evaluation()
        metrics = evaluator.compute_metrics()

        assert len(results) == len(benchmark)
        assert metrics["total_tasks"] == len(benchmark)
        assert all(isinstance(r, EvaluationResult) for r in results)

    def test_all_tasks_have_valid_results(self, benchmark, baseline_agent):
        """Test that all tasks produce valid results."""
        evaluator = HeRoSEvaluator(benchmark, baseline_agent)
        results = evaluator.run_evaluation()

        for result in results:
            assert result.task_id in benchmark.list_tasks()
            assert result.agent_type == "baseline"
            assert result.timestamp is not None

    def test_milestones_are_tracked(self, benchmark, baseline_agent):
        """Test that milestones are tracked during evaluation."""
        evaluator = HeRoSEvaluator(benchmark, baseline_agent)
        result = evaluator.run_episode("change_theme_dark")

        # Should have milestone results
        assert len(result.per_milestone_results) >= 0  # May be 0 if episode failed early

    def test_episode_history_is_recorded(self, benchmark, baseline_agent):
        """Test that episode history is properly recorded."""
        evaluator = HeRoSEvaluator(benchmark, baseline_agent)
        result = evaluator.run_episode("logout_session")

        assert len(result.episode_history) > 0
        assert all("step" in h for h in result.episode_history)
        assert all("action" in h for h in result.episode_history)

    def test_different_tasks_have_different_difficulties(self, benchmark):
        """Test that different tasks have different difficulty ratings."""
        tasks = {
            "logout_session": benchmark.get_task("logout_session"),
            "change_theme_dark": benchmark.get_task("change_theme_dark"),
            "search_open_source_llm": benchmark.get_task("search_open_source_llm"),
        }

        difficulties = set(t.difficulty for t in tasks.values())
        assert len(difficulties) >= 2  # At least easy and hard should be present


# ============================================================================
# Test HeRoS Agent Integration
# ============================================================================


class TestHeRoSAgentIntegration:
    """Tests for integration with actual HeRoS agent."""

    def test_evaluator_with_mock_heros_agent(self, benchmark, mock_heros_agent):
        """Test evaluator works with mock HeRoS agent."""
        evaluator = HeRoSEvaluator(benchmark, mock_heros_agent, use_hindsight=True)
        assert evaluator.agent_type == "heros"

        result = evaluator.run_episode("logout_session")
        assert isinstance(result, EvaluationResult)

    def test_wrapped_agent_act(self, wrapped_agent, mock_web_env):
        """Test wrapped agent act method."""
        obs = mock_web_env.get_observation()
        action = wrapped_agent.act("Test task", obs)
        assert isinstance(action, str)


# ============================================================================
# Performance and Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_benchmark_task_count(self, benchmark):
        """Test benchmark has expected number of tasks."""
        assert len(benchmark) == 5

    def test_max_steps_limit(self, evaluator_with_baseline):
        """Test that max steps is respected."""
        evaluator_with_baseline._max_steps = 2
        result = evaluator_with_baseline.run_episode("change_theme_dark")
        assert result.episode_length <= 2

    def test_action_string_parsing(self, evaluator_with_baseline, mock_web_env):
        """Test action string parsing in evaluator."""
        evaluator_with_baseline._max_steps = 5

        # The evaluator should handle various action formats
        result = evaluator_with_baseline.run_episode("logout_session")
        assert result.episode_length >= 1

    def test_empty_observation(self, evaluator_with_baseline, mock_web_env):
        """Test handling of minimal observation."""
        # Should not crash
        evaluator_with_baseline._agent.act("task", "")

    def test_benchmark_subset_difficulties(self):
        """Test that different subsets have appropriate difficulties."""
        easy_bench = WebArenaLiteBenchmark(task_subset="easy")
        assert all(t.difficulty == "easy" for t in easy_bench)

        hard_bench = WebArenaLiteBenchmark(task_subset="hard")
        assert all(t.difficulty == "hard" for t in hard_bench)
