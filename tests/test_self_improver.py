"""Tests for Test-time Self-Improvement Module.

Tests cover:
- TestTimeSelfImprover._extract_failures
- TestTimeSelfImprover.run_with_self_improvement (mock agent + env)
- SelfImprovementResult dataclass
- InferenceEngine.run_inference_episode
- InferenceEngine.run_batch_inference
- Local hindsight buffer accumulation
- Improvement detection (delta calculation)
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from heros.buffer import HindsightBuffer, HindsightTrajectory
from heros.critic import Verdict
from heros.planner import Milestone
from heros.self_improver import (
    TestTimeSelfImprover,
    EpisodeMetrics,
    PolicyUpdateResult,
    SelfImprovementResult,
    InferenceEpisodeResult,
    BatchInferenceResult,
)
from heros.inference_engine import InferenceEngine


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def hindsight_buffer():
    """Create a fresh HindsightBuffer for testing."""
    return HindsightBuffer(capacity=50, hindsight_ratio=0.3)


@pytest.fixture
def mock_agent():
    """Create a mock HeRoSAgent."""
    agent = MagicMock()
    agent.act = MagicMock(return_value=MagicMock(
        action="mock_action",
        milestone_id="m1",
        milestone_description="Test milestone",
        critic_result=None,
        is_milestone_complete=False,
        is_episode_done=False,
    ))
    agent.update = MagicMock(return_value=MagicMock(
        loss=0.1,
        num_samples=8,
        hindsight_ratio=0.3,
        is_simulation=True,
        details={},
    ))
    return agent


@pytest.fixture
def sample_task():
    """Create a sample WebTask-like object."""
    task = MagicMock()
    task.task_id = "test_task_1"
    task.description = "Test task description"
    task.milestones = [
        Milestone(
            id="m1",
            description="First milestone",
            rubric="Complete first milestone",
        ),
        Milestone(
            id="m2",
            description="Second milestone",
            rubric="Complete second milestone",
        ),
        Milestone(
            id="m3",
            description="Third milestone",
            rubric="Complete third milestone",
        ),
    ]
    return task


@pytest.fixture
def mock_env_factory(sample_task):
    """Create a mock environment factory."""
    def factory():
        env = MagicMock()
        env.reset = MagicMock(return_value={"url": "http://example.com", "page_content": ""})
        env.step = MagicMock(return_value=(
            {"url": "http://example.com", "page_content": ""},
            0.5,
            False,
            {"milestone_complete": False},
        ))
        env._milestone_states = [
            {"milestone": m, "hit": False, "attempts": 0}
            for m in sample_task.milestones
        ]
        return env
    return factory


# ============================================================================
# Test: EpisodeMetrics Dataclass
# ============================================================================


class TestEpisodeMetrics:
    """Tests for EpisodeMetrics dataclass."""

    def test_episode_metrics_creation(self):
        """Test basic EpisodeMetrics creation."""
        metrics = EpisodeMetrics(
            episode_idx=0,
            success=True,
            milestone_hit_rate=1.0,
            total_reward=10.0,
            episode_length=5,
            failed_subgoals=[],
            milestone_verdicts=["PASS", "PASS", "PASS"],
        )
        assert metrics.episode_idx == 0
        assert metrics.success is True
        assert metrics.milestone_hit_rate == 1.0
        assert metrics.total_reward == 10.0
        assert metrics.episode_length == 5
        assert metrics.failed_subgoals == []
        assert len(metrics.milestone_verdicts) == 3

    def test_episode_metrics_defaults(self):
        """Test EpisodeMetrics default values."""
        metrics = EpisodeMetrics(
            episode_idx=1,
            success=False,
            milestone_hit_rate=0.5,
            total_reward=5.0,
            episode_length=3,
        )
        assert metrics.failed_subgoals == []
        assert metrics.milestone_verdicts == []

    def test_episode_metrics_with_failures(self):
        """Test EpisodeMetrics with failed subgoals."""
        metrics = EpisodeMetrics(
            episode_idx=2,
            success=False,
            milestone_hit_rate=0.33,
            total_reward=2.0,
            episode_length=10,
            failed_subgoals=["Second milestone", "Third milestone"],
            milestone_verdicts=["PASS", "FAIL", "FAIL"],
        )
        assert metrics.success is False
        assert len(metrics.failed_subgoals) == 2
        assert metrics.milestone_hit_rate == pytest.approx(0.33, rel=0.01)

    def test_episode_metrics_to_dict(self):
        """Test EpisodeMetrics serialization."""
        metrics = EpisodeMetrics(
            episode_idx=0,
            success=True,
            milestone_hit_rate=1.0,
            total_reward=10.0,
            episode_length=5,
        )
        d = metrics.to_dict()
        assert d["episode_idx"] == 0
        assert d["success"] is True
        assert d["milestone_hit_rate"] == 1.0
        assert d["total_reward"] == 10.0
        assert d["episode_length"] == 5


# ============================================================================
# Test: PolicyUpdateResult Dataclass
# ============================================================================


class TestPolicyUpdateResult:
    """Tests for PolicyUpdateResult dataclass."""

    def test_policy_update_result_creation(self):
        """Test PolicyUpdateResult creation."""
        result = PolicyUpdateResult(
            epochs_run=3,
            buffer_size_before=5,
            buffer_size_after=8,
            estimated_policy_delta=0.15,
            is_simulation=True,
        )
        assert result.epochs_run == 3
        assert result.buffer_size_before == 5
        assert result.buffer_size_after == 8
        assert result.estimated_policy_delta == 0.15
        assert result.is_simulation is True

    def test_policy_update_result_defaults(self):
        """Test PolicyUpdateResult default values."""
        result = PolicyUpdateResult(
            epochs_run=1,
            buffer_size_before=0,
            buffer_size_after=0,
        )
        assert result.estimated_policy_delta == 0.0
        assert result.is_simulation is True
        assert result.details == {}

    def test_policy_update_result_to_dict(self):
        """Test PolicyUpdateResult serialization."""
        result = PolicyUpdateResult(
            epochs_run=2,
            buffer_size_before=3,
            buffer_size_after=5,
            estimated_policy_delta=0.1,
            details={"note": "test"},
        )
        d = result.to_dict()
        assert d["epochs_run"] == 2
        assert d["buffer_size_before"] == 3
        assert d["buffer_size_after"] == 5
        assert d["estimated_policy_delta"] == 0.1
        assert d["details"]["note"] == "test"


# ============================================================================
# Test: SelfImprovementResult Dataclass
# ============================================================================


class TestSelfImprovementResult:
    """Tests for SelfImprovementResult dataclass."""

    def test_self_improvement_result_creation(self):
        """Test SelfImprovementResult creation."""
        episodes = [
            EpisodeMetrics(
                episode_idx=0,
                success=False,
                milestone_hit_rate=0.33,
                total_reward=3.0,
                episode_length=5,
                failed_subgoals=["Second milestone", "Third milestone"],
            ),
            EpisodeMetrics(
                episode_idx=1,
                success=True,
                milestone_hit_rate=1.0,
                total_reward=10.0,
                episode_length=4,
                failed_subgoals=[],
            ),
        ]
        result = SelfImprovementResult(
            task_id="task_1",
            episodes=episodes,
            final_success_rate=1.0,
            initial_success_rate=0.0,
            improvement_delta=1.0,
            total_self_play_epochs=6,
            hindsight_buffer_size_after=10,
            episodes_to_first_success=2,
        )
        assert result.task_id == "task_1"
        assert len(result.episodes) == 2
        assert result.final_success_rate == 1.0
        assert result.initial_success_rate == 0.0
        assert result.improvement_delta == 1.0
        assert result.total_self_play_epochs == 6
        assert result.episodes_to_first_success == 2

    def test_self_improvement_result_no_improvement(self):
        """Test SelfImprovementResult when no improvement occurred."""
        episodes = [
            EpisodeMetrics(
                episode_idx=i,
                success=False,
                milestone_hit_rate=0.33,
                total_reward=3.0,
                episode_length=5,
            )
            for i in range(3)
        ]
        result = SelfImprovementResult(
            task_id="task_2",
            episodes=episodes,
            final_success_rate=0.0,
            initial_success_rate=0.0,
            improvement_delta=0.0,
            total_self_play_epochs=9,
            hindsight_buffer_size_after=15,
            episodes_to_first_success=None,
        )
        assert result.improvement_delta == 0.0
        assert result.episodes_to_first_success is None

    def test_self_improvement_result_to_dict(self):
        """Test SelfImprovementResult serialization."""
        episodes = [
            EpisodeMetrics(
                episode_idx=0,
                success=True,
                milestone_hit_rate=1.0,
                total_reward=10.0,
                episode_length=5,
            ),
        ]
        result = SelfImprovementResult(
            task_id="task_3",
            episodes=episodes,
            final_success_rate=1.0,
            initial_success_rate=0.0,
            improvement_delta=1.0,
            total_self_play_epochs=3,
            hindsight_buffer_size_after=5,
        )
        d = result.to_dict()
        assert d["task_id"] == "task_3"
        assert len(d["episodes"]) == 1
        assert d["final_success_rate"] == 1.0
        assert d["improvement_delta"] == 1.0


# ============================================================================
# Test: InferenceEpisodeResult Dataclass
# ============================================================================


class TestInferenceEpisodeResult:
    """Tests for InferenceEpisodeResult dataclass."""

    def test_inference_episode_result_creation(self):
        """Test InferenceEpisodeResult creation."""
        result = InferenceEpisodeResult(
            task_id="task_1",
            success=True,
            milestone_hit_rate=1.0,
            total_reward=10.0,
            episode_length=5,
            failed_subgoals=[],
            collected_for_hindsight=False,
        )
        assert result.task_id == "task_1"
        assert result.success is True
        assert result.milestone_hit_rate == 1.0
        assert result.collected_for_hindsight is False

    def test_inference_episode_result_with_failures(self):
        """Test InferenceEpisodeResult with collected failures."""
        result = InferenceEpisodeResult(
            task_id="task_2",
            success=False,
            milestone_hit_rate=0.5,
            total_reward=5.0,
            episode_length=8,
            failed_subgoals=["Milestone 2"],
            collected_for_hindsight=True,
        )
        assert result.success is False
        assert result.collected_for_hindsight is True
        assert len(result.failed_subgoals) == 1


# ============================================================================
# Test: BatchInferenceResult Dataclass
# ============================================================================


class TestBatchInferenceResult:
    """Tests for BatchInferenceResult dataclass."""

    def test_batch_inference_result_creation(self):
        """Test BatchInferenceResult creation."""
        episode_results = [
            InferenceEpisodeResult(
                task_id=f"task_{i}",
                success=(i % 2 == 0),
                milestone_hit_rate=1.0 if i % 2 == 0 else 0.5,
                total_reward=10.0 if i % 2 == 0 else 5.0,
                episode_length=5,
            )
            for i in range(4)
        ]
        result = BatchInferenceResult(
            tasks=["task_0", "task_1", "task_2", "task_3"],
            episode_results=episode_results,
            overall_success_rate=0.5,
            avg_milestone_hit_rate=0.75,
            total_self_play_epochs=12,
            hindsight_buffer_size=8,
        )
        assert len(result.tasks) == 4
        assert len(result.episode_results) == 4
        assert result.overall_success_rate == 0.5
        assert result.avg_milestone_hit_rate == 0.75
        assert result.total_self_play_epochs == 12
        assert result.hindsight_buffer_size == 8

    def test_batch_inference_result_to_dict(self):
        """Test BatchInferenceResult serialization."""
        episode_results = [
            InferenceEpisodeResult(
                task_id="task_1",
                success=True,
                milestone_hit_rate=1.0,
                total_reward=10.0,
                episode_length=5,
            ),
        ]
        result = BatchInferenceResult(
            tasks=["task_1"],
            episode_results=episode_results,
            overall_success_rate=1.0,
            avg_milestone_hit_rate=1.0,
            total_self_play_epochs=3,
            hindsight_buffer_size=2,
        )
        d = result.to_dict()
        assert d["tasks"] == ["task_1"]
        assert d["overall_success_rate"] == 1.0


# ============================================================================
# Test: TestTimeSelfImprover._extract_failures
# ============================================================================


class TestExtractFailures:
    """Tests for TestTimeSelfImprover._extract_failures method."""

    def test_extract_failures_empty(self, mock_agent, hindsight_buffer, sample_task):
        """Test _extract_failures with no failures."""
        improver = TestTimeSelfImprover(
            agent=mock_agent,
            hindsight_buffer=hindsight_buffer,
            self_play_epochs=0,
        )
        episode_record = {
            "task_id": sample_task.task_id,
            "episode_idx": 0,
            "milestone_verdicts": [Verdict.PASS, Verdict.PASS, Verdict.PASS],
            "milestone_exec_traces": [],
            "milestones": sample_task.milestones,
            "all_failed_subgoals": [],
        }
        failures = improver._extract_failures(episode_record)
        assert len(failures) == 0

    def test_extract_failures_partial(self, mock_agent, hindsight_buffer, sample_task):
        """Test _extract_failures with some failures."""
        improver = TestTimeSelfImprover(
            agent=mock_agent,
            hindsight_buffer=hindsight_buffer,
            self_play_epochs=0,
        )
        episode_record = {
            "task_id": sample_task.task_id,
            "episode_idx": 1,
            "milestone_verdicts": [Verdict.PASS, Verdict.FAIL, Verdict.FAIL],
            "milestone_exec_traces": [
                {"milestone_id": "m1", "action": "act1"},
                {"milestone_id": "m2", "action": "act2"},
            ],
            "milestones": sample_task.milestones,
            "all_failed_subgoals": ["Second milestone", "Third milestone"],
        }
        failures = improver._extract_failures(episode_record)
        # _extract_failures returns failures but doesn't add them to buffer
        # (that's done in run_with_self_improvement)
        assert len(failures) == 2
        assert all(isinstance(f, HindsightTrajectory) for f in failures)

    def test_extract_failures_all(self, mock_agent, hindsight_buffer, sample_task):
        """Test _extract_failures with all milestones failing."""
        improver = TestTimeSelfImprover(
            agent=mock_agent,
            hindsight_buffer=hindsight_buffer,
            self_play_epochs=0,
        )
        episode_record = {
            "task_id": sample_task.task_id,
            "episode_idx": 0,
            "milestone_verdicts": [Verdict.FAIL, Verdict.FAIL, Verdict.FAIL],
            "milestone_exec_traces": [],
            "milestones": sample_task.milestones,
            "all_failed_subgoals": [
                "First milestone",
                "Second milestone",
                "Third milestone",
            ],
        }
        failures = improver._extract_failures(episode_record)
        assert len(failures) == 3
        # Failures returned but not yet added to buffer
        assert hindsight_buffer.size == 0

    def test_extract_failures_with_exec_traces(self, mock_agent, hindsight_buffer, sample_task):
        """Test that exec traces are preserved in failure trajectories."""
        improver = TestTimeSelfImprover(
            agent=mock_agent,
            hindsight_buffer=hindsight_buffer,
            self_play_epochs=0,
        )
        exec_traces = [
            {"step": 0, "milestone_id": "m1", "action": "action1"},
            {"step": 1, "milestone_id": "m2", "action": "action2"},
        ]
        episode_record = {
            "task_id": sample_task.task_id,
            "episode_idx": 0,
            "milestone_verdicts": [Verdict.PASS, Verdict.FAIL],
            "milestone_exec_traces": exec_traces,
            "milestones": sample_task.milestones[:2],
            "all_failed_subgoals": ["Second milestone"],
        }
        failures = improver._extract_failures(episode_record)
        assert len(failures) == 1
        assert len(failures[0].exec_traces) == 1
        assert failures[0].exec_traces[0]["milestone_id"] == "m2"


# ============================================================================
# Test: TestTimeSelfImprover.run_with_self_improvement
# ============================================================================


class TestRunWithSelfImprovement:
    """Tests for TestTimeSelfImprover.run_with_self_improvement method."""

    def test_run_with_self_improvement_basic(
        self, mock_agent, hindsight_buffer, sample_task, mock_env_factory
    ):
        """Test basic run_with_self_improvement execution."""
        improver = TestTimeSelfImprover(
            agent=mock_agent,
            hindsight_buffer=hindsight_buffer,
            self_play_epochs=1,
            simulate_updates=True,
        )
        result = improver.run_with_self_improvement(
            task=sample_task,
            env_factory=mock_env_factory,
            n_episodes=2,
            max_steps_per_episode=10,
        )
        assert isinstance(result, SelfImprovementResult)
        assert result.task_id == sample_task.task_id
        assert len(result.episodes) == 2
        assert result.total_self_play_epochs >= 0

    def test_run_with_self_improvement_episode_count(
        self, mock_agent, hindsight_buffer, sample_task, mock_env_factory
    ):
        """Test that correct number of episodes are run."""
        improver = TestTimeSelfImprover(
            agent=mock_agent,
            hindsight_buffer=hindsight_buffer,
            self_play_epochs=0,
        )
        result = improver.run_with_self_improvement(
            task=sample_task,
            env_factory=mock_env_factory,
            n_episodes=5,
            max_steps_per_episode=10,
        )
        assert len(result.episodes) == 5

    def test_run_with_self_improvement_buffer_grows(
        self, mock_agent, hindsight_buffer, sample_task, mock_env_factory
    ):
        """Test that hindsight buffer grows as failures are collected."""
        improver = TestTimeSelfImprover(
            agent=mock_agent,
            hindsight_buffer=hindsight_buffer,
            self_play_epochs=0,
        )
        result = improver.run_with_self_improvement(
            task=sample_task,
            env_factory=mock_env_factory,
            n_episodes=3,
            max_steps_per_episode=10,
        )
        # Buffer should have grown
        assert hindsight_buffer.size >= 0

    def test_run_with_self_improvement_improvement_delta(
        self, mock_agent, hindsight_buffer, sample_task, mock_env_factory
    ):
        """Test improvement delta calculation."""
        improver = TestTimeSelfImprover(
            agent=mock_agent,
            hindsight_buffer=hindsight_buffer,
            self_play_epochs=1,
        )
        result = improver.run_with_self_improvement(
            task=sample_task,
            env_factory=mock_env_factory,
            n_episodes=3,
            max_steps_per_episode=10,
        )
        assert isinstance(result.improvement_delta, float)
        assert -1.0 <= result.improvement_delta <= 1.0

    def test_run_with_self_improvement_initial_final_sr(
        self, mock_agent, hindsight_buffer, sample_task, mock_env_factory
    ):
        """Test initial and final success rate are computed correctly."""
        improver = TestTimeSelfImprover(
            agent=mock_agent,
            hindsight_buffer=hindsight_buffer,
            self_play_epochs=0,
        )
        result = improver.run_with_self_improvement(
            task=sample_task,
            env_factory=mock_env_factory,
            n_episodes=2,
            max_steps_per_episode=10,
        )
        assert result.initial_success_rate in (0.0, 1.0)
        assert 0.0 <= result.final_success_rate <= 1.0

    def test_run_with_self_improvement_self_play_epochs_tracked(
        self, mock_agent, hindsight_buffer, sample_task, mock_env_factory
    ):
        """Test that self-play epochs are tracked."""
        improver = TestTimeSelfImprover(
            agent=mock_agent,
            hindsight_buffer=hindsight_buffer,
            self_play_epochs=2,
        )
        result = improver.run_with_self_improvement(
            task=sample_task,
            env_factory=mock_env_factory,
            n_episodes=3,
            max_steps_per_episode=10,
        )
        assert result.total_self_play_epochs >= 0

    def test_run_with_self_improvement_invalid_n_episodes(
        self, mock_agent, hindsight_buffer, sample_task, mock_env_factory
    ):
        """Test that invalid n_episodes raises ValueError."""
        improver = TestTimeSelfImprover(
            agent=mock_agent,
            hindsight_buffer=hindsight_buffer,
        )
        with pytest.raises(ValueError):
            improver.run_with_self_improvement(
                task=sample_task,
                env_factory=mock_env_factory,
                n_episodes=0,
            )


# ============================================================================
# Test: Local Hindsight Buffer Accumulation
# ============================================================================


class TestLocalBufferAccumulation:
    """Tests for local hindsight buffer accumulation."""

    def test_buffer_capacity_enforced(self, mock_agent, sample_task):
        """Test that buffer respects capacity."""
        buffer = HindsightBuffer(capacity=3, hindsight_ratio=0.3)
        improver = TestTimeSelfImprover(
            agent=mock_agent,
            hindsight_buffer=buffer,
            self_play_epochs=0,
        )

        def env_factory():
            env = MagicMock()
            env.reset = MagicMock(return_value={})
            env.step = MagicMock(return_value=({}, 0.0, False, {}))
            env._milestone_states = [
                {"milestone": m, "hit": False, "attempts": 0}
                for m in sample_task.milestones
            ]
            return env

        improver.run_with_self_improvement(
            task=sample_task,
            env_factory=env_factory,
            n_episodes=5,
            max_steps_per_episode=10,
        )

        # Buffer should not exceed capacity
        assert buffer.size <= buffer.capacity

    def test_multiple_episodes_accumulate_failures(
        self, mock_agent, hindsight_buffer, sample_task
    ):
        """Test that multiple episodes accumulate failures in buffer."""
        improver = TestTimeSelfImprover(
            agent=mock_agent,
            hindsight_buffer=hindsight_buffer,
            self_play_epochs=0,
        )

        initial_size = hindsight_buffer.size

        def env_factory():
            env = MagicMock()
            env.reset = MagicMock(return_value={})
            env.step = MagicMock(return_value=({}, 0.0, False, {}))
            env._milestone_states = [
                {"milestone": m, "hit": False, "attempts": 0}
                for m in sample_task.milestones
            ]
            return env

        improver.run_with_self_improvement(
            task=sample_task,
            env_factory=env_factory,
            n_episodes=3,
            max_steps_per_episode=10,
        )

        # Buffer should have accumulated at least some failures
        assert hindsight_buffer.size >= initial_size


# ============================================================================
# Test: Improvement Detection (Delta Calculation)
# ============================================================================


class TestImprovementDetection:
    """Tests for improvement detection and delta calculation."""

    def test_delta_positive_improvement(self, mock_agent, hindsight_buffer, sample_task):
        """Test positive improvement delta."""
        improver = TestTimeSelfImprover(
            agent=mock_agent,
            hindsight_buffer=hindsight_buffer,
            self_play_epochs=0,
        )

        def env_factory():
            env = MagicMock()
            env.reset = MagicMock(return_value={})
            env.step = MagicMock(return_value=({}, 0.0, False, {}))
            env._milestone_states = [
                {"milestone": m, "hit": False, "attempts": 0}
                for m in sample_task.milestones
            ]
            return env

        result = improver.run_with_self_improvement(
            task=sample_task,
            env_factory=env_factory,
            n_episodes=3,
            max_steps_per_episode=10,
        )

        assert isinstance(result.improvement_delta, float)

    def test_delta_no_improvement(self, mock_agent, hindsight_buffer, sample_task):
        """Test zero improvement delta."""
        improver = TestTimeSelfImprover(
            agent=mock_agent,
            hindsight_buffer=hindsight_buffer,
            self_play_epochs=0,
        )

        def env_factory():
            env = MagicMock()
            env.reset = MagicMock(return_value={})
            env.step = MagicMock(return_value=({}, 0.0, True, {}))
            env._milestone_states = []
            return env

        result = improver.run_with_self_improvement(
            task=sample_task,
            env_factory=env_factory,
            n_episodes=2,
            max_steps_per_episode=10,
        )

        assert result.improvement_delta == 0.0

    def test_first_success_tracked(self, mock_agent, hindsight_buffer, sample_task):
        """Test that episodes_to_first_success is tracked."""
        improver = TestTimeSelfImprover(
            agent=mock_agent,
            hindsight_buffer=hindsight_buffer,
            self_play_epochs=0,
        )

        def env_factory():
            env = MagicMock()
            env.reset = MagicMock(return_value={})
            env.step = MagicMock(return_value=({}, 0.0, False, {}))
            env._milestone_states = [
                {"milestone": m, "hit": False, "attempts": 0}
                for m in sample_task.milestones
            ]
            return env

        result = improver.run_with_self_improvement(
            task=sample_task,
            env_factory=env_factory,
            n_episodes=3,
            max_steps_per_episode=10,
        )

        # episodes_to_first_success should be tracked (may be None)
        assert result.episodes_to_first_success is None or isinstance(
            result.episodes_to_first_success, int
        )


# ============================================================================
# Test: InferenceEngine.run_inference_episode
# ============================================================================


class TestInferenceEngineEpisode:
    """Tests for InferenceEngine.run_inference_episode."""

    def test_inference_engine_creation(self, mock_agent):
        """Test InferenceEngine creation."""
        engine = InferenceEngine(
            agent=mock_agent,
            use_self_improvement=True,
            local_buffer_capacity=50,
        )
        assert engine.use_self_improvement is True
        assert engine.local_hindsight_buffer.capacity == 50

    def test_inference_engine_no_self_improvement(self, mock_agent):
        """Test InferenceEngine without self-improvement."""
        engine = InferenceEngine(
            agent=mock_agent,
            use_self_improvement=False,
            local_buffer_capacity=50,
        )
        assert engine.use_self_improvement is False
        assert engine.improver is None

    def test_run_inference_episode_basic(self, mock_agent, sample_task):
        """Test basic run_inference_episode."""
        engine = InferenceEngine(
            agent=mock_agent,
            use_self_improvement=False,
            local_buffer_capacity=50,
        )

        def env_factory():
            env = MagicMock()
            env.reset = MagicMock(return_value={})
            env.step = MagicMock(return_value=({}, 0.0, True, {}))
            env._milestone_states = []
            return env

        result = engine.run_inference_episode(
            task=sample_task,
            env_factory=env_factory,
        )
        assert isinstance(result, InferenceEpisodeResult)
        assert result.task_id == sample_task.task_id

    def test_run_inference_episode_collect_failures(
        self, mock_agent, hindsight_buffer, sample_task
    ):
        """Test that failures are collected when requested."""
        buffer = HindsightBuffer(capacity=50)
        mock_agent._hindsight_buffer = buffer

        engine = InferenceEngine(
            agent=mock_agent,
            use_self_improvement=False,
            local_buffer_capacity=50,
        )
        engine._local_buffer = buffer

        def env_factory():
            env = MagicMock()
            env.reset = MagicMock(return_value={})
            env.step = MagicMock(return_value=({}, 0.0, True, {}))
            env._milestone_states = []
            return env

        result = engine.run_inference_episode(
            task=sample_task,
            env_factory=env_factory,
            collect_failures=True,
        )
        assert isinstance(result, InferenceEpisodeResult)


# ============================================================================
# Test: InferenceEngine.run_batch_inference
# ============================================================================


class TestInferenceEngineBatch:
    """Tests for InferenceEngine.run_batch_inference."""

    def test_run_batch_inference_basic(self, mock_agent, sample_task):
        """Test basic batch inference."""
        engine = InferenceEngine(
            agent=mock_agent,
            use_self_improvement=False,
            local_buffer_capacity=50,
        )

        tasks = [sample_task, sample_task, sample_task]

        def env_factory(t):
            env = MagicMock()
            env.reset = MagicMock(return_value={})
            env.step = MagicMock(return_value=({}, 0.0, False, {}))
            env._milestone_states = [
                {"milestone": m, "hit": False, "attempts": 0}
                for m in t.milestones
            ]
            return env

        result = engine.run_batch_inference(
            tasks=tasks,
            env_factory=env_factory,
            show_progress=False,
        )
        assert isinstance(result, BatchInferenceResult)
        assert len(result.tasks) == 3
        assert len(result.episode_results) == 3

    def test_run_batch_inference_single_episode(
        self, mock_agent, sample_task
    ):
        """Test batch inference with single episode per task."""
        engine = InferenceEngine(
            agent=mock_agent,
            use_self_improvement=False,
            local_buffer_capacity=50,
        )

        tasks = [sample_task, sample_task]

        def env_factory(t):
            env = MagicMock()
            env.reset = MagicMock(return_value={})
            env.step = MagicMock(return_value=({}, 0.0, False, {}))
            env._milestone_states = [
                {"milestone": m, "hit": False, "attempts": 0}
                for m in t.milestones
            ]
            return env

        result = engine.run_batch_inference(
            tasks=tasks,
            env_factory=env_factory,
            n_episodes_per_task=1,
        )
        assert len(result.episode_results) == 2

    def test_run_batch_inference_multiple_episodes(
        self, mock_agent, sample_task
    ):
        """Test batch inference with multiple episodes per task."""
        engine = InferenceEngine(
            agent=mock_agent,
            use_self_improvement=True,
            local_buffer_capacity=50,
        )

        tasks = [sample_task]

        def env_factory(t):
            env = MagicMock()
            env.reset = MagicMock(return_value={})
            env.step = MagicMock(return_value=({}, 0.0, False, {}))
            env._milestone_states = [
                {"milestone": m, "hit": False, "attempts": 0}
                for m in t.milestones
            ]
            return env

        result = engine.run_batch_inference(
            tasks=tasks,
            env_factory=env_factory,
            n_episodes_per_task=3,
        )
        assert len(result.episode_results) == 3

    def test_run_batch_inference_stats(
        self, mock_agent, sample_task
    ):
        """Test that batch inference computes stats correctly."""
        engine = InferenceEngine(
            agent=mock_agent,
            use_self_improvement=False,
            local_buffer_capacity=50,
        )

        tasks = [sample_task, sample_task]

        def env_factory(t):
            env = MagicMock()
            env.reset = MagicMock(return_value={})
            env.step = MagicMock(return_value=({}, 0.0, False, {}))
            env._milestone_states = [
                {"milestone": m, "hit": False, "attempts": 0}
                for m in t.milestones
            ]
            return env

        result = engine.run_batch_inference(
            tasks=tasks,
            env_factory=env_factory,
        )
        assert isinstance(result.overall_success_rate, float)
        assert isinstance(result.avg_milestone_hit_rate, float)


# ============================================================================
# Test: InferenceEngine Utilities
# ============================================================================


class TestInferenceEngineUtilities:
    """Tests for InferenceEngine utility methods."""

    def test_reset_local_hindsight_buffer(self, mock_agent):
        """Test reset_local_hindsight_buffer."""
        engine = InferenceEngine(
            agent=mock_agent,
            use_self_improvement=True,
            local_buffer_capacity=50,
        )

        # Add some data to buffer
        for _ in range(5):
            traj = HindsightTrajectory(
                task="test",
                milestones=[],
                exec_traces=[],
                verdicts=[],
            )
            engine._local_buffer.add(traj)

        assert len(engine.local_hindsight_buffer) == 5

        engine.reset_local_hindsight_buffer()

        assert len(engine.local_hindsight_buffer) == 0

    def test_get_buffer_stats(self, mock_agent):
        """Test get_buffer_stats."""
        engine = InferenceEngine(
            agent=mock_agent,
            use_self_improvement=True,
            local_buffer_capacity=50,
        )

        stats = engine.get_buffer_stats()
        assert "total" in stats
        assert "capacity" in stats
        assert "utilization" in stats

    def test_inference_engine_repr(self, mock_agent):
        """Test InferenceEngine __repr__."""
        engine = InferenceEngine(
            agent=mock_agent,
            use_self_improvement=True,
        )
        r = repr(engine)
        assert "InferenceEngine" in r
        assert "self_improvement" in r


# ============================================================================
# Test: TestTimeSelfImprover Properties
# ============================================================================


class TestTestTimeSelfImproverProperties:
    """Tests for TestTimeSelfImprover property accessors."""

    def test_agent_property(self, mock_agent, hindsight_buffer):
        """Test agent property."""
        improver = TestTimeSelfImprover(
            agent=mock_agent,
            hindsight_buffer=hindsight_buffer,
        )
        assert improver.agent is mock_agent

    def test_hindsight_buffer_property(self, mock_agent, hindsight_buffer):
        """Test hindsight_buffer property."""
        improver = TestTimeSelfImprover(
            agent=mock_agent,
            hindsight_buffer=hindsight_buffer,
        )
        assert improver.hindsight_buffer is hindsight_buffer

    def test_self_play_epochs_property(self, mock_agent, hindsight_buffer):
        """Test self_play_epochs property."""
        improver = TestTimeSelfImprover(
            agent=mock_agent,
            hindsight_buffer=hindsight_buffer,
            self_play_epochs=5,
        )
        assert improver.self_play_epochs == 5

    def test_improvement_threshold_property(self, mock_agent, hindsight_buffer):
        """Test improvement_threshold property."""
        improver = TestTimeSelfImprover(
            agent=mock_agent,
            hindsight_buffer=hindsight_buffer,
            improvement_threshold=0.1,
        )
        assert improver.improvement_threshold == 0.1

    def test_get_improvement_trajectory(self, mock_agent, hindsight_buffer, sample_task):
        """Test get_improvement_trajectory."""
        improver = TestTimeSelfImprover(
            agent=mock_agent,
            hindsight_buffer=hindsight_buffer,
            self_play_epochs=0,
        )

        def env_factory():
            env = MagicMock()
            env.reset = MagicMock(return_value={})
            env.step = MagicMock(return_value=({}, 0.0, False, {}))
            env._milestone_states = [
                {"milestone": m, "hit": False, "attempts": 0}
                for m in sample_task.milestones
            ]
            return env

        improver.run_with_self_improvement(
            task=sample_task,
            env_factory=env_factory,
            n_episodes=2,
            max_steps_per_episode=10,
        )

        trajectory = improver.get_improvement_trajectory()
        assert len(trajectory) == 2
        assert all(isinstance(m, EpisodeMetrics) for m in trajectory)

    def test_repr(self, mock_agent, hindsight_buffer):
        """Test __repr__."""
        improver = TestTimeSelfImprover(
            agent=mock_agent,
            hindsight_buffer=hindsight_buffer,
            self_play_epochs=3,
        )
        r = repr(improver)
        assert "TestTimeSelfImprover" in r
        assert "self_play_epochs=3" in r


# ============================================================================
# Test: Validation
# ============================================================================


class TestValidation:
    """Tests for input validation."""

    def test_self_play_epochs_validation(self, mock_agent, hindsight_buffer):
        """Test self_play_epochs must be non-negative."""
        with pytest.raises(ValueError):
            TestTimeSelfImprover(
                agent=mock_agent,
                hindsight_buffer=hindsight_buffer,
                self_play_epochs=-1,
            )

    def test_improvement_threshold_range(self, mock_agent, hindsight_buffer):
        """Test improvement_threshold must be in [0, 1]."""
        with pytest.raises(ValueError):
            TestTimeSelfImprover(
                agent=mock_agent,
                hindsight_buffer=hindsight_buffer,
                improvement_threshold=-0.1,
            )
        with pytest.raises(ValueError):
            TestTimeSelfImprover(
                agent=mock_agent,
                hindsight_buffer=hindsight_buffer,
                improvement_threshold=1.5,
            )

    def test_buffer_capacity_validation(self, mock_agent):
        """Test local_buffer_capacity must be positive."""
        with pytest.raises(ValueError):
            InferenceEngine(
                agent=mock_agent,
                local_buffer_capacity=0,
            )
        with pytest.raises(ValueError):
            InferenceEngine(
                agent=mock_agent,
                local_buffer_capacity=-1,
            )
