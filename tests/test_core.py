"""Comprehensive tests for HeRoS core RL training loop components.

Tests HeRoSEnv, HeRoSAgent, PPOTrainer, and TrainingLogger.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

import heros
from heros.planner import Milestone, SubgoalPlan, SubgoalPlanner
from heros.critic import CriticResult, MilestoneCritic, Verdict
from heros.buffer import HindsightBuffer, HindsightTrajectory
from heros.trainer import HindsightTrainer, UpdateResult
from heros.env import HeRoSEnv, MilestoneStatus
from heros.agent import HeRoSAgent, ActResult
from heros.core import PPOTrainer
from heros.logging_utils import TrainingMetrics, TrainingLogger


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_task_fn():
    """Create a mock task factory function."""
    def task_fn():
        return {
            "observation": "initial_state",
            "task": "Test task: accomplish subgoals A, B, C",
        }
    return task_fn


@pytest.fixture
def mock_planner():
    """Create a mock planner with predefined milestones."""
    planner = MagicMock(spec=SubgoalPlanner)
    planner.plan.return_value = SubgoalPlan(
        task="Test task",
        milestones=[
            Milestone(id="m1", description="Subgoal A", rubric="A completed", expected_output=""),
            Milestone(id="m2", description="Subgoal B", rubric="B completed", expected_output=""),
            Milestone(id="m3", description="Subgoal C", rubric="C completed", expected_output=""),
        ],
    )
    return planner


@pytest.fixture
def mock_critic():
    """Create a mock milestone critic."""
    critic = MagicMock(spec=MilestoneCritic)
    critic.review.return_value = CriticResult(
        verdict=Verdict.PASS,
        feedback="Looks good",
        confidence=0.9,
        reward_signal=1.0,
    )
    return critic


@pytest.fixture
def hindsight_buffer():
    """Create a fresh HindsightBuffer."""
    return HindsightBuffer(capacity=100, hindsight_ratio=0.3, seed=42)


@pytest.fixture
def hindsight_trainer(hindsight_buffer):
    """Create a HindsightTrainer."""
    return HindsightTrainer(buffer=hindsight_buffer, learning_rate=1e-5)


@pytest.fixture
def heros_env(mock_task_fn, mock_planner, mock_critic, hindsight_buffer):
    """Create a HeRoSEnv with mocked dependencies."""
    return HeRoSEnv(
        task_fn=mock_task_fn,
        planner=mock_planner,
        critic=mock_critic,
        hindsight_buffer=hindsight_buffer,
    )


@pytest.fixture
def heros_agent(mock_planner, mock_critic, hindsight_buffer, hindsight_trainer, heros_env):
    """Create a HeRoSAgent with mocked dependencies."""
    return HeRoSAgent(
        planner=mock_planner,
        critic=mock_critic,
        hindsight_buffer=hindsight_buffer,
        trainer=hindsight_trainer,
        env=heros_env,
    )


@pytest.fixture
def ppo_trainer(heros_agent):
    """Create a PPOTrainer."""
    return PPOTrainer(
        agent=heros_agent,
        hindsight_ratio=0.3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        seed=42,
    )


# ---------------------------------------------------------------------------
# HeRoSEnv Tests
# ---------------------------------------------------------------------------


class TestHeRoSEnvInit:
    """Tests for HeRoSEnv initialization."""

    def test_init_with_valid_args(self, mock_task_fn, mock_planner, mock_critic, hindsight_buffer):
        env = HeRoSEnv(
            task_fn=mock_task_fn,
            planner=mock_planner,
            critic=mock_critic,
            hindsight_buffer=hindsight_buffer,
        )
        assert env._task_fn is mock_task_fn
        assert env._planner is mock_planner
        assert env._critic is mock_critic
        assert env._hindsight_buffer is hindsight_buffer

    def test_init_with_invalid_task_fn(self, mock_planner, mock_critic, hindsight_buffer):
        with pytest.raises(TypeError, match="task_fn must be callable"):
            HeRoSEnv(task_fn="not_callable", planner=mock_planner, critic=mock_critic, hindsight_buffer=hindsight_buffer)

    def test_init_with_invalid_planner(self, mock_task_fn, mock_critic, hindsight_buffer):
        with pytest.raises(TypeError, match="planner must be a SubgoalPlanner"):
            HeRoSEnv(task_fn=mock_task_fn, planner="not_planner", critic=mock_critic, hindsight_buffer=hindsight_buffer)

    def test_init_with_invalid_critic(self, mock_task_fn, mock_planner, hindsight_buffer):
        with pytest.raises(TypeError, match="critic must be a MilestoneCritic"):
            HeRoSEnv(task_fn=mock_task_fn, planner=mock_planner, critic="not_critic", hindsight_buffer=hindsight_buffer)

    def test_init_with_invalid_buffer(self, mock_task_fn, mock_planner, mock_critic):
        with pytest.raises(TypeError, match="hindsight_buffer must be a HindsightBuffer"):
            HeRoSEnv(task_fn=mock_task_fn, planner=mock_planner, critic=mock_critic, hindsight_buffer="not_buffer")


class TestHeRoSEnvReset:
    """Tests for HeRoSEnv.reset()."""

    def test_reset_returns_observation(self, heros_env):
        obs = heros_env.reset()
        assert isinstance(obs, dict)
        assert "observation" in obs
        assert "task" in obs
        assert "milestones" in obs
        assert "active_milestone" in obs

    def test_reset_sets_current_plan(self, heros_env):
        heros_env.reset()
        assert heros_env.current_plan is not None
        assert len(heros_env.current_plan.milestones) == 3

    def test_reset_initializes_milestone_statuses(self, heros_env):
        heros_env.reset()
        assert len(heros_env.milestone_statuses) == 3
        assert heros_env.milestone_statuses[0].status == "active"
        assert heros_env.milestone_statuses[1].status == "pending"
        assert heros_env.milestone_statuses[2].status == "pending"

    def test_reset_resets_episode_state(self, heros_env):
        heros_env.reset()
        assert heros_env.episode_step_count == 0
        assert heros_env.episode_reward == 0.0

    def test_reset_with_planning_error_graceful_degradation(self, mock_task_fn, mock_planner, mock_critic, hindsight_buffer):
        mock_planner.plan.side_effect = Exception("Planning failed")
        env = HeRoSEnv(task_fn=mock_task_fn, planner=mock_planner, critic=mock_critic, hindsight_buffer=hindsight_buffer)
        obs = env.reset()
        # Should create minimal plan
        assert env.current_plan is not None
        assert len(env.current_plan.milestones) >= 1


class TestHeRoSEnvStep:
    """Tests for HeRoSEnv.step()."""

    def test_step_returns_tuple(self, heros_env):
        heros_env.reset()
        result = heros_env.step("test action")
        assert isinstance(result, tuple)
        assert len(result) == 4
        obs, reward, done, info = result
        assert isinstance(obs, dict)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_step_increments_step_count(self, heros_env):
        heros_env.reset()
        heros_env.step("action1")
        assert heros_env.episode_step_count == 1
        heros_env.step("action2")
        assert heros_env.episode_step_count == 2

    def test_step_records_exec_trace(self, heros_env):
        heros_env.reset()
        heros_env.step("action1")
        assert len(heros_env._episode_exec_traces) == 1

    def test_step_requires_reset_first(self, heros_env):
        with pytest.raises(RuntimeError, match="Environment not reset"):
            heros_env.step("action")

    def test_multiple_steps_advance_milestones(self, heros_env, mock_critic):
        heros_env.reset()
        # First step passes m1
        mock_critic.review.return_value = CriticResult(verdict=Verdict.PASS, feedback="", confidence=0.9, reward_signal=1.0)
        heros_env.step("action1")
        assert heros_env.current_milestone_idx == 1
        # Second step passes m2
        heros_env.step("action2")
        assert heros_env.current_milestone_idx == 2

    def test_failed_milestone_advances_to_next(self, heros_env, mock_critic):
        heros_env.reset()
        mock_critic.review.return_value = CriticResult(verdict=Verdict.FAIL, feedback="", confidence=0.9, reward_signal=0.0)
        heros_env.step("action1")
        # Still advances because milestone is "complete" (failed)
        assert heros_env.current_milestone_idx == 1

    def test_partial_verdict_stays_on_same_milestone(self, heros_env, mock_critic):
        heros_env.reset()
        mock_critic.review.return_value = CriticResult(verdict=Verdict.PARTIAL, feedback="", confidence=0.5, reward_signal=0.5)
        heros_env.step("action1")
        # Does NOT advance
        assert heros_env.current_milestone_idx == 0


class TestHeRoSMilestoneStatus:
    """Tests for HeRoSEnv milestone tracking."""

    def test_get_milestone_status_returns_dict(self, heros_env):
        heros_env.reset()
        status = heros_env.get_milestone_status()
        assert isinstance(status, dict)
        assert "current_index" in status
        assert "total" in status
        assert "passed" in status
        assert "failed" in status
        assert "pending" in status
        assert "active" in status

    def test_get_milestone_status_values(self, heros_env):
        heros_env.reset()
        status = heros_env.get_milestone_status()
        assert status["current_index"] == 0
        assert status["total"] == 3
        assert status["passed"] == 0
        assert status["failed"] == 0
        assert status["pending"] == 2
        assert status["active"] is not None

    def test_get_milestone_status_empty_env(self):
        mock_planner = MagicMock(spec=SubgoalPlanner)
        mock_planner.plan.side_effect = Exception("Planning failed")
        env = HeRoSEnv(
            task_fn=lambda: {"task": "x"},
            planner=mock_planner,
            critic=MagicMock(spec=MilestoneCritic),
            hindsight_buffer=HindsightBuffer(),
        )
        # Reset creates minimal plan via graceful degradation
        env.reset()
        status = env.get_milestone_status()
        # Graceful degradation creates a minimal single-milestone plan
        assert status["total"] >= 1


class TestHeRoSBufferIntegration:
    """Tests for HeRoSEnv buffer integration."""

    def test_add_to_buffer(self, heros_env, hindsight_buffer):
        heros_env.reset()
        traj = heros_env.create_trajectory_from_episode()
        heros_env.add_to_buffer(traj)
        assert len(hindsight_buffer) >= 1

    def test_add_to_buffer_type_check(self, heros_env):
        with pytest.raises(TypeError, match="Expected HindsightTrajectory"):
            heros_env.add_to_buffer("not a trajectory")

    def test_create_trajectory_from_episode(self, heros_env):
        heros_env.reset()
        heros_env.step("action1")
        traj = heros_env.create_trajectory_from_episode()
        assert isinstance(traj, HindsightTrajectory)
        # Task comes from the env's task_fn
        assert "Test task" in traj.task
        assert len(traj.milestones) == 3

    def test_create_trajectory_requires_reset(self, heros_env):
        with pytest.raises(RuntimeError, match="No episode in progress"):
            heros_env.create_trajectory_from_episode()


# ---------------------------------------------------------------------------
# HeRoSAgent Tests
# ---------------------------------------------------------------------------


class TestHeRoSAgentInit:
    """Tests for HeRoSAgent initialization."""

    def test_init_with_valid_args(self, mock_planner, mock_critic, hindsight_buffer, hindsight_trainer):
        agent = HeRoSAgent(
            planner=mock_planner,
            critic=mock_critic,
            hindsight_buffer=hindsight_buffer,
            trainer=hindsight_trainer,
        )
        assert agent.planner is mock_planner
        assert agent.critic is mock_critic
        assert agent.hindsight_buffer is hindsight_buffer
        assert agent.trainer is hindsight_trainer

    def test_init_with_env(self, mock_planner, mock_critic, hindsight_buffer, hindsight_trainer, heros_env):
        agent = HeRoSAgent(
            planner=mock_planner,
            critic=mock_critic,
            hindsight_buffer=hindsight_buffer,
            trainer=hindsight_trainer,
            env=heros_env,
        )
        assert agent.env is heros_env

    def test_init_with_invalid_planner(self, mock_critic, hindsight_buffer, hindsight_trainer):
        with pytest.raises(TypeError, match="planner must be a SubgoalPlanner"):
            HeRoSAgent(planner="x", critic=mock_critic, hindsight_buffer=hindsight_buffer, trainer=hindsight_trainer)

    def test_init_with_invalid_critic(self, mock_planner, hindsight_buffer, hindsight_trainer):
        with pytest.raises(TypeError, match="critic must be a MilestoneCritic"):
            HeRoSAgent(planner=mock_planner, critic="x", hindsight_buffer=hindsight_buffer, trainer=hindsight_trainer)

    def test_init_with_invalid_buffer(self, mock_planner, mock_critic, hindsight_trainer):
        with pytest.raises(TypeError, match="hindsight_buffer must be a HindsightBuffer"):
            HeRoSAgent(planner=mock_planner, critic=mock_critic, hindsight_buffer="x", trainer=hindsight_trainer)


class TestHeRoSAgentAct:
    """Tests for HeRoSAgent.act()."""

    def test_act_returns_act_result(self, heros_agent, heros_env):
        heros_env.reset()
        obs = heros_env._build_observation()
        result = heros_agent.act(obs)
        assert isinstance(result, ActResult)
        assert isinstance(result.action, str)
        assert isinstance(result.milestone_id, str)
        assert isinstance(result.milestone_description, str)

    def test_act_with_no_active_milestone(self, heros_agent):
        result = heros_agent.act({"active_milestone": None})
        assert result.action == ""
        assert result.milestone_id == ""
        assert result.is_episode_done is False

    def test_act_calls_critic(self, heros_agent, heros_env):
        heros_env.reset()
        obs = heros_env._build_observation()
        heros_agent.act(obs)
        heros_env._critic.review.assert_called()

    def test_act_result_has_critic_result(self, heros_agent, heros_env):
        heros_env.reset()
        obs = heros_env._build_observation()
        result = heros_agent.act(obs)
        assert result.critic_result is not None

    def test_act_episode_done_at_max_steps(self, heros_agent):
        obs = {
            "active_milestone": {"id": "m1", "description": "test", "rubric": "x"},
            "task": "test",
            "step_count": 100,
        }
        result = heros_agent.act(obs)
        assert result.is_episode_done is True


class TestHeRoSAgentUpdate:
    """Tests for HeRoSAgent.update()."""

    def test_update_returns_update_result(self, heros_agent, hindsight_buffer):
        # Add some trajectories
        for i in range(5):
            traj = HindsightTrajectory(
                task=f"task_{i}",
                milestones=[Milestone(id="m1", description="x", rubric="y", expected_output="")],
                exec_traces=[],
                verdicts=[Verdict.PASS],
                unmet_rubrics=[],
            )
            hindsight_buffer.add(traj)
        result = heros_agent.update()
        assert isinstance(result, UpdateResult)
        assert isinstance(result.loss, float)

    def test_update_with_empty_buffer(self, heros_agent):
        result = heros_agent.update()
        assert result.num_samples == 0
        assert result.is_simulation is True

    def test_update_increments_update_count(self, heros_agent, hindsight_buffer):
        for i in range(3):
            traj = HindsightTrajectory(
                task=f"task_{i}",
                milestones=[Milestone(id="m1", description="x", rubric="y", expected_output="")],
                exec_traces=[],
                verdicts=[Verdict.FAIL],
                unmet_rubrics=["rubric1"],
            )
            hindsight_buffer.add(traj)
        initial_count = heros_agent.update_count
        heros_agent.update()
        assert heros_agent.update_count == initial_count + 1


class TestHeRoSAgentRunEpisode:
    """Tests for HeRoSAgent.run_episode()."""

    def test_run_episode_returns_summary(self, heros_agent, heros_env, mock_critic):
        heros_env.reset()
        # Make critic return FAIL to avoid infinite loop
        mock_critic.review.return_value = CriticResult(
            verdict=Verdict.FAIL, feedback="", confidence=0.9, reward_signal=0.0
        )
        summary = heros_agent.run_episode()
        assert isinstance(summary, dict)
        assert "episode_reward" in summary
        assert "step_count" in summary
        assert "milestones_passed" in summary
        assert "milestones_failed" in summary

    def test_run_episode_increments_episode_count(self, heros_agent, heros_env, mock_critic):
        heros_env.reset()
        mock_critic.review.return_value = CriticResult(verdict=Verdict.FAIL, feedback="", confidence=0.9, reward_signal=0.0)
        initial = heros_agent.episode_count
        heros_agent.run_episode()
        assert heros_agent.episode_count == initial + 1


# ---------------------------------------------------------------------------
# PPOTrainer Tests
# ---------------------------------------------------------------------------


class TestPPOTrainerInit:
    """Tests for PPOTrainer initialization."""

    def test_init_with_valid_args(self, heros_agent):
        trainer = PPOTrainer(agent=heros_agent, hindsight_ratio=0.3, gamma=0.99)
        assert trainer.agent is heros_agent
        assert trainer.hindsight_ratio == 0.3
        assert trainer.gamma == 0.99
        assert trainer.gae_lambda == 0.95

    def test_init_invalid_hindsight_ratio(self, heros_agent):
        with pytest.raises(ValueError, match="hindsight_ratio must be between"):
            PPOTrainer(agent=heros_agent, hindsight_ratio=1.5)

    def test_init_invalid_gamma(self, heros_agent):
        with pytest.raises(ValueError, match="gamma must be between"):
            PPOTrainer(agent=heros_agent, gamma=2.0)

    def test_init_invalid_gae_lambda(self, heros_agent):
        with pytest.raises(ValueError, match="gae_lambda must be between"):
            PPOTrainer(agent=heros_agent, gae_lambda=-0.5)

    def test_init_invalid_clip_epsilon(self, heros_agent):
        with pytest.raises(ValueError, match="clip_epsilon must be non-negative"):
            PPOTrainer(agent=heros_agent, clip_epsilon=-0.1)

    def test_hindsight_ratio_setter(self, heros_agent):
        trainer = PPOTrainer(agent=heros_agent)
        trainer.hindsight_ratio = 0.5
        assert trainer.hindsight_ratio == 0.5

    def test_hindsight_ratio_setter_bounds(self, heros_agent):
        trainer = PPOTrainer(agent=heros_agent)
        with pytest.raises(ValueError):
            trainer.hindsight_ratio = -0.1


class TestPPOTrainerComputeAdvantages:
    """Tests for PPOTrainer.compute_advantages()."""

    def test_compute_advantages_empty(self, ppo_trainer):
        result = ppo_trainer.compute_advantages([])
        assert result == []

    def test_compute_advantages_single_reward(self, ppo_trainer):
        result = ppo_trainer.compute_advantages([1.0])
        assert len(result) == 1

    def test_compute_advantages_multiple_rewards(self, ppo_trainer):
        rewards = [0.0, 0.5, 1.0, 0.5, 1.0]
        result = ppo_trainer.compute_advantages(rewards)
        assert len(result) == len(rewards)

    def test_compute_advantages_normalized(self, ppo_trainer):
        rewards = [1.0, 1.0, 1.0, 1.0, 1.0]
        result = ppo_trainer.compute_advantages(rewards)
        # Normalized advantages should have mean ~0
        mean_adv = sum(result) / len(result)
        assert abs(mean_adv) < 1e-6

    def test_compute_advantages_gae_shape(self, ppo_trainer):
        rewards = [0.0, 1.0, 0.0, 1.0]
        result = ppo_trainer.compute_advantages(rewards)
        assert len(result) == len(rewards)
        assert all(isinstance(x, float) for x in result)


class TestPPOTrainerComputeReturns:
    """Tests for PPOTrainer.compute_returns()."""

    def test_compute_returns_empty(self, ppo_trainer):
        result = ppo_trainer.compute_returns([])
        assert result == []

    def test_compute_returns_single(self, ppo_trainer):
        result = ppo_trainer.compute_returns([1.0])
        assert len(result) == 1
        assert result[0] == 1.0

    def test_compute_returns_discounted(self, ppo_trainer):
        rewards = [1.0, 1.0, 1.0]
        result = ppo_trainer.compute_returns(rewards)
        # R_t = r_t + gamma * R_{t+1}
        # R_2 = 1.0
        # R_1 = 1.0 + 0.99 * 1.0 = 1.99
        # R_0 = 1.0 + 0.99 * 1.99 = 2.9701
        assert abs(result[2] - 1.0) < 1e-6
        assert abs(result[1] - 1.99) < 1e-5
        assert abs(result[0] - 2.9701) < 1e-5


class TestPPOTrainerTrainStep:
    """Tests for PPOTrainer.train_step()."""

    def test_train_step_empty_trajectories(self, ppo_trainer):
        metrics = ppo_trainer.train_step([])
        assert isinstance(metrics, TrainingMetrics)
        assert metrics.num_trajectories == 0
        assert metrics.hindsight_utilization == 0.0

    def test_train_step_with_trajectories(self, ppo_trainer, hindsight_buffer):
        for i in range(3):
            traj = HindsightTrajectory(
                task=f"task_{i}",
                milestones=[Milestone(id="m1", description="x", rubric="y", expected_output="")],
                exec_traces=[],
                verdicts=[Verdict.PASS],
                unmet_rubrics=[],
            )
            hindsight_buffer.add(traj)
        trajectories = hindsight_buffer.sample(3)
        metrics = ppo_trainer.train_step(trajectories)
        assert metrics.num_trajectories == 3
        assert metrics.training_step == 1

    def test_train_step_with_hindsight_trajectories(self, ppo_trainer, hindsight_buffer):
        # Add standard trajectory
        traj1 = HindsightTrajectory(
            task="task_1",
            milestones=[Milestone(id="m1", description="x", rubric="y", expected_output="")],
            exec_traces=[],
            verdicts=[Verdict.PASS],
            unmet_rubrics=[],
        )
        hindsight_buffer.add(traj1)
        # Add hindsight-enhanced trajectory
        traj2 = HindsightTrajectory(
            task="task_2",
            milestones=[Milestone(id="m1", description="x", rubric="y", expected_output="")],
            exec_traces=[],
            verdicts=[Verdict.FAIL],
            unmet_rubrics=["rubric1"],
            is_hindsight_enhanced=True,
        )
        hindsight_buffer.add(traj2)
        trajectories = hindsight_buffer.sample(2)
        metrics = ppo_trainer.train_step(trajectories)
        assert metrics.num_trajectories == 2
        assert metrics.hindsight_utilization >= 0.0

    def test_train_step_increments_training_step(self, ppo_trainer, hindsight_buffer):
        traj = HindsightTrajectory(
            task="task",
            milestones=[Milestone(id="m1", description="x", rubric="y", expected_output="")],
            exec_traces=[],
            verdicts=[Verdict.PASS],
            unmet_rubrics=[],
        )
        hindsight_buffer.add(traj)
        trajectories = hindsight_buffer.sample(1)
        initial = ppo_trainer.training_step
        ppo_trainer.train_step(trajectories)
        assert ppo_trainer.training_step == initial + 1


class TestPPOTrainerPPOUpdate:
    """Tests for PPOTrainer.ppo_update()."""

    def test_ppo_update_empty(self, ppo_trainer):
        result = ppo_trainer.ppo_update([], [], [])
        assert result == 0.0

    def test_ppo_update_with_data(self, ppo_trainer):
        old_log_probs = [0.0, 0.0, 0.0]
        new_log_probs = [0.1, 0.1, 0.1]
        advantages = [1.0, 0.5, -0.5]
        result = ppo_trainer.ppo_update(old_log_probs, new_log_probs, advantages)
        assert isinstance(result, float)
        assert result >= 0.0

    def test_ppo_update_mismatched_lengths(self, ppo_trainer):
        with pytest.raises(ValueError, match="All input lists must have the same length"):
            ppo_trainer.ppo_update([0.0], [0.0, 0.0], [1.0])


class TestPPOTrainerSampleBatch:
    """Tests for PPOTrainer.sample_training_batch()."""

    def test_sample_batch_from_buffer(self, ppo_trainer, hindsight_buffer):
        for i in range(5):
            traj = HindsightTrajectory(
                task=f"task_{i}",
                milestones=[Milestone(id="m1", description="x", rubric="y", expected_output="")],
                exec_traces=[],
                verdicts=[Verdict.PASS],
                unmet_rubrics=[],
            )
            hindsight_buffer.add(traj)
        batch = ppo_trainer.sample_training_batch(batch_size=3)
        assert len(batch) == 3


# ---------------------------------------------------------------------------
# TrainingLogger Tests
# ---------------------------------------------------------------------------


class TestTrainingMetricsInit:
    """Tests for TrainingMetrics initialization."""

    def test_init_with_valid_args(self):
        metrics = TrainingMetrics(
            milestone_success_rate=0.8,
            hindsight_utilization=0.3,
            avg_reward=0.75,
            policy_loss=0.25,
            training_step=1,
        )
        assert metrics.milestone_success_rate == 0.8
        assert metrics.hindsight_utilization == 0.3
        assert metrics.avg_reward == 0.75
        assert metrics.policy_loss == 0.25
        assert metrics.training_step == 1

    def test_init_clamping(self):
        metrics = TrainingMetrics(
            milestone_success_rate=1.5,  # Should clamp to 1.0
            hindsight_utilization=-0.5,  # Should clamp to 0.0
            avg_reward=0.75,
            policy_loss=-1.0,  # Should clamp to 0.0
            training_step=0,
        )
        assert metrics.milestone_success_rate == 1.0
        assert metrics.hindsight_utilization == 0.0
        assert metrics.policy_loss == 0.0

    def test_init_default_timestamp(self):
        metrics = TrainingMetrics(0.0, 0.0, 0.0, 0.0)
        assert metrics.timestamp is not None


class TestTrainingMetricsSerialization:
    """Tests for TrainingMetrics serialization."""

    def test_to_dict(self):
        metrics = TrainingMetrics(
            milestone_success_rate=0.8,
            hindsight_utilization=0.3,
            avg_reward=0.75,
            policy_loss=0.25,
            training_step=1,
            num_trajectories=10,
            num_hindsight=3,
        )
        d = metrics.to_dict()
        assert isinstance(d, dict)
        assert d["milestone_success_rate"] == 0.8
        assert d["hindsight_utilization"] == 0.3
        assert d["avg_reward"] == 0.75
        assert d["policy_loss"] == 0.25
        assert d["training_step"] == 1
        assert d["num_trajectories"] == 10
        assert d["num_hindsight"] == 3

    def test_from_dict(self):
        d = {
            "milestone_success_rate": 0.8,
            "hindsight_utilization": 0.3,
            "avg_reward": 0.75,
            "policy_loss": 0.25,
            "training_step": 1,
            "num_trajectories": 10,
            "num_hindsight": 3,
        }
        metrics = TrainingMetrics.from_dict(d)
        assert metrics.milestone_success_rate == 0.8
        assert metrics.training_step == 1

    def test_roundtrip(self):
        original = TrainingMetrics(
            milestone_success_rate=0.8,
            hindsight_utilization=0.3,
            avg_reward=0.75,
            policy_loss=0.25,
            training_step=42,
            num_trajectories=10,
            num_hindsight=3,
        )
        d = original.to_dict()
        restored = TrainingMetrics.from_dict(d)
        assert restored.milestone_success_rate == original.milestone_success_rate
        assert restored.training_step == original.training_step


class TestTrainingLoggerInit:
    """Tests for TrainingLogger initialization."""

    def test_init_defaults(self):
        logger = TrainingLogger()
        assert logger.total_steps == 0
        assert logger.total_episodes == 0

    def test_init_with_name(self):
        logger = TrainingLogger(name="test")
        assert logger._name == "test"

    def test_init_with_log_dir(self, tmp_path):
        logger = TrainingLogger(log_dir=tmp_path, file_logging=True)
        assert logger._log_dir is not None


class TestTrainingLoggerLogging:
    """Tests for TrainingLogger logging methods."""

    def test_log_step(self):
        logger = TrainingLogger(console_logging=False)
        metrics = TrainingMetrics(0.8, 0.3, 0.75, 0.25, training_step=1)
        logger.log_step(metrics)
        assert logger.total_steps == 1
        assert len(logger.step_metrics) == 1

    def test_log_episode(self):
        logger = TrainingLogger(console_logging=False)
        metrics = TrainingMetrics(0.8, 0.3, 0.75, 0.25, training_step=1)
        logger.log_episode(metrics)
        assert logger.total_episodes == 1
        assert len(logger.episode_metrics) == 1

    def test_log_step_invalid_type(self):
        logger = TrainingLogger(console_logging=False)
        with pytest.raises(TypeError, match="Expected TrainingMetrics"):
            logger.log_step("not metrics")

    def test_log_episode_invalid_type(self):
        logger = TrainingLogger(console_logging=False)
        with pytest.raises(TypeError, match="Expected TrainingMetrics"):
            logger.log_episode("not metrics")


class TestTrainingLoggerStatistics:
    """Tests for TrainingLogger statistics methods."""

    def test_get_summary_empty(self):
        logger = TrainingLogger()
        summary = logger.get_summary()
        assert summary["total_steps"] == 0
        assert summary["total_episodes"] == 0

    def test_get_summary_with_data(self):
        logger = TrainingLogger(console_logging=False)
        for i in range(5):
            metrics = TrainingMetrics(0.8, 0.3, 0.75, 0.25, training_step=i)
            logger.log_step(metrics)
        summary = logger.get_summary()
        assert summary["total_steps"] == 5
        assert summary["avg_milestone_success_rate"] == 0.8

    def test_get_recent_metrics(self):
        logger = TrainingLogger(console_logging=False)
        for i in range(10):
            metrics = TrainingMetrics(0.8, 0.3, 0.75, 0.25, training_step=i)
            logger.log_step(metrics)
        recent = logger.get_recent_metrics(n=5)
        assert len(recent) == 5
        assert recent[0].training_step == 5

    def test_get_recent_metrics_as_dicts(self):
        logger = TrainingLogger(console_logging=False)
        for i in range(3):
            metrics = TrainingMetrics(0.8, 0.3, 0.75, 0.25, training_step=i)
            logger.log_step(metrics)
        recent = logger.get_recent_metrics(n=3, as_dicts=True)
        assert all(isinstance(d, dict) for d in recent)


class TestTrainingLoggerSaveLoad:
    """Tests for TrainingLogger save and load."""

    def test_save_to_file(self, tmp_path):
        logger = TrainingLogger(console_logging=False, file_logging=False)
        for i in range(3):
            metrics = TrainingMetrics(0.8, 0.3, 0.75, 0.25, training_step=i)
            logger.log_step(metrics)
        save_path = tmp_path / "metrics.json"
        logger.save(save_path)
        assert save_path.exists()

    def test_load_from_file(self, tmp_path):
        # Save
        logger1 = TrainingLogger(console_logging=False, file_logging=False, name="test")
        for i in range(3):
            metrics = TrainingMetrics(0.8, 0.3, 0.75, 0.25, training_step=i)
            logger1.log_step(metrics)
        save_path = tmp_path / "metrics.json"
        logger1.save(save_path)
        # Load
        logger2 = TrainingLogger.load(save_path)
        assert logger2.total_steps == 3

    def test_reset(self):
        logger = TrainingLogger(console_logging=False)
        for i in range(3):
            metrics = TrainingMetrics(0.8, 0.3, 0.75, 0.25, training_step=i)
            logger.log_step(metrics)
        logger.reset()
        assert logger.total_steps == 0
        assert len(logger.step_metrics) == 0


# ---------------------------------------------------------------------------
# End-to-End Integration Tests
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_full_pipeline(self, hindsight_buffer, hindsight_trainer, mock_planner, mock_critic, tmp_path):
        """Test complete env → agent → buffer → trainer pipeline."""
        # 1. Create env
        def task_fn():
            return {"observation": "state", "task": "Complete tasks A, B, C"}

        env = HeRoSEnv(
            task_fn=task_fn,
            planner=mock_planner,
            critic=mock_critic,
            hindsight_buffer=hindsight_buffer,
        )

        # 2. Create agent
        agent = HeRoSAgent(
            planner=mock_planner,
            critic=mock_critic,
            hindsight_buffer=hindsight_buffer,
            trainer=hindsight_trainer,
            env=env,
        )

        # 3. Create trainer
        ppo_trainer = PPOTrainer(agent=agent, hindsight_ratio=0.3, gamma=0.99, seed=42)

        # 4. Run episode
        env.reset()
        obs = env._build_observation()

        for _ in range(5):
            act_result = agent.act(obs)
            if act_result.is_episode_done:
                break
            obs, reward, done, info = env.step(act_result.action)

        # 5. Create trajectory from episode
        traj = env.create_trajectory_from_episode()
        assert isinstance(traj, HindsightTrajectory)

        # 6. Add to buffer if failed
        if traj.is_failed:
            env.add_to_buffer(traj)

        # 7. Sample from buffer
        batch = hindsight_buffer.sample(batch_size=32)
        assert isinstance(batch, list)

        # 8. Train with PPO
        if batch:
            metrics = ppo_trainer.train_step(batch)
            assert isinstance(metrics, TrainingMetrics)
            assert metrics.num_trajectories >= 0

        # 9. Log metrics
        logger = TrainingLogger(console_logging=False, file_logging=False)
        metrics = TrainingMetrics(
            milestone_success_rate=0.5,
            hindsight_utilization=0.3,
            avg_reward=0.6,
            policy_loss=0.4,
            training_step=1,
            num_trajectories=len(batch) if batch else 0,
        )
        logger.log_step(metrics)
        logger.log_episode(metrics)

        # 10. Verify logger state
        assert logger.total_steps == 1
        assert logger.total_episodes == 1
        summary = logger.get_summary()
        assert summary["total_steps"] == 1

    def test_agent_update_with_hindsight_buffer(self, hindsight_buffer, hindsight_trainer, mock_planner, mock_critic, heros_env):
        """Test agent.update() with trajectories in hindsight buffer."""
        # Add some failed trajectories
        for i in range(5):
            traj = HindsightTrajectory(
                task=f"task_{i}",
                milestones=[Milestone(id="m1", description="x", rubric="y", expected_output="")],
                exec_traces=[],
                verdicts=[Verdict.FAIL],
                unmet_rubrics=[f"rubric_{i}"],
                is_hindsight_enhanced=True,
            )
            hindsight_buffer.add(traj)

        agent = HeRoSAgent(
            planner=mock_planner,
            critic=mock_critic,
            hindsight_buffer=hindsight_buffer,
            trainer=hindsight_trainer,
            env=heros_env,
        )

        result = agent.update()
        assert isinstance(result, UpdateResult)
        # Buffer has 5 trajectories, sample(32) returns at most 5
        assert result.num_samples <= 5
        # All 5 added were hindsight_enhanced so ratio is 1.0
        assert result.hindsight_ratio == 1.0

    def test_ppo_trainer_with_empty_buffer(self, heros_agent):
        """Test PPO trainer handles empty buffer gracefully."""
        ppo_trainer = PPOTrainer(agent=heros_agent, hindsight_ratio=0.3)
        metrics = ppo_trainer.train_step([])
        assert metrics.num_trajectories == 0
        assert metrics.hindsight_utilization == 0.0


class TestMilestoneStatus:
    """Tests for MilestoneStatus dataclass."""

    def test_mark_active(self):
        ms = MilestoneStatus(milestone=Milestone(id="m1", description="x", rubric="y"))
        assert ms.status == "pending"
        ms.mark_active()
        assert ms.status == "active"

    def test_mark_passed(self):
        ms = MilestoneStatus(milestone=Milestone(id="m1", description="x", rubric="y"))
        result = CriticResult(Verdict.PASS, "good", 0.9, 1.0)
        ms.mark_passed(result)
        assert ms.status == "passed"
        assert ms.critic_result is result

    def test_mark_failed(self):
        ms = MilestoneStatus(milestone=Milestone(id="m1", description="x", rubric="y"))
        result = CriticResult(Verdict.FAIL, "bad", 0.9, 0.0)
        ms.mark_failed(result)
        assert ms.status == "failed"
        assert ms.critic_result is result


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
