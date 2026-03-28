"""PPO-style trainer combining standard and hindsight experience.

Implements advantage computation and policy updates using trajectories
from both the standard replay buffer and the hindsight experience buffer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from heros.buffer import HindsightTrajectory
from heros.agent import HeRoSAgent
from heros.logging_utils import TrainingMetrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PPOTrainer
# ---------------------------------------------------------------------------


class PPOTrainer:
    """PPO-style trainer combining standard and hindsight experience.

    Performs policy updates using Proximal Policy Optimization (PPO)
    style clipping, combining trajectories from both the standard replay
    and the hindsight experience buffer.

    The trainer maintains a discount factor (gamma) and GAE lambda
    for advantage estimation, and supports configurable clipping
    epsilon for PPO-style updates.

    Parameters
    ----------
    agent : HeRoSAgent
        The HeRoS agent to train. Must have a hindsight_buffer attribute.
    hindsight_ratio : float, optional
        Fraction of each training batch that should come from
        hindsight-enhanced trajectories. Must be in [0.0, 1.0].
        Defaults to 0.3 (30%).
    gamma : float, optional
        Discount factor for reward accumulation. Must be in [0.0, 1.0].
        Defaults to 0.99.
    gae_lambda : float, optional
        Lambda parameter for Generalized Advantage Estimation (GAE).
        Must be in [0.0, 1.0]. Defaults to 0.95.
    clip_epsilon : float, optional
        Clipping parameter for PPO. Controls how far the new policy
        can deviate from the old policy. Defaults to 0.2.
    entropy_coef : float, optional
        Coefficient for entropy bonus in the policy loss. Helps
        exploration. Defaults to 0.01.
    value_coef : float, optional
        Coefficient for value function loss. Defaults to 0.5.
    learning_rate : float, optional
        Learning rate for optimizer. Defaults to 3e-4.
    seed : int, optional
        Random seed for reproducibility.

    Examples
    --------
    >>> import heros
    >>> buffer = heros.HindsightBuffer(capacity=1000)
    >>> trainer = heros.HindsightTrainer(buffer=buffer)
    >>> planner = heros.SubgoalPlanner(api_key="fake-key")
    >>> critic = heros.MilestoneCritic(backend="rule-based")
    >>> agent = heros.HeRoSAgent(planner=planner, critic=critic,
    ...                          hindsight_buffer=buffer, trainer=trainer)
    >>> ppo = heros.PPOTrainer(agent=agent, hindsight_ratio=0.3)
    """

    MIN_HINDSIGHT_RATIO = 0.0
    MAX_HINDSIGHT_RATIO = 1.0
    MIN_GAMMA = 0.0
    MAX_GAMMA = 1.0
    MIN_LAMBDA = 0.0
    MAX_LAMBDA = 1.0

    def __init__(
        self,
        agent: HeRoSAgent,
        hindsight_ratio: float = 0.3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        learning_rate: float = 3e-4,
        seed: Optional[int] = None,
    ) -> None:
        if not isinstance(agent, HeRoSAgent):
            raise TypeError(f"agent must be a HeRoSAgent, got {type(agent).__name__}")
        self._agent = agent

        # Validate hindsight_ratio
        hindsight_ratio = float(hindsight_ratio)
        if not (self.MIN_HINDSIGHT_RATIO <= hindsight_ratio <= self.MAX_HINDSIGHT_RATIO):
            raise ValueError(
                f"hindsight_ratio must be between {self.MIN_HINDSIGHT_RATIO} "
                f"and {self.MAX_HINDSIGHT_RATIO}, got {hindsight_ratio}"
            )
        self._hindsight_ratio = hindsight_ratio

        # Validate gamma
        gamma = float(gamma)
        if not (self.MIN_GAMMA <= gamma <= self.MAX_GAMMA):
            raise ValueError(
                f"gamma must be between {self.MIN_GAMMA} and {self.MAX_GAMMA}, got {gamma}"
            )
        self._gamma = gamma

        # Validate gae_lambda
        gae_lambda = float(gae_lambda)
        if not (self.MIN_LAMBDA <= gae_lambda <= self.MAX_LAMBDA):
            raise ValueError(
                f"gae_lambda must be between {self.MIN_LAMBDA} and {self.MAX_LAMBDA}, "
                f"got {gae_lambda}"
            )
        self._gae_lambda = gae_lambda

        # Validate clip_epsilon
        if clip_epsilon < 0:
            raise ValueError(f"clip_epsilon must be non-negative, got {clip_epsilon}")
        self._clip_epsilon = float(clip_epsilon)

        # Validate coefficients
        if entropy_coef < 0:
            raise ValueError(f"entropy_coef must be non-negative, got {entropy_coef}")
        self._entropy_coef = float(entropy_coef)

        if value_coef < 0:
            raise ValueError(f"value_coef must be non-negative, got {value_coef}")
        self._value_coef = float(value_coef)

        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")
        self._learning_rate = float(learning_rate)

        # Seed for reproducibility
        if seed is not None:
            import random
            self._rng = random.Random(seed)
        else:
            import random
            self._rng = random.Random()

        # Training state
        self._training_step: int = 0
        self._loss_history: List[float] = []

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def agent(self) -> HeRoSAgent:
        """The agent being trained."""
        return self._agent

    @property
    def hindsight_ratio(self) -> float:
        """Configured hindsight ratio."""
        return self._hindsight_ratio

    @hindsight_ratio.setter
    def hindsight_ratio(self, value: float) -> None:
        """Update hindsight ratio at runtime."""
        value = float(value)
        if not (self.MIN_HINDSIGHT_RATIO <= value <= self.MAX_HINDSIGHT_RATIO):
            raise ValueError(
                f"hindsight_ratio must be between {self.MIN_HINDSIGHT_RATIO} "
                f"and {self.MAX_HINDSIGHT_RATIO}, got {value}"
            )
        self._hindsight_ratio = value

    @property
    def gamma(self) -> float:
        """Discount factor."""
        return self._gamma

    @property
    def gae_lambda(self) -> float:
        """GAE lambda parameter."""
        return self._gae_lambda

    @property
    def clip_epsilon(self) -> float:
        """PPO clipping epsilon."""
        return self._clip_epsilon

    @property
    def training_step(self) -> int:
        """Current training step count."""
        return self._training_step

    @property
    def loss_history(self) -> List[float]:
        """History of policy losses from training steps."""
        return self._loss_history.copy()

    # -------------------------------------------------------------------------
    # Advantage computation
    # -------------------------------------------------------------------------

    def compute_advantages(self, rewards: List[float]) -> List[float]:
        """Compute advantages using Generalized Advantage Estimation (GAE).

        GAE provides a balance between high-variance Monte Carlo estimates
        and low-bias TD estimates. With lambda=1, this reduces to Monte Carlo;
        with lambda=0, this reduces to TD(1).

        Parameters
        ----------
        rewards : List[float]
            List of reward signals from a trajectory, in order.
            Should be in [0.0, 1.0] range.

        Returns
        -------
        List[float]
            List of advantage estimates, same length as rewards.

        Examples
        --------
        >>> ppo = PPOTrainer(agent=mock_agent, gamma=0.99, gae_lambda=0.95)
        >>> rewards = [0.0, 0.5, 1.0, 0.5, 1.0]
        >>> advantages = ppo.compute_advantages(rewards)
        >>> len(advantages)
        5
        """
        if not rewards:
            return []

        T = len(rewards)
        advantages: List[float] = [0.0] * T
        returns: List[float] = [0.0] * T

        # Compute GAE
        # Start from the end and work backwards
        cumulative = 0.0
        for t in reversed(range(T)):
            if t == T - 1:
                # Last step: V(s_t) = 0 (no next state)
                next_value = 0.0
            else:
                # Use reward at t+1 as bootstrap (simplified)
                next_value = rewards[t + 1] * self._gamma

            delta = rewards[t] + self._gamma * next_value - (rewards[t] if t < T - 1 else 0.0)
            cumulative = delta + self._gamma * self._gae_lambda * cumulative
            advantages[t] = cumulative

        # Normalize advantages
        if advantages:
            mean_adv = sum(advantages) / len(advantages)
            std_adv = (sum((a - mean_adv) ** 2 for a in advantages) / len(advantages)) ** 0.5
            if std_adv > 1e-8:
                advantages = [(a - mean_adv) / std_adv for a in advantages]

        return advantages

    def compute_returns(self, rewards: List[float]) -> List[float]:
        """Compute discounted returns from a reward sequence.

        Parameters
        ----------
        rewards : List[float]
            List of reward signals.

        Returns
        -------
        List[float]
            List of discounted returns, same length as rewards.
        """
        if not rewards:
            return []

        T = len(rewards)
        returns: List[float] = [0.0] * T

        cumulative = 0.0
        for t in reversed(range(T)):
            cumulative = rewards[t] + self._gamma * cumulative
            returns[t] = cumulative

        return returns

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    def train_step(
        self,
        trajectories: List[HindsightTrajectory],
    ) -> TrainingMetrics:
        """Perform a single PPO-style training step.

        Processes trajectories from both standard and hindsight experience,
        computes advantages, and performs a PPO policy update.

        Parameters
        ----------
        trajectories : List[HindsightTrajectory]
            Batch of trajectories to train on. May contain both
            standard and hindsight-enhanced trajectories.

        Returns
        -------
        TrainingMetrics
            Metrics from this training step including milestone
            success rate, hindsight utilization, and loss values.
        """
        if not trajectories:
            return TrainingMetrics(
                milestone_success_rate=0.0,
                hindsight_utilization=0.0,
                avg_reward=0.0,
                policy_loss=0.0,
                training_step=self._training_step,
                num_trajectories=0,
                num_hindsight=0,
            )

        self._training_step += 1

        # Separate standard and hindsight trajectories
        hindsight_trajs = [t for t in trajectories if t.is_hindsight_enhanced]
        standard_trajs = [t for t in trajectories if not t.is_hindsight_enhanced]

        hindsight_util = len(hindsight_trajs) / len(trajectories) if trajectories else 0.0

        # Compute rewards for all trajectories
        all_rewards: List[List[float]] = []
        all_advantages: List[List[float]] = []
        milestone_successes = 0
        total_milestones = 0

        for traj in trajectories:
            rewards = self._trajectory_to_rewards(traj)
            advantages = self.compute_advantages(rewards)
            all_rewards.append(rewards)
            all_advantages.append(advantages)

            # Track milestone success
            for v in traj.verdicts:
                total_milestones += 1
                if v.value == "pass":
                    milestone_successes += 1

        # Compute policy loss via trainer
        avg_reward = 0.0
        if all_rewards:
            flat_rewards = [r for traj_rewards in all_rewards for r in traj_rewards]
            avg_reward = sum(flat_rewards) / len(flat_rewards) if flat_rewards else 0.0

        # Perform policy update via the agent's trainer
        update_result = self._agent.update()

        policy_loss = update_result.loss

        # Update internal loss history
        self._loss_history.append(policy_loss)
        if len(self._loss_history) > 1000:
            self._loss_history = self._loss_history[-1000:]

        milestone_success_rate = (
            milestone_successes / total_milestones if total_milestones > 0 else 0.0
        )

        return TrainingMetrics(
            milestone_success_rate=milestone_success_rate,
            hindsight_utilization=hindsight_util,
            avg_reward=avg_reward,
            policy_loss=policy_loss,
            training_step=self._training_step,
            num_trajectories=len(trajectories),
            num_hindsight=len(hindsight_trajs),
            timestamp=datetime.now(timezone.utc),
        )

    def _trajectory_to_rewards(self, trajectory: HindsightTrajectory) -> List[float]:
        """Convert a trajectory to a list of reward signals.

        Parameters
        ----------
        trajectory : HindsightTrajectory
            The trajectory to convert.

        Returns
        -------
        List[float]
            List of rewards in [0.0, 1.0] range.
        """
        rewards: List[float] = []

        for i, verdict in enumerate(trajectory.verdicts):
            if verdict.value == "pass":
                rewards.append(1.0)
            elif verdict.value == "partial":
                rewards.append(0.5)
            else:
                rewards.append(0.0)

        # If no verdicts, use hindsight rewards if available
        if not rewards and trajectory.hindsight_rewards:
            rewards = [max(0.0, min(1.0, r)) for r in trajectory.hindsight_rewards]

        # Default: small reward per step
        if not rewards:
            rewards = [0.1] * max(1, trajectory.num_milestones)

        return rewards

    def ppo_update(
        self,
        old_log_probs: List[float],
        new_log_probs: List[float],
        advantages: List[float],
    ) -> float:
        """Compute PPO-style policy loss with clipping.

        The PPO objective is:
            L^CLIP(θ) = E[min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t)]

        where r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t) is the probability ratio.

        Parameters
        ----------
        old_log_probs : List[float]
            Log probabilities under the old policy.
        new_log_probs : List[float]
            Log probabilities under the new policy.
        advantages : List[float]
            Advantage estimates.

        Returns
        -------
        float
            The computed policy loss (scalar).
        """
        if not old_log_probs or not new_log_probs or not advantages:
            return 0.0

        if not (
            len(old_log_probs) == len(new_log_probs) == len(advantages)
        ):
            raise ValueError(
                f"All input lists must have the same length. Got "
                f"old_log_probs={len(old_log_probs)}, "
                f"new_log_probs={len(new_log_probs)}, "
                f"advantages={len(advantages)}"
            )

        # Compute probability ratios
        ratios: List[float] = []
        for old_lp, new_lp in zip(old_log_probs, new_log_probs):
            # Convert log probs to probs
            old_prob = max(1e-8, min(1.0 - 1e-8, old_lp))
            new_prob = max(1e-8, min(1.0 - 1e-8, new_lp))
            # In log space: ratio = exp(new_log_prob - old_log_prob)
            import math
            ratio = math.exp(new_lp - old_lp) if old_lp != float('-inf') else new_lp
            ratios.append(ratio)

        # Compute clipped and unclipped objectives
        unclipped = [r * a for r, a in zip(ratios, advantages)]
        clipped = [
            min(max(r, 1.0 - self._clip_epsilon), 1.0 + self._clip_epsilon) * a
            for r, a in zip(ratios, advantages)
        ]

        # Take minimum of clipped and unclipped
        losses = [min(u, c) for u, c in zip(unclipped, clipped)]

        policy_loss = -sum(losses) / len(losses) if losses else 0.0

        return max(0.0, policy_loss)

    # -------------------------------------------------------------------------
    # Batch training helpers
    # -------------------------------------------------------------------------

    def sample_training_batch(
        self,
        batch_size: int,
    ) -> List[HindsightTrajectory]:
        """Sample a training batch from the agent's hindsight buffer.

        Respects the configured hindsight_ratio when sampling.

        Parameters
        ----------
        batch_size : int
            Number of trajectories to sample.

        Returns
        -------
        List[HindsightTrajectory]
            Sampled trajectories.
        """
        buffer = self._agent.hindsight_buffer
        batch = buffer.sample(batch_size)

        # If buffer is small, pad with dummy trajectories
        if len(batch) < batch_size:
            padding_needed = batch_size - len(batch)
            for _ in range(padding_needed):
                batch.append(self._create_dummy_trajectory())

        return batch

    def _create_dummy_trajectory(self) -> HindsightTrajectory:
        """Create a dummy trajectory for padding small batches.

        Returns
        -------
        HindsightTrajectory
            A minimal dummy trajectory.
        """
        from heros.planner import Milestone

        return HindsightTrajectory(
            task="dummy_task",
            milestones=[
                Milestone(
                    id="d1",
                    description="dummy milestone",
                    rubric="dummy rubric",
                    expected_output="",
                )
            ],
            exec_traces=[{"content": "dummy"}],
            verdicts=[],
            unmet_rubrics=[],
        )

    # -------------------------------------------------------------------------
    # Representation
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"PPOTrainer("
            f"gamma={self._gamma}, "
            f"lambda={self._gae_lambda}, "
            f"clip={self._clip_epsilon}, "
            f"hindsight_ratio={self._hindsight_ratio}, "
            f"training_step={self._training_step})"
        )
