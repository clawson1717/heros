"""HeRoS full agent: planner + actor + critic + hindsight buffer.

Integrates all HeRoS components into a single agent that can plan,
act, evaluate, and learn from both standard and hindsight experience.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from heros.planner import SubgoalPlanner, SubgoalPlan, Milestone
from heros.critic import MilestoneCritic, CriticResult, Verdict
from heros.buffer import HindsightBuffer, HindsightTrajectory
from heros.trainer import HindsightTrainer, UpdateResult
from heros.env import HeRoSEnv

import logging

logger = logging.getLogger(__name__)


@dataclass
class ActResult:
    """Result of an agent act() call.

    Attributes
    ----------
    action : str
        The action (or action string) selected by the agent.
    milestone_id : str
        ID of the milestone the agent is currently working on.
    milestone_description : str
        Description of the current milestone.
    critic_result : CriticResult, optional
        The critic's evaluation of the previous action, if available.
    is_milestone_complete : bool
        Whether the last milestone has been completed (passed or failed).
    is_episode_done : bool
        Whether the episode is complete.
    """

    action: str
    milestone_id: str
    milestone_description: str
    critic_result: Optional[CriticResult] = None
    is_milestone_complete: bool = False
    is_episode_done: bool = False


class HeRoSAgent:
    """Full HeRoS agent integrating planner, actor, critic, and hindsight buffer.

    The agent operates in a loop:
    1. Plan: Decompose task into milestones using the planner
    2. Act: Select actions targeting the current milestone
    3. Evaluate: Use the critic to assess milestone completion
    4. Store: Add failed trajectories to the hindsight buffer
    5. Learn: Periodically update policy via the trainer

    Parameters
    ----------
    planner : SubgoalPlanner
        The subgoal planner for decomposing tasks.
    critic : MilestoneCritic
        The milestone critic for evaluating actions.
    hindsight_buffer : HindsightBuffer
        The hindsight experience buffer.
    trainer : HindsightTrainer
        The trainer for policy updates.
    env : HeRoSEnv, optional
        The environment wrapper. If provided, the agent can run
        full episodes. If None, act() must be called manually.

    Attributes
    ----------
    update_count : int
        Number of policy updates performed.
    episode_count : int
        Number of episodes completed.
    """

    def __init__(
        self,
        planner: SubgoalPlanner,
        critic: MilestoneCritic,
        hindsight_buffer: HindsightBuffer,
        trainer: HindsightTrainer,
        env: Optional[HeRoSEnv] = None,
    ) -> None:
        if not isinstance(planner, SubgoalPlanner):
            raise TypeError(
                f"planner must be a SubgoalPlanner, got {type(planner).__name__}"
            )
        self._planner = planner

        if not isinstance(critic, MilestoneCritic):
            raise TypeError(
                f"critic must be a MilestoneCritic, got {type(critic).__name__}"
            )
        self._critic = critic

        if not isinstance(hindsight_buffer, HindsightBuffer):
            raise TypeError(
                f"hindsight_buffer must be a HindsightBuffer, "
                f"got {type(hindsight_buffer).__name__}"
            )
        self._hindsight_buffer = hindsight_buffer

        if not isinstance(trainer, HindsightTrainer):
            raise TypeError(
                f"trainer must be a HindsightTrainer, got {type(trainer).__name__}"
            )
        self._trainer = trainer

        if env is not None and not isinstance(env, HeRoSEnv):
            raise TypeError(f"env must be a HeRoSEnv, got {type(env).__name__}")
        self._env = env

        # State tracking
        self._update_count: int = 0
        self._episode_count: int = 0
        self._last_critic_result: Optional[CriticResult] = None

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def planner(self) -> SubgoalPlanner:
        """The subgoal planner."""
        return self._planner

    @property
    def critic(self) -> MilestoneCritic:
        """The milestone critic."""
        return self._critic

    @property
    def hindsight_buffer(self) -> HindsightBuffer:
        """The hindsight experience buffer."""
        return self._hindsight_buffer

    @property
    def trainer(self) -> HindsightTrainer:
        """The policy trainer."""
        return self._trainer

    @property
    def env(self) -> Optional[HeRoSEnv]:
        """The environment wrapper, if any."""
        return self._env

    @property
    def update_count(self) -> int:
        """Number of policy updates performed."""
        return self._update_count

    @property
    def episode_count(self) -> int:
        """Number of episodes completed."""
        return self._episode_count

    # -------------------------------------------------------------------------
    # Acting
    # -------------------------------------------------------------------------

    def act(self, obs: Dict[str, Any]) -> ActResult:
        """Select an action based on the current observation.

        Uses the planner and critic to select actions targeting the
        current milestone. The action is a string representation
        of the agent's planned response.

        In rule-based mode (default), the agent constructs a simple
        action string based on the active milestone. In LLM mode,
        this could invoke the planner to generate actions.

        Parameters
        ----------
        obs : Dict[str, Any]
            The milestone-augmented observation from HeRoSEnv. Should
            contain at minimum:
            - ``active_milestone``: dict with milestone info
            - ``task``: task description
            - ``milestones``: list of all milestones

        Returns
        -------
        ActResult
            The result of the act() call, including the selected action
            and milestone information.
        """
        active_milestone = obs.get("active_milestone")

        if not active_milestone:
            return ActResult(
                action="",
                milestone_id="",
                milestone_description="",
                is_episode_done=False,
            )

        milestone_id = active_milestone.get("id", "")
        milestone_desc = active_milestone.get("description", "")
        rubric = active_milestone.get("rubric", "")
        task = obs.get("task", "")

        # Generate action based on milestone
        # Rule-based fallback: simple action construction
        action = self._generate_action(task, milestone_desc, rubric)

        # Evaluate the action using critic if we have execution trace
        # (For rule-based, we evaluate the planned action directly)
        critic_result = None
        if action:
            critic_result = self._critic.review(
                milestone_description=milestone_desc,
                rubric=rubric,
                execution_trace=action,
            )
            self._last_critic_result = critic_result

        # Determine if milestone is complete
        is_complete = (
            critic_result is not None
            and critic_result.verdict in (Verdict.PASS, Verdict.FAIL)
        )

        # Get episode done status
        is_done = obs.get("step_count", 0) >= 100  # Max steps default

        return ActResult(
            action=action,
            milestone_id=milestone_id,
            milestone_description=milestone_desc,
            critic_result=critic_result,
            is_milestone_complete=is_complete,
            is_episode_done=is_done,
        )

    def _generate_action(
        self,
        task: str,
        milestone_desc: str,
        rubric: str,
    ) -> str:
        """Generate an action for the given milestone.

        Rule-based fallback action generation. Constructs a simple
        action string that addresses the milestone description.

        Parameters
        ----------
        task : str
            The overall task description.
        milestone_desc : str
            The current milestone description.
        rubric : str
            The milestone's pass/fail rubric.

        Returns
        -------
        str
            A generated action string.
        """
        # Simple rule-based action construction
        action_parts = [
            f"Task: {task[:100]}",
            f"Milestone: {milestone_desc[:200]}",
            f"Rubric: {rubric[:200]}",
            "Action: Executing milestone step.",
        ]
        return " | ".join(action_parts)

    # -------------------------------------------------------------------------
    # Learning
    # -------------------------------------------------------------------------

    def update(self) -> UpdateResult:
        """Perform a policy update from the hindsight buffer.

        Samples a batch of trajectories from the hindsight buffer
        (respecting the configured hindsight ratio) and performs
        a policy update via the trainer.

        Returns
        -------
        UpdateResult
            The result of the policy update.
        """
        # Sample from buffer
        batch_size = 32
        trajectories = self._hindsight_buffer.sample(batch_size)

        if not trajectories:
            # Return a dummy result if buffer is empty
            return UpdateResult(
                loss=0.0,
                num_samples=0,
                hindsight_ratio=self._hindsight_buffer.hindsight_ratio,
                is_simulation=True,
                details={"message": "Buffer empty, no update performed"},
            )

        # Perform update
        result = self._trainer.update_policy(trajectories)
        self._update_count += 1

        return result

    def learn_from_episode(
        self,
        batch_size: int = 32,
        add_trajectory_to_buffer: bool = True,
    ) -> UpdateResult:
        """Run an episode and learn from the hindsight experience.

        If an environment is configured, runs a complete episode,
        then samples from the buffer and performs a policy update.

        Parameters
        ----------
        batch_size : int, optional
            Number of trajectories to sample for the update.
            Defaults to 32.
        add_trajectory_to_buffer : bool, optional
            Whether to add the completed trajectory to the buffer.
            Defaults to True.

        Returns
        -------
        UpdateResult
            The result of the policy update.
        """
        if self._env is None:
            raise RuntimeError("No environment configured for learning.")

        # Run episode
        obs = self._env.reset()
        done = False
        trajectories_collected: List[HindsightTrajectory] = []

        while not done:
            act_result = self.act(obs)
            if act_result.is_episode_done:
                break

            # Step in environment
            next_obs, reward, done, info = self._env.step(act_result.action)

            # Check if milestone complete and we should create trajectory
            if act_result.is_milestone_complete:
                traj = self._env.create_trajectory_from_episode()
                if traj.is_failed and add_trajectory_to_buffer:
                    self._hindsight_buffer.add(traj)
                trajectories_collected.append(traj)

            obs = next_obs

        # Add final trajectory
        if add_trajectory_to_buffer:
            final_traj = self._env.create_trajectory_from_episode()
            if final_traj.is_failed:
                self._hindsight_buffer.add(final_traj)

        self._episode_count += 1

        # Perform update from buffer
        return self.update()

    # -------------------------------------------------------------------------
    # Episode management
    # -------------------------------------------------------------------------

    def run_episode(self) -> Dict[str, Any]:
        """Run a single episode and return summary statistics.

        Requires an environment to be configured.

        Returns
        -------
        Dict[str, Any]
            Episode summary with:
            - ``episode_reward``: total reward accumulated
            - ``step_count``: number of steps taken
            - ``milestones_passed``: count of passed milestones
            - ``milestones_failed``: count of failed milestones
            - ``milestone_status``: full milestone status dict
        """
        if self._env is None:
            raise RuntimeError("No environment configured. Set env= in constructor.")

        obs = self._env.reset()
        done = False
        step_count = 0
        total_reward = 0.0

        while not done:
            act_result = self.act(obs)
            if act_result.is_episode_done:
                break

            next_obs, reward, done, info = self._env.step(act_result.action)
            total_reward += reward
            step_count += 1
            obs = next_obs

        self._episode_count += 1

        milestone_status = self._env.get_milestone_status()

        return {
            "episode_reward": total_reward,
            "step_count": step_count,
            "milestones_passed": milestone_status["passed"],
            "milestones_failed": milestone_status["failed"],
            "milestone_status": milestone_status,
        }

    # -------------------------------------------------------------------------
    # Representation
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"HeRoSAgent("
            f"update_count={self._update_count}, "
            f"episode_count={self._episode_count}, "
            f"buffer_size={len(self._hindsight_buffer)})"
        )
