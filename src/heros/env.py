"""HeRoS environment wrapper with milestone tracking.

Wraps any task environment and integrates with the HeRoS planning,
critic, and hindsight systems to produce milestone-aware observations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from heros.planner import Milestone, SubgoalPlan, SubgoalPlanner
from heros.critic import CriticResult, MilestoneCritic, Verdict
from heros.buffer import HindsightBuffer, HindsightTrajectory

import logging

logger = logging.getLogger(__name__)


@dataclass
class MilestoneStatus:
    """Tracks the status of a single milestone within an episode.

    Attributes
    ----------
    milestone : Milestone
        The milestone being tracked.
    status : str
        One of "pending", "active", "passed", "failed".
    critic_result : CriticResult, optional
        The critic's verdict and feedback after evaluation.
    step_count : int
        Number of environment steps taken on this milestone.
    """

    milestone: Milestone
    status: str = "pending"
    critic_result: Optional[CriticResult] = None
    step_count: int = 0

    def mark_active(self) -> None:
        """Mark this milestone as the currently active one."""
        self.status = "active"

    def mark_passed(self, result: CriticResult) -> None:
        """Mark this milestone as passed with critic result."""
        self.status = "passed"
        self.critic_result = result

    def mark_failed(self, result: CriticResult) -> None:
        """Mark this milestone as failed with critic result."""
        self.status = "failed"
        self.critic_result = result


class HeRoSEnv:
    """Environment wrapper with milestone tracking for HeRoS RL loop.

    Wraps a task environment and integrates with the HeRoS planning,
    critic, and hindsight systems. Tracks milestone progress throughout
    each episode and produces milestone-augmented observations.

    Parameters
    ----------
    task_fn : Callable
        A callable that returns a reset observation dict when invoked
        with no arguments (i.e., ``task_fn()``). The returned dict should
        contain at minimum an ``"observation"`` key with the raw env
        observation. The task_fn may also accept keyword arguments for
        configuring the task.
    planner : SubgoalPlanner
        The subgoal planner for decomposing tasks into milestones.
    critic : MilestoneCritic
        The milestone critic for evaluating milestone completion.
    hindsight_buffer : HindsightBuffer
        The hindsight experience buffer for storing failed trajectories.

    Attributes
    ----------
    current_plan : SubgoalPlan, optional
        The current task plan with ordered milestones.
    milestone_statuses : List[MilestoneStatus]
        Status tracking for each milestone in the current plan.
    current_milestone_idx : int
        Index of the currently active milestone (0-indexed).
    episode_step_count : int
        Total environment steps taken in the current episode.
    episode_reward : float
        Cumulative reward in the current episode.

    Examples
    --------
    >>> import heros
    >>> def task_factory():
    ...     return {"observation": "initial_state", "task": "Example task"}
    >>> planner = heros.SubgoalPlanner(planning_depth=3, api_key="fake-key")
    >>> critic = heros.MilestoneCritic(backend="rule-based")
    >>> buffer = heros.HindsightBuffer(capacity=100)
    >>> env = heros.HeRoSEnv(task_fn=task_factory, planner=planner,
    ...                      critic=critic, hindsight_buffer=buffer)
    >>> obs = env.reset()
    >>> print(obs.keys())
    dict_keys(['observation', 'task', 'milestones', 'active_milestone'])
    """

    def __init__(
        self,
        task_fn: Callable,
        planner: SubgoalPlanner,
        critic: MilestoneCritic,
        hindsight_buffer: HindsightBuffer,
    ) -> None:
        if not callable(task_fn):
            raise TypeError(f"task_fn must be callable, got {type(task_fn).__name__}")
        self._task_fn = task_fn

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

        # State
        self._current_plan: Optional[SubgoalPlan] = None
        self._milestone_statuses: List[MilestoneStatus] = []
        self._current_milestone_idx: int = 0
        self._episode_step_count: int = 0
        self._episode_reward: float = 0.0

        # Episode tracking for buffer
        self._episode_exec_traces: List[Dict[str, Any]] = []
        self._episode_verdicts: List[Verdict] = []
        self._episode_unmet_rubrics: List[str] = []
        self._current_task: str = ""

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def current_plan(self) -> Optional[SubgoalPlan]:
        """Current task plan, or None if no plan has been set."""
        return self._current_plan

    @property
    def milestone_statuses(self) -> List[MilestoneStatus]:
        """Status tracking for milestones in the current plan."""
        return self._milestone_statuses

    @property
    def current_milestone_idx(self) -> int:
        """Index of the currently active milestone."""
        return self._current_milestone_idx

    @property
    def episode_step_count(self) -> int:
        """Total environment steps in the current episode."""
        return self._episode_step_count

    @property
    def episode_reward(self) -> float:
        """Cumulative reward in the current episode."""
        return self._episode_reward

    @property
    def hindsight_buffer(self) -> HindsightBuffer:
        """The hindsight experience buffer."""
        return self._hindsight_buffer

    # -------------------------------------------------------------------------
    # Core environment interface
    # -------------------------------------------------------------------------

    def reset(self) -> Dict[str, Any]:
        """Reset the environment and start a new episode.

        Calls the task factory to get a new task observation, then uses
        the planner to decompose the task into milestones.

        Returns
        -------
        Dict[str, Any]
            An observation dictionary containing:
            - ``observation``: raw env observation from task_fn
            - ``task``: task description string
            - ``milestones``: list of milestone dicts
            - ``active_milestone``: the first (active) milestone
            - ``plan``: the full SubgoalPlan object
        """
        # Get fresh observation from task
        raw_obs = self._task_fn()
        if not isinstance(raw_obs, dict):
            raise TypeError(
                f"task_fn() must return a dict, got {type(raw_obs).__name__}"
            )

        # Extract task description
        self._current_task = raw_obs.get("task", "")
        if not self._current_task:
            raise ValueError("task_fn() observation must have a 'task' key with non-empty value")

        # Plan milestones
        try:
            self._current_plan = self._planner.plan(self._current_task)
        except Exception as e:
            logger.warning("Planning failed, using empty plan: %s", e)
            # Graceful degradation: create a minimal single-milestone plan
            self._current_plan = SubgoalPlan(
                task=self._current_task,
                milestones=[
                    Milestone(
                        id="m1",
                        description="Complete the task",
                        rubric="Task is completed",
                        expected_output="",
                    )
                ],
            )

        # Initialize milestone statuses
        self._milestone_statuses = [
            MilestoneStatus(milestone=m, status="pending")
            for m in self._current_plan.milestones
        ]

        if self._milestone_statuses:
            self._milestone_statuses[0].mark_active()
            self._current_milestone_idx = 0
        else:
            self._current_milestone_idx = 0

        # Reset episode tracking
        self._episode_step_count = 0
        self._episode_reward = 0.0
        self._episode_exec_traces = []
        self._episode_verdicts = []
        self._episode_unmet_rubrics = []

        return self._build_observation()

    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Take a step in the environment.

        Executes the given action, evaluates the current milestone using
        the critic, advances milestones as needed, and returns the
        milestone-augmented observation.

        Parameters
        ----------
        action : str
            The action to execute. This is typically the agent's output
            or execution trace.

        Returns
        -------
        Tuple[Dict[str, Any], float, bool, Dict[str, Any]]
            A tuple of (observation, reward, done, info) where:
            - observation: milestone-augmented observation dict
            - reward: reward from the critic's milestone evaluation
            - done: True if the episode is finished (all milestones complete or failed)
            - info: additional information including milestone status
        """
        if not self._current_plan:
            raise RuntimeError(
                "Environment not reset. Call reset() before step()."
            )

        self._episode_step_count += 1

        # Record execution trace
        self._episode_exec_traces.append({"action": action, "step": self._episode_step_count})

        # Evaluate current milestone
        reward, done, info = self._evaluate_current_milestone(action)

        self._episode_reward += reward

        # Advance to next milestone if needed
        if not done and self._current_milestone_idx < len(self._milestone_statuses) - 1:
            if info.get("milestone_complete", False):
                self._current_milestone_idx += 1
                self._milestone_statuses[self._current_milestone_idx].mark_active()

        # Check if episode is done
        if self._current_milestone_idx >= len(self._milestone_statuses) - 1:
            # Check if last milestone passed or all milestones evaluated
            last_status = self._milestone_statuses[-1]
            if last_status.status in ("passed", "failed"):
                done = True

        info["milestone_index"] = self._current_milestone_idx
        info["total_milestones"] = len(self._milestone_statuses)

        return self._build_observation(), reward, done, info

    def _evaluate_current_milestone(self, action: str) -> Tuple[float, bool, Dict[str, Any]]:
        """Evaluate the current milestone using the critic.

        Parameters
        ----------
        action : str
            The execution trace or action string to evaluate.

        Returns
        -------
        Tuple[float, bool, Dict[str, Any]]
            (reward, done, info) from evaluating the current milestone.
        """
        info: Dict[str, Any] = {}

        if self._current_milestone_idx >= len(self._milestone_statuses):
            return 0.0, True, info

        current_status = self._milestone_statuses[self._current_milestone_idx]
        milestone = current_status.milestone

        # Use the action as the execution trace for evaluation
        exec_trace = action if action else ""

        # Call the critic
        critic_result = self._critic.review(
            milestone_description=milestone.description,
            rubric=milestone.rubric,
            execution_trace=exec_trace,
        )

        # Record verdict
        self._episode_verdicts.append(critic_result.verdict)

        # Handle unmet rubrics (failed milestones)
        if critic_result.verdict == Verdict.FAIL:
            current_status.mark_failed(critic_result)
            self._episode_unmet_rubrics.append(milestone.rubric)
            info["milestone_complete"] = True
            info["milestone_passed"] = False
            return critic_result.reward_signal, False, info

        elif critic_result.verdict == Verdict.PASS:
            current_status.mark_passed(critic_result)
            info["milestone_complete"] = True
            info["milestone_passed"] = True
            # Check if all milestones are done
            all_done = all(s.status in ("passed", "failed") for s in self._milestone_statuses)
            return critic_result.reward_signal, all_done, info

        else:  # PARTIAL
            # For partial, we stay on the same milestone
            info["milestone_complete"] = False
            info["milestone_passed"] = False
            return critic_result.reward_signal, False, info

    def _build_observation(self) -> Dict[str, Any]:
        """Build the milestone-augmented observation dict.

        Returns
        -------
        Dict[str, Any]
            Observation dict with milestone information.
        """
        active_milestone = None
        if 0 <= self._current_milestone_idx < len(self._milestone_statuses):
            ms = self._milestone_statuses[self._current_milestone_idx]
            active_milestone = {
                "id": ms.milestone.id,
                "description": ms.milestone.description,
                "rubric": ms.milestone.rubric,
                "expected_output": ms.milestone.expected_output,
                "status": ms.status,
            }

        return {
            "observation": self._current_task,
            "task": self._current_task,
            "milestones": [
                {
                    "id": s.milestone.id,
                    "description": s.milestone.description,
                    "rubric": s.milestone.rubric,
                    "status": s.status,
                }
                for s in self._milestone_statuses
            ],
            "active_milestone": active_milestone,
            "plan": self._current_plan,
            "step_count": self._episode_step_count,
            "episode_reward": self._episode_reward,
        }

    # -------------------------------------------------------------------------
    # Milestone tracking
    # -------------------------------------------------------------------------

    def get_milestone_status(self) -> Dict[str, Any]:
        """Get the current milestone status for the episode.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - ``current_index``: index of the active milestone
            - ``total``: total number of milestones
            - ``passed``: count of passed milestones
            - ``failed``: count of failed milestones
            - ``pending``: count of pending milestones
            - ``active``: description of the current active milestone
            - ``milestones``: list of per-milestone status dicts
        """
        if not self._milestone_statuses:
            return {
                "current_index": 0,
                "total": 0,
                "passed": 0,
                "failed": 0,
                "pending": 0,
                "active": None,
                "milestones": [],
            }

        passed = sum(1 for s in self._milestone_statuses if s.status == "passed")
        failed = sum(1 for s in self._milestone_statuses if s.status == "failed")
        pending = sum(1 for s in self._milestone_statuses if s.status == "pending")
        active_count = sum(1 for s in self._milestone_statuses if s.status == "active")

        active_status = None
        if 0 <= self._current_milestone_idx < len(self._milestone_statuses):
            s = self._milestone_statuses[self._current_milestone_idx]
            active_status = {
                "id": s.milestone.id,
                "description": s.milestone.description,
                "rubric": s.milestone.rubric,
                "status": s.status,
            }

        return {
            "current_index": self._current_milestone_idx,
            "total": len(self._milestone_statuses),
            "passed": passed,
            "failed": failed,
            "pending": pending,
            "active": active_status,
            "milestones": [
                {
                    "id": s.milestone.id,
                    "description": s.milestone.description,
                    "status": s.status,
                    "verdict": (
                        s.critic_result.verdict.value if s.critic_result else None
                    ),
                    "feedback": (
                        s.critic_result.feedback if s.critic_result else None
                    ),
                }
                for s in self._milestone_statuses
            ],
        }

    # -------------------------------------------------------------------------
    # Buffer integration
    # -------------------------------------------------------------------------

    def add_to_buffer(self, trajectory: HindsightTrajectory) -> None:
        """Add a completed trajectory to the hindsight buffer.

        Parameters
        ----------
        trajectory : HindsightTrajectory
            The trajectory to add. Should have been built from the
            current episode's execution traces and verdicts.
        """
        if not isinstance(trajectory, HindsightTrajectory):
            raise TypeError(
                f"Expected HindsightTrajectory, got {type(trajectory).__name__}"
            )
        self._hindsight_buffer.add(trajectory)

    def create_trajectory_from_episode(self) -> HindsightTrajectory:
        """Create a HindsightTrajectory from the current episode.

        Builds a trajectory object using the accumulated execution
        traces, verdicts, and unmet rubrics from the current episode.

        Returns
        -------
        HindsightTrajectory
            A trajectory representing the current episode's execution.
        """
        if not self._current_plan:
            raise RuntimeError(
                "No episode in progress. Call reset() first."
            )

        milestones = [s.milestone for s in self._milestone_statuses]

        return HindsightTrajectory(
            task=self._current_task,
            milestones=milestones,
            exec_traces=self._episode_exec_traces,
            verdicts=self._episode_verdicts,
            unmet_rubrics=self._episode_unmet_rubrics,
        )

    # -------------------------------------------------------------------------
    # Utility
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"HeRoSEnv("
            f"task={self._current_task[:30]!r}..., "
            f"milestones={len(self._milestone_statuses)}, "
            f"step={self._episode_step_count})"
        )
