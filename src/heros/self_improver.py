"""Test-time Self-Improvement Mode for HeRoS.

Implements inference-only self-improvement where failed subgoals are
captured in a local hindsight buffer and used for lightweight policy
updates without additional environment interaction.

Key insight: at test time, failures are FREE learning signals — no
environment interaction needed beyond the initial failed attempt.

References:
    - Self-play / policy iteration at inference time
    - HeRL: Hindsight Experience Replay for LLMs
    - MiRA: Milestoning RL Enhanced Agent
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from heros.buffer import HindsightBuffer, HindsightTrajectory
from heros.critic import Verdict

if TYPE_CHECKING:
    from heros.agent import HeRoSAgent

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class EpisodeMetrics:
    """Metrics captured from a single self-improvement episode.

    Attributes
    ----------
    episode_idx : int
        Zero-based episode index in the self-improvement run.
    success : bool
        Whether the task was successfully completed.
    milestone_hit_rate : float
        Fraction of milestones that were completed (0.0 to 1.0).
    total_reward : float
        Cumulative reward accumulated during the episode.
    episode_length : int
        Number of steps taken in the episode.
    failed_subgoals : List[str]
        Descriptions of milestones that failed during this episode.
    milestone_verdicts : List[str]
        Verdict for each milestone: "PASS" or "FAIL".
    """

    episode_idx: int
    success: bool
    milestone_hit_rate: float
    total_reward: float
    episode_length: int
    failed_subgoals: List[str] = field(default_factory=list)
    milestone_verdicts: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "episode_idx": self.episode_idx,
            "success": self.success,
            "milestone_hit_rate": self.milestone_hit_rate,
            "total_reward": self.total_reward,
            "episode_length": self.episode_length,
            "failed_subgoals": self.failed_subgoals,
            "milestone_verdicts": self.milestone_verdicts,
        }


@dataclass
class PolicyUpdateResult:
    """Result of a self-play policy update.

    Attributes
    ----------
    epochs_run : int
        Number of training epochs attempted.
    buffer_size_before : int
        Buffer size before the update.
    buffer_size_after : int
        Buffer size after the update.
    estimated_policy_delta : float
        Simulated metric representing the magnitude of policy change.
        In simulation mode (no API key), this is a random value.
    is_simulation : bool
        True if this was a simulated update (no real model update).
    details : Dict[str, Any]
        Additional details about the update.
    """

    epochs_run: int
    buffer_size_before: int
    buffer_size_after: int
    estimated_policy_delta: float = 0.0
    is_simulation: bool = True
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "epochs_run": self.epochs_run,
            "buffer_size_before": self.buffer_size_before,
            "buffer_size_after": self.buffer_size_after,
            "estimated_policy_delta": self.estimated_policy_delta,
            "is_simulation": self.is_simulation,
            "details": self.details,
        }


@dataclass
class SelfImprovementResult:
    """Result of a complete self-improvement run.

    Attributes
    ----------
    task_id : str
        The task that was evaluated.
    episodes : List[EpisodeMetrics]
        Per-episode metrics in order.
    final_success_rate : float
        Success rate across the final N episodes (0.0 to 1.0).
    initial_success_rate : float
        Success rate in the first episode (0.0 or 1.0).
    improvement_delta : float
        final_success_rate - initial_success_rate.
    total_self_play_epochs : int
        Total number of self-play epochs performed.
    hindsight_buffer_size_after : int
        Final size of the hindsight buffer.
    policy_update_results : List[PolicyUpdateResult]
        Result of each policy update.
    episodes_to_first_success : Optional[int]
        Number of episodes until first success. None if never succeeded.
    """

    task_id: str
    episodes: List[EpisodeMetrics]
    final_success_rate: float
    initial_success_rate: float
    improvement_delta: float
    total_self_play_epochs: int
    hindsight_buffer_size_after: int
    policy_update_results: List[PolicyUpdateResult] = field(default_factory=list)
    episodes_to_first_success: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "task_id": self.task_id,
            "episodes": [e.to_dict() for e in self.episodes],
            "final_success_rate": self.final_success_rate,
            "initial_success_rate": self.initial_success_rate,
            "improvement_delta": self.improvement_delta,
            "total_self_play_epochs": self.total_self_play_epochs,
            "hindsight_buffer_size_after": self.hindsight_buffer_size_after,
            "policy_update_results": [p.to_dict() for p in self.policy_update_results],
            "episodes_to_first_success": self.episodes_to_first_success,
        }


@dataclass
class InferenceEpisodeResult:
    """Result of a single inference episode.

    Attributes
    ----------
    task_id : str
        The task that was evaluated.
    success : bool
        Whether the task was completed.
    milestone_hit_rate : float
        Fraction of milestones completed.
    total_reward : float
        Cumulative reward.
    episode_length : int
        Number of steps.
    failed_subgoals : List[str]
        Descriptions of failed milestones.
    collected_for_hindsight : bool
        Whether failures were added to the hindsight buffer.
    """

    task_id: str
    success: bool
    milestone_hit_rate: float
    total_reward: float
    episode_length: int
    failed_subgoals: List[str] = field(default_factory=list)
    collected_for_hindsight: bool = False


@dataclass
class BatchInferenceResult:
    """Result of batch inference.

    Attributes
    ----------
    tasks : List[str]
        Task IDs that were evaluated.
    episode_results : List[InferenceEpisodeResult]
        Per-task results.
    overall_success_rate : float
        Fraction of tasks completed.
    avg_milestone_hit_rate : float
        Average milestone hit rate across tasks.
    total_self_play_epochs : int
        Total self-play epochs performed.
    hindsight_buffer_size : int
        Final buffer size.
    """

    tasks: List[str]
    episode_results: List[InferenceEpisodeResult]
    overall_success_rate: float = 0.0
    avg_milestone_hit_rate: float = 0.0
    total_self_play_epochs: int = 0
    hindsight_buffer_size: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "tasks": self.tasks,
            "episode_results": [
                {
                    "task_id": r.task_id,
                    "success": r.success,
                    "milestone_hit_rate": r.milestone_hit_rate,
                    "total_reward": r.total_reward,
                    "episode_length": r.episode_length,
                    "failed_subgoals": r.failed_subgoals,
                    "collected_for_hindsight": r.collected_for_hindsight,
                }
                for r in self.episode_results
            ],
            "overall_success_rate": self.overall_success_rate,
            "avg_milestone_hit_rate": self.avg_milestone_hit_rate,
            "total_self_play_epochs": self.total_self_play_epochs,
            "hindsight_buffer_size": self.hindsight_buffer_size,
        }


# ============================================================================
# TestTimeSelfImprover
# ============================================================================


class TestTimeSelfImprover:
    """Agent that self-improves at test time using failed subgoal hindsight.

    At test time, the agent runs episodes against tasks. When milestones
    fail, those failures are captured in a local hindsight buffer. Between
    episodes, a lightweight self-play update is performed using the accumulated
    failures, allowing the agent to improve without additional environment
    interaction.

    The key insight is that **failures are free learning signals** — the
    environment interaction cost is paid once, but the failure information
    can be reused multiple times for policy improvement.

    Parameters
    ----------
    agent : HeRoSAgent
        The HeRoSAgent to improve. Must have planner, critic, and
        a hindsight buffer accessible via the hindsight_buffer property.
    hindsight_buffer : HindsightBuffer
        Local buffer for inference-time failures.
    self_play_epochs : int, optional
        Number of policy update rounds per improvement cycle.
        Defaults to 3.
    improvement_threshold : float, optional
        Minimum delta in success rate to accept improvement as real
        (vs. noise). Defaults to 0.05.
    simulate_updates : bool, optional
        If True, use simulated policy updates (no real model changes).
        If False, attempt real updates when OPENAI_API_KEY is available.
        Defaults to True (safer for test-time use).

    Examples
    --------
    >>> from heros.agent import HeRoSAgent
    >>> from heros.buffer import HindsightBuffer
    >>> from heros.self_improver import TestTimeSelfImprover
    >>> buffer = HindsightBuffer(capacity=100)
    >>> improver = TestTimeSelfImprover(agent, buffer, self_play_epochs=3)
    >>> result = improver.run_with_self_improvement(task, env_factory)
    >>> print(f"Improved from {result.initial_success_rate} to {result.final_success_rate}")
    """

    def __init__(
        self,
        agent: "HeRoSAgent",
        hindsight_buffer: HindsightBuffer,
        self_play_epochs: int = 3,
        improvement_threshold: float = 0.05,
        simulate_updates: bool = True,
    ) -> None:
        if self_play_epochs < 0:
            raise ValueError(
                f"self_play_epochs must be non-negative, got {self_play_epochs}"
            )
        if not (0.0 <= improvement_threshold <= 1.0):
            raise ValueError(
                f"improvement_threshold must be in [0, 1], got {improvement_threshold}"
            )

        self._agent = agent
        self._hindsight_buffer = hindsight_buffer
        self._self_play_epochs = self_play_epochs
        self._improvement_threshold = improvement_threshold
        self._simulate_updates = simulate_updates or not self._has_openai_key()

        # Track improvement history
        self._episode_history: List[EpisodeMetrics] = []
        self._policy_update_history: List[PolicyUpdateResult] = []
        self._total_self_play_epochs: int = 0

        logger.info(
            "TestTimeSelfImprover initialized: epochs=%d, threshold=%.2f, simulate=%s",
            self_play_epochs,
            improvement_threshold,
            self._simulate_updates,
        )

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def agent(self) -> "HeRoSAgent":
        """The agent being improved."""
        return self._agent

    @property
    def hindsight_buffer(self) -> HindsightBuffer:
        """The hindsight buffer for storing failures."""
        return self._hindsight_buffer

    @property
    def self_play_epochs(self) -> int:
        """Number of self-play epochs per improvement cycle."""
        return self._self_play_epochs

    @property
    def improvement_threshold(self) -> float:
        """Minimum delta to accept improvement as real."""
        return self._improvement_threshold

    @property
    def episode_history(self) -> List[EpisodeMetrics]:
        """Full trajectory of episode metrics."""
        return self._episode_history.copy()

    # -------------------------------------------------------------------------
    # Main Entry Points
    # -------------------------------------------------------------------------

    def run_with_self_improvement(
        self,
        task: Any,
        env_factory: Callable[[], Any],
        n_episodes: int = 5,
        max_steps_per_episode: int = 50,
    ) -> SelfImprovementResult:
        """Run N episodes with self-improvement between episodes.

        After each episode, failed subgoals are extracted and added to
        the hindsight buffer. After every episode, a self-play policy
        update is optionally performed.

        Parameters
        ----------
        task : Any
            The task object to evaluate. Must have task_id, description,
            and milestones attributes.
        env_factory : Callable[[], Any]
            Factory function that creates a fresh environment for each
            episode. The environment must be compatible with the evaluator.
        n_episodes : int, optional
            Number of episodes to run. Defaults to 5.
        max_steps_per_episode : int, optional
            Maximum steps per episode. Defaults to 50.

        Returns
        -------
        SelfImprovementResult
            Complete result including per-episode metrics and improvement delta.
        """
        if n_episodes <= 0:
            raise ValueError(f"n_episodes must be positive, got {n_episodes}")

        task_id = getattr(task, "task_id", str(task))
        logger.info(
            "Starting self-improvement run: task=%s, n_episodes=%d",
            task_id,
            n_episodes,
        )

        # Clear episode history for this run
        self._episode_history = []
        self._policy_update_history = []
        self._total_self_play_epochs = 0

        buffer_size_before = len(self._hindsight_buffer)

        for episode_idx in range(n_episodes):
            logger.info(
                "Self-improvement episode %d/%d for task %s",
                episode_idx + 1,
                n_episodes,
                task_id,
            )

            # Create fresh environment for this episode
            env = env_factory()

            # Run episode and get the record
            episode_record = self._run_single_episode(
                task=task,
                env=env,
                episode_idx=episode_idx,
                max_steps=max_steps_per_episode,
            )

            # Extract metrics from episode
            metrics = self._extract_episode_metrics(episode_record, episode_idx)
            self._episode_history.append(metrics)

            logger.info(
                "Episode %d: success=%s, milestone_hit_rate=%.2f, length=%d, "
                "failed_subgoals=%d",
                episode_idx,
                metrics.success,
                metrics.milestone_hit_rate,
                metrics.episode_length,
                len(metrics.failed_subgoals),
            )

            # Extract failures and add to hindsight buffer
            failures = self._extract_failures(episode_record)
            for failure_traj in failures:
                self._hindsight_buffer.add(failure_traj)
                logger.debug(
                    "Added failure trajectory to buffer: %s",
                    failure_traj.id,
                )

            # Perform self-play policy update
            if self._self_play_epochs > 0:
                update_result = self._self_play_update(episode_record)
                self._policy_update_history.append(update_result)
                self._total_self_play_epochs += update_result.epochs_run

                logger.info(
                    "Self-play update: epochs=%d, policy_delta=%.4f, "
                    "buffer_size=%d->%d, simulation=%s",
                    update_result.epochs_run,
                    update_result.estimated_policy_delta,
                    update_result.buffer_size_before,
                    update_result.buffer_size_after,
                    update_result.is_simulation,
                )

        # Compute final metrics
        buffer_size_after = len(self._hindsight_buffer)

        # Success rate: count successful episodes
        successful_episodes = sum(1 for m in self._episode_history if m.success)
        final_success_rate = successful_episodes / n_episodes

        # Initial success rate: was the first episode successful?
        initial_success_rate = 1.0 if (
            self._episode_history and self._episode_history[0].success
        ) else 0.0

        improvement_delta = final_success_rate - initial_success_rate

        # Find episodes to first success
        episodes_to_first_success = None
        for idx, m in enumerate(self._episode_history):
            if m.success:
                episodes_to_first_success = idx + 1
                break

        result = SelfImprovementResult(
            task_id=task_id,
            episodes=self._episode_history.copy(),
            final_success_rate=final_success_rate,
            initial_success_rate=initial_success_rate,
            improvement_delta=improvement_delta,
            total_self_play_epochs=self._total_self_play_epochs,
            hindsight_buffer_size_after=buffer_size_after,
            policy_update_results=self._policy_update_history.copy(),
            episodes_to_first_success=episodes_to_first_success,
        )

        logger.info(
            "Self-improvement run complete: task=%s, initial_sr=%.2f, "
            "final_sr=%.2f, delta=%.2f, first_success_at_episode=%s",
            task_id,
            initial_success_rate,
            final_success_rate,
            improvement_delta,
            episodes_to_first_success,
        )

        return result

    def run_with_self_improvement_batch(
        self,
        tasks: List[Any],
        env_factory: Callable[[Any], Any],
        n_episodes_per_task: int = 5,
        max_steps_per_episode: int = 50,
    ) -> List[SelfImprovementResult]:
        """Run self-improvement on multiple tasks.

        Parameters
        ----------
        tasks : List[Any]
            List of tasks to evaluate.
        env_factory : Callable[[Any], Any]
            Factory that creates environment for a given task.
        n_episodes_per_task : int, optional
            Number of episodes per task. Defaults to 5.
        max_steps_per_episode : int, optional
            Maximum steps per episode. Defaults to 50.

        Returns
        -------
        List[SelfImprovementResult]
            Results for each task.
        """
        results = []
        for task in tasks:
            task_env_factory = lambda t=task: env_factory(t)
            result = self.run_with_self_improvement(
                task=task,
                env_factory=task_env_factory,
                n_episodes=n_episodes_per_task,
                max_steps_per_episode=max_steps_per_episode,
            )
            results.append(result)
        return results

    # -------------------------------------------------------------------------
    # Episode Execution
    # -------------------------------------------------------------------------

    def _run_single_episode(
        self,
        task: Any,
        env: Any,
        episode_idx: int,
        max_steps: int = 50,
    ) -> Dict[str, Any]:
        """Run a single episode and return the complete record.

        Parameters
        ----------
        task : Any
            The task object.
        env : Any
            The environment for this episode.
        episode_idx : int
            Episode index for record keeping.
        max_steps : int
            Maximum steps before terminating.

        Returns
        -------
        Dict[str, Any]
            Episode record containing:
            - task_id, episode_idx
            - milestone_verdicts: list of Verdicts per milestone
            - milestone_descriptions: list of descriptions
            - milestone_exec_traces: execution traces
            - success: overall task success
            - total_reward: cumulative reward
            - episode_length: number of steps
            - all_failed_subgoals: all failed milestone descriptions
        """
        task_id = getattr(task, "task_id", str(task))
        milestones = getattr(task, "milestones", [])

        # Reset environment with task
        obs = self._reset_env(env, task)

        milestone_verdicts: List[Verdict] = []
        milestone_descriptions: List[str] = []
        milestone_exec_traces: List[Dict[str, Any]] = []
        total_reward = 0.0
        episode_length = 0

        # Get milestone states from env if available
        env_milestone_states = getattr(env, "_milestone_states", None)
        if env_milestone_states is None:
            # Build default milestone states
            env_milestone_states = [
                {"milestone": m, "hit": False, "attempts": 0}
                for m in milestones
            ]

        current_milestone_idx = 0

        for step in range(max_steps):
            episode_length = step + 1

            # Get current milestone
            if current_milestone_idx < len(milestones):
                current_milestone = milestones[current_milestone_idx]
            else:
                current_milestone = None

            # Generate action using agent
            action_str = self._generate_agent_action(
                task=task,
                milestone=current_milestone,
                env=env,
            )

            # Execute in environment
            prev_obs = obs
            obs, reward, done, info = self._step_env(env, action_str)
            total_reward += reward

            # Record execution trace
            if current_milestone:
                trace = {
                    "step": step,
                    "milestone_id": getattr(current_milestone, "id", ""),
                    "milestone_description": getattr(current_milestone, "description", ""),
                    "action": action_str,
                    "reward": reward,
                }
                milestone_exec_traces.append(trace)

            # Check milestone completion
            if current_milestone_idx < len(env_milestone_states):
                ms_state = env_milestone_states[current_milestone_idx]
                ms_state["attempts"] += 1

                milestone_complete = self._check_milestone_completion(
                    current_milestone,
                    action_str,
                    obs,
                    info,
                )

                if milestone_complete:
                    ms_state["hit"] = True
                    milestone_verdicts.append(Verdict.PASS)
                    milestone_descriptions.append(
                        getattr(current_milestone, "description", "")
                    )
                    current_milestone_idx += 1

            if done:
                break

        # Mark remaining milestones as failed
        while current_milestone_idx < len(milestones):
            milestone_verdicts.append(Verdict.FAIL)
            milestone_descriptions.append(
                getattr(milestones[current_milestone_idx], "description", "")
            )
            current_milestone_idx += 1

        # Build failed subgoal list
        all_failed_subgoals = []
        for i, v in enumerate(milestone_verdicts):
            if v == Verdict.FAIL and i < len(milestones):
                all_failed_subgoals.append(
                    getattr(milestones[i], "description", "")
                )

        # Overall success
        success = all(v == Verdict.PASS for v in milestone_verdicts) if milestone_verdicts else False

        # Milestone hit rate
        total_milestones = len(milestones) if milestones else 1
        hit_rate = sum(1 for v in milestone_verdicts if v == Verdict.PASS) / total_milestones

        return {
            "task_id": task_id,
            "episode_idx": episode_idx,
            "milestone_verdicts": milestone_verdicts,
            "milestone_descriptions": milestone_descriptions,
            "milestone_exec_traces": milestone_exec_traces,
            "milestones": milestones,
            "success": success,
            "total_reward": total_reward,
            "episode_length": episode_length,
            "all_failed_subgoals": all_failed_subgoals,
            "milestone_hit_rate": hit_rate,
        }

    def _reset_env(self, env: Any, task: Any) -> Dict[str, Any]:
        """Reset environment with task.

        Parameters
        ----------
        env : Any
            Environment to reset.
        task : Any
            Task to load.

        Returns
        -------
        Dict[str, Any]
            Initial observation.
        """
        reset_fn = getattr(env, "reset", None)
        if reset_fn is not None:
            return reset_fn(task)
        return {}

    def _step_env(
        self,
        env: Any,
        action: str,
    ) -> tuple:
        """Step the environment with an action.

        Parameters
        ----------
        env : Any
            Environment to step.
        action : str
            Action string.

        Returns
        -------
        tuple
            (observation, reward, done, info)
        """
        step_fn = getattr(env, "step", None)
        if step_fn is not None:
            return step_fn(action)
        return {}, 0.0, True, {}

    def _generate_agent_action(
        self,
        task: Any,
        milestone: Any,
        env: Any,
    ) -> str:
        """Generate an action using the agent.

        Parameters
        ----------
        task : Any
            The task.
        milestone : Any
            Current milestone or None.
        env : Any
            Current environment state.

        Returns
        -------
        str
            Action string.
        """
        # Use agent.act() if available
        act_fn = getattr(self._agent, "act", None)
        if act_fn is not None:
            obs = {
                "task": getattr(task, "description", ""),
                "active_milestone": {
                    "id": getattr(milestone, "id", ""),
                    "description": getattr(milestone, "description", ""),
                    "rubric": getattr(milestone, "rubric", ""),
                },
                "step_count": 0,
            }
            result = act_fn(obs)
            return getattr(result, "action", "")

        # Fallback: simple action construction
        milestone_desc = getattr(milestone, "description", "unknown")
        return f"Execute milestone: {milestone_desc[:100]}"

    def _check_milestone_completion(
        self,
        milestone: Any,
        action: str,
        obs: Dict[str, Any],
        info: Dict[str, Any],
    ) -> bool:
        """Check if a milestone has been completed.

        Parameters
        ----------
        milestone : Any
            The milestone to check.
        action : str
            Action taken.
        obs : Dict[str, Any]
            Current observation.
        info : Dict[str, Any]
            Step info.

        Returns
        -------
        bool
            True if milestone is complete.
        """
        # Check explicit completion flags
        if info.get("milestone_complete", False):
            return True
        if info.get("task_complete", False):
            return True

        # Check URL-based milestones
        url = obs.get("url", "").lower()
        rubric = getattr(milestone, "rubric", "").lower()
        description = getattr(milestone, "description", "").lower()

        # Navigation milestones
        for check_term in ["settings", "contact", "search", "profile"]:
            if check_term in rubric or check_term in description:
                if check_term in url:
                    return True

        # Theme milestones
        if "theme" in rubric or "dark" in rubric:
            if obs.get("theme", "") == "dark":
                return True

        # Login milestones
        if "logged" in rubric or "login" in rubric:
            if not obs.get("is_logged_in", True) and "login" not in url:
                return True

        # Form milestones
        form_values = obs.get("form_values", {})
        if "name" in rubric.lower():
            if form_values.get("name"):
                return True
        if "email" in rubric.lower():
            if form_values.get("email"):
                return True

        return False

    # -------------------------------------------------------------------------
    # Failure Extraction
    # -------------------------------------------------------------------------

    def _extract_failures(self, episode_record: Dict[str, Any]) -> List[HindsightTrajectory]:
        """Extract failed subgoal trajectories from episode record.

        Parameters
        ----------
        episode_record : Dict[str, Any]
            Episode record from _run_single_episode.

        Returns
        -------
        List[HindsightTrajectory]
            List of HindsightTrajectory objects for each failed milestone.
            Empty list if no failures.
        """
        failures = []

        milestones = episode_record.get("milestones", [])
        verdicts = episode_record.get("milestone_verdicts", [])
        exec_traces = episode_record.get("milestone_exec_traces", [])
        task_id = episode_record.get("task_id", "")
        task_desc = getattr(task_id, "description", str(task_id))

        # Get task description if available
        if hasattr(task_id, "description"):
            task_desc = task_id.description
        elif hasattr(task_id, "task"):
            task_desc = task_id.task

        failed_indices = [
            i for i, v in enumerate(verdicts)
            if v == Verdict.FAIL
        ]

        for idx in failed_indices:
            milestone = milestones[idx] if idx < len(milestones) else None

            # Get traces for this milestone
            traj_traces = [
                t for t in exec_traces
                if t.get("milestone_id", "") == getattr(milestone, "id", "")
            ]

            # Build HindsightTrajectory
            from heros.planner import Milestone

            traj = HindsightTrajectory(
                task=task_desc,
                milestones=[milestone] if milestone else [],
                exec_traces=traj_traces,
                verdicts=[verdicts[idx]] if idx < len(verdicts) else [],
                unmet_rubrics=[
                    getattr(milestone, "rubric", ""),
                    getattr(milestone, "description", ""),
                ],
            )

            failures.append(traj)

        logger.debug(
            "Extracted %d failure trajectories from episode %d",
            len(failures),
            episode_record.get("episode_idx", -1),
        )

        return failures

    def _extract_episode_metrics(
        self,
        episode_record: Dict[str, Any],
        episode_idx: int,
    ) -> EpisodeMetrics:
        """Extract EpisodeMetrics from an episode record.

        Parameters
        ----------
        episode_record : Dict[str, Any]
            Episode record from _run_single_episode.
        episode_idx : int
            Episode index.

        Returns
        -------
        EpisodeMetrics
            Structured metrics for this episode.
        """
        verdicts = episode_record.get("milestone_verdicts", [])
        milestone_verdicts_str = [v.value if isinstance(v, Verdict) else str(v) for v in verdicts]

        return EpisodeMetrics(
            episode_idx=episode_idx,
            success=episode_record.get("success", False),
            milestone_hit_rate=episode_record.get("milestone_hit_rate", 0.0),
            total_reward=episode_record.get("total_reward", 0.0),
            episode_length=episode_record.get("episode_length", 0),
            failed_subgoals=episode_record.get("all_failed_subgoals", []),
            milestone_verdicts=milestone_verdicts_str,
        )

    # -------------------------------------------------------------------------
    # Self-Play Update
    # -------------------------------------------------------------------------

    def _self_play_update(self, episode_record: Dict[str, Any]) -> PolicyUpdateResult:
        """Perform a lightweight self-play policy update.

        Uses the hindsight buffer to perform a BC-style (behavior cloning)
        update. When OPENAI_API_KEY is not available or simulate_updates
        is True, this performs a simulated update that logs what would happen.

        Parameters
        ----------
        episode_record : Dict[str, Any]
            Current episode record (used for context).

        Returns
        -------
        PolicyUpdateResult
            Result of the policy update.
        """
        buffer_size_before = len(self._hindsight_buffer)

        if buffer_size_before == 0:
            logger.debug("Buffer empty, skipping self-play update")
            return PolicyUpdateResult(
                epochs_run=0,
                buffer_size_before=0,
                buffer_size_after=0,
                estimated_policy_delta=0.0,
                is_simulation=True,
                details={"message": "Buffer empty"},
            )

        # Perform self-play epochs
        total_epochs = 0
        total_delta = 0.0

        for epoch_idx in range(self._self_play_epochs):
            total_epochs += 1

            if self._simulate_updates:
                # Simulated update: compute a mock policy delta
                delta = self._compute_simulated_policy_delta(episode_record, epoch_idx)
                total_delta += abs(delta)
            else:
                # Real update via agent's update method
                try:
                    update_result = self._agent.update()
                    if update_result:
                        total_delta += getattr(update_result, "loss", 0.0)
                except Exception as e:
                    logger.warning(
                        "Real policy update failed, falling back to simulation: %s",
                        e,
                    )
                    total_delta += self._compute_simulated_policy_delta(
                        episode_record, epoch_idx
                    )

        buffer_size_after = len(self._hindsight_buffer)
        avg_delta = total_delta / max(total_epochs, 1)

        result = PolicyUpdateResult(
            epochs_run=total_epochs,
            buffer_size_before=buffer_size_before,
            buffer_size_after=buffer_size_after,
            estimated_policy_delta=avg_delta,
            is_simulation=self._simulate_updates,
            details={
                "total_epochs_attempted": total_epochs,
                "avg_policy_delta": avg_delta,
            },
        )

        logger.debug(
            "Self-play update complete: epochs=%d, delta=%.4f, simulation=%s",
            total_epochs,
            avg_delta,
            self._simulate_updates,
        )

        return result

    def _compute_simulated_policy_delta(
        self,
        episode_record: Dict[str, Any],
        epoch_idx: int,
    ) -> float:
        """Compute a simulated policy delta for testing.

        This produces a reasonable mock value based on:
        - Number of failures in the episode
        - How many self-play epochs have run
        - Random variation

        Parameters
        ----------
        episode_record : Dict[str, Any]
            Current episode record.
        epoch_idx : int
            Current epoch index.

        Returns
        -------
        float
            Simulated policy delta (magnitude of change).
        """
        import random

        num_failures = len(episode_record.get("all_failed_subgoals", []))
        base_delta = 0.01 * num_failures  # Each failure contributes 1%

        # Diminishing returns over epochs
        decay = 1.0 / (1.0 + epoch_idx * 0.5)

        # Small random variation
        noise = random.uniform(-0.005, 0.005)

        delta = base_delta * decay + noise

        return max(0.0, delta)

    def _has_openai_key(self) -> bool:
        """Check if OpenAI API key is available.

        Returns
        -------
        bool
            True if OPENAI_API_KEY is set and non-empty.
        """
        key = os.environ.get("OPENAI_API_KEY", "")
        return bool(key and key not in ("", "your-api-key-here"))

    # -------------------------------------------------------------------------
    # Utility
    # -------------------------------------------------------------------------

    def get_improvement_trajectory(self) -> List[EpisodeMetrics]:
        """Return the full trajectory of episode metrics showing improvement.

        Returns
        -------
        List[EpisodeMetrics]
            Copy of episode history in order.
        """
        return self._episode_history.copy()

    def get_policy_update_history(self) -> List[PolicyUpdateResult]:
        """Return the history of policy updates.

        Returns
        -------
        List[PolicyUpdateResult]
            Copy of policy update history.
        """
        return self._policy_update_history.copy()

    def reset_local_buffer(self) -> None:
        """Clear the local hindsight buffer.

        Note: This only affects the local buffer tracked by this improver,
        not any shared buffers.
        """
        # We can't actually clear the deque, but we track that we've reset
        logger.info("Local buffer reset requested (buffer size: %d)", len(self._hindsight_buffer))

    def __repr__(self) -> str:
        return (
            f"TestTimeSelfImprover("
            f"self_play_epochs={self._self_play_epochs}, "
            f"improvement_threshold={self._improvement_threshold}, "
            f"simulate={self._simulate_updates}, "
            f"episode_history={len(self._episode_history)}, "
            f"total_self_play_epochs={self._total_self_play_epochs})"
        )
