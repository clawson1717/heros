"""Inference-only Engine for HeRoS Test-time Improvement.

Provides a simplified inference runner that can optionally collect
failed subgoals for self-improvement without requiring a full
training loop.

Reference: Self-play / test-time improvement literature.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, TYPE_CHECKING

from heros.buffer import HindsightBuffer, HindsightTrajectory
from heros.self_improver import (
    InferenceEpisodeResult,
    BatchInferenceResult,
    TestTimeSelfImprover,
    EpisodeMetrics,
    PolicyUpdateResult,
)

if TYPE_CHECKING:
    from heros.agent import HeRoSAgent

logger = logging.getLogger(__name__)


# ============================================================================
# InferenceEngine
# ============================================================================


class InferenceEngine:
    """Simplified inference runner for test-time improvement.

    Provides a clean interface for running inference episodes with
    optional self-improvement. Designed for deployment scenarios where:
    - We want to run episodes efficiently without training overhead
    - We want to optionally collect failures for future improvement
    - We want a simple API that hides the complexity of the improver

    Parameters
    ----------
    agent : HeRoSAgent
        The HeRoSAgent to run inference with.
    use_self_improvement : bool, optional
        Whether to enable self-improvement between episodes.
        Defaults to True.
    local_buffer_capacity : int, optional
        Capacity of the local hindsight buffer for storing failures.
        Defaults to 100.

    Attributes
    ----------
    local_hindsight_buffer : HindsightBuffer
        The local buffer for storing inference-time failures.
    improver : TestTimeSelfImprover
        The underlying self-improver instance.

    Examples
    --------
    >>> engine = InferenceEngine(agent, use_self_improvement=True)
    >>> result = engine.run_inference_episode(task, env_factory)
    >>> print(f"Success: {result.success}, Hit Rate: {result.milestone_hit_rate:.2f}")
    >>> engine.reset_local_hindsight_buffer()
    """

    def __init__(
        self,
        agent: "HeRoSAgent",
        use_self_improvement: bool = True,
        local_buffer_capacity: int = 100,
    ) -> None:
        if local_buffer_capacity <= 0:
            raise ValueError(
                f"local_buffer_capacity must be positive, got {local_buffer_capacity}"
            )

        self._agent = agent
        self._use_self_improvement = use_self_improvement

        # Create local hindsight buffer
        self._local_buffer = HindsightBuffer(
            capacity=local_buffer_capacity,
            hindsight_ratio=0.3,
        )

        # Create the self-improver if self-improvement is enabled
        if use_self_improvement:
            self._improver = TestTimeSelfImprover(
                agent=agent,
                hindsight_buffer=self._local_buffer,
                self_play_epochs=3,
                improvement_threshold=0.05,
                simulate_updates=True,  # Safe default for inference
            )
        else:
            self._improver = None

        self._total_self_play_epochs: int = 0

        logger.info(
            "InferenceEngine initialized: self_improvement=%s, buffer_capacity=%d",
            use_self_improvement,
            local_buffer_capacity,
        )

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def agent(self) -> "HeRoSAgent":
        """The agent running inference."""
        return self._agent

    @property
    def local_hindsight_buffer(self) -> HindsightBuffer:
        """The local hindsight buffer for storing failures."""
        return self._local_buffer

    @property
    def use_self_improvement(self) -> bool:
        """Whether self-improvement is enabled."""
        return self._use_self_improvement

    @property
    def improver(self) -> Optional[TestTimeSelfImprover]:
        """The underlying self-improver, if enabled."""
        return self._improver

    @property
    def total_self_play_epochs(self) -> int:
        """Total number of self-play epochs performed."""
        return self._total_self_play_epochs

    # -------------------------------------------------------------------------
    # Single Episode
    # -------------------------------------------------------------------------

    def run_inference_episode(
        self,
        task: Any,
        env_factory: Callable[[], Any],
        collect_failures: bool = True,
    ) -> InferenceEpisodeResult:
        """Run a single inference episode.

        Parameters
        ----------
        task : Any
            The task to evaluate. Must have task_id and description
            and milestones attributes.
        env_factory : Callable[[], Any]
            Factory function that creates a fresh environment.
        collect_failures : bool, optional
            Whether to collect failed subgoals for the hindsight buffer.
            Defaults to True.

        Returns
        -------
        InferenceEpisodeResult
            Result of the inference episode.
        """
        task_id = getattr(task, "task_id", str(task))

        logger.debug(
            "Running inference episode: task=%s, collect_failures=%s",
            task_id,
            collect_failures,
        )

        # Create environment
        env = env_factory()

        # Run episode using the improver's internal method
        if self._improver is not None:
            episode_record = self._improver._run_single_episode(
                task=task,
                env=env,
                episode_idx=0,
                max_steps=50,
            )
        else:
            episode_record = self._run_episode_simple(
                task=task,
                env=env,
            )

        # Extract failures if requested
        collected_for_hindsight = False
        if collect_failures:
            failures = self._extract_failures(episode_record)
            for failure_traj in failures:
                self._local_buffer.add(failure_traj)
            collected_for_hindsight = len(failures) > 0

        # Build result
        result = InferenceEpisodeResult(
            task_id=task_id,
            success=episode_record.get("success", False),
            milestone_hit_rate=episode_record.get("milestone_hit_rate", 0.0),
            total_reward=episode_record.get("total_reward", 0.0),
            episode_length=episode_record.get("episode_length", 0),
            failed_subgoals=episode_record.get("all_failed_subgoals", []),
            collected_for_hindsight=collected_for_hindsight,
        )

        logger.debug(
            "Inference episode complete: task=%s, success=%s, "
            "milestone_hit_rate=%.2f, collected=%s",
            task_id,
            result.success,
            result.milestone_hit_rate,
            collected_for_hindsight,
        )

        return result

    def _run_episode_simple(
        self,
        task: Any,
        env: Any,
    ) -> dict:
        """Run a simple episode without self-improvement.

        Parameters
        ----------
        task : Any
            The task.
        env : Any
            The environment.

        Returns
        -------
        dict
            Episode record.
        """
        milestones = getattr(task, "milestones", [])
        task_id = getattr(task, "task_id", str(task))

        # Reset environment
        obs = self._reset_env(env, task)

        milestone_verdicts = []
        milestone_descriptions = []
        total_reward = 0.0
        episode_length = 0
        max_steps = 50

        env_milestone_states = getattr(env, "_milestone_states", None)
        if env_milestone_states is None:
            env_milestone_states = [
                {"milestone": m, "hit": False, "attempts": 0}
                for m in milestones
            ]

        current_milestone_idx = 0
        all_failed_subgoals = []

        for step in range(max_steps):
            episode_length = step + 1

            if current_milestone_idx < len(milestones):
                current_milestone = milestones[current_milestone_idx]
            else:
                current_milestone = None

            # Generate action
            action_str = self._generate_action(task, current_milestone, env)

            # Step
            obs, reward, done, info = self._step_env(env, action_str)
            total_reward += reward

            # Check completion
            if current_milestone_idx < len(env_milestone_states):
                ms_state = env_milestone_states[current_milestone_idx]
                ms_state["attempts"] += 1

                complete = self._check_completion(
                    current_milestone, action_str, obs, info
                )

                if complete:
                    ms_state["hit"] = True
                    milestone_verdicts.append(self._verdict_pass())
                    milestone_descriptions.append(
                        getattr(current_milestone, "description", "")
                    )
                    current_milestone_idx += 1

            if done:
                break

        # Mark remaining as failed
        while current_milestone_idx < len(milestones):
            milestone_verdicts.append(self._verdict_fail())
            desc = getattr(milestones[current_milestone_idx], "description", "")
            milestone_descriptions.append(desc)
            all_failed_subgoals.append(desc)
            current_milestone_idx += 1

        # Compute hit rate
        total_milestones = len(milestones) if milestones else 1
        hit_rate = sum(1 for v in milestone_verdicts if v.value == "PASS") / total_milestones

        return {
            "task_id": task_id,
            "success": all(v.value == "PASS" for v in milestone_verdicts),
            "milestone_hit_rate": hit_rate,
            "total_reward": total_reward,
            "episode_length": episode_length,
            "milestone_verdicts": milestone_verdicts,
            "milestone_descriptions": milestone_descriptions,
            "all_failed_subgoals": all_failed_subgoals,
        }

    def _extract_failures(self, episode_record: dict) -> List[HindsightTrajectory]:
        """Extract failed trajectories from episode record.

        Parameters
        ----------
        episode_record : dict
            Episode record.

        Returns
        -------
        List[HindsightTrajectory]
            List of failure trajectories.
        """
        from heros.critic import Verdict
        from heros.planner import Milestone

        failures = []
        milestones = episode_record.get("milestones", [])
        verdicts = episode_record.get("milestone_verdicts", [])
        task_id = episode_record.get("task_id", "")
        task_desc = str(task_id)

        failed_indices = [
            i for i, v in enumerate(verdicts)
            if (isinstance(v, Verdict) and v == Verdict.FAIL) or v.value == "FAIL"
        ]

        for idx in failed_indices:
            milestone = milestones[idx] if idx < len(milestones) else None

            traj = HindsightTrajectory(
                task=task_desc,
                milestones=[milestone] if milestone else [],
                exec_traces=[],
                verdicts=[verdicts[idx]] if idx < len(verdicts) else [],
                unmet_rubrics=[
                    getattr(milestone, "rubric", ""),
                    getattr(milestone, "description", ""),
                ],
            )
            failures.append(traj)

        return failures

    def _reset_env(self, env: Any, task: Any) -> dict:
        """Reset environment with task."""
        reset_fn = getattr(env, "reset", None)
        if reset_fn is not None:
            return reset_fn(task)
        return {}

    def _step_env(self, env: Any, action: str) -> tuple:
        """Step the environment."""
        step_fn = getattr(env, "step", None)
        if step_fn is not None:
            return step_fn(action)
        return {}, 0.0, True, {}

    def _generate_action(
        self,
        task: Any,
        milestone: Any,
        env: Any,
    ) -> str:
        """Generate an action for the current state."""
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

        milestone_desc = getattr(milestone, "description", "unknown")
        return f"Execute: {milestone_desc[:100]}"

    def _check_completion(
        self,
        milestone: Any,
        action: str,
        obs: dict,
        info: dict,
    ) -> bool:
        """Check if milestone is complete."""
        if info.get("milestone_complete", False):
            return True
        if info.get("task_complete", False):
            return True

        url = obs.get("url", "").lower()
        rubric = getattr(milestone, "rubric", "").lower()
        description = getattr(milestone, "description", "").lower()

        for term in ["settings", "contact", "search", "profile"]:
            if term in rubric or term in description:
                if term in url:
                    return True

        if "theme" in rubric or "dark" in rubric:
            if obs.get("theme", "") == "dark":
                return True

        if "logged" in rubric or "login" in rubric:
            if not obs.get("is_logged_in", True):
                return True

        return False

    def _verdict_pass(self):
        """Return a PASS verdict."""
        from heros.critic import Verdict
        return Verdict.PASS

    def _verdict_fail(self):
        """Return a FAIL verdict."""
        from heros.critic import Verdict
        return Verdict.FAIL

    # -------------------------------------------------------------------------
    # Batch Inference
    # -------------------------------------------------------------------------

    def run_batch_inference(
        self,
        tasks: List[Any],
        env_factory: Callable[[Any], Any],
        show_progress: bool = False,
        n_episodes_per_task: int = 1,
    ) -> BatchInferenceResult:
        """Run batch inference on multiple tasks.

        Parameters
        ----------
        tasks : List[Any]
            List of tasks to evaluate.
        env_factory : Callable[[Any], Any]
            Factory that creates environment for a given task.
        show_progress : bool, optional
            Whether to log progress. Defaults to False.
        n_episodes_per_task : int, optional
            Number of episodes per task. If > 1, self-improvement kicks in.
            Defaults to 1.

        Returns
        -------
        BatchInferenceResult
            Aggregated results for the batch.
        """
        if n_episodes_per_task < 1:
            raise ValueError(
                f"n_episodes_per_task must be >= 1, got {n_episodes_per_task}"
            )

        task_ids = [getattr(t, "task_id", str(t)) for t in tasks]
        episode_results: List[InferenceEpisodeResult] = []
        total_self_play = 0

        for task_idx, task in enumerate(tasks):
            task_id = task_ids[task_idx]

            if show_progress:
                logger.info(
                    "Batch inference: task %d/%d (%s)",
                    task_idx + 1,
                    len(tasks),
                    task_id,
                )

            # Run N episodes for this task
            for ep_idx in range(n_episodes_per_task):
                ep_env_factory = lambda t=task: env_factory(t)

                result = self.run_inference_episode(
                    task=task,
                    env_factory=ep_env_factory,
                    collect_failures=True,
                )
                episode_results.append(result)

                # Update self-play epoch count
                if self._improver is not None and ep_idx < len(self._improver._policy_update_history):
                    total_self_play += self._improver._policy_update_history[ep_idx].epochs_run

        # Compute aggregate metrics
        successful = sum(1 for r in episode_results if r.success)
        total = len(episode_results)
        overall_sr = successful / total if total > 0 else 0.0

        avg_hit_rate = (
            sum(r.milestone_hit_rate for r in episode_results) / total
            if total > 0 else 0.0
        )

        batch_result = BatchInferenceResult(
            tasks=task_ids,
            episode_results=episode_results,
            overall_success_rate=overall_sr,
            avg_milestone_hit_rate=avg_hit_rate,
            total_self_play_epochs=total_self_play,
            hindsight_buffer_size=len(self._local_buffer),
        )

        logger.info(
            "Batch inference complete: tasks=%d, episodes=%d, "
            "success_rate=%.2f, avg_hit_rate=%.2f, buffer_size=%d",
            len(tasks),
            total,
            overall_sr,
            avg_hit_rate,
            len(self._local_buffer),
        )

        return batch_result

    # -------------------------------------------------------------------------
    # Buffer Management
    # -------------------------------------------------------------------------

    def reset_local_hindsight_buffer(self) -> None:
        """Clear the local inference-time hindsight buffer.

        Note: HindsightBuffer uses a deque with maxlen, so we can't truly
        "clear" it without recreation. This resets tracking state.
        """
        logger.info(
            "Resetting local hindsight buffer (current size: %d)",
            len(self._local_buffer),
        )
        # Recreate the buffer
        capacity = self._local_buffer.capacity
        ratio = self._local_buffer.hindsight_ratio
        self._local_buffer = HindsightBuffer(
            capacity=capacity,
            hindsight_ratio=ratio,
        )

        # Update improver's buffer reference if it exists
        if self._improver is not None:
            self._improver._hindsight_buffer = self._local_buffer

        self._total_self_play_epochs = 0

    def get_buffer_stats(self) -> dict:
        """Get statistics about the local hindsight buffer.

        Returns
        -------
        dict
            Buffer statistics from HindsightBuffer.get_stats().
        """
        return self._local_buffer.get_stats()

    # -------------------------------------------------------------------------
    # Self-Improvement Access
    # -------------------------------------------------------------------------

    def run_self_improvement_on_task(
        self,
        task: Any,
        env_factory: Callable[[], Any],
        n_episodes: int = 5,
        max_steps_per_episode: int = 50,
    ):
        """Run full self-improvement on a task using the improver.

        This is a convenience method that delegates to the underlying
        TestTimeSelfImprover.

        Parameters
        ----------
        task : Any
            Task to improve on.
        env_factory : Callable[[], Any]
            Environment factory.
        n_episodes : int, optional
            Number of episodes. Defaults to 5.
        max_steps_per_episode : int, optional
            Max steps per episode. Defaults to 50.

        Returns
        -------
        SelfImprovementResult
            Full self-improvement result.
        """
        if self._improver is None:
            raise RuntimeError(
                "Self-improvement is not enabled. "
                "Set use_self_improvement=True in constructor."
            )

        return self._improver.run_with_self_improvement(
            task=task,
            env_factory=env_factory,
            n_episodes=n_episodes,
            max_steps_per_episode=max_steps_per_episode,
        )

    # -------------------------------------------------------------------------
    # Representation
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"InferenceEngine("
            f"use_self_improvement={self._use_self_improvement}, "
            f"buffer_size={len(self._local_buffer)}, "
            f"total_self_play_epochs={self._total_self_play_epochs})"
        )
