"""HeRoS Evaluation Harness for WebArena-Lite / MiniWoB Benchmarks.

Provides:
- EvaluationResult: Structured result dataclass
- HeRoSEvaluator: Main evaluation orchestrator
- Metrics computation for comparing baseline vs HeRoS agents
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from heros.benchmark import (
    WebArenaLiteBenchmark,
    WebTask,
    MockWebEnv,
    WebAction,
    EvaluationAction,
    Milestone,
)

import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Evaluation Result
# ============================================================================


@dataclass
class EvaluationResult:
    """Result of a single evaluation episode.

    Attributes
    ----------
    task_id : str
        The task that was evaluated.
    agent_type : str
        Type of agent used: "baseline" or "heros".
    completion : bool
        Whether the task was successfully completed.
    milestone_hit_rate : float
        Fraction of milestones that were completed (0.0 to 1.0).
    total_reward : float
        Cumulative reward accumulated during the episode.
    episode_length : int
        Number of steps taken in the episode.
    hindsight_applied : bool
        Whether hindsight was enabled for this run.
    per_milestone_results : List[Dict[str, Any]]
        Detailed results for each milestone.
    task_description : str
        The task description for reference.
    episode_history : List[Dict[str, Any]]
        History of actions taken during the episode.
    timestamp : str
        ISO-8601 timestamp of when the evaluation started.
    error : Optional[str]
        Error message if the episode failed unexpectedly.
    """

    task_id: str
    agent_type: str
    completion: bool = False
    milestone_hit_rate: float = 0.0
    total_reward: float = 0.0
    episode_length: int = 0
    hindsight_applied: bool = False
    per_milestone_results: List[Dict[str, Any]] = field(default_factory=list)
    task_description: str = ""
    episode_history: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "task_id": self.task_id,
            "agent_type": self.agent_type,
            "completion": self.completion,
            "milestone_hit_rate": self.milestone_hit_rate,
            "total_reward": self.total_reward,
            "episode_length": self.episode_length,
            "hindsight_applied": self.hindsight_applied,
            "per_milestone_results": self.per_milestone_results,
            "task_description": self.task_description,
            "episode_history": self.episode_history,
            "timestamp": self.timestamp,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EvaluationResult":
        """Deserialize from a dictionary."""
        return cls(
            task_id=d["task_id"],
            agent_type=d["agent_type"],
            completion=d.get("completion", False),
            milestone_hit_rate=d.get("milestone_hit_rate", 0.0),
            total_reward=d.get("total_reward", 0.0),
            episode_length=d.get("episode_length", 0),
            hindsight_applied=d.get("hindsight_applied", False),
            per_milestone_results=d.get("per_milestone_results", []),
            task_description=d.get("task_description", ""),
            episode_history=d.get("episode_history", []),
            timestamp=d.get("timestamp", ""),
            error=d.get("error"),
        )


# ============================================================================
# HeRoS Evaluator
# ============================================================================


class HeRoSEvaluator:
    """Evaluation harness for WebArena-Lite / MiniWoB benchmarks.

    Runs episodes with either a baseline agent or a HeRoS agent and
    computes comprehensive metrics including:
    - Task completion rate
    - Milestone hit rate
    - Hindsight improvement delta
    - Average reward and episode length

    Parameters
    ----------
    benchmark : WebArenaLiteBenchmark
        The benchmark to evaluate on.
    agent : HeRoSAgent | BaselineAgent
        The agent to evaluate. Must be either a HeRoSAgent or BaselineAgent.
    use_hindsight : bool, optional
        Whether to enable hindsight for HeRoS agents. Defaults to True.
    max_steps : int, optional
        Maximum steps per episode. Defaults to 20.

    Attributes
    ----------
    results : List[EvaluationResult]
        All evaluation results collected so far.

    Examples
    --------
    >>> benchmark = WebArenaLiteBenchmark(task_subset="mini")
    >>> baseline = BaselineAgent()
    >>> evaluator = HeRoSEvaluator(benchmark, baseline)
    >>> results = evaluator.run_evaluation()
    >>> metrics = evaluator.compute_metrics(results)
    >>> print(f"Completion rate: {metrics['completion_rate']:.2%}")
    """

    def __init__(
        self,
        benchmark: WebArenaLiteBenchmark,
        agent: Any,  # HeRoSAgent | BaselineAgent
        use_hindsight: bool = True,
        max_steps: int = 20,
    ) -> None:
        if not isinstance(benchmark, WebArenaLiteBenchmark):
            raise TypeError(
                f"benchmark must be a WebArenaLiteBenchmark, got {type(benchmark).__name__}"
            )
        self._benchmark = benchmark

        # Agent type detection
        agent_type = getattr(agent, "__class__", None)
        if agent_type is None:
            raise TypeError("agent must have a __class__ attribute")

        agent_name = agent_type.__name__.lower()
        if "baseline" in agent_name:
            self._agent_type = "baseline"
        elif "heros" in agent_name or "wrapped" in agent_name:
            self._agent_type = "heros"
        else:
            # Default to heros if unknown but has the right interface
            self._agent_type = "heros"
            logger.warning(
                "Unknown agent type %s, assuming 'heros'",
                agent_type.__name__,
            )

        self._agent = agent
        self._use_hindsight = use_hindsight
        self._max_steps = max_steps

        self._results: List[EvaluationResult] = []

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def benchmark(self) -> WebArenaLiteBenchmark:
        """The benchmark being evaluated."""
        return self._benchmark

    @property
    def agent(self) -> Any:
        """The agent being evaluated."""
        return self._agent

    @property
    def agent_type(self) -> str:
        """Type of agent: 'baseline' or 'heros'."""
        return self._agent_type

    @property
    def use_hindsight(self) -> bool:
        """Whether hindsight is enabled."""
        return self._use_hindsight

    @property
    def max_steps(self) -> int:
        """Maximum steps per episode."""
        return self._max_steps

    @property
    def results(self) -> List[EvaluationResult]:
        """All evaluation results collected so far."""
        return self._results

    # -------------------------------------------------------------------------
    # Episode Running
    # -------------------------------------------------------------------------

    def run_episode(self, task_id: str) -> EvaluationResult:
        """Run a single evaluation episode for a task.

        Parameters
        ----------
        task_id : str
            The task to evaluate.

        Returns
        -------
        EvaluationResult
            The result of the episode.
        """
        task = self._benchmark.get_task(task_id)
        env = self._benchmark.create_env_for_task(task_id)

        return self._run_episode_with_env(task, env)

    def _run_episode_with_env(
        self,
        task: WebTask,
        env: MockWebEnv,
    ) -> EvaluationResult:
        """Run an episode with a pre-configured environment.

        Parameters
        ----------
        task : WebTask
            The task to evaluate.
        env : MockWebEnv
            Pre-configured environment.

        Returns
        -------
        EvaluationResult
            The result of the episode.
        """
        obs = env.reset(task)
        episode_history: List[Dict[str, Any]] = []
        total_reward = 0.0
        step_count = 0

        milestones = task.milestones
        milestone_states = [
            {"milestone": m, "hit": False, "attempts": 0}
            for m in milestones
        ]
        current_milestone_idx = 0

        per_milestone_results: List[Dict[str, Any]] = []

        try:
            while step_count < self._max_steps:
                step_count += 1

                # Get agent action
                action_str = self._agent.act(
                    task=task.description,
                    observation=self._format_observation(obs),
                )

                # Parse action
                web_action = self._parse_action_string(action_str, env, current_milestone_idx, milestones)

                # Execute action
                prev_obs = obs
                obs, reward, done, info = env.step(web_action.action)

                total_reward += reward
                episode_history.append({
                    "step": step_count,
                    "action": web_action.action.to_dict(),
                    "milestone_id": web_action.milestone_id,
                    "reward": reward,
                    "observation_url": obs.get("url", ""),
                })

                # Check milestone progress
                if current_milestone_idx < len(milestone_states):
                    ms_state = milestone_states[current_milestone_idx]
                    ms_state["attempts"] += 1

                    # Check if milestone was completed
                    milestone_complete = self._check_milestone_complete(
                        milestones[current_milestone_idx],
                        web_action,
                        obs,
                        info,
                    )

                    if milestone_complete:
                        ms_state["hit"] = True
                        per_milestone_results.append({
                            "milestone_id": milestones[current_milestone_idx].id,
                            "milestone_description": milestones[current_milestone_idx].description,
                            "hit": True,
                            "attempts": ms_state["attempts"],
                        })
                        current_milestone_idx += 1

                if done:
                    break

            # Final milestone results for any not yet evaluated
            while current_milestone_idx < len(milestone_states):
                ms_state = milestone_states[current_milestone_idx]
                per_milestone_results.append({
                    "milestone_id": milestones[current_milestone_idx].id,
                    "milestone_description": milestones[current_milestone_idx].description,
                    "hit": ms_state["hit"],
                    "attempts": ms_state["attempts"],
                })
                current_milestone_idx += 1

        except Exception as e:
            logger.error("Episode failed with error: %s", e)
            return EvaluationResult(
                task_id=task.task_id,
                agent_type=self._agent_type,
                task_description=task.description,
                error=str(e),
                episode_history=episode_history,
                hindsight_applied=self._use_hindsight,
            )

        # Determine completion
        milestones_hit = sum(1 for ms in milestone_states if ms["hit"])
        total_milestones = len(milestones)
        milestone_hit_rate = milestones_hit / total_milestones if total_milestones > 0 else 0.0
        completion = milestones_hit == total_milestones

        return EvaluationResult(
            task_id=task.task_id,
            agent_type=self._agent_type,
            completion=completion,
            milestone_hit_rate=milestone_hit_rate,
            total_reward=total_reward,
            episode_length=step_count,
            hindsight_applied=self._use_hindsight,
            per_milestone_results=per_milestone_results,
            task_description=task.description,
            episode_history=episode_history,
        )

    def _format_observation(self, obs: Dict[str, Any]) -> str:
        """Format an observation for the agent.

        Parameters
        ----------
        obs : Dict[str, Any]
            Raw observation from MockWebEnv.

        Returns
        -------
        str
            Formatted observation string.
        """
        lines = [
            f"URL: {obs.get('url', 'unknown')}",
            f"Page Content: {obs.get('page_content', '').strip()[:200]}",
            f"Logged in: {obs.get('is_logged_in', False)}",
            f"Theme: {obs.get('theme', 'unknown')}",
            "",
            "Clickable Elements:",
        ]

        for elem in obs.get("clickable_elements", []):
            lines.append(f"  - [{elem.get('id', '')}] {elem.get('text', '')} ({elem.get('type', '')})")

        forms = obs.get("available_forms", [])
        if forms:
            lines.append("")
            lines.append("Available Forms:")
            for form in forms:
                lines.append(f"  - {form.get('id', '')}: {', '.join(form.get('fields', []))}")

        return "\n".join(lines)

    def _parse_action_string(
        self,
        action_str: str,
        env: MockWebEnv,
        milestone_idx: int,
        milestones: List[Milestone],
    ) -> EvaluationAction:
        """Parse an agent's action string into a WebAction.

        Parameters
        ----------
        action_str : str
            Raw action string from the agent.
        env : MockWebEnv
            Current environment state.
        milestone_idx : int
            Current milestone index.
        milestones : List[Milestone]
            All milestones for the task.

        Returns
        -------
        EvaluationAction
            Parsed evaluation action.
        """
        action_str_lower = action_str.lower().strip()

        milestone_id = ""
        if milestone_idx < len(milestones):
            milestone_id = milestones[milestone_idx].id

        # Try to determine action type and target from string
        action_type = "click"
        target = ""
        value = None

        # Check for action type keywords
        if "type " in action_str_lower or "enter " in action_str_lower:
            action_type = "type"
            # Try to extract target and value
            parts = action_str.split()
            for i, part in enumerate(parts):
                if part.lower() in ("type", "enter") and i + 1 < len(parts):
                    target = parts[i + 1].strip("[]\"':")
                    if i + 2 < len(parts):
                        value = " ".join(parts[i + 2:]).strip("[]\"':")
                    break
        elif "navigate" in action_str_lower or "go to" in action_str_lower or "->" in action_str:
            action_type = "navigate"
            # Extract URL/path
            if "->" in action_str:
                value = action_str.split("->")[-1].strip()
            else:
                for word in ["navigate", "go to", "go", "visit"]:
                    if word in action_str_lower:
                        idx = action_str_lower.find(word)
                        value = action_str[idx + len(word):].strip()
                        break
        elif "submit" in action_str_lower or "press submit" in action_str_lower:
            action_type = "submit"
        elif "select" in action_str_lower:
            action_type = "select"
            # Try to extract value
            words = action_str.split()
            for i, w in enumerate(words):
                if w.lower() in ("dark", "light", "option"):
                    if i + 1 < len(words):
                        value = words[i + 1].lower()
                        break
            if value is None and "dark" in action_str_lower:
                value = "dark"
            elif value is None and "light" in action_str_lower:
                value = "light"
        elif "check" in action_str_lower:
            action_type = "check"
        elif "uncheck" in action_str_lower:
            action_type = "uncheck"

        # Default target selection based on URL and milestone
        if not target:
            url = env.current_url.lower()
            if "settings" in url:
                target = "#settings-link"
            elif "contact" in url:
                target = "#contact-form"
            elif "search" in url:
                target = "#search-field"
            else:
                # Check clickable elements
                for elem in env._get_clickable_elements():
                    elem_text = elem.get("text", "").lower()
                    elem_id = elem.get("id", "").lower()
                    if "settings" in elem_text or "settings" in elem_id:
                        target = f"#{elem_id}"
                        break
                    elif "contact" in elem_text or "contact" in elem_id:
                        target = f"#{elem_id}"
                        break
                    elif "search" in elem_text or "search" in elem_id:
                        target = f"#{elem_id}"
                        break
                    elif "logout" in elem_text or "logout" in elem_id:
                        target = f"#{elem_id}"
                        break

            if not target:
                target = "#main-content"

        if value is None:
            value = ""

        web_action = WebAction(
            action_type=action_type,
            target=target,
            value=value,
            label=action_str[:100],
        )

        return EvaluationAction(
            action=web_action,
            milestone_id=milestone_id,
            reasoning=action_str,
            success=False,
        )

    def _check_milestone_complete(
        self,
        milestone: Milestone,
        action: EvaluationAction,
        obs: Dict[str, Any],
        info: Dict[str, Any],
    ) -> bool:
        """Check if a milestone has been completed.

        Parameters
        ----------
        milestone : Milestone
            The milestone to check.
        action : EvaluationAction
            The action that was taken.
        obs : Dict[str, Any]
            Current observation.
        info : Dict[str, Any]
            Step info dict.

        Returns
        -------
        bool
            True if the milestone is complete.
        """
        # Check info for explicit milestone completion
        if info.get("milestone_complete", False):
            return True

        if info.get("task_complete", False):
            return True

        # Check URL for navigation milestones
        url = obs.get("url", "").lower()
        rubric = milestone.rubric.lower()

        if "settings" in rubric and "settings" in url:
            return True
        if "contact" in rubric and "contact" in url:
            return True
        if "search" in rubric and "search" in url:
            return True
        if "logout" in rubric or "logged in" in rubric:
            if "login" in url and not obs.get("is_logged_in", True):
                return True

        # Check theme change
        if "theme" in rubric.lower() or "dark" in rubric.lower():
            if obs.get("theme", "") == "dark":
                return True

        # Check form fields
        form_values = obs.get("form_values", {})
        if "name" in rubric.lower() and form_values.get("name"):
            if "alice" in form_values.get("name", "").lower():
                return True
        if "email" in rubric.lower() and form_values.get("email"):
            if "alice@example.com" in form_values.get("email", ""):
                return True

        # Check search query
        if "search" in rubric.lower() and obs.get("search_query"):
            return True

        return False

    # -------------------------------------------------------------------------
    # Full Evaluation
    # -------------------------------------------------------------------------

    def run_evaluation(
        self,
        task_ids: Optional[List[str]] = None,
    ) -> List[EvaluationResult]:
        """Run evaluation on a set of tasks.

        Parameters
        ----------
        task_ids : List[str], optional
            Specific task IDs to evaluate. If None, evaluates all tasks
            in the benchmark.

        Returns
        -------
        List[EvaluationResult]
            Results for each evaluated task.
        """
        if task_ids is None:
            task_ids = self._benchmark.list_tasks()

        results = []
        for task_id in task_ids:
            logger.info("Evaluating task: %s with %s agent", task_id, self._agent_type)
            try:
                result = self.run_episode(task_id)
                results.append(result)
                self._results.append(result)
            except Exception as e:
                logger.error("Failed to evaluate task %s: %s", task_id, e)
                error_result = EvaluationResult(
                    task_id=task_id,
                    agent_type=self._agent_type,
                    error=str(e),
                    hindsight_applied=self._use_hindsight,
                )
                results.append(error_result)
                self._results.append(error_result)

        return results

    # -------------------------------------------------------------------------
    # Metrics Computation
    # -------------------------------------------------------------------------

    def compute_metrics(
        self,
        results: Optional[List[EvaluationResult]] = None,
    ) -> Dict[str, Any]:
        """Compute aggregate metrics from evaluation results.

        Parameters
        ----------
        results : List[EvaluationResult], optional
            Results to compute metrics from. If None, uses all stored results.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - completion_rate: Fraction of tasks completed (0.0 to 1.0)
            - avg_milestone_hit_rate: Average milestone hit rate across tasks
            - avg_reward: Average total reward per episode
            - avg_episode_length: Average episode length in steps
            - hindsight_delta: Difference in completion rate between
              hindsight-enabled and disabled (if computable)
            - per_task_results: Individual task results
            - agent_type: The agent type evaluated
        """
        if results is None:
            results = self._results

        if not results:
            return {
                "completion_rate": 0.0,
                "avg_milestone_hit_rate": 0.0,
                "avg_reward": 0.0,
                "avg_episode_length": 0.0,
                "hindsight_delta": None,
                "per_task_results": [],
                "agent_type": self._agent_type,
                "total_tasks": 0,
            }

        # Filter out error results for metrics
        valid_results = [r for r in results if r.error is None]

        if not valid_results:
            return {
                "completion_rate": 0.0,
                "avg_milestone_hit_rate": 0.0,
                "avg_reward": 0.0,
                "avg_episode_length": 0.0,
                "hindsight_delta": None,
                "per_task_results": [r.to_dict() for r in results],
                "agent_type": self._agent_type,
                "total_tasks": len(results),
                "errors": [r.error for r in results if r.error],
            }

        # Basic metrics
        total = len(valid_results)
        completed = sum(1 for r in valid_results if r.completion)
        completion_rate = completed / total if total > 0 else 0.0

        avg_milestone_hit_rate = (
            sum(r.milestone_hit_rate for r in valid_results) / total
            if total > 0 else 0.0
        )
        avg_reward = (
            sum(r.total_reward for r in valid_results) / total
            if total > 0 else 0.0
        )
        avg_episode_length = (
            sum(r.episode_length for r in valid_results) / total
            if total > 0 else 0.0
        )

        # Hindsight delta computation
        hindsight_delta = None
        hindsight_enabled = [r for r in valid_results if r.hindsight_applied]
        hindsight_disabled = [r for r in valid_results if not r.hindsight_applied]

        if hindsight_enabled and hindsight_disabled:
            enabled_rate = sum(1 for r in hindsight_enabled if r.completion) / len(hindsight_enabled)
            disabled_rate = sum(1 for r in hindsight_disabled if r.completion) / len(hindsight_disabled)
            hindsight_delta = enabled_rate - disabled_rate

        return {
            "completion_rate": completion_rate,
            "avg_milestone_hit_rate": avg_milestone_hit_rate,
            "avg_reward": avg_reward,
            "avg_episode_length": avg_episode_length,
            "hindsight_delta": hindsight_delta,
            "per_task_results": [r.to_dict() for r in valid_results],
            "agent_type": self._agent_type,
            "total_tasks": total,
            "completed_tasks": completed,
        }

    def compare_agents(
        self,
        results_baseline: List[EvaluationResult],
        results_heros: List[EvaluationResult],
    ) -> Dict[str, Any]:
        """Compare results between baseline and HeRoS agents.

        Parameters
        ----------
        results_baseline : List[EvaluationResult]
            Results from baseline agent evaluation.
        results_heros : List[EvaluationResult]
            Results from HeRoS agent evaluation.

        Returns
        -------
        Dict[str, Any]
            Comparison metrics including improvement deltas.
        """
        metrics_baseline = self.compute_metrics(results_baseline)
        metrics_heros = self.compute_metrics(results_heros)

        completion_delta = (
            metrics_heros["completion_rate"] - metrics_baseline["completion_rate"]
        )
        milestone_delta = (
            metrics_heros["avg_milestone_hit_rate"] - metrics_baseline["avg_milestone_hit_rate"]
        )
        reward_delta = metrics_heros["avg_reward"] - metrics_baseline["avg_reward"]
        length_delta = metrics_heros["avg_episode_length"] - metrics_baseline["avg_episode_length"]

        return {
            "baseline_metrics": metrics_baseline,
            "heros_metrics": metrics_heros,
            "improvement": {
                "completion_rate_delta": completion_delta,
                "milestone_hit_rate_delta": milestone_delta,
                "avg_reward_delta": reward_delta,
                "avg_episode_length_delta": length_delta,
            },
            "summary": (
                f"HeRoS improved completion rate by {completion_delta:.1%} "
                f"({metrics_baseline['completion_rate']:.1%} -> {metrics_heros['completion_rate']:.1%})"
            ),
        }

    # -------------------------------------------------------------------------
    # Utility
    # -------------------------------------------------------------------------

    def get_task_results(self, task_id: str) -> List[EvaluationResult]:
        """Get all results for a specific task.

        Parameters
        ----------
        task_id : str
            The task ID to look up.

        Returns
        -------
        List[EvaluationResult]
            All results for the specified task.
        """
        return [r for r in self._results if r.task_id == task_id]

    def clear_results(self) -> None:
        """Clear all stored results."""
        self._results = []

    def export_results(self, path: str) -> None:
        """Export results to a JSON file.

        Parameters
        ----------
        path : str
            File path to write to.
        """
        import json

        data = {
            "agent_type": self._agent_type,
            "use_hindsight": self._use_hindsight,
            "max_steps": self._max_steps,
            "results": [r.to_dict() for r in self._results],
            "metrics": self.compute_metrics(),
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info("Exported %d results to %s", len(self._results), path)

    def __repr__(self) -> str:
        return (
            f"HeRoSEvaluator("
            f"agent_type={self._agent_type!r}, "
            f"hindsight={self._use_hindsight}, "
            f"max_steps={self._max_steps}, "
            f"results={len(self._results)})"
        )
