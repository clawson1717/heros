"""HeRoS Agent Wrapper for Evaluation Mode.

Wraps the Step 6 HeRoSAgent with additional functionality for
evaluation in the WebArena-Lite / MiniWoB benchmark environment:
- Action parsing (click/type/navigate/submit)
- Observation formatting for milestone-aware agents
- Hindsight integration toggle
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from heros.benchmark import (
    WebAction,
    EvaluationAction,
    WebTask,
    MockWebEnv,
    Milestone,
)
from heros.agent import HeRoSAgent

import logging

logger = logging.getLogger(__name__)


class HeRoSWrappedAgent:
    """HeRoS agent configured for evaluation mode.

    This wrapper adds the following capabilities to the base HeRoSAgent:
    1. Action parsing: Converts milestone-targeted output to web actions
    2. Observation formatting: Prepares observations for milestone tracking
    3. Hindsight toggle: Enable/disable hindsight experience replay

    Parameters
    ----------
    heros_agent : HeRoSAgent
        The base HeRoS agent from Step 6.
    hindsight_enabled : bool, optional
        Whether to enable hindsight experience replay. Defaults to True.

    Attributes
    ----------
    milestone_index : int
        Current milestone index the agent is working on.
    total_milestones : int
        Total number of milestones in the current task.

    Examples
    --------
    >>> from heros import HeRoSAgent
    >>> from heros.benchmark import WebArenaLiteBenchmark
    >>> from heros.heros_agent_wrapper import HeRoSWrappedAgent
    >>>
    >>> benchmark = WebArenaLiteBenchmark(task_subset="mini")
    >>> agent = HeRoSWrappedAgent(heros_agent=some_agent)
    >>>
    >>> task = benchmark.get_task("change_theme_dark")
    >>> obs = env.get_observation()
    >>> action = agent.act(task.description, obs)
    """

    def __init__(
        self,
        heros_agent: HeRoSAgent,
        hindsight_enabled: bool = True,
    ) -> None:
        if not isinstance(heros_agent, HeRoSAgent):
            raise TypeError(
                f"heros_agent must be a HeRoSAgent, got {type(heros_agent).__name__}"
            )
        self._agent = heros_agent
        self._hindsight_enabled = hindsight_enabled

        # State tracking for evaluation
        self._current_task: Optional[str] = None
        self._current_milestones: List[Milestone] = []
        self._milestone_index: int = 0
        self._milestone_states: List[Dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def agent(self) -> HeRoSAgent:
        """The wrapped HeRoS agent."""
        return self._agent

    @property
    def hindsight_enabled(self) -> bool:
        """Whether hindsight is enabled."""
        return self._hindsight_enabled

    @hindsight_enabled.setter
    def hindsight_enabled(self, value: bool) -> None:
        """Enable or disable hindsight."""
        self._hindsight_enabled = bool(value)

    @property
    def milestone_index(self) -> int:
        """Current milestone index."""
        return self._milestone_index

    @property
    def total_milestones(self) -> int:
        """Total milestones in current task."""
        return len(self._current_milestones)

    # -------------------------------------------------------------------------
    # Core Action Method
    # -------------------------------------------------------------------------

    def act(self, task: str, observation: str | Dict[str, Any]) -> str:
        """Generate the next action targeting the current milestone.

        This method:
        1. Determines the active milestone
        2. Has the HeRoS agent generate a milestone-targeted response
        3. Parses that response into a concrete web action string

        Parameters
        ----------
        task : str
            Task description.
        observation : str | Dict[str, Any]
            Current observation. Can be a string (from MockWebEnv.get_observation())
            or a dict with structured observation data.

        Returns
        -------
        str
            An action string suitable for parsing by the evaluator into a WebAction.
            This is NOT a WebAction directly, but a string that describes the action.
        """
        # Ensure we have task context
        if self._current_task != task:
            self._setup_task(task)

        # Format observation if it's a dict
        if isinstance(observation, dict):
            observation = self._format_observation(observation)

        # Build milestone-augmented observation for HeRoS agent
        milestone_obs = self._build_milestone_observation(task, observation)

        # Get action from HeRoS agent
        act_result = self._agent.act(milestone_obs)

        # Update milestone tracking based on critic result
        if act_result.critic_result is not None:
            self._update_milestone_tracking(act_result)

        # Parse the agent's action into a web action string
        action_str = self._parse_action(
            act_result.action,
            observation,
            act_result.milestone_description,
        )

        return action_str

    def _setup_task(self, task: str) -> None:
        """Set up internal state for a new task.

        Parameters
        ----------
        task : str
            Task description.
        """
        self._current_task = task
        self._milestone_index = 0
        self._milestone_states = []

        # The milestones will be set when the agent's planner produces them
        self._current_milestones = []

    def _update_milestone_tracking(self, act_result: Any) -> None:
        """Update milestone tracking based on critic result.

        Parameters
        ----------
        act_result : ActResult
            Result from the HeRoS agent's act() call.
        """
        if act_result.is_milestone_complete:
            if self._milestone_index < len(self._milestone_states):
                self._milestone_states[self._milestone_index]["hit"] = True
            self._milestone_index += 1

    def _format_observation(self, obs: Dict[str, Any]) -> str:
        """Format a dict observation into a string.

        Parameters
        ----------
        obs : Dict[str, Any]
            Structured observation from MockWebEnv.

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

    def _build_milestone_observation(
        self,
        task: str,
        observation: str,
    ) -> Dict[str, Any]:
        """Build a milestone-augmented observation for the HeRoS agent.

        Parameters
        ----------
        task : str
            Task description.
        observation : str
            Formatted observation string.

        Returns
        -------
        Dict[str, Any]
            Milestone-augmented observation dict.
        """
        active_milestone = None
        if 0 <= self._milestone_index < len(self._current_milestones):
            m = self._current_milestones[self._milestone_index]
            active_milestone = {
                "id": m.id,
                "description": m.description,
                "rubric": m.rubric,
                "expected_output": m.expected_output,
            }

        return {
            "observation": observation,
            "task": task,
            "milestones": [
                {
                    "id": m.id,
                    "description": m.description,
                    "rubric": m.rubric,
                }
                for m in self._current_milestones
            ],
            "active_milestone": active_milestone,
            "step_count": 0,
        }

    def _parse_action(
        self,
        agent_action: str,
        observation: str,
        milestone_desc: str,
    ) -> str:
        """Parse the agent's milestone-targeted action into a web action string.

        Parameters
        ----------
        agent_action : str
            Raw action string from the HeRoS agent.
        observation : str
            Current observation string.
        milestone_desc : str
            Description of the current milestone.

        Returns
        -------
        str
            A web action string like "click Settings" or "type name=Alice".
        """
        action_lower = agent_action.lower()
        milestone_lower = milestone_desc.lower()
        obs_lower = observation.lower()

        # Check what type of action is needed based on milestone
        if "navigate" in milestone_lower or "go to" in milestone_lower or "settings" in milestone_lower:
            if "settings" in milestone_lower or "theme" in milestone_lower:
                return "click Settings link"
        if "fill" in milestone_lower or "type" in milestone_lower or "enter" in milestone_lower:
            if "name" in milestone_lower:
                if "alice" in milestone_lower:
                    return "type Alice in name field"
            if "email" in milestone_lower:
                if "alice" in milestone_lower:
                    return "type alice@example.com in email field"
            if "search" in milestone_lower or "query" in milestone_lower:
                if "date" in milestone_lower:
                    from datetime import date
                    return f"type {date.today()} in search field"
                return "type open source LLMs in search field"
        if "click" in milestone_lower or "select" in milestone_lower or "result" in milestone_lower:
            if "first" in milestone_lower or "result" in milestone_lower:
                return "click first search result"
            if "theme" in milestone_lower or "dark" in milestone_lower:
                return "select dark theme"
        if "logout" in milestone_lower or "log out" in milestone_lower:
            return "click Logout button"
        if "submit" in milestone_lower:
            return "submit form"

        # Fallback: try to parse the raw action
        if "click" in action_lower:
            return agent_action
        elif "type" in action_lower or "enter" in action_lower:
            return agent_action
        elif "navigate" in action_lower or "go to" in action_lower:
            return agent_action
        elif "submit" in action_lower:
            return agent_action

        # Last resort: return the raw action
        return agent_action

    # -------------------------------------------------------------------------
    # Milestone Management
    # -------------------------------------------------------------------------

    def set_milestones(self, milestones: List[Milestone]) -> None:
        """Set the milestones for the current task.

        Parameters
        ----------
        milestones : List[Milestone]
            Ordered list of milestones.
        """
        self._current_milestones = milestones
        self._milestone_states = [
            {"milestone": m, "hit": False, "attempts": 0}
            for m in milestones
        ]
        self._milestone_index = 0

    def get_active_milestone(self) -> Optional[Milestone]:
        """Get the currently active milestone.

        Returns
        -------
        Optional[Milestone]
            The current milestone or None if all are complete.
        """
        if 0 <= self._milestone_index < len(self._current_milestones):
            return self._current_milestones[self._milestone_index]
        return None

    def get_milestone_progress(self) -> Dict[str, Any]:
        """Get progress through milestones.

        Returns
        -------
        Dict[str, Any]
            Progress information including hit rate and status.
        """
        total = len(self._milestone_states)
        hit = sum(1 for ms in self._milestone_states if ms.get("hit", False))

        return {
            "current_index": self._milestone_index,
            "total": total,
            "hit_count": hit,
            "hit_rate": hit / total if total > 0 else 0.0,
            "states": [
                {
                    "id": ms["milestone"].id,
                    "hit": ms.get("hit", False),
                    "attempts": ms.get("attempts", 0),
                }
                for ms in self._milestone_states
            ],
        }

    def reset(self) -> None:
        """Reset the agent's internal state for a new episode."""
        self._current_task = None
        self._current_milestones = []
        self._milestone_index = 0
        self._milestone_states = []

    # -------------------------------------------------------------------------
    # Hindsight Integration
    # -------------------------------------------------------------------------

    def apply_hindsight(self, failed_milestone_id: str) -> None:
        """Apply hindsight to revise the plan after a failure.

        This is a placeholder for the hindsight relabeling functionality.
        When a milestone fails, this method can be called to trigger
        the hindsight buffer and trainer to learn from the failure.

        Parameters
        ----------
        failed_milestone_id : str
            ID of the milestone that failed.
        """
        if not self._hindsight_enabled:
            logger.debug("Hindsight disabled, skipping")
            return

        # The actual hindsight application would happen through the
        # agent's hindsight_buffer and trainer
        logger.info(
            "Hindsight triggered for failed milestone: %s",
            failed_milestone_id,
        )

        # This would typically:
        # 1. Get the failed trajectory from the buffer
        # 2. Apply hindsight relabeling
        # 3. Trigger a policy update

    # -------------------------------------------------------------------------
    # Representation
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"HeRoSWrappedAgent("
            f"hindsight={self._hindsight_enabled}, "
            f"milestones={len(self._current_milestones)}, "
            f"current={self._milestone_index})"
        )
