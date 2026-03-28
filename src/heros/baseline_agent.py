"""Baseline Agent: Raw LLM agent without milestones or hindsight.

This module provides a baseline comparison agent that uses a plain LLM
to select actions based solely on the task description and current
observation — without any milestone planning, critic evaluation, or
hindsight relabeling.

This serves as the control group for evaluating HeRoS's improvement
over a standard approach.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

import logging

logger = logging.getLogger(__name__)


class BaselineAgent:
    """Raw LLM agent with NO milestone planning, NO hindsight, NO critic.

    This agent takes a simple approach:
    1. Receives task description and current observation
    2. Generates the next action directly from the LLM
    3. No milestone tracking, no critic, no hindsight

    This baseline represents a "vanilla" LLM agent that attempts to
    solve tasks through direct action selection without any of the
    HeRoS enhancements.

    Parameters
    ----------
    model_name : str, optional
        The OpenAI model to use. Defaults to "gpt-4o-mini".
    api_key : str, optional
        OpenAI API key. If not provided, reads from OPENAI_API_KEY env var.
    temperature : float, optional
        LLM sampling temperature. Defaults to 0.7.
    max_tokens : int, optional
        Maximum tokens in LLM response. Defaults to 256.

    Attributes
    ----------
    model_name : str
        The model being used.
    action_count : int
        Total number of actions this agent has taken.

    Examples
    --------
    >>> agent = BaselineAgent(model_name="gpt-4o-mini")
    >>> task = "Navigate to settings and change theme to dark"
    >>> obs = "URL: https://example.com/home\\nClickable: [Settings] [Search]"
    >>> action = agent.act(task, obs)
    >>> print(action)
    "Click [Settings] to navigate to settings page"
    """

    DEFAULT_MODEL = "gpt-4o-mini"
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_MAX_TOKENS = 256

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        if not isinstance(model_name, str):
            raise TypeError(f"model_name must be a str, got {type(model_name).__name__}")
        self._model_name = model_name

        if not isinstance(temperature, (int, float)):
            raise TypeError(f"temperature must be a float, got {type(temperature).__name__}")
        if not (0.0 <= temperature <= 2.0):
            raise ValueError(f"temperature must be in [0.0, 2.0], got {temperature}")
        self._temperature = float(temperature)

        if not isinstance(max_tokens, int):
            raise TypeError(f"max_tokens must be an int, got {type(max_tokens).__name__}")
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")
        self._max_tokens = max_tokens

        # Resolve API key
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._has_api_key = bool(resolved_key)

        if self._has_api_key:
            from openai import OpenAI
            self._client = OpenAI(api_key=resolved_key)
        else:
            self._client = None
            logger.warning(
                "No OpenAI API key found. BaselineAgent will use rule-based "
                "fallback responses."
            )

        # Action tracking
        self._action_count: int = 0

        # System prompt for the baseline agent
        self._system_prompt = self._build_system_prompt()

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        """The model name being used."""
        return self._model_name

    @property
    def action_count(self) -> int:
        """Total number of actions taken."""
        return self._action_count

    @property
    def has_api_key(self) -> bool:
        """True if an API key is configured."""
        return self._has_api_key

    # -------------------------------------------------------------------------
    # Core Action Method
    # -------------------------------------------------------------------------

    def act(self, task: str, observation: str) -> str:
        """Generate the next action based on task and observation.

        This is the main method for generating actions. It takes the
        task description and current observation, then generates an
        action string using either the LLM (if available) or a
        rule-based fallback.

        Parameters
        ----------
        task : str
            Natural language description of the task goal.
        observation : str
            Current state observation (URL, page content, clickable elements, etc.).

        Returns
        -------
        str
            The generated action as a string. This should be parseable
            by the evaluation harness into a WebAction.

        Notes
        -----
        The returned action string should describe what the agent wants
        to do. Examples:
        - "Click the Settings link"
        - "Type 'Alice' in the name field"
        - "Navigate to /settings"
        - "Submit the search form"
        """
        self._action_count += 1

        if self._client is not None:
            return self._act_with_llm(task, observation)
        else:
            return self._act_rule_based(task, observation)

    def _act_with_llm(self, task: str, observation: str) -> str:
        """Generate action using the LLM.

        Parameters
        ----------
        task : str
            Task description.
        observation : str
            Current observation.

        Returns
        -------
        str
            Generated action string.
        """
        user_prompt = self._build_user_prompt(task, observation)

        try:
            response = self._client.chat.completions.create(
                model=self._model_name,
                messages=[
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )

            content = response.choices[0].message.content
            if content is None:
                return self._act_rule_based(task, observation)

            return content.strip()

        except Exception as e:
            logger.warning("LLM call failed: %s, using rule-based fallback", e)
            return self._act_rule_based(task, observation)

    def _act_rule_based(self, task: str, observation: str) -> str:
        """Generate action using simple rule-based heuristics.

        This is a fallback when no LLM is available. It uses keyword
        matching and simple heuristics to generate actions.

        Parameters
        ----------
        task : str
            Task description.
        observation : str
            Current observation.

        Returns
        -------
        str
            A rule-based action string.
        """
        task_lower = task.lower()
        obs_lower = observation.lower()

        # Parse observation for elements
        has_settings = "settings" in obs_lower
        has_contact = "contact" in obs_lower
        has_search = "search" in obs_lower
        has_logout = "logout" in obs_lower
        has_form = "form" in obs_lower or "field" in obs_lower
        is_logged_in = "logged in: true" in obs_lower or "is_logged_in: true" in obs_lower

        # Theme-related tasks
        if "theme" in task_lower or "dark" in task_lower or "light" in task_lower:
            if has_settings:
                if "dark" in task_lower:
                    return "Select the dark theme option in settings"
                elif "light" in task_lower:
                    return "Select the light theme option in settings"
                else:
                    return "Navigate to settings to change theme"
            else:
                return "Click the Settings link"

        # Contact form tasks
        if "contact" in task_lower or ("form" in task_lower and "fill" in task_lower):
            if has_contact and has_form:
                # Determine what to fill
                if "name" in task_lower and "alice" in task_lower:
                    return "Type 'Alice' in the name field"
                elif "email" in task_lower and "alice" in task_lower:
                    return "Type 'alice@example.com' in the email field"
                elif "name" in task_lower:
                    return "Type the required name in the name field"
                elif "email" in task_lower:
                    return "Type the required email in the email field"
                else:
                    return "Fill in the contact form fields"
            elif has_contact:
                return "Fill in the contact form"
            else:
                return "Click the Contact link"

        # Search tasks
        if "search" in task_lower:
            if "llm" in task_lower or "open source" in task_lower:
                if has_search:
                    return "Type 'open source LLMs' in the search field and submit"
                else:
                    return "Click the Search link, then type the query"
            elif "date" in task_lower:
                from datetime import date
                today = str(date.today())
                if has_search:
                    return f"Type '{today}' in the search field"
                else:
                    return "Click Search, then enter today's date"
            elif has_search:
                return "Enter the search query"
            else:
                return "Click the Search link"

        # Logout task
        if "logout" in task_lower or "log out" in task_lower:
            if is_logged_in and has_logout:
                return "Click the Logout button"
            else:
                return "Find and click the Logout option"

        # Default: try to find a link to click
        if has_settings:
            return "Click the Settings link"
        elif has_contact:
            return "Click the Contact link"
        elif has_search:
            return "Click the Search link"
        elif has_logout:
            return "Click the Logout button"
        else:
            return "Examine the page and take appropriate action"

    # -------------------------------------------------------------------------
    # Prompt Building
    # -------------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        """Build the system prompt for the baseline agent.

        Returns
        -------
        str
            System prompt text.
        """
        return """You are a web navigation assistant. Given a task description and 
the current page state, you must decide on the next action to take.

You can perform these actions:
- click [element]: Click on a link, button, or element
- type [field] [text]: Type text into an input field
- navigate [url]: Navigate to a URL
- submit [form]: Submit a form

IMPORTANT RULES:
1. Be specific about which element to interact with
2. Only perform one action at a time
3. If you see a Settings link, use it to access settings
4. If you need to fill a form, identify the correct fields
5. Return ONLY the action you want to take, nothing else

Example:
Task: Navigate to settings and change theme to dark
Observation: URL: /home, Clickable: [Home] [Search] [Contact] [Settings]
Action: click Settings

Now you generate the action."""

    def _build_user_prompt(self, task: str, observation: str) -> str:
        """Build the user prompt for action generation.

        Parameters
        ----------
        task : str
            Task description.
        observation : str
            Current observation.

        Returns
        -------
        str
            Formatted user prompt.
        """
        return f"""Task: {task}

Current Page State:
{observation}

What action should I take? Return ONLY the action."""

    # -------------------------------------------------------------------------
    # Reset
    # -------------------------------------------------------------------------

    def reset(self) -> None:
        """Reset the agent's internal state.

        This resets action counts and any other tracked state.
        Does not reset API configuration.
        """
        self._action_count = 0

    # -------------------------------------------------------------------------
    # Representation
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"BaselineAgent("
            f"model={self._model_name!r}, "
            f"actions={self._action_count}, "
            f"has_key={self._has_api_key})"
        )
