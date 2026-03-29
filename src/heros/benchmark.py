"""WebArena-Lite / MiniWoB-style Web Navigation Benchmark.

Defines a set of realistic web navigation tasks with ground-truth milestone
decompositions for evaluating HeRoS agents against baseline agents.

This benchmark provides:
- WebTask: Individual task definitions with milestones
- WebArenaLiteBenchmark: Collection of tasks with milestone retrieval
- MockWebEnv: Simulated web environment for offline evaluation
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Literal, Optional, Tuple

from heros.planner import Milestone

import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Web Action Types
# ============================================================================


@dataclass
class WebAction:
    """A web interaction action.

    Attributes
    ----------
    action_type : str
        One of: "click", "type", "navigate", "submit", "select", "check", "uncheck"
    target : str
        The element identifier (CSS selector, XPath, or semantic identifier).
    value : Optional[str]
        For type actions, the text to type. For navigate, the URL or path.
    label : str
        Human-readable description of the action for logging/debugging.
    """

    action_type: str
    target: str
    value: Optional[str] = None
    label: str = ""

    def __str__(self) -> str:
        if self.action_type == "navigate":
            return f"NAVIGATE -> {self.value}"
        elif self.action_type == "type":
            return f"TYPE [{self.target}] = '{self.value}'"
        elif self.action_type == "submit":
            return f"SUBMIT [{self.target}]"
        else:
            return f"{self.action_type.upper()} [{self.target}]"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type,
            "target": self.target,
            "value": self.value,
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WebAction":
        return cls(
            action_type=d.get("action_type", ""),
            target=d.get("target", ""),
            value=d.get("value"),
            label=d.get("label", ""),
        )


@dataclass
class EvaluationAction:
    """An action taken by an agent during evaluation.

    Attributes
    ----------
    action : WebAction
        The web action to execute.
    milestone_id : str
        The milestone ID this action targets.
    reasoning : str
        Agent's reasoning for selecting this action.
    success : bool
        Whether the action was successful (reached intended state).
    """

    action: WebAction
    milestone_id: str = ""
    reasoning: str = ""
    success: bool = False


# ============================================================================
# WebTask Definition
# ============================================================================


@dataclass
class WebTask:
    """A single web navigation task with ground-truth milestones.

    Attributes
    ----------
    task_id : str
        Unique identifier for this task.
    description : str
        Natural language description of the task goal.
    target_url : str
        The initial URL/state from which the task begins.
    milestones : List[Milestone]
        Ordered subgoals required to complete the task.
    expected_actions : List[WebAction]
        Ground truth action sequence for task completion.
    initial_state : Dict[str, Any]
        Initial mock web state (page content, form values, etc.).
    success_criteria : str
        Description of what constitutes successful task completion.
    difficulty : str
        One of: "easy", "medium", "hard"
    """

    task_id: str
    description: str
    target_url: str
    milestones: List[Milestone] = field(default_factory=list)
    expected_actions: List[WebAction] = field(default_factory=list)
    initial_state: Dict[str, Any] = field(default_factory=dict)
    success_criteria: str = ""
    difficulty: str = "medium"

    def __post_init__(self) -> None:
        if self.difficulty not in ("easy", "medium", "hard"):
            raise ValueError(f"difficulty must be 'easy', 'medium', or 'hard', got {self.difficulty}")

    def milestone_count(self) -> int:
        """Return the number of milestones."""
        return len(self.milestones)

    def is_action_sequence_correct(self, actions: List[WebAction]) -> bool:
        """Check if an action sequence matches the expected sequence."""
        if len(actions) != len(self.expected_actions):
            return False
        for actual, expected in zip(actions, self.expected_actions):
            if actual.action_type != expected.action_type:
                return False
            if actual.target != expected.target:
                return False
        return True


# ============================================================================
# Mock Web Environment
# ============================================================================


class MockWebEnv:
    """Simulated web environment for offline benchmark evaluation.

    Maintains state including:
    - Current URL
    - Page content
    - Form values
    - Clickable elements
    - Session state (logged in/out, etc.)

    Parameters
    ----------
    initial_state : Dict[str, Any]
        Initial web state configuration.
    task : WebTask, optional
        The task this environment is configured for.

    Examples
    --------
    >>> env = MockWebEnv()
    >>> env.reset()
    >>> obs = env.get_observation()
    >>> action = WebAction("click", "#settings-link", label="Click settings")
    >>> obs, reward, done, info = env.step(action)
    """

    def __init__(
        self,
        initial_state: Optional[Dict[str, Any]] = None,
        task: Optional[WebTask] = None,
    ) -> None:
        self._task = task
        self._state: Dict[str, Any] = {}
        self._history: List[Dict[str, Any]] = []
        self._step_count: int = 0

        if initial_state:
            self._state = copy.deepcopy(initial_state)
        else:
            self._state = self._default_state()

    # -------------------------------------------------------------------------
    # State Properties
    # -------------------------------------------------------------------------

    @property
    def current_url(self) -> str:
        """Current URL or path in the browser."""
        return self._state.get("url", "https://example.com/home")

    @property
    def page_content(self) -> str:
        """Current page content as text."""
        return self._state.get("page_content", "")

    @property
    def form_values(self) -> Dict[str, str]:
        """Current form field values."""
        return self._state.get("form_values", {})

    @property
    def is_logged_in(self) -> bool:
        """Whether a user is currently logged in."""
        return self._state.get("is_logged_in", False)

    @property
    def theme(self) -> str:
        """Current theme setting (e.g., 'light', 'dark')."""
        return self._state.get("theme", "light")

    @property
    def step_count(self) -> int:
        """Number of steps taken in current episode."""
        return self._step_count

    # -------------------------------------------------------------------------
    # Core Interface
    # -------------------------------------------------------------------------

    def reset(self, task: Optional[WebTask] = None) -> Dict[str, Any]:
        """Reset the environment to initial state.

        Parameters
        ----------
        task : WebTask, optional
            Task to configure the environment for.

        Returns
        -------
        Dict[str, Any]
            Initial observation dictionary.
        """
        if task is not None:
            self._task = task
            self._state = self._task_to_state(task)
        elif self._task is not None:
            self._state = self._task_to_state(self._task)
        else:
            self._state = self._default_state()

        self._history = []
        self._step_count = 0

        return self.get_observation()

    def step(self, action: WebAction) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Execute a web action and return the result.

        Parameters
        ----------
        action : WebAction
            The action to execute.

        Returns
        -------
        Tuple[Dict[str, Any], float, bool, Dict[str, Any]]
            (observation, reward, done, info)
        """
        self._step_count += 1
        self._history.append({
            "step": self._step_count,
            "action": action.to_dict(),
        })

        reward = 0.0
        done = False
        base_info: Dict[str, Any] = {"action_executed": str(action)}

        # Process action based on type
        if action.action_type == "click":
            reward, done, info = self._handle_click(action)
        elif action.action_type == "type":
            reward, done, info = self._handle_type(action)
        elif action.action_type == "navigate":
            reward, done, info = self._handle_navigate(action)
        elif action.action_type == "submit":
            reward, done, info = self._handle_submit(action)
        elif action.action_type == "select":
            reward, done, info = self._handle_select(action)
        elif action.action_type == "check":
            reward, done, info = self._handle_check(action, checked=True)
        elif action.action_type == "uncheck":
            reward, done, info = self._handle_check(action, checked=False)
        else:
            info = {}
            info["error"] = f"Unknown action type: {action.action_type}"

        # Merge handler info with base info (preserve action_executed)
        if info:
            info = {**base_info, **info}
        else:
            info = base_info

        obs = self.get_observation()
        return obs, reward, done, info

    def get_observation(self) -> Dict[str, Any]:
        """Get the current observation for the agent.

        Returns
        -------
        Dict[str, Any]
            Observation dict with web state information.
        """
        return {
            "url": self.current_url,
            "page_content": self.page_content,
            "form_values": self.form_values.copy(),
            "is_logged_in": self.is_logged_in,
            "theme": self.theme,
            "step_count": self._step_count,
            "clickable_elements": self._get_clickable_elements(),
            "available_forms": self._get_available_forms(),
        }

    # -------------------------------------------------------------------------
    # Action Handlers
    # -------------------------------------------------------------------------

    def _handle_click(self, action: WebAction) -> Tuple[float, bool, Dict[str, Any]]:
        """Handle a click action."""
        info: Dict[str, Any] = {}
        reward = 0.0
        done = False

        target = action.target.lower()

        # Settings navigation
        if "settings" in target or "gear" in target:
            if self.current_url != "https://example.com/settings":
                self._state["url"] = "https://example.com/settings"
                self._state["page_content"] = self._settings_page_content()
                reward = 0.1
                info["milestone_progress"] = "Navigated to settings"
            else:
                reward = 0.05
        # Theme toggle
        elif "theme" in target or "dark" in target or "light" in target:
            if self.current_url == "https://example.com/settings":
                if "dark" in target:
                    self._state["theme"] = "dark"
                    reward = 0.5
                    info["milestone_complete"] = True
                    info["milestone_id"] = "m2"
                elif "light" in target:
                    self._state["theme"] = "light"
                    reward = 0.5
                    info["milestone_complete"] = True
                    info["milestone_id"] = "m2"
                else:
                    reward = 0.1
            else:
                reward = -0.1
                info["error"] = "Must be on settings page to change theme"
        # Navigation links
        elif "home" in target:
            self._state["url"] = "https://example.com/home"
            self._state["page_content"] = self._home_page_content()
            reward = 0.1
        elif "contact" in target:
            self._state["url"] = "https://example.com/contact"
            self._state["page_content"] = self._contact_page_content()
            reward = 0.1
            info["milestone_progress"] = "Navigated to contact form"
        # Logout
        elif "logout" in target or "sign out" in target:
            if self.is_logged_in:
                self._state["is_logged_in"] = False
                self._state["url"] = "https://example.com/login"
                self._state["page_content"] = self._login_page_content()
                reward = 1.0
                done = True
                info["task_complete"] = True
            else:
                reward = -0.1
                info["error"] = "Not logged in"
        # Search result click
        elif "result" in target or "search" in target:
            if "1" in target or "first" in target:
                self._state["url"] = "https://example.com/results/llm-article"
                self._state["page_content"] = "Article: Open Source LLMs - A Comprehensive Guide..."
                reward = 1.0
                done = True
                info["task_complete"] = True
        else:
            reward = 0.0
            info["note"] = f"Click on unknown target: {target}"

        return reward, done, info

    def _handle_type(self, action: WebAction) -> Tuple[float, bool, Dict[str, Any]]:
        """Handle a type action."""
        info: Dict[str, Any] = {}
        reward = 0.0
        done = False

        target = action.target.lower()
        value = action.value or ""

        # Form field typing
        if "name" in target:
            forms = self._state.get("form_values", {})
            forms["name"] = value
            self._state["form_values"] = forms
            reward = 0.2
            info["field_filled"] = "name"
        elif "email" in target:
            forms = self._state.get("form_values", {})
            forms["email"] = value
            self._state["form_values"] = forms
            reward = 0.2
            info["field_filled"] = "email"
        elif "search" in target:
            self._state["search_query"] = value
            reward = 0.1
            info["field_filled"] = "search_query"
        elif "query" in target or "date" in target:
            self._state["search_query"] = value
            reward = 0.3
            info["milestone_progress"] = "Search query entered"
        else:
            # Generic form field
            forms = self._state.get("form_values", {})
            forms[target] = value
            self._state["form_values"] = forms
            reward = 0.1

        return reward, done, info

    def _handle_navigate(self, action: WebAction) -> Tuple[float, bool, Dict[str, Any]]:
        """Handle a navigation action."""
        info: Dict[str, Any] = {}
        reward = 0.0
        done = False

        value = (action.value or "").lower()

        if "settings" in value:
            self._state["url"] = "https://example.com/settings"
            self._state["page_content"] = self._settings_page_content()
            reward = 0.3
            info["milestone_progress"] = "Navigated to settings"
        elif "contact" in value:
            self._state["url"] = "https://example.com/contact"
            self._state["page_content"] = self._contact_page_content()
            reward = 0.3
            info["milestone_progress"] = "Navigated to contact"
        elif "home" in value:
            self._state["url"] = "https://example.com/home"
            self._state["page_content"] = self._home_page_content()
            reward = 0.1
        elif "search" in value or "result" in value:
            self._state["url"] = "https://example.com/search"
            self._state["page_content"] = self._search_results_content()
            reward = 0.2
            info["milestone_progress"] = "Navigated to search results"
        else:
            # Try to parse as URL
            if "://" in value:
                self._state["url"] = value
                reward = 0.1
            else:
                self._state["url"] = f"https://example.com/{value}"
                reward = 0.1

        return reward, done, info

    def _handle_submit(self, action: WebAction) -> Tuple[float, bool, Dict[str, Any]]:
        """Handle a form submit action."""
        info: Dict[str, Any] = {}
        reward = 0.0
        done = False

        forms = self.form_values
        target = action.target.lower()

        # Contact form submission
        if "contact" in self.current_url.lower() or "contact" in target:
            if forms.get("name") and forms.get("email"):
                self._state["form_submitted"] = True
                self._state["url"] = "https://example.com/contact/success"
                self._state["page_content"] = "Thank you! Your message has been sent."
                reward = 1.0
                done = True
                info["task_complete"] = True
                info["milestone_complete"] = True
            else:
                reward = -0.2
                info["error"] = "Form fields incomplete"
        # Search submission
        elif "search" in self.current_url.lower() or "search" in target:
            if self._state.get("search_query"):
                self._state["url"] = "https://example.com/search/results"
                self._state["page_content"] = self._search_results_content()
                reward = 0.3
                info["milestone_progress"] = "Search results loaded"
            else:
                reward = -0.2
                info["error"] = "No search query entered"
        # Login form
        elif "login" in self.current_url.lower():
            self._state["is_logged_in"] = True
            self._state["url"] = "https://example.com/home"
            self._state["page_content"] = self._home_page_content()
            reward = 0.5
            info["milestone_progress"] = "Logged in"
        else:
            reward = 0.0
            info["note"] = "Submit on unknown page"

        return reward, done, info

    def _handle_select(self, action: WebAction) -> Tuple[float, bool, Dict[str, Any]]:
        """Handle a dropdown select action."""
        info: Dict[str, Any] = {}
        reward = 0.0
        done = False

        target = action.target.lower()
        value = action.value or ""

        if "theme" in target or "color" in target:
            if self.current_url == "https://example.com/settings":
                self._state["theme"] = value
                reward = 0.5
                info["milestone_complete"] = True
                info["milestone_id"] = "m2"
            else:
                reward = -0.1
                info["error"] = "Must be on settings page"
        else:
            reward = 0.1

        return reward, done, info

    def _handle_check(self, action: WebAction, checked: bool) -> Tuple[float, bool, Dict[str, Any]]:
        """Handle a checkbox toggle action."""
        info: Dict[str, Any] = {}
        reward = 0.1
        done = False
        info["checked"] = checked
        return reward, done, info

    # -------------------------------------------------------------------------
    # Page Content Generators
    # -------------------------------------------------------------------------

    def _default_state(self) -> Dict[str, Any]:
        """Return a default initial state."""
        return {
            "url": "https://example.com/home",
            "page_content": self._home_page_content(),
            "form_values": {},
            "is_logged_in": True,
            "theme": "light",
            "search_query": "",
        }

    def _task_to_state(self, task: WebTask) -> Dict[str, Any]:
        """Convert a WebTask to initial state."""
        state = copy.deepcopy(task.initial_state)
        if "url" not in state:
            state["url"] = task.target_url
        if "page_content" not in state:
            state["page_content"] = ""
        return state

    def _home_page_content(self) -> str:
        return """
        Welcome to Example.com
        =====================
        [Home] [Search] [Contact] [Settings] [Logout]
        
        Latest News
        -----------
        - Article 1: Tech Updates
        - Article 2: Web Development Tips
        
        User: john.doe@email.com (Logged in)
        """

    def _settings_page_content(self) -> str:
        return """
        Settings
        ========
        [Back to Home]
        
        Account Settings
        ----------------
        Theme: 
          [ ] Light  [x] Dark
          [Select theme: dropdown]
        
        Notifications:
          [x] Email notifications
          [ ] SMS notifications
        
        Language:
          [English v]
        
        [Save Settings] [Cancel]
        """

    def _contact_page_content(self) -> str:
        return """
        Contact Us
        ==========
        [Back to Home]
        
        Fill out the form below:
        
        Name: [________________]
        Email: [________________]
        Message: [________________________]
                       [________________________]
        
        [Submit] [Clear]
        """

    def _search_results_content(self) -> str:
        return """
        Search Results for: "open source llms"
        =====================================
        
        Found 3 results:
        
        1. Open Source LLMs: A Comprehensive Guide [Read More]
        2. Top 10 Open Source Language Models [Read More]  
        3. Building with Open Source LLMs [Read More]
        
        [Back to Search]
        """

    def _login_page_content(self) -> str:
        return """
        Login
        =====
        [Back to Home]
        
        Email: [________________]
        Password: [________________]
        
        [Login] [Forgot Password?]
        
        Don't have an account? [Sign Up]
        """

    def _get_clickable_elements(self) -> List[Dict[str, str]]:
        """Return list of available clickable elements."""
        url = self.current_url.lower()

        elements = []

        if "home" in url:
            elements = [
                {"id": "nav-home", "text": "Home", "type": "link"},
                {"id": "nav-search", "text": "Search", "type": "link"},
                {"id": "nav-contact", "text": "Contact", "type": "link"},
                {"id": "nav-settings", "text": "Settings", "type": "link"},
                {"id": "nav-logout", "text": "Logout", "type": "link"},
            ]
        elif "settings" in url:
            elements = [
                {"id": "btn-back", "text": "Back to Home", "type": "link"},
                {"id": "btn-save", "text": "Save Settings", "type": "button"},
                {"id": "theme-light", "text": "Light", "type": "radio"},
                {"id": "theme-dark", "text": "Dark", "type": "radio"},
                {"id": "theme-select", "text": "Select theme", "type": "dropdown"},
            ]
        elif "contact" in url:
            elements = [
                {"id": "btn-back", "text": "Back to Home", "type": "link"},
                {"id": "field-name", "text": "Name", "type": "input"},
                {"id": "field-email", "text": "Email", "type": "input"},
                {"id": "field-message", "text": "Message", "type": "textarea"},
                {"id": "btn-submit", "text": "Submit", "type": "button"},
                {"id": "btn-clear", "text": "Clear", "type": "button"},
            ]
        elif "search" in url:
            elements = [
                {"id": "btn-back", "text": "Back to Search", "type": "link"},
                {"id": "result-1", "text": "Open Source LLMs: A Comprehensive Guide", "type": "link"},
                {"id": "result-2", "text": "Top 10 Open Source Language Models", "type": "link"},
                {"id": "result-3", "text": "Building with Open Source LLMs", "type": "link"},
            ]
        elif "login" in url:
            elements = [
                {"id": "btn-back", "text": "Back to Home", "type": "link"},
                {"id": "field-email", "text": "Email", "type": "input"},
                {"id": "field-password", "text": "Password", "type": "input"},
                {"id": "btn-login", "text": "Login", "type": "button"},
            ]

        return elements

    def _get_available_forms(self) -> List[Dict[str, Any]]:
        """Return list of forms on current page."""
        url = self.current_url.lower()

        if "contact" in url:
            return [{
                "id": "contact-form",
                "action": "/contact",
                "method": "post",
                "fields": ["name", "email", "message"],
            }]
        elif "login" in url:
            return [{
                "id": "login-form",
                "action": "/login",
                "method": "post",
                "fields": ["email", "password"],
            }]
        elif "search" in url:
            return [{
                "id": "search-form",
                "action": "/search",
                "method": "get",
                "fields": ["q"],
            }]

        return []

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def get_state_snapshot(self) -> Dict[str, Any]:
        """Get a snapshot of the current state for debugging."""
        return {
            "url": self.current_url,
            "page_content": self.page_content,
            "form_values": self.form_values.copy(),
            "is_logged_in": self.is_logged_in,
            "theme": self.theme,
            "step_count": self._step_count,
            "history": copy.deepcopy(self._history),
        }

    def __repr__(self) -> str:
        return (
            f"MockWebEnv(url={self.current_url!r}, "
            f"logged_in={self.is_logged_in}, "
            f"theme={self.theme!r}, "
            f"step={self._step_count})"
        )


# ============================================================================
# WebArena-Lite Benchmark
# ============================================================================


class WebArenaLiteBenchmark:
    """Collection of WebArena-Lite / MiniWoB-style web navigation tasks.

    This benchmark provides a standardized set of tasks for evaluating
    agents on web interaction problems. Each task has:
    - Natural language description
    - Ground-truth milestone decomposition
    - Expected action sequence
    - Mock environment configuration

    Parameters
    ----------
    task_subset : str, optional
        Which task subset to load. Options:
        - "mini": 5 core tasks (default)
        - "full": All available tasks
        - "easy": Subset of easy tasks
        - "medium": Subset of medium tasks
        - "hard": Subset of hard tasks

    Attributes
    ----------
    tasks : Dict[str, WebTask]
        All loaded tasks indexed by task_id.

    Examples
    --------
    >>> benchmark = WebArenaLiteBenchmark(task_subset="mini")
    >>> task_ids = benchmark.list_tasks()
    >>> task = benchmark.get_task("change_theme_dark")
    >>> milestones = benchmark.get_milestone_for_task("change_theme_dark")
    """

    def __init__(self, task_subset: str = "mini") -> None:
        if task_subset not in ("mini", "full", "easy", "medium", "hard"):
            raise ValueError(
                f"task_subset must be one of 'mini', 'full', 'easy', 'medium', 'hard', "
                f"got {task_subset}"
            )
        self._task_subset = task_subset
        self._tasks: Dict[str, WebTask] = {}
        self._load_tasks()

    # -------------------------------------------------------------------------
    # Task Access
    # -------------------------------------------------------------------------

    def get_task(self, task_id: str) -> WebTask:
        """Get a task by its ID.

        Parameters
        ----------
        task_id : str
            The unique task identifier.

        Returns
        -------
        WebTask
            The task definition.

        Raises
        ------
        KeyError
            If task_id is not found in the benchmark.
        """
        if task_id not in self._tasks:
            available = list(self._tasks.keys())
            raise KeyError(
                f"Task '{task_id}' not found. Available tasks: {available}"
            )
        return self._tasks[task_id]

    def list_tasks(self) -> List[str]:
        """List all available task IDs in the current subset.

        Returns
        -------
        List[str]
            List of task IDs.
        """
        return list(self._tasks.keys())

    def get_tasks_by_difficulty(self, difficulty: str) -> List[WebTask]:
        """Get all tasks of a specific difficulty.

        Parameters
        ----------
        difficulty : str
            One of "easy", "medium", "hard".

        Returns
        -------
        List[WebTask]
            Tasks matching the specified difficulty.
        """
        if difficulty not in ("easy", "medium", "hard"):
            raise ValueError(f"difficulty must be 'easy', 'medium', or 'hard', got {difficulty}")
        return [t for t in self._tasks.values() if t.difficulty == difficulty]

    def get_milestone_for_task(self, task_id: str) -> List[Milestone]:
        """Get the milestone decomposition for a task.

        Parameters
        ----------
        task_id : str
            The task identifier.

        Returns
        -------
        List[Milestone]
            Ordered list of milestones for the task.
        """
        task = self.get_task(task_id)
        return task.milestones

    def create_env_for_task(self, task_id: str) -> MockWebEnv:
        """Create a MockWebEnv configured for a specific task.

        Parameters
        ----------
        task_id : str
            The task identifier.

        Returns
        -------
        MockWebEnv
            Environment ready to run the task.
        """
        task = self.get_task(task_id)
        return MockWebEnv(task=task)

    # -------------------------------------------------------------------------
    # Task Loading
    # -------------------------------------------------------------------------

    def _load_tasks(self) -> None:
        """Load the task definitions for the configured subset."""
        if self._task_subset == "mini":
            tasks_to_load = [
                self._task_change_theme,
                self._task_contact_form,
                self._task_search_llm,
                self._task_logout,
                self._task_search_date,
            ]
        elif self._task_subset == "easy":
            tasks_to_load = [self._task_logout]
        elif self._task_subset == "medium":
            tasks_to_load = [self._task_change_theme, self._task_contact_form]
        elif self._task_subset == "hard":
            tasks_to_load = [self._task_search_llm, self._task_search_date]
        else:  # full
            tasks_to_load = [
                self._task_change_theme,
                self._task_contact_form,
                self._task_search_llm,
                self._task_logout,
                self._task_search_date,
            ]

        for task in tasks_to_load:
            self._tasks[task.task_id] = task

    # -------------------------------------------------------------------------
    # Task Definitions
    # -------------------------------------------------------------------------

    @property
    def _task_change_theme(self) -> WebTask:
        """Task: Navigate to settings and change theme to dark."""
        return WebTask(
            task_id="change_theme_dark",
            description="Navigate to the settings page and change the theme to dark mode.",
            target_url="https://example.com/home",
            difficulty="medium",
            success_criteria="Theme is set to 'dark' on the settings page.",
            milestones=[
                Milestone(
                    id="m1",
                    description="Navigate to the settings page",
                    rubric="Agent has navigated to the settings page (URL contains 'settings')",
                    expected_output="URL: https://example.com/settings",
                ),
                Milestone(
                    id="m2",
                    description="Change theme selection to dark",
                    rubric="Theme is set to 'dark' in the settings",
                    expected_output="Theme setting: dark",
                ),
            ],
            expected_actions=[
                WebAction("click", "#nav-settings", label="Navigate to settings"),
                WebAction("select", "#theme-select", value="dark", label="Select dark theme"),
            ],
            initial_state={
                "url": "https://example.com/home",
                "page_content": "Welcome to Example.com [Home] [Search] [Contact] [Settings] [Logout]",
                "form_values": {},
                "is_logged_in": True,
                "theme": "light",
            },
        )

    @property
    def _task_contact_form(self) -> WebTask:
        """Task: Find contact form and fill in name and email."""
        return WebTask(
            task_id="contact_form_fill",
            description="Find the contact form and fill in name=Alice and email=alice@example.com.",
            target_url="https://example.com/home",
            difficulty="medium",
            success_criteria="Contact form is filled with name='Alice' and email='alice@example.com'.",
            milestones=[
                Milestone(
                    id="m1",
                    description="Navigate to the contact page",
                    rubric="Agent has navigated to the contact page (URL contains 'contact')",
                    expected_output="URL: https://example.com/contact",
                ),
                Milestone(
                    id="m2",
                    description="Fill in the name field with 'Alice'",
                    rubric="Name field contains 'Alice'",
                    expected_output="Form field: name=Alice",
                ),
                Milestone(
                    id="m3",
                    description="Fill in the email field with 'alice@example.com'",
                    rubric="Email field contains 'alice@example.com'",
                    expected_output="Form field: email=alice@example.com",
                ),
            ],
            expected_actions=[
                WebAction("click", "#nav-contact", label="Navigate to contact"),
                WebAction("type", "#field-name", value="Alice", label="Type name"),
                WebAction("type", "#field-email", value="alice@example.com", label="Type email"),
            ],
            initial_state={
                "url": "https://example.com/home",
                "page_content": "Welcome to Example.com [Home] [Search] [Contact] [Settings] [Logout]",
                "form_values": {},
                "is_logged_in": True,
                "theme": "light",
            },
        )

    @property
    def _task_search_llm(self) -> WebTask:
        """Task: Search for 'open source LLMs' and click the first result."""
        return WebTask(
            task_id="search_open_source_llm",
            description="Search for 'open source LLMs' and click the first result.",
            target_url="https://example.com/home",
            difficulty="hard",
            success_criteria="Agent clicked on the first search result about open source LLMs.",
            milestones=[
                Milestone(
                    id="m1",
                    description="Navigate to the search page",
                    rubric="Agent has navigated to the search functionality",
                    expected_output="URL contains 'search'",
                ),
                Milestone(
                    id="m2",
                    description="Enter 'open source LLMs' in search query",
                    rubric="Search query field contains 'open source LLMs'",
                    expected_output="Search query: open source LLMs",
                ),
                Milestone(
                    id="m3",
                    description="Submit search or navigate to results",
                    rubric="Search results are displayed",
                    expected_output="Results page with links",
                ),
                Milestone(
                    id="m4",
                    description="Click on the first result",
                    rubric="Agent clicked on the first search result link",
                    expected_output="Navigated to article page",
                ),
            ],
            expected_actions=[
                WebAction("click", "#nav-search", label="Navigate to search"),
                WebAction("type", "#search-field", value="open source LLMs", label="Enter search query"),
                WebAction("submit", "#search-form", label="Submit search"),
                WebAction("click", "#result-1", label="Click first result"),
            ],
            initial_state={
                "url": "https://example.com/home",
                "page_content": "Welcome to Example.com [Home] [Search] [Contact] [Settings] [Logout]",
                "form_values": {},
                "is_logged_in": True,
                "theme": "light",
                "search_query": "",
            },
        )

    @property
    def _task_logout(self) -> WebTask:
        """Task: Log out from the current session."""
        return WebTask(
            task_id="logout_session",
            description="Log out from the current session.",
            target_url="https://example.com/home",
            difficulty="easy",
            success_criteria="User is logged out and redirected to login page.",
            milestones=[
                Milestone(
                    id="m1",
                    description="Click the logout button",
                    rubric="Agent clicked on the logout button/link",
                    expected_output="is_logged_in=False, URL contains 'login'",
                ),
            ],
            expected_actions=[
                WebAction("click", "#nav-logout", label="Click logout"),
            ],
            initial_state={
                "url": "https://example.com/home",
                "page_content": "Welcome to Example.com [Home] [Search] [Contact] [Settings] [Logout]",
                "form_values": {},
                "is_logged_in": True,
                "theme": "light",
            },
        )

    @property
    def _task_search_date(self) -> WebTask:
        """Task: Submit a search query with the current date."""
        return WebTask(
            task_id="search_with_date",
            description=f"Submit a search query with today's date ({date.today()}).",
            target_url="https://example.com/search",
            difficulty="hard",
            success_criteria="Search query includes the current date.",
            milestones=[
                Milestone(
                    id="m1",
                    description="Enter the current date in the search query",
                    rubric="Search query field contains today's date",
                    expected_output=f"Search query contains: {date.today()}",
                ),
                Milestone(
                    id="m2",
                    description="Submit the search",
                    rubric="Search was submitted successfully",
                    expected_output="Search results page displayed",
                ),
            ],
            expected_actions=[
                WebAction("type", "#search-field", value=str(date.today()), label="Enter today's date"),
                WebAction("submit", "#search-form", label="Submit search"),
            ],
            initial_state={
                "url": "https://example.com/search",
                "page_content": "Search [________________] [Search]",
                "form_values": {},
                "is_logged_in": True,
                "theme": "light",
                "search_query": "",
            },
        )

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Get benchmark statistics.

        Returns
        -------
        Dict[str, Any]
            Statistics including task count, difficulty distribution, etc.
        """
        tasks = list(self._tasks.values())
        easy = sum(1 for t in tasks if t.difficulty == "easy")
        medium = sum(1 for t in tasks if t.difficulty == "medium")
        hard = sum(1 for t in tasks if t.difficulty == "hard")

        return {
            "subset": self._task_subset,
            "total_tasks": len(tasks),
            "easy_count": easy,
            "medium_count": medium,
            "hard_count": hard,
            "task_ids": list(self._tasks.keys()),
        }

    def __repr__(self) -> str:
        return (
            f"WebArenaLiteBenchmark("
            f"subset={self._task_subset!r}, "
            f"tasks={len(self._tasks)})"
        )

    def __len__(self) -> int:
        return len(self._tasks)

    def __iter__(self):
        return iter(self._tasks.values())

    def __getitem__(self, task_id: str) -> WebTask:
        return self.get_task(task_id)
