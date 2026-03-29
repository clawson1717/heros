"""WebArena-Lite / MiniWoB-style Benchmark for HeRoS Evaluation.

Defines web navigation tasks with milestone rubrics and a simulated
DOM environment (MockWebEnv) for evaluation without a real browser.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# WebTask definition
# ---------------------------------------------------------------------------


@dataclass
class WebTask:
    """A single web navigation task with milestone rubrics.

    Attributes
    ----------
    task_id : str
        Unique identifier for this task (e.g., "click_button_sequence").
    description : str
        Natural language description of the task goal.
    initial_url : str
        URL the task starts at.
    gold_action_sequence : List[str]
        The canonical correct action sequence to complete the task.
        Each action is a string like "click[id=btn1]" or "type[id=input1,text=hello]".
    milestone_rubrics : List[Dict[str, str]]
        List of milestone rubrics, each containing:
        - "description": what this milestone checks
        - "rubric": pass/fail criteria text
        - "expected_output": expected DOM state or observation after milestone
    max_steps : int
        Maximum number of agent steps allowed for this task.
    """

    task_id: str
    description: str
    initial_url: str
    gold_action_sequence: List[str] = field(default_factory=list)
    milestone_rubrics: List[Dict[str, str]] = field(default_factory=list)
    max_steps: int = 20


# ---------------------------------------------------------------------------
# MockWebEnv — simulated DOM environment
# ---------------------------------------------------------------------------


class MockWebEnv:
    """Simulated DOM environment for web navigation evaluation.

    Provides a simple state machine that models a web page with:
    - current_url: the current URL/path
    - page_state: dict of element states (visibility, values, checked status)
    - visited_links: history of navigation

    Supported actions:
    - click(element_id): click an element by ID
    - type(element_id, text): type text into an input field
    - navigate(url): navigate to a URL
    - submit(): submit the current form
    - go_back(): navigate back in history
    - check(option): check a checkbox or select an option
    - uncheck(option): uncheck a checkbox

    Parameters
    ----------
    initial_url : str, optional
        Starting URL. Defaults to "https://example.com/".
    page_config : Dict[str, Any], optional
        Initial page configuration dict. If None, uses a default landing page.

    Examples
    --------
    >>> env = MockWebEnv(initial_url="https://example.com/login")
    >>> obs = env.reset()
    >>> print(obs)
    "URL: https://example.com/login | Elements: [btn_submit, input_username, ..."
    >>> obs, reward, done, info = env.step("click[id=btn_submit]")
    """

    def __init__(
        self,
        initial_url: str = "https://example.com/",
        page_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._initial_url = initial_url
        self._page_config = page_config or self._default_page_config()

        # Core state
        self._current_url: str = self._initial_url
        self._page_state: Dict[str, Any] = {}
        self._visited_links: List[str] = []
        self._nav_history: List[str] = []
        self._step_count: int = 0
        self._form_data: Dict[str, str] = {}
        self._submitted: bool = False

        # Initialize page state from config
        self._init_page_from_config()

    def _default_page_config(self) -> Dict[str, Any]:
        """Return a default page configuration."""
        return {
            "title": "Example Domain",
            "elements": [
                {"id": "heading", "type": "text", "text": "Welcome", "visible": True},
                {"id": "btn_start", "type": "button", "text": "Get Started", "visible": True},
                {"id": "link_about", "type": "link", "text": "About", "href": "/about", "visible": True},
            ],
            "url": self._initial_url,
        }

    def _init_page_from_config(self) -> None:
        """Initialize page state from configuration."""
        self._page_state = {}
        for elem in self._page_config.get("elements", []):
            eid = elem["id"]
            self._page_state[eid] = {
                "type": elem.get("type", "unknown"),
                "text": elem.get("text", ""),
                "visible": elem.get("visible", True),
                "href": elem.get("href", ""),
                "value": elem.get("value", ""),
                "checked": elem.get("checked", False),
                "disabled": elem.get("disabled", False),
                "options": elem.get("options", []),
                "selected": elem.get("selected", ""),
            }

    def _config_for_url(self, url: str) -> Optional[Dict[str, Any]]:
        """Return page config for a given URL path."""
        configs = {
            "https://example.com/": {
                "title": "Home",
                "elements": [
                    {"id": "heading", "type": "text", "text": "Welcome to Example", "visible": True},
                    {"id": "btn_start", "type": "button", "text": "Get Started", "visible": True},
                    {"id": "link_about", "type": "link", "text": "About", "href": "/about", "visible": True},
                    {"id": "link_settings", "type": "link", "text": "Settings", "href": "/settings", "visible": True},
                ],
                "url": "https://example.com/",
            },
            "https://example.com/about": {
                "title": "About",
                "elements": [
                    {"id": "heading", "type": "text", "text": "About Us", "visible": True},
                    {"id": "link_home", "type": "link", "text": "Home", "href": "/", "visible": True},
                    {"id": "link_contact", "type": "link", "text": "Contact", "href": "/contact", "visible": True},
                ],
                "url": "https://example.com/about",
            },
            "https://example.com/login": {
                "title": "Login",
                "elements": [
                    {"id": "heading", "type": "text", "text": "Login", "visible": True},
                    {"id": "input_username", "type": "input", "text": "", "visible": True, "placeholder": "Username"},
                    {"id": "input_password", "type": "input", "text": "", "visible": True, "placeholder": "Password", "secret": True},
                    {"id": "btn_submit", "type": "button", "text": "Submit", "visible": True},
                    {"id": "link_register", "type": "link", "text": "Register", "href": "/register", "visible": True},
                ],
                "url": "https://example.com/login",
            },
            "https://example.com/register": {
                "title": "Register",
                "elements": [
                    {"id": "heading", "type": "text", "text": "Create Account", "visible": True},
                    {"id": "input_email", "type": "input", "text": "", "visible": True, "placeholder": "Email"},
                    {"id": "input_username", "type": "input", "text": "", "visible": True, "placeholder": "Username"},
                    {"id": "input_password", "type": "input", "text": "", "visible": True, "placeholder": "Password", "secret": True},
                    {"id": "btn_register", "type": "button", "text": "Register", "visible": True},
                ],
                "url": "https://example.com/register",
            },
            "https://example.com/contact": {
                "title": "Contact",
                "elements": [
                    {"id": "heading", "type": "text", "text": "Contact Us", "visible": True},
                    {"id": "input_name", "type": "input", "text": "", "visible": True, "placeholder": "Your Name"},
                    {"id": "input_email", "type": "input", "text": "", "visible": True, "placeholder": "Email"},
                    {"id": "input_message", "type": "textarea", "text": "", "visible": True, "placeholder": "Message"},
                    {"id": "btn_send", "type": "button", "text": "Send", "visible": True},
                ],
                "url": "https://example.com/contact",
            },
            "https://example.com/settings": {
                "title": "Settings",
                "elements": [
                    {"id": "heading", "type": "text", "text": "Settings", "visible": True},
                    {"id": "toggle_notifications", "type": "checkbox", "text": "Enable Notifications", "visible": True, "checked": False},
                    {"id": "toggle_dark_mode", "type": "checkbox", "text": "Dark Mode", "visible": True, "checked": False},
                    {"id": "select_language", "type": "select", "text": "Language", "visible": True, "options": ["English", "Spanish", "French"], "selected": "English"},
                    {"id": "btn_save", "type": "button", "text": "Save", "visible": True},
                ],
                "url": "https://example.com/settings",
            },
            "https://example.com/dashboard": {
                "title": "Dashboard",
                "elements": [
                    {"id": "heading", "type": "text", "text": "Dashboard", "visible": True},
                    {"id": "link_profile", "type": "link", "text": "Profile", "href": "/profile", "visible": True},
                    {"id": "link_inbox", "type": "link", "text": "Inbox", "href": "/inbox", "visible": True},
                    {"id": "btn_logout", "type": "button", "text": "Logout", "visible": True},
                ],
                "url": "https://example.com/dashboard",
            },
            "https://example.com/profile": {
                "title": "Profile",
                "elements": [
                    {"id": "heading", "type": "text", "text": "Your Profile", "visible": True},
                    {"id": "input_bio", "type": "textarea", "text": "", "visible": True, "placeholder": "Bio"},
                    {"id": "btn_update", "type": "button", "text": "Update", "visible": True},
                ],
                "url": "https://example.com/profile",
            },
        }

        # Handle relative paths
        if url.startswith("/"):
            url = "https://example.com" + url

        return configs.get(url)

    # -------------------------------------------------------------------------
    # Core environment interface
    # -------------------------------------------------------------------------

    def reset(self) -> str:
        """Reset the environment to initial state.

        Returns
        -------
        str
            Initial observation string describing the page.
        """
        self._current_url = self._initial_url
        self._step_count = 0
        self._visited_links = [self._initial_url]
        self._nav_history = []
        self._form_data = {}
        self._submitted = False

        config = self._config_for_url(self._initial_url)
        if config:
            self._page_config = config
            self._init_page_from_config()
        else:
            self._init_page_from_config()

        return self._build_observation()

    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        """Execute a web action.

        Parameters
        ----------
        action : str
            Action string like "click[id=btn_submit]" or "type[id=input_username,text=hello]".

        Returns
        -------
        Tuple[str, float, bool, Dict[str, Any]]
            observation, reward, done, info
        """
        self._step_count += 1
        info: Dict[str, Any] = {"action": action, "step": self._step_count}

        # Parse action
        action_lower = action.strip().lower()

        if action_lower.startswith("click"):
            obs = self._do_click(action)
        elif action_lower.startswith("type"):
            obs = self._do_type(action)
        elif action_lower.startswith("navigate"):
            obs = self._do_navigate(action)
        elif action_lower.startswith("submit"):
            obs = self._do_submit()
        elif action_lower.startswith("go_back"):
            obs = self._do_go_back()
        elif action_lower.startswith("check"):
            obs = self._do_check(action)
        elif action_lower.startswith("uncheck"):
            obs = self._do_uncheck(action)
        else:
            obs = self._build_observation()
            info["error"] = f"Unknown action: {action}"

        # Check if done (max steps or task complete)
        done = self._step_count >= 50  # Hard cap

        # Simple reward: 1.0 if action was valid, 0.0 otherwise
        reward = 0.0 if info.get("error") else 0.1

        return obs, reward, done, info

    def _build_observation(self) -> str:
        """Build the observation string describing current page state."""
        parts = [f"URL: {self._current_url}"]

        visible_elements = []
        for eid, state in self._page_state.items():
            if state.get("visible", True):
                etype = state.get("type", "unknown")
                text = state.get("text", "")
                checked = state.get("checked", False)
                value = state.get("value", "")

                if etype == "input":
                    display = f"{eid}(input, value='{value}')"
                elif etype == "checkbox":
                    display = f"{eid}(checkbox, checked={checked})"
                elif etype == "select":
                    display = f"{eid}(select, selected={state.get('selected', '')})"
                elif etype == "button":
                    display = f"{eid}(button, text='{text}')"
                elif etype == "link":
                    display = f"{eid}(link, text='{text}', href='{state.get('href', '')}')"
                elif etype == "textarea":
                    display = f"{eid}(textarea, value='{value[:50]}')"
                else:
                    display = f"{eid}({etype}, text='{text}')"
                visible_elements.append(display)

        if visible_elements:
            parts.append("Elements: " + ", ".join(visible_elements))
        else:
            parts.append("Elements: [none]")

        parts.append(f"FormData: {self._form_data}")
        parts.append(f"Submitted: {self._submitted}")

        return " | ".join(parts)

    # -------------------------------------------------------------------------
    # Action handlers
    # -------------------------------------------------------------------------

    def _parse_element_action(self, action: str, param_name: str = "id") -> Optional[str]:
        """Extract element ID from action string like 'click[id=btn_submit]'."""
        import re
        match = re.search(rf"{param_name}=([a-zA-Z0-9_]+)", action)
        return match.group(1) if match else None

    def _parse_type_action(self, action: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract element ID and text from 'type[id=input1,text=hello]'."""
        import re
        id_match = re.search(r"id=([a-zA-Z0-9_]+)", action)
        text_match = re.search(r"text=([^]]+)", action)
        eid = id_match.group(1) if id_match else None
        text = text_match.group(1).strip() if text_match else None
        return eid, text

    def _do_click(self, action: str) -> str:
        """Handle click action."""
        eid = self._parse_element_action(action, "id")
        if not eid:
            return self._build_observation()

        if eid not in self._page_state:
            return self._build_observation()

        state = self._page_state[eid]
        etype = state.get("type", "")

        # Handle navigation links
        if etype == "link":
            href = state.get("href", "")
            if href:
                target = href if href.startswith("http") else "https://example.com" + href
                self._nav_history.append(self._current_url)
                self._current_url = target
                self._visited_links.append(target)
                config = self._config_for_url(target)
                if config:
                    self._page_config = config
                    self._init_page_from_config()
                else:
                    self._page_state = {}

        # Handle buttons — some trigger navigation or state changes
        elif etype == "button":
            # Button-specific side effects based on ID
            if eid in ("btn_start", "btn_submit", "btn_register", "btn_send", "btn_save", "btn_update"):
                # Mark form as submitted for submit buttons
                if eid in ("btn_submit", "btn_register", "btn_send"):
                    self._submitted = True
                # Navigation buttons
                if eid == "btn_start":
                    self._nav_history.append(self._current_url)
                    self._current_url = "https://example.com/dashboard"
                    config = self._config_for_url(self._current_url)
                    if config:
                        self._page_config = config
                        self._init_page_from_config()

        return self._build_observation()

    def _do_type(self, action: str) -> str:
        """Handle type action."""
        eid, text = self._parse_type_action(action)
        if not eid or text is None:
            return self._build_observation()

        if eid in self._page_state:
            self._page_state[eid]["value"] = text
            self._form_data[eid] = text

        return self._build_observation()

    def _do_navigate(self, action: str) -> str:
        """Handle navigate action."""
        import re
        match = re.search(r"navigate\[url=([^\]]+)\]", action)
        if not match:
            return self._build_observation()

        url = match.group(1).strip()
        if not url.startswith("http"):
            url = "https://example.com/" + url.lstrip("/")

        self._nav_history.append(self._current_url)
        self._current_url = url
        self._visited_links.append(url)

        config = self._config_for_url(url)
        if config:
            self._page_config = config
            self._init_page_from_config()
        else:
            self._page_state = {}

        return self._build_observation()

    def _do_submit(self) -> str:
        """Handle submit action."""
        self._submitted = True
        return self._build_observation()

    def _do_go_back(self) -> str:
        """Handle go_back action."""
        if self._nav_history:
            prev = self._nav_history.pop()
            self._current_url = prev
            config = self._config_for_url(prev)
            if config:
                self._page_config = config
                self._init_page_from_config()
            else:
                self._page_state = {}
        return self._build_observation()

    def _do_check(self, action: str) -> str:
        """Handle check action."""
        eid = self._parse_element_action(action, "option")
        if not eid:
            # Try to find a checkbox by ID
            eid = self._parse_element_action(action, "id")

        if eid and eid in self._page_state:
            state = self._page_state[eid]
            if state.get("type") == "checkbox":
                state["checked"] = True
            elif state.get("type") == "select":
                state["selected"] = eid

        return self._build_observation()

    def _do_uncheck(self, action: str) -> str:
        """Handle uncheck action."""
        eid = self._parse_element_action(action, "option")
        if not eid:
            eid = self._parse_element_action(action, "id")

        if eid and eid in self._page_state:
            state = self._page_state[eid]
            if state.get("type") == "checkbox":
                state["checked"] = False

        return self._build_observation()

    # -------------------------------------------------------------------------
    # State accessors
    # -------------------------------------------------------------------------

    @property
    def current_url(self) -> str:
        """Current URL."""
        return self._current_url

    @property
    def page_state(self) -> Dict[str, Any]:
        """Current page element state."""
        return dict(self._page_state)

    @property
    def visited_links(self) -> List[str]:
        """History of visited URLs."""
        return list(self._visited_links)

    @property
    def step_count(self) -> int:
        """Number of steps taken."""
        return self._step_count

    @property
    def is_submitted(self) -> bool:
        """Whether the current form has been submitted."""
        return self._submitted

    def get_element_state(self, element_id: str) -> Optional[Dict[str, Any]]:
        """Get the state of a specific element."""
        return dict(self._page_state.get(element_id, {}))

    def __repr__(self) -> str:
        return f"MockWebEnv(url={self._current_url}, steps={self._step_count})"


# ---------------------------------------------------------------------------
# WebArenaLiteBenchmark — task subsets
# ---------------------------------------------------------------------------


class WebArenaLiteBenchmark:
    """WebArena-Lite task benchmark with configurable difficulty subsets.

    Provides pre-defined web navigation tasks at various difficulty levels:
    - mini: 5 tasks (core interaction patterns)
    - easy: 8 tasks
    - medium: 12 tasks
    - full: 20 tasks

    Attributes
    ----------
    SUBSETS : Dict[str, int]
        Mapping of subset name to number of tasks.
    """

    SUBSETS = {
        "mini": 5,
        "easy": 8,
        "medium": 12,
        "full": 20,
    }

    # All available tasks
    _ALL_TASKS: List[WebTask] = []

    def __init__(self, subset: str = "mini") -> None:
        if subset not in self.SUBSETS:
            raise ValueError(
                f"Unknown subset '{subset}'. "
                f"Available: {list(self.SUBSETS.keys())}"
            )
        self._subset = subset
        self._tasks: List[WebTask] = []
        self._init_tasks()
        self._tasks = self._tasks[: self.SUBSETS[subset]]

    def _init_tasks(self) -> None:
        """Initialize all tasks."""
        self._tasks = [
            # Task 1: click_button_sequence
            WebTask(
                task_id="click_button_sequence",
                description="Navigate to the website and click the 'Get Started' button to reach the dashboard.",
                initial_url="https://example.com/",
                gold_action_sequence=[
                    "click[id=btn_start]",
                ],
                milestone_rubrics=[
                    {
                        "description": "Clicked the Get Started button",
                        "rubric": "Agent clicked the btn_start element",
                        "expected_output": "Dashboard page loaded",
                    },
                    {
                        "description": "Reached dashboard",
                        "rubric": "Current URL is /dashboard or contains 'dashboard'",
                        "expected_output": "URL contains 'dashboard'",
                    },
                ],
                max_steps=10,
            ),
            # Task 2: fill_login_form
            WebTask(
                task_id="fill_login_form",
                description="Go to the login page, enter 'testuser' as username and 'secret123' as password, then submit.",
                initial_url="https://example.com/",
                gold_action_sequence=[
                    "navigate[url=/login]",
                    "type[id=input_username,text=testuser]",
                    "type[id=input_password,text=secret123]",
                    "click[id=btn_submit]",
                ],
                milestone_rubrics=[
                    {
                        "description": "Navigated to login page",
                        "rubric": "Current URL contains 'login'",
                        "expected_output": "URL contains 'login'",
                    },
                    {
                        "description": "Entered username",
                        "rubric": "input_username has value 'testuser'",
                        "expected_output": "Username field filled",
                    },
                    {
                        "description": "Entered password",
                        "rubric": "input_password has value 'secret123'",
                        "expected_output": "Password field filled",
                    },
                    {
                        "description": "Form submitted",
                        "rubric": "submitted flag is True",
                        "expected_output": "Form submitted successfully",
                    },
                ],
                max_steps=15,
            ),
            # Task 3: navigate_menu
            WebTask(
                task_id="navigate_menu",
                description="Start at the home page, navigate to About page, then to Contact page.",
                initial_url="https://example.com/",
                gold_action_sequence=[
                    "click[id=link_about]",
                    "click[id=link_contact]",
                ],
                milestone_rubrics=[
                    {
                        "description": "Navigated to About page",
                        "rubric": "Current URL contains 'about'",
                        "expected_output": "URL contains 'about'",
                    },
                    {
                        "description": "Navigated to Contact page",
                        "rubric": "Current URL contains 'contact'",
                        "expected_output": "URL contains 'contact'",
                    },
                ],
                max_steps=10,
            ),
            # Task 4: fill_contact_form
            WebTask(
                task_id="fill_contact_form",
                description="Navigate to the Contact page and fill out the contact form with name 'Alice', email 'alice@example.com', and message 'Hello!', then send.",
                initial_url="https://example.com/",
                gold_action_sequence=[
                    "click[id=link_contact]",
                    "type[id=input_name,text=Alice]",
                    "type[id=input_email,text=alice@example.com]",
                    "type[id=input_message,text=Hello!]",
                    "click[id=btn_send]",
                ],
                milestone_rubrics=[
                    {
                        "description": "Navigated to Contact page",
                        "rubric": "Current URL contains 'contact'",
                        "expected_output": "URL contains 'contact'",
                    },
                    {
                        "description": "Filled name field",
                        "rubric": "input_name has value 'Alice'",
                        "expected_output": "Name field filled",
                    },
                    {
                        "description": "Filled email field",
                        "rubric": "input_email has value 'alice@example.com'",
                        "expected_output": "Email field filled",
                    },
                    {
                        "description": "Filled message field",
                        "rubric": "input_message contains 'Hello!'",
                        "expected_output": "Message field filled",
                    },
                    {
                        "description": "Form sent",
                        "rubric": "submitted flag is True",
                        "expected_output": "Form submitted",
                    },
                ],
                max_steps=20,
            ),
            # Task 5: toggle_settings
            WebTask(
                task_id="toggle_settings",
                description="Navigate to Settings, enable notifications, enable dark mode, and save settings.",
                initial_url="https://example.com/",
                gold_action_sequence=[
                    "click[id=link_settings]",
                    "check[id=toggle_notifications]",
                    "check[id=toggle_dark_mode]",
                    "click[id=btn_save]",
                ],
                milestone_rubrics=[
                    {
                        "description": "Navigated to Settings page",
                        "rubric": "Current URL contains 'settings'",
                        "expected_output": "URL contains 'settings'",
                    },
                    {
                        "description": "Notifications enabled",
                        "rubric": "toggle_notifications checked is True",
                        "expected_output": "Notifications checkbox enabled",
                    },
                    {
                        "description": "Dark mode enabled",
                        "rubric": "toggle_dark_mode checked is True",
                        "expected_output": "Dark mode checkbox enabled",
                    },
                    {
                        "description": "Settings saved",
                        "rubric": "submitted flag is True or btn_save was clicked",
                        "expected_output": "Settings saved",
                    },
                ],
                max_steps=15,
            ),
            # Task 6: register_account
            WebTask(
                task_id="register_account",
                description="Navigate to the Register page and create an account with email 'newuser@test.com', username 'newuser', password 'Pass123!'.",
                initial_url="https://example.com/",
                gold_action_sequence=[
                    "navigate[url=/register]",
                    "type[id=input_email,text=newuser@test.com]",
                    "type[id=input_username,text=newuser]",
                    "type[id=input_password,text=Pass123!]",
                    "click[id=btn_register]",
                ],
                milestone_rubrics=[
                    {
                        "description": "Navigated to Register page",
                        "rubric": "Current URL contains 'register'",
                        "expected_output": "URL contains 'register'",
                    },
                    {
                        "description": "Entered email",
                        "rubric": "input_email has value 'newuser@test.com'",
                        "expected_output": "Email entered",
                    },
                    {
                        "description": "Entered username",
                        "rubric": "input_username has value 'newuser'",
                        "expected_output": "Username entered",
                    },
                    {
                        "description": "Entered password",
                        "rubric": "input_password has value 'Pass123!'",
                        "expected_output": "Password entered",
                    },
                    {
                        "description": "Registration submitted",
                        "rubric": "submitted flag is True",
                        "expected_output": "Account created",
                    },
                ],
                max_steps=20,
            ),
            # Task 7: back_navigation
            WebTask(
                task_id="back_navigation",
                description="Start at home, navigate to About, then go back to home.",
                initial_url="https://example.com/",
                gold_action_sequence=[
                    "click[id=link_about]",
                    "go_back",
                ],
                milestone_rubrics=[
                    {
                        "description": "Navigated to About",
                        "rubric": "'about' in current_url",
                        "expected_output": "At About page",
                    },
                    {
                        "description": "Went back to home",
                        "rubric": "current_url is '/' or 'https://example.com/'",
                        "expected_output": "Back at home",
                    },
                ],
                max_steps=10,
            ),
            # Task 8: multi_form_interaction
            WebTask(
                task_id="multi_form_interaction",
                description="Go to login page and enter credentials, then go to register page and enter details, without submitting either.",
                initial_url="https://example.com/",
                gold_action_sequence=[
                    "navigate[url=/login]",
                    "type[id=input_username,text=alice]",
                    "type[id=input_password,text=pass123]",
                    "navigate[url=/register]",
                    "type[id=input_email,text=alice@test.com]",
                    "type[id=input_username,text=alice]",
                    "type[id=input_password,text=pass123]",
                ],
                milestone_rubrics=[
                    {
                        "description": "Filled login username",
                        "rubric": "input_username value contains 'alice'",
                        "expected_output": "Login username filled",
                    },
                    {
                        "description": "Navigated to register",
                        "rubric": "'register' in current_url",
                        "expected_output": "At register page",
                    },
                    {
                        "description": "Filled register email",
                        "rubric": "input_email value contains 'alice@test.com'",
                        "expected_output": "Register email filled",
                    },
                ],
                max_steps=20,
            ),
        ]

        # Add more tasks to reach 'full' subset size
        if len(self._tasks) < 20:
            for i in range(len(self._tasks), 20):
                task_id = f"task_{i+1}"
                self._tasks.append(
                    WebTask(
                        task_id=task_id,
                        description=f"Web navigation task number {i+1}",
                        initial_url="https://example.com/",
                        gold_action_sequence=[],
                        milestone_rubrics=[
                            {
                                "description": f"Completed task {i+1}",
                                "rubric": "Task completed",
                                "expected_output": "Task done",
                            }
                        ],
                        max_steps=10,
                    )
                )

    @property
    def tasks(self) -> List[WebTask]:
        """List of tasks in this benchmark subset."""
        return list(self._tasks)

    @property
    def subset(self) -> str:
        """The current subset name."""
        return self._subset

    def get_task(self, task_id: str) -> Optional[WebTask]:
        """Get a specific task by ID."""
        for task in self._tasks:
            if task.task_id == task_id:
                return task
        return None

    def create_env_factory(self) -> Callable[[], MockWebEnv]:
        """Create a factory that produces fresh MockWebEnv instances.

        Returns
        -------
        Callable[[], MockWebEnv]
            A factory function that creates a new MockWebEnv.
        """
        def factory() -> MockWebEnv:
            return MockWebEnv()
        return factory

    def __repr__(self) -> str:
        return f"WebArenaLiteBenchmark(subset={self._subset}, tasks={len(self._tasks)})"
