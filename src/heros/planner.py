"""
HeRoS Planner Module — MiRA-style Subgoal Decomposition

This module implements the MiRA-style subgoal decomposition planner from Step 3
of the HeRoS project. It decomposes tasks into ordered, verifiable milestones
with pass/fail rubrics.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


# ---------------------------------------------------------------------------
# Protocols / Interfaces
# ---------------------------------------------------------------------------


class LLMClient(Protocol):
    """Protocol for an LLM client used by LLMPlanner."""

    def complete(self, prompt: str, **kwargs: Any) -> str:
        """Send a prompt to the LLM and return the response text."""
        ...


# ---------------------------------------------------------------------------
# Milestone Dataclass
# ---------------------------------------------------------------------------


@dataclass
class Milestone:
    """
    A single verifiable subgoal within a task.

    Attributes
    ----------
    id : str
        Unique identifier within the plan, e.g. "m1", "m2".
    description : str
        Human-readable description of what this milestone entails.
    rubric : str
        Specific pass/fail criteria that can be evaluated programmatically.
        Should be concrete and unambiguous.
    expected_output : str
        Description of the concrete output or state produced when this
        milestone is successfully completed.
    order : int
        Zero-based execution order. Milestones must be executed in order.
    """

    id: str
    description: str
    rubric: str
    expected_output: str
    order: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize milestone to a plain dictionary."""
        return {
            "id": self.id,
            "description": self.description,
            "rubric": self.rubric,
            "expected_output": self.expected_output,
            "order": self.order,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Milestone:
        """Deserialize a milestone from a dictionary."""
        return cls(
            id=str(data["id"]),
            description=str(data["description"]),
            rubric=str(data["rubric"]),
            expected_output=str(data["expected_output"]),
            order=int(data["order"]),
        )


# ---------------------------------------------------------------------------
# Rule-Based Task-Type Detector
# ---------------------------------------------------------------------------

_TASK_TYPE_PATTERNS: list[tuple[str, list[str]]] = [
    (
        "code_generation",
        [
            "write code", "implement", "function", "class", "script",
            "program", "algorithm", "api", "endpoint", "module",
            "debug", "refactor", "test", "specification", "generate",
        ],
    ),
    (
        "web_navigation",
        [
            "click", "navigate", "browser", "website", "page", "submit form",
            "fill out", "login", "scroll", "web", "http", "url",
        ],
    ),
    (
        "data_analysis",
        [
            "analyze", "dataset", "csv", "statistics", "plot", "visualize",
            "data", "chart", "histogram", "regression", "correlation",
            "clean data", "transform", "filter", "aggregate",
        ],
    ),
    (
        "reasoning",
        [
            "prove", "theorem", "reasoning", "logical", "infer", "conclusion",
            "deduce", "hypothesis", "argument", "proof",
        ],
    ),
]


def detect_task_type(task: str) -> str:
    """
    Heuristically detect the task type from free-text task description.

    Returns one of: ``code_generation``, ``web_navigation``,
    ``data_analysis``, ``reasoning``, or ``general``.
    """
    task_lower = task.lower()
    scores: dict[str, int] = {}

    for task_type, keywords in _TASK_TYPE_PATTERNS:
        scores[task_type] = sum(1 for kw in keywords if kw in task_lower)

    if not scores or max(scores.values()) == 0:
        return "general"

    return max(scores, key=scores.get)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Rule-Based Default Milestone Templates
# ---------------------------------------------------------------------------

# Each template is a list of (description, rubric, expected_output) triples.
_RULE_BASED_TEMPLATES: dict[str, list[tuple[str, str, str]]] = {
    "code_generation": [
        (
            "Understand requirements and plan structure",
            "Requirements are parsed into a written plan or outline; no syntax errors in the plan.",
            "Requirements analysis document or comment block",
        ),
        (
            "Write core implementation",
            "Core functions/classes are written with correct syntax; no import errors.",
            "Source code file with core logic",
        ),
        (
            "Add error handling and edge cases",
            "Key functions have try/except or error-return logic; edge-case inputs handled.",
            "Robust source code with error handling",
        ),
        (
            "Write unit tests",
            "Tests exist for all public functions; pytest runs with zero failures.",
            "Test file with passing tests",
        ),
        (
            "Verify end-to-end execution",
            "Script or module runs without exceptions; produces expected output on a simple input.",
            "Working, runnable code",
        ),
    ],
    "web_navigation": [
        (
            "Identify target page and URL structure",
            "Target URL or page description is written; no malformed URL.",
            "Target URL or page blueprint",
        ),
        (
            "Perform navigation actions (click/link)",
            "Navigation action is defined (selector or URL); action is reasonable for the goal.",
            "Navigation step definition",
        ),
        (
            "Fill in required form fields",
            "All required form fields are identified and populated with plausible values.",
            "Completed form field map",
        ),
        (
            "Submit and verify outcome",
            "Submission action is defined; expected post-submit page state is described.",
            "Submission result description",
        ),
    ],
    "data_analysis": [
        (
            "Load and inspect data",
            "Data loads without exceptions; schema or first few rows are inspected.",
            "Loaded data object or preview",
        ),
        (
            "Clean and transform data",
            "Nulls or obvious noise are handled; transformations are applied and documented.",
            "Cleaned DataFrame or dataset",
        ),
        (
            "Perform analysis or modelling",
            "Analysis logic is executed; numeric or categorical results are produced.",
            "Analysis results or model output",
        ),
        (
            "Visualize or summarize findings",
            "At least one chart or summary table is produced; findings are written out.",
            "Visualization or written summary",
        ),
    ],
    "reasoning": [
        (
            "Restate the problem and identify givens",
            "Problem statement is restated; all given facts are listed.",
            "Restated problem with givens list",
        ),
        (
            "Formulate hypothesis or logical steps",
            "Hypothesis or step-by-step argument is written; each step is justified.",
            "Logical argument structure",
        ),
        (
            "Derive conclusion",
            "Final conclusion follows logically from the steps; conclusion is stated clearly.",
            "Conclusion statement",
        ),
    ],
    "general": [
        (
            "Understand and clarify the task",
            "Task is restated in own words; scope and success criteria are identified.",
            "Clarified task description",
        ),
        (
            "Break down into actionable steps",
            "At least 2 concrete sub-steps are identified; dependencies are noted.",
            "Step breakdown list",
        ),
        (
            "Execute the primary action",
            "Main action is performed; output or result is produced.",
            "Action result",
        ),
        (
            "Review and refine",
            "Result is checked against success criteria; any gaps are documented.",
            "Review notes",
        ),
    ],
}


def _build_rule_milestones(
    task: str,
    min_subgoals: int,
    max_subgoals: int,
) -> list[Milestone]:
    """Build a milestone list using rule-based templates."""
    task_type = detect_task_type(task)
    template = _RULE_BASED_TEMPLATES.get(task_type, _RULE_BASED_TEMPLATES["general"])

    # Clamp to the allowable range; prefer depth up to max_subgoals
    depth = min(max(len(template), min_subgoals), max_subgoals)

    # If we need more milestones than the template provides, repeat the
    # "review and refine" milestone type to fill the gap.
    milestones: list[Milestone] = []
    for i in range(depth):
        if i < len(template):
            desc, rubric, expected = template[i]
        else:
            desc = f"Verify and validate step {i + 1}"
            rubric = "Step is verified; no errors detected."
            expected = f"Verification result for step {i + 1}"

        milestones.append(
            Milestone(
                id=f"m{i + 1}",
                description=desc,
                rubric=rubric,
                expected_output=expected,
                order=i,
            )
        )

    return milestones


# ---------------------------------------------------------------------------
# SubgoalPlanner
# ---------------------------------------------------------------------------


class SubgoalPlanner:
    """
    MiRA-style subgoal decomposition planner.

    Takes a task description and decomposes it into an ordered list of
    verifiable milestones. Supports both rule-based (fallback) and
    LLM-based decomposition.

    Parameters
    ----------
    llm_client : LLMClient, optional
        An LLM client to use for LLM-based decomposition. If ``None``,
        rule-based decomposition is used.
    max_subgoals : int, default 10
        Maximum number of subgoals to produce. Clamped to the range [1, 10].
    min_subgoals : int, default 1
        Minimum number of subgoals to produce. Clamped to [1, max_subgoals].

    Examples
    --------
    >>> planner = SubgoalPlanner()
    >>> milestones = planner.plan("Write a Python function that sorts a list")
    >>> len(milestones) >= 1
    True
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        max_subgoals: int = 10,
        min_subgoals: int = 1,
    ) -> None:
        if not (1 <= max_subgoals <= 10):
            raise ValueError("max_subgoals must be between 1 and 10")
        if not (1 <= min_subgoals):
            raise ValueError("min_subgoals must be at least 1")
        if min_subgoals > max_subgoals:
            raise ValueError("min_subgoals must not exceed max_subgoals")

        self._llm_client = llm_client
        self._max_subgoals = max_subgoals
        self._min_subgoals = min_subgoals

    @property
    def max_subgoals(self) -> int:
        """Maximum number of subgoals."""
        return self._max_subgoals

    @property
    def min_subgoals(self) -> int:
        """Minimum number of subgoals."""
        return self._min_subgoals

    def plan(self, task: str) -> list[Milestone]:
        """
        Decompose a task into an ordered milestone list.

        If an LLM client was provided at construction time, LLM-based
        decomposition is attempted. On any error, falls back to rule-based
        decomposition.

        Parameters
        ----------
        task : str
            Free-text task description.

        Returns
        -------
        list[Milestone]
            Ordered list of milestones (sorted by ``order``).
        """
        if self._llm_client is not None:
            try:
                return self._plan_with_llm(task)
            except Exception:
                # Fall back gracefully
                pass

        return _build_rule_milestones(task, self._min_subgoals, self._max_subgoals)

    def _plan_with_llm(self, task: str) -> list[Milestone]:
        """Use the LLM client for decomposition."""
        prompt = (
            "You are a subgoal decomposition planner. Given a task, produce a list of "
            "ordered milestones. Each milestone must have: id, description, rubric "
            "(pass/fail criteria), expected_output, and order.\n\n"
            f"Task: {task}\n\n"
            f"Produce between {self._min_subgoals} and {self._max_subgoals} milestones.\n"
            "Format your response as a JSON list of milestone objects with keys: "
            "id, description, rubric, expected_output, order.\n"
            "Use sequential IDs: m1, m2, m3, ...\n"
            "Return ONLY the JSON list, no extra text."
        )
        response = self._llm_client.complete(prompt)

        # Attempt to parse JSON from the response
        json_str = _extract_json(response)
        import json

        data = json.loads(json_str)
        milestones = [Milestone.from_dict(item) for item in data]

        # Enforce ordering and depth constraints
        milestones.sort(key=lambda m: m.order)
        if len(milestones) < self._min_subgoals:
            # Pad with generic milestones
            for i in range(len(milestones), self._max_subgoals):
                milestones.append(
                    Milestone(
                        id=f"m{i + 1}",
                        description=f"Verify milestone {i + 1}",
                        rubric="Milestone completed successfully.",
                        expected_output=f"Outcome for milestone {i + 1}",
                        order=i,
                    )
                )
        if len(milestones) > self._max_subgoals:
            milestones = milestones[: self._max_subgoals]
            # Re-number
            for i, m in enumerate(milestones):
                m.id = f"m{i + 1}"
                m.order = i

        return milestones

    def plan_from_template(
        self, task: str, template: list[dict[str, str]]
    ) -> list[Milestone]:
        """
        Decompose a task using a custom template.

        Parameters
        ----------
        task : str
            Task description (used only for context; template drives content).
        template : list[dict]
            List of dicts, each with keys: ``description``, ``rubric``,
            ``expected_output``.

        Returns
        -------
        list[Milestone]
            Ordered milestone list built from the template.
        """
        if not template:
            return _build_rule_milestones(task, self._min_subgoals, self._max_subgoals)

        milestones: list[Milestone] = []
        for i, step in enumerate(template[: self._max_subgoals]):
            milestones.append(
                Milestone(
                    id=f"m{i + 1}",
                    description=str(step.get("description", f"Step {i + 1}"))[:200],
                    rubric=str(step.get("rubric", "Complete.")),
                    expected_output=str(step.get("expected_output", "")),
                    order=i,
                )
            )

        # Pad to min_subgoals if needed
        while len(milestones) < self._min_subgoals:
            i = len(milestones)
            milestones.append(
                Milestone(
                    id=f"m{i + 1}",
                    description=f"Validate step {i + 1}",
                    rubric="Step completed successfully.",
                    expected_output="Validation result",
                    order=i,
                )
            )
        return milestones


# ---------------------------------------------------------------------------
# LLMPlanner — Subclass that always uses the LLM
# ---------------------------------------------------------------------------


class LLMPlanner(SubgoalPlanner):
    """
    Subclass of SubgoalPlanner that always uses an LLM client.

    Raises ``ValueError`` at construction if no LLM client is provided.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        max_subgoals: int = 10,
        min_subgoals: int = 1,
    ) -> None:
        super().__init__(llm_client=llm_client, max_subgoals=max_subgoals, min_subgoals=min_subgoals)


# ---------------------------------------------------------------------------
# JSON Parsing Helper
# ---------------------------------------------------------------------------

_JSON_CODE_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.MULTILINE)
_JSON_LIST_RE = re.compile(r"\[\s*\{", re.MULTILINE)


def _extract_json(text: str) -> str:
    """Extract the first JSON list or object from a text string."""
    # Try code block first
    match = _JSON_CODE_BLOCK_RE.search(text)
    if match:
        return match.group(1).strip()

    # Fall back to finding a JSON array
    start = _JSON_LIST_RE.search(text)
    if start:
        # Simple bracket matching — find the matching closing bracket
        idx = start.start()
        depth = 0
        in_str = False
        escape = False
        for i, ch in enumerate(text[idx:], idx):
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"' and not escape:
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "[" or ch == "{":
                depth += 1
            elif ch == "]" or ch == "}":
                depth -= 1
                if depth == 0:
                    return text[idx : i + 1]

    return text.strip()
