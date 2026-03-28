"""MiRA-style Subgoal Decomposition Module.

Plans a task into ordered milestones (subgoals) with rubrics.
"""

import json
import logging
import os
import re
import textwrap
from dataclasses import dataclass, field
from typing import List, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)


class PlannerError(Exception):
    """Base exception for planner errors."""
    pass


class APIKeyMissingError(PlannerError):
    """Raised when the OpenAI API key is not configured."""
    pass


class PlanParsingError(PlannerError):
    """Raised when the LLM response cannot be parsed into a valid plan."""
    pass


class PlanValidationError(PlannerError):
    """Raised when the parsed plan is missing required fields."""
    pass


@dataclass
class Milestone:
    """A single subgoal with pass/fail rubric."""
    id: str
    description: str
    rubric: str  # Pass/fail criteria
    expected_output: str = ""


@dataclass
class SubgoalPlan:
    """A task decomposed into ordered milestones."""
    task: str
    milestones: List[Milestone] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "milestones": [
                {
                    "id": m.id,
                    "description": m.description,
                    "rubric": m.rubric,
                    "expected_output": m.expected_output,
                }
                for m in self.milestones
            ],
        }


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

PLANNING_SYSTEM_PROMPT = textwrap.dedent("""\
    You are a MiRA-style subgoal decomposition planner.
    Your job is to decompose a high-level task into a small number of \
    ordered, measurable milestones (subgoals).

    For each milestone you must produce:
      - id         : a short unique identifier, e.g. "m1", "m2" ...
      - description: plain-English description of what this milestone \
                     requires
      - rubric     : concrete pass/fail criteria that can be used by \
                     an automated critic to judge completion
      - expected_output: what concrete artifact or result completing \
                         this milestone produces

    Rules:
      - Return between {min_depth} and {max_depth} milestones (inclusive).
      - Milestones must be ordered: each one builds on the previous.
      - The last milestone should represent full task completion.
      - Every milestone's rubric must be specific enough to evaluate \
        automatically.
      - Think step by step but only output the final JSON.
    """).strip()


PLANNING_USER_PROMPT_TEMPLATE = textwrap.dedent("""\
    Decompose the following task into ordered milestones:

    Task: {task}

    Return ONLY a JSON object with this shape:
    {{
      "milestones": [
        {{
          "id": "m1",
          "description": "...",
          "rubric": "...",
          "expected_output": "..."
        }},
        ...
      ]
    }}
    Do not include any text outside the JSON block.
    """).strip()


REPLANNING_SYSTEM_PROMPT = textwrap.dedent("""\
    You are a MiRA-style subgoal decomposition planner.
    A previously generated plan failed at milestone "{failed_id}".
    Revise the remaining plan so that the failed milestone is either:
      - Rewritten with a simpler or better-specified version, OR
      - Split into multiple smaller milestones.

    Return a complete revised plan (all milestones, not just the revised \
    ones) as a JSON object with the same shape as before.
    Think step by step about why the milestone failed and how to fix it, \
    but only output the final JSON.
    """).strip()


REPLANNING_USER_PROMPT_TEMPLATE = textwrap.dedent("""\
    Original task: {task}

    Failed milestone: {failed_id}
    Original description: {failed_description}
    Original rubric: {failed_rubric}

    Return ONLY a JSON object with the complete revised plan:
    {{
      "milestones": [
        {{
          "id": "m1",
          "description": "...",
          "rubric": "...",
          "expected_output": "..."
        }},
        ...
      ]
    }}
    Do not include any text outside the JSON block.
    """).strip()


# ---------------------------------------------------------------------------
# SubgoalPlanner
# ---------------------------------------------------------------------------

class SubgoalPlanner:
    """Decomposes a task into ordered subgoals with rubrics using an LLM.

    Parameters
    ----------
    planning_depth : int, default 5
        Desired number of milestones (1–10). The LLM will aim for this
        number but may return slightly more or fewer depending on the task.

    model : str, default "gpt-4o-mini"
        OpenAI model to use for decomposition.

    api_key : str, optional
        OpenAI API key. If omitted, reads from the ``OPENAI_API_KEY``
        environment variable.

    Attributes
    ----------
    planning_depth : int
    model : str
    """

    DEFAULT_MODEL = "gpt-4o-mini"
    MIN_DEPTH = 1
    MAX_DEPTH = 10

    def __init__(
        self,
        planning_depth: int = 5,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
    ):
        if not isinstance(planning_depth, int):
            raise TypeError(
                f"planning_depth must be an int, got {type(planning_depth).__name__}"
            )
        if not (self.MIN_DEPTH <= planning_depth <= self.MAX_DEPTH):
            raise ValueError(
                f"planning_depth must be between {self.MIN_DEPTH} and "
                f"{self.MAX_DEPTH}, got {planning_depth}"
            )

        self.planning_depth = planning_depth
        self.model = model

        # Resolve API key
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not resolved_key:
            raise APIKeyMissingError(
                "OpenAI API key is not configured. Either set the "
                "OPENAI_API_KEY environment variable or pass api_key "
                "to SubgoalPlanner(...)."
            )

        self._client = OpenAI(api_key=resolved_key)

    # -----------------------------------------------------------------------
    # Core planning methods
    # -----------------------------------------------------------------------

    def plan(self, task: str) -> SubgoalPlan:
        """Decompose a task into ordered milestones.

        Parameters
        ----------
        task : str
            High-level task description.

        Returns
        -------
        SubgoalPlan
            A plan containing ordered :class:`Milestone` objects.

        Raises
        ------
        APIKeyMissingError
            If no OpenAI API key is configured.
        PlanParsingError
            If the LLM response cannot be parsed as JSON.
        PlanValidationError
            If the parsed JSON is missing required fields.
        """
        system_prompt = PLANNING_SYSTEM_PROMPT.format(
            min_depth=max(1, self.planning_depth - 2),
            max_depth=min(10, self.planning_depth + 2),
        )
        user_prompt = PLANNING_USER_PROMPT_TEMPLATE.format(task=task)

        raw_response = self._call_llm(system_prompt, user_prompt)
        plan_dict = self._parse_response(raw_response)
        self._validate_plan_dict(plan_dict)
        milestones = self._build_milestones(plan_dict)

        return SubgoalPlan(task=task, milestones=milestones)

    def replan(
        self,
        plan: SubgoalPlan,
        failed_milestone_id: str,
    ) -> SubgoalPlan:
        """Replan after a milestone failure.

        Revises the remaining plan so that the failed milestone is either
        rewritten with clearer specification or split into smaller steps.

        Parameters
        ----------
        plan : SubgoalPlan
            The current (failed) plan.
        failed_milestone_id : str
            ID of the milestone that failed.

        Returns
        -------
        SubgoalPlan
            A revised plan.

        Raises
        ------
        ValueError
            If ``failed_milestone_id`` is not found in the plan.
        APIKeyMissingError
            If no OpenAI API key is configured.
        PlanParsingError
            If the LLM response cannot be parsed as JSON.
        """
        # Find the failed milestone
        failed = None
        for m in plan.milestones:
            if m.id == failed_milestone_id:
                failed = m
                break

        if failed is None:
            raise ValueError(
                f"Milestone id '{failed_milestone_id}' not found in plan. "
                f"Available ids: {[m.id for m in plan.milestones]}"
            )

        system_prompt = REPLANNING_SYSTEM_PROMPT.format(failed_id=failed.id)
        user_prompt = REPLANNING_USER_PROMPT_TEMPLATE.format(
            task=plan.task,
            failed_id=failed.id,
            failed_description=failed.description,
            failed_rubric=failed.rubric,
        )

        raw_response = self._call_llm(system_prompt, user_prompt)
        plan_dict = self._parse_response(raw_response)
        self._validate_plan_dict(plan_dict)
        milestones = self._build_milestones(plan_dict)

        return SubgoalPlan(task=plan.task, milestones=milestones)

    # -----------------------------------------------------------------------
    # LLM interaction helpers
    # -----------------------------------------------------------------------

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call the OpenAI model and return the raw response text."""
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,  # Low temperature for deterministic planning
                max_tokens=2048,
            )
            content = response.choices[0].message.content
            if content is None:
                raise PlanParsingError("LLM returned an empty response.")
            return content

        except Exception as e:
            logger.error("LLM API call failed: %s", e)
            raise PlannerError(f"LLM API call failed: {e}") from e

    # -----------------------------------------------------------------------
    # Parsing helpers
    # -----------------------------------------------------------------------

    def _parse_response(self, raw: str) -> dict:
        """Extract a JSON object from the LLM response.

        Handles cases where the LLM wraps the JSON in ```json fences
        or in plain text.
        """
        # Try stripping code fences first
        stripped = raw.strip()
        fence_pattern = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)
        match = fence_pattern.match(stripped)
        if match:
            stripped = match.group(1).strip()

        # Try direct JSON parse
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

        # Try to extract the first JSON object from the text
        json_start = stripped.find("{")
        json_end = stripped.rfind("}")
        if json_start != -1 and json_end != -1:
            candidate = stripped[json_start : json_end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        raise PlanParsingError(
            f"Could not parse LLM response as JSON. Response was:\n{raw[:500]}"
        )

    def _validate_plan_dict(self, d: dict) -> None:
        """Validate that the parsed dict has the expected structure."""
        if not isinstance(d, dict):
            raise PlanValidationError(
                f"Expected a JSON object, got {type(d).__name__}"
            )

        milestones = d.get("milestones")
        if not isinstance(milestones, list):
            raise PlanValidationError(
                "Parsed JSON must have a 'milestones' key containing a list. "
                f"Got keys: {list(d.keys())}"
            )

        if len(milestones) == 0:
            raise PlanValidationError("Milestones list cannot be empty.")

        for i, m in enumerate(milestones):
            if not isinstance(m, dict):
                raise PlanValidationError(
                    f"Milestone {i} is not an object (got {type(m).__name__})."
                )
            for field_name in ("id", "description", "rubric"):
                if not m.get(field_name):
                    raise PlanValidationError(
                        f"Milestone {i} is missing required field '{field_name}' "
                        f"(value: {m!r})."
                    )

    def _build_milestones(self, d: dict) -> List[Milestone]:
        """Convert a validated JSON dict into Milestone objects."""
        milestones = []
        for m in d["milestones"]:
            milestones.append(
                Milestone(
                    id=str(m["id"]),
                    description=str(m["description"]),
                    rubric=str(m["rubric"]),
                    expected_output=str(m.get("expected_output", "")),
                )
            )
        return milestones
