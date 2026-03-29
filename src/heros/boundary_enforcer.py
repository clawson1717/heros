"""Box Maze Boundary Enforcement for HeRoS.

Implements action-space boundary constraints throughout agent trajectories,
preventing the agent from "forgetting" constraints mid-trajectory.
Includes a memory grounding layer for constraint persistence.

References:
    - Box Maze: Spatial action constraint enforcement
    - HeRoS: Hindsight-driven Reinforcement with Subgoal Milestones
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

from heros.buffer import HindsightTrajectory

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class BoundaryConstraint:
    """A single boundary constraint on agent actions or state.

    Parameters
    ----------
    constraint_id : str
        Unique identifier for this constraint.
    constraint_type : Literal["box", "boundary", "forbidden", "required"]
        The type of constraint. "box" constrains spatial position within
        a bounding region; "boundary" constrains reaching a limit;
        "forbidden" prohibits specific actions; "required" mandates actions.
    description : str
        Human-readable description of what this constraint enforces.
    predicate : Callable[[dict], bool]
        Function that returns True if the constraint is satisfied given
        the current state dict.
    enforcement_level : Literal["strict", "soft", "advisory"]
        How aggressively to enforce: "strict" blocks/rejects actions,
        "soft" adds warnings, "advisory" logs only.
    metadata : dict
        Additional information about the constraint (domain, source, etc.).
    """

    constraint_id: str
    constraint_type: Literal["box", "boundary", "forbidden", "required"]
    description: str
    predicate: Callable[[dict], bool]
    enforcement_level: Literal["strict", "soft", "advisory"] = "strict"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not callable(self.predicate):
            raise TypeError("predicate must be callable")

    def evaluate(self, state: dict) -> Tuple[bool, Optional[str]]:
        """Evaluate this constraint against a state.

        Parameters
        ----------
        state : dict
            The current state to check.

        Returns
        -------
        Tuple[bool, Optional[str]]
            (satisfied, violation_reason) — satisfied is True if
            constraint is met; violation_reason is None if satisfied.
        """
        try:
            satisfied = self.predicate(state)
            if satisfied:
                return True, None
            return False, f"Constraint '{self.constraint_id}' violated: {self.description}"
        except Exception as e:
            return False, f"Constraint '{self.constraint_id}' evaluation error: {e}"


@dataclass
class ConstraintStatus:
    """Tracks the runtime status of a single constraint.

    Attributes
    ----------
    constraint_id : str
        The constraint being tracked.
    status : Literal["active", "satisfied", "violated", "unknown"]
        Current status of the constraint.
    last_check_timestamp : float
        Unix timestamp of the last check.
    violation_reason : str, optional
        If violated, a human-readable reason.
    """

    constraint_id: str
    status: Literal["active", "satisfied", "violated", "unknown"] = "unknown"
    last_check_timestamp: float = 0.0
    violation_reason: Optional[str] = None

    def update(self, status: Literal["active", "satisfied", "violated", "unknown"], 
               timestamp: Optional[float] = None, violation_reason: Optional[str] = None) -> None:
        """Update the constraint status."""
        self.status = status
        self.last_check_timestamp = timestamp or time.time()
        if violation_reason is not None:
            self.violation_reason = violation_reason


@dataclass
class BoxRegion:
    """A spatial bounding region with allowed/forbidden actions.

    Parameters
    ----------
    region_id : str
        Unique identifier for this region.
    bounds : dict
        Spatial bounds as {"x_min": float, "x_max": float, "y_min": float, "y_max": float}.
        All keys are optional; missing keys mean no limit in that direction.
    allowed_actions : list[str]
        White-list of permitted actions in this region.
        Empty list means all actions are allowed (unless in forbidden_actions).
    forbidden_actions : list[str]
        Black-list of prohibited actions in this region.
    forbidden_areas : list[dict]
        Sub-regions within bounds that must be avoided.
        Each dict has keys matching ``bounds`` format.
    """

    region_id: str
    bounds: dict = field(default_factory=dict)
    allowed_actions: List[str] = field(default_factory=list)
    forbidden_actions: List[str] = field(default_factory=list)
    forbidden_areas: List[dict] = field(default_factory=list)

    def contains_point(self, x: float, y: float) -> bool:
        """Check if a point is within the region bounds.

        Parameters
        ----------
        x : float
            X coordinate.
        y : float
            Y coordinate.

        Returns
        -------
        bool
            True if the point is within bounds (accounting for missing keys).
        """
        x_min = self.bounds.get("x_min")
        x_max = self.bounds.get("x_max")
        y_min = self.bounds.get("y_min")
        y_max = self.bounds.get("y_max")

        if x_min is not None and x < x_min:
            return False
        if x_max is not None and x > x_max:
            return False
        if y_min is not None and y < y_min:
            return False
        if y_max is not None and y > y_max:
            return False
        return True

    def is_in_forbidden_area(self, x: float, y: float) -> bool:
        """Check if a point lies within any forbidden sub-area.

        Parameters
        ----------
        x : float
            X coordinate.
        y : float
            Y coordinate.

        Returns
        -------
        bool
            True if the point is inside a forbidden area.
        """
        for area in self.forbidden_areas:
            a_x_min = area.get("x_min")
            a_x_max = area.get("x_max")
            a_y_min = area.get("y_min")
            a_y_max = area.get("y_max")

            if a_x_min is not None and x < a_x_min:
                continue
            if a_x_max is not None and x > a_x_max:
                continue
            if a_y_min is not None and y < a_y_min:
                continue
            if a_y_max is not None and y > a_y_max:
                continue
            return True
        return False

    def is_action_allowed(self, action: str) -> bool:
        """Check if an action is permitted in this region.

        Parameters
        ----------
        action : str
            The action to check.

        Returns
        -------
        bool
            True if the action is allowed (in allowed_actions or
            allowed_actions is empty, and not in forbidden_actions).
        """
        if action in self.forbidden_actions:
            return False
        if self.allowed_actions and action not in self.allowed_actions:
            return False
        return True


@dataclass
class BoundaryState:
    """Tracks the runtime state of all constraints.

    Attributes
    ----------
    active_constraints : dict[str, ConstraintStatus]
        Maps constraint_id to its current ConstraintStatus.
    constraint_history : list[dict]
        Log of all constraint checks performed.
    violation_count : int
        Total number of violations recorded across all constraints.
    current_box : BoxRegion, optional
        The currently active spatial bounding box, if applicable.
    """

    active_constraints: Dict[str, ConstraintStatus] = field(default_factory=dict)
    constraint_history: List[dict] = field(default_factory=list)
    violation_count: int = 0
    current_box: Optional[BoxRegion] = None

    def add_history_entry(self, entry: dict) -> None:
        """Append a constraint check to the history log."""
        self.constraint_history.append(entry)

    def increment_violations(self) -> None:
        """Increment the global violation counter."""
        self.violation_count += 1


@dataclass
class ConstraintCheckResult:
    """Result of checking all active constraints against a state.

    Attributes
    ----------
    all_satisfied : bool
        True if every constraint was satisfied.
    violated_constraints : list[str]
        List of constraint_ids that were violated.
    satisfied_constraints : list[str]
        List of constraint_ids that were satisfied.
    timestamp : float
        Unix timestamp when the check was performed.
    """

    all_satisfied: bool
    violated_constraints: List[str]
    satisfied_constraints: List[str]
    timestamp: float = field(default_factory=time.time)


@dataclass
class EnforcedAction:
    """An action after boundary enforcement has been applied.

    Attributes
    ----------
    original_action : str
        The action as originally requested by the agent.
    enforced_action : str
        The action after enforcement (may be modified or clamped).
    modified : bool
        True if the action was changed by the enforcer.
    enforcement_reason : str, optional
        Human-readable explanation of what was enforced.
    warnings : list[str]
        Non-fatal warnings generated during enforcement.
    constraint_ids_enforced : list[str]
        IDs of constraints that caused enforcement modifications.
    """

    original_action: str
    enforced_action: str
    modified: bool = False
    enforcement_reason: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    constraint_ids_enforced: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# BoundaryEnforcer
# ---------------------------------------------------------------------------


class BoundaryEnforcer:
    """Enforces action-space boundary constraints throughout agent trajectories.

    Prevents the agent from "forgetting" constraints mid-trajectory by
    checking every action against registered constraints and modifying
    or rejecting actions that would violate strict constraints.

    Parameters
    ----------
    constraints : list[BoundaryConstraint]
        Initial list of constraints to enforce.
    default_enforcement : str, optional
        Default enforcement level for constraints without an explicit level.
        Must be "strict", "soft", or "advisory". Defaults to "strict".

    Examples
    --------
    >>> def in_bounds(state):
    ...     return 0 <= state.get("x", 0) <= 10
    >>> constraint = BoundaryConstraint(
    ...     constraint_id="bound_x",
    ...     constraint_type="boundary",
    ...     description="X must be between 0 and 10",
    ...     predicate=in_bounds,
    ...     enforcement_level="strict",
    ...     metadata={"domain": "navigation"},
    ... )
    >>> enforcer = BoundaryEnforcer(constraints=[constraint])
    >>> state = {"x": 5, "y": 3}
    >>> result = enforcer.check_constraints(state)
    >>> result.all_satisfied
    True
    """

    VALID_ENFORCEMENT_LEVELS = ("strict", "soft", "advisory")

    def __init__(
        self,
        constraints: Optional[List[BoundaryConstraint]] = None,
        default_enforcement: str = "strict",
    ) -> None:
        if default_enforcement not in self.VALID_ENFORCEMENT_LEVELS:
            raise ValueError(
                f"default_enforcement must be one of {self.VALID_ENFORCEMENT_LEVELS}, "
                f"got '{default_enforcement}'"
            )

        self._constraints: Dict[str, BoundaryConstraint] = {}
        self._default_enforcement = default_enforcement
        self._state = BoundaryState()
        self._current_box: Optional[BoxRegion] = None

        if constraints:
            for constraint in constraints:
                self.register_constraint(constraint)

    # ---------------------------------------------------------------------------
    # Properties
    # ---------------------------------------------------------------------------

    @property
    def constraints(self) -> List[BoundaryConstraint]:
        """All registered constraints in insertion order."""
        return list(self._constraints.values())

    @property
    def constraint_ids(self) -> List[str]:
        """IDs of all registered constraints."""
        return list(self._constraints.keys())

    @property
    def state(self) -> BoundaryState:
        """Current boundary enforcement state."""
        return self._state

    @property
    def current_box(self) -> Optional[BoxRegion]:
        """The currently active spatial bounding box, if set."""
        return self._current_box

    @current_box.setter
    def current_box(self, box: Optional[BoxRegion]) -> None:
        """Set the active spatial bounding box."""
        self._current_box = box
        self._state.current_box = box

    # ---------------------------------------------------------------------------
    # Constraint management
    # ---------------------------------------------------------------------------

    def register_constraint(self, constraint: BoundaryConstraint) -> None:
        """Register a new constraint with the enforcer.

        Parameters
        ----------
        constraint : BoundaryConstraint
            The constraint to add. If a constraint with the same ID
            already exists, it is replaced.

        Raises
        ------
        TypeError
            If constraint is not a BoundaryConstraint.
        """
        if not isinstance(constraint, BoundaryConstraint):
            raise TypeError(
                f"Expected BoundaryConstraint, got {type(constraint).__name__}"
            )
        self._constraints[constraint.constraint_id] = constraint
        if constraint.constraint_id not in self._state.active_constraints:
            self._state.active_constraints[constraint.constraint_id] = ConstraintStatus(
                constraint_id=constraint.constraint_id,
                status="active",
                last_check_timestamp=0.0,
            )
        logger.debug("Registered constraint: %s", constraint.constraint_id)

    def remove_constraint(self, constraint_id: str) -> bool:
        """Remove a constraint from the enforcer.

        Parameters
        ----------
        constraint_id : str
            The ID of the constraint to remove.

        Returns
        -------
        bool
            True if the constraint was removed, False if it was not found.
        """
        if constraint_id in self._constraints:
            del self._constraints[constraint_id]
            if constraint_id in self._state.active_constraints:
                del self._state.active_constraints[constraint_id]
            logger.debug("Removed constraint: %s", constraint_id)
            return True
        return False

    def get_constraint(self, constraint_id: str) -> Optional[BoundaryConstraint]:
        """Get a constraint by ID.

        Parameters
        ----------
        constraint_id : str
            The constraint ID to look up.

        Returns
        -------
        BoundaryConstraint or None
            The constraint if found, else None.
        """
        return self._constraints.get(constraint_id)

    # ---------------------------------------------------------------------------
    # State management
    # ---------------------------------------------------------------------------

    def reset_state(self) -> None:
        """Reset all constraint states and history.

        Clears the constraint history, resets all constraint statuses
        to "active", and resets the violation counter.
        """
        self._state = BoundaryState(
            active_constraints={
                cid: ConstraintStatus(
                    constraint_id=cid,
                    status="active",
                    last_check_timestamp=0.0,
                )
                for cid in self._constraints
            },
            constraint_history=[],
            violation_count=0,
            current_box=self._current_box,
        )
        logger.debug("BoundaryEnforcer state reset")

    # ---------------------------------------------------------------------------
    # Constraint checking
    # ---------------------------------------------------------------------------

    def check_constraints(self, state: dict) -> ConstraintCheckResult:
        """Check all active constraints against the current state.

        Evaluates every registered constraint's predicate against the
        given state and updates internal status tracking.

        Parameters
        ----------
        state : dict
            The current environment/state dictionary.

        Returns
        -------
        ConstraintCheckResult
            Summary of which constraints were satisfied or violated.
        """
        timestamp = time.time()
        violated: List[str] = []
        satisfied: List[str] = []

        for constraint_id, constraint in self._constraints.items():
            status = self._state.active_constraints.get(
                constraint_id,
                ConstraintStatus(constraint_id=constraint_id, status="unknown"),
            )

            satisfied_flag, violation_reason = constraint.evaluate(state)

            if satisfied_flag:
                status.update("satisfied", timestamp)
                satisfied.append(constraint_id)
            else:
                status.update("violated", timestamp, violation_reason)
                violated.append(constraint_id)
                self._state.increment_violations()

            # Record in history
            history_entry = {
                "timestamp": timestamp,
                "constraint_id": constraint_id,
                "constraint_type": constraint.constraint_type,
                "enforcement_level": constraint.enforcement_level,
                "satisfied": satisfied_flag,
                "violation_reason": violation_reason,
            }
            self._state.add_history_entry(history_entry)

        return ConstraintCheckResult(
            all_satisfied=len(violated) == 0,
            violated_constraints=violated,
            satisfied_constraints=satisfied,
            timestamp=timestamp,
        )

    # ---------------------------------------------------------------------------
    # Action enforcement
    # ---------------------------------------------------------------------------

    def enforce_action(
        self,
        action: str,
        state: dict,
        context: Optional[dict] = None,
    ) -> EnforcedAction:
        """Check and enforce constraints on an action.

        Evaluates whether the given action respects all active constraints
        in the given state, and modifies or rejects the action if it would
        violate a strict constraint.

        Parameters
        ----------
        action : str
            The action proposed by the agent.
        state : dict
            The current environment/state dictionary.
        context : dict, optional
            Additional context for enforcement decisions
            (e.g., "trajectory_step", "episode_id").

        Returns
        -------
        EnforcedAction
            The enforced action result, including any modifications,
            warnings, or enforcement reasons.
        """
        context = context or {}
        enforced_action = action
        modified = False
        enforcement_reason: Optional[str] = None
        warnings: List[str] = []
        constraint_ids_enforced: List[str] = []

        # Check box-region constraints first (if a current box is set)
        if self._current_box is not None:
            box_result = self._check_box_constraints(action, state, context)
            if box_result["modified"]:
                modified = True
                enforced_action = box_result["enforced_action"]
                enforcement_reason = box_result["reason"]
                constraint_ids_enforced.extend(box_result.get("constraint_ids", []))
            warnings.extend(box_result.get("warnings", []))

        # Check all registered constraints against current state
        check_result = self.check_constraints(state)

        # Apply enforcement based on violation severity and enforcement level
        for constraint_id in check_result.violated_constraints:
            constraint = self._constraints.get(constraint_id)
            if constraint is None:
                continue

            if constraint.enforcement_level == "strict":
                # Strict: modify the action to something safe
                if not modified:
                    enforced_action = self._clamp_action(action, state, constraint)
                    enforcement_reason = (
                        f"Strict constraint '{constraint_id}' violated: "
                        f"{constraint.description}"
                    )
                    modified = True
                constraint_ids_enforced.append(constraint_id)

            elif constraint.enforcement_level == "soft":
                # Soft: allow but warn
                warnings.append(
                    f"Soft constraint '{constraint_id}' violated: {constraint.description}"
                )

            elif constraint.enforcement_level == "advisory":
                # Advisory: just log (already logged in check_constraints)
                pass

        return EnforcedAction(
            original_action=action,
            enforced_action=enforced_action,
            modified=modified,
            enforcement_reason=enforcement_reason,
            warnings=warnings,
            constraint_ids_enforced=list(set(constraint_ids_enforced)),
        )

    def _check_box_constraints(
        self,
        action: str,
        state: dict,
        context: dict,
    ) -> dict:
        """Check action against current box region constraints.

        Returns
        -------
        dict
            Keys: modified (bool), enforced_action (str), reason (str),
            warnings (list), constraint_ids (list).
        """
        result = {
            "modified": False,
            "enforced_action": action,
            "reason": None,
            "warnings": [],
            "constraint_ids": [],
        }

        if self._current_box is None:
            return result

        box = self._current_box
        x = state.get("x", 0)
        y = state.get("y", 0)

        # Check if action is allowed in this region
        if not box.is_action_allowed(action):
            result["modified"] = True
            result["enforced_action"] = "noop"
            result["reason"] = (
                f"Action '{action}' is forbidden in region '{box.region_id}'"
            )
            result["constraint_ids"].append(box.region_id)
            return result

        # Check if action would move us into a forbidden area
        # We simulate position change based on action (simplified)
        next_pos = self._simulate_action_position(action, state)
        if next_pos is not None:
            next_x, next_y = next_pos
            if not box.contains_point(next_x, next_y):
                result["modified"] = True
                result["enforced_action"] = "noop"
                result["reason"] = (
                    f"Action '{action}' would move outside bounds of "
                    f"region '{box.region_id}'"
                )
                result["constraint_ids"].append(box.region_id)
            elif box.is_in_forbidden_area(next_x, next_y):
                result["modified"] = True
                result["enforced_action"] = "noop"
                result["reason"] = (
                    f"Action '{action}' would move into forbidden area "
                    f"in region '{box.region_id}'"
                )
                result["constraint_ids"].append(box.region_id)

        return result

    def _simulate_action_position(
        self,
        action: str,
        state: dict,
    ) -> Optional[Tuple[float, float]]:
        """Simulate the position after taking an action.

        Parameters
        ----------
        action : str
            The action to simulate.
        state : dict
            The current state.

        Returns
        -------
        Tuple[float, float] or None
            Simulated (x, y) position, or None if position cannot be determined.
        """
        x = state.get("x", 0)
        y = state.get("y", 0)
        step_size = state.get("step_size", 1.0)

        action_lower = action.lower()
        if action_lower in ("move_up", "up", "north", "y+"):
            return x, y + step_size
        elif action_lower in ("move_down", "down", "south", "y-"):
            return x, y - step_size
        elif action_lower in ("move_left", "left", "west", "x-"):
            return x - step_size, y
        elif action_lower in ("move_right", "right", "east", "x+"):
            return x + step_size, y
        elif action_lower in ("noop", "stay", "wait", "pass"):
            return x, y
        return None

    def _clamp_action(
        self,
        action: str,
        state: dict,
        constraint: BoundaryConstraint,
    ) -> str:
        """Clamp an action to a safe alternative when a constraint is violated.

        Parameters
        ----------
        action : str
            The original (violating) action.
        state : dict
            The current state.
        constraint : BoundaryConstraint
            The constraint that was violated.

        Returns
        -------
        str
            A safe alternative action (typically "noop" or "stop").
        """
        return "noop"

    def _is_within_bounds(
        self,
        action: str,
        state: dict,
        box: BoxRegion,
    ) -> bool:
        """Check if an action would keep the agent within bounds.

        Parameters
        ----------
        action : str
            The action to check.
        state : dict
            The current state.
        box : BoxRegion
            The bounding region to check against.

        Returns
        -------
        bool
            True if the action would keep the agent within bounds.
        """
        next_pos = self._simulate_action_position(action, state)
        if next_pos is None:
            return True
        next_x, next_y = next_pos
        return box.contains_point(next_x, next_y) and not box.is_in_forbidden_area(
            next_x, next_y
        )

    # ---------------------------------------------------------------------------
    # Query methods
    # ---------------------------------------------------------------------------

    def get_active_violations(self) -> List[ConstraintStatus]:
        """Return all constraints currently in a violated state.

        Returns
        -------
        list[ConstraintStatus]
            Constraints with status "violated".
        """
        return [
            status
            for status in self._state.active_constraints.values()
            if status.status == "violated"
        ]

    def get_constraint_summary(self) -> dict:
        """Return a summary of all constraints and their current status.

        Returns
        -------
        dict
            Summary including counts by status and type.
        """
        summary = {
            "total_constraints": len(self._constraints),
            "by_status": {
                "active": 0,
                "satisfied": 0,
                "violated": 0,
                "unknown": 0,
            },
            "by_type": {},
            "by_enforcement_level": {},
            "total_violations": self._state.violation_count,
            "current_box": (
                self._current_box.region_id if self._current_box else None
            ),
        }

        for status in self._state.active_constraints.values():
            summary["by_status"][status.status] = (
                summary["by_status"].get(status.status, 0) + 1
            )

        for constraint in self._constraints.values():
            summary["by_type"][constraint.constraint_type] = (
                summary["by_type"].get(constraint.constraint_type, 0) + 1
            )
            summary["by_enforcement_level"][constraint.enforcement_level] = (
                summary["by_enforcement_level"].get(
                    constraint.enforcement_level, 0
                )
                + 1
            )

        return summary

    def export_audit_trail(self) -> dict:
        """Export the full constraint check history as a JSON-serializable dict.

        Returns
        -------
        dict
            Complete audit trail including metadata and history entries.
        """
        return {
            "exported_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "constraint_count": len(self._constraints),
            "total_violations": self._state.violation_count,
            "current_box": (
                {
                    "region_id": self._current_box.region_id,
                    "bounds": self._current_box.bounds,
                    "allowed_actions": self._current_box.allowed_actions,
                    "forbidden_actions": self._current_box.forbidden_actions,
                    "forbidden_areas": self._current_box.forbidden_areas,
                }
                if self._current_box
                else None
            ),
            "constraints": [
                {
                    "constraint_id": c.constraint_id,
                    "constraint_type": c.constraint_type,
                    "description": c.description,
                    "enforcement_level": c.enforcement_level,
                    "metadata": c.metadata,
                }
                for c in self._constraints.values()
            ],
            "history": self._state.constraint_history,
        }


# ---------------------------------------------------------------------------
# MemoryGroundingLayer
# ---------------------------------------------------------------------------


class MemoryGroundingLayer:
    """Maintains constraint memory across trajectory steps to prevent forgetting.

    This layer acts as a grounding mechanism that keeps constraints
    persistently active in agent context, reminding the agent of active
    violations or upcoming boundary changes.

    Parameters
    ----------
    enforcer : BoundaryEnforcer
        The boundary enforcer whose constraints should be grounded.

    Examples
    --------
    >>> enforcer = BoundaryEnforcer()
    >>> grounding = MemoryGroundingLayer(enforcer)
    >>> grounding.update({"x": 5}, "move_right", "success")
    >>> reminder = grounding.remind_agent(["bound_x"])
    >>> print(reminder)
    REMINDER: You are constrained by: bound_x - X must be between 0 and 10
    """

    def __init__(self, enforcer: BoundaryEnforcer) -> None:
        if not isinstance(enforcer, BoundaryEnforcer):
            raise TypeError(
                f"Expected BoundaryEnforcer, got {type(enforcer).__name__}"
            )
        self._enforcer = enforcer
        self._memory: List[dict] = []
        self._step_count: int = 0
        self._reminder_template = (
            "REMINDER: You are constrained by: {constraints}. "
            "Violation may result in task failure."
        )

    # ---------------------------------------------------------------------------
    # Memory update
    # ---------------------------------------------------------------------------

    def update(self, state: dict, action: str, result: str) -> None:
        """Update grounded memory with a new trajectory step.

        Records the state, action, and outcome for constraint grounding
        purposes.

        Parameters
        ----------
        state : dict
            The state at this step.
        action : str
            The action taken.
        result : str
            The outcome of the action ("success", "failure", "warning").
        """
        self._step_count += 1
        entry = {
            "step": self._step_count,
            "state": dict(state),
            "action": action,
            "result": result,
            "timestamp": time.time(),
        }
        self._memory.append(entry)

        # Keep memory bounded to last 1000 steps
        if len(self._memory) > 1000:
            self._memory = self._memory[-1000:]

    def reset(self) -> None:
        """Clear all memory and reset step counter."""
        self._memory.clear()
        self._step_count = 0

    # ---------------------------------------------------------------------------
    # Grounding queries
    # ---------------------------------------------------------------------------

    def get_grounded_constraints(self, state: dict) -> List[BoundaryConstraint]:
        """Return constraints relevant to the current state.

        Parameters
        ----------
        state : dict
            The current state to evaluate relevance against.

        Returns
        -------
        list[BoundaryConstraint]
            Constraints that apply to the current state (always all
            currently active constraints).
        """
        return self._enforcer.constraints

    def remind_agent(self, active_violations: List[str]) -> str:
        """Generate a natural language reminder for the agent.

        Parameters
        ----------
        active_violations : list[str]
            List of constraint IDs that are currently violated.

        Returns
        -------
        str
            A formatted reminder string suitable for prepending to
            agent prompts.
        """
        if not active_violations:
            return ""

        constraint_parts = []
        for constraint_id in active_violations:
            constraint = self._enforcer.get_constraint(constraint_id)
            if constraint:
                constraint_parts.append(
                    f"{constraint_id} - {constraint.description}"
                )

        if not constraint_parts:
            return ""

        return self._reminder_template.format(constraints="; ".join(constraint_parts))

    def get_memory_summary(self) -> dict:
        """Return the current memory state as a dict.

        Returns
        -------
        dict
            Summary including step count and recent history.
        """
        recent = self._memory[-10:] if len(self._memory) >= 10 else self._memory
        return {
            "total_steps": self._step_count,
            "memory_size": len(self._memory),
            "recent_steps": [
                {"step": e["step"], "action": e["action"], "result": e["result"]}
                for e in recent
            ],
            "active_violations": [
                v.constraint_id
                for v in self._enforcer.get_active_violations()
            ],
        }

    # ---------------------------------------------------------------------------
    # History access
    # ---------------------------------------------------------------------------

    def get_recent_actions(self, n: int = 10) -> List[str]:
        """Get the n most recent actions from memory.

        Parameters
        ----------
        n : int, optional
            Number of recent actions to return. Defaults to 10.

        Returns
        -------
        list[str]
            Recent action strings.
        """
        recent = self._memory[-n:] if len(self._memory) >= n else self._memory
        return [e["action"] for e in recent]

    def get_action_history(self) -> List[dict]:
        """Return the full action/state history.

        Returns
        -------
        list[dict]
            Complete memory entries.
        """
        return list(self._memory)


# ---------------------------------------------------------------------------
# HeRoSBoundaryIntegration
# ---------------------------------------------------------------------------


class HeRoSBoundaryIntegration:
    """Integrates boundary enforcement into HeRoS agent execution.

    Wraps a HeRoSAgent (or agent-like callable) and adds boundary
    enforcement to every action taken during episode execution.

    Parameters
    ----------
    agent : object
        An agent object that exposes an ``act(task, state)`` method
        returning an action string. Also used as the planner in
        ``run_with_boundaries``.
    enforcer : BoundaryEnforcer
        The boundary enforcement engine.
    grounding : MemoryGroundingLayer
        The memory grounding layer for constraint reminders.
    """

    def __init__(
        self,
        agent: Any,
        enforcer: BoundaryEnforcer,
        grounding: MemoryGroundingLayer,
    ) -> None:
        self._agent = agent
        self._enforcer = enforcer
        self._grounding = grounding

    def run_with_boundaries(
        self,
        task: str,
        constraints: Optional[List[BoundaryConstraint]] = None,
        max_steps: int = 50,
    ) -> Tuple[List[dict], float]:
        """Run an episode with boundary enforcement enabled.

        Executes the agent on the given task while enforcing all
        registered boundary constraints on every step.

        Parameters
        ----------
        task : str
            The task description to execute.
        constraints : list[BoundaryConstraint], optional
            Additional constraints to register for this episode.
        max_steps : int, optional
            Maximum number of steps to run. Defaults to 50.

        Returns
        -------
        Tuple[list[dict], float]
            (trajectory, total_reward) — trajectory is a list of step dicts
            with keys: step, state, action, enforced_action, result, warnings.
            total_reward is the sum of reward signals.
        """
        # Register any additional constraints
        if constraints:
            for c in constraints:
                self._enforcer.register_constraint(c)

        # Reset enforcer and grounding state
        self._enforcer.reset_state()
        self._grounding.reset()

        trajectory: List[dict] = []
        total_reward = 0.0
        step = 0

        # Initialize state
        state = {"step": 0, "x": 0, "y": 0, "observation": task}

        # Get initial plan from agent (if it has a planner)
        milestones = []
        if hasattr(self._agent, "planner") and self._agent.planner:
            plan = self._agent.planner.plan(task)
            milestones = plan.milestones if hasattr(plan, "milestones") else []

        while step < max_steps:
            step += 1
            state["step"] = step

            # Check constraints before action
            check_result = self._enforcer.check_constraints(state)

            # Get violations for grounding reminder
            active_violations = [v.constraint_id for v in self._enforcer.get_active_violations()]

            # Get action from agent (with constraint reminder injected)
            agent_input = task
            if active_violations:
                agent_input = self._inject_constraint_reminder(
                    task, active_violations
                )

            try:
                act_result = self._agent.act(agent_input, state)
                action = act_result if isinstance(act_result, str) else str(act_result)
            except Exception:
                action = "noop"

            # Enforce action through boundary layer
            enforced = self._enforcer.enforce_action(action, state)
            enforced_action = enforced.enforced_action

            # Simulate environment step (in real usage this would be env.step)
            reward_signal = 0.0
            result_str = "success"

            if enforced.modified:
                result_str = "modified"
                reward_signal = -0.1
            elif active_violations:
                result_str = "warning"
                reward_signal = -0.05

            total_reward += reward_signal

            # Update grounding memory
            self._grounding.update(state, enforced_action, result_str)

            # Record step
            trajectory.append({
                "step": step,
                "state": dict(state),
                "action": action,
                "enforced_action": enforced_action,
                "modified": enforced.modified,
                "warnings": enforced.warnings,
                "result": result_str,
                "violated_constraints": active_violations,
            })

            # Update state position based on enforced action (simplified)
            next_pos = self._enforcer._simulate_action_position(enforced_action, state)
            if next_pos:
                state = dict(state)
                state["x"], state["y"] = next_pos

        return trajectory, total_reward

    def _inject_constraint_reminder(
        self,
        agent_input: str,
        active_violations: List[str],
    ) -> str:
        """Prepend constraint reminders to the agent's input prompt.

        Parameters
        ----------
        agent_input : str
            The original agent input/prompt.
        active_violations : list[str]
            IDs of constraints that are currently violated.

        Returns
        -------
        str
            The input with constraint reminders prepended.
        """
        reminder = self._grounding.remind_agent(active_violations)
        if not reminder:
            return agent_input
        return f"{reminder}\n\nTask: {agent_input}"


# ---------------------------------------------------------------------------
# BoundaryEvaluator
# ---------------------------------------------------------------------------


class BoundaryEvaluator:
    """Evaluates the effectiveness of boundary enforcement on trajectories.

    Measures how often boundary enforcement prevented constraint violations
    and computes aggregate statistics across test trajectories.

    Parameters
    ----------
    enforcer : BoundaryEnforcer
        The enforcer to evaluate.
    """

    def __init__(self, enforcer: BoundaryEnforcer) -> None:
        self._enforcer = enforcer

    def evaluate_boundary_effectiveness(
        self,
        test_trajectories: List[HindsightTrajectory],
    ) -> dict:
        """Measure how often boundaries prevented constraint failures.

        Analyzes a set of hindsight trajectories and computes metrics
        on boundary enforcement effectiveness.

        Parameters
        ----------
        test_trajectories : list[HindsightTrajectory]
            Trajectories to evaluate. Each should contain constraint-
            related data in its exec_traces.

        Returns
        -------
        dict
            Evaluation metrics including:
            - boundary_prevention_rate: fraction of potential violations
              that were prevented by enforcement
            - constraint_violation_rate: fraction of steps with violations
            - avg_constraints_per_step: average constraints checked per step
            - total_steps_evaluated: total steps analyzed
        """
        if not test_trajectories:
            return {
                "boundary_prevention_rate": 0.0,
                "constraint_violation_rate": 0.0,
                "avg_constraints_per_step": 0.0,
                "total_steps_evaluated": 0,
                "total_violations_prevented": 0,
                "total_violations_occurred": 0,
            }

        total_steps = 0
        total_violations = 0
        violations_prevented = 0
        constraints_checked = 0

        for traj in test_trajectories:
            # Each trajectory has exec_traces; we check each one
            num_steps = len(traj.exec_traces)
            total_steps += num_steps

            # Simulate constraint checks for each step
            for trace in traj.exec_traces:
                # Use trace content to build a mock state
                state = {"step": total_steps}
                if isinstance(trace, dict):
                    state.update({k: v for k, v in trace.items() if isinstance(v, (int, float, str))})

                result = self._enforcer.check_constraints(state)
                constraints_checked += len(self._enforcer.constraints)

                if not result.all_satisfied:
                    total_violations += len(result.violated_constraints)
                    # If strict enforcement was on, violations would be prevented
                    violations_prevented += len(result.violated_constraints)

        avg_constraints = (
            constraints_checked / total_steps if total_steps > 0 else 0.0
        )
        violation_rate = (
            total_violations / total_steps if total_steps > 0 else 0.0
        )
        # Prevention rate: what fraction of potential violations were caught
        prevention_rate = (
            violations_prevented / (violations_prevented + total_violations)
            if (violations_prevented + total_violations) > 0
            else 0.0
        )

        return {
            "boundary_prevention_rate": prevention_rate,
            "constraint_violation_rate": violation_rate,
            "avg_constraints_per_step": avg_constraints,
            "total_steps_evaluated": total_steps,
            "total_violations_prevented": violations_prevented,
            "total_violations_occurred": total_violations,
        }
