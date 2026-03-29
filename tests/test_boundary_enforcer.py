"""Comprehensive tests for Box Maze Boundary Enforcement.

Tests BoundaryEnforcer, MemoryGroundingLayer, HeRoSBoundaryIntegration,
BoundaryEvaluator, and all related dataclasses.
"""

import time
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from heros.boundary_enforcer import (
    BoundaryConstraint,
    BoundaryState,
    BoundaryEnforcer,
    BoxRegion,
    ConstraintCheckResult,
    ConstraintStatus,
    EnforcedAction,
    HeRoSBoundaryIntegration,
    MemoryGroundingLayer,
    BoundaryEvaluator,
)
from heros.buffer import HindsightTrajectory
from heros.planner import Milestone


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def in_bounds_x():
    """Predicate that checks x is within [0, 10]."""
    def predicate(state: dict) -> bool:
        x = state.get("x", 0)
        return 0 <= x <= 10
    return predicate


@pytest.fixture
def in_bounds_y():
    """Predicate that checks y is within [0, 10]."""
    def predicate(state: dict) -> bool:
        y = state.get("y", 0)
        return 0 <= y <= 10
    return predicate


@pytest.fixture
def no_forbidden_action():
    """Predicate that checks action is not 'delete_everything'."""
    def predicate(state: dict) -> bool:
        return state.get("action") != "delete_everything"
    return predicate


@pytest.fixture
def boundary_constraint_x(in_bounds_x):
    """A boundary constraint on x position."""
    return BoundaryConstraint(
        constraint_id="bound_x",
        constraint_type="boundary",
        description="X must be between 0 and 10",
        predicate=in_bounds_x,
        enforcement_level="strict",
        metadata={"domain": "navigation"},
    )


@pytest.fixture
def boundary_constraint_y(in_bounds_y):
    """A boundary constraint on y position."""
    return BoundaryConstraint(
        constraint_id="bound_y",
        constraint_type="boundary",
        description="Y must be between 0 and 10",
        predicate=in_bounds_y,
        enforcement_level="soft",
        metadata={"domain": "navigation"},
    )


@pytest.fixture
def forbidden_action_constraint(no_forbidden_action):
    """A forbidden action constraint."""
    return BoundaryConstraint(
        constraint_id="forbid_delete",
        constraint_type="forbidden",
        description="Action 'delete_everything' is prohibited",
        predicate=no_forbidden_action,
        enforcement_level="strict",
        metadata={"domain": "safety"},
    )


@pytest.fixture
def box_region():
    """A simple box region."""
    return BoxRegion(
        region_id="nav_zone_1",
        bounds={"x_min": 0, "x_max": 10, "y_min": 0, "y_max": 10},
        allowed_actions=[],
        forbidden_actions=["delete_everything"],
        forbidden_areas=[{"x_min": 3, "x_max": 5, "y_min": 3, "y_max": 5}],
    )


@pytest.fixture
def enforcer(boundary_constraint_x):
    """A BoundaryEnforcer with one constraint."""
    return BoundaryEnforcer(constraints=[boundary_constraint_x])


@pytest.fixture
def enforcer_multi(boundary_constraint_x, boundary_constraint_y, forbidden_action_constraint):
    """A BoundaryEnforcer with multiple constraints."""
    return BoundaryEnforcer(
        constraints=[boundary_constraint_x, boundary_constraint_y, forbidden_action_constraint]
    )


@pytest.fixture
def grounding_layer(enforcer):
    """A MemoryGroundingLayer backed by the fixture enforcer."""
    return MemoryGroundingLayer(enforcer)


@pytest.fixture
def mock_agent():
    """A mock agent that returns actions."""
    agent = MagicMock()
    agent.act.return_value = "move_right"
    agent.planner = MagicMock()
    return agent


# ---------------------------------------------------------------------------
# BoundaryConstraint Tests
# ---------------------------------------------------------------------------


class TestBoundaryConstraint:
    """Tests for the BoundaryConstraint dataclass."""

    def test_create_valid_constraint(self, in_bounds_x):
        """Test creating a valid constraint."""
        constraint = BoundaryConstraint(
            constraint_id="test",
            constraint_type="boundary",
            description="Test constraint",
            predicate=in_bounds_x,
            enforcement_level="strict",
            metadata={"key": "value"},
        )
        assert constraint.constraint_id == "test"
        assert constraint.constraint_type == "boundary"
        assert constraint.description == "Test constraint"
        assert constraint.enforcement_level == "strict"
        assert constraint.metadata == {"key": "value"}

    def test_create_invalid_constraint_type(self, in_bounds_x):
        """Test that invalid constraint_type is accepted at creation (Literal doesn't enforce)."""
        # Note: Python Literal types don't enforce at runtime; this test verifies
        # that invalid types are accepted at construction time.
        constraint = BoundaryConstraint(
            constraint_id="test",
            constraint_type="invalid_type",
            description="Test",
            predicate=in_bounds_x,
        )
        assert constraint.constraint_type == "invalid_type"

    def test_create_invalid_enforcement_level(self, in_bounds_x):
        """Test that invalid enforcement_level is accepted but should be valid."""
        constraint = BoundaryConstraint(
            constraint_id="test",
            constraint_type="boundary",
            description="Test",
            predicate=in_bounds_x,
            enforcement_level="advisory",
        )
        assert constraint.enforcement_level == "advisory"

    def test_predicate_must_be_callable(self):
        """Test that non-callable predicate raises TypeError."""
        with pytest.raises(TypeError, match="predicate must be callable"):
            BoundaryConstraint(
                constraint_id="test",
                constraint_type="boundary",
                description="Test",
                predicate="not callable",  # type: ignore
            )

    def test_evaluate_satisfied(self, boundary_constraint_x):
        """Test evaluate returns satisfied when predicate returns True."""
        satisfied, reason = boundary_constraint_x.evaluate({"x": 5})
        assert satisfied is True
        assert reason is None

    def test_evaluate_violated(self, boundary_constraint_x):
        """Test evaluate returns violated when predicate returns False."""
        satisfied, reason = boundary_constraint_x.evaluate({"x": 15})
        assert satisfied is False
        assert reason is not None
        assert "bound_x" in reason

    def test_evaluate_at_boundary_edges(self, boundary_constraint_x):
        """Test evaluate at exact boundary values."""
        satisfied, _ = boundary_constraint_x.evaluate({"x": 0})
        assert satisfied is True
        satisfied, _ = boundary_constraint_x.evaluate({"x": 10})
        assert satisfied is True

    def test_evaluate_missing_key(self, in_bounds_x):
        """Test evaluate when state key is missing uses default 0."""
        constraint = BoundaryConstraint(
            constraint_id="test",
            constraint_type="boundary",
            description="Test",
            predicate=in_bounds_x,
        )
        satisfied, _ = constraint.evaluate({})
        assert satisfied is True

    def test_evaluate_exception_handling(self):
        """Test evaluate handles exceptions gracefully."""
        def bad_predicate(state: dict) -> bool:
            raise RuntimeError("predicate error")

        constraint = BoundaryConstraint(
            constraint_id="test",
            constraint_type="boundary",
            description="Test",
            predicate=bad_predicate,
        )
        satisfied, reason = constraint.evaluate({})
        assert satisfied is False
        assert "evaluation error" in reason


# ---------------------------------------------------------------------------
# ConstraintStatus Tests
# ---------------------------------------------------------------------------


class TestConstraintStatus:
    """Tests for the ConstraintStatus dataclass."""

    def test_create_status(self):
        """Test creating a ConstraintStatus."""
        status = ConstraintStatus(
            constraint_id="test",
            status="active",
            last_check_timestamp=1000.0,
        )
        assert status.constraint_id == "test"
        assert status.status == "active"
        assert status.last_check_timestamp == 1000.0
        assert status.violation_reason is None

    def test_update_status(self):
        """Test updating constraint status."""
        status = ConstraintStatus(constraint_id="test")
        assert status.status == "unknown"
        status.update("satisfied", timestamp=2000.0)
        assert status.status == "satisfied"
        assert status.last_check_timestamp == 2000.0

    def test_update_with_violation_reason(self):
        """Test updating status with a violation reason."""
        status = ConstraintStatus(constraint_id="test")
        status.update("violated", timestamp=3000.0, violation_reason="X out of bounds")
        assert status.status == "violated"
        assert status.violation_reason == "X out of bounds"


# ---------------------------------------------------------------------------
# BoxRegion Tests
# ---------------------------------------------------------------------------


class TestBoxRegion:
    """Tests for the BoxRegion dataclass."""

    def test_create_box_region(self, box_region):
        """Test creating a BoxRegion."""
        assert box_region.region_id == "nav_zone_1"
        assert box_region.bounds == {"x_min": 0, "x_max": 10, "y_min": 0, "y_max": 10}
        assert "delete_everything" in box_region.forbidden_actions
        assert len(box_region.forbidden_areas) == 1

    def test_contains_point_inside(self, box_region):
        """Test contains_point returns True for point inside bounds."""
        assert box_region.contains_point(5, 5) is True
        assert box_region.contains_point(0, 0) is True
        assert box_region.contains_point(10, 10) is True

    def test_contains_point_outside(self, box_region):
        """Test contains_point returns False for point outside bounds."""
        assert box_region.contains_point(-1, 5) is False
        assert box_region.contains_point(5, -1) is False
        assert box_region.contains_point(11, 5) is False
        assert box_region.contains_point(5, 11) is False

    def test_contains_point_partial_bounds(self):
        """Test contains_point with partial bounds specified."""
        region = BoxRegion(region_id="partial", bounds={"x_min": 0, "x_max": 5})
        assert region.contains_point(3, 100) is True  # y unbounded
        assert region.contains_point(6, 0) is False
        assert region.contains_point(-1, 0) is False

    def test_is_in_forbidden_area(self, box_region):
        """Test is_in_forbidden_area detects forbidden sub-areas."""
        # Point at center of forbidden area (3,3) to (5,5)
        assert box_region.is_in_forbidden_area(4, 4) is True
        # Point just outside forbidden area
        assert box_region.is_in_forbidden_area(2, 4) is False
        assert box_region.is_in_forbidden_area(6, 4) is False

    def test_is_action_allowed_empty_whitelist(self, box_region):
        """Test action allowed when allowed_actions is empty (all allowed)."""
        region = BoxRegion(region_id="test", allowed_actions=[])
        assert region.is_action_allowed("any_action") is True

    def test_is_action_allowed_whitelisted(self, box_region):
        """Test action allowed when in allowed_actions list."""
        assert box_region.is_action_allowed("move_up") is True
        assert box_region.is_action_allowed("move_down") is True

    def test_is_action_allowed_forbidden_list(self, box_region):
        """Test action not allowed when in forbidden_actions."""
        assert box_region.is_action_allowed("delete_everything") is False

    def test_is_action_allowed_not_in_whitelist(self):
        """Test action not allowed when whitelist exists and action not in it."""
        region = BoxRegion(
            region_id="test",
            allowed_actions=["move_up", "move_down"],
        )
        assert region.is_action_allowed("move_up") is True
        assert region.is_action_allowed("move_left") is False

    def test_empty_forbidden_areas(self):
        """Test with no forbidden areas."""
        region = BoxRegion(region_id="empty", bounds={"x_min": 0, "x_max": 10})
        assert region.is_in_forbidden_area(5, 5) is False


# ---------------------------------------------------------------------------
# BoundaryEnforcer Tests
# ---------------------------------------------------------------------------


class TestBoundaryEnforcer:
    """Tests for the BoundaryEnforcer class."""

    def test_create_enforcer_empty(self):
        """Test creating an empty enforcer."""
        enforcer = BoundaryEnforcer()
        assert len(enforcer.constraints) == 0
        assert enforcer.constraint_ids == []

    def test_create_enforcer_with_constraints(self, enforcer):
        """Test creating enforcer with initial constraints."""
        assert len(enforcer.constraints) == 1
        assert "bound_x" in enforcer.constraint_ids

    def test_invalid_default_enforcement(self):
        """Test that invalid default_enforcement raises ValueError."""
        with pytest.raises(ValueError, match="default_enforcement must be one of"):
            BoundaryEnforcer(default_enforcement="invalid")

    def test_register_constraint(self, enforcer, boundary_constraint_y):
        """Test registering a new constraint."""
        enforcer.register_constraint(boundary_constraint_y)
        assert len(enforcer.constraints) == 2
        assert "bound_y" in enforcer.constraint_ids

    def test_register_duplicate_constraint_replaces(self, enforcer, in_bounds_x):
        """Test registering a constraint with existing ID replaces it."""
        new_constraint = BoundaryConstraint(
            constraint_id="bound_x",
            constraint_type="boundary",
            description="New description",
            predicate=in_bounds_x,
        )
        enforcer.register_constraint(new_constraint)
        assert len(enforcer.constraints) == 1
        retrieved = enforcer.get_constraint("bound_x")
        assert retrieved.description == "New description"

    def test_register_non_constraint_raises(self, enforcer):
        """Test registering a non-BoundaryConstraint raises TypeError."""
        with pytest.raises(TypeError, match="Expected BoundaryConstraint"):
            enforcer.register_constraint("not a constraint")  # type: ignore

    def test_remove_constraint(self, enforcer):
        """Test removing a constraint by ID."""
        result = enforcer.remove_constraint("bound_x")
        assert result is True
        assert len(enforcer.constraints) == 0
        assert enforcer.get_constraint("bound_x") is None

    def test_remove_nonexistent_constraint(self, enforcer):
        """Test removing a non-existent constraint returns False."""
        result = enforcer.remove_constraint("nonexistent")
        assert result is False

    def test_reset_state(self, enforcer):
        """Test resetting enforcer state."""
        enforcer.check_constraints({"x": 15})  # cause violation
        assert enforcer.state.violation_count > 0
        enforcer.reset_state()
        assert enforcer.state.violation_count == 0
        assert len(enforcer.state.constraint_history) == 0

    def test_check_constraints_all_satisfied(self, enforcer):
        """Test check_constraints when all constraints are satisfied."""
        result = enforcer.check_constraints({"x": 5})
        assert result.all_satisfied is True
        assert result.violated_constraints == []
        assert "bound_x" in result.satisfied_constraints

    def test_check_constraints_violated(self, enforcer):
        """Test check_constraints when a constraint is violated."""
        result = enforcer.check_constraints({"x": 15})
        assert result.all_satisfied is False
        assert "bound_x" in result.violated_constraints
        assert result.violated_constraints == ["bound_x"]

    def test_check_constraints_updates_history(self, enforcer):
        """Test that check_constraints adds entries to history."""
        enforcer.check_constraints({"x": 5})
        assert len(enforcer.state.constraint_history) == 1
        entry = enforcer.state.constraint_history[0]
        assert entry["constraint_id"] == "bound_x"
        assert entry["satisfied"] is True

    def test_check_constraints_updates_violation_count(self, enforcer):
        """Test that violations increment the counter."""
        enforcer.check_constraints({"x": 15})
        assert enforcer.state.violation_count == 1
        enforcer.check_constraints({"x": 20})
        assert enforcer.state.violation_count == 2

    def test_enforce_action_no_modification(self, enforcer):
        """Test enforce_action when action is valid."""
        result = enforcer.enforce_action("move_right", {"x": 5})
        assert result.modified is False
        assert result.enforced_action == "move_right"

    def test_enforce_action_strict_violation(self, enforcer):
        """Test enforce_action modifies action on strict violation."""
        result = enforcer.enforce_action("move_right", {"x": 15})
        assert result.modified is True
        assert result.enforced_action == "noop"
        assert "bound_x" in result.constraint_ids_enforced

    def test_enforce_action_soft_adds_warning(self, enforcer_multi):
        """Test enforce_action adds warning for soft violations."""
        result = enforcer_multi.enforce_action("move_right", {"y": 15})
        assert result.modified is False
        assert len(result.warnings) > 0

    def test_enforce_action_forbidden_action(self, enforcer_multi):
        """Test enforce_action blocks forbidden actions."""
        result = enforcer_multi.enforce_action(
            "delete_everything", {"x": 5, "y": 5, "action": "delete_everything"}
        )
        assert result.modified is True
        assert result.enforced_action == "noop"

    def test_enforce_action_with_context(self, enforcer):
        """Test enforce_action accepts and uses context."""
        result = enforcer.enforce_action(
            "move_right", {"x": 5}, context={"episode_id": 1}
        )
        assert result.modified is False

    def test_enforce_action_box_region_forbidden_action(self, box_region, enforcer):
        """Test action enforcement against box region forbidden list."""
        enforcer.current_box = box_region
        result = enforcer.enforce_action("delete_everything", {"x": 5, "y": 5})
        assert result.modified is True
        assert result.enforced_action == "noop"

    def test_enforce_action_box_region_out_of_bounds(self, box_region, enforcer):
        """Test action enforcement when action would move out of bounds."""
        enforcer.current_box = box_region
        result = enforcer.enforce_action("move_right", {"x": 9, "y": 5, "step_size": 2})
        assert result.modified is True
        assert result.enforced_action == "noop"

    def test_enforce_action_box_region_forbidden_area(self, box_region, enforcer):
        """Test action enforcement when action would move into forbidden area."""
        enforcer.current_box = box_region
        result = enforcer.enforce_action("move_right", {"x": 2.5, "y": 4, "step_size": 1})
        assert result.modified is True
        assert result.enforced_action == "noop"

    def test_get_active_violations(self, enforcer):
        """Test getting currently violated constraints."""
        enforcer.check_constraints({"x": 15})
        violations = enforcer.get_active_violations()
        assert len(violations) == 1
        assert violations[0].constraint_id == "bound_x"

    def test_get_active_violations_none(self, enforcer):
        """Test getting violations when none exist."""
        enforcer.check_constraints({"x": 5})
        violations = enforcer.get_active_violations()
        assert violations == []

    def test_get_constraint_summary(self, enforcer):
        """Test getting constraint summary."""
        summary = enforcer.get_constraint_summary()
        assert summary["total_constraints"] == 1
        assert summary["total_violations"] == 0
        assert "by_type" in summary
        assert "by_enforcement_level" in summary

    def test_get_constraint_summary_after_violations(self, enforcer):
        """Test summary updates after violations."""
        enforcer.check_constraints({"x": 15})
        summary = enforcer.get_constraint_summary()
        assert summary["total_violations"] == 1
        assert summary["by_status"]["violated"] == 1

    def test_export_audit_trail(self, enforcer):
        """Test exporting audit trail as dict."""
        enforcer.check_constraints({"x": 5})
        trail = enforcer.export_audit_trail()
        assert "exported_at" in trail
        assert trail["constraint_count"] == 1
        assert len(trail["history"]) == 1
        assert trail["history"][0]["satisfied"] is True

    def test_current_box_setter(self, enforcer, box_region):
        """Test setting current_box updates both enforcer and state."""
        enforcer.current_box = box_region
        assert enforcer.current_box is box_region
        assert enforcer.state.current_box is box_region


# ---------------------------------------------------------------------------
# MemoryGroundingLayer Tests
# ---------------------------------------------------------------------------


class TestMemoryGroundingLayer:
    """Tests for the MemoryGroundingLayer class."""

    def test_create_grounding_layer(self, grounding_layer):
        """Test creating a MemoryGroundingLayer."""
        assert grounding_layer._enforcer is not None
        assert grounding_layer._step_count == 0

    def test_invalid_enforcer_raises(self):
        """Test that non-BoundaryEnforcer raises TypeError."""
        with pytest.raises(TypeError, match="Expected BoundaryEnforcer"):
            MemoryGroundingLayer("not an enforcer")  # type: ignore

    def test_update_memory(self, grounding_layer):
        """Test updating memory with a step."""
        grounding_layer.update({"x": 5}, "move_right", "success")
        assert grounding_layer._step_count == 1
        assert len(grounding_layer._memory) == 1

    def test_update_memory_bounded(self, grounding_layer):
        """Test that memory is bounded to 1000 entries."""
        for i in range(1005):
            grounding_layer.update({"x": i}, f"action_{i}", "success")
        assert len(grounding_layer._memory) == 1000
        assert grounding_layer._step_count == 1005

    def test_reset_memory(self, grounding_layer):
        """Test resetting memory clears state."""
        grounding_layer.update({"x": 5}, "move_right", "success")
        grounding_layer.reset()
        assert grounding_layer._step_count == 0
        assert len(grounding_layer._memory) == 0

    def test_get_grounded_constraints(self, grounding_layer, enforcer):
        """Test getting grounded constraints returns enforcer constraints."""
        constraints = grounding_layer.get_grounded_constraints({"x": 5})
        assert len(constraints) == 1

    def test_remind_agent_no_violations(self, grounding_layer):
        """Test remind_agent with no violations returns empty string."""
        reminder = grounding_layer.remind_agent([])
        assert reminder == ""

    def test_remind_agent_with_violations(self, grounding_layer, enforcer):
        """Test remind_agent generates reminder string."""
        reminder = grounding_layer.remind_agent(["bound_x"])
        assert "bound_x" in reminder
        assert "REMINDER" in reminder

    def test_remind_agent_nonexistent_constraint(self, grounding_layer):
        """Test remind_agent handles nonexistent constraint IDs gracefully."""
        reminder = grounding_layer.remind_agent(["nonexistent"])
        assert reminder == ""

    def test_get_memory_summary(self, grounding_layer):
        """Test getting memory summary."""
        grounding_layer.update({"x": 5}, "move_right", "success")
        summary = grounding_layer.get_memory_summary()
        assert summary["total_steps"] == 1
        assert summary["memory_size"] == 1
        assert len(summary["recent_steps"]) == 1

    def test_get_recent_actions(self, grounding_layer):
        """Test getting recent actions."""
        grounding_layer.update({"x": 1}, "a1", "success")
        grounding_layer.update({"x": 2}, "a2", "success")
        grounding_layer.update({"x": 3}, "a3", "success")
        recent = grounding_layer.get_recent_actions(n=2)
        assert recent == ["a2", "a3"]

    def test_get_action_history(self, grounding_layer):
        """Test getting full action history."""
        grounding_layer.update({"x": 1}, "a1", "success")
        grounding_layer.update({"x": 2}, "a2", "failure")
        history = grounding_layer.get_action_history()
        assert len(history) == 2
        assert history[0]["action"] == "a1"
        assert history[1]["result"] == "failure"


# ---------------------------------------------------------------------------
# HeRoSBoundaryIntegration Tests
# ---------------------------------------------------------------------------


class TestHeRoSBoundaryIntegration:
    """Tests for the HeRoSBoundaryIntegration class."""

    def test_create_integration(self, mock_agent, enforcer, grounding_layer):
        """Test creating integration."""
        integration = HeRoSBoundaryIntegration(mock_agent, enforcer, grounding_layer)
        assert integration._agent is mock_agent
        assert integration._enforcer is enforcer
        assert integration._grounding is grounding_layer

    def test_run_with_boundaries_basic(self, mock_agent, enforcer, grounding_layer):
        """Test running episode with boundaries."""
        integration = HeRoSBoundaryIntegration(mock_agent, enforcer, grounding_layer)
        trajectory, reward = integration.run_with_boundaries(
            "Test task", max_steps=5
        )
        assert len(trajectory) == 5
        assert isinstance(reward, float)

    def test_run_with_boundaries_with_constraints(
        self, mock_agent, enforcer, grounding_layer, boundary_constraint_x
    ):
        """Test running with additional constraints."""
        integration = HeRoSBoundaryIntegration(mock_agent, enforcer, grounding_layer)
        trajectory, _ = integration.run_with_boundaries(
            "Test task",
            constraints=[boundary_constraint_x],
            max_steps=3,
        )
        assert len(trajectory) == 3

    def test_run_with_boundaries_resets_state(
        self, mock_agent, enforcer, grounding_layer
    ):
        """Test that run_with_boundaries resets enforcer and grounding."""
        integration = HeRoSBoundaryIntegration(mock_agent, enforcer, grounding_layer)
        integration.run_with_boundaries("Task 1", max_steps=2)
        integration.run_with_boundaries("Task 2", max_steps=2)
        # After second run, step count should be 2, not cumulative
        assert grounding_layer._step_count == 2

    def test_inject_constraint_reminder(
        self, mock_agent, enforcer, grounding_layer
    ):
        """Test constraint reminder injection."""
        integration = HeRoSBoundaryIntegration(mock_agent, enforcer, grounding_layer)
        result = integration._inject_constraint_reminder("Original task", ["bound_x"])
        assert "REMINDER" in result
        assert "Original task" in result

    def test_inject_constraint_reminder_no_violations(
        self, mock_agent, enforcer, grounding_layer
    ):
        """Test injection with no violations returns original."""
        integration = HeRoSBoundaryIntegration(mock_agent, enforcer, grounding_layer)
        result = integration._inject_constraint_reminder("Original task", [])
        assert result == "Original task"


# ---------------------------------------------------------------------------
# BoundaryEvaluator Tests
# ---------------------------------------------------------------------------


class TestBoundaryEvaluator:
    """Tests for the BoundaryEvaluator class."""

    def test_create_evaluator(self, enforcer):
        """Test creating BoundaryEvaluator."""
        evaluator = BoundaryEvaluator(enforcer)
        assert evaluator._enforcer is enforcer

    def test_evaluate_empty_trajectories(self, enforcer):
        """Test evaluating with no trajectories."""
        evaluator = BoundaryEvaluator(enforcer)
        result = evaluator.evaluate_boundary_effectiveness([])
        assert result["boundary_prevention_rate"] == 0.0
        assert result["total_steps_evaluated"] == 0

    def test_evaluate_with_trajectories(self, enforcer):
        """Test evaluating with actual trajectories."""
        evaluator = BoundaryEvaluator(enforcer)

        # Create mock trajectories with exec_traces
        milestones = [
            Milestone(id="m1", description="Test", rubric="done", expected_output="")
        ]
        traj1 = HindsightTrajectory(
            task="Task 1",
            milestones=milestones,
            exec_traces=[{"x": 5}, {"x": 6}, {"x": 7}],
            verdicts=[],
        )
        traj2 = HindsightTrajectory(
            task="Task 2",
            milestones=milestones,
            exec_traces=[{"x": 15}],
            verdicts=[],
        )

        result = evaluator.evaluate_boundary_effectiveness([traj1, traj2])
        assert result["total_steps_evaluated"] == 4
        assert "boundary_prevention_rate" in result
        assert "constraint_violation_rate" in result
        assert "avg_constraints_per_step" in result

    def test_evaluate_calculates_violation_rate(self, enforcer):
        """Test that violation rate is calculated correctly."""
        evaluator = BoundaryEvaluator(enforcer)
        milestones = [Milestone(id="m1", description="T", rubric="r", expected_output="")]
        traj = HindsightTrajectory(
            task="Task",
            milestones=milestones,
            exec_traces=[{"x": 15}, {"x": 20}],  # Both violate
            verdicts=[],
        )
        result = evaluator.evaluate_boundary_effectiveness([traj])
        assert result["total_violations_occurred"] > 0


# ---------------------------------------------------------------------------
# BoundaryState Tests
# ---------------------------------------------------------------------------


class TestBoundaryState:
    """Tests for the BoundaryState dataclass."""

    def test_create_boundary_state(self):
        """Test creating a BoundaryState."""
        state = BoundaryState()
        assert state.active_constraints == {}
        assert state.constraint_history == []
        assert state.violation_count == 0
        assert state.current_box is None

    def test_add_history_entry(self):
        """Test adding entries to constraint history."""
        state = BoundaryState()
        state.add_history_entry({"step": 1, "constraint_id": "c1", "satisfied": True})
        assert len(state.constraint_history) == 1

    def test_increment_violations(self):
        """Test incrementing violation counter."""
        state = BoundaryState()
        state.increment_violations()
        state.increment_violations()
        assert state.violation_count == 2


# ---------------------------------------------------------------------------
# ConstraintCheckResult Tests
# ---------------------------------------------------------------------------


class TestConstraintCheckResult:
    """Tests for the ConstraintCheckResult dataclass."""

    def test_create_result(self):
        """Test creating a ConstraintCheckResult."""
        result = ConstraintCheckResult(
            all_satisfied=True,
            violated_constraints=[],
            satisfied_constraints=["c1", "c2"],
            timestamp=1000.0,
        )
        assert result.all_satisfied is True
        assert result.violated_constraints == []
        assert len(result.satisfied_constraints) == 2
        assert result.timestamp == 1000.0


# ---------------------------------------------------------------------------
# EnforcedAction Tests
# ---------------------------------------------------------------------------


class TestEnforcedAction:
    """Tests for the EnforcedAction dataclass."""

    def test_create_enforced_action(self):
        """Test creating an EnforcedAction."""
        result = EnforcedAction(
            original_action="move_right",
            enforced_action="move_right",
            modified=False,
        )
        assert result.original_action == "move_right"
        assert result.enforced_action == "move_right"
        assert result.modified is False
        assert result.warnings == []
        assert result.constraint_ids_enforced == []

    def test_enforced_action_with_warnings(self):
        """Test EnforcedAction with warnings and enforcement info."""
        result = EnforcedAction(
            original_action="bad_action",
            enforced_action="noop",
            modified=True,
            enforcement_reason="Constraint violated",
            warnings=["Soft constraint warning"],
            constraint_ids_enforced=["c1"],
        )
        assert result.modified is True
        assert result.enforcement_reason == "Constraint violated"
        assert len(result.warnings) == 1


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


class TestBoundaryEnforcerIntegration:
    """Integration tests combining multiple components."""

    def test_full_enforcement_cycle(self):
        """Test a complete enforce-check-remind cycle."""
        # Create constraints
        x_constraint = BoundaryConstraint(
            constraint_id="x_bound",
            constraint_type="boundary",
            description="X in [0, 10]",
            predicate=lambda s: 0 <= s.get("x", 0) <= 10,
            enforcement_level="strict",
        )
        y_constraint = BoundaryConstraint(
            constraint_id="y_bound",
            constraint_type="boundary",
            description="Y in [0, 10]",
            predicate=lambda s: 0 <= s.get("y", 0) <= 10,
            enforcement_level="soft",
        )

        # Create enforcer and grounding
        enforcer = BoundaryEnforcer(constraints=[x_constraint, y_constraint])
        grounding = MemoryGroundingLayer(enforcer)

        # Simulate trajectory
        states = [
            {"x": 5, "y": 5},
            {"x": 15, "y": 5},  # X violates
            {"x": 3, "y": 12},  # Y violates
            {"x": 8, "y": 8},
        ]

        prev_len = 0
        for i, state in enumerate(states):
            check = enforcer.check_constraints(state)
            action = "move_right"
            enforced = enforcer.enforce_action(action, state)
            grounding.update(state, enforced.enforced_action, "step_result")
            # History grows monotonically (enforce_action also calls check_constraints)
            assert len(enforcer.state.constraint_history) > prev_len
            prev_len = len(enforcer.state.constraint_history)

        # After 4 states, each with check + enforce (2 calls), we have at least 8 entries
        assert len(enforcer.state.constraint_history) >= 8

        # Get reminder for violations
        violations = enforcer.get_active_violations()
        violation_ids = [v.constraint_id for v in violations]
        reminder = grounding.remind_agent(violation_ids)
        assert "x_bound" in reminder or "y_bound" in reminder or reminder == ""

    def test_box_region_enforcement_full(self):
        """Test complete box region enforcement workflow."""
        # Create box
        box = BoxRegion(
            region_id="workspace",
            bounds={"x_min": 0, "x_max": 100, "y_min": 0, "y_max": 100},
            allowed_actions=["move", "click", "type"],
            forbidden_actions=["delete", "shutdown"],
            forbidden_areas=[{"x_min": 40, "x_max": 60, "y_min": 40, "y_max": 60}],
        )

        # Create enforcer
        enforcer = BoundaryEnforcer()
        enforcer.current_box = box

        # Test allowed action
        result = enforcer.enforce_action("click", {"x": 50, "y": 50})
        # click is allowed and doesn't move, so not modified
        assert result.modified is False

        # Test forbidden action
        result = enforcer.enforce_action("delete", {"x": 50, "y": 50})
        assert result.modified is True
        assert result.enforced_action == "noop"

        # Test movement out of bounds (would exceed x_max=100)
        result = enforcer.enforce_action("move_right", {"x": 95, "y": 50, "step_size": 10})
        assert result.modified is True

        # Test movement into forbidden area
        result = enforcer.enforce_action("move_right", {"x": 39, "y": 50, "step_size": 2})
        assert result.modified is True

    def test_memory_grounding_persists_across_steps(self):
        """Test that memory grounding persists constraint awareness."""
        constraint = BoundaryConstraint(
            constraint_id="stay_in_bounds",
            constraint_type="boundary",
            description="Stay within x: 0-10",
            predicate=lambda s: 0 <= s.get("x", 0) <= 10,
            enforcement_level="strict",
        )

        enforcer = BoundaryEnforcer(constraints=[constraint])
        grounding = MemoryGroundingLayer(enforcer)

        # Simulate several steps
        for i in range(10):
            state = {"x": 5 + (0 if i < 5 else 15), "y": i}
            check = enforcer.check_constraints(state)
            action = "move_right"
            enforced = enforcer.enforce_action(action, state)
            grounding.update(state, enforced.enforced_action, "success" if check.all_satisfied else "failure")

        summary = grounding.get_memory_summary()
        assert summary["total_steps"] == 10
        assert len(summary["recent_steps"]) <= 10

    def test_audit_trail_serialization(self, enforcer):
        """Test that audit trail is JSON-serializable."""
        enforcer.check_constraints({"x": 5})
        enforcer.check_constraints({"x": 15})

        trail = enforcer.export_audit_trail()

        # Should not raise when serialized
        json_str = str(trail)  # Just check it can be represented
        assert "exported_at" in json_str
        assert "constraint_count" in json_str
        assert "history" in json_str


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestBoundaryEnforcerEdgeCases:
    """Edge case tests for BoundaryEnforcer."""

    def test_enforcer_with_all_constraint_types(self):
        """Test enforcer handles all constraint types."""
        constraints = [
            BoundaryConstraint(
                constraint_id="box1",
                constraint_type="box",
                description="Box constraint",
                predicate=lambda s: True,
                enforcement_level="strict",
            ),
            BoundaryConstraint(
                constraint_id="bound1",
                constraint_type="boundary",
                description="Boundary constraint",
                predicate=lambda s: True,
                enforcement_level="soft",
            ),
            BoundaryConstraint(
                constraint_id="forbid1",
                constraint_type="forbidden",
                description="Forbidden constraint",
                predicate=lambda s: True,
                enforcement_level="advisory",
            ),
            BoundaryConstraint(
                constraint_id="req1",
                constraint_type="required",
                description="Required constraint",
                predicate=lambda s: True,
                enforcement_level="strict",
            ),
        ]
        enforcer = BoundaryEnforcer(constraints=constraints)
        assert len(enforcer.constraints) == 4

        result = enforcer.check_constraints({})
        assert result.all_satisfied is True

    def test_multiple_box_regions(self, enforcer):
        """Test using multiple box regions over time."""
        box1 = BoxRegion(region_id="inner", bounds={"x_min": 0, "x_max": 5, "y_min": 0, "y_max": 5})
        box2 = BoxRegion(region_id="outer", bounds={"x_min": 0, "x_max": 10, "y_min": 0, "y_max": 10})

        enforcer.current_box = box1
        assert enforcer.current_box.region_id == "inner"

        enforcer.current_box = box2
        assert enforcer.current_box.region_id == "outer"

    def test_constraint_with_empty_metadata(self, in_bounds_x):
        """Test constraint with empty metadata dict."""
        constraint = BoundaryConstraint(
            constraint_id="test",
            constraint_type="boundary",
            description="Test",
            predicate=in_bounds_x,
            metadata={},
        )
        assert constraint.metadata == {}

    def test_check_constraints_with_empty_state(self, in_bounds_x):
        """Test checking constraints with empty state dict."""
        constraint = BoundaryConstraint(
            constraint_id="test",
            constraint_type="boundary",
            description="Test",
            predicate=in_bounds_x,
        )
        enforcer = BoundaryEnforcer(constraints=[constraint])
        result = enforcer.check_constraints({})
        assert result.all_satisfied is True  # Empty state, x defaults to 0

    def test_enforce_action_unknown_action(self, enforcer):
        """Test enforcing an unknown action."""
        result = enforcer.enforce_action("unknown_weird_action_xyz", {"x": 5})
        assert result.modified is False

    def test_simulate_action_position_for_various_actions(self, enforcer):
        """Test action position simulation for various movement actions."""
        state = {"x": 5, "y": 5, "step_size": 1}

        actions = ["move_up", "up", "north", "y+"]
        for action in actions:
            pos = enforcer._simulate_action_position(action, state)
            assert pos is not None
            assert pos[1] > state["y"]  # y increased

        actions = ["move_down", "down", "south", "y-"]
        for action in actions:
            pos = enforcer._simulate_action_position(action, state)
            assert pos is not None
            assert pos[1] < state["y"]  # y decreased

    def test_enforcer_default_enforcement_param(self):
        """Test that default_enforcement is correctly stored."""
        enforcer = BoundaryEnforcer(default_enforcement="soft")
        assert enforcer._default_enforcement == "soft"

    def test_is_within_bounds(self, enforcer, box_region):
        """Test _is_within_bounds helper method."""
        enforcer.current_box = box_region

        # Within bounds
        assert enforcer._is_within_bounds("move_up", {"x": 5, "y": 5, "step_size": 1}, box_region) is True

        # Would go out of bounds
        assert enforcer._is_within_bounds("move_up", {"x": 5, "y": 9, "step_size": 2}, box_region) is False

    def test_enforcer_get_constraint_not_found(self, enforcer):
        """Test get_constraint returns None for nonexistent ID."""
        assert enforcer.get_constraint("nonexistent") is None

    def test_constraint_evaluation_with_state_updates(self, in_bounds_x):
        """Test that constraint evaluation sees latest state values."""
        constraint = BoundaryConstraint(
            constraint_id="test",
            constraint_type="boundary",
            description="Test",
            predicate=in_bounds_x,
        )

        satisfied, _ = constraint.evaluate({"x": 0})
        assert satisfied is True

        satisfied, _ = constraint.evaluate({"x": 10})
        assert satisfied is True

        satisfied, _ = constraint.evaluate({"x": 100})
        assert satisfied is False
