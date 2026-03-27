"""Basic skeleton tests for HeRoS project structure."""

import os
import pathlib
import sys

import pytest

# Ensure the src directory is on the path
REPO_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def test_package_import():
    """Package can be imported."""
    import heros
    assert heros.__version__ == "0.1.0"


def test_planner_import():
    """Planner module can be imported."""
    from heros.planner import SubgoalPlanner, Milestone, SubgoalPlan
    assert SubgoalPlanner is not None
    assert Milestone is not None
    assert SubgoalPlan is not None


def test_critic_import():
    """Critic module can be imported."""
    from heros.critic import MilestoneCritic, CriticResult, Verdict
    assert MilestoneCritic is not None
    assert CriticResult is not None
    assert Verdict is not None


def test_planner_stub():
    """Planner produces a valid plan."""
    from heros.planner import SubgoalPlanner
    planner = SubgoalPlanner(planning_depth=3)
    plan = planner.plan("Test task")
    assert plan.task == "Test task"
    assert len(plan.milestones) >= 1


def test_critic_stub():
    """Critic produces a valid result."""
    from heros.critic import MilestoneCritic, Verdict
    critic = MilestoneCritic()
    result = critic.review(
        milestone_description="Test milestone",
        rubric="Must contain keyword",
        execution_trace="This contains keyword",
    )
    assert result.verdict in [Verdict.PASS, Verdict.FAIL, Verdict.PARTIAL]
    assert 0.0 <= result.reward_signal <= 1.0


def test_config_dir_exists():
    """configs/ directory exists."""
    configs = REPO_ROOT / "configs"
    assert configs.exists()
    assert configs.is_dir()


def test_data_dir_exists():
    """data/ directory exists."""
    data = REPO_ROOT / "data"
    assert data.exists()
    assert data.is_dir()


def test_eval_dir_exists():
    """eval/ directory exists."""
    eval_dir = REPO_ROOT / "eval"
    assert eval_dir.exists()
    assert eval_dir.is_dir()


def test_scripts_dir_exists():
    """scripts/ directory exists."""
    scripts = REPO_ROOT / "scripts"
    assert scripts.exists()
    assert scripts.is_dir()
