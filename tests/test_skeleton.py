"""Skeleton tests for HeRoS project structure."""

import pathlib

import pytest

# Ensure the package is importable
import heros


class TestPackageImports:
    """Verify basic package imports."""

    def test_version_exists(self):
        """Package exposes __version__."""
        assert hasattr(heros, "__version__")
        assert isinstance(heros.__version__, str)
        assert heros.__version__ == "0.1.0"

    def test_planner_importable(self):
        """SubgoalPlanner is importable from heros."""
        from heros import SubgoalPlanner
        assert SubgoalPlanner is not None

    def test_critic_importable(self):
        """MilestoneCritic is importable from heros."""
        from heros import MilestoneCritic
        assert MilestoneCritic is not None


class TestProjectStructure:
    """Verify expected directories exist."""

    def test_configs_dir_exists(self):
        """configs/ directory exists."""
        configs = pathlib.Path(__file__).parent.parent / "configs"
        assert configs.is_dir()

    def test_data_dir_structure(self):
        """data/ directory exists and has expected subdirs."""
        data = pathlib.Path(__file__).parent.parent / "data"
        assert data.is_dir()

    def test_logs_dir_exists(self):
        """logs/ directory exists."""
        logs = pathlib.Path(__file__).parent.parent / "logs"
        assert logs.is_dir()

    def test_scripts_dir_exists(self):
        """scripts/ directory exists."""
        scripts = pathlib.Path(__file__).parent.parent / "scripts"
        assert scripts.is_dir()

    def test_eval_dir_exists(self):
        """eval/ directory exists."""
        eval_dir = pathlib.Path(__file__).parent.parent / "eval"
        assert eval_dir.is_dir()


class TestStubBehavior:
    """Verify stub components have basic runnable behavior."""

    def test_planner_instantiation(self):
        """Planner can be instantiated and used with a mock API key."""
        from unittest.mock import MagicMock, patch
        from heros import SubgoalPlanner

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(
            content='{"milestones":[{"id":"m1","description":"Write a test","rubric":"Test written","expected_output":"test.py"}]}'
        ))]

        with patch("heros.planner.OpenAI") as mock_openai_cls:
            mock_openai_cls.return_value.chat.completions.create = MagicMock(
                return_value=mock_response
            )
            planner = SubgoalPlanner(planning_depth=3, api_key="sk-test")
            plan = planner.plan("Write a test")
            assert plan.task == "Write a test"
            assert len(plan.milestones) >= 1

    def test_critic_instantiation(self):
        """Critic can be instantiated and used."""
        from heros import MilestoneCritic
        critic = MilestoneCritic(backend="rule-based")
        result = critic.review(
            milestone_description="Write a test",
            rubric="test written successfully",
            execution_trace="test written successfully and passed",
        )
        assert result.verdict is not None
        assert 0.0 <= result.reward_signal <= 1.0
