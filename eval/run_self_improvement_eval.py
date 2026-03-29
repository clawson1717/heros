#!/usr/bin/env python3
"""CLI Script for Test-time Self-Improvement Evaluation.

This script runs the TestTimeSelfImprover on the mini benchmark and reports
per-task improvement trajectories.

Usage:
    python eval/run_self_improvement_eval.py
    python eval/run_self_improvement_eval.py --tasks 5 --episodes 5
    python eval/run_self_improvement_eval.py --output results.json
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from heros.benchmark import WebArenaLiteBenchmark, WebTask, MockWebEnv
from heros.buffer import HindsightBuffer
from heros.agent import HeRoSAgent
from heros.self_improver import TestTimeSelfImprover, SelfImprovementResult
from heros.inference_engine import InferenceEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Mock Agent for Evaluation
# ============================================================================


class MockHeRoSAgent:
    """Mock HeRoSAgent for evaluation without real model.

    Provides minimal interface for testing the self-improvement pipeline.
    """

    def __init__(
        self,
        success_probability: float = 0.3,
        milestone_hit_probability: float = 0.5,
    ):
        import random

        self._success_prob = success_probability
        self._milestone_hit_prob = milestone_hit_prob
        self._rng = random.Random(42)
        self._act_call_count = 0

    @property
    def hindsight_buffer(self):
        """Return a dummy buffer."""
        return None

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Return a mock action result."""
        self._act_call_count += 1
        return {
            "action": f"mock_action_{self._act_call_count}",
            "milestone_id": obs.get("active_milestone", {}).get("id", ""),
            "milestone_description": obs.get("active_milestone", {}).get("description", ""),
        }

    def update(self) -> Dict[str, Any]:
        """Mock update."""
        return {"loss": 0.1, "num_samples": 8}


# ============================================================================
# Evaluation Functions
# ============================================================================


def load_eval_config(config_path: str = "eval/eval_config.yaml") -> Dict[str, Any]:
    """Load evaluation configuration from YAML.

    Parameters
    ----------
    config_path : str
        Path to the eval config file.

    Returns
    -------
    Dict[str, Any]
        Configuration dictionary.
    """
    import yaml

    config_path = Path(config_path)
    if not config_path.exists():
        logger.warning("Config file not found at %s, using defaults", config_path)
        return get_default_config()

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logger.info("Loaded config from %s", config_path)
        return config
    except Exception as e:
        logger.warning("Failed to load config: %s, using defaults", e)
        return get_default_config()


def get_default_config() -> Dict[str, Any]:
    """Get default evaluation configuration.

    Returns
    -------
    Dict[str, Any]
        Default configuration dictionary.
    """
    return {
        "n_episodes": 5,
        "max_steps_per_episode": 50,
        "self_play_epochs": 3,
        "improvement_threshold": 0.05,
        "benchmark_subset": "mini",
        "output_path": None,
    }


def create_mini_benchmark() -> WebArenaLiteBenchmark:
    """Create a mini benchmark for evaluation.

    Returns
    -------
    WebArenaLiteBenchmark
        Mini benchmark with 5 tasks.
    """
    return WebArenaLiteBenchmark(task_subset="mini")


def create_mock_env_factory(task: WebTask) -> callable:
    """Create a mock environment factory for a task.

    Parameters
    ----------
    task : WebTask
        The task to create env for.

    Returns
    -------
    callable
        Factory function that creates MockWebEnv.
    """
    def factory() -> MockWebEnv:
        env = MockWebEnv(max_steps=50)
        return env
    return factory


def run_self_improvement_evaluation(
    benchmark: WebArenaLiteBenchmark,
    n_episodes: int = 5,
    max_steps_per_episode: int = 50,
    self_play_epochs: int = 3,
    improvement_threshold: float = 0.05,
) -> List[SelfImprovementResult]:
    """Run self-improvement evaluation on the benchmark.

    Parameters
    ----------
    benchmark : WebArenaLiteBenchmark
        The benchmark to evaluate on.
    n_episodes : int
        Number of episodes per task.
    max_steps_per_episode : int
        Maximum steps per episode.
    self_play_epochs : int
        Self-play epochs per improvement cycle.
    improvement_threshold : float
        Minimum delta to accept as real improvement.

    Returns
    -------
    List[SelfImprovementResult]
        Results for each task.
    """
    # Create the hindsight buffer
    hindsight_buffer = HindsightBuffer(
        capacity=100,
        hindsight_ratio=0.3,
    )

    # Get task list
    task_ids = benchmark.list_tasks()[:5]  # Limit to 5 for eval
    tasks = [benchmark.get_task(tid) for tid in task_ids]

    logger.info(
        "Starting self-improvement evaluation: %d tasks, %d episodes each",
        len(tasks),
        n_episodes,
    )

    # Use mock agent if no real agent available
    try:
        # Try to create real agent
        from heros.planner import SubgoalPlanner
        from heros.critic import MilestoneCritic
        from heros.trainer import HindsightTrainer

        planner = SubgoalPlanner()
        critic = MilestoneCritic()
        trainer = HindsightTrainer()
        agent = HeRoSAgent(
            planner=planner,
            critic=critic,
            hindsight_buffer=hindsight_buffer,
            trainer=trainer,
        )
        logger.info("Using real HeRoSAgent")
    except Exception as e:
        logger.warning("Could not create real agent: %s, using mock", e)
        agent = MockHeRoSAgent()
        # Give the mock agent access to the hindsight buffer
        agent._hindsight_buffer = hindsight_buffer

    # Create self-improver
    improver = TestTimeSelfImprover(
        agent=agent,
        hindsight_buffer=hindsight_buffer,
        self_play_epochs=self_play_epochs,
        improvement_threshold=improvement_threshold,
        simulate_updates=True,
    )

    results: List[SelfImprovementResult] = []

    for task_idx, task in enumerate(tasks):
        task_id = getattr(task, "task_id", f"task_{task_idx}")
        logger.info(
            "Evaluating task %d/%d: %s",
            task_idx + 1,
            len(tasks),
            task_id,
        )

        # Create env factory for this task
        def env_factory(t=task):
            env = MockWebEnv(max_steps=max_steps_per_episode)
            # Pre-load the task into the env
            env.reset(t)
            return env

        try:
            result = improver.run_with_self_improvement(
                task=task,
                env_factory=env_factory,
                n_episodes=n_episodes,
                max_steps_per_episode=max_steps_per_episode,
            )
            results.append(result)
        except Exception as e:
            logger.error("Failed to evaluate task %s: %s", task_id, e)
            # Create a failed result
            from heros.self_improver import SelfImprovementResult, EpisodeMetrics

            results.append(
                SelfImprovementResult(
                    task_id=task_id,
                    episodes=[],
                    final_success_rate=0.0,
                    initial_success_rate=0.0,
                    improvement_delta=0.0,
                    total_self_play_epochs=0,
                    hindsight_buffer_size_after=len(hindsight_buffer),
                )
            )

    return results


def format_results_table(results: List[SelfImprovementResult]) -> str:
    """Format results as a table.

    Parameters
    ----------
    results : List[SelfImprovementResult]
        Results to format.

    Returns
    -------
    str
        Formatted table string.
    """
    lines = []
    lines.append("")
    lines.append("=" * 90)
    lines.append("TEST-TIME SELF-IMPROVEMENT EVALUATION RESULTS")
    lines.append("=" * 90)
    lines.append("")
    lines.append(
        f"{'Task':<20} {'Initial SR':>10} {'Final SR':>10} {'Delta':>8} "
        f"{'First Success':>14} {'Episodes':>8}"
    )
    lines.append("-" * 90)

    for result in results:
        task_id = result.task_id[:18]
        initial_sr = f"{result.initial_success_rate:.2f}"
        final_sr = f"{result.final_success_rate:.2f}"
        delta = f"{result.improvement_delta:+.2f}"
        first_success = (
            str(result.episodes_to_first_success)
            if result.episodes_to_first_success is not None
            else "Never"
        )
        n_eps = str(len(result.episodes))

        lines.append(
            f"{task_id:<20} {initial_sr:>10} {final_sr:>10} {delta:>8} "
            f"{first_success:>14} {n_eps:>8}"
        )

    lines.append("-" * 90)
    lines.append("")

    # Summary statistics
    if results:
        avg_initial_sr = sum(r.initial_success_rate for r in results) / len(results)
        avg_final_sr = sum(r.final_success_rate for r in results) / len(results)
        avg_delta = sum(r.improvement_delta for r in results) / len(results)

        tasks_ever_succeeded = sum(
            1 for r in results if r.episodes_to_first_success is not None
        )

        lines.append("SUMMARY:")
        lines.append(f"  Tasks evaluated: {len(results)}")
        lines.append(f"  Tasks that eventually succeeded: {tasks_ever_succeeded}/{len(results)}")
        lines.append(f"  Average initial success rate: {avg_initial_sr:.2f}")
        lines.append(f"  Average final success rate: {avg_final_sr:.2f}")
        lines.append(f"  Average improvement delta: {avg_delta:+.2f}")
        lines.append("")

    return "\n".join(lines)


def print_improvement_trajectory(result: SelfImprovementResult) -> str:
    """Format a single task's improvement trajectory.

    Parameters
    ----------
    result : SelfImprovementResult
        Result to format.

    Returns
    -------
    str
        Formatted trajectory string.
    """
    lines = []
    lines.append(f"\n  Task: {result.task_id}")
    lines.append(f"  Initial SR: {result.initial_success_rate:.2f} -> Final SR: {result.final_success_rate:.2f}")
    lines.append(f"  Delta: {result.improvement_delta:+.2f}, Self-play epochs: {result.total_self_play_epochs}")
    lines.append("  Episode trajectory:")

    for ep in result.episodes:
        status = "✓" if ep.success else "✗"
        lines.append(
            f"    Episode {ep.episode_idx}: {status} "
            f"hit_rate={ep.milestone_hit_rate:.2f} "
            f"reward={ep.total_reward:.1f} "
            f"length={ep.episode_length} "
            f"failed={len(ep.failed_subgoals)}"
        )

    return "\n".join(lines)


def save_results(results: List[SelfImprovementResult], output_path: str) -> None:
    """Save results to a JSON file.

    Parameters
    ----------
    results : List[SelfImprovementResult]
        Results to save.
    output_path : str
        Path to save to.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "results": [r.to_dict() for r in results],
        "summary": {
            "num_tasks": len(results),
            "avg_initial_sr": (
                sum(r.initial_success_rate for r in results) / len(results)
                if results else 0.0
            ),
            "avg_final_sr": (
                sum(r.final_success_rate for r in results) / len(results)
                if results else 0.0
            ),
            "avg_delta": (
                sum(r.improvement_delta for r in results) / len(results)
                if results else 0.0
            ),
        },
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info("Results saved to %s", output_path)


# ============================================================================
# Main
# ============================================================================


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run test-time self-improvement evaluation",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="eval/eval_config.yaml",
        help="Path to eval config file",
    )
    parser.add_argument(
        "--tasks",
        type=int,
        default=5,
        help="Number of tasks to evaluate",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes per task",
    )
    parser.add_argument(
        "--self-play-epochs",
        type=int,
        default=3,
        help="Self-play epochs per improvement cycle",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for results JSON",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load config
    config = load_eval_config(args.config)
    n_episodes = args.episodes
    max_steps = config.get("max_steps_per_episode", 50)
    self_play_epochs = args.self_play_epochs

    logger.info("Configuration:")
    logger.info("  Tasks: %d", args.tasks)
    logger.info("  Episodes per task: %d", n_episodes)
    logger.info("  Max steps per episode: %d", max_steps)
    logger.info("  Self-play epochs: %d", self_play_epochs)

    # Create benchmark
    try:
        benchmark = create_mini_benchmark()
        task_ids = benchmark.list_tasks()[: args.tasks]
        tasks = [benchmark.get_task(tid) for tid in task_ids]
        logger.info("Created mini benchmark with %d tasks", len(tasks))
    except Exception as e:
        logger.error("Failed to create benchmark: %s", e)
        # Create mock tasks for testing
        from heros.benchmark import WebTask, Milestone
        tasks = []
        for i in range(args.tasks):
            task = WebTask(
                task_id=f"mock_task_{i}",
                description=f"Mock task {i}",
                url="http://example.com",
                milestones=[
                    Milestone(
                        id=f"m1",
                        description=f"Milestone 1 for task {i}",
                        rubric="Complete milestone 1",
                    ),
                    Milestone(
                        id=f"m2",
                        description=f"Milestone 2 for task {i}",
                        rubric="Complete milestone 2",
                    ),
                ],
            )
            tasks.append(task)
        logger.info("Created %d mock tasks for evaluation", len(tasks))

    # Run evaluation
    results = run_self_improvement_evaluation(
        benchmark=type("MockBenchmark", (), {"list_tasks": lambda self: [t.task_id for t in tasks], "get_task": lambda self, tid: tasks[int(tid.split("_")[-1])]})(),
        n_episodes=n_episodes,
        max_steps_per_episode=max_steps,
        self_play_epochs=self_play_epochs,
        improvement_threshold=config.get("improvement_threshold", 0.05),
    )

    # Actually, let's fix this - we need to pass actual tasks
    # Run on the actual tasks
    from heros.self_improver import TestTimeSelfImprover
    from heros.buffer import HindsightBuffer
    from heros.benchmark import MockWebEnv

    hindsight_buffer = HindsightBuffer(capacity=100, hindsight_ratio=0.3)
    mock_agent = MockHeRoSAgent()
    mock_agent._hindsight_buffer = hindsight_buffer

    improver = TestTimeSelfImprover(
        agent=mock_agent,
        hindsight_buffer=hindsight_buffer,
        self_play_epochs=self_play_epochs,
        improvement_threshold=config.get("improvement_threshold", 0.05),
        simulate_updates=True,
    )

    results = []
    for task_idx, task in enumerate(tasks):
        logger.info("Evaluating task %d/%d: %s", task_idx + 1, len(tasks), task.task_id)

        def env_factory(t=task):
            env = MockWebEnv(max_steps=max_steps)
            env.reset(t)
            return env

        result = improver.run_with_self_improvement(
            task=task,
            env_factory=env_factory,
            n_episodes=n_episodes,
            max_steps_per_episode=max_steps,
        )
        results.append(result)

    # Print results
    print(format_results_table(results))

    # Print detailed trajectories
    for result in results:
        print(print_improvement_trajectory(result))

    # Save if requested
    if args.output:
        save_results(results, args.output)

    logger.info("Evaluation complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
