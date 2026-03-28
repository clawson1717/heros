#!/usr/bin/env python3
"""CLI script for running HeRoS WebArena-Lite / MiniWoB evaluations.

Usage:
    python -m heros.eval.run_evaluation --agent baseline
    python -m heros.eval.run_evaluation --agent heros --hindsight
    python -m heros.eval.run_evaluation --compare --output results.json
    python -m heros.eval.run_evaluation --benchmark full --tasks change_theme_dark logout_session

Examples:
    # Run baseline evaluation on mini benchmark
    python -m heros.eval.run_evaluation --agent baseline

    # Run HeRoS evaluation with hindsight enabled
    python -m heros.eval.run_evaluation --agent heros --hindsight

    # Compare baseline vs HeRoS
    python -m heros.eval.run_evaluation --compare

    # Run specific tasks on full benchmark
    python -m heros.eval.run_evaluation --benchmark full --tasks change_theme_dark contact_form_fill
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import heros
from heros import (
    WebArenaLiteBenchmark,
    BaselineAgent,
    HeRoSWrappedAgent,
    HeRoSEvaluator,
)
from heros.planner import SubgoalPlanner
from heros.critic import MilestoneCritic
from heros.buffer import HindsightBuffer
from heros.trainer import HindsightTrainer
from heros.agent import HeRoSAgent

logger = logging.getLogger(__name__)


# ============================================================================
# Utility Functions
# ============================================================================


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the evaluation script."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load evaluation configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "configs" / "eval_config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        logger.warning("Config file not found at %s, using defaults", config_path)
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_baseline_agent(config: Dict[str, Any]) -> BaselineAgent:
    """Create a BaselineAgent from configuration."""
    agent_config = config.get("agents", {}).get("baseline", {})

    return BaselineAgent(
        model_name=agent_config.get("model_name", "gpt-4o-mini"),
        api_key=os.environ.get("OPENAI_API_KEY"),  # Will use env var
        temperature=agent_config.get("temperature", 0.7),
        max_tokens=agent_config.get("max_tokens", 256),
    )


def create_heros_agent(config: Dict[str, Any]) -> HeRoSWrappedAgent:
    """Create a HeRoSWrappedAgent from configuration."""
    agent_config = config.get("agents", {}).get("heros", {})
    hindsight_enabled = agent_config.get("use_hindsight", True)

    # Create the underlying HeRoS agent components
    # For evaluation, we create a minimal working agent
    planner = SubgoalPlanner(
        planning_depth=3,
        api_key=os.environ.get("OPENAI_API_KEY", "dummy"),
    )
    critic = MilestoneCritic(backend="rule-based")
    buffer = HindsightBuffer(capacity=100)
    trainer = HindsightTrainer(buffer=buffer)

    # Create the base HeRoS agent
    base_agent = HeRoSAgent(
        planner=planner,
        critic=critic,
        hindsight_buffer=buffer,
        trainer=trainer,
    )

    # Wrap it for evaluation
    return HeRoSWrappedAgent(base_agent, hindsight_enabled=hindsight_enabled)


def run_evaluation(
    benchmark: WebArenaLiteBenchmark,
    agent: Any,
    agent_type: str,
    max_steps: int = 20,
    task_ids: Optional[List[str]] = None,
) -> List[heros.EvaluationResult]:
    """Run evaluation with the specified agent."""
    evaluator = HeRoSEvaluator(
        benchmark=benchmark,
        agent=agent,
        use_hindsight=getattr(agent, "hindsight_enabled", True),
        max_steps=max_steps,
    )

    return evaluator.run_evaluation(task_ids=task_ids)


def print_results(
    results: List[heros.EvaluationResult],
    agent_type: str,
    metrics: Dict[str, Any],
    verbose: bool = False,
) -> None:
    """Print evaluation results to console."""
    print("\n" + "=" * 60)
    print(f"Evaluation Results: {agent_type.upper()} Agent")
    print("=" * 60)

    print(f"\nTotal tasks evaluated: {metrics['total_tasks']}")
    print(f"Completion rate: {metrics['completion_rate']:.1%}")
    print(f"Avg milestone hit rate: {metrics['avg_milestone_hit_rate']:.1%}")
    print(f"Avg reward: {metrics['avg_reward']:.2f}")
    print(f"Avg episode length: {metrics['avg_episode_length']:.1f} steps")

    if metrics.get("hindsight_delta") is not None:
        print(f"Hindsight delta: {metrics['hindsight_delta']:+.1%}")

    if verbose:
        print("\nPer-task results:")
        for r in metrics.get("per_task_results", []):
            status = "✓" if r["completion"] else "✗"
            print(f"  {status} {r['task_id']}: hit_rate={r['milestone_hit_rate']:.0%}, "
                  f"reward={r['total_reward']:.2f}, steps={r['episode_length']}")


def save_results(
    results: List[heros.EvaluationResult],
    metrics: Dict[str, Any],
    output_path: str,
    agent_type: str,
) -> None:
    """Save evaluation results to a JSON file."""
    output_data = {
        "agent_type": agent_type,
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics,
        "results": [r.to_dict() for r in results],
    }

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    logger.info("Results saved to %s", output_path)


# ============================================================================
# Main CLI
# ============================================================================


def main() -> int:
    """Main entry point for the evaluation CLI."""
    parser = argparse.ArgumentParser(
        description="Run HeRoS WebArena-Lite / MiniWoB evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--agent",
        choices=["baseline", "heros"],
        help="Agent type to evaluate",
    )
    parser.add_argument(
        "--benchmark",
        choices=["mini", "full", "easy", "medium", "hard"],
        default="mini",
        help="Benchmark task subset (default: mini)",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        help="Specific task IDs to evaluate",
    )
    parser.add_argument(
        "--hindsight",
        action="store_true",
        help="Enable hindsight (for HeRoS agent)",
    )
    parser.add_argument(
        "--no-hindsight",
        dest="hindsight",
        action="store_false",
        help="Disable hindsight",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=20,
        help="Max steps per episode (default: 20)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare baseline vs HeRoS agents",
    )
    parser.add_argument(
        "--config",
        help="Path to evaluation config YAML",
    )
    parser.add_argument(
        "--output",
        help="Output file path for results (JSON)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress non-essential output",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(verbose=args.verbose)

    # Load configuration
    config = load_config(args.config)

    # Validate arguments
    if args.compare:
        # Comparison mode: evaluate both agents
        pass
    elif args.agent is None:
        parser.error("--agent is required (or use --compare to run both)")

    try:
        # Create benchmark
        benchmark = WebArenaLiteBenchmark(task_subset=args.benchmark)
        logger.info("Loaded benchmark: %s with %d tasks", args.benchmark, len(benchmark))

        # Determine output settings
        save_output = args.output is not None
        verbose = args.verbose and not args.quiet

        if args.compare:
            # Run comparison: baseline vs heros
            print("\n" + "=" * 60)
            print("Running Baseline Agent Evaluation")
            print("=" * 60)

            baseline_agent = create_baseline_agent(config)
            baseline_results = run_evaluation(
                benchmark,
                baseline_agent,
                "baseline",
                max_steps=args.max_steps,
                task_ids=args.tasks,
            )

            baseline_evaluator = HeRoSEvaluator(
                benchmark, baseline_agent, use_hindsight=False, max_steps=args.max_steps
            )
            baseline_metrics = baseline_evaluator.compute_metrics(baseline_results)

            if verbose:
                print_results(baseline_results, "baseline", baseline_metrics, verbose=True)

            print("\n" + "=" * 60)
            print("Running HeRoS Agent Evaluation")
            print("=" * 60)

            heros_agent = create_heros_agent(config)
            heros_results = run_evaluation(
                benchmark,
                heros_agent,
                "heros",
                max_steps=args.max_steps,
                task_ids=args.tasks,
            )

            heros_evaluator = HeRoSEvaluator(
                benchmark, heros_agent, use_hindsight=args.hindsight, max_steps=args.max_steps
            )
            heros_metrics = heros_evaluator.compute_metrics(heros_results)

            if verbose:
                print_results(heros_results, "heros", heros_metrics, verbose=True)

            # Compare results
            comparison = baseline_evaluator.compare_agents(baseline_results, heros_results)

            print("\n" + "=" * 60)
            print("COMPARISON SUMMARY")
            print("=" * 60)
            print(f"\n{comparison['summary']}")
            print(f"\nBaseline completion rate:  {comparison['baseline_metrics']['completion_rate']:.1%}")
            print(f"HeRoS completion rate:     {comparison['heros_metrics']['completion_rate']:.1%}")
            print(f"Improvement:               {comparison['improvement']['completion_rate_delta']:+.1%}")
            print(f"\nBaseline milestone hit:    {comparison['baseline_metrics']['avg_milestone_hit_rate']:.1%}")
            print(f"HeRoS milestone hit:       {comparison['heros_metrics']['avg_milestone_hit_rate']:.1%}")
            print(f"Improvement:               {comparison['improvement']['milestone_hit_rate_delta']:+.1%}")

            # Save comparison results
            if save_output:
                save_results(
                    baseline_results + heros_results,
                    {"comparison": comparison, "baseline": baseline_metrics, "heros": heros_metrics},
                    args.output,
                    "comparison",
                )
                print(f"\nResults saved to: {args.output}")

        else:
            # Single agent evaluation
            agent_type = args.agent

            if agent_type == "baseline":
                agent = create_baseline_agent(config)
            else:
                agent = create_heros_agent(config)

            logger.info("Evaluating %s agent", agent_type)

            results = run_evaluation(
                benchmark,
                agent,
                agent_type,
                max_steps=args.max_steps,
                task_ids=args.tasks,
            )

            evaluator = HeRoSEvaluator(
                benchmark,
                agent,
                use_hindsight=args.hindsight,
                max_steps=args.max_steps,
            )
            metrics = evaluator.compute_metrics(results)

            print_results(results, agent_type, metrics, verbose=verbose)

            # Save results
            if save_output:
                save_results(results, metrics, args.output, agent_type)
                print(f"\nResults saved to: {args.output}")

        print()
        return 0

    except Exception as e:
        logger.error("Evaluation failed: %s", e)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
