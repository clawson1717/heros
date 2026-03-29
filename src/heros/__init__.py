"""HeRoS: Hindsight-driven Reinforcement with Subgoal Milestones."""

__version__ = "0.1.0"
__author__ = "HeRoS Team"

from heros.planner import SubgoalPlanner
from heros.critic import MilestoneCritic
from heros.buffer import HindsightBuffer, HindsightTrajectory
from heros.trainer import HindsightTrainer, UpdateResult
from heros.env import HeRoSEnv, MilestoneStatus
from heros.agent import HeRoSAgent, ActResult
from heros.core import PPOTrainer
from heros.logging_utils import TrainingMetrics, TrainingLogger

# Benchmark and evaluation
from heros.benchmark import (
    WebArenaLiteBenchmark,
    WebTask,
    MockWebEnv,
    WebAction,
    EvaluationAction,
)
from heros.evaluator import HeRoSEvaluator, EvaluationResult
from heros.baseline_agent import BaselineAgent
from heros.heros_agent_wrapper import HeRoSWrappedAgent

# Interpretability and Audit Trail (Step 8)
from heros.interpretability import (
    MilestoneDecisionType,
    MilestoneDecisionLogger,
    FunctionalEquivalenceResult,
    FunctionalInterchangeabilityCheck,
    MilestoneAuditEntry,
    RewardAuditTrail,
    RewardAuditor,
    BufferCompositionAnalyzer,
    plot_buffer_composition,
)

__all__ = [
    # Planner
    "SubgoalPlanner",
    # Critic
    "MilestoneCritic",
    # Buffer
    "HindsightBuffer",
    "HindsightTrajectory",
    # Trainer
    "HindsightTrainer",
    "UpdateResult",
    # Env
    "HeRoSEnv",
    "MilestoneStatus",
    # Agent
    "HeRoSAgent",
    "ActResult",
    # Core
    "PPOTrainer",
    # Logging
    "TrainingMetrics",
    "TrainingLogger",
    # Benchmark
    "WebArenaLiteBenchmark",
    "WebTask",
    "MockWebEnv",
    "WebAction",
    "EvaluationAction",
    # Evaluator
    "HeRoSEvaluator",
    "EvaluationResult",
    # Agents
    "BaselineAgent",
    "HeRoSWrappedAgent",
    # Interpretability (Step 8)
    "MilestoneDecisionType",
    "MilestoneDecisionLogger",
    "FunctionalEquivalenceResult",
    "FunctionalInterchangeabilityCheck",
    "MilestoneAuditEntry",
    "RewardAuditTrail",
    "RewardAuditor",
    "BufferCompositionAnalyzer",
    "plot_buffer_composition",
    # Version
    "__version__",
]
