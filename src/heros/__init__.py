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
    # Version
    "__version__",
]
