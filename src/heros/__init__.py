"""HeRoS: Hindsight-driven Reinforcement with Subgoal Milestones."""

__version__ = "0.1.0"
__author__ = "HeRoS Team"

from heros.planner import SubgoalPlanner
from heros.critic import MilestoneCritic
from heros.buffer import HindsightBuffer, HindsightTrajectory
from heros.trainer import HindsightTrainer, UpdateResult

__all__ = [
    "SubgoalPlanner",
    "MilestoneCritic",
    "HindsightBuffer",
    "HindsightTrajectory",
    "HindsightTrainer",
    "UpdateResult",
    "__version__",
]
