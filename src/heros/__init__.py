"""HeRoS — Hindsight-driven Reinforcement with Subgoal Milestones."""

__version__ = "0.1.0"

from .planner import SubgoalPlanner, Milestone, LLMPlanner
from .critic import Verdict, CriticResult, MilestoneCritic, RewardAuditor
