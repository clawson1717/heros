"""OS-Themis-style Milestone Critic Agent.

Reviews execution traces against milestone rubrics and produces verdicts.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Verdict(Enum):
    PASS = "pass"
    FAIL = "fail"
    PARTIAL = "partial"


@dataclass
class CriticResult:
    """Output from the milestone critic."""
    verdict: Verdict
    feedback: str
    confidence: float  # 0.0 to 1.0
    reward_signal: float  # 0.0 to 1.0 derived from verdict


class MilestoneCritic:
    """Reviews a milestone execution against its rubric.

    This is a stub implementation. The full version will use an LLM
    to perform OS-Themis-style milestone verification.
    """

    def __init__(self, backend: str = "rule-based"):
        self.backend = backend

    def review(
        self,
        milestone_description: str,
        rubric: str,
        execution_trace: str,
    ) -> CriticResult:
        """Review an execution against a milestone rubric.

        Args:
            milestone_description: What the milestone was supposed to do.
            rubric: The pass/fail criteria.
            execution_trace: What actually happened.

        Returns:
            A CriticResult with verdict, feedback, and reward signal.
        """
        # Stub: rule-based critic that checks for keyword presence
        # Full implementation will use LLM-as-critic
        rubric_keywords = rubric.lower().split()
        trace_lower = execution_trace.lower()

        matched = sum(1 for kw in rubric_keywords if kw in trace_lower)
        score = min(matched / max(len(rubric_keywords), 1), 1.0)

        if score >= 0.8:
            verdict = Verdict.PASS
            reward = 1.0
            feedback = "All rubric criteria met."
        elif score >= 0.4:
            verdict = Verdict.PARTIAL
            reward = 0.5
            feedback = "Partially met rubric criteria."
        else:
            verdict = Verdict.FAIL
            reward = 0.0
            feedback = "Rubric criteria not met."

        return CriticResult(
            verdict=verdict,
            feedback=feedback,
            confidence=0.5,  # Low confidence for rule-based stub
            reward_signal=reward,
        )

    def audit_reward(self, verdict: Verdict) -> float:
        """Convert a milestone verdict into a dense RL reward signal.

        Args:
            verdict: The milestone verdict.

        Returns:
            Reward in [0.0, 1.0].
        """
        mapping = {Verdict.PASS: 1.0, Verdict.PARTIAL: 0.5, Verdict.FAIL: 0.0}
        return mapping.get(verdict, 0.0)
