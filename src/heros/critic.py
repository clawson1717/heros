"""OS-Themis-style Milestone Critic Agent."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

# === Enums and Dataclasses ===

class Verdict(Enum):
    PASS = "pass"
    FAIL = "fail"
    PARTIAL = "partial"


@dataclass
class CriticResult:
    verdict: Verdict
    feedback: str
    confidence: float  # 0.0 to 1.0
    reward_signal: float  # 0.0 to 1.0


# === LLM Client Protocol ===
class LLMClient(Protocol):
    def complete(self, prompt: str, **kwargs: Any) -> str: ...


# === MilestoneCritic ===
class MilestoneCritic:
    """
    OS-Themis-style milestone critic.

    Reviews execution traces against milestone rubrics and produces verdicts.
    Supports two backends: "rule-based" (keyword matching) and "llm" (LLM-as-critic).

    Parameters
    ----------
    backend : str
        Either "rule-based" (default) or "llm".
    llm_client : LLMClient, optional
        Required if backend="llm". Protocol-compatible with OpenAI, Anthropic, etc.
    """

    def __init__(
        self,
        backend: str = "rule-based",
        llm_client: LLMClient | None = None,
    ) -> None:
        if backend not in ("rule-based", "llm"):
            raise ValueError(f"Unknown backend: {backend}")
        if backend == "llm" and llm_client is None:
            raise ValueError("llm backend requires an llm_client")
        self._backend = backend
        self._llm_client = llm_client

    def review(
        self,
        milestone_description: str,
        rubric: str,
        execution_trace: str,
    ) -> CriticResult:
        """
        Review an execution against a milestone rubric.

        Parameters
        ----------
        milestone_description : str
            What the milestone was supposed to accomplish.
        rubric : str
            The pass/fail criteria for this milestone.
        execution_trace : str
            The actual execution output/trace.

        Returns
        -------
        CriticResult
            Contains verdict, feedback, confidence, and reward_signal.
        """
        if self._backend == "llm":
            return self._review_with_llm(milestone_description, rubric, execution_trace)
        return self._review_rule_based(milestone_description, rubric, execution_trace)

    def _review_rule_based(
        self,
        milestone_description: str,
        rubric: str,
        execution_trace: str,
    ) -> CriticResult:
        """Rule-based review using keyword/phrase overlap scoring."""
        import re

        # Build a combined text corpus
        combined = f"{milestone_description} {rubric}".lower()
        trace_lower = execution_trace.lower()

        # Extract "important" words (length > 3, not common stopwords)
        stopwords = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
            "been", "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "should", "could", "may", "might", "can", "this", "that",
            "these", "those", "it", "its", "not", "no",
        }
        words = [w for w in re.findall(r"\b\w+\b", combined) if w not in stopwords and len(w) > 3]

        if not words:
            # Fallback: simple substring check
            overlap = sum(1 for phrase in rubric.lower().split() if phrase in trace_lower)
            score = min(overlap / max(len(rubric.split()), 1), 1.0)
        else:
            matched = sum(1 for w in words if w in trace_lower)
            score = matched / len(words)

        if score >= 0.7:
            verdict = Verdict.PASS
            reward = 1.0
            feedback = "All rubric criteria appear to be met."
        elif score >= 0.35:
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
            confidence=min(score, 1.0),
            reward_signal=reward,
        )

    def _review_with_llm(
        self,
        milestone_description: str,
        rubric: str,
        execution_trace: str,
    ) -> CriticResult:
        """LLM-as-critic using a structured prompt."""
        prompt = (
            "You are an OS-Themis-style milestone critic. Given a milestone, its rubric, "
            "and an execution trace, produce a verdict (PASS, PARTIAL, or FAIL), "
            "feedback, and a confidence score (0.0 to 1.0).\n\n"
            f"Milestone: {milestone_description}\n"
            f"Rubric: {rubric}\n"
            f"Execution Trace:\n{execution_trace}\n\n"
            "Respond in JSON with keys: verdict (one of PASS, PARTIAL, FAIL), "
            "feedback (string), confidence (float 0.0-1.0).\n"
            "Return ONLY the JSON object, no extra text."
        )
        response = self._llm_client.complete(prompt)

        # Parse JSON
        import json
        import re

        json_match = re.search(r"\{[\s\S]*\}", response)
        if not json_match:
            # Fall back to rule-based
            return self._review_rule_based(milestone_description, rubric, execution_trace)

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError:
            return self._review_rule_based(milestone_description, rubric, execution_trace)

        verdict_str = data.get("verdict", "FAIL").upper()
        try:
            verdict = Verdict[verdict_str]
        except KeyError:
            verdict = Verdict.FAIL

        confidence = float(data.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))

        reward_map = {Verdict.PASS: 1.0, Verdict.PARTIAL: 0.5, Verdict.FAIL: 0.0}
        reward = reward_map.get(verdict, 0.0)

        return CriticResult(
            verdict=verdict,
            feedback=str(data.get("feedback", "")),
            confidence=confidence,
            reward_signal=reward,
        )


# === RewardAuditor ===
class RewardAuditor:
    """
    Converts milestone verdicts and critic results into dense RL reward signals.

    From OS-Themis: converts milestone-level verdicts into per-step reward signals
    for policy gradient training.
    """

    def __init__(self, weights: dict[Verdict, float] | None = None) -> None:
        # Default: PASS=1.0, PARTIAL=0.5, FAIL=0.0
        self._weights = weights or {
            Verdict.PASS: 1.0,
            Verdict.PARTIAL: 0.5,
            Verdict.FAIL: 0.0,
        }

    def audit(self, result: CriticResult) -> float:
        """
        Convert a CriticResult into a reward signal.

        Parameters
        ----------
        result : CriticResult
            The output from MilestoneCritic.review().

        Returns
        -------
        float
            Reward signal in [0.0, 1.0].
        """
        base = self._weights.get(result.verdict, 0.0)
        # Optionally scale by confidence for more nuanced signals
        return base * result.confidence

    def audit_verdict(self, verdict: Verdict) -> float:
        """Convert a bare verdict into a reward signal."""
        return self._weights.get(verdict, 0.0)

    def batch_audit(self, results: list[CriticResult]) -> list[float]:
        """Audit a batch of critic results."""
        return [self.audit(r) for r in results]
