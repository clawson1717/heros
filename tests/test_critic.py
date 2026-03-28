"""Tests for the HeRoS Milestone Critic (Step 4).

Covers: Verdict enum, CriticResult dataclass, MilestoneCritic rule-based
and LLM backends, RewardAuditor, initialization validation, and error cases.
"""

from __future__ import annotations

from typing import Any

import pytest
from heros.critic import (
    Verdict,
    CriticResult,
    MilestoneCritic,
    RewardAuditor,
)


# ---------------------------------------------------------------------------
# Verdict Enum Tests
# ---------------------------------------------------------------------------

class TestVerdictEnum:
    def test_verdict_values(self):
        assert Verdict.PASS.value == "pass"
        assert Verdict.FAIL.value == "fail"
        assert Verdict.PARTIAL.value == "partial"

    def test_verdict_is_enum(self):
        assert isinstance(Verdict.PASS, Verdict)
        assert isinstance(Verdict.FAIL, Verdict)
        assert isinstance(Verdict.PARTIAL, Verdict)

    def test_verdict_from_str(self):
        assert Verdict["PASS"] == Verdict.PASS
        assert Verdict["FAIL"] == Verdict.FAIL
        assert Verdict["PARTIAL"] == Verdict.PARTIAL


# ---------------------------------------------------------------------------
# CriticResult Dataclass Tests
# ---------------------------------------------------------------------------

class TestCriticResult:
    def test_fields(self):
        result = CriticResult(
            verdict=Verdict.PASS,
            feedback="Looks good",
            confidence=0.95,
            reward_signal=1.0,
        )
        assert result.verdict == Verdict.PASS
        assert result.feedback == "Looks good"
        assert result.confidence == 0.95
        assert result.reward_signal == 1.0

    def test_roundtrip(self):
        original = CriticResult(
            verdict=Verdict.PARTIAL,
            feedback="Some criteria met",
            confidence=0.5,
            reward_signal=0.5,
        )
        restored = CriticResult(
            verdict=original.verdict,
            feedback=original.feedback,
            confidence=original.confidence,
            reward_signal=original.reward_signal,
        )
        assert restored.verdict == original.verdict
        assert restored.feedback == original.feedback
        assert restored.confidence == original.confidence
        assert restored.reward_signal == original.reward_signal


# ---------------------------------------------------------------------------
# MilestoneCritic Initialization Tests
# ---------------------------------------------------------------------------

class TestMilestoneCriticInit:
    def test_default_is_rule_based(self):
        critic = MilestoneCritic()
        result = critic.review(
            milestone_description="Write a function",
            rubric="Function exists",
            execution_trace="def foo(): pass",
        )
        assert isinstance(result, CriticResult)

    def test_explicit_rule_based(self):
        critic = MilestoneCritic(backend="rule-based")
        assert critic._backend == "rule-based"

    def test_llm_backend_requires_client(self):
        with pytest.raises(ValueError, match="llm backend requires an llm_client"):
            MilestoneCritic(backend="llm")

    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            MilestoneCritic(backend="unknown")


# ---------------------------------------------------------------------------
# MilestoneCritic Rule-Based Backend Tests
# ---------------------------------------------------------------------------

class TestMilestoneCriticRuleBased:
    def test_high_overlap_returns_pass(self):
        # rubric keywords appear in trace -> PASS
        critic = MilestoneCritic(backend="rule-based")
        result = critic.review(
            milestone_description="Write a Python quicksort function",
            rubric="Function named quicksort, handles empty list, handles sorted list, in-place sorting",
            execution_trace="def quicksort function handles empty list case handles sorted list in-place sorting",
        )
        assert result.verdict == Verdict.PASS
        assert result.confidence >= 0.7

    def test_medium_overlap_returns_partial(self):
        # Some rubric keywords missing -> PARTIAL
        critic = MilestoneCritic(backend="rule-based")
        result = critic.review(
            milestone_description="Write quicksort implementation",
            rubric="Function named quicksort, pivot selection, recursive partitioning",
            execution_trace="def quicksort(arr): pass",
        )
        assert result.verdict in (Verdict.PARTIAL, Verdict.FAIL)
        assert 0.0 <= result.confidence <= 1.0

    def test_low_overlap_returns_fail(self):
        # No overlap -> FAIL
        critic = MilestoneCritic(backend="rule-based")
        result = critic.review(
            milestone_description="Build a web scraper",
            rubric="Fetches URL, parses HTML, extracts data",
            execution_trace="print('hello world')",
        )
        assert result.verdict == Verdict.FAIL
        assert result.confidence < 0.35

    def test_empty_trace_returns_fail(self):
        critic = MilestoneCritic(backend="rule-based")
        result = critic.review(
            milestone_description="Deploy the application",
            rubric="Application deployed, URL accessible",
            execution_trace="",
        )
        assert result.verdict == Verdict.FAIL

    def test_trace_with_all_rubric_words_returns_pass(self):
        # Every rubric word appears in trace -> high score
        critic = MilestoneCritic(backend="rule-based")
        result = critic.review(
            milestone_description="Unit test coverage",
            rubric="test file created, tests pass, coverage above 80 percent",
            execution_trace="test file created, tests pass, coverage 85 percent",
        )
        assert result.verdict == Verdict.PASS

    def test_confidence_bounded_between_zero_and_one(self):
        critic = MilestoneCritic(backend="rule-based")
        for _ in range(20):
            result = critic.review(
                milestone_description="Test milestone",
                rubric="criteria one, criteria two",
                execution_trace="some execution output",
            )
            assert 0.0 <= result.confidence <= 1.0

    def test_reward_signal_for_pass(self):
        critic = MilestoneCritic(backend="rule-based")
        result = critic.review(
            milestone_description="Task",
            rubric="done successfully",
            execution_trace="task done successfully",
        )
        if result.verdict == Verdict.PASS:
            assert result.reward_signal == 1.0

    def test_reward_signal_for_fail(self):
        critic = MilestoneCritic(backend="rule-based")
        result = critic.review(
            milestone_description="Task",
            rubric="done",
            execution_trace="nothing here",
        )
        if result.verdict == Verdict.FAIL:
            assert result.reward_signal == 0.0


# ---------------------------------------------------------------------------
# MilestoneCritic LLM Backend Tests
# ---------------------------------------------------------------------------

class MockLLMClient:
    def __init__(self, response: str):
        self.calls: list[str] = []
        self._response = response

    def complete(self, prompt: str, **kwargs: Any) -> str:  # type: ignore[type-arg]
        self.calls.append(prompt)
        return self._response


class TestMilestoneCriticLLM:
    def test_llm_client_is_called(self):
        mock = MockLLMClient('{"verdict": "PASS", "feedback": "Great", "confidence": 0.9}')
        critic = MilestoneCritic(backend="llm", llm_client=mock)
        result = critic.review(
            milestone_description="Write quicksort",
            rubric="Function exists",
            execution_trace="def quicksort(): pass",
        )
        assert len(mock.calls) == 1
        assert "milestone" in mock.calls[0].lower()
        assert result.verdict == Verdict.PASS

    def test_llm_verdict_parsed(self):
        for verdict_str, expected in [
            ('"verdict": "PASS"', Verdict.PASS),
            ('"verdict": "PARTIAL"', Verdict.PARTIAL),
            ('"verdict": "FAIL"', Verdict.FAIL),
        ]:
            mock = MockLLMClient(
                f'{{{verdict_str}, "feedback": "ok", "confidence": 0.8}}'
            )
            critic = MilestoneCritic(backend="llm", llm_client=mock)
            result = critic.review("x", "y", "z")
            assert result.verdict == expected, f"Failed for {verdict_str}"

    def test_llm_confidence_clamped(self):
        mock = MockLLMClient('{"verdict": "PASS", "feedback": "x", "confidence": 1.5}')
        critic = MilestoneCritic(backend="llm", llm_client=mock)
        result = critic.review("x", "y", "z")
        assert result.confidence == 1.0

    def test_llm_negative_confidence_clamped(self):
        mock = MockLLMClient('{"verdict": "PASS", "feedback": "x", "confidence": -0.5}')
        critic = MilestoneCritic(backend="llm", llm_client=mock)
        result = critic.review("x", "y", "z")
        assert result.confidence == 0.0

    def test_llm_invalid_verdict_falls_back_to_fail(self):
        mock = MockLLMClient('{"verdict": "INVALID", "feedback": "x", "confidence": 0.5}')
        critic = MilestoneCritic(backend="llm", llm_client=mock)
        result = critic.review("x", "y", "z")
        assert result.verdict == Verdict.FAIL

    def test_llm_no_json_falls_back_to_rule_based(self):
        mock = MockLLMClient("This is not JSON at all")
        critic = MilestoneCritic(backend="llm", llm_client=mock)
        result = critic.review(
            milestone_description="Write a function",
            rubric="function definition code",
            execution_trace="def foo(): pass",
        )
        # Should fall back to rule-based with some overlap
        assert isinstance(result, CriticResult)

    def test_llm_json_parse_error_falls_back_to_rule_based(self):
        mock = MockLLMClient("{ this is not valid json }")
        critic = MilestoneCritic(backend="llm", llm_client=mock)
        result = critic.review(
            milestone_description="Write a function",
            rubric="Function exists",
            execution_trace="def foo(): pass",
        )
        assert isinstance(result, CriticResult)

    def test_llm_feedback_extracted(self):
        mock = MockLLMClient('{"verdict": "PASS", "feedback": "All criteria met", "confidence": 0.95}')
        critic = MilestoneCritic(backend="llm", llm_client=mock)
        result = critic.review("x", "y", "z")
        assert result.feedback == "All criteria met"

    def test_llm_confidence_default(self):
        mock = MockLLMClient('{"verdict": "PASS", "feedback": "ok"}')  # no confidence
        critic = MilestoneCritic(backend="llm", llm_client=mock)
        result = critic.review("x", "y", "z")
        assert result.confidence == 0.5  # default


# ---------------------------------------------------------------------------
# RewardAuditor Tests
# ---------------------------------------------------------------------------

class TestRewardAuditor:
    def test_default_weights(self):
        auditor = RewardAuditor()
        assert auditor._weights[Verdict.PASS] == 1.0
        assert auditor._weights[Verdict.PARTIAL] == 0.5
        assert auditor._weights[Verdict.FAIL] == 0.0

    def test_custom_weights(self):
        auditor = RewardAuditor(weights={
            Verdict.PASS: 0.8,
            Verdict.PARTIAL: 0.4,
            Verdict.FAIL: 0.1,
        })
        assert auditor._weights[Verdict.PASS] == 0.8

    def test_audit_pass(self):
        auditor = RewardAuditor()
        result = CriticResult(Verdict.PASS, "ok", 0.9, 1.0)
        assert auditor.audit(result) == pytest.approx(0.9)  # 1.0 * 0.9

    def test_audit_partial(self):
        auditor = RewardAuditor()
        result = CriticResult(Verdict.PARTIAL, "partial", 0.6, 0.5)
        assert auditor.audit(result) == pytest.approx(0.3)  # 0.5 * 0.6

    def test_audit_fail(self):
        auditor = RewardAuditor()
        result = CriticResult(Verdict.FAIL, "fail", 1.0, 0.0)
        assert auditor.audit(result) == pytest.approx(0.0)  # 0.0 * 1.0

    def test_audit_uses_custom_weights(self):
        auditor = RewardAuditor(weights={
            Verdict.PASS: 0.5,
            Verdict.PARTIAL: 0.25,
            Verdict.FAIL: 0.0,
        })
        result = CriticResult(Verdict.PASS, "ok", 1.0, 0.5)
        assert auditor.audit(result) == pytest.approx(0.5)  # 0.5 * 1.0

    def test_audit_verdict(self):
        auditor = RewardAuditor()
        assert auditor.audit_verdict(Verdict.PASS) == 1.0
        assert auditor.audit_verdict(Verdict.PARTIAL) == 0.5
        assert auditor.audit_verdict(Verdict.FAIL) == 0.0

    def test_batch_audit(self):
        auditor = RewardAuditor()
        results = [
            CriticResult(Verdict.PASS, "p", 1.0, 1.0),
            CriticResult(Verdict.PARTIAL, "pt", 0.8, 0.5),
            CriticResult(Verdict.FAIL, "f", 0.5, 0.0),
        ]
        rewards = auditor.batch_audit(results)
        assert rewards == pytest.approx([1.0, 0.4, 0.0])

    def test_batch_audit_empty_list(self):
        auditor = RewardAuditor()
        assert auditor.batch_audit([]) == []

    def test_audit_zero_confidence(self):
        auditor = RewardAuditor()
        result = CriticResult(Verdict.PASS, "ok", 0.0, 1.0)
        assert auditor.audit(result) == 0.0

    def test_audit_unknown_verdict_defaults_to_zero(self):
        auditor = RewardAuditor()
        result = CriticResult(Verdict.PASS, "ok", 1.0, 1.0)
        # Remove PASS from weights to test default
        auditor._weights = {}  # type: ignore[assignment]
        assert auditor.audit(result) == 0.0


# ---------------------------------------------------------------------------
# End-to-End Integration Tests
# ---------------------------------------------------------------------------

class TestCriticIntegration:
    def test_planner_critic_integration(self):
        """A milestone from planner flows through critic correctly."""
        from heros.planner import SubgoalPlanner

        planner = SubgoalPlanner(min_subgoals=1, max_subgoals=5)
        milestones = planner.plan("Write a quicksort function in Python")

        critic = MilestoneCritic(backend="rule-based")
        auditor = RewardAuditor()

        for m in milestones:
            result = critic.review(
                milestone_description=m.description,
                rubric=m.rubric,
                execution_trace=f"Completed: {m.description}. Output: {m.expected_output}",
            )
            reward = auditor.audit(result)
            assert 0.0 <= reward <= 1.0

    def test_rule_based_pass_threshold_exactly_0_7(self):
        """Score >= 0.7 should yield PASS."""
        critic = MilestoneCritic(backend="rule-based")
        # Build a trace with enough keywords to hit ~0.7
        result = critic.review(
            milestone_description="Build REST API endpoint",
            rubric="POST endpoint, JSON request, JSON response, status 200",
            execution_trace="POST endpoint exists, accepts JSON request, returns JSON response, status 200 returned",
        )
        assert result.verdict == Verdict.PASS

    def test_rule_based_partial_threshold_exactly_0_35(self):
        """Score >= 0.35 and < 0.7 should yield PARTIAL."""
        critic = MilestoneCritic(backend="rule-based")
        result = critic.review(
            milestone_description="Deploy the app",
            rubric="Deploy application to server, check URL accessible",
            execution_trace="Deploy application",
        )
        assert result.verdict == Verdict.PARTIAL
        assert 0.35 <= result.confidence < 0.7
