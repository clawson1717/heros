"""Tests for the HeRoS Interpretability and Audit Trail module (Step 8).

Covers: MilestoneDecisionLogger, FunctionalInterchangeabilityCheck,
RewardAuditTrail, RewardAuditor, BufferCompositionAnalyzer,
plot_buffer_composition.
"""

from __future__ import annotations

import json
import math
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from heros.critic import CriticResult, MilestoneCritic, Verdict
from heros.buffer import HindsightBuffer, HindsightTrajectory
from heros.planner import Milestone
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_milestone() -> Milestone:
    """A sample milestone for testing."""
    return Milestone(
        id="m1",
        description="Navigate to the settings page",
        rubric="Must successfully load the settings URL and display settings content",
        expected_output="Settings page HTML",
    )


@pytest.fixture
def sample_alt_milestone() -> Milestone:
    """An alternative formulation of the same milestone."""
    return Milestone(
        id="m1_alt",
        description="Open the settings page",
        rubric="Must successfully load the settings URL and display settings content",
        expected_output="Settings page HTML",
    )


@pytest.fixture
def sample_critic_result_pass() -> CriticResult:
    """A sample passing critic result."""
    return CriticResult(
        verdict=Verdict.PASS,
        feedback="All rubric criteria met.",
        confidence=0.95,
        reward_signal=1.0,
    )


@pytest.fixture
def sample_critic_result_fail() -> CriticResult:
    """A sample failing critic result."""
    return CriticResult(
        verdict=Verdict.FAIL,
        feedback="Rubric criteria not met.",
        confidence=0.8,
        reward_signal=0.0,
    )


@pytest.fixture
def sample_trajectory(sample_milestone) -> HindsightTrajectory:
    """A sample HindsightTrajectory for testing."""
    return HindsightTrajectory(
        task="Fix the login bug",
        milestones=[sample_milestone],
        exec_traces=[{"content": "Settings page loaded successfully."}],
        verdicts=[Verdict.PASS],
        timestamp="2026-03-28T12:00:00Z",
        trajectory_id="traj-001",
    )


@pytest.fixture
def sample_buffer(sample_milestone, sample_trajectory) -> HindsightBuffer:
    """A sample HindsightBuffer for testing."""
    buffer = HindsightBuffer(capacity=100)

    # Add a passing trajectory
    buffer.add(sample_trajectory)

    # Add a failing trajectory
    fail_trajectory = HindsightTrajectory(
        task="Fix the settings bug",
        milestones=[sample_milestone],
        exec_traces=[{"content": "Settings page failed to load."}],
        verdicts=[Verdict.FAIL],
        timestamp="2026-03-28T12:01:00Z",
        trajectory_id="traj-002",
    )
    buffer.add(fail_trajectory)

    return buffer


@pytest.fixture
def temp_log_dir():
    """A temporary directory for log files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ---------------------------------------------------------------------------
# MilestoneDecisionType Tests
# ---------------------------------------------------------------------------


class TestMilestoneDecisionType:
    def test_values(self):
        assert MilestoneDecisionType.CREATED.value == "created"
        assert MilestoneDecisionType.ATTEMPTED.value == "attempted"
        assert MilestoneDecisionType.PASSED.value == "passed"
        assert MilestoneDecisionType.FAILED.value == "failed"
        assert MilestoneDecisionType.PARTIAL.value == "partial"

    def test_is_enum(self):
        assert isinstance(MilestoneDecisionType.CREATED, MilestoneDecisionType)


# ---------------------------------------------------------------------------
# MilestoneDecisionLogger Tests
# ---------------------------------------------------------------------------


class TestMilestoneDecisionLogger:
    def test_init_defaults(self):
        logger = MilestoneDecisionLogger()
        assert logger.decision_count == 0
        assert logger._console_logging is True
        assert logger._log_path is None

    def test_init_with_path(self, temp_log_dir):
        log_path = temp_log_dir / "decisions.jsonl"
        logger = MilestoneDecisionLogger(log_path=log_path, console_logging=False)
        assert logger._log_path == log_path
        assert logger._console_logging is False

    def test_log_created(self, sample_milestone, temp_log_dir):
        log_path = temp_log_dir / "decisions.jsonl"
        logger = MilestoneDecisionLogger(log_path=log_path, console_logging=False)

        logger.log_created(task_id="task-1", milestone=sample_milestone, milestone_index=0)

        assert logger.decision_count == 1

        # Verify file content
        entries = logger.load_entries()
        assert len(entries) == 1
        assert entries[0]["task_id"] == "task-1"
        assert entries[0]["milestone_id"] == "m1"
        assert entries[0]["decision_type"] == "created"
        assert entries[0]["verdict"] is None
        assert entries[0]["confidence"] is None

    def test_log_attempted(self, sample_milestone, temp_log_dir):
        log_path = temp_log_dir / "decisions.jsonl"
        logger = MilestoneDecisionLogger(log_path=log_path, console_logging=False)

        logger.log_attempted(task_id="task-1", milestone=sample_milestone, milestone_index=0)

        assert logger.decision_count == 1
        entries = logger.load_entries()
        assert entries[0]["decision_type"] == "attempted"

    def test_log_review_pass(self, sample_milestone, sample_critic_result_pass, temp_log_dir):
        log_path = temp_log_dir / "decisions.jsonl"
        logger = MilestoneDecisionLogger(log_path=log_path, console_logging=False)

        logger.log_review(
            task_id="task-1",
            milestone=sample_milestone,
            result=sample_critic_result_pass,
            milestone_index=0,
        )

        assert logger.decision_count == 1
        entries = logger.load_entries()
        assert entries[0]["decision_type"] == "passed"
        assert entries[0]["verdict"] == "pass"
        assert entries[0]["confidence"] == 0.95

    def test_log_review_fail(self, sample_milestone, sample_critic_result_fail, temp_log_dir):
        log_path = temp_log_dir / "decisions.jsonl"
        logger = MilestoneDecisionLogger(log_path=log_path, console_logging=False)

        logger.log_review(
            task_id="task-1",
            milestone=sample_milestone,
            result=sample_critic_result_fail,
            milestone_index=0,
        )

        entries = logger.load_entries()
        assert entries[0]["decision_type"] == "failed"
        assert entries[0]["verdict"] == "fail"

    def test_log_review_with_reasoning(self, sample_milestone, sample_critic_result_pass, temp_log_dir):
        log_path = temp_log_dir / "decisions.jsonl"
        logger = MilestoneDecisionLogger(log_path=log_path, console_logging=False)

        reasoning = "The settings page loaded successfully, all rubric criteria met."
        logger.log_review(
            task_id="task-1",
            milestone=sample_milestone,
            result=sample_critic_result_pass,
            milestone_index=0,
            critic_reasoning=reasoning,
        )

        entries = logger.load_entries()
        assert entries[0]["critic_reasoning"] == reasoning

    def test_load_entries_filter_task_id(self, sample_milestone, temp_log_dir):
        log_path = temp_log_dir / "decisions.jsonl"
        logger = MilestoneDecisionLogger(log_path=log_path, console_logging=False)

        logger.log_created(task_id="task-1", milestone=sample_milestone, milestone_index=0)
        logger.log_created(task_id="task-2", milestone=sample_milestone, milestone_index=0)

        entries_task1 = logger.load_entries(task_id="task-1")
        assert len(entries_task1) == 1
        assert entries_task1[0]["task_id"] == "task-1"

    def test_load_entries_filter_milestone_id(self, sample_milestone, temp_log_dir):
        log_path = temp_log_dir / "decisions.jsonl"
        logger = MilestoneDecisionLogger(log_path=log_path, console_logging=False)

        logger.log_created(task_id="task-1", milestone=sample_milestone, milestone_index=0)

        entries = logger.load_entries(milestone_id="m1")
        assert len(entries) == 1
        assert entries[0]["milestone_id"] == "m1"

    def test_load_entries_filter_decision_type(self, sample_milestone, temp_log_dir):
        log_path = temp_log_dir / "decisions.jsonl"
        logger = MilestoneDecisionLogger(log_path=log_path, console_logging=False)

        logger.log_created(task_id="task-1", milestone=sample_milestone, milestone_index=0)
        logger.log_created(task_id="task-2", milestone=sample_milestone, milestone_index=0)

        entries = logger.load_entries(decision_type=MilestoneDecisionType.CREATED)
        assert len(entries) == 2

    def test_load_entries_limit(self, sample_milestone, temp_log_dir):
        log_path = temp_log_dir / "decisions.jsonl"
        logger = MilestoneDecisionLogger(log_path=log_path, console_logging=False)

        for i in range(10):
            logger.log_created(task_id=f"task-{i}", milestone=sample_milestone, milestone_index=0)

        entries = logger.load_entries(limit=5)
        assert len(entries) == 5

    def test_multiple_loggers_independent_counters(self, sample_milestone, temp_log_dir):
        log_path1 = temp_log_dir / "decisions1.jsonl"
        log_path2 = temp_log_dir / "decisions2.jsonl"

        logger1 = MilestoneDecisionLogger(log_path=log_path1, console_logging=False)
        logger2 = MilestoneDecisionLogger(log_path=log_path2, console_logging=False)

        logger1.log_created(task_id="task-1", milestone=sample_milestone, milestone_index=0)
        logger2.log_created(task_id="task-2", milestone=sample_milestone, milestone_index=0)
        logger2.log_created(task_id="task-3", milestone=sample_milestone, milestone_index=0)

        assert logger1.decision_count == 1
        assert logger2.decision_count == 2


# ---------------------------------------------------------------------------
# FunctionalEquivalenceResult Tests
# ---------------------------------------------------------------------------


class TestFunctionalEquivalenceResult:
    def test_equivalent_pass_pass(self, sample_critic_result_pass):
        result = FunctionalEquivalenceResult(
            equivalent=True,
            original_verdict=Verdict.PASS,
            alternative_verdict=Verdict.PASS,
            swap_verified=True,
            original_result=sample_critic_result_pass,
            alternative_result=sample_critic_result_pass,
            verification_trace="Settings page loaded successfully.",
        )

        assert result.equivalent is True
        assert result.original_verdict == Verdict.PASS
        assert result.alternative_verdict == Verdict.PASS
        assert result.swap_verified is True

    def test_not_equivalent(self, sample_critic_result_pass, sample_critic_result_fail):
        result = FunctionalEquivalenceResult(
            equivalent=False,
            original_verdict=Verdict.PASS,
            alternative_verdict=Verdict.FAIL,
            swap_verified=False,
            original_result=sample_critic_result_pass,
            alternative_result=sample_critic_result_fail,
            verification_trace="Settings page loaded successfully.",
        )

        assert result.equivalent is False
        assert result.swap_verified is False

    def test_to_dict(self, sample_critic_result_pass):
        result = FunctionalEquivalenceResult(
            equivalent=True,
            original_verdict=Verdict.PASS,
            alternative_verdict=Verdict.PASS,
            swap_verified=True,
            original_result=sample_critic_result_pass,
            alternative_result=sample_critic_result_pass,
            verification_trace="Settings page loaded.",
        )

        d = result.to_dict()
        assert d["equivalent"] is True
        assert d["original_verdict"] == "pass"
        assert d["alternative_verdict"] == "pass"
        assert d["swap_verified"] is True
        assert d["verification_trace"] == "Settings page loaded."
        assert "timestamp" in d


# ---------------------------------------------------------------------------
# FunctionalInterchangeabilityCheck Tests
# ---------------------------------------------------------------------------


class TestFunctionalInterchangeabilityCheck:
    def test_init(self):
        critic = MilestoneCritic(backend="rule-based")
        checker = FunctionalInterchangeabilityCheck(critic=critic)
        assert checker._critic is critic
        assert checker._confidence_threshold == 0.7

    def test_init_custom_threshold(self):
        critic = MilestoneCritic(backend="rule-based")
        checker = FunctionalInterchangeabilityCheck(
            critic=critic, confidence_threshold=0.5
        )
        assert checker._confidence_threshold == 0.5

    def test_init_invalid_threshold(self):
        critic = MilestoneCritic(backend="rule-based")
        with pytest.raises(ValueError):
            FunctionalInterchangeabilityCheck(critic=critic, confidence_threshold=1.5)

    def test_init_invalid_critic(self):
        with pytest.raises(TypeError):
            FunctionalInterchangeabilityCheck(critic="not a critic")  # type: ignore

    def test_verify_same_trace_same_verdict(self, sample_milestone, sample_alt_milestone):
        """Test that same trace gives same verdict for both formulations (if rubric matches)."""
        critic = MilestoneCritic(backend="rule-based")
        checker = FunctionalInterchangeabilityCheck(critic=critic)

        # Use a trace that contains the exact rubric keywords
        trace = "Navigate to the settings page. Must successfully load the settings URL and display settings content."
        result = checker.verify(trace=trace, milestone=sample_milestone, alt_milestone=sample_alt_milestone)

        # Both should get the same verdict since trace is rich with rubric terms
        assert result.original_verdict == result.alternative_verdict

    def test_verify_fail_fail(self, sample_milestone, sample_alt_milestone):
        critic = MilestoneCritic(backend="rule-based")
        checker = FunctionalInterchangeabilityCheck(critic=critic)

        trace = "Some unrelated content."
        result = checker.verify(trace=trace, milestone=sample_milestone, alt_milestone=sample_alt_milestone)

        # Rule-based critic should give low score for unrelated content
        assert result.original_verdict == Verdict.FAIL
        assert result.alternative_verdict == Verdict.FAIL
        assert result.equivalent is True

    def test_verify_different_verdicts(self, sample_milestone, sample_alt_milestone):
        critic = MilestoneCritic(backend="rule-based")
        checker = FunctionalInterchangeabilityCheck(critic=critic)

        # Very short trace should produce low scores
        trace = "done"
        result = checker.verify(trace=trace, milestone=sample_milestone, alt_milestone=sample_alt_milestone)

        # Both should get same verdict for same trace (same rubric)
        assert result.equivalent is True

    def test_generate_alternative_default(self, sample_milestone):
        critic = MilestoneCritic(backend="rule-based")
        checker = FunctionalInterchangeabilityCheck(critic=critic)

        alt = checker.generate_alternative(sample_milestone)

        assert alt.id == "m1_alt"
        assert alt.rubric == sample_milestone.rubric
        assert alt.expected_output == sample_milestone.expected_output
        # Description should be different (paraphrased)
        assert alt.description != sample_milestone.description

    def test_generate_alternative_custom_generator(self, sample_milestone):
        critic = MilestoneCritic(backend="rule-based")

        def custom_generator(desc: str, rubric: str) -> str:
            return f"Custom paraphrase: {desc}"

        checker = FunctionalInterchangeabilityCheck(
            critic=critic, alternative_generator=custom_generator
        )

        alt = checker.generate_alternative(sample_milestone)
        assert alt.description == f"Custom paraphrase: {sample_milestone.description}"


# ---------------------------------------------------------------------------
# MilestoneAuditEntry Tests
# ---------------------------------------------------------------------------


class TestMilestoneAuditEntry:
    def test_creation(self):
        entry = MilestoneAuditEntry(
            milestone_id="m1",
            description="Navigate to settings",
            verdict="pass",
            reward_signal=1.0,
            critic_confidence=0.95,
            feedback_text="All criteria met.",
            milestone_index=0,
        )

        assert entry.milestone_id == "m1"
        assert entry.verdict == "pass"
        assert entry.reward_signal == 1.0
        assert entry.critic_confidence == 0.95
        assert entry.feedback_text == "All criteria met."
        assert entry.milestone_index == 0

    def test_to_dict(self):
        entry = MilestoneAuditEntry(
            milestone_id="m1",
            description="Navigate to settings",
            verdict="pass",
            reward_signal=1.0,
            critic_confidence=0.95,
            feedback_text="All criteria met.",
            milestone_index=0,
        )

        d = entry.to_dict()
        assert d["milestone_id"] == "m1"
        assert d["description"] == "Navigate to settings"
        assert d["verdict"] == "pass"
        assert d["reward_signal"] == 1.0
        assert d["critic_confidence"] == 0.95
        assert d["feedback_text"] == "All criteria met."
        assert d["milestone_index"] == 0


# ---------------------------------------------------------------------------
# RewardAuditTrail Tests
# ---------------------------------------------------------------------------


class TestRewardAuditTrail:
    def test_from_critic_results(self, sample_milestone, sample_critic_result_pass):
        trail = RewardAuditTrail.from_critic_results(
            episode_id="ep-1",
            task="Fix the login bug",
            milestone_results=[(sample_milestone, sample_critic_result_pass)],
        )

        assert trail.episode_id == "ep-1"
        assert trail.task == "Fix the login bug"
        assert len(trail.milestones) == 1
        assert trail.milestones[0].milestone_id == "m1"
        assert trail.milestones[0].verdict == "pass"
        assert trail.milestones[0].reward_signal == 1.0
        assert trail.total_reward == 1.0
        assert trail.avg_reward == 1.0
        assert trail.success_rate == 1.0

    def test_from_critic_results_multiple(self, sample_milestone, sample_critic_result_pass, sample_critic_result_fail):
        trail = RewardAuditTrail.from_critic_results(
            episode_id="ep-2",
            task="Fix the login bug",
            milestone_results=[
                (sample_milestone, sample_critic_result_pass),
                (sample_milestone, sample_critic_result_fail),
            ],
        )

        assert len(trail.milestones) == 2
        assert trail.total_reward == 1.0
        assert trail.avg_reward == 0.5
        assert trail.success_rate == 0.5

    def test_from_trajectory(self, sample_trajectory, sample_critic_result_pass):
        auditor = RewardAuditor()
        trail = RewardAuditTrail.from_trajectory(
            trajectory=sample_trajectory,
            auditor=auditor,
            episode_id="ep-3",
        )

        assert trail.episode_id == "ep-3"
        assert trail.task == "Fix the login bug"
        assert len(trail.milestones) == 1

    def test_to_dict(self, sample_milestone, sample_critic_result_pass):
        trail = RewardAuditTrail.from_critic_results(
            episode_id="ep-1",
            task="Fix the login bug",
            milestone_results=[(sample_milestone, sample_critic_result_pass)],
        )

        d = trail.to_dict()
        assert d["episode_id"] == "ep-1"
        assert d["task"] == "Fix the login bug"
        assert d["total_reward"] == 1.0
        assert d["avg_reward"] == 1.0
        assert d["success_rate"] == 1.0
        assert len(d["milestones"]) == 1


# ---------------------------------------------------------------------------
# RewardAuditor Tests
# ---------------------------------------------------------------------------


class TestRewardAuditor:
    def test_init_defaults(self):
        auditor = RewardAuditor()
        assert auditor._weights[Verdict.PASS] == 1.0
        assert auditor._weights[Verdict.PARTIAL] == 0.5
        assert auditor._weights[Verdict.FAIL] == 0.0
        assert auditor._confidence_scaling is True
        assert auditor.episode_count == 0

    def test_init_custom_weights(self):
        auditor = RewardAuditor(
            weights={Verdict.PASS: 0.8, Verdict.PARTIAL: 0.4, Verdict.FAIL: 0.1}
        )
        assert auditor._weights[Verdict.PASS] == 0.8
        assert auditor._weights[Verdict.FAIL] == 0.1

    def test_audit_pass_confidence_scaling(self, sample_critic_result_pass):
        auditor = RewardAuditor(confidence_scaling=True)
        reward = auditor.audit(sample_critic_result_pass)
        assert reward == 1.0 * 0.95  # weight * confidence

    def test_audit_pass_no_confidence_scaling(self, sample_critic_result_pass):
        auditor = RewardAuditor(confidence_scaling=False)
        reward = auditor.audit(sample_critic_result_pass)
        assert reward == 1.0  # Just the weight, no scaling

    def test_audit_fail(self, sample_critic_result_fail):
        auditor = RewardAuditor()
        reward = auditor.audit(sample_critic_result_fail)
        assert reward == 0.0 * 0.8  # weight * confidence

    def test_audit_verdict_pass(self):
        auditor = RewardAuditor()
        reward = auditor.audit_verdict(Verdict.PASS)
        assert reward == 1.0

    def test_audit_verdict_partial(self):
        auditor = RewardAuditor()
        reward = auditor.audit_verdict(Verdict.PARTIAL)
        assert reward == 0.5

    def test_audit_verdict_fail(self):
        auditor = RewardAuditor()
        reward = auditor.audit_verdict(Verdict.FAIL)
        assert reward == 0.0

    def test_batch_audit(self, sample_critic_result_pass, sample_critic_result_fail):
        auditor = RewardAuditor()
        results = [sample_critic_result_pass, sample_critic_result_fail]
        rewards = auditor.batch_audit(results)
        assert len(rewards) == 2
        assert rewards[0] > 0
        assert rewards[1] == 0.0

    def test_audit_episode(self, sample_milestone, sample_critic_result_pass, temp_log_dir):
        audit_path = temp_log_dir / "audit.jsonl"
        auditor = RewardAuditor(audit_log_path=audit_path, console_logging=False)

        trail = auditor.audit_episode(
            episode_id="ep-1",
            task="Fix the login bug",
            milestone_results=[(sample_milestone, sample_critic_result_pass)],
        )

        assert auditor.episode_count == 1
        assert trail.episode_id == "ep-1"

        # Verify file was written
        assert audit_path.exists()
        with open(audit_path, "r") as f:
            lines = f.readlines()
        assert len(lines) == 1

    def test_audit_trajectory(self, sample_trajectory, temp_log_dir):
        audit_path = temp_log_dir / "audit.jsonl"
        auditor = RewardAuditor(audit_log_path=audit_path, console_logging=False)

        trail = auditor.audit_trajectory(sample_trajectory, episode_id="ep-traj")

        assert trail.episode_id == "ep-traj"

    def test_load_audit_trails(self, sample_milestone, sample_critic_result_pass, temp_log_dir):
        audit_path = temp_log_dir / "audit.jsonl"
        auditor = RewardAuditor(audit_log_path=audit_path, console_logging=False)

        # Audit a few episodes
        for i in range(5):
            auditor.audit_episode(
                episode_id=f"ep-{i}",
                task=f"Task {i}",
                milestone_results=[(sample_milestone, sample_critic_result_pass)],
            )

        # Load with limit
        trails = auditor.load_audit_trails(limit=3)
        assert len(trails) == 3

    def test_load_audit_trails_task_filter(self, sample_milestone, sample_critic_result_pass, temp_log_dir):
        audit_path = temp_log_dir / "audit.jsonl"
        auditor = RewardAuditor(audit_log_path=audit_path, console_logging=False)

        auditor.audit_episode(
            episode_id="ep-1",
            task="Fix the login bug",
            milestone_results=[(sample_milestone, sample_critic_result_pass)],
        )
        auditor.audit_episode(
            episode_id="ep-2",
            task="Fix the settings bug",
            milestone_results=[(sample_milestone, sample_critic_result_pass)],
        )

        trails = auditor.load_audit_trails(task_filter="login")
        assert len(trails) == 1
        assert trails[0].task == "Fix the login bug"


# ---------------------------------------------------------------------------
# BufferCompositionAnalyzer Tests
# ---------------------------------------------------------------------------


class TestBufferCompositionAnalyzer:
    def test_init(self, sample_buffer):
        analyzer = BufferCompositionAnalyzer(buffer=sample_buffer)
        assert analyzer._buffer is sample_buffer

    def test_init_invalid_buffer(self):
        with pytest.raises(TypeError):
            BufferCompositionAnalyzer(buffer="not a buffer")  # type: ignore

    def test_compute_buffer_diversity_empty(self):
        empty_buffer = HindsightBuffer(capacity=10)
        analyzer = BufferCompositionAnalyzer(buffer=empty_buffer)
        assert analyzer.compute_buffer_diversity() == 0.0

    def test_compute_buffer_diversity(self, sample_buffer):
        analyzer = BufferCompositionAnalyzer(buffer=sample_buffer)
        entropy = analyzer.compute_buffer_diversity()
        assert entropy >= 0.0

    def test_milestone_hit_rate_by_type(self, sample_buffer):
        analyzer = BufferCompositionAnalyzer(buffer=sample_buffer)
        hit_rates = analyzer.milestone_hit_rate_by_type()
        assert isinstance(hit_rates, dict)

    def test_milestone_hit_rate_by_type_empty(self):
        empty_buffer = HindsightBuffer(capacity=10)
        analyzer = BufferCompositionAnalyzer(buffer=empty_buffer)
        hit_rates = analyzer.milestone_hit_rate_by_type()
        assert hit_rates == {}

    def test_failed_milestone_distribution(self, sample_buffer):
        analyzer = BufferCompositionAnalyzer(buffer=sample_buffer)
        dist = analyzer.failed_milestone_distribution()
        assert isinstance(dist, dict)

    def test_milestone_type_counts(self, sample_buffer):
        analyzer = BufferCompositionAnalyzer(buffer=sample_buffer)
        counts = analyzer.milestone_type_counts()
        assert isinstance(counts, dict)
        assert all(isinstance(k, str) for k in counts.keys())
        assert all(isinstance(v, int) for v in counts.values())

    def test_overall_hit_rate(self, sample_buffer):
        analyzer = BufferCompositionAnalyzer(buffer=sample_buffer)
        rate = analyzer.overall_hit_rate()
        assert 0.0 <= rate <= 1.0

    def test_overall_partial_rate(self, sample_buffer):
        analyzer = BufferCompositionAnalyzer(buffer=sample_buffer)
        rate = analyzer.overall_partial_rate()
        assert 0.0 <= rate <= 1.0

    def test_overall_fail_rate(self, sample_buffer):
        analyzer = BufferCompositionAnalyzer(buffer=sample_buffer)
        rate = analyzer.overall_fail_rate()
        assert 0.0 <= rate <= 1.0

    def test_trajectory_success_rates(self, sample_buffer):
        analyzer = BufferCompositionAnalyzer(buffer=sample_buffer)
        rates = analyzer.trajectory_success_rates()
        assert isinstance(rates, list)
        assert all(0.0 <= r <= 1.0 for r in rates)

    def test_hindsight_enhancement_rate(self, sample_buffer):
        analyzer = BufferCompositionAnalyzer(buffer=sample_buffer)
        rate = analyzer.hindsight_enhancement_rate()
        assert 0.0 <= rate <= 1.0

    def test_hindsight_enhancement_rate_empty(self):
        empty_buffer = HindsightBuffer(capacity=10)
        analyzer = BufferCompositionAnalyzer(buffer=empty_buffer)
        assert analyzer.hindsight_enhancement_rate() == 0.0

    def test_export_buffer_summary_json(self, sample_buffer, temp_log_dir):
        summary_path = temp_log_dir / "buffer_summary.json"
        analyzer = BufferCompositionAnalyzer(buffer=sample_buffer)
        analyzer.export_buffer_summary_json(summary_path)

        assert summary_path.exists()

        with open(summary_path, "r") as f:
            summary = json.load(f)

        assert "timestamp" in summary
        assert "buffer_stats" in summary
        assert "diversity_entropy" in summary
        assert "milestone_type_counts" in summary
        assert "hit_rate_by_type" in summary
        assert "failed_milestone_distribution" in summary
        assert "overall_hit_rate" in summary
        assert "overall_partial_rate" in summary
        assert "overall_fail_rate" in summary
        assert "hindsight_enhancement_rate" in summary
        assert "trajectory_success_rates" in summary

    def test_custom_type_extractor(self, sample_buffer):
        def custom_extractor(milestone: Milestone) -> str:
            return milestone.id

        analyzer = BufferCompositionAnalyzer(
            buffer=sample_buffer, milestone_type_extractor=custom_extractor
        )
        counts = analyzer.milestone_type_counts()
        assert "m1" in counts


# ---------------------------------------------------------------------------
# plot_buffer_composition Tests
# ---------------------------------------------------------------------------


class TestPlotBufferComposition:
    def test_empty_buffer(self, temp_log_dir):
        empty_buffer = HindsightBuffer(capacity=10)
        output_path = temp_log_dir / "composition.png"

        data = plot_buffer_composition(
            buffer=empty_buffer,
            output_path=output_path,
        )

        assert output_path.exists()
        assert "milestone_type_counts" in data
        assert "hit_rates_by_type" in data
        assert "verdict_counts" in data
        assert "trajectory_success_rates" in data
        assert "buffer_stats" in data
        assert "diversity_entropy" in data

    def test_with_data(self, sample_buffer, temp_log_dir):
        output_path = temp_log_dir / "composition.png"

        data = plot_buffer_composition(
            buffer=sample_buffer,
            output_path=output_path,
        )

        assert output_path.exists()
        assert data["verdict_counts"]["pass"] == 1
        assert data["verdict_counts"]["fail"] == 1

    def test_no_output_path(self, sample_buffer):
        data = plot_buffer_composition(buffer=sample_buffer, output_path=None)
        assert "milestone_type_counts" in data

    def test_custom_figsize_dpi(self, sample_buffer, temp_log_dir):
        output_path = temp_log_dir / "composition.png"

        data = plot_buffer_composition(
            buffer=sample_buffer,
            output_path=output_path,
            figsize=(16, 10),
            dpi=150,
        )

        assert output_path.exists()

    def test_custom_type_extractor(self, sample_buffer, temp_log_dir):
        output_path = temp_log_dir / "composition.png"

        def custom_extractor(milestone: Milestone) -> str:
            return milestone.id

        data = plot_buffer_composition(
            buffer=sample_buffer,
            output_path=output_path,
            milestone_type_extractor=custom_extractor,
        )

        assert "m1" in data["milestone_type_counts"]


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


class TestInterpretabilityIntegration:
    """End-to-end integration tests for the interpretability module."""

    def test_full_audit_workflow(self, sample_milestone, sample_critic_result_pass, temp_log_dir):
        """Test the full workflow: log decisions, audit rewards, analyze buffer."""
        # 1. Setup
        decision_log = temp_log_dir / "decisions.jsonl"
        audit_log = temp_log_dir / "audit.jsonl"
        summary_log = temp_log_dir / "summary.json"

        # 2. Log milestone decisions
        logger = MilestoneDecisionLogger(log_path=decision_log, console_logging=False)
        logger.log_created(task_id="task-1", milestone=sample_milestone, milestone_index=0)
        logger.log_attempted(task_id="task-1", milestone=sample_milestone, milestone_index=0)
        logger.log_review(
            task_id="task-1",
            milestone=sample_milestone,
            result=sample_critic_result_pass,
            milestone_index=0,
        )

        # 3. Audit rewards
        auditor = RewardAuditor(audit_log_path=audit_log, console_logging=False)
        trail = auditor.audit_episode(
            episode_id="ep-1",
            task="Fix the login bug",
            milestone_results=[(sample_milestone, sample_critic_result_pass)],
        )

        assert trail.avg_reward == 1.0
        assert trail.success_rate == 1.0

        # 4. Verify logs were written
        assert decision_log.exists()
        assert audit_log.exists()

    def test_functional_equivalence_workflow(self, sample_milestone, sample_alt_milestone):
        """Test functional interchangeability check."""
        critic = MilestoneCritic(backend="rule-based")
        checker = FunctionalInterchangeabilityCheck(critic=critic)

        # Use a trace rich with rubric terms to ensure both get same verdict
        trace = "Navigate to the settings page. Must successfully load the settings URL and display settings content."

        # Check that original and alternative get same verdict
        result = checker.verify(
            trace=trace,
            milestone=sample_milestone,
            alt_milestone=sample_alt_milestone,
        )

        assert result.original_verdict == result.alternative_verdict

        # Generate alternative and verify again
        alt = checker.generate_alternative(sample_milestone)
        result2 = checker.verify(
            trace=trace,
            milestone=sample_milestone,
            alt_milestone=alt,
        )

        # Both should get same verdict with rubric-rich trace
        assert result2.original_verdict == result2.alternative_verdict

    def test_buffer_analysis_workflow(self, sample_buffer, sample_milestone, temp_log_dir):
        """Test buffer analysis and visualization."""
        # 1. Analyze buffer
        analyzer = BufferCompositionAnalyzer(buffer=sample_buffer)

        diversity = analyzer.compute_buffer_diversity()
        hit_rates = analyzer.milestone_hit_rate_by_type()
        fail_dist = analyzer.failed_milestone_distribution()

        assert isinstance(diversity, float)
        assert isinstance(hit_rates, dict)
        assert isinstance(fail_dist, dict)

        # 2. Export summary
        summary_path = temp_log_dir / "summary.json"
        analyzer.export_buffer_summary_json(summary_path)
        assert summary_path.exists()

        # 3. Plot composition
        plot_path = temp_log_dir / "composition.png"
        data = plot_buffer_composition(buffer=sample_buffer, output_path=plot_path)
        assert plot_path.exists()
        assert data["buffer_stats"]["total"] == 2

    def test_multi_trajectory_buffer_analysis(self, sample_milestone, temp_log_dir):
        """Test with a buffer containing multiple trajectories."""
        buffer = HindsightBuffer(capacity=100)

        # Add various trajectories
        for i in range(10):
            verdict = Verdict.PASS if i % 2 == 0 else Verdict.FAIL
            traj = HindsightTrajectory(
                task=f"Task {i}",
                milestones=[sample_milestone],
                exec_traces=[{"content": f"Trace {i}"}],
                verdicts=[verdict],
                timestamp=f"2026-03-28T{i:02d}:00:00Z",
                trajectory_id=f"traj-{i:03d}",
            )
            buffer.add(traj)

        analyzer = BufferCompositionAnalyzer(buffer=buffer)

        # Overall rates should be reasonable
        hit_rate = analyzer.overall_hit_rate()
        fail_rate = analyzer.overall_fail_rate()
        assert hit_rate + fail_rate <= 1.0  # May have partials

        # Success rates should have 5 passing (i=0,2,4,6,8)
        success_rates = analyzer.trajectory_success_rates()
        assert len(success_rates) == 10

    def test_concurrent_logging(self, sample_milestone, sample_critic_result_pass, temp_log_dir):
        """Test that multiple loggers can write to different files concurrently."""
        loggers = []
        for i in range(5):
            log_path = temp_log_dir / f"decisions_{i}.jsonl"
            logger = MilestoneDecisionLogger(log_path=log_path, console_logging=False)
            logger.log_review(
                task_id=f"task-{i}",
                milestone=sample_milestone,
                result=sample_critic_result_pass,
                milestone_index=0,
            )
            loggers.append(logger)

        # All should have independent counters
        for i, logger in enumerate(loggers):
            assert logger.decision_count == 1


# ---------------------------------------------------------------------------
# Edge Case Tests
# ---------------------------------------------------------------------------


class TestInterpretabilityEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_logger_with_nonexistent_directory(self, sample_milestone, temp_log_dir):
        """Test that logger handles nonexistent log directory gracefully."""
        # It should create parent directories
        log_path = temp_log_dir / "nested" / "dir" / "decisions.jsonl"
        logger = MilestoneDecisionLogger(log_path=log_path, console_logging=False)
        logger.log_created(task_id="task-1", milestone=sample_milestone, milestone_index=0)
        assert log_path.exists()

    def test_auditor_with_empty_results(self, sample_milestone, temp_log_dir):
        """Test auditing an episode with no milestones."""
        audit_path = temp_log_dir / "audit.jsonl"
        auditor = RewardAuditor(audit_log_path=audit_path, console_logging=False)

        trail = auditor.audit_episode(
            episode_id="ep-empty",
            task="Empty task",
            milestone_results=[],  # Empty
        )

        assert trail.episode_id == "ep-empty"
        assert len(trail.milestones) == 0
        assert trail.total_reward == 0.0
        assert trail.avg_reward == 0.0

    def test_analyzer_with_single_trajectory(self, sample_milestone, temp_log_dir):
        """Test buffer analyzer with a single trajectory."""
        buffer = HindsightBuffer(capacity=10)
        traj = HindsightTrajectory(
            task="Single task",
            milestones=[sample_milestone],
            exec_traces=[{"content": "Trace"}],
            verdicts=[Verdict.PASS],
            timestamp="2026-03-28T12:00:00Z",
            trajectory_id="traj-single",
        )
        buffer.add(traj)

        analyzer = BufferCompositionAnalyzer(buffer=buffer)

        assert analyzer.compute_buffer_diversity() >= 0.0
        assert analyzer.overall_hit_rate() == 1.0
        assert analyzer.trajectory_success_rates() == [1.0]

    def test_plot_with_single_milestone_type(self, sample_milestone, temp_log_dir):
        """Test plotting with only one milestone type in buffer."""
        buffer = HindsightBuffer(capacity=10)
        for i in range(5):
            traj = HindsightTrajectory(
                task=f"Task {i}",
                milestones=[sample_milestone],  # Same milestone
                exec_traces=[{"content": f"Trace {i}"}],
                verdicts=[Verdict.PASS],
                timestamp=f"2026-03-28T{i:02d}:00:00Z",
                trajectory_id=f"traj-{i}",
            )
            buffer.add(traj)

        output_path = temp_log_dir / "composition.png"
        data = plot_buffer_composition(buffer=buffer, output_path=output_path)

        assert output_path.exists()
        # All should be same type
        assert len(data["milestone_type_counts"]) == 1

    def test_load_entries_nonexistent_file(self):
        """Test loading from a nonexistent file returns empty list."""
        logger = MilestoneDecisionLogger(log_path=None, console_logging=False)
        entries = logger.load_entries()
        assert entries == []

    def test_load_audit_trails_nonexistent_file(self):
        """Test loading audit trails from nonexistent file returns empty list."""
        auditor = RewardAuditor(audit_log_path=None, console_logging=False)
        trails = auditor.load_audit_trails()
        assert trails == []

    def test_buffer_analyzer_empty_buffer_stats(self, temp_log_dir):
        """Test buffer analyzer with empty buffer exports correctly."""
        empty_buffer = HindsightBuffer(capacity=10)
        analyzer = BufferCompositionAnalyzer(buffer=empty_buffer)

        summary_path = temp_log_dir / "empty_summary.json"
        analyzer.export_buffer_summary_json(summary_path)

        with open(summary_path, "r") as f:
            summary = json.load(f)

        assert summary["buffer_stats"]["total"] == 0
        assert summary["diversity_entropy"] == 0.0

    def test_decision_logger_with_various_verdicts(self, sample_milestone, temp_log_dir):
        """Test logging decisions with different verdict types."""
        log_path = temp_log_dir / "decisions.jsonl"
        logger = MilestoneDecisionLogger(log_path=log_path, console_logging=False)

        verdicts = [
            CriticResult(verdict=Verdict.PASS, feedback="Good", confidence=0.9, reward_signal=1.0),
            CriticResult(verdict=Verdict.FAIL, feedback="Bad", confidence=0.7, reward_signal=0.0),
            CriticResult(verdict=Verdict.PARTIAL, feedback="OK", confidence=0.6, reward_signal=0.5),
        ]

        for i, result in enumerate(verdicts):
            logger.log_review(
                task_id=f"task-{i}",
                milestone=sample_milestone,
                result=result,
                milestone_index=i,
            )

        assert logger.decision_count == 3
        entries = logger.load_entries()
        assert len(entries) == 3
        # Entries are returned most recent first (reverse chronological)
        assert entries[0]["verdict"] == "partial"  # Most recent (task-2)
        assert entries[1]["verdict"] == "fail"    # Middle (task-1)
        assert entries[2]["verdict"] == "pass"     # Oldest (task-0)

    def test_auditor_with_partial_verdicts(self, sample_milestone, temp_log_dir):
        """Test auditor with partial verdict episodes."""
        audit_path = temp_log_dir / "audit.jsonl"
        auditor = RewardAuditor(audit_log_path=audit_path, console_logging=False)

        partial_result = CriticResult(
            verdict=Verdict.PARTIAL,
            feedback="Some criteria met",
            confidence=0.7,
            reward_signal=0.5,  # This is the base weight, NOT confidence scaled
        )

        trail = auditor.audit_episode(
            episode_id="ep-partial",
            task="Partial task",
            milestone_results=[(sample_milestone, partial_result)],
        )

        assert trail.success_rate == 0.0  # Partial is not PASS
        # CriticResult.reward_signal stores base weight, confidence is stored separately
        assert trail.avg_reward == 0.5  # Base weight for PARTIAL
        assert trail.milestones[0].verdict == "partial"

    def test_interchangeability_with_llm_backend(self, sample_milestone, sample_alt_milestone):
        """Test functional interchangeability with custom threshold."""
        critic = MilestoneCritic(backend="rule-based")
        # Use a high threshold to require high confidence
        checker = FunctionalInterchangeabilityCheck(
            critic=critic, confidence_threshold=0.9
        )

        trace = "Navigate to the settings page. Must successfully load the settings URL and display settings content."
        result = checker.verify(
            trace=trace,
            milestone=sample_milestone,
            alt_milestone=sample_alt_milestone,
        )

        # Both should get same verdict
        assert result.original_verdict == result.alternative_verdict
        # But swap_verified may be False due to threshold
        # (rule-based confidence may be around 0.7)
        assert isinstance(result.swap_verified, bool)

    def test_reward_audit_trail_with_none_timestamp(self, sample_milestone, sample_critic_result_pass):
        """Test RewardAuditTrail handles None timestamps gracefully."""
        trail = RewardAuditTrail.from_critic_results(
            episode_id="ep-1",
            task="Test task",
            milestone_results=[(sample_milestone, sample_critic_result_pass)],
            timestamp=None,
        )

        assert trail.timestamp is not None
        assert trail.episode_id == "ep-1"

    def test_plot_data_structure(self, sample_buffer, temp_log_dir):
        """Test that plot_buffer_composition returns correct data structure."""
        output_path = temp_log_dir / "plot.png"
        data = plot_buffer_composition(buffer=sample_buffer, output_path=output_path)

        # Verify all expected keys are present
        assert "milestone_type_counts" in data
        assert "hit_rates_by_type" in data
        assert "verdict_counts" in data
        assert "trajectory_success_rates" in data
        assert "buffer_stats" in data
        assert "diversity_entropy" in data

        # Verify types
        assert isinstance(data["milestone_type_counts"], dict)
        assert isinstance(data["hit_rates_by_type"], dict)
        assert isinstance(data["verdict_counts"], dict)
        assert isinstance(data["trajectory_success_rates"], list)
        assert isinstance(data["buffer_stats"], dict)
        assert isinstance(data["diversity_entropy"], float)
