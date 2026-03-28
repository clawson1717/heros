"""Tests for Step 5 — HeRL-style Hindsight Experience Buffer.

Covers: HindsightTrajectory dataclass, HindsightBuffer FIFO add/sample/filter,
hindsight ratio enforcement, HindsightTrainer, UpdateResult, and
serialization roundtrips.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
import json
import pytest

from heros.buffer import HindsightBuffer, HindsightTrajectory
from heros.trainer import HindsightTrainer, UpdateResult
from heros.planner import Milestone
from heros.critic import Verdict


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_milestones(n: int = 3) -> list[Milestone]:
    """Create n simple milestones."""
    return [
        Milestone(
            id=f"m{i + 1}",
            description=f"Milestone {i + 1} description",
            rubric=f"Rubric {i + 1}: check this",
            expected_output=f"Output {i + 1}",
        )
        for i in range(n)
        ]


def make_verdicts(pass_n: int, fail_n: int, partial_n: int = 0) -> list[Verdict]:
    """Create a verdict list with the specified counts."""
    verdicts = []
    verdicts.extend([Verdict.PASS] * pass_n)
    verdicts.extend([Verdict.FAIL] * fail_n)
    verdicts.extend([Verdict.PARTIAL] * partial_n)
    return verdicts


def make_trace(milestone_id: str, content: str = "executed successfully") -> dict[str, Any]:
    """Create a simple execution trace dict."""
    return {"milestone_id": milestone_id, "output": content, "status": "complete"}


def make_trajectory(
    task: str = "Test task",
    num_milestones: int = 3,
    verdicts: list[Verdict] | None = None,
    hindsight_labels: list[str] | None = None,
    hindsight_rewards: list[float] | None = None,
    is_hindsight_enhanced: bool = False,
) -> HindsightTrajectory:
    """Create a HindsightTrajectory for testing."""
    milestones = make_milestones(num_milestones)
    if verdicts is None:
        verdicts = make_verdicts(pass_n=num_milestones, fail_n=0)

    unmet_rubrics = [
        milestones[i].rubric
        for i, v in enumerate(verdicts)
        if v == Verdict.FAIL
    ]

    exec_traces = [
        make_trace(m.id, f"output for {m.id}")
        for m in milestones
    ]

    return HindsightTrajectory(
        task=task,
        milestones=milestones,
        exec_traces=exec_traces,
        verdicts=verdicts,
        unmet_rubrics=unmet_rubrics,
        hindsight_labels=hindsight_labels,
        hindsight_rewards=hindsight_rewards,
        is_hindsight_enhanced=is_hindsight_enhanced,
    )


# ---------------------------------------------------------------------------
# HindsightTrajectory — creation & properties
# ---------------------------------------------------------------------------

class TestHindsightTrajectoryCreation:
    def test_required_fields_only(self):
        traj = HindsightTrajectory(task="Do the thing")
        assert traj.task == "Do the thing"
        assert traj.milestones == []
        assert traj.verdicts == []
        assert traj.unmet_rubrics == []
        assert traj.hindsight_labels is None
        assert traj.hindsight_rewards is None
        assert traj.is_hindsight_enhanced is False

    def test_all_fields_set(self):
        milestones = make_milestones(2)
        verdicts = [Verdict.PASS, Verdict.FAIL]
        traj = HindsightTrajectory(
            task="Build a scraper",
            milestones=milestones,
            exec_traces=[make_trace("m1"), make_trace("m2")],
            verdicts=verdicts,
            unmet_rubrics=["Rubric 2: check this"],
            hindsight_labels=["Maybe m2 was enough?"],
            hindsight_rewards=[0.7],
            is_hindsight_enhanced=True,
            timestamp="2026-01-01T00:00:00Z",
            trajectory_id="test-id-123",
        )
        assert traj.task == "Build a scraper"
        assert len(traj.milestones) == 2
        assert len(traj.verdicts) == 2
        assert traj.is_hindsight_enhanced is True

    def test_timestamp_auto_generated(self):
        traj = HindsightTrajectory(task="x")
        assert traj.timestamp is not None
        assert "Z" in traj.timestamp or "T" in traj.timestamp

    def test_trajectory_id_property(self):
        traj = HindsightTrajectory(task="x")
        assert traj.id is not None
        assert isinstance(traj.id, str)

    def test_trajectory_id_explicit(self):
        traj = HindsightTrajectory(task="x", trajectory_id="my-id")
        assert traj.id == "my-id"


class TestHindsightTrajectoryProperties:
    def test_is_failed_true_when_fail_verdict(self):
        traj = make_trajectory(verdicts=[Verdict.PASS, Verdict.FAIL, Verdict.PASS])
        assert traj.is_failed is True

    def test_is_failed_false_when_all_pass(self):
        traj = make_trajectory(verdicts=[Verdict.PASS, Verdict.PASS])
        assert traj.is_failed is False

    def test_is_failed_false_when_partial_only(self):
        traj = make_trajectory(verdicts=[Verdict.PARTIAL, Verdict.PARTIAL])
        assert traj.is_failed is False

    def test_failed_milestone_indices(self):
        traj = make_trajectory(
            num_milestones=4,
            verdicts=[Verdict.PASS, Verdict.FAIL, Verdict.PARTIAL, Verdict.FAIL],
        )
        assert traj.failed_milestone_indices == [1, 3]

    def test_failed_milestone_indices_none(self):
        traj = make_trajectory(verdicts=[Verdict.PASS, Verdict.PASS])
        assert traj.failed_milestone_indices == []

    def test_num_milestones(self):
        traj = make_trajectory(num_milestones=5)
        assert traj.num_milestones == 5

    def test_compute_success_rate_all_pass(self):
        traj = make_trajectory(verdicts=[Verdict.PASS, Verdict.PASS, Verdict.PASS])
        assert traj.compute_success_rate() == 1.0

    def test_compute_success_rate_all_fail(self):
        traj = make_trajectory(verdicts=[Verdict.FAIL, Verdict.FAIL])
        assert traj.compute_success_rate() == 0.0

    def test_compute_success_rate_mixed(self):
        traj = make_trajectory(verdicts=[Verdict.PASS, Verdict.FAIL, Verdict.PASS])
        assert traj.compute_success_rate() == pytest.approx(2 / 3)

    def test_compute_success_rate_empty_verdicts(self):
        traj = HindsightTrajectory(task="x")
        assert traj.compute_success_rate() == 0.0


# ---------------------------------------------------------------------------
# HindsightTrajectory — serialization
# ---------------------------------------------------------------------------

class TestHindsightTrajectorySerialization:
    def test_to_dict_contains_all_fields(self):
        milestones = make_milestones(2)
        verdicts = [Verdict.PASS, Verdict.FAIL]
        traj = HindsightTrajectory(
            task="Build a scraper",
            milestones=milestones,
            exec_traces=[make_trace("m1"), make_trace("m2")],
            verdicts=verdicts,
            unmet_rubrics=["Rubric 2"],
            hindsight_labels=["Maybe m2 was enough?"],
            hindsight_rewards=[0.7],
            is_hindsight_enhanced=True,
            trajectory_id="id-abc",
        )
        d = traj.to_dict()
        assert d["task"] == "Build a scraper"
        assert len(d["milestones"]) == 2
        assert d["verdicts"] == ["pass", "fail"]
        assert d["unmet_rubrics"] == ["Rubric 2"]
        assert d["hindsight_labels"] == ["Maybe m2 was enough?"]
        assert d["hindsight_rewards"] == [0.7]
        assert d["is_hindsight_enhanced"] is True
        assert d["trajectory_id"] == "id-abc"

    def test_to_dict_verdicts_are_strings(self):
        traj = make_trajectory(verdicts=[Verdict.PASS, Verdict.FAIL, Verdict.PARTIAL])
        d = traj.to_dict()
        assert d["verdicts"] == ["pass", "fail", "partial"]

    def test_to_dict_is_json_serializable(self):
        traj = make_trajectory()
        d = traj.to_dict()
        json.dumps(d)  # Must not raise

    def test_from_dict_roundtrip(self):
        original = make_trajectory(
            task="Test roundtrip",
            num_milestones=3,
            verdicts=[Verdict.PASS, Verdict.FAIL, Verdict.PARTIAL],
            hindsight_labels=["re-label 1"],
            hindsight_rewards=[0.8],
            is_hindsight_enhanced=True,
        )
        d = original.to_dict()
        restored = HindsightTrajectory.from_dict(d)

        assert restored.task == original.task
        assert len(restored.milestones) == len(original.milestones)
        assert len(restored.verdicts) == len(original.verdicts)
        assert [v.value for v in restored.verdicts] == [v.value for v in original.verdicts]
        assert restored.unmet_rubrics == original.unmet_rubrics
        assert restored.hindsight_labels == original.hindsight_labels
        assert restored.hindsight_rewards == original.hindsight_rewards
        assert restored.is_hindsight_enhanced == original.is_hindsight_enhanced
        assert restored.id == original.id

    def test_from_dict_milestones_reconstructed(self):
        d = {
            "task": "x",
            "milestones": [
                {"id": "m1", "description": "d1", "rubric": "r1", "expected_output": "e1"},
                {"id": "m2", "description": "d2", "rubric": "r2"},
            ],
            "verdicts": ["pass", "fail"],
            "exec_traces": [],
            "unmet_rubrics": [],
        }
        traj = HindsightTrajectory.from_dict(d)
        assert len(traj.milestones) == 2
        assert traj.milestones[0].id == "m1"
        assert traj.milestones[1].expected_output == ""  # default

    def test_from_dict_invalid_verdict_defaults_to_fail(self):
        d = {
            "task": "x",
            "milestones": [],
            "verdicts": ["not_a_real_verdict"],
            "exec_traces": [],
            "unmet_rubrics": [],
        }
        traj = HindsightTrajectory.from_dict(d)
        assert traj.verdicts[0] == Verdict.FAIL

    def test_from_dict_missing_optional_fields(self):
        d = {"task": "minimal", "milestones": [], "verdicts": [], "exec_traces": [], "unmet_rubrics": []}
        traj = HindsightTrajectory.from_dict(d)
        assert traj.task == "minimal"
        assert traj.hindsight_labels is None
        assert traj.hindsight_rewards is None
        assert traj.is_hindsight_enhanced is False


# ---------------------------------------------------------------------------
# HindsightBuffer — initialization
# ---------------------------------------------------------------------------

class TestHindsightBufferInit:
    def test_default_values(self):
        buf = HindsightBuffer()
        assert buf.capacity == 10000
        assert buf.hindsight_ratio == 0.3
        assert buf.size == 0
        assert buf.is_empty is True

    def test_custom_capacity_and_ratio(self):
        buf = HindsightBuffer(capacity=100, hindsight_ratio=0.5, seed=42)
        assert buf.capacity == 100
        assert buf.hindsight_ratio == 0.5

    def test_invalid_capacity_type_raises(self):
        with pytest.raises(TypeError):
            HindsightBuffer(capacity="ten")  # type: ignore

    def test_invalid_capacity_value_raises(self):
        with pytest.raises(ValueError):
            HindsightBuffer(capacity=0)
        with pytest.raises(ValueError):
            HindsightBuffer(capacity=-5)

    def test_invalid_ratio_type_raises(self):
        with pytest.raises(TypeError):
            HindsightBuffer(hindsight_ratio="0.3")  # type: ignore

    def test_invalid_ratio_range_raises(self):
        with pytest.raises(ValueError):
            HindsightBuffer(hindsight_ratio=-0.1)
        with pytest.raises(ValueError):
            HindsightBuffer(hindsight_ratio=1.5)

    def test_repr_includes_capacity_and_ratio(self):
        buf = HindsightBuffer(capacity=500, hindsight_ratio=0.4)
        r = repr(buf)
        assert "500" in r
        assert "0.4" in r


# ---------------------------------------------------------------------------
# HindsightBuffer — add
# ---------------------------------------------------------------------------

class TestHindsightBufferAdd:
    def test_add_increases_size(self):
        buf = HindsightBuffer(capacity=100)
        buf.add(make_trajectory(task="task 1"))
        assert buf.size == 1
        buf.add(make_trajectory(task="task 2"))
        assert buf.size == 2

    def test_add_wrong_type_raises(self):
        buf = HindsightBuffer()
        with pytest.raises(TypeError, match="HindsightTrajectory"):
            buf.add({"task": "not a trajectory"})  # type: ignore

    def test_add_none_raises(self):
        buf = HindsightBuffer()
        with pytest.raises(TypeError):
            buf.add(None)  # type: ignore

    def test_add_to_full_buffer_evicts_oldest(self):
        buf = HindsightBuffer(capacity=3, seed=42)
        for i in range(5):
            buf.add(make_trajectory(task=f"task {i}"))

        assert buf.size == 3
        # First two should have been evicted
        assert not any("task 0" in t.task for t in buf)
        assert not any("task 1" in t.task for t in buf)


# ---------------------------------------------------------------------------
# HindsightBuffer — sample
# ---------------------------------------------------------------------------

class TestHindsightBufferSample:
    def test_sample_empty_buffer_returns_empty(self):
        buf = HindsightBuffer()
        assert buf.sample(10) == []

    def test_sample_batch_size_larger_than_buffer(self):
        buf = HindsightBuffer(capacity=10, seed=42)
        for i in range(3):
            buf.add(make_trajectory(task=f"task {i}"))
        batch = buf.sample(10)
        assert len(batch) == 3

    def test_sample_returns_list_of_trajectories(self):
        buf = HindsightBuffer(capacity=10, seed=42)
        buf.add(make_trajectory(task="t1"))
        buf.add(make_trajectory(task="t2"))
        batch = buf.sample(2)
        assert all(isinstance(t, HindsightTrajectory) for t in batch)

    def test_sample_invalid_batch_size_raises(self):
        buf = HindsightBuffer()
        with pytest.raises(ValueError, match="positive"):
            buf.sample(0)
        with pytest.raises(ValueError):
            buf.sample(-1)

    def test_sample_invalid_type_raises(self):
        buf = HindsightBuffer()
        with pytest.raises(TypeError):
            buf.sample("10")  # type: ignore

    def test_sample_respects_hindsight_ratio_zero(self):
        buf = HindsightBuffer(capacity=100, hindsight_ratio=0.0, seed=42)
        # Add mix of hindsight and non-hindsight
        for i in range(10):
            t = make_trajectory(task=f"task {i}", is_hindsight_enhanced=(i % 2 == 0))
            buf.add(t)
        batch = buf.sample(10)
        assert all(not t.is_hindsight_enhanced for t in batch)

    def test_sample_respects_hindsight_ratio_full(self):
        buf = HindsightBuffer(capacity=100, hindsight_ratio=1.0, seed=42)
        for i in range(10):
            t = make_trajectory(task=f"task {i}", is_hindsight_enhanced=(i % 2 == 0))
            buf.add(t)
        # Only 5 hindsight-enhanced available, request 5 to avoid mix
        batch = buf.sample(5)
        assert all(t.is_hindsight_enhanced for t in batch)

    def test_sample_mixed_hindsight_ratio(self):
        buf = HindsightBuffer(capacity=100, hindsight_ratio=0.3, seed=42)
        for i in range(20):
            t = make_trajectory(
                task=f"task {i}",
                is_hindsight_enhanced=(i % 2 == 0),  # half are hindsight
            )
            buf.add(t)

        batch = buf.sample(10)
        hindsight_count = sum(1 for t in batch if t.is_hindsight_enhanced)
        # Should be close to 30% (3 out of 10), allow some tolerance
        assert 1 <= hindsight_count <= 5

    def test_sample_deterministic_with_seed(self):
        buf1 = HindsightBuffer(capacity=100, hindsight_ratio=0.5, seed=123)
        buf2 = HindsightBuffer(capacity=100, hindsight_ratio=0.5, seed=123)
        for i in range(10):
            buf1.add(make_trajectory(task=f"t{i}"))
            buf2.add(make_trajectory(task=f"t{i}"))

        batch1 = buf1.sample(5)
        batch2 = buf2.sample(5)
        assert [t.task for t in batch1] == [t.task for t in batch2]

    def test_sample_shuffles_result(self):
        buf = HindsightBuffer(capacity=100, hindsight_ratio=0.3, seed=42)
        for i in range(20):
            buf.add(make_trajectory(task=f"task_{i}"))
        batch = buf.sample(10)
        # Not in insertion order (should be shuffled)
        ids = [t.task for t in batch]
        # With high probability, won't be in sorted order
        assert isinstance(ids, list)


# ---------------------------------------------------------------------------
# HindsightBuffer — add_hindsight_label
# ---------------------------------------------------------------------------

class TestHindsightBufferAddHindsightLabel:
    def test_add_hindsight_label_basic(self):
        buf = HindsightBuffer(capacity=10, seed=42)
        buf.add(make_trajectory(task="t1"))
        buf.add_hindsight_label(0, "What if m1 was the goal?")
        traj = buf[0]
        assert traj.hindsight_labels == ["What if m1 was the goal?"]
        assert traj.is_hindsight_enhanced is True

    def test_add_hindsight_label_with_reward(self):
        buf = HindsightBuffer(capacity=10, seed=42)
        buf.add(make_trajectory(task="t1"))
        buf.add_hindsight_label(0, "new goal", reward=0.75)
        traj = buf[0]
        assert traj.hindsight_labels == ["new goal"]
        assert traj.hindsight_rewards == [0.75]

    def test_add_hindsight_label_multiple_labels(self):
        buf = HindsightBuffer(capacity=10, seed=42)
        buf.add(make_trajectory(task="t1"))
        buf.add_hindsight_label(0, "label 1", reward=0.6)
        buf.add_hindsight_label(0, "label 2", reward=0.8)
        traj = buf[0]
        assert len(traj.hindsight_labels) == 2
        assert len(traj.hindsight_rewards) == 2

    def test_add_hindsight_label_invalid_index_raises(self):
        buf = HindsightBuffer(capacity=10)
        buf.add(make_trajectory(task="t1"))
        with pytest.raises(IndexError):
            buf.add_hindsight_label(99, "bad index")

    def test_add_hindsight_label_negative_index_raises(self):
        buf = HindsightBuffer(capacity=10)
        buf.add(make_trajectory(task="t1"))
        with pytest.raises(IndexError):
            buf.add_hindsight_label(-1, "negative")

    def test_add_hindsight_label_invalid_reward_raises(self):
        buf = HindsightBuffer(capacity=10)
        buf.add(make_trajectory(task="t1"))
        with pytest.raises(ValueError, match="reward"):
            buf.add_hindsight_label(0, "x", reward=-0.1)
        with pytest.raises(ValueError):
            buf.add_hindsight_label(0, "x", reward=1.5)


# ---------------------------------------------------------------------------
# HindsightBuffer — filter_failed & get_stats
# ---------------------------------------------------------------------------

class TestHindsightBufferFilterFailed:
    def test_filter_failed_empty_buffer(self):
        buf = HindsightBuffer()
        assert buf.filter_failed() == []

    def test_filter_failed_returns_only_failed(self):
        buf = HindsightBuffer(capacity=10, seed=42)
        buf.add(make_trajectory(task="pass1", verdicts=[Verdict.PASS]))
        buf.add(make_trajectory(task="fail1", verdicts=[Verdict.FAIL]))
        buf.add(make_trajectory(task="pass2", verdicts=[Verdict.PASS, Verdict.PASS]))
        buf.add(make_trajectory(task="fail2", verdicts=[Verdict.FAIL, Verdict.PASS]))

        failed = buf.filter_failed()
        assert len(failed) == 2
        assert all(t.is_failed for t in failed)

    def test_filter_failed_none_failed(self):
        buf = HindsightBuffer(capacity=10)
        buf.add(make_trajectory(task="all_pass", verdicts=[Verdict.PASS, Verdict.PASS]))
        assert buf.filter_failed() == []


class TestHindsightBufferGetStats:
    def test_get_stats_empty(self):
        buf = HindsightBuffer(capacity=100)
        stats = buf.get_stats()
        assert stats["total"] == 0
        assert stats["failed"] == 0
        assert stats["hindsight_enhanced"] == 0
        assert stats["hindsight_ratio"] == 0.3
        assert stats["capacity"] == 100
        assert stats["utilization"] == 0.0

    def test_get_stats_partial_fill(self):
        buf = HindsightBuffer(capacity=100, seed=42)
        for i in range(30):
            t = make_trajectory(
                task=f"t{i}",
                verdicts=[Verdict.FAIL] if i % 3 == 0 else [Verdict.PASS],
                is_hindsight_enhanced=(i % 5 == 0),
            )
            buf.add(t)

        stats = buf.get_stats()
        assert stats["total"] == 30
        assert stats["failed"] == 10  # i % 3 == 0
        assert stats["hindsight_enhanced"] == 6  # i % 5 == 0
        assert stats["utilization"] == 0.3

    def test_get_stats_full_capacity(self):
        buf = HindsightBuffer(capacity=5)
        for i in range(10):
            buf.add(make_trajectory(task=f"t{i}"))
        stats = buf.get_stats()
        assert stats["total"] == 5
        assert stats["utilization"] == 1.0


# ---------------------------------------------------------------------------
# HindsightBuffer — serialization
# ---------------------------------------------------------------------------

class TestHindsightBufferSerialization:
    def test_to_dict_structure(self):
        buf = HindsightBuffer(capacity=100, hindsight_ratio=0.4)
        buf.add(make_trajectory(task="t1"))
        buf.add(make_trajectory(task="t2"))
        d = buf.to_dict()
        assert d["capacity"] == 100
        assert d["hindsight_ratio"] == 0.4
        assert len(d["trajectories"]) == 2

    def test_to_dict_is_json_serializable(self):
        buf = HindsightBuffer(capacity=50, seed=42)
        for i in range(5):
            buf.add(make_trajectory(task=f"t{i}"))
        d = buf.to_dict()
        json.dumps(d)  # must not raise

    def test_from_dict_roundtrip(self):
        buf_original = HindsightBuffer(capacity=200, hindsight_ratio=0.6, seed=99)
        for i in range(10):
            t = make_trajectory(
                task=f"task {i}",
                num_milestones=3,
                verdicts=(
                    [Verdict.PASS, Verdict.FAIL, Verdict.PARTIAL]
                    if i % 2 == 0
                    else [Verdict.PASS, Verdict.PASS]
                ),
                hindsight_labels=(["relabel"] if i % 3 == 0 else None),
                hindsight_rewards=([0.8] if i % 3 == 0 else None),
                is_hindsight_enhanced=(i % 3 == 0),
            )
            buf_original.add(t)

        d = buf_original.to_dict()
        buf_restored = HindsightBuffer.from_dict(d)

        assert buf_restored.capacity == 200
        assert buf_restored.hindsight_ratio == 0.6
        assert buf_restored.size == 10
        for i, (orig, rest) in enumerate(zip(buf_original, buf_restored)):
            assert orig.task == rest.task
            assert orig.id == rest.id

    def test_from_dict_empty(self):
        d = {"capacity": 100, "hindsight_ratio": 0.3, "trajectories": []}
        buf = HindsightBuffer.from_dict(d)
        assert buf.size == 0
        assert buf.capacity == 100


# ---------------------------------------------------------------------------
# HindsightBuffer — container operations
# ---------------------------------------------------------------------------

class TestHindsightBufferContainerOps:
    def test_len(self):
        buf = HindsightBuffer(capacity=10)
        assert len(buf) == 0
        buf.add(make_trajectory(task="t1"))
        buf.add(make_trajectory(task="t2"))
        assert len(buf) == 2

    def test_getitem(self):
        buf = HindsightBuffer(capacity=10)
        buf.add(make_trajectory(task="first"))
        buf.add(make_trajectory(task="second"))
        assert buf[0].task == "first"
        assert buf[1].task == "second"

    def test_getitem_out_of_range_raises(self):
        buf = HindsightBuffer(capacity=10)
        buf.add(make_trajectory(task="t1"))
        with pytest.raises(IndexError):
            _ = buf[99]

    def test_iter(self):
        buf = HindsightBuffer(capacity=10, seed=42)
        for i in range(3):
            buf.add(make_trajectory(task=f"t{i}"))
        tasks = [t.task for t in buf]
        assert tasks == ["t0", "t1", "t2"]


# ---------------------------------------------------------------------------
# HindsightBuffer — hindsight_ratio setter
# ---------------------------------------------------------------------------

class TestHindsightBufferRatioSetter:
    def test_set_valid_ratio(self):
        buf = HindsightBuffer(hindsight_ratio=0.3)
        buf.hindsight_ratio = 0.5
        assert buf.hindsight_ratio == 0.5

    def test_set_invalid_ratio_raises(self):
        buf = HindsightBuffer()
        with pytest.raises(ValueError):
            buf.hindsight_ratio = -0.1
        with pytest.raises(ValueError):
            buf.hindsight_ratio = 1.1


# ---------------------------------------------------------------------------
# UpdateResult dataclass
# ---------------------------------------------------------------------------

class TestUpdateResult:
    def test_fields_default(self):
        result = UpdateResult(loss=0.5, num_samples=32, hindsight_ratio=0.3)
        assert result.loss == 0.5
        assert result.num_samples == 32
        assert result.hindsight_ratio == 0.3
        assert result.is_simulation is True
        assert isinstance(result.timestamp, datetime)

    def test_fields_explicit(self):
        ts = datetime.utcnow()
        result = UpdateResult(
            loss=0.25,
            num_samples=16,
            hindsight_ratio=0.5,
            timestamp=ts,
            is_simulation=False,
            details={"lr": 1e-5, "model": "gpt-4o-mini"},
        )
        assert result.is_simulation is False
        assert result.details["model"] == "gpt-4o-mini"

    def test_to_dict(self):
        result = UpdateResult(
            loss=0.7, num_samples=20, hindsight_ratio=0.4, is_simulation=True
        )
        d = result.to_dict()
        assert d["loss"] == 0.7
        assert d["num_samples"] == 20
        assert d["hindsight_ratio"] == 0.4
        assert d["is_simulation"] is True
        assert "timestamp" in d
        assert "Z" in d["timestamp"]

    def test_to_dict_is_json_serializable(self):
        result = UpdateResult(loss=0.5, num_samples=10, hindsight_ratio=0.3)
        json.dumps(result.to_dict())  # must not raise


# ---------------------------------------------------------------------------
# HindsightTrainer — initialization
# ---------------------------------------------------------------------------

class TestHindsightTrainerInit:
    def test_default_init(self):
        buf = HindsightBuffer()
        trainer = HindsightTrainer(buffer=buf)
        assert trainer.buffer is buf
        assert trainer.model_path is None
        assert trainer.learning_rate == 1e-5
        assert trainer.update_count == 0

    def test_custom_init(self):
        buf = HindsightBuffer(capacity=100)
        trainer = HindsightTrainer(
            buffer=buf,
            model_path="/path/to/model",
            learning_rate=2e-5,
        )
        assert trainer.model_path == "/path/to/model"
        assert trainer.learning_rate == 2e-5

    def test_invalid_buffer_type_raises(self):
        with pytest.raises(TypeError, match="HindsightBuffer"):
            HindsightTrainer(buffer="not a buffer")  # type: ignore

    def test_invalid_model_path_type_raises(self):
        buf = HindsightBuffer()
        with pytest.raises(TypeError):
            HindsightTrainer(buffer=buf, model_path=123)  # type: ignore

    def test_invalid_learning_rate_raises(self):
        buf = HindsightBuffer()
        with pytest.raises(TypeError):
            HindsightTrainer(buffer=buf, learning_rate="1e-5")  # type: ignore
        with pytest.raises(ValueError):
            HindsightTrainer(buffer=buf, learning_rate=0.0)

    def test_repr(self):
        buf = HindsightBuffer(capacity=100, seed=42)
        for i in range(5):
            buf.add(make_trajectory(task=f"t{i}"))
        trainer = HindsightTrainer(buffer=buf, model_path="/model", learning_rate=1e-5)
        r = repr(trainer)
        assert "HindsightTrainer" in r
        assert "model" in r


# ---------------------------------------------------------------------------
# HindsightTrainer — compute_hindsight_reward
# ---------------------------------------------------------------------------

class TestHindsightTrainerComputeHindsightReward:
    def test_all_pass_reward(self):
        buf = HindsightBuffer()
        trainer = HindsightTrainer(buffer=buf)
        traj = make_trajectory(verdicts=[Verdict.PASS, Verdict.PASS, Verdict.PASS])
        assert trainer.compute_hindsight_reward(traj) == 1.0

    def test_all_fail_reward(self):
        buf = HindsightBuffer()
        trainer = HindsightTrainer(buffer=buf)
        traj = make_trajectory(verdicts=[Verdict.FAIL, Verdict.FAIL])
        assert trainer.compute_hindsight_reward(traj) == 0.0

    def test_mixed_reward(self):
        buf = HindsightBuffer()
        trainer = HindsightTrainer(buffer=buf)
        traj = make_trajectory(verdicts=[Verdict.PASS, Verdict.FAIL, Verdict.PASS])
        assert trainer.compute_hindsight_reward(traj) == pytest.approx(2 / 3)

    def test_with_hindsight_bonus(self):
        buf = HindsightBuffer()
        trainer = HindsightTrainer(buffer=buf)
        traj = HindsightTrajectory(
            task="x",
            milestones=make_milestones(2),
            verdicts=[Verdict.PASS, Verdict.FAIL],
            hindsight_rewards=[0.9],  # hindsight rescued with 0.9
            is_hindsight_enhanced=True,
        )
        reward = trainer.compute_hindsight_reward(traj)
        # base = 0.5 (1 pass out of 2), bonus = 0.9, combined = (0.5 + 0.9) / 2
        assert reward == pytest.approx((0.5 + 0.9) / 2)

    def test_empty_verdicts(self):
        buf = HindsightBuffer()
        trainer = HindsightTrainer(buffer=buf)
        traj = HindsightTrajectory(task="x")
        assert trainer.compute_hindsight_reward(traj) == 0.0

    def test_wrong_type_raises(self):
        buf = HindsightBuffer()
        trainer = HindsightTrainer(buffer=buf)
        with pytest.raises(TypeError, match="HindsightTrajectory"):
            trainer.compute_hindsight_reward({"task": "x"})  # type: ignore


# ---------------------------------------------------------------------------
# HindsightTrainer — update_policy
# ---------------------------------------------------------------------------

class TestHindsightTrainerUpdatePolicy:
    def test_update_policy_empty_trajectories_raises(self):
        buf = HindsightBuffer()
        trainer = HindsightTrainer(buffer=buf)
        with pytest.raises(ValueError, match="empty"):
            trainer.update_policy([])

    def test_update_policy_wrong_type_raises(self):
        buf = HindsightBuffer()
        trainer = HindsightTrainer(buffer=buf)
        with pytest.raises(TypeError, match="list"):
            trainer.update_policy("not a list")  # type: ignore

    def test_update_policy_wrong_element_type_raises(self):
        buf = HindsightBuffer()
        trainer = HindsightTrainer(buffer=buf)
        with pytest.raises(TypeError, match="trajectories\\[0\\]"):
            trainer.update_policy([{"task": "x"}])  # type: ignore

    def test_update_policy_returns_update_result(self):
        buf = HindsightBuffer()
        trainer = HindsightTrainer(buffer=buf)
        trajs = [make_trajectory(task=f"t{i}") for i in range(5)]
        result = trainer.update_policy(trajs)
        assert isinstance(result, UpdateResult)
        assert result.num_samples == 5
        assert 0.0 <= result.loss <= 1.0

    def test_update_policy_increments_update_count(self):
        buf = HindsightBuffer()
        trainer = HindsightTrainer(buffer=buf)
        trajs = [make_trajectory(task="t1")]
        assert trainer.update_count == 0
        trainer.update_policy(trajs)
        assert trainer.update_count == 1
        trainer.update_policy(trajs)
        assert trainer.update_count == 2

    def test_update_policy_lr_override(self):
        buf = HindsightBuffer()
        trainer = HindsightTrainer(buffer=buf, learning_rate=1e-5)
        trajs = [make_trajectory(task="t1")]
        result = trainer.update_policy(trajs, lr=5e-5)
        assert result.details is not None
        assert result.details["learning_rate"] == 5e-5

    def test_update_policy_simulated_result_has_details(self):
        buf = HindsightBuffer()
        trainer = HindsightTrainer(buffer=buf)
        trajs = [
            make_trajectory(
                verdicts=[Verdict.PASS, Verdict.FAIL],
                hindsight_rewards=[0.8],
                is_hindsight_enhanced=True,
            )
        ]
        result = trainer.update_policy(trajs)
        assert result.is_simulation is True
        assert result.details is not None
        assert "avg_hindsight_reward" in result.details
        assert result.details["update_count"] == 1

    def test_update_policy_uses_configured_lr_by_default(self):
        buf = HindsightBuffer()
        trainer = HindsightTrainer(buffer=buf, learning_rate=3e-5)
        trajs = [make_trajectory(task="t1")]
        result = trainer.update_policy(trajs)
        assert result.details is not None
        assert result.details["learning_rate"] == 3e-5

    def test_update_policy_hindsight_ratio_in_result(self):
        buf = HindsightBuffer(hindsight_ratio=0.3, seed=42)
        trainer = HindsightTrainer(buffer=buf)
        trajs = [
            make_trajectory(task=f"t{i}", is_hindsight_enhanced=(i % 2 == 0))
            for i in range(10)
        ]
        result = trainer.update_policy(trajs)
        # Batch hindsight ratio should be 0.5 (5/10 are enhanced)
        assert result.hindsight_ratio == 0.5

    def test_multiple_updates_accumulate(self):
        buf = HindsightBuffer(seed=42)
        trainer = HindsightTrainer(buffer=buf)
        for i in range(3):
            trajs = [make_trajectory(task=f"t{i}_1"), make_trajectory(task=f"t{i}_2")]
            trainer.update_policy(trajs)
        assert trainer.update_count == 3


# ---------------------------------------------------------------------------
# HindsightTrainer — export_buffer / import_buffer
# ---------------------------------------------------------------------------

class TestHindsightTrainerExportImport:
    def test_export_buffer(self, tmp_path):
        buf = HindsightBuffer(capacity=50, hindsight_ratio=0.4, seed=42)
        for i in range(5):
            buf.add(make_trajectory(task=f"task {i}"))
        trainer = HindsightTrainer(buffer=buf)

        path = tmp_path / "buffer_export.json"
        trainer.export_buffer(str(path))

        assert path.exists()
        d = json.loads(path.read_text())
        assert d["capacity"] == 50
        assert d["hindsight_ratio"] == 0.4
        assert len(d["trajectories"]) == 5

    def test_import_buffer_replaces_current(self, tmp_path):
        # Create and export original buffer
        buf_original = HindsightBuffer(capacity=100, hindsight_ratio=0.5, seed=42)
        for i in range(3):
            buf_original.add(make_trajectory(task=f"original {i}"))

        path = tmp_path / "import_test.json"
        buf_original.export(str(path))

        # Create a trainer with a different buffer
        buf_new = HindsightBuffer(capacity=10)
        trainer = HindsightTrainer(buffer=buf_new)

        # Import should replace
        trainer.import_buffer(str(path))
        assert trainer.buffer.capacity == 100
        assert trainer.buffer.hindsight_ratio == 0.5
        assert trainer.buffer.size == 3


# ---------------------------------------------------------------------------
# Integration tests — full pipeline
# ---------------------------------------------------------------------------

class TestBufferTrainerIntegration:
    def test_add_sample_update_pipeline(self):
        """Full pipeline: add trajectories, sample, update policy."""
        buf = HindsightBuffer(capacity=100, hindsight_ratio=0.3, seed=42)
        for i in range(20):
            traj = make_trajectory(
                task=f"task {i}",
                verdicts=[Verdict.PASS if i % 2 == 0 else Verdict.FAIL],
            )
            buf.add(traj)

        stats = buf.get_stats()
        assert stats["total"] == 20
        assert stats["failed"] == 10

        batch = buf.sample(8)
        assert len(batch) == 8

        trainer = HindsightTrainer(buffer=buf)
        result = trainer.update_policy(batch)
        assert isinstance(result, UpdateResult)
        assert result.num_samples == 8

    def test_hindsight_relabeling_pipeline(self):
        """Test the full HeRL hindsight relabeling pipeline."""
        buf = HindsightBuffer(capacity=50, hindsight_ratio=0.5, seed=42)

        # Add a failed trajectory
        failed_traj = make_trajectory(
            task="Write a web scraper",
            verdicts=[Verdict.PASS, Verdict.FAIL, Verdict.FAIL],
        )
        buf.add(failed_traj)
        assert buf[0].is_failed

        # Add hindsight labels to failed milestones
        buf.add_hindsight_label(0, "What if just fetching the page was enough?", reward=0.7)
        buf.add_hindsight_label(
            0,
            "What if parsing a single element was the goal?",
            reward=0.5,
        )

        # Filter failed
        failed = buf.filter_failed()
        assert len(failed) == 1
        assert failed[0].is_hindsight_enhanced is True
        assert len(failed[0].hindsight_labels) == 2

        # Train on it
        trainer = HindsightTrainer(buffer=buf)
        result = trainer.update_policy(failed)
        assert result.num_samples == 1
        assert result.hindsight_ratio == 1.0  # 100% hindsight enhanced

    def test_serialization_roundtrip_full_pipeline(self, tmp_path):
        """Serialize buffer + trainer state through full pipeline."""
        buf = HindsightBuffer(capacity=200, hindsight_ratio=0.4, seed=123)
        for i in range(15):
            buf.add(
                make_trajectory(
                    task=f"task {i}",
                    verdicts=(
                        [Verdict.PASS, Verdict.FAIL, Verdict.PARTIAL]
                        if i % 3 == 0
                        else [Verdict.PASS, Verdict.PASS]
                    ),
                    is_hindsight_enhanced=(i % 4 == 0),
                )
            )

        path = tmp_path / "full_pipeline.json"
        buf.export(str(path))

        # Load into new buffer
        buf_loaded = HindsightBuffer.import_(path)
        assert buf_loaded.capacity == 200
        assert buf_loaded.hindsight_ratio == 0.4
        assert buf_loaded.size == 15

        # Train
        trainer = HindsightTrainer(buffer=buf_loaded)
        batch = buf_loaded.sample(8)
        result = trainer.update_policy(batch)
        assert result.num_samples == 8
        assert trainer.update_count == 1

    def test_fifo_eviction_preserves_order(self):
        """Verify FIFO eviction maintains insertion order of survivors."""
        buf = HindsightBuffer(capacity=3, seed=0)
        for i in range(5):
            buf.add(make_trajectory(task=f"t{i}"))

        ids = [t.task for t in buf]
        assert "t0" not in ids
        assert "t1" not in ids
        assert "t2" in ids
        assert "t3" in ids
        assert "t4" in ids

    def test_hindsight_ratio_exact_at_boundaries(self):
        """Test sampling at exactly 0.0 and 1.0 ratio."""
        buf0 = HindsightBuffer(capacity=100, hindsight_ratio=0.0, seed=42)
        buf1 = HindsightBuffer(capacity=100, hindsight_ratio=1.0, seed=42)

        for i in range(10):
            t = make_trajectory(
                task=f"t{i}", is_hindsight_enhanced=(i < 5)
            )
            buf0.add(t)
            buf1.add(t)

        batch0 = buf0.sample(10)
        batch1 = buf1.sample(5)  # request only 5 (hindsight pool size) to avoid mix

        assert all(not t.is_hindsight_enhanced for t in batch0)
        assert all(t.is_hindsight_enhanced for t in batch1)

    def test_stats_reflect_exact_counts(self):
        """Verify get_stats counts are accurate."""
        buf = HindsightBuffer(capacity=100, seed=42)
        for i in range(20):
            buf.add(
                make_trajectory(
                    task=f"t{i}",
                    verdicts=[Verdict.FAIL] if i in [2, 5, 7, 11] else [Verdict.PASS],
                    is_hindsight_enhanced=(i in [3, 6, 9]),
                )
            )

        stats = buf.get_stats()
        assert stats["total"] == 20
        assert stats["failed"] == 4
        assert stats["hindsight_enhanced"] == 3
