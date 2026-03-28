"""HeRL-style Hindsight Experience Buffer.

Stores failed trajectories and enables policy improvement beyond the
current response distribution via hindsight relabeling.

References:
    - HeRL: Hindsight Experience Replay for LLMs
    - MiRA: Milestoning RL Enhanced Agent
"""

from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional

from heros.planner import Milestone
from heros.critic import Verdict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class HindsightTrajectory:
    """A single trajectory for the hindsight experience buffer.

    Captures a complete execution attempt against a task, including
    planned milestones, execution traces, critic verdicts, and any
    unmet rubrics that can be used for hindsight relabeling.

    Parameters
    ----------
    task : str
        The original task description.
    milestones : List[Milestone]
        Planned milestones (subgoals) for the task.
    exec_traces : List[Dict]
        Execution traces per milestone. Each dict should contain at minimum
        a "content" or "output" key with the trace text.
    verdicts : List[Verdict]
        Critic verdicts per milestone.
    unmet_rubrics : List[str]
        Descriptions of failed milestone rubrics (rubric text or description).
    hindsight_labels : Optional[List[str]], optional
        Re-labeled goals from hindsight relabeling. If a milestone failed,
        this contains alternative goal descriptions ("what if this subgoal
        was the goal?").
    hindsight_rewards : Optional[List[float]], optional
        Reward signals for each relabeled trajectory, in [0.0, 1.0].
    is_hindsight_enhanced : bool, optional
        Whether this trajectory has been hindsight-enhanced (relabeled).
        Defaults to False.
    timestamp : Optional[str], optional
        ISO-8601 timestamp of when the trajectory was created.
        Defaults to current UTC time.
    trajectory_id : Optional[str], optional
        Unique identifier for this trajectory. Auto-generated if not provided.

    Attributes
    ----------
    is_failed : bool
        True if any milestone verdict is FAIL.
    failed_milestone_indices : List[int]
        Indices of milestones that failed.
    """

    task: str
    milestones: List[Milestone] = field(default_factory=list)
    exec_traces: List[Dict[str, Any]] = field(default_factory=list)
    verdicts: List[Verdict] = field(default_factory=list)
    unmet_rubrics: List[str] = field(default_factory=list)
    hindsight_labels: Optional[List[str]] = None
    hindsight_rewards: Optional[List[float]] = None
    is_hindsight_enhanced: bool = False
    timestamp: Optional[str] = None
    trajectory_id: Optional[str] = None

    # Auto-generated on first access (lazily initialized)
    _id_cache: Optional[str] = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        # Auto-generate trajectory_id if not provided
        if self.trajectory_id is None and self._id_cache is None:
            import uuid

            self._id_cache = str(uuid.uuid4())[:8]

    @property
    def id(self) -> str:
        """Unique trajectory identifier."""
        if self.trajectory_id is not None:
            return self.trajectory_id
        if self._id_cache is None:
            import uuid

            self._id_cache = str(uuid.uuid4())[:8]
        return self._id_cache

    @property
    def is_failed(self) -> bool:
        """True if any milestone verdict is FAIL."""
        return any(v == Verdict.FAIL for v in self.verdicts)

    @property
    def failed_milestone_indices(self) -> List[int]:
        """Indices of milestones that received a FAIL verdict."""
        return [i for i, v in enumerate(self.verdicts) if v == Verdict.FAIL]

    @property
    def num_milestones(self) -> int:
        """Number of milestones in this trajectory."""
        return len(self.milestones)

    def compute_success_rate(self) -> float:
        """Fraction of milestones that passed."""
        if not self.verdicts:
            return 0.0
        passed = sum(1 for v in self.verdicts if v == Verdict.PASS)
        return passed / len(self.verdicts)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dictionary.

        Returns
        -------
        Dict[str, Any]
            Serializable dictionary representation.
        """
        return {
            "task": self.task,
            "milestones": [
                {
                    "id": m.id,
                    "description": m.description,
                    "rubric": m.rubric,
                    "expected_output": m.expected_output,
                }
                for m in self.milestones
            ],
            "exec_traces": self.exec_traces,
            "verdicts": [v.value for v in self.verdicts],
            "unmet_rubrics": self.unmet_rubrics,
            "hindsight_labels": self.hindsight_labels,
            "hindsight_rewards": self.hindsight_rewards,
            "is_hindsight_enhanced": self.is_hindsight_enhanced,
            "timestamp": self.timestamp,
            "trajectory_id": self.id,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HindsightTrajectory":
        """Deserialize from a JSON-compatible dictionary.

        Parameters
        ----------
        d : Dict[str, Any]
            Dictionary produced by :meth:`to_dict`.

        Returns
        -------
        HindsightTrajectory
            Reconstructed trajectory object.
        """
        milestones = [
            Milestone(
                id=m["id"],
                description=m["description"],
                rubric=m["rubric"],
                expected_output=m.get("expected_output", ""),
            )
            for m in d.get("milestones", [])
        ]

        verdicts = []
        for v_str in d.get("verdicts", []):
            try:
                verdicts.append(Verdict(v_str))
            except ValueError:
                verdicts.append(Verdict.FAIL)

        return cls(
            task=d.get("task", ""),
            milestones=milestones,
            exec_traces=d.get("exec_traces", []),
            verdicts=verdicts,
            unmet_rubrics=d.get("unmet_rubrics", []),
            hindsight_labels=d.get("hindsight_labels"),
            hindsight_rewards=d.get("hindsight_rewards"),
            is_hindsight_enhanced=d.get("is_hindsight_enhanced", False),
            timestamp=d.get("timestamp"),
            trajectory_id=d.get("trajectory_id"),
        )


# ---------------------------------------------------------------------------
# HindsightBuffer
# ---------------------------------------------------------------------------


class HindsightBuffer:
    """FIFO hindsight experience buffer with configurable hindsight sampling ratio.

    Stores failed trajectories and supports hindsight relabeling for
    policy improvement beyond the current response distribution.

    Parameters
    ----------
    capacity : int, optional
        Maximum number of trajectories to store. Uses FIFO eviction when
        full. Defaults to 10000.
    hindsight_ratio : float, optional
        Fraction of each sampled batch that should come from
        hindsight-enhanced (relabeled) trajectories. Must be in [0.0, 1.0].
        Defaults to 0.3 (30%).
    seed : int, optional
        Random seed for reproducibility.

    Examples
    --------
    >>> buffer = HindsightBuffer(capacity=1000, hindsight_ratio=0.3, seed=42)
    >>> buffer.add(trajectory)
    >>> batch = buffer.sample(batch_size=32)
    >>> stats = buffer.get_stats()
    """

    MIN_HINDSIGHT_RATIO = 0.0
    MAX_HINDSIGHT_RATIO = 1.0
    DEFAULT_CAPACITY = 10000
    DEFAULT_HINDSIGHT_RATIO = 0.3

    def __init__(
        self,
        capacity: int = DEFAULT_CAPACITY,
        hindsight_ratio: float = DEFAULT_HINDSIGHT_RATIO,
        seed: Optional[int] = None,
    ) -> None:
        if not isinstance(capacity, int):
            raise TypeError(f"capacity must be an int, got {type(capacity).__name__}")
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self._capacity = capacity

        if not isinstance(hindsight_ratio, (int, float)):
            raise TypeError(
                f"hindsight_ratio must be a float, got {type(hindsight_ratio).__name__}"
            )
        if not (self.MIN_HINDSIGHT_RATIO <= hindsight_ratio <= self.MAX_HINDSIGHT_RATIO):
            raise ValueError(
                f"hindsight_ratio must be between {self.MIN_HINDSIGHT_RATIO} "
                f"and {self.MAX_HINDSIGHT_RATIO}, got {hindsight_ratio}"
            )
        self._hindsight_ratio = float(hindsight_ratio)

        # Seed for reproducible sampling
        import random

        self._rng = random.Random(seed)

        # Internal storage: deque for O(1) append/pop with maxlen for FIFO
        self._buffer: Deque[HindsightTrajectory] = deque(maxlen=capacity)
        self._add_order: List[int] = []  # Track insertion order for stable indexing
        self._id_to_idx: Dict[str, int] = {}  # trajectory_id -> current index
        self._next_idx: int = 0

    # ---------------------------------------------------------------------------
    # Properties
    # ---------------------------------------------------------------------------

    @property
    def capacity(self) -> int:
        """Maximum buffer capacity."""
        return self._capacity

    @property
    def hindsight_ratio(self) -> float:
        """Configured hindsight sampling ratio."""
        return self._hindsight_ratio

    @hindsight_ratio.setter
    def hindsight_ratio(self, value: float) -> None:
        """Update hindsight ratio (runtime configuration)."""
        if not (self.MIN_HINDSIGHT_RATIO <= value <= self.MAX_HINDSIGHT_RATIO):
            raise ValueError(
                f"hindsight_ratio must be between {self.MIN_HINDSIGHT_RATIO} "
                f"and {self.MAX_HINDSIGHT_RATIO}, got {value}"
            )
        self._hindsight_ratio = float(value)

    @property
    def size(self) -> int:
        """Current number of trajectories in the buffer."""
        return len(self._buffer)

    @property
    def is_empty(self) -> bool:
        """True if the buffer contains no trajectories."""
        return len(self._buffer) == 0

    # ---------------------------------------------------------------------------
    # Core operations
    # ---------------------------------------------------------------------------

    def add(self, trajectory: HindsightTrajectory) -> None:
        """Add a trajectory to the buffer.

        Uses FIFO eviction when the buffer is at capacity — the oldest
        trajectory is removed to make room for the new one.

        Parameters
        ----------
        trajectory : HindsightTrajectory
            The trajectory to add. Must not be None.

        Raises
        ------
        TypeError
            If trajectory is not a HindsightTrajectory.
        """
        if not isinstance(trajectory, HindsightTrajectory):
            raise TypeError(
                f"Expected HindsightTrajectory, got {type(trajectory).__name__}"
            )

        # Evict oldest if at capacity
        if len(self._buffer) >= self._capacity:
            oldest = self._buffer[0]
            if oldest.id in self._id_to_idx:
                del self._id_to_idx[oldest.id]

        # Remove from front (oldest), add to back (newest)
        # Using deque with maxlen handles FIFO automatically
        self._buffer.append(trajectory)
        self._id_to_idx[trajectory.id] = len(self._buffer) - 1

    def sample(self, batch_size: int) -> List[HindsightTrajectory]:
        """Sample a batch of trajectories respecting the hindsight ratio.

        The batch contains approximately ``hindsight_ratio * batch_size``
        hindsight-enhanced trajectories (if available) and the remainder
        from the general buffer.

        Parameters
        ----------
        batch_size : int
            Number of trajectories to sample. Must be positive.

        Returns
        -------
        List[HindsightTrajectory]
            A list of sampled trajectories. May be smaller than batch_size
            if the buffer does not contain enough trajectories.

        Raises
        ------
        ValueError
            If batch_size is not positive.
        """
        if not isinstance(batch_size, int):
            raise TypeError(
                f"batch_size must be an int, got {type(batch_size).__name__}"
            )
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        if len(self._buffer) == 0:
            return []

        actual_batch_size = min(batch_size, len(self._buffer))
        hindsight_count = int(round(actual_batch_size * self._hindsight_ratio))
        hindsight_count = min(hindsight_count, actual_batch_size)

        # Separate hindsight-enhanced and non-hindsight trajectories
        hindsight_trajs = [t for t in self._buffer if t.is_hindsight_enhanced]
        non_hindsight_trajs = [t for t in self._buffer if not t.is_hindsight_enhanced]

        sampled: List[HindsightTrajectory] = []

        # Sample from hindsight-enhanced pool
        hindsight_available = len(hindsight_trajs)
        if hindsight_count > 0 and hindsight_available > 0:
            n_hindsight_sample = min(hindsight_count, hindsight_available)
            sampled.extend(self._rng.sample(hindsight_trajs, n_hindsight_sample))

        # Fill remainder from non-hindsight pool
        remaining = actual_batch_size - len(sampled)
        non_hindsight_available = len(non_hindsight_trajs)
        if remaining > 0 and non_hindsight_available > 0:
            n_non_sample = min(remaining, non_hindsight_available)
            sampled.extend(self._rng.sample(non_hindsight_trajs, n_non_sample))

        # Shuffle the final batch so hindsight/regular are interleaved
        self._rng.shuffle(sampled)

        return sampled

    def add_hindsight_label(
        self,
        trajectory_idx: int,
        new_label: str,
        reward: Optional[float] = None,
    ) -> None:
        """Relabel a failed trajectory with hindsight (HeRL-style).

        Takes an unmet rubric from the trajectory and adds it as a
        hindsight label, effectively re-framing the failed trajectory
        as a successful one with a different (simpler) goal.

        Parameters
        ----------
        trajectory_idx : int
            Index of the trajectory in the buffer (0 = oldest).
        new_label : str
            The new hindsight label (alternative goal description).
        reward : float, optional
            The hindsight reward for this relabeled trajectory.
            Should be in [0.0, 1.0].

        Raises
        ------
        IndexError
            If trajectory_idx is out of range.
        ValueError
            If reward is outside [0.0, 1.0].
        """
        if trajectory_idx < 0 or trajectory_idx >= len(self._buffer):
            raise IndexError(
                f"trajectory_idx {trajectory_idx} out of range "
                f"[0, {len(self._buffer) - 1}]"
            )

        if reward is not None and not (0.0 <= reward <= 1.0):
            raise ValueError(f"reward must be in [0.0, 1.0], got {reward}")

        trajectory = self._buffer[trajectory_idx]

        # Initialize hindsight fields if not already set
        if trajectory.hindsight_labels is None:
            trajectory.hindsight_labels = []
        if trajectory.hindsight_rewards is None:
            trajectory.hindsight_rewards = []

        trajectory.hindsight_labels.append(new_label)
        if reward is not None:
            trajectory.hindsight_rewards.append(reward)

        trajectory.is_hindsight_enhanced = True

    def get_stats(self) -> Dict[str, Any]:
        """Return buffer statistics.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - ``total``: total number of trajectories
            - ``failed``: number of trajectories with at least one FAIL verdict
            - ``hindsight_enhanced``: number of hindsight-enhanced trajectories
            - ``hindsight_ratio``: configured hindsight sampling ratio
            - ``capacity``: buffer capacity
            - ``utilization``: fraction of capacity currently used
        """
        total = len(self._buffer)
        failed = sum(1 for t in self._buffer if t.is_failed)
        hindsight_enhanced = sum(1 for t in self._buffer if t.is_hindsight_enhanced)

        return {
            "total": total,
            "failed": failed,
            "hindsight_enhanced": hindsight_enhanced,
            "hindsight_ratio": self._hindsight_ratio,
            "capacity": self._capacity,
            "utilization": total / self._capacity if self._capacity > 0 else 0.0,
        }

    def filter_failed(self) -> List[HindsightTrajectory]:
        """Return all trajectories with unmet (failed) rubrics.

        Returns
        -------
        List[HindsightTrajectory]
            All trajectories in the buffer that have at least one
            FAIL verdict.
        """
        return [t for t in self._buffer if t.is_failed]

    # ---------------------------------------------------------------------------
    # Serialization
    # ---------------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the buffer to a JSON-compatible dictionary.

        Returns
        -------
        Dict[str, Any]
            Serializable dictionary representation of the buffer.
        """
        return {
            "capacity": self._capacity,
            "hindsight_ratio": self._hindsight_ratio,
            "trajectories": [t.to_dict() for t in self._buffer],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HindsightBuffer":
        """Deserialize a buffer from a JSON-compatible dictionary.

        Parameters
        ----------
        d : Dict[str, Any]
            Dictionary produced by :meth:`to_dict`.

        Returns
        -------
        HindsightBuffer
            Reconstructed buffer object.
        """
        capacity = d.get("capacity", cls.DEFAULT_CAPACITY)
        hindsight_ratio = d.get("hindsight_ratio", cls.DEFAULT_HINDSIGHT_RATIO)
        buffer = cls(capacity=capacity, hindsight_ratio=hindsight_ratio)

        for traj_dict in d.get("trajectories", []):
            trajectory = HindsightTrajectory.from_dict(traj_dict)
            buffer.add(trajectory)

        return buffer

    def export(self, path: str) -> None:
        """Export the buffer to a JSON file.

        Parameters
        ----------
        path : str
            File path to write the JSON to.
        """
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info("Exported buffer with %d trajectories to %s", len(self._buffer), path)

    @classmethod
    def import_(cls, path: str) -> "HindsightBuffer":
        """Import a buffer from a JSON file.

        Parameters
        ----------
        path : str
            File path to read the JSON from.

        Returns
        -------
        HindsightBuffer
            Loaded buffer object.
        """
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        logger.info("Imported buffer from %s", path)
        return cls.from_dict(d)

    # ---------------------------------------------------------------------------
    # Iterator & container helpers
    # ---------------------------------------------------------------------------

    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self._buffer)

    def __getitem__(self, index: int) -> HindsightTrajectory:
        """Index into the buffer (oldest = 0)."""
        return self._buffer[index]

    def __iter__(self):
        """Iterate over trajectories (oldest to newest)."""
        return iter(self._buffer)

    def __repr__(self) -> str:
        return (
            f"HindsightBuffer(capacity={self._capacity}, "
            f"size={len(self._buffer)}, "
            f"hindsight_ratio={self._hindsight_ratio})"
        )
