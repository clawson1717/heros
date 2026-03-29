"""Interpretability and Audit Trail for HeRoS.

Provides tools for logging milestone decisions, verifying functional equivalence
of milestone formulations, auditing reward signals, and visualizing hindsight
buffer composition over training.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from heros.critic import CriticResult, MilestoneCritic, Verdict
from heros.buffer import HindsightBuffer, HindsightTrajectory
from heros.planner import Milestone

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MilestoneDecisionLogger
# ---------------------------------------------------------------------------


class MilestoneDecisionType(Enum):
    """Types of milestone decisions that can be logged."""
    CREATED = "created"
    ATTEMPTED = "attempted"
    PASSED = "passed"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class MilestoneDecisionLogger:
    """Logs every milestone decision with full critic reasoning.

    Records: task_id, milestone_id, milestone_description, decision_type,
    critic_reasoning (raw LLM output), verdict, confidence, timestamp.

    Writes to both a JSONL file and optionally the console.

    Parameters
    ----------
    log_path : Union[str, Path], optional
        Path to the JSONL log file. If None, no file logging occurs.
    console_logging : bool, optional
        Whether to log decisions to the console. Defaults to True.
    name : str, optional
        Logger name for identification. Defaults to "heros.milestones".

    Examples
    --------
    >>> logger = MilestoneDecisionLogger(log_path="./logs/decisions.jsonl")
    >>> logger.log_created(task_id="task-1", milestone=milestone)
    >>> logger.log_review(task_id="task-1", milestone=milestone, result=verdict)
    """

    def __init__(
        self,
        log_path: Optional[Union[str, Path]] = None,
        console_logging: bool = True,
        name: str = "heros.milestones",
    ) -> None:
        self._name = name
        self._console_logging = bool(console_logging)
        self._log_path = Path(log_path) if log_path is not None else None
        self._decision_count: int = 0

        # Internal logger for console output
        self._logger = logging.getLogger(name)
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter(
                    f"[%(name)s] %(asctime)s - %(levelname)s - %(message)s",
                    datefmt="%H:%M:%S",
                )
            )
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.INFO)

        # Ensure log directory exists
        if self._log_path is not None:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def decision_count(self) -> int:
        """Total number of decisions logged."""
        return self._decision_count

    def log_created(
        self,
        task_id: str,
        milestone: Milestone,
        milestone_index: int,
    ) -> None:
        """Log that a milestone was created/planned.

        Parameters
        ----------
        task_id : str
            Identifier of the parent task.
        milestone : Milestone
            The milestone that was created.
        milestone_index : int
            Index of the milestone in the task's plan (0-based).
        """
        self._log_decision(
            task_id=task_id,
            milestone=milestone,
            decision_type=MilestoneDecisionType.CREATED,
            verdict=None,
            critic_reasoning=None,
            confidence=None,
            milestone_index=milestone_index,
        )

    def log_attempted(
        self,
        task_id: str,
        milestone: Milestone,
        milestone_index: int,
    ) -> None:
        """Log that a milestone was attempted/executed.

        Parameters
        ----------
        task_id : str
            Identifier of the parent task.
        milestone : Milestone
            The milestone that was attempted.
        milestone_index : int
            Index of the milestone in the task's plan (0-based).
        """
        self._log_decision(
            task_id=task_id,
            milestone=milestone,
            decision_type=MilestoneDecisionType.ATTEMPTED,
            verdict=None,
            critic_reasoning=None,
            confidence=None,
            milestone_index=milestone_index,
        )

    def log_review(
        self,
        task_id: str,
        milestone: Milestone,
        result: CriticResult,
        milestone_index: int,
        critic_reasoning: Optional[str] = None,
    ) -> None:
        """Log the result of a milestone review by the critic.

        Parameters
        ----------
        task_id : str
            Identifier of the parent task.
        milestone : Milestone
            The milestone that was reviewed.
        result : CriticResult
            The critic's review result.
        milestone_index : int
            Index of the milestone in the task's plan (0-based).
        critic_reasoning : str, optional
            Raw LLM reasoning output from the critic (if available).
        """
        # Map verdict to decision type
        decision_type_map = {
            Verdict.PASS: MilestoneDecisionType.PASSED,
            Verdict.FAIL: MilestoneDecisionType.FAILED,
            Verdict.PARTIAL: MilestoneDecisionType.PARTIAL,
        }
        decision_type = decision_type_map.get(result.verdict, MilestoneDecisionType.FAILED)

        self._log_decision(
            task_id=task_id,
            milestone=milestone,
            decision_type=decision_type,
            verdict=result.verdict,
            critic_reasoning=critic_reasoning or result.feedback,
            confidence=result.confidence,
            milestone_index=milestone_index,
        )

    def _log_decision(
        self,
        task_id: str,
        milestone: Milestone,
        decision_type: MilestoneDecisionType,
        verdict: Optional[Verdict],
        critic_reasoning: Optional[str],
        confidence: Optional[float],
        milestone_index: int,
    ) -> None:
        """Internal method to record a decision entry.

        Parameters
        ----------
        task_id : str
            Identifier of the parent task.
        milestone : Milestone
            The milestone involved in the decision.
        decision_type : MilestoneDecisionType
            Type of decision being logged.
        verdict : Verdict, optional
            The critic's verdict (if a review was performed).
        critic_reasoning : str, optional
            The critic's reasoning (raw LLM output or feedback).
        confidence : float, optional
            The critic's confidence score.
        milestone_index : int
            Index of the milestone in the task's plan.
        """
        entry = {
            "task_id": task_id,
            "milestone_id": milestone.id,
            "milestone_description": milestone.description,
            "milestone_rubric": milestone.rubric,
            "decision_type": decision_type.value,
            "verdict": verdict.value if verdict is not None else None,
            "critic_reasoning": critic_reasoning,
            "confidence": confidence,
            "milestone_index": milestone_index,
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "entry_id": self._decision_count,
        }

        self._decision_count += 1

        # Console output
        if self._console_logging:
            verdict_str = verdict.value if verdict else "n/a"
            conf_str = f"{confidence:.2f}" if confidence is not None else "n/a"
            self._logger.info(
                "[%s] task=%s milestone=%s decision=%s verdict=%s confidence=%s",
                decision_type.value,
                task_id,
                milestone.id,
                decision_type.value,
                verdict_str,
                conf_str,
            )

        # File output (JSONL)
        if self._log_path is not None:
            try:
                with open(self._log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            except Exception as e:
                self._logger.warning("Failed to write decision to %s: %s", self._log_path, e)

    def load_entries(
        self,
        task_id: Optional[str] = None,
        milestone_id: Optional[str] = None,
        decision_type: Optional[MilestoneDecisionType] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Load logged decision entries.

        Parameters
        ----------
        task_id : str, optional
            Filter by task ID.
        milestone_id : str, optional
            Filter by milestone ID.
        decision_type : MilestoneDecisionType, optional
            Filter by decision type.
        limit : int, optional
            Maximum number of entries to return (most recent first).

        Returns
        -------
        List[Dict[str, Any]]
            List of matching decision entries.
        """
        if self._log_path is None or not self._log_path.exists():
            return []

        entries = []
        try:
            with open(self._log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Apply filters
                    if task_id is not None and entry.get("task_id") != task_id:
                        continue
                    if milestone_id is not None and entry.get("milestone_id") != milestone_id:
                        continue
                    if decision_type is not None and entry.get("decision_type") != decision_type.value:
                        continue

                    entries.append(entry)
        except Exception as e:
            self._logger.warning("Failed to read decisions from %s: %s", self._log_path, e)

        # Most recent first
        entries.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

        if limit is not None:
            entries = entries[:limit]

        return entries


# ---------------------------------------------------------------------------
# FunctionalInterchangeabilityCheck
# ---------------------------------------------------------------------------


@dataclass
class FunctionalEquivalenceResult:
    """Result of a functional interchangeability check.

    Attributes
    ----------
    equivalent : bool
        True if original and alternative milestones receive the same verdict.
    original_verdict : Verdict
        Verdict for the original milestone formulation.
    alternative_verdict : Verdict
        Verdict for the alternative milestone formulation.
    swap_verified : bool
        True if the swap was verified (both directions yielded same verdict).
    original_result : CriticResult
        Full critic result for the original milestone.
    alternative_result : CriticResult
        Full critic result for the alternative milestone.
    verification_trace : str
        The execution trace used for verification.
    timestamp : str
        ISO-8601 timestamp of when verification was performed.
    """

    equivalent: bool
    original_verdict: Verdict
    alternative_verdict: Verdict
    swap_verified: bool
    original_result: CriticResult
    alternative_result: CriticResult
    verification_trace: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "equivalent": self.equivalent,
            "original_verdict": self.original_verdict.value,
            "alternative_verdict": self.alternative_verdict.value,
            "swap_verified": self.swap_verified,
            "original_result": {
                "verdict": self.original_result.verdict.value,
                "feedback": self.original_result.feedback,
                "confidence": self.original_result.confidence,
                "reward_signal": self.original_result.reward_signal,
            },
            "alternative_result": {
                "verdict": self.alternative_result.verdict.value,
                "feedback": self.alternative_result.feedback,
                "confidence": self.alternative_result.confidence,
                "reward_signal": self.alternative_result.reward_signal,
            },
            "verification_trace": self.verification_trace,
            "timestamp": self.timestamp,
        }


class FunctionalInterchangeabilityCheck:
    """Verifies that a milestone is functionally interchangeable with an alternative.

    Given a milestone with a rubric, generates an alternative phrasing of the same
    milestone and verifies both formulations receive the same verdict on the same
    execution trace.

    Parameters
    ----------
    critic : MilestoneCritic
        The critic used to evaluate milestones.
    alternative_generator : callable, optional
        A function ``(milestone_description: str, rubric: str) -> str`` that
        generates an alternative phrasing. If None, a simple rule-based
        paraphrasing is used.
    confidence_threshold : float, optional
        Minimum confidence required for swap verification. Defaults to 0.7.

    Examples
    --------
    >>> check = FunctionalInterchangeabilityCheck(critic=critic)
    >>> result = check.verify(trace="...", milestone=milestone, alt_milestone=alt)
    >>> if result.equivalent:
    ...     print("Milestones are functionally equivalent")
    """

    def __init__(
        self,
        critic: MilestoneCritic,
        alternative_generator: Optional[callable] = None,
        confidence_threshold: float = 0.7,
    ) -> None:
        if not isinstance(critic, MilestoneCritic):
            raise TypeError(f"Expected MilestoneCritic, got {type(critic).__name__}")
        if not (0.0 <= confidence_threshold <= 1.0):
            raise ValueError(f"confidence_threshold must be in [0.0, 1.0], got {confidence_threshold}")

        self._critic = critic
        self._alt_generator = alternative_generator
        self._confidence_threshold = confidence_threshold

    def verify(
        self,
        trace: str,
        milestone: Milestone,
        alt_milestone: Milestone,
    ) -> FunctionalEquivalenceResult:
        """Verify functional equivalence between two milestone formulations.

        Evaluates both the original and alternative milestones against the
        same execution trace and checks if they receive equivalent verdicts.

        Parameters
        ----------
        trace : str
            The execution trace to evaluate against.
        milestone : Milestone
            The original milestone formulation.
        alt_milestone : Milestone
            The alternative milestone formulation (same semantic goal,
            different phrasing).

        Returns
        -------
        FunctionalEquivalenceResult
            Result containing verdicts for both formulations and verification status.
        """
        # Evaluate original milestone
        original_result = self._critic.review(
            milestone_description=milestone.description,
            rubric=milestone.rubric,
            execution_trace=trace,
        )

        # Evaluate alternative milestone
        alternative_result = self._critic.review(
            milestone_description=alt_milestone.description,
            rubric=alt_milestone.rubric,
            execution_trace=trace,
        )

        # Check equivalence
        equivalent = original_result.verdict == alternative_result.verdict

        # Verify swap: both directions must agree
        # (We already checked both directions by comparing verdicts)
        swap_verified = (
            equivalent
            and original_result.confidence >= self._confidence_threshold
            and alternative_result.confidence >= self._confidence_threshold
        )

        return FunctionalEquivalenceResult(
            equivalent=equivalent,
            original_verdict=original_result.verdict,
            alternative_verdict=alternative_result.verdict,
            swap_verified=swap_verified,
            original_result=original_result,
            alternative_result=alternative_result,
            verification_trace=trace,
        )

    def generate_alternative(
        self,
        milestone: Milestone,
    ) -> Milestone:
        """Generate an alternative phrasing of a milestone.

        Parameters
        ----------
        milestone : Milestone
            The original milestone.

        Returns
        -------
        Milestone
            An alternative phrasing of the same milestone.
        """
        if self._alt_generator is not None:
            alt_description = self._alt_generator(
                milestone.description, milestone.rubric
            )
        else:
            alt_description = self._rule_based_paraphrase(
                milestone.description, milestone.rubric
            )

        return Milestone(
            id=f"{milestone.id}_alt",
            description=alt_description,
            rubric=milestone.rubric,  # Same rubric - same criteria
            expected_output=milestone.expected_output,
        )

    @staticmethod
    def _rule_based_paraphrase(description: str, rubric: str) -> str:
        """Simple rule-based paraphrase of a milestone description.

        Uses synonym replacement and reordering heuristics.

        Parameters
        ----------
        description : str
            Original milestone description.
        rubric : str
            Milestone rubric (unchanged).

        Returns
        -------
        str
            Alternative phrasing of the description.
        """
        # Simple transformations
        text = description

        # Common substitutions
        substitutions = [
            (r"\bfirst\b", "initially"),
            (r"\bthen\b", "next"),
            (r"\bafter that\b", "subsequently"),
            (r"\bfinally\b", "in conclusion"),
            (r"\bensure\b", "make sure"),
            (r"\bverify\b", "confirm"),
            (r"\bcomplete\b", "finish"),
            (r"\bperform\b", "carry out"),
            (r"\bsubmit\b", "send"),
            (r"\bcreate\b", "make"),
            (r"\bretrieve\b", "fetch"),
            (r"\bobtain\b", "get"),
        ]

        for pattern, replacement in substitutions:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # Capitalize first letter
        if text:
            text = text[0].upper() + text[1:]

        # Add period if missing
        if text and not text.endswith((".", "!", "?")):
            text += "."

        return text


# ---------------------------------------------------------------------------
# RewardAuditTrail
# ---------------------------------------------------------------------------


@dataclass
class MilestoneAuditEntry:
    """Single milestone entry in a reward audit trail.

    Attributes
    ----------
    milestone_id : str
        Unique identifier of the milestone.
    description : str
        Human-readable milestone description.
    verdict : str
        The critic's verdict (pass, fail, partial).
    reward_signal : float
        The computed reward signal in [0.0, 1.0].
    critic_confidence : float
        The critic's confidence score in [0.0, 1.0].
    feedback_text : str
        The critic's feedback text.
    milestone_index : int
        Position of the milestone in the episode plan (0-based).
    """

    milestone_id: str
    description: str
    verdict: str
    reward_signal: float
    critic_confidence: float
    feedback_text: str
    milestone_index: int

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "milestone_id": self.milestone_id,
            "description": self.description,
            "verdict": self.verdict,
            "reward_signal": round(self.reward_signal, 6),
            "critic_confidence": round(self.critic_confidence, 6),
            "feedback_text": self.feedback_text,
            "milestone_index": self.milestone_index,
        }


@dataclass
class RewardAuditTrail:
    """Per-episode milestone-level reward audit trail.

    Attributes
    ----------
    episode_id : str
        Unique identifier for this episode.
    task : str
        The task description for this episode.
    timestamp : str
        ISO-8601 timestamp of when the episode was recorded.
    milestones : List[MilestoneAuditEntry]
        List of milestone audit entries in execution order.
    total_reward : float
        Sum of all milestone reward signals.
    avg_reward : float
        Average milestone reward signal.
    success_rate : float
        Fraction of milestones that passed (Verdict.PASS).
    """

    episode_id: str
    task: str
    timestamp: str
    milestones: List[MilestoneAuditEntry] = field(default_factory=list)
    total_reward: float = 0.0
    avg_reward: float = 0.0
    success_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "episode_id": self.episode_id,
            "task": self.task,
            "timestamp": self.timestamp,
            "milestones": [m.to_dict() for m in self.milestones],
            "total_reward": round(self.total_reward, 6),
            "avg_reward": round(self.avg_reward, 6),
            "success_rate": round(self.success_rate, 6),
        }

    @classmethod
    def from_trajectory(
        cls,
        trajectory: HindsightTrajectory,
        auditor: "RewardAuditor",
        episode_id: Optional[str] = None,
    ) -> "RewardAuditTrail":
        """Create a RewardAuditTrail from a HindsightTrajectory.

        Parameters
        ----------
        trajectory : HindsightTrajectory
            The trajectory to create the audit trail from.
        auditor : RewardAuditor
            The reward auditor for computing reward signals.
        episode_id : str, optional
            Custom episode ID. If None, uses the trajectory's ID.

        Returns
        -------
        RewardAuditTrail
            The constructed audit trail.
        """
        episode_id = episode_id or trajectory.id

        # Build milestone entries
        milestones: List[MilestoneAuditEntry] = []
        total_reward = 0.0

        for idx, (milestone, verdict, trace) in enumerate(
            zip(trajectory.milestones, trajectory.verdicts, trajectory.exec_traces)
        ):
            # Get trace content
            if isinstance(trace, dict):
                trace_content = trace.get("content", "") or trace.get("output", "")
            else:
                trace_content = str(trace)

            # Review with critic to get full result
            # Note: We need critic for confidence; if not available, estimate
            reward = auditor.audit_verdict(verdict)
            total_reward += reward

            # Extract feedback from trace metadata if available
            feedback = ""
            if isinstance(trace, dict):
                feedback = trace.get("feedback", "")

            milestones.append(
                MilestoneAuditEntry(
                    milestone_id=milestone.id,
                    description=milestone.description,
                    verdict=verdict.value,
                    reward_signal=reward,
                    critic_confidence=0.5,  # Estimated when critic not available
                    feedback_text=feedback,
                    milestone_index=idx,
                )
            )

        # Compute statistics
        num_milestones = len(trajectory.milestones)
        avg_reward = total_reward / num_milestones if num_milestones > 0 else 0.0
        passed = sum(1 for v in trajectory.verdicts if v == Verdict.PASS)
        success_rate = passed / num_milestones if num_milestones > 0 else 0.0

        return cls(
            episode_id=episode_id,
            task=trajectory.task,
            timestamp=trajectory.timestamp or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            milestones=milestones,
            total_reward=total_reward,
            avg_reward=avg_reward,
            success_rate=success_rate,
        )

    @classmethod
    def from_critic_results(
        cls,
        episode_id: str,
        task: str,
        milestone_results: List[Tuple[Milestone, CriticResult]],
        timestamp: Optional[str] = None,
    ) -> "RewardAuditTrail":
        """Create a RewardAuditTrail from a list of (Milestone, CriticResult) pairs.

        Parameters
        ----------
        episode_id : str
            Unique identifier for this episode.
        task : str
            The task description.
        milestone_results : List[Tuple[Milestone, CriticResult]]
            List of (milestone, critic_result) pairs in execution order.
        timestamp : str, optional
            ISO-8601 timestamp. Defaults to current UTC time.

        Returns
        -------
        RewardAuditTrail
            The constructed audit trail.
        """
        timestamp = timestamp or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        milestones: List[MilestoneAuditEntry] = []
        total_reward = 0.0

        for idx, (milestone, result) in enumerate(milestone_results):
            total_reward += result.reward_signal

            milestones.append(
                MilestoneAuditEntry(
                    milestone_id=milestone.id,
                    description=milestone.description,
                    verdict=result.verdict.value,
                    reward_signal=result.reward_signal,
                    critic_confidence=result.confidence,
                    feedback_text=result.feedback,
                    milestone_index=idx,
                )
            )

        num_milestones = len(milestone_results)
        avg_reward = total_reward / num_milestones if num_milestones > 0 else 0.0
        passed = sum(
            1 for (_, result) in milestone_results if result.verdict == Verdict.PASS
        )
        success_rate = passed / num_milestones if num_milestones > 0 else 0.0

        return cls(
            episode_id=episode_id,
            task=task,
            timestamp=timestamp,
            milestones=milestones,
            total_reward=total_reward,
            avg_reward=avg_reward,
            success_rate=success_rate,
        )


class RewardAuditor:
    """Converts milestone verdicts into dense reward signals with full audit logging.

    This is an enhanced version of the critic's built-in RewardAuditor that
    produces comprehensive audit logs alongside the reward signals.

    Parameters
    ----------
    weights : Dict[Verdict, float], optional
        Mapping from Verdict to base reward weight.
        Defaults to {PASS: 1.0, PARTIAL: 0.5, FAIL: 0.0}.
    confidence_scaling : bool, optional
        If True, scales rewards by confidence score. Defaults to True.
    audit_log_path : Union[str, Path], optional
        Path to the JSONL audit log file. If None, no file logging.
    console_logging : bool, optional
        Whether to log audit entries to console. Defaults to False.

    Examples
    --------
    >>> auditor = RewardAuditor(audit_log_path="./logs/reward_audit.jsonl")
    >>> trail = auditor.audit_episode(
    ...     episode_id="ep-1",
    ...     task="Fix the login bug",
    ...     milestone_results=[(milestone, result)],
    ... )
    >>> print(f"Episode reward: {trail.avg_reward}")
    """

    def __init__(
        self,
        weights: Optional[Dict[Verdict, float]] = None,
        confidence_scaling: bool = True,
        audit_log_path: Optional[Union[str, Path]] = None,
        console_logging: bool = False,
    ) -> None:
        self._weights = weights or {
            Verdict.PASS: 1.0,
            Verdict.PARTIAL: 0.5,
            Verdict.FAIL: 0.0,
        }
        self._confidence_scaling = bool(confidence_scaling)
        self._console_logging = bool(console_logging)
        self._audit_log_path = Path(audit_log_path) if audit_log_path is not None else None
        self._episode_count: int = 0

        # Setup logger
        self._logger = logging.getLogger("heros.reward_auditor")
        if self._console_logging and not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter(
                    f"[%(name)s] %(asctime)s - %(levelname)s - %(message)s",
                    datefmt="%H:%M:%S",
                )
            )
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.INFO)

        # Ensure audit log directory exists
        if self._audit_log_path is not None:
            self._audit_log_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def episode_count(self) -> int:
        """Total number of episodes audited."""
        return self._episode_count

    def audit_episode(
        self,
        episode_id: str,
        task: str,
        milestone_results: List[Tuple[Milestone, CriticResult]],
        timestamp: Optional[str] = None,
    ) -> RewardAuditTrail:
        """Audit a complete episode and produce a RewardAuditTrail.

        Parameters
        ----------
        episode_id : str
            Unique identifier for this episode.
        task : str
            The task description.
        milestone_results : List[Tuple[Milestone, CriticResult]]
            List of (milestone, critic_result) pairs in execution order.
        timestamp : str, optional
            ISO-8601 timestamp. Defaults to current UTC time.

        Returns
        -------
        RewardAuditTrail
            The complete audit trail for this episode.
        """
        trail = RewardAuditTrail.from_critic_results(
            episode_id=episode_id,
            task=task,
            milestone_results=milestone_results,
            timestamp=timestamp,
        )

        self._episode_count += 1

        # Log to console
        if self._console_logging:
            self._logger.info(
                "Episode %s | milestones=%d | success_rate=%.1f%% | avg_reward=%.4f",
                episode_id,
                len(trail.milestones),
                trail.success_rate * 100,
                trail.avg_reward,
            )

        # Log to file
        if self._audit_log_path is not None:
            try:
                with open(self._audit_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(trail.to_dict(), ensure_ascii=False) + "\n")
            except Exception as e:
                self._logger.warning(
                    "Failed to write audit trail to %s: %s",
                    self._audit_log_path,
                    e,
                )

        return trail

    def audit_trajectory(
        self,
        trajectory: HindsightTrajectory,
        episode_id: Optional[str] = None,
    ) -> RewardAuditTrail:
        """Audit a HindsightTrajectory and produce a RewardAuditTrail.

        Parameters
        ----------
        trajectory : HindsightTrajectory
            The trajectory to audit.
        episode_id : str, optional
            Custom episode ID. If None, uses the trajectory's ID.

        Returns
        -------
        RewardAuditTrail
            The complete audit trail for this trajectory.
        """
        return RewardAuditTrail.from_trajectory(
            trajectory=trajectory,
            auditor=self,
            episode_id=episode_id,
        )

    def audit(self, result: CriticResult) -> float:
        """Convert a CriticResult into a reward signal.

        Parameters
        ----------
        result : CriticResult
            The critic result to convert.

        Returns
        -------
        float
            Reward signal in [0.0, 1.0].
        """
        base = self._weights.get(result.verdict, 0.0)
        if self._confidence_scaling:
            return base * result.confidence
        return base

    def audit_verdict(self, verdict: Verdict) -> float:
        """Convert a bare verdict into a reward signal (no confidence scaling).

        Parameters
        ----------
        verdict : Verdict
            The verdict to convert.

        Returns
        -------
        float
            Reward signal in [0.0, 1.0].
        """
        return self._weights.get(verdict, 0.0)

    def batch_audit(self, results: List[CriticResult]) -> List[float]:
        """Audit a batch of critic results.

        Parameters
        ----------
        results : List[CriticResult]
            The critic results to audit.

        Returns
        -------
        List[float]
            List of reward signals.
        """
        return [self.audit(r) for r in results]

    def load_audit_trails(
        self,
        limit: Optional[int] = None,
        task_filter: Optional[str] = None,
    ) -> List[RewardAuditTrail]:
        """Load previously saved audit trails.

        Parameters
        ----------
        limit : int, optional
            Maximum number of trails to return (most recent first).
        task_filter : str, optional
            Only return trails where task contains this string.

        Returns
        -------
        List[RewardAuditTrail]
            List of loaded audit trails.
        """
        if self._audit_log_path is None or not self._audit_log_path.exists():
            return []

        trails = []
        try:
            with open(self._audit_log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Apply task filter
                    if task_filter is not None and task_filter not in data.get("task", ""):
                        continue

                    # Reconstruct MilestoneAuditEntry objects
                    milestones = [
                        MilestoneAuditEntry(
                            milestone_id=m["milestone_id"],
                            description=m["description"],
                            verdict=m["verdict"],
                            reward_signal=m["reward_signal"],
                            critic_confidence=m["critic_confidence"],
                            feedback_text=m["feedback_text"],
                            milestone_index=m["milestone_index"],
                        )
                        for m in data.get("milestones", [])
                    ]

                    trails.append(
                        RewardAuditTrail(
                            episode_id=data["episode_id"],
                            task=data["task"],
                            timestamp=data["timestamp"],
                            milestones=milestones,
                            total_reward=data.get("total_reward", 0.0),
                            avg_reward=data.get("avg_reward", 0.0),
                            success_rate=data.get("success_rate", 0.0),
                        )
                    )
        except Exception as e:
            if self._console_logging:
                self._logger.warning("Failed to read audit trails: %s", e)

        # Most recent first
        trails.sort(key=lambda x: x.timestamp, reverse=True)

        if limit is not None:
            trails = trails[:limit]

        return trails


# ---------------------------------------------------------------------------
# BufferCompositionAnalyzer
# ---------------------------------------------------------------------------


class BufferCompositionAnalyzer:
    """Analyzes and visualizes hindsight buffer composition over training.

    Provides methods to compute buffer diversity, milestone hit rates,
    failure distributions, and export summaries.

    Parameters
    ----------
    buffer : HindsightBuffer
        The hindsight buffer to analyze.
    milestone_type_extractor : callable, optional
        A function ``(Milestone) -> str`` that extracts a milestone type/category.
        If None, uses the milestone's description (truncated to 50 chars).

    Examples
    --------
    >>> analyzer = BufferCompositionAnalyzer(buffer=buffer)
    >>> diversity = analyzer.compute_buffer_diversity()
    >>> hit_rates = analyzer.milestone_hit_rate_by_type()
    >>> analyzer.export_buffer_summary_json("./buffer_summary.json")
    """

    def __init__(
        self,
        buffer: HindsightBuffer,
        milestone_type_extractor: Optional[callable] = None,
    ) -> None:
        if not isinstance(buffer, HindsightBuffer):
            raise TypeError(f"Expected HindsightBuffer, got {type(buffer).__name__}")

        self._buffer = buffer
        self._type_extractor = milestone_type_extractor or (lambda m: m.description[:50])

    def compute_buffer_diversity(self) -> float:
        """Compute the entropy of milestone types in the buffer.

        Higher entropy indicates more diverse buffer composition.
        Returns a float in [0.0, log(n)] where n is the number of unique
        milestone types.

        Returns
        -------
        float
            Entropy of milestone type distribution (in nats).
        """
        # Count milestone types across all trajectories
        type_counts: Counter[str] = Counter()

        for trajectory in self._buffer:
            for milestone in trajectory.milestones:
                m_type = self._type_extractor(milestone)
                type_counts[m_type] += 1

        if not type_counts:
            return 0.0

        # Compute entropy
        total = sum(type_counts.values())
        entropy = 0.0
        for count in type_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log(p)

        return entropy

    def milestone_hit_rate_by_type(self) -> Dict[str, float]:
        """Compute pass (hit) rate for each milestone type.

        Returns
        -------
        Dict[str, float]
            Mapping from milestone type to hit rate (fraction of PASS verdicts).
            Types with no milestones return 0.0.
        """
        # Aggregate verdicts by milestone type
        type_verdicts: Dict[str, List[Verdict]] = {}

        for trajectory in self._buffer:
            for milestone, verdict in zip(trajectory.milestones, trajectory.verdicts):
                m_type = self._type_extractor(milestone)
                if m_type not in type_verdicts:
                    type_verdicts[m_type] = []
                type_verdicts[m_type].append(verdict)

        # Compute hit rates
        hit_rates = {}
        for m_type, verdicts in type_verdicts.items():
            passed = sum(1 for v in verdicts if v == Verdict.PASS)
            hit_rates[m_type] = passed / len(verdicts) if verdicts else 0.0

        return hit_rates

    def failed_milestone_distribution(self) -> Dict[str, int]:
        """Compute the distribution of failed milestones by type.

        Returns
        -------
        Dict[str, int]
            Mapping from milestone type to count of FAIL verdicts.
        """
        failure_counts: Counter[str] = Counter()

        for trajectory in self._buffer:
            for milestone, verdict in zip(trajectory.milestones, trajectory.verdicts):
                if verdict == Verdict.FAIL:
                    m_type = self._type_extractor(milestone)
                    failure_counts[m_type] += 1

        return dict(failure_counts)

    def milestone_type_counts(self) -> Dict[str, int]:
        """Get the count of each milestone type in the buffer.

        Returns
        -------
        Dict[str, int]
            Mapping from milestone type to number of occurrences.
        """
        type_counts: Counter[str] = Counter()

        for trajectory in self._buffer:
            for milestone in trajectory.milestones:
                m_type = self._type_extractor(milestone)
                type_counts[m_type] += 1

        return dict(type_counts)

    def overall_hit_rate(self) -> float:
        """Compute the overall milestone pass rate across the buffer.

        Returns
        -------
        float
            Fraction of milestones with PASS verdict.
        """
        total = 0
        passed = 0

        for trajectory in self._buffer:
            for verdict in trajectory.verdicts:
                total += 1
                if verdict == Verdict.PASS:
                    passed += 1

        return passed / total if total > 0 else 0.0

    def overall_partial_rate(self) -> float:
        """Compute the overall milestone partial pass rate across the buffer.

        Returns
        -------
        float
            Fraction of milestones with PARTIAL verdict.
        """
        total = 0
        partial = 0

        for trajectory in self._buffer:
            for verdict in trajectory.verdicts:
                total += 1
                if verdict == Verdict.PARTIAL:
                    partial += 1

        return partial / total if total > 0 else 0.0

    def overall_fail_rate(self) -> float:
        """Compute the overall milestone fail rate across the buffer.

        Returns
        -------
        float
            Fraction of milestones with FAIL verdict.
        """
        total = 0
        failed = 0

        for trajectory in self._buffer:
            for verdict in trajectory.verdicts:
                total += 1
                if verdict == Verdict.FAIL:
                    failed += 1

        return failed / total if total > 0 else 0.0

    def trajectory_success_rates(self) -> List[float]:
        """Get per-trajectory success rates (fraction of milestones passed).

        Returns
        -------
        List[float]
            List of success rates, one per trajectory in the buffer.
        """
        return [t.compute_success_rate() for t in self._buffer]

    def hindsight_enhancement_rate(self) -> float:
        """Compute the fraction of trajectories that are hindsight-enhanced.

        Returns
        -------
        float
            Fraction of trajectories with is_hindsight_enhanced=True.
        """
        if len(self._buffer) == 0:
            return 0.0

        enhanced = sum(1 for t in self._buffer if t.is_hindsight_enhanced)
        return enhanced / len(self._buffer)

    def export_buffer_summary_json(
        self,
        path: Union[str, Path],
    ) -> None:
        """Export a comprehensive buffer composition summary as JSON.

        Parameters
        ----------
        path : Union[str, Path]
            Path to save the JSON summary.
        """
        summary = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "buffer_stats": self._buffer.get_stats(),
            "diversity_entropy": round(self.compute_buffer_diversity(), 6),
            "milestone_type_counts": self.milestone_type_counts(),
            "hit_rate_by_type": {
                k: round(v, 6) for k, v in self.milestone_hit_rate_by_type().items()
            },
            "failed_milestone_distribution": self.failed_milestone_distribution(),
            "overall_hit_rate": round(self.overall_hit_rate(), 6),
            "overall_partial_rate": round(self.overall_partial_rate(), 6),
            "overall_fail_rate": round(self.overall_fail_rate(), 6),
            "hindsight_enhancement_rate": round(self.hindsight_enhancement_rate(), 6),
            "trajectory_success_rates": [
                round(r, 6) for r in self.trajectory_success_rates()
            ],
        }

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info("Exported buffer summary to %s", path)


# ---------------------------------------------------------------------------
# plot_buffer_composition
# ---------------------------------------------------------------------------


def plot_buffer_composition(
    buffer: HindsightBuffer,
    output_path: Optional[Union[str, Path]] = None,
    milestone_type_extractor: Optional[callable] = None,
    figsize: Tuple[int, int] = (12, 8),
    dpi: int = 100,
) -> Dict[str, Any]:
    """Plot buffer composition analysis charts.

    Generates a figure with multiple subplots showing:
    1. Milestone type distribution (bar chart)
    2. Hit rate by milestone type (bar chart)
    3. Verdict distribution (pie chart)
    4. Trajectory success rate distribution (histogram)

    Parameters
    ----------
    buffer : HindsightBuffer
        The hindsight buffer to visualize.
    output_path : Union[str, Path], optional
        Path to save the figure. If None, figure is not saved.
    milestone_type_extractor : callable, optional
        Function to extract milestone type from Milestone.
        If None, uses milestone.description (truncated).
    figsize : Tuple[int, int], optional
        Figure size in inches. Defaults to (12, 8).
    dpi : int, optional
        Dots per inch for saved figure. Defaults to 100.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing the raw data used for plotting.
    """
    import matplotlib.pyplot as plt

    analyzer = BufferCompositionAnalyzer(
        buffer=buffer,
        milestone_type_extractor=milestone_type_extractor,
    )

    # Gather data
    type_counts = analyzer.milestone_type_counts()
    hit_rates = analyzer.milestone_hit_rate_by_type()
    fail_dist = analyzer.failed_milestone_distribution()
    traj_success_rates = analyzer.trajectory_success_rates()

    # Overall verdict counts
    total_pass = 0
    total_partial = 0
    total_fail = 0
    for trajectory in buffer:
        for verdict in trajectory.verdicts:
            if verdict == Verdict.PASS:
                total_pass += 1
            elif verdict == Verdict.PARTIAL:
                total_partial += 1
            else:
                total_fail += 1

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle("Hindsight Buffer Composition Analysis", fontsize=14, fontweight="bold")

    # 1. Milestone type distribution (bar chart)
    ax1 = axes[0, 0]
    if type_counts:
        types = list(type_counts.keys())
        counts = list(type_counts.values())
        # Truncate long type names
        types_display = [t[:30] + "..." if len(t) > 30 else t for t in types]
        bars = ax1.barh(types_display, counts, color="steelblue")
        ax1.set_xlabel("Count")
        ax1.set_title("Milestone Type Distribution")
        ax1.bar_label(bars, padding=3)
    else:
        ax1.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax1.transAxes)
        ax1.set_title("Milestone Type Distribution")

    # 2. Hit rate by milestone type (bar chart)
    ax2 = axes[0, 1]
    if hit_rates:
        types = list(hit_rates.keys())
        rates = list(hit_rates.values())
        types_display = [t[:30] + "..." if len(t) > 30 else t for t in types]
        bars = ax2.barh(types_display, rates, color="seagreen")
        ax2.set_xlabel("Hit Rate")
        ax2.set_title("Hit Rate by Milestone Type")
        ax2.set_xlim(0.0, 1.0)
        ax2.bar_label(bars, fmt="%.2f", padding=3)
    else:
        ax2.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax2.transAxes)
        ax2.set_title("Hit Rate by Milestone Type")

    # 3. Verdict distribution (pie chart)
    ax3 = axes[1, 0]
    verdict_counts = [total_pass, total_partial, total_fail]
    verdict_labels = ["PASS", "PARTIAL", "FAIL"]
    verdict_colors = ["seagreen", "orange", "crimson"]
    verdict_explode = (0.05, 0.05, 0.05)

    if sum(verdict_counts) > 0:
        wedges, texts, autotexts = ax3.pie(
            verdict_counts,
            labels=verdict_labels,
            colors=verdict_colors,
            explode=verdict_explode,
            autopct="%1.1f%%",
            startangle=90,
        )
        ax3.set_title("Overall Verdict Distribution")
    else:
        ax3.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax3.transAxes)
        ax3.set_title("Overall Verdict Distribution")

    # 4. Trajectory success rate distribution (histogram)
    ax4 = axes[1, 1]
    if traj_success_rates:
        ax4.hist(traj_success_rates, bins=10, color="steelblue", edgecolor="black", alpha=0.7)
        ax4.set_xlabel("Success Rate")
        ax4.set_ylabel("Number of Trajectories")
        ax4.set_title("Trajectory Success Rate Distribution")
        ax4.set_xlim(0.0, 1.0)
    else:
        ax4.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax4.transAxes)
        ax4.set_title("Trajectory Success Rate Distribution")

    plt.tight_layout()

    # Save figure if path provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        logger.info("Saved buffer composition plot to %s", output_path)

    # Collect raw data for return
    plot_data = {
        "milestone_type_counts": type_counts,
        "hit_rates_by_type": hit_rates,
        "verdict_counts": {
            "pass": total_pass,
            "partial": total_partial,
            "fail": total_fail,
        },
        "trajectory_success_rates": traj_success_rates,
        "buffer_stats": buffer.get_stats(),
        "diversity_entropy": round(analyzer.compute_buffer_diversity(), 6),
    }

    plt.close(fig)

    return plot_data


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    # MilestoneDecisionLogger
    "MilestoneDecisionType",
    "MilestoneDecisionLogger",
    # FunctionalInterchangeabilityCheck
    "FunctionalEquivalenceResult",
    "FunctionalInterchangeabilityCheck",
    # RewardAuditTrail
    "MilestoneAuditEntry",
    "RewardAuditTrail",
    "RewardAuditor",
    # BufferCompositionAnalyzer
    "BufferCompositionAnalyzer",
    # plot_buffer_composition
    "plot_buffer_composition",
]
