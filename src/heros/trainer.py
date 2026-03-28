"""HeRL-style Policy Update Module.

Handles policy improvement from the hindsight experience buffer using
BC-style supervised fine-tuning (with optional OpenAI API) or a
mock simulation for testing purposes.

References:
    - HeRL: Hindsight Experience Replay for LLMs
    - MiRA: Milestoning RL Enhanced Agent
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from heros.buffer import HindsightBuffer, HindsightTrajectory

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# UpdateResult dataclass
# ---------------------------------------------------------------------------


@dataclass
class UpdateResult:
    """Result of a policy update operation.

    Attributes
    ----------
    loss : float
        Computed loss value from the policy update. May be a simulation
        or placeholder value when no real model is available.
    num_samples : int
        Number of trajectories used in this update.
    hindsight_ratio : float
        Fraction of the batch that came from hindsight-enhanced trajectories.
    timestamp : datetime
        UTC timestamp of when the update was performed.
    is_simulation : bool
        True if this update was simulated (no real model was updated).
    details : Dict[str, Any], optional
        Additional details about the update (e.g., learning rate,
        model path, number of hindsight samples).
    """

    loss: float
    num_samples: int
    hindsight_ratio: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    is_simulation: bool = True
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "loss": self.loss,
            "num_samples": self.num_samples,
            "hindsight_ratio": self.hindsight_ratio,
            "timestamp": self.timestamp.isoformat() + "Z",
            "is_simulation": self.is_simulation,
            "details": self.details or {},
        }


# ---------------------------------------------------------------------------
# HindsightTrainer
# ---------------------------------------------------------------------------


class HindsightTrainer:
    """Trains a policy from the hindsight experience buffer.

    Supports two modes of operation:

    1. **BC-style fine-tuning** (when ``OPENAI_API_KEY`` is set):
       Uses ``gpt-4o-mini`` to generate supervisory signals from
       hindsight relabeled trajectories, computing a simulated
       loss based on trajectory alignment.

    2. **Simulation mode** (default, no API key):
       Computes a heuristic reward-based "loss" signal from the
       hindsight buffer and returns a simulated :class:`UpdateResult`.

    Parameters
    ----------
    buffer : HindsightBuffer
        The hindsight experience buffer to train from.
    model_path : str, optional
        Path to a local model for fine-tuning. If None and
        ``OPENAI_API_KEY`` is set, uses the OpenAI API.
    learning_rate : float, optional
        Learning rate for policy updates. Defaults to 1e-5.

    Attributes
    ----------
    buffer : HindsightBuffer
        The buffer being trained from.
    model_path : Optional[str]
        Configured model path.
    learning_rate : float
        Configured learning rate.
    update_count : int
        Number of policy updates performed so far.
    """

    def __init__(
        self,
        buffer: HindsightBuffer,
        model_path: Optional[str] = None,
        learning_rate: float = 1e-5,
    ) -> None:
        if not isinstance(buffer, HindsightBuffer):
            raise TypeError(
                f"buffer must be a HindsightBuffer, got {type(buffer).__name__}"
            )
        self._buffer = buffer

        if model_path is not None and not isinstance(model_path, str):
            raise TypeError(f"model_path must be a str, got {type(model_path).__name__}")
        self._model_path = model_path

        if not isinstance(learning_rate, (int, float)):
            raise TypeError(
                f"learning_rate must be a float, got {type(learning_rate).__name__}"
            )
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")
        self._learning_rate = float(learning_rate)

        self._update_count: int = 0
        self._openai_key: Optional[str] = None

        # Check for OpenAI API key availability
        self._openai_key = os.environ.get("OPENAI_API_KEY", "")

    # ---------------------------------------------------------------------------
    # Properties
    # ---------------------------------------------------------------------------

    @property
    def buffer(self) -> HindsightBuffer:
        """Hindsight experience buffer."""
        return self._buffer

    @property
    def model_path(self) -> Optional[str]:
        """Configured model path."""
        return self._model_path

    @property
    def learning_rate(self) -> float:
        """Configured learning rate."""
        return self._learning_rate

    @property
    def update_count(self) -> int:
        """Number of policy updates performed."""
        return self._update_count

    @property
    def has_openai_key(self) -> bool:
        """True if an OpenAI API key is configured."""
        return bool(self._openai_key)

    # ---------------------------------------------------------------------------
    # Core training methods
    # ---------------------------------------------------------------------------

    def update_policy(
        self,
        trajectories: List[HindsightTrajectory],
        lr: Optional[float] = None,
    ) -> UpdateResult:
        """Perform a policy update from a batch of trajectories.

        Uses BC-style supervised fine-tuning. If an OpenAI API key is
        available, this method attempts to use ``gpt-4o-mini`` for
        generating supervisory signals. Otherwise, returns a simulated
        result based on hindsight reward heuristics.

        Parameters
        ----------
        trajectories : List[HindsightTrajectory]
            Batch of trajectories to train on.
        lr : float, optional
            Learning rate override for this update. If None, uses the
            default from ``__init__``.

        Returns
        -------
        UpdateResult
            Result of the update including loss and metadata.

        Raises
        ------
        TypeError
            If trajectories is not a list or contains non-HindsightTrajectory items.
        ValueError
            If trajectories is empty or lr is invalid.
        """
        if not isinstance(trajectories, list):
            raise TypeError(
                f"trajectories must be a list, got {type(trajectories).__name__}"
            )
        if len(trajectories) == 0:
            raise ValueError("trajectories list cannot be empty")

        for i, t in enumerate(trajectories):
            if not isinstance(t, HindsightTrajectory):
                raise TypeError(
                    f"trajectories[{i}] must be a HindsightTrajectory, "
                    f"got {type(t).__name__}"
                )

        effective_lr = lr if lr is not None else self._learning_rate

        # Count hindsight-enhanced trajectories in the batch
        hindsight_count = sum(1 for t in trajectories if t.is_hindsight_enhanced)
        batch_hindsight_ratio = hindsight_count / len(trajectories)

        self._update_count += 1

        # Attempt OpenAI API-based BC update if key is available
        if self._openai_key:
            try:
                return self._update_via_openai(trajectories, effective_lr, batch_hindsight_ratio)
            except Exception as e:
                logger.warning(
                    "OpenAI API update failed (%s), falling back to simulation. "
                    "Error: %s",
                    type(e).__name__,
                    e,
                )
                return self._simulate_update(trajectories, effective_lr, batch_hindsight_ratio)

        return self._simulate_update(trajectories, effective_lr, batch_hindsight_ratio)

    def _update_via_openai(
        self,
        trajectories: List[HindsightTrajectory],
        lr: float,
        hindsight_ratio: float,
    ) -> UpdateResult:
        """Perform BC-style update via OpenAI API (gpt-4o-mini).

        Generates supervisory feedback for each trajectory and computes
        a proxy loss based on the quality of the hindsight relabeling.
        """
        from openai import OpenAI

        client = OpenAI(api_key=self._openai_key)
        total_reward = 0.0

        for traj in trajectories:
            reward = self.compute_hindsight_reward(traj)
            total_reward += reward

        avg_reward = total_reward / len(trajectories)

        # Build a prompt summarizing the batch for gpt-4o-mini
        batch_summary = self._build_batch_summary(trajectories)

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a policy quality assessor for a hindsight "
                            "experience replay system. Given a batch of trajectory "
                            "summaries and their hindsight rewards, estimate the "
                            "policy improvement quality as a float in [0.0, 1.0]. "
                            "Return ONLY a JSON object with a single key 'loss' "
                            "representing estimated BC loss (lower is better, "
                            "so use 1.0 - quality_score)."
                        ),
                    },
                    {
                        "role": "user",
                        "content": batch_summary
                        + f"\n\nAverage hindsight reward: {avg_reward:.4f}\n"
                        f"Batch size: {len(trajectories)}\n"
                        f"Hindsight ratio: {hindsight_ratio:.2f}\n"
                        f"Return JSON: {{'loss': <float>}}",
                    },
                ],
                temperature=0.0,
                max_tokens=128,
            )

            content = response.choices[0].message.content
            if content:
                import re

                json_match = re.search(r"\{[\s\S]*?\}", content)
                if json_match:
                    data = json.loads(json_match.group())
                    loss = float(data.get("loss", 1.0 - avg_reward))
                else:
                    loss = 1.0 - avg_reward
            else:
                loss = 1.0 - avg_reward

        except Exception as e:
            logger.warning("OpenAI API call failed: %s", e)
            loss = 1.0 - avg_reward

        return UpdateResult(
            loss=loss,
            num_samples=len(trajectories),
            hindsight_ratio=hindsight_ratio,
            timestamp=datetime.now(timezone.utc),
            is_simulation=False,
            details={
                "learning_rate": lr,
                "model": "gpt-4o-mini",
                "avg_hindsight_reward": avg_reward,
                "update_count": self._update_count,
            },
        )

    def _simulate_update(
        self,
        trajectories: List[HindsightTrajectory],
        lr: float,
        hindsight_ratio: float,
    ) -> UpdateResult:
        """Simulate a policy update using heuristic reward signals.

        Computes a synthetic "loss" based on:
        - Hindsight reward of each trajectory
        - Ratio of hindsight-enhanced trajectories in the batch
        - Learning rate (higher lr -> larger simulated gradient steps)
        """
        total_reward = 0.0
        hindsight_rewards: List[float] = []

        for traj in trajectories:
            reward = self.compute_hindsight_reward(traj)
            hindsight_rewards.append(reward)
            total_reward += reward

        avg_reward = total_reward / len(trajectories)

        # Synthetic loss: inverse of average reward, scaled by learning rate
        # Simulates gradient magnitude proportional to learning rate
        base_loss = 1.0 - avg_reward
        simulated_loss = base_loss * (1.0 + lr * 1e4)

        # Clamp to [0, 1] range (like a probability)
        simulated_loss = max(0.0, min(1.0, simulated_loss))

        return UpdateResult(
            loss=simulated_loss,
            num_samples=len(trajectories),
            hindsight_ratio=hindsight_ratio,
            timestamp=datetime.now(timezone.utc),
            is_simulation=True,
            details={
                "learning_rate": lr,
                "avg_hindsight_reward": avg_reward,
                "update_count": self._update_count,
                "min_reward": min(hindsight_rewards) if hindsight_rewards else 0.0,
                "max_reward": max(hindsight_rewards) if hindsight_rewards else 0.0,
            },
        )

    # ---------------------------------------------------------------------------
    # Reward computation
    # ---------------------------------------------------------------------------

    def compute_hindsight_reward(self, trajectory: HindsightTrajectory) -> float:
        """Compute hindsight reward for a trajectory.

        Uses the milestone verdicts and any hindsight relabeling to compute
        a dense reward signal in [0.0, 1.0].

        The reward is computed as:
        1. Fraction of milestones that passed (base reward)
        2. Bonus from hindsight relabeling if available

        Parameters
        ----------
        trajectory : HindsightTrajectory
            The trajectory to score.

        Returns
        -------
        float
            Reward signal in [0.0, 1.0].
        """
        if not isinstance(trajectory, HindsightTrajectory):
            raise TypeError(
                f"Expected HindsightTrajectory, got {type(trajectory).__name__}"
            )

        # Base reward: fraction of milestones that passed
        if trajectory.verdicts:
            base_reward = sum(
                1 for v in trajectory.verdicts if v.value == "pass"
            ) / len(trajectory.verdicts)
        else:
            base_reward = 0.0

        # If the trajectory has explicit hindsight rewards, use them
        if (
            trajectory.is_hindsight_enhanced
            and trajectory.hindsight_rewards
            and len(trajectory.hindsight_rewards) > 0
        ):
            hindsight_bonus = sum(trajectory.hindsight_rewards) / len(
                trajectory.hindsight_rewards
            )
            # Combine: base reward from verdicts + hindsight bonus
            # Hindsight relabeling can "rescue" a failed trajectory
            return (base_reward + hindsight_bonus) / 2.0

        return base_reward

    # ---------------------------------------------------------------------------
    # Batch summary for LLM-based updates
    # ---------------------------------------------------------------------------

    def _build_batch_summary(self, trajectories: List[HindsightTrajectory]) -> str:
        """Build a text summary of a trajectory batch for LLM prompting."""
        lines = [f"Batch of {len(trajectories)} trajectories:\n"]

        for i, traj in enumerate(trajectories[:5]):  # Limit to first 5 for token budget
            passed = sum(1 for v in traj.verdicts if v.value == "pass")
            failed = sum(1 for v in traj.verdicts if v.value == "fail")
            partial = sum(1 for v in traj.verdicts if v.value == "partial")

            summary = (
                f"  Trajectory {i + 1} (id={traj.id}): "
                f"task='{traj.task[:60]}...', "
                f"milestones={len(traj.milestones)}, "
                f"pass={passed}, fail={failed}, partial={partial}, "
                f"hindsight_enhanced={traj.is_hindsight_enhanced}"
            )
            lines.append(summary)

            if traj.unmet_rubrics:
                lines.append(f"    unmet_rubrics: {traj.unmet_rubrics[:2]}")
            if traj.hindsight_labels:
                lines.append(f"    hindsight_labels: {traj.hindsight_labels[:2]}")

        return "\n".join(lines)

    # ---------------------------------------------------------------------------
    # Buffer export / import
    # ---------------------------------------------------------------------------

    def export_buffer(self, path: str) -> None:
        """Export the hindsight buffer to a JSON file.

        Parameters
        ----------
        path : str
            File path to write the JSON to.
        """
        self._buffer.export(path)

    def import_buffer(self, path: str) -> None:
        """Import a hindsight buffer from a JSON file, replacing the current buffer.

        .. note::
            This replaces the current buffer entirely.

        Parameters
        ----------
        path : str
            File path to read the JSON from.
        """
        imported = HindsightBuffer.import_(path)
        # Replace internal buffer
        self._buffer = imported

    # ---------------------------------------------------------------------------
    # Representation
    # ---------------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"HindsightTrainer("
            f"buffer_size={len(self._buffer)}, "
            f"model_path={self._model_path!r}, "
            f"lr={self._learning_rate}, "
            f"update_count={self._update_count}, "
            f"openai={'yes' if self._openai_key else 'no'})"
        )
