"""Training metrics and logging utilities for HeRoS.

Provides dataclasses for tracking training progress and classes
for logging metrics to various backends (console, file, etc.).
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TrainingMetrics
# ---------------------------------------------------------------------------


@dataclass
class TrainingMetrics:
    """Metrics from a single training step or episode.

    Attributes
    ----------
    milestone_success_rate : float
        Fraction of milestones that passed in this step/episode.
        In [0.0, 1.0].
    hindsight_utilization : float
        Fraction of trajectories in the batch that were
        hindsight-enhanced. In [0.0, 1.0].
    avg_reward : float
        Average reward signal across trajectories. In [0.0, 1.0].
    policy_loss : float
        Computed policy loss from the update. Non-negative scalar.
    training_step : int
        Global training step counter.
    num_trajectories : int
        Number of trajectories in this training batch.
    num_hindsight : int
        Number of hindsight-enhanced trajectories in the batch.
    timestamp : datetime, optional
        UTC timestamp of when these metrics were recorded.
    episode_reward : float, optional
        Total episodic reward, if available.
    entropy : float, optional
        Policy entropy, if computed.
    value_loss : float, optional
        Value function loss, if computed.
    """

    milestone_success_rate: float
    hindsight_utilization: float
    avg_reward: float
    policy_loss: float
    training_step: int = 0
    num_trajectories: int = 0
    num_hindsight: int = 0
    timestamp: Optional[datetime] = None
    episode_reward: Optional[float] = None
    entropy: Optional[float] = None
    value_loss: Optional[float] = None

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)

        # Clamp values to valid ranges
        self.milestone_success_rate = max(0.0, min(1.0, self.milestone_success_rate))
        self.hindsight_utilization = max(0.0, min(1.0, self.hindsight_utilization))
        self.avg_reward = max(0.0, min(1.0, self.avg_reward))
        self.policy_loss = max(0.0, self.policy_loss)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metrics to a JSON-compatible dictionary.

        Returns
        -------
        Dict[str, Any]
            Serializable dictionary representation.
        """
        return {
            "milestone_success_rate": round(self.milestone_success_rate, 6),
            "hindsight_utilization": round(self.hindsight_utilization, 6),
            "avg_reward": round(self.avg_reward, 6),
            "policy_loss": round(self.policy_loss, 6),
            "training_step": self.training_step,
            "num_trajectories": self.num_trajectories,
            "num_hindsight": self.num_hindsight,
            "timestamp": (
                self.timestamp.isoformat() + "Z"
                if self.timestamp
                else datetime.now(timezone.utc).isoformat() + "Z"
            ),
            "episode_reward": (
                round(self.episode_reward, 6) if self.episode_reward is not None else None
            ),
            "entropy": round(self.entropy, 6) if self.entropy is not None else None,
            "value_loss": round(self.value_loss, 6) if self.value_loss is not None else None,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainingMetrics":
        """Deserialize from a JSON-compatible dictionary.

        Parameters
        ----------
        d : Dict[str, Any]
            Dictionary produced by :meth:`to_dict`.

        Returns
        -------
        TrainingMetrics
            Reconstructed metrics object.
        """
        timestamp = None
        if d.get("timestamp"):
            try:
                ts_str = d["timestamp"].replace("Z", "+00:00")
                timestamp = datetime.fromisoformat(ts_str)
            except ValueError:
                timestamp = None

        return cls(
            milestone_success_rate=d.get("milestone_success_rate", 0.0),
            hindsight_utilization=d.get("hindsight_utilization", 0.0),
            avg_reward=d.get("avg_reward", 0.0),
            policy_loss=d.get("policy_loss", 0.0),
            training_step=d.get("training_step", 0),
            num_trajectories=d.get("num_trajectories", 0),
            num_hindsight=d.get("num_hindsight", 0),
            timestamp=timestamp,
            episode_reward=d.get("episode_reward"),
            entropy=d.get("entropy"),
            value_loss=d.get("value_loss"),
        )

    def __repr__(self) -> str:
        return (
            f"TrainingMetrics("
            f"step={self.training_step}, "
            f"success_rate={self.milestone_success_rate:.2%}, "
            f"hindsight_util={self.hindsight_utilization:.2%}, "
            f"avg_reward={self.avg_reward:.4f}, "
            f"policy_loss={self.policy_loss:.4f})"
        )


# ---------------------------------------------------------------------------
# TrainingLogger
# ---------------------------------------------------------------------------


class TrainingLogger:
    """Logs training metrics to console, file, or custom handlers.

    Supports logging at two granularities:
    - Step-level: each training step's metrics
    - Episode-level: aggregated episode statistics

    Parameters
    ----------
    log_dir : Union[str, Path], optional
        Directory to save metric logs. If None, no file logging occurs.
    console_logging : bool, optional
        Whether to log metrics to the console. Defaults to True.
    file_logging : bool, optional
        Whether to save metrics to JSON files. Defaults to True.
    metrics_history_size : int, optional
        Maximum number of step-level metrics to keep in memory.
        Defaults to 10000.
    name : str, optional
        Logger name for identification. Defaults to "heros".

    Examples
    --------
    >>> logger = TrainingLogger(log_dir="./logs", console_logging=True)
    >>> metrics = TrainingMetrics(
    ...     milestone_success_rate=0.8,
    ...     hindsight_utilization=0.3,
    ...     avg_reward=0.75,
    ...     policy_loss=0.25,
    ...     training_step=1,
    ... )
    >>> logger.log_step(metrics)
    >>> logger.save("./final_metrics.json")
    """

    def __init__(
        self,
        log_dir: Optional[Union[str, Path]] = None,
        console_logging: bool = True,
        file_logging: bool = True,
        metrics_history_size: int = 10000,
        name: str = "heros",
    ) -> None:
        self._name = name
        self._console_logging = bool(console_logging)
        self._file_logging = bool(file_logging)
        self._metrics_history_size = metrics_history_size

        # History storage
        self._step_metrics: List[TrainingMetrics] = []
        self._episode_metrics: List[TrainingMetrics] = []

        # Aggregated statistics
        self._total_steps: int = 0
        self._total_episodes: int = 0

        # Configure logging
        self._logger = logging.getLogger(f"{name}.training")
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

        # Setup file logging
        self._log_dir: Optional[Path] = None
        if log_dir is not None and self._file_logging:
            self._log_dir = Path(log_dir)
            self._log_dir.mkdir(parents=True, exist_ok=True)
            self._step_log_path = self._log_dir / f"{name}_steps.jsonl"
            self._episode_log_path = self._log_dir / f"{name}_episodes.jsonl"

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def step_metrics(self) -> List[TrainingMetrics]:
        """History of step-level metrics."""
        return self._step_metrics.copy()

    @property
    def episode_metrics(self) -> List[TrainingMetrics]:
        """History of episode-level metrics."""
        return self._episode_metrics.copy()

    @property
    def total_steps(self) -> int:
        """Total number of training steps logged."""
        return self._total_steps

    @property
    def total_episodes(self) -> int:
        """Total number of episodes logged."""
        return self._total_episodes

    # -------------------------------------------------------------------------
    # Logging methods
    # -------------------------------------------------------------------------

    def log_step(self, metrics: TrainingMetrics) -> None:
        """Log metrics from a single training step.

        Parameters
        ----------
        metrics : TrainingMetrics
            Metrics from the training step.
        """
        if not isinstance(metrics, TrainingMetrics):
            raise TypeError(
                f"Expected TrainingMetrics, got {type(metrics).__name__}"
            )

        self._step_metrics.append(metrics)
        self._total_steps += 1

        # Trim history if needed
        if len(self._step_metrics) > self._metrics_history_size:
            self._step_metrics = self._step_metrics[-self._metrics_history_size:]

        # Console output
        if self._console_logging:
            self._logger.info(
                "Step %d | success=%.1f%% | hindsight=%.1f%% | "
                "reward=%.4f | loss=%.4f",
                metrics.training_step,
                metrics.milestone_success_rate * 100,
                metrics.hindsight_utilization * 100,
                metrics.avg_reward,
                metrics.policy_loss,
            )

        # File output
        if self._file_logging and self._log_dir is not None:
            self._append_to_jsonl(self._step_log_path, metrics.to_dict())

    def log_episode(self, metrics: TrainingMetrics) -> None:
        """Log metrics from a complete episode.

        Parameters
        ----------
        metrics : TrainingMetrics
            Aggregated metrics from the episode.
        """
        if not isinstance(metrics, TrainingMetrics):
            raise TypeError(
                f"Expected TrainingMetrics, got {type(metrics).__name__}"
            )

        self._episode_metrics.append(metrics)
        self._total_episodes += 1

        # Trim history if needed
        if len(self._episode_metrics) > self._metrics_history_size:
            self._episode_metrics = self._episode_metrics[-self._metrics_history_size:]

        # Console output
        if self._console_logging:
            episode_str = ""
            if metrics.episode_reward is not None:
                episode_str = f" | episode_reward={metrics.episode_reward:.4f}"

            self._logger.info(
                "Episode %d | success=%.1f%% | hindsight=%.1f%% | "
                "reward=%.4f | loss=%.4f%s",
                self._total_episodes,
                metrics.milestone_success_rate * 100,
                metrics.hindsight_utilization * 100,
                metrics.avg_reward,
                metrics.policy_loss,
                episode_str,
            )

        # File output
        if self._file_logging and self._log_dir is not None:
            self._append_to_jsonl(self._episode_log_path, metrics.to_dict())

    def _append_to_jsonl(self, path: Path, data: Dict[str, Any]) -> None:
        """Append a dictionary to a JSONL file.

        Parameters
        ----------
        path : Path
            Path to the JSONL file.
        data : Dict[str, Any]
            Data to append.
        """
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
        except Exception as e:
            self._logger.warning("Failed to write to %s: %s", path, e)

    # -------------------------------------------------------------------------
    # Statistics and aggregation
    # -------------------------------------------------------------------------

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all logged metrics.

        Returns
        -------
        Dict[str, Any]
            Summary statistics including averages and totals.
        """
        step_count = len(self._step_metrics)
        episode_count = len(self._episode_metrics)

        if step_count == 0:
            return {
                "total_steps": 0,
                "total_episodes": 0,
                "avg_milestone_success_rate": 0.0,
                "avg_hindsight_utilization": 0.0,
                "avg_reward": 0.0,
                "avg_policy_loss": 0.0,
            }

        # Aggregate step metrics
        avg_success = sum(m.milestone_success_rate for m in self._step_metrics) / step_count
        avg_hindsight = sum(m.hindsight_utilization for m in self._step_metrics) / step_count
        avg_reward = sum(m.avg_reward for m in self._step_metrics) / step_count
        avg_loss = sum(m.policy_loss for m in self._step_metrics) / step_count

        # Episode metrics if available
        episode_reward = None
        if episode_count > 0:
            rewards = [
                m.episode_reward
                for m in self._episode_metrics
                if m.episode_reward is not None
            ]
            if rewards:
                episode_reward = sum(rewards) / len(rewards)

        return {
            "total_steps": self._total_steps,
            "total_episodes": self._total_episodes,
            "avg_milestone_success_rate": round(avg_success, 6),
            "avg_hindsight_utilization": round(avg_hindsight, 6),
            "avg_reward": round(avg_reward, 6),
            "avg_policy_loss": round(avg_loss, 6),
            "latest_episode_reward": episode_reward,
        }

    def get_recent_metrics(
        self,
        n: int = 100,
        as_dicts: bool = False,
    ) -> Union[List[TrainingMetrics], List[Dict[str, Any]]]:
        """Get the most recent N step metrics.

        Parameters
        ----------
        n : int, optional
            Number of recent metrics to return. Defaults to 100.
        as_dicts : bool, optional
            If True, return dicts instead of TrainingMetrics objects.
            Defaults to False.

        Returns
        -------
        Union[List[TrainingMetrics], List[Dict[str, Any]]]
            Most recent N metrics.
        """
        recent = self._step_metrics[-n:]
        if as_dicts:
            return [m.to_dict() for m in recent]
        return recent

    # -------------------------------------------------------------------------
    # Save and load
    # -------------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Save all logged metrics to a JSON file.

        Parameters
        ----------
        path : Union[str, Path]
            Path to save the metrics.
        """
        path = Path(path)

        data = {
            "name": self._name,
            "saved_at": datetime.now(timezone.utc).isoformat() + "Z",
            "summary": self.get_summary(),
            "step_metrics": [m.to_dict() for m in self._step_metrics],
            "episode_metrics": [m.to_dict() for m in self._episode_metrics],
        }

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self._logger.info("Saved metrics to %s", path)
        except Exception as e:
            self._logger.error("Failed to save metrics to %s: %s", path, e)
            raise

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TrainingLogger":
        """Load a TrainingLogger from a saved JSON file.

        Parameters
        ----------
        path : Union[str, Path]
            Path to the saved metrics file.

        Returns
        -------
        TrainingLogger
            Loaded logger with all metrics restored.
        """
        path = Path(path)

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.error("Failed to load metrics from %s: %s", path, e)
            raise

        # Create a new logger (can't fully restore internal state)
        log_dir = path.parent
        name = data.get("name", "heros")
        logger_obj = cls(log_dir=log_dir, name=name)

        # Restore metrics
        for m_dict in data.get("step_metrics", []):
            logger_obj._step_metrics.append(TrainingMetrics.from_dict(m_dict))

        for m_dict in data.get("episode_metrics", []):
            logger_obj._episode_metrics.append(TrainingMetrics.from_dict(m_dict))

        summary = data.get("summary", {})
        logger_obj._total_steps = summary.get("total_steps", len(logger_obj._step_metrics))
        logger_obj._total_episodes = summary.get(
            "total_episodes", len(logger_obj._episode_metrics)
        )

        return logger_obj

    # -------------------------------------------------------------------------
    # Utility
    # -------------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all logged metrics."""
        self._step_metrics.clear()
        self._episode_metrics.clear()
        self._total_steps = 0
        self._total_episodes = 0
        self._logger.info("Metrics history cleared.")

    def __repr__(self) -> str:
        return (
            f"TrainingLogger("
            f"name={self._name!r}, "
            f"steps={self._total_steps}, "
            f"episodes={self._total_episodes})"
        )
