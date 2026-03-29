"""Step 9: Hybrid Uncertainty Estimation Integration.

Two-sample hybrid estimator: embedding cosine similarity + LLM verbalized confidence
for milestone critic confidence calibration.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np

from heros.critic import CriticResult, MilestoneCritic, Verdict
from heros.planner import Milestone


# ---------------------------------------------------------------------------
# HybridUncertaintyEstimator
# ---------------------------------------------------------------------------


class HybridUncertaintyEstimator:
    """Two-sample hybrid estimator: embedding cosine similarity + LLM verbalized confidence.

    Parameters
    ----------
    embedding_model : str
        OpenAI embedding model to use. Defaults to "text-embedding-3-small".
    alpha : float
        Weight for verbalized confidence in the fused score.
        ``fused = alpha * verbalized + (1 - alpha) * cosine``.
        Higher fused values = more uncertain.
    """

    def __init__(self, embedding_model: str = "text-embedding-3-small", alpha: float = 0.5):
        self._embedding_model = embedding_model
        self._alpha = alpha

    def estimate(self, text1: str, text2: str) -> tuple[float, float, float]:
        """Estimate uncertainty between two text samples.

        Parameters
        ----------
        text1 : str
            First text (e.g., milestone description).
        text2 : str
            Second text (e.g., execution trace).

        Returns
        -------
        tuple[float, float, float]
            (cosine_score, verbalized_confidence, fused_uncertainty)

            - cosine_score: embedding cosine similarity (0-1, higher = more similar)
            - verbalized_confidence: LLM confidence if available, else 0.5 (0-1)
            - fused_uncertainty: weighted combination (0-1, higher = more uncertain)
        """
        cosine_score = self._compute_cosine_similarity(text1, text2)
        # For now verbalized is a placeholder; callers inject LLM responses
        verbalized_confidence = 0.5  # Will be overridden by _extract when LLM used
        fused_uncertainty = self._alpha * verbalized_confidence + (1 - self._alpha) * (
            1.0 - cosine_score
        )
        return cosine_score, verbalized_confidence, fused_uncertainty

    def estimate_with_llm(
        self, text1: str, text2: str, llm_response: str
    ) -> tuple[float, float, float]:
        """Estimate uncertainty with an LLM verbalized confidence response.

        Parameters
        ----------
        text1 : str
            First text.
        text2 : str
            Second text.
        llm_response : str
            Raw LLM response containing "I am X% confident" text.

        Returns
        -------
        tuple[float, float, float]
            (cosine_score, verbalized_confidence, fused_uncertainty)
        """
        cosine_score = self._compute_cosine_similarity(text1, text2)
        verbalized_confidence = self._extract_verbalized_confidence(llm_response)
        fused_uncertainty = self._alpha * verbalized_confidence + (1 - self._alpha) * (
            1.0 - cosine_score
        )
        return cosine_score, verbalized_confidence, fused_uncertainty

    def _compute_cosine_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two text embeddings."""
        emb1 = self._get_embedding(text1)
        emb2 = self._get_embedding(text2)
        dot = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.clip(dot / (norm1 * norm2), 0.0, 1.0))

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding vector for text.

        Uses OpenAI embeddings API if OPENAI_API_KEY is set,
        otherwise returns a deterministic mock embedding based on text hash.
        """
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if api_key:
            try:
                from openai import OpenAI

                client = OpenAI(api_key=api_key)
                response = client.embeddings.create(
                    model=self._embedding_model,
                    input=text,
                )
                data = response.data[0].embedding
                return np.array(data, dtype=np.float64)
            except Exception:
                pass  # Fall through to mock

        # Deterministic mock embedding based on text hash
        # Produces a fixed-dimension vector (1536-dim like text-embedding-3-small)
        dim = 1536
        seed = hash(text) % (2**32)
        rng = np.random.RandomState(seed)
        return rng.rand(dim).astype(np.float64)

    def _extract_verbalized_confidence(self, llm_response: str) -> float:
        """Extract confidence percentage from LLM response.

        Matches patterns like "I am 85% confident" or "I'm 85 confident".
        Returns float in [0.0, 1.0].
        """
        pattern = r"(?:i\s+am|i'm)\s*(\d+(?:\.\d+)?)\s*%?\s*confident"
        match = re.search(pattern, llm_response.lower())
        if match:
            value = float(match.group(1))
            # If the number is > 1, assume it's a percentage (e.g., 85)
            if value > 1.0:
                value = value / 100.0
            return float(np.clip(value, 0.0, 1.0))
        return 0.5  # Default if no confidence found


# ---------------------------------------------------------------------------
# CalibrationDataset & CalibrationPair
# ---------------------------------------------------------------------------


@dataclass
class CalibrationPair:
    """A single calibration data point.

    Attributes
    ----------
    predicted_confidence : float
        Model's predicted confidence (0-1).
    correct : bool
        Whether the prediction was correct (true label).
    """

    predicted_confidence: float
    correct: bool


class CalibrationDataset:
    """A collection of calibration pairs for evaluating model calibration."""

    def __init__(self) -> None:
        self._pairs: list[CalibrationPair] = []

    def add(self, predicted_confidence: float, correct: bool) -> None:
        """Add a calibration pair.

        Parameters
        ----------
        predicted_confidence : float
            Model confidence in [0, 1].
        correct : bool
            Whether the prediction was correct.
        """
        self._pairs.append(
            CalibrationPair(
                predicted_confidence=float(np.clip(predicted_confidence, 0.0, 1.0)),
                correct=bool(correct),
            )
        )

    def to_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (confidences, correctness) as numpy arrays.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (confidences, correctness) where correctness is 0/1.
        """
        confidences = np.array([p.predicted_confidence for p in self._pairs], dtype=np.float64)
        correctness = np.array([int(p.correct) for p in self._pairs], dtype=np.int32)
        return confidences, correctness

    def split(
        self, train_ratio: float = 0.7
    ) -> tuple[CalibrationDataset, CalibrationDataset]:
        """Split dataset into train and test sets sequentially (no shuffle).

        Parameters
        ----------
        train_ratio : float
            Fraction of data to use for training (default 0.7).

        Returns
        -------
        tuple[CalibrationDataset, CalibrationDataset]
            (train_dataset, test_dataset)
        """
        n = len(self._pairs)
        split_idx = int(n * train_ratio)
        train_pairs = self._pairs[:split_idx]
        test_pairs = self._pairs[split_idx:]

        train_ds = CalibrationDataset()
        test_ds = CalibrationDataset()
        for p in train_pairs:
            train_ds._pairs.append(p)
        for p in test_pairs:
            test_ds._pairs.append(p)

        return train_ds, test_ds

    def __len__(self) -> int:
        return len(self._pairs)


# ---------------------------------------------------------------------------
# CalibrationCurve & ECE
# ---------------------------------------------------------------------------


@dataclass
class CalibrationCurve:
    """Calibration curve data points.

    Attributes
    ----------
    confidences : np.ndarray
        Mean predicted confidence per bin.
    accuracies : np.ndarray
        Fraction correct per bin.
    bin_counts : np.ndarray
        Number of samples per bin.
    """

    confidences: np.ndarray
    accuracies: np.ndarray
    bin_counts: np.ndarray


def compute_expected_calibration_error(
    confidences: np.ndarray,
    correctness: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE) using equal-width bins.

    ECE = sum_i (bin_count_i / n) * |accuracy_i - confidence_i|

    Parameters
    ----------
    confidences : np.ndarray
        Predicted confidence scores (0-1).
    correctness : np.ndarray
        Binary correctness values (0 or 1).
    n_bins : int
        Number of equal-width bins (default 10).

    Returns
    -------
    float
        ECE score (0 = perfectly calibrated, higher = more miscalibrated).
    """
    if len(confidences) == 0:
        return 0.0

    confidences = np.asarray(confidences, dtype=np.float64)
    correctness = np.asarray(correctness, dtype=np.int32)

    # Equal-width binning
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confidences)

    for i in range(n_bins):
        # Samples in this bin
        mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
        # Last bin: include right edge
        if i == n_bins - 1:
            mask = (confidences >= bin_edges[i]) & (confidences <= bin_edges[i + 1])

        bin_count = int(np.sum(mask))
        if bin_count == 0:
            continue

        bin_conf = confidences[mask]
        bin_acc = correctness[mask]

        avg_confidence = float(np.mean(bin_conf))
        avg_accuracy = float(np.mean(bin_acc))

        ece += (bin_count / n) * abs(avg_accuracy - avg_confidence)

    return float(ece)


# ---------------------------------------------------------------------------
# AUROC
# ---------------------------------------------------------------------------


@dataclass
class AUROCMetrics:
    """AUROC evaluation metrics.

    Attributes
    ----------
    auroc : float
        Area under the ROC curve (0.5 = random, 1.0 = perfect).
    precision : float
        Precision at the optimal threshold.
    recall : float
        Recall (sensitivity) at the optimal threshold.
    f1 : float
        F1 score at the optimal threshold.
    optimal_threshold : float
        Threshold selected via Youden's J statistic.
    """

    auroc: float
    precision: float
    recall: float
    f1: float
    optimal_threshold: float


def compute_auroc(
    confidences: np.ndarray,
    correctness: np.ndarray,
) -> dict:
    """Compute AUROC and related metrics using Youden's J for threshold selection.

    Parameters
    ----------
    confidences : np.ndarray
        Predicted confidence scores (0-1).
    correctness : np.ndarray
        Binary correctness values (0 or 1).

    Returns
    -------
    dict
        Dictionary with auroc, precision, recall, f1, optimal_threshold.
    """
    try:
        from sklearn.metrics import auc, precision_recall_curve, roc_curve
    except ImportError:
        return {
            "auroc": 0.5,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "optimal_threshold": 0.5,
        }

    if len(confidences) < 2:
        return {
            "auroc": 0.5,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "optimal_threshold": 0.5,
        }

    # Check if all labels are the same
    if np.std(correctness) == 0:
        return {
            "auroc": float("nan"),
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "optimal_threshold": 0.5,
        }

    confidences = np.asarray(confidences, dtype=np.float64)
    correctness = np.asarray(correctness, dtype=np.int32)

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(correctness, confidences)
    auroc = float(auc(fpr, tpr))

    # Compute precision-recall curve
    precision_arr, recall_arr, pr_thresholds = precision_recall_curve(
        correctness, confidences
    )

    # Find optimal threshold using Youden's J = sensitivity + specificity - 1
    # = tpr + (1 - fpr) - 1 = tpr - fpr
    j_scores = tpr - fpr
    best_j_idx = int(np.argmax(j_scores))
    optimal_threshold = float(np.clip(thresholds[best_j_idx], 0.0, 1.0))

    # Compute metrics at optimal threshold
    predictions = (confidences >= optimal_threshold).astype(int)

    tp = int(np.sum((predictions == 1) & (correctness == 1)))
    fp = int(np.sum((predictions == 1) & (correctness == 0)))
    fn = int(np.sum((predictions == 0) & (correctness == 1)))

    precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_val = (
        2 * precision_val * recall_val / (precision_val + recall_val)
        if (precision_val + recall_val) > 0
        else 0.0
    )

    return {
        "auroc": auroc,
        "precision": precision_val,
        "recall": recall_val,
        "f1": f1_val,
        "optimal_threshold": optimal_threshold,
    }


# ---------------------------------------------------------------------------
# CalibratedMilestoneCritic
# ---------------------------------------------------------------------------


class CalibratedMilestoneCritic:
    """Wraps MilestoneCritic with uncertainty-calibrated verdicts.

    Uses Platt scaling (LogisticRegression) on a calibration dataset
    to adjust the base critic's confidence scores.

    Parameters
    ----------
    base_critic : MilestoneCritic
        The underlying milestone critic.
    embedding_model : str
        OpenAI embedding model for uncertainty estimation.
    alpha : float
        Weight for verbalized confidence in the fused uncertainty.
    calibration_dataset : CalibrationDataset, optional
        Dataset of (confidence, correct) pairs for Platt scaling.
        If None, no calibration adjustment is applied.
    """

    def __init__(
        self,
        base_critic: MilestoneCritic,
        embedding_model: str = "text-embedding-3-small",
        alpha: float = 0.5,
        calibration_dataset: Optional[CalibrationDataset] = None,
    ) -> None:
        self._base_critic = base_critic
        self._embedding_model = embedding_model
        self._alpha = alpha
        self._calibration_dataset = calibration_dataset
        self._platt_scaler: Optional["LogisticRegression"] = None

        if calibration_dataset is not None and len(calibration_dataset) >= 2:
            self._fit_platt_scaler(calibration_dataset)

    def _fit_platt_scaler(self, dataset: CalibrationDataset) -> None:
        """Fit a LogisticRegression (Platt scaling) on calibration data."""
        try:
            from sklearn.linear_model import LogisticRegression

            confs, correct = dataset.to_arrays()
            if len(np.unique(correct)) < 2:
                return  # Need both classes

            X = confs.reshape(-1, 1)
            y = correct

            self._platt_scaler = LogisticRegression()
            self._platt_scaler.fit(X, y)
        except Exception:
            self._platt_scaler = None

    def review(
        self,
        milestone: Milestone,
        execution_trace: str,
    ) -> CriticResult:
        """Review milestone completion with calibrated confidence.

        Parameters
        ----------
        milestone : Milestone
            The milestone being evaluated.
        execution_trace : str
            Execution trace to review.

        Returns
        -------
        CriticResult
            Critic result with calibrated confidence.
        """
        result = self._base_critic.review(
            milestone_description=milestone.description,
            rubric=milestone.rubric,
            execution_trace=execution_trace,
        )

        calibrated_confidence = self._apply_calibration(result.confidence)

        return CriticResult(
            verdict=result.verdict,
            feedback=result.feedback,
            confidence=calibrated_confidence,
            reward_signal=result.reward_signal,
        )

    def _apply_calibration(self, raw_confidence: float) -> float:
        """Apply Platt scaling calibration to a raw confidence.

        Parameters
        ----------
        raw_confidence : float
            Raw confidence from the base critic.

        Returns
        -------
        float
            Calibrated confidence in [0, 1].
        """
        if self._platt_scaler is None:
            return raw_confidence

        try:
            X = np.array([[raw_confidence]], dtype=np.float64)
            calibrated = float(self._platt_scaler.predict_proba(X)[0, 1])
            return float(np.clip(calibrated, 0.0, 1.0))
        except Exception:
            return raw_confidence


# ---------------------------------------------------------------------------
# UncertaintyAwareRewardAuditor
# ---------------------------------------------------------------------------


class UncertaintyAwareRewardAuditor:
    """Adjusts reward signals by uncertainty (less trust for high-uncertainty verdicts).

    Uses the HybridUncertaintyEstimator to compute uncertainty between milestone
    description and execution trace, then dampens the reward accordingly.

    Parameters
    ----------
    base_critic : MilestoneCritic
        The underlying milestone critic.
    uncertainty_estimator : HybridUncertaintyEstimator
        The uncertainty estimator for computing uncertainty factors.
    """

    def __init__(
        self,
        base_critic: MilestoneCritic,
        uncertainty_estimator: HybridUncertaintyEstimator,
    ) -> None:
        self._base_critic = base_critic
        self._uncertainty_estimator = uncertainty_estimator

    def audit(
        self,
        milestone: Milestone,
        execution_trace: str,
        base_reward: float,
    ) -> tuple[float, float]:
        """Audit a milestone verdict and adjust reward by uncertainty.

        Parameters
        ----------
        milestone : Milestone
            The milestone being evaluated.
        execution_trace : str
            Execution trace to evaluate.
        base_reward : float
            Base reward from milestone verdict (0-1).

        Returns
        -------
        tuple[float, float]
            (adjusted_reward, uncertainty) where:
            - adjusted_reward = base_reward * (1 - uncertainty_factor)
            - uncertainty = fused_uncertainty from estimator (0-1)
        """
        result = self._base_critic.review(
            milestone_description=milestone.description,
            rubric=milestone.rubric,
            execution_trace=execution_trace,
        )

        cosine_score, verbalized_conf, fused_uncertainty = self._uncertainty_estimator.estimate(
            milestone.description, execution_trace
        )

        # Dampen reward by uncertainty factor
        uncertainty_factor = fused_uncertainty
        adjusted_reward = base_reward * (1.0 - uncertainty_factor)

        return float(np.clip(adjusted_reward, 0.0, 1.0)), float(fused_uncertainty)


# ---------------------------------------------------------------------------
# evaluate_calibration
# ---------------------------------------------------------------------------


def evaluate_calibration(
    calibrated_critic: CalibratedMilestoneCritic,
    evaluation_tasks: list[tuple[Milestone, str]],
) -> dict:
    """Evaluate calibration quality of a CalibratedMilestoneCritic.

    Parameters
    ----------
    calibrated_critic : CalibratedMilestoneCritic
        The calibrated critic to evaluate.
    evaluation_tasks : list[tuple[Milestone, str]]
        List of (milestone, execution_trace) pairs for evaluation.

    Returns
    -------
    dict
        Dictionary with:
        - auroc: float
        - precision: float
        - recall: float
        - f1: float
        - optimal_threshold: float
        - ece: float (Expected Calibration Error)
        - n_evaluated: int
        - calibration_curve: dict with confidences, accuracies, bin_counts
    """
    if not evaluation_tasks:
        return {
            "auroc": 0.5,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "optimal_threshold": 0.5,
            "ece": 0.0,
            "n_evaluated": 0,
            "calibration_curve": {
                "confidences": [],
                "accuracies": [],
                "bin_counts": [],
            },
        }

    confidences_list: list[float] = []
    correctness_list: list[int] = []

    for milestone, execution_trace in evaluation_tasks:
        result = calibrated_critic.review(milestone, execution_trace)
        # Determine correctness: PASS verdicts are correct, FAIL are incorrect
        is_correct = 1 if result.verdict == Verdict.PASS else 0
        confidences_list.append(result.confidence)
        correctness_list.append(is_correct)

    confidences_arr = np.array(confidences_list, dtype=np.float64)
    correctness_arr = np.array(correctness_list, dtype=np.int32)

    # Compute AUROC
    auroc_metrics = compute_auroc(confidences_arr, correctness_arr)

    # Compute ECE
    ece = compute_expected_calibration_error(confidences_arr, correctness_arr, n_bins=10)

    # Compute calibration curve
    calibration_curve = _compute_calibration_curve(confidences_arr, correctness_arr, n_bins=10)

    return {
        "auroc": auroc_metrics["auroc"],
        "precision": auroc_metrics["precision"],
        "recall": auroc_metrics["recall"],
        "f1": auroc_metrics["f1"],
        "optimal_threshold": auroc_metrics["optimal_threshold"],
        "ece": ece,
        "n_evaluated": len(evaluation_tasks),
        "calibration_curve": {
            "confidences": calibration_curve.confidences.tolist(),
            "accuracies": calibration_curve.accuracies.tolist(),
            "bin_counts": calibration_curve.bin_counts.tolist(),
        },
    }


def _compute_calibration_curve(
    confidences: np.ndarray,
    correctness: np.ndarray,
    n_bins: int = 10,
) -> CalibrationCurve:
    """Compute calibration curve data points for equal-width bins."""
    if len(confidences) == 0:
        return CalibrationCurve(
            confidences=np.array([], dtype=np.float64),
            accuracies=np.array([], dtype=np.float64),
            bin_counts=np.array([], dtype=np.int32),
        )

    confidences = np.asarray(confidences, dtype=np.float64)
    correctness = np.asarray(correctness, dtype=np.int32)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_confidences: list[float] = []
    bin_accuracies: list[float] = []
    bin_counts_list: list[int] = []

    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
        else:
            mask = (confidences >= bin_edges[i]) & (confidences <= bin_edges[i + 1])

        bc = int(np.sum(mask))
        bin_counts_list.append(bc)
        if bc == 0:
            bin_confidences.append((bin_edges[i] + bin_edges[i + 1]) / 2.0)
            bin_accuracies.append(0.0)
        else:
            bin_confidences.append(float(np.mean(confidences[mask])))
            bin_accuracies.append(float(np.mean(correctness[mask])))

    return CalibrationCurve(
        confidences=np.array(bin_confidences, dtype=np.float64),
        accuracies=np.array(bin_accuracies, dtype=np.float64),
        bin_counts=np.array(bin_counts_list, dtype=np.int32),
    )
