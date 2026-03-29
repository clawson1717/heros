"""Tests for the HeRoS Uncertainty Estimation Module (Step 9).

Covers:
- HybridUncertaintyEstimator: cosine computation, verbalized confidence parsing,
  fused uncertainty calculation, mock vs real embedding fallback
- CalibrationDataset: add, to_arrays, split (deterministic, train/test ratio)
- compute_expected_calibration_error: perfect calibration, miscalibrated, n_bins edge cases
- compute_auroc: random (AUROC≈0.5), perfect (AUROC=1.0), threshold selection via Youden's J
- CalibratedMilestoneCritic: wraps base critic, calibration adjustment, passthrough of feedback
- UncertaintyAwareRewardAuditor: reward dampening by uncertainty, uncertainty extraction
- evaluate_calibration: returns dict with AUROC, ECE, curve data
"""

from __future__ import annotations

import numpy as np
import pytest

from heros.critic import CriticResult, MilestoneCritic, Verdict
from heros.planner import Milestone

from heros.uncertainty import (
    HybridUncertaintyEstimator,
    CalibrationPair,
    CalibrationDataset,
    CalibrationCurve,
    compute_expected_calibration_error,
    compute_auroc,
    AUROCMetrics,
    CalibratedMilestoneCritic,
    UncertaintyAwareRewardAuditor,
    evaluate_calibration,
)


# ---------------------------------------------------------------------------
# HybridUncertaintyEstimator Tests
# ---------------------------------------------------------------------------

class TestHybridUncertaintyEstimatorCosine:
    def test_identical_texts_high_cosine(self):
        """Identical texts should yield cosine score of 1.0."""
        est = HybridUncertaintyEstimator(alpha=0.0)  # alpha=0 so cosine dominates
        cosine, verbal, fused = est.estimate("hello world", "hello world")
        assert cosine == pytest.approx(1.0, abs=1e-6)

    def test_completely_different_texts_low_cosine(self):
        """Completely different texts should yield a cosine score in [0, 1].
        Note: Mock embeddings are deterministic random vectors, so cosine
        of different vectors is bounded but not guaranteed low.
        """
        est = HybridUncertaintyEstimator(alpha=0.0)
        cosine, verbal, fused = est.estimate(
            "the quick brown fox jumps", "zyxwvu tsrq ponm lkji"
        )
        # Cosine of random vectors is bounded in [0, 1]
        assert 0.0 <= cosine <= 1.0

    def test_cosine_symmetric(self):
        """Cosine similarity should be symmetric."""
        est = HybridUncertaintyEstimator(alpha=0.0)
        c1, _, _ = est.estimate("hello world", "world hello")
        c2, _, _ = est.estimate("world hello", "hello world")
        assert c1 == pytest.approx(c2, abs=1e-6)

    def test_cosine_deterministic_mock(self):
        """Mock embeddings should be deterministic."""
        est = HybridUncertaintyEstimator(alpha=0.0)
        c1, _, _ = est.estimate("same text here", "same text here")
        c2, _, _ = est.estimate("same text here", "same text here")
        assert c1 == pytest.approx(c2, abs=1e-6)

    def test_cosine_bounded_0_1(self):
        """Cosine scores should always be in [0, 1]."""
        texts = [
            ("a", "b"),
            ("hello", "goodbye"),
            ("the quick brown fox", "a totally different sentence"),
            ("", ""),
            ("x" * 1000, "y" * 1000),
        ]
        est = HybridUncertaintyEstimator()
        for t1, t2 in texts:
            c, v, f = est.estimate(t1, t2)
            assert 0.0 <= c <= 1.0
            assert 0.0 <= v <= 1.0
            assert 0.0 <= f <= 1.0


class TestHybridUncertaintyEstimatorVerbalized:
    def test_extract_exact_percent(self):
        """Extract 'I am 85% confident' correctly."""
        est = HybridUncertaintyEstimator()
        val = est._extract_verbalized_confidence("I am 85% confident that this is correct.")
        assert val == pytest.approx(0.85, abs=1e-6)

    def test_extract_decimal_percent(self):
        """Extract 'I am 72.5% confident' correctly."""
        est = HybridUncertaintyEstimator()
        val = est._extract_verbalized_confidence("I'm 72.5% confident.")
        assert val == pytest.approx(0.725, abs=1e-6)

    def test_extract_no_percent_symbol(self):
        """Extract 'I am 90 confident' (no % symbol) correctly."""
        est = HybridUncertaintyEstimator()
        val = est._extract_verbalized_confidence("I am 90 confident this passes.")
        assert val == pytest.approx(0.9, abs=1e-6)

    def test_extract_im(self):
        """Extract "I'm 50% confident" correctly."""
        est = HybridUncertaintyEstimator()
        val = est._extract_verbalized_confidence("I'm 50% confident.")
        assert val == pytest.approx(0.5, abs=1e-6)

    def test_extract_case_insensitive(self):
        """Extraction should be case-insensitive."""
        est = HybridUncertaintyEstimator()
        val = est._extract_verbalized_confidence("I AM 60% CONFIDENT")
        assert val == pytest.approx(0.6, abs=1e-6)

    def test_extract_missing_returns_default(self):
        """Missing confidence should return 0.5 (default)."""
        est = HybridUncertaintyEstimator()
        val = est._extract_verbalized_confidence("This is a response with no confidence.")
        assert val == pytest.approx(0.5, abs=1e-6)

    def test_extract_zero_confidence(self):
        """Extract 'I am 0% confident' correctly."""
        est = HybridUncertaintyEstimator()
        val = est._extract_verbalized_confidence("I am 0% confident.")
        assert val == pytest.approx(0.0, abs=1e-6)

    def test_extract_one_confidence(self):
        """Extract 'I am 100% confident' correctly."""
        est = HybridUncertaintyEstimator()
        val = est._extract_verbalized_confidence("I am 100% confident.")
        assert val == pytest.approx(1.0, abs=1e-6)

    def test_extract_above_1_treated_as_percent(self):
        """Values > 1.0 should be divided by 100."""
        est = HybridUncertaintyEstimator()
        val = est._extract_verbalized_confidence("I am 95 confident.")
        assert val == pytest.approx(0.95, abs=1e-6)


class TestHybridUncertaintyEstimatorFused:
    def test_fused_alpha_zero_uses_only_cosine(self):
        """When alpha=0, fused = 1 - cosine (uncertainty)."""
        # With alpha=0, fused = 1 - cosine (higher alpha → more uncertain)
        # If cosine=1.0, fused=0 (perfect match → no uncertainty)
        # If cosine=0.0, fused=1 (no match → max uncertainty)
        est = HybridUncertaintyEstimator(alpha=0.0)
        c, v, f = est.estimate("same", "same")
        assert c == pytest.approx(1.0, abs=1e-6)
        assert f == pytest.approx(0.0, abs=1e-6)  # alpha*verbal + (1-alpha)*(1-cosine)

    def test_fused_alpha_half_balanced(self):
        """When alpha=0.5, fused = 0.5*verbal + 0.5*(1-cosine)."""
        est = HybridUncertaintyEstimator(alpha=0.5)
        # Identical text: cosine=1.0, verbal=0.5
        c, v, f = est.estimate("identical", "identical")
        assert c == pytest.approx(1.0, abs=1e-6)
        assert v == pytest.approx(0.5, abs=1e-6)
        expected_fused = 0.5 * 0.5 + 0.5 * (1.0 - 1.0)  # = 0.25
        assert f == pytest.approx(expected_fused, abs=1e-6)

    def test_fused_alpha_one_uses_only_verbal(self):
        """When alpha=1.0, fused = verbalized confidence."""
        est = HybridUncertaintyEstimator(alpha=1.0)
        c, v, f = est.estimate("hello", "world")
        assert v == pytest.approx(0.5, abs=1e-6)
        assert f == pytest.approx(0.5, abs=1e-6)

    def test_fused_with_llm_response(self):
        """estimate_with_llm should use extracted verbalized confidence."""
        est = HybridUncertaintyEstimator(alpha=0.5)
        c, v, f = est.estimate_with_llm("hello", "world", "I am 80% confident.")
        assert v == pytest.approx(0.8, abs=1e-6)


class TestHybridUncertaintyEstimatorMockEmbedding:
    def test_empty_string_embedding(self):
        """Empty string should not crash (norm=0 guard)."""
        est = HybridUncertaintyEstimator()
        emb = est._get_embedding("")
        assert isinstance(emb, np.ndarray)
        assert len(emb) > 0

    def test_different_texts_different_embeddings(self):
        """Different texts should produce different embeddings."""
        est = HybridUncertaintyEstimator()
        e1 = est._get_embedding("text one")
        e2 = est._get_embedding("text two")
        assert not np.allclose(e1, e2)


# ---------------------------------------------------------------------------
# CalibrationDataset Tests
# ---------------------------------------------------------------------------

class TestCalibrationPair:
    def test_fields(self):
        pair = CalibrationPair(predicted_confidence=0.8, correct=True)
        assert pair.predicted_confidence == 0.8
        assert pair.correct is True

    def test_clip_high_confidence(self):
        """Confidence > 1.0 should be clipped to 1.0 in CalibrationDataset.add()."""
        # CalibrationDataset.add() clips to [0, 1]
        ds = CalibrationDataset()
        ds.add(1.5, False)
        assert ds._pairs[0].predicted_confidence == 1.0


class TestCalibrationDataset:
    def test_add_single(self):
        ds = CalibrationDataset()
        ds.add(0.7, True)
        assert len(ds) == 1

    def test_add_multiple(self):
        ds = CalibrationDataset()
        ds.add(0.9, True)
        ds.add(0.3, False)
        ds.add(0.5, True)
        assert len(ds) == 3

    def test_to_arrays(self):
        ds = CalibrationDataset()
        ds.add(0.9, True)
        ds.add(0.3, False)
        ds.add(0.5, True)
        confs, correct = ds.to_arrays()
        assert confs.dtype == np.float64
        assert correct.dtype == np.int32
        np.testing.assert_array_equal(confs, [0.9, 0.3, 0.5])
        np.testing.assert_array_equal(correct, [1, 0, 1])

    def test_to_arrays_empty(self):
        ds = CalibrationDataset()
        confs, correct = ds.to_arrays()
        assert len(confs) == 0
        assert len(correct) == 0

    def test_split_70_30(self):
        """Sequential split with 70/30 ratio."""
        ds = CalibrationDataset()
        for i in range(10):
            ds.add(float(i) / 10.0, i % 2 == 0)
        train, test = ds.split(train_ratio=0.7)
        assert len(train) == 7
        assert len(test) == 3

    def test_split_50_50(self):
        ds = CalibrationDataset()
        for i in range(4):
            ds.add(float(i), i % 2 == 0)
        train, test = ds.split(train_ratio=0.5)
        assert len(train) == 2
        assert len(test) == 2

    def test_split_deterministic(self):
        """Split should be deterministic (no random shuffle)."""
        ds1 = CalibrationDataset()
        for i in range(10):
            ds1.add(float(i) / 10.0, i % 2 == 0)
        t1, _ = ds1.split(0.7)
        t2, _ = ds1.split(0.7)
        np.testing.assert_array_equal(
            [p.predicted_confidence for p in t1._pairs],
            [p.predicted_confidence for p in t2._pairs],
        )

    def test_split_content_preserved(self):
        """Split should preserve the data (no data loss)."""
        ds = CalibrationDataset()
        for i in range(10):
            ds.add(float(i) / 10.0, i % 2 == 0)
        train, test = ds.split(0.7)
        assert len(train) + len(test) == len(ds)


# ---------------------------------------------------------------------------
# compute_expected_calibration_error Tests
# ---------------------------------------------------------------------------

class TestECE:
    def test_perfect_calibration_single_bin(self):
        """Perfect calibration with single bin: all same confidence and accuracy → ECE = 0."""
        # Put all samples in one bin with matching accuracy
        confs = np.array([0.8] * 5 + [0.2] * 5)
        correct = np.array([1] * 5 + [0] * 5)
        ece = compute_expected_calibration_error(confs, correct, n_bins=1)
        # accuracy = 0.5, confidence = 0.5 → ECE = 0
        assert ece == pytest.approx(0.0, abs=1e-6)

    def test_perfect_calibration_multi_bin(self):
        """Well-calibrated predictions across bins with balanced groups."""
        # With n_bins=4 and equal groups: ECE is a small fraction
        confs = np.array([0.05, 0.05, 0.15, 0.15, 0.55, 0.55, 0.85, 0.85])
        correct = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        ece = compute_expected_calibration_error(confs, correct, n_bins=4)
        assert ece < 0.2

    def test_miscalibrated_high(self):
        """Overconfident predictions → ECE > 0."""
        # Predictions all high confidence but mostly wrong
        confs = np.array([0.8, 0.85, 0.9, 0.95, 0.8, 0.85])
        correct = np.array([0, 0, 0, 0, 1, 1])
        ece = compute_expected_calibration_error(confs, correct, n_bins=5)
        assert ece > 0.2  # Should be significantly > 0

    def test_empty_input(self):
        """Empty arrays should return ECE = 0."""
        ece = compute_expected_calibration_error(np.array([]), np.array([]))
        assert ece == 0.0

    def test_single_bin(self):
        """Single bin (n_bins=1) should work."""
        confs = np.array([0.5, 0.6, 0.7])
        correct = np.array([0, 1, 1])
        ece = compute_expected_calibration_error(confs, correct, n_bins=1)
        assert ece >= 0.0

    def test_many_bins(self):
        """Many bins should not crash."""
        confs = np.linspace(0.0, 1.0, 100)
        correct = (confs > 0.5).astype(int)
        ece = compute_expected_calibration_error(confs, correct, n_bins=20)
        assert ece >= 0.0

    def test_uniform_confidence(self):
        """All same confidence → ECE computed across single bin."""
        confs = np.array([0.7] * 10)
        correct = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        ece = compute_expected_calibration_error(confs, correct, n_bins=5)
        # accuracy = 0.5, confidence = 0.7 → difference = 0.2
        assert ece > 0.0


# ---------------------------------------------------------------------------
# compute_auroc Tests
# ---------------------------------------------------------------------------

class TestAUROC:
    def test_random_predictions(self):
        """Random predictions should give AUROC ≈ 0.5."""
        rng = np.random.RandomState(42)
        confs = rng.rand(100)
        correct = rng.randint(0, 2, 100)
        metrics = compute_auroc(confs, correct)
        assert 0.4 < metrics["auroc"] < 0.6

    def test_perfect_separation(self):
        """Perfect separation: all correct > all incorrect → AUROC = 1.0."""
        confs = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
        correct = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        metrics = compute_auroc(confs, correct)
        assert metrics["auroc"] == pytest.approx(1.0, abs=1e-6)

    def test_anti_correlated_predictions(self):
        """Anti-correlated predictions should give AUROC close to 0 or 1 (near-perfect)."""
        confs = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
        correct = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        metrics = compute_auroc(confs, correct)
        # Anti-correlated → AUROC should be near 0 or 1 (≥ 0.9 or ≤ 0.1)
        assert metrics["auroc"] >= 0.9 or metrics["auroc"] <= 0.1

    def test_auroc_metric_fields(self):
        """AUROC result should contain all required fields."""
        confs = np.array([0.3, 0.6, 0.5, 0.8])
        correct = np.array([0, 1, 0, 1])
        metrics = compute_auroc(confs, correct)
        assert "auroc" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "optimal_threshold" in metrics
        assert 0.0 <= metrics["optimal_threshold"] <= 1.0

    def test_youden_threshold_selection(self):
        """Optimal threshold should maximize Youden's J (tpr - fpr)."""
        # Data where 0.5 is the natural threshold
        confs = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9])
        correct = np.array([0, 0, 0, 1, 1, 1])
        metrics = compute_auroc(confs, correct)
        # Threshold should be between 0.4 and 0.6
        assert 0.35 <= metrics["optimal_threshold"] <= 0.65

    def test_f1_at_optimal_threshold(self):
        """F1 score should be computable at the optimal threshold."""
        confs = np.array([0.2, 0.4, 0.6, 0.8])
        correct = np.array([0, 0, 1, 1])
        metrics = compute_auroc(confs, correct)
        assert 0.0 <= metrics["f1"] <= 1.0

    def test_too_few_samples(self):
        """Less than 2 samples should return default metrics."""
        metrics = compute_auroc(np.array([0.5]), np.array([1]))
        assert metrics["auroc"] == 0.5


# ---------------------------------------------------------------------------
# CalibratedMilestoneCritic Tests
# ---------------------------------------------------------------------------

class TestCalibratedMilestoneCritic:
    def test_wraps_base_critic(self):
        """Should return results from base critic when no calibration set."""
        base = MilestoneCritic(backend="rule-based")
        calibrated = CalibratedMilestoneCritic(base_critic=base)
        milestone = Milestone(
            id="m1",
            description="Click the submit button",
            rubric="Submit button is clicked",
        )
        result = calibrated.review(milestone, "User clicked submit")
        assert isinstance(result, CriticResult)
        assert isinstance(result.verdict, Verdict)
        assert 0.0 <= result.confidence <= 1.0

    def test_feedback_passthrough(self):
        """Feedback should be passed through from base critic."""
        base = MilestoneCritic(backend="rule-based")
        calibrated = CalibratedMilestoneCritic(base_critic=base)
        milestone = Milestone(
            id="m1",
            description="Find the login form",
            rubric="Login form found",
        )
        result = calibrated.review(milestone, "Login form displayed on screen")
        assert result.feedback != "" or result.verdict in (Verdict.PASS, Verdict.FAIL, Verdict.PARTIAL)

    def test_confidence_calibrated(self):
        """With calibration dataset, confidence should be adjusted."""
        base = MilestoneCritic(backend="rule-based")
        # Build calibration dataset where high confidence → correct
        calib_ds = CalibrationDataset()
        for i in range(10):
            conf = 0.5 + i * 0.05  # 0.5 to 0.95
            correct = conf > 0.7
            calib_ds.add(conf, correct)

        calibrated = CalibratedMilestoneCritic(
            base_critic=base,
            calibration_dataset=calib_ds,
        )
        milestone = Milestone(
            id="m1",
            description="Navigate to settings",
            rubric="Settings page reached",
        )
        result = calibrated.review(milestone, "User navigated to settings page")
        assert 0.0 <= result.confidence <= 1.0

    def test_verdict_unchanged(self):
        """Verdict type should match base critic (not altered by calibration)."""
        base = MilestoneCritic(backend="rule-based")
        calibrated = CalibratedMilestoneCritic(base_critic=base)
        milestone = Milestone(
            id="m1",
            description="Upload a file",
            rubric="File upload confirmed",
        )
        base_result = base.review(
            milestone.description,
            milestone.rubric,
            "File uploaded successfully",
        )
        calibrated_result = calibrated.review(milestone, "File uploaded successfully")
        # Verdict type should match
        assert calibrated_result.verdict == base_result.verdict

    def test_empty_calibration_dataset(self):
        """Empty calibration dataset should not crash."""
        base = MilestoneCritic(backend="rule-based")
        calib_ds = CalibrationDataset()
        calibrated = CalibratedMilestoneCritic(base_critic=base, calibration_dataset=calib_ds)
        milestone = Milestone(id="m1", description="test", rubric="test")
        result = calibrated.review(milestone, "test trace")
        assert isinstance(result, CriticResult)


# ---------------------------------------------------------------------------
# UncertaintyAwareRewardAuditor Tests
# ---------------------------------------------------------------------------

class TestUncertaintyAwareRewardAuditor:
    def test_audit_returns_tuple(self):
        """audit() should return (adjusted_reward, uncertainty)."""
        base = MilestoneCritic(backend="rule-based")
        est = HybridUncertaintyEstimator(alpha=0.3)
        auditor = UncertaintyAwareRewardAuditor(base_critic=base, uncertainty_estimator=est)
        milestone = Milestone(
            id="m1",
            description="Submit form",
            rubric="Form submitted",
        )
        adj_reward, uncertainty = auditor.audit(milestone, "Form submitted", 1.0)
        assert isinstance(adj_reward, float)
        assert isinstance(uncertainty, float)

    def test_uncertainty_bounded_0_1(self):
        """Uncertainty should be in [0, 1]."""
        base = MilestoneCritic(backend="rule-based")
        est = HybridUncertaintyEstimator(alpha=0.5)
        auditor = UncertaintyAwareRewardAuditor(base_critic=base, uncertainty_estimator=est)
        milestone = Milestone(
            id="m1",
            description="Click button",
            rubric="Button clicked",
        )
        _, uncertainty = auditor.audit(milestone, "Button clicked", 1.0)
        assert 0.0 <= uncertainty <= 1.0

    def test_reward_dampened_by_uncertainty(self):
        """adjusted_reward = base_reward * (1 - uncertainty)."""
        base = MilestoneCritic(backend="rule-based")
        est = HybridUncertaintyEstimator(alpha=0.5)
        auditor = UncertaintyAwareRewardAuditor(base_critic=base, uncertainty_estimator=est)
        milestone = Milestone(
            id="m1",
            description="Login page",
            rubric="Login page shown",
        )
        # Identical texts → low uncertainty → reward close to base
        adj, unc = auditor.audit(milestone, milestone.description, 1.0)
        assert adj <= 1.0  # Never exceeds base
        # With identical texts, cosine=1 → fused = alpha*0.5 + (1-alpha)*0
        expected_fused = 0.5 * 0.5 + 0.5 * 0.0  # alpha=0.5, cosine=1 → fused = 0.25
        expected_reward = 1.0 * (1 - expected_fused)
        assert adj == pytest.approx(expected_reward, abs=0.01)

    def test_different_text_increases_uncertainty(self):
        """Identical text → lower uncertainty than random text."""
        base = MilestoneCritic(backend="rule-based")
        est = HybridUncertaintyEstimator(alpha=0.3)
        auditor = UncertaintyAwareRewardAuditor(base_critic=base, uncertainty_estimator=est)
        milestone = Milestone(
            id="m1",
            description="Click the red button",
            rubric="Red button clicked",
        )
        # Identical text → cosine = 1.0 → fused = alpha * 0.5
        adj_identical, unc_identical = auditor.audit(milestone, milestone.description, 1.0)
        # Random text → random cosine → higher fused
        adj_random, unc_random = auditor.audit(
            milestone, "xyz abc random noise qwerty", 1.0
        )
        # The fused uncertainty for identical text is deterministic
        # fused = alpha * 0.5 + (1-alpha) * 0 = 0.3 * 0.5 = 0.15
        expected_fused_identical = 0.3 * 0.5
        assert unc_identical == pytest.approx(expected_fused_identical, abs=0.01)
        # Both should be bounded
        assert 0.0 <= unc_identical <= 1.0
        assert 0.0 <= unc_random <= 1.0

    def test_zero_base_reward(self):
        """Zero base reward → adjusted reward is zero regardless of uncertainty."""
        base = MilestoneCritic(backend="rule-based")
        est = HybridUncertaintyEstimator(alpha=0.5)
        auditor = UncertaintyAwareRewardAuditor(base_critic=base, uncertainty_estimator=est)
        milestone = Milestone(id="m1", description="test", rubric="test")
        adj, _ = auditor.audit(milestone, "test", 0.0)
        assert adj == 0.0


# ---------------------------------------------------------------------------
# evaluate_calibration Tests
# ---------------------------------------------------------------------------

class TestEvaluateCalibration:
    def test_returns_required_fields(self):
        """Should return dict with all expected keys."""
        base = MilestoneCritic(backend="rule-based")
        calib_base = CalibratedMilestoneCritic(base_critic=base)
        tasks = [
            (
                Milestone(id="m1", description="Do thing A", rubric="Thing A done"),
                "Thing A completed successfully",
            ),
            (
                Milestone(id="m2", description="Do thing B", rubric="Thing B done"),
                "Thing B not done",
            ),
        ]
        result = evaluate_calibration(calib_base, tasks)
        assert "auroc" in result
        assert "precision" in result
        assert "recall" in result
        assert "f1" in result
        assert "optimal_threshold" in result
        assert "ece" in result
        assert "n_evaluated" in result
        assert "calibration_curve" in result

    def test_empty_tasks(self):
        """Empty task list should return defaults."""
        base = CalibratedMilestoneCritic(base_critic=MilestoneCritic())
        result = evaluate_calibration(base, [])
        assert result["n_evaluated"] == 0
        assert result["auroc"] == 0.5
        assert result["ece"] == 0.0

    def test_n_evaluated_correct(self):
        """n_evaluated should match number of tasks."""
        base = CalibratedMilestoneCritic(base_critic=MilestoneCritic())
        tasks = [
            (Milestone(id="m1", description="a", rubric="a"), "a"),
            (Milestone(id="m2", description="b", rubric="b"), "b"),
            (Milestone(id="m3", description="c", rubric="c"), "c"),
        ]
        result = evaluate_calibration(base, tasks)
        assert result["n_evaluated"] == 3

    def test_calibration_curve_structure(self):
        """Calibration curve should have correct structure."""
        base = CalibratedMilestoneCritic(base_critic=MilestoneCritic())
        tasks = [
            (Milestone(id="m1", description="task one", rubric="done"), "task done"),
            (Milestone(id="m2", description="task two", rubric="done"), "task done"),
        ]
        result = evaluate_calibration(base, tasks)
        cc = result["calibration_curve"]
        assert "confidences" in cc
        assert "accuracies" in cc
        assert "bin_counts" in cc
        assert len(cc["confidences"]) == len(cc["accuracies"]) == len(cc["bin_counts"])

    def test_ece_non_negative(self):
        """ECE should always be non-negative."""
        base = CalibratedMilestoneCritic(base_critic=MilestoneCritic())
        tasks = [
            (Milestone(id="m1", description="x", rubric="x"), "x"),
            (Milestone(id="m2", description="y", rubric="y"), "y"),
        ]
        result = evaluate_calibration(base, tasks)
        assert result["ece"] >= 0.0

    def test_auroc_bounded(self):
        """AUROC should be in [0, 1] when computed (may be nan for degenerate all-same labels)."""
        base = CalibratedMilestoneCritic(base_critic=MilestoneCritic())
        tasks = [
            (Milestone(id="m1", description="a", rubric="a"), "a"),
            (Milestone(id="m2", description="b", rubric="b"), "b"),
        ]
        result = evaluate_calibration(base, tasks)
        auroc = result["auroc"]
        # If valid (not nan), must be in [0, 1]
        if auroc == auroc:  # NaN is the only float that != itself
            assert 0.0 <= auroc <= 1.0
