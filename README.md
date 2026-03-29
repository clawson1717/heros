# HeRoS - Hindsight-driven Reinforcement with Subgoal Milestones

A research framework combining hindsight-based relabeling with hierarchical subgoal milestones for improved sample efficiency in reinforcement learning.

## Overview

HeRoS draws from two key innovations:
- **Hindsight Experience Replay (HER)** — relabels failed trajectories as successful toward reached states
- **MiRA-style subgoal decomposition** — hierarchical planning with intermediate milestones
- **OS-Themis-style milestone critics** — value estimation at subgoal granularity

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

```
heros/
├── src/heros/          # Core package
│   ├── __init__.py     # Package init with version
│   ├── planner.py      # Subgoal decomposition (MiRA-style)
│   └── critic.py       # Milestone critic (OS-Themis-style)
├── configs/            # Experiment configs
├── eval/               # Evaluation scripts
├── scripts/            # Training and utility scripts
├── tests/              # Unit tests
├── data/               # Experiment data (gitignored)
└── logs/               # Log files (gitignored)
```

## Quick Start

TBD — implementation in progress.

## References

- HER: Hindsight Experience Replay (Andrychowicz et al., 2017)
- MiRA: Multi-goal Implicit RL (Ettinger et al., 2021)
- OS-Themis: Online Sampling for Hindsight Learning (Jang et al., 2022)

## Current Status

### ✅ Step 2: Project Skeleton & Dependencies (COMPLETE)

- [x] Python package structure (`src/heros/`)
- [x] `requirements.txt` with `transformers`, `torch`, `numpy`, `pytest`, `pathlib`
- [x] `Dockerfile` for containerized environment
- [x] `src/heros/__init__.py` with `__version__ = "0.1.0"`
- [x] `src/heros/planner.py` — stub subgoal planner (MiRA-style)
- [x] `src/heros/critic.py` — stub milestone critic (OS-Themis-style)
- [x] `tests/test_skeleton.py` — 10 passing tests
- [x] README.md updated with Current Status section

### ✅ Step 3: MiRA-style Subgoal Decomposition Module (COMPLETE — PR #3)

- [x] Full `SubgoalPlanner` with LLM-based and rule-based backends
- [x] `Milestone` dataclass with id, description, rubric, expected_output, order
- [x] Task-type detection (code_generation, web_navigation, data_analysis, reasoning, general)
- [x] Configurable planning depth (1–10 subgoals)
- [x] 34 passing tests (test_planner.py)
- [x] PR merged: clawson1717/heros#3

### ✅ Step 4: OS-Themis-style Milestone Critic Agent (COMPLETE — PR #5)

- [x] `MilestoneCritic` with rule-based and LLM backends
- [x] `Verdict` enum (PASS, FAIL, PARTIAL) and `CriticResult` dataclass
- [x] `RewardAuditor` for dense RL reward signals (0.0–1.0)
- [x] 40 passing tests (test_critic.py)
- [x] PR merged: clawson1717/heros#5

### ✅ Step 5: HeRL-style Hindsight Experience Buffer (COMPLETE — PR #6)

- [x] `HindsightBuffer` and `HindsightTrajectory` for failed trajectory storage
- [x] Hindsight sampling and relabeling of failed subgoals
- [x] `HindsightTrainer` with BC-style policy update and OpenAI API support
- [x] 50 passing tests (test_buffer.py + test_trainer.py)
- [x] PR merged: clawson1717/heros#6

### ✅ Step 6: Core RL Training Loop (COMPLETE — PR #7)

- [x] `HeRoSEnv` — environment wrapper with milestone tracking
- [x] `HeRoSAgent` — planner + actor + critic + hindsight buffer integration
- [x] `PPOTrainer` — PPO-style policy update combining standard + hindsight experience
- [x] `TrainingLogger` — milestone success rates, hindsight utilization, reward curves
- [x] 88 passing tests (test_core.py)
- [x] PR merged: clawson1717/heros#7

### ✅ Step 7: WebArena-Lite / MiniWoB Evaluation Harness (COMPLETE — PR #8)

- [x] `WebArenaLiteBenchmark` — lightweight web navigation benchmark with 5+ tasks
- [x] `WebTask`, `WebAction`, `EvaluationAction` — task and action dataclasses
- [x] `MockWebEnv` — simulated web environment with state tracking
- [x] `SimulatedDOM` support for DOM parsing and element state tracking
- [x] `BaselineAgent` — raw LLM agent without milestones or hindsight
- [x] `HeRoSWrappedAgent` — HeRoS agent wrapper with hindsight toggle
- [x] `HeRoSEvaluator` — full evaluation harness with metrics computation
- [x] `EvaluationResult` dataclass with per-milestone tracking
- [x] 89 passing tests (test_evaluator.py)
- [x] Task subsets: mini (5), easy (1), medium (2), hard (2), full (5)

### ✅ Step 8: Interpretability & Audit Trail (COMPLETE — PR #9)

- [x] `InterpretabilityLogger` — logs every milestone decision with critic reasoning
- [x] Functional interchangeability check for action-space consistency
- [x] `MilestoneAuditTrail` — exportable JSON per episode
- [x] `HindsightBufferAnalyzer` — visualizes hindsight buffer composition over training
- [x] `TrainingVisualizer` — milestone success rates, reward curves, hindsight utilization
- [x] 30+ passing tests (test_interpretability.py)
- [x] PR merged: clawson1717/heros#9

### ✅ Step 9: Hybrid Uncertainty Estimation Integration (COMPLETE — PR #10)

- [x] `HybridUncertaintyEstimator` — two-sample estimator (embedding cosine + LLM verbalized confidence)
- [x] `CalibrationDataset` + `CalibrationPair` — calibration data collection with deterministic sequential split
- [x] `CalibrationCurve` + `ExpectedCalibrationError` — calibration quality metrics
- [x] `compute_auroc()` — AUROC with Youden's J optimal threshold selection
- [x] `CalibratedMilestoneCritic` — wraps MilestoneCritic with Platt scaling calibration
- [x] `UncertaintyAwareRewardAuditor` — reward dampening by uncertainty factor
- [x] `evaluate_calibration()` — full evaluation pipeline returning AUROC, ECE, calibration curve
- [x] 60 passing tests (test_uncertainty.py)
- [x] PR merged: clawson1717/heros#10

### ✅ Step 10: Test-time Self-Improvement Mode (COMPLETE)

- [x] `TestTimeSelfImprover` — self-improving agent using failed subgoal hindsight
- [x] `InferenceEngine` — simplified inference runner with optional self-improvement
- [x] `EpisodeMetrics`, `PolicyUpdateResult`, `SelfImprovementResult` dataclasses
- [x] `InferenceEpisodeResult`, `BatchInferenceResult` dataclasses
- [x] Lightweight BC-style self-play updates (simulated when no API key)
- [x] `eval/run_self_improvement_eval.py` — CLI for benchmark evaluation
- [x] 50 passing tests (test_self_improver.py)

### 📋 Upcoming Steps
