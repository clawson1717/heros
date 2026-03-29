# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-03-29

### Added

#### Core Framework
- **`src/heros/__init__.py`** — Package init with `__version__ = "0.1.0"`, public API exports
- **`src/heros/planner.py`** — MiRA-style `SubgoalPlanner` with LLM and rule-based backends; `Milestone` dataclass with id, description, rubric, expected_output, order; task-type detection (code_generation, web_navigation, data_analysis, reasoning, general); configurable planning depth (1–10 subgoals)
- **`src/heros/critic.py`** — OS-Themis-style `MilestoneCritic` with LLM and rule-based backends; `Verdict` enum (PASS, FAIL, PARTIAL); `CriticResult` dataclass; `RewardAuditor` producing dense rewards (0.0–1.0)
- **`src/heros/buffer.py`** — HER-style `HindsightBuffer` (capacity-based trajectory storage) and `HindsightTrajectory`; hindsight sampling and relabeling of failed subgoals with unmet rubrics as pseudo-goals
- **`src/heros/trainer.py`** — `HindsightTrainer` with BC-style policy update and OpenAI API support; configurable hindsight ratio (e.g. 30% of batches from hindsight buffer)
- **`src/heros/core.py`** — `PPOTrainer` combining standard + hindsight experience; `HeRoSEnv` environment wrapper with milestone tracking; `HeRoSAgent` integrating planner + actor + critic + hindsight buffer
- **`src/heros/logging_utils.py`** — `TrainingLogger` tracking milestone success rates, hindsight utilization, reward curves

#### Evaluation
- **`src/heros/benchmark.py`** — `WebArenaLiteBenchmark` (lightweight web navigation, 5+ tasks); `WebTask`, `WebAction`, `EvaluationAction` dataclasses; `MockWebEnv` simulated environment; `SimulatedDOM` for DOM parsing and element state tracking
- **`src/heros/baseline_agent.py`** — `BaselineAgent` (raw LLM without milestones or hindsight)
- **`src/heros/heros_agent_wrapper.py`** — `HeRoSWrappedAgent` HeRoS agent wrapper with hindsight toggle
- **`src/heros/evaluator.py`** — `HeRoSEvaluator` with full evaluation harness; `EvaluationResult` dataclass with per-milestone tracking; `compute_metrics()`, `compare_agents()`, `run_evaluation()`
- **`eval/run_evaluation.py`** — CLI for WebArena-Lite benchmark evaluation
- **`eval/run_self_improvement_eval.py`** — CLI for self-improvement benchmark evaluation

#### Uncertainty (Step 9)
- **`src/heros/uncertainty.py`** — `HybridUncertaintyEstimator` (two-sample: embedding cosine + LLM verbalized confidence); `CalibrationDataset`, `CalibrationPair`, `CalibrationCurve`, `ExpectedCalibrationError`; `compute_auroc()` with Youden's J optimal threshold; `CalibratedMilestoneCritic` with Platt scaling; `UncertaintyAwareRewardAuditor` with uncertainty-based reward dampening; `evaluate_calibration()` pipeline

#### Self-Improvement (Step 10)
- **`src/heros/self_improver.py`** — `TestTimeSelfImprover` (self-improving agent using failed subgoal hindsight); `InferenceEngine` with optional self-improvement; `EpisodeMetrics`, `PolicyUpdateResult`, `SelfImprovementResult`, `InferenceEpisodeResult`, `BatchInferenceResult` dataclasses; lightweight BC-style self-play updates (simulated when no API key)

#### Boundary Enforcement (Step 11)
- **`src/heros/boundary_enforcer.py`** — `BoundaryConstraint`, `BoundaryState`, `ConstraintStatus`, `BoxRegion` dataclasses; `BoundaryEnforcer` with strict/soft/advisory constraint enforcement and audit trail; `MemoryGroundingLayer` for constraint memory across trajectory steps with natural language reminders; `HeRoSBoundaryIntegration` wrapping HeRoSAgent with boundary enforcement; `BoundaryEvaluator` computing boundary_prevention_rate and constraint_violation_rate; `ConstraintCheckResult`, `EnforcedAction` dataclasses

#### Interpretability (Step 8)
- **`src/heros/interpretability.py`** — `InterpretabilityLogger` logging every milestone decision with critic reasoning; functional interchangeability check for action-space consistency; `MilestoneAuditTrail` (JSON export per episode); `HindsightBufferAnalyzer` visualizing hindsight buffer composition over training; `TrainingVisualizer` for milestone success rates, reward curves, hindsight utilization

#### Tests
- **`tests/test_skeleton.py`** — 10 passing project structure tests
- **`tests/test_planner.py`** — 34 passing tests for MiRA-style subgoal decomposition
- **`tests/test_critic.py`** — 40 passing tests for OS-Themis-style milestone critic
- **`tests/test_buffer.py`** — 50 combined passing tests for hindsight buffer and trainer
- **`tests/test_core.py`** — 88 passing tests for core RL loop
- **`tests/test_evaluator.py`** — 89 passing tests for WebArena-Lite evaluation harness
- **`tests/test_interpretability.py`** — 30+ passing tests for interpretability components
- **`tests/test_uncertainty.py`** — 60 passing tests for hybrid uncertainty estimation
- **`tests/test_self_improver.py`** — 50 passing tests for test-time self-improvement
- **`tests/test_boundary_enforcer.py`** — 91 passing tests for boundary enforcement

#### Configuration & Infrastructure
- **`pyproject.toml`** — PEP 517/518 build system with setuptools, editable install via `pip install -e .`
- **`requirements.txt`** — Dependencies: transformers, torch, numpy, pytest, pathlib2
- **`Dockerfile`** — Containerized development environment
- **`configs/eval_config.yaml`** — Evaluation harness configuration
- **`scripts/`** — Training and utility scripts directory

#### Documentation
- **`README.md`** — Full-featured README with architecture diagram, quick start, benchmark results, roadmap, references
- **`CHANGELOG.md`** — This changelog (v0.1.0 initial release)
- **`CONTRIBUTING.md`** — Contributing guidelines
- **`docs/arxiv-draft.md`** — ArXiv technical report draft

### Changed

- **`README.md`** — Overwritten with comprehensive documentation including architecture diagram, quick start guide, benchmark results, full roadmap, and references
- **`PLAN.md`** — Updated to reflect completion of all 12 steps

### Fixed

- **`src/heros/benchmark.py`** — Restored correct benchmark, wrapper, and evaluator files after accidental truncation in Step 10

### Security

- All LLM-backed components include rule-based fallback when no API key is available, ensuring reproducibility and testability without external dependencies
