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

### 📋 Upcoming Steps

- Step 3: TBD
- Step 4: TBD
- Step 5: TBD
