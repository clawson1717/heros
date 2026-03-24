# HeRoS — Hindsight-driven Reinforcement with Subgoal Milestones

> Combining HeRL, OS-Themis, and MiRA for self-improving LLM agents that plan, verify, fail, and learn.

[![Project Stage: Planning](https://img.shields.io/badge/stage-planning-blue)](#)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](#)

---

## 🔗 TL;DR

**HeRoS** fuses three ArXiv techniques into a single self-improving agent framework:

1. **HeRL** — Failed trajectories with unmet rubrics become *hindsight experience* that expands the policy's exploration beyond its current response distribution.
2. **OS-Themis** — A multi-agent *milestone critic* decomposes every task into verifiable checkpoints with RL reward auditing.
3. **MiRA** — *Subgoal-driven planning* with dense milestone reward signals enables structured inference-time reasoning.

**Result:** An agent that decomposes tasks into milestones, executes each one, has its work critically reviewed, learns from every failure, and self-improves at test time.

---

## 📐 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        HeRoS Framework                        │
│                                                              │
│   Task ──▶ ┌──────────────┐                                  │
│            │    Planner   │  MiRA-style subgoal             │
│            │  (Decompose) │  decomposition into milestones   │
│            └──────┬───────┘                                  │
│                   │ milestone list (id, rubric, description)│
│                   ▼                                          │
│            ┌──────────────┐                                  │
│            │    Actor     │  LLM executes each milestone      │
│            │   (LLM/Agent)│  producing execution traces       │
│            └──────┬───────┘                                  │
│                   │ execution traces                         │
│                   ▼                                          │
│            ┌──────────────┐      OS-Themis-style:             │
│            │ Milestone    │      critic reviews each trace   │
│            │   Critic     │      against rubric → verdict +   │
│            │  (OS-Themis) │      reward audit signal         │
│            └──────┬───────┘                                   │
│                   │ verdicts (pass/fail/partial)             │
│          ┌────────┴────────┐                                │
│          ▼                  ▼                                │
│   ┌────────────┐    ┌──────────────┐                         │
│   │  Reward    │    │  Hindsight   │  HeRL-style:            │
│   │  Auditor   │    │   Buffer     │  failed trajectories     │
│   │  (RL Sig.) │    │  (HeRL)      │  with unmet rubrics      │
│   └─────┬──────┘    └──────┬───────┘  → policy improvement   │
│         │                 │                                  │
│         └────────┬────────┘                                  │
│                  ▼                                           │
│           ┌────────────┐                                     │
│           │   Policy   │  PPO / SFT from standard +          │
│           │  Updater   │  hindsight experience               │
│           └────────────┘                                     │
│                                                              │
│   [Test-time Self-Improvement: Failed subgoals from the      │
│    current episode are added to the hindsight buffer —      │
│    no new environment interactions needed.]                 │
└─────────────────────────────────────────────────────────────┘
```

### The Three-Layer Feedback Loop

| Layer | Component | Source Paper | Role |
|---|---|---|---|
| **Planning** | Subgoal Decomposer | MiRA | Breaks task into ordered, verifiable milestones |
| **Verification** | Milestone Critic + Auditor | OS-Themis | Reviews traces, produces audited reward signals |
| **Learning** | Hindsight Buffer + Policy | HeRL | Stores failed rubric-missed trajectories, trains policy beyond current distribution |

---

## 🚀 Why HeRoS?

Current LLM agents fail in two ways that existing frameworks don't address:

1. **Silent mid-task failures** — An agent executing a 10-step plan fails at step 4, but has no mechanism to learn from *why* it failed. It just continues (or stops) with no feedback signal.

2. **Sparse reward only at task completion** — Standard RL only provides reward when the final task is done. For tasks with 10 steps, this means 9 failures produce zero learning signal.

HeRoS solves both by:
- **Verifiable milestones** (OS-Themis) — Every step has a pass/fail criterion
- **Dense reward signals** (MiRA) — Every milestone produces a reward
- **Hindsight from failure** (HeRL) — Failed milestones are relabeled and used for policy improvement

---

## 📋 Implementation Roadmap

| Step | Task | Status |
|:----:|------|:------:|
| 1 | Planning — architecture, combination design | ✅ DONE |
| 2 | Project skeleton & dependencies | 🔲 TODO |
| 3 | MiRA-style subgoal decomposition module | 🔲 TODO |
| 4 | OS-Themis-style milestone critic agent | 🔲 TODO |
| 5 | HeRL-style hindsight experience buffer | 🔲 TODO |
| 6 | Core RL training loop (PPO + hindsight) | 🔲 TODO |
| 7 | WebArena-Lite / MiniWoB evaluation harness | 🔲 TODO |
| 8 | Interpretability & reward audit trail logging | 🔲 TODO |
| 9 | Hybrid uncertainty estimation for critic calibration | 🔲 TODO |
| 10 | Test-time self-improvement mode | 🔲 TODO |
| 11 | Box Maze boundary enforcement (optional extension) | 🔲 TODO |
| 12 | Paper, documentation & open source release | 🔲 TODO |

---

## 🧩 Technique Combination Details

### HeRL — Hindsight Experience Guided RL
From the ArXiv paper on HeRL: failed trajectories with *unmet rubrics* are re-used as positive training examples by imagining "what if this unmet subgoal was actually the intended goal?" This guides the policy to explore desired responses beyond its current distribution, enabling test-time self-improvement.

**Used in HeRoS for:** The learning signal. Every failed milestone generates a hindsight example that expands the policy.

### OS-Themis — Multi-agent Critic with Milestone Decomposition
From the ArXiv paper on OS-Themis: trajectories are decomposed into verifiable milestones with a dedicated critic/reviewer agent that audits RL rewards. Achieved +10.3% on AndroidWorld.

**Used in HeRoS for:** The structure. Every task is decomposed into milestone checkpoints, each with an associated rubric verified by the critic.

### MiRA — Milestoning RL Enhanced Agent
From the ArXiv paper on MiRA: subgoal-driven RL boosting Gemma3-12B from 6.4% → 43.0% on WebArena-Lite through inference-time planning and dense milestone rewards.

**Used in HeRoS for:** The planning. MiRA-style inference-time subgoal decomposition generates the milestone sequence before execution.

---

## 📁 Project Structure

```
heros/
├── README.md
├── LICENSE
├── requirements.txt
├── Dockerfile
├── src/
│   └── heros/
│       ├── __init__.py
│       ├── planner.py          # MiRA-style subgoal decomposition
│       ├── actor.py             # LLM actor agent
│       ├── critic.py            # OS-Themis-style milestone critic
│       ├── hindsight_buffer.py  # HeRL-style hindsight experience
│       ├── reward_auditor.py    # OS-Themis reward auditing
│       ├── policy_updater.py    # PPO/SFT policy training
│       ├── env.py               # HeRoS environment wrapper
│       └── heros_agent.py       # Main agent class
├── configs/
│   ├── base.yaml
│   ├── webarena_lite.yaml
│   └── self_improve.yaml
├── eval/
│   ├── harness.py              # WebArena-Lite / MiniWoB harness
│   ├── metrics.py              # Milestone hit rate, task comp, etc.
│   └── baselines.py            # Raw LLM, MiRA-only, etc.
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── self_improve.py
├── notebooks/
│   └── demo.ipynb
└── tests/
    ├── test_planner.py
    ├── test_critic.py
    └── test_hindsight.py
```

---

## 📊 Expected Outcomes

Based on the individual paper results:

| Baseline | Task Completion |
|---|---|
| Raw LLM agent | ~6–15% |
| MiRA only (WebArena-Lite) | 43.0% |
| OS-Themis only (AndroidWorld) | +10.3% over baseline |
| **HeRoS (combined)** | **Expected >50%** (synergistic gain from combined techniques) |

HeRoS should exceed MiRA's performance because:
- OS-Themis critics catch false-positive milestone completions (reducing reward noise)
- HeRL hindsight experience accelerates learning on hard subgoals
- Dense milestone rewards prevent reward sparsity

---

## 🔬 Key Research Questions

1. Does hindsight experience from *failed subgoals* improve policy more than hindsight from *failed full tasks*?
2. How does the critic's verification quality interact with hindsight buffer quality?
3. Can test-time self-improvement (no new env interactions) achieve meaningful gains?
4. How does hybrid uncertainty estimation affect critic calibration?

---

## 📖 Papers Referenced

- **HeRL**: "HeRL: Hindsight Experience Guided Reinforcement Learning for LLMs" — ArXiv 2026-03-23
- **OS-Themis**: "OS-Themis: Multi-agent Critic Framework for GUI Agent RL" — arXiv:2603.19191
- **MiRA**: "MiRA: Milestoning your Reinforcement Learning Enhanced Agent" — ArXiv 2026-03-23
- **Box Maze** (optional): arXiv:2603.19182
- **Hybrid Uncertainty Estimation** (optional): arXiv:2603.19118

---

## 📝 License

MIT License. See `LICENSE` file.

---

*HeRoS: Plan with milestones. Execute. Criticize. Learn from failure. Repeat.*
