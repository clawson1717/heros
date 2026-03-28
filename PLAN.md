# HeRoS — Hindsight-driven Reinforcement with Subgoal Milestones

## Step 1 = [DONE] Planning Complete

---

## Project Overview

**What problem does this solve?**
LLM-based agents performing complex multi-step tasks (GUI automation, code generation, autonomous web navigation) currently fail silently mid-task. When a subgoal fails, the agent rarely learns from it — no hindsight experience is captured, no verifiable milestone signals guide the policy, and the agent cannot self-correct at test time.

**What is the novel combination?**
Three techniques from distinct ArXiv papers are fused in a way that has never been done:

1. **HeRL** (Hindsight Experience Guided RL for LLMs) — Uses failed trajectories with *unmet rubrics* as hindsight experience to guide policy exploration beyond current response distribution.
2. **OS-Themis** (Multi-agent Critic with Milestone Decomposition) — Decomposes agent trajectories into verifiable milestones; uses a review/critic mechanism for RL reward auditing.
3. **MiRA** (Milestoning RL Enhanced Agent) — Subgoal-driven RL framework with dense milestone-based reward signals enabling inference-time planning.

**Why is this novel?**
No prior work combines rubric-grounded hindsight experience from failed subgoals with an external multi-agent critic architecture and dense milestone reward signals. HeRL provides the *learning signal from failure*, OS-Themis provides the *verifiable milestone structure*, and MiRA provides the *inference-time subgoal planning* — together enabling an agent that plans, verifies, fails, learns, and self-improves at test time.

**Who would use this and why?**
- Researchers building autonomous LLM agents (GUI, code synthesis, robotics)
- RL practitioners needing interpretable milestone-based reward signals
- Developers wanting self-correcting agents that learn from every failed attempt

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    HeRoS Framework                   │
│                                                      │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────┐ │
│  │   Planner   │───▶│   Milestone  │───▶│  Actor  │ │
│  │  (MiRA)     │    │   Critic     │    │  (LLM)  │ │
│  │  Subgoal    │    │ (OS-Themis)  │    │         │ │
│  │  Decompose  │    │  Verify &    │    │ Execute │ │
│  │             │    │  Audit       │    │ Subgoal │ │
│  └─────────────┘    └──────────────┘    └────┬────┘ │
│         ▲                    ▲               │      │
│         │      ┌────────────┴───────────┐   │      │
│         │      │   Hindsight Buffer     │   │      │
│         │      │   (HeRL)               │   │      │
│         │      │ Failed rubric-missed   │   │      │
│         │      │ trajectories → policy   │◀──┘      │
│         │      │ improvement             │          │
│         └──────┴─────────────────────────┘          │
│                   ▲                                  │
│              [Test-time                              │
│               Self-                                  │
│               Improve]                              │
└─────────────────────────────────────────────────────┘
```

### Core Loop

1. **Plan**: MiRA-style subgoal decomposition — the agent receives a task and decomposes it into an ordered sequence of subgoals (milestones).
2. **Execute**: Actor (LLM) attempts to complete each subgoal.
3. **Verify**: OS-Themis-style milestone critic reviews the execution trace against the milestone rubric. Produces a binary or scalar verdict + feedback.
4. **Hindsight**: HeRL-style hindsight experience — if a subgoal fails (unmet rubric), the failed trajectory with the unmet rubric is stored in a hindsight buffer and used to train the policy beyond its current distribution.
5. **Audit**: OS-Themis reward auditor converts milestone verdicts into dense RL reward signals.
6. **Repeat / Self-Improve**: At test time, failed subgoals from the current episode are added to the hindsight buffer, enabling self-improvement without new environment interactions.

---

## Implementation Roadmap

### Step 1 = [DONE] Planning
- [x] Identify novel technique combination (HeRL + OS-Themis + MiRA)
- [x] Define project scope and architecture
- [x] Write PLAN.md and README.md

### Step 2 = [DONE] Project Skeleton & Dependencies
- [x] Set up Python project structure (`src/heros/`, `configs/`, `eval/`, `scripts/`) ✅
- [x] Dependencies: `transformers`, `torch`, `numpy`, `json`, `pathlib` ✅
- [x] Docker/environment setup file ✅
- **PR:** #1 merged ✅ (`b105709`)

### Step 3 = [DONE] MiRA-style Subgoal Decomposition Module
- [x] Implement subgoal planner: task prompt → ordered milestone list ✅
- [x] Each milestone has: `id`, `description`, `rubric` (pass/fail criteria), `expected_output` ✅
- [x] Support for configurable planning depth (1-10 subgoals) ✅
- [x] Unit tests on task decomposition quality ✅ (37 passing tests)
- **PR:** #3 merged ✅ (`633acd0`)

### Step 4 — OS-Themis-style Milestone Critic Agent [DONE]
- [x] Implement critic agent that receives: `milestone rubric` + `execution trace`
- [x] Critic outputs: `verdict` (pass/fail/partially), `feedback`, `confidence`
- [x] Support both LLM-as-critic and rule-based critic backends
- [x] Implement reward auditor: milestone verdict → dense reward signal (0.0 to 1.0)
- [x] Unit tests for critic quality (40 passing tests)
- **PR:** #5 merged ✅ (`18427c4`)

### Step 5 — HeRL-style Hindsight Experience Buffer
- Implement trajectory storage: `task`, `milestones`, `exec_traces`, `verdicts`, `unmet_rubrics`
- Hindsight sampling: failed trajectories re-labeled with "what if this subgoal was the goal?"
- Policy update from hindsight buffer (PPO-style or BC-style supervised fine-tuning)
- Configurable hindsight ratio (e.g., 30% of batches from hindsight buffer)

### Step 6 — Core RL Training Loop
- Implement `HeRoSEnv` — wraps any task environment with milestone tracking
- Implement `HeRoSAgent` — planner + actor + critic + hindsight buffer
- PPO-style policy update combining standard and hindsight experience
- Logging: milestone success rates, hindsight utilization, reward curves

### Step 7 — WebArena-Lite / MiniWoB Evaluation Harness
- Port evaluation to WebArena-Lite-style web navigation tasks
- Implement milestone decomposition for web interaction tasks (click, type, navigate, submit)
- Baseline: raw LLM agent (no milestones, no hindsight)
- Metrics: task completion rate, milestone hit rate, hindsight improvement delta

### Step 8 — Interpretability & Audit Trail
- Log every milestone decision with critic reasoning
- Implement functional interchangeability check (from "Pitfalls in Interpretability")
- Export milestone-level reward audit trail (JSON per episode)
- Visualize hindsight buffer composition over training

### Step 9 — Hybrid Uncertainty Estimation Integration
- Integrate the two-sample hybrid estimator (embedding cosine + verbalized confidence) for milestone critic confidence calibration
- Reduces false-positive milestone passes
- Evaluate AUROC improvement on milestone verification

### Step 10 — Test-time Self-Improvement Mode
- Implement inference-only mode: agent runs task, fails subgoals are added to local hindsight buffer
- Policy optionally updated between inference episodes (self-play style)
- Benchmark improvement over N inference episodes with no additional training data

### Step 11 — Box Maze Boundary Enforcement (Optional Extension)
- Integrate Box Maze-style boundary enforcement for agent action space
- Memory grounding layer to prevent the agent from "forgetting" constraints mid-trajectory
- Reduces boundary failure rates (applicable to GUI/robotics domains)

### Step 12 — Paper, Documentation & Open Source Release
- Write comprehensive README with benchmarks, architecture diagrams
- Publish ArXiv preprint / technical report
- Release pretrained checkpoints and evaluation harness
- Submit to relevant venues (NeurIPS, ICML, ICLR workshops)

---

## Key Innovations Summary

| Feature | HeRL | OS-Themis | MiRA | **HeRoS (novel)** |
|---|---|---|---|---|
| Hindsight from failed trajectories | ✅ | — | — | ✅ |
| Multi-agent milestone critics | — | ✅ | — | ✅ |
| RL reward auditing | — | ✅ | — | ✅ |
| Subgoal-driven planning | — | — | ✅ | ✅ |
| Dense milestone rewards | — | — | ✅ | ✅ |
| **Test-time self-improvement** | ✅ | — | — | ✅ |
| **Combined milestone critics + hindsight** | — | — | — | ✅ |
| **Hindsight-guided policy exploration** | ✅ | — | — | ✅ |
