# Manufacturing Autoresearch

## Overview

This project implements an **iterative MILP self-repair and lesson-learning system** for manufacturing scheduling problems.

Instead of generating a single optimization model, the system:

1. Generates a baseline MILP
2. Executes and evaluates it
3. Diagnoses issues
4. Selects relevant modeling lessons
5. Uses an LLM to propose targeted fixes
6. Repeats until an acceptable solution is found
7. Learns reusable adaptive lessons from successful repairs
8. Deduplicates and merges similar adaptive lessons over time

---

## Key Idea

> Combine structured evaluation + reusable modeling knowledge + LLM reasoning + adaptive lesson learning to iteratively repair optimization models.

This system behaves like an **autonomous MILP debugger** that can also accumulate reusable modeling knowledge across runs.

---

## Current Status

The system currently supports:

- **iterative MILP repair**
- **LLM-based lesson selection**
- **persistent adaptive lesson memory**
- **lesson usage / success statistics**
- **adaptive lesson creation after successful repairs**
- **deduplication and safe merge of similar adaptive lessons**
- **per-iteration curation reports**

Implemented phases:

- **Phase 1 — Persistent lesson bookkeeping**
  - adaptive lesson memory
  - lesson stats tracking
  - runtime merge of core + adaptive lessons

- **Phase 2 — Stats-aware lesson selection**
  - heuristic selector uses lesson stats
  - LLM selector sees historical lesson performance
  - adaptive lessons participate in selection

- **Phase 3 — Adaptive lesson creation**
  - successful accepted repairs can generate new adaptive lessons
  - lessons are stored persistently in memory

- **Phase 4a — Deduplication and safe merge policy**
  - exact duplicates are skipped
  - near-duplicates can be merged into existing adaptive lessons
  - lesson curation decisions are logged

---

## Architecture

### Repair Loop

`baseline -> evaluate -> repair_signal -> select_lessons -> propose -> repeat`

### Learning Loop Extension

After successful accepted repairs:

`accepted repair -> curate lessons -> validate -> add or merge -> persist`

### Components

- **Evaluation Engine**
  Detects feasibility, rule violations, and objective mismatch.

- **Repair Signal**
  Structured diagnostics such as:
  - failure type
  - missing objective terms
  - violated rules
  - infeasibility hints

- **Modeling Lessons**
  Reusable MILP modeling patterns, for example tardiness, sparse indexing, changeover modeling, and objective construction.

- **Lesson Selector**
  - heuristic fallback
  - **LLM-based selector (main)**
  - uses both **core lessons** and **adaptive lessons**
  - can incorporate historical lesson stats

- **LLM Proposer**
  - reasoning phase: what to fix
  - code generation phase: apply the fix

- **Lesson Memory**
  Persistent store for adaptive lessons.

- **Lesson Stats**
  Tracks lesson usage and empirical success.

- **Lesson Curator**
  Extracts new adaptive lessons after successful repairs and applies validation / deduplication / merge logic.

---

## Core vs Adaptive Lessons

### Core Lessons
Core lessons are fixed, version-controlled, and never modified automatically.

### Adaptive Lessons
Adaptive lessons are:
- created from successful accepted repairs
- stored persistently
- reused in future selection
- deduplicated and merged when similar patterns already exist

This separation keeps the system stable while still enabling learning.

---

## LLM Lesson Selector

The selector receives:

- problem definition
- repair signal
- proposer guidance
- run memory (history)
- available lessons
- current MILP code
- lesson history / performance metadata

Returns:

```json
{
  "selected_lessons": [
    {
      "lesson_id": "...",
      "priority": 10,
      "reason": "..."
    }
  ]
}
```

Priorities and reasons are logged for interpretability.

### Stats-aware behavior

The selector can take lesson history into account, including:
- times selected
- success rate
- historical lesson score

This helps the system prefer lessons that are both relevant **and** historically useful.

---

## Adaptive Lesson Learning

After a successful accepted repair, the system can ask an LLM curator to extract reusable modeling lessons.

Example adaptive lesson structure:

```json
{
  "lesson_id": "changeover_modeling_with_aux_vars",
  "title": "Model changeovers explicitly with auxiliary variables",
  "applies_when": {
    "failure_type": "objective_mismatch",
    "missing_objective_terms": ["changeover_penalty"],
    "traceback_contains": [],
    "violated_rules": [],
    "objective_ids": [],
    "always": false
  },
  "lesson": "Introduce explicit transition variables and linking constraints to model changeover penalties in the MILP objective.",
  "recommended_actions": [
    "Add binary transition variables",
    "Link transition variables to assignments",
    "Include changeover penalties directly in the objective"
  ],
  "anti_patterns": [
    "Injecting changeover cost without explicit linking variables"
  ],
  "priority_hint": 5,
  "tags": ["adaptive", "changeover", "objective"]
}
```

### Deduplication and merging

When the curator proposes lessons, the system can:
- **accept** a genuinely new lesson
- **skip** an exact duplicate
- **merge** a near-duplicate into an existing adaptive lesson

This prevents uncontrolled lesson growth.

---

## Logging

Each iteration produces:

- `iteration_X.json` -> full state
- `iteration_XXX_summary.json` -> compact summary
- `adaptive_lesson_memory_snapshot.json` -> current adaptive lesson memory
- `adaptive_lesson_stats_snapshot.json` -> current lesson stats
- `iteration_XXX_lesson_curation_report.json` -> lesson curation decisions for accepted repairs

Includes:

- solver status
- objective values
- failure type
- selected lessons
- lesson reasons (LLM and heuristic)
- adaptive lesson proposals
- accepted / merged / rejected curation outcomes

Example:

```text
lessons=tardiness_requires_slot_based_penalty (reason...)
```

---

## 📊 Benchmark Results

We evaluate the repair loop on a small suite of custom scheduling problems designed to isolate key modeling challenges:

- assignment feasibility
- sparse eligibility constraints
- tardiness penalties
- changeover costs
- worker capacity constraints

### Summary

| Benchmark | Description | Solved | Iterations | Final Objective | Notes |
|---|---|---:|---:|---:|---|
| `assignment_small` | Basic assignment with costs | ✅ | 0 | 4.0 | Baseline already acceptable |
| `eligibility_sparse` | Sparse eligibility constraints | ✅ | 0 | 6.0 | Tests safe indexing |
| `worker_capacity_small` | Slot-based worker limits | ✅ | 0 | 4.0 | Resource aggregation |
| `changeover_small` | Sequence-dependent changeovers | ✅ | 2 | 8.0 | Requires auxiliary variables |
| `tardiness_small` | Due dates + tardiness penalties | ✅ | 3 | 10.0 | Multi-step repair |

### Research-inspired benchmark expansion

To evaluate cross-problem transfer and lesson reuse, the next benchmark suite is based on classic scheduling problem families used in research, including small instances inspired by:

- **single machine tardiness** (`1||∑T_j`)
- **parallel machine makespan** (`P||Cmax`)
- **flow shop scheduling**
- **job shop scheduling**
- **sequence-dependent changeovers / setups**
- **resource- and skill-constrained scheduling**

Suggested benchmark set:

| Benchmark | Type | What it teaches |
|---|---|---|
| `single_machine_tardiness` | 1 machine | tardiness modeling |
| `parallel_machine_assignment` | parallel machines | load balancing / makespan |
| `flow_shop_small` | flow shop | sequencing across machines |
| `job_shop_tiny` | job shop | precedence + machine conflicts |
| `changeover_sequence` | sequence-dependent setup | auxiliary transition variables |
| `worker_capacity_multi_skill` | resource constrained | aggregation + sparse indexing |

These benchmarks are intended to stress-test:
- cross-instance lesson reuse
- adaptive lesson creation quality
- duplicate handling
- convergence behavior across diverse scheduling structures

---

## 🔍 Observations

- **Immediate success cases**
  Simple feasibility and constraint problems are solved by the baseline MILP without requiring repair.

- **Structured repair behavior**
  More complex problems (tardiness, changeover) require multiple iterations, where the system:
  - detects missing objective terms or constraints
  - selects relevant modeling lessons
  - incrementally improves the MILP

- **Lesson-guided improvement**
  The selector consistently chooses relevant lessons, such as:
  - `tardiness_requires_slot_based_penalty`
  - `changeover_requires_auxiliary_variables`
  - `set_objective_once`

- **Adaptive learning behavior**
  Successful repairs can generate reusable adaptive lessons, which are then:
  - stored
  - reused
  - deduplicated
  - merged if they overlap with existing knowledge

- **Convergence pattern**

```text
infeasible -> feasible but incorrect objective -> accepted solution
```

---

## 🧠 Code Explanation

The system is built as a modular pipeline implementing an iterative MILP repair and lesson-learning loop.

### Main Loop (`graph.py`)

Controls the full process:

`baseline -> evaluate -> repair_signal -> select_lessons -> propose -> repeat`

Responsibilities:
- run state management
- lesson selection (heuristic + LLM)
- LLM calls (reasoning + code generation)
- adaptive lesson curation after accepted repairs
- iteration logging

### LLM Interaction (`llm.py`)

Handles all LLM logic:

- lesson selection (with reasons and priorities)
- reasoning (diagnosis)
- code generation (MILP update)
- lesson curation after successful repairs

### Evaluation (`evaluator.py`)

Checks:
- feasibility
- rule satisfaction
- objective correctness

Produces structured feedback.

### Repair Signal

Core feedback structure containing:
- failure type
- violated rules
- missing objective terms
- infeasibility hints

### Modeling Lessons

- `modeling_lessons.py` defines reusable core patterns
- `lesson_selector.py` selects lessons (LLM + fallback)
- `lesson_memory.py` stores adaptive lessons
- `lesson_stats.py` tracks lesson performance
- `lesson_curator.py` validates, deduplicates, and classifies new adaptive lesson proposals

### Execution Pipeline

- `preflight.py` validates generated code
- `execution.py` runs the MILP
- `selector.py` accepts or rejects candidates

### Logging

Stores:
- full iteration logs
- compact summaries
- selected lessons and reasons
- adaptive lesson curation reports
- lesson memory and stats snapshots

### Design Principle

The system separates:
- evaluation
- knowledge (lessons)
- generation (LLM)
- learning / memory
- lesson quality control

This enables iterative self-improvement while preserving stability.

---

## 🚀 Future Directions

- **Phase 4b — Adaptive Lesson Quality Control**
  Deactivate or archive weak adaptive lessons based on repeated poor performance.

- **Benchmarking**
  Evaluate on larger datasets and research benchmark families (e.g. OR-Library, PSPLIB-style or Taillard-inspired scheduling instances).

- **Ablation Studies**
  Analyze the impact of:
  - lesson selection
  - repair signals
  - adaptive lesson memory
  - deduplication / merge policy
  - reasoning vs direct code generation

- **More Complex Problems**
  Extend to multi-stage scheduling, resource planning, batching, and stochastic optimization.

- **Efficiency**
  Reduce iterations via smarter stopping criteria, caching, lesson pruning, and stronger selector calibration.

---

## Example Result

Tardiness benchmark:

- Baseline objective: **15**
- Final objective: **10**
- Solved in **1 iteration**

The system correctly:
- detected objective mismatch
- selected relevant lessons
- fixed objective construction

Example adaptive learning behavior:
- a successful changeover repair can generate a reusable adaptive lesson about transition variables
- a near-duplicate future proposal can be merged instead of being re-added

---

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .

cp .env.example .env
# add your OPENAI_API_KEY

export PYTHONPATH=src
python -m manufacturing_autoresearch.main \
  --problem configs/benchmarks/changeover_small.json \
  --out runs/changeover_test \
  --max-iters 3
```

---

## Requirements

- Python 3.10+
- PuLP
- OpenAI Python SDK
- python-dotenv

---

## Summary

An **autonomous MILP debugger and lesson-learning system** that improves optimization models using:

- structured feedback
- reusable modeling knowledge
- LLM-guided reasoning
- adaptive lesson memory
- stats-aware lesson selection
- adaptive lesson curation with deduplication and merge logic
