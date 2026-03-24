# Manufacturing Autoresearch

## Overview

This project implements an **iterative MILP self-repair system** for manufacturing scheduling problems.

Instead of generating a single optimization model, the system:

1. Generates a baseline MILP  
2. Executes and evaluates it  
3. Diagnoses issues  
4. Selects relevant modeling lessons  
5. Uses an LLM to propose targeted fixes  
6. Repeats until an acceptable solution is found  

---

## Key Idea

> Combine structured evaluation + reusable modeling knowledge + LLM reasoning to iteratively repair optimization models.

This system behaves like an **autonomous MILP debugger**.

---

## Architecture

### Repair Loop

`baseline -> evaluate -> repair_signal -> select_lessons -> propose -> repeat`

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
  Reusable MILP modeling patterns, for example tardiness, sparse indexing, and objective construction.

- **Lesson Selector**
  - heuristic fallback
  - **LLM-based selector (main)**

- **LLM Proposer**
  - reasoning phase: what to fix
  - code generation phase: apply the fix

---

## LLM Lesson Selector

The selector receives:

- problem definition  
- repair signal  
- proposer guidance  
- run memory (history)  
- available lessons  
- **current MILP code**

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

---

## Logging

Each iteration produces:

- `iteration_X.json` -> full state
- `iteration_XXX_summary.json` -> compact summary

Includes:

- solver status
- objective values
- failure type
- selected lessons
- **lesson reasons (LLM and heuristic)**

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

---

### 🔍 Observations

- **Immediate success cases**  
  Simple feasibility and constraint problems are solved by the baseline MILP without requiring repair.

- **Structured repair behavior**  
  More complex problems (tardiness, changeover) require multiple iterations, where the system:
  - detects missing objective terms or constraints
  - selects relevant modeling lessons
  - incrementally improves the MILP

- **Lesson-guided improvement**  
  The LLM-based lesson selector consistently chooses relevant lessons, such as:
  - `tardiness_requires_slot_based_penalty`
  - `changeover_requires_auxiliary_variables`
  - `set_objective_once`

- **Convergence pattern**

```text
infeasible -> feasible but incorrect objective -> accepted solution
```

---

## 🧠 Code Explanation

The system is built as a modular pipeline implementing an iterative MILP repair loop.

### Main Loop (`graph.py`)

Controls the full process:

`baseline -> evaluate -> repair_signal -> select_lessons -> propose -> repeat`

Responsibilities:
- run state management
- lesson selection (heuristic + LLM)
- LLM calls (reasoning + code generation)
- iteration logging

### LLM Interaction (`llm.py`)

Handles all LLM logic:

- lesson selection (with reasons and priorities)
- reasoning (diagnosis)
- code generation (MILP update)

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

- `modeling_lessons.py` defines reusable patterns  
- `lesson_selector.py` selects lessons (LLM + fallback)

### Execution Pipeline

- `preflight.py` validates generated code  
- `execution.py` runs the MILP  
- `selector.py` accepts or rejects candidates  

### Logging

Stores:
- full iteration logs
- compact summaries
- selected lessons and reasons

### Design Principle

The system separates:
- evaluation
- knowledge (lessons)
- generation (LLM)

This enables iterative self-improvement.

---

## 🚀 Future Directions

- **Adaptive Lesson Learning**  
  Learn new modeling lessons from successful runs and track their effectiveness, while keeping core lessons fixed.

- **Benchmarking**  
  Evaluate on larger datasets (e.g. PSPLIB, OR-Library) and compare against heuristic and no-lesson baselines.

- **Ablation Studies**  
  Analyze the impact of lesson selection, repair signals, and reasoning vs direct code generation.

- **More Complex Problems**  
  Extend to multi-stage scheduling, resource planning, and stochastic optimization.

- **Efficiency**  
  Reduce iterations via smarter stopping criteria, caching, and improved lesson selection.

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

An **autonomous MILP debugger** that improves optimization models using:

- structured feedback
- reusable modeling knowledge
- LLM-guided reasoning
