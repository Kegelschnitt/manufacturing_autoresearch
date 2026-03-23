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
  --problem configs/changeover_problem.json \
  --out runs/example \
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
