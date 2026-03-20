# Manufacturing Auto-Research (MILP Self-Improving System)

This project implements a **self-improving MILP (Mixed-Integer Linear Programming) generation loop** powered by an LLM.

The system iteratively:
1. Generates a MILP model
2. Executes it
3. Evaluates feasibility and objective correctness
4. Uses structured feedback to improve the model

---

## 🚀 Overview

The goal is to automatically construct correct and optimal MILP formulations for manufacturing scheduling problems.

Key idea:
> The LLM does not just generate code — it **learns from evaluation feedback** and incrementally improves the optimization model.

---

## 🔁 Iterative Loop

Each iteration performs:

1. **Propose**
   - LLM generates a new MILP formulation

2. **Preflight**
   - Validate syntax and structure

3. **Execute**
   - Solve using PuLP

4. **Evaluate**
   - Check:
     - hard constraints (rules)
     - objective correctness
     - feasibility

5. **Repair Signal**
   - Structured feedback describing:
     - rule violations
     - missing objective terms
     - runtime errors
     - structural issues

6. **Update Best Solution**
   - Accept candidate if it improves evaluation score

---

## 🧠 Key Components

### `graph.py`
Core orchestration loop:
- Runs iterations
- Tracks best solution
- Generates repair signals
- Logs progress

---

### `llm.py`
Handles LLM interaction:
- **Reasoning step** → what to fix
- **Code generation step** → improved MILP

---

### `evaluation_rules.py`
Defines **hard constraints**:
- job assigned once
- machine capacity
- eligibility

Each rule includes:
- description
- evaluation logic
- modeling hints

---

### `objective_terms.py`
Defines **objective components**:
- assignment cost
- changeover penalty

Each term includes:
- modeling hints
- required variables
- dependencies

---

### `proposer_guidance.py`
Builds structured input for the LLM:
- active rules
- active objective terms
- modeling hints
- latest evaluation feedback

---

### `evaluator.py`
Evaluates solver output:
- feasibility
- rule satisfaction
- objective correctness

---

### `baseline.py`
Provides initial MILP model.

---

## 📦 Problem Format

Problems are defined as JSON:

```json
{
  "name": "...",
  "entities": {
    "jobs": [...],
    "machines": [...],
    "slots": [...]
  },
  "parameters": {
    "cost": {...},
    "eligible": {...},
    "changeover_penalty": 3
  },
  "evaluation_framework": {
    "hard_rules": [...],
    "objective": {...}
  }
}