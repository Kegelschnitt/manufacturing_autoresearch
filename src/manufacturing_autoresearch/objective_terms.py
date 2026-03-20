from __future__ import annotations

from .types import Assignment, ProblemDefinition


def assignment_cost(problem: ProblemDefinition, assignments: list[Assignment]) -> float:
    cost = problem.parameters.get("cost", {})
    total = 0.0
    for a in assignments:
        total += float(cost.get(f"{a.job}|{a.machine}|{a.slot}", 0.0))
    return total


def changeover_penalty(problem: ProblemDefinition, assignments: list[Assignment]) -> float:
    penalty = float(problem.parameters.get("changeover_penalty", 0.0) or 0.0)
    if penalty <= 0:
        return 0.0
    by_machine: dict[str, dict[int, str]] = {}
    for a in assignments:
        by_machine.setdefault(a.machine, {})[a.slot] = a.job
    total = 0.0
    for _, slot_to_job in by_machine.items():
        slots = sorted(slot_to_job)
        for s1, s2 in zip(slots, slots[1:]):
            if s2 == s1 + 1 and slot_to_job[s1] != slot_to_job[s2]:
                total += penalty
    return total


def build_objective_registry():
    return {
        "assignment_cost": assignment_cost,
        "changeover_penalty": changeover_penalty,
    }


def compute_objective_terms(problem: ProblemDefinition, assignments: list[Assignment]) -> dict[str, float]:
    registry = build_objective_registry()
    terms = {"assignment_cost": registry["assignment_cost"](problem, assignments)}
    objective_id = str(problem.evaluation_framework.get("objective", {}).get("id", "")).lower()
    if "changeover" in objective_id:
        terms["changeover_penalty"] = registry["changeover_penalty"](problem, assignments)
    return terms
