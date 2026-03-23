from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .types import Assignment, ProblemDefinition


ObjectiveEvaluator = Callable[[ProblemDefinition, list[Assignment]], float]


@dataclass
class ObjectiveTermSpec:
    term_id: str
    description: str
    evaluator: ObjectiveEvaluator
    modeling_hint: str = ""
    suggested_variables: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    applies_to_objectives: list[str] = field(default_factory=list)

    def __call__(self, problem: ProblemDefinition, assignments: list[Assignment]) -> float:
        return self.evaluator(problem, assignments)


def assignment_cost(problem: ProblemDefinition, assignments: list[Assignment]) -> float:
    cost = problem.parameters.get("cost", {}) or {}
    total = 0.0
    for a in assignments:
        total += float(cost.get(f"{a.job}|{a.machine}|{a.slot}", 0.0))
    return total


def changeover_penalty(problem: ProblemDefinition, assignments: list[Assignment]) -> float:
    """
    Evaluator semantics:
    - Group assignments by machine.
    - Look at occupied slots on each machine.
    - For each consecutive occupied slot pair s and s+1:
        add one penalty iff the assigned jobs differ.
    - No penalty if one of the slots is empty.
    - No penalty if the same job continues.
    """
    penalty = float(problem.parameters.get("changeover_penalty", 0.0) or 0.0)
    if penalty <= 0:
        return 0.0

    by_machine: dict[str, dict[int, str]] = {}
    for a in assignments:
        by_machine.setdefault(a.machine, {})[int(a.slot)] = a.job

    total = 0.0
    for _, slot_to_job in by_machine.items():
        slots = sorted(slot_to_job)
        for s1, s2 in zip(slots, slots[1:]):
            if s2 == s1 + 1 and slot_to_job[s1] != slot_to_job[s2]:
                total += penalty

    return total


def tardiness_penalty(problem: ProblemDefinition, assignments: list[Assignment]) -> float:
    due_slot = problem.parameters.get("due_slot", {}) or {}
    tardiness_penalty_map = problem.parameters.get("tardiness_penalty", {}) or {}

    total = 0.0
    for a in assignments:
        due = int(due_slot.get(a.job, 0) or 0)
        assigned_slot = int(a.slot)
        tardiness = max(0, assigned_slot - due)
        penalty = float(tardiness_penalty_map.get(a.job, 0.0) or 0.0)
        total += penalty * tardiness

    return total


def build_objective_registry() -> dict[str, ObjectiveTermSpec]:
    return {
        "assignment_cost": ObjectiveTermSpec(
            term_id="assignment_cost",
            description="Sum of assignment costs for all selected job-machine-slot assignments.",
            evaluator=assignment_cost,
            modeling_hint="Add the cost coefficient for each selected assignment variable directly to the MILP objective.",
            suggested_variables=[],
            depends_on=["x[job, machine, slot]", "cost[job|machine|slot]"],
            applies_to_objectives=["min_total_cost", "min_total_cost_with_changeover", "min_total_cost_with_tardiness"],
        ),
        "changeover_penalty": ObjectiveTermSpec(
            term_id="changeover_penalty",
            description=(
                "A changeover occurs on machine m between consecutive occupied slots "
                "t and t+1 if the assigned jobs are different. The objective adds one "
                "changeover_penalty for each such transition."
            ),
            evaluator=changeover_penalty,
            modeling_hint=(
                "Introduce binary transition variables z[(machine, slot, prev_job, next_job)] "
                "only for valid assignment pairs where x[(prev_job, machine, slot)] exists "
                "and x[(next_job, machine, slot+1)] exists, and only for prev_job != next_job. "
                "Add linking constraints z <= x_prev, z <= x_next, and z >= x_prev + x_next - 1. "
                "Then add changeover_penalty * sum(z) to the MILP objective. "
                "Use eligibility-safe sparse indexing and never reference nonexistent x keys."
            ),
            suggested_variables=["z[machine, slot, prev_job, next_job]"],
            depends_on=[
                "x[job, machine, slot]",
                "consecutive occupied slots",
                "prev_job != next_job",
                "changeover_penalty parameter",
                "sparse eligibility-safe indexing",
            ],
            applies_to_objectives=["min_total_cost_with_changeover"],
        ),
        "tardiness_penalty": ObjectiveTermSpec(
            term_id="tardiness_penalty",
            description=(
                "Each job assigned after its due slot incurs tardiness equal to "
                "max(0, assigned_slot - due_slot[job]). The objective adds tardiness_penalty[job] "
                "times that tardiness."
            ),
            evaluator=tardiness_penalty,
            modeling_hint=(
                "Introduce assignment-slot-driven tardiness terms. A simple time-indexed formulation "
                "can directly add tardiness_penalty[job] * max(0, slot - due_slot[job]) as the "
                "coefficient of x[job, machine, slot] in the objective. Alternatively, introduce a "
                "nonnegative tardiness variable T[job] with linking constraints and add "
                "tardiness_penalty[job] * T[job] to the objective."
            ),
            suggested_variables=["T[job] (optional tardiness variable)"],
            depends_on=[
                "x[job, machine, slot]",
                "due_slot[job]",
                "tardiness_penalty[job]",
                "assigned slot of each job",
            ],
            applies_to_objectives=["min_total_cost_with_tardiness"],
        ),
    }


def compute_objective_terms(problem: ProblemDefinition, assignments: list[Assignment]) -> dict[str, float]:
    registry = build_objective_registry()

    objective = problem.evaluation_framework.get("objective", {}) or {}
    objective_id = str(objective.get("id", "")).lower()

    terms: dict[str, float] = {}

    for term_id, spec in registry.items():
        applies = list(spec.applies_to_objectives or [])
        if not applies or objective_id in applies:
            terms[term_id] = spec(problem, assignments)

    return terms
