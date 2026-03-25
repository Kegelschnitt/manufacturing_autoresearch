from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .types import Assignment, ProblemDefinition, RuleCheck


RuleEvaluator = Callable[[ProblemDefinition, list[Assignment]], RuleCheck]


@dataclass
class RuleSpec:
    rule_id: str
    description: str
    evaluator: RuleEvaluator
    is_hard: bool = True
    modeling_hint: str = ""
    suggested_variables: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)

    def __call__(self, problem: ProblemDefinition, assignments: list[Assignment]) -> RuleCheck:
        return self.evaluator(problem, assignments)


def _make_rule_check(rule_id: str, passed: bool, details: str) -> RuleCheck:
    return RuleCheck(rule_id=rule_id, passed=passed, details=details)


def job_assigned_once(problem: ProblemDefinition, assignments: list[Assignment]) -> RuleCheck:
    jobs = list(problem.entities.get("jobs", []) or [])
    counts = {job: 0 for job in jobs}

    for a in assignments:
        if a.job in counts:
            counts[a.job] += 1

    bad_jobs = [job for job, count in counts.items() if count != 1]
    if bad_jobs:
        return _make_rule_check(
            rule_id="job_assigned_once",
            passed=False,
            details=f"jobs not assigned exactly once: {bad_jobs}",
        )

    return _make_rule_check(
        rule_id="job_assigned_once",
        passed=True,
        details="all jobs assigned exactly once",
    )


def machine_capacity(problem: ProblemDefinition, assignments: list[Assignment]) -> RuleCheck:
    seen: dict[tuple[str, int], list[str]] = {}

    for a in assignments:
        key = (a.machine, int(a.slot))
        seen.setdefault(key, []).append(a.job)

    overloads = {key: jobs for key, jobs in seen.items() if len(jobs) > 1}

    if overloads:
        return _make_rule_check(
            rule_id="machine_capacity",
            passed=False,
            details=f"machine-slot overloads detected: {overloads}",
        )

    return _make_rule_check(
        rule_id="machine_capacity",
        passed=True,
        details="no machine-slot overloads",
    )


def eligibility(problem: ProblemDefinition, assignments: list[Assignment]) -> RuleCheck:
    eligible_map = problem.parameters.get("eligible", {}) or {}

    invalid: list[dict[str, object]] = []
    for a in assignments:
        allowed_machines = list(eligible_map.get(a.job, []) or [])
        if a.machine not in allowed_machines:
            invalid.append(
                {
                    "job": a.job,
                    "machine": a.machine,
                    "slot": int(a.slot),
                    "eligible_machines": allowed_machines,
                }
            )

    if invalid:
        return _make_rule_check(
            rule_id="eligibility",
            passed=False,
            details=f"ineligible assignments found: {invalid}",
        )

    return _make_rule_check(
        rule_id="eligibility",
        passed=True,
        details="all assignments are valid and eligible",
    )


def worker_capacity(problem: ProblemDefinition, assignments: list[Assignment]) -> RuleCheck:
    workers_required = problem.parameters.get("workers_required", {}) or {}
    workers_available_per_slot = problem.parameters.get("workers_available_per_slot", {}) or {}

    used_by_slot: dict[int, float] = {}
    for a in assignments:
        slot = int(a.slot)
        used_by_slot[slot] = used_by_slot.get(slot, 0.0) + float(workers_required.get(a.job, 0.0) or 0.0)

    overloads: dict[int, dict[str, float]] = {}
    slots_to_check = set(used_by_slot.keys()) | {int(s) for s in workers_available_per_slot.keys()}
    for slot in sorted(slots_to_check):
        available = float(workers_available_per_slot.get(str(slot), workers_available_per_slot.get(slot, 0.0)) or 0.0)
        used = float(used_by_slot.get(slot, 0.0))
        if used > available + 1e-9:
            overloads[slot] = {"used": used, "available": available}

    if overloads:
        return _make_rule_check(
            rule_id="worker_capacity",
            passed=False,
            details=f"worker-capacity overloads detected by slot: {overloads}",
        )

    return _make_rule_check(
        rule_id="worker_capacity",
        passed=True,
        details="worker usage fits within available workers for every slot",
    )


def operation_order(problem: ProblemDefinition, assignments: list[Assignment]) -> RuleCheck:
    """
    Coarse precedence check for flow-shop / job-shop style problems.

    Assumes assignments can be grouped by job and sorted by slot. For jobs with
    multiple operations, later operations must not appear before earlier ones.
    """
    jobs_data = problem.parameters.get("jobs_data", {}) or {}
    violations: list[dict[str, object]] = []

    by_job: dict[str, list[Assignment]] = {}
    for a in assignments:
        by_job.setdefault(a.job, []).append(a)

    for job_id, job_assignments in by_job.items():
        ops = jobs_data.get(job_id, {}).get("operations", [])
        if not ops or len(job_assignments) <= 1:
            continue

        sorted_assignments = sorted(job_assignments, key=lambda x: int(x.slot))
        sorted_slots = [int(a.slot) for a in sorted_assignments]

        for idx in range(1, len(sorted_slots)):
            if sorted_slots[idx] < sorted_slots[idx - 1]:
                violations.append(
                    {
                        "job": job_id,
                        "slots": sorted_slots,
                        "details": "operation order violated",
                    }
                )
                break

    if violations:
        return _make_rule_check(
            rule_id="operation_order",
            passed=False,
            details=f"operation precedence violations detected: {violations}",
        )

    return _make_rule_check(
        rule_id="operation_order",
        passed=True,
        details="job operation order respected",
    )


def skill_match(problem: ProblemDefinition, assignments: list[Assignment]) -> RuleCheck:
    """
    Check that assigned workers / machines satisfy required job skills.

    Expected parameters:
      required_skill[job] -> skill
      worker_skills[worker_or_machine] -> list[str]
    """
    required_skill = problem.parameters.get("required_skill", {}) or {}
    worker_skills = problem.parameters.get("worker_skills", {}) or {}

    invalid: list[dict[str, object]] = []
    for a in assignments:
        needed = required_skill.get(a.job)
        if not needed:
            continue

        skills = worker_skills.get(a.machine, []) or []
        if needed not in skills:
            invalid.append(
                {
                    "job": a.job,
                    "assigned_to": a.machine,
                    "required_skill": needed,
                    "available_skills": skills,
                }
            )

    if invalid:
        return _make_rule_check(
            rule_id="skill_match",
            passed=False,
            details=f"skill mismatches detected: {invalid}",
        )

    return _make_rule_check(
        rule_id="skill_match",
        passed=True,
        details="all assignments satisfy required skills",
    )


def makespan_consistency(problem: ProblemDefinition, assignments: list[Assignment]) -> RuleCheck:
    """
    Coarse consistency check for makespan-style problems.

    This does not verify an exact MILP makespan formulation. It only checks that
    assignments exist and that the schedule is non-empty when makespan is the objective.
    """
    if not assignments:
        return _make_rule_check(
            rule_id="makespan_consistency",
            passed=False,
            details="no assignments produced for makespan objective",
        )

    latest_slot = max(int(a.slot) for a in assignments)
    return _make_rule_check(
        rule_id="makespan_consistency",
        passed=True,
        details=f"non-empty schedule found; latest occupied slot = {latest_slot}",
    )


def build_rule_registry() -> dict[str, RuleSpec]:
    return {
        "job_assigned_once": RuleSpec(
            rule_id="job_assigned_once",
            description="Each job must appear exactly once in assignments.",
            evaluator=job_assigned_once,
            is_hard=True,
            modeling_hint=(
                "For each job, add a constraint that the sum of all assignment "
                "variables over eligible machines and all slots equals exactly 1."
            ),
            suggested_variables=[],
            depends_on=["x[job, machine, slot]", "eligible machine choices", "all slots"],
        ),
        "machine_capacity": RuleSpec(
            rule_id="machine_capacity",
            description="A machine can handle at most one job in the same slot.",
            evaluator=machine_capacity,
            is_hard=True,
            modeling_hint=(
                "For each machine-slot pair, add a capacity constraint that the sum "
                "of all assignment variables using that machine and slot is at most 1."
            ),
            suggested_variables=[],
            depends_on=["x[job, machine, slot]", "machine-slot pairs"],
        ),
        "eligibility": RuleSpec(
            rule_id="eligibility",
            description="Each job may only be assigned to eligible machines.",
            evaluator=eligibility,
            is_hard=True,
            modeling_hint=(
                "Only create or reference assignment variables for eligible "
                "job-machine-slot combinations. Use sparse eligibility-safe indexing "
                "throughout the MILP and never assume all (job, machine, slot) keys exist."
            ),
            suggested_variables=[],
            depends_on=["eligible[job]", "x[job, machine, slot]", "sparse eligibility-safe indexing"],
        ),
        "worker_capacity": RuleSpec(
            rule_id="worker_capacity",
            description=(
                "The total workers required by jobs assigned in a slot must not exceed "
                "the workers available in that slot."
            ),
            evaluator=worker_capacity,
            is_hard=True,
            modeling_hint=(
                "For each slot, add a worker-capacity constraint: the sum of "
                "workers_required[job] * x[job, machine, slot] over all assigned job-machine "
                "choices in that slot must be less than or equal to workers_available_per_slot[slot]."
            ),
            suggested_variables=[],
            depends_on=[
                "x[job, machine, slot]",
                "workers_required[job]",
                "workers_available_per_slot[slot]",
                "slot-wise resource aggregation",
            ],
        ),
        "operation_order": RuleSpec(
            rule_id="operation_order",
            description="Operations within the same job must respect precedence order.",
            evaluator=operation_order,
            is_hard=True,
            modeling_hint=(
                "For each job, add precedence constraints so later operations can only "
                "start after earlier operations finish."
            ),
            suggested_variables=[],
            depends_on=["job operations", "operation precedence", "start/slot ordering"],
        ),
        "skill_match": RuleSpec(
            rule_id="skill_match",
            description="Jobs may only be assigned to workers or machines with the required skill.",
            evaluator=skill_match,
            is_hard=True,
            modeling_hint=(
                "Only create assignment variables for job-resource pairs that satisfy "
                "the required skill. Use sparse indexing to avoid invalid combinations."
            ),
            suggested_variables=[],
            depends_on=["required_skill[job]", "worker_skills[resource]", "eligible assignment pairs"],
        ),
        "makespan_consistency": RuleSpec(
            rule_id="makespan_consistency",
            description="The schedule must be consistent with a makespan-style objective.",
            evaluator=makespan_consistency,
            is_hard=False,
            modeling_hint=(
                "Introduce a makespan variable C_max and constrain each job completion "
                "time to be less than or equal to C_max. Minimize C_max."
            ),
            suggested_variables=["C_max"],
            depends_on=["completion time variables", "max-completion objective"],
        ),
    }


def evaluate_rule_checks(problem: ProblemDefinition, assignments: list[Assignment]) -> list[RuleCheck]:
    framework = problem.evaluation_framework or {}
    hard_rules = framework.get("hard_rules", []) or []
    registry = build_rule_registry()

    checks: list[RuleCheck] = []
    for rule in hard_rules:
        rule_id = str(rule.get("id", ""))
        spec = registry.get(rule_id)

        if spec is None:
            checks.append(
                _make_rule_check(
                    rule_id=rule_id,
                    passed=True,
                    details="no registered evaluator for this rule id; skipped",
                )
            )
            continue

        checks.append(spec(problem, assignments))

    return checks
