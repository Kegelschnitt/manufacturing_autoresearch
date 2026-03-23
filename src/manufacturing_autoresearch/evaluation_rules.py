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

    overloads = {
        key: jobs
        for key, jobs in seen.items()
        if len(jobs) > 1
    }

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
            depends_on=[
                "x[job, machine, slot]",
                "eligible machine choices",
                "all slots",
            ],
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
            depends_on=[
                "x[job, machine, slot]",
                "machine-slot pairs",
            ],
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
            depends_on=[
                "eligible[job]",
                "x[job, machine, slot]",
                "sparse eligibility-safe indexing",
            ],
        ),
    }


def evaluate_rule_checks(
    problem: ProblemDefinition,
    assignments: list[Assignment],
) -> list[RuleCheck]:
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