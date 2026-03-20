from __future__ import annotations

from collections import Counter
from typing import Callable

from .types import Assignment, ProblemDefinition, RuleCheck


RuleFn = Callable[[ProblemDefinition, list[Assignment]], RuleCheck]


def rule_job_assigned_once(problem: ProblemDefinition, assignments: list[Assignment]) -> RuleCheck:
    jobs = problem.entities.get("jobs", [])
    counts = Counter(a.job for a in assignments)
    bad = [j for j in jobs if counts.get(j, 0) != 1]
    return RuleCheck(
        rule_id="job_assigned_once",
        passed=not bad,
        details="all jobs assigned exactly once" if not bad else f"jobs not assigned exactly once: {bad}",
    )


def rule_machine_capacity(problem: ProblemDefinition, assignments: list[Assignment]) -> RuleCheck:
    counts = Counter((a.machine, a.slot) for a in assignments)
    bad = [f"{m}@{s}" for (m, s), c in counts.items() if c > 1]
    return RuleCheck(
        rule_id="machine_capacity",
        passed=not bad,
        details="no machine-slot overloads" if not bad else f"machine-slot overloads: {bad}",
    )


def rule_eligibility(problem: ProblemDefinition, assignments: list[Assignment]) -> RuleCheck:
    eligible = problem.parameters.get("eligible", {})
    bad = [f"{a.job}->{a.machine}" for a in assignments if a.machine not in eligible.get(a.job, [])]
    return RuleCheck(
        rule_id="eligibility",
        passed=not bad,
        details="all assignments are valid and eligible" if not bad else f"ineligible assignments: {bad}",
    )


def build_rule_registry() -> dict[str, RuleFn]:
    return {
        "job_assigned_once": rule_job_assigned_once,
        "machine_capacity": rule_machine_capacity,
        "eligibility": rule_eligibility,
    }
