from __future__ import annotations

from typing import Any

from .evaluation_rules import build_rule_registry
from .objective_terms import compute_objective_terms
from .types import EvaluationReport, ProblemDefinition, RuleCheck, SolverResult


OK_STATUSES = {"optimal", "feasible"}
TOL = 1e-9


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _diagnose_infeasibility(problem: ProblemDefinition, result: SolverResult, checks: list[RuleCheck]) -> list[dict[str, Any]]:
    diagnosis: list[dict[str, Any]] = []
    params = problem.parameters or {}
    entities = problem.entities or {}

    jobs = list(entities.get("jobs", []) or [])
    machines = list(entities.get("machines", []) or [])
    slots = [int(s) for s in (entities.get("slots", []) or [])]
    eligible = params.get("eligible", {}) or {}

    # Global worker-demand shortage.
    hard_rule_ids = {str(r.get("id", "")) for r in (problem.evaluation_framework.get("hard_rules", []) or [])}
    if "worker_capacity" in hard_rule_ids:
        workers_required = params.get("workers_required", {}) or {}
        workers_available_per_slot = params.get("workers_available_per_slot", {}) or {}

        total_required = sum(_safe_float(workers_required.get(job, 0.0)) for job in jobs)
        total_available = sum(
            _safe_float(workers_available_per_slot.get(str(slot), workers_available_per_slot.get(slot, 0.0)))
            for slot in slots
        )
        if total_required > total_available + TOL:
            diagnosis.append(
                {
                    "type": "worker_capacity_global_shortage",
                    "message": (
                        f"Total worker demand is {total_required:g}, but total available worker capacity over all slots is {total_available:g}."
                    ),
                    "suggested_fix": (
                        "Increase workers_available_per_slot, add more slots, reduce workers_required, or schedule fewer required jobs."
                    ),
                }
            )

    # Global machine-slot capacity shortage.
    max_assignments = len(machines) * len(slots)
    if jobs and max_assignments < len(jobs):
        diagnosis.append(
            {
                "type": "global_machine_slot_shortage",
                "message": (
                    f"There are {len(jobs)} jobs that must each be assigned once, but only {max_assignments} total machine-slot positions are available."
                ),
                "suggested_fix": "Add more slots or machines, or reduce the number of jobs that must be assigned.",
            }
        )

    # Job with no eligible machine at all.
    jobs_without_eligible = [job for job in jobs if not list(eligible.get(job, []) or [])]
    if jobs_without_eligible:
        diagnosis.append(
            {
                "type": "jobs_without_eligible_machine",
                "message": f"Some jobs have no eligible machine at all: {jobs_without_eligible}.",
                "suggested_fix": "Expand the eligible machine set for these jobs or relax eligibility constraints.",
            }
        )

    # Job with no feasible machine-slot placements due to zero slot horizon or no eligibility.
    impossible_jobs: list[str] = []
    for job in jobs:
        allowed = list(eligible.get(job, []) or [])
        feasible_pairs = [(m, s) for m in allowed for s in slots]
        if len(feasible_pairs) == 0:
            impossible_jobs.append(job)
    if impossible_jobs:
        diagnosis.append(
            {
                "type": "jobs_without_any_machine_slot_placement",
                "message": f"Some jobs have no feasible machine-slot placement under the current horizon and eligibility map: {impossible_jobs}.",
                "suggested_fix": "Add slots, relax eligibility, or remove impossible jobs from the instance.",
            }
        )

    # Surface first failed rule checks as structured diagnostics too.
    for check in checks:
        if not check.passed:
            diagnosis.append(
                {
                    "type": f"failed_rule::{check.rule_id}",
                    "message": check.details,
                    "suggested_fix": "Adjust the problem data or strengthen the MILP so this hard rule can be satisfied.",
                }
            )

    # Deduplicate while preserving order.
    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for item in diagnosis:
        key = (str(item.get("type", "")), str(item.get("message", "")))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def evaluate_result(problem: ProblemDefinition, result: SolverResult) -> EvaluationReport:
    registry = build_rule_registry()
    requested = [r["id"] for r in problem.evaluation_framework.get("hard_rules", [])]
    checks: list[RuleCheck] = []
    for rule_id in requested:
        fn = registry.get(rule_id)
        if fn is None:
            checks.append(RuleCheck(rule_id=rule_id, passed=False, details="no evaluator implemented for rule"))
        else:
            checks.append(fn(problem, result.assignments))

    solver_status_ok = result.solver_status in OK_STATUSES
    objective_terms = compute_objective_terms(problem, result.assignments)
    computed = float(sum(objective_terms.values()))
    reported = result.objective_value

    violations = []
    if not solver_status_ok:
        violations.append(f"solver status is '{result.solver_status}', expected feasible/optimal")
    violations.extend(c.details for c in checks if not c.passed)

    objective_consistent = reported is None or abs(float(reported) - computed) <= TOL
    if reported is not None and not objective_consistent:
        violations.append(
            "reported objective does not match evaluator-computed objective; model likely omits active objective terms"
        )

    is_feasible = solver_status_ok and all(c.passed for c in checks)
    is_acceptable = is_feasible and objective_consistent
    infeasibility_diagnosis = [] if is_feasible else _diagnose_infeasibility(problem, result, checks)

    return EvaluationReport(
        is_feasible=is_feasible,
        is_acceptable=is_acceptable,
        solver_status_ok=solver_status_ok,
        computed_objective_value=computed,
        reported_objective_value=reported,
        objective_terms=objective_terms,
        rule_checks=checks,
        violations=violations,
        infeasibility_diagnosis=infeasibility_diagnosis,
        summary="accepted" if is_acceptable else "violations found",
    )
