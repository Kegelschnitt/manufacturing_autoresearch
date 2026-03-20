from __future__ import annotations

from .evaluation_rules import build_rule_registry
from .objective_terms import compute_objective_terms
from .types import EvaluationReport, ProblemDefinition, RuleCheck, SolverResult


OK_STATUSES = {"optimal", "feasible"}


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
    violations = []
    if not solver_status_ok:
        violations.append(f"solver status is '{result.solver_status}', expected feasible/optimal")
    violations.extend(c.details for c in checks if not c.passed)
    is_feasible = solver_status_ok and all(c.passed for c in checks)
    is_acceptable = is_feasible
    return EvaluationReport(
        is_feasible=is_feasible,
        is_acceptable=is_acceptable,
        solver_status_ok=solver_status_ok,
        computed_objective_value=computed,
        reported_objective_value=result.objective_value,
        objective_terms=objective_terms,
        rule_checks=checks,
        violations=violations,
        summary="accepted" if is_acceptable else "violations found",
    )
