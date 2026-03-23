from __future__ import annotations

from typing import Any

from .evaluation_rules import build_rule_registry
from .objective_terms import build_objective_registry


OBJECTIVE_TO_TERMS: dict[str, list[str]] = {
    "min_total_cost": ["assignment_cost"],
    "min_total_cost_with_changeover": ["assignment_cost", "changeover_penalty"],
    "min_total_cost_with_tardiness": ["assignment_cost", "tardiness_penalty"],
}


def _rule_definition_snapshot(spec: Any, fallback_rule_id: str, fallback_description: str) -> dict[str, Any]:
    return {
        "rule_id": getattr(spec, "rule_id", fallback_rule_id),
        "description": getattr(spec, "description", fallback_description),
        "is_hard": bool(getattr(spec, "is_hard", True)),
        "modeling_hint": getattr(spec, "modeling_hint", ""),
        "suggested_variables": list(getattr(spec, "suggested_variables", []) or []),
        "depends_on": list(getattr(spec, "depends_on", []) or []),
    }


def _objective_definition_snapshot(term_id: str, spec: Any) -> dict[str, Any]:
    if hasattr(spec, "term_id") or hasattr(spec, "description"):
        return {
            "term": getattr(spec, "term_id", term_id),
            "description": getattr(spec, "description", ""),
            "modeling_hint": getattr(spec, "modeling_hint", ""),
            "suggested_variables": list(getattr(spec, "suggested_variables", []) or []),
            "depends_on": list(getattr(spec, "depends_on", []) or []),
        }
    return {
        "term": term_id,
        "description": getattr(spec, "__doc__", "") or "",
        "modeling_hint": "Add this term directly to the MILP objective.",
        "suggested_variables": [],
        "depends_on": [],
    }


def extract_proposer_guidance(problem: dict[str, Any], evaluation: dict[str, Any] | None = None) -> dict[str, Any]:
    framework = problem.get("evaluation_framework", {}) or {}
    hard_rules = framework.get("hard_rules", []) or []
    objective = framework.get("objective", {}) or {}

    rule_registry = build_rule_registry()
    objective_registry = build_objective_registry()

    active_rule_ids: list[str] = [str(rule.get("id", "")) for rule in hard_rules if str(rule.get("id", ""))]
    objective_id = str(objective.get("id", ""))
    active_term_ids = list(OBJECTIVE_TO_TERMS.get(objective_id, []))

    active_rules_guidance: list[dict[str, Any]] = []
    for rule in hard_rules:
        rule_id = str(rule.get("id", ""))
        spec = rule_registry.get(rule_id)
        if spec is None:
            active_rules_guidance.append(
                {
                    "rule_id": rule_id,
                    "description": rule.get("description", ""),
                    "type": "hard_constraint",
                    "modeling_hint": "Implement this requirement directly as MILP constraints.",
                }
            )
        else:
            active_rules_guidance.append(
                {
                    "rule_id": getattr(spec, "rule_id", rule_id),
                    "description": getattr(spec, "description", rule.get("description", "")),
                    "type": "hard_constraint" if getattr(spec, "is_hard", True) else "soft_rule",
                    "modeling_hint": getattr(spec, "modeling_hint", "Implement this requirement directly as MILP constraints."),
                    "suggested_variables": list(getattr(spec, "suggested_variables", []) or []),
                    "depends_on": list(getattr(spec, "depends_on", []) or []),
                }
            )

    active_objective_guidance: list[dict[str, Any]] = []
    for term_id in active_term_ids:
        spec = objective_registry.get(term_id)
        active_objective_guidance.append(_objective_definition_snapshot(term_id, spec))

    guidance: dict[str, Any] = {
        "problem_active_rule_ids": active_rule_ids,
        "problem_active_objective_id": objective_id,
        "active_rules_guidance": active_rules_guidance,
        "active_objective_guidance": active_objective_guidance,
        "available_rule_definitions": {
            rule_id: _rule_definition_snapshot(spec, rule_id, getattr(spec, "description", ""))
            for rule_id, spec in rule_registry.items()
        },
        "available_objective_term_definitions": {
            term_id: _objective_definition_snapshot(term_id, spec)
            for term_id, spec in objective_registry.items()
        },
    }

    if evaluation is not None:
        guidance["latest_rule_results"] = [
            {
                "rule_id": item.get("rule_id", ""),
                "passed": bool(item.get("passed", False)),
                "details": item.get("details", ""),
            }
            for item in evaluation.get("rule_checks", [])
        ]
        guidance["latest_objective_terms"] = dict(evaluation.get("objective_terms", {}) or {})
        guidance["latest_violations_from_guidance"] = list(evaluation.get("violations", []) or [])
        guidance["latest_acceptability_from_guidance"] = {
            "is_feasible": bool(evaluation.get("is_feasible", False)),
            "is_acceptable": bool(evaluation.get("is_acceptable", False)),
            "solver_status_ok": bool(evaluation.get("solver_status_ok", False)),
            "summary": evaluation.get("summary", ""),
        }

    return guidance
