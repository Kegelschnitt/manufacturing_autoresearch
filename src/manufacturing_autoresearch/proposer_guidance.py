from __future__ import annotations

from typing import Any

from .evaluation_rules import build_rule_registry
from .objective_terms import build_objective_registry


OBJECTIVE_TO_TERMS: dict[str, list[str]] = {
    "min_total_cost": ["assignment_cost"],
    "min_total_cost_with_changeover": ["assignment_cost", "changeover_penalty"],
}


def extract_proposer_guidance(
    problem: dict[str, Any],
    evaluation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build a compact proposer-facing summary from the active evaluation framework.

    This intentionally exposes semantic/modeling guidance instead of raw evaluator
    implementation details, so evaluation_rules.py and objective_terms.py remain
    easy to refactor later.
    """
    framework = problem.get("evaluation_framework", {})
    hard_rules = framework.get("hard_rules", [])
    objective = framework.get("objective", {})

    rule_registry = build_rule_registry()
    objective_registry = build_objective_registry()

    active_rules_guidance: list[dict[str, Any]] = []
    for rule in hard_rules:
        rule_id = rule.get("id", "")
        spec = rule_registry.get(rule_id)
        if spec is None:
            active_rules_guidance.append(
                {
                    "rule_id": rule_id,
                    "description": rule.get("description", ""),
                    "type": "hard_constraint",
                    "modeling_hint": (
                        "Implement this requirement directly as MILP constraints."
                    ),
                }
            )
            continue

        active_rules_guidance.append(
            {
                "rule_id": getattr(spec, "rule_id", rule_id),
                "description": getattr(spec, "description", rule.get("description", "")),
                "type": "hard_constraint" if getattr(spec, "is_hard", True) else "soft_rule",
                "modeling_hint": getattr(
                    spec,
                    "modeling_hint",
                    "Implement this requirement directly as MILP constraints.",
                ),
            }
        )

    objective_id = objective.get("id", "")
    active_objective_guidance: list[dict[str, Any]] = []
    for term_id in OBJECTIVE_TO_TERMS.get(objective_id, []):
        spec = objective_registry.get(term_id)
        if spec is None:
            active_objective_guidance.append(
                {
                    "term": term_id,
                    "description": "",
                    "modeling_hint": "Add this term directly to the MILP objective.",
                    "suggested_variables": [],
                    "depends_on": [],
                }
            )
            continue

        active_objective_guidance.append(
            {
                "term": getattr(spec, "term_id", term_id),
                "description": getattr(spec, "description", ""),
                "modeling_hint": getattr(
                    spec,
                    "modeling_hint",
                    "Add this term directly to the MILP objective.",
                ),
                "suggested_variables": list(
                    getattr(spec, "suggested_variables", []) or []
                ),
                "depends_on": list(getattr(spec, "depends_on", []) or []),
            }
        )

    guidance: dict[str, Any] = {
        "active_rules_guidance": active_rules_guidance,
        "active_objective_guidance": active_objective_guidance,
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
        guidance["latest_objective_terms"] = dict(
            evaluation.get("objective_terms", {}) or {}
        )

    return guidance
