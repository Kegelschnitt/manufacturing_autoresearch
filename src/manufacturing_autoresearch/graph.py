from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from rich import print

from .baseline import baseline_program
from .evaluator import evaluate_result
from .execution import execute_program
from .llm import LLMClient
from .logging_utils import dump_json, ensure_dir
from .preflight import run_preflight
from .proposer_guidance import extract_proposer_guidance
from .selector import should_accept_candidate
from .types import IterationRecord, ProblemDefinition, RunState, Settings, SolverResult


def _normalized(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _objective_value_for_log(evaluation) -> float | None:
    value = _safe_float(getattr(evaluation, "computed_objective_value", None))
    if value is not None:
        return value
    return _safe_float(getattr(evaluation, "reported_objective_value", None))


def _compact_eval_summary(evaluation) -> dict[str, Any]:
    return {
        "is_feasible": bool(getattr(evaluation, "is_feasible", False)),
        "is_acceptable": bool(getattr(evaluation, "is_acceptable", False)),
        "solver_status_ok": bool(getattr(evaluation, "solver_status_ok", False)),
        "computed_objective_value": _safe_float(getattr(evaluation, "computed_objective_value", None)),
        "reported_objective_value": _safe_float(getattr(evaluation, "reported_objective_value", None)),
        "objective_terms": dict(getattr(evaluation, "objective_terms", {}) or {}),
        "violations": list(getattr(evaluation, "violations", []) or []),
        "summary": getattr(evaluation, "summary", ""),
    }


def _extract_structure_features(code: str) -> dict[str, Any]:
    features: dict[str, Any] = {
        "imports": [],
        "function_names": [],
        "lpvariable_targets": [],
        "constraint_names": [],
        "objective_labels": [],
        "context_keys": [],
        "mentions": {
            "lpvariable": "LpVariable" in code,
            "lpsum": "lpSum" in code,
            "changeover": "changeover" in code.lower(),
            "constraint_capacity": "capacity_" in code,
            "constraint_job_once": "job_once_" in code,
        },
    }
    try:
        tree = ast.parse(code)
    except Exception:
        return features

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            features["function_names"].append(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                features["imports"].append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for alias in node.names:
                features["imports"].append(f"{mod}:{alias.name}")
        elif isinstance(node, ast.Call):
            func_name = ""
            if isinstance(node.func, ast.Attribute):
                func_name = node.func.attr
            elif isinstance(node.func, ast.Name):
                func_name = node.func.id

            if func_name == "LpVariable" and node.args:
                first = node.args[0]
                if isinstance(first, ast.JoinedStr):
                    text_bits = []
                    for value in first.values:
                        if isinstance(value, ast.Constant) and isinstance(value.value, str):
                            text_bits.append(value.value)
                    features["lpvariable_targets"].append("".join(text_bits))
                elif isinstance(first, ast.Constant) and isinstance(first.value, str):
                    features["lpvariable_targets"].append(first.value)

        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) and target.value.id == "context":
                    key = None
                    sl = target.slice
                    if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
                        key = sl.value
                    elif hasattr(ast, "Index") and isinstance(sl, ast.Index) and isinstance(sl.value, ast.Constant) and isinstance(sl.value.value, str):
                        key = sl.value.value
                    if key is not None:
                        features["context_keys"].append(key)

    for label in ["total_cost", "total_assignment_cost", "total_cost_with_changeover", "total_cost_with_setup"]:
        if label in code:
            features["objective_labels"].append(label)

    for prefix in ["capacity_", "job_once_", "link", "linking", "changeover_link_"]:
        if prefix in code:
            features["constraint_names"].append(prefix)

    for key, value in features.items():
        if isinstance(value, list):
            features[key] = sorted(set(value))
    return features


def _candidate_changed_structure(candidate_code: str, reference_code: str) -> bool:
    candidate_norm = _normalized(candidate_code)
    reference_norm = _normalized(reference_code)
    if candidate_norm == reference_norm:
        return False
    return _extract_structure_features(candidate_norm) != _extract_structure_features(reference_norm)


def _top_violation(evaluation) -> str | None:
    violations = list(getattr(evaluation, "violations", []) or [])
    return violations[0] if violations else None


def _build_run_memory(state: RunState) -> dict[str, Any]:
    recent_iterations: list[dict[str, Any]] = []
    for rec in state.history[-3:]:
        recent_iterations.append(
            {
                "iteration": rec.iteration,
                "accepted": rec.accepted_as_best,
                "solver_status": rec.candidate_result.solver_status,
                "computed_objective_value": _safe_float(rec.candidate_evaluation.computed_objective_value),
                "reported_objective_value": _safe_float(rec.candidate_evaluation.reported_objective_value),
                "violations": list(rec.candidate_evaluation.violations or []),
                "failure_type": rec.repair_signal.get("failure_type", ""),
                "candidate_changed_structure": bool(rec.repair_signal.get("candidate_changed_structure", False)),
            }
        )
    return {
        "current_iteration": state.current_iteration,
        "best_evaluation": _compact_eval_summary(state.best_evaluation),
        "recent_iterations": recent_iterations,
    }


def _classify_failure(candidate_preflight, candidate_result, candidate_evaluation) -> str:
    if not candidate_preflight.ok:
        return "preflight_error"
    if candidate_result.solver_status in {"runtime_error", "execution_error", "preflight_error"}:
        return "runtime_error"
    if any(not rc.passed for rc in candidate_evaluation.rule_checks):
        return "rule_violation"
    if not candidate_evaluation.is_acceptable:
        return "objective_mismatch"
    return "no_improvement"


def _build_repair_signal(
    problem,
    candidate_program,
    candidate_result,
    candidate_evaluation,
    decision,
    previous_best_program,
    candidate_preflight,
    proposer_guidance: dict[str, Any],
    candidate_changed_structure: bool,
    accepted: bool,
):
    if accepted:
        return {
            "failure_type": None,
            "summary": "Accepted solution found.",
            "must_fix": [],
            "violated_rules": [],
            "missing_objective_terms": [],
            "objective_mismatch": {
                "reported_objective_value": candidate_evaluation.reported_objective_value,
                "computed_objective_value": candidate_evaluation.computed_objective_value,
            },
            "current_assignments": [a.model_dump() for a in candidate_result.assignments],
            "forbidden_patterns": ["```", "data['problem']", 'data["problem"]'],
            "last_traceback": "\n".join(candidate_result.notes),
            "modeling_guidance": [
                "Current best candidate is accepted.",
                "Further iterations should only continue if you intentionally search for a strictly better evaluator objective.",
            ],
            "latest_rule_checks": [rc.model_dump() for rc in candidate_evaluation.rule_checks],
            "latest_objective_terms": dict(candidate_evaluation.objective_terms),
            "problem_objective": problem.evaluation_framework.get("objective", {}),
            "current_best_code_preview": _normalized(candidate_program.model_logic)[:4000],
            "candidate_changed_structure": candidate_changed_structure,
            "problem_active_rule_ids": list(proposer_guidance.get("problem_active_rule_ids", [])),
            "problem_active_objective_id": proposer_guidance.get("problem_active_objective_id", ""),
            "active_rules_guidance": proposer_guidance.get("active_rules_guidance", []),
            "active_objective_guidance": proposer_guidance.get("active_objective_guidance", []),
            "latest_rule_results_from_guidance": proposer_guidance.get("latest_rule_results", []),
            "latest_objective_terms_from_guidance": proposer_guidance.get("latest_objective_terms", {}),
            "latest_violations_from_guidance": proposer_guidance.get("latest_violations_from_guidance", []),
            "latest_acceptability_from_guidance": proposer_guidance.get("latest_acceptability_from_guidance", {}),
        }

    failure_type = _classify_failure(candidate_preflight, candidate_result, candidate_evaluation)

    violated_rules = [
        {"rule_id": rc.rule_id, "details": rc.details}
        for rc in candidate_evaluation.rule_checks
        if not rc.passed
    ]

    objective_terms = dict(candidate_evaluation.objective_terms)
    missing_objective_terms = []
    if failure_type == "objective_mismatch":
        reported = float(candidate_evaluation.reported_objective_value or 0.0)
        assignment_cost = float(objective_terms.get("assignment_cost", 0.0))
        if abs(reported - assignment_cost) <= 1e-9:
            for term, value in objective_terms.items():
                if term != "assignment_cost" and abs(float(value)) > 1e-9:
                    missing_objective_terms.append(
                        {
                            "term": term,
                            "reported_contribution": 0.0,
                            "computed_contribution": float(value),
                            "details": "This term appears active in the evaluator but is missing from the solver objective.",
                        }
                    )
        else:
            for term, value in objective_terms.items():
                missing_objective_terms.append(
                    {
                        "term": term,
                        "reported_contribution": None,
                        "computed_contribution": float(value),
                        "details": "Evaluator detected this active objective contribution, but the solver-reported objective does not match the full computed objective.",
                    }
                )

    if (not candidate_changed_structure) and failure_type in {"objective_mismatch", "no_improvement"}:
        failure_type = "no_structural_change"

    if failure_type == "rule_violation":
        must_fix = [
            "One or more hard rules are violated.",
            "Change the MILP constraints so the solver cannot produce assignments violating those rules.",
            "Do not rely on extract_assignments to filter invalid assignments after solving.",
        ]
        summary = "Hard-rule violations were detected; the MILP constraints must be strengthened."
    elif failure_type == "objective_mismatch":
        missing_terms_text = ", ".join(m["term"] for m in missing_objective_terms) or "active evaluation terms"
        must_fix = [
            f"The MILP objective currently omits or mis-models: {missing_terms_text}.",
            "Modify the MILP objective itself, not just output formatting.",
            "Add auxiliary variables and linking constraints if needed to represent missing objective terms linearly.",
            "Return an objective_value equal to the optimized full MILP objective.",
        ]
        summary = "The current MILP is feasible but does not optimize the full evaluation objective."
    elif failure_type == "no_structural_change":
        must_fix = [
            "The candidate repeated the current baseline or made no meaningful structural change.",
            "Change the MILP objective or constraints, not just comments or formatting.",
            "For objective mismatch, explicitly introduce variables/constraints for the missing term and add it to the objective.",
        ]
        summary = "Candidate made no meaningful structural change."
    elif failure_type == "runtime_error":
        must_fix = [
            "Fix syntax/runtime issues first.",
            "Return valid executable Python for build_model and extract_assignments.",
            "Preserve existing working behavior while repairing execution.",
        ]
        summary = "Generated program failed during execution."
    elif failure_type == "preflight_error":
        must_fix = [
            "Fix preflight issues before changing model structure.",
            "Return only raw Python code with the required functions.",
        ]
        summary = "Generated program failed preflight checks."
    else:
        must_fix = [
            "Preserve the current best behavior and improve only one weakness.",
            "If the objective is already modeled correctly, search for a strictly better feasible objective.",
        ]
        summary = decision.reason

    return {
        "failure_type": failure_type,
        "summary": summary,
        "must_fix": must_fix,
        "violated_rules": violated_rules,
        "missing_objective_terms": missing_objective_terms,
        "objective_mismatch": {
            "reported_objective_value": candidate_evaluation.reported_objective_value,
            "computed_objective_value": candidate_evaluation.computed_objective_value,
        },
        "current_assignments": [a.model_dump() for a in candidate_result.assignments],
        "forbidden_patterns": ["```", "data['problem']", 'data["problem"]'],
        "last_traceback": "\n".join(candidate_result.notes),
        "modeling_guidance": [
            "Keep existing working feasibility constraints unless they directly cause a violation.",
            "If hard rules are violated, change the MILP constraints rather than only changing output formatting.",
            "If objective terms are missing, change the MILP objective and add auxiliary variables/constraints if needed.",
            "Do not fake compliance by filtering invalid assignments in extract_assignments after solving.",
            "A candidate that does not materially change the MILP structure may be rejected as no_structural_change.",
        ],
        "latest_rule_checks": [rc.model_dump() for rc in candidate_evaluation.rule_checks],
        "latest_objective_terms": objective_terms,
        "problem_objective": problem.evaluation_framework.get("objective", {}),
        "current_best_code_preview": _normalized(previous_best_program.model_logic)[:4000],
        "candidate_changed_structure": candidate_changed_structure,
        "problem_active_rule_ids": list(proposer_guidance.get("problem_active_rule_ids", [])),
        "problem_active_objective_id": proposer_guidance.get("problem_active_objective_id", ""),
        "active_rules_guidance": proposer_guidance.get("active_rules_guidance", []),
        "active_objective_guidance": proposer_guidance.get("active_objective_guidance", []),
        "latest_rule_results_from_guidance": proposer_guidance.get("latest_rule_results", []),
        "latest_objective_terms_from_guidance": proposer_guidance.get("latest_objective_terms", {}),
        "latest_violations_from_guidance": proposer_guidance.get("latest_violations_from_guidance", []),
        "latest_acceptability_from_guidance": proposer_guidance.get("latest_acceptability_from_guidance", {}),
    }


def _write_iteration_debug(run_dir: Path, iteration_index: int, payload: dict[str, Any]) -> None:
    debug_dir = run_dir / "iteration_debug"
    ensure_dir(debug_dir)
    path = debug_dir / f"iteration_{iteration_index:03d}_summary.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _log_start(problem: ProblemDefinition, settings: Settings) -> None:
    print(f"[start] problem={problem.name} | max_iterations={settings.max_iterations}")


def _log_baseline(best_result, best_evaluation) -> None:
    print(
        f"[baseline] status={best_result.solver_status} | "
        f"feasible={best_evaluation.is_feasible} | "
        f"acceptable={best_evaluation.is_acceptable} | "
        f"objective={_objective_value_for_log(best_evaluation)}"
    )


def _log_iteration(iteration_number: int, candidate_result, candidate_evaluation, accepted: bool, repair_signal: dict[str, Any]) -> None:
    top_violation = _top_violation(candidate_evaluation)
    objective_terms = dict(getattr(candidate_evaluation, "objective_terms", {}) or {})
    terms_text = ", ".join(f"{k}={v}" for k, v in objective_terms.items()) if objective_terms else "no_terms"
    msg = (
        f"[iter {iteration_number}] status={candidate_result.solver_status} | "
        f"feasible={candidate_evaluation.is_feasible} | "
        f"acceptable={candidate_evaluation.is_acceptable} | "
        f"objective={_objective_value_for_log(candidate_evaluation)} | "
        f"accepted={accepted} | "
        f"failure_type={repair_signal.get('failure_type')} | "
        f"changed_structure={repair_signal.get('candidate_changed_structure')} | "
        f"terms=({terms_text})"
    )
    if top_violation:
        msg += f" | top_violation={top_violation}"
    print(msg)


def build_graph(settings: Settings, run_dir: Path):
    ensure_dir(run_dir)
    llm = LLMClient()

    def run(problem: ProblemDefinition) -> RunState:
        _log_start(problem, settings)

        best_program = baseline_program()
        best_preflight = run_preflight(best_program)
        best_program.model_logic = best_preflight.normalized_code
        best_result = execute_program(problem, best_program)
        best_evaluation = evaluate_result(problem, best_result)
        _log_baseline(best_result, best_evaluation)

        state = RunState(
            problem=problem,
            current_iteration=0,
            max_iterations=settings.max_iterations,
            best_program=best_program,
            best_preflight=best_preflight,
            best_result=best_result,
            best_evaluation=best_evaluation,
            latest_program=best_program,
            latest_preflight=best_preflight,
            latest_result=best_result,
            latest_evaluation=best_evaluation,
        )

        for i in range(settings.max_iterations):
            previous_best_program = state.best_program.model_copy(deep=True)
            proposer_guidance = extract_proposer_guidance(
                problem=problem.model_dump(),
                evaluation=state.best_evaluation.model_dump(),
            )
            run_memory = _build_run_memory(state)

            candidate_program = llm.propose(
                problem=problem,
                current_best=state.best_program,
                repair_signal=state.repair_signal,
                proposer_guidance=proposer_guidance,
                run_memory=run_memory,
            )
            candidate_preflight = run_preflight(candidate_program)
            candidate_program.model_logic = candidate_preflight.normalized_code

            candidate_changed_structure = _candidate_changed_structure(
                candidate_code=candidate_program.model_logic,
                reference_code=previous_best_program.model_logic,
            )
            same_as_best = _normalized(candidate_program.model_logic) == _normalized(previous_best_program.model_logic)

            if same_as_best and state.repair_signal.get("failure_type") in {"objective_mismatch", "no_structural_change"}:
                candidate_result = SolverResult(
                    solver_status="rejected_without_execution",
                    objective_value=None,
                    assignments=[],
                    notes=["Candidate code was effectively identical to current best and was rejected before execution."],
                )
                candidate_evaluation = state.best_evaluation
                decision = should_accept_candidate(candidate_evaluation, state.best_evaluation)
                decision.accepted = False
                decision.reason = "Candidate repeated current best without structural change; skipping execution."
                accepted = False
            else:
                if candidate_preflight.ok:
                    candidate_result = execute_program(problem, candidate_program)
                else:
                    candidate_result = SolverResult(
                        solver_status="preflight_error",
                        objective_value=None,
                        assignments=[],
                        notes=candidate_preflight.errors,
                    )
                candidate_evaluation = evaluate_result(problem, candidate_result)
                decision = should_accept_candidate(candidate_evaluation, state.best_evaluation)
                accepted = decision.accepted

            repair_signal = _build_repair_signal(
                problem=problem,
                candidate_program=candidate_program,
                candidate_result=candidate_result,
                candidate_evaluation=candidate_evaluation,
                decision=decision,
                previous_best_program=previous_best_program,
                candidate_preflight=candidate_preflight,
                proposer_guidance=proposer_guidance,
                candidate_changed_structure=candidate_changed_structure,
                accepted=accepted,
            )

            if accepted:
                state.best_program = candidate_program
                state.best_preflight = candidate_preflight
                state.best_result = candidate_result
                state.best_evaluation = candidate_evaluation

            rec = IterationRecord(
                iteration=i,
                candidate_program=candidate_program,
                candidate_preflight=candidate_preflight,
                candidate_result=candidate_result,
                candidate_evaluation=candidate_evaluation,
                accepted_as_best=accepted,
                decision=decision,
                best_evaluation_after=state.best_evaluation,
                repair_signal=repair_signal,
            )
            state.history.append(rec)
            state.current_iteration = i + 1
            state.repair_signal = repair_signal
            state.latest_program = candidate_program
            state.latest_preflight = candidate_preflight
            state.latest_result = candidate_result
            state.latest_evaluation = candidate_evaluation

            dump_json(run_dir / f"iteration_{i}.json", rec)
            _write_iteration_debug(
                run_dir=run_dir,
                iteration_index=i,
                payload={
                    "iteration": i,
                    "accepted": accepted,
                    "decision_reason": decision.reason,
                    "candidate_solver_status": candidate_result.solver_status,
                    "candidate_changed_structure": candidate_changed_structure,
                    "repair_signal": repair_signal,
                    "candidate_evaluation": _compact_eval_summary(candidate_evaluation),
                    "best_evaluation_after": _compact_eval_summary(state.best_evaluation),
                },
            )
            _log_iteration(i + 1, candidate_result, candidate_evaluation, accepted, repair_signal)

            if state.best_evaluation.is_acceptable:
                state.stop = True
                state.final_message = "Accepted solution found from current best solver."
                state.repair_signal = {
                    "failure_type": None,
                    "summary": "Accepted solution found.",
                    "latest_rule_checks": [rc.model_dump() for rc in state.best_evaluation.rule_checks],
                    "latest_objective_terms": dict(state.best_evaluation.objective_terms),
                    "current_assignments": [a.model_dump() for a in state.best_result.assignments],
                    "candidate_changed_structure": candidate_changed_structure,
                }
                break

        if not state.stop:
            state.final_message = "Maximum iterations reached."
        return state

    return run
