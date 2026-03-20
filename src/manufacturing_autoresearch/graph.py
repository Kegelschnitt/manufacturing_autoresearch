
from __future__ import annotations

from pathlib import Path

from .baseline import baseline_program
from .evaluator import evaluate_result
from .execution import execute_program
from .llm import LLMClient
from .logging_utils import dump_json, ensure_dir
from .preflight import run_preflight
from .selector import should_accept_candidate
from .types import IterationRecord, ProblemDefinition, RunState, Settings, SolverResult


def _build_repair_signal(
    problem: ProblemDefinition,
    candidate_result: SolverResult,
    candidate_evaluation,
    decision,
) -> dict:
    violated_rules = [
        {"rule_id": rc.rule_id, "details": rc.details}
        for rc in candidate_evaluation.rule_checks
        if not rc.passed
    ]

    missing_objective_terms = []
    reported_obj = candidate_evaluation.reported_objective_value
    computed_obj = candidate_evaluation.computed_objective_value
    if reported_obj is not None and abs(float(reported_obj) - float(computed_obj)) > 1e-9:
        for term, value in candidate_evaluation.objective_terms.items():
            if float(value) != 0.0:
                missing_objective_terms.append(
                    {
                        "term": term,
                        "reported_contribution": None,
                        "computed_contribution": float(value),
                        "details": "Evaluator detected this active objective contribution, but the solver-reported objective does not match the full computed objective.",
                    }
                )

    if candidate_result.solver_status == "preflight_error":
        failure_type = "preflight_error"
    elif candidate_result.solver_status in {"runtime_error", "execution_error"}:
        failure_type = "runtime_error"
    elif violated_rules:
        failure_type = "rule_violation"
    elif not candidate_evaluation.is_acceptable:
        failure_type = "objective_mismatch"
    elif not decision.accepted:
        failure_type = "no_improvement"
    else:
        failure_type = "accepted"

    must_fix = []
    if failure_type == "preflight_error":
        must_fix.extend([
            "Return valid raw Python only.",
            "Ensure both required functions exist: build_model(data) and extract_assignments(data, context).",
        ])
    elif failure_type == "runtime_error":
        must_fix.extend([
            "Fix the runtime error before changing the model structure further.",
            "Keep input access at the top level of data; never use data['problem'] or data[\"problem\"].",
        ])
    elif failure_type == "rule_violation":
        must_fix.extend([
            "Add or strengthen MILP constraints so violated hard rules cannot occur in solved assignments.",
            "Preserve currently satisfied hard rules.",
        ])
    elif failure_type == "objective_mismatch":
        must_fix.extend([
            "The MILP objective currently omits one or more active evaluation objective terms.",
            "Modify the MILP objective itself, not just output formatting.",
            "Return an objective_value equal to the optimized full MILP objective.",
        ])
    elif failure_type == "no_improvement":
        must_fix.extend([
            "Preserve the current best feasible behavior and improve only one weak point.",
            "Prefer a minimal targeted modification over a full rewrite.",
        ])

    signal = {
        "failure_type": failure_type,
        "summary": decision.reason,
        "must_fix": must_fix,
        "violated_rules": violated_rules,
        "missing_objective_terms": missing_objective_terms,
        "objective_mismatch": {
            "reported_objective_value": reported_obj,
            "computed_objective_value": computed_obj,
        },
        "current_assignments": [a.model_dump() for a in candidate_result.assignments],
        "forbidden_patterns": ["```", "data['problem']", 'data["problem"]'],
        "last_traceback": "\n".join(candidate_result.notes),
        "modeling_guidance": [
            "Keep existing working feasibility constraints unless they directly cause a violation.",
            "If hard rules are violated, change the MILP constraints rather than only changing output formatting.",
            "If objective terms are missing, change the MILP objective and add auxiliary variables/constraints if needed.",
            "Do not fake compliance by filtering invalid assignments in extract_assignments after solving.",
        ],
        "latest_rule_checks": [rc.model_dump() for rc in candidate_evaluation.rule_checks],
        "latest_objective_terms": dict(candidate_evaluation.objective_terms),
        "problem_objective": problem.evaluation_framework.get("objective", {}),
    }
    return signal


def build_graph(settings: Settings, run_dir: Path):
    ensure_dir(run_dir)
    llm = LLMClient()

    def run(problem: ProblemDefinition) -> RunState:
        best_program = baseline_program()
        best_preflight = run_preflight(best_program)
        best_program.model_logic = best_preflight.normalized_code
        best_result = execute_program(problem, best_program)
        best_evaluation = evaluate_result(problem, best_result)

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

        state.repair_signal = _build_repair_signal(
            problem=problem,
            candidate_result=best_result,
            candidate_evaluation=best_evaluation,
            decision=type("Decision", (), {"accepted": False, "reason": "Initial baseline evaluation."})(),
        )

        for i in range(settings.max_iterations):
            candidate_program = llm.propose(
                problem=problem,
                current_best=state.best_program,
                repair_signal=state.repair_signal,
                latest_evaluation=state.latest_evaluation,
                latest_result=state.latest_result,
            )

            candidate_preflight = run_preflight(candidate_program)
            candidate_program.model_logic = candidate_preflight.normalized_code

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

            if accepted:
                state.best_program = candidate_program
                state.best_preflight = candidate_preflight
                state.best_result = candidate_result
                state.best_evaluation = candidate_evaluation

            repair_signal = _build_repair_signal(
                problem=problem,
                candidate_result=candidate_result,
                candidate_evaluation=candidate_evaluation,
                decision=decision,
            )

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

            if state.best_evaluation.is_acceptable:
                state.stop = True
                state.final_message = "Accepted solution found from current best solver."
                break

        if not state.stop:
            state.final_message = "Maximum iterations reached."
        return state

    return run
