
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


def _normalized(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


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


def _build_repair_signal(problem, candidate_program, candidate_result, candidate_evaluation, decision, best_program, candidate_preflight):
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
                            "details": "This term appears active in the evaluator but is missing from the solver objective."
                        }
                    )
        else:
            for term, value in objective_terms.items():
                missing_objective_terms.append(
                    {
                        "term": term,
                        "reported_contribution": None,
                        "computed_contribution": float(value),
                        "details": "Evaluator detected this active objective contribution, but the solver-reported objective does not match the full computed objective."
                    }
                )

    novelty = _normalized(candidate_program.model_logic) != _normalized(best_program.model_logic)

    if not novelty and failure_type in {"objective_mismatch", "no_improvement"}:
        failure_type = "no_structural_change"

    if failure_type == "rule_violation":
        must_fix = [
            "One or more hard rules are violated.",
            "Change the MILP constraints so the solver cannot produce assignments violating those rules.",
            "Do not rely on extract_assignments to filter invalid assignments after solving."
        ]
        summary = "Hard-rule violations were detected; the MILP constraints must be strengthened."
    elif failure_type == "objective_mismatch":
        missing_terms_text = ", ".join(m["term"] for m in missing_objective_terms) or "active evaluation terms"
        must_fix = [
            f"The MILP objective currently omits or mis-models: {missing_terms_text}.",
            "Modify the MILP objective itself, not just output formatting.",
            "Add auxiliary variables and linking constraints if needed to represent missing objective terms linearly.",
            "Return an objective_value equal to the optimized full MILP objective."
        ]
        summary = "The current MILP is feasible but does not optimize the full evaluation objective."
    elif failure_type == "no_structural_change":
        must_fix = [
            "The candidate repeated the current baseline or made no meaningful structural change.",
            "Change the MILP objective or constraints, not just comments or formatting.",
            "For objective mismatch, explicitly introduce variables/constraints for the missing term and add it to the objective."
        ]
        summary = "Candidate made no meaningful structural change."
    elif failure_type == "runtime_error":
        must_fix = [
            "Fix syntax/runtime issues first.",
            "Return valid executable Python for build_model and extract_assignments.",
            "Preserve existing working behavior while repairing execution."
        ]
        summary = "Generated program failed during execution."
    elif failure_type == "preflight_error":
        must_fix = [
            "Fix preflight issues before changing model structure.",
            "Return only raw Python code with the required functions."
        ]
        summary = "Generated program failed preflight checks."
    else:
        must_fix = [
            "Preserve the current best behavior and improve only one weakness.",
            "If the objective is already modeled correctly, search for a strictly better feasible objective."
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
            "A candidate that does not materially change the MILP structure may be rejected as no_structural_change."
        ],
        "latest_rule_checks": [rc.model_dump() for rc in candidate_evaluation.rule_checks],
        "latest_objective_terms": objective_terms,
        "problem_objective": problem.evaluation_framework.get("objective", {}),
        "current_best_code_preview": _normalized(best_program.model_logic)[:4000],
        "candidate_changed_structure": novelty,
    }


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

        for i in range(settings.max_iterations):
            candidate_program = llm.propose(problem, state.best_program, state.repair_signal)
            candidate_preflight = run_preflight(candidate_program)
            candidate_program.model_logic = candidate_preflight.normalized_code

            same_as_best = _normalized(candidate_program.model_logic) == _normalized(state.best_program.model_logic)

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

            if accepted:
                state.best_program = candidate_program
                state.best_preflight = candidate_preflight
                state.best_result = candidate_result
                state.best_evaluation = candidate_evaluation

            repair_signal = _build_repair_signal(
                problem=problem,
                candidate_program=candidate_program,
                candidate_result=candidate_result,
                candidate_evaluation=candidate_evaluation,
                decision=decision,
                best_program=state.best_program,
                candidate_preflight=candidate_preflight,
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
