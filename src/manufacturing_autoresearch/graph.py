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
            if candidate_preflight.ok:
                candidate_result = execute_program(problem, candidate_program)
            else:
                candidate_result = SolverResult(solver_status="preflight_error", objective_value=None, assignments=[], notes=candidate_preflight.errors)
            candidate_evaluation = evaluate_result(problem, candidate_result)
            decision = should_accept_candidate(candidate_evaluation, state.best_evaluation)
            accepted = decision.accepted
            if accepted:
                state.best_program = candidate_program
                state.best_preflight = candidate_preflight
                state.best_result = candidate_result
                state.best_evaluation = candidate_evaluation
            if not candidate_evaluation.is_feasible:
                failure_type = "infeasible_or_invalid"
            elif not candidate_evaluation.is_acceptable:
                failure_type = "objective_mismatch_or_incomplete_model"
            elif not accepted:
                failure_type = "no_improvement"
            else:
                failure_type = "accepted"
            repair_signal = {
                "failure_type": failure_type,
                "must_fix": ["Preserve the current best behavior and improve only one weakness."],
                "forbidden_patterns": ["```", "data['problem']", 'data["problem"]'],
                "last_traceback": "\n" .join(candidate_result.notes),
                "summary": decision.reason,
            }
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
