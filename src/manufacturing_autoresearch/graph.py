from __future__ import annotations

import json
from pathlib import Path

from .baseline import baseline_program
from .evaluator import evaluate_result
from .execution import execute_program
from .modeling_lessons import build_modeling_lessons, lesson_to_prompt_dict
from .llm import LLMClient
from .logging_utils import dump_json, ensure_dir
from .preflight import run_preflight
from .proposer_guidance import extract_proposer_guidance
from .selector import should_accept_candidate
from .types import IterationRecord, ProblemDefinition, RunState, Settings, SolverResult
from .lesson_selector import select_lessons

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


def _build_run_memory(state: RunState) -> dict:
    recent = state.history[-3:]

    persistent_failure_types = []
    persistent_missing_terms = set()

    rules_already_satisfied = []
    if state.best_evaluation:
        rules_already_satisfied = [
            rc.rule_id for rc in state.best_evaluation.rule_checks if rc.passed
        ]

    for rec in recent:
        ft = rec.repair_signal.get("failure_type")
        if ft:
            persistent_failure_types.append(ft)

        for item in rec.repair_signal.get("missing_objective_terms", []):
            term = item.get("term")
            if term:
                persistent_missing_terms.add(term)

    return {
        "iterations_seen": state.current_iteration,
        "persistent_failure_types": persistent_failure_types,
        "persistent_missing_objective_terms": sorted(persistent_missing_terms),
        "rules_already_satisfied": rules_already_satisfied,
        "recent_attempt_summaries": [
            {
                "iteration": rec.iteration,
                "failure_type": rec.repair_signal.get("failure_type"),
                "candidate_changed_structure": rec.repair_signal.get("candidate_changed_structure"),
                "missing_objective_terms": [
                    item.get("term")
                    for item in rec.repair_signal.get("missing_objective_terms", [])
                ],
                "selected_modeling_lesson_ids": [
                    item.get("lesson_id")
                    for item in (rec.selected_modeling_lessons or [])
                    if isinstance(item, dict) and item.get("lesson_id")
                ],
            }
            for rec in recent
        ],
    }


def _objective_terms_text(candidate_evaluation) -> str:
    terms = candidate_evaluation.objective_terms or {}
    if not terms:
        return "terms=(none)"
    inner = ", ".join(f"{k}={float(v)}" for k, v in terms.items())
    return f"terms=({inner})"


def _print_start(problem: ProblemDefinition, settings: Settings) -> None:
    print(f"[start] problem={problem.name} | max_iterations={settings.max_iterations}")


def _print_baseline(best_result, best_evaluation) -> None:
    print(
        "[baseline] "
        f"status={best_result.solver_status} | "
        f"feasible={best_evaluation.is_feasible} | "
        f"acceptable={best_evaluation.is_acceptable} | "
        f"objective={best_evaluation.computed_objective_value}"
    )


def _print_live_iteration_update(
    i,
    candidate_result,
    candidate_evaluation,
    decision,
    repair_signal,
) -> None:
    terms_text = _objective_terms_text(candidate_evaluation)
    print(
        f"[iter {i + 1}] "
        f"status={candidate_result.solver_status} | "
        f"feasible={candidate_evaluation.is_feasible} | "
        f"acceptable={candidate_evaluation.is_acceptable} | "
        f"objective={candidate_evaluation.computed_objective_value} | "
        f"accepted={decision.accepted} | "
        f"failure_type={repair_signal.get('failure_type')} | "
        f"changed_structure={repair_signal.get('candidate_changed_structure')} | "
        f"{terms_text}"
    )

    violations = candidate_evaluation.violations or []
    if violations:
        print(f"  top_violation={violations[0]}")

    diagnosis = candidate_evaluation.infeasibility_diagnosis or []
    if diagnosis:
        print(f"  infeasibility_hint={diagnosis[0].get('message', '')}")

    selected_lessons = repair_signal.get("selected_modeling_lessons", []) or []
    if selected_lessons:
        lesson_bits = []
        for item in selected_lessons[:3]:
            lid = item.get("lesson_id", "")
            reason = (
                item.get("selection_reason")
                or item.get("llm_reason")
                or ""
            )

            if lid and reason:
                # optional: truncate long reasons
                reason = reason[:80] + "..." if len(reason) > 80 else reason
                lesson_bits.append(f"{lid} ({reason})")
            elif lid:
                lesson_bits.append(lid)

        if lesson_bits:
            print(f"  lessons={'; '.join(lesson_bits)}")


def _build_repair_signal(
    problem,
    candidate_program,
    candidate_result,
    candidate_evaluation,
    decision,
    best_program,
    candidate_preflight,
    proposer_guidance,
    selected_modeling_lessons,
):
    if decision.accepted and candidate_evaluation.is_acceptable:
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
            "modeling_guidance": [],
            "latest_rule_checks": [rc.model_dump() for rc in candidate_evaluation.rule_checks],
            "latest_objective_terms": dict(candidate_evaluation.objective_terms),
            "problem_objective": problem.evaluation_framework.get("objective", {}),
            "current_best_code_preview": _normalized(best_program.model_logic)[:4000],
            "candidate_changed_structure": _normalized(candidate_program.model_logic) != _normalized(best_program.model_logic),
            "selected_modeling_lessons": selected_modeling_lessons,
            "infeasibility_diagnosis": list(candidate_evaluation.infeasibility_diagnosis or []),
            "problem_active_rule_ids": proposer_guidance.get("problem_active_rule_ids", []),
            "problem_active_objective_id": proposer_guidance.get("problem_active_objective_id", ""),
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

    novelty = _normalized(candidate_program.model_logic) != _normalized(best_program.model_logic)

    if not novelty and failure_type in {"objective_mismatch", "no_improvement"}:
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
        "current_best_code_preview": _normalized(best_program.model_logic)[:4000],
        "candidate_changed_structure": novelty,
        "selected_modeling_lessons": selected_modeling_lessons,
        "problem_active_rule_ids": proposer_guidance.get("problem_active_rule_ids", []),
        "problem_active_objective_id": proposer_guidance.get("problem_active_objective_id", ""),
        "infeasibility_diagnosis": list(candidate_evaluation.infeasibility_diagnosis or []),
        "active_rules_guidance": proposer_guidance.get("active_rules_guidance", []),
        "active_objective_guidance": proposer_guidance.get("active_objective_guidance", []),
        "latest_rule_results_from_guidance": proposer_guidance.get("latest_rule_results", []),
        "latest_objective_terms_from_guidance": proposer_guidance.get("latest_objective_terms", {}),
    }


def build_graph(settings: Settings, run_dir: Path):
    ensure_dir(run_dir)
    llm = LLMClient()

    def run(problem: ProblemDefinition) -> RunState:
        _print_start(problem, settings)

        best_program = baseline_program()
        best_preflight = run_preflight(best_program)
        best_program.model_logic = best_preflight.normalized_code
        best_result = execute_program(problem, best_program)
        best_evaluation = evaluate_result(problem, best_result)

        _print_baseline(best_result, best_evaluation)

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

        # Early stop if the baseline is already acceptable.
        if best_evaluation.is_feasible and best_evaluation.is_acceptable:
            print("[stop] baseline already acceptable | no repair iterations needed")

            state.stop = True
            state.final_message = "Baseline solution already acceptable; no repair iterations needed."
            state.repair_signal = {
                "failure_type": None,
                "summary": "Baseline solution already acceptable.",
                "must_fix": [],
                "violated_rules": [],
                "missing_objective_terms": [],
                "objective_mismatch": {
                    "reported_objective_value": best_evaluation.reported_objective_value,
                    "computed_objective_value": best_evaluation.computed_objective_value,
                },
                "current_assignments": [a.model_dump() for a in best_result.assignments],
                "forbidden_patterns": ["```", "data['problem']", 'data["problem"]'],
                "last_traceback": "\n".join(best_result.notes or []),
                "modeling_guidance": [],
                "latest_rule_checks": [rc.model_dump() for rc in best_evaluation.rule_checks],
                "latest_objective_terms": dict(best_evaluation.objective_terms or {}),
                "problem_objective": problem.evaluation_framework.get("objective", {}),
                "current_best_code_preview": _normalized(best_program.model_logic)[:4000],
                "candidate_changed_structure": False,
                "selected_modeling_lessons": [],
                "infeasibility_diagnosis": list(best_evaluation.infeasibility_diagnosis or []),
                "problem_active_rule_ids": [
                    str(rule.get("id", ""))
                    for rule in problem.evaluation_framework.get("hard_rules", [])
                    if rule.get("id")
                ],
                "problem_active_objective_id": str(
                    (problem.evaluation_framework.get("objective", {}) or {}).get("id", "")
                ),
            }
            return state

        for i in range(settings.max_iterations):
            proposer_guidance = extract_proposer_guidance(
                problem=problem.model_dump(),
                evaluation=state.best_evaluation.model_dump() if state.best_evaluation else None,
            )
            run_memory = _build_run_memory(state)

            available_lessons = [
                lesson_to_prompt_dict(lesson)
                for lesson in build_modeling_lessons().values()
            ]
            selected_modeling_lessons = select_lessons(
                llm=llm,
                problem=problem,
                repair_signal=state.repair_signal,
                proposer_guidance=proposer_guidance,
                available_lessons=available_lessons,
                run_memory=run_memory,
            )
            proposer_guidance["selected_modeling_lessons"] = selected_modeling_lessons
                        
            candidate_program, reasoning_plan = llm.propose(
                problem=problem,
                current_best=state.best_program,
                repair_signal=state.repair_signal,
                proposer_guidance=proposer_guidance,
                run_memory=run_memory,
                selected_modeling_lessons=selected_modeling_lessons,
            )

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
                if isinstance(reasoning_plan, dict):
                    candidate_result.notes.append("reasoning_plan=" + json.dumps(reasoning_plan, ensure_ascii=False))
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
                if isinstance(reasoning_plan, dict):
                    candidate_result.notes.append("reasoning_plan=" + json.dumps(reasoning_plan, ensure_ascii=False))
                candidate_evaluation = evaluate_result(problem, candidate_result)
                decision = should_accept_candidate(candidate_evaluation, state.best_evaluation)
                accepted = decision.accepted

            best_program_for_signal = state.best_program
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
                best_program=best_program_for_signal,
                candidate_preflight=candidate_preflight,
                proposer_guidance=proposer_guidance,
                selected_modeling_lessons=selected_modeling_lessons,
            )

            state.latest_program = candidate_program
            state.latest_preflight = candidate_preflight
            state.latest_result = candidate_result
            state.latest_evaluation = candidate_evaluation
            state.repair_signal = repair_signal
            state.current_iteration = i + 1

            record = IterationRecord(
                iteration=i,
                candidate_program=candidate_program,
                candidate_preflight=candidate_preflight,
                candidate_result=candidate_result,
                candidate_evaluation=candidate_evaluation,
                accepted_as_best=accepted,
                decision=decision,
                best_evaluation_after=state.best_evaluation,
                repair_signal=repair_signal,
                selected_modeling_lessons=selected_modeling_lessons,
                reasoning_plan=reasoning_plan if isinstance(reasoning_plan, dict) else {},
            )
            state.history.append(record)

            dump_json(run_dir / f"iteration_{i}.json", record.model_dump())
            dump_json(
                run_dir / f"iteration_{i:03d}_summary.json",
                {
                    "iteration": i,
                    "candidate_solver_status": candidate_result.solver_status,
                    "candidate_feasible": candidate_evaluation.is_feasible,
                    "candidate_acceptable": candidate_evaluation.is_acceptable,
                    "candidate_computed_objective": candidate_evaluation.computed_objective_value,
                    "candidate_reported_objective": candidate_evaluation.reported_objective_value,
                    "accepted": accepted,
                    "decision_reason": decision.reason,
                    "failure_type": repair_signal.get("failure_type"),
                    "selected_modeling_lesson_ids": [
                        item.get("lesson_id")
                        for item in selected_modeling_lessons
                        if item.get("lesson_id")
                    ],
                    "selected_modeling_lesson_reasons": [
                        {
                            "lesson_id": item.get("lesson_id"),
                            "reason": item.get("selection_reason") or item.get("llm_reason", ""),
                            "priority_hint": item.get("priority_hint"),
                            "llm_reason": item.get("llm_reason", ""),
                            "llm_priority": item.get("llm_priority"),
                        }
                        for item in selected_modeling_lessons
                        if item.get("lesson_id")
                    ],
                    "missing_objective_terms": [
                        item.get("term")
                        for item in repair_signal.get("missing_objective_terms", [])
                    ],
                    "top_violation": (candidate_evaluation.violations or [None])[0],
                    "infeasibility_diagnosis": list(candidate_evaluation.infeasibility_diagnosis or []),
                },
            )

            _print_live_iteration_update(
                i=i,
                candidate_result=candidate_result,
                candidate_evaluation=candidate_evaluation,
                decision=decision,
                repair_signal=repair_signal,
            )

            if state.best_evaluation.is_acceptable and accepted:
                state.stop = True
                state.final_message = "Accepted solution found from current best solver."
                break

        if not state.stop:
            state.final_message = "Maximum iterations reached."
        return state

    return run
