from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from rich import print

from .graph import build_graph
from .logging_utils import dump_json, ensure_dir
from .types import ProblemDefinition, Settings

app = typer.Typer(add_completion=False)


def load_problem(path: Path) -> ProblemDefinition:
    with path.open("r", encoding="utf-8") as f:
        return ProblemDefinition.model_validate(json.load(f))


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _build_short_terminal_summary(final_state_dict: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    best_eval = final_state_dict.get("best_evaluation") or {}
    latest_eval = final_state_dict.get("latest_evaluation") or {}
    latest_signal = final_state_dict.get("repair_signal") or {}
    best_result = final_state_dict.get("best_result") or {}

    latest_violations = latest_eval.get("violations") or []

    return {
        "final_message": final_state_dict.get("final_message"),
        "iterations_completed": final_state_dict.get("current_iteration"),
        "best_solver_status": best_result.get("solver_status"),
        "best_is_feasible": best_eval.get("is_feasible"),
        "best_is_acceptable": best_eval.get("is_acceptable"),
        "best_computed_objective": best_eval.get("computed_objective_value"),
        "best_reported_objective": best_eval.get("reported_objective_value"),
        "latest_failure_type": latest_signal.get("failure_type"),
        "latest_summary": latest_signal.get("summary"),
        "latest_top_violation": latest_violations[0] if latest_violations else None,
        "detailed_report_path": str(out_dir / "final_state_full.json"),
        "final_model_path": str(out_dir / "final_milp_model.py"),
        "final_explanation_path": str(out_dir / "final_milp_explanation.md"),
        "final_program_proposal_path": str(out_dir / "final_program_proposal.json"),
    }


def _build_final_milp_explanation(final_state_dict: dict[str, Any]) -> str:
    problem = final_state_dict.get("problem") or {}
    evaluation_framework = problem.get("evaluation_framework") or {}
    objective = evaluation_framework.get("objective") or {}

    best_program = final_state_dict.get("best_program") or {}
    best_result = final_state_dict.get("best_result") or {}
    best_eval = final_state_dict.get("best_evaluation") or {}
    repair_signal = final_state_dict.get("repair_signal") or {}

    lines: list[str] = []

    lines.append("# Final Accepted MILP Model")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append(f"- Problem: `{problem.get('name', '')}`")
    lines.append(f"- Solver status: `{best_result.get('solver_status', '')}`")
    lines.append(f"- Reported objective value: `{best_result.get('objective_value', None)}`")
    lines.append(f"- Evaluator-computed objective value: `{best_eval.get('computed_objective_value', None)}`")
    lines.append("")

    if objective:
        lines.append("## Target Objective")
        lines.append("")
        lines.append(f"- Objective ID: `{objective.get('id', '')}`")
        lines.append(f"- Description: {objective.get('description', '')}")
        lines.append("")

    lines.append("## Accepted Program Explanation")
    lines.append("")
    lines.append(best_program.get("explanation", ""))
    lines.append("")

    lines.append("## Objective Term Breakdown")
    lines.append("")
    objective_terms = best_eval.get("objective_terms") or {}
    if objective_terms:
        for term_name, term_value in objective_terms.items():
            lines.append(f"- `{term_name}`: `{term_value}`")
    else:
        lines.append("- No objective term breakdown available.")
    lines.append("")

    lines.append("## Rule Check Results")
    lines.append("")
    rule_checks = best_eval.get("rule_checks") or []
    if rule_checks:
        for item in rule_checks:
            status = "passed" if item.get("passed", False) else "failed"
            lines.append(f"- `{item.get('rule_id', '')}`: `{status}` — {item.get('details', '')}")
    else:
        lines.append("- No rule checks recorded.")
    lines.append("")

    lines.append("## Final Assignments")
    lines.append("")
    assignments = best_result.get("assignments") or []
    if assignments:
        for item in assignments:
            lines.append(
                f"- job `{item.get('job')}` -> machine `{item.get('machine')}` at slot `{item.get('slot')}`"
            )
    else:
        lines.append("- No assignments recorded.")
    lines.append("")

    lines.append("## Final Notes")
    lines.append("")
    notes = best_result.get("notes") or []
    if notes:
        for note in notes:
            lines.append(f"- {note}")
    else:
        lines.append("- No notes.")
    lines.append("")

    lines.append("## Latest Repair Signal Snapshot")
    lines.append("")
    lines.append(f"- Failure type: `{repair_signal.get('failure_type', '')}`")
    lines.append(f"- Summary: {repair_signal.get('summary', '')}")
    lines.append("")

    lines.append("## Final Model Code")
    lines.append("")
    lines.append("```python")
    lines.append(best_program.get("model_logic", ""))
    lines.append("```")
    lines.append("")

    return "\n".join(lines)


def _save_final_milp_artifacts(out_dir: Path, final_state_dict: dict[str, Any]) -> None:
    best_program = final_state_dict.get("best_program") or {}
    best_result = final_state_dict.get("best_result") or {}
    best_eval = final_state_dict.get("best_evaluation") or {}

    if not best_program:
        return

    _write_json(out_dir / "final_program_proposal.json", best_program)

    model_logic = best_program.get("model_logic", "")
    if model_logic:
        header = '"""Final accepted MILP model generated by manufacturing_autoresearch."""\n\n'
        _write_text(out_dir / "final_milp_model.py", header + model_logic + "\n")

    explanation_md = _build_final_milp_explanation(final_state_dict)
    _write_text(out_dir / "final_milp_explanation.md", explanation_md)

    _write_json(
        out_dir / "final_summary.json",
        {
            "final_message": final_state_dict.get("final_message"),
            "iterations_completed": final_state_dict.get("current_iteration"),
            "solver_status": best_result.get("solver_status"),
            "objective_value": best_result.get("objective_value"),
            "computed_objective_value": best_eval.get("computed_objective_value"),
            "reported_objective_value": best_eval.get("reported_objective_value"),
            "objective_terms": best_eval.get("objective_terms") or {},
            "assignments": best_result.get("assignments") or [],
            "notes": best_result.get("notes") or [],
            "summary": best_eval.get("summary"),
            "violations": best_eval.get("violations") or [],
        },
    )


@app.command()
def main(
    problem: Path = typer.Option(..., exists=True, dir_okay=False),
    out: Path = typer.Option(...),
    max_iters: int = typer.Option(5),
) -> None:
    settings = Settings(max_iterations=max_iters)
    ensure_dir(out)

    runner = build_graph(settings=settings, run_dir=out)
    problem_def = load_problem(problem)
    final_state = runner(problem_def)

    final_state_dict = final_state.model_dump()

    dump_json(out / "final_state.json", final_state)
    _write_json(out / "final_state_full.json", final_state_dict)
    _save_final_milp_artifacts(out, final_state_dict)

    summary = _build_short_terminal_summary(final_state_dict, out)

    print("\n[bold green]Run finished.[/bold green]")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    app()