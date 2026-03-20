from __future__ import annotations

import json
from pathlib import Path

import typer
from rich import print

from .graph import build_graph
from .logging_utils import dump_json, ensure_dir
from .types import ProblemDefinition, Settings

app = typer.Typer(add_completion=False)


def load_problem(path: Path) -> ProblemDefinition:
    with path.open("r", encoding="utf-8") as f:
        return ProblemDefinition.model_validate(json.load(f))


def _build_short_terminal_summary(final_state, out_dir: Path) -> dict:
    best_eval = final_state.best_evaluation
    latest_eval = final_state.latest_evaluation
    latest_signal = final_state.repair_signal or {}

    return {
        "final_message": final_state.final_message,
        "iterations_completed": final_state.current_iteration,
        "best_solver_status": final_state.best_result.solver_status if final_state.best_result else None,
        "best_is_feasible": best_eval.is_feasible if best_eval else None,
        "best_is_acceptable": best_eval.is_acceptable if best_eval else None,
        "best_computed_objective": best_eval.computed_objective_value if best_eval else None,
        "best_reported_objective": best_eval.reported_objective_value if best_eval else None,
        "latest_failure_type": latest_signal.get("failure_type"),
        "latest_summary": latest_signal.get("summary"),
        "latest_top_violation": (
            latest_eval.violations[0] if latest_eval and latest_eval.violations else None
        ),
        "detailed_report_path": str(out_dir / "final_state_full.json"),
    }


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

    dump_json(out / "final_state.json", final_state)

    summary = _build_short_terminal_summary(final_state, out)

    print("\n[bold green]Run finished.[/bold green]")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    app()