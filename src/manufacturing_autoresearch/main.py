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
    print("Run finished.")
    print(final_state.model_dump())


if __name__ == "__main__":
    app()
