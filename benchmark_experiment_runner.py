#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_get(d: dict[str, Any], *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def run_problem(problem_path: Path, out_dir: Path, max_iters: int, python_exe: str) -> dict[str, Any]:
    cmd = [
        python_exe,
        "-m",
        "manufacturing_autoresearch.main",
        "--problem",
        str(problem_path),
        "--out",
        str(out_dir),
        "--max-iters",
        str(max_iters),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "cmd": cmd,
    }


def parse_run(run_dir: Path, problem_name: str, pass_idx: int) -> dict[str, Any]:
    final_summary_path = run_dir / "final_summary.json"
    final_state_path = run_dir / "final_state_full.json"

    summary = load_json(final_summary_path) if final_summary_path.exists() else None
    final_state = load_json(final_state_path) if final_state_path.exists() else None

    if summary is None and final_state is None:
        return {
            "problem": problem_name,
            "pass": pass_idx,
            "run_dir": str(run_dir),
            "status": "missing_final_outputs",
            "accepted_solution_found": False,
            "best_feasible_candidate_found": False,
            "iterations_completed": None,
            "best_solver_status": None,
            "best_is_feasible": None,
            "best_is_acceptable": None,
            "best_computed_objective": None,
            "latest_failure_type": None,
            "latest_summary": None,
            "adaptive_added_count": 0,
            "adaptive_merged_count": 0,
            "adaptive_rejected_count": 0,
            "selected_lessons_total": 0,
            "selected_lessons_unique": 0,
            "curation_reports": 0,
        }

    if summary is not None:
        accepted_solution_found = bool(summary.get("accepted_solution_found"))
        best_feasible_candidate_found = bool(summary.get("best_feasible_candidate_found"))
        iterations_completed = summary.get("iterations_completed")
        best_solver_status = summary.get("best_solver_status")
        best_is_feasible = summary.get("best_is_feasible")
        best_is_acceptable = summary.get("best_is_acceptable")
        best_computed_objective = summary.get("best_computed_objective")
        latest_failure_type = summary.get("latest_failure_type")
        latest_summary = summary.get("latest_summary")
        status_name = "ok_final_summary"
    else:
        best_eval = final_state.get("best_evaluation", {}) if isinstance(final_state, dict) else {}
        accepted_solution_found = bool(best_eval.get("is_acceptable"))
        best_feasible_candidate_found = bool(best_eval.get("is_feasible"))
        iterations_completed = final_state.get("current_iteration") if isinstance(final_state, dict) else None
        best_solver_status = safe_get(final_state, "best_result", "solver_status")
        best_is_feasible = best_eval.get("is_feasible")
        best_is_acceptable = best_eval.get("is_acceptable")
        best_computed_objective = best_eval.get("computed_objective_value")
        latest_failure_type = safe_get(final_state, "repair_signal", "failure_type")
        latest_summary = safe_get(final_state, "repair_signal", "summary") or final_state.get("final_message")
        status_name = "ok_final_state_full"

    curation_reports = sorted(run_dir.glob("iteration_*_lesson_curation_report.json"))
    adaptive_added_count = 0
    adaptive_merged_count = 0
    adaptive_rejected_count = 0

    for rp in curation_reports:
        report = load_json(rp)
        adaptive_added_count += len(report.get("accepted_new_lessons", []) or [])
        adaptive_merged_count += len(report.get("merge_updates", []) or [])
        adaptive_rejected_count += len(report.get("rejected_candidates", []) or [])

    summaries = sorted(run_dir.glob("iteration_*_summary.json"))
    selected_lesson_ids = []
    for sp in summaries:
        summary_row = load_json(sp)
        selected_lesson_ids.extend(summary_row.get("selected_modeling_lesson_ids", []) or [])

    return {
        "problem": problem_name,
        "pass": pass_idx,
        "run_dir": str(run_dir),
        "status": status_name,
        "accepted_solution_found": accepted_solution_found,
        "best_feasible_candidate_found": best_feasible_candidate_found,
        "iterations_completed": iterations_completed,
        "best_solver_status": best_solver_status,
        "best_is_feasible": best_is_feasible,
        "best_is_acceptable": best_is_acceptable,
        "best_computed_objective": best_computed_objective,
        "latest_failure_type": latest_failure_type,
        "latest_summary": latest_summary,
        "adaptive_added_count": adaptive_added_count,
        "adaptive_merged_count": adaptive_merged_count,
        "adaptive_rejected_count": adaptive_rejected_count,
        "selected_lessons_total": len(selected_lesson_ids),
        "selected_lessons_unique": len(set(selected_lesson_ids)),
        "curation_reports": len(curation_reports),
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "problem",
        "pass",
        "run_dir",
        "status",
        "accepted_solution_found",
        "best_feasible_candidate_found",
        "iterations_completed",
        "best_solver_status",
        "best_is_feasible",
        "best_is_acceptable",
        "best_computed_objective",
        "latest_failure_type",
        "latest_summary",
        "adaptive_added_count",
        "adaptive_merged_count",
        "adaptive_rejected_count",
        "selected_lessons_total",
        "selected_lessons_unique",
        "curation_reports",
        "process_returncode",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_summary(rows: list[dict[str, Any]], path: Path) -> None:
    by_problem: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_problem.setdefault(row["problem"], []).append(row)

    solved = sum(1 for row in rows if row["accepted_solution_found"])
    iteration_values = [row["iterations_completed"] for row in rows if isinstance(row["iterations_completed"], int)]
    summary = {
        "runs_total": len(rows),
        "runs_solved": solved,
        "solve_rate": solved / len(rows) if rows else 0.0,
        "avg_iterations_completed": (sum(iteration_values) / len(iteration_values)) if iteration_values else None,
        "median_iterations_completed": statistics.median(iteration_values) if iteration_values else None,
        "problems": {},
    }

    for problem, entries in by_problem.items():
        summary["problems"][problem] = {
            "passes": len(entries),
            "solved_any_pass": any(e["accepted_solution_found"] for e in entries),
            "best_objective_seen": min(
                [e["best_computed_objective"] for e in entries if isinstance(e["best_computed_objective"], (int, float))],
                default=None,
            ),
            "adaptive_added_total": sum(int(e["adaptive_added_count"] or 0) for e in entries),
            "adaptive_merged_total": sum(int(e["adaptive_merged_count"] or 0) for e in entries),
            "adaptive_rejected_total": sum(int(e["adaptive_rejected_count"] or 0) for e in entries),
        }

    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def make_plots(rows: list[dict[str, Any]], out_dir: Path) -> list[Path]:
    out_paths = []

    labels = [f'{r["problem"]}\npass {r["pass"]}' for r in rows]
    vals = [r["iterations_completed"] if isinstance(r["iterations_completed"], int) else 0 for r in rows]
    plt.figure(figsize=(max(8, len(rows) * 0.8), 4.5))
    plt.bar(labels, vals)
    plt.ylabel("Iterations completed")
    plt.title("Iterations by benchmark run")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    p1 = out_dir / "iterations_by_run.png"
    plt.savefig(p1, dpi=160, bbox_inches="tight")
    plt.close()
    out_paths.append(p1)

    labels2 = [f'{r["problem"]}\npass {r["pass"]}' for r in rows]
    vals2 = [1 if r["accepted_solution_found"] else 0 for r in rows]
    plt.figure(figsize=(max(8, len(rows) * 0.8), 4.5))
    plt.bar(labels2, vals2)
    plt.ylabel("Solved (1=yes)")
    plt.title("Solved status by run")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    p2 = out_dir / "solved_status_by_run.png"
    plt.savefig(p2, dpi=160, bbox_inches="tight")
    plt.close()
    out_paths.append(p2)

    labels3 = [f'{r["problem"]}\npass {r["pass"]}' for r in rows]
    added = [int(r["adaptive_added_count"] or 0) for r in rows]
    merged = [int(r["adaptive_merged_count"] or 0) for r in rows]
    rejected = [int(r["adaptive_rejected_count"] or 0) for r in rows]
    x = list(range(len(rows)))
    plt.figure(figsize=(max(8, len(rows) * 0.8), 4.8))
    plt.bar(x, added, label="added")
    plt.bar(x, merged, bottom=added, label="merged")
    bottoms = [a + m for a, m in zip(added, merged)]
    plt.bar(x, rejected, bottom=bottoms, label="rejected")
    plt.xticks(x, labels3, rotation=45, ha="right")
    plt.ylabel("Count")
    plt.title("Adaptive lesson curation outcomes")
    plt.legend()
    plt.tight_layout()
    p3 = out_dir / "adaptive_curation_outcomes.png"
    plt.savefig(p3, dpi=160, bbox_inches="tight")
    plt.close()
    out_paths.append(p3)

    return out_paths


def main():
    parser = argparse.ArgumentParser(description="Run manufacturing_autoresearch benchmarks with retries and reporting.")
    parser.add_argument("--bench-dir", default="configs/benchmarks", help="Directory containing benchmark JSON files.")
    parser.add_argument("--runs-dir", default="runs/benchmark_experiments", help="Output directory for benchmark runs.")
    parser.add_argument("--max-iters", type=int, default=5, help="Max iterations per problem run.")
    parser.add_argument("--passes", type=int, default=2, help="Number of passes over failed problems.")
    parser.add_argument("--python", default=sys.executable, help="Python executable to use.")
    args = parser.parse_args()

    bench_dir = Path(args.bench_dir)
    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    problems = sorted(bench_dir.glob("*.json"))
    if not problems:
        raise SystemExit(f"No benchmark JSON files found in {bench_dir}")

    remaining = problems[:]
    all_rows: list[dict[str, Any]] = []

    for pass_idx in range(1, args.passes + 1):
        if not remaining:
            break

        next_remaining = []
        for problem_path in remaining:
            name = problem_path.stem
            run_dir = runs_dir / f"pass_{pass_idx}_{name}"
            run_dir.mkdir(parents=True, exist_ok=True)

            print(f"[pass {pass_idx}] running {name}")
            proc_info = run_problem(problem_path, run_dir, args.max_iters, args.python)

            (run_dir / "process_stdout.txt").write_text(proc_info["stdout"], encoding="utf-8")
            (run_dir / "process_stderr.txt").write_text(proc_info["stderr"], encoding="utf-8")
            (run_dir / "process_meta.json").write_text(
                json.dumps(
                    {
                        "returncode": proc_info["returncode"],
                        "cmd": proc_info["cmd"],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            row = parse_run(run_dir, name, pass_idx)
            row["process_returncode"] = proc_info["returncode"]
            all_rows.append(row)

            if not row["accepted_solution_found"]:
                next_remaining.append(problem_path)

        remaining = next_remaining

    csv_path = runs_dir / "benchmark_results.csv"
    summary_path = runs_dir / "benchmark_summary.json"
    write_csv(all_rows, csv_path)
    write_summary(all_rows, summary_path)
    plot_paths = make_plots(all_rows, runs_dir)

    markdown_lines = [
        "# Benchmark Experiment Report",
        "",
        f"- Benchmarks directory: `{bench_dir}`",
        f"- Runs directory: `{runs_dir}`",
        f"- Max iterations per run: `{args.max_iters}`",
        f"- Passes: `{args.passes}`",
        "",
        "## Result files",
        "",
        f"- CSV: `{csv_path.name}`",
        f"- Summary JSON: `{summary_path.name}`",
    ]
    for p in plot_paths:
        markdown_lines.append(f"- Plot: `{p.name}`")
    markdown_lines.extend([
        "",
        "## Runs",
        "",
        "| Problem | Pass | Solved | Iterations | Objective | Added | Merged | Rejected | Return code |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for r in all_rows:
        markdown_lines.append(
            f'| {r["problem"]} | {r["pass"]} | '
            f'{"✅" if r["accepted_solution_found"] else "❌"} | '
            f'{r["iterations_completed"] if r["iterations_completed"] is not None else "-"} | '
            f'{r["best_computed_objective"] if r["best_computed_objective"] is not None else "-"} | '
            f'{r["adaptive_added_count"]} | {r["adaptive_merged_count"]} | {r["adaptive_rejected_count"]} | '
            f'{r.get("process_returncode", "-")} |'
        )
    (runs_dir / "REPORT.md").write_text("\n".join(markdown_lines), encoding="utf-8")

    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")
    for p in plot_paths:
        print(f"Wrote {p}")
    print(f"Wrote {runs_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
