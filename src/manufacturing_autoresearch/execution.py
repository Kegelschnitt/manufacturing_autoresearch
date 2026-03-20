from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

import pulp

from .types import Assignment, ProblemDefinition, ProgramProposal, SolverResult


def execute_program(problem: ProblemDefinition, program: ProgramProposal) -> SolverResult:
    ns: dict[str, Any] = {"pulp": pulp}
    try:
        exec(program.model_logic, ns, ns)
        build_model = ns["build_model"]
        extract_assignments = ns["extract_assignments"]
        prob, context = build_model(problem.model_dump())
        status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
        assignments_raw, notes = extract_assignments(problem.model_dump(), context)
        assignments = [Assignment.model_validate(a) for a in assignments_raw]
        status_str = pulp.LpStatus.get(status, str(status)).lower()
        if status_str == "optimal":
            obj = float(pulp.value(prob.objective)) if prob.objective is not None else None
        else:
            obj = None
        return SolverResult(
            solver_status=status_str,
            objective_value=obj,
            assignments=assignments,
            notes=list(notes or []),
        )
    except Exception:
        return SolverResult(
            solver_status="runtime_error",
            objective_value=None,
            assignments=[],
            notes=[traceback.format_exc()],
        )
