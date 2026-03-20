from __future__ import annotations

from .types import ProgramProposal


def baseline_program() -> ProgramProposal:
    code = '''def build_model(data):
    import pulp

    # Top-level problem data
    jobs = data["entities"]["jobs"]
    machines = data["entities"]["machines"]
    slots = data["entities"]["slots"]
    eligible = data["parameters"]["eligible"]
    cost = data["parameters"]["cost"]

    # Create the optimization problem.
    # Baseline objective: minimize assignment cost only.
    #
    # IMPORTANT:
    # This baseline does NOT yet include richer manufacturing objective terms
    # such as changeover/setup penalties, tardiness, labor usage, etc.
    # Future improvements should EXTEND the objective, not replace the
    # working assignment logic below.
    prob = pulp.LpProblem(data["name"], pulp.LpMinimize)

    # Core assignment decision variables:
    # x[(job, machine, slot)] = 1 if job is assigned to machine at slot, else 0.
    #
    # We only create variables for eligible machine choices.
    x = {}

    for job in jobs:
        for machine in eligible.get(job, []):
            for slot in slots:
                x[(job, machine, slot)] = pulp.LpVariable(
                    f"assign_{job}_{machine}_{slot}",
                    lowBound=0,
                    upBound=1,
                    cat="Binary",
                )

    # ------------------------------------------------------------------
    # OBJECTIVE: assignment cost only
    # ------------------------------------------------------------------
    # This is the currently working baseline objective.
    # The evaluator may later require additional terms, e.g.:
    #   - changeover penalties
    #   - setup penalties
    #   - tardiness penalties
    #   - overtime penalties
    #
    # If such terms are required, they must be added HERE in the MILP
    # objective with proper auxiliary variables and linking constraints.
    prob += pulp.lpSum(
        float(cost[f"{job}|{machine}|{slot}"]) * x[(job, machine, slot)]
        for (job, machine, slot) in x
    ), "total_assignment_cost"

    # ------------------------------------------------------------------
    # HARD RULE 1: each job must be assigned exactly once
    # ------------------------------------------------------------------
    # A valid solution must place every job on exactly one eligible
    # machine-slot pair.
    for job in jobs:
        prob += pulp.lpSum(
            x[(job, machine, slot)]
            for machine in eligible.get(job, [])
            for slot in slots
        ) == 1, f"job_once_{job}"

    # ------------------------------------------------------------------
    # HARD RULE 2: machine-slot capacity
    # ------------------------------------------------------------------
    # A machine can process at most one job in the same slot.
    for machine in machines:
        for slot in slots:
            prob += pulp.lpSum(
                x[(job, machine, slot)]
                for job in jobs
                if (job, machine, slot) in x
            ) <= 1, f"capacity_{machine}_{slot}"

    # ------------------------------------------------------------------
    # EXTENSION POINT: richer manufacturing logic
    # ------------------------------------------------------------------
    # If the evaluator reports missing manufacturing terms or violated
    # rules, extend the MILP HERE by introducing additional variables and
    # constraints.
    #
    # Example: changeover penalties between consecutive slots
    # ------------------------------------------------------
    # If the true objective penalizes switching jobs on the same machine
    # between slot t and t+1, then future versions should:
    #
    # 1. Introduce auxiliary variables describing whether a machine has
    #    a changeover between consecutive slots.
    # 2. Add linking constraints connecting those variables to x.
    # 3. Add the penalty term to the objective.
    #
    # IMPORTANT:
    # Do not try to "fix" missing objective terms in extract_assignments.
    # The MILP itself must optimize the correct objective.

    context = {
        "x": x,
    }

    return prob, context


def extract_assignments(data, context):
    # This function only reads the solved MILP result.
    #
    # IMPORTANT:
    # - Do NOT enforce constraints here.
    # - Do NOT filter out "bad" assignments to fake compliance.
    # - All required logic must already be encoded in build_model().
    assignments = []

    for (job, machine, slot), var in context["x"].items():
        if var.varValue is not None and float(var.varValue) > 0.5:
            assignments.append({
                "job": job,
                "machine": machine,
                "slot": int(slot),
            })

    # Notes are optional diagnostics only.
    notes = [
        "Baseline MILP minimizes assignment cost only.",
        "Additional manufacturing objective terms must be modeled in build_model().",
    ]

    return assignments, notes
'''
    return ProgramProposal(
        explanation="Known-good baseline MILP for assignment feasibility and assignment-cost minimization. This baseline intentionally omits richer manufacturing objective terms such as changeover penalties. Extend the marked extension points when evaluator feedback requires additional terms or constraints.",
        model_logic=code,
        expected_failure_modes=["suboptimal_objective", "missing_complex_constraints"],
    )
