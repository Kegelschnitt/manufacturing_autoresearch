from __future__ import annotations

from .types import ProgramProposal


def baseline_program() -> ProgramProposal:
    code = '''def build_model(data):
    jobs = data["entities"]["jobs"]
    machines = data["entities"]["machines"]
    slots = data["entities"]["slots"]
    eligible = data["parameters"]["eligible"]
    cost = data["parameters"]["cost"]

    prob = pulp.LpProblem(data["name"], pulp.LpMinimize)
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

    prob += pulp.lpSum(
        float(cost[f"{job}|{machine}|{slot}"]) * x[(job, machine, slot)]
        for (job, machine, slot) in x
    ), "total_cost"

    for job in jobs:
        prob += pulp.lpSum(
            x[(job, machine, slot)]
            for machine in eligible.get(job, [])
            for slot in slots
        ) == 1, f"job_once_{job}"

    for machine in machines:
        for slot in slots:
            prob += pulp.lpSum(
                x[(job, machine, slot)]
                for job in jobs
                if (job, machine, slot) in x
            ) <= 1, f"capacity_{machine}_{slot}"

    return prob, {"x": x}


def extract_assignments(data, context):
    assignments = []
    for (job, machine, slot), var in context["x"].items():
        if var.varValue is not None and float(var.varValue) > 0.5:
            assignments.append({"job": job, "machine": machine, "slot": int(slot)})
    notes = []
    return assignments, notes
'''
    return ProgramProposal(
        explanation="Known-good baseline assignment MILP. Use this as the fallback anchor and improve it incrementally.",
        model_logic=code,
        expected_failure_modes=["suboptimal_objective", "missing_complex_constraints"],
    )
