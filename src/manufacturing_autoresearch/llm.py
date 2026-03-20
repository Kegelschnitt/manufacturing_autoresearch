
from __future__ import annotations

import json
import os

from .types import EvaluationReport, ProblemDefinition, ProgramProposal, SolverResult


_SYSTEM_PROMPT = """You are improving a Python MILP generator for manufacturing scheduling.

Return ONLY valid raw Python code for exactly two functions:

def build_model(data):
    ...
    return prob, context

def extract_assignments(data, context):
    ...
    return assignments, notes

Rules:
- Return raw Python only. No markdown fences. No explanations.
- Do not access data["problem"] or data['problem'].
- The input data already has top-level keys: name, description, entities, parameters, evaluation_framework.
- Preserve existing working behavior unless the repair signal requires a change.
- If hard rules are violated, fix the MILP constraints.
- If objective terms are missing, fix the MILP objective so the optimized solution reflects those terms.
- Do not merely post-process assignments to fake compliance.
- The objective optimized by the MILP must match the objective_value returned by the solver harness.
- Prefer minimal targeted edits over full rewrites.
"""


def _jsonable(obj):
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return obj


class LLMClient:
    def __init__(self) -> None:
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self._client = None
        if self.api_key:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except Exception:
                self._client = None

    def _build_user_prompt(
        self,
        problem: ProblemDefinition,
        current_best: ProgramProposal,
        repair_signal: dict,
        latest_evaluation: EvaluationReport | None,
        latest_result: SolverResult | None,
    ) -> str:
        failure_type = repair_signal.get("failure_type", "")
        prompt = f"""You are given:

1. The problem definition
2. The current best MILP code
3. The latest repair signal
4. The latest evaluator report
5. The latest solver result

Your task:
- Produce an improved MILP model.
- Keep all currently satisfied hard rules satisfied.
- Fix violated hard rules if any.
- If the evaluator reports missing objective terms, incorporate those terms directly into the MILP objective.
- If the current model is feasible but not acceptable, make the smallest structural change needed to address the evaluator's complaint.

Problem definition:
{json.dumps(_jsonable(problem), indent=2)}

Current best code:
{current_best.model_logic}

Latest evaluator report:
{json.dumps(_jsonable(latest_evaluation), indent=2)}

Latest solver result:
{json.dumps(_jsonable(latest_result), indent=2)}

Latest repair signal:
{json.dumps(repair_signal, indent=2)}

Required behavior:
- If failure_type == "rule_violation", add or strengthen constraints.
- If failure_type == "objective_mismatch", add the missing objective terms to the MILP objective.
- If failure_type == "no_improvement", preserve feasibility and improve only the weakest objective component.
- If failure_type == "runtime_error" or "preflight_error", fix syntax/runtime issues first before changing model structure.

Important:
- The evaluation framework is the source of truth.
- The returned objective_value must correspond to the optimized MILP objective.
- Keep the code executable in the existing harness.
- Return only Python code.
"""
        if failure_type == "objective_mismatch":
            prompt += """
The current MILP is feasible but incomplete.

The evaluator found that the optimized objective used by the MILP does not match the true evaluation objective.

You must modify the MILP objective itself, not only the output formatting.

For each missing objective term:
- identify what additional variables or linearization are needed
- add those variables if necessary
- add linking constraints if necessary
- include the term in the minimized objective

Example:
- a changeover penalty may require auxiliary binary variables for machine transitions between consecutive slots

Do not remove already-working feasibility constraints unless necessary.
"""
        if failure_type == "rule_violation":
            prompt += """
The evaluator found hard-rule violations.

You must change the MILP so the solver cannot produce assignments that violate these rules.

For each violated rule:
- identify the structural reason the current model allows the violation
- add or strengthen constraints to prevent it
- preserve currently satisfied rules

Do not rely on extract_assignments to filter away invalid assignments after solving.
"""
        return prompt

    def propose(
        self,
        problem: ProblemDefinition,
        current_best: ProgramProposal,
        repair_signal: dict,
        latest_evaluation: EvaluationReport | None = None,
        latest_result: SolverResult | None = None,
    ) -> ProgramProposal:
        if self._client is None:
            return current_best

        user_prompt = self._build_user_prompt(
            problem=problem,
            current_best=current_best,
            repair_signal=repair_signal,
            latest_evaluation=latest_evaluation,
            latest_result=latest_result,
        )

        try:
            response = self._client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            text = (getattr(response, "output_text", "") or "").strip()
            if not text:
                return current_best
            return ProgramProposal(
                explanation="LLM-proposed targeted MILP refinement based on repair signal.",
                model_logic=text,
                expected_failure_modes=current_best.expected_failure_modes,
            )
        except Exception:
            return current_best
