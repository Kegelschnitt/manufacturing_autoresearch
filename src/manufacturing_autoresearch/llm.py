from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from .types import ProblemDefinition, ProgramProposal

load_dotenv()

SYSTEM_PROMPT = """You are improving a Python MILP generator for manufacturing scheduling.

Return ONLY raw valid Python code implementing exactly these functions:

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
- Preserve existing working feasibility constraints unless the repair signal requires a targeted structural change.
- If hard rules are violated, fix the MILP constraints.
- If objective terms are missing, fix the MILP objective itself so the optimized solution reflects those terms.
- Do not merely post-process assignments to fake compliance.
- The solver_result objective_value must reflect the actual optimized objective used by the MILP.
- Prefer minimal targeted edits over full rewrites.
- If the previous candidate made no structural change, you must materially change the objective or constraints.
- Use proposer_guidance and run_memory to avoid repeating previous mistakes.
"""


class LLMClient:
    def __init__(self) -> None:
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None

    def _fallback(self, current_best: ProgramProposal) -> ProgramProposal:
        return current_best

    def propose(
        self,
        problem: ProblemDefinition,
        current_best: ProgramProposal,
        repair_signal: dict,
        proposer_guidance: dict | None = None,
        run_memory: dict | None = None,
    ) -> ProgramProposal:
        if not self.client:
            return self._fallback(current_best)

        user_prompt: dict[str, Any] = {
            "task": "Produce an improved MILP model.",
            "requirements": [
                "Keep all currently satisfied hard rules satisfied.",
                "Fix violated hard rules by changing constraints.",
                "If the evaluator reports missing objective terms, incorporate those terms directly into the MILP objective.",
                "If the current model is feasible but not acceptable, make the smallest structural change needed to address the evaluator complaint.",
                "A candidate that only reformats or restates the current MILP is not acceptable.",
                "Do not produce code identical to the current best code.",
            ],
            "problem_definition": problem.model_dump(),
            "current_best_code": current_best.model_logic,
            "repair_signal": repair_signal,
            "proposer_guidance": proposer_guidance or {},
            "run_memory": run_memory or {},
            "required_behavior_by_failure_type": {
                "rule_violation": "Add or strengthen MILP constraints so the violation becomes impossible.",
                "objective_mismatch": "Add the missing objective terms to the MILP objective and introduce auxiliary variables/constraints if needed.",
                "no_structural_change": "Make a real structural MILP change; comments or formatting-only edits are not enough.",
                "no_improvement": "Preserve feasibility and improve the weakest objective component.",
                "runtime_error": "Fix syntax/runtime issues first.",
                "preflight_error": "Fix preflight issues first.",
            },
            "special_guidance": {
                "objective_mismatch": [
                    "Modify the MILP objective itself, not only the output formatting.",
                    "For each missing objective term, identify what additional variables or linearization are needed.",
                    "A changeover penalty may require auxiliary binary variables for machine transitions between consecutive slots.",
                ],
                "rule_violation": [
                    "Do not rely on extract_assignments to filter away invalid assignments after solving.",
                    "The MILP itself must enforce all hard rules.",
                ],
            },
        }

        try:
            response = self.client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)},
                ],
            )
            text = (response.output_text or "").strip()
            if not text:
                return self._fallback(current_best)
            return ProgramProposal(
                explanation="LLM-improved MILP candidate",
                model_logic=text,
                expected_failure_modes=[],
            )
        except Exception:
            return self._fallback(current_best)
