from __future__ import annotations

import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from .types import ProblemDefinition, ProgramProposal


load_dotenv()


REASONING_SYSTEM_PROMPT = """You are a MILP modeling assistant for manufacturing scheduling.

Your job is NOT to write code yet.
Your first job is to analyze the current MILP and describe the minimum structural MILP changes needed.

Return ONLY valid JSON with this schema:
{
  "diagnosis": "string",
  "keep_unchanged": ["string"],
  "structural_changes": ["string"],
  "new_variables": ["string"],
  "new_constraints": ["string"],
  "objective_update": "string",
  "anti_patterns": ["string"]
}

Rules:
- Be concrete and mathematically actionable.
- Use the proposer guidance as the main semantic source of truth.
- Do not assume the meaning of rules or objective terms only from their names if source-level guidance is available.
- Read the provided rule/objective definitions and source code snippets before deciding how to modify the MILP.
- If an objective term is missing, explain what auxiliary variables and linking constraints are needed.
- If hard rules are already satisfied, explicitly say they should be preserved.
- Preserve sparse indexing safety: if a variable dictionary only contains eligible keys, never assume all (job, machine, slot) tuples exist.
- When describing changes, explicitly mention whether constraints must guard with checks like `(job, machine, slot) in x`.
- Do not return code.
- Do not return markdown.
"""


CODEGEN_SYSTEM_PROMPT = """You are improving a Python MILP generator for manufacturing scheduling.

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
- Preserve existing working feasibility constraints unless the reasoning explicitly says to change them.
- If hard rules are violated, fix the MILP constraints.
- If objective terms are missing, fix the MILP objective itself so the optimized solution reflects those terms.
- Introduce auxiliary variables and linking constraints when required by the reasoning plan.
- Do not merely post-process assignments to fake compliance.
- The solver_result objective_value must reflect the actual optimized objective used by the MILP.
- Prefer minimal targeted edits over full rewrites.
- If the previous candidate made no structural change, you must materially change the objective or constraints.
- A candidate that only adds comments, renames variables, or reformats code is invalid.

Critical modeling rules:
- Respect sparse indexing: if x only contains eligible decision variables, never reference x[(job, machine, slot)] unless that key exists.
- When summing over x, either iterate over existing keys or guard with `if (job, machine, slot) in x`.
- Do not invent dense variable access for ineligible machine-job pairs.
- If modeling transition/changeover logic, ensure the formulation matches the evaluator semantics rather than guessing from the term name alone.
- Use the provided proposer guidance and source definitions to infer the intended math.
"""


class LLMClient:
    def __init__(self) -> None:
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None

    def _fallback(self, current_best: ProgramProposal) -> ProgramProposal:
        return current_best

    def _build_reasoning_prompt(
        self,
        problem: ProblemDefinition,
        current_best: ProgramProposal,
        repair_signal: dict,
        proposer_guidance: dict,
        run_memory: dict | None = None,
    ) -> dict:
        return {
            "task": "Analyze the current MILP and describe the minimum structural change needed.",
            "problem_definition": problem.model_dump(),
            "current_best_code": current_best.model_logic,
            "repair_signal": repair_signal,
            "proposer_guidance": proposer_guidance,
            "run_memory": run_memory or {},
            "required_behavior": [
                "Explain what is missing or wrong in the current MILP.",
                "State what must remain unchanged if it already works.",
                "Use proposer_guidance.problem_active_rule_ids and proposer_guidance.problem_active_objective_id to identify what is active for this problem.",
                "Use proposer_guidance.available_rule_definitions and proposer_guidance.available_objective_term_definitions as the first semantic reference for modeling.",
                "If those structured definitions are incomplete, inspect proposer_guidance.rules_source_code and proposer_guidance.objective_terms_source_code.",
                "Do not assume a rule or objective term meaning only from its name if code-level guidance is available.",
                "If objective terms are missing, describe what auxiliary variables and linking constraints are needed.",
                "If the loop is stuck in no_structural_change, propose an actual structural MILP modification.",
                "Respect sparse indexing and explicitly mention eligibility-safe indexing when relevant.",
            ],
            "special_attention_checks": [
                "Does the current MILP reference only valid keys in x?",
                "Do proposed new constraints require guards like `(job, machine, slot) in x`?",
                "Does the intended formulation actually match the evaluator objective semantics?",
                "Are currently satisfied hard constraints preserved?",
            ],
        }

    def reason_about_fix(
        self,
        problem: ProblemDefinition,
        current_best: ProgramProposal,
        repair_signal: dict,
        proposer_guidance: dict,
        run_memory: dict | None = None,
    ) -> dict:
        if not self.client:
            return {
                "diagnosis": "No client available.",
                "keep_unchanged": [],
                "structural_changes": [],
                "new_variables": [],
                "new_constraints": [],
                "objective_update": "",
                "anti_patterns": [],
            }

        user_prompt = self._build_reasoning_prompt(
            problem=problem,
            current_best=current_best,
            repair_signal=repair_signal,
            proposer_guidance=proposer_guidance,
            run_memory=run_memory,
        )

        try:
            response = self.client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": REASONING_SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)},
                ],
            )
            text = (response.output_text or "").strip()
            parsed = json.loads(text)
            if not isinstance(parsed, dict):
                raise ValueError("Reasoning response was not a dict.")
            return parsed
        except Exception:
            return {
                "diagnosis": "Reasoning step failed.",
                "keep_unchanged": [],
                "structural_changes": [],
                "new_variables": [],
                "new_constraints": [],
                "objective_update": "",
                "anti_patterns": [],
            }

    def _build_codegen_prompt(
        self,
        problem: ProblemDefinition,
        current_best: ProgramProposal,
        repair_signal: dict,
        proposer_guidance: dict,
        reasoning_plan: dict,
        run_memory: dict | None = None,
    ) -> dict:
        return {
            "task": "Produce an improved MILP model from the reasoning plan.",
            "problem_definition": problem.model_dump(),
            "current_best_code": current_best.model_logic,
            "repair_signal": repair_signal,
            "proposer_guidance": proposer_guidance,
            "run_memory": run_memory or {},
            "reasoning_plan": reasoning_plan,
            "requirements": [
                "Keep all currently satisfied hard rules satisfied unless reasoning explicitly requires change.",
                "Fix violated hard rules by changing constraints.",
                "If the evaluator reports missing objective terms, incorporate those terms directly into the MILP objective.",
                "If the current model is feasible but not acceptable, make the smallest structural change needed to address the evaluator complaint.",
                "Use proposer_guidance.problem_active_rule_ids and proposer_guidance.problem_active_objective_id to determine what is active in this problem.",
                "Use proposer_guidance.available_rule_definitions and proposer_guidance.available_objective_term_definitions as semantic guidance.",
                "If those are incomplete, consult proposer_guidance.rules_source_code and proposer_guidance.objective_terms_source_code before deciding the formulation.",
                "If the reasoning_plan specifies new variables or new constraints, implement them in the code.",
                "Do not return code equivalent to the current_best_code.",
                "Preserve sparse eligibility-safe indexing.",
                "Never reference x[(job, machine, slot)] unless that key exists in x.",
            ],
            "special_guidance": {
                "objective_mismatch": [
                    "Modify the MILP objective itself, not only the output formatting.",
                    "For each missing objective term, identify what additional variables or linearization are needed.",
                    "If source-level rule/objective guidance is available, use that to determine the correct semantics.",
                    "Do not guess a mathematical formulation if the supplied guidance/source text already implies a specific one.",
                ],
                "no_structural_change": [
                    "You must change the MILP structure.",
                    "Add new variables, constraints, or objective terms as required by reasoning_plan.",
                    "Do not return comments-only or formatting-only changes.",
                ],
                "runtime_error": [
                    "Repair runtime errors before optimizing quality.",
                    "If a previous candidate failed due to invalid dictionary access, rewrite the new formulation to use eligibility-safe indexing.",
                    "Guard sparse variable access with membership tests or iterate only over existing keys.",
                ],
            },
        }

    def generate_code_from_reasoning(
        self,
        problem: ProblemDefinition,
        current_best: ProgramProposal,
        repair_signal: dict,
        proposer_guidance: dict,
        reasoning_plan: dict,
        run_memory: dict | None = None,
    ) -> ProgramProposal:
        if not self.client:
            return self._fallback(current_best)

        user_prompt = self._build_codegen_prompt(
            problem=problem,
            current_best=current_best,
            repair_signal=repair_signal,
            proposer_guidance=proposer_guidance,
            reasoning_plan=reasoning_plan,
            run_memory=run_memory,
        )

        try:
            response = self.client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": CODEGEN_SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)},
                ],
            )
            text = (response.output_text or "").strip()
            if not text:
                return self._fallback(current_best)

            return ProgramProposal(
                explanation="LLM-improved MILP candidate generated from reasoning plan",
                model_logic=text,
                expected_failure_modes=[],
            )
        except Exception:
            return self._fallback(current_best)

    def propose(
        self,
        problem: ProblemDefinition,
        current_best: ProgramProposal,
        repair_signal: dict,
        proposer_guidance: dict,
        run_memory: dict | None = None,
    ) -> tuple[ProgramProposal, dict]:
        reasoning_plan = self.reason_about_fix(
            problem=problem,
            current_best=current_best,
            repair_signal=repair_signal,
            proposer_guidance=proposer_guidance,
            run_memory=run_memory,
        )

        candidate = self.generate_code_from_reasoning(
            problem=problem,
            current_best=current_best,
            repair_signal=repair_signal,
            proposer_guidance=proposer_guidance,
            reasoning_plan=reasoning_plan,
            run_memory=run_memory,
        )

        return candidate, reasoning_plan