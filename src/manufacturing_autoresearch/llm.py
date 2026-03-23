from __future__ import annotations

import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from .types import ProblemDefinition, ProgramProposal


load_dotenv()


LESSON_SELECTOR_SYSTEM_PROMPT = """You are selecting modeling lessons for a MILP repair loop.

Return ONLY valid JSON with this schema:
{
  "selected_lessons": [
    {
      "lesson_id": "string",
      "priority": 1,
      "reason": "string"
    }
  ],
  "rejected_lessons": [
    {
      "lesson_id": "string",
      "reason": "string"
    }
  ]
}

Rules:
- Choose at most 4 lessons.
- Prefer lessons that match the active objective, failure type, violations, warnings, and traceback.
- Reject irrelevant lessons.
- Prefer specific lessons over generic ones when possible.
"""


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
- Use proposer guidance and selected modeling lessons as the main semantic source of truth.
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
- Use selected modeling lessons as reusable modeling patterns, but adapt them to the current problem instead of copying blindly.
"""


class LLMClient:
    def __init__(self) -> None:
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None

    def _fallback(self, current_best: ProgramProposal) -> ProgramProposal:
        return current_best

    def reason_about_fix(
        self,
        problem: ProblemDefinition,
        current_best: ProgramProposal,
        repair_signal: dict,
        proposer_guidance: dict,
        run_memory: dict | None = None,
        selected_modeling_lessons: list[dict] | None = None,
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

        user_prompt = {
            "task": "Analyze the current MILP and describe the minimum structural change needed.",
            "problem_definition": problem.model_dump(),
            "current_best_code": current_best.model_logic,
            "repair_signal": repair_signal,
            "proposer_guidance": proposer_guidance,
            "run_memory": run_memory or {},
            "selected_modeling_lessons": selected_modeling_lessons or [],
            "required_behavior": [
                "Explain what is missing or wrong in the current MILP.",
                "State what must remain unchanged if it already works.",
                "If objective terms are missing, describe what auxiliary variables and linking constraints are needed.",
                "If the loop is stuck in no_structural_change, propose an actual structural MILP modification.",
            ],
        }

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
        except Exception as exc:
            return {
                "diagnosis": f"Reasoning step failed: {type(exc).__name__}",
                "keep_unchanged": [],
                "structural_changes": [],
                "new_variables": [],
                "new_constraints": [],
                "objective_update": "",
                "anti_patterns": [],
            }

    def generate_code_from_reasoning(
        self,
        problem: ProblemDefinition,
        current_best: ProgramProposal,
        repair_signal: dict,
        proposer_guidance: dict,
        reasoning_plan: dict,
        run_memory: dict | None = None,
        selected_modeling_lessons: list[dict] | None = None,
    ) -> ProgramProposal:
        if not self.client:
            return self._fallback(current_best)

        user_prompt = {
            "task": "Produce an improved MILP model from the reasoning plan.",
            "problem_definition": problem.model_dump(),
            "current_best_code": current_best.model_logic,
            "repair_signal": repair_signal,
            "proposer_guidance": proposer_guidance,
            "run_memory": run_memory or {},
            "selected_modeling_lessons": selected_modeling_lessons or [],
            "reasoning_plan": reasoning_plan,
            "requirements": [
                "Keep all currently satisfied hard rules satisfied unless reasoning explicitly requires change.",
                "Fix violated hard rules by changing constraints.",
                "If the evaluator reports missing objective terms, incorporate those terms directly into the MILP objective.",
                "If the current model is feasible but not acceptable, make the smallest structural change needed to address the evaluator complaint.",
                "Use proposer_guidance.active_rules_guidance and proposer_guidance.active_objective_guidance as the semantic specification of what the MILP must satisfy and optimize.",
                "Use selected modeling lessons as relevant patterns to avoid repeated mistakes.",
                "If the reasoning_plan specifies new variables or new constraints, implement them in the code.",
                "Do not return code equivalent to the current_best_code.",
            ],
            "special_guidance": {
                "objective_mismatch": [
                    "Modify the MILP objective itself, not only the output formatting.",
                    "For each missing objective term, identify what additional variables or linearization are needed.",
                ],
                "no_structural_change": [
                    "You must change the MILP structure.",
                    "Add new variables, constraints, or objective terms as required by reasoning_plan.",
                    "Do not return comments-only or formatting-only changes.",
                ],
            },
        }

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
                explanation="LLM-improved MILP candidate generated from reasoning plan and modeling lessons",
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
        selected_modeling_lessons: list[dict] | None = None,
    ) -> tuple[ProgramProposal, dict]:
        reasoning_plan = self.reason_about_fix(
            problem=problem,
            current_best=current_best,
            repair_signal=repair_signal,
            proposer_guidance=proposer_guidance,
            run_memory=run_memory,
            selected_modeling_lessons=selected_modeling_lessons,
        )

        candidate = self.generate_code_from_reasoning(
            problem=problem,
            current_best=current_best,
            repair_signal=repair_signal,
            proposer_guidance=proposer_guidance,
            reasoning_plan=reasoning_plan,
            run_memory=run_memory,
            selected_modeling_lessons=selected_modeling_lessons,
        )

        return candidate, reasoning_plan
    
    def select_lessons(
        self,
        problem,
        repair_signal,
        proposer_guidance,
        available_lessons,
        run_memory=None,
    ):
        if not self.client:
            return []

        user_prompt = {
            "problem_definition": problem.model_dump(),
            "repair_signal": repair_signal,
            "proposer_guidance": proposer_guidance,
            "available_lessons": available_lessons,
            "run_memory": run_memory or {},
        }

        try:
            response = self.client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": LESSON_SELECTOR_SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)},
                ],
            )
            text = (response.output_text or "").strip()
            parsed = json.loads(text)
            selected = parsed.get("selected_lessons", [])
            selected_ids = [item["lesson_id"] for item in selected if "lesson_id" in item]

            lesson_map = {lesson["lesson_id"]: lesson for lesson in available_lessons}
            resolved = []
            for item in selected:
                lid = item.get("lesson_id")
                if lid in lesson_map:
                    lesson = dict(lesson_map[lid])
                    lesson["llm_priority"] = item.get("priority")
                    lesson["llm_reason"] = item.get("reason", "")
                    resolved.append(lesson)
            return resolved
        except Exception:
            return []
