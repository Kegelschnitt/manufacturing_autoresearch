from __future__ import annotations

import json
import os

from dotenv import load_dotenv
from openai import OpenAI
from rich import text

from .types import ProblemDefinition, ProgramProposal


load_dotenv()

LESSON_CURATOR_SYSTEM_PROMPT = """You are extracting reusable adaptive modeling lessons from a successful MILP repair.

Return ONLY a JSON object with this shape:
{
  "new_lessons": [
    {
      "lesson_id": "string",
      "title": "string",
      "applies_when": {
        "failure_type": "string or null",
        "missing_objective_terms": ["string"],
        "traceback_contains": ["string"],
        "violated_rules": ["string"],
        "objective_ids": ["string"],
        "always": false
      },
      "lesson": "string",
      "recommended_actions": ["string"],
      "anti_patterns": ["string"],
      "priority_hint": 5,
      "tags": ["string"]
    }
  ]
}

If no reusable lesson was learned, return:
{"new_lessons": []}

Follow the structure strictly.
- Do not include markdown fences.
- Do not include explanations outside the JSON.

Rules:
- Propose at most 2 lessons.
- Only propose lessons if the successful repair revealed a reusable modeling pattern.
- Do not restate an existing core lesson.
- Do not create benchmark-specific lessons.
- Do not mention file paths, run directories, or exact benchmark names.
- Use structured applies_when fields such as failure_type, missing_objective_terms, violated_rules, traceback_contains, or objective_ids.
- Prefer narrow reusable lessons over generic advice.
- If no reusable lesson was learned, return {"new_lessons": []}.
- Do not restate an existing lesson with only minor wording changes.
- If an existing lesson already captures the same modeling pattern, return no new lesson for that pattern.
- Prefer returning {"new_lessons": []} over proposing a near-duplicate lesson.
- Only propose a new lesson when the pattern is clearly distinct from existing lessons.
- Avoid generic “best practice” lessons unless the repair revealed a specific reusable MILP modeling pattern.
"""

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
- Use current_best_code to detect concrete modeling mistakes when relevant.
- If the code appears to overwrite the objective, prioritize lessons about setting the objective exactly once.
- If the code appears to reference sparse variables unsafely, prioritize sparse-indexing lessons.
- Each lesson may include historical_success_rate, historical_score, historical_times_selected, and historical_times_failed.
- Use historical_score and historical_success_rate only as secondary signals.
- Do not prefer a historically strong but irrelevant lesson over a clearly relevant lesson.
- If two lessons are similarly relevant, prefer the one with stronger historical evidence.
- Do not unfairly reject unseen lessons with little or no history if they are strongly relevant.
- Prefer the smallest lesson set that plausibly addresses the current failure; do not include extra lessons unless they have clear evidence from the repair signal or code.
- Avoid repeatedly selecting the same bundle of lessons unless each lesson is independently justified by the current repair signal.
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
    
    def _json_chat(self, system_prompt, user_prompt, schema=None, temperature=0.2):
        if self.client is None:
            raise ValueError("OPENAI_API_KEY is not set")

        request_kwargs = {
            "model": self.model,
            "temperature": temperature,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)},
            ],
        }

        if schema is not None:
            request_kwargs["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "structured_response",
                    "strict": True,
                    "schema": schema,
                }
            }

        response = self.client.responses.create(**request_kwargs)

        text = getattr(response, "output_text", None)

        print("[debug] _json_chat raw output_text repr:", repr(text))

        if not text or not str(text).strip():
            print("[debug] _json_chat full response object:", response)
            raise ValueError("Empty response from model")

        cleaned = str(text).strip()

        if cleaned.startswith("```json"):
            cleaned = cleaned[len("```json"):].strip()
        elif cleaned.startswith("```"):
            cleaned = cleaned[len("```"):].strip()

        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()

        print("[debug] _json_chat cleaned text repr:", repr(cleaned))

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            print("[debug] _json_chat failed to parse JSON")
            print("[debug] _json_chat cleaned text:", cleaned)
            raise
    
    def _attach_lesson_history(self, available_lessons, lesson_stats=None):
        enriched = []

        for lesson in available_lessons or []:
            if not isinstance(lesson, dict):
                continue

            item = dict(lesson)
            lesson_id = item.get("lesson_id")

            if lesson_stats is not None and lesson_id:
                stats = lesson_stats.get_lesson_stats(lesson_id)
            else:
                stats = {
                    "times_selected": 0,
                    "times_successful": 0,
                    "times_failed": 0,
                    "success_rate": 0.0,
                    "score": 0.0,
                }

            item["historical_times_selected"] = int(stats.get("times_selected", 0) or 0)
            item["historical_times_successful"] = int(stats.get("times_successful", 0) or 0)
            item["historical_times_failed"] = int(stats.get("times_failed", 0) or 0)
            item["historical_success_rate"] = float(stats.get("success_rate", 0.0) or 0.0)
            item["historical_score"] = float(stats.get("score", 0.0) or 0.0)

            enriched.append(item)

        return enriched

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
        current_best_code=None,
        run_memory=None,
        lesson_stats=None,
    ):
        if not self.client:
            print("[debug] LLM lesson selector: no client available")
            return []

        enriched_lessons = self._attach_lesson_history(
            available_lessons=available_lessons,
            lesson_stats=lesson_stats,
        )

        user_prompt = {
            "problem_definition": problem.model_dump(),
            "repair_signal": repair_signal,
            "proposer_guidance": proposer_guidance,
            "available_lessons": enriched_lessons,
            "current_best_code": current_best_code or "",
            "run_memory": run_memory or {},
            "selection_guidance": {
                "prefer_relevance_first": True,
                "use_history_as_secondary_signal": True,
                "allow_exploration_for_unseen_but_relevant_lessons": True,
            },
        }

        try:
            response = self.client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": LESSON_SELECTOR_SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)},
                ],
            )

            # --- CLEAN MARKDOWN FENCES ---
            def _clean_json(text: str) -> str:
                text = text.strip()
                if text.startswith("```"):
                    parts = text.split("```")
                    if len(parts) >= 2:
                        text = parts[1]
                    if text.startswith("json"):
                        text = text[len("json"):].strip()
                return text
            
            text = _clean_json(response.output_text or "")
            parsed = json.loads(text)
            selected = parsed.get("selected_lessons", [])

            lesson_map = {
                lesson.get("lesson_id"): lesson
                for lesson in available_lessons
                if isinstance(lesson, dict) and lesson.get("lesson_id")
            }

            resolved = []
            for item in selected:
                lid = item.get("lesson_id")
                if lid in lesson_map:
                    lesson = dict(lesson_map[lid])
                    lesson["llm_priority"] = item.get("priority")
                    lesson["llm_reason"] = item.get("reason", "")
                    resolved.append(lesson)

            # print("[debug] LLM lesson selector resolved ids:", [x.get("lesson_id") for x in resolved])
            # print("[debug] LLM lesson selector resolved reasons:", [x.get("llm_reason") for x in resolved])
            return resolved
        except Exception as exc:
            print(f"[debug] LLM lesson selector failed: {type(exc).__name__}: {exc}")
            return []
        
    def curate_lessons(
        self,
        problem,
        selected_modeling_lessons,
        repair_signal_before,
        reasoning_plan,
        before_code,
        after_code,
        before_evaluation,
        after_evaluation,
        existing_lessons=None,
        run_summary=None,
    ):
        """
        Extract reusable adaptive modeling lessons from a successful accepted repair.

        Returns:
            list[dict]: normalized candidate lesson dicts, or [] if none were proposed / parsing failed.
        """
        if self.client is None:
            return []

        def _safe_model_dump(obj):
            if obj is None:
                return None
            if hasattr(obj, "model_dump"):
                try:
                    return obj.model_dump()
                except Exception:
                    pass
            if isinstance(obj, dict):
                return obj
            return {"value": str(obj)}

        def _truncate(text, limit=12000):
            text = "" if text is None else str(text)
            if len(text) <= limit:
                return text
            return text[:limit] + "\n...[truncated]..."

        def _compact_existing_lessons(existing):
            compact = []
            for lesson in existing or []:
                if not isinstance(lesson, dict):
                    continue
                compact.append(
                    {
                        "lesson_id": lesson.get("lesson_id"),
                        "title": lesson.get("title") or lesson.get("lesson"),
                        "lesson": lesson.get("lesson"),
                        "tags": lesson.get("tags", []),
                        "applies_when": lesson.get("applies_when", {}),
                    }
                )
            return compact

        def _compact_selected_lessons(selected):
            compact = []
            for lesson in selected or []:
                if not isinstance(lesson, dict):
                    continue
                compact.append(
                    {
                        "lesson_id": lesson.get("lesson_id"),
                        "title": lesson.get("title") or lesson.get("lesson"),
                        "selection_reason": lesson.get("selection_reason"),
                        "historical_success_rate": lesson.get("historical_success_rate"),
                        "historical_score": lesson.get("historical_score"),
                    }
                )
            return compact

        user_prompt = {
            "task": "Extract reusable adaptive modeling lessons from a successful MILP repair.",
            "problem_definition": _safe_model_dump(problem),
            "repair_signal_before": repair_signal_before or {},
            "selected_modeling_lessons": _compact_selected_lessons(selected_modeling_lessons),
            "reasoning_plan": reasoning_plan if isinstance(reasoning_plan, dict) else {"value": str(reasoning_plan)},
            "before_evaluation": _safe_model_dump(before_evaluation),
            "after_evaluation": _safe_model_dump(after_evaluation),
            "before_code": _truncate(before_code, limit=12000),
            "after_code": _truncate(after_code, limit=12000),
            "existing_lessons_summary": _compact_existing_lessons(existing_lessons),
            "run_summary": run_summary or {},
            "curation_rules": {
                "max_new_lessons": 2,
                "prefer_narrow_reusable_lessons": True,
                "avoid_problem_specific_lessons": True,
                "avoid_duplicate_core_or_adaptive_lessons": True,
                "return_empty_if_no_new_reusable_pattern": True,
            },
        }

        try:
            response = self.client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": LESSON_CURATOR_SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)},
                ],
                text={"format": {"type": "json_object"}},
            )

            text = getattr(response, "output_text", None)
            if not text or not str(text).strip():
                return []

            cleaned = str(text).strip()

            if cleaned.startswith("```json"):
                cleaned = cleaned[len("```json"):].strip()
            elif cleaned.startswith("```"):
                cleaned = cleaned[len("```"):].strip()

            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].strip()

            parsed = json.loads(cleaned)

            if not isinstance(parsed, dict):
                return []

            new_lessons = parsed.get("new_lessons", [])
            if not isinstance(new_lessons, list):
                return []

            normalized = []
            for candidate in new_lessons:
                if not isinstance(candidate, dict):
                    continue

                lesson_id = candidate.get("lesson_id")
                title = candidate.get("title")
                applies_when = candidate.get("applies_when")
                lesson_text = candidate.get("lesson")
                recommended_actions = candidate.get("recommended_actions", [])
                anti_patterns = candidate.get("anti_patterns", [])
                priority_hint = candidate.get("priority_hint", 5)
                tags = candidate.get("tags", [])

                if not lesson_id or not isinstance(lesson_id, str):
                    continue
                if not title or not isinstance(title, str):
                    continue
                if not lesson_text or not isinstance(lesson_text, str):
                    continue
                if not isinstance(applies_when, dict):
                    continue
                if not isinstance(recommended_actions, list):
                    recommended_actions = []
                if not isinstance(anti_patterns, list):
                    anti_patterns = []
                if not isinstance(tags, list):
                    tags = []

                anti_patterns_list = [str(x).strip() for x in anti_patterns if str(x).strip()]
                recommended_actions_list = [str(x).strip() for x in recommended_actions if str(x).strip()]
                tags_list = [str(x).strip() for x in tags if str(x).strip()]

                normalized.append(
                    {
                        "lesson_id": lesson_id.strip(),
                        "title": title.strip(),
                        "applies_when": applies_when,
                        "lesson": lesson_text.strip(),
                        "recommended_actions": recommended_actions_list,
                        "anti_patterns": anti_patterns_list,
                        "priority_hint": int(priority_hint) if str(priority_hint).isdigit() else 5,
                        "tags": tags_list,
                        "lesson_type": "adaptive",
                        "protected": False,
                        "version": 1,
                    }
                )

            return normalized

        except Exception:
            return []
