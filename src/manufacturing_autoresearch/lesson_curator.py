from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ValidationResult:
    ok: bool
    reason: str | None = None


class LessonCurator:
    """
    Phase 3 curator:
    - asks the LLM for new adaptive lessons after accepted repairs
    - validates candidate lessons aggressively
    - filters out duplicates / benchmark-specific / overly generic proposals
    """

    def __init__(self, memory):
        self.memory = memory

    def propose_lessons(
        self,
        llm,
        problem,
        selected_modeling_lessons: list[dict[str, Any]] | None,
        repair_signal_before: dict[str, Any] | None,
        reasoning_plan: dict[str, Any] | None,
        before_code: str | None,
        after_code: str | None,
        before_evaluation: Any,
        after_evaluation: Any,
        run_summary: dict[str, Any] | None = None,
        existing_lessons: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Ask the LLM to extract reusable adaptive lessons from a successful repair.
        Returns raw candidate lesson dicts. Validation is handled separately.
        """
        if llm is None:
            return []

        try:
            lessons = llm.curate_lessons(
                problem=problem,
                selected_modeling_lessons=selected_modeling_lessons or [],
                repair_signal_before=repair_signal_before or {},
                reasoning_plan=reasoning_plan or {},
                before_code=before_code or "",
                after_code=after_code or "",
                before_evaluation=self._to_jsonable(before_evaluation),
                after_evaluation=self._to_jsonable(after_evaluation),
                existing_lessons=existing_lessons or self.memory.as_prompt_lessons(),
                run_summary=run_summary or {},
            )

        except Exception as exc:
            print(f"[debug] lesson curator failed: {type(exc).__name__}: {exc}")
            return []

        if not isinstance(lessons, list):
            print(f"[debug] curator returned non-list type: {type(lessons).__name__}")
            return []

        lessons = [lesson for lesson in lessons if isinstance(lesson, dict)]
        return lessons

    def filter_new_lessons(
        self,
        proposed_lessons: list[dict[str, Any]] | None,
        problem_name: str,
        existing_core_lessons: list[dict[str, Any]] | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        """
        Validate, classify, and normalize candidate lessons.

        Returns:
            accepted_new_lessons, merge_updates, rejected_candidates
        """
        proposed_lessons = proposed_lessons or []
        existing_core_lessons = existing_core_lessons or []

        core_ids = {
            lesson.get("lesson_id")
            for lesson in existing_core_lessons
            if isinstance(lesson, dict) and lesson.get("lesson_id")
        }

        accepted: list[dict[str, Any]] = []
        merge_updates: list[dict[str, Any]] = []
        rejected_candidates: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        for candidate in proposed_lessons:
            normalized = self._normalize_candidate(candidate)
            lesson_id = normalized.get("lesson_id")

            if not lesson_id:
                rejected_candidates.append(
                    {
                        "lesson_id": None,
                        "action": "reject",
                        "reason": "missing lesson_id after normalization",
                    }
                )
                continue

            if lesson_id in seen_ids:
                rejected_candidates.append(
                    {
                        "lesson_id": lesson_id,
                        "action": "duplicate_skip",
                        "reason": "duplicate lesson_id within same proposal batch",
                    }
                )
                continue

            action, existing, reason = self.classify_candidate(
                candidate=normalized,
                problem_name=problem_name,
                core_ids=core_ids,
            )

            if action == "accept":
                accepted.append(normalized)
                seen_ids.add(lesson_id)
                continue

            if action == "merge_update" and existing is not None:
                target_lesson_id = existing.get("lesson_id")
                if target_lesson_id:
                    merge_updates.append(
                        {
                            "target_lesson_id": target_lesson_id,
                            "candidate": normalized,
                            "reason": reason,
                        }
                    )
                else:
                    rejected_candidates.append(
                        {
                            "lesson_id": lesson_id,
                            "action": "reject",
                            "reason": "merge target missing lesson_id",
                        }
                    )
                continue

            rejected_candidates.append(
                {
                    "lesson_id": lesson_id,
                    "action": action,
                    "reason": reason,
                }
            )

        return accepted, merge_updates, rejected_candidates

    def validate_candidate(
        self,
        candidate: dict[str, Any],
        problem_name: str,
        core_ids: set[str] | None = None,
    ) -> ValidationResult:
        core_ids = core_ids or set()

        lesson_id = candidate.get("lesson_id")
        title = candidate.get("title")
        lesson_text = candidate.get("lesson")
        applies_when = candidate.get("applies_when")
        recommended_actions = candidate.get("recommended_actions")
        anti_patterns = candidate.get("anti_patterns")

        if not isinstance(lesson_id, str) or not lesson_id.strip():
            return ValidationResult(False, "missing lesson_id")

        if not isinstance(title, str) or not title.strip():
            return ValidationResult(False, "missing title")

        if not isinstance(lesson_text, str) or len(lesson_text.strip()) < 40:
            return ValidationResult(False, "lesson text too short")

        if not isinstance(applies_when, dict) or not applies_when:
            return ValidationResult(False, "applies_when must be a non-empty dict")

        if not isinstance(recommended_actions, list):
            return ValidationResult(False, "recommended_actions must be a list")

        if not isinstance(anti_patterns, list):
            return ValidationResult(False, "anti_patterns must be a list")

        if lesson_id in core_ids:
            return ValidationResult(False, "lesson_id already exists in core lessons")

        if self._looks_too_generic(title, lesson_text):
            return ValidationResult(False, "lesson is too generic")

        if self._looks_problem_specific(candidate, problem_name):
            return ValidationResult(False, "lesson appears benchmark-specific")

        if applies_when.get("always") is True:
            return ValidationResult(False, "adaptive lessons should not use always=true in Phase 3")

        return ValidationResult(True, None)
    
    def classify_candidate(
        self,
        candidate: dict[str, Any],
        problem_name: str,
        core_ids: set[str] | None = None,
    ) -> tuple[str, dict[str, Any] | None, str | None]:
        core_ids = core_ids or set()

        validation = self.validate_candidate(
            candidate=candidate,
            problem_name=problem_name,
            core_ids=core_ids,
        )
        if not validation.ok:
            return "reject", None, validation.reason

        lesson_id = candidate.get("lesson_id")
        if lesson_id in core_ids:
            return "reject", None, "lesson_id already exists in core lessons"

        existing = None
        if hasattr(self.memory, "find_duplicate_or_similar"):
            existing = self.memory.find_duplicate_or_similar(candidate)

        if existing is None:
            return "accept", None, None

        existing_id = existing.get("lesson_id")
        if existing_id == lesson_id:
            return "duplicate_skip", existing, "lesson_id already exists in adaptive memory"

        return "merge_update", existing, "similar adaptive lesson already exists"

    def _normalize_candidate(self, candidate: dict[str, Any]) -> dict[str, Any]:
        lesson_id = str(candidate.get("lesson_id", "")).strip()
        title = str(candidate.get("title", "")).strip()
        lesson_text = str(candidate.get("lesson", candidate.get("description", ""))).strip()

        applies_when = candidate.get("applies_when")
        if not isinstance(applies_when, dict):
            applies_when = {}

        recommended_actions = candidate.get("recommended_actions")
        if isinstance(recommended_actions, str):
            recommended_actions = [recommended_actions]
        elif not isinstance(recommended_actions, list):
            recommended_actions = []

        anti_patterns = candidate.get("anti_patterns")
        if isinstance(anti_patterns, str):
            anti_patterns = [anti_patterns]
        elif not isinstance(anti_patterns, list):
            anti_patterns = []

        tags = candidate.get("tags")
        if not isinstance(tags, list):
            tags = []

        priority_hint = candidate.get("priority_hint", 5)
        try:
            priority_hint = int(priority_hint)
        except Exception:
            priority_hint = 5

        normalized = {
            "lesson_id": lesson_id,
            "title": title,
            "applies_when": applies_when,
            "lesson": lesson_text,
            "recommended_actions": [str(x).strip() for x in recommended_actions if str(x).strip()],
            "anti_patterns": [str(x).strip() for x in anti_patterns if str(x).strip()],
            "priority_hint": priority_hint,
            "tags": [str(x).strip() for x in tags if str(x).strip()],
            "lesson_type": "adaptive",
            "protected": False,
            "version": int(candidate.get("version", 1) or 1),
        }

        if "adaptive" not in normalized["tags"]:
            normalized["tags"].append("adaptive")

        return normalized

    def _looks_too_generic(self, title: str, lesson_text: str) -> bool:
        combined = f"{title} {lesson_text}".lower()

        generic_markers = [
            "be careful",
            "fix the model",
            "fix infeasibility",
            "use correct constraints",
            "set the right objective",
            "improve the model",
            "make the code better",
            "repair the bug",
            "ensure correctness",
        ]

        return any(marker in combined for marker in generic_markers)

    def _looks_problem_specific(self, candidate: dict[str, Any], problem_name: str) -> bool:
        fields = [
            candidate.get("lesson_id", ""),
            candidate.get("title", ""),
            candidate.get("lesson", ""),
            " ".join(candidate.get("recommended_actions", []) or []),
            " ".join(candidate.get("anti_patterns", []) or []),
        ]
        blob = " ".join(str(x) for x in fields).lower()

        problem_name = (problem_name or "").strip().lower()
        bad_markers = [
            problem_name,
            "runs/",
            ".json",
            ".py",
            "configs/benchmarks",
        ]

        return any(marker and marker in blob for marker in bad_markers)

    def _to_jsonable(self, value: Any) -> Any:
        if value is None:
            return None

        if isinstance(value, (str, int, float, bool)):
            return value

        if isinstance(value, list):
            return [self._to_jsonable(v) for v in value]

        if isinstance(value, dict):
            return {str(k): self._to_jsonable(v) for k, v in value.items()}

        if hasattr(value, "model_dump"):
            try:
                return value.model_dump()
            except Exception:
                pass

        if hasattr(value, "dict"):
            try:
                return value.dict()
            except Exception:
                pass

        if hasattr(value, "__dict__"):
            try:
                return {
                    str(k): self._to_jsonable(v)
                    for k, v in vars(value).items()
                    if not str(k).startswith("_")
                }
            except Exception:
                pass

        return str(value)