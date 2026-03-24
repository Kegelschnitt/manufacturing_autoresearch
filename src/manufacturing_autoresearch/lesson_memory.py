from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class LessonMemory:
    def __init__(self, storage_path: str | Path | None = None):
        base_dir = Path(__file__).resolve().parent / "adaptive_lessons"
        self.storage_path = Path(storage_path) if storage_path is not None else base_dir / "lesson_memory.json"
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.lessons: dict[str, dict[str, Any]] = {}
        self.load()

    def _empty_payload(self) -> dict[str, list[dict[str, Any]]]:
        return {"lessons": []}

    def load(self) -> dict[str, dict[str, Any]]:
        if not self.storage_path.exists():
            self.save()
            return self.lessons

        try:
            payload = json.loads(self.storage_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = self._empty_payload()

        raw_lessons = payload.get("lessons", []) if isinstance(payload, dict) else []
        normalized: dict[str, dict[str, Any]] = {}
        for item in raw_lessons:
            if not isinstance(item, dict):
                continue
            lesson_id = str(item.get("lesson_id", "")).strip()
            if not lesson_id:
                continue
            normalized[lesson_id] = self._normalize_lesson_record(item)

        self.lessons = normalized
        return self.lessons

    def save(self) -> None:
        payload = {
            "lessons": [self.lessons[k] for k in sorted(self.lessons)]
        }
        self.storage_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _normalize_lesson_record(self, record: dict[str, Any]) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        return {
            "lesson_id": str(record.get("lesson_id", "")).strip(),
            "title": str(record.get("title") or record.get("lesson_id") or "Adaptive lesson").strip(),
            "applies_when": dict(record.get("applies_when") or {}),
            "lesson": str(record.get("lesson") or record.get("description") or "").strip(),
            "recommended_actions": list(record.get("recommended_actions") or []),
            "anti_patterns": list(record.get("anti_patterns") or []),
            "priority_hint": int(record.get("priority_hint", 5) or 5),
            "tags": list(record.get("tags") or ["adaptive"]),
            "lesson_type": "adaptive",
            "protected": False,
            "version": int(record.get("version", 1) or 1),
            "created_at": str(record.get("created_at") or now),
            "updated_at": str(record.get("updated_at") or now),
        }
    
    def _normalize_text(self, text: str) -> str:
        text = "" if text is None else str(text)
        text = text.lower().replace("_", " ").replace("-", " ")
        return " ".join(text.split())

    def _normalize_list_of_strings(self, values) -> list[str]:
        if not isinstance(values, list):
            return []
        cleaned = []
        for value in values:
            norm = self._normalize_text(value)
            if norm:
                cleaned.append(norm)
        return sorted(set(cleaned))

    def _normalize_applies_when(self, applies_when) -> dict[str, Any]:
        if not isinstance(applies_when, dict):
            applies_when = {}

        return {
            "failure_type": self._normalize_text(applies_when.get("failure_type", "")) or None,
            "missing_objective_terms": self._normalize_list_of_strings(
                applies_when.get("missing_objective_terms", [])
            ),
            "traceback_contains": self._normalize_list_of_strings(
                applies_when.get("traceback_contains", [])
            ),
            "violated_rules": self._normalize_list_of_strings(
                applies_when.get("violated_rules", [])
            ),
            "objective_ids": self._normalize_list_of_strings(
                applies_when.get("objective_ids", [])
            ),
            "always": bool(applies_when.get("always", False)),
        }
    
    def candidate_signature(self, record: dict[str, Any]) -> dict[str, Any]:
        return {
            "title": self._normalize_text(record.get("title", "")),
            "lesson": self._normalize_text(record.get("lesson", "")),
            "applies_when": self._normalize_applies_when(record.get("applies_when", {})),
        }
    
    def find_duplicate_or_similar(self, candidate: dict[str, Any]) -> dict[str, Any] | None:
        candidate_id = str(candidate.get("lesson_id", "")).strip()
        if candidate_id and candidate_id in self.lessons:
            return self.lessons[candidate_id]

        candidate_sig = self.candidate_signature(candidate)
        candidate_title = candidate_sig["title"]
        candidate_lesson = candidate_sig["lesson"]
        candidate_applies = candidate_sig["applies_when"]

        for existing in self.lessons.values():
            existing_sig = self.candidate_signature(existing)

            if candidate_title and candidate_title == existing_sig["title"]:
                return existing

            if candidate_lesson and candidate_lesson == existing_sig["lesson"]:
                return existing

            same_failure = (
                candidate_applies.get("failure_type") == existing_sig["applies_when"].get("failure_type")
            )
            same_missing_terms = (
                candidate_applies.get("missing_objective_terms")
                == existing_sig["applies_when"].get("missing_objective_terms")
            )

            if same_failure and same_missing_terms and candidate_title == existing_sig["title"]:
                return existing

        return None
    
    def merge_into_existing(self, lesson_id: str, candidate: dict[str, Any]) -> None:
        if lesson_id not in self.lessons:
            return

        existing = self.lessons[lesson_id]

        def _merge_unique_strings(old_values, new_values):
            old_values = old_values if isinstance(old_values, list) else []
            new_values = new_values if isinstance(new_values, list) else []
            merged = []
            seen = set()

            for value in old_values + new_values:
                text = str(value).strip()
                key = self._normalize_text(text)
                if text and key and key not in seen:
                    seen.add(key)
                    merged.append(text)

            return merged

        existing["tags"] = _merge_unique_strings(existing.get("tags", []), candidate.get("tags", []))
        existing["recommended_actions"] = _merge_unique_strings(
            existing.get("recommended_actions", []),
            candidate.get("recommended_actions", []),
        )
        existing["anti_patterns"] = _merge_unique_strings(
            existing.get("anti_patterns", []),
            candidate.get("anti_patterns", []),
        )

        from datetime import datetime, timezone
        existing["updated_at"] = datetime.now(timezone.utc).isoformat()

        self.lessons[lesson_id] = self._normalize_lesson_record(existing)
        self.save()

    def add_lesson(self, lesson_id: str, data: dict[str, Any]) -> None:
        record = dict(data or {})
        record["lesson_id"] = lesson_id
        record = self._normalize_lesson_record(record)
        self.lessons[lesson_id] = record
        self.save()

    def update_lesson(self, lesson_id: str, data: dict[str, Any]) -> None:
        current = dict(self.lessons.get(lesson_id, {}))
        current.update(data or {})
        current["lesson_id"] = lesson_id
        current["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.lessons[lesson_id] = self._normalize_lesson_record(current)
        self.save()

    def remove_lesson(self, lesson_id: str) -> None:
        if lesson_id in self.lessons:
            del self.lessons[lesson_id]
            self.save()

    def get_lessons(self) -> dict[str, dict[str, Any]]:
        return dict(self.lessons)

    def as_prompt_lessons(self) -> list[dict[str, Any]]:
        return [self.lessons[k] for k in sorted(self.lessons)]
    
    def has_lesson(self, lesson_id: str) -> bool:
        return lesson_id in self.lessons

    def get_lesson_ids(self) -> set[str]:
        return set(self.lessons.keys())
    