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