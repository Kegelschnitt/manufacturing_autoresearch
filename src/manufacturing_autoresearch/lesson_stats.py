from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class LessonStats:
    def __init__(self, storage_path: str | Path | None = None):
        base_dir = Path(__file__).resolve().parent / "adaptive_lessons"
        self.storage_path = Path(storage_path) if storage_path is not None else base_dir / "lesson_stats.json"
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.usage: dict[str, dict[str, Any]] = {}
        self.load()

    def _default_entry(self) -> dict[str, Any]:
        return {
            "times_selected": 0,
            "times_successful": 0,
            "times_failed": 0,
            "last_used": None,
            "last_outcome": None,
        }
    
    def _refresh_derived_fields(self, entry: dict[str, Any]) -> None:
        successful = int(entry.get("times_successful", 0) or 0)
        failed = int(entry.get("times_failed", 0) or 0)
        resolved = successful + failed
        success_rate = (successful / resolved) if resolved else 0.0
        score = success_rate * math.log1p(resolved)
        entry["success_rate"] = success_rate
        entry["score"] = score

    def load(self) -> dict[str, dict[str, Any]]:
        if not self.storage_path.exists():
            self.save()
            return self.usage

        try:
            payload = json.loads(self.storage_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}

        normalized: dict[str, dict[str, Any]] = {}
        if isinstance(payload, dict):
            for lesson_id, stats in payload.items():
                if not isinstance(stats, dict):
                    continue
                entry = self._default_entry()
                entry.update(stats)
                entry["times_selected"] = int(entry.get("times_selected", 0) or 0)
                entry["times_successful"] = int(entry.get("times_successful", 0) or 0)
                entry["times_failed"] = int(entry.get("times_failed", 0) or 0)
                self._refresh_derived_fields(entry)
                normalized[str(lesson_id)] = entry

        self.usage = normalized
        return self.usage

    def save(self) -> None:
        self.storage_path.write_text(
            json.dumps(self.usage, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _ensure(self, lesson_id: str) -> dict[str, Any]:
        if lesson_id not in self.usage:
            self.usage[lesson_id] = self._default_entry()
        return self.usage[lesson_id]

    def record_usage(self, lesson_id: str) -> None:
        entry = self._ensure(lesson_id)
        entry["times_selected"] += 1
        entry["last_used"] = datetime.now(timezone.utc).isoformat()
        self._refresh_derived_fields(entry)
        self.save()

    def record_outcome(self, lesson_id: str, success: bool) -> None:
        entry = self._ensure(lesson_id)
        if success:
            entry["times_successful"] += 1
            entry["last_outcome"] = "success"
        else:
            entry["times_failed"] += 1
            entry["last_outcome"] = "failure"
        self._refresh_derived_fields(entry)
        self.save()

    def get_stats(self) -> dict[str, dict[str, Any]]:
        enriched: dict[str, dict[str, Any]] = {}
        for lesson_id, entry in self.usage.items():
            current = dict(entry)
            self._refresh_derived_fields(current)
            enriched[lesson_id] = current
        return enriched

    def get_lesson_stats(self, lesson_id: str) -> dict[str, Any]:
        stats = self.get_stats()
        default_entry = self._default_entry()
        self._refresh_derived_fields(default_entry)
        return stats.get(lesson_id, default_entry)
