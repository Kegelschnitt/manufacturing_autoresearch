from __future__ import annotations

from typing import Any

from .modeling_lessons import build_modeling_lessons, lesson_to_prompt_dict


def _traceback_contains(repair_signal: dict[str, Any], needles: list[str]) -> bool:
    haystack = str(repair_signal.get("last_traceback", "") or "")
    return any(needle in haystack for needle in needles)


def _missing_terms(repair_signal: dict[str, Any]) -> set[str]:
    return {
        item.get("term")
        for item in (repair_signal.get("missing_objective_terms", []) or [])
        if item.get("term")
    }


def select_modeling_lessons(
    repair_signal: dict[str, Any] | None,
    proposer_guidance: dict[str, Any] | None = None,
    run_memory: dict[str, Any] | None = None,
    max_lessons: int = 5,
) -> list[dict[str, Any]]:
    repair_signal = repair_signal or {}
    proposer_guidance = proposer_guidance or {}
    run_memory = run_memory or {}

    registry = build_modeling_lessons()
    failure_type = repair_signal.get("failure_type")
    missing_terms = _missing_terms(repair_signal)

    selected: list[str] = []

    for lesson_id, lesson in registry.items():
        applies = lesson.applies_when or {}

        if applies.get("always"):
            selected.append(lesson_id)
            continue

        if applies.get("failure_type") == failure_type:
            selected.append(lesson_id)
            continue

        expected_missing = set(applies.get("missing_objective_terms", []))
        if expected_missing and (expected_missing & missing_terms):
            selected.append(lesson_id)
            continue

        traceback_needles = applies.get("traceback_contains", [])
        if traceback_needles and _traceback_contains(repair_signal, traceback_needles):
            selected.append(lesson_id)
            continue

    if not selected:
        selected = [
            "preserve_working_hard_constraints",
            "dont_repair_in_extract_assignments",
        ]

    deduped = []
    seen = set()
    for lesson_id in selected:
        if lesson_id not in seen and lesson_id in registry:
            seen.add(lesson_id)
            deduped.append(lesson_id)

    return [lesson_to_prompt_dict(registry[lesson_id]) for lesson_id in deduped[:max_lessons]]
