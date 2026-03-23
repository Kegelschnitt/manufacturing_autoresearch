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

def select_lessons(
    repair_signal,
    proposer_guidance=None,
    run_memory=None,
    llm=None,
    problem=None,
    available_lessons=None,
    current_best_code=None,
):
    if available_lessons is None:
        available_lessons = [
            lesson_to_prompt_dict(lesson)
            for lesson in build_modeling_lessons().values()
        ]

    if llm is not None and problem is not None:
        try:
            selected = llm.select_lessons(
                problem=problem,
                repair_signal=repair_signal,
                proposer_guidance=proposer_guidance,
                available_lessons=available_lessons,
                current_best_code=proposer_guidance.get("current_best_code_preview", ""),
                run_memory=run_memory or {},
            )
            if selected:
                return selected
        except Exception as exc:
            print(f"[debug] LLM lesson selector failed: {type(exc).__name__}: {exc}")

    return heuristic_select_lessons(
        repair_signal=repair_signal,
        proposer_guidance=proposer_guidance,
        run_memory=run_memory,
    )


def heuristic_select_lessons(
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

    selected: list[tuple[str, str]] = []

    for lesson_id, lesson in registry.items():
        applies = lesson.applies_when or {}

        if applies.get("always"):
            selected.append((lesson_id, "always-applicable guardrail"))
            continue

        if applies.get("failure_type") == failure_type:
            selected.append((lesson_id, f"matched failure_type={failure_type}"))
            continue

        expected_missing = set(applies.get("missing_objective_terms", []))
        if expected_missing and (expected_missing & missing_terms):
            matched = sorted(expected_missing & missing_terms)
            selected.append(
                (lesson_id, f"matched missing objective term(s): {', '.join(matched)}")
            )
            continue

        traceback_needles = applies.get("traceback_contains", [])
        if traceback_needles and _traceback_contains(repair_signal, traceback_needles):
            selected.append(
                (lesson_id, f"matched traceback pattern(s): {', '.join(traceback_needles)}")
            )
            continue

    if not selected:
        selected = [
            ("preserve_working_hard_constraints", "fallback default lesson"),
            ("dont_repair_in_extract_assignments", "fallback default lesson"),
        ]

    deduped: list[tuple[str, str]] = []
    seen = set()
    for lesson_id, reason in selected:
        if lesson_id not in seen and lesson_id in registry:
            seen.add(lesson_id)
            deduped.append((lesson_id, reason))

    result: list[dict[str, Any]] = []
    for lesson_id, reason in deduped[:max_lessons]:
        item = lesson_to_prompt_dict(registry[lesson_id])
        item["selection_reason"] = reason
        result.append(item)

    return result