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

def _lesson_stats_for(lesson_id: str, lesson_stats) -> dict[str, Any]:
    if lesson_stats is None:
        return {
            "times_selected": 0,
            "times_successful": 0,
            "times_failed": 0,
            "success_rate": 0.0,
            "score": 0.0,
        }
    return lesson_stats.get_lesson_stats(lesson_id)


def _relevance_score(lesson: dict[str, Any], repair_signal: dict[str, Any]) -> tuple[float, str | None]:
    applies = lesson.get("applies_when") or {}
    failure_type = repair_signal.get("failure_type")
    missing_terms = _missing_terms(repair_signal)

    if applies.get("always"):
        return 0.25, "always-applicable guardrail"

    if applies.get("failure_type") == failure_type and failure_type:
        return 1.0, f"matched failure_type={failure_type}"

    expected_missing = set(applies.get("missing_objective_terms", []) or [])
    if expected_missing and (expected_missing & missing_terms):
        matched = sorted(expected_missing & missing_terms)
        return 0.9, f"matched missing objective term(s): {', '.join(matched)}"

    traceback_needles = applies.get("traceback_contains", []) or []
    if traceback_needles and _traceback_contains(repair_signal, traceback_needles):
        return 0.85, f"matched traceback pattern(s): {', '.join(traceback_needles)}"

    return 0.0, None

def select_lessons(
    repair_signal,
    proposer_guidance=None,
    run_memory=None,
    llm=None,
    problem=None,
    available_lessons=None,
    current_best_code=None,
    lesson_stats=None,
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
                lesson_stats=lesson_stats,
            )
            if selected:
                return selected
        except Exception as exc:
            print(f"[debug] LLM lesson selector failed: {type(exc).__name__}: {exc}")

    return heuristic_select_lessons(
        repair_signal=repair_signal,
        proposer_guidance=proposer_guidance,
        run_memory=run_memory,
        available_lessons=available_lessons,
        lesson_stats=lesson_stats,
    )


def heuristic_select_lessons(
    repair_signal: dict[str, Any] | None,
    proposer_guidance: dict[str, Any] | None = None,
    run_memory: dict[str, Any] | None = None,
    available_lessons: list[dict[str, Any]] | None = None,
    lesson_stats=None,
    max_lessons: int = 5,
) -> list[dict[str, Any]]:
        repair_signal = repair_signal or {}
        proposer_guidance = proposer_guidance or {}
        run_memory = run_memory or {}

        if available_lessons is None:
            available_lessons = [
                lesson_to_prompt_dict(lesson)
                for lesson in build_modeling_lessons().values()
            ]

        ranked: list[tuple[float, dict[str, Any], str]] = []

        for lesson in available_lessons:
            if not isinstance(lesson, dict):
                continue

            lesson_id = lesson.get("lesson_id")
            if not lesson_id:
                continue

            relevance, reason = _relevance_score(lesson, repair_signal)
            if relevance <= 0.0:
                continue

            stats = _lesson_stats_for(lesson_id, lesson_stats)
            stats_score = float(stats.get("score", 0.0) or 0.0)
            times_selected = int(stats.get("times_selected", 0) or 0)

            exploration_bonus = 0.15 if times_selected == 0 else 0.0
            final_score = relevance + 0.35 * stats_score + exploration_bonus

            item = dict(lesson)
            item["selection_reason"] = reason
            item["selection_score"] = final_score
            item["historical_success_rate"] = float(stats.get("success_rate", 0.0) or 0.0)
            item["historical_score"] = stats_score
            item["historical_times_selected"] = times_selected

            ranked.append((final_score, item, lesson_id))

        if not ranked:
            fallback_ids = [
                "preserve_working_hard_constraints",
                "dont_repair_in_extract_assignments",
            ]
            fallback_items: list[dict[str, Any]] = []
            lesson_map = {
                lesson.get("lesson_id"): lesson
                for lesson in available_lessons
                if isinstance(lesson, dict) and lesson.get("lesson_id")
            }

            for lesson_id in fallback_ids:
                lesson = lesson_map.get(lesson_id)
                if lesson:
                    item = dict(lesson)
                    item["selection_reason"] = "fallback default lesson"
                    item["selection_score"] = 0.0
                    fallback_items.append(item)

            return fallback_items[:max_lessons]

        ranked.sort(key=lambda x: x[0], reverse=True)

        deduped: list[dict[str, Any]] = []
        seen = set()
        for _, item, lesson_id in ranked:
            if lesson_id in seen:
                continue
            seen.add(lesson_id)
            deduped.append(item)

        return deduped[:max_lessons]