from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelingLesson:
    lesson_id: str
    title: str
    applies_when: dict[str, Any] = field(default_factory=dict)
    lesson: str = ""
    recommended_actions: list[str] = field(default_factory=list)
    anti_patterns: list[str] = field(default_factory=list)


def build_modeling_lessons() -> dict[str, ModelingLesson]:
    lessons = [
        ModelingLesson(
            lesson_id="objective_mismatch_fix_objective",
            title="Fix the MILP objective itself",
            applies_when={"failure_type": "objective_mismatch"},
            lesson="When the evaluator objective and solver objective differ, the MILP objective itself must change. Output formatting is not a valid fix.",
            recommended_actions=[
                "Modify the linear objective in build_model.",
                "Introduce auxiliary variables if the missing term is not directly linear in existing variables.",
                "Return an objective value equal to the actual optimized MILP objective.",
            ],
            anti_patterns=[
                "Changing only notes or explanation text",
                "Patching values in extract_assignments",
            ],
        ),
        ModelingLesson(
            lesson_id="no_structural_change_make_real_edit",
            title="Comments are not structural changes",
            applies_when={"failure_type": "no_structural_change"},
            lesson="If the loop flags no_structural_change, the next candidate must add or modify variables, constraints, or the objective.",
            recommended_actions=[
                "Add a new variable family, linking constraints, or an objective term.",
                "Make the smallest real MILP change needed to address the failure.",
            ],
            anti_patterns=[
                "Only reformatting code",
                "Only renaming variables",
                "Only adding comments",
            ],
        ),
        ModelingLesson(
            lesson_id="runtime_error_sparse_indexing",
            title="Respect sparse indexing",
            applies_when={"failure_type": "runtime_error", "traceback_contains": ["KeyError"]},
            lesson="Assignment variables are often created only for eligible job-machine-slot combinations. Never assume every x[(job, machine, slot)] exists.",
            recommended_actions=[
                "Guard variable references with membership checks like `(job, machine, slot) in x`.",
                "Build auxiliary variables only for valid existing x-pairs.",
                "Iterate over existing x keys where possible.",
            ],
            anti_patterns=[
                "Dense indexing over all jobs, machines, and slots without checking membership",
                "Referencing x[(job, machine, slot)] for ineligible pairs",
            ],
        ),
        ModelingLesson(
            lesson_id="preserve_working_hard_constraints",
            title="Preserve already satisfied hard rules",
            applies_when={"always": True},
            lesson="If a hard rule already passes in the current best solution, preserve that working structure unless the violation directly requires changing it.",
            recommended_actions=[
                "Keep existing job-once and capacity logic if those checks already pass.",
                "Target only the weakest part of the MILP first.",
            ],
            anti_patterns=[
                "Rewriting all feasibility constraints unnecessarily",
            ],
        ),
        ModelingLesson(
            lesson_id="changeover_requires_auxiliary_variables",
            title="Model changeovers with explicit transition logic",
            applies_when={"missing_objective_terms": ["changeover_penalty"]},
            lesson="Changeover penalties usually require explicit transition variables and linking constraints rather than a formatting fix.",
            recommended_actions=[
                "Introduce transition or setup indicator variables.",
                "Link them to the assignment variables with linear inequalities.",
                "Add the changeover term directly to the MILP objective.",
            ],
            anti_patterns=[
                "Assuming the evaluator will infer changeovers from notes",
                "Adding a scalar constant to the objective without linking variables",
            ],
        ),
        ModelingLesson(
            lesson_id="dont_repair_in_extract_assignments",
            title="Do not repair in extract_assignments",
            applies_when={"always": True},
            lesson="extract_assignments should read the solved model result, not enforce feasibility or patch objective logic.",
            recommended_actions=[
                "Encode all required constraints and objective terms in build_model.",
            ],
            anti_patterns=[
                "Filtering invalid assignments after solving",
                "Injecting missing costs in extract_assignments",
            ],
        ),
        ModelingLesson(
            lesson_id="runtime_first_fix_execution",
            title="Fix runtime issues before optimization",
            applies_when={"failure_type": "runtime_error"},
            lesson="If the generated program does not execute, repair execution before attempting deeper modeling improvements.",
            recommended_actions=[
                "Resolve syntax and indexing issues first.",
                "Preserve the previous working logic while fixing execution.",
            ],
            anti_patterns=[
                "Adding more complexity before fixing the crash",
            ],
        ),
    ]
    return {lesson.lesson_id: lesson for lesson in lessons}


def lesson_to_prompt_dict(lesson: ModelingLesson) -> dict[str, Any]:
    return {
        "lesson_id": lesson.lesson_id,
        "title": lesson.title,
        "applies_when": dict(lesson.applies_when),
        "lesson": lesson.lesson,
        "recommended_actions": list(lesson.recommended_actions),
        "anti_patterns": list(lesson.anti_patterns),
    }
