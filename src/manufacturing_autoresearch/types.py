from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ConfigDict


class CoreModel(BaseModel):
    model_config = ConfigDict(protected_namespaces=())


class ProblemDefinition(CoreModel):
    name: str
    description: str
    entities: dict[str, list[Any]]
    parameters: dict[str, Any]
    evaluation_framework: dict[str, Any]


class ProgramProposal(CoreModel):
    explanation: str
    model_logic: str
    expected_failure_modes: list[str] = Field(default_factory=list)


class PreflightReport(CoreModel):
    ok: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    normalized_code: str = ""


class Assignment(CoreModel):
    job: str
    machine: str
    slot: int


class SolverResult(CoreModel):
    solver_status: str
    objective_value: float | None = None
    assignments: list[Assignment] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class RuleCheck(CoreModel):
    rule_id: str
    passed: bool
    details: str


class EvaluationReport(CoreModel):
    is_feasible: bool
    is_acceptable: bool
    solver_status_ok: bool
    computed_objective_value: float
    reported_objective_value: float | None = None
    objective_terms: dict[str, float] = Field(default_factory=dict)
    rule_checks: list[RuleCheck] = Field(default_factory=list)
    violations: list[str] = Field(default_factory=list)
    infeasibility_diagnosis: list[dict[str, Any]] = Field(default_factory=list)
    summary: str


class CandidateDecision(CoreModel):
    accepted: bool
    reason: str
    candidate_score: tuple[int, int, int, float]
    best_score_before: tuple[int, int, int, float]


class IterationRecord(CoreModel):
    iteration: int
    candidate_program: ProgramProposal
    candidate_preflight: PreflightReport
    candidate_result: SolverResult
    candidate_evaluation: EvaluationReport
    accepted_as_best: bool
    decision: CandidateDecision
    best_evaluation_after: EvaluationReport
    repair_signal: dict[str, Any]
    selected_modeling_lessons: list[dict[str, Any]] = Field(default_factory=list)
    reasoning_plan: dict[str, Any] = Field(default_factory=dict)


class RunState(CoreModel):
    problem: ProblemDefinition
    current_iteration: int = 0
    max_iterations: int = 5
    history: list[IterationRecord] = Field(default_factory=list)
    repair_signal: dict[str, Any] = Field(default_factory=dict)
    best_program: ProgramProposal
    best_preflight: PreflightReport
    best_result: SolverResult
    best_evaluation: EvaluationReport
    latest_program: ProgramProposal
    latest_preflight: PreflightReport
    latest_result: SolverResult
    latest_evaluation: EvaluationReport
    stop: bool = False
    final_message: str = ""


class Settings(CoreModel):
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    max_iterations: int = 5
    run_dir: Path | None = None
