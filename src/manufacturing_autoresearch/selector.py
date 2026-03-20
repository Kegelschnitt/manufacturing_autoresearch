from __future__ import annotations

from .types import CandidateDecision, EvaluationReport


def _score(ev: EvaluationReport) -> tuple[int, int, int, float]:
    passed = sum(1 for c in ev.rule_checks if c.passed)
    return (
        1 if ev.is_acceptable else 0,
        1 if ev.is_feasible else 0,
        passed,
        -float(ev.computed_objective_value),
    )


def should_accept_candidate(candidate: EvaluationReport, best: EvaluationReport) -> CandidateDecision:
    cs = _score(candidate)
    bs = _score(best)
    accepted = cs > bs
    reason = "Candidate beat current best." if accepted else "Candidate did not beat current best; keeping fallback."
    return CandidateDecision(accepted=accepted, reason=reason, candidate_score=cs, best_score_before=bs)
