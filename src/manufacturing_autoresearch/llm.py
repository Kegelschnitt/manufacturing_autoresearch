from __future__ import annotations

from .types import ProgramProposal, ProblemDefinition


class LLMClient:
    def propose(self, problem: ProblemDefinition, current_best: ProgramProposal, repair_signal: dict) -> ProgramProposal:
        # Safe fallback: return current best unchanged. This keeps the package deterministic
        # even without an API key, while preserving the baseline-plus-fallback architecture.
        return current_best
