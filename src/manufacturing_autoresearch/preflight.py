from __future__ import annotations

import ast

from .types import PreflightReport, ProgramProposal

FORBIDDEN_PATTERNS = ["```", "data['problem']", 'data["problem"]']


def strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        t = "\n".join(lines)
    return t.strip()


def run_preflight(program: ProgramProposal) -> PreflightReport:
    code = strip_code_fences(program.model_logic)
    errors: list[str] = []
    warnings: list[str] = []
    for pat in FORBIDDEN_PATTERNS:
        if pat in code:
            errors.append(f"forbidden pattern present: {pat}")
    try:
        ast.parse(code)
    except SyntaxError as e:
        errors.append(f"syntax error: {e}")
    for required in ["def build_model", "def extract_assignments"]:
        if required not in code:
            errors.append(f"missing required function: {required}")
    return PreflightReport(ok=not errors, errors=errors, warnings=warnings, normalized_code=code)
