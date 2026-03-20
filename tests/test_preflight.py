from manufacturing_autoresearch.baseline import baseline_program
from manufacturing_autoresearch.preflight import strip_code_fences, validate_program


def test_strip_code_fences():
    text = "```python\nprint('x')\n```"
    assert strip_code_fences(text) == "print('x')"


def test_baseline_passes_preflight():
    report = validate_program(baseline_program())
    assert report.ok, report.errors
