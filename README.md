# Manufacturing Autoresearch V3

This version uses a **baseline-plus-fallback** design.

Core idea:
- start from a known-good baseline MILP implementation
- ask the LLM only for **small edits** to the model logic
- validate generated code before execution
- keep the **best working version**
- reject regressions and fall back automatically

## Run

```bash
pip install -e .
export PYTHONPATH=src
python -m manufacturing_autoresearch.main \
  --problem configs/example_problem.json \
  --out runs/example_run \
  --max-iters 5
```

## What is fixed compared with earlier versions

- baseline solver always exists
- code fences are stripped automatically
- preflight checks use `ast.parse`
- invalid candidates are rejected before execution
- worse candidates are not allowed to replace the best working solver

## Important design choice

The strict contract is on:
- `problem.json`
- `result.json`
- deterministic evaluation

The LLM is only asked to modify the modeling logic inside two functions:
- `build_model(data)`
- `extract_assignments(data, context)`
