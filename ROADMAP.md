# Manufacturing Autoresearch Roadmap

## Phase 1 — Persistent Lesson Bookkeeping ✅
- lesson_memory.json
- lesson_stats.json
- usage + outcome tracking
- runtime merging of core + adaptive lessons

## Phase 2 — Stats-Aware Lesson Selection ✅
- heuristic selector uses stats
- LLM selector sees historical performance
- adaptive lessons participate in selection

## Phase 2.1 — Selection Refinement (in progress)
- reduce redundant lesson bundling
- prefer minimal sufficient lesson sets

## Phase 3 — Adaptive Lesson Creation (next)
- generate new lessons after successful repairs
- store in lesson_memory.json
- basic filtering/validation