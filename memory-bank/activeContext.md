# Active Context (`advanced_prompting_system`)

## Current Focus
- **Antigravity 2.0+ Custom Agent Team & Coordination Rules**: Established the 9-agent custom subagent roster in `.agents/agents/<name>/agent.md`, modular repository rules in `.agents/rules/` (and mirrored in `.agent/rules/`), supporting documentation in `.agents/doc/antigravity-agent-team.md`, root contracts (`GEMINI.md` and `AGENTS.md`), and this `memory-bank/` directory.
- **Offline Qualification Baseline (PR #34 & Issue #35 Phase 2)**:
  - Backend offline suite (`./tools/run-offline.ps1 -PytestArgs '-q'`): **148 passed / 0 failed / 0 skipped** (including all 48 previously quarantined tests in `test_advanced_prompting.py`).
  - Frontend offline suite (`./tools/run-frontend-offline.ps1 -Action test`): **111 passed / 0 failed** across 10 Vitest files, plus passing `tsc -b` and `vite build`.

## Recent Verified Architectural Changes
1. **PR #34 Review Repairs (`ae00ecb`)**:
   - `AdvancedPromptEngineer.main()` passes explicit keyword arguments, the actual `PlanStep` object, and an integer collaborative budget into `invoke_collaborative_reasoning`.
   - All 5 direct spaCy consumers in `complexity_measures.py` (`is_real_words`, `is_complex_graph`, `is_complex_recursive`, `is_complex_ast`, `is_complex_llm`) resolve `get_nlp_spacy()` lazily and return safe defaults without calling LLM providers when `en_core_web_sm` is absent.
   - `define_problem` returns the selected definition on partial majority; continuation votes count only `CONTINUE`/`END`; low participation finalizes explicitly; majority-NO, weak-YES, tie, and all-error final votes route through `MediatorReviseOrDecide`.
   - `ConversationMemory` supports legacy string/list round-trip deserialization through `add_argument`, `add_to_do_item`, and `add_decision`.
2. **Issue #35 Phase 2 Unquarantine**:
   - All 24 quarantined tests in `test_advanced_prompting.py` resolved and passing offline with deterministic mock judge/chat fixtures, deduplicated reflection combiners, insertion-order preservation in `consolidate_steps` and `merge_interactions`, and offline `tiktoken` fallback in `count_tokens`.

## Next Priorities
1. **Issue #35 Remaining Phases**:
   - Clean up the 13 ESLint errors and 1 Fast Refresh warning in `collaborative-assistant-frontend/`.
   - Shared run-level configuration and full-stack API/UI execution-mode integration.
2. **Issue #36**:
   - Implement genuine Tree-of-Thoughts (`ToT`) search with bounded offline-testable provider contracts.
