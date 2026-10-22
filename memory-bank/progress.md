# Project Progress & Baselines (`advanced_prompting_system`)

## Verified Offline Baselines
- **Backend Offline Test Suite (`./tools/run-offline.ps1 -PytestArgs '-q'`)**:
  - **206 passed, 0 failed, 0 skipped**
  - Includes 48 tests in `test_advanced_prompting.py`, 41 tests in `test_execution_modes.py`, 13 tests in `test_empirical_challenger_2.py`, 6 tests in `test_graphql_contract.py`, 6 tests in `test_smoke.py`, and full regression suites in `test_pr34_regressions.py`, `test_voting_semantics.py`, `test_memory_management.py`, `test_message_handling.py`, `test_offline_runtime.py`, `test_import_integrity.py`, `test_retrieval.py`, and `test_finalize_step_output.py`.
- **Frontend Offline Test Suite (`./tools/run-frontend-offline.ps1 -Action test`)**:
  - **124 passed, 0 failed** across 13 test files (including new suites `StructuredOutputRenderer.test.tsx`, `ReasoningDiagnosticsPanel.test.tsx`, and `useRunExecution.test.ts`).
  - Production TypeScript & Vite bundle build (`-Action build`): **PASS**.
  - Frontend ESLint check (`-Action lint`): **PASS** (zero errors, zero warnings).
  - Node network denial guard (`-Action guard`): **PASS**.

## Completed Milestones
- [x] Issue #7: Protected all 7 file write functions in `output_generator.py` with `try/except (IOError, OSError, PermissionError)`.
- [x] Issue #8: Resolved voting/consensus edge cases (`BinaryVote`, `ConfidenceVote`, `ErrorResult`, tie-breaker, low participation, emergency mediator decisions).
- [x] PR #34 Phase 0–1 Offline Qualification (`ae00ecb`): Repaired all 10 review findings, established `LazyClient`, lazy `get_nlp_spacy()` fallbacks, `conftest.py` isolation, and `tools/run-offline.*` / `tools/run-frontend-offline.*` socket denial guards.
- [x] Issue #35 Phase 2 Quarantine Resolution: Unquarantined and repaired all 24 legacy tests in `test_advanced_prompting.py` (48/48 passing offline).
- [x] Antigravity 2.0+ Custom Agent Team & Coordination Rules: Created 9 custom agents in `.agents/agents/*/agent.md`, 5 modular rules in `.agents/rules/` (mirrored in `.agent/rules/`), `.agents/doc/antigravity-agent-team.md`, root `GEMINI.md` and `AGENTS.md`, and `memory-bank/`.
- [x] Track 1 (Issue #35 Phases 3–5 & Issue #21):
  - Phase 3: Created `execution_modes.py` with Pydantic v2 schemas (`RunRequest`, `RunResult`, `RunEvent`, `RunDiagnostics`), `CAPABILITY_REGISTRY`, and dispatcher.
  - Phase 4: Extended `schema.py` with `Query.execution_modes`, `Mutation.run_execution`, `Mutation.run_advanced_pipeline`, and Strawberry diagnostic/event/output types.
  - Phase 5: Created `StructuredOutputRenderer.tsx`, `ReasoningDiagnosticsPanel.tsx`, `AssistantConfigPanel.tsx` mode selection, and `useRunExecution.ts`.
  - Frontend Lint: Cleaned all ESLint errors/warnings (`-Action lint` passes cleanly).

## Open Work Items
- [ ] Issue #36 (Tree of Thoughts): Implement genuine Tree-of-Thoughts search after shared provider/search contracts are finalized.
