# Project Progress & Baselines (`advanced_prompting_system`)

## Verified Offline Baselines
- **Backend Offline Test Suite (`./tools/run-offline.ps1 -PytestArgs '-q'`)**:
  - **148 passed, 0 failed, 0 skipped**
  - Includes all 48 tests in `test_advanced_prompting.py` (unquarantined in Issue #35 Phase 2), `test_pr34_regressions.py`, `test_voting_semantics.py`, `test_memory_management.py`, `test_message_handling.py`, `test_offline_runtime.py`, `test_import_integrity.py`, `test_graphql_contract.py`, `test_retrieval.py`, `test_smoke.py`, and `test_finalize_step_output.py`.
- **Frontend Offline Test Suite (`./tools/run-frontend-offline.ps1 -Action test`)**:
  - **111 passed, 0 failed** across 10 test files.
  - Production TypeScript & Vite bundle build (`-Action build`): **PASS**.
  - Node network denial guard (`-Action guard`): **PASS**.

## Completed Milestones
- [x] Issue #7: Protected all 7 file write functions in `output_generator.py` with `try/except (IOError, OSError, PermissionError)`.
- [x] Issue #8: Resolved voting/consensus edge cases (`BinaryVote`, `ConfidenceVote`, `ErrorResult`, tie-breaker, low participation, emergency mediator decisions).
- [x] PR #34 Phase 0–1 Offline Qualification (`ae00ecb`): Repaired all 10 review findings, established `LazyClient`, lazy `get_nlp_spacy()` fallbacks, `conftest.py` isolation, and `tools/run-offline.*` / `tools/run-frontend-offline.*` socket denial guards.
- [x] Issue #35 Phase 2 Quarantine Resolution: Unquarantined and repaired all 24 legacy tests in `test_advanced_prompting.py` (48/48 passing offline).
- [x] Antigravity 2.0+ Custom Agent Team & Coordination Rules: Created 9 custom agents in `.agents/agents/*/agent.md`, 5 modular rules in `.agents/rules/` (mirrored in `.agent/rules/`), `.agents/doc/antigravity-agent-team.md`, root `GEMINI.md` and `AGENTS.md`, and `memory-bank/`.

## Open Work Items
- [ ] Issue #35 (Frontend Lint Debt): Resolve 13 ESLint errors and 1 Fast Refresh warning in `collaborative-assistant-frontend/`.
- [ ] Issue #35 (Phase 3 API/UI Integration): Shared run-level configuration, execution-mode registry, and browser path matrix.
- [ ] Issue #36 (Tree of Thoughts): Implement genuine Tree-of-Thoughts search after shared provider/search contracts are finalized.
