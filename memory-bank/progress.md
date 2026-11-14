# Project Progress & Baselines (`advanced_prompting_system`)

## Verified Offline Baselines
- **Backend + Integration + Browser E2E Suite (`./tools/run-offline.ps1 -PytestArgs '-q'`)**:
  - **243 passed, 0 failed, 0 skipped** (43.38s on Python 3.12.2 Windows)
  - Includes 9 unmocked HTTP `/graphql` integration tests (`test_graphql_integration.py`), 6 offline Chromium Playwright E2E tests (`test_browser_e2e.py`), plus full unit/contract/regression suites (`test_advanced_prompting.py`, `test_execution_modes.py`, `test_graphql_contract.py`, `test_issue35_pr1_regressions.py`, `test_pr34_regressions.py`, `test_voting_semantics.py`, `test_memory_management.py`, `test_message_handling.py`, `test_offline_runtime.py`, `test_import_integrity.py`, `test_retrieval.py`, `test_smoke.py`, etc.).
- **Frontend Offline Test Suite (`./tools/run-frontend-offline.ps1 -Action test`)**:
  - **126 passed, 0 failed** across 15 Vitest test files.
  - Production TypeScript & Vite bundle build (`-Action build`): **PASS**.
  - Frontend ESLint check (`-Action lint`): **PASS**.
  - Node network denial guard (`-Action guard`): **PASS**.
- **CI Workflow**: `.github/workflows/test.yml` runs `backend-test` (`--ignore=test_browser_e2e.py -q` on Python 3.11/3.12), `frontend-test` (Node 22), and `browser-e2e` (Chromium + `test_browser_e2e.py` + `py_compile` + ledger artifact upload).

## Completed Milestones
- [x] Issue #7: Protected all 7 file write functions in `output_generator.py` with `try/except (IOError, OSError, PermissionError)`.
- [x] Issue #8: Resolved voting/consensus edge cases (`BinaryVote`, `ConfidenceVote`, `ErrorResult`, tie-breaker, low participation, emergency mediator decisions).
- [x] PR #34 Phase 0–1 Offline Qualification (`ae00ecb`): Repaired all 10 review findings, established `LazyClient`, lazy `get_nlp_spacy()` fallbacks, `conftest.py` isolation, and `tools/run-offline.*` / `tools/run-frontend-offline.*` socket denial guards.
- [x] Issue #35 Phase 2 Quarantine Resolution: Unquarantined and repaired all 24 legacy tests in `test_advanced_prompting.py` (48/48 passing offline).
- [x] Track 1 & PR #38 Hardening (Issue #35 Phases 3–5 & Issue #21): Added `execution_modes.py`, Strawberry types, frontend `ReasoningDiagnosticsPanel.tsx` / `StructuredOutputRenderer.tsx` / `useRunExecution.ts`, and security hardening.
- [x] Issue #35 Stacked Resolution (PR #39, PR #40, PR 3; resolving #9, #10, #11): Unmocked GraphQL HTTP integration (`test_graphql_integration.py`), offline Playwright Chromium E2E (`test_browser_e2e.py`), production `Dockerfile` / `wsgi.py` / `.env.example` / `pyproject.toml`, CI `browser-e2e` job, rewritten `README.md`, and `docs/issue35-qualification.md`. (Docker image build deferred locally due to host Docker Desktop daemon not running.)

## Open Work Items
- [ ] Issue #36 (Tree of Thoughts): Keep `tree_of_thought` gated (`enabled=False`) until genuine Tree-of-Thoughts search is implemented.
- [ ] Issues #20 & #22: Remain open outside Issue #35 closure scope.
