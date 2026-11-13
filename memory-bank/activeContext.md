# Active Context (`advanced_prompting_system`)

## Current Focus
- **Issue #35 Completion (Stacked PRs #39, #40, PR 3)**:
  - **PR #39 (`fix/issue-35-engine-artifacts-provider` -> `main`)**: Engine retrieval propagation, `provider_client` model policy & budget/cancellation scoping, per-run `.aps_artifacts/<run_id>/` isolation.
  - **PR #40 (`feat/issue-35-graphql-security-react-lifecycle` -> PR #39)**: Unified `execution_modes` (`collaborative`, `advanced_pipeline`, gated `tree_of_thought`), GraphQL security hardening (loopback CORS allowlist, optional Bearer auth, per-IP rate limiting, query depth/token limiters, error sanitization), and React 19 UI lifecycle (`useRunExecution`, `ReasoningDiagnosticsPanel`, abort-wait isolation, draft restoration).
  - **PR 3 (`feat/issue-35-browser-e2e-packaging-docs` -> PR #40)**: Unmocked HTTP `/graphql` -> real orchestration integration suite (`test_graphql_integration.py`, `tools/e2e/`), offline Playwright Chromium E2E suite (`test_browser_e2e.py`), CI `browser-e2e` job (`.github/workflows/test.yml`), Cloud Run-compliant multi-stage `Dockerfile`, `.dockerignore`, `wsgi.py`, `.env.example`, updated `pyproject.toml`, `README.md`, `docs/issue35-qualification.md`, and `docs/offline-test-inventory.md`.
- **Verified Offline Qualification Baseline (Post Issue #35 PR 3)**:
  - Backend + HTTP GraphQL integration + Playwright Chromium E2E (`./tools/run-offline.ps1 -PytestArgs '-q'`): **243 passed / 0 failed** (43.38s).
  - Frontend offline Vitest suite (`./tools/run-frontend-offline.ps1 -Action test`): **126 passed / 0 failed** across 15 Vitest files (12.27s).
  - Frontend offline `guard`, `lint`, and `build`: **PASS** (`index.css` external Google Fonts `@import` removed for zero external browser requests offline).

## Recent Verified Architectural Changes
1. **Unmocked Pipeline & Browser Bug Fixes (`advanced_prompting.py`, `schema.py`, `index.css`)**:
   - Guarded `AdvancedPromptEngineer.main()` against `<= 0.0` complexity / empty `Plan` when optional spaCy `en_core_web_sm` is absent; fixed `judge_step_completion` candidate step evaluation, `best_interaction.answer` copy, and `self.judge_final_answer(task_object, main_interaction)`.
   - Generalized `schema._sanitize_error_message(exc: Exception | str)` and applied it to `run_result.error.error_message`.
   - Removed external `fonts.googleapis.com` `@import` from `collaborative-assistant-frontend/src/index.css`.
2. **Offline E2E & WSGI Packaging (`tools/e2e/`, `test_browser_e2e.py`, `wsgi.py`, `Dockerfile`)**:
   - `FakeProviderClient` + `ProviderCallLedger` + `OfflineE2EServer` (`127.0.0.1`) drive real Flask `/graphql` + built Vite SPA inside headless Chromium with fail-closed `page.route` blocking.
   - `wsgi:app` binds `0.0.0.0:${PORT:-8000}` (`waitress`/Flask fallback locally, `gunicorn` in multi-stage `Dockerfile` with `PORT=8080`, non-root `appuser`, `.[server]`).

## Next Priorities
1. **Issue #36 (`tree_of_thought`)**:
   - Remains strictly gated (`enabled=False`) pending genuine Tree-of-Thoughts implementation; Issues `#20` and `#22` also remain open.
