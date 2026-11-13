# Issue #35 Offline Qualification & Release Readiness Report

## Overview

This document summarizes the verification architecture, real integration defects uncovered and fixed, and offline qualification baselines across the stacked pull requests resolving **Issue #35** and its completed dependencies (**#9**, **#10**, **#11**):

1. **PR #39 (`fix/issue-35-engine-artifacts-provider` -> `main`)**:
   - Engine retrieval propagation, `provider_client` model policy & budget/cancellation hardening, per-run `.aps_artifacts/<run_id>/` isolation, `conftest.py` cleanup.
2. **PR #40 (`feat/issue-35-graphql-security-react-lifecycle` -> PR #39)**:
   - Unified `execution_modes` dispatcher (`collaborative`, `advanced_pipeline`, gated `tree_of_thought`), Strawberry GraphQL security hardening (loopback CORS allowlist, optional Bearer auth, per-IP rate limiting, query depth & token limiters, sanitized error reporting), and React 19 UI lifecycle (`useRunExecution`, `ReasoningDiagnosticsPanel`, abort-wait isolation, draft restoration).
3. **PR 3 (`feat/issue-35-browser-e2e-packaging-docs` -> PR #40)**:
   - Unmocked HTTP `/graphql` -> real orchestration integration suite (`test_graphql_integration.py`, `tools/e2e/provider_fixture.py`, `tools/e2e/server_runner.py`).
   - Offline Chromium Playwright E2E suite (`test_browser_e2e.py`) with fail-closed non-loopback route blocking and `ProviderCallLedger` audit trail under `.qualification/logs/`.
   - CI `browser-e2e` job in [`.github/workflows/test.yml`](../.github/workflows/test.yml), multi-stage Cloud Run-compliant [`Dockerfile`](../Dockerfile), [`.dockerignore`](../.dockerignore), [`wsgi.py`](../wsgi.py), [`.env.example`](../.env.example), [`pyproject.toml`](../pyproject.toml), updated [`README.md`](../README.md), and [`docs/offline-test-inventory.md`](offline-test-inventory.md).

---

## Real Defects Uncovered & Fixed During Integration / Browser Testing

Running unmocked Flask HTTP `/graphql` and real Chromium browser flows against the built Vite bundle revealed the following production defects, all resolved and regression-tested:

1. **Empty `Plan` / Zero-Complexity Pipeline Abort in `AdvancedPromptEngineer.main()` (`advanced_prompting.py`)**:
   - When optional spaCy `en_core_web_sm` is absent, complexity assessment could produce `<= 0.0` with zero generated plan steps, causing `main()` to return early without reasoning. Added safe floor/fallback 1-step `Plan` execution.
2. **Pre-Merge Step Completion Evaluation & Final Answer Propagation (`advanced_prompting.py`)**:
   - `main()` previously evaluated `judge_step_completion` against empty `steps_this_planstep` before `best_interaction.steps` merged, and passed raw `str` `task` into `judge_final_answer` instead of `Task` `task_object` (`AttributeError: 'str' object has no attribute 'output_type'`). Fixed candidate step evaluation, answer copy, and `task_object` forwarding.
3. **Unredacted Execution Error Propagation in GraphQL (`schema.py`)**:
   - `_build_run_execution_response` previously forwarded `run_result.error.error_message` without routing through `_sanitize_error_message()`. Extended `_sanitize_error_message()` to accept `Exception | str` and applied it consistently to execution errors.
4. **External CDN Request on Offline Page Load (`collaborative-assistant-frontend/src/index.css`)**:
   - Playwright's fail-closed browser route guard caught an external `https://fonts.googleapis.com/...` CSS `@import` firing on every page load offline (and emitting `ERR_BLOCKED_BY_CLIENT` console errors). Removed the external font import so the built SPA makes zero outbound external requests offline.

---

## Offline Qualification Harness & Isolation Invariants

- **Python Socket & Env Guard ([`tools/offline/sitecustomize.py`](../tools/offline/sitecustomize.py))**: Strips provider API keys/tokens, permits only loopback (`localhost` / `127.0.0.1` / `::1`) sockets required by local HTTP/Playwright drivers, and raises `NetworkForbidden(BaseException)` on any non-loopback DNS/TCP/UDP attempt.
- **Node Socket Guard ([`tools/offline/node-network-guard.cjs`](../tools/offline/node-network-guard.cjs))**: Verified via `./tools/run-frontend-offline.ps1 -Action guard`.
- **Browser Loopback Route Guard ([`test_browser_e2e.py`](../test_browser_e2e.py))**: Intercepts `page.route("**/*", ...)` in Chromium and aborts + fails the test on any non-loopback request while asserting zero `pageerror` or `console` errors on positive flows.
- **Deterministic Provider Double & Ledger ([`tools/e2e/provider_fixture.py`](../tools/e2e/provider_fixture.py))**: Implements OpenAI-compatible `chat.completions.create`, `beta.chat.completions.parse`, and `embeddings.create`, logs every invocation to `ProviderCallLedger` (saved to `.qualification/logs/browser_e2e_latest_ledger.json`), and simulates outage, all-error voting, XSS payload escaping, empty model output, and slow execution scenarios via `X-APS-E2E-Scenario`.

---

## Verified Qualification Baselines

### Backend, Integration & Browser E2E (Python 3.12.2, Windows)

| Command | Result |
| :--- | :--- |
| `./tools/run-offline.ps1 -PytestArgs '-q'` | **243 passed, 0 failed** (43.64s) |
| `./tools/run-offline.ps1 -PytestArgs 'test_graphql_integration.py','-v'` | **9 passed, 0 failed** (7.99s) |
| `./tools/run-offline.ps1 -PytestArgs 'test_browser_e2e.py','-v'` | **6 passed, 0 failed** (22.40s) |
| `.qualification\venv\Scripts\python.exe -m py_compile wsgi.py tools/e2e/provider_fixture.py tools/e2e/server_runner.py` | **Exited 0** (`PY_COMPILE_OK`) |

### Frontend (Node 22 / Vitest 3.1.4 / Vite 6.3.5 / ESLint)

| Command | Result |
| :--- | :--- |
| `./tools/run-frontend-offline.ps1 -Action guard` | Passed (`Node network denial verified`) |
| `./tools/run-frontend-offline.ps1 -Action lint` | Passed (exit code 0) |
| `./tools/run-frontend-offline.ps1 -Action test` | **15 test files, 126 passed, 0 failed** (12.10s) |
| `./tools/run-frontend-offline.ps1 -Action build` | Built `collaborative-assistant-frontend/dist/` cleanly (5.44s) |

---

## Explicit Scope Boundaries & Unexecuted Environment Checks

- **Kept Intentionally Open / Gated**:
  - **Issue #36 (`tree_of_thought`)**: Remains disabled in backend capability discovery (`enabled=False`) and UI (`<option disabled>`), verified with zero provider calls (`test_http_graphql_tree_of_thought_gated_zero_provider_calls`, `test_browser_e2e_tree_of_thought_disabled_in_ui_zero_provider_calls`).
  - **Issues #20 and #22**: Not closed by this stack.
- **Docker Daemon Unavailable Locally**: `docker info` failed to connect to `//./pipe/dockerDesktopLinuxEngine`; therefore [`Dockerfile`](../Dockerfile) was validated structurally for multi-stage Cloud Run compliance (`PORT=8080`, `0.0.0.0`, non-root `appuser`, `.[server]` without dev deps), not locally image-built.
- **Live OpenAI Smoke Run**: Intentionally excluded from offline CI (`OFFLINE_MODE=1` and stripped credentials). Owner may perform an optional final post-merge smoke run with a live `.env` key.
