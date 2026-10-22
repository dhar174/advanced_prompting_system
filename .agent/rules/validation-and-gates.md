---
trigger: model_decision
description: "Mandatory offline qualification test commands, socket-denial guard checks, backend pytest suites, and frontend Vitest/ESLint/Vite gates."
---

# Validation & Release Gates Rules

This document defines the strict offline verification criteria, test execution suites, and quality standards required for all code changes in `advanced_prompting_system`.

---

## 1. Backend Offline Qualification Suite (No API Keys or Network Permitted)

All backend tests MUST be run through the repository's offline wrapper (`./tools/run-offline.ps1` on Windows or `./tools/run-offline.sh` on Linux/macOS). Bare `pytest` intentionally fails with `pytest.UsageError` because `sys._aps_network_guard` is required before collection.

### Full Backend Regression Gate
```powershell
./tools/run-offline.ps1 -PytestArgs '-q'
```
- **Required Baseline**: **148 passed / 0 failed / 0 skipped** (~25 seconds).
- Strips all `*_API_KEY`, `*_ACCESS_TOKEN`, `*_AUTH_TOKEN`, `OPENAI_ADMIN_KEY`, and `HF_TOKEN` variables.
- Preloads `tools/offline/sitecustomize.py` (`sys._aps_network_guard = True`), raising `NetworkForbidden(BaseException)` on any external DNS, TCP, or UDP socket attempt.
- Enforces `forbid_unconfigured_provider` and `isolated_runtime_directory` (`tmp_path`) via `conftest.py`.

### Per-Subsystem Backend Suites
| Test Suite | Command | Focus & Invariants Verified |
| :--- | :--- | :--- |
| **Core Prompting & Adapters** | `./tools/run-offline.ps1 -PytestArgs 'test_advanced_prompting.py','test_finalize_step_output.py','-v'` | All 48 unquarantined tests in `TestParseResponse`, `TestAdvancedPromptEngineer`, `TestAdvancedPromptingAdapterMethods`, `TestFinalizePlanStepOutput`, plus `test_finalize_step_output.py`. |
| **PR #34 Regressions & Voting** | `./tools/run-offline.ps1 -PytestArgs 'test_pr34_regressions.py','test_voting_semantics.py','-v'` | `PlanStep` real adapter binding, partial-majority definition, continuation vote errors, low-participation finalization, unanimous YES / majority NO / tie / all-error / weak YES production branches, `BinaryVote(vote=False)` truthiness. |
| **Memory & Message Safety** | `./tools/run-offline.ps1 -PytestArgs 'test_memory_management.py','test_message_handling.py','-v'` | `ConversationMemory` typed models (`ArgumentItem`, `DecisionItem`, `ToDoItem`), legacy round-trips, compaction, `normalize_message` rejecting `ErrorResult`. |
| **Offline Isolation & Imports** | `./tools/run-offline.ps1 -PytestArgs 'test_offline_runtime.py','test_import_integrity.py','-v'` | Fresh-process import purity, network denial guard, `LazyClient` deferred resolution, `get_nlp_spacy()` caching/fallback, optional NLTK fallback, `pydantic.ValidationError` non-shadowing. |
| **GraphQL, Retrieval & Smoke** | `./tools/run-offline.ps1 -PytestArgs 'test_graphql_contract.py','test_retrieval.py','test_smoke.py','-v'` | Strawberry GraphQL `runConversation` and `submitFeedback` schema contracts, Flask `/graphql` startup `{ hello }`, RAG retrieval, and smoke suite. |

---

## 2. Frontend Offline Verification Gates (`collaborative-assistant-frontend/`)

All frontend verification commands MUST run through `./tools/run-frontend-offline.ps1` (or `./tools/run-frontend-offline.sh`), which preloads `tools/offline/node-network-guard.cjs`.

### Gate 1: Node Network Denial Probe
```powershell
./tools/run-frontend-offline.ps1 -Action guard
```
- **Required**: Exit code `0` (`guard-probe.cjs` verifies external connections are blocked while loopback works).

### Gate 2: Vitest Component & Hook Suite
```powershell
./tools/run-frontend-offline.ps1 -Action test
```
- **Required Baseline**: **111 passed / 0 failed** across 10 test files (`ConversationPage.test.tsx`, `AssistantConfigPanel.test.tsx`, `ChatInput.test.tsx`, `MessageBubble.test.tsx`, `useRunConversation.test.ts`, `Button.test.tsx`, `Card.test.tsx`, `Input.test.tsx`, `Label.test.tsx`, `Textarea.test.tsx`).

### Gate 3: TypeScript & Vite Production Build
```powershell
./tools/run-frontend-offline.ps1 -Action build
```
- **Required**: `tsc -b` and `vite build` exit `0` with zero type errors.

### Gate 4: ESLint Check
```powershell
./tools/run-frontend-offline.ps1 -Action lint
```
- Tracks frontend lint compliance (13 historical lint errors and 1 Fast Refresh warning documented in `docs/pr34-offline-qualification.md`). Any touched frontend file must be lint-clean.

---

## 3. Git & Artifact Hygiene Gate

Before completing any task, verify repository cleanliness:
```powershell
git status --short
git -c core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol diff --check
```
- Ensure zero untracked `debug_log_complexity_*.txt`, `logging_debug_log_complexity_*.txt`, `.qualification/` artifacts, or temporary output files are staged or left dirty.
- Mirror parity: Whenever a rule in `.agents/rules/` is updated, keep its mirror in `.agent/rules/` identical.
