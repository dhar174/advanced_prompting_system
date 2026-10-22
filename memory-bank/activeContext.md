# Active Context (`advanced_prompting_system`)

## Current Focus
- **Track 1: End-to-End Execution Modes & UI Integration (Issue #35 Phases 3–5 & Issue #21)**:
  - Phase 3: Created `execution_modes.py` with strongly typed Pydantic v2 schemas (`RunRequest`, `RunResult`, `RunEvent`, `RunDiagnostics`, `PlanStepDiagnostic`, `SubtaskDiagnostic`, `ReflectionDiagnostic`, `AgentAttribution`), `CAPABILITY_REGISTRY`, and dispatcher.
  - Phase 4: Extended `schema.py` with Strawberry GraphQL types, `Query.execution_modes`, `Mutation.run_execution`, and `Mutation.run_advanced_pipeline`.
  - Phase 5: Built frontend components `StructuredOutputRenderer.tsx`, `ReasoningDiagnosticsPanel.tsx`, `AssistantConfigPanel.tsx` (execution mode selector), and custom hook `useRunExecution.ts`.
- **Offline Qualification Baseline (Track 1 Certified)**:
  - Backend offline suite (`./tools/run-offline.ps1 -PytestArgs '-q'`): **206 passed / 0 failed / 0 skipped** (up from 148 baseline, +58 new unit and contract tests in `test_execution_modes.py`, `test_empirical_challenger_2.py`, and `test_graphql_contract.py`).
  - Frontend offline suite (`./tools/run-frontend-offline.ps1 -Action test`): **124 passed / 0 failed** across 13 Vitest files (up from 111 baseline, +13 new unit tests in `StructuredOutputRenderer.test.tsx`, `ReasoningDiagnosticsPanel.test.tsx`, and `useRunExecution.test.ts`).
  - Frontend offline bundle build (`-Action build`): **PASS** (zero TypeScript or Vite bundle errors).
  - Frontend offline lint (`-Action lint`): **PASS** (zero ESLint errors or warnings).
  - Frontend offline guard (`-Action guard`): **PASS** (Node outbound network denial verified).

## Recent Verified Architectural Changes
1. **Track 1 Execution Modes & GraphQL Telemetry (`execution_modes.py`, `schema.py`)**:
   - `RunRequest` provides bidirectional synchronization between `task_prompt` and polymorphic `conversation` inputs.
   - Fail-closed gating for `tree_of_thought` returns deterministic `RunResult(status="gated")` without calling LLM providers.
   - `schema.py` exposes `Query.execution_modes`, `Mutation.run_execution`, and `Mutation.run_advanced_pipeline` while preserving `run_conversation` and `submit_feedback` 100% backward-compatible.
2. **Frontend Mode Switching & Observability (`collaborative-assistant-frontend/`)**:
   - `AssistantConfigPanel.tsx` dynamically adapts controls between Collaborative Debate (`collaborative`), Standalone Pipeline (`advanced_pipeline`), and disabled `tree_of_thought` (Issue #36 badge).
   - `ReasoningDiagnosticsPanel.tsx` visualizes complexity scores, step budgets, reflections, rewards, plan subtasks, agent attributions, and event logs.
   - `StructuredOutputRenderer.tsx` formats output by `OutputType` (code containers, JSON pretty-print, CSV table/raw preview, copy to clipboard).

## Next Priorities
1. **Issue #36**:
   - Implement genuine Tree-of-Thoughts (`ToT`) search with bounded offline-testable provider contracts.
