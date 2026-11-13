# Project: Track 1 — End-to-End Execution Modes & UI Integration

## Architecture
- **Layer 1: Unified Run Contracts & Execution Engine (`execution_modes.py`)**:
  - Pydantic v2 schemas: `RunRequest`, `RunResult`, `RunEvent`, `RunDiagnostics`, `PlanStepDiagnostic`, `SubtaskDiagnostic`, `ReflectionDiagnostic`, `AgentAttribution`, `ExecutionModeCapability`.
  - Capability Registry: `CAPABILITY_REGISTRY` mapping mode capabilities for `collaborative`, `advanced_pipeline`, and `tree_of_thought` (gated Issue #36).
  - Dispatcher: `execute_run(request: RunRequest) -> RunResult` routing to specialized handlers with lazy engine imports and fail-closed gating.
- **Layer 2: GraphQL API (`schema.py`)**:
  - Strawberry GraphQL schema exposing native types without top-level circular imports.
  - Queries: `Query.hello`, `Query.execution_modes`.
  - Mutations: `Mutation.run_conversation`, `Mutation.submit_feedback`, `Mutation.run_execution`, `Mutation.run_advanced_pipeline`, `Mutation.run_reasoning_task`.
- **Layer 3: React 19 Frontend (`collaborative-assistant-frontend/src/`)**:
  - Apollo Client operations and `useRunExecution` hook.
  - `AssistantConfigPanel`: execution mode selector (`collaborative`, `advanced_pipeline`, disabled `tree_of_thought` with Issue #36 badge) and dynamic controls with backward-compatible defaults.
  - `ReasoningDiagnosticsPanel`: multi-tab dashboard for complexity, budgets, plan steps, subtasks, reflections with rewards, agent attributions, and chronological events.
  - `StructuredOutputRenderer`: OutputType-aware formatting for code, JSON, CSV (table/raw), Markdown, and plain text fallback.
- **Layer 4: Verification & Qualification Harness**:
  - Backend: `tools/run-offline.ps1` with socket-denial audit hooks (`sitecustomize.py`) and provider guard (`conftest.py`).
  - Frontend: `tools/run-frontend-offline.ps1` with Node network denial guard (`node-network-guard.cjs`).

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | `RunRequest` & Validators | Strongly typed Pydantic v2 request schema with bidirectional prompt/conversation validators | M1 | survey |
| 2 | Diagnostic & Telemetry Schemas | `SubtaskDiagnostic`, `PlanStepDiagnostic`, `ReflectionDiagnostic`, `AgentAttribution`, `RunDiagnostics`, `RunEvent` | M1 | survey |
| 3 | `RunResult` & Capability Models | Unified result schema and `ExecutionModeCapability` metadata | M1 | survey |
| 4 | `CAPABILITY_REGISTRY` | Mode capability registry defining collaborative, advanced_pipeline, and gated tree_of_thought | M1 | survey |
| 5 | Gated `tree_of_thought` Dispatcher | Deterministic gated return with `ModeGatedError` and `mode_gated` event, zero provider/network calls | M1 | survey |
| 6 | Collaborative Mode Dispatcher | Wraps `run_conversation`, normalizes history, extracts speaker attributions and OutputType | M1 | survey |
| 7 | Advanced Pipeline Dispatcher | Wraps `AdvancedPromptEngineer.main`, extracts complexity, plan steps, subtasks, reflections, and events | M1 | survey |
| 8 | Backend Unit Test Suite | Authoritative `test_execution_modes.py` with >=20 tests verifying validation, registry, gating, and extraction | M1 | survey |
| 9 | Strawberry GraphQL Schema Extensions | Extend `schema.py` with native Strawberry diagnostic, event, and input/output types | M2 | survey |
| 10 | `Query.execution_modes` | Expose registered capabilities and gating status in GraphQL query | M2 | survey |
| 11 | `Mutation.run_execution` | Multi-mode execution mutation accepting `RunExecutionInput` and returning `RunExecutionResponse` | M2 | survey |
| 12 | `Mutation.run_advanced_pipeline` | Convenience mutation for direct standalone pipeline execution | M2 | survey |
| 13 | `Mutation.run_reasoning_task` & Telemetry | Strategy-based mutation returning steps, reflections, and `TelemetryMetrics` (R1/R2) | M2 | survey |
| 14 | GraphQL Contract Tests | Extend `test_graphql_contract.py` covering new query and mutations while ensuring backward compatibility | M2 | survey |
| 15 | Frontend TypeScript Types & Operations | `src/graphql/graphqlTypes.ts` and `src/graphql/executionOperations.ts` | M3 | survey |
| 16 | Apollo Hook `useRunExecution` | Custom React hook for unified execution operations | M3 | survey |
| 17 | `AssistantConfigPanel` Mode Selector | UI mode toggle with dynamic controls and disabled Tree of Thoughts option | M3 | survey |
| 18 | `ReasoningDiagnosticsPanel` | Dashboard visualizing complexity, budgets, plan steps, reflections, attributions, and events | M3 | survey |
| 19 | `StructuredOutputRenderer` | Syntax-highlighted code, pretty JSON, dual-view CSV, Markdown, and plain text fallback | M3 | survey |
| 20 | Frontend Integration & Lint Cleanup | Integrate into `FeedbackDisplay` and `ConversationPage`, resolve all ESLint issues | M3 | survey |
| 21 | Frontend Vitest Test Suites | Comprehensive unit & integration tests across new components, mode switching, and hooks | M3 | survey |
| 22 | Full Offline Qualification & Adversarial Hardening | Pass 100% backend (>=170) and frontend (>=140) offline tests with zero network calls and adversarial verification | M4 | survey |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1 | Phase 3 Shared Run Contracts & Capability Registry | Create `execution_modes.py` and `test_execution_modes.py` | none | IN_PROGRESS |
| 2 | Phase 4 GraphQL API & Schema Extensions | Extend `schema.py` with native types, queries, and mutations; update `test_graphql_contract.py` | M1 | PLANNED |
| 3 | Phase 4 & 5 Frontend Mode Selection, Diagnostics & OutputType | Extend frontend types, operations, hooks, components (`AssistantConfigPanel`, `ReasoningDiagnosticsPanel`, `StructuredOutputRenderer`), page integration, lint fixes, and Vitest suites | M2 | PLANNED |
| 4 | Final Milestone: Full E2E Offline Qualification & Coverage Hardening | Run complete guarded offline test suites (backend & frontend), adversarial verification, and audit gate | M1, M2, M3 | PLANNED |

## Interface Contracts

### Layer 1 (`execution_modes.py`) ↔ Layer 2 (`schema.py`)
- `schema.py` does NOT import `execution_modes.py` at top-level.
- In resolver methods:
  - `Query.execution_modes`: `from execution_modes import CAPABILITY_REGISTRY` -> returns `List[ExecutionModeCapabilityType]`.
  - `Mutation.run_execution`: `from execution_modes import execute_run, RunRequest` -> constructs `RunRequest` and calls `execute_run(request)` -> maps `RunResult` to `RunExecutionResponse`.
  - `Mutation.run_advanced_pipeline`: calls `execute_run(RunRequest(mode="advanced_pipeline", ...))` -> maps `RunResult` to `RunExecutionResponse`.
  - `Mutation.run_reasoning_task`: calls `execute_run` or `AdvancedPromptEngineer` with chosen strategy -> returns `ReasoningTaskResponse` with `TelemetryMetrics`.

### Layer 2 (`schema.py`) ↔ Layer 3 (`collaborative-assistant-frontend/`)
- GraphQL Query: `executionModes { id label description enabled gatedIssue supportsMultiAgent supportsStepPlanning }`.
- GraphQL Mutation: `runExecution(input: RunExecutionInput!) -> RunExecutionResponse!`.
- GraphQL Mutation: `runAdvancedPipeline(taskPrompt: String!, ...) -> RunExecutionResponse!`.
- GraphQL Mutation: `runReasoningTask(input: TaskInput!) -> ReasoningTaskResponse!`.
- Backward compatibility: `runConversation(...) -> RunConversationResponse!`, `submitFeedback(...) -> FeedbackResponse!`, `hello -> String!`.

### Component Props Contracts (`collaborative-assistant-frontend/`)
- `AssistantConfigPanelProps`:
  - Existing 9 props remain required or preserved.
  - New props optional with defaults: `executionMode?: ExecutionModeId` (default `'collaborative'`), `onExecutionModeChange?: (mode: ExecutionModeId) => void`, `maxSteps?: number`, `onMaxStepsChange?: (val: number) => void`, `agentCount?: number`, `onAgentCountChange?: (val: number) => void`.
- `FeedbackDisplayProps`:
  - Existing 3 props: `currentQuestions`, `finalOutput`, `runConversationLoading`.
  - New optional prop: `outputType?: OutputTypeInfo`.
  - Renders `<h3 ...>Final Output:</h3>` wrapping `StructuredOutputRenderer`.

## Code Layout
- Backend Source:
  - `execution_modes.py` (Repository root)
  - `schema.py` (Repository root)
  - `conversation_manager.py` (Repository root)
  - `advanced_prompting.py` (Repository root)
- Backend Tests:
  - `test_execution_modes.py` (Repository root)
  - `test_graphql_contract.py` (Repository root)
- Frontend Source:
  - `collaborative-assistant-frontend/src/graphql/graphqlTypes.ts`
  - `collaborative-assistant-frontend/src/graphql/executionOperations.ts`
  - `collaborative-assistant-frontend/src/hooks/useRunExecution.ts`
  - `collaborative-assistant-frontend/src/components/AssistantConfigPanel.tsx`
  - `collaborative-assistant-frontend/src/components/ReasoningDiagnosticsPanel.tsx`
  - `collaborative-assistant-frontend/src/components/StructuredOutputRenderer.tsx`
  - `collaborative-assistant-frontend/src/components/FeedbackDisplay.tsx`
  - `collaborative-assistant-frontend/src/pages/ConversationPage.tsx`
  - `collaborative-assistant-frontend/src/components/ui/Button.tsx` (ESLint fix)
- Frontend Tests:
  - `collaborative-assistant-frontend/src/components/ReasoningDiagnosticsPanel.test.tsx`
  - `collaborative-assistant-frontend/src/components/StructuredOutputRenderer.test.tsx`
  - `collaborative-assistant-frontend/src/components/AssistantConfigPanel.test.tsx`
  - `collaborative-assistant-frontend/src/pages/ConversationPage.test.tsx`
  - `collaborative-assistant-frontend/src/hooks/useRunExecution.test.ts`
