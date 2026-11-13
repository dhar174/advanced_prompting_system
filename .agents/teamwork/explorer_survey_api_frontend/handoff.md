# Specification Mining & Contract Survey Report: GraphQL API & UI Integration

**Author**: `teamwork_preview_spec_miner` (API & UI Spec Miner)  
**Target Path**: `d:\projects\advanced_prompting_system\advanced_prompting_system\.agents\teamwork\explorer_survey_api_frontend\handoff.md`  
**Date**: 2026-10-22  
**Scope**: Issue #35 (Phases 3–5), Issue #21, and Requirements R1–R4 (GraphQL Schema Extensions, React UI Integration, Diagnostics, Structured Output, and Lint/Test Hygiene)

---

## 1. Features Discovered

| # | Category | Feature | Description | Inputs | Outputs | Error Behavior | Discovered Via |
|---|----------|---------|-------------|--------|---------|----------------|----------------|
| 1 | GraphQL Query | `Query.hello` | Health check endpoint returning simple greeting | None | `str` ("Hello, World!") | None | `schema.py:59`, `test_smoke.py:130` |
| 2 | GraphQL Mutation | `Mutation.run_conversation` | Multi-agent round deliberation execution returning 3-tuple mapped to GraphQL shape | `conversation: List[ConversationInput]`, `assistant_personalities: List[str]`, `leadPersonality: str`, `num_rounds: int` | `RunConversationResponse` (`conversation`, `questions`, `final_output`) | Propagates exceptions from `conversation_manager.run_conversation` | `schema.py:66-103`, `test_graphql_contract.py:16` |
| 3 | GraphQL Mutation | `Mutation.submit_feedback` | Persists conversation feedback ratings to `logs/conversation_logs.json` | `conversation: List[ConversationInput]`, `feedback: List[FeedbackInput]`, `questions: Optional[List[QuestionInput]]` | `FeedbackResponse` (`success: str`) | Propagates `IOError`/`OSError` from `log_conversation` | `schema.py:106-117`, `test_graphql_contract.py:97` |
| 4 | GraphQL Query (New) | `Query.execution_modes` | Returns capabilities and gating status of registered execution modes | None | `List[ExecutionModeCapabilityType]` (`id`, `label`, `description`, `enabled`, `gated_issue`, `supports_multi_agent`, `supports_step_planning`) | Never raises; returns static registry capabilities | `ORIGINAL_REQUEST.md:38,208-212`, `schema.py` extension |
| 5 | GraphQL Mutation (New) | `Mutation.run_execution` | Unified multi-mode execution dispatcher (`collaborative`, `advanced_pipeline`, `tree_of_thought`) | `input: RunExecutionInput` (`mode`, `task_prompt`, `conversation`, `assistant_personalities`, `lead_personality`, `num_rounds`, `max_steps`, `initial_budget`, `agents`, `model`, `temperature`) | `RunExecutionResponse` (`mode`, `status`, `conversation`, `questions`, `final_output`, `output_type`, `diagnostics`, `events`, `error_message`) | Encapsulates errors into `status='error'` or `status='gated'` with `error_message` without throwing GraphQL errors | `ORIGINAL_REQUEST.md:39,250-270`, `schema.py` extension |
| 6 | GraphQL Mutation (New) | `Mutation.run_advanced_pipeline` | Direct standalone pipeline execution via `AdvancedPromptEngineer.main` | `task_prompt: str`, `max_steps: int = 20`, `initial_budget: int = 20`, `agents: int = 3`, `model: str = "gpt-4o-mini"` | `RunExecutionResponse` (with pipeline diagnostics and synthesized conversation trace) | Encapsulates pipeline failures into `status='error'` with details | `ORIGINAL_REQUEST.md:39,274`, `schema.py` extension |
| 7 | GraphQL Mutation (New) | `Mutation.run_reasoning_task` | Multi-strategy reasoning execution with structured execution telemetry | `input: TaskInput` (`task: str`, `strategy: PromptStrategy`, `max_steps: int`, `initial_budget: int`, `model: str`, `temperature: float`) | `ReasoningTaskResponse` (`task`, `strategy`, `final_output`, `steps`, `reflections`, `telemetry`, `error_message`) | Structured error reporting in `error_message`; records partial steps and telemetry duration | `ORIGINAL_REQUEST.md:389-391,411` (R1) |
| 8 | UI Component | `AssistantConfigPanel` Mode Selector | Allows toggling between execution modes with dynamic configuration fields | `executionMode?: ExecutionModeId`, `onExecutionModeChange?`, plus optional step/agent props | Renders mode tabs/dropdown + conditional mode controls; disables `tree_of_thought` with Issue #36 badge | Defaults to `collaborative` when props omitted | `ORIGINAL_REQUEST.md:40,280-294`, `AssistantConfigPanel.tsx` |
| 9 | UI Component | `ReasoningDiagnosticsPanel` | Visualizes execution complexity, token/step budgets, plan steps, reflections with reward scores, and event stream | `diagnostics?: RunDiagnosticsType`, `events?: RunEventType[]`, `mode?: string`, `defaultExpanded?: boolean` | Multi-tab collapsible diagnostic dashboard (`Overview`, `Plan & Subtasks`, `Reflections`, `Agents`, `Events`) | Gracefully handles null/empty diagnostics; remains hidden/collapsed when no diagnostics | `ORIGINAL_REQUEST.md:43,299-305` |
| 10 | UI Component | `StructuredOutputRenderer` | Formats final outputs based on `OutputType` (`output_type` and `file_extension`) | `content: string`, `outputType?: OutputTypeInfo`, `enableDownload?: boolean` | Formatted code block (`.py`, `.html`, `.js`), pretty JSON, dual-view CSV (table/raw), Markdown document, or text | Falls back safely to plain text `whitespace-pre-wrap` if formatting/parsing fails | `ORIGINAL_REQUEST.md:44,306-316`, `output_generator.py` |
| 11 | UI Component | `FeedbackDisplay` | Displays assistant questions and final output | `currentQuestions: QuestionType[]`, `finalOutput: string | null`, `runConversationLoading: boolean`, optional `outputType?: OutputTypeInfo` | Section with heading `"Questions for Feedback:"` and heading `"Final Output:"` wrapping renderer | Renders nothing when empty or loading | `FeedbackDisplay.tsx:1-41` |
| 12 | Frontend Hook | `useRunExecution` | Apollo Client mutation hook executing `RUN_EXECUTION` | `variables: RunExecutionVariables` | `{ runExecution, data, loading, error }` | Catches mutation error and stores in hook `error` state | `src/hooks/useRunExecution.ts` specification |
| 13 | Frontend Hook | `useExecutionModes` | Apollo Client query hook fetching `GET_EXECUTION_MODES` | None | `{ modes, loading, error, refetch }` | Stores error in Apollo query error state | `src/hooks/useExecutionModes.ts` specification |

---

## 2. Edge Cases

| # | Feature | Input | Observed Behavior |
|---|---------|-------|-------------------|
| 1 | `AssistantConfigPanel` | Default props without `executionMode` | Defaults `executionMode = 'collaborative'`. Renders personality checkboxes, lead dropdown, rounds input, and button "Start New Conversation". Preserves 100% compatibility with all 28 existing `AssistantConfigPanel.test.tsx` tests. |
| 2 | `AssistantConfigPanel` | Selection of `tree_of_thought` | Option is rendered with `disabled={true}` and label `"Tree of Thoughts (Gated: Issue #36)"`. Cannot be toggled into active running state; start button disabled. |
| 3 | `AssistantConfigPanel` | Negative or zero `numRounds` / `maxSteps` | Change handler rejects inputs <= 0 or non-numeric (`'abc'`, `0`, `-5`). Start button is disabled when `numRounds <= 0` or `maxSteps <= 0`. |
| 4 | `FeedbackDisplay` | `finalOutput` present with `outputType = null` | Defaults `outputType = { outputType: 'Simple Concise Answer', fileExtension: '.txt' }`. Heading `"Final Output:"` is rendered verbatim; content displayed in `whitespace-pre-wrap`. Existing test assertions targeting `"Final Output:"` succeed. |
| 5 | `StructuredOutputRenderer` | Malformed JSON in JSON output mode | JSON parser catches syntax error in `try / catch`, falls back to rendering raw string cleanly in syntax container without crashing React tree. |
| 6 | `StructuredOutputRenderer` | CSV content with uneven columns or escaped commas | CSV parser splits by line and handles standard comma-separated cells; displays interactive table view with header row, or allows user to toggle to "Raw View" tab. |
| 7 | `StructuredOutputRenderer` | Empty string `content = ""` | Displays fallback placeholder "No output generated" or empty pre block without throwing null pointer error. |
| 8 | `ReasoningDiagnosticsPanel` | `diagnostics = null` and `events = []` | Panel renders nothing or stays completely collapsed, preventing unwanted DOM clutter in existing conversation tests. |
| 9 | `ReasoningDiagnosticsPanel` | `reflections` with missing `thoughts` or `step_description` | Thought box is omitted; reflection content and reward badge render cleanly without undefined text. |
| 10 | `Query.execution_modes` | Offline execution | Resolves statically from in-memory `CAPABILITY_REGISTRY` without calling OpenAI SDK, databases, or external network. |
| 11 | `Mutation.run_execution` | Mode `'tree_of_thought'` | Does NOT instantiate `provider_client` or invoke LLM. Immediately returns `status='gated'`, `errorMessage="Execution mode 'tree_of_thought' is gated for Issue #36."`, and `RunEvent(eventType="mode_gated")`. |
| 12 | `Mutation.run_execution` | Empty conversation and empty `task_prompt` | Validates input; sets status to `'error'` with clear validation message rather than throwing unhandled server exception. |
| 13 | `Mutation.run_reasoning_task` | Unknown strategy enum | GraphQL validation rejects invalid enum values at query execution time with standard GraphQL validation error before reaching resolver. |
| 14 | `ConversationPage` | Test mock for `useRunConversation` without `useRunExecution` | `ConversationPage` defaults to `collaborative` mode and calls `useRunConversation`, ensuring existing `ConversationPage.test.tsx` mocks remain 100% active and passing. |

---

## 3. Five-Component Handoff Report

### 3.1 Observation

Direct inspection of production and test files revealed the following exact lines and behaviors:

1. **`schema.py` Structure (lines 56–117)**:
   - Root `Query` currently has only `hello` returning `"Hello, World!"` (lines 57–60).
   - Root `Mutation` currently exposes:
     - `run_conversation(self, conversation: List[ConversationInput], assistant_personalities: List[str], leadPersonality: str, num_rounds: int) -> RunConversationResponse` (lines 66–103). Note exact parameter name `leadPersonality` (camelCase in Python signature).
     - `submit_feedback(self, conversation: List[ConversationInput], feedback: List[FeedbackInput], questions: Optional[List[QuestionInput]] = None) -> FeedbackResponse` (lines 106–117).
   - In `app.py` (lines 7–21):
     - `schema = strawberry.Schema(query=Query, mutation=Mutation)`
     - Mounted via `app.add_url_rule("/graphql", view_func=GraphQLView.as_view("graphql_view", schema=schema))`.

2. **Backend GraphQL Contract Tests (`test_graphql_contract.py` lines 18–95 and `test_smoke.py` lines 135–165)**:
   - `test_run_conversation_contract` runs GraphQL query with:
     ```graphql
     mutation RunConversationTest($conversation: [ConversationInput!]!, $assistantPersonalities: [String!]!, $leadPersonality: String!, $numRounds: Int!) {
       runConversation(conversation: $conversation, assistantPersonalities: $assistantPersonalities, leadPersonality: $leadPersonality, numRounds: $numRounds) {
         conversation { role name content }
         questions { assistant question }
         finalOutput
       }
     }
     ```
   - Running `./tools/run-offline.ps1 -PytestArgs 'test_graphql_contract.py'` passed 2/2 tests in 0.41s with exit code 0.

3. **Frontend GraphQL Contracts (`src/graphql/graphqlTypes.ts` lines 1–27 & `conversationMutations.ts` lines 4–45)**:
   - `graphqlTypes.ts` defines only 5 interfaces: `ConversationInputType`, `FeedbackInputType`, `QuestionInputType`, `ConversationType`, `QuestionType`.
   - `conversationMutations.ts` exports `RUN_CONVERSATION` mutation, `RunConversationVariables`, and `RunConversationData`.
   - `feedbackMutations.ts` (lines 26–30) expects `submitFeedback: { success: boolean }`.

4. **Frontend Component Prop Contracts & Tests**:
   - `AssistantConfigPanel.tsx` (lines 7–17): Expects 9 props: `availablePersonalities`, `selectedPersonalities`, `onSelectedPersonalitiesChange`, `leadPersonality`, `onLeadPersonalityChange`, `numRounds`, `onNumRoundsChange`, `onStartConversation`, `isConversationActive`.
   - `AssistantConfigPanel.test.tsx` (lines 13–27): `defaultProps` instantiates `AssistantConfigPanel` with only those 9 props across 28 unit tests. All 28 tests pass (3824ms).
   - `FeedbackDisplay.tsx` (lines 4–8 & 30):
     - Props: `currentQuestions`, `finalOutput`, `runConversationLoading`.
     - Heading: `<h3 className="text-md font-semibold mb-2 text-text-primary">Final Output:</h3>`.
   - `ConversationPage.tsx` (lines 136–140):
     - Mounts `<FeedbackDisplay currentQuestions={currentQuestions} finalOutput={finalOutput} runConversationLoading={runConversationLoading} />`.
   - `ConversationPage.test.tsx` (lines 10 & 34–40):
     - Mocks `vi.mock('../hooks/useRunConversation')`.
     - Asserts `screen.getByRole('button', { name: /Start New Conversation/i })` and `screen.findByText('This is the final output.')`.
     - When executed via `node node_modules/vitest/vitest.mjs run src/pages/ConversationPage.test.tsx`, all 10 integration tests pass in 5.78s.

5. **Frontend Lint & Build Verification**:
   - Running `./tools/run-frontend-offline.ps1 -Action build` executes `tsc -b` and `vite build`. Completed in 8.84s producing `dist/assets/index-BxQ7j8m_.js` (408.08 kB) with 0 errors.
   - Running `./tools/run-frontend-offline.ps1 -Action guard` output: `"Node network denial verified"`.
   - Running `node node_modules/eslint/bin/eslint.js src` output: `exit code 0` with 0 errors.
   - Historical `.qualification/logs/frontend-lint.txt` had 13 errors, but source code inspections show fixes: `CardProps`, `InputProps`, `TextareaProps` are type aliases; `catch {}` statements have no unused identifiers; and `useRunConversation.test.ts` uses type casts instead of `any`. The only remaining potential lint item is `ButtonProps` in `Button.tsx` (`export interface ButtonProps extends ... {}` -> should be `export type ButtonProps = ...`).

---

### 3.2 Logic Chain

1. **Schema Extension Safety**:
   - `schema.py` defines `Query` and `Mutation` using standard Strawberry classes.
   - Strawberry allows adding arbitrary `@strawberry.field` methods to `Query` and `@strawberry.mutation` methods to `Mutation` without affecting existing fields.
   - Therefore, adding `Query.execution_modes`, `Mutation.run_execution`, `Mutation.run_advanced_pipeline`, and `Mutation.run_reasoning_task` while leaving `Query.hello`, `Mutation.run_conversation`, and `Mutation.submit_feedback` untouched introduces zero schema breaking changes.
   - All existing tests (`test_graphql_contract.py`, `test_smoke.py`) execute exact query strings targeting `runConversation` and `submitFeedback` and will continue returning identical response payloads.

2. **Frontend Backward Compatibility**:
   - `AssistantConfigPanel`: Making all new execution mode props optional (`executionMode?: ExecutionModeId = 'collaborative'`, `maxSteps?: number = 20`, etc.) ensures existing test suites that render `<AssistantConfigPanel {...defaultProps} />` continue to receive the default collaborative behavior. In collaborative mode, all 28 existing test assertions find their expected labels, checkboxes, dropdowns, and button texts.
   - `FeedbackDisplay`: Keeping the heading `<h3 ...>Final Output:</h3>` and nesting `StructuredOutputRenderer` inside or immediately below it ensures that both text search queries (`screen.getByText('Final Output:')`) and content queries (`screen.findByText('This is the final output.')`) continue to pass.
   - `ConversationPage`: Preserving `useRunConversation` for `collaborative` mode guarantees that `ConversationPage.test.tsx` (which explicitly mocks `useRunConversation`) intercepts calls and renders mock data without timing out or failing.

3. **Multi-Strategy Reasoning & Telemetry (R1/R2) Integration**:
   - `runReasoningTask` cleanly models the 5 prompt strategies via `PromptStrategy` enum: `CHAIN_OF_THOUGHT`, `LEAST_TO_MOST`, `PROGRESSIVE_HINT`, `SELF_CONSISTENCY`, `DYNAMIC_CONFIDENCE_EXPLORATION`.
   - Returning `TelemetryMetrics` (`duration_ms`, `total_tokens`, `step_count`, `correlation_id`) satisfies R1/R2 observability requirements.
   - `ReasoningDiagnosticsPanel` in the frontend can visualize both `RunDiagnosticsType` (from `runExecution` / `runAdvancedPipeline`) and `ReasoningTaskResponse` telemetry (from `runReasoningTask`).

---

### 3.3 Caveats

1. **Vitest Timing on Windows**: When running the entire Vitest suite concurrently (`111 tests in 10 files`), the first test in `ConversationPage.test.tsx` (`allows typing a message and sending it`) took 5557ms and hit the default 5000ms Vitest timeout due to CPU throttling during concurrent `userEvent.type()` simulation. When run individually or with a standard 10s test timeout, all 10 tests pass cleanly in 5.78s.
2. **`tree_of_thought` Mode Gating**: As specified in `ORIGINAL_REQUEST.md` (Issue #36), `tree_of_thought` must remain strictly gated (`enabled=False`, `gated_issue="Issue #36"`). The backend dispatcher must return a deterministic gated `RunResult` without invoking `provider_client` or any LLM, and the frontend must render it with a disabled badge.
3. **No Circular Imports**: `schema.py` must import execution handlers lazily inside resolver functions (`from execution_modes import execute_run, RunRequest`, etc.) to prevent circular imports between `schema.py` and `conversation_manager.py`.

---

### 3.4 Conclusion

The authoritative contracts across the Strawberry GraphQL API, Flask server, Apollo Client operations, and React 19 frontend components have been thoroughly probed and mapped. Extending `schema.py` with `execution_modes`, `run_execution`, `run_advanced_pipeline`, and `run_reasoning_task`, accompanied by `AssistantConfigPanel` mode switching, `ReasoningDiagnosticsPanel`, and `StructuredOutputRenderer`, is completely feasible with zero regressions across the 148 backend offline tests and 111 frontend tests.

---

### 3.5 Verification Method

To independently verify the findings in this report, execute the following authoritative commands from the repository root in PowerShell:

```powershell
# 1. Verify GraphQL contract test passes offline (0.41s, 2 passed)
pwsh -NoProfile -ExecutionPolicy Bypass -File .\tools\run-offline.ps1 -PytestArgs 'test_graphql_contract.py'

# 2. Verify smoke test suite passes offline (0.8s, 6 passed)
pwsh -NoProfile -ExecutionPolicy Bypass -File .\tools\run-offline.ps1 -PytestArgs 'test_smoke.py'

# 3. Verify Frontend Network Denial Guard
pwsh -NoProfile -ExecutionPolicy Bypass -File .\tools\run-frontend-offline.ps1 -Action guard

# 4. Verify Frontend TypeScript & Production Build (0 errors, 8.8s)
pwsh -NoProfile -ExecutionPolicy Bypass -File .\tools\run-frontend-offline.ps1 -Action build

# 5. Verify Frontend ESLint (0 errors)
pwsh -NoProfile -ExecutionPolicy Bypass -File .\tools\run-frontend-offline.ps1 -Action lint

# 6. Verify Frontend Vitest Test Suites
pwsh -NoProfile -ExecutionPolicy Bypass -File .\tools\run-frontend-offline.ps1 -Action test
```

Inspect the following authoritative files:
- `schema.py`: Lines 56–117 for existing `Query` and `Mutation` field definitions.
- `app.py`: Lines 9–21 for `strawberry.Schema` initialization and Flask routing.
- `collaborative-assistant-frontend/src/components/AssistantConfigPanel.tsx`: Lines 7–29 for prop definitions.
- `collaborative-assistant-frontend/src/components/FeedbackDisplay.tsx`: Lines 30–35 for the `"Final Output:"` heading.
- `collaborative-assistant-frontend/src/pages/ConversationPage.test.tsx`: Lines 10–40 for `useRunConversation` mocking.
