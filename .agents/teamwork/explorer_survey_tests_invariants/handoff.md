# Comprehensive Handoff Report: Test Harness, Existing Test Suites, and Red-Green Test Matrix Design

**Role**: Test & Invariants Explorer (`teamwork_preview_explorer`)  
**Target Milestone**: Track 1 — End-to-End Execution Modes & UI Integration (Issue #35 Phases 3–5 & Issue #21, Requirements R1–R4)  
**Report Path**: `d:\projects\advanced_prompting_system\advanced_prompting_system\.agents\teamwork\explorer_survey_tests_invariants\handoff.md`  

---

## 1. Observation

### 1.1 Offline Test Harness & Network Isolation Verification
We directly executed and inspected the offline qualification wrappers and guard mechanisms:

1. **Python Network Isolation Guard (`tools/offline/sitecustomize.py`)**:
   - `sys.addaudithook(audit)` audits `socket.getaddrinfo`, `socket.connect`, and `socket.sendto`.
   - Any external (non-loopback) host raises `NetworkForbidden(BaseException)`:
     ```python
     class NetworkForbidden(BaseException):
         pass
     ```
     Because `NetworkForbidden` inherits from `BaseException` (not `Exception`), general application `try ... except Exception:` blocks cannot swallow accidental network calls.
   - Removes all `API_KEY`, `ACCESS_TOKEN`, `AUTH_TOKEN`, `OPENAI_ADMIN_KEY`, and `HF_TOKEN` from `os.environ`.
   - Sets `sys._aps_network_guard = True`.

2. **Pytest Offline Policy & Boundaries (`conftest.py`)**:
   - `pytest_configure` checks `sys._aps_network_guard`; fails with `pytest.UsageError` if bare pytest is run without the harness.
   - `forbid_unconfigured_provider`: Autouse fixture replaces `provider_client.get_client` with a function that calls `pytest.fail('Unconfigured SDK call: inject a provider response at this test boundary')`. Any unmocked or leaked provider invocation fails immediately.
   - `isolated_runtime_directory`: Autouse fixture executes `monkeypatch.chdir(tmp_path)` to keep generated artifacts, logs, and `conversation_history.json` away from repository source trees.
   - `conversation_case`: Shared fixture providing mock chat completions and `beta.chat.completions.parse` for `OutputType` and `MediatorReviseOrDecide`.

3. **Node Network Denial Guard (`tools/offline/node-network-guard.cjs`)**:
   - Overrides `net.Socket.prototype.connect`, `dns.lookup`, and `dgram.Socket.prototype.send`.
   - Non-loopback addresses throw `Error('Offline test attempted external network: ' + host)`.
   - Verified via `powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\run-frontend-offline.ps1 -Action guard`:
     `Output: Node network denial verified` (Exit code: 0).

### 1.2 Existing Test Suite Baselines
We ran the full backend and frontend offline test suites:

- **Backend Pytest Offline Baseline**:
  - Command: `powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\run-offline.ps1 -PytestArgs @('-q')`
  - Result: **148 passed, 0 failed, 2 deprecation warnings in 90.02s** (Exit code: 0).
  - Breakdown across 11 collected test files:
    - `test_advanced_prompting.py`: 48 passed (all 24 previously quarantined parser/reasoning tests unquarantined and passing offline)
    - `test_finalize_step_output.py`: 1 passed
    - `test_graphql_contract.py`: 2 passed (`test_run_conversation_contract`, `test_submit_feedback_contract`)
    - `test_import_integrity.py`: 7 passed
    - `test_memory_management.py`: 20 passed
    - `test_message_handling.py`: 13 passed
    - `test_offline_runtime.py`: 10 passed (including clean-process import isolation and socket denial assertion)
    - `test_pr34_regressions.py`: 29 passed
    - `test_retrieval.py`: 6 passed
    - `test_smoke.py`: 6 passed
    - `test_voting_semantics.py`: 6 passed
    - Total: 48 + 1 + 2 + 7 + 20 + 13 + 10 + 29 + 6 + 6 + 6 = **148 tests**.

- **Frontend Vitest Offline Baseline**:
  - Command: `powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\run-frontend-offline.ps1 -Action test`
  - Result: **111 passed across 10 test files in 96.87s** (Exit code: 0).
  - Breakdown:
    - `src/components/MessageBubble.test.tsx`: 14 passed
    - `src/components/ui/Card.test.tsx`: 13 passed
    - `src/components/ui/Button.test.tsx`: 11 passed
    - `src/components/AssistantConfigPanel.test.tsx`: 28 passed
    - `src/components/ChatInput.test.tsx`: 9 passed
    - `src/components/ui/Input.test.tsx`: 7 passed
    - `src/hooks/useRunConversation.test.ts`: 6 passed
    - `src/components/ui/Textarea.test.tsx`: 8 passed
    - `src/components/ui/Label.test.tsx`: 5 passed
    - `src/pages/ConversationPage.test.tsx`: 10 passed
    - Total: **111 tests**.

- **Frontend Lint & Build Baseline**:
  - `powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\run-frontend-offline.ps1 -Action lint` -> **0 errors, 0 warnings** (Exit code: 0).
  - `powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\run-frontend-offline.ps1 -Action build` -> `tsc -b` and `vite build` completed cleanly; generated `dist/` bundle (Exit code: 0).

### 1.3 Key Contract & Schema Observations
1. `schema.py`:
   - Strawberry schema defines `Query.hello`, `Mutation.run_conversation`, `Mutation.submit_feedback`.
   - Imports `conversation_manager.run_conversation` and `feedback_manager.log_conversation` lazily inside mutation resolvers.
   - `test_graphql_contract.py` executes queries synchronously via `self.schema.execute_sync(query, variable_values=variables)`. No HTTP server or network sockets are created.
2. `AssistantConfigPanel.tsx` and `AssistantConfigPanel.test.tsx`:
   - In `AssistantConfigPanel.test.tsx`, `defaultProps` defines only 8 props: `availablePersonalities`, `selectedPersonalities`, `onSelectedPersonalitiesChange`, `leadPersonality`, `onLeadPersonalityChange`, `numRounds`, `onNumRoundsChange`, `onStartConversation`, `isConversationActive`.
   - Any new props added to `AssistantConfigPanelProps` (e.g., `executionMode`, `onExecutionModeChange`, `maxSteps`, `onMaxStepsChange`, `agentCount`, `onAgentCountChange`) **must be optional with defaults**; making any new prop required would immediately break 28 existing frontend tests.
3. `FeedbackDisplay.tsx` and `ConversationPage.test.tsx`:
   - `FeedbackDisplay.tsx` renders `<h3 ...>Final Output:</h3>` followed by the output text.
   - `ConversationPage.test.tsx` searches for the output text rendered in `FeedbackDisplay`.
   - Any structured output component replacing plain output must maintain the heading `"Final Output:"` and render the content so existing queries continue to match.

---

## 2. Logic Chain

1. **Zero Import-Side-Effect Invariant**:
   - *Observation*: `test_offline_runtime.py::test_fresh_process_imports_and_network_denial` verifies that importing backend modules executes zero network calls and constructs zero `openai.OpenAI()` clients.
   - *Logic*: The new module `execution_modes.py` and modified `schema.py` must maintain this property. `execution_modes.py` must define Pydantic schemas and registry definitions without eagerly importing or executing heavy model code, and route execution via lazy adapters.
   - *Inference*: `test_execution_modes.py` must include a fresh-process import test asserting that importing `execution_modes` does not trigger client creation or network traffic.

2. **Gated Mode Isolation Invariant (`tree_of_thought`)**:
   - *Observation*: Issue #36 is reserved for the genuine Tree of Thoughts implementation. `forbid_unconfigured_provider` in `conftest.py` fails any test where `provider_client.get_client` is called without an explicit test fixture.
   - *Logic*: When `execute_run(RunRequest(mode="tree_of_thought"))` is invoked, it must return a gated `RunResult` immediately (`status="gated"`, `error=ErrorResult(...)`, `events=[RunEvent(event_type="mode_gated", ...)]`) without touching `provider_client.get_client()`, `run_conversation()`, or `AdvancedPromptEngineer()`.
   - *Inference*: A dedicated unit test in `test_execution_modes.py` running without any mock on `provider_client` will automatically fail if `tree_of_thought` mode touches any provider SDK, verifying the gating contract fail-closed.

3. **Public Contract Immutability**:
   - *Observation*: `conversation_manager.run_conversation` returns `(conversation_history, questions_asked, final_output)` 3-tuple, tested in `test_smoke.py` and `test_pr34_regressions.py`.
   - *Logic*: `execution_modes.py`'s `_execute_collaborative` handler must call `run_conversation(...)` and unpack the 3-tuple without mutating `run_conversation`'s signature or return type.
   - *Inference*: Existing tests in `test_smoke.py`, `test_pr34_regressions.py`, and `test_graphql_contract.py` will remain completely unaffected.

4. **Multi-Strategy Reasoning & Telemetry (R1 & R2)**:
   - *Observation*: Requirements R1 & R2 require a GraphQL mutation (`runReasoningTask` / `runExecution` / `runAdvancedPipeline`) returning final output, intermediate steps, reflections, and execution telemetry metrics (duration/latency, token expenditure, step counts, request correlation ID).
   - *Logic*: In `execution_modes.py`, `RunDiagnostics` and `RunEvent` capture step counts, reflections with rewards, complexity scores, and ordered events. Adding `duration_ms: float`, `token_count: int`, and `correlation_id: str` into `RunDiagnostics` / `TelemetryMetrics` and computing duration via `time.perf_counter()` provides deterministic offline telemetry without external dependencies.
   - *Inference*: We can test telemetry generation completely offline, asserting non-negative duration, positive token count (via `AdvancedPromptEngineer.count_tokens` offline fallback), and valid UUID/correlation string.

5. **Frontend Regression Safety**:
   - *Observation*: `ConversationPage.test.tsx` tests the complete conversation flow using mocked `useRunConversation`.
   - *Logic*: By retaining `useRunConversation` for collaborative debate while introducing `useRunExecution` for unified execution, and defaulting `executionMode` to `'collaborative'`, all 10 integration tests in `ConversationPage.test.tsx` will pass unchanged.
   - *Inference*: New tests for `ReasoningDiagnosticsPanel`, `StructuredOutputRenderer`, and mode switching will be added in new dedicated test files, increasing coverage without modifying existing test baselines.

---

## 3. Caveats

1. **Offline Token Counting Heuristic**:
   - When running offline under `APS_OFFLINE_TESTS=1`, `tiktoken` cannot download BPE tokenizers and raises `NetworkForbidden`. `AdvancedPromptEngineer.count_tokens` catches this and uses a deterministic heuristic (`len(text.split()) * 1.33` or `len(text) // 4`). Tests for token telemetry must assert structural properties (positive integer) rather than exact live tokenizer counts.
2. **Offline Embedding Fallback**:
   - In offline qualification mode (`OFFLINE_MODE=1`), `complexity_measures.get_embedding` returns deterministic seeded synthetic vectors. Plan generation and complexity scoring tests prove structural data flow and schema validation, not live semantic similarity or reasoning quality.
3. **Execution Durations on Windows**:
   - Running the full pytest backend suite takes ~90 seconds and the frontend Vitest suite takes ~96 seconds on Windows due to JSDOM virtualization and cold imports of NLP/ML libraries. During development, developers should execute focused test suites (e.g., `./tools/run-offline.ps1 -PytestArgs @('test_execution_modes.py', '-v')`).
4. **Tree of Thoughts Capability**:
   - `tree_of_thought` is intentionally gated pending Issue #36. Tests must verify that the mode is correctly declared as `enabled=False` with `gated_issue="Issue #36"`, and that invoking it returns a gated status with zero provider calls.

---

## 4. Conclusion & Red-Green Test Matrix Design

The test architecture for Track 1 is structured across three layers:
1. Backend Unit Tests (`test_execution_modes.py`)
2. GraphQL Contract Tests (`test_graphql_contract.py`)
3. Frontend Vitest Tests (`ReasoningDiagnosticsPanel.test.tsx`, `StructuredOutputRenderer.test.tsx`, `AssistantConfigPanel.test.tsx`)
Plus adversarial and invariant verification suites.

### 4.1 Backend Unit Test Matrix (`test_execution_modes.py`)

A new authoritative test module `test_execution_modes.py` with approximately 20–25 unit tests structured under `unittest.TestCase` or `pytest`:

| Test Class / Group | Test Method | Purpose & Verification |
|---|---|---|
| **`TestRunRequestValidation`** | `test_default_initialization` | Verify default fields: `mode="collaborative"`, `num_rounds=3`, `max_steps=20`, `initial_budget=20`, `agents=3`, `model="gpt-4o-mini"`. |
| | `test_task_prompt_derived_from_conversation` | Provide `conversation=[{"role": "user", "content": "Analyze market"}]`, assert `task_prompt == "Analyze market"`. |
| | `test_conversation_populated_from_task_prompt` | Provide `task_prompt="Build API"`, `conversation=[]`, assert `conversation == [{"role": "user", "content": "Build API", "name": "User"}]`. |
| | `test_invalid_mode_rejected` | Provide `mode="quantum_reasoning"`, assert Pydantic `ValidationError` raised. |
| | `test_bounds_validation` | Test `num_rounds < 1`, `max_steps < 1`, `agents < 0`, assert validation behavior. |
| **`TestDiagnosticsAndEventSchemas`** | `test_subtask_and_planstep_diagnostic_serialization` | Instantiate `PlanStepDiagnostic` with nested `SubtaskDiagnostic`, serialize to dict, verify round-trip. |
| | `test_reflection_diagnostic_fields` | Verify `thoughts`, `content`, `reward`, `step_number`, `remaining_budget`. |
| | `test_run_event_iso8601_timestamp` | Assert `RunEvent.timestamp` conforms to ISO-8601 UTC format. |
| | `test_run_diagnostics_telemetry_fields` | Verify `complexity_score`, `initial_budget`, `adjusted_budget`, `final_reward`, `duration_ms`, `token_count`, `correlation_id`. |
| **`TestCapabilityRegistry`** | `test_registry_contains_three_modes` | Assert keys `{"collaborative", "advanced_pipeline", "tree_of_thought"}` exist in `CAPABILITY_REGISTRY`. |
| | `test_collaborative_capability_metadata` | `enabled=True`, `gated_issue=None`, `supports_multi_agent=True`, `supports_step_planning=False`. |
| | `test_advanced_pipeline_capability_metadata` | `enabled=True`, `gated_issue=None`, `supports_multi_agent=True`, `supports_step_planning=True`. |
| | `test_tree_of_thought_capability_metadata` | `enabled=False`, `gated_issue="Issue #36"`, `supports_multi_agent=False`, `supports_step_planning=True`. |
| **`TestExecuteRunGatedMode`** | `test_tree_of_thought_gated_returns_immediately` | Run `execute_run(RunRequest(mode="tree_of_thought", task_prompt="Test"))`. Assert `result.status == "gated"`, `result.error.error_type == "ModeGatedError"`, `result.events[0].event_type == "mode_gated"`. |
| | `test_tree_of_thought_never_calls_provider` | Execute under `forbid_unconfigured_provider` without any mocked provider; assert test passes without triggering provider failure. |
| **`TestExecuteRunCollaborative`** | `test_collaborative_execution_success` | Mock `conversation_manager.run_conversation` returning 3-tuple; assert `RunResult(status="completed", mode="collaborative")`, output type resolved, speaker attributions generated. |
| | `test_collaborative_event_sequence` | Assert events `["mode_selected", "agent_contribution", "output_finalized"]` emitted in order. |
| | `test_collaborative_file_output_type_sniffing` | Given final output pointing to `.json` or `.py` file content, verify `output_type` matches detected type. |
| **`TestExecuteRunAdvancedPipeline`** | `test_advanced_pipeline_execution_success` | Mock `AdvancedPromptEngineer.main` returning an `Interaction` with steps and reflections; assert `RunResult` has structured `diagnostics` with plan steps, reflections, and attributions. |
| | `test_advanced_pipeline_agents_zero` | Test `request.agents = 0`; verify `assigned_agents` list is empty and no collaborative sub-agents are invoked. |
| | `test_advanced_pipeline_conversation_synthesis` | Verify `result.conversation` contains a synthesized transcript (user prompt, step milestones, final output). |
| | `test_advanced_pipeline_event_sequence` | Assert events `["mode_selected", "complexity_assessed", "plan_generated", "plan_step_started", "plan_step_completed", "reflection_recorded", "output_finalized"]` emitted. |
| **`TestAdversarialAndErrorHandling`** | `test_pipeline_empty_plan_handling` | When `task.plan` has no steps, assert `RunResult` returns gracefully with `final_output` without unhandled exception. |
| | `test_collaborative_exception_wrapped` | When `run_conversation` raises an exception, assert `RunResult(status="error", error=ErrorResult(...))` returned. |
| | `test_import_purity_execution_modes` | Assert importing `execution_modes` executes zero network calls and constructs zero `openai.OpenAI()` instances. |

### 4.2 GraphQL Contract Test Matrix (`test_graphql_contract.py`)

Extensions to `test_graphql_contract.py`:

| Test Method | GraphQL Operation Tested | Verification |
|---|---|---|
| `test_execution_modes_query_contract` | `query GetExecutionModes { executionModes { id label description enabled gatedIssue supportsMultiAgent supportsStepPlanning } }` | Verifies all 3 modes returned with correct types and values matching `CAPABILITY_REGISTRY`. |
| `test_run_execution_collaborative_contract` | `mutation RunExecution($input: RunExecutionInput!) { runExecution(input: $input) { mode status conversation { role name content } questions { assistant question } finalOutput outputType { outputType fileExtension } diagnostics { finalReward planSteps { stepNumber stepName } } events { eventId eventType summary } error { errorType errorMessage } } }` | Executes with mocked `execute_run`; validates full GraphQL serialization of collaborative mode response. |
| `test_run_execution_advanced_pipeline_contract` | `mutation RunExecution($input: RunExecutionInput!) { runExecution(input: $input) { mode status finalOutput outputType { outputType fileExtension } diagnostics { complexityScore initialBudget adjustedBudget finalReward planSteps { stepNumber stepName assignedAgents subtasks { subtaskNumber subtaskName completed } } reflections { stepNumber content reward remainingBudget } agentAttributions { agentName role roundOrStep contributionSummary } } events { eventId eventType summary } } }` | Executes with mocked `execute_run`; validates complete diagnostics and telemetry field mapping. |
| `test_run_execution_tree_of_thought_gated_contract` | `mutation RunExecution($input: RunExecutionInput!) { runExecution(input: $input) { mode status finalOutput error { errorType errorMessage } events { eventType summary } } }` | Verifies `mode="tree_of_thought"` returns `status="gated"`, gating error message, and `mode_gated` event. |
| `test_run_advanced_pipeline_mutation_contract` | `mutation RunAdvancedPipeline($taskPrompt: String!, $maxSteps: Int, $initialBudget: Int, $agents: Int, $model: String) { runAdvancedPipeline(...) { mode status finalOutput diagnostics { complexityScore planSteps { stepNumber } } } }` | Verifies standalone convenience mutation signature and output shape. |
| `test_run_reasoning_task_mutation_contract` (R1) | `mutation RunReasoningTask($task: String!, $strategy: PromptStrategy!, $maxSteps: Int) { runReasoningTask(...) { finalOutput intermediateSteps reflections telemetry { durationMs tokenCount stepCount correlationId } } }` | Verifies strategy-based mutation and telemetry block contract. |
| `test_backward_compatibility_run_conversation` | Existing `runConversation` mutation | Asserts existing `runConversation` test passes completely unchanged. |
| `test_backward_compatibility_submit_feedback` | Existing `submitFeedback` mutation | Asserts existing `submitFeedback` test passes completely unchanged. |

### 4.3 Frontend Vitest Test Matrix

#### A. `src/components/ReasoningDiagnosticsPanel.test.tsx` (New — ~10 tests)
1. `renders_complexity_and_budget_summary`: Displays formatted `complexityScore` (with Low/Moderate/High badge), `initialBudget` -> `adjustedBudget`, and `finalReward`.
2. `renders_plan_steps_with_status_badges`: Renders each step from `planSteps` with step number, name, description, and `Completed` / `In Progress` badge.
3. `renders_nested_subtasks_checklist`: Renders checklist items from `subtasks` under each step with completion status.
4. `renders_assigned_agent_pills`: Renders pills for each agent in `assignedAgents`.
5. `renders_reflections_with_reward_scores`: Renders reflection content, thoughts block, and colored reward score badge.
6. `renders_agent_attributions`: Renders speaker contribution summaries from `agentAttributions`.
7. `renders_chronological_events_timeline`: Renders event items from `events` with timestamps and summaries.
8. `collapsible_accordion_sections`: Allows expanding/collapsing individual sections (Plan Steps, Reflections, Event Stream).
9. `handles_empty_diagnostics_gracefully`: Renders empty placeholder or collapses cleanly when `planSteps`, `reflections`, and `events` are empty.
10. `handles_null_optional_fields`: Gracefully renders when `complexityScore`, `finalReward`, or `thoughts` are null/undefined.

#### B. `src/components/StructuredOutputRenderer.test.tsx` (New — ~10 tests)
1. `renders_code_snippet_container`: For `outputType: { outputType: "Python Script", fileExtension: ".py" }`, renders dark-themed code block with syntax container `<pre><code data-testid="structured-code-output">`.
2. `displays_language_badge_and_extension`: Renders pill showing language/type and file extension badge (e.g., `Python Script (.py)`).
3. `renders_pretty_printed_json`: For valid JSON string and `fileExtension: ".json"`, pretty-prints formatted JSON.
4. `falls_back_gracefully_on_invalid_json`: If JSON parsing fails, renders raw text without crashing.
5. `renders_csv_table_preview`: For `fileExtension: ".csv"`, parses rows and renders an HTML preview table plus toggle for raw CSV text.
6. `renders_markdown_formatted_text`: For `Simple Concise Answer` or `.txt`, parses fenced code blocks, bullet lists, and bold text.
7. `copy_button_copies_to_clipboard`: Clicking the `Copy` button writes output content to `navigator.clipboard.writeText` and shows copied confirmation.
8. `preserves_final_output_heading`: Verifies the container or parent retains `Final Output:` heading so existing queries in `ConversationPage.test.tsx` continue finding it.
9. `handles_empty_output_content`: Renders cleanly when content is empty string.
10. `handles_missing_output_type_gracefully`: Defaults to `{ outputType: 'Simple Concise Answer', fileExtension: '.txt' }` when `outputType` prop is undefined.

#### C. `src/components/AssistantConfigPanel.test.tsx` (Additions — ~6 tests)
1. `renders_execution_mode_selector`: Renders mode selection control (`collaborative`, `advanced_pipeline`, `tree_of_thought`).
2. `tree_of_thought_mode_is_disabled`: Verifies the `tree_of_thought` option is disabled with badge `"Issue #36"`.
3. `switching_to_advanced_pipeline_shows_pipeline_controls`: When `advanced_pipeline` is selected, renders `maxSteps` and `agentCount` inputs.
4. `calls_on_execution_mode_change`: Clicking a mode option triggers `onExecutionModeChange(mode)`.
5. `button_label_adapts_to_mode`: Button text displays `"Start New Conversation"` for collaborative and `"Start Pipeline Session"` (or unified action) for pipeline mode.
6. **Regression Invariant Test**: Renders `<AssistantConfigPanel {...defaultProps} />` with only the original 8 props; asserts it renders in collaborative mode with all original controls intact.

#### D. `src/pages/ConversationPage.test.tsx` (Additions — ~4 tests)
1. `defaults_to_collaborative_mode`: Verifies default page state is collaborative debate.
2. `dispatches_run_execution_or_pipeline`: Selecting pipeline mode and sending a prompt triggers the pipeline execution mutation/hook.
3. `renders_reasoning_diagnostics_panel_when_diagnostics_returned`: When response contains `diagnostics`, `ReasoningDiagnosticsPanel` is mounted and rendered.
4. `renders_structured_output_renderer_for_final_output`: Verifies final output is rendered via `StructuredOutputRenderer` while keeping `Final Output:` visible.

#### E. `src/hooks/useRunExecution.test.ts` (New — ~4 tests)
1. `executes_run_execution_mutation`: Calls Apollo mutation `RUN_EXECUTION` and returns data.
2. `executes_run_advanced_pipeline_mutation`: Calls Apollo mutation `RUN_ADVANCED_PIPELINE`.
3. `handles_loading_and_error_states`: Sets `loading: true` during query and populates `error` on failure.
4. `normalizes_execution_response`: Formats diagnostics and events for component consumption.

---

## 5. Verification Method

### 5.1 Authoritative Verification Commands
All verification must be run from the repository root in PowerShell using the isolated offline wrappers:

```powershell
# 1. Backend Offline Collection & Targeted Execution
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\run-offline.ps1 -PytestArgs @('--collect-only', '-q')
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\run-offline.ps1 -PytestArgs @('test_execution_modes.py', '-v')
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\run-offline.ps1 -PytestArgs @('test_graphql_contract.py', 'test_smoke.py', '-v')

# 2. Full Backend Offline Suite Gate (Target: >= 170 passed, 0 failed)
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\run-offline.ps1 -PytestArgs @('-q')

# 3. Frontend Offline Isolation & Guard Probe
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\run-frontend-offline.ps1 -Action guard

# 4. Frontend Vitest Suite Gate (Target: >= 140 passed across >= 13 files, 0 failed)
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\run-frontend-offline.ps1 -Action test

# 5. Frontend Production Bundle Build & Typecheck
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\run-frontend-offline.ps1 -Action build

# 6. Frontend Lint Hygiene Gate
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\run-frontend-offline.ps1 -Action lint

# 7. Whitespace & Git Diff Hygiene Check
git -c core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol diff --check
git status --short
```

### 5.2 Invalidation Conditions
The verification is invalidated and must be rejected if:
1. Bare `pytest` is run instead of `tools/run-offline.ps1` (violates `conftest.py` policy).
2. Any test triggers an external socket connection, DNS lookup, or HTTP request (`NetworkForbidden` or Node network guard raised).
3. Any test causes `provider_client.get_client` to be called without an explicit injected mock (`forbid_unconfigured_provider` fails).
4. `tree_of_thought` execution mode attempts to instantiate or call an LLM client.
5. Existing 148 backend tests or 111 frontend tests regress.
6. `schema.py` changes break existing `runConversation`, `submitFeedback`, or `hello` contracts.
7. Any file in `.agents/teamwork/` is used for non-metadata files (source code, tests, or application data).
