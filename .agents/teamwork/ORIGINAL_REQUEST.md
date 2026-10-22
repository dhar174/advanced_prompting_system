# Original User Request

## 2026-10-22T05:01:46Z

# Antigravity Execution Plan & Coordinator Prompt: Track 1 — End-to-End Execution Modes & UI Integration (Issue #35 Phases 3–5 & Issue #21)

> **Target Repository**: `advanced_prompting_system`
> **Target Primary Agent**: `aps-repo-coordinator` (Google Antigravity 2.0+ Lead Orchestrator & Single Production Writer)
> **Scope**: Issue #35 (Phases 3–5) & Issue #21 — Shared Run Contracts & Capability Registry, GraphQL API & UI Mode Selection, and Reasoning Diagnostics & Structured Output Rendering
> **Baseline Preservation**: 148/148 backend offline tests (`./tools/run-offline.ps1`) + 111/111 frontend Vitest tests (`./tools/run-frontend-offline.ps1`)

## How to Use This Document

Copy the entire **Prompt Block for `aps-repo-coordinator`** below and paste it directly into Antigravity with `aps-repo-coordinator` selected as the active agent (or invoke via `/goal` or `/plan`). The prompt is pre-wired to trigger the repository's mandatory 5-stage subagent orchestration lifecycle, enforce the one-production-writer policy, and preserve all PR #34 / Issue #35 offline qualification guards.

## Prompt Block for `aps-repo-coordinator`

```markdown
@aps-repo-coordinator Execute Track 1: End-to-End Execution Modes & UI Integration (Issue #35 Phases 3–5 & Issue #21) across the Python backend, Strawberry GraphQL API, and React 19 + TypeScript frontend.

You MUST operate strictly as the Single Production Writer under the repository's 5-Stage Orchestration Lifecycle defined in `AGENTS.md`, `GEMINI.md`, and `.agents/rules/team-coordination.md`. Read `AGENTS.md` and `docs/pr34-offline-qualification.md` first, delegate read-only discovery, contract auditing, domain analysis, and adversarial QA to your specialized subagents via `invoke_subagent`, synthesize their findings into an `implementation_plan.md`, and apply all production code and test edits yourself.

---

### 1. Mission & Track 1 Deliverables

Complete the three high-priority phases of Track 1 without breaking any existing offline tests (148 backend, 111 frontend) or cross-module contracts:

1. **Phase 3 — Shared Run Contracts & Capability Registry (`execution_modes.py`)**:
   - Define strongly typed Pydantic v2 schemas: `RunRequest`, `RunResult`, and `RunEvent` (along with supporting diagnostic schemas `PlanStepDiagnostic`, `SubtaskDiagnostic`, `ReflectionDiagnostic`, `AgentAttribution`, `RunDiagnostics`, and `ExecutionModeCapability`).
   - Implement a unified backend capability registry (`CAPABILITY_REGISTRY`) and dispatcher (`execute_run(request: RunRequest) -> RunResult`) defining three intentional execution modes:
     - `collaborative`: Multi-agent round deliberation via `conversation_manager.run_conversation`.
     - `advanced_pipeline`: Standalone task-planning and multi-strategy reasoning pipeline via `AdvancedPromptEngineer.main`.
     - `tree_of_thought`: Placeholder explicitly gated for Issue #36 (`enabled=False`, `gated_issue="Issue #36"`), returning a deterministic gated `RunResult` / `ErrorResult` without invoking any LLM or network call.

2. **Phase 4 — GraphQL API & UI Mode Selection (`schema.py` & `collaborative-assistant-frontend/`, Issue #35 Phase 4 & Issue #21)**:
   - Extend `schema.py` (while keeping `Query.hello`, `Mutation.run_conversation`, and `Mutation.submit_feedback` 100% backward-compatible) with:
     - `Query.execution_modes` (`executionModes`) returning registered mode capabilities and gating metadata.
     - `Mutation.run_execution` (`runExecution`) accepting unified run requests across modes, plus `Mutation.run_advanced_pipeline` (`runAdvancedPipeline`) for direct standalone pipeline execution.
   - Update the React 19 + TypeScript frontend to add an Execution Mode Selector (`Collaborative Multi-Agent` vs. `Standalone Pipeline`, plus the visibly gated `Tree of Thoughts (Issue #36)` badge) that dynamically adapts configuration controls and dispatches the selected execution path.

3. **Phase 5 — Reasoning Diagnostics & Structured Output Rendering (`collaborative-assistant-frontend/` & backend adapters, Issue #35 Phase 5 & Issue #21)**:
   - Report and render intermediate `PlanStep` / `Subtask` execution status, step thoughts and reflections with reward scores, multi-agent personality attributions, complexity scores, and ordered `RunEvent` telemetry in a dedicated `ReasoningDiagnosticsPanel` component.
   - Render final outputs according to `OutputType` (`output_type` and `file_extension`) via a `StructuredOutputRenderer` component supporting syntax-formatted code/script blocks (`Python Script`, `Code Snippet`, `JSON`, `HTML`, `CSV`) with language/extension badges and copy-to-clipboard, structured Markdown rendering (`Simple Concise Answer`, `Text File`, `PDF` text preview), and plain-text fallback.

---

### 2. Non-Negotiable Architectural & Safety Invariants

1. **One-Production-Writer Discipline**:
   - Only `aps-repo-coordinator` writes to production and test files (`memory-steward` updates `memory-bank/` only at Stage 5 closeout).
2. **Lazy SDK & Import-Time Purity**:
   - Importing `execution_modes`, `advanced_prompting`, `conversation_manager`, `complexity_measures`, `app`, or `schema` must NEVER instantiate `openai.OpenAI()`, require `OPENAI_API_KEY`, download NLTK/HuggingFace/tiktoken assets, or eagerly call `spacy.load("en_core_web_sm")`.
   - Route all LLM calls through `provider_client.LazyClient`.
   - Preserve lazy `get_nlp_spacy()` resolution across all 5 direct spaCy consumers in `complexity_measures.py` (`is_real_words`, `is_complex_graph`, `is_complex_recursive`, `is_complex_ast`, `is_complex_llm`).
3. **Backward Compatibility of Existing Public Signatures**:
   - `conversation_manager.run_conversation(problem_statement, selected_personalities, lead_personality, num_rounds=3, context=None)` MUST continue returning the exact 3-tuple `(conversation_history, questions_asked, final_output)`.
   - `AdvancedPromptEngineer.main(self, task: str) -> Interaction` MUST continue returning an `Interaction` instance and passing explicit keyword arguments, the actual `PlanStep` object, and an integer collaborative budget into `invoke_collaborative_reasoning` / `collaborative_reasoning_main`.
   - `schema.Mutation.run_conversation` (`runConversation`) and `schema.Mutation.submit_feedback` (`submitFeedback`) MUST preserve their exact GraphQL field names and behavior for `test_graphql_contract.py` and existing `useRunConversation` / `useSubmitFeedback` hooks.
   - Default UI state in `ConversationPage.tsx` and `AssistantConfigPanel.tsx` MUST remain `collaborative` mode with existing labels/controls accessible so all 111 existing Vitest tests continue passing without regression.
4. **Resilient Voting & Error Semantics**:
   - Never check `if vote:` on `BinaryVote`, `ConfidenceVote`, or `ErrorResult` (all Pydantic `BaseModel` instances are truthy). Always check `isinstance(vote, BinaryVote) and vote.vote is True`.
   - `normalize_message()` must continue rejecting `ErrorResult` with `ValueError`.
5. **Offline Guard & File Hygiene**:
   - Never weaken `conftest.py`, `tools/offline/sitecustomize.py`, or `tools/offline/node-network-guard.cjs`.
   - Always execute backend tests via `./tools/run-offline.ps1` and frontend checks via `./tools/run-frontend-offline.ps1`.
   - Never wire imports to historical/non-certifying files (`test_b.py`, `test_c.py`, `output_generator_backup.py`, `output_generator_fixed.py`, `historical_verification/`, `finance_manager_final_a.py`, `personal_finance_*.py`, `meeting_plan.py`, `project_visualizations_plan.py`, `mediator_summary_next_steps.py`).

---

### 3. Stage-by-Stage Subagent Orchestration Plan

Execute the following 5 stages in order. Do not poll subagents in a loop; await Antigravity's reactive wakeup after each `invoke_subagent` turn.

#### Stage 1: Pre-Flight Context Recovery (`memory-scout`)
- **Action**: Invoke `memory-scout` with task context: `"Track 1: Issue #35 Phases 3–5 & Issue #21 — Shared Run Contracts (RunRequest, RunResult, RunEvent), Execution Mode Capability Registry (collaborative, advanced_pipeline, tree_of_thought), GraphQL API & React UI Mode Selection, and Reasoning Diagnostics & OutputType Rendering."`
- **Deliverable**: Memory Brief summarizing prior PR #34 / Issue #35 Phase 2 decisions, baselines (148 backend, 111 frontend), and open lint/contract notes.

#### Stage 2: Cross-Module Contract Scoping (`architecture-contract-scout`)
- **Action**: Invoke `architecture-contract-scout` to audit the exact seams across:
  - `advanced_prompting.py` (`PromptEngineeringConfig` lines 1354–1434, `Reflection` lines 1460–1483, `Step` lines 1485–1613, `FinalStepOutput` lines 1615–1650, `FinalPlanStepOutput` lines 1652–1663, `Task` lines 1665–1744, `Interaction` lines 1746–1900, `AdvancedPromptEngineer.determine_output_type_from_content` lines 1922–1968, `invoke_collaborative_reasoning` lines 1969–2018, `select_personalities_for_planstep` lines 2019–2045, `main` lines 7612–8037).
  - `conversation_manager.py` (`ErrorResult` lines 59–65, `OutputType` lines 114–120, `ConversationMemory` lines 290–572, `output_type_determination` lines 1214–1240, `run_conversation` lines 2394–3433).
  - `complexity_measures.py` (`Subtask`, `PlanStep`, `Plan` lines 724–774).
  - `schema.py` (lines 1–117) and `app.py` (lines 1–43).
- **Deliverable**: Architecture & Contract Brief + coordinator-authored `implementation_plan.md` locking in exact schemas before modifying production code.

#### Stage 3: Domain Specialist Delegation (Wave 3A Parallel -> Wave 3B Sequential)

- **Wave 3A (Dispatch in Parallel)**:
  1. **`prompt-reasoning-architect`**:
     - Inspect `advanced_prompting.py` (`AdvancedPromptEngineer.main`, `Interaction`, `Task`, `Step`, `Reflection`, `FinalStepOutput`, `FinalPlanStepOutput`, `PromptEngineeringConfig`).
     - Specify the exact adapter logic for `advanced_pipeline` mode so `execute_run` can invoke `AdvancedPromptEngineer(config).main(task)` (or a thin non-breaking instrumentation hook on `AdvancedPromptEngineer`), extract `task_object.complexity`, `task_object.plan` (`PlanStep` and `Subtask` items), `interaction.steps`, `interaction.reflections`, `interaction.final_reward`, `task_object.output_type`, `last_conversation_history`, and `select_personalities_for_planstep` attributions, and map them cleanly into `RunResult`, `RunDiagnostics`, and ordered `RunEvent` objects without mutating `main(task) -> Interaction` return semantics.
  2. **`consensus-conversation-specialist`**:
     - Inspect `conversation_manager.py` (`run_conversation`, `OutputType`, `ErrorResult`, `ConversationMemory`).
     - Specify how `collaborative` mode in `execution_modes.py` wraps `run_conversation`, normalizes `conversation_history` messages, extracts round-by-round speaker attributions (`role`, `name`, `round`), maps `questions_asked`, resolves the final `OutputType` (via `determine_output_type_from_content` or `DEFAULT_OUTPUT_TYPE` / content sniffing on `final_output`), and emits `RunEvent` items while leaving `run_conversation`'s 3-tuple return unchanged.
  3. **`complexity-nlp-engineer`**:
     - Inspect `complexity_measures.py` (`Plan`, `PlanStep`, `Subtask`, `assess_complexity` seam) to verify serialization of `PlanStep` (`step_number`, `step_name`, `step_description`, `description`, `step_explanation`, `step_output`, `completed`, `subtasks`) and `Subtask` into `PlanStepDiagnostic` / `SubtaskDiagnostic` and ensure zero eager spaCy/NLTK side effects when constructing diagnostics.
  4. **`offline-qa-gatekeeper`**:
     - Design the red-green offline test matrix covering Phase 3 (`test_execution_modes.py`), Phase 4 (`test_graphql_contract.py` + frontend hook/mode selector tests), and Phase 5 (diagnostics + `OutputType` renderer tests), including adversarial cases (`tree_of_thought` gated call must not touch `provider_client.get_client`, `ErrorResult` handling, empty/malformed plans, `agents=0` vs `agents>0` in `advanced_pipeline`).

- **Wave 3B (Dispatch after Wave 3A contract synthesis)**:
  5. **`graphql-frontend-specialist`**:
     - Inspect `schema.py` and `collaborative-assistant-frontend/src/`.
     - Review the finalized `RunRequest`, `RunResult`, `RunEvent`, and `RunDiagnostics` schemas from Wave 3A and design:
       - Exact Strawberry GraphQL input/output types, `Query.execution_modes`, `Mutation.run_execution`, and `Mutation.run_advanced_pipeline` in `schema.py`.
       - Frontend TypeScript interfaces (`graphqlTypes.ts`), GraphQL operations (`conversationMutations.ts` / `executionOperations.ts`), custom hooks (`useRunExecution.ts`, `useExecutionModes.ts`), UI mode selector in `AssistantConfigPanel.tsx`, `ReasoningDiagnosticsPanel.tsx`, `StructuredOutputRenderer.tsx`, and integration in `ConversationPage.tsx` and `FeedbackDisplay.tsx`.

#### Stage 4: Guarded Offline Qualification & Adversarial Review (`offline-qa-gatekeeper`)
- **Action**:
  - Run focused and full backend offline test suites via `./tools/run-offline.ps1`.
  - Run frontend guard, test, build, and lint checks via `./tools/run-frontend-offline.ps1`.
  - Dispatch `offline-qa-gatekeeper` for independent adversarial audit of all diffs and test execution logs.

#### Stage 5: Post-Verification Knowledge Closeout (`memory-steward`)
- **Action**: Once Stage 4 passes with 0 failures and `READY` verdict from `offline-qa-gatekeeper`, invoke `memory-steward` to record the Track 1 architectural contracts and updated test counts to `mem0ry4ai` (`project:advanced_prompting_system`) and update `memory-bank/activeContext.md`, `memory-bank/progress.md`, and `memory-bank/systemPatterns.md`.

---

### 4. Detailed Technical Blueprint (Phases 3, 4, and 5)

#### Phase 3 Specification: Shared Run Contracts & Capability Registry (`execution_modes.py`)

Create `execution_modes.py` at the repository root (with lazy imports of `conversation_manager` and `advanced_prompting` inside execution handlers or preserving `LazyClient` import purity).

1. **Typed Pydantic v2 Schemas (`execution_modes.py`)**:
   - `ExecutionModeId = Literal["collaborative", "advanced_pipeline", "tree_of_thought"]`
   - `RunEventType = Literal["mode_selected", "problem_defined", "complexity_assessed", "plan_generated", "plan_step_started", "plan_step_completed", "agent_contribution", "reflection_recorded", "output_finalized", "mode_gated", "run_error"]`
   - `RunEvent(BaseModel)`:
     - `event_id: str`
     - `event_type: str` (`RunEventType`)
     - `timestamp: str` (ISO-8601 UTC string)
     - `mode: str` (`ExecutionModeId`)
     - `step_number: Optional[int] = None`
     - `plan_step_number: Optional[int] = None`
     - `agent_name: Optional[str] = None`
     - `summary: str`
     - `details: Dict[str, Any] = Field(default_factory=dict)`
   - `SubtaskDiagnostic(BaseModel)`:
     - `subtask_number: int`
     - `subtask_name: str`
     - `subtask_description: str`
     - `subtask_output: str = ""`
     - `completed: bool = False`
   - `PlanStepDiagnostic(BaseModel)`:
     - `step_number: int`
     - `step_name: str`
     - `step_description: str`
     - `step_explanation: str = ""`
     - `step_output: str = ""`
     - `completed: bool = False`
     - `assigned_agents: List[str] = Field(default_factory=list)`
     - `subtasks: List[SubtaskDiagnostic] = Field(default_factory=list)`
   - `ReflectionDiagnostic(BaseModel)`:
     - `step_number: int`
     - `plan_step_number: Optional[int] = None`
     - `step_description: str = ""`
     - `thoughts: Optional[str] = None`
     - `content: str`
     - `reward: float`
     - `remaining_budget: int = 0`
   - `AgentAttribution(BaseModel)`:
     - `agent_name: str`
     - `role: str = "assistant"`
     - `round_or_step: int = 1`
     - `contribution_summary: str`
   - `RunDiagnostics(BaseModel)`:
     - `complexity_score: Optional[float] = None`
     - `initial_budget: Optional[int] = None`
     - `adjusted_budget: Optional[int] = None`
     - `final_reward: Optional[float] = None`
     - `plan_steps: List[PlanStepDiagnostic] = Field(default_factory=list)`
     - `reflections: List[ReflectionDiagnostic] = Field(default_factory=list)`
     - `agent_attributions: List[AgentAttribution] = Field(default_factory=list)`
   - `RunRequest(BaseModel)`:
     - `mode: ExecutionModeId = "collaborative"`
     - `task_prompt: Optional[str] = None`
     - `conversation: List[Dict[str, Any]] = Field(default_factory=list)`
     - `assistant_personalities: List[str] = Field(default_factory=lambda: ["Helpful Assistant"])`
     - `lead_personality: str = "Helpful Assistant"`
     - `num_rounds: int = 3`
     - `max_steps: int = 20`
     - `initial_budget: int = 20`
     - `agents: int = 3`
     - `model: str = "gpt-4o-mini"`
     - `temperature: float = 0.7`
     - `context: Optional[str] = None`
     - Include a `@model_validator(mode="after")` that derives `task_prompt` from the latest user entry in `conversation` if `task_prompt` is not explicitly provided, and populates a single user message in `conversation` if only `task_prompt` is provided.
   - `RunResult(BaseModel)`:
     - `mode: ExecutionModeId`
     - `status: Literal["completed", "gated", "error"]`
     - `final_output: str = ""`
     - `output_type: OutputType = Field(default_factory=lambda: OutputType(output_type="Simple Concise Answer", file_extension=".txt"))`
     - `conversation: List[Dict[str, Any]] = Field(default_factory=list)`
     - `questions: List[Dict[str, str]] = Field(default_factory=list)`
     - `diagnostics: RunDiagnostics = Field(default_factory=RunDiagnostics)`
     - `events: List[RunEvent] = Field(default_factory=list)`
     - `error: Optional[ErrorResult] = None`
   - `ExecutionModeCapability(BaseModel)`:
     - `id: ExecutionModeId`
     - `label: str`
     - `description: str`
     - `enabled: bool`
     - `gated_issue: Optional[str] = None`
     - `supports_multi_agent: bool`
     - `supports_step_planning: bool`

2. **Unified Capability Registry & Mode Handlers (`execution_modes.py`)**:
   - `CAPABILITY_REGISTRY: Dict[ExecutionModeId, ExecutionModeCapability]`:
     - `"collaborative"`: `enabled=True`, `gated_issue=None`, `supports_multi_agent=True`, `supports_step_planning=False`, backed by `conversation_manager.run_conversation`.
     - `"advanced_pipeline"`: `enabled=True`, `gated_issue=None`, `supports_multi_agent=True`, `supports_step_planning=True`, backed by `AdvancedPromptEngineer.main`.
     - `"tree_of_thought"`: `enabled=False`, `gated_issue="Issue #36"`, `supports_multi_agent=False`, `supports_step_planning=True`, placeholder gated for Issue #36.
   - `execute_run(request: RunRequest) -> RunResult`:
     - If `request.mode == "tree_of_thought"`:
       - Do NOT call `get_client()`, `run_conversation`, or `AdvancedPromptEngineer`.
       - Return `RunResult(mode="tree_of_thought", status="gated", final_output="Tree of Thoughts execution mode is currently gated pending Issue #36.", output_type=OutputType(output_type="Simple Concise Answer", file_extension=".txt"), error=ErrorResult(error_type="ModeGatedError", error_message="Execution mode 'tree_of_thought' is gated for Issue #36."), events=[RunEvent(event_type="mode_gated", ...)])`.
     - If `request.mode == "collaborative"`:
       - Invoke `conversation_manager.run_conversation(request.conversation, request.assistant_personalities, request.lead_personality, request.num_rounds, context=request.context)`.
       - Unpack `(conversation_history, questions_asked, final_output)`.
       - Build `AgentAttribution` entries from `conversation_history` (excluding raw system instructions) and `RunEvent` items (`mode_selected`, `agent_contribution`, `output_finalized`).
       - Resolve `output_type`: if `final_output` is a file path or content string, infer `OutputType` via `AdvancedPromptEngineer(PromptEngineeringConfig()).determine_output_type_from_content(final_output, final_output if os.path.exists(str(final_output)) else "", None)` or parse `"Output Type:"` from history if present, defaulting cleanly to `OutputType(output_type="Simple Concise Answer", file_extension=".txt")`.
       - If `final_output` points to an existing generated file on disk, optionally read its content or surface the content clearly while keeping `final_output` non-None string.
     - If `request.mode == "advanced_pipeline"`:
       - Build `PromptEngineeringConfig(max_steps=request.max_steps, initial_budget=request.initial_budget, model=request.model, temperature=request.temperature, agents=request.agents)`.
       - Instantiate `engineer = AdvancedPromptEngineer(config)` and execute `interaction = engineer.main(request.task_prompt)`.
       - Extract `task_obj = interaction.task` (or `engineer.task_object`):
         - `complexity_score = float(getattr(task_obj, "complexity", 0.0) or 0.0)`
         - `plan_steps`: Map `task_obj.plan.steps` (`PlanStep`) and their `subtasks` (`Subtask`) into `PlanStepDiagnostic` and `SubtaskDiagnostic`, attaching `assigned_agents=engineer.select_personalities_for_planstep(pstep, task_obj) if config.agents > 0 else []`.
         - `reflections`: Map `interaction.steps` and `interaction.reflections` into `ReflectionDiagnostic` (matching `step.step_number`, `step.plan_step_number`, `step.description`, `step.thoughts`, `step.remaining_budget`, and `reflection.content` / `reflection.reward`).
         - `agent_attributions`: Combine `engineer.last_conversation_history` (when collaborative sub-steps ran) and plan-step personality selections into `AgentAttribution` records.
         - `final_output`: Use `interaction.answer` if non-empty; otherwise synthesize from `interaction.planstep_outputs` or `interaction.steps[-1].description` so `final_output` is always a meaningful string.
         - `output_type`: Use `task_obj.output_type` if it is a valid `OutputType`, or refine via `engineer.determine_output_type_from_content(final_output, "", task_obj)`.
         - `conversation`: Synthesize a readable transcript of the pipeline execution (user prompt + step/agent milestone messages + final output) so the chat thread in the UI also displays the pipeline progression naturally alongside the structured diagnostics panel.
         - Emit ordered `RunEvent` records (`mode_selected`, `complexity_assessed`, `plan_generated`, `plan_step_started`/`plan_step_completed`, `reflection_recorded`, `output_finalized`).

---

#### Phase 4 Specification: GraphQL API & UI Mode Selection (Issue #35 Phase 4 & Issue #21)

1. **GraphQL Schema Extensions (`schema.py`)**:
   - Keep existing `ConversationInput`, `FeedbackInput`, `QuestionInput`, `Question`, `Conversation`, `Feedback`, `RunConversationResponse`, `FeedbackResponse`, `Query.hello`, `Mutation.run_conversation`, and `Mutation.submit_feedback` completely intact.
   - Add Strawberry GraphQL types:
     - `@strawberry.type class OutputTypeInfo`: `output_type: str`, `file_extension: str`
     - `@strawberry.type class SubtaskDiagnosticType`: `subtask_number: int`, `subtask_name: str`, `subtask_description: str`, `subtask_output: str`, `completed: bool`
     - `@strawberry.type class PlanStepDiagnosticType`: `step_number: int`, `step_name: str`, `step_description: str`, `step_explanation: str`, `step_output: str`, `completed: bool`, `assigned_agents: List[str]`, `subtasks: List[SubtaskDiagnosticType]`
     - `@strawberry.type class ReflectionDiagnosticType`: `step_number: int`, `plan_step_number: Optional[int]`, `step_description: str`, `thoughts: Optional[str]`, `content: str`, `reward: float`, `remaining_budget: int`
     - `@strawberry.type class AgentAttributionType`: `agent_name: str`, `role: str`, `round_or_step: int`, `contribution_summary: str`
     - `@strawberry.type class RunDiagnosticsType`: `complexity_score: Optional[float]`, `initial_budget: Optional[int]`, `adjusted_budget: Optional[int]`, `final_reward: Optional[float]`, `plan_steps: List[PlanStepDiagnosticType]`, `reflections: List[ReflectionDiagnosticType]`, `agent_attributions: List[AgentAttributionType]`
     - `@strawberry.type class RunEventTypeGQL`: `event_id: str`, `event_type: str`, `timestamp: str`, `mode: str`, `step_number: Optional[int]`, `plan_step_number: Optional[int]`, `agent_name: Optional[str]`, `summary: str`
     - `@strawberry.type class ExecutionModeCapabilityType`: `id: str`, `label: str`, `description: str`, `enabled: bool`, `gated_issue: Optional[str]`, `supports_multi_agent: bool`, `supports_step_planning: bool`
     - `@strawberry.input class RunExecutionInput`:
       - `mode: str = "collaborative"`
       - `task_prompt: Optional[str] = None`
       - `conversation: Optional[List[ConversationInput]] = None`
       - `assistant_personalities: Optional[List[str]] = None`
       - `lead_personality: Optional[str] = "Helpful Assistant"`
       - `num_rounds: int = 3`
       - `max_steps: int = 20`
       - `initial_budget: int = 20`
       - `agents: int = 3`
       - `model: str = "gpt-4o-mini"`
     - `@strawberry.type class RunExecutionResponse`:
       - `mode: str`
       - `status: str`
       - `conversation: List[Conversation]`
       - `questions: Optional[List[Question]]`
       - `final_output: str`
       - `output_type: OutputTypeInfo`
       - `diagnostics: RunDiagnosticsType`
       - `events: List[RunEventTypeGQL]`
       - `error_message: Optional[str] = None`
   - Add GraphQL Query & Mutations:
     - `Query.execution_modes(self) -> List[ExecutionModeCapabilityType]`
     - `Mutation.run_execution(self, input: RunExecutionInput) -> RunExecutionResponse`
     - `Mutation.run_advanced_pipeline(self, task_prompt: str, max_steps: int = 20, initial_budget: int = 20, agents: int = 3, model: str = "gpt-4o-mini") -> RunExecutionResponse`

2. **React Frontend Execution Mode Selection (`collaborative-assistant-frontend/src/`)**:
   - Extend `src/graphql/graphqlTypes.ts` with TypeScript interfaces for `ExecutionModeId`, `ExecutionModeCapabilityType`, `OutputTypeInfo`, `PlanStepDiagnosticType`, `SubtaskDiagnosticType`, `ReflectionDiagnosticType`, `AgentAttributionType`, `RunDiagnosticsType`, `RunEventType`, `RunExecutionInputType`, and `RunExecutionResponseType`.
   - Add GraphQL queries/mutations in `src/graphql/mutations/conversationMutations.ts` (or `src/graphql/executionOperations.ts`): `GET_EXECUTION_MODES`, `RUN_EXECUTION`, and `RUN_ADVANCED_PIPELINE`.
   - Create `src/hooks/useRunExecution.ts` (and keep `src/hooks/useRunConversation.ts` intact).
   - Update `src/components/AssistantConfigPanel.tsx`:
     - Accept optional props (with safe defaults so existing unit tests that render `<AssistantConfigPanel ... />` with only the original 8 props pass without modification!):
       - `executionMode?: ExecutionModeId` (default `'collaborative'`)
       - `onExecutionModeChange?: (mode: ExecutionModeId) => void`
       - `maxSteps?: number` (default `10`)
       - `onMaxStepsChange?: (steps: number) => void`
       - `agentCount?: number` (default `3`)
       - `onAgentCountChange?: (agents: number) => void`
     - Render an accessible **Execution Mode** selector (`role="radiogroup"` or `<select id="execution-mode">` / segmented toggle buttons) showing:
       - **Collaborative Debate** (`collaborative`) — Multi-agent round deliberation & consensus voting.
       - **Standalone Pipeline** (`advanced_pipeline`) — Complexity-driven step planning, reflections & reward judging.
       - **Tree of Thoughts** (`tree_of_thought`, disabled with `"Issue #36"` badge).
     - When `executionMode === 'collaborative'`, display the existing Personality checkboxes, Lead Personality `<select>`, and Number of Rounds `<Input>`, and keep button text `'Start New Conversation'` / `'Conversation in Progress...'`.
     - When `executionMode === 'advanced_pipeline'`, display pipeline configuration controls (`Max Reasoning Steps`, `Collaborative Agents per Step`) and allow starting/running the pipeline (`'Start Pipeline Session'` or unified start button).

---

#### Phase 5 Specification: Reasoning Diagnostics & Structured Output Rendering (Issue #35 Phase 5 & Issue #21)

1. **Reasoning Diagnostics Component (`src/components/ReasoningDiagnosticsPanel.tsx`)**:
   - Create a dedicated, collapsible/tabbed diagnostic panel rendered inside `ConversationPage.tsx` (or `FeedbackDisplay.tsx`) whenever `diagnostics` or `events` are present:
     - **Complexity & Reward Summary Bar**: Displays `complexityScore` (formatted numeric score + visual badge: Low / Moderate / High), `initialBudget` -> `adjustedBudget`, and `finalReward` confidence score.
     - **Execution Plan & Subtasks (`planSteps`)**: Displays each `PlanStepDiagnostic` with step number, name, description, expected output, completion badge (`Completed` / `In Progress`), assigned agent pills (`assignedAgents`), and nested `SubtaskDiagnostic` checklist items.
     - **Reasoning Steps & Reflections (`reflections`)**: Displays intermediate `Step` thoughts (`<thinking>`), step descriptions, remaining budget countdown, and `Reflection` feedback with color-coded reward badge (`reward`).
     - **Agent Attributions (`agentAttributions`) & Event Log (`events`)**: Displays speaker/agent attribution pills and the chronological `RunEvent` stream.

2. **OutputType-Aware Renderer (`src/components/StructuredOutputRenderer.tsx`)**:
   - Create `StructuredOutputRenderer.tsx` and integrate it into `FeedbackDisplay.tsx` (while keeping the heading `"Final Output:"` so existing `FeedbackDisplay` / `ConversationPage` tests continue to find it!):
     - Inspect `outputType` (`OutputTypeInfo`, defaulting to `{ outputType: 'Simple Concise Answer', fileExtension: '.txt' }`).
     - Display a metadata header pill showing `outputType.outputType` and `outputType.fileExtension` alongside a `Copy` button.
     - **Code & Structured Data Modes** (`Python Script`, `Code Snippet`, `JSON`, `HTML`, `CSV`, or extensions `.py`, `.json`, `.html`, `.csv`, `.js`, `.ts`):
       - Render inside a dark-themed syntax/code container (`<pre><code data-testid="structured-code-output">...</code></pre>`) with line numbers or language badge.
       - For `JSON`, pretty-print valid JSON safely (`JSON.stringify(JSON.parse(content), null, 2)` fallback to raw string).
       - For `CSV`, render both a clean tabular preview (if comma-separated rows parse cleanly) and raw CSV source view.
     - **Markdown & Document Modes** (`Simple Concise Answer`, `Text File`, `PDF`):
       - Parse and render fenced code blocks (` ```lang ... ``` `), headings, bullet lists, and bold text cleanly while preserving `whitespace-pre-wrap` readability.

3. **Frontend Lint Debt Cleanup (Issue #35)**:
   - While touching `collaborative-assistant-frontend/src/`, resolve the tracked ESLint issues (unused catch bindings, explicit `any` types in tests, empty interfaces in `src/components/ui/`) so `./tools/run-frontend-offline.ps1 -Action lint` passes cleanly alongside `test` and `build`.

---

### 5. Mandatory Verification Commands & Acceptance Gate

Before invoking `memory-steward` in Stage 5, execute and verify all of the following commands from the repository root:

```powershell
# 1. Backend Offline Test Suite (must pass all 148 existing tests + new Track 1 contract tests)
./tools/run-offline.ps1 -PytestArgs '--collect-only','-q'
./tools/run-offline.ps1 -PytestArgs 'test_execution_modes.py','test_graphql_contract.py','test_smoke.py','-v'
./tools/run-offline.ps1 -PytestArgs '-q'

# 2. Frontend Offline Guard, Unit/Integration Tests, Build, and Lint
./tools/run-frontend-offline.ps1 -Action guard
./tools/run-frontend-offline.ps1 -Action test
./tools/run-frontend-offline.ps1 -Action build
./tools/run-frontend-offline.ps1 -Action lint

# 3. Git Diff & Whitespace Hygiene
git -c core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol diff --check
git status --short
```

All tests must pass with zero network calls, zero unconfigured provider calls, and `offline-qa-gatekeeper` verdict `READY`.

---

## Architectural Reference & Seam Map for Track 1

| Layer / File | Existing State in Repo | Track 1 Target State (Phases 3–5 & Issue #21) | Owning Subagent(s) |
| :--- | :--- | :--- | :--- |
| **`execution_modes.py` (New)** | Does not exist; `schema.py` calls `conversation_manager.run_conversation` directly. | Defines `RunRequest`, `RunResult`, `RunEvent`, `RunDiagnostics`, `PlanStepDiagnostic`, `SubtaskDiagnostic`, `ReflectionDiagnostic`, `AgentAttribution`, `ExecutionModeCapability`, `CAPABILITY_REGISTRY` (`collaborative`, `advanced_pipeline`, `tree_of_thought`), and `execute_run()`. | `architecture-contract-scout`, `prompt-reasoning-architect`, `consensus-conversation-specialist` |
| **`advanced_prompting.py`** | `AdvancedPromptEngineer.main(task)` (line 7612) returns `Interaction` with `task` (`Task.complexity`, `Task.plan`, `Task.output_type`), `steps` (`Step`), `reflections` (`Reflection`), `planstep_outputs`, `answer`, `final_reward`. | Preserves `main(task) -> Interaction` contract; surfaces `task_object`, `last_conversation_history`, and `select_personalities_for_planstep` cleanly to `execution_modes.py` for `advanced_pipeline` runs. | `prompt-reasoning-architect`, `complexity-nlp-engineer` |
| **`conversation_manager.py`** | `run_conversation(...)` (line 2394) returns 3-tuple `(conversation_history, questions_asked, final_output)`. Defines `OutputType` (line 114) and `ErrorResult` (line 59). | Preserves 3-tuple return contract; `execution_modes.py` wraps `run_conversation` for `collaborative` mode and extracts speaker attributions, questions, and `OutputType`. | `consensus-conversation-specialist` |
| **`schema.py`** | Exposes `Query.hello`, `Mutation.run_conversation`, `Mutation.submit_feedback` (lines 56–117). | Retains existing fields unchanged; adds `Query.execution_modes`, `Mutation.run_execution`, `Mutation.run_advanced_pipeline`, and Strawberry diagnostic/event/output types. | `graphql-frontend-specialist` |
| **`ConversationPage.tsx` & `AssistantConfigPanel.tsx`** | Hardcoded to `useRunConversation` with personality checkboxes, lead dropdown, and rounds input. | Adds Execution Mode selector (`collaborative`, `advanced_pipeline`, gated `tree_of_thought`), mode-specific config inputs, and dispatches either `runConversation` or `runExecution` / `runAdvancedPipeline` while preserving default collaborative test behavior. | `graphql-frontend-specialist` |
| **`ReasoningDiagnosticsPanel.tsx` & `StructuredOutputRenderer.tsx` (New)** | `FeedbackDisplay.tsx` only renders `questions` and a plain-text `finalOutput` div. | Renders intermediate `PlanStep`/`Subtask` progress, `Reflection` rewards, `AgentAttribution` badges, `complexityScore`, `RunEvent` timeline, and `OutputType`-aware formatting (code blocks, JSON/CSV, Markdown, text). | `graphql-frontend-specialist` |
| **`test_execution_modes.py` (New) & `test_graphql_contract.py`** | 148 backend tests + 111 frontend Vitest tests. | Adds unit and contract tests for `RunRequest`/`RunResult`/`RunEvent`, all 3 registry modes (including gated `tree_of_thought`), new GraphQL queries/mutations, and frontend mode switching, diagnostics, and `OutputType` rendering. | `offline-qa-gatekeeper` |

## Subagent Delegation Topology (Quick Reference)

1. **`memory-scout` (Stage 1 — Pre-Flight Bookend)**:
   - Queries `mem0ry4ai` (`project:advanced_prompting_system`) and reads `memory-bank/activeContext.md` + `memory-bank/progress.md`.
2. **`architecture-contract-scout` (Stage 2 — Seam Audit)**:
   - Verifies that `execution_modes.py` introduces zero circular imports between `schema.py`, `conversation_manager.py`, `advanced_prompting.py`, and `complexity_measures.py` (note that `conversation_manager.py` line 17 currently has `from schema import ConversationInput` and `advanced_prompting.py` line 18 has `from conversation_manager import output_type_determination, OutputType` — `execution_modes.py` must import carefully or lazily to avoid circular import traps).
3. **`prompt-reasoning-architect` + `consensus-conversation-specialist` + `complexity-nlp-engineer` (Stage 3 — Wave 3A Parallel)**:
   - Design the exact data extraction mapping from `Interaction` / `Task` / `Plan` and `run_conversation` into `RunResult`, `RunDiagnostics`, and `RunEvent`.
4. **`graphql-frontend-specialist` (Stage 3 — Wave 3B Sequential)**:
   - Designs the Strawberry GraphQL types/resolvers and React 19 UI components (`AssistantConfigPanel`, `ReasoningDiagnosticsPanel`, `StructuredOutputRenderer`, `ConversationPage`) + ESLint fixes.
5. **`offline-qa-gatekeeper` (Stage 3 Test Design & Stage 4 Adversarial Release Gate)**:
   - Audits all changes under `./tools/run-offline.ps1` and `./tools/run-frontend-offline.ps1`.
6. **`memory-steward` (Stage 5 — Post-Flight Bookend)**:
   - Persists Track 1 completion state to `mem0ry4ai` and `memory-bank/`.

# The Above Plan is ground-truth for subagent delegation and execution flow. Use the Teamwork Project Prompt as the authoritative guide for implementation and verification based on the outlined stages and responsibilities of each subagent according to the plan above.

# Teamwork Project Prompt — Draft

> Status: Launched
> Goal: Craft prompt → get user approval → delegate to teamwork_preview
> Requested team: Full multi-agent team with domain specialists

Wire up `AdvancedPromptEngineer` multi-strategy reasoning execution modes into the Strawberry GraphQL API with structured observability metrics (latency, token expenditure, and execution traces) and provide React frontend controls for strategy selection and telemetry inspection.

Working directory: d:\projects\advanced_prompting_system\advanced_prompting_system
Integrity mode: development

## Requirements

### R1. GraphQL Reasoning & Telemetry Mutation
Provide a dedicated Strawberry GraphQL mutation (`runReasoningTask`) that accepts a task specification, a reasoning strategy selection (`CHAIN_OF_THOUGHT`, `LEAST_TO_MOST`, `PROGRESSIVE_HINT`, `SELF_CONSISTENCY`, `DYNAMIC_CONFIDENCE_EXPLORATION`), and optional configuration parameters. The mutation must return the final output, intermediate steps, reflections, and structured execution telemetry (latency/duration, token expenditure, step counts, and request correlation ID).

### R2. Backend Observability & Workload Profiling
Replace unstructured console logging in the API and orchestration layers with structured JSON events. Propagate request correlation IDs throughout execution and collect telemetry metrics for agent debate rounds and reasoning pipeline steps.

### R3. Frontend Strategy Selection & Telemetry Visualization
Provide UI controls in the React/Apollo client application enabling users to configure and trigger reasoning tasks by strategy, view intermediate reasoning steps and reflections, and inspect execution telemetry metrics.

### R4. Offline Qualification & Architectural Invariants
All additions must maintain strict repository invariants: zero import-time side effects (no network calls, external downloads, or eager OpenAI client instantiations), fail-closed error handling, full compatibility with the existing 148 backend offline tests and 111 frontend tests, and new dedicated offline tests covering the new mutation and UI components.

## Verification Resources
- Backend offline test harness: `./tools/run-offline.ps1 -PytestArgs '-q'`
- Backend targeted contract suite: `./tools/run-offline.ps1 -PytestArgs 'test_graphql_contract.py','-v'`
- Frontend offline test runner: `./tools/run-frontend-offline.ps1 -Action test`
- Frontend offline production build: `./tools/run-frontend-offline.ps1 -Action build`
- Frontend offline network denial guard: `./tools/run-frontend-offline.ps1 -Action guard`

## Acceptance Criteria

### GraphQL & Backend Verification
- [ ] Strawberry GraphQL schema defines `runReasoningTask` mutation, `PromptStrategy` enum, `TaskInput`, and structured response types including `TelemetryMetrics`.
- [ ] Offline backend test suite passes completely with >= 150 tests passing and 0 failures via `./tools/run-offline.ps1 -PytestArgs '-q'`.
- [ ] Dedicated unit and contract tests in `test_graphql_contract.py` verify `runReasoningTask` and telemetry generation offline without external network dependencies.
- [ ] Zero import-time side effects verified: importing `schema`, `app`, `advanced_prompting`, or `conversation_manager` initiates no network requests and requires no API keys.

### Frontend & UI Verification
- [ ] Apollo client queries/mutations and React UI components provide strategy selection and telemetry display.
- [ ] Frontend Vitest test suite passes with 0 failures via `./tools/run-frontend-offline.ps1 -Action test`.
- [ ] Frontend production build succeeds cleanly via `./tools/run-frontend-offline.ps1 -Action build` with zero TypeScript or bundling errors.
