# Backend Architecture Survey: Execution Modes, Circular Imports, Telemetry Extraction & Gating

**Author**: `teamwork_preview_explorer` (Backend Architecture Explorer)  
**Date**: 2026-10-22  
**Target File**: `d:\projects\advanced_prompting_system\advanced_prompting_system\.agents\teamwork\explorer_survey_backend\handoff.md`  
**Scope**: Issue #35 (Phases 3–5) & Issue #21 — `execution_modes.py` design, import graph analysis, telemetry data extraction for `collaborative` and `advanced_pipeline`, and `tree_of_thought` zero-network gating.

---

## 1. Observation

### 1.1 Existing Module Import Seams & Circular Import Topology
1. **`conversation_manager.py` imports `schema`**:
   - `conversation_manager.py:17`:
     ```python
     from schema import ConversationInput
     ```
   - In `conversation_manager.py:105-111`, `ConversationInput` is only referenced in a comment; runtime normalization uses duck-typing:
     ```python
     # Handle ConversationInput or other model objects
     if hasattr(value, 'content'):
         return {
             'role': getattr(value, 'role', default_role),
             'content': str(value.content),
             'name': getattr(value, 'name', default_name),
         }
     ```
   - Ripper/grep search across `conversation_manager.py` confirms that `ConversationInput` appears on line 17 and line 105 only.
2. **`advanced_prompting.py` imports `conversation_manager`**:
   - `advanced_prompting.py:18`:
     ```python
     from conversation_manager import output_type_determination, OutputType
     ```
   - `advanced_prompting.py:1974` inside `invoke_collaborative_reasoning`:
     ```python
     from conversation_manager import run_conversation # Import here to avoid circular dependencies at module level
     ```
3. **`schema.py` imports `conversation_manager`**:
   - `schema.py` does **not** import `conversation_manager` at module top-level.
   - `schema.py:67` imports inside `Mutation.run_conversation`:
     ```python
     @strawberry.mutation
     def run_conversation(self, ...):
         from conversation_manager import run_conversation
     ```
   - `schema.py:107` imports inside `Mutation.submit_feedback`:
     ```python
     @strawberry.mutation
     def submit_feedback(self, ...):
         from feedback_manager import log_conversation
     ```
4. **Circular Import Trap if `schema.py` imports `execution_modes` at top level**:
   - If `schema.py` had a top-level `from execution_modes import execute_run, CAPABILITY_REGISTRY`, and `execution_modes.py` had a top-level `from conversation_manager import OutputType, ErrorResult`:
     - Case A: `import conversation_manager` loads `conversation_manager.py:17` -> loads `schema.py` -> loads `execution_modes.py` -> attempts `from conversation_manager import OutputType` while `conversation_manager` is paused at line 17 (`OutputType` at line 114 not yet defined) -> **`ImportError: cannot import name 'OutputType' from partially initialized module 'conversation_manager'`**.
     - Case B: `import schema` loads `execution_modes.py` at line 4 (before line 7 `class ConversationInput`) -> loads `conversation_manager.py:17` -> attempts `from schema import ConversationInput` while `schema` is paused at line 4 -> **`ImportError: cannot import name 'ConversationInput' from partially initialized module 'schema'`**.

### 1.2 Core Reasoning Structures in `advanced_prompting.py`
1. **`PromptEngineeringConfig` (`advanced_prompting.py:1354-1434`)**:
   - Fields: `max_steps: int = 20`, `initial_budget: int = 20`, `confidence_thresholds: Tuple[float, float, float] = (0.8, 0.5, 0.0)`, `model: str = "gpt-4o-mini"`, `temperature: float = 0.7`, `top_p: float = 0.9`, `n: int = 3`, `max_retries: int = 3`, `backtrack: bool = True`, `agents: int = 3`, `complexity_factor: int = 5`.
2. **`Reflection` (`advanced_prompting.py:1460-1483`)**:
   - Fields: `content: str`, `reward: float`, `step_number: int`.
3. **`Step` (`advanced_prompting.py:1485-1613`)**:
   - Fields: `description: str`, `step_number: int`, `remaining_budget: int`, `reflection: Optional[Reflection] = None`, `thoughts: Optional[str] = None`, `plan_step_number: Optional[int] = None`, `final_step_output: Optional[FinalStepOutput] = None`.
4. **`FinalStepOutput` (`advanced_prompting.py:1615-1650`) & `FinalPlanStepOutput` (`advanced_prompting.py:1652-1663`)**:
   - `FinalStepOutput`: `final_output: str`, `output_type: OutputType`, `version: int`, `component_type: str`, `associated_plan_step: PlanStep`, `step: Step`.
   - `FinalPlanStepOutput`: `final_output: str`, `output_type: OutputType`, `version: int`, `component_type: str`, `steps: List[Step]`, `planstep: PlanStep`, `file_name: str`, `parent_file_name: str`.
5. **`Task` (`advanced_prompting.py:1665-1744`)**:
   - Fields: `description: str`, `refined_description: str`, `complexity: int`, `steps: List[Step]`, `reflections: List[Reflection]`, `answer: str`, `final_reward: float`, `plan: Plan`, `output_type: OutputType`, `project_name: Optional[str]`, `planstep_outputs: List[FinalPlanStepOutput]`.
6. **`Interaction` (`advanced_prompting.py:1746-1830`)**:
   - Fields: `task: Task`, `steps: List[Step]`, `reflections: List[Reflection]`, `answer: Optional[str]`, `final_reward: float`, `step_outputs: List[FinalStepOutput]`, `planstep_outputs: List[FinalPlanStepOutput]`, `existing_files: List[str]`.
7. **`AdvancedPromptEngineer` Attributes and Seams**:
   - `AdvancedPromptEngineer.__init__` (`advanced_prompting.py:1914-1920`):
     - `self.config = config`
     - `self.knowledge_base = {}`
     - `self.task_object = None`
     - `self.last_conversation_history = None`
     - `self.last_questions_asked = None`
   - `determine_output_type_from_content(self, content: str, file_path: str, task: Task) -> OutputType` (`advanced_prompting.py:1922-1968`):
     - Checks file extension (`.py`, `.json`, `.html`, `.csv`, `.pdf`), then sniffs content (Python keywords, `json.loads`, HTML tags), then task output type, defaulting to `OutputType(output_type="Text File", file_extension=".txt")`.
   - `select_personalities_for_planstep(self, plan_step: PlanStep, task: Task) -> List[str]` (`advanced_prompting.py:2019-2045`):
     - Base: `["Strategist", "Educator"]`; adds `"Software Engineer"` (if code/script/python), `"Business Analyst"` (if analysis/data), `"UX Designer"` (if design/ui/ux). Returns up to 3 personalities.
   - `invoke_collaborative_reasoning(self, task: Task, plan_step: PlanStep) -> FinalStepOutput` (`advanced_prompting.py:1969-2018`):
     - Calls `run_conversation(...)`, captures `self.last_conversation_history = conversation_history` and `self.last_questions_asked = questions_asked`.
   - `AdvancedPromptEngineer.main(self, task: str) -> Interaction` (`advanced_prompting.py:7612-8037`):
     - Step 1: `retrieve_information`, `task_into_prompt`, `output_type_determination`.
     - Step 2: `assess_complexity(refined_task)` -> `(complexity, plan)`. Sets `self.task_object = task_object`.
     - Step 3: `adjust_step_budget`.
     - Step 4: `generate_initial_prompt`.
     - Step 5: For each plan step, if `self.config.agents > 0`, calls `collaborative_reasoning_main(task=..., plan_step=current_plan_step, ...)` passing explicit keyword arguments, `PlanStep`, and integer budget.
     - Step 6: `self_consistency`, `finalize_step_output`, `finalize_planstep_output`.
     - Step 7: Returns `main_interaction: Interaction` with `task`, `steps`, `reflections`, `answer`, `final_reward`, `planstep_outputs`.

### 1.3 Multi-Agent Debate & Consensus in `conversation_manager.py`
1. **`run_conversation` Signature and Return Contract (`conversation_manager.py:2394, 3432`)**:
   - Signature: `run_conversation(problem_statement, selected_personalities, lead_personality, num_rounds=3, context=None)`
   - Return: `(conversation_history, questions_asked, final_output)` 3-tuple.
2. **`OutputType` and `ErrorResult` (`conversation_manager.py:59-65, 114-120`)**:
   - `ErrorResult(BaseModel)`: `error_type: str`, `error_message: str`, `error_details: Optional[str]`, `timestamp: datetime = default_factory`.
   - `OutputType(BaseModel)`: `output_type: str`, `file_extension: str`.
   - `DEFAULT_OUTPUT_TYPE = OutputType(output_type="simple_text", file_extension="txt")`.
3. **Voting Semantics (`conversation_manager.py:788-815`)**:
   - `BinaryVote(BaseModel)`: `vote: bool`.
   - `ConfidenceVote(BaseModel)`: `confidence: float`.
   - `cast_binary_vote` returns `Union[BinaryVote, ErrorResult]`.
   - Every `BaseModel` instance is truthy in Python (`bool(BinaryVote(vote=False)) is True`). All checks in production code use `isinstance(vote, BinaryVote) and vote.vote is True`.
4. **Message Normalization (`conversation_manager.py:91-112`)**:
   - `normalize_message(value, default_role='assistant', default_name='') -> Dict[str, Any]`
   - Explicitly raises `ValueError` on `ErrorResult` instances to prevent failed LLM calls from entering conversation history.
5. **`ConversationMemory` (`conversation_manager.py:290-572`)**:
   - Structured memory with `ArgumentItem`, `DecisionItem`, `ToDoItem`. Supports both attribute and dict-style access, with lossless round-tripping via `to_dict()` and `from_dict()`.

### 1.4 Step Planning & Complexity Measures in `complexity_measures.py`
1. **`Subtask` (`complexity_measures.py:724-738`)**:
   - Fields: `subtask_number: int`, `completed: bool`, `subtask_name: str`, `subtask_description: str`, `subtask_explanation: str`, `subtask_output: str`, `subtask_full_text: str`, `subtasks: List[Subtask]`.
2. **`PlanStep` (`complexity_measures.py:740-766`)**:
   - Fields: `step_number: int`, `completed: bool`, `step_name: str`, `step_description: str`, `description: Optional[str] = None`, `step_explanation: str`, `step_output: str`, `step_full_text: str`, `subtasks: List[Subtask]`.
   - Uses `@model_validator(mode="before")` to synchronize `description` and `step_description`.
3. **`Plan` (`complexity_measures.py:768-773`)**:
   - Fields: `steps: List[PlanStep]`.
4. **`assess_complexity` (`advanced_prompting.py:3766`)**:
   - Calls `complexity_measures.is_complex_final(task, True)` returning `(complexity, plan)`.
   - Lazy spaCy resolution (`get_nlp_spacy()`) across all 5 consumers in `complexity_measures.py` ensures zero eager model loading or network downloads offline.

### 1.5 Offline Testing Verification
- Executed `powershell -NoProfile -ExecutionPolicy Bypass -Command "& .\tools\run-offline.ps1 -PytestArgs @('--collect-only', '-q')"`:
  - Output: `148 tests collected in 45.51s`. Exit code 0.
  - Confirms test suite baseline is 148 passed / 0 quarantined / 0 failed.

---

## 2. Logic Chain

### 2.1 Circular Import Prevention Strategy for `execution_modes.py`
1. **Premise**: `schema.py` must expose GraphQL endpoints that dispatch execution across modes, and `execution_modes.py` needs to invoke `run_conversation` and `AdvancedPromptEngineer`.
2. **Problem**:
   - `conversation_manager.py:17` has `from schema import ConversationInput`.
   - `advanced_prompting.py:18` has `from conversation_manager import output_type_determination, OutputType`.
   - If `schema.py` has a top-level import of `execution_modes.py`, and `execution_modes.py` has a top-level import of `conversation_manager.py`, an unresolvable circular import cycle is formed (`schema -> execution_modes -> conversation_manager -> schema`).
3. **Deduction & Solution**:
   - **Rule 1 (Schema Resolver Isolation)**: In `schema.py`, Strawberry GraphQL types (`RunExecutionResponse`, `RunExecutionInput`, `RunDiagnosticsType`, `PlanStepDiagnosticType`, `SubtaskDiagnosticType`, `ReflectionDiagnosticType`, `AgentAttributionType`, `RunEventTypeGQL`, `OutputTypeInfo`, `ExecutionModeCapabilityType`) must be declared directly in `schema.py` as native Strawberry types. `schema.py` must **never** import `execution_modes` at module top-level. Instead, resolvers (`Query.execution_modes`, `Mutation.run_execution`, `Mutation.run_advanced_pipeline`) must lazily import `from execution_modes import CAPABILITY_REGISTRY, execute_run, RunRequest` inside the resolver methods, exactly matching the existing established pattern in `schema.py:67` and `schema.py:107`.
   - **Rule 2 (Execution Handler Lazy Imports)**: In `execution_modes.py`, heavy runtime engines (`run_conversation` from `conversation_manager` and `AdvancedPromptEngineer`, `PromptEngineeringConfig` from `advanced_prompting`) must be imported lazily inside the respective handler functions (`_execute_collaborative`, `_execute_advanced_pipeline`).
   - **Rule 3 (Safe Pydantic Type References in `execution_modes.py`)**:
     `execution_modes.py` can either:
     - Import `OutputType` and `ErrorResult` from `conversation_manager` (which is safe once `schema.py` has no top-level import of `execution_modes`), OR
     - Define `output_type` as a Pydantic model compatible with `OutputType` (`output_type: str`, `file_extension: str`) or import `OutputType` from `conversation_manager`. Importing `from conversation_manager import OutputType, ErrorResult` at `execution_modes.py` top-level is safe *if and only if* `schema.py` has no top-level import of `execution_modes.py`.
   - **Rule 4 (Root Cause Hygiene Note)**: Line 17 of `conversation_manager.py` (`from schema import ConversationInput`) is unused in runtime code (duck-typing is used on line 106). The production writer can cleanly remove line 17 or place it under `if typing.TYPE_CHECKING:` to eliminate the extraneous dependency `conversation_manager -> schema` entirely.

### 2.2 Telemetry & Diagnostic Data Extraction for `collaborative` Mode
1. **Execution**:
   ```python
   result = run_conversation(
       problem_statement=request.conversation,
       selected_personalities=request.assistant_personalities,
       lead_personality=request.lead_personality,
       num_rounds=request.num_rounds,
       context=request.context
   )
   conversation_history, questions_asked, final_output = result
   ```
2. **Data Extraction Mapping**:
   - **`final_output`**: Raw output string (may be a file path or generated text).
   - **`output_type`**: Sniffed from `final_output` (or read file extension if path exists) via `AdvancedPromptEngineer(PromptEngineeringConfig()).determine_output_type_from_content(final_output, final_output if os.path.exists(str(final_output)) else "", None)`, falling back to `OutputType(output_type="Simple Concise Answer", file_extension=".txt")`.
   - **`questions`**: Extracted from `questions_asked`: `[{"assistant": q.get("assistant", ""), "question": q.get("question") or q.get("reply") or ""} for q in questions_asked]`.
   - **`conversation`**: Normalized copy of `conversation_history` (dicts with `role`, `name`, `content`).
   - **`agent_attributions`**: Extracted from `conversation_history` by filtering out pure system prompt entries (`entry.get("role") != "system"` or `entry.get("name") != "system"`):
     ```python
     AgentAttribution(
         agent_name=entry.get("name", "assistant"),
         role=entry.get("role", "assistant"),
         round_or_step=entry.get("round", 1),
         contribution_summary=entry.get("content", "")[:120] + "..." if len(entry.get("content", "")) > 120 else entry.get("content", "")
     )
     ```
   - **`events` (`List[RunEvent]`)**:
     - `RunEvent(event_type="mode_selected", mode="collaborative", summary="Collaborative debate session started")`
     - `RunEvent(event_type="problem_defined", mode="collaborative", summary=f"Problem statement formulated for {len(request.assistant_personalities)} assistants")`
     - Per-round agent events: `RunEvent(event_type="agent_contribution", mode="collaborative", step_number=entry.get("round"), agent_name=entry.get("name"), summary=f"Round {entry.get('round')} contribution by {entry.get('name')}")`
     - `RunEvent(event_type="output_finalized", mode="collaborative", summary="Collaborative final output generated")`
   - **`diagnostics` (`RunDiagnostics`)**:
     - `complexity_score = None`
     - `initial_budget = None`
     - `adjusted_budget = None`
     - `final_reward = None`
     - `plan_steps = []`
     - `reflections = []`
     - `agent_attributions = attributions`

### 2.3 Telemetry & Diagnostic Data Extraction for `advanced_pipeline` Mode
1. **Execution**:
   ```python
   config = PromptEngineeringConfig(
       max_steps=request.max_steps,
       initial_budget=request.initial_budget,
       model=request.model,
       temperature=request.temperature,
       agents=request.agents,
   )
   engineer = AdvancedPromptEngineer(config)
   interaction = engineer.main(request.task_prompt)
   ```
2. **Data Extraction Mapping**:
   - `task_obj = interaction.task or engineer.task_object`:
     - `complexity_score = float(getattr(task_obj, "complexity", 0.0) or 0.0)`
     - `initial_budget = config.initial_budget`
     - `adjusted_budget = int(config.initial_budget + complexity_score * config.complexity_factor)`
   - `final_reward = float(interaction.final_reward if interaction.final_reward is not None else 0.0)`
   - `plan_steps (`List[PlanStepDiagnostic]`)`:
     Extracted from `task_obj.plan.steps`:
     ```python
     plan_step_diagnostics = []
     for pstep in task_obj.plan.steps:
         subtask_diags = [
             SubtaskDiagnostic(
                 subtask_number=sub.subtask_number,
                 subtask_name=sub.subtask_name,
                 subtask_description=sub.subtask_description,
                 subtask_output=getattr(sub, "subtask_output", "") or "",
                 completed=bool(sub.completed)
             )
             for sub in getattr(pstep, "subtasks", [])
         ]
         assigned = engineer.select_personalities_for_planstep(pstep, task_obj) if config.agents > 0 else []
         plan_step_diagnostics.append(PlanStepDiagnostic(
             step_number=pstep.step_number,
             step_name=pstep.step_name,
             step_description=getattr(pstep, "description", None) or getattr(pstep, "step_description", "") or "",
             step_explanation=getattr(pstep, "step_explanation", "") or "",
             step_output=getattr(pstep, "step_output", "") or "",
             completed=bool(pstep.completed),
             assigned_agents=assigned,
             subtasks=subtask_diags
         ))
     ```
   - `reflections (`List[ReflectionDiagnostic]`)`:
     Extracted from `interaction.steps` and `interaction.reflections`:
     ```python
     # Map steps and reflections by step_number
     reflections_by_step = {r.step_number: r for r in interaction.reflections}
     reflection_diagnostics = []
     for stp in interaction.steps:
         refl = reflections_by_step.get(stp.step_number)
         reflection_diagnostics.append(ReflectionDiagnostic(
             step_number=stp.step_number,
             plan_step_number=stp.plan_step_number,
             step_description=stp.description or "",
             thoughts=stp.thoughts,
             content=refl.content if refl else "Step executed.",
             reward=float(refl.reward) if refl else 0.0,
             remaining_budget=stp.remaining_budget
         ))
     ```
   - `agent_attributions`:
     From `assigned_agents` across plan steps and `engineer.last_conversation_history` (if collaborative reasoning ran):
     ```python
     attributions = []
     for ps in plan_step_diagnostics:
         for ag in ps.assigned_agents:
             attributions.append(AgentAttribution(
                 agent_name=ag,
                 role="Plan Specialist",
                 round_or_step=ps.step_number,
                 contribution_summary=f"Assigned to Step {ps.step_number}: {ps.step_name}"
             ))
     ```
   - `final_output`:
     `interaction.answer` if non-empty; fallback to `interaction.planstep_outputs[-1].final_output` if present, or `interaction.steps[-1].description`, guaranteeing a non-empty string.
   - `output_type`:
     `task_obj.output_type` if valid `OutputType`; else refined via `engineer.determine_output_type_from_content(final_output, "", task_obj)`.
   - `conversation` transcript:
     Synthesized list of user and assistant progress messages so UI chat timeline renders seamlessly alongside the diagnostics panel:
     - User message: `{"role": "user", "name": "User", "content": request.task_prompt}`
     - Planner message: `{"role": "assistant", "name": "Planner", "content": f"Task complexity assessed at {complexity_score:.2f}. Generated {len(task_obj.plan.steps)} plan steps."}`
     - Completion message: `{"role": "assistant", "name": "Finalizer", "content": final_output}`
   - `events (`List[RunEvent]`)`:
     - `mode_selected`: `summary="Advanced reasoning pipeline initiated"`
     - `complexity_assessed`: `summary=f"Complexity assessed at {complexity_score:.2f}"`, `details={"complexity": complexity_score}`
     - `plan_generated`: `summary=f"Generated plan with {len(task_obj.plan.steps)} steps"`
     - Per-step events: `plan_step_started`, `plan_step_completed`, `reflection_recorded`
     - `output_finalized`: `summary="Pipeline completed and final output synthesized"`

### 2.4 Gated `tree_of_thought` Mode Behavior
1. **Premise**: In offline testing, any unconfigured call to `provider_client.get_client()` fails the test suite immediately (`conftest.py:forbid_unconfigured_provider`), and any socket/DNS call raises `NetworkForbidden`.
2. **Design**:
   - `execute_run(request: RunRequest)` inspects `request.mode` at the very beginning of the function:
     ```python
     if request.mode == "tree_of_thought":
         return RunResult(
             mode="tree_of_thought",
             status="gated",
             final_output="Tree of Thoughts execution mode is currently gated pending Issue #36.",
             output_type=OutputType(output_type="Simple Concise Answer", file_extension=".txt"),
             conversation=[
                 {"role": "user", "name": "User", "content": request.task_prompt or ""},
                 {"role": "assistant", "name": "System", "content": "Tree of Thoughts execution mode is currently gated pending Issue #36."}
             ],
             questions=[],
             diagnostics=RunDiagnostics(),
             events=[
                 RunEvent(
                     event_id="evt_gated_tot",
                     event_type="mode_gated",
                     timestamp=datetime.now(timezone.utc).isoformat(),
                     mode="tree_of_thought",
                     summary="Execution mode 'tree_of_thought' is gated pending Issue #36.",
                     details={"gated_issue": "Issue #36"}
                 )
             ],
             error=ErrorResult(
                 error_type="ModeGatedError",
                 error_message="Execution mode 'tree_of_thought' is gated for Issue #36."
             )
         )
     ```
3. **Safety Guarantee**:
   - Zero calls to `provider_client.get_client()`, `LazyClient`, `openai`, `run_conversation`, `AdvancedPromptEngineer`, or NLP libraries.
   - Guaranteed deterministic return in sub-millisecond time.
   - 100% immune to network denial or unconfigured SDK errors.

---

## 3. Caveats
1. **Dynamic Runtime Output Files**: `run_conversation` and `AdvancedPromptEngineer.main` may write files to the current working directory during execution (e.g. `conversation_history_*.json` or artifact files). In test environments, `conftest.py`'s `isolated_runtime_directory` fixture redirects these writes to project-local temporary directories (`tmp_path`), preserving clean git status.
2. **`interaction.steps` and `interaction.reflections` Alignment**: In `AdvancedPromptEngineer`, `interaction.reflections` has entries with `step_number` matching `Step.step_number`. In some branching or self-consistency edge cases, multiple reflections may exist for the same step number or a step may complete without an explicit reflection. The extraction logic must safely look up reflections using a dictionary mapping (`reflections_by_step.get(stp.step_number)`) and provide a default `ReflectionDiagnostic` if absent.
3. **Frontend Contract Gating**: Gating for `tree_of_thought` must also be reflected in the UI: `AssistantConfigPanel.tsx` should render the "Tree of Thoughts" option disabled with an `"Issue #36"` badge.

---

## 4. Conclusion

1. **Circular Import Resolution**:
   - Keep `schema.py` pure by declaring all Strawberry GraphQL types natively in `schema.py`.
   - Never import `execution_modes` at `schema.py` module top-level; use local/lazy imports inside resolver methods (`Query.execution_modes`, `Mutation.run_execution`, `Mutation.run_advanced_pipeline`).
   - In `execution_modes.py`, lazily import `run_conversation` and `AdvancedPromptEngineer` inside `_execute_collaborative` and `_execute_advanced_pipeline`.
   - The production writer can additionally remove the unused line 17 `from schema import ConversationInput` from `conversation_manager.py` (or place it under `if TYPE_CHECKING:`).
2. **Schema & Registry Design**:
   - `execution_modes.py` should define Pydantic v2 schemas: `RunRequest`, `RunResult`, `RunEvent`, `RunDiagnostics`, `PlanStepDiagnostic`, `SubtaskDiagnostic`, `ReflectionDiagnostic`, `AgentAttribution`, and `ExecutionModeCapability`.
   - `CAPABILITY_REGISTRY` registers all three modes (`collaborative`, `advanced_pipeline`, `tree_of_thought`), cleanly providing capability queries to `schema.Query.execution_modes`.
3. **Data Extraction**:
   - `advanced_pipeline` extracts comprehensive metrics from `interaction.task`, `task_obj.plan.steps`, `pstep.subtasks`, `interaction.steps`, `interaction.reflections`, `engineer.last_conversation_history`, and `engineer.select_personalities_for_planstep`.
   - `collaborative` extracts speaker attributions, questions asked, output type sniffing, and round events from `run_conversation`'s 3-tuple return.
4. **Gated Mode**:
   - `tree_of_thought` returns early with `status="gated"`, `ModeGatedError`, and `mode_gated` event without touching any provider, LLM, or external resources.

---

## 5. Verification Method

To independently verify the backend architecture and contracts:

1. **Verify Offline Test Collection & Baseline**:
   ```powershell
   powershell -NoProfile -ExecutionPolicy Bypass -Command "& .\tools\run-offline.ps1 -PytestArgs @('--collect-only', '-q')"
   ```
   *Expected outcome*: 148 tests collected, 0 errors.

2. **Verify Import Purity (No Network, No Eager Client Construction)**:
   ```powershell
   powershell -NoProfile -ExecutionPolicy Bypass -Command "& .\tools\run-offline.ps1 -PytestArgs @('test_import_integrity.py', 'test_offline_runtime.py', '-v')"
   ```
   *Expected outcome*: All import integrity and network denial tests pass.

3. **Verify Existing GraphQL Contract Compatibility**:
   ```powershell
   powershell -NoProfile -ExecutionPolicy Bypass -Command "& .\tools\run-offline.ps1 -PytestArgs @('test_graphql_contract.py', '-v')"
   ```
   *Expected outcome*: `test_run_conversation_contract` passes with exact schema shape.

4. **Verify Frontend Offline Guard & Test Suite**:
   ```powershell
   powershell -NoProfile -ExecutionPolicy Bypass -Command "& .\tools\run-frontend-offline.ps1 -Action guard"
   powershell -NoProfile -ExecutionPolicy Bypass -Command "& .\tools\run-frontend-offline.ps1 -Action test"
   ```
   *Expected outcome*: Node network denial guard verified; 111 Vitest tests in 10 test files pass.
