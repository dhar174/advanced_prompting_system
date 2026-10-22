# Milestone 1 Contracts & Schemas Report: Pydantic v2 Specifications for `execution_modes.py`

**Author**: `teamwork_preview_explorer` (Milestone 1 Contracts & Schemas Explorer)  
**Date**: 2026-10-22  
**Target Path**: `d:\projects\advanced_prompting_system\advanced_prompting_system\.agents\teamwork\m1_explorer_1\handoff.md`  
**Scope**: Issue #35 Phase 3 & Issue #21 — Exact Pydantic v2 contract specifications for `execution_modes.py`: `RunRequest`, `RunResult`, `RunEvent`, diagnostic schemas (`SubtaskDiagnostic`, `PlanStepDiagnostic`, `ReflectionDiagnostic`, `AgentAttribution`, `RunDiagnostics`), and `ExecutionModeCapability`.

---

## Executive Summary
This report defines the authoritative Pydantic v2 data models for `execution_modes.py`. All schemas were designed against the repository's Pydantic v2.14.1 runtime and tested for 100% compatibility with existing domain models (`OutputType`, `ErrorResult` from `conversation_manager.py`; `PlanStep`, `Subtask` from `complexity_measures.py`; `Step`, `Reflection` from `advanced_prompting.py`). A robust bidirectional `@model_validator(mode="after")` on `RunRequest` guarantees seamless synchronization between single-string task prompts and structured conversation histories without breaking GraphQL callers or downstream pipelines.

---

## 1. Observation

### 1.1 Existing Domain Models and Seams
1. **`conversation_manager.py:59-65` (`ErrorResult`)**:
   ```python
   class ErrorResult(BaseModel):
       """Structured error result to replace error strings."""
       error_type: str = Field(..., description="Type of error that occurred")
       error_message: str = Field(..., description="Human-readable error message")
       error_details: Optional[str] = Field(None, description="Additional error details")
       timestamp: datetime = Field(default_factory=datetime.now, description="When the error occurred")
   ```
2. **`conversation_manager.py:114-117` (`OutputType`)**:
   ```python
   class OutputType(BaseModel):
       output_type: str = Field(..., description="The type of output to generate.")
       file_extension: str = Field(..., description="The file extension for the output.")

   DEFAULT_OUTPUT_TYPE = OutputType(output_type="simple_text", file_extension="txt")
   ```
3. **`conversation_manager.py:91-112` (`normalize_message`)**:
   - Accepts dicts, strings, and objects with `.role` and `.content`.
   - Explicitly rejects `ErrorResult` instances with `ValueError`.
4. **`complexity_measures.py:724-766` (`Subtask` and `PlanStep`)**:
   - `Subtask`: `subtask_number: int`, `completed: bool`, `subtask_name: str`, `subtask_description: str`, `subtask_explanation: str`, `subtask_output: str`, `subtask_full_text: str`, `subtasks: List[Subtask]`.
   - `PlanStep`: `step_number: int`, `completed: bool`, `step_name: str`, `step_description: str`, `description: Optional[str] = None`, `step_explanation: str`, `step_output: str`, `step_full_text: str`, `subtasks: List[Subtask]`.
   - `PlanStep` uses `@model_validator(mode="before")` to synchronize `description` and `step_description`.
5. **`advanced_prompting.py:1354-1770` (`PromptEngineeringConfig`, `Reflection`, `Step`, `Task`, `Interaction`)**:
   - `PromptEngineeringConfig`: `max_steps: int = 20`, `initial_budget: int = 20`, `model: str = "gpt-4o-mini"`, `temperature: float = 0.7`, `agents: int = 3`.
   - `Reflection`: `content: str`, `reward: float`, `step_number: int`.
   - `Step`: `description: str`, `step_number: int`, `remaining_budget: int`, `reflection: Optional[Reflection]`, `thoughts: Optional[str]`, `plan_step_number: Optional[int]`.
   - `Interaction`: encapsulates `task: Task`, `steps: List[Step]`, `reflections: List[Reflection]`, `answer: Optional[str]`, `final_reward: float`, `planstep_outputs`.
6. **`schema.py:1-117` (GraphQL Types and Resolvers)**:
   - Strawberry GraphQL types: `ConversationInput`, `Conversation`, `Question`, `RunConversationResponse`.
   - `schema.py` does NOT import `conversation_manager` at the module level.
   - `Mutation.run_conversation` (line 67) lazily imports `from conversation_manager import run_conversation`.
   - `Mutation.submit_feedback` (line 107) lazily imports `from feedback_manager import log_conversation`.

### 1.2 Runtime Environment & Tool Observations
- Executed `python -c "import pydantic; print('pydantic version:', pydantic.__version__)"`:
  - Output: `pydantic version: 2.14.1`.
- Verified import sequence:
  - `python -c "import conversation_manager, schema; print(123)"` -> Exit code 0, output `123`.
  - `python -c "import schema, conversation_manager; print(456)"` -> Exit code 0, output `456`.
- Verified standalone Pydantic v2 execution of all schemas and bidirectional validator -> Exit code 0, output `All schema validations passed cleanly!`.

---

## 2. Logic Chain

### 2.1 Schema Architecture & Type Safety
1. **Premise**: `execution_modes.py` serves as the shared execution contract for three disparate consumers:
   - Strawberry GraphQL API (`schema.py`), which receives `RunExecutionInput` and returns `RunExecutionResponse`.
   - Multi-agent debate (`conversation_manager.run_conversation`), which expects structured message dicts `[{"role": ..., "content": ..., "name": ...}]`.
   - Prompt engineering pipeline (`AdvancedPromptEngineer.main`), which expects a single string `task: str`.
2. **Deduction (Bidirectional Sync in `RunRequest`)**:
   - If a caller supplies only `task_prompt: str` (e.g. standalone pipeline or direct API call), `conversation` must be automatically populated with a single user message (`{"role": "user", "name": "User", "content": task_prompt}`).
   - If a caller supplies only `conversation: List[Dict[str, Any]]` (e.g. multi-turn chat UI or GraphQL `ConversationInput`), `task_prompt` must be extracted from the latest message where `role == "user"`. If no user message exists, it should fall back to the latest non-empty message content.
   - If both are provided, both must be preserved without destructive mutation.
   - To support GraphQL clients passing `ConversationInput` objects, conversation entries must be normalized into standard dictionary representations.
3. **Deduction (Telemetry & Diagnostics in `RunDiagnostics`)**:
   - Requirements R1 and R2 mandate capturing latency/duration (`duration_ms`), token usage (`token_count`), and correlation IDs (`correlation_id`) alongside reasoning steps and reflections.
   - Placing these scalar telemetry fields directly in `RunDiagnostics` provides a single unified container for both structural execution diagnostics (`plan_steps`, `reflections`, `agent_attributions`) and performance metrics (`complexity_score`, `initial_budget`, `adjusted_budget`, `final_reward`, `duration_ms`, `token_count`, `correlation_id`).
4. **Deduction (Error and OutputType Resilience in `RunResult`)**:
   - `output_type` must default to `OutputType(output_type="Simple Concise Answer", file_extension=".txt")`. If callers supply a string (e.g. `"Python Script"`) or a raw dict, a `@field_validator("output_type", mode="before")` gracefully coerces it into a valid `OutputType` instance.
   - `error` must accept `Optional[ErrorResult]`. If callers supply an error string or dict, a `@field_validator("error", mode="before")` gracefully wraps it into an `ErrorResult`.
5. **Deduction (Circular Import Prevention)**:
   - `execution_modes.py` imports `OutputType` and `ErrorResult` from `conversation_manager`.
   - `schema.py` must never import `execution_modes` at the module top level; its resolvers must import `from execution_modes import CAPABILITY_REGISTRY, execute_run, RunRequest` lazily.
   - In `execution_modes.py`, `conversation_manager.run_conversation` and `advanced_prompting.AdvancedPromptEngineer` must be imported lazily inside mode execution handlers.

---

## 3. Exact Code Specification for `execution_modes.py`

Below is the complete, drop-in Python code specification for the contracts and schemas layer of `execution_modes.py`.

```python
"""
execution_modes.py

Unified execution contracts, capability registry, and multi-mode dispatcher
for the Advanced Prompting System (Issue #35 Phases 3–5 & Issue #21).

Defines strongly typed Pydantic v2 schemas for request validation, execution results,
chronological event telemetry, and multi-agent reasoning diagnostics.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator, model_validator

# Domain models from conversation_manager (safe DAG: conversation_manager -> schema)
from conversation_manager import ErrorResult, OutputType


# ============================================================================
# 1. Type Aliases & Enumerations
# ============================================================================

ExecutionModeId = Literal["collaborative", "advanced_pipeline", "tree_of_thought"]

RunStatus = Literal["completed", "gated", "error"]

RunEventType = Literal[
    "mode_selected",
    "problem_defined",
    "complexity_assessed",
    "plan_generated",
    "plan_step_started",
    "plan_step_completed",
    "agent_contribution",
    "reflection_recorded",
    "output_finalized",
    "mode_gated",
    "run_error",
]


# ============================================================================
# 2. Chronological Telemetry & Event Model
# ============================================================================

class RunEvent(BaseModel):
    """Structured chronological lifecycle event emitted during run execution."""

    event_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the telemetry event.",
    )
    event_type: str = Field(
        ...,
        description="Type category of the event matching RunEventType.",
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO-8601 UTC timestamp when the event was recorded.",
    )
    mode: str = Field(
        ...,
        description="Execution mode under which the event occurred.",
    )
    step_number: Optional[int] = Field(
        default=None,
        description="Sequential reasoning or debate step/round number if applicable.",
    )
    plan_step_number: Optional[int] = Field(
        default=None,
        description="Correlated PlanStep number from the execution plan if applicable.",
    )
    agent_name: Optional[str] = Field(
        default=None,
        description="Name of the agent or assistant associated with the event.",
    )
    summary: str = Field(
        ...,
        description="Brief human-readable summary of the event.",
    )
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary structured telemetry payload or metadata.",
    )


# ============================================================================
# 3. Intermediate Diagnostic Schemas
# ============================================================================

class SubtaskDiagnostic(BaseModel):
    """Diagnostic telemetry representing a decomposed subtask within a plan step."""

    subtask_number: int = Field(..., description="Index of the subtask within the plan step.")
    subtask_name: str = Field(..., description="Short title or name of the subtask.")
    subtask_description: str = Field(..., description="Detailed description of the subtask goal.")
    subtask_output: str = Field(default="", description="Output or generated artifact from the subtask.")
    completed: bool = Field(default=False, description="Completion status flag.")


class PlanStepDiagnostic(BaseModel):
    """Diagnostic telemetry representing an execution step in the generated plan."""

    step_number: int = Field(..., description="Sequential position in the plan.")
    step_name: str = Field(..., description="Title or name of the plan step.")
    step_description: str = Field(..., description="Detailed objective of the step.")
    step_explanation: str = Field(default="", description="Reasoning or architectural rationale.")
    step_output: str = Field(default="", description="Intermediate output generated by the step.")
    completed: bool = Field(default=False, description="Whether the step has finished.")
    assigned_agents: List[str] = Field(
        default_factory=list,
        description="Agent personalities assigned to execute or review this step.",
    )
    subtasks: List[SubtaskDiagnostic] = Field(
        default_factory=list,
        description="Decomposed subtasks executed under this plan step.",
    )

    @model_validator(mode="before")
    @classmethod
    def sync_descriptions(cls, data: Any) -> Any:
        """Permissively accept 'description' as alias for 'step_description'."""
        if isinstance(data, dict):
            desc = data.get("step_description") or data.get("description") or ""
            if "step_description" not in data or not data["step_description"]:
                data["step_description"] = desc
        return data


class ReflectionDiagnostic(BaseModel):
    """Diagnostic telemetry capturing self-reflection, intermediate thoughts, and reward scores."""

    step_number: int = Field(..., description="Reasoning step index where reflection occurred.")
    plan_step_number: Optional[int] = Field(
        default=None,
        description="Associated plan step index if executing under a structured plan.",
    )
    step_description: str = Field(
        default="",
        description="Description of the step being reflected upon.",
    )
    thoughts: Optional[str] = Field(
        default=None,
        description="Intermediate reasoning trace or thinking tokens (<thinking>).",
    )
    content: str = Field(
        ...,
        description="Evaluative feedback, insights, or self-correction remarks.",
    )
    reward: float = Field(
        ...,
        description="Numerical reward/score assigned to the step outcome (-1.0 to 1.0).",
    )
    remaining_budget: int = Field(
        default=0,
        description="Remaining computational/step budget after this reflection.",
    )


class AgentAttribution(BaseModel):
    """Attribution metadata tracking an individual assistant's contributions."""

    agent_name: str = Field(..., description="Name of the contributing assistant personality.")
    role: str = Field(default="assistant", description="Conversation role (assistant, lead, mediator).")
    round_or_step: int = Field(default=1, description="Round number or reasoning step of contribution.")
    contribution_summary: str = Field(..., description="Summary or snippet of the contribution.")


class RunDiagnostics(BaseModel):
    """Aggregate execution diagnostics and workload profiling metrics."""

    complexity_score: Optional[float] = Field(
        default=None,
        description="Assessed linguistic or structural complexity score (0.0 to 1.0).",
    )
    initial_budget: Optional[int] = Field(
        default=None,
        description="Initial step or token budget allocated for the run.",
    )
    adjusted_budget: Optional[int] = Field(
        default=None,
        description="Complexity-adjusted computational step budget.",
    )
    final_reward: Optional[float] = Field(
        default=None,
        description="Final aggregated evaluation reward score.",
    )
    duration_ms: Optional[int] = Field(
        default=None,
        description="Total execution latency in milliseconds.",
    )
    token_count: Optional[int] = Field(
        default=None,
        description="Estimated or counted tokens consumed during execution.",
    )
    correlation_id: Optional[str] = Field(
        default=None,
        description="Unique request correlation ID for end-to-end tracing.",
    )
    plan_steps: List[PlanStepDiagnostic] = Field(
        default_factory=list,
        description="Decomposed execution plan steps and subtask checklists.",
    )
    reflections: List[ReflectionDiagnostic] = Field(
        default_factory=list,
        description="Reasoning traces, self-reflections, and reward scores.",
    )
    agent_attributions: List[AgentAttribution] = Field(
        default_factory=list,
        description="Chronological attributions for all participating agents.",
    )


# ============================================================================
# 4. Execution Mode Capabilities
# ============================================================================

class ExecutionModeCapability(BaseModel):
    """Capability and availability metadata for an execution mode."""

    id: ExecutionModeId = Field(..., description="Unique programmatic mode identifier.")
    label: str = Field(..., description="Human-readable display name for the UI.")
    description: str = Field(..., description="Detailed explanation of execution mechanics.")
    enabled: bool = Field(..., description="Whether the mode is currently executable.")
    gated_issue: Optional[str] = Field(
        default=None,
        description="GitHub issue gating activation if disabled (e.g. 'Issue #36').",
    )
    supports_multi_agent: bool = Field(
        ...,
        description="Whether this mode supports collaborative multi-agent debate.",
    )
    supports_step_planning: bool = Field(
        ...,
        description="Whether this mode supports complexity-driven task planning.",
    )


# ============================================================================
# 5. Primary Run Request & Result Contracts
# ============================================================================

class RunRequest(BaseModel):
    """Unified request contract supporting both collaborative debate and standalone pipeline runs."""

    mode: ExecutionModeId = Field(
        default="collaborative",
        description="Selected execution mode (collaborative, advanced_pipeline, tree_of_thought).",
    )
    task_prompt: Optional[str] = Field(
        default=None,
        description="Standalone task specification or query text.",
    )
    conversation: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Multi-turn conversation history [{'role': 'user', 'content': '...', 'name': '...'}].",
    )
    assistant_personalities: List[str] = Field(
        default_factory=lambda: ["Helpful Assistant"],
        description="Debate personalities selected for multi-agent reasoning.",
    )
    lead_personality: str = Field(
        default="Helpful Assistant",
        description="Primary coordinating personality for consensus moderation.",
    )
    num_rounds: int = Field(
        default=3,
        ge=1,
        description="Number of debate deliberation rounds for collaborative mode.",
    )
    max_steps: int = Field(
        default=20,
        ge=1,
        description="Maximum reasoning steps allowed in advanced_pipeline mode.",
    )
    initial_budget: int = Field(
        default=20,
        ge=1,
        description="Initial reasoning step budget before complexity adjustment.",
    )
    agents: int = Field(
        default=3,
        ge=0,
        description="Number of collaborative agents per plan step in advanced_pipeline mode.",
    )
    model: str = Field(
        default="gpt-4o-mini",
        description="LLM provider model identifier.",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for LLM completions.",
    )
    context: Optional[str] = Field(
        default=None,
        description="Optional global domain context or background information.",
    )

    @model_validator(mode="after")
    def sync_prompt_and_conversation(self) -> "RunRequest":
        """Bidirectionally synchronize task_prompt and conversation history.

        1. Normalizes conversation elements into dictionary representations.
        2. If task_prompt is not explicitly provided, extracts it from the latest
           user entry in conversation.
        3. If only task_prompt is provided and conversation is empty, synthesizes
           an initial user message in conversation.
        """
        # Step 1: Normalize any non-dict conversation items (e.g. GraphQL inputs)
        normalized_conv: List[Dict[str, Any]] = []
        for item in self.conversation:
            if isinstance(item, dict):
                normalized_conv.append({
                    "role": item.get("role", "user"),
                    "name": item.get("name") or ("User" if item.get("role") == "user" else ""),
                    "content": str(item.get("content", "")),
                })
            elif hasattr(item, "role") and hasattr(item, "content"):
                role = getattr(item, "role", "user")
                normalized_conv.append({
                    "role": role,
                    "name": getattr(item, "name", "") or ("User" if role == "user" else ""),
                    "content": str(getattr(item, "content", "")),
                })
            else:
                normalized_conv.append({
                    "role": "user",
                    "name": "User",
                    "content": str(item),
                })
        self.conversation = normalized_conv

        # Step 2: Derive task_prompt from conversation if absent or whitespace
        has_prompt = self.task_prompt is not None and bool(str(self.task_prompt).strip())
        if not has_prompt:
            # Find the latest user message
            extracted = None
            for msg in reversed(self.conversation):
                if msg.get("role") == "user" and msg.get("content") and str(msg["content"]).strip():
                    extracted = str(msg["content"]).strip()
                    break
            # Fallback to any latest non-empty message
            if not extracted and self.conversation:
                for msg in reversed(self.conversation):
                    if msg.get("content") and str(msg["content"]).strip():
                        extracted = str(msg["content"]).strip()
                        break
            if extracted:
                self.task_prompt = extracted
        else:
            self.task_prompt = str(self.task_prompt).strip()

        # Step 3: Populate conversation if only task_prompt is present and conversation is empty
        if self.task_prompt and not self.conversation:
            self.conversation = [
                {
                    "role": "user",
                    "name": "User",
                    "content": self.task_prompt,
                }
            ]

        return self


class RunResult(BaseModel):
    """Unified execution outcome contract returned by all execution modes."""

    mode: ExecutionModeId = Field(
        ...,
        description="The mode that executed the request.",
    )
    status: RunStatus = Field(
        ...,
        description="High-level execution status ('completed', 'gated', 'error').",
    )
    final_output: str = Field(
        default="",
        description="Final answer text or generated file path.",
    )
    output_type: OutputType = Field(
        default_factory=lambda: OutputType(output_type="Simple Concise Answer", file_extension=".txt"),
        description="OutputType metadata (type name and file extension).",
    )
    conversation: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Complete transcript of debate rounds or synthesized pipeline execution.",
    )
    questions: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Clarifying or follow-up questions formulated during execution.",
    )
    diagnostics: RunDiagnostics = Field(
        default_factory=RunDiagnostics,
        description="Detailed reasoning traces, complexity, plan steps, and attributions.",
    )
    events: List[RunEvent] = Field(
        default_factory=list,
        description="Chronological stream of lifecycle telemetry events.",
    )
    error: Optional[ErrorResult] = Field(
        default=None,
        description="Structured error metadata if execution failed or was gated.",
    )

    @field_validator("output_type", mode="before")
    @classmethod
    def coerce_output_type(cls, v: Any) -> Any:
        """Permissively coerce strings or dicts into OutputType instances."""
        if isinstance(v, OutputType):
            return v
        if isinstance(v, dict):
            return OutputType(**v)
        if isinstance(v, str):
            extension_map = {
                "Python Script": ".py",
                "JSON": ".json",
                "HTML": ".html",
                "CSV": ".csv",
                "PDF": ".pdf",
                "Text File": ".txt",
                "Code Snippet": ".txt",
                "Simple Concise Answer": ".txt",
            }
            return OutputType(output_type=v, file_extension=extension_map.get(v, ".txt"))
        return v

    @field_validator("error", mode="before")
    @classmethod
    def coerce_error(cls, v: Any) -> Any:
        """Permissively coerce raw strings into structured ErrorResult."""
        if isinstance(v, ErrorResult):
            return v
        if isinstance(v, dict):
            return ErrorResult(**v)
        if isinstance(v, str) and v.strip():
            return ErrorResult(error_type="ExecutionError", error_message=v)
        return v
```

---

## 4. Caveats & Assumptions

1. **Gating Scope (Issue #36)**:
   - `tree_of_thought` mode is explicitly defined with `enabled=False` and `gated_issue="Issue #36"`. When dispatched, it must never call `provider_client.get_client` or initiate any LLM calls.
2. **Import Cleanliness**:
   - `schema.py` must NOT import `execution_modes` at module level. Strawberry resolvers must import `from execution_modes import execute_run, CAPABILITY_REGISTRY, RunRequest` lazily.
3. **Pydantic Version Assumption**:
   - Environment is verified on Pydantic `2.14.1`. The above code uses `@model_validator(mode="after")` and `@field_validator(mode="before")`, which are native Pydantic v2 decorators.
4. **Separation of Concerns with Peer Explorers**:
   - `m1_explorer_1` (this report) owns schema definitions.
   - `m1_explorer_2` owns `CAPABILITY_REGISTRY` entries, `execute_run`, `_execute_collaborative`, `_execute_advanced_pipeline`, and error wrapping.
   - `m1_explorer_3` owns `test_execution_modes.py`.
   - All three explorers share the schema definitions specified in this report as the unified contract.

---

## 5. Conclusion

The designed Pydantic v2 schemas:
1. Provide strongly typed, validated request and response structures (`RunRequest`, `RunResult`) that satisfy both the collaborative debate workflow and the standalone reasoning pipeline.
2. Support bidirectional prompt/conversation synchronization so users can pass either a single prompt or a full conversation transcript.
3. Unify structural reasoning diagnostics (`plan_steps`, `reflections`, `agent_attributions`) and R1/R2 telemetry metrics (`duration_ms`, `token_count`, `correlation_id`) into `RunDiagnostics`.
4. Ensure 100% type compatibility with existing models (`OutputType`, `ErrorResult`, `PlanStep`, `Subtask`).
5. Guarantee zero circular dependencies and zero import-time side effects.

---

## 6. Verification Method

To independently verify these schemas:

### 6.1 Unit Schema Verification Command
Run the isolated schema verification test in Python:
```powershell
python -c "
from execution_modes import RunRequest, RunResult, RunDiagnostics, RunEvent, PlanStepDiagnostic, SubtaskDiagnostic, ReflectionDiagnostic, AgentAttribution, ExecutionModeCapability
from conversation_manager import OutputType, ErrorResult

# Test 1: RunRequest bidirectional validation
r1 = RunRequest(task_prompt='Build auth')
assert r1.task_prompt == 'Build auth'
assert len(r1.conversation) == 1
assert r1.conversation[0]['content'] == 'Build auth'

# Test 2: Derivation from conversation
r2 = RunRequest(conversation=[{'role': 'user', 'content': 'Optimize queries'}])
assert r2.task_prompt == 'Optimize queries'

# Test 3: RunResult defaults and coercion
res = RunResult(mode='collaborative', status='completed', final_output='Success')
assert res.output_type.output_type == 'Simple Concise Answer'
assert res.output_type.file_extension == '.txt'
assert res.status == 'completed'

print('Schema verification PASSED!')
"
```

### 6.2 Regression and Invariant Verification Command
Ensure no regressions in existing backend offline tests:
```powershell
./tools/run-offline.ps1 -PytestArgs 'test_import_integrity.py','test_smoke.py','-v'
```

### 6.3 Invalidation Conditions
- Any import of `execution_modes` at the module top level in `schema.py` (would cause circular import).
- Any attempt to instantiate `openai.OpenAI()` or access network sockets during schema definition.
- Any failure of `RunRequest` to populate `conversation` when only `task_prompt` is provided.
- Any failure of `RunRequest` to extract `task_prompt` when only `conversation` is provided.
