# Milestone 1 Registry & Dispatcher Architecture Report

**Author**: `teamwork_preview_explorer` (Milestone 1 Registry & Dispatcher Explorer)  
**Date**: 2026-10-22  
**Target Path**: `d:\projects\advanced_prompting_system\advanced_prompting_system\.agents\teamwork\m1_explorer_2\handoff.md`  
**Scope**: Milestone 1 (Issue #35 Phase 3) — Capability Registry (`CAPABILITY_REGISTRY`) and Unified Dispatcher (`execute_run`) in `execution_modes.py`.

---

## 1. Observation

### 1.1 Existing Public Signatures and Return Contracts
1. **`conversation_manager.run_conversation` (`conversation_manager.py:2394-2408, 3432`)**:
   - Signature:
     ```python
     def run_conversation(
         problem_statement, selected_personalities, lead_personality, num_rounds=3, context=None
     ):
     ```
   - Normalization: lines 2398-2408 accept string, list of dicts (`role`, `content`), single dict, or object.
   - Return contract: strictly returns 3-tuple `(conversation_history, questions_asked, final_output)`.
   - `conversation_history`: list of message dictionaries with keys `role`, `name`, `round`, `content`.
   - `questions_asked`: list of dicts with `assistant`, `question`, and optional `reply`.
   - `final_output`: generated string or filepath (lines 3409–3410).
2. **`conversation_manager.py` Models (`conversation_manager.py:59-65, 114-120`)**:
   - `ErrorResult`:
     ```python
     class ErrorResult(BaseModel):
         error_type: str = Field(..., description="Type of error that occurred")
         error_message: str = Field(..., description="Human-readable error message")
         error_details: Optional[str] = Field(None, description="Additional error details")
         timestamp: datetime = Field(default_factory=datetime.now, description="When the error occurred")
     ```
   - `OutputType`:
     ```python
     class OutputType(BaseModel):
         output_type: str = Field(..., description="The type of output to generate.")
         file_extension: str = Field(..., description="The file extension for the output.")
     ```
   - `DEFAULT_OUTPUT_TYPE = OutputType(output_type="simple_text", file_extension="txt")`.
   - Message normalization (`conversation_manager.py:91-112`): `normalize_message(value)` raises `ValueError` on `ErrorResult` instances.
3. **`AdvancedPromptEngineer` and Reasoning Structures (`advanced_prompting.py:1354-2045, 7612-8037`)**:
   - `PromptEngineeringConfig`: `max_steps: int = 20`, `initial_budget: int = 20`, `model: str = "gpt-4o-mini"`, `temperature: float = 0.7`, `agents: int = 3`, `complexity_factor: int = 5`.
   - `AdvancedPromptEngineer.main(task: str) -> Interaction`:
     - Sets `self.task_object = task_object` where `task_object.complexity` is int/float and `task_object.plan` is `Plan(steps=[PlanStep, ...])`.
     - Returns `main_interaction: Interaction` containing:
       - `task: Task` (with `complexity`, `plan`, `output_type`, `planstep_outputs`)
       - `steps: List[Step]` (with `step_number`, `plan_step_number`, `description`, `thoughts`, `remaining_budget`, `reflection`)
       - `reflections: List[Reflection]` (with `content`, `reward`, `step_number`)
       - `answer: Optional[str]`
       - `final_reward: float`
       - `planstep_outputs: List[FinalPlanStepOutput]`
   - `determine_output_type_from_content(self, content: str, file_path: str, task: Task) -> OutputType`:
     - Checks file extension (`.py`, `.json`, `.html`, `.csv`, `.pdf`), content heuristics (keywords, `json.loads`, HTML tags), task's output type, and falls back to `OutputType(output_type="Text File", file_extension=".txt")`.
   - `select_personalities_for_planstep(self, plan_step: PlanStep, task: Task) -> List[str]`:
     - Dynamically selects domain-specific assistants (e.g., `"Software Engineer"`, `"Business Analyst"`, `"UX Designer"`, `"Strategist"`, `"Educator"`).
   - `self.last_conversation_history`: populated when collaborative reasoning sub-steps execute.
4. **Offline Isolation Guard & Invariants (`conftest.py` & `tools/offline/sitecustomize.py`)**:
   - `conftest.py:forbid_unconfigured_provider`: unscripted calls to `provider_client.get_client()` immediately raise a test failure.
   - `sitecustomize.py`: blocks external network sockets via `NetworkForbidden(BaseException)`.
   - `148/148` backend offline tests pass when run via `./tools/run-offline.ps1`.

---

## 2. Logic Chain

### 2.1 Registry Design (`CAPABILITY_REGISTRY`)
1. **Goal**: Provide a single, authoritative metadata registry of all execution modes supported or planned by the system.
2. **Modes & Requirements**:
   - `"collaborative"`: Fully operational (`enabled=True`), multi-agent round deliberation via `conversation_manager.run_conversation`. Supports multi-agent (`supports_multi_agent=True`), does not perform autonomous step planning (`supports_step_planning=False`).
   - `"advanced_pipeline"`: Fully operational (`enabled=True`), autonomous step planning and multi-strategy reasoning via `AdvancedPromptEngineer.main`. Supports both multi-agent planstep delegation (`supports_multi_agent=True`) and step planning (`supports_step_planning=True`).
   - `"tree_of_thought"`: Explicitly gated for Issue #36 (`enabled=False`, `gated_issue="Issue #36"`), single-agent tree search (`supports_multi_agent=False`, `supports_step_planning=True`).
3. **Interface Alignment**:
   - Registry entries map directly to `ExecutionModeCapability` (Pydantic model) and Strawberry GraphQL's `ExecutionModeCapabilityType` (`Query.execution_modes`), fulfilling Feature 4 and Feature 10 in `PROJECT.md`.

### 2.2 Dispatcher Execution Architecture (`execute_run`)
1. **Flow & Routing**:
   - Correlation ID: Every run request receives a unique identifier (`run_<hex12>`) and millisecond-level execution timer (`duration_ms`).
   - Route by `request.mode`:
     - `"tree_of_thought"` -> `_execute_tree_of_thought`
     - `"collaborative"` -> `_execute_collaborative`
     - `"advanced_pipeline"` -> `_execute_advanced_pipeline`
     - Unknown mode -> raises `ValueError` caught by outer error handler.
2. **Zero-Network Gating for `tree_of_thought`**:
   - Because `tree_of_thought` is gated for Issue #36, the dispatcher must NEVER instantiate `provider_client.get_client()`, invoke `run_conversation`, or execute `AdvancedPromptEngineer`.
   - Calling `_execute_tree_of_thought` directly returns `RunResult` with `status="gated"`, `ErrorResult(error_type="ModeGatedError")`, and a `mode_gated` `RunEvent`.
   - Sub-millisecond execution, zero provider touches, passes adversarial offline tests under `forbid_unconfigured_provider`.
3. **`collaborative` Telemetry & Extraction**:
   - Wraps `conversation_manager.run_conversation` with local lazy import (`from conversation_manager import run_conversation`).
   - Emits `mode_selected` and `problem_defined` events.
   - Inspects `raw_final_output`: if `isinstance(raw_final_output, ErrorResult)`, gracefully wraps it into `status="error"` with `run_error` event.
   - Normalizes `conversation_history` into standard dictionaries and extracts `agent_attributions` and per-round `agent_contribution` events.
   - Maps `questions_asked` into `questions: List[Dict[str, str]]`.
   - Resolves `output_type`: if `final_output` points to an existing file on disk, reads the content safely and resolves `OutputType` via content/extension sniffing, defaulting cleanly to `Simple Concise Answer` (`.txt`).
   - Emits `output_finalized` event.
4. **`advanced_pipeline` Telemetry & Extraction**:
   - Wraps `AdvancedPromptEngineer` with local lazy import (`from advanced_prompting import AdvancedPromptEngineer, PromptEngineeringConfig`).
   - Configures `PromptEngineeringConfig` with `request.max_steps`, `request.initial_budget`, `request.agents`, `request.model`, `request.temperature`.
   - Executes `interaction = engineer.main(task_prompt)`.
   - Extracts:
     - `complexity_score`, `initial_budget`, `adjusted_budget`, and `final_reward`.
     - `plan_steps`: maps `PlanStep` and `Subtask` objects to `PlanStepDiagnostic` and `SubtaskDiagnostic`, resolving assigned agents via `engineer.select_personalities_for_planstep`.
     - `reflections`: maps `interaction.steps` and `interaction.reflections` into `ReflectionDiagnostic` with step numbers, thoughts, content, rewards, and remaining budgets.
     - `agent_attributions`: records both plan specialists and collaborative sub-round contributors from `engineer.last_conversation_history`.
     - `final_output`: prioritizes `interaction.answer`, fallback to `planstep_outputs[-1].final_output` or step descriptions.
     - `output_type`: resolves from `task_obj.output_type` or `_infer_output_type`.
     - `conversation`: synthesizes a clean, readable multi-turn transcript (User -> Planner -> Step Milestones -> Finalizer) so the React UI chat thread renders pipeline progress seamlessly.
     - Emits ordered telemetry events: `mode_selected`, `complexity_assessed`, `plan_generated`, `plan_step_started`, `plan_step_completed`, `reflection_recorded`, `output_finalized`.
5. **Fail-Closed Error Handling**:
   - Catches `except Exception as e:` (preserving `NetworkForbidden(BaseException)`).
   - Wraps the exception into `status="error"` with structured `ErrorResult` (capturing `error_type`, `error_message`, and traceback in `error_details`).
   - Emits a `run_error` event with correlation details.
   - Returns valid, non-None `RunResult` ensuring downstream GraphQL resolvers and API callers never crash.

---

## 3. Caveats

1. **Import Seam Discipline**:
   - `schema.py` must NOT import `execution_modes.py` at top level. In resolvers (`Query.execution_modes`, `Mutation.run_execution`, `Mutation.run_advanced_pipeline`), use local lazy imports: `from execution_modes import CAPABILITY_REGISTRY, execute_run, RunRequest`.
   - In `execution_modes.py`, `run_conversation` and `AdvancedPromptEngineer` must be imported inside `_execute_collaborative` and `_execute_advanced_pipeline`.
2. **`OutputType` and `ErrorResult` Sharing**:
   - Both models are cleanly imported from `conversation_manager` (`from conversation_manager import OutputType, ErrorResult`). This is 100% safe because `conversation_manager.py` does not import `execution_modes.py`.
3. **Disk Output Reading**:
   - When `final_output` in `collaborative` mode is a valid filepath on disk, `_execute_collaborative` attempts to read its text content with UTF-8 encoding. If reading fails or permissions are denied, it preserves the original filepath string.
4. **Preservation of `NetworkForbidden`**:
   - The error handler in `execute_run` catches `Exception`, NOT `BaseException`. This ensures that `NetworkForbidden` (which inherits directly from `BaseException`) correctly bubbles up and trips the offline network guard in test environments.

---

## 4. Conclusion & Recommended Implementation

The exact code recommendations below should be placed into `execution_modes.py` by the single production writer (`aps-repo-coordinator`).

### 4.1 Recommended Python Implementation for `execution_modes.py`

```python
"""
Execution Modes, Capability Registry & Unified Dispatcher for the Advanced Prompting System.
Provides strongly typed Pydantic v2 contracts and dispatcher logic across:
- 'collaborative': Multi-agent round deliberation & consensus voting via conversation_manager
- 'advanced_pipeline': Standalone task-planning and multi-strategy reasoning via AdvancedPromptEngineer
- 'tree_of_thought': Zero-network fail-closed gated placeholder for Issue #36
"""

from __future__ import annotations

import os
import time
import uuid
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator

# Re-use authoritative existing types from conversation_manager without circular imports
from conversation_manager import OutputType, ErrorResult


# =====================================================================
# 1. Type Aliases & Literal Enums
# =====================================================================

ExecutionModeId = Literal["collaborative", "advanced_pipeline", "tree_of_thought"]

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


# =====================================================================
# 2. Diagnostic & Telemetry Schemas
# =====================================================================

class SubtaskDiagnostic(BaseModel):
    """Diagnostic details for an individual subtask within a plan step."""
    subtask_number: int = Field(..., description="1-indexed subtask sequence number.")
    subtask_name: str = Field(..., description="Short name or title of the subtask.")
    subtask_description: str = Field(..., description="Detailed description of the subtask objective.")
    subtask_output: str = Field(default="", description="Output or artifact generated by the subtask.")
    completed: bool = Field(default=False, description="Whether this subtask has been completed.")


class PlanStepDiagnostic(BaseModel):
    """Diagnostic details for an execution plan step."""
    step_number: int = Field(..., description="1-indexed plan step number.")
    step_name: str = Field(..., description="Name or title of the plan step.")
    step_description: str = Field(..., description="Detailed description of what this step accomplishes.")
    step_explanation: str = Field(default="", description="Rationale or strategy explanation for this step.")
    step_output: str = Field(default="", description="Direct output generated by this plan step.")
    completed: bool = Field(default=False, description="Whether this step completed successfully.")
    assigned_agents: List[str] = Field(default_factory=list, description="Debate or domain specialists assigned to this step.")
    subtasks: List[SubtaskDiagnostic] = Field(default_factory=list, description="Nested subtask items for this step.")


class ReflectionDiagnostic(BaseModel):
    """Diagnostic record of intermediate step thoughts, reflections, and reward evaluation."""
    step_number: int = Field(..., description="Sequential reasoning step number.")
    plan_step_number: Optional[int] = Field(default=None, description="Associated high-level plan step number, if any.")
    step_description: str = Field(default="", description="Description of the action taken in this step.")
    thoughts: Optional[str] = Field(default=None, description="Intermediate thoughts or chain-of-thought scratchpad.")
    content: str = Field(..., description="Critique, reflection, or assessment content.")
    reward: float = Field(..., description="Evaluated reward score for this reasoning step (-1.0 to 1.0).")
    remaining_budget: int = Field(default=0, description="Step budget remaining after this reflection.")


class AgentAttribution(BaseModel):
    """Attribution metadata for contributing assistants or debate personas."""
    agent_name: str = Field(..., description="Name of the persona or assistant.")
    role: str = Field(default="assistant", description="Role or classification (e.g. 'assistant', 'user', 'Plan Specialist').")
    round_or_step: int = Field(default=1, description="Round or step number of the contribution.")
    contribution_summary: str = Field(..., description="Summary or snippet of the contribution made.")


class RunDiagnostics(BaseModel):
    """Comprehensive diagnostic container for reasoning traces and workload metrics."""
    complexity_score: Optional[float] = Field(default=None, description="Assessed linguistic or structural complexity (0.0 to 1.0+).")
    initial_budget: Optional[int] = Field(default=None, description="Initial reasoning step budget.")
    adjusted_budget: Optional[int] = Field(default=None, description="Budget adjusted based on task complexity.")
    final_reward: Optional[float] = Field(default=None, description="Final consensus or answer judge reward score.")
    plan_steps: List[PlanStepDiagnostic] = Field(default_factory=list, description="Execution plan steps and subtasks.")
    reflections: List[ReflectionDiagnostic] = Field(default_factory=list, description="Intermediate thoughts and reflection rewards.")
    agent_attributions: List[AgentAttribution] = Field(default_factory=list, description="Agent and persona contributions.")
    duration_ms: Optional[int] = Field(default=None, description="Total run duration in milliseconds.")
    token_count: Optional[int] = Field(default=None, description="Estimated or tracked token expenditure.")
    correlation_id: Optional[str] = Field(default=None, description="Request correlation identifier.")


class RunEvent(BaseModel):
    """Chronological execution event for telemetry and frontend timeline rendering."""
    event_id: str = Field(..., description="Unique event identifier.")
    event_type: str = Field(..., description="Event category matching RunEventType.")
    timestamp: str = Field(..., description="ISO-8601 UTC timestamp string.")
    mode: str = Field(..., description="Execution mode active when event occurred.")
    step_number: Optional[int] = Field(default=None, description="Associated reasoning step number.")
    plan_step_number: Optional[int] = Field(default=None, description="Associated high-level plan step number.")
    agent_name: Optional[str] = Field(default=None, description="Name of the acting agent, if applicable.")
    summary: str = Field(..., description="Human-readable event summary.")
    details: Dict[str, Any] = Field(default_factory=dict, description="Structured event metadata.")


# =====================================================================
# 3. Capability Metadata & Unified Run Contracts
# =====================================================================

class ExecutionModeCapability(BaseModel):
    """Descriptor defining capability, availability, and features of an execution mode."""
    id: ExecutionModeId = Field(..., description="Unique execution mode identifier.")
    label: str = Field(..., description="Human-friendly mode title.")
    description: str = Field(..., description="Summary of mode operation and architecture.")
    enabled: bool = Field(..., description="Whether this mode is currently active and executable.")
    gated_issue: Optional[str] = Field(default=None, description="Tracking issue if mode is disabled/gated.")
    supports_multi_agent: bool = Field(..., description="Whether the mode orchestrates multi-agent deliberations.")
    supports_step_planning: bool = Field(..., description="Whether the mode performs autonomous step planning.")


class RunRequest(BaseModel):
    """Unified execution request contract across all execution modes."""
    mode: ExecutionModeId = Field(default="collaborative", description="Target execution mode.")
    task_prompt: Optional[str] = Field(default=None, description="Core user prompt or problem statement.")
    conversation: List[Dict[str, Any]] = Field(default_factory=list, description="Input conversation thread.")
    assistant_personalities: List[str] = Field(
        default_factory=lambda: ["Helpful Assistant"],
        description="Assigned assistant personas for collaborative debate."
    )
    lead_personality: str = Field(default="Helpful Assistant", description="Lead moderator or finalizer persona.")
    num_rounds: int = Field(default=3, description="Number of debate rounds for collaborative mode.")
    max_steps: int = Field(default=20, description="Max reasoning steps for pipeline mode.")
    initial_budget: int = Field(default=20, description="Initial reasoning step budget for pipeline mode.")
    agents: int = Field(default=3, description="Collaborative agents per plan step in pipeline mode.")
    model: str = Field(default="gpt-4o-mini", description="LLM provider model identifier.")
    temperature: float = Field(default=0.7, description="Sampling temperature.")
    context: Optional[str] = Field(default=None, description="Optional system or domain context.")

    @model_validator(mode="after")
    def sync_prompt_and_conversation(self) -> "RunRequest":
        """Bidirectional validator syncing task_prompt and conversation."""
        if not self.task_prompt and self.conversation:
            for msg in reversed(self.conversation):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    self.task_prompt = str(msg.get("content", ""))
                    break
            if not self.task_prompt and self.conversation:
                first = self.conversation[0]
                if isinstance(first, dict):
                    self.task_prompt = str(first.get("content", ""))
                else:
                    self.task_prompt = str(first)
        elif self.task_prompt and not self.conversation:
            self.conversation = [{"role": "user", "name": "User", "content": self.task_prompt}]
        return self


class RunResult(BaseModel):
    """Unified execution result returned by all execution modes."""
    mode: ExecutionModeId = Field(..., description="Execution mode utilized.")
    status: Literal["completed", "gated", "error"] = Field(..., description="Execution outcome status.")
    final_output: str = Field(default="", description="Synthesized text or output content.")
    output_type: OutputType = Field(
        default_factory=lambda: OutputType(output_type="Simple Concise Answer", file_extension=".txt"),
        description="Detected or configured output type and extension."
    )
    conversation: List[Dict[str, Any]] = Field(default_factory=list, description="Transcript of execution conversation.")
    questions: List[Dict[str, str]] = Field(default_factory=list, description="Extracted questions asked by assistants.")
    diagnostics: RunDiagnostics = Field(default_factory=RunDiagnostics, description="Detailed reasoning diagnostics.")
    events: List[RunEvent] = Field(default_factory=list, description="Chronological timeline telemetry events.")
    error: Optional[ErrorResult] = Field(default=None, description="Error details if status is 'error' or 'gated'.")


# =====================================================================
# 4. Capability Registry
# =====================================================================

CAPABILITY_REGISTRY: Dict[ExecutionModeId, ExecutionModeCapability] = {
    "collaborative": ExecutionModeCapability(
        id="collaborative",
        label="Collaborative Multi-Agent Debate",
        description="Multi-agent round deliberation and consensus voting via conversation manager.",
        enabled=True,
        gated_issue=None,
        supports_multi_agent=True,
        supports_step_planning=False,
    ),
    "advanced_pipeline": ExecutionModeCapability(
        id="advanced_pipeline",
        label="Standalone Reasoning Pipeline",
        description="Complexity-driven step planning, reflections, reward judging, and multi-strategy reasoning.",
        enabled=True,
        gated_issue=None,
        supports_multi_agent=True,
        supports_step_planning=True,
    ),
    "tree_of_thought": ExecutionModeCapability(
        id="tree_of_thought",
        label="Tree of Thoughts (Issue #36)",
        description="Tree of thoughts exploration gated pending Issue #36.",
        enabled=False,
        gated_issue="Issue #36",
        supports_multi_agent=False,
        supports_step_planning=True,
    ),
}


def get_capability(mode: str) -> Optional[ExecutionModeCapability]:
    """Retrieve capability descriptor for a given execution mode ID."""
    return CAPABILITY_REGISTRY.get(mode)  # type: ignore[arg-type]


def list_capabilities() -> List[ExecutionModeCapability]:
    """Return all registered execution mode capabilities."""
    return list(CAPABILITY_REGISTRY.values())


# =====================================================================
# 5. Internal Helper Functions
# =====================================================================

def _create_run_event(
    event_type: str,
    mode: str,
    summary: str,
    step_number: Optional[int] = None,
    plan_step_number: Optional[int] = None,
    agent_name: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
) -> RunEvent:
    """Construct a strongly typed RunEvent with ISO-8601 UTC timestamp."""
    return RunEvent(
        event_id=f"evt_{uuid.uuid4().hex[:8]}",
        event_type=event_type,
        timestamp=datetime.now(timezone.utc).isoformat(),
        mode=mode,
        step_number=step_number,
        plan_step_number=plan_step_number,
        agent_name=agent_name,
        summary=summary,
        details=details or {},
    )


def _infer_output_type(content: str, file_path: str = "", task_obj: Optional[Any] = None) -> OutputType:
    """
    Safely determine OutputType from content or file path using AdvancedPromptEngineer,
    defaulting cleanly to 'Simple Concise Answer' (.txt).
    """
    try:
        from advanced_prompting import AdvancedPromptEngineer, PromptEngineeringConfig
        engineer = AdvancedPromptEngineer(PromptEngineeringConfig())
        inferred = engineer.determine_output_type_from_content(
            content=content,
            file_path=file_path if (file_path and os.path.exists(str(file_path))) else "",
            task=task_obj,
        )
        if inferred and isinstance(inferred, OutputType):
            if inferred.output_type in ("Text File", "simple_text") and not (file_path and os.path.isfile(str(file_path))):
                return OutputType(output_type="Simple Concise Answer", file_extension=".txt")
            return inferred
    except Exception:
        pass
    return OutputType(output_type="Simple Concise Answer", file_extension=".txt")


def _normalize_conversation_history(history: List[Any]) -> List[Dict[str, Any]]:
    """Ensure all conversation history entries are plain dictionaries with role, name, content."""
    normalized: List[Dict[str, Any]] = []
    for item in history:
        if isinstance(item, dict):
            entry = {
                "role": str(item.get("role", "assistant")),
                "name": str(item.get("name", "")),
                "content": str(item.get("content", "")),
            }
            if "round" in item:
                entry["round"] = item["round"]
            normalized.append(entry)
        elif hasattr(item, "content"):
            normalized.append({
                "role": getattr(item, "role", "assistant"),
                "name": getattr(item, "name", ""),
                "content": str(item.content),
            })
        elif isinstance(item, str):
            normalized.append({"role": "assistant", "name": "", "content": item})
    return normalized


# =====================================================================
# 6. Mode Execution Handlers
# =====================================================================

def _execute_tree_of_thought(
    request: RunRequest,
    correlation_id: str,
    start_time: float,
) -> RunResult:
    """
    Zero-network fail-closed gating for Tree of Thoughts execution mode (Issue #36).
    Guarantees no provider client, LLM calls, or network sockets are accessed.
    """
    duration_ms = int((time.perf_counter() - start_time) * 1000)
    gating_message = "Tree of Thoughts execution mode is currently gated pending Issue #36."

    conversation_transcript: List[Dict[str, Any]] = []
    if request.task_prompt:
        conversation_transcript.append({
            "role": "user",
            "name": "User",
            "content": request.task_prompt,
        })
    conversation_transcript.append({
        "role": "assistant",
        "name": "System",
        "content": gating_message,
    })

    event = _create_run_event(
        event_type="mode_gated",
        mode="tree_of_thought",
        summary="Execution mode 'tree_of_thought' is gated pending Issue #36.",
        details={"gated_issue": "Issue #36", "correlation_id": correlation_id},
    )

    error_obj = ErrorResult(
        error_type="ModeGatedError",
        error_message="Execution mode 'tree_of_thought' is gated for Issue #36.",
        error_details=gating_message,
        timestamp=datetime.now(timezone.utc),
    )

    return RunResult(
        mode="tree_of_thought",
        status="gated",
        final_output=gating_message,
        output_type=OutputType(output_type="Simple Concise Answer", file_extension=".txt"),
        conversation=conversation_transcript,
        questions=[],
        diagnostics=RunDiagnostics(
            duration_ms=duration_ms,
            correlation_id=correlation_id,
        ),
        events=[event],
        error=error_obj,
    )


def _execute_collaborative(
    request: RunRequest,
    correlation_id: str,
    start_time: float,
) -> RunResult:
    """
    Execute collaborative debate mode via conversation_manager.run_conversation.
    Extracts attributions, questions, OutputType, and timeline telemetry.
    """
    from conversation_manager import run_conversation

    events: List[RunEvent] = []

    # 1. Mode selected event
    events.append(_create_run_event(
        event_type="mode_selected",
        mode="collaborative",
        summary="Collaborative multi-agent debate session initiated.",
        details={
            "assistant_personalities": request.assistant_personalities,
            "lead_personality": request.lead_personality,
            "num_rounds": request.num_rounds,
        },
    ))

    # 2. Problem defined event
    events.append(_create_run_event(
        event_type="problem_defined",
        mode="collaborative",
        summary=f"Problem statement formulated for {len(request.assistant_personalities)} debate participants.",
        details={"task_prompt": request.task_prompt},
    ))

    # 3. Invoke run_conversation
    raw_history, raw_questions, raw_final_output = run_conversation(
        problem_statement=request.conversation if request.conversation else (request.task_prompt or ""),
        selected_personalities=request.assistant_personalities,
        lead_personality=request.lead_personality,
        num_rounds=request.num_rounds,
        context=request.context,
    )

    # 4. Check for ErrorResult from run_conversation
    if isinstance(raw_final_output, ErrorResult):
        error_obj = raw_final_output
        err_msg = error_obj.error_message
        events.append(_create_run_event(
            event_type="run_error",
            mode="collaborative",
            summary=f"Collaborative deliberation returned error: {err_msg}",
            details={"error_type": error_obj.error_type, "error_details": error_obj.error_details},
        ))
        duration_ms = int((time.perf_counter() - start_time) * 1000)
        return RunResult(
            mode="collaborative",
            status="error",
            final_output=err_msg,
            output_type=OutputType(output_type="Simple Concise Answer", file_extension=".txt"),
            conversation=_normalize_conversation_history(raw_history or []),
            questions=[],
            diagnostics=RunDiagnostics(duration_ms=duration_ms, correlation_id=correlation_id),
            events=events,
            error=error_obj,
        )

    # 5. Normalize conversation history & extract agent attributions
    normalized_history = _normalize_conversation_history(raw_history or [])
    agent_attributions: List[AgentAttribution] = []

    for entry in normalized_history:
        role = entry.get("role", "assistant")
        name = entry.get("name", "")
        rnd = int(entry.get("round", 1) or 1)
        content = entry.get("content", "")

        # Emit agent contribution event for assistants
        if role == "assistant":
            agent_name = name or "Assistant"
            events.append(_create_run_event(
                event_type="agent_contribution",
                mode="collaborative",
                step_number=rnd,
                agent_name=agent_name,
                summary=f"Round {rnd} contribution from {agent_name}",
                details={"round": rnd, "agent_name": agent_name},
            ))

        # Build attributions (excluding purely anonymous system messages)
        if role in ("assistant", "user") or (role == "system" and name and name.lower() not in ("system", "user")):
            summary_snippet = content[:140] + "..." if len(content) > 140 else content
            agent_attributions.append(AgentAttribution(
                agent_name=name or ("User" if role == "user" else "Assistant"),
                role=role,
                round_or_step=rnd,
                contribution_summary=summary_snippet or "Debate contribution.",
            ))

    # 6. Extract questions
    questions_list: List[Dict[str, str]] = []
    for q in (raw_questions or []):
        if isinstance(q, dict):
            questions_list.append({
                "assistant": str(q.get("assistant", "")),
                "question": str(q.get("question") or q.get("reply") or ""),
            })

    # 7. Output Type resolution and file content check
    final_output_str = str(raw_final_output or "")
    file_path = final_output_str if os.path.exists(final_output_str) else ""
    if file_path and os.path.isfile(file_path):
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                disk_content = f.read()
            if disk_content.strip():
                final_output_str = disk_content
        except Exception:
            pass

    resolved_output_type = _infer_output_type(content=final_output_str, file_path=file_path)

    # 8. Output finalized event
    events.append(_create_run_event(
        event_type="output_finalized",
        mode="collaborative",
        summary="Collaborative consensus reached and output finalized.",
        details={"output_type": resolved_output_type.output_type, "file_extension": resolved_output_type.file_extension},
    ))

    duration_ms = int((time.perf_counter() - start_time) * 1000)
    diagnostics = RunDiagnostics(
        agent_attributions=agent_attributions,
        duration_ms=duration_ms,
        correlation_id=correlation_id,
    )

    return RunResult(
        mode="collaborative",
        status="completed",
        final_output=final_output_str,
        output_type=resolved_output_type,
        conversation=normalized_history,
        questions=questions_list,
        diagnostics=diagnostics,
        events=events,
        error=None,
    )


def _execute_advanced_pipeline(
    request: RunRequest,
    correlation_id: str,
    start_time: float,
) -> RunResult:
    """
    Execute standalone reasoning pipeline via AdvancedPromptEngineer.main.
    Extracts complexity, plan steps, subtasks, reflections with rewards, attributions,
    synthesizes chat transcript, and produces timeline telemetry events.
    """
    from advanced_prompting import AdvancedPromptEngineer, PromptEngineeringConfig

    events: List[RunEvent] = []

    # 1. Mode selected event
    events.append(_create_run_event(
        event_type="mode_selected",
        mode="advanced_pipeline",
        summary="Advanced reasoning pipeline initiated.",
        details={
            "max_steps": request.max_steps,
            "initial_budget": request.initial_budget,
            "agents": request.agents,
            "model": request.model,
        },
    ))

    # 2. Build configuration and instantiate engineer
    config = PromptEngineeringConfig(
        max_steps=request.max_steps,
        initial_budget=request.initial_budget,
        model=request.model,
        temperature=request.temperature,
        agents=request.agents,
    )
    engineer = AdvancedPromptEngineer(config)

    # 3. Execute main reasoning pipeline
    task_prompt = request.task_prompt or ""
    if not task_prompt and request.conversation:
        task_prompt = str(request.conversation[-1].get("content", ""))

    interaction = engineer.main(task_prompt)

    # 4. Extract task object and complexity
    task_obj = getattr(interaction, "task", None) or getattr(engineer, "task_object", None)
    complexity_score = float(getattr(task_obj, "complexity", 0.0) or 0.0) if task_obj else 0.0
    initial_budget = int(config.initial_budget)
    complexity_factor = getattr(config, "complexity_factor", 5)
    adjusted_budget = int(initial_budget + complexity_score * complexity_factor)
    final_reward = float(getattr(interaction, "final_reward", 0.0) if getattr(interaction, "final_reward", None) is not None else 0.0)

    events.append(_create_run_event(
        event_type="complexity_assessed",
        mode="advanced_pipeline",
        summary=f"Task complexity assessed at {complexity_score:.2f}.",
        details={
            "complexity_score": complexity_score,
            "initial_budget": initial_budget,
            "adjusted_budget": adjusted_budget,
        },
    ))

    # 5. Extract plan steps and subtasks
    plan = getattr(task_obj, "plan", None) if task_obj else None
    plan_steps_raw = getattr(plan, "steps", []) if plan else []

    events.append(_create_run_event(
        event_type="plan_generated",
        mode="advanced_pipeline",
        summary=f"Generated execution plan with {len(plan_steps_raw)} steps.",
        details={"step_count": len(plan_steps_raw)},
    ))

    plan_step_diagnostics: List[PlanStepDiagnostic] = []
    agent_attributions: List[AgentAttribution] = []

    for pstep in plan_steps_raw:
        step_num = getattr(pstep, "step_number", 1)
        step_name = getattr(pstep, "step_name", "") or f"Step {step_num}"
        desc = getattr(pstep, "description", None) or getattr(pstep, "step_description", "") or ""
        exp = getattr(pstep, "step_explanation", "") or ""
        out = getattr(pstep, "step_output", "") or ""
        completed = bool(getattr(pstep, "completed", False))

        # Extract assigned agents if agents > 0
        assigned_agents: List[str] = []
        if config.agents > 0 and hasattr(engineer, "select_personalities_for_planstep") and task_obj:
            try:
                assigned_agents = engineer.select_personalities_for_planstep(pstep, task_obj) or []
            except Exception:
                assigned_agents = []

        # Record agent attributions
        for ag in assigned_agents:
            agent_attributions.append(AgentAttribution(
                agent_name=ag,
                role="Plan Specialist",
                round_or_step=step_num,
                contribution_summary=f"Assigned to Step {step_num}: {step_name}",
            ))

        # Extract subtasks
        subtasks_diag: List[SubtaskDiagnostic] = []
        for sub in (getattr(pstep, "subtasks", []) or []):
            sub_num = getattr(sub, "subtask_number", 1)
            subtasks_diag.append(SubtaskDiagnostic(
                subtask_number=sub_num,
                subtask_name=getattr(sub, "subtask_name", "") or f"Subtask {sub_num}",
                subtask_description=getattr(sub, "subtask_description", "") or "",
                subtask_output=getattr(sub, "subtask_output", "") or "",
                completed=bool(getattr(sub, "completed", False)),
            ))

        plan_step_diagnostics.append(PlanStepDiagnostic(
            step_number=step_num,
            step_name=step_name,
            step_description=desc,
            step_explanation=exp,
            step_output=out,
            completed=completed,
            assigned_agents=assigned_agents,
            subtasks=subtasks_diag,
        ))

        # Per-step events
        events.append(_create_run_event(
            event_type="plan_step_started",
            mode="advanced_pipeline",
            plan_step_number=step_num,
            summary=f"Plan step {step_num} started: {step_name}",
            details={"step_name": step_name, "assigned_agents": assigned_agents},
        ))
        if completed:
            events.append(_create_run_event(
                event_type="plan_step_completed",
                mode="advanced_pipeline",
                plan_step_number=step_num,
                summary=f"Plan step {step_num} completed: {step_name}",
                details={"step_name": step_name, "output_preview": out[:100]},
            ))

    # Also capture attributions from collaborative sub-rounds if executed
    last_collab_history = getattr(engineer, "last_conversation_history", None) or []
    for entry in last_collab_history:
        if isinstance(entry, dict) and entry.get("role") == "assistant":
            name = entry.get("name", "Assistant")
            rnd = int(entry.get("round", 1) or 1)
            cnt = entry.get("content", "")
            agent_attributions.append(AgentAttribution(
                agent_name=name,
                role="Collaborative Sub-step",
                round_or_step=rnd,
                contribution_summary=cnt[:120] + "..." if len(cnt) > 120 else cnt,
            ))

    # 6. Extract reflections with rewards
    raw_reflections = getattr(interaction, "reflections", []) or []
    reflections_by_step = {getattr(r, "step_number", None): r for r in raw_reflections}
    reflection_diagnostics: List[ReflectionDiagnostic] = []

    for stp in (getattr(interaction, "steps", []) or []):
        s_num = getattr(stp, "step_number", 1)
        p_num = getattr(stp, "plan_step_number", None)
        refl = reflections_by_step.get(s_num)

        # Determine reward & content
        if refl:
            rew = float(getattr(refl, "reward", 0.0) if getattr(refl, "reward", None) is not None else 0.0)
            cnt = getattr(refl, "content", "Step executed.") or "Step executed."
        elif getattr(stp, "reflection", None):
            rew = float(getattr(stp.reflection, "reward", 0.0) if getattr(stp.reflection, "reward", None) is not None else 0.0)
            cnt = getattr(stp.reflection, "content", "Step executed.") or "Step executed."
        else:
            rew = 0.0
            cnt = "Step completed."

        reflection_diagnostics.append(ReflectionDiagnostic(
            step_number=s_num,
            plan_step_number=p_num,
            step_description=getattr(stp, "description", "") or "",
            thoughts=getattr(stp, "thoughts", None),
            content=cnt,
            reward=rew,
            remaining_budget=int(getattr(stp, "remaining_budget", 0) or 0),
        ))

        events.append(_create_run_event(
            event_type="reflection_recorded",
            mode="advanced_pipeline",
            step_number=s_num,
            plan_step_number=p_num,
            summary=f"Step {s_num} reflection recorded (reward: {rew:.2f}).",
            details={"reward": rew, "remaining_budget": getattr(stp, "remaining_budget", 0)},
        ))

    # 7. Final Output & Output Type
    final_output_str = getattr(interaction, "answer", None)
    if not final_output_str or not str(final_output_str).strip():
        planstep_outputs = getattr(interaction, "planstep_outputs", []) or []
        if planstep_outputs and hasattr(planstep_outputs[-1], "final_output"):
            final_output_str = str(planstep_outputs[-1].final_output)
        elif getattr(interaction, "steps", []):
            final_output_str = str(interaction.steps[-1].description)
        else:
            final_output_str = "Advanced reasoning pipeline completed successfully."

    # Determine output type
    resolved_output_type: Optional[OutputType] = None
    if task_obj and hasattr(task_obj, "output_type") and isinstance(task_obj.output_type, OutputType):
        resolved_output_type = task_obj.output_type
    else:
        resolved_output_type = _infer_output_type(content=final_output_str, task_obj=task_obj)

    events.append(_create_run_event(
        event_type="output_finalized",
        mode="advanced_pipeline",
        summary="Pipeline reasoning completed and final output synthesized.",
        details={"output_type": resolved_output_type.output_type, "file_extension": resolved_output_type.file_extension},
    ))

    # 8. Synthesize readable chat transcript for UI
    conversation_transcript: List[Dict[str, Any]] = [
        {"role": "user", "name": "User", "content": task_prompt},
        {
            "role": "assistant",
            "name": "Planner",
            "content": f"Task complexity assessed at {complexity_score:.2f}. Generated {len(plan_step_diagnostics)} plan steps with initial budget {initial_budget} (adjusted to {adjusted_budget}).",
        },
    ]

    for ps in plan_step_diagnostics:
        if ps.completed or ps.step_output:
            conversation_transcript.append({
                "role": "assistant",
                "name": ps.step_name or f"Step {ps.step_number}",
                "content": ps.step_output or ps.step_description,
            })

    conversation_transcript.append({
        "role": "assistant",
        "name": "Finalizer",
        "content": final_output_str,
    })

    duration_ms = int((time.perf_counter() - start_time) * 1000)
    diagnostics = RunDiagnostics(
        complexity_score=complexity_score,
        initial_budget=initial_budget,
        adjusted_budget=adjusted_budget,
        final_reward=final_reward,
        plan_steps=plan_step_diagnostics,
        reflections=reflection_diagnostics,
        agent_attributions=agent_attributions,
        duration_ms=duration_ms,
        correlation_id=correlation_id,
    )

    return RunResult(
        mode="advanced_pipeline",
        status="completed",
        final_output=final_output_str,
        output_type=resolved_output_type,
        conversation=conversation_transcript,
        questions=[],
        diagnostics=diagnostics,
        events=events,
        error=None,
    )


# =====================================================================
# 7. Unified Dispatcher Entry Point
# =====================================================================

def execute_run(request: RunRequest) -> RunResult:
    """
    Unified execution dispatcher for the Advanced Prompting System.
    Dispatches to 'collaborative', 'advanced_pipeline', or gated 'tree_of_thought'.
    Guarantees fail-closed error handling wrapping unhandled exceptions into status='error'.
    """
    correlation_id = f"run_{uuid.uuid4().hex[:12]}"
    start_time = time.perf_counter()

    try:
        if request.mode == "tree_of_thought":
            return _execute_tree_of_thought(request, correlation_id, start_time)
        elif request.mode == "collaborative":
            return _execute_collaborative(request, correlation_id, start_time)
        elif request.mode == "advanced_pipeline":
            return _execute_advanced_pipeline(request, correlation_id, start_time)
        else:
            raise ValueError(f"Unknown or unsupported execution mode: '{request.mode}'")
    except Exception as e:
        duration_ms = int((time.perf_counter() - start_time) * 1000)
        err_type = type(e).__name__
        err_msg = str(e) or err_type
        err_details = traceback.format_exc()

        error_obj = ErrorResult(
            error_type=err_type,
            error_message=err_msg,
            error_details=err_details,
            timestamp=datetime.now(timezone.utc),
        )

        error_event = _create_run_event(
            event_type="run_error",
            mode=getattr(request, "mode", "unknown"),
            summary=f"Execution error in {getattr(request, 'mode', 'unknown')} mode: {err_msg}",
            details={
                "error_type": err_type,
                "error_message": err_msg,
                "error_details": err_details,
                "correlation_id": correlation_id,
            },
        )

        # Build fallback conversation transcript
        fallback_conv: List[Dict[str, Any]] = []
        if getattr(request, "conversation", None):
            fallback_conv = _normalize_conversation_history(request.conversation)
        elif getattr(request, "task_prompt", None):
            fallback_conv = [{"role": "user", "name": "User", "content": request.task_prompt}]

        return RunResult(
            mode=getattr(request, "mode", "collaborative"),
            status="error",
            final_output="",
            output_type=OutputType(output_type="Simple Concise Answer", file_extension=".txt"),
            conversation=fallback_conv,
            questions=[],
            diagnostics=RunDiagnostics(
                duration_ms=duration_ms,
                correlation_id=correlation_id,
            ),
            events=[error_event],
            error=error_obj,
        )
```

---

## 5. Verification Method

To verify the registry and dispatcher independently once implemented:

1. **Verify Schema & Registry Imports without Side Effects**:
   ```powershell
   powershell -NoProfile -ExecutionPolicy Bypass -Command "& .\tools\run-offline.ps1 -PytestArgs @('-c', 'pytest.ini', '--import-mode=importlib', '-k', 'test_import_integrity', '-v')"
   ```
   *Expected outcome*: Zero socket calls, zero eager LLM client instantiations.

2. **Verify Tree of Thoughts Zero-Network Fail-Closed Gating**:
   ```python
   # In test_execution_modes.py
   def test_execute_run_tree_of_thought_gated():
       req = RunRequest(mode="tree_of_thought", task_prompt="Explore reasoning tree")
       res = execute_run(req)
       assert res.mode == "tree_of_thought"
       assert res.status == "gated"
       assert "gated pending Issue #36" in res.final_output
       assert res.error is not None
       assert res.error.error_type == "ModeGatedError"
       assert len(res.events) == 1
       assert res.events[0].event_type == "mode_gated"
   ```
   *Expected outcome*: Passes offline with `forbid_unconfigured_provider` active.

3. **Verify Collaborative Mode Dispatching & Attributions**:
   ```python
   # In test_execution_modes.py (with mocked run_conversation)
   def test_execute_run_collaborative(monkeypatch):
       mock_history = [
           {"role": "user", "round": 0, "content": "Test problem"},
           {"role": "assistant", "name": "Analytical Thinker", "round": 1, "content": "Analysis"},
       ]
       monkeypatch.setattr(
           "conversation_manager.run_conversation",
           lambda *args, **kwargs: (mock_history, [{"assistant": "Analytical Thinker", "question": "Why?"}], "Final answer")
       )
       req = RunRequest(mode="collaborative", task_prompt="Test problem")
       res = execute_run(req)
       assert res.status == "completed"
       assert res.final_output == "Final answer"
       assert len(res.diagnostics.agent_attributions) >= 1
       assert res.diagnostics.agent_attributions[0].agent_name == "User"
       assert res.diagnostics.agent_attributions[1].agent_name == "Analytical Thinker"
   ```

4. **Verify Advanced Pipeline Dispatching & Diagnostics**:
   ```python
   # In test_execution_modes.py (with mocked AdvancedPromptEngineer)
   def test_execute_run_advanced_pipeline(monkeypatch):
       # Mock AdvancedPromptEngineer.main to return Interaction fixture
       ...
       req = RunRequest(mode="advanced_pipeline", task_prompt="Plan a web app")
       res = execute_run(req)
       assert res.status == "completed"
       assert res.diagnostics.complexity_score is not None
       assert len(res.diagnostics.plan_steps) > 0
       assert len(res.events) >= 5
   ```

5. **Verify Error Handling & Wrapping**:
   ```python
   def test_execute_run_unhandled_exception(monkeypatch):
       monkeypatch.setattr("conversation_manager.run_conversation", lambda *a, **k: 1 / 0)
       req = RunRequest(mode="collaborative", task_prompt="Trigger crash")
       res = execute_run(req)
       assert res.status == "error"
       assert res.error is not None
       assert res.error.error_type == "ZeroDivisionError"
       assert any(e.event_type == "run_error" for e in res.events)
   ```

6. **Full Offline Test Suite Run**:
   ```powershell
   powershell -NoProfile -ExecutionPolicy Bypass -Command "& .\tools\run-offline.ps1 -PytestArgs @('-q')"
   ```
   *Expected outcome*: All existing 148 backend tests + all new `test_execution_modes.py` tests pass cleanly with 0 failures.
