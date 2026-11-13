"""
execution_modes.py

Unified execution contracts, capability registry, and multi-mode dispatcher
for the Advanced Prompting System (Issue #35 Phases 3–5 & Issue #21).

Defines strongly typed Pydantic v2 schemas for request validation, execution results,
chronological event telemetry, and multi-agent reasoning diagnostics.
"""

from __future__ import annotations

import contextlib
import inspect
import json
import logging
import os
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator, model_validator

# Authoritative domain models (safe DAG: assistant_personalities/output_generator/conversation_manager -> execution_modes -> schema)
from assistant_personalities import assistant_personalities as PERSONALITY_CATALOG
from conversation_manager import ErrorResult, OutputType
from output_generator import (
    ArtifactRef,
    artifact_run_scope,
    is_tracked_artifact,
    list_run_artifacts,
    read_tracked_artifact,
)
import provider_client
from provider_client import (
    DisallowedModelError,
    RunBudgetExceededError,
    RunCancelledError,
    RunTimeoutError,
    get_active_execution_context,
    get_effective_model,
    provider_execution_scope,
)

_logger = logging.getLogger(__name__)


# ============================================================================
# 1. Type Aliases, Literal Enums & Server Provider Policy
# ============================================================================

ExecutionModeId = Literal["collaborative", "advanced_pipeline", "tree_of_thought"]

RunStatus = Literal[
    "completed",
    "gated",
    "error",
    "cancelled",
    "budget_exhausted",
    "timeout",
    "incomplete",
]

MAX_TASK_PROMPT_CHARS = 16000
MAX_CONTEXT_CHARS = 16000
MAX_CONVERSATION_TURNS = 100
MAX_NUM_ROUNDS = 10
MAX_PIPELINE_STEPS = 50
MAX_INITIAL_BUDGET = 50
MAX_AGENTS = 10
_DEFAULT_ALLOWED_MODELS = ("gpt-4o-mini", "gpt-4o", "o3-mini")


class ProviderConfig(BaseModel):
    """Server-owned provider configuration, model allowlist, and execution ceilings."""

    default_model: str = Field(default="gpt-4o-mini")
    allowed_models: List[str] = Field(default_factory=lambda: list(_DEFAULT_ALLOWED_MODELS))
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    timeout_seconds: float = Field(default=120.0, gt=0.0, le=600.0)
    max_calls: int = Field(default=40, ge=1, le=200)


class ProviderUnavailableError(RuntimeError):
    """Raised when live provider execution is requested without configured credentials."""


class EmptyPipelineError(RuntimeError):
    """Raised when the reasoning pipeline yields neither steps nor a final answer."""


_ERROR_SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_\-]{3,}"),
    re.compile(r"(?i)\b(bearer\s+)[A-Za-z0-9._\-]{4,}"),
    re.compile(
        r"(?i)\b((?:api[_-]?key|token|secret|password|authorization)\s*[:=]\s*['\"]?)[^\s'\",;]+"
    ),
)
_ERROR_PATH_PATTERNS = (
    re.compile(r'[Ff]ile\s+"[^"]+"'),
    re.compile(r"[A-Za-z]:[\\/][^\s'\",;)\]]+"),
    re.compile(r"\\\\[^\s'\",;)\]]+"),
    re.compile(
        r"(?<![A-Za-z0-9_])/(?:home|Users|tmp|var|app|srv|etc|opt|usr|root|mnt|workspace|workspaces)(?:/[^\s'\",;)\]]*)?"
    ),
)


def sanitize_public_error_message(
    exc: Union[Exception, str, None],
    *,
    default_message: str = "Execution failed.",
    max_length: int = 400,
) -> str:
    """Return a concise public-safe error message with secrets and filesystem paths redacted."""
    if isinstance(exc, Exception):
        raw = str(exc or "").strip() or exc.__class__.__name__
    elif exc is None:
        raw = ""
    else:
        raw = str(exc).strip()
    msg = raw or default_message

    msg = _ERROR_SECRET_PATTERNS[0].sub("[REDACTED]", msg)
    msg = _ERROR_SECRET_PATTERNS[1].sub(r"\1[REDACTED]", msg)
    msg = _ERROR_SECRET_PATTERNS[2].sub(r"\1[REDACTED]", msg)

    msg = _ERROR_PATH_PATTERNS[0].sub('File "[PATH]"', msg)
    for path_pat in _ERROR_PATH_PATTERNS[1:]:
        msg = path_pat.sub("[PATH]", msg)

    msg = re.sub(r"Traceback \(most recent call last\):?", "", msg, flags=re.IGNORECASE)
    msg = re.sub(r'File "\[PATH\]"(?:,\s*line\s*\d+)?(?:,\s*in\s+[A-Za-z0-9_<>.]+)?', "", msg)

    msg = re.sub(r"\s+", " ", msg).strip()
    if len(msg) > max_length:
        msg = msg[: max_length - 3] + "..."
    return msg or default_message


def _build_safe_error_artifacts(
    exc: Union[Exception, ErrorResult, str],
    *,
    mode: str,
    stop_reason: str,
    correlation_id: str,
    error_type: Optional[str] = None,
    summary_prefix: Optional[str] = None,
) -> Tuple[ErrorResult, RunEvent]:
    """Build one sanitized public ErrorResult and matching run_error RunEvent."""
    if isinstance(exc, ErrorResult):
        resolved_type = error_type or exc.error_type or "ExecutionError"
        safe_msg = sanitize_public_error_message(
            exc.error_message, default_message=resolved_type
        )
        safe_details = (
            sanitize_public_error_message(exc.error_details, default_message=safe_msg)
            if exc.error_details
            else safe_msg
        )
    elif isinstance(exc, Exception):
        resolved_type = error_type or type(exc).__name__
        safe_msg = sanitize_public_error_message(exc, default_message=resolved_type)
        safe_details = safe_msg
    else:
        resolved_type = error_type or "ExecutionError"
        safe_msg = sanitize_public_error_message(exc, default_message=resolved_type)
        safe_details = safe_msg

    prefix = summary_prefix or f"Execution error in {mode} mode"
    safe_summary = f"{prefix}: {safe_msg}"

    error_obj = ErrorResult(
        error_type=resolved_type,
        error_message=safe_msg,
        error_details=safe_details,
        timestamp=datetime.now(timezone.utc),
    )
    error_event = _create_run_event(
        event_type="run_error",
        mode=mode,
        summary=safe_summary,
        details={
            "error_type": resolved_type,
            "error_message": safe_msg,
            "correlation_id": correlation_id,
            "stop_reason": stop_reason,
        },
    )
    return error_obj, error_event


def get_provider_config() -> ProviderConfig:
    """Resolve server-owned provider policy from environment variables."""
    default_model = os.environ.get("APS_DEFAULT_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    raw_allowed = os.environ.get("APS_ALLOWED_MODELS", "")
    if raw_allowed.strip():
        allowed = [m.strip() for m in raw_allowed.split(",") if m.strip()]
    else:
        allowed = list(_DEFAULT_ALLOWED_MODELS)
    if default_model not in allowed:
        allowed.insert(0, default_model)

    def _safe_float(env_key: str, default: float, lo: float, hi: float) -> float:
        raw = os.environ.get(env_key, "")
        if not raw.strip():
            return default
        try:
            val = float(raw)
            return min(max(val, lo), hi)
        except ValueError:
            return default

    def _safe_int(env_key: str, default: int, lo: int, hi: int) -> int:
        raw = os.environ.get(env_key, "")
        if not raw.strip():
            return default
        try:
            val = int(raw)
            return min(max(val, lo), hi)
        except ValueError:
            return default

    return ProviderConfig(
        default_model=default_model,
        allowed_models=allowed,
        temperature=_safe_float("APS_DEFAULT_TEMPERATURE", 0.7, 0.0, 2.0),
        timeout_seconds=_safe_float("APS_PROVIDER_TIMEOUT_SECONDS", 120.0, 0.05, 600.0),
        max_calls=_safe_int("APS_PROVIDER_MAX_CALLS", 40, 1, 200),
    )


def validate_provider_model(
    requested_model: Optional[str], config: Optional[ProviderConfig] = None
) -> str:
    """Resolve and strictly enforce the server-owned model allowlist."""
    cfg = config or get_provider_config()
    candidate = str(requested_model or "").strip() or cfg.default_model
    if candidate not in cfg.allowed_models:
        raise DisallowedModelError(
            f"Requested model '{candidate}' is not permitted. Allowed models: {', '.join(cfg.allowed_models)}."
        )
    return candidate

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
        default_factory=lambda: f"evt_{uuid.uuid4().hex[:8]}",
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
    subtask_description: str = Field(default="", description="Detailed description of the subtask goal.")
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
        description="Assessed linguistic or structural complexity score (0.0 to 1.0+).",
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
    duration_ms: Optional[float] = Field(
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
        description="Chronological attributions for contributing agents.",
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
    ready: bool = Field(
        default=True,
        description="Whether runtime dependencies/credentials are ready to execute this mode.",
    )
    unavailable_reason: Optional[str] = Field(
        default=None,
        description="Human-readable reason when mode is gated or not currently ready.",
    )
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
    supports_cancellation: bool = Field(
        default=False,
        description="Whether mid-run server-side cancellation is supported.",
    )
    supports_resume: bool = Field(
        default=False,
        description="Whether pausing and resuming an active run is supported.",
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
    conversation: List[Any] = Field(
        default_factory=list,
        description="Multi-turn conversation history [{'role': 'user', 'content': '...', 'name': '...'}].",
    )
    assistant_personalities: List[str] = Field(
        default_factory=lambda: ["Analytical Thinker", "Mediator"],
        description="Debate personalities selected for multi-agent reasoning.",
    )
    lead_personality: str = Field(
        default="Mediator",
        description="Primary coordinating personality for consensus moderation.",
    )
    num_rounds: int = Field(
        default=3,
        ge=1,
        le=MAX_NUM_ROUNDS,
        description="Number of debate deliberation rounds for collaborative mode.",
    )
    max_steps: int = Field(
        default=20,
        ge=1,
        le=MAX_PIPELINE_STEPS,
        description="Maximum reasoning steps allowed in advanced_pipeline mode.",
    )
    initial_budget: int = Field(
        default=20,
        ge=1,
        le=MAX_INITIAL_BUDGET,
        description="Initial reasoning step budget before complexity adjustment.",
    )
    agents: int = Field(
        default=3,
        ge=0,
        le=MAX_AGENTS,
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
    correlation_id: Optional[str] = Field(
        default=None,
        description="Optional request correlation identifier.",
    )

    @field_validator("task_prompt")
    @classmethod
    def check_task_prompt_length(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and len(str(v)) > MAX_TASK_PROMPT_CHARS:
            raise ValueError(f"task_prompt exceeds maximum allowed length of {MAX_TASK_PROMPT_CHARS} characters.")
        return v

    @field_validator("context")
    @classmethod
    def check_context_length(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and len(str(v)) > MAX_CONTEXT_CHARS:
            raise ValueError(f"context exceeds maximum allowed length of {MAX_CONTEXT_CHARS} characters.")
        return v

    @field_validator("assistant_personalities")
    @classmethod
    def validate_personalities(cls, v: List[str]) -> List[str]:
        cleaned: List[str] = []
        for item in v or []:
            name = str(item or "").strip()
            if not name:
                continue
            if name not in PERSONALITY_CATALOG:
                raise ValueError(f"Unknown assistant personality '{name}'.")
            if name not in cleaned:
                cleaned.append(name)
        if not cleaned:
            raise ValueError("At least one valid assistant personality must be selected.")
        return cleaned

    @field_validator("lead_personality")
    @classmethod
    def validate_lead_personality(cls, v: str) -> str:
        name = str(v or "").strip()
        if not name or name not in PERSONALITY_CATALOG:
            raise ValueError(f"Unknown lead personality '{name}'.")
        return name

    @model_validator(mode="after")
    def sync_prompt_and_conversation(self) -> "RunRequest":
        """Bidirectionally synchronize task_prompt and conversation history.

        1. Normalizes conversation elements into dictionary representations.
        2. Ignores UI 'system' banners when deriving task_prompt from conversation.
        3. If only task_prompt is provided and conversation is empty, synthesizes
           an initial user message in conversation.
        """
        if len(self.conversation) > MAX_CONVERSATION_TURNS:
            raise ValueError(
                f"conversation exceeds maximum allowed length of {MAX_CONVERSATION_TURNS} messages."
            )

        if (
            self.lead_personality not in self.assistant_personalities
            and self.lead_personality != "Mediator"
        ):
            raise ValueError(
                f"lead_personality '{self.lead_personality}' must be in assistant_personalities or 'Mediator'."
            )

        normalized_conv: List[Dict[str, Any]] = []
        for item in self.conversation:
            if isinstance(item, dict):
                role = item.get("role") or "user"
                name = item.get("name") or ("User" if role == "user" else "")
                content_val = item.get("content", "")
                content = "" if content_val is None else str(content_val)
                normalized_conv.append({
                    "role": role,
                    "name": name,
                    "content": content,
                })
            elif hasattr(item, "role") and hasattr(item, "content"):
                raw_role = getattr(item, "role", "user")
                role = raw_role if raw_role else "user"
                name_val = getattr(item, "name", "")
                name = name_val or ("User" if role == "user" else "")
                raw_content = getattr(item, "content", "")
                content = "" if raw_content is None else str(raw_content)
                normalized_conv.append({
                    "role": role,
                    "name": name,
                    "content": content,
                })
            else:
                normalized_conv.append({
                    "role": "user",
                    "name": "User",
                    "content": "" if item is None else str(item),
                })
        self.conversation = normalized_conv

        if self.task_prompt is not None:
            self.task_prompt = str(self.task_prompt).strip()

        has_prompt = bool(self.task_prompt)
        if not has_prompt:
            extracted = None
            for msg in reversed(self.conversation):
                if (
                    str(msg.get("role", "")).lower() == "user"
                    and msg.get("content")
                    and str(msg["content"]).strip()
                ):
                    extracted = str(msg["content"]).strip()
                    break
            if not extracted and self.conversation:
                for msg in reversed(self.conversation):
                    if (
                        str(msg.get("role", "")).lower() != "system"
                        and msg.get("content")
                        and str(msg["content"]).strip()
                    ):
                        extracted = str(msg["content"]).strip()
                        break
            if extracted:
                self.task_prompt = extracted

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
        description="The requested execution mode.",
    )
    effective_mode: Optional[ExecutionModeId] = Field(
        default=None,
        description="The actual mode that executed the request.",
    )
    status: RunStatus = Field(
        ...,
        description="High-level execution outcome status.",
    )
    stop_reason: Optional[str] = Field(
        default=None,
        description="Machine-readable reason execution terminated (e.g. 'final_answer', 'gated', 'budget_exhausted', 'error').",
    )
    final_output: str = Field(
        default="",
        description="Final answer text or generated artifact preview (empty on error/gated).",
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
    artifacts: List[ArtifactRef] = Field(
        default_factory=list,
        description="Run-scoped generated output artifact descriptors with SHA-256 checksums.",
    )
    usage: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Measured usage telemetry if available, or explicit availability status.",
    )
    error: Optional[ErrorResult] = Field(
        default=None,
        description="Structured error metadata if execution failed or was gated.",
    )

    @model_validator(mode="after")
    def ensure_effective_mode(self) -> "RunResult":
        if self.effective_mode is None:
            self.effective_mode = self.mode
        return self

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


# ============================================================================
# 6. Capability Registry
# ============================================================================

CAPABILITY_REGISTRY: Dict[ExecutionModeId, ExecutionModeCapability] = {
    "collaborative": ExecutionModeCapability(
        id="collaborative",
        label="Collaborative Debate",
        description="Multi-agent round deliberation and consensus voting via conversation manager.",
        enabled=True,
        ready=True,
        unavailable_reason=None,
        gated_issue=None,
        supports_multi_agent=True,
        supports_step_planning=False,
        supports_cancellation=False,
        supports_resume=False,
    ),
    "advanced_pipeline": ExecutionModeCapability(
        id="advanced_pipeline",
        label="Standalone Pipeline",
        description="Complexity-driven step planning, reflections, reward judging, and multi-strategy reasoning.",
        enabled=True,
        ready=True,
        unavailable_reason=None,
        gated_issue=None,
        supports_multi_agent=True,
        supports_step_planning=True,
        supports_cancellation=False,
        supports_resume=False,
    ),
    "tree_of_thought": ExecutionModeCapability(
        id="tree_of_thought",
        label="Tree of Thoughts",
        description="Tree of thoughts exploration gated pending Issue #36.",
        enabled=False,
        ready=False,
        unavailable_reason="Tree of Thoughts execution mode is gated pending Issue #36.",
        gated_issue="Issue #36",
        supports_multi_agent=False,
        supports_step_planning=True,
        supports_cancellation=False,
        supports_resume=False,
    ),
}


def get_capability(mode: str) -> Optional[ExecutionModeCapability]:
    """Retrieve capability descriptor for a given execution mode ID."""
    return CAPABILITY_REGISTRY.get(mode)  # type: ignore[arg-type]


def list_capabilities(check_provider: bool = False) -> List[ExecutionModeCapability]:
    """Return execution mode capabilities, optionally reflecting live provider readiness."""
    caps = [cap.model_copy() for cap in CAPABILITY_REGISTRY.values()]
    enforce = check_provider or os.environ.get("APS_ENFORCE_PROVIDER_READY") == "1"
    if enforce:
        import provider_client

        if not provider_client.has_configured_provider():
            for cap in caps:
                if cap.enabled:
                    cap.ready = False
                    cap.unavailable_reason = (
                        "LLM provider credentials (OPENAI_API_KEY) are not configured on the server."
                    )
    return caps


# ============================================================================
# 7. Internal Helpers
# ============================================================================

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
    Safely determine OutputType from content or file path, defaulting to 'Simple Concise Answer' (.txt).
    """
    try:
        # Check direct JSON structure first
        if content and isinstance(content, str):
            stripped = content.strip()
            if (stripped.startswith("{") and stripped.endswith("}")) or (stripped.startswith("[") and stripped.endswith("]")):
                try:
                    json.loads(stripped)
                    return OutputType(output_type="JSON", file_extension=".json")
                except Exception:
                    pass

        from advanced_prompting import AdvancedPromptEngineer, PromptEngineeringConfig
        from provider_client import get_effective_model

        engineer = AdvancedPromptEngineer(PromptEngineeringConfig(model=get_effective_model()))
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
            entry: Dict[str, Any] = {
                "role": str(item.get("role", "assistant")),
                "name": str(item.get("name", "")),
                "content": str(item.get("content", "")),
            }
            if "round" in item:
                entry["round"] = item["round"]
            normalized.append(entry)
        elif hasattr(item, "content"):
            entry = {
                "role": getattr(item, "role", "assistant"),
                "name": getattr(item, "name", ""),
                "content": str(item.content),
            }
            if hasattr(item, "round"):
                entry["round"] = getattr(item, "round")
            normalized.append(entry)
        elif isinstance(item, str):
            normalized.append({"role": "assistant", "name": "", "content": item})
    return normalized


# ============================================================================
# 8. Execution Handlers
# ============================================================================

def _execute_tree_of_thought(
    request: RunRequest,
    correlation_id: str,
    start_time: float,
) -> RunResult:
    """
    Zero-network fail-closed gating for Tree of Thoughts execution mode (Issue #36).
    Guarantees no provider client, LLM calls, or network sockets are accessed.
    """
    duration_ms = (time.perf_counter() - start_time) * 1000
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
        effective_mode="tree_of_thought",
        status="gated",
        stop_reason="gated",
        final_output="",
        output_type=OutputType(output_type="Simple Concise Answer", file_extension=".txt"),
        conversation=conversation_transcript,
        questions=[],
        diagnostics=RunDiagnostics(
            duration_ms=duration_ms,
            correlation_id=correlation_id,
        ),
        events=[event],
        artifacts=[],
        usage={"available": False, "reason": "Mode gated pending Issue #36"},
        error=error_obj,
    )


def _strip_system_banners(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Exclude UI/system banners before passing conversation history to runtime engines."""
    return [
        msg
        for msg in (history or [])
        if str(msg.get("role", "")).lower() != "system"
        and msg.get("content") is not None
        and str(msg.get("content", "")).strip() != ""
    ]


def _execute_collaborative(
    request: RunRequest,
    correlation_id: str,
    start_time: float,
    provider_config: Optional[ProviderConfig] = None,
) -> RunResult:
    """
    Execute collaborative debate mode via conversation_manager.run_conversation.
    Extracts attributions, questions, OutputType, artifacts, and timeline telemetry.
    """
    from conversation_manager import run_conversation

    cfg = provider_config or get_provider_config()
    resolved_model = validate_provider_model(request.model, cfg)
    filtered_history = _strip_system_banners(request.conversation)
    if not (request.task_prompt and request.task_prompt.strip()) and not filtered_history:
        raise ValueError("Collaborative mode requires a non-empty user task prompt.")

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
            "model": resolved_model,
        },
    ))

    # 2. Problem defined event
    events.append(_create_run_event(
        event_type="problem_defined",
        mode="collaborative",
        summary=f"Problem statement formulated for {len(request.assistant_personalities)} debate participants.",
        details={"task_prompt": request.task_prompt},
    ))

    # 3. Invoke run_conversation cleanly (forwarding provider kwargs when supported by callable)
    problem_input: Any = filtered_history if filtered_history else (request.task_prompt or "")
    rc_kwargs: Dict[str, Any] = {
        "problem_statement": problem_input,
        "selected_personalities": request.assistant_personalities,
        "lead_personality": request.lead_personality,
        "num_rounds": request.num_rounds,
        "context": request.context,
    }
    try:
        sig = inspect.signature(run_conversation)
        params = sig.parameters
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        if has_var_kw or "model" in params:
            rc_kwargs["model"] = resolved_model
        if has_var_kw or "temperature" in params:
            rc_kwargs["temperature"] = request.temperature
        if has_var_kw or "timeout_seconds" in params:
            rc_kwargs["timeout_seconds"] = cfg.timeout_seconds
        if has_var_kw or "max_calls" in params:
            rc_kwargs["max_calls"] = cfg.max_calls
    except (TypeError, ValueError):
        pass

    scope_cm = (
        contextlib.nullcontext()
        if provider_client.get_active_execution_context() is not None
        else provider_client.provider_execution_scope(
            default_model=resolved_model,
            allowed_models=cfg.allowed_models,
            temperature=request.temperature,
            timeout_seconds=cfg.timeout_seconds,
            max_calls=cfg.max_calls,
        )
    )
    with scope_cm:
        raw_history, raw_questions, raw_final_output = run_conversation(**rc_kwargs)

    # 4. Check for ErrorResult from run_conversation
    if isinstance(raw_final_output, ErrorResult):
        error_obj, error_event = _build_safe_error_artifacts(
            raw_final_output,
            mode="collaborative",
            stop_reason="provider_error",
            correlation_id=correlation_id,
            error_type=raw_final_output.error_type,
            summary_prefix="Collaborative deliberation returned error",
        )
        events.append(error_event)
        duration_ms = (time.perf_counter() - start_time) * 1000
        return RunResult(
            mode="collaborative",
            effective_mode="collaborative",
            status="error",
            stop_reason="provider_error",
            final_output="",
            output_type=OutputType(output_type="Simple Concise Answer", file_extension=".txt"),
            conversation=_normalize_conversation_history(raw_history or []),
            questions=[],
            diagnostics=RunDiagnostics(duration_ms=duration_ms, correlation_id=correlation_id),
            events=events,
            artifacts=list_run_artifacts(run_id=correlation_id),
            usage={"available": False, "reason": "Provider telemetry unavailable on error"},
            error=error_obj,
        )

    # 5. Normalize conversation history & extract assistant attributions
    normalized_history = _normalize_conversation_history(raw_history or [])
    agent_attributions: List[AgentAttribution] = []

    for entry in normalized_history:
        role = entry.get("role", "assistant")
        name = entry.get("name", "")
        rnd = int(entry.get("round", 1) or 1)
        content = entry.get("content", "")

        # Extract only assistant/specialist persona contributions
        if role == "assistant" and name.lower() not in ("system", "user"):
            agent_name = name or "Assistant"
            events.append(_create_run_event(
                event_type="agent_contribution",
                mode="collaborative",
                step_number=rnd,
                agent_name=agent_name,
                summary=f"Round {rnd} contribution from {agent_name}",
                details={"round": rnd, "agent_name": agent_name},
            ))

            summary_snippet = content[:140] + "..." if len(content) > 140 else content
            agent_attributions.append(AgentAttribution(
                agent_name=agent_name,
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

    # 7. Output Type resolution and strictly run-scoped artifact content reading
    final_output_str = str(raw_final_output or "")
    file_path = ""
    candidate_path = final_output_str.strip()
    run_artifacts = list_run_artifacts(run_id=correlation_id)

    if candidate_path and is_tracked_artifact(candidate_path, run_id=correlation_id):
        abs_candidate = os.path.realpath(os.path.abspath(candidate_path))
        cwd = os.path.realpath(os.path.abspath(os.getcwd()))
        is_safe_subpath = os.path.commonpath([abs_candidate, cwd]) == cwd
        if not is_safe_subpath:
            try:
                import tempfile
                td = os.path.realpath(os.path.abspath(tempfile.gettempdir()))
                is_safe_subpath = os.path.commonpath([abs_candidate, td]) == td
            except Exception:
                is_safe_subpath = False
        if not is_safe_subpath:
            try:
                from output_generator import get_artifacts_base_dir
                art_base = get_artifacts_base_dir()
                is_safe_subpath = os.path.commonpath([abs_candidate, art_base]) == art_base
            except Exception:
                is_safe_subpath = False

        if is_safe_subpath and os.path.isfile(abs_candidate):
            file_path = abs_candidate
            preview = read_tracked_artifact(abs_candidate, run_id=correlation_id)
            if preview is not None and preview.strip():
                final_output_str = preview
            elif abs_candidate.lower().endswith(".pdf"):
                final_output_str = f"Generated PDF artifact: {os.path.basename(abs_candidate)}"

    resolved_output_type = _infer_output_type(content=final_output_str, file_path=file_path)

    has_output = bool(final_output_str.strip()) or bool(run_artifacts)
    final_status: RunStatus = "completed" if has_output else "incomplete"
    stop_reason = "final_answer" if has_output else "empty_output"

    # 8. Output finalized event
    events.append(_create_run_event(
        event_type="output_finalized",
        mode="collaborative",
        summary=(
            "Collaborative consensus reached and output finalized."
            if has_output
            else "Collaborative deliberation completed without non-empty output."
        ),
        details={
            "output_type": resolved_output_type.output_type,
            "file_extension": resolved_output_type.file_extension,
            "stop_reason": stop_reason,
        },
    ))

    duration_ms = (time.perf_counter() - start_time) * 1000
    diagnostics = RunDiagnostics(
        agent_attributions=agent_attributions,
        duration_ms=duration_ms,
        correlation_id=correlation_id,
    )

    return RunResult(
        mode="collaborative",
        effective_mode="collaborative",
        status=final_status,
        stop_reason=stop_reason,
        final_output=final_output_str,
        output_type=resolved_output_type,
        conversation=normalized_history,
        questions=questions_list,
        diagnostics=diagnostics,
        events=events,
        artifacts=run_artifacts,
        usage={"available": False, "reason": "Provider token usage tracking not reported by client calls"},
        error=None,
    )


def _execute_advanced_pipeline(
    request: RunRequest,
    correlation_id: str,
    start_time: float,
    provider_config: Optional[ProviderConfig] = None,
) -> RunResult:
    """
    Execute standalone reasoning pipeline via AdvancedPromptEngineer.main.
    Extracts complexity, plan steps, subtasks, reflections with rewards, attributions,
    synthesizes chat transcript, artifacts, and produces honest completion status.
    """
    from advanced_prompting import AdvancedPromptEngineer, PromptEngineeringConfig

    cfg = provider_config or get_provider_config()
    resolved_model = validate_provider_model(request.model, cfg)

    task_prompt = (request.task_prompt or "").strip()
    if not task_prompt and request.conversation:
        for msg in reversed(_strip_system_banners(request.conversation)):
            if str(msg.get("role", "")).lower() == "user" and str(msg.get("content", "")).strip():
                task_prompt = str(msg["content"]).strip()
                break
    if not task_prompt:
        raise ValueError("Advanced pipeline mode requires a non-empty user task prompt.")

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
            "model": resolved_model,
        },
    ))

    # 2. Build configuration and instantiate engineer
    config_kwargs: Dict[str, Any] = {
        "max_steps": request.max_steps,
        "initial_budget": request.initial_budget,
        "model": resolved_model,
        "temperature": request.temperature,
        "agents": request.agents,
        "max_sc_calls": cfg.max_calls,
    }
    try:
        cfg_params = inspect.signature(PromptEngineeringConfig).parameters
        if "max_calls" in cfg_params:
            config_kwargs["max_calls"] = cfg.max_calls
        if "timeout_seconds" in cfg_params:
            config_kwargs["timeout_seconds"] = cfg.timeout_seconds
    except (TypeError, ValueError):
        pass
    config = PromptEngineeringConfig(**config_kwargs)
    engineer = AdvancedPromptEngineer(config)

    # 3. Execute main reasoning pipeline inside request-scoped execution context
    scope_cm = (
        contextlib.nullcontext()
        if provider_client.get_active_execution_context() is not None
        else provider_client.provider_execution_scope(
            default_model=resolved_model,
            allowed_models=cfg.allowed_models,
            temperature=request.temperature,
            timeout_seconds=cfg.timeout_seconds,
            max_calls=cfg.max_calls,
        )
    )
    with scope_cm:
        interaction = engineer.main(task_prompt)
        ctx = provider_client.get_active_execution_context()
        if ctx is not None and ctx.termination_reason == "disallowed_model":
            raise DisallowedModelError(
                ctx.termination_message or "Requested model is not permitted by provider policy."
            )

    # 4. Extract task object and complexity
    task_obj = getattr(interaction, "task", None) or getattr(engineer, "task_object", None)
    complexity_score = float(getattr(task_obj, "complexity", 0.0) or 0.0) if task_obj else 0.0
    initial_budget = int(config.initial_budget)
    complexity_factor = getattr(config, "complexity_factor", 5)
    raw_adjusted_budget = int(initial_budget + complexity_score * complexity_factor)
    adjusted_budget = min(raw_adjusted_budget, request.max_steps) if request.max_steps > 0 else raw_adjusted_budget
    final_reward = float(getattr(interaction, "final_reward", 0.0) if getattr(interaction, "final_reward", None) is not None else 0.0)

    events.append(_create_run_event(
        event_type="complexity_assessed",
        mode="advanced_pipeline",
        summary=f"Task complexity assessed at {complexity_score:.2f} (budget: {adjusted_budget}).",
        details={
            "complexity_score": complexity_score,
            "initial_budget": initial_budget,
            "adjusted_budget": adjusted_budget,
            "max_steps": request.max_steps,
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

    if config.agents == 0 and not agent_attributions:
        agent_attributions.append(AgentAttribution(
            agent_name="Advanced Prompt Engineer",
            role="Solo Reasoning Engine",
            round_or_step=1,
            contribution_summary="Executed solo prompt reasoning pipeline without collaborative agent delegation.",
        ))

    # 6. Extract reflections with rewards
    raw_reflections = getattr(interaction, "reflections", []) or []
    reflections_by_step = {getattr(r, "step_number", None): r for r in raw_reflections}
    reflection_diagnostics: List[ReflectionDiagnostic] = []

    for stp in (getattr(interaction, "steps", []) or []):
        s_num = getattr(stp, "step_number", 1)
        p_num = getattr(stp, "plan_step_number", None)
        refl = reflections_by_step.get(s_num)

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

    # 7. Honest completion evaluation & Output Type resolution
    raw_answer = getattr(interaction, "answer", None)
    answer_text = str(raw_answer).strip() if raw_answer is not None else ""
    planstep_outputs = getattr(interaction, "planstep_outputs", []) or []
    raw_steps = getattr(interaction, "steps", []) or []
    sc_term = getattr(engineer, "_last_sc_termination", {}) or {}
    sc_reason = sc_term.get("reason")
    interaction_status = getattr(interaction, "status", None)
    interaction_stop_reason = getattr(interaction, "stop_reason", None)
    active_ctx = provider_client.get_active_execution_context()
    ctx_reason = getattr(active_ctx, "termination_reason", None) if active_ctx is not None else None
    effective_term_reason = interaction_stop_reason or sc_reason or ctx_reason
    all_plan_steps_complete = (
        all(ps.completed for ps in plan_step_diagnostics)
        if plan_step_diagnostics
        else True
    )
    run_artifacts = list_run_artifacts(run_id=correlation_id)

    final_output_str = ""
    final_status: RunStatus
    stop_reason: str
    error_obj: Optional[ErrorResult] = None

    _non_complete_reasons = ("budget_exhausted", "timeout", "cancelled", "incomplete")

    if answer_text:
        final_output_str = str(raw_answer)
    elif planstep_outputs and hasattr(planstep_outputs[-1], "final_output") and str(planstep_outputs[-1].final_output or "").strip():
        final_output_str = str(planstep_outputs[-1].final_output)
    elif raw_steps:
        final_output_str = str(getattr(raw_steps[-1], "description", "") or "")

    if interaction_status in _non_complete_reasons or effective_term_reason in _non_complete_reasons:
        if final_output_str or raw_steps or planstep_outputs:
            candidate_status = (
                interaction_status
                if interaction_status in _non_complete_reasons
                else effective_term_reason
            )
            final_status = candidate_status  # type: ignore[assignment]
            stop_reason = str(effective_term_reason or candidate_status)
        else:
            candidate_status = (
                interaction_status
                if interaction_status in ("budget_exhausted", "timeout", "cancelled")
                else (
                    effective_term_reason
                    if effective_term_reason in ("budget_exhausted", "timeout", "cancelled")
                    else "error"
                )
            )
            stop_reason = str(effective_term_reason or candidate_status)
            err = RuntimeError(
                getattr(active_ctx, "termination_message", None)
                or f"Pipeline stopped ({stop_reason}) before producing output."
            )
            duration_ms = (time.perf_counter() - start_time) * 1000
            error_obj, error_event = _build_safe_error_artifacts(
                err,
                mode="advanced_pipeline",
                stop_reason=stop_reason,
                correlation_id=correlation_id,
                error_type="RunTimeoutError" if stop_reason == "timeout" else (
                    "RunBudgetExceededError" if stop_reason == "budget_exhausted" else type(err).__name__
                ),
            )
            events.append(error_event)
            return RunResult(
                mode="advanced_pipeline",
                effective_mode="advanced_pipeline",
                status=candidate_status,  # type: ignore[arg-type]
                stop_reason=stop_reason,
                final_output="",
                output_type=OutputType(output_type="Simple Concise Answer", file_extension=".txt"),
                conversation=_normalize_conversation_history(request.conversation),
                questions=[],
                diagnostics=RunDiagnostics(
                    complexity_score=complexity_score,
                    initial_budget=initial_budget,
                    adjusted_budget=adjusted_budget,
                    final_reward=final_reward,
                    plan_steps=plan_step_diagnostics,
                    reflections=reflection_diagnostics,
                    agent_attributions=agent_attributions,
                    duration_ms=duration_ms,
                    correlation_id=correlation_id,
                ),
                events=events,
                artifacts=run_artifacts,
                usage={"available": False, "reason": "Provider token usage tracking not reported by client calls"},
                error=error_obj,
            )
    elif answer_text:
        final_output_str = str(raw_answer)
        final_status = "completed"
        stop_reason = "final_answer"
    elif planstep_outputs and hasattr(planstep_outputs[-1], "final_output") and str(planstep_outputs[-1].final_output or "").strip():
        final_output_str = str(planstep_outputs[-1].final_output)
        if not all_plan_steps_complete:
            final_status = "incomplete"
            stop_reason = "incomplete_plan_steps"
        else:
            final_status = "completed"
            stop_reason = "planstep_output"
    elif raw_steps:
        final_output_str = str(getattr(raw_steps[-1], "description", "") or "")
        final_status = "incomplete"
        stop_reason = sc_reason or "missing_final_answer"
    else:
        err = EmptyPipelineError(
            "Advanced reasoning pipeline produced no reasoning steps or final output."
        )
        duration_ms = (time.perf_counter() - start_time) * 1000
        error_obj, error_event = _build_safe_error_artifacts(
            err,
            mode="advanced_pipeline",
            stop_reason="empty_pipeline",
            correlation_id=correlation_id,
            error_type=type(err).__name__,
        )
        events.append(error_event)
        return RunResult(
            mode="advanced_pipeline",
            effective_mode="advanced_pipeline",
            status="error",
            stop_reason="empty_pipeline",
            final_output="",
            output_type=OutputType(output_type="Simple Concise Answer", file_extension=".txt"),
            conversation=_normalize_conversation_history(request.conversation),
            questions=[],
            diagnostics=RunDiagnostics(
                complexity_score=complexity_score,
                initial_budget=initial_budget,
                adjusted_budget=adjusted_budget,
                final_reward=final_reward,
                plan_steps=plan_step_diagnostics,
                reflections=reflection_diagnostics,
                agent_attributions=agent_attributions,
                duration_ms=duration_ms,
                correlation_id=correlation_id,
            ),
            events=events,
            artifacts=run_artifacts,
            usage={"available": False, "reason": "Provider token usage tracking not reported by client calls"},
            error=error_obj,
        )

    resolved_output_type: Optional[OutputType] = None
    if task_obj and hasattr(task_obj, "output_type") and isinstance(task_obj.output_type, OutputType):
        resolved_output_type = task_obj.output_type
    else:
        resolved_output_type = _infer_output_type(content=final_output_str, task_obj=task_obj)

    events.append(_create_run_event(
        event_type="output_finalized",
        mode="advanced_pipeline",
        summary=(
            "Pipeline reasoning completed and final output synthesized."
            if final_status == "completed"
            else f"Pipeline stopped ({stop_reason}) without a complete final answer."
        ),
        details={
            "output_type": resolved_output_type.output_type,
            "file_extension": resolved_output_type.file_extension,
            "stop_reason": stop_reason,
        },
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

    duration_ms = (time.perf_counter() - start_time) * 1000
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
        effective_mode="advanced_pipeline",
        status=final_status,
        stop_reason=stop_reason,
        final_output=final_output_str,
        output_type=resolved_output_type,
        conversation=conversation_transcript,
        questions=[],
        diagnostics=diagnostics,
        events=events,
        artifacts=run_artifacts,
        usage={"available": False, "reason": "Provider token usage tracking not reported by client calls"},
        error=error_obj,
    )


# ============================================================================
# 9. Unified Dispatcher Entry Point
# ============================================================================

def execute_run(request: RunRequest) -> RunResult:
    """
    Unified execution dispatcher for the Advanced Prompting System.
    Dispatches to 'collaborative', 'advanced_pipeline', or gated 'tree_of_thought'.
    Guarantees fail-closed error handling wrapping unhandled exceptions into status='error'.
    """
    correlation_id = request.correlation_id or f"run_{uuid.uuid4().hex[:12]}"
    start_time = time.perf_counter()

    try:
        if request.mode == "tree_of_thought":
            return _execute_tree_of_thought(request, correlation_id, start_time)

        if request.mode not in ("collaborative", "advanced_pipeline"):
            raise ValueError(f"Unknown or unsupported execution mode: '{request.mode}'")

        provider_config = get_provider_config()
        resolved_model = validate_provider_model(request.model, provider_config)

        if os.environ.get("APS_ENFORCE_PROVIDER_READY") == "1":
            if not provider_client.has_configured_provider():
                raise ProviderUnavailableError(
                    "LLM provider credentials (OPENAI_API_KEY) are not configured on the server."
                )

        with artifact_run_scope(correlation_id), provider_client.provider_execution_scope(
            default_model=resolved_model,
            allowed_models=provider_config.allowed_models,
            temperature=getattr(request, "temperature", None),
            timeout_seconds=provider_config.timeout_seconds,
            max_calls=provider_config.max_calls,
        ):
            if request.mode == "collaborative":
                return _execute_collaborative(
                    request, correlation_id, start_time, provider_config=provider_config
                )
            else:
                return _execute_advanced_pipeline(
                    request, correlation_id, start_time, provider_config=provider_config
                )
    except Exception as e:
        duration_ms = (time.perf_counter() - start_time) * 1000
        err_type = type(e).__name__
        raw_err_msg = str(e) or err_type
        _logger.exception("Execution run %s failed (%s)", correlation_id, err_type)

        result_status: RunStatus = "error"
        if err_type == "DisallowedModelError":
            stop_reason = "disallowed_model"
        elif err_type == "ProviderUnavailableError":
            stop_reason = "provider_unavailable"
        elif err_type == "RunBudgetExceededError":
            if "timed out" in raw_err_msg.lower() or "timeout" in raw_err_msg.lower():
                stop_reason = "timeout"
                result_status = "timeout"
            else:
                stop_reason = "budget_exhausted"
                result_status = "budget_exhausted"
        elif err_type == "RunCancelledError":
            stop_reason = "cancelled"
            result_status = "cancelled"
        elif err_type in ("TimeoutError", "RunTimeoutError", "APITimeoutError") or "timeout" in err_type.lower():
            stop_reason = "timeout"
            result_status = "timeout"
        elif err_type == "ValueError":
            stop_reason = "invalid_input"
        else:
            stop_reason = "runtime_error"

        error_obj, error_event = _build_safe_error_artifacts(
            e,
            mode=getattr(request, "mode", "unknown"),
            stop_reason=stop_reason,
            correlation_id=correlation_id,
            error_type=err_type,
        )

        fallback_conv: List[Dict[str, Any]] = []
        if getattr(request, "conversation", None):
            fallback_conv = _normalize_conversation_history(request.conversation)
        elif getattr(request, "task_prompt", None):
            fallback_conv = [{"role": "user", "name": "User", "content": request.task_prompt}]

        return RunResult(
            mode=getattr(request, "mode", "collaborative"),
            effective_mode=getattr(request, "mode", "collaborative"),
            status=result_status,
            stop_reason=stop_reason,
            final_output="",
            output_type=OutputType(output_type="Simple Concise Answer", file_extension=".txt"),
            conversation=fallback_conv,
            questions=[],
            diagnostics=RunDiagnostics(
                duration_ms=duration_ms,
                correlation_id=correlation_id,
            ),
            events=[error_event],
            artifacts=list_run_artifacts(run_id=correlation_id),
            usage={"available": False, "reason": "Provider telemetry unavailable on error"},
            error=error_obj,
        )
