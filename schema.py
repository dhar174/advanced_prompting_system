from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional
import strawberry

logger = logging.getLogger(__name__)

# ============================================================================
# Existing Types (Preserved 100% Backward-Compatible)
# ============================================================================

# Define the Conversation input type
@strawberry.input
class ConversationInput:
    role: str
    name: Optional[str] = None
    content: str

# Define the Feedback input type
@strawberry.input
class FeedbackInput:
    assistant: str
    rating: int
    comments: Optional[str] = None

# Define the Question input type
@strawberry.input
class QuestionInput:
    assistant: str
    question: str

# Define the Question output type
@strawberry.type
class Question:
    assistant: str
    question: str

# Define the Conversation output type (this stays as a type, not input)
@strawberry.type
class Conversation:
    role: str
    name: Optional[str] = None
    content: str

# Define the Feedback output type
@strawberry.type
class Feedback:
    assistant: str
    rating: int

# Define the response types for running a conversation
@strawberry.type
class RunConversationResponse:
    conversation: List[Conversation]
    questions: Optional[List[Question]]
    final_output: str

# Define the response type for submitting feedback
@strawberry.type
class FeedbackResponse:
    success: str


# ============================================================================
# Execution Modes & Diagnostic Types (Issue #35 Phases 3–5 & Issue #21)
# ============================================================================

@strawberry.type
class OutputTypeInfo:
    output_type: str
    file_extension: str

@strawberry.type
class SubtaskDiagnosticType:
    subtask_number: int
    subtask_name: str
    subtask_description: str
    subtask_output: str
    completed: bool

@strawberry.type
class PlanStepDiagnosticType:
    step_number: int
    step_name: str
    step_description: str
    step_explanation: str
    step_output: str
    completed: bool
    assigned_agents: List[str]
    subtasks: List[SubtaskDiagnosticType]

@strawberry.type
class ReflectionDiagnosticType:
    step_number: int
    plan_step_number: Optional[int]
    step_description: str
    thoughts: Optional[str]
    content: str
    reward: float
    remaining_budget: int

@strawberry.type
class AgentAttributionType:
    agent_name: str
    role: str
    round_or_step: int
    contribution_summary: str

@strawberry.type
class RunDiagnosticsType:
    complexity_score: Optional[float] = None
    initial_budget: Optional[int] = None
    adjusted_budget: Optional[int] = None
    final_reward: Optional[float] = None
    duration_ms: Optional[float] = None
    token_count: Optional[int] = None
    correlation_id: Optional[str] = None
    plan_steps: List[PlanStepDiagnosticType] = strawberry.field(default_factory=list)
    reflections: List[ReflectionDiagnosticType] = strawberry.field(default_factory=list)
    agent_attributions: List[AgentAttributionType] = strawberry.field(default_factory=list)

@strawberry.type
class RunEventTypeGQL:
    event_id: str
    event_type: str
    timestamp: str
    mode: str
    step_number: Optional[int] = None
    plan_step_number: Optional[int] = None
    agent_name: Optional[str] = None
    summary: str

@strawberry.type
class ArtifactRefType:
    artifact_id: str
    run_id: str
    filename: str
    path: str
    mime_type: str
    size_bytes: int
    sha256: str


@strawberry.type
class PersonalityCapabilityType:
    id: str
    instructions: str


@strawberry.type
class ProviderPolicyType:
    default_model: str
    allowed_models: List[str]
    default_temperature: float


@strawberry.type
class ExecutionModeCapabilityType:
    id: str
    label: str
    description: str
    enabled: bool
    gated_issue: Optional[str] = None
    supports_multi_agent: bool
    supports_step_planning: bool
    ready: bool = True
    unavailable_reason: Optional[str] = None
    supports_cancellation: bool = False
    supports_resume: bool = False
    min_num_rounds: Optional[int] = None
    max_num_rounds: Optional[int] = None
    max_steps: Optional[int] = None
    max_agents: Optional[int] = None

@strawberry.input
class RunExecutionInput:
    mode: str = "collaborative"
    task_prompt: Optional[str] = None
    conversation: Optional[List[ConversationInput]] = None
    assistant_personalities: Optional[List[str]] = None
    lead_personality: Optional[str] = "Mediator"
    num_rounds: int = 3
    max_steps: int = 20
    initial_budget: int = 20
    agents: int = 3
    model: Optional[str] = None
    temperature: float = 0.7
    context: Optional[str] = None

@strawberry.type
class RunExecutionResponse:
    mode: str
    status: str
    conversation: List[Conversation]
    questions: Optional[List[Question]]
    final_output: str
    output_type: OutputTypeInfo
    diagnostics: RunDiagnosticsType
    events: List[RunEventTypeGQL]
    error_message: Optional[str] = None
    effective_mode: Optional[str] = None
    stop_reason: Optional[str] = None
    artifacts: List[ArtifactRefType] = strawberry.field(default_factory=list)
    usage_available: bool = False


# ============================================================================
# Helper Conversion Functions
# ============================================================================

def _build_run_execution_response(run_result: Any) -> RunExecutionResponse:
    """Map domain RunResult to Strawberry RunExecutionResponse."""
    conversation_objs = [
        Conversation(
            role=entry.get("role", "assistant"),
            name=entry.get("name") or "",
            content=entry.get("content", ""),
        )
        for entry in run_result.conversation
    ]

    question_objs = [
        Question(
            assistant=q.get("assistant", ""),
            question=q.get("question") or q.get("reply") or "",
        )
        for q in run_result.questions
    ]

    output_type_obj = OutputTypeInfo(
        output_type=getattr(run_result.output_type, "output_type", "Simple Concise Answer"),
        file_extension=getattr(run_result.output_type, "file_extension", ".txt"),
    )

    diag = run_result.diagnostics
    plan_steps_gql: List[PlanStepDiagnosticType] = []
    for step in diag.plan_steps:
        subtasks_gql = [
            SubtaskDiagnosticType(
                subtask_number=sub.subtask_number,
                subtask_name=sub.subtask_name,
                subtask_description=sub.subtask_description,
                subtask_output=sub.subtask_output,
                completed=sub.completed,
            )
            for sub in step.subtasks
        ]
        plan_steps_gql.append(
            PlanStepDiagnosticType(
                step_number=step.step_number,
                step_name=step.step_name,
                step_description=step.step_description,
                step_explanation=step.step_explanation,
                step_output=step.step_output,
                completed=step.completed,
                assigned_agents=step.assigned_agents,
                subtasks=subtasks_gql,
            )
        )

    reflections_gql = [
        ReflectionDiagnosticType(
            step_number=refl.step_number,
            plan_step_number=refl.plan_step_number,
            step_description=refl.step_description,
            thoughts=refl.thoughts,
            content=refl.content,
            reward=refl.reward,
            remaining_budget=refl.remaining_budget,
        )
        for refl in diag.reflections
    ]

    attributions_gql = [
        AgentAttributionType(
            agent_name=attr.agent_name,
            role=attr.role,
            round_or_step=attr.round_or_step,
            contribution_summary=attr.contribution_summary,
        )
        for attr in diag.agent_attributions
    ]

    diagnostics_gql = RunDiagnosticsType(
        complexity_score=diag.complexity_score,
        initial_budget=diag.initial_budget,
        adjusted_budget=diag.adjusted_budget,
        final_reward=diag.final_reward,
        duration_ms=diag.duration_ms,
        token_count=diag.token_count,
        correlation_id=diag.correlation_id,
        plan_steps=plan_steps_gql,
        reflections=reflections_gql,
        agent_attributions=attributions_gql,
    )

    from execution_modes import sanitize_public_error_message

    events_gql = [
        RunEventTypeGQL(
            event_id=evt.event_id,
            event_type=evt.event_type,
            timestamp=evt.timestamp,
            mode=evt.mode,
            step_number=evt.step_number,
            plan_step_number=evt.plan_step_number,
            agent_name=evt.agent_name,
            summary=sanitize_public_error_message(
                evt.summary, default_message=evt.event_type
            ),
        )
        for evt in run_result.events
    ]

    error_msg = (
        sanitize_public_error_message(run_result.error.error_message)
        if run_result.error
        else None
    )
    artifacts_gql = [
        ArtifactRefType(
            artifact_id=a.artifact_id,
            run_id=a.run_id,
            filename=a.filename,
            path=f"artifacts/{a.run_id}/{a.filename}",
            mime_type=a.mime_type,
            size_bytes=int(a.size_bytes),
            sha256=a.sha256,
        )
        for a in (getattr(run_result, "artifacts", None) or [])
    ]
    usage_payload = getattr(run_result, "usage", None)
    usage_available = bool(
        isinstance(usage_payload, dict)
        and (usage_payload.get("available") is True or "total_tokens" in usage_payload)
    )

    return RunExecutionResponse(
        mode=run_result.mode,
        status=run_result.status,
        conversation=conversation_objs,
        questions=question_objs,
        final_output=run_result.final_output,
        output_type=output_type_obj,
        diagnostics=diagnostics_gql,
        events=events_gql,
        error_message=error_msg,
        effective_mode=getattr(run_result, "effective_mode", None) or run_result.mode,
        stop_reason=getattr(run_result, "stop_reason", None),
        artifacts=artifacts_gql,
        usage_available=usage_available,
    )


def _sanitize_error_message(exc: Exception) -> str:
    """Return a concise, safe error message without leaking tracebacks, keys, or filesystem paths."""
    from execution_modes import sanitize_public_error_message

    return sanitize_public_error_message(exc)


def _build_error_run_response(
    mode: str, exc: Exception, stop_reason: str = "error"
) -> RunExecutionResponse:
    """Construct a structured error RunExecutionResponse from a validation or execution failure."""
    return RunExecutionResponse(
        mode=str(mode or "collaborative"),
        status="error",
        conversation=[],
        questions=[],
        final_output="",
        output_type=OutputTypeInfo(output_type="Simple Concise Answer", file_extension=".txt"),
        diagnostics=RunDiagnosticsType(),
        events=[],
        error_message=_sanitize_error_message(exc),
        effective_mode=str(mode or "collaborative"),
        stop_reason=stop_reason,
        artifacts=[],
        usage_available=False,
    )


# ============================================================================
# Root Query Class
# ============================================================================

@strawberry.type
class Query:
    @strawberry.field
    def hello(self) -> str:
        return "Hello, World!"

    @strawberry.field
    def execution_modes(self) -> List[ExecutionModeCapabilityType]:
        """Return registered execution modes and capability/readiness flags."""
        from execution_modes import list_capabilities

        capabilities: List[ExecutionModeCapabilityType] = []
        for cap in list_capabilities(check_provider=True):
            capabilities.append(
                ExecutionModeCapabilityType(
                    id=cap.id,
                    label=cap.label,
                    description=cap.description,
                    enabled=cap.enabled,
                    gated_issue=cap.gated_issue,
                    supports_multi_agent=cap.supports_multi_agent,
                    supports_step_planning=cap.supports_step_planning,
                    ready=bool(getattr(cap, "ready", cap.enabled)),
                    unavailable_reason=getattr(cap, "unavailable_reason", None),
                    supports_cancellation=bool(getattr(cap, "supports_cancellation", False)),
                    supports_resume=bool(getattr(cap, "supports_resume", False)),
                    min_num_rounds=getattr(cap, "min_num_rounds", None),
                    max_num_rounds=getattr(cap, "max_num_rounds", None),
                    max_steps=getattr(cap, "max_steps", None),
                    max_agents=getattr(cap, "max_agents", None),
                )
            )
        return capabilities

    @strawberry.field
    def personalities(self) -> List[PersonalityCapabilityType]:
        """Return authoritative backend assistant personalities catalog."""
        from execution_modes import PERSONALITY_CATALOG

        return [
            PersonalityCapabilityType(id=name, instructions=instructions)
            for name, instructions in PERSONALITY_CATALOG.items()
        ]

    @strawberry.field
    def provider_policy(self) -> ProviderPolicyType:
        """Return server-owned model allowlist and default inference policy."""
        from execution_modes import get_provider_config

        cfg = get_provider_config()
        return ProviderPolicyType(
            default_model=cfg.default_model,
            allowed_models=list(cfg.allowed_models),
            default_temperature=float(cfg.temperature),
        )


# ============================================================================
# Root Mutation Class
# ============================================================================

@strawberry.type
class Mutation:
    @strawberry.mutation
    def run_conversation(
        self,
        conversation: List[ConversationInput],
        assistant_personalities: List[str],
        leadPersonality: str,
        num_rounds: int,
    ) -> RunConversationResponse:
        """Legacy conversation mutation routed through unified execution_modes policy & run scope."""
        from execution_modes import (
            MAX_NUM_ROUNDS,
            MAX_TASK_PROMPT_CHARS,
            PERSONALITY_CATALOG,
            RunRequest,
            execute_run,
            get_provider_config,
            sanitize_public_error_message,
        )

        if not conversation:
            raise ValueError("Conversation must contain at least one message.")
        if num_rounds < 1 or num_rounds > MAX_NUM_ROUNDS:
            raise ValueError(f"num_rounds must be between 1 and {MAX_NUM_ROUNDS}.")
        cleaned_personalities = [str(p or "").strip() for p in (assistant_personalities or []) if str(p or "").strip()]
        if not cleaned_personalities:
            raise ValueError("At least one assistant personality must be selected.")
        unknown_p = [p for p in cleaned_personalities if p not in PERSONALITY_CATALOG]
        if unknown_p:
            raise ValueError(f"Unknown assistant personality: {unknown_p[0]}")
        lead_clean = str(leadPersonality or "").strip()
        if lead_clean not in PERSONALITY_CATALOG:
            raise ValueError(f"Unknown lead personality: {lead_clean}")

        conversation_dicts = [
            {"role": c.role, "name": c.name, "content": c.content}
            for c in conversation
            if str(c.role or "").strip().lower() != "system"
        ]
        if not conversation_dicts:
            raise ValueError("Conversation must contain at least one non-system message.")
        if sum(len(str(c.get("content", ""))) for c in conversation_dicts) > MAX_TASK_PROMPT_CHARS * 4:
            raise ValueError("Conversation payload exceeds maximum allowed size.")

        logger.info(
            json.dumps({
                "event": "run_conversation_started",
                "lead_personality": lead_clean,
                "personalities": cleaned_personalities,
                "num_rounds": num_rounds,
                "history_length": len(conversation_dicts),
            })
        )

        try:
            cfg = get_provider_config()
            req = RunRequest(
                mode="collaborative",
                conversation=conversation_dicts,
                assistant_personalities=cleaned_personalities,
                lead_personality=lead_clean,
                num_rounds=num_rounds,
                model=cfg.default_model,
                temperature=cfg.temperature,
            )
            result = execute_run(req)
        except ValueError as exc:
            raise ValueError(_sanitize_error_message(exc)) from None
        except Exception as exc:
            logger.exception("run_conversation failed")
            raise RuntimeError(f"Conversation execution failed: {_sanitize_error_message(exc)}") from None

        if result.status != "completed" or result.error is not None:
            raw_err = (
                result.error.error_message
                if result.error and result.error.error_message
                else f"Conversation execution did not complete ({result.stop_reason or result.status})."
            )
            safe_err = sanitize_public_error_message(raw_err)
            if result.stop_reason in ("invalid_input", "disallowed_model"):
                raise ValueError(safe_err)
            raise RuntimeError(f"Conversation execution failed: {safe_err}")

        conversation_objs = [
            Conversation(
                role=entry.get("role", "assistant"),
                name=entry.get("name") or "",
                content=entry.get("content", ""),
            )
            for entry in result.conversation
        ]

        question_objs = [
            Question(
                assistant=q.get("assistant", ""),
                question=q.get("question") or q.get("reply") or "",
            )
            for q in result.questions
        ]

        return RunConversationResponse(
            conversation=conversation_objs,
            questions=question_objs,
            final_output=result.final_output,
        )

    @strawberry.mutation
    def submit_feedback(
        self,
        conversation: List[ConversationInput],
        feedback: List[FeedbackInput],
        questions: Optional[List[QuestionInput]] = None,
    ) -> FeedbackResponse:
        """Submit validated user feedback for conversation evaluation."""
        import feedback_manager

        conversation_dicts = [
            {"role": c.role, "name": c.name, "content": c.content}
            for c in conversation
            if str(c.role or "").strip().lower() != "system"
        ]
        if not conversation_dicts:
            raise ValueError("Conversation must contain at least one non-system message.")
        feedback_dicts = [
            {
                "assistant": f.assistant,
                "rating": f.rating,
                "comments": getattr(f, "comments", None),
            }
            for f in feedback
        ]
        question_dicts = (
            [{"assistant": q.assistant, "question": q.question} for q in questions]
            if questions is not None
            else None
        )

        try:
            feedback_manager.log_conversation(conversation_dicts, feedback_dicts, question_dicts)
        except ValueError as exc:
            raise ValueError(_sanitize_error_message(exc)) from None
        except Exception as exc:
            logger.exception("submit_feedback failed")
            raise RuntimeError(f"Failed to record feedback: {_sanitize_error_message(exc)}") from None
        return FeedbackResponse(success="Feedback submitted successfully.")

    @strawberry.mutation
    def run_execution(self, input: RunExecutionInput) -> RunExecutionResponse:
        """Unified execution dispatcher supporting collaborative, advanced_pipeline, and gated tree_of_thought."""
        from execution_modes import (
            DisallowedModelError,
            RunRequest,
            execute_run,
            get_provider_config,
        )

        conv_payload = []
        if input.conversation:
            conv_payload = [
                {"role": c.role, "name": c.name, "content": c.content}
                for c in input.conversation
            ]

        personalities = input.assistant_personalities or ["Analytical Thinker", "Mediator"]
        lead = input.lead_personality or "Mediator"
        try:
            cfg = get_provider_config()
            selected_model = input.model if (input.model and str(input.model).strip()) else cfg.default_model
            req = RunRequest(
                mode=input.mode,  # type: ignore[arg-type]
                task_prompt=input.task_prompt,
                conversation=conv_payload,
                assistant_personalities=personalities,
                lead_personality=lead,
                num_rounds=input.num_rounds,
                max_steps=input.max_steps,
                initial_budget=input.initial_budget,
                agents=input.agents,
                model=selected_model,
                temperature=input.temperature,
                context=input.context,
            )
            result = execute_run(req)
            return _build_run_execution_response(result)
        except DisallowedModelError as exc:
            return _build_error_run_response(input.mode, exc, stop_reason="disallowed_model")
        except ValueError as exc:
            return _build_error_run_response(input.mode, exc)
        except Exception as exc:
            logger.exception("run_execution failed")
            return _build_error_run_response(input.mode, exc)

    @strawberry.mutation
    def run_advanced_pipeline(
        self,
        task_prompt: str,
        max_steps: int = 20,
        initial_budget: int = 20,
        agents: int = 3,
        model: Optional[str] = None,
    ) -> RunExecutionResponse:
        """Direct execution mutation for standalone advanced prompt pipeline."""
        from execution_modes import (
            DisallowedModelError,
            RunRequest,
            execute_run,
            get_provider_config,
        )

        try:
            cfg = get_provider_config()
            selected_model = model if (model and str(model).strip()) else cfg.default_model
            req = RunRequest(
                mode="advanced_pipeline",
                task_prompt=task_prompt,
                max_steps=max_steps,
                initial_budget=initial_budget,
                agents=agents,
                model=selected_model,
            )
            result = execute_run(req)
            return _build_run_execution_response(result)
        except DisallowedModelError as exc:
            return _build_error_run_response("advanced_pipeline", exc, stop_reason="disallowed_model")
        except ValueError as exc:
            return _build_error_run_response("advanced_pipeline", exc)
        except Exception as exc:
            logger.exception("run_advanced_pipeline failed")
            return _build_error_run_response("advanced_pipeline", exc)
