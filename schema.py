from __future__ import annotations

import logging
import json
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
class ExecutionModeCapabilityType:
    id: str
    label: str
    description: str
    enabled: bool
    gated_issue: Optional[str] = None
    supports_multi_agent: bool
    supports_step_planning: bool

@strawberry.input
class RunExecutionInput:
    mode: str = "collaborative"
    task_prompt: Optional[str] = None
    conversation: Optional[List[ConversationInput]] = None
    assistant_personalities: Optional[List[str]] = None
    lead_personality: Optional[str] = "Helpful Assistant"
    num_rounds: int = 3
    max_steps: int = 20
    initial_budget: int = 20
    agents: int = 3
    model: str = "gpt-4o-mini"
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

    events_gql = [
        RunEventTypeGQL(
            event_id=evt.event_id,
            event_type=evt.event_type,
            timestamp=evt.timestamp,
            mode=evt.mode,
            step_number=evt.step_number,
            plan_step_number=evt.plan_step_number,
            agent_name=evt.agent_name,
            summary=evt.summary,
        )
        for evt in run_result.events
    ]

    error_msg = run_result.error.error_message if run_result.error else None

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
        """Return registered execution modes and gating status."""
        from execution_modes import CAPABILITY_REGISTRY

        capabilities: List[ExecutionModeCapabilityType] = []
        for cap in CAPABILITY_REGISTRY.values():
            capabilities.append(
                ExecutionModeCapabilityType(
                    id=cap.id,
                    label=cap.label,
                    description=cap.description,
                    enabled=cap.enabled,
                    gated_issue=cap.gated_issue,
                    supports_multi_agent=cap.supports_multi_agent,
                    supports_step_planning=cap.supports_step_planning,
                )
            )
        return capabilities


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
        """Legacy conversation mutation preserved for full backward compatibility."""
        from conversation_manager import run_conversation

        conversation_dicts = [
            {"role": c.role, "name": c.name, "content": c.content}
            for c in conversation
        ]

        logger.info(
            json.dumps({
                "event": "run_conversation_started",
                "lead_personality": leadPersonality,
                "personalities": assistant_personalities,
                "num_rounds": num_rounds,
                "history_length": len(conversation_dicts),
            })
        )

        result = run_conversation(
            conversation_dicts,
            assistant_personalities,
            leadPersonality,
            num_rounds,
        )

        conversation_history, questions_asked, final_output = result

        conversation_objs = [
            Conversation(
                role=entry.get("role", "assistant"),
                name=entry.get("name") or "",
                content=entry.get("content", ""),
            )
            for entry in conversation_history
        ]

        question_objs = [
            Question(
                assistant=q.get("assistant", ""),
                question=q.get("question") or q.get("reply") or "",
            )
            for q in questions_asked
        ]

        return RunConversationResponse(
            conversation=conversation_objs,
            questions=question_objs,
            final_output=final_output,
        )

    @strawberry.mutation
    def submit_feedback(
        self,
        conversation: List[ConversationInput],
        feedback: List[FeedbackInput],
        questions: Optional[List[QuestionInput]] = None,
    ) -> FeedbackResponse:
        """Submit user feedback for conversation evaluation."""
        from feedback_manager import log_conversation

        conversation_dicts = [
            {"role": c.role, "name": c.name, "content": c.content}
            for c in conversation
        ]

        log_conversation(conversation_dicts, feedback, questions)
        return FeedbackResponse(success="Feedback submitted successfully.")

    @strawberry.mutation
    def run_execution(self, input: RunExecutionInput) -> RunExecutionResponse:
        """Unified execution dispatcher supporting collaborative, advanced_pipeline, and gated tree_of_thought."""
        from execution_modes import RunRequest, execute_run

        conv_payload = []
        if input.conversation:
            conv_payload = [
                {"role": c.role, "name": c.name, "content": c.content}
                for c in input.conversation
            ]

        personalities = input.assistant_personalities or ["Helpful Assistant"]
        lead = input.lead_personality or "Helpful Assistant"

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
            model=input.model,
            temperature=input.temperature,
            context=input.context,
        )

        result = execute_run(req)
        return _build_run_execution_response(result)

    @strawberry.mutation
    def run_advanced_pipeline(
        self,
        task_prompt: str,
        max_steps: int = 20,
        initial_budget: int = 20,
        agents: int = 3,
        model: str = "gpt-4o-mini",
    ) -> RunExecutionResponse:
        """Direct execution mutation for standalone advanced prompt pipeline."""
        from execution_modes import RunRequest, execute_run

        req = RunRequest(
            mode="advanced_pipeline",
            task_prompt=task_prompt,
            max_steps=max_steps,
            initial_budget=initial_budget,
            agents=agents,
            model=model,
        )

        result = execute_run(req)
        return _build_run_execution_response(result)
