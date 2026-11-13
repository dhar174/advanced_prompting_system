"""
Unit and integration test suite for execution modes, capability registry,
telemetry extraction, and fail-closed adversarial gating (Track 1 / Issue #35 Phase 3).
"""

import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

import pytest
from pydantic import ValidationError

from execution_modes import (
    CAPABILITY_REGISTRY,
    AgentAttribution,
    ExecutionModeCapability,
    PlanStepDiagnostic,
    ReflectionDiagnostic,
    RunDiagnostics,
    RunEvent,
    RunRequest,
    RunResult,
    SubtaskDiagnostic,
    execute_run,
)
from conversation_manager import OutputType, ErrorResult
from complexity_measures import Plan, PlanStep, Subtask
from advanced_prompting import Task, Step, Reflection, Interaction, PromptEngineeringConfig
from schema import ConversationInput
from output_generator import clear_tracked_artifacts, track_generated_artifact, is_tracked_artifact


# ============================================================================
# Shared Fixtures
# ============================================================================

@pytest.fixture
def mock_collaborative_tuple():
    """Mock return 3-tuple from conversation_manager.run_conversation."""
    conversation_history = [
        {"role": "system", "name": "system", "content": "You are a multi-agent team."},
        {"role": "user", "name": "User", "content": "Analyze database performance."},
        {
            "role": "assistant",
            "name": "Database Administrator",
            "content": "Identified slow queries in index scan.",
            "round": 1,
        },
        {
            "role": "assistant",
            "name": "Software Engineer",
            "content": "Added composite index on (tenant_id, created_at).",
            "round": 1,
        },
        {
            "role": "assistant",
            "name": "Mediator",
            "content": "Consensus reached: composite indexing resolved performance bottleneck.",
            "round": 2,
        },
    ]
    questions_asked = [
        {"assistant": "Database Administrator", "question": "What is the current p99 query latency?"}
    ]
    final_output = "Consensus output: Composite index added on (tenant_id, created_at)."
    return conversation_history, questions_asked, final_output


@pytest.fixture
def mock_pipeline_interaction():
    """Mock Interaction returned from AdvancedPromptEngineer.main."""
    subtask = Subtask(
        subtask_number=1,
        completed=True,
        subtask_name="Define architecture",
        subtask_description="Define components",
        subtask_explanation="High-level architecture definition",
        subtask_output="Architecture definition done",
        subtask_full_text="1. Define architecture",
        subtasks=[],
    )
    plan_step1 = PlanStep(
        step_number=1,
        completed=True,
        step_name="Step 1: System Design",
        step_description="Design the complete system architecture",
        description="Design the complete system architecture",
        step_explanation="High-level architecture",
        step_output="Architecture document",
        step_full_text="Step 1: System Design",
        subtasks=[subtask],
    )
    plan_step2 = PlanStep(
        step_number=2,
        completed=False,
        step_name="Step 2: Implementation",
        step_description="Implement backend services",
        description="Implement backend services",
        step_explanation="Implementation details",
        step_output="",
        step_full_text="Step 2: Implementation",
        subtasks=[],
    )
    plan = Plan(steps=[plan_step1, plan_step2])

    task_obj = Task(
        description="Build scalable API",
        refined_description="Build scalable API with microservices",
        complexity=2,
        steps=[],
        reflections=[],
        answer="Pipeline final answer: API built successfully",
        final_reward=0.92,
        plan=plan,
        output_type=OutputType(output_type="Python Script", file_extension=".py"),
    )

    refl1 = Reflection(
        content="Architecture design is solid and scalable.",
        reward=0.9,
        step_number=1,
    )
    refl2 = Reflection(
        content="Implementation plan meets requirements.",
        reward=0.94,
        step_number=2,
    )

    step1 = Step(
        description="Execute architecture design",
        step_number=1,
        remaining_budget=15,
        thoughts="Evaluating trade-offs between monolith and microservices...",
        plan_step_number=1,
        reflection=refl1,
    )
    step2 = Step(
        description="Execute implementation setup",
        step_number=2,
        remaining_budget=10,
        thoughts="Setting up FastAPI skeleton and database connections...",
        plan_step_number=2,
        reflection=refl2,
    )

    interaction = Interaction(
        task=task_obj,
        steps=[step1, step2],
        reflections=[refl1, refl2],
        answer="Pipeline final answer: API built successfully",
        final_reward=0.92,
    )
    return interaction


# ============================================================================
# 1. TestRunRequestValidation
# ============================================================================

class TestRunRequestValidation:
    """Verifies schema validation and bidirectional prompt/conversation derivation."""

    def test_default_initialization(self):
        """Verify default values across all RunRequest fields."""
        req = RunRequest()
        assert req.mode == "collaborative"
        assert req.task_prompt is None
        assert req.conversation == []
        assert req.assistant_personalities == ["Helpful Assistant"]
        assert req.lead_personality == "Helpful Assistant"
        assert req.num_rounds == 3
        assert req.max_steps == 20
        assert req.initial_budget == 20
        assert req.agents == 3
        assert req.model == "gpt-4o-mini"
        assert req.temperature == 0.7
        assert req.context is None
        assert req.correlation_id is None

    def test_task_prompt_derived_from_single_user_message(self):
        """When conversation has a user message, task_prompt is derived automatically."""
        req = RunRequest(
            conversation=[{"role": "user", "name": "User", "content": "Analyze quarterly budget"}]
        )
        assert req.task_prompt == "Analyze quarterly budget"
        assert len(req.conversation) == 1

    def test_task_prompt_derived_from_latest_user_message_in_history(self):
        """In multi-turn conversations, task_prompt is derived from the latest user message."""
        req = RunRequest(
            conversation=[
                {"role": "user", "name": "User", "content": "First prompt"},
                {"role": "assistant", "name": "Strategist", "content": "Clarifying question?"},
                {"role": "user", "name": "User", "content": "Revised follow-up prompt"},
            ]
        )
        assert req.task_prompt == "Revised follow-up prompt"

    def test_conversation_populated_from_task_prompt(self):
        """When only task_prompt is provided, conversation is populated with a user message."""
        req = RunRequest(task_prompt="Implement OAuth2 authentication")
        assert len(req.conversation) == 1
        assert req.conversation[0]["role"] == "user"
        assert req.conversation[0]["name"] == "User"
        assert req.conversation[0]["content"] == "Implement OAuth2 authentication"

    def test_explicit_task_prompt_and_conversation_both_retained(self):
        """When both task_prompt and conversation are provided, neither is overwritten."""
        req = RunRequest(
            task_prompt="Custom prompt override",
            conversation=[{"role": "user", "name": "Alice", "content": "Original message"}],
        )
        assert req.task_prompt == "Custom prompt override"
        assert req.conversation[0]["content"] == "Original message"

    def test_invalid_execution_mode_raises_validation_error(self):
        """Invalid execution mode strings are rejected with ValidationError."""
        with pytest.raises(ValidationError):
            RunRequest(mode="invalid_quantum_mode")  # type: ignore[arg-type]

    def test_numeric_bounds_validation(self):
        """Zero or negative step/round counts are rejected by validator constraints."""
        with pytest.raises(ValidationError):
            RunRequest(num_rounds=0)
        with pytest.raises(ValidationError):
            RunRequest(max_steps=0)
        with pytest.raises(ValidationError):
            RunRequest(agents=-1)

    def test_conversation_strawberry_and_object_items(self):
        """Objects with .role and .content (e.g., Strawberry ConversationInput, SimpleNamespace) are normalized into dicts."""
        user_item = ConversationInput(role="user", name="Alice", content="Input from Strawberry")
        asst_item = ConversationInput(role="assistant", name=None, content="Assistant response")
        custom_obj = SimpleNamespace(role="user", name="", content="Namespace content")

        req = RunRequest(conversation=[user_item, asst_item, custom_obj])
        assert len(req.conversation) == 3

        # Explicit name and role preserved
        assert req.conversation[0] == {
            "role": "user",
            "name": "Alice",
            "content": "Input from Strawberry",
        }
        # Assistant role with None name defaults name to empty string
        assert req.conversation[1] == {
            "role": "assistant",
            "name": "",
            "content": "Assistant response",
        }
        # User role with empty name defaults name to "User"
        assert req.conversation[2] == {
            "role": "user",
            "name": "User",
            "content": "Namespace content",
        }
        # task_prompt is derived from the latest user message
        assert req.task_prompt == "Namespace content"

    def test_conversation_raw_string_and_primitive_items(self):
        """Raw strings and primitives in conversation default to role='user' and name='User'."""
        req = RunRequest(conversation=["Plain text user message", 42])
        assert len(req.conversation) == 2
        assert req.conversation[0] == {
            "role": "user",
            "name": "User",
            "content": "Plain text user message",
        }
        assert req.conversation[1] == {
            "role": "user",
            "name": "User",
            "content": "42",
        }
        assert req.task_prompt == "42"

    def test_conversation_empty_and_partial_dict_defaults(self):
        """Empty dict {} and partial dicts in conversation default role='user' and name='User'."""
        req_empty = RunRequest(conversation=[{}])
        assert len(req_empty.conversation) == 1
        assert req_empty.conversation[0] == {
            "role": "user",
            "name": "User",
            "content": "",
        }

        # Partial dictionaries missing role, name, or content
        req_partial = RunRequest(
            conversation=[
                {"role": "user"},
                {"role": "assistant"},
                {"content": "Just content"},
                {"name": "Admin", "content": "Notice"},
            ]
        )
        assert len(req_partial.conversation) == 4
        # User without name -> name="User", empty content
        assert req_partial.conversation[0] == {"role": "user", "name": "User", "content": ""}
        # Assistant without name -> name="", empty content
        assert req_partial.conversation[1] == {"role": "assistant", "name": "", "content": ""}
        # Content only -> role defaults to "user", name defaults to "User"
        assert req_partial.conversation[2] == {"role": "user", "name": "User", "content": "Just content"}
        # Name and content -> role defaults to "user", custom name preserved
        assert req_partial.conversation[3] == {"role": "user", "name": "Admin", "content": "Notice"}

    def test_whitespace_only_task_prompt_stripped_empty(self):
        """Whitespace-only task_prompt is stripped to empty string and does not synthesize conversation."""
        req = RunRequest(task_prompt="   \t\n  ", conversation=[])
        assert req.task_prompt == ""
        assert req.conversation == []

    def test_whitespace_only_task_prompt_derives_from_conversation(self):
        """When task_prompt is whitespace-only, it falls back to extracting user message from conversation."""
        req = RunRequest(
            task_prompt="   \t  \n",
            conversation=[{"role": "user", "name": "User", "content": "Extracted query"}],
        )
        assert req.task_prompt == "Extracted query"
        assert len(req.conversation) == 1
        assert req.conversation[0]["content"] == "Extracted query"

    def test_task_prompt_whitespace_padding_stripped(self):
        """Leading and trailing whitespace on task_prompt is stripped upon initialization."""
        req = RunRequest(task_prompt="   Deploy Kubernetes cluster   ")
        assert req.task_prompt == "Deploy Kubernetes cluster"
        assert len(req.conversation) == 1
        assert req.conversation[0]["content"] == "Deploy Kubernetes cluster"


# ============================================================================
# 2. TestDiagnosticsAndEventSchemas
# ============================================================================

class TestDiagnosticsAndEventSchemas:
    """Verifies schema definitions, field types, and serialization of telemetry models."""

    def test_subtask_diagnostic_serialization(self):
        """SubtaskDiagnostic fields match and serialize cleanly."""
        sub = SubtaskDiagnostic(
            subtask_number=1,
            subtask_name="Init Database",
            subtask_description="Run migrations",
            subtask_output="Migration applied",
            completed=True,
        )
        assert sub.subtask_number == 1
        assert sub.subtask_name == "Init Database"
        assert sub.completed is True
        dump = sub.model_dump()
        assert dump["subtask_number"] == 1
        assert dump["subtask_output"] == "Migration applied"

    def test_plan_step_diagnostic_with_nested_subtasks(self):
        """PlanStepDiagnostic preserves nested SubtaskDiagnostic items and assigned agents."""
        sub = SubtaskDiagnostic(
            subtask_number=1,
            subtask_name="Subtask A",
            subtask_description="Subtask A Desc",
            completed=True,
        )
        step = PlanStepDiagnostic(
            step_number=1,
            step_name="Step One",
            step_description="First major step",
            step_explanation="Detailed explanation",
            step_output="Output text",
            completed=True,
            assigned_agents=["Strategist", "Coding Guru"],
            subtasks=[sub],
        )
        assert step.step_number == 1
        assert len(step.subtasks) == 1
        assert step.subtasks[0].subtask_name == "Subtask A"
        assert step.assigned_agents == ["Strategist", "Coding Guru"]
        dump = step.model_dump()
        assert len(dump["subtasks"]) == 1
        assert dump["subtasks"][0]["subtask_number"] == 1

    def test_reflection_diagnostic_fields_and_defaults(self):
        """ReflectionDiagnostic fields and optional defaults behave predictably."""
        refl = ReflectionDiagnostic(
            step_number=2,
            plan_step_number=1,
            step_description="Review code",
            thoughts="Code is clean and modular",
            content="Passed quality review",
            reward=0.88,
            remaining_budget=12,
        )
        assert refl.step_number == 2
        assert refl.plan_step_number == 1
        assert refl.thoughts == "Code is clean and modular"
        assert refl.reward == 0.88
        assert refl.remaining_budget == 12

        # Check default optional fields
        refl_def = ReflectionDiagnostic(step_number=1, content="Step OK", reward=1.0)
        assert refl_def.thoughts is None
        assert refl_def.plan_step_number is None
        assert refl_def.remaining_budget == 0

    def test_agent_attribution_fields(self):
        """AgentAttribution models store attribution metadata with defaults."""
        attr = AgentAttribution(
            agent_name="Skeptical Critic",
            role="critic",
            round_or_step=2,
            contribution_summary="Critiqued scalability assumptions",
        )
        assert attr.agent_name == "Skeptical Critic"
        assert attr.role == "critic"
        assert attr.round_or_step == 2

        # Default role and round
        attr_def = AgentAttribution(
            agent_name="General Assistant", contribution_summary="Answered user question"
        )
        assert attr_def.role == "assistant"
        assert attr_def.round_or_step == 1

    def test_run_event_iso8601_timestamp(self):
        """RunEvent timestamp conforms to ISO-8601 format and details defaults to empty dict."""
        now_iso = datetime.now(timezone.utc).isoformat()
        evt = RunEvent(
            event_id="evt_001",
            event_type="mode_selected",
            timestamp=now_iso,
            mode="collaborative",
            summary="Mode selected successfully",
        )
        assert evt.event_id == "evt_001"
        assert evt.event_type == "mode_selected"
        assert evt.details == {}
        # Parse ISO-8601 string to ensure standard format
        parsed_dt = datetime.fromisoformat(evt.timestamp)
        assert parsed_dt is not None

    def test_run_diagnostics_telemetry_fields(self):
        """RunDiagnostics stores complexity, budget, duration, tokens, and correlation ID."""
        diag = RunDiagnostics(
            complexity_score=2.5,
            initial_budget=20,
            adjusted_budget=32,
            final_reward=0.95,
            duration_ms=145.2,
            token_count=350,
            correlation_id="corr-test-123",
        )
        assert diag.complexity_score == 2.5
        assert diag.initial_budget == 20
        assert diag.adjusted_budget == 32
        assert diag.final_reward == 0.95
        assert diag.duration_ms == 145.2
        assert diag.token_count == 350
        assert diag.correlation_id == "corr-test-123"
        assert diag.plan_steps == []
        assert diag.reflections == []
        assert diag.agent_attributions == []


# ============================================================================
# 3. TestCapabilityRegistry
# ============================================================================

class TestCapabilityRegistry:
    """Verifies CAPABILITY_REGISTRY mode definitions, flags, and gating metadata."""

    def test_registry_contains_three_modes(self):
        """Registry contains exactly the three defined modes."""
        assert set(CAPABILITY_REGISTRY.keys()) == {
            "collaborative",
            "advanced_pipeline",
            "tree_of_thought",
        }

    def test_collaborative_capability_metadata(self):
        """Collaborative mode metadata conforms to Track 1 Phase 3 specification."""
        cap = CAPABILITY_REGISTRY["collaborative"]
        assert cap.id == "collaborative"
        assert cap.label == "Collaborative Debate"
        assert cap.enabled is True
        assert cap.gated_issue is None
        assert cap.supports_multi_agent is True
        assert cap.supports_step_planning is False

    def test_advanced_pipeline_capability_metadata(self):
        """Advanced pipeline mode metadata conforms to Track 1 Phase 3 specification."""
        cap = CAPABILITY_REGISTRY["advanced_pipeline"]
        assert cap.id == "advanced_pipeline"
        assert cap.label == "Standalone Pipeline"
        assert cap.enabled is True
        assert cap.gated_issue is None
        assert cap.supports_multi_agent is True
        assert cap.supports_step_planning is True

    def test_tree_of_thought_capability_metadata(self):
        """Tree of Thoughts mode is explicitly gated for Issue #36."""
        cap = CAPABILITY_REGISTRY["tree_of_thought"]
        assert cap.id == "tree_of_thought"
        assert cap.label == "Tree of Thoughts"
        assert cap.enabled is False
        assert cap.gated_issue == "Issue #36"
        assert cap.supports_multi_agent is False
        assert cap.supports_step_planning is True


# ============================================================================
# 4. TestExecuteRunGatedMode (Adversarial Zero Provider Calls)
# ============================================================================

class TestExecuteRunGatedMode:
    """Verifies fail-closed gating for tree_of_thought mode with adversarial assertions."""

    def test_tree_of_thought_gated_status_and_error(self):
        """Invoking tree_of_thought returns status='gated' with ModeGatedError."""
        req = RunRequest(mode="tree_of_thought", task_prompt="Explore reasoning tree")
        result = execute_run(req)

        assert result.mode == "tree_of_thought"
        assert result.status == "gated"
        assert "Issue #36" in result.final_output
        assert result.error is not None
        assert result.error.error_type == "ModeGatedError"
        assert "Issue #36" in result.error.error_message

    def test_tree_of_thought_emits_mode_gated_event(self):
        """Invoking tree_of_thought records a 'mode_gated' event with issue metadata."""
        req = RunRequest(mode="tree_of_thought", task_prompt="Test task")
        result = execute_run(req)

        assert len(result.events) >= 1
        gated_events = [e for e in result.events if e.event_type == "mode_gated"]
        assert len(gated_events) == 1
        assert gated_events[0].details.get("gated_issue") == "Issue #36"

    def test_tree_of_thought_adversarial_zero_provider_calls(self, monkeypatch):
        """Adversarial check: tree_of_thought NEVER calls provider_client or LLM."""
        import provider_client

        spy_client = Mock(side_effect=AssertionError("Provider client should not be called in gated mode"))
        monkeypatch.setattr(provider_client, "get_client", spy_client)

        req = RunRequest(mode="tree_of_thought", task_prompt="Zero provider test")
        result = execute_run(req)

        assert result.status == "gated"
        spy_client.assert_not_called()


# ============================================================================
# 5. TestExecuteRunCollaborative
# ============================================================================

class TestExecuteRunCollaborative:
    """Verifies collaborative mode execution, attribution mapping, and event telemetry."""

    def test_collaborative_execution_success(self, mock_collaborative_tuple, monkeypatch):
        """Collaborative execution unpacks run_conversation 3-tuple successfully."""
        import conversation_manager

        monkeypatch.setattr(
            conversation_manager, "run_conversation", lambda *a, **kw: mock_collaborative_tuple
        )

        req = RunRequest(
            mode="collaborative",
            task_prompt="Analyze database performance",
            assistant_personalities=["Database Administrator", "Software Engineer"],
            lead_personality="Mediator",
            num_rounds=2,
        )
        result = execute_run(req)

        assert result.status == "completed"
        assert result.mode == "collaborative"
        assert result.final_output == mock_collaborative_tuple[2]
        assert len(result.conversation) == len(mock_collaborative_tuple[0])
        assert result.output_type.output_type == "Simple Concise Answer"

    def test_collaborative_agent_attributions_extracted(self, mock_collaborative_tuple, monkeypatch):
        """Assistant contributions are extracted into agent_attributions; system messages excluded."""
        import conversation_manager

        monkeypatch.setattr(
            conversation_manager, "run_conversation", lambda *a, **kw: mock_collaborative_tuple
        )

        req = RunRequest(mode="collaborative", task_prompt="Analyze database performance")
        result = execute_run(req)

        attributions = result.diagnostics.agent_attributions
        assert len(attributions) == 3
        # Ensure system and user messages were excluded
        assert not any(attr.agent_name == "system" for attr in attributions)
        names = [attr.agent_name for attr in attributions]
        assert names == ["Database Administrator", "Software Engineer", "Mediator"]
        assert attributions[0].round_or_step == 1
        assert attributions[2].round_or_step == 2

    def test_collaborative_questions_extracted(self, mock_collaborative_tuple, monkeypatch):
        """Questions asked during debate are extracted into result.questions."""
        import conversation_manager

        monkeypatch.setattr(
            conversation_manager, "run_conversation", lambda *a, **kw: mock_collaborative_tuple
        )

        req = RunRequest(mode="collaborative", task_prompt="Analyze database performance")
        result = execute_run(req)

        assert len(result.questions) == 1
        assert result.questions[0]["assistant"] == "Database Administrator"
        assert "p99" in result.questions[0]["question"]

    def test_collaborative_event_sequence(self, mock_collaborative_tuple, monkeypatch):
        """Ordered events are recorded: mode_selected, agent contributions, and output_finalized."""
        import conversation_manager

        monkeypatch.setattr(
            conversation_manager, "run_conversation", lambda *a, **kw: mock_collaborative_tuple
        )

        req = RunRequest(mode="collaborative", task_prompt="Analyze database performance")
        result = execute_run(req)

        event_types = [e.event_type for e in result.events]
        assert "mode_selected" in event_types
        assert "agent_contribution" in event_types
        assert "output_finalized" in event_types
        assert event_types[0] == "mode_selected"
        assert event_types[-1] == "output_finalized"

    def test_collaborative_output_type_sniffing(self, monkeypatch):
        """Output type sniffing detects structured JSON content in final output."""
        import conversation_manager

        json_history = [{"role": "assistant", "name": "API Guru", "content": '{"status": "ok"}'}]
        json_output = '{"service": "auth", "port": 8080, "status": "healthy"}'
        monkeypatch.setattr(
            conversation_manager,
            "run_conversation",
            lambda *a, **kw: (json_history, [], json_output),
        )

        req = RunRequest(mode="collaborative", task_prompt="Generate auth service config")
        result = execute_run(req)

        assert result.status == "completed"
        assert result.output_type.output_type == "JSON" or result.output_type.file_extension == ".json"

    def test_collaborative_correlation_id_and_duration_telemetry(
        self, mock_collaborative_tuple, monkeypatch
    ):
        """Correlation ID and execution duration are recorded in diagnostics telemetry."""
        import conversation_manager

        monkeypatch.setattr(
            conversation_manager, "run_conversation", lambda *a, **kw: mock_collaborative_tuple
        )

        req = RunRequest(
            mode="collaborative",
            task_prompt="Measure latency",
            correlation_id="corr-telemetry-456",
        )
        result = execute_run(req)

        assert result.diagnostics.correlation_id == "corr-telemetry-456"
        assert result.diagnostics.duration_ms is not None
        assert result.diagnostics.duration_ms >= 0.0


# ============================================================================
# 6. TestExecuteRunAdvancedPipeline
# ============================================================================

class TestExecuteRunAdvancedPipeline:
    """Verifies advanced pipeline execution, reasoning diagnostics, and step telemetry."""

    def test_advanced_pipeline_execution_success(self, mock_pipeline_interaction, monkeypatch):
        """Advanced pipeline execution wraps AdvancedPromptEngineer.main successfully."""
        from advanced_prompting import AdvancedPromptEngineer

        monkeypatch.setattr(
            AdvancedPromptEngineer, "main", lambda self, prompt: mock_pipeline_interaction
        )

        req = RunRequest(
            mode="advanced_pipeline",
            task_prompt="Build scalable API",
            max_steps=15,
            initial_budget=20,
            agents=2,
        )
        result = execute_run(req)

        assert result.status == "completed"
        assert result.mode == "advanced_pipeline"
        assert result.final_output == mock_pipeline_interaction.answer
        assert result.output_type.output_type == "Python Script"
        assert result.output_type.file_extension == ".py"

    def test_advanced_pipeline_diagnostics_extraction(
        self, mock_pipeline_interaction, monkeypatch
    ):
        """Pipeline complexity, plan steps, subtasks, reflections, and rewards are mapped."""
        from advanced_prompting import AdvancedPromptEngineer

        monkeypatch.setattr(
            AdvancedPromptEngineer, "main", lambda self, prompt: mock_pipeline_interaction
        )

        req = RunRequest(mode="advanced_pipeline", task_prompt="Build scalable API")
        result = execute_run(req)

        diag = result.diagnostics
        assert diag.complexity_score == 2.0
        assert diag.final_reward == 0.92
        assert len(diag.plan_steps) == 2
        assert diag.plan_steps[0].step_name == "Step 1: System Design"
        assert len(diag.plan_steps[0].subtasks) == 1
        assert diag.plan_steps[0].subtasks[0].subtask_name == "Define architecture"

        assert len(diag.reflections) == 2
        assert diag.reflections[0].reward == 0.9
        assert diag.reflections[0].remaining_budget == 15
        assert diag.reflections[1].reward == 0.94

    def test_advanced_pipeline_agents_zero(self, mock_pipeline_interaction, monkeypatch):
        """When agents=0, assigned_agents list is empty and collaborative reasoning is skipped."""
        from advanced_prompting import AdvancedPromptEngineer

        monkeypatch.setattr(
            AdvancedPromptEngineer, "main", lambda self, prompt: mock_pipeline_interaction
        )

        req = RunRequest(mode="advanced_pipeline", task_prompt="Solo prompt", agents=0)
        result = execute_run(req)

        for step in result.diagnostics.plan_steps:
            assert step.assigned_agents == []

    def test_advanced_pipeline_conversation_synthesis(
        self, mock_pipeline_interaction, monkeypatch
    ):
        """Synthesized chat transcript includes user prompt, planner progress, and final answer."""
        from advanced_prompting import AdvancedPromptEngineer

        monkeypatch.setattr(
            AdvancedPromptEngineer, "main", lambda self, prompt: mock_pipeline_interaction
        )

        req = RunRequest(mode="advanced_pipeline", task_prompt="Synthesize transcript test")
        result = execute_run(req)

        conv = result.conversation
        assert len(conv) >= 2
        assert conv[0]["role"] == "user"
        assert conv[0]["content"] == "Synthesize transcript test"
        assert any(
            msg["role"] == "assistant" and "Pipeline final answer" in msg["content"]
            for msg in conv
        )

    def test_advanced_pipeline_event_sequence(self, mock_pipeline_interaction, monkeypatch):
        """Ordered events are recorded across the pipeline reasoning lifecycle."""
        from advanced_prompting import AdvancedPromptEngineer

        monkeypatch.setattr(
            AdvancedPromptEngineer, "main", lambda self, prompt: mock_pipeline_interaction
        )

        req = RunRequest(mode="advanced_pipeline", task_prompt="Build scalable API")
        result = execute_run(req)

        event_types = [e.event_type for e in result.events]
        assert "mode_selected" in event_types
        assert "complexity_assessed" in event_types
        assert "plan_generated" in event_types
        assert "output_finalized" in event_types
        assert event_types[0] == "mode_selected"
        assert event_types[-1] == "output_finalized"


# ============================================================================
# 7. TestAdversarialAndErrorHandling
# ============================================================================

class TestAdversarialAndErrorHandling:
    """Verifies edge cases, unhandled exception wrapping, and clean-process import purity."""

    def test_pipeline_empty_plan_handling(self, mock_pipeline_interaction, monkeypatch):
        """Pipeline handles empty plan steps gracefully without raising IndexError."""
        from advanced_prompting import AdvancedPromptEngineer

        # Strip all plan steps from task
        mock_pipeline_interaction.task.plan.steps = []
        monkeypatch.setattr(
            AdvancedPromptEngineer, "main", lambda self, prompt: mock_pipeline_interaction
        )

        req = RunRequest(mode="advanced_pipeline", task_prompt="Task with empty plan")
        result = execute_run(req)

        assert result.status == "completed"
        assert result.diagnostics.plan_steps == []
        assert result.final_output == mock_pipeline_interaction.answer

    def test_collaborative_exception_wrapped(self, monkeypatch):
        """Uncaught exceptions during collaborative debate are safely wrapped into ErrorResult."""
        import conversation_manager

        monkeypatch.setattr(
            conversation_manager,
            "run_conversation",
            Mock(side_effect=RuntimeError("Simulated LLM rate limit failure")),
        )

        req = RunRequest(mode="collaborative", task_prompt="Failing task")
        result = execute_run(req)

        assert result.status == "error"
        assert result.error is not None
        assert result.error.error_type == "RuntimeError"
        assert "Simulated LLM rate limit failure" in result.error.error_message
        assert any(e.event_type == "run_error" for e in result.events)

    def test_advanced_pipeline_exception_wrapped(self, monkeypatch):
        """Uncaught exceptions during pipeline reasoning are safely wrapped into ErrorResult."""
        from advanced_prompting import AdvancedPromptEngineer

        monkeypatch.setattr(
            AdvancedPromptEngineer,
            "main",
            Mock(side_effect=ValueError("Invalid task prompt syntax")),
        )

        req = RunRequest(mode="advanced_pipeline", task_prompt="Failing task")
        result = execute_run(req)

        assert result.status == "error"
        assert result.error is not None
        assert result.error.error_type == "ValueError"
        assert "Invalid task prompt syntax" in result.error.error_message
        assert any(e.event_type == "run_error" for e in result.events)

    def test_fresh_process_import_purity(self):
        """In a clean subprocess, importing execution_modes makes zero network calls and constructs no clients."""
        code = r'''
import os, sys, socket
assert sys._aps_network_guard
assert not os.getenv('OPENAI_API_KEY')
import openai, nltk, spacy
def forbidden(*a, **kw):
    raise AssertionError('Import tried to initialize a provider or download/load a model')
openai.OpenAI = forbidden
nltk.download = forbidden
spacy.load = forbidden

# Import execution_modes
import execution_modes
assert hasattr(execution_modes, 'CAPABILITY_REGISTRY')
assert hasattr(execution_modes, 'execute_run')
assert hasattr(execution_modes, 'RunRequest')
assert hasattr(execution_modes, 'RunResult')
print('execution_modes_import_purity_verified')
'''
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=os.path.dirname(__file__),
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        assert "execution_modes_import_purity_verified" in result.stdout

    def test_collaborative_artifact_reads_constrained_to_tracked_artifacts(self, monkeypatch, tmp_path):
        """
        Security verification (PR #38 review finding 1):
        Arbitrary file reads are strictly constrained. If the model outputs an existing server
        file path (e.g. pyproject.toml or a path outside the workspace) without it having been
        explicitly generated and tracked by output_generator, the server NEVER reads that file.
        """
        import conversation_manager
        clear_tracked_artifacts()

        # Existing file on disk that was NOT produced by output_generator
        sensitive_file = tmp_path / "sensitive_config.json"
        sensitive_file.write_text('{"api_secret": "super_secret_token"}', encoding="utf-8")

        # Mock run_conversation returning the path to this existing file
        monkeypatch.setattr(
            conversation_manager,
            "run_conversation",
            Mock(return_value=([{"role": "assistant", "content": "Done"}], [], str(sensitive_file))),
        )

        req = RunRequest(mode="collaborative", task_prompt="What is in the config?")
        res = execute_run(req)

        assert res.status == "completed"
        # Since sensitive_file was NOT explicitly tracked by output_generator,
        # its content must NOT be read into final_output! The raw string remains intact.
        assert res.final_output == str(sensitive_file)
        assert "super_secret_token" not in res.final_output

        # Now test a genuinely tracked artifact created via track_generated_artifact
        artifact_file = tmp_path / "final_output.json"
        artifact_file.write_text('{"summary": "safe generated analysis"}', encoding="utf-8")
        track_generated_artifact(str(artifact_file))

        monkeypatch.setattr(
            conversation_manager,
            "run_conversation",
            Mock(return_value=([{"role": "assistant", "content": "Done"}], [], str(artifact_file))),
        )

        res2 = execute_run(req)
        assert res2.status == "completed"
        # Tracked artifact content IS safely read
        assert '{"summary": "safe generated analysis"}' in res2.final_output
        clear_tracked_artifacts()

    def test_pipeline_step_limit_enforcement(self, monkeypatch, mock_pipeline_interaction):
        """
        Enforce advertised step limits (PR #38 review finding 2):
        When callers specify lower max_steps or initial_budget, the execution loop
        and reported diagnostics strictly cap the adjusted budget to max_steps.
        """
        from advanced_prompting import AdvancedPromptEngineer, PromptEngineeringConfig

        created_configs = []
        original_init = PromptEngineeringConfig.__init__

        def capturing_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            created_configs.append(self)

        monkeypatch.setattr(PromptEngineeringConfig, "__init__", capturing_init)

        # Mock AdvancedPromptEngineer.main
        def fake_main(task_str):
            engineer = capturing_init.__self__ if hasattr(capturing_init, "__self__") else None
            return mock_pipeline_interaction

        monkeypatch.setattr(AdvancedPromptEngineer, "main", Mock(return_value=mock_pipeline_interaction))

        req = RunRequest(
            mode="advanced_pipeline",
            task_prompt="Constrained step task",
            max_steps=2,
            initial_budget=2,
            agents=1,
        )
        res = execute_run(req)

        assert res.status == "completed"
        assert res.diagnostics is not None
        # Diagnostics step budget must be capped at max_steps (2)
        assert res.diagnostics.adjusted_budget <= 2
        # Mode selected and complexity assessed events reflect the cap
        comp_event = next(e for e in res.events if e.event_type == "complexity_assessed")
        assert comp_event.details["max_steps"] == 2
        assert comp_event.details["adjusted_budget"] <= 2

    def test_pipeline_zero_agents_solo_run(self, monkeypatch, mock_pipeline_interaction):
        """
        Support zero-agent solo reasoning (PR #38 review finding 3):
        When agents=0 is configured, AdvancedPromptEngineer.main and execution_modes
        execute the solo run cleanly without raising UnboundLocalError.
        """
        from advanced_prompting import AdvancedPromptEngineer

        monkeypatch.setattr(AdvancedPromptEngineer, "main", Mock(return_value=mock_pipeline_interaction))

        req = RunRequest(
            mode="advanced_pipeline",
            task_prompt="Solo reasoning task",
            agents=0,
            max_steps=5,
        )
        res = execute_run(req)

        assert res.status == "completed"
        assert res.error is None
        assert res.diagnostics is not None
        # Must have solo attribution recorded
        assert any(
            attr.agent_name == "Advanced Prompt Engineer" and attr.role == "Solo Reasoning Engine"
            for attr in res.diagnostics.agent_attributions
        )
