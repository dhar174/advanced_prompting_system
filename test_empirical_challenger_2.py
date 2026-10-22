"""
test_empirical_challenger_2.py

Empirical stress tests for execution_modes.py:
1. Error handling & invalid tuples in run_conversation
2. Negative rewards & None attributes in AdvancedPromptEngineer.main
3. Strict ISO-8601 UTC timestamps & JSON-serializability of RunEvent detail payloads
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
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


# ============================================================================
# Challenge Suite 1: run_conversation Returning ErrorResult or Invalid Tuple
# ============================================================================

class TestCollaborativeErrorAndTupleFidelity:
    """Empirically challenge collaborative mode handling of ErrorResult and malformed returns."""

    def test_run_conversation_returns_error_result_in_3tuple(self, monkeypatch):
        """When run_conversation returns (history, questions, ErrorResult), execute_run returns status='error'."""
        import conversation_manager

        history = [
            {"role": "user", "name": "User", "content": "Analyze system load."},
            {"role": "assistant", "name": "System", "content": "Partial progress."},
        ]
        questions = []
        error_result = ErrorResult(
            error_type="ConsensusError",
            error_message="Tie vote could not be resolved by mediator.",
            error_details="Detailed consensus deadlock traceback.",
        )

        monkeypatch.setattr(
            conversation_manager,
            "run_conversation",
            Mock(return_value=(history, questions, error_result)),
        )

        req = RunRequest(mode="collaborative", task_prompt="Analyze system load.")
        res = execute_run(req)

        assert res.mode == "collaborative"
        assert res.status == "error"
        assert res.error is not None
        assert res.error.error_type == "ConsensusError"
        assert "Tie vote could not be resolved" in res.error.error_message
        assert res.final_output == "Tie vote could not be resolved by mediator."
        assert any(e.event_type == "run_error" for e in res.events)
        assert res.diagnostics.correlation_id is not None
        assert res.diagnostics.duration_ms is not None

    def test_run_conversation_returns_naked_error_result(self, monkeypatch):
        """When run_conversation returns an ErrorResult directly (not 3-tuple), execute_run wraps it fail-closed."""
        import conversation_manager

        error_result = ErrorResult(
            error_type="ExecutionFatalError",
            error_message="Provider client failed fatally.",
        )

        monkeypatch.setattr(
            conversation_manager,
            "run_conversation",
            Mock(return_value=error_result),
        )

        req = RunRequest(mode="collaborative", task_prompt="Analyze system load.")
        res = execute_run(req)

        assert res.status == "error"
        assert res.error is not None
        assert res.error.error_type == "ValueError"
        assert "unpack" in res.error.error_message.lower()
        assert any(e.event_type == "run_error" for e in res.events)

    def test_run_conversation_returns_2tuple(self, monkeypatch):
        """When run_conversation returns a 2-tuple, execute_run wraps ValueError into status='error'."""
        import conversation_manager

        monkeypatch.setattr(
            conversation_manager,
            "run_conversation",
            Mock(return_value=([], [])),
        )

        req = RunRequest(mode="collaborative", task_prompt="Analyze system load.")
        res = execute_run(req)

        assert res.status == "error"
        assert res.error is not None
        assert res.error.error_type == "ValueError"
        assert any(e.event_type == "run_error" for e in res.events)

    def test_run_conversation_returns_4tuple(self, monkeypatch):
        """When run_conversation returns a 4-tuple, execute_run wraps ValueError into status='error'."""
        import conversation_manager

        monkeypatch.setattr(
            conversation_manager,
            "run_conversation",
            Mock(return_value=([], [], "output", "extra_element")),
        )

        req = RunRequest(mode="collaborative", task_prompt="Analyze system load.")
        res = execute_run(req)

        assert res.status == "error"
        assert res.error is not None
        assert res.error.error_type == "ValueError"
        assert any(e.event_type == "run_error" for e in res.events)

    def test_run_conversation_returns_none_tuple(self, monkeypatch):
        """When run_conversation returns (None, None, None), execute_run completes with safe empty outputs."""
        import conversation_manager

        monkeypatch.setattr(
            conversation_manager,
            "run_conversation",
            Mock(return_value=(None, None, None)),
        )

        req = RunRequest(mode="collaborative", task_prompt="Analyze system load.")
        res = execute_run(req)

        assert res.status == "completed"
        assert res.final_output == ""
        assert res.conversation == []
        assert res.questions == []
        assert res.output_type.output_type == "Simple Concise Answer"

    def test_run_conversation_history_with_error_result_objects(self, monkeypatch):
        """When conversation history contains an ErrorResult item, history normalizer does not crash."""
        import conversation_manager

        history = [
            {"role": "user", "name": "User", "content": "Query"},
            ErrorResult(error_type="TransientError", error_message="Failed attempt"),
            {"role": "assistant", "name": "Analyst", "content": "Recovered response", "round": 1},
        ]
        monkeypatch.setattr(
            conversation_manager,
            "run_conversation",
            Mock(return_value=(history, [], "Final success")),
        )

        req = RunRequest(mode="collaborative", task_prompt="Query")
        res = execute_run(req)

        assert res.status == "completed"
        # Verify valid messages are retained and normalizer did not crash on ErrorResult
        assert any(m["role"] == "user" for m in res.conversation)
        assert any(m["name"] == "Analyst" for m in res.conversation)


# ============================================================================
# Challenge Suite 2: AdvancedPromptEngineer Steps with Negative Rewards or None
# ============================================================================

class TestPipelineNegativeRewardsAndNoneAttributes:
    """Empirically challenge advanced pipeline mode handling of negative rewards and None attributes."""

    def test_pipeline_negative_reward_and_final_reward(self, monkeypatch):
        """Steps and final rewards with negative values (-1.0 to 0.0) are accurately preserved."""
        from advanced_prompting import AdvancedPromptEngineer

        step1 = Step(
            step_number=1,
            description="Initial flawed hypothesis",
            thoughts="Attempting direct calculation without validation",
            remaining_budget=15,
        )
        refl1 = Reflection(
            step_number=1,
            content="Hypothesis failed constraint check; negative reward assigned.",
            reward=-0.85,
        )

        step2 = Step(
            step_number=2,
            description="Corrected formulation",
            thoughts="Using dynamic programming approach",
            remaining_budget=10,
        )
        refl2 = Reflection(
            step_number=2,
            content="Corrected path validated.",
            reward=0.90,
        )

        task_obj = Task(
            description="Optimize complex routing",
            refined_description="Optimize complex routing with bounds",
            complexity=3,
            steps=[step1, step2],
            reflections=[refl1, refl2],
            answer="Final optimal route found",
            final_reward=-0.25,
            plan=Plan(steps=[]),
            output_type=OutputType(output_type="Simple Concise Answer", file_extension=".txt"),
        )

        interaction = Interaction(
            task=task_obj,
            steps=[step1, step2],
            reflections=[refl1, refl2],
            answer="Final optimal route found",
            final_reward=-0.25,
        )

        monkeypatch.setattr(AdvancedPromptEngineer, "main", Mock(return_value=interaction))

        req = RunRequest(mode="advanced_pipeline", task_prompt="Optimize complex routing")
        res = execute_run(req)

        assert res.status == "completed"
        assert res.diagnostics.final_reward == -0.25
        assert len(res.diagnostics.reflections) == 2

        # Step 1 reflection check
        r1 = res.diagnostics.reflections[0]
        assert r1.step_number == 1
        assert r1.reward == -0.85
        assert r1.content == "Hypothesis failed constraint check; negative reward assigned."

        # Step 2 reflection check
        r2 = res.diagnostics.reflections[1]
        assert r2.step_number == 2
        assert r2.reward == 0.90

        # Event stream check for negative reward
        refl_events = [e for e in res.events if e.event_type == "reflection_recorded"]
        assert len(refl_events) == 2
        assert refl_events[0].details["reward"] == -0.85
        assert "-0.85" in refl_events[0].summary

    def test_pipeline_step_with_none_attributes_graceful_handling(self, monkeypatch):
        """Step objects containing None description, thoughts, or reflection do not crash the pipeline."""
        from advanced_prompting import AdvancedPromptEngineer

        step = Step(
            step_number=1,
            description="",
            thoughts=None,
            remaining_budget=5,
        )
        # No reflection attached, no reflection in interaction.reflections
        task_obj = Task(
            description="Task with empty step attributes",
            refined_description="Refined",
            complexity=1,
            steps=[step],
            reflections=[],
            answer="Answer",
            final_reward=None,
            plan=Plan(steps=[]),
            output_type=OutputType(output_type="Simple Concise Answer", file_extension=".txt"),
        )

        interaction = Interaction(
            task=task_obj,
            steps=[step],
            reflections=[],
            answer="Answer",
            final_reward=None,
        )

        monkeypatch.setattr(AdvancedPromptEngineer, "main", Mock(return_value=interaction))

        req = RunRequest(mode="advanced_pipeline", task_prompt="Task with empty step")
        res = execute_run(req)

        assert res.status == "completed"
        assert res.diagnostics.final_reward == 0.0  # None coerced to 0.0
        assert len(res.diagnostics.reflections) == 1
        r = res.diagnostics.reflections[0]
        assert r.step_number == 1
        assert r.thoughts is None
        assert r.reward == 0.0
        assert r.content == "Step completed."

    def test_pipeline_step_with_explicit_none_step_number(self, monkeypatch):
        """When an ill-formed step has step_number=None, execute_run fail-closed wraps validation error."""
        from advanced_prompting import AdvancedPromptEngineer

        class MalformedStep:
            step_number = None
            plan_step_number = None
            description = "Malformed step"
            thoughts = None
            remaining_budget = 0
            reflection = None

        task_obj = Task(
            description="Malformed task",
            refined_description="Refined",
            complexity=1,
            steps=[],
            reflections=[],
            answer="Answer",
            final_reward=0.5,
            plan=Plan(steps=[]),
            output_type=OutputType(output_type="Simple Concise Answer", file_extension=".txt"),
        )

        interaction = Interaction(
            task=task_obj,
            steps=[MalformedStep()],  # type: ignore
            reflections=[],
            answer="Answer",
            final_reward=0.5,
        )

        monkeypatch.setattr(AdvancedPromptEngineer, "main", Mock(return_value=interaction))

        req = RunRequest(mode="advanced_pipeline", task_prompt="Malformed step test")
        res = execute_run(req)

        # step_number: int is required by ReflectionDiagnostic. If step_number is None,
        # Pydantic raises ValidationError, which execute_run catches fail-closed into status='error'.
        assert res.status == "error"
        assert res.error is not None
        assert res.error.error_type == "ValidationError"
        assert any(e.event_type == "run_error" for e in res.events)

    def test_pipeline_plan_step_with_none_optional_attributes(self, monkeypatch):
        """PlanStep with None or empty step_explanation and step_output deserializes safely."""
        from advanced_prompting import AdvancedPromptEngineer

        plan_step = PlanStep(
            step_number=1,
            completed=False,
            step_name="Step One",
            step_description="Description One",
            description="Description One",
            step_explanation="",
            step_output="",
            step_full_text="Step 1",
            subtasks=[],
        )
        plan = Plan(steps=[plan_step])

        task_obj = Task(
            description="Task with empty plan explanations",
            refined_description="Refined",
            complexity=1,
            steps=[],
            reflections=[],
            answer="Done",
            final_reward=0.8,
            plan=plan,
            output_type=OutputType(output_type="Simple Concise Answer", file_extension=".txt"),
        )
        interaction = Interaction(
            task=task_obj,
            steps=[],
            reflections=[],
            answer="Done",
            final_reward=0.8,
        )

        monkeypatch.setattr(AdvancedPromptEngineer, "main", Mock(return_value=interaction))

        req = RunRequest(mode="advanced_pipeline", task_prompt="Plan step test")
        res = execute_run(req)

        assert res.status == "completed"
        assert len(res.diagnostics.plan_steps) == 1
        ps = res.diagnostics.plan_steps[0]
        assert ps.step_number == 1
        assert ps.step_name == "Step One"
        assert ps.step_explanation == ""
        assert ps.step_output == ""
        assert ps.completed is False


# ============================================================================
# Challenge Suite 3: RunEvent ISO-8601 UTC Timestamps & JSON Serializability
# ============================================================================

class TestRunEventFidelityAndSerialization:
    """Empirically challenge RunEvent timestamp ISO-8601 UTC compliance and JSON payload serializability."""

    def test_run_event_default_timestamp_is_strict_iso8601_utc(self):
        """Default RunEvent timestamp must be strict ISO-8601 UTC with offset +00:00 or Z."""
        evt = RunEvent(
            event_type="mode_selected",
            mode="collaborative",
            summary="Testing timestamp",
        )

        assert evt.timestamp is not None
        # Must be parseable by fromisoformat
        dt = datetime.fromisoformat(evt.timestamp)
        assert dt is not None
        # Must be timezone-aware
        assert dt.tzinfo is not None
        # Must strictly have UTC offset == 0
        assert dt.utcoffset().total_seconds() == 0

    def test_all_emitted_run_events_have_strict_iso8601_utc_timestamps(self, monkeypatch):
        """Verify across all execution modes that every emitted RunEvent has strict ISO-8601 UTC timestamps."""
        import conversation_manager
        from advanced_prompting import AdvancedPromptEngineer

        # Mock collaborative
        monkeypatch.setattr(
            conversation_manager,
            "run_conversation",
            Mock(return_value=([
                {"role": "assistant", "name": "Analyst", "content": "Contribution", "round": 1}
            ], [], "Final output")),
        )

        # Mock advanced pipeline
        task_obj = Task(
            description="Task",
            refined_description="Task",
            complexity=1,
            steps=[Step(step_number=1, description="Step 1", thoughts="T1", remaining_budget=5)],
            reflections=[Reflection(step_number=1, content="Good", reward=0.8)],
            answer="Pipeline answer",
            final_reward=0.8,
            plan=Plan(steps=[PlanStep(
                step_number=1,
                completed=True,
                step_name="S1",
                step_description="D1",
                description="D1",
                step_explanation="",
                step_output="Out",
                step_full_text="1",
                subtasks=[],
            )]),
            output_type=OutputType(output_type="Simple Concise Answer", file_extension=".txt"),
        )
        interaction = Interaction(
            task=task_obj,
            steps=task_obj.steps,
            reflections=task_obj.reflections,
            answer="Pipeline answer",
            final_reward=0.8,
        )
        monkeypatch.setattr(AdvancedPromptEngineer, "main", Mock(return_value=interaction))

        # 1. Gated mode events
        res_gated = execute_run(RunRequest(mode="tree_of_thought", task_prompt="Gated test"))
        # 2. Collaborative mode events
        res_collab = execute_run(RunRequest(mode="collaborative", task_prompt="Collab test"))
        # 3. Advanced pipeline mode events
        res_pipeline = execute_run(RunRequest(mode="advanced_pipeline", task_prompt="Pipeline test"))
        # 4. Error mode events
        monkeypatch.setattr(conversation_manager, "run_conversation", Mock(side_effect=RuntimeError("Simulated boom")))
        res_error = execute_run(RunRequest(mode="collaborative", task_prompt="Error test"))

        all_events: List[RunEvent] = res_gated.events + res_collab.events + res_pipeline.events + res_error.events
        assert len(all_events) >= 12

        for evt in all_events:
            assert isinstance(evt, RunEvent)
            dt = datetime.fromisoformat(evt.timestamp)
            assert dt is not None, f"Failed to parse timestamp '{evt.timestamp}' on event '{evt.event_type}'"
            assert dt.tzinfo is not None, f"Timestamp '{evt.timestamp}' is naive on event '{evt.event_type}'"
            assert dt.utcoffset().total_seconds() == 0, f"Timestamp '{evt.timestamp}' is not UTC on event '{evt.event_type}'"

    def test_all_emitted_run_events_details_are_strictly_json_serializable(self, monkeypatch):
        """Verify across all execution modes that details payloads in every emitted event are strictly JSON serializable."""
        import conversation_manager
        from advanced_prompting import AdvancedPromptEngineer

        monkeypatch.setattr(
            conversation_manager,
            "run_conversation",
            Mock(return_value=([
                {"role": "assistant", "name": "Analyst", "content": "Contribution", "round": 1}
            ], [], "Final output")),
        )

        task_obj = Task(
            description="Task",
            refined_description="Task",
            complexity=1,
            steps=[Step(step_number=1, description="Step 1", thoughts="T1", remaining_budget=5)],
            reflections=[Reflection(step_number=1, content="Good", reward=0.8)],
            answer="Pipeline answer",
            final_reward=0.8,
            plan=Plan(steps=[PlanStep(
                step_number=1,
                completed=True,
                step_name="S1",
                step_description="D1",
                description="D1",
                step_explanation="",
                step_output="Out",
                step_full_text="1",
                subtasks=[],
            )]),
            output_type=OutputType(output_type="Simple Concise Answer", file_extension=".txt"),
        )
        interaction = Interaction(
            task=task_obj,
            steps=task_obj.steps,
            reflections=task_obj.reflections,
            answer="Pipeline answer",
            final_reward=0.8,
        )
        monkeypatch.setattr(AdvancedPromptEngineer, "main", Mock(return_value=interaction))

        res_gated = execute_run(RunRequest(mode="tree_of_thought", task_prompt="Gated test"))
        res_collab = execute_run(RunRequest(mode="collaborative", task_prompt="Collab test"))
        res_pipeline = execute_run(RunRequest(mode="advanced_pipeline", task_prompt="Pipeline test"))

        monkeypatch.setattr(conversation_manager, "run_conversation", Mock(side_effect=RuntimeError("Simulated error")))
        res_error = execute_run(RunRequest(mode="collaborative", task_prompt="Error test"))

        all_results = [res_gated, res_collab, res_pipeline, res_error]

        for r in all_results:
            # 1. Test model_dump_json() on the entire RunResult
            dumped_json = r.model_dump_json()
            assert dumped_json is not None
            parsed = json.loads(dumped_json)
            assert parsed["mode"] == r.mode
            assert parsed["status"] == r.status

            # 2. Test json.dumps directly on evt.details for every single event
            for evt in r.events:
                assert isinstance(evt.details, dict)
                # Must serialize with standard json.dumps (not custom pydantic serializer)
                serialized = json.dumps(evt.details)
                deserialized = json.loads(serialized)
                assert isinstance(deserialized, dict)
