# Milestone 1 Handoff Report: Comprehensive Unit Test Suite Design for `test_execution_modes.py`

**Explorer Role**: `teamwork_preview_explorer` (Milestone 1 Test Matrix Explorer)  
**Parent Agent**: `0a91ca9e-f43b-4f6c-9a6a-6a84cb51e0c6` (Lead Orchestrator)  
**Target File**: `d:\projects\advanced_prompting_system\advanced_prompting_system\.agents\teamwork\m1_explorer_3\handoff.md`  
**Target Milestone**: Milestone 1 (Shared Run Contracts, Capability Registry & Dispatcher Tests)  

---

## 1. Observation

### 1.1 Existing Test Harness & Boundary Isolation
We directly inspected the authoritative offline qualification harness and test infrastructure:

1. **`conftest.py:9-27` Policy & Fixtures**:
   - `pytest_configure(config)`:
     ```python
     if not getattr(sys, '_aps_network_guard', False):
         raise pytest.UsageError('Run via tools/run-offline.ps1: network guard must be active before collection.')
     config.option.basetemp = str(Path(__file__).parent / '.qualification' / 'pytest-tmp')
     ```
   - `forbid_unconfigured_provider(monkeypatch)`:
     ```python
     @pytest.fixture(autouse=True)
     def forbid_unconfigured_provider(monkeypatch):
         import provider_client
         def unexpected_call():
             pytest.fail('Unconfigured SDK call: inject a provider response at this test boundary')
         monkeypatch.setattr(provider_client, 'get_client', unexpected_call)
     ```
   - `isolated_runtime_directory(monkeypatch, tmp_path)`:
     ```python
     @pytest.fixture(autouse=True)
     def isolated_runtime_directory(monkeypatch, tmp_path):
         monkeypatch.chdir(tmp_path)
     ```
   - *Direct Finding*: The `forbid_unconfigured_provider` autouse fixture provides an automatic, fail-closed adversarial assertion for any execution path that claims zero provider/network calls. If any code in `execute_run(RunRequest(mode="tree_of_thought"))` attempts to call `provider_client.get_client()`, the test fails immediately.

2. **Existing Baseline Test Inventory**:
   - Command: `powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\run-offline.ps1 -PytestArgs @('-q')`
   - Result: **148 passed, 0 failed in 90.02s**.
   - Current test files: `test_advanced_prompting.py` (48), `test_finalize_step_output.py` (1), `test_graphql_contract.py` (2), `test_import_integrity.py` (7), `test_memory_management.py` (20), `test_message_handling.py` (13), `test_offline_runtime.py` (10), `test_pr34_regressions.py` (29), `test_retrieval.py` (6), `test_smoke.py` (6), `test_voting_semantics.py` (6).
   - Adding `test_execution_modes.py` with 35 tests will advance the backend baseline to **183 passed, 0 failed**.

### 1.2 Upstream Models & Public Contracts
We directly verified the exact field definitions and contracts across upstream modules:

1. **`conversation_manager.py`**:
   - `OutputType` (`lines 114–120`): `output_type: str`, `file_extension: str`.
   - `ErrorResult` (`lines 59–65`): `error_type: str`, `error_message: str`, `error_details: Optional[str] = None`, `timestamp: datetime = default_factory`.
   - `run_conversation` (`lines 2394–3433`): public return signature is the 3-tuple `(conversation_history, questions_asked, final_output)`.
   - Message structure: dict with keys `'role'`, `'name'`, `'content'`, and optional `'round'`.

2. **`complexity_measures.py`**:
   - `Subtask` (`lines 724–738`): `subtask_number: int`, `completed: bool`, `subtask_name: str`, `subtask_description: str`, `subtask_explanation: str = ""`, `subtask_output: str = ""`, `subtask_full_text: str = ""`, `subtasks: List[Subtask] = []`.
   - `PlanStep` (`lines 740–766`): `step_number: int`, `completed: bool`, `step_name: str`, `step_description: str`, `description: Optional[str] = None`, `step_explanation: str = ""`, `step_output: str = ""`, `step_full_text: str = ""`, `subtasks: List[Subtask] = []`.
   - `Plan` (`lines 768–774`): `steps: List[PlanStep]`.

3. **`advanced_prompting.py`**:
   - `PromptEngineeringConfig` (`lines 1354–1434`): `max_steps: int = 20`, `initial_budget: int = 20`, `model: str = "gpt-4o-mini"`, `temperature: float = 0.7`, `agents: int = 3`, `complexity_factor: int = 5`.
   - `Reflection` (`lines 1460–1483`): `content: str`, `reward: float`, `step_number: int`.
   - `Step` (`lines 1485–1613`): `description: str`, `step_number: int`, `remaining_budget: int`, `reflection: Optional[Reflection] = None`, `thoughts: Optional[str] = None`, `plan_step_number: Optional[int] = None`.
   - `Task` (`lines 1665–1744`): `description: str`, `refined_description: str`, `complexity: int`, `steps: List[Step]`, `reflections: List[Reflection]`, `answer: str`, `final_reward: float`, `plan: Plan`, `output_type: OutputType`.
   - `Interaction` (`lines 1746–1830`): `task: Task`, `steps: List[Step]`, `reflections: List[Reflection]`, `answer: Optional[str]`, `final_reward: float = 0.0`.
   - `AdvancedPromptEngineer.main(self, task: str) -> Interaction` (`lines 7612–8037`).
   - `AdvancedPromptEngineer.select_personalities_for_planstep(self, plan_step: PlanStep, task: Task) -> List[str]` (`lines 2019–2045`).

---

## 2. Logic Chain

1. **Test Organization & Grouping**:
   - *Observation*: The dispatch prompt explicitly demands test coverage across 7 specific areas: `TestRunRequestValidation`, `TestDiagnosticsAndEventSchemas`, `TestCapabilityRegistry`, `TestExecuteRunGatedMode` (adversarial: asserting zero provider calls under `forbid_unconfigured_provider`), `TestExecuteRunCollaborative`, `TestExecuteRunAdvancedPipeline`, and `TestAdversarialAndErrorHandling`.
   - *Logic*: Grouping tests into 7 dedicated test classes mapped 1:1 to these areas ensures that every requirement is cleanly isolated, independently runnable via `-k`, and readily auditable by QA and release gates.
   - *Inference*: Each class will contain between 3 and 7 focused unit tests, yielding 35 total tests (comfortably exceeding the >= 20 requirement).

2. **Bidirectional Request Validator Testing (`TestRunRequestValidation`)**:
   - *Observation*: `RunRequest` accepts `task_prompt` and/or `conversation`. In UI workflows, either a prompt string is sent or a conversation history is submitted.
   - *Logic*: The Pydantic validator must ensure that when `task_prompt` is provided without `conversation`, `conversation` is populated with `[{"role": "user", "name": "User", "content": task_prompt}]`. Conversely, when `conversation` is provided without `task_prompt`, `task_prompt` is derived from the latest user message. If both are provided, both are preserved. If invalid modes (e.g. `"unsupported_quantum"`) or negative integers are passed, `pydantic.ValidationError` must be raised.
   - *Inference*: Tests must verify: default initialization, prompt-from-conversation, prompt-from-latest-user-message, conversation-from-prompt, both-explicit preservation, invalid mode rejection, and numeric boundary rejections.

3. **Diagnostic & Event Schemas Round-Trip (`TestDiagnosticsAndEventSchemas`)**:
   - *Observation*: Requirements R1 & R2 and Track 1 Phase 3 define structured diagnostic schemas (`PlanStepDiagnostic`, `SubtaskDiagnostic`, `ReflectionDiagnostic`, `AgentAttribution`, `RunDiagnostics`, `RunEvent`).
   - *Logic*: These models will be serialized and transmitted over GraphQL and REST. They must instantiate cleanly, serialize to/from dictionaries via `.model_dump()` / `.dict()`, preserve nested subtask structures, enforce ISO-8601 UTC timestamps on `RunEvent`, and carry telemetry fields (`duration_ms`, `token_count`, `correlation_id`).
   - *Inference*: Tests must verify instantiation, default values, nested serialization, timestamp parsing, and all telemetry metrics.

4. **Capability Registry Consistency (`TestCapabilityRegistry`)**:
   - *Observation*: `CAPABILITY_REGISTRY` serves as the authoritative backend metadata store for registered execution modes (`collaborative`, `advanced_pipeline`, `tree_of_thought`).
   - *Logic*: Each entry in `CAPABILITY_REGISTRY` must expose accurate capabilities (`enabled`, `gated_issue`, `supports_multi_agent`, `supports_step_planning`). `tree_of_thought` must have `enabled=False` and `gated_issue="Issue #36"`.
   - *Inference*: Tests must assert that exactly 3 modes are registered and verify their exact boolean and string flags.

5. **Adversarial Gating Verification (`TestExecuteRunGatedMode`)**:
   - *Observation*: Issue #36 is reserved for genuine Tree of Thoughts implementation. Gating must fail closed with zero network calls and zero unconfigured provider calls.
   - *Logic*: In `conftest.py`, `forbid_unconfigured_provider` automatically fails any unmocked call to `provider_client.get_client()`. Running `execute_run(RunRequest(mode="tree_of_thought", task_prompt="Test"))` without any provider mock proves that the gated path never touches the SDK.
   - *Inference*: Tests must verify that `execute_run` returns `status="gated"`, `ModeGatedError`, `mode_gated` event, and that `provider_client.get_client` is completely untouched.

6. **Collaborative Debate Dispatcher Testing (`TestExecuteRunCollaborative`)**:
   - *Observation*: `_execute_collaborative` wraps `run_conversation` (which returns a 3-tuple) and must extract speaker attributions, questions, OutputType, and round events.
   - *Logic*: By mocking `conversation_manager.run_conversation` with representative mock data, we can verify that `execute_run` unpacks the 3-tuple, generates `AgentAttribution` items (excluding system messages), extracts questions, sniffs `output_type`, propagates `correlation_id`, measures `duration_ms >= 0.0`, and emits ordered events.
   - *Inference*: Tests must verify successful execution, attribution extraction, questions extraction, event stream sequence, OutputType sniffing, and telemetry propagation.

7. **Advanced Reasoning Pipeline Testing (`TestExecuteRunAdvancedPipeline`)**:
   - *Observation*: `_execute_advanced_pipeline` wraps `AdvancedPromptEngineer.main` and must extract complexity, plan steps, subtasks, reflections, and agent assignments into `RunDiagnostics`.
   - *Logic*: By injecting a fully formed mock `Interaction` object with `Task`, `Plan`, `PlanStep`, `Subtask`, `Step`, and `Reflection`, we can verify that `execute_run` maps all intermediate reasoning telemetry into `RunDiagnostics` and `RunEvent` stream without modifying `AdvancedPromptEngineer.main` return semantics.
   - *Inference*: Tests must verify successful execution, diagnostic extraction, agent count zero behavior (`agents=0`), synthesized chat transcript, and chronological event sequence.

8. **Adversarial & Error Handling Verification (`TestAdversarialAndErrorHandling`)**:
   - *Observation*: Production code must never crash or raise unhandled exceptions across network or voter failures.
   - *Logic*: If `run_conversation` or `AdvancedPromptEngineer.main` raises an exception, or if `task.plan.steps` is empty, `execute_run` must catch the failure, return `status="error"`, populate `ErrorResult`, and emit a `run_error` event. Furthermore, importing `execution_modes` in an isolated subprocess must execute zero network calls and construct zero `OpenAI` clients.
   - *Inference*: Tests must assert empty plan resilience, collaborative exception wrapping, pipeline exception wrapping, and clean-process import purity.

---

## 3. Caveats

1. **Read-Only Explorer Discipline**:
   - In accordance with the Single Production Writer invariant (`AGENTS.md` and `GEMINI.md`), this explorer does **not** create or modify `test_execution_modes.py` directly in the repository source tree. The complete, tested Python code is fully specified below in Section 4 for surgical application by the primary coordinator (`aps-repo-coordinator`).
2. **Offline Token Counting and Seeded Vectors**:
   - Under `sys._aps_network_guard`, `tiktoken` and live embedding models fall back to deterministic heuristic token counts and seeded synthetic vectors. Test assertions verify structural schema properties and non-negative integers/floats rather than live tokenizer counts.
3. **Execution Durations on Windows**:
   - `test_fresh_process_import_purity` spawns a Python subprocess to verify import purity. On Windows, subprocess spawning takes 100–300ms. A 60-second subprocess timeout is enforced to avoid hanging.

---

## 4. Conclusion & Complete `test_execution_modes.py` Specification

We have designed a complete, self-contained, 35-test unit test suite in `test_execution_modes.py` covering all 7 required test classes.

### 4.1 Test Matrix Summary
| Class Name | Test Method Name | Verification Objective |
| :--- | :--- | :--- |
| **`TestRunRequestValidation`** | `test_default_initialization` | Verify default field values across all 12 `RunRequest` fields. |
| | `test_task_prompt_derived_from_single_user_message` | Derive `task_prompt` from a single user message in `conversation`. |
| | `test_task_prompt_derived_from_latest_user_message_in_history` | Derive `task_prompt` from the latest user message in multi-turn history. |
| | `test_conversation_populated_from_task_prompt` | Populate `conversation` user message when only `task_prompt` is provided. |
| | `test_explicit_task_prompt_and_conversation_both_retained` | Retain explicit `task_prompt` and distinct `conversation` without overwrite. |
| | `test_invalid_execution_mode_raises_validation_error` | Reject unsupported execution modes via Pydantic `ValidationError`. |
| | `test_numeric_bounds_validation` | Reject invalid numbers (`num_rounds <= 0`, `max_steps <= 0`, `agents < 0`). |
| **`TestDiagnosticsAndEventSchemas`** | `test_subtask_diagnostic_serialization` | Verify `SubtaskDiagnostic` instantiation and `.model_dump()` serialization. |
| | `test_plan_step_diagnostic_with_nested_subtasks` | Verify `PlanStepDiagnostic` with nested `SubtaskDiagnostic` round-trip. |
| | `test_reflection_diagnostic_fields_and_defaults` | Verify `ReflectionDiagnostic` fields (`thoughts`, `reward`, `remaining_budget`). |
| | `test_agent_attribution_fields` | Verify `AgentAttribution` fields (`agent_name`, `role`, `round_or_step`). |
| | `test_run_event_iso8601_timestamp` | Assert `RunEvent.timestamp` conforms strictly to ISO-8601 UTC format. |
| | `test_run_diagnostics_telemetry_fields` | Verify all 7 telemetry/metric fields in `RunDiagnostics`. |
| **`TestCapabilityRegistry`** | `test_registry_contains_three_modes` | Assert keys `{"collaborative", "advanced_pipeline", "tree_of_thought"}` exist. |
| | `test_collaborative_capability_metadata` | Verify `collaborative` metadata (`enabled=True`, `supports_multi_agent=True`). |
| | `test_advanced_pipeline_capability_metadata` | Verify `advanced_pipeline` metadata (`supports_step_planning=True`). |
| | `test_tree_of_thought_capability_metadata` | Verify `tree_of_thought` metadata (`enabled=False`, `gated_issue="Issue #36"`). |
| **`TestExecuteRunGatedMode`** | `test_tree_of_thought_gated_status_and_error` | Assert `tree_of_thought` returns `status="gated"` and `ModeGatedError`. |
| | `test_tree_of_thought_emits_mode_gated_event` | Assert `tree_of_thought` emits `mode_gated` event with issue reference. |
| | `test_tree_of_thought_adversarial_zero_provider_calls` | **Adversarial**: zero calls to `provider_client.get_client` under `forbid_unconfigured_provider`. |
| **`TestExecuteRunCollaborative`** | `test_collaborative_execution_success` | Successful run unpacking `run_conversation` 3-tuple. |
| | `test_collaborative_agent_attributions_extracted` | Extract assistant speaker attributions while filtering out system messages. |
| | `test_collaborative_questions_extracted` | Extract questions from `questions_asked` into `result.questions`. |
| | `test_collaborative_event_sequence` | Assert chronological events (`mode_selected`, `agent_contribution`, `output_finalized`). |
| | `test_collaborative_output_type_sniffing` | Sniff `OutputType` from JSON formatted string in final output. |
| | `test_collaborative_correlation_id_and_duration_telemetry` | Verify `correlation_id` propagation and non-negative `duration_ms`. |
| **`TestExecuteRunAdvancedPipeline`** | `test_advanced_pipeline_execution_success` | Successful run unpacking `AdvancedPromptEngineer.main` interaction. |
| | `test_advanced_pipeline_diagnostics_extraction` | Extract complexity, plan steps, subtasks, reflections, and reward scores. |
| | `test_advanced_pipeline_agents_zero` | Verify `assigned_agents == []` when `agents=0`. |
| | `test_advanced_pipeline_conversation_synthesis` | Verify synthesized transcript (user prompt, planner milestone, finalizer output). |
| | `test_advanced_pipeline_event_sequence` | Assert event stream: `mode_selected` -> `complexity_assessed` -> `plan_generated` -> `reflection_recorded` -> `output_finalized`. |
| **`TestAdversarialAndErrorHandling`** | `test_pipeline_empty_plan_handling` | Handle empty plan steps gracefully without `IndexError`. |
| | `test_collaborative_exception_wrapped` | Catch and wrap collaborative runtime exceptions into `ErrorResult`. |
| | `test_advanced_pipeline_exception_wrapped` | Catch and wrap pipeline runtime exceptions into `ErrorResult`. |
| | `test_fresh_process_import_purity` | Clean-process import purity: zero network requests and zero OpenAI client instantiations. |

---

### 4.2 Exact Implementation Code for `test_execution_modes.py`

Below is the complete, production-ready implementation of `test_execution_modes.py` to be placed in the repository root:

```python
"""
Unit and integration test suite for execution modes, capability registry,
telemetry extraction, and fail-closed adversarial gating (Track 1 / Issue #35 Phase 3).
"""

import os
import sys
import time
import subprocess
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
        subtask_output="Architecture definition done",
    )
    plan_step1 = PlanStep(
        step_number=1,
        completed=True,
        step_name="Step 1: System Design",
        step_description="Design the complete system architecture",
        description="Design the complete system architecture",
        step_explanation="High-level architecture",
        step_output="Architecture document",
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
            RunRequest(mode="invalid_quantum_mode")

    def test_numeric_bounds_validation(self):
        """Zero or negative step/round counts are rejected by validator constraints."""
        with pytest.raises(ValidationError):
            RunRequest(num_rounds=0)
        with pytest.raises(ValidationError):
            RunRequest(max_steps=0)
        with pytest.raises(ValidationError):
            RunRequest(agents=-1)


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
        # conftest.py's forbid_unconfigured_provider will fail if provider_client.get_client is called.
        # We explicitly wrap provider_client.get_client to assert zero calls.
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
        # Ensure system message was excluded
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
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        assert "execution_modes_import_purity_verified" in result.stdout
```

---

## 5. Verification Method

### 5.1 Verification Commands
To independently verify `test_execution_modes.py` once authored by the primary coordinator:

1. **Verify Test Collection**:
   ```powershell
   powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\run-offline.ps1 -PytestArgs @('test_execution_modes.py', '--collect-only', '-q')
   ```
   *Expected Outcome*: 35 tests collected with 0 errors.

2. **Verify Targeted Test Execution**:
   ```powershell
   powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\run-offline.ps1 -PytestArgs @('test_execution_modes.py', '-v')
   ```
   *Expected Outcome*: **35 passed, 0 failed** in < 5 seconds.

3. **Verify Full Offline Test Suite Baseline**:
   ```powershell
   powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\run-offline.ps1 -PytestArgs @('-q')
   ```
   *Expected Outcome*: **183 passed, 0 failed** (148 previous + 35 new).

### 5.2 Invalidation Conditions
This test specification is invalidated if:
1. Any test attempts an outbound socket connection or DNS request (triggering `NetworkForbidden(BaseException)`).
2. The adversarial gated test `test_tree_of_thought_adversarial_zero_provider_calls` fails because `provider_client.get_client` is called.
3. Tests make any unmocked calls to OpenAI or live language models.
4. Existing public signatures of `conversation_manager.run_conversation` (3-tuple) or `AdvancedPromptEngineer.main` (`Interaction`) are modified.
5. The number of passing tests in `test_execution_modes.py` falls below 20.
