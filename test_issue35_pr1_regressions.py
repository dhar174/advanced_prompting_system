"""PR 1 regression tests for Issue #35: reasoning engine, conversation manager,
run-scoped artifacts/writers, and provider/run-status execution contracts."""

from __future__ import annotations

import copy
import csv
import os
import time
from types import SimpleNamespace

import pytest

import assistant_personalities
import conversation_manager
import execution_modes
import output_generator
import provider_client
from advanced_prompting import (
    AdvancedPromptEngineer,
    FinalPlanStepOutput,
    FinalStepOutput,
    Interaction,
    OutputType,
    PromptEngineeringConfig,
    Reflection,
    Step,
    Task,
)
from complexity_measures import Plan, PlanStep
from conversation_manager import (
    BinaryVote,
    ConfidenceVote,
    ConversationMemory,
    RunBudgetExceededError,
    RunCancelledError,
)
from execution_modes import (
    RunRequest,
    execute_run,
    list_capabilities,
    validate_provider_model,
)


def _make_engineer(**cfg_overrides) -> AdvancedPromptEngineer:
    cfg = PromptEngineeringConfig(**cfg_overrides)
    return AdvancedPromptEngineer(cfg)


def _make_task(desc: str = "Solve test problem") -> Task:
    return Task(
        description=desc,
        refined_description=desc,
        complexity=1,
        steps=[],
        reflections=[],
        answer="",
        final_reward=0.0,
        plan=Plan(
            steps=[
                PlanStep(
                    step_number=1,
                    completed=False,
                    step_name="Step 1",
                    step_description=desc,
                    step_explanation="Explain",
                    step_output="Out",
                    step_full_text=desc,
                    subtasks=[],
                )
            ],
            step_budget=2,
        ),
        output_type=OutputType(output_type="Simple Concise Answer", file_extension=".txt"),
    )


class TestPR1ReasoningEngineFixes:
    """Verify PR 1 fixes in advanced_prompting.py."""

    def test_self_consistency_isolates_candidate_paths_and_parses_revised_response(
        self, monkeypatch
    ):
        engineer = _make_engineer(n=2, initial_budget=2, max_sc_calls=10)
        observed_message_histories = []

        # Path 1 gets low reward (0.2) -> triggers revision -> revised XML returns answer and reward 0.9.
        # Path 2 must start from base_messages without Path 1's assistant/system mutations.
        responses = iter(
            [
                "<step>Path 1 initial weak step</step><thinking>weak</thinking><count>1</count>",
                "<step>Path 1 revised step</step><thinking>revised</thinking><count>1</count>"
                "<reward>0.9</reward><answer>Path 1 revised answer</answer>"
                "<final_reward>0.9</final_reward><agent_response>Path 1 revised final</agent_response>",
                "<step>Path 2 clean step</step><thinking>clean</thinking><count>1</count>"
                "<reward>0.85</reward><answer>Path 2 clean answer</answer>"
                "<final_reward>0.85</final_reward><agent_response>Path 2 clean final</agent_response>",
            ]
        )

        def fake_call_openai(messages, temperature=0.7, stop_sequence=None):
            observed_message_histories.append(copy.deepcopy(messages))
            return next(responses)

        judge_rewards = iter([0.2, 0.9, 0.85])

        def fake_judge_step(current_step, task):
            return Reflection(
                content="judged",
                reward=next(judge_rewards, 0.9),
                step_number=current_step.step_number,
            )

        monkeypatch.setattr(engineer, "call_openai", fake_call_openai)
        monkeypatch.setattr(engineer, "judge_step", fake_judge_step)
        monkeypatch.setattr(
            engineer,
            "judge_step_completion",
            lambda *a, **kw: (True, 1),
        )
        monkeypatch.setattr(
            engineer,
            "choose_best_response",
            lambda responses, *a, **kw: responses[0],
        )

        base_messages = [{"role": "user", "content": "Solve test problem"}]
        base_snapshot = copy.deepcopy(base_messages)
        task_obj = _make_task("Solve test problem")
        ot = OutputType(output_type="Simple Concise Answer", file_extension=".txt")

        interaction = engineer.self_consistency(
            task=task_obj,
            prompt="Solve test problem",
            existing_interaction=None,
            messages=base_messages,
            output_type=ot,
            step_budget=1,
            step_number=0,
            plan_step_number=1,
            consistency_multiplier=1,
        )

        # Caller's original messages list must remain unmodified.
        assert base_messages == base_snapshot
        # Path 2's first call must see only base_snapshot (isolated from Path 1's initial+revision calls).
        path2_first_call = observed_message_histories[2]
        assert path2_first_call == base_snapshot
        # Final answer must come from revised response, not stale weak step.
        assert "Path 1 revised answer" in interaction.answer
        assert interaction.steps[0].description == "Path 1 revised step"

    def test_self_consistency_empty_responses_do_not_raise_max_value_error(
        self, monkeypatch
    ):
        engineer = _make_engineer(n=1, initial_budget=1, max_sc_calls=5)
        monkeypatch.setattr(engineer, "call_openai", lambda *a, **kw: "")

        interaction = engineer.self_consistency(
            task=_make_task("Empty provider test"),
            prompt="Empty provider test",
            existing_interaction=None,
            messages=[{"role": "user", "content": "Empty provider test"}],
            output_type=OutputType(output_type="Simple Concise Answer", file_extension=".txt"),
            step_budget=1,
            step_number=0,
            plan_step_number=1,
            consistency_multiplier=1,
        )

        assert isinstance(interaction, Interaction)
        assert interaction.final_reward == 0.0
        assert interaction.answer == ""

    def test_self_consistency_enforces_max_calls_cancel_and_deadline(self, monkeypatch):
        engineer = _make_engineer(n=3, initial_budget=5, max_sc_calls=2)
        calls = {"count": 0}

        def endless_steps(*a, **kw):
            calls["count"] += 1
            return "<step>Keep stepping</step><count>1</count>"

        monkeypatch.setattr(engineer, "call_openai", endless_steps)
        monkeypatch.setattr(
            engineer,
            "judge_step",
            lambda step, task: Reflection(
                content="ok", reward=0.9, step_number=step.step_number
            ),
        )
        monkeypatch.setattr(
            engineer,
            "choose_best_response",
            lambda responses, *a, **kw: responses[0],
        )

        task_obj = _make_task("Budget cap test")
        ot = OutputType(output_type="Simple Concise Answer", file_extension=".txt")

        engineer.self_consistency(
            task=task_obj,
            prompt="Budget cap test",
            existing_interaction=None,
            messages=[{"role": "user", "content": "Budget cap test"}],
            output_type=ot,
            step_budget=5,
            step_number=0,
            plan_step_number=1,
            consistency_multiplier=3,
            max_calls=2,
        )
        assert calls["count"] == 2
        assert engineer._last_sc_termination["reason"] == "budget_exhausted"

        # Cancellation check
        calls["count"] = 0
        engineer.self_consistency(
            task=task_obj,
            prompt="Cancel test",
            existing_interaction=None,
            messages=[{"role": "user", "content": "Cancel test"}],
            output_type=ot,
            step_budget=5,
            step_number=0,
            plan_step_number=1,
            consistency_multiplier=2,
            cancel_check=lambda: True,
        )
        assert calls["count"] == 0
        assert engineer._last_sc_termination["reason"] == "cancelled"

        # Expired deadline
        engineer.self_consistency(
            task=task_obj,
            prompt="Timeout test",
            existing_interaction=None,
            messages=[{"role": "user", "content": "Timeout test"}],
            output_type=ot,
            step_budget=5,
            step_number=0,
            plan_step_number=1,
            consistency_multiplier=2,
            deadline_monotonic=time.monotonic() - 1.0,
        )
        assert engineer._last_sc_termination["reason"] == "timeout"

    def test_refine_prompt_triggers_at_zero_reward_and_main_step9_succeeds(
        self, monkeypatch
    ):
        engineer = _make_engineer(n=1, initial_budget=3)
        base_task = _make_task("Zero reward task")
        zero_interaction = Interaction(
            task=base_task,
            steps=[Step(step_number=1, description="s", remaining_budget=1, plan_step_number=1)],
            answer="",
            final_reward=0.0,
            reflections=[Reflection(content="Need better precision", reward=0.1, step_number=1)],
        )
        refined = engineer.refine_prompt(
            interaction=zero_interaction,
            task="Zero reward task",
            complexity=2,
            prompt="Start with a 3-step budget.",
        )
        assert "Reflection: Need better precision" in refined

        # Verify main() handles low/zero reward without AssertionError and invokes Step 9 refine_prompt cleanly.
        plan = Plan(
            steps=[
                PlanStep(
                    step_number=1,
                    completed=False,
                    step_name="Analyze",
                    step_description="Analyze constraints",
                    step_explanation="Analyze",
                    step_output="Constraints list",
                    step_full_text="Analyze constraints",
                    subtasks=[],
                )
            ],
            step_budget=2,
        )
        import advanced_prompting

        monkeypatch.setattr(engineer, "retrieve_information", lambda *a, **kw: "")
        monkeypatch.setattr(engineer, "task_into_prompt", lambda d: d)
        monkeypatch.setattr(
            engineer,
            "convert_planstep_to_prompt",
            lambda *a, **kw: "Focus on Analyze constraints",
        )
        monkeypatch.setattr(
            advanced_prompting,
            "output_type_determination",
            lambda *a, **kw: OutputType(
                output_type="Simple Concise Answer", file_extension=".txt"
            ),
        )
        monkeypatch.setattr(engineer, "assess_complexity", lambda *a, **kw: (1.0, plan))
        monkeypatch.setattr(engineer, "name_project", lambda *a, **kw: "Test Project")
        monkeypatch.setattr(engineer, "adjust_step_budget", lambda *a, **kw: 2)
        monkeypatch.setattr(
            engineer,
            "generate_initial_prompt",
            lambda *a, **kw: ("System", "Start with a 3-step budget."),
        )
        monkeypatch.setattr(
            engineer,
            "dynamic_confidence_exploration",
            lambda interaction, *a, **kw: interaction,
        )
        monkeypatch.setattr(
            engineer, "judge_step_completion", lambda *a, **kw: (True, 1)
        )
        monkeypatch.setattr(
            engineer,
            "finalize_step_output",
            lambda stp, *a, **kw: FinalStepOutput(
                final_output="done",
                output_type=OutputType(
                    output_type="Simple Concise Answer", file_extension=".txt"
                ),
                version=1,
                component_type="Standalone File",
                associated_plan_step=plan.steps[0],
                step=stp,
            ),
        )
        monkeypatch.setattr(
            engineer,
            "finalize_planstep_output",
            lambda steps_arg, *a, **kw: FinalPlanStepOutput(
                final_output="done",
                output_type=OutputType(
                    output_type="Simple Concise Answer", file_extension=".txt"
                ),
                version=1,
                component_type="Standalone File",
                steps=steps_arg or [],
                planstep=plan.steps[0],
                file_name="planstep.txt",
                parent_file_name="",
            ),
        )
        monkeypatch.setattr(
            engineer,
            "judge_final_answer",
            lambda task_desc, interaction: interaction,
        )

        sc_calls = []

        def fake_sc(*a, **kw):
            sc_calls.append(kw)
            if len(sc_calls) == 1:
                return Interaction(
                    task=base_task,
                    steps=[
                        Step(
                            step_number=1,
                            description="Draft",
                            remaining_budget=1,
                            plan_step_number=1,
                        )
                    ],
                    answer="Zero reward draft",
                    final_reward=0.0,
                    reflections=[Reflection(content="Revise depth", reward=0.0, step_number=1)],
                )
            return Interaction(
                task=base_task,
                steps=[
                    Step(
                        step_number=1,
                        description="Refined",
                        remaining_budget=1,
                        plan_step_number=1,
                    )
                ],
                answer="Refined final answer",
                final_reward=0.9,
                reflections=[],
            )

        monkeypatch.setattr(engineer, "self_consistency", fake_sc)
        collab_calls = []
        monkeypatch.setattr(
            engineer,
            "collaborative_reasoning_main",
            lambda *a, **kw: (
                collab_calls.append(kw)
                or Interaction(
                    task=base_task,
                    steps=[],
                    reflections=[Reflection(content="Revise depth", reward=0.0, step_number=1)],
                    answer="collab",
                    final_reward=0.0,
                )
            ),
        )

        final_interaction = engineer.main("Low reward initial task")
        assert len(collab_calls) >= 1 and all(isinstance(c["step_budget"], int) for c in collab_calls)
        assert len(sc_calls) >= 2 and "Reflection: Revise depth" in sc_calls[1]["prompt"]
        assert isinstance(final_interaction, Interaction)

    def test_collaborative_reasoning_parses_revised_xml_without_index_shadowing(
        self, monkeypatch
    ):
        engineer = _make_engineer(n=1, initial_budget=2, agents=1)
        responses = iter(
            [
                "<step>Draft collab step</step><thinking>draft</thinking><count>1</count>",
                "<step>Revised collab step</step><thinking>better</thinking><count>1</count>"
                "<reward>0.9</reward><agent_response>Revised agent answer</agent_response>",
            ]
        )
        judge_rewards = iter([0.2, 0.9])
        monkeypatch.setattr(engineer, "call_openai", lambda *a, **kw: next(responses))
        monkeypatch.setattr(
            engineer,
            "judge_step",
            lambda step, task: Reflection(
                content="collab judged",
                reward=next(judge_rewards, 0.9),
                step_number=step.step_number,
            ),
        )
        monkeypatch.setattr(
            engineer,
            "judge_step_completion",
            lambda *a, **kw: (True, 1),
        )

        ot = OutputType(output_type="Simple Concise Answer", file_extension=".txt")
        task_obj = _make_task("Collab task")
        task_obj.plan = Plan(
            steps=[
                PlanStep(
                    step_number=1,
                    completed=False,
                    step_name="Collab Step",
                    step_description="Collaborate cleanly",
                    step_explanation="Explain",
                    step_output="Out",
                    step_full_text="Full",
                    subtasks=[],
                )
            ],
            step_budget=2,
        )

        interactions = engineer.collaborative_reasoning(
            task=task_obj,
            prompt="Collaborate cleanly",
            existing_interaction=None,
            output_type=ot,
            step_budget=2,
            step_number=0,
            plan_step_num=1,
        )
        assert isinstance(interactions, list) and len(interactions) == 1
        assert "Revised agent answer" in interactions[0].answer
        assert interactions[0].steps[0].description == "Revised collab step"


class TestPR1ConversationManagerFixes:
    """Verify PR 1 fixes in conversation_manager.py."""

    def test_extract_prompt_skips_system_banner_and_rejects_empty(self):
        prompt, prior = conversation_manager._extract_prompt_and_prior_turns(
            [
                {"role": "system", "content": "Collaborative debate instructions."},
                {"role": "user", "content": "First user question"},
                {"role": "assistant", "name": "Mediator", "content": "First summary"},
                {"role": "user", "content": "Follow-up user question"},
            ]
        )
        assert prompt == "Follow-up user question"
        assert len(prior) == 2
        assert prior[0]["content"] == "First user question"

        with pytest.raises(ValueError, match="no actionable user task prompt"):
            conversation_manager.run_conversation(
                [{"role": "system", "content": "Banner only"}],
                ["Analytical Thinker"],
                "Mediator",
            )

    def test_run_conversation_does_not_mutate_global_personalities_and_accumulates_questions(
        self, monkeypatch
    ):
        before_keys = set(assistant_personalities.assistant_personalities.keys())
        seen_models = []

        class _FakeCompletions:
            def create(self, **kwargs):
                seen_models.append(kwargs.get("model"))
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="Synthesized reply"))]
                )

        class _FakeBetaCompletions:
            def parse(self, **kwargs):
                seen_models.append(kwargs.get("model"))
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                parsed=OutputType(
                                    output_type="Simple Concise Answer",
                                    file_extension=".txt",
                                )
                            )
                        )
                    ]
                )

        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=_FakeCompletions()),
            beta=SimpleNamespace(chat=SimpleNamespace(completions=_FakeBetaCompletions())),
        )
        monkeypatch.setattr(conversation_manager, "client", fake_client)

        monkeypatch.setattr(
            conversation_manager,
            "define_problem",
            lambda *a, **kw: "Refined problem statement",
        )
        monkeypatch.setattr(
            conversation_manager,
            "calculate_priority",
            lambda *a, **kw: 10,
        )

        def fake_get_assistant_response(assistant_name, *args, **kwargs):
            seen_models.append(conversation_manager._resolve_model())
            return {"role": "assistant", "content": f"{assistant_name} response"}

        monkeypatch.setattr(
            conversation_manager,
            "get_assistant_response",
            fake_get_assistant_response,
        )
        monkeypatch.setattr(
            conversation_manager,
            "summarize_conversation",
            lambda *a, **kw: "Round summary",
        )

        extract_count = {"n": 0}

        def fake_extract_information(content, memory, personalities, rnd):
            extract_count["n"] += 1
            if extract_count["n"] == 1:
                memory.direct_replies = {
                    "Analytical Thinker": "clarification question?"
                }
            return memory

        monkeypatch.setattr(
            conversation_manager,
            "extract_information",
            fake_extract_information,
        )
        monkeypatch.setattr(
            conversation_manager,
            "analyze_to_do_list",
            lambda memory, msgs: memory,
        )
        monkeypatch.setattr(
            conversation_manager,
            "cast_binary_vote",
            lambda *a, **kw: BinaryVote(
                vote=False, reasoning="Ready to conclude"
            ),
        )
        monkeypatch.setattr(
            conversation_manager,
            "cast_confidence_vote",
            lambda *a, **kw: ConfidenceVote(
                vote=True, confidence=0.95, reasoning="Strong consensus"
            ),
        )
        monkeypatch.setattr(
            conversation_manager,
            "output_type_determination",
            lambda *a, **kw: OutputType(
                output_type="Simple Concise Answer", file_extension=".txt"
            ),
        )
        monkeypatch.setattr(
            conversation_manager,
            "finalize_output",
            lambda *a, **kw: "Final consensus output",
        )
        monkeypatch.setattr(
            conversation_manager,
            "test_finalize_output",
            lambda *a, **kw: "Final consensus output",
        )
        monkeypatch.setattr(
            conversation_manager,
            "analyze_final_output",
            lambda *a, **kw: SimpleNamespace(
                is_complete=True, explanations=[], corrections_needed=[]
            ),
        )

        # Round 1 asks a question AND receives END vote -> must still preserve Round 1's question!
        history, questions, final_out = conversation_manager.run_conversation(
            [{"role": "user", "content": "Compare indexing strategies"}],
            ["Analytical Thinker", "Nonexistent Persona"],
            "Mediator",
            num_rounds=2,
            model="gpt-4o",
        )

        assert set(assistant_personalities.assistant_personalities.keys()) == before_keys
        assert "Nonexistent Persona" not in assistant_personalities.assistant_personalities
        assert seen_models and all(m == "gpt-4o" for m in seen_models)
        assert len(questions) >= 1
        assert final_out == "Final consensus output"
        assert any(m.get("role") == "user" for m in history)

    def test_define_problem_votes_every_assistant_inside_inner_loop(self, monkeypatch):
        votes_cast = []

        class _FakeDefCompletions:
            def create(self, **kwargs):
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content=f"Candidate definition")
                        )
                    ]
                )

        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=_FakeDefCompletions())
        )
        monkeypatch.setattr(conversation_manager, "client", fake_client)

        def fake_confidence_vote(assistant_name, messages):
            votes_cast.append(assistant_name)
            return ConfidenceVote(vote=True, confidence=0.9, reasoning="Agree")

        monkeypatch.setattr(
            conversation_manager, "cast_confidence_vote", fake_confidence_vote
        )
        monkeypatch.setattr(
            conversation_manager,
            "cast_binary_vote",
            lambda *a, **kw: BinaryVote(vote=True, reasoning="Confirmed"),
        )

        selected = {
            "Analytical Thinker": "p1",
            "Skeptical Critic": "p2",
        }
        chosen = conversation_manager.define_problem(
            problem_statement="Define X",
            selected_personalities=selected,
        )
        assert chosen == "Candidate definition"
        # Both assistants vote on every candidate definition inside the inner loop (2 x 2 = 4).
        assert votes_cast == [
            "Analytical Thinker",
            "Skeptical Critic",
            "Analytical Thinker",
            "Skeptical Critic",
        ]


class TestPR1OutputGeneratorAndArtifacts:
    """Verify PR 1 fixes in output_generator.py."""

    def test_pdf_html_csv_writers_and_run_scoped_isolation(self, tmp_path):
        with output_generator.artifact_run_scope("run_alpha"):
            alpha_dir = output_generator.get_run_artifact_dir("run_alpha")
            pdf_path = output_generator.generate_pdf_output(
                "Quarterly Report\nRevenue (USD): $100",
                "report.pdf",
            )
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
            assert pdf_bytes.startswith(b"%PDF-1.4")
            assert b"%%EOF" in pdf_bytes[-32:]
            assert b"Quarterly Report" in pdf_bytes

            html_path = output_generator.generate_html_output(
                "<script>alert('xss')</script>",
                "safe.html",
            )
            with open(html_path, "r", encoding="utf-8") as f:
                html_text = f.read()
            assert html_text.startswith("<!DOCTYPE html>")
            assert "<script>" not in html_text
            assert "&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;" in html_text

            csv_path = output_generator.generate_csv_output(
                [{"col": 'a, "b"\nc', "val": 1}],
                "rows.csv",
            )
            with open(csv_path, "r", encoding="utf-8", newline="") as f:
                parsed = list(csv.reader(f))
            assert parsed == [["col", "val"], ['a, "b"\nc', "1"]]

            alpha_artifacts = output_generator.list_run_artifacts("run_alpha")
            assert len(alpha_artifacts) == 3
            assert os.path.commonpath([alpha_dir, pdf_path]) == alpha_dir

            # Traversal and arbitrary external absolute path writes rejected inside run scope
            with pytest.raises(PermissionError):
                output_generator.resolve_output_filepath("../escape.txt")
            with pytest.raises(PermissionError):
                output_generator.resolve_output_filepath(str(tmp_path / "outside.txt"))

            # Symlink escape rejected when OS allows symlinks
            outside_target = tmp_path / "host_secret.txt"
            outside_target.write_text("secret", encoding="utf-8")
            symlink_path = os.path.join(alpha_dir, "link_escape.txt")
            try:
                os.symlink(str(outside_target), symlink_path)
            except (OSError, NotImplementedError):
                pass
            else:
                with pytest.raises(PermissionError):
                    output_generator.resolve_output_filepath("link_escape.txt")
                assert output_generator.read_tracked_artifact(symlink_path) is None

        # Cross-run access denied even with absolute path
        with output_generator.artifact_run_scope("run_beta"):
            assert output_generator.read_tracked_artifact(pdf_path) is None
            assert output_generator.list_run_artifacts("run_beta") == []


class TestPR1ProviderConfigAndExecutionModes:
    """Verify PR 1 provider policy, gated integrity, and RunStatus contracts."""

    def test_validate_provider_model_and_disallowed_model_in_execute_run(self, monkeypatch):
        monkeypatch.setenv("APS_DEFAULT_MODEL", "gpt-4o-mini")
        monkeypatch.setenv("APS_ALLOWED_MODELS", "gpt-4o-mini,gpt-4o")

        assert validate_provider_model(None) == "gpt-4o-mini"
        assert validate_provider_model("gpt-4o") == "gpt-4o"
        with pytest.raises(execution_modes.DisallowedModelError):
            validate_provider_model("gpt-evil-unlisted")

        req = RunRequest(
            mode="collaborative",
            task_prompt="Test model policy",
            model="gpt-evil-unlisted",
        )
        res = execute_run(req)
        assert res.status == "error"
        assert res.stop_reason == "disallowed_model"
        assert res.final_output == ""
        assert res.error is not None
        assert "not permitted" in res.error.error_message

    def test_enforce_provider_ready_flag_and_capability_readiness(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        # Autouse conftest guard must NOT be mistaken for a configured live/fake provider
        assert provider_client.has_configured_provider() is False

        caps = list_capabilities(check_provider=True)
        collab = next(c for c in caps if c.id == "collaborative")
        assert collab.ready is False
        assert "OPENAI_API_KEY" in (collab.unavailable_reason or "")

        monkeypatch.setenv("APS_ENFORCE_PROVIDER_READY", "1")
        res = execute_run(RunRequest(mode="collaborative", task_prompt="Needs key"))
        assert res.status == "error"
        assert res.stop_reason == "provider_unavailable"
        assert res.final_output == ""

    def test_budget_exhausted_cancelled_timeout_and_incomplete_statuses(self, monkeypatch):
        import advanced_prompting
        import conversation_manager

        def raise_budget(*a, **kw):
            raise RunBudgetExceededError("Exceeded call budget of 2")

        def raise_timeout(*a, **kw):
            raise RunBudgetExceededError("Run timed out after 1.0s")

        def raise_cancel(*a, **kw):
            raise RunCancelledError("Client cancelled")

        monkeypatch.setattr(conversation_manager, "run_conversation", raise_budget)
        res_budget = execute_run(RunRequest(mode="collaborative", task_prompt="Budget"))
        assert res_budget.status == "budget_exhausted"
        assert res_budget.stop_reason == "budget_exhausted"

        monkeypatch.setattr(conversation_manager, "run_conversation", raise_timeout)
        res_timeout = execute_run(RunRequest(mode="collaborative", task_prompt="Timeout"))
        assert res_timeout.status == "timeout"
        assert res_timeout.stop_reason == "timeout"

        monkeypatch.setattr(conversation_manager, "run_conversation", raise_cancel)
        res_cancel = execute_run(RunRequest(mode="collaborative", task_prompt="Cancel"))
        assert res_cancel.status == "cancelled"
        assert res_cancel.stop_reason == "cancelled"

        monkeypatch.setattr(conversation_manager, "run_conversation", lambda *a, **kw: ([], [], ""))
        res_incomplete = execute_run(RunRequest(mode="collaborative", task_prompt="Empty"))
        assert res_incomplete.status == "incomplete"
        assert res_incomplete.stop_reason == "empty_output"

        monkeypatch.setattr(
            advanced_prompting.AdvancedPromptEngineer,
            "main",
            lambda self, task_desc: Interaction(
                task=_make_task(task_desc), steps=[], reflections=[], answer="", final_reward=0.0
            ),
        )
        res_empty_pipe = execute_run(RunRequest(mode="advanced_pipeline", task_prompt="Empty pipeline"))
        assert res_empty_pipe.status == "error"
        assert res_empty_pipe.stop_reason == "empty_pipeline"
        assert res_empty_pipe.final_output == ""
