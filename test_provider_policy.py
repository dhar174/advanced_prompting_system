"""Deterministic regression tests for PR #39 request-scoped provider policy,
shared call ledger, monotonic timeout propagation, and public error sanitization.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import execution_modes
import provider_client
from execution_modes import RunRequest, execute_run


class _OptionsAwareClient:
    """Fake OpenAI-like client implementing class-level ``with_options``."""

    def __init__(self, handler, *, options_log=None):
        self._handler = handler
        self._options_log = options_log if options_log is not None else []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )
        self.beta = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(parse=self._parse)
            )
        )

    def with_options(self, **kwargs):
        self._options_log.append(dict(kwargs))
        return _OptionsAwareClient(self._handler, options_log=self._options_log)

    def _create(self, **kwargs):
        return self._handler("create", kwargs)

    def _parse(self, **kwargs):
        return self._handler("parse", kwargs)


def test_shared_ledger_max_calls_one_blocks_collaborative_second_call(monkeypatch):
    """APS_PROVIDER_MAX_CALLS=1 must allow at most one outbound SDK call in collaborative mode."""
    monkeypatch.setenv("APS_PROVIDER_MAX_CALLS", "1")
    monkeypatch.setenv("APS_DEFAULT_MODEL", "gpt-4o-mini")
    monkeypatch.delenv("APS_ALLOWED_MODELS", raising=False)

    calls = []

    def handler(kind, kwargs):
        calls.append((kind, kwargs))
        # Return a valid parse-like response on call #1 (define_problem), then fail on call #2
        msg = SimpleNamespace(
            parsed=SimpleNamespace(vote=True),
            content="Agreed problem statement.",
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    fake_client = _OptionsAwareClient(handler)
    monkeypatch.setattr(provider_client, "get_client", lambda: fake_client)

    res = execute_run(
        RunRequest(
            mode="collaborative",
            task_prompt="Design an offline test strategy",
            assistant_personalities=["Analytical Thinker"],
            lead_personality="Analytical Thinker",
            num_rounds=1,
        )
    )

    assert len(calls) == 1
    assert res.status == "budget_exhausted"
    assert res.stop_reason == "budget_exhausted"
    assert provider_client.get_active_execution_context() is None


def test_shared_ledger_max_calls_one_blocks_advanced_pipeline_second_call(monkeypatch):
    """APS_PROVIDER_MAX_CALLS=1 must allow at most one outbound SDK call across advanced_pipeline."""
    monkeypatch.setenv("APS_PROVIDER_MAX_CALLS", "1")
    monkeypatch.setenv("APS_DEFAULT_MODEL", "gpt-4o-mini")
    monkeypatch.delenv("APS_ALLOWED_MODELS", raising=False)

    calls = []

    def handler(kind, kwargs):
        calls.append((kind, kwargs))
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="Plan step output", parsed=None)
                )
            ]
        )

    fake_client = _OptionsAwareClient(handler)
    monkeypatch.setattr(provider_client, "get_client", lambda: fake_client)

    import advanced_prompting
    import complexity_measures

    # Stub complexity/NLP plan generation to make deterministic LLM calls via LazyClient
    def fake_generate_plan(task_desc, *args, **kwargs):
        advanced_prompting.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "plan call 1"}],
        )
        advanced_prompting.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "plan call 2"}],
        )
        return []

    monkeypatch.setattr(complexity_measures, "generate_plan", fake_generate_plan)
    monkeypatch.setattr(
        advanced_prompting.AdvancedPromptEngineer,
        "main",
        lambda self, task_prompt: (
            self.call_openai("step 1"),
            self.call_openai("step 2"),
        ),
    )

    res = execute_run(
        RunRequest(
            mode="advanced_pipeline",
            task_prompt="Solve multi-step reasoning task",
            max_steps=3,
            agents=0,
        )
    )

    assert len(calls) == 1
    assert res.status == "budget_exhausted"
    assert res.stop_reason == "budget_exhausted"
    assert provider_client.get_active_execution_context() is None


def test_nondefault_allowlist_gpt4o_enforced_including_output_type_determination(monkeypatch):
    """When APS_ALLOWED_MODELS=gpt-4o (without gpt-4o-mini), gpt-4o-mini is never sent."""
    monkeypatch.setenv("APS_DEFAULT_MODEL", "gpt-4o")
    monkeypatch.setenv("APS_ALLOWED_MODELS", "gpt-4o")
    monkeypatch.setenv("APS_PROVIDER_MAX_CALLS", "10")

    observed_models = []

    def handler(kind, kwargs):
        observed_models.append(kwargs.get("model"))
        if kind == "parse":
            parsed = SimpleNamespace(
                output_type="Simple Concise Answer",
                file_extension=".txt",
            )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed, content="{}"))]
            )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Done", parsed=None))]
        )

    fake_client = _OptionsAwareClient(handler)
    monkeypatch.setattr(provider_client, "get_client", lambda: fake_client)

    import advanced_prompting
    import conversation_manager

    # 1. Explicit disallowed request model fails closed at boundary
    disallowed_res = execute_run(
        RunRequest(
            mode="collaborative",
            task_prompt="Test allowlist",
            model="gpt-4o-mini",
        )
    )
    assert disallowed_res.status == "error"
    assert disallowed_res.stop_reason == "disallowed_model"
    assert observed_models == []

    # 2. Within an allowed gpt-4o request scope, output_type_determination & determine_output_type
    # must resolve to gpt-4o (never hardcoded gpt-4o-mini).
    with provider_client.provider_execution_scope(
        default_model="gpt-4o",
        allowed_models=("gpt-4o",),
        max_calls=5,
        timeout_seconds=10.0,
    ):
        ot = conversation_manager.output_type_determination("Summarize quarterly metrics")
        assert ot.output_type == "Simple Concise Answer"

        engineer = advanced_prompting.AdvancedPromptEngineer(
            advanced_prompting.PromptEngineeringConfig(model=provider_client.get_effective_model())
        )
        inferred = engineer.determine_output_type_from_content(
            "Summarize quarterly metrics", "", None
        )
        assert inferred.output_type in ("Text File", "Simple Concise Answer")
        assert execution_modes._infer_output_type("Summarize quarterly metrics", None).output_type in (
            "Text File",
            "Simple Concise Answer",
        )

        # Explicit disallowed call through LazyClient raises DisallowedModelError pre-dispatch
        with pytest.raises(provider_client.DisallowedModelError):
            advanced_prompting.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "forbidden"}],
            )

    assert observed_models
    assert all(m == "gpt-4o" for m in observed_models)


def test_nested_conversation_scope_shares_ledger_and_cleans_up_in_finally(monkeypatch):
    """Nested _conversation_run_scope shares parent budget/deadline/allowlist without reset."""
    calls = []

    def handler(kind, kwargs):
        calls.append(kwargs.get("model"))
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok", parsed=None))]
        )

    fake_client = _OptionsAwareClient(handler)
    monkeypatch.setattr(provider_client, "get_client", lambda: fake_client)

    import conversation_manager

    with pytest.raises(provider_client.RunBudgetExceededError):
        with provider_client.provider_execution_scope(
            default_model="gpt-4o",
            allowed_models=("gpt-4o",),
            timeout_seconds=5.0,
            max_calls=2,
        ) as outer_ctx:
            outer_deadline = outer_ctx.deadline_monotonic

            conversation_manager.client.chat.completions.create(
                model=provider_client.get_effective_model(),
                messages=[{"role": "user", "content": "1"}],
            )

            # Nested scope asks for looser max_calls=20 and timeout=99, which MUST NOT expand outer scope
            with conversation_manager._conversation_run_scope(
                model=None,
                temperature=0.2,
                timeout_seconds=99.0,
                max_calls=20,
            ) as inner_ctx:
                assert provider_client.get_active_execution_context() is outer_ctx
                assert inner_ctx["max_calls"] == 2
                assert inner_ctx["calls_made"] == 1
                assert inner_ctx["deadline_monotonic"] == outer_deadline

                conversation_manager.client.chat.completions.create(
                    model=provider_client.get_effective_model(),
                    messages=[{"role": "user", "content": "2"}],
                )
                # 3rd call must be blocked BEFORE calling handler
                conversation_manager.client.chat.completions.create(
                    model=provider_client.get_effective_model(),
                    messages=[{"role": "user", "content": "3"}],
                )

    assert len(calls) == 2
    assert provider_client.get_active_execution_context() is None


def test_timeout_propagates_remaining_deadline_and_disables_sdk_retries(monkeypatch):
    """Remaining monotonic deadline flows into client with_options/timeout and halts slow runs."""
    options_log = []
    calls = []

    def slow_handler(kind, kwargs):
        calls.append(kwargs)
        time.sleep(0.08)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="slow", parsed=None))]
        )

    fake_client = _OptionsAwareClient(slow_handler, options_log=options_log)
    monkeypatch.setattr(provider_client, "get_client", lambda: fake_client)
    monkeypatch.setenv("APS_PROVIDER_TIMEOUT_SECONDS", "0.05")
    monkeypatch.setenv("APS_PROVIDER_MAX_CALLS", "5")
    monkeypatch.setenv("APS_DEFAULT_MODEL", "gpt-4o-mini")
    monkeypatch.delenv("APS_ALLOWED_MODELS", raising=False)

    res = execute_run(
        RunRequest(
            mode="collaborative",
            task_prompt="Slow task timeout check",
            assistant_personalities=["Analytical Thinker"],
            lead_personality="Analytical Thinker",
            num_rounds=1,
        )
    )

    assert len(calls) == 1
    assert len(options_log) == 1
    assert options_log[0]["max_retries"] == 0
    assert 0.0 < options_log[0]["timeout"] <= 0.05
    assert res.status == "timeout"
    assert res.stop_reason == "timeout"
    assert provider_client.get_active_execution_context() is None


def test_timeout_kwarg_injected_only_inside_active_scope(monkeypatch):
    """Callable without with_options receives timeout inside scope, untouched outside scope."""
    received_kwargs = []

    def raw_create(*, model, messages=None, timeout=None):
        received_kwargs.append({"model": model, "timeout": timeout})
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])

    raw_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=raw_create))
    )
    monkeypatch.setattr(provider_client, "get_client", lambda: raw_client)

    lazy = provider_client.LazyClient()
    # Outside scope: timeout is NOT injected
    lazy.chat.completions.create(model="fixture")
    assert received_kwargs[-1] == {"model": "fixture", "timeout": None}

    # Inside scope: remaining monotonic deadline IS injected
    with provider_client.provider_execution_scope(
        default_model="gpt-4o",
        allowed_models=("gpt-4o",),
        timeout_seconds=1.5,
        max_calls=2,
    ):
        lazy.chat.completions.create(model="gpt-4o")

    assert received_kwargs[-1]["model"] == "gpt-4o"
    assert 0.0 < received_kwargs[-1]["timeout"] <= 1.5


def test_public_error_sanitization_blocks_secrets_and_paths_in_all_modes(monkeypatch):
    """Secrets, Bearer tokens, and file paths must never appear in error or events[].summary."""
    secret_tok = "sk-live-secret-999988887777"
    bearer_tok = "Bearer tok_abc123xyz"
    win_path = r"C:\Users\alice\private\keys.txt"
    posix_path = "/home/runner/work/secret_repo/config.py"
    raw_msg = (
        f'OpenAI auth failed using {secret_tok} and {bearer_tok} '
        f'at File "{posix_path}", line 42 reading {win_path}'
    )

    # 1. Raised exception in advanced_pipeline
    import advanced_prompting

    monkeypatch.setattr(
        advanced_prompting.AdvancedPromptEngineer,
        "main",
        lambda self, task_prompt: (_ for _ in ()).throw(RuntimeError(raw_msg)),
    )
    res_adv = execute_run(
        RunRequest(mode="advanced_pipeline", task_prompt="Trigger error sanitization")
    )
    assert res_adv.status == "error"
    assert res_adv.error is not None
    serialized_adv = json.dumps(res_adv.model_dump(mode="json"))
    for sensitive in (secret_tok, "tok_abc123xyz", win_path, posix_path, "Traceback"):
        assert sensitive not in serialized_adv

    # 2. Returned ErrorResult in collaborative mode
    import conversation_manager

    err_obj = conversation_manager.ErrorResult(
        error_type="ProviderAuthError",
        error_message=raw_msg,
        error_details=f"Traceback (most recent call last):\n  File \"{posix_path}\", line 10\n{secret_tok}",
    )
    monkeypatch.setattr(
        conversation_manager,
        "run_conversation",
        lambda *args, **kwargs: ([], [], err_obj),
    )
    res_collab = execute_run(
        RunRequest(mode="collaborative", task_prompt="Trigger collaborative error sanitization")
    )
    assert res_collab.status == "error"
    assert res_collab.error is not None
    serialized_collab = json.dumps(res_collab.model_dump(mode="json"))
    for sensitive in (secret_tok, "tok_abc123xyz", win_path, posix_path, "Traceback"):
        assert sensitive not in serialized_collab


def test_advanced_pipeline_honest_completion_on_partial_budget_exhaustion(monkeypatch):
    """Step-1 output plus mid-plan budget exhaustion must NOT report status='completed'."""
    import advanced_prompting

    step1 = SimpleNamespace(
        step_number=1,
        step_name="Gather inputs",
        step_description="Gather",
        step_explanation="Explained",
        step_output="Partial step 1 output",
        completed=True,
        agents=["Analytical Thinker"],
        subtasks=[],
    )
    step2 = SimpleNamespace(
        step_number=2,
        step_name="Synthesize",
        step_description="Synthesize",
        step_explanation="",
        step_output="",
        completed=False,
        agents=["Synthesizer"],
        subtasks=[],
    )
    task_obj = SimpleNamespace(
        complexity=1.0,
        plan=SimpleNamespace(steps=[step1, step2]),
        output_type=execution_modes.OutputType(
            output_type="Simple Concise Answer", file_extension=".txt"
        ),
    )
    ps_out1 = SimpleNamespace(final_output="Partial step 1 answer")
    partial_interaction = SimpleNamespace(
        task=task_obj,
        steps=[SimpleNamespace(step_number=1, description="Step 1", thoughts=None, reflection=None)],
        reflections=[],
        planstep_outputs=[ps_out1],
        answer="Partial step 1 answer",
        status="budget_exhausted",
        stop_reason="budget_exhausted",
    )

    monkeypatch.setattr(
        advanced_prompting.AdvancedPromptEngineer,
        "main",
        lambda self, task_prompt: partial_interaction,
    )

    res = execute_run(
        RunRequest(mode="advanced_pipeline", task_prompt="Multi-step test")
    )
    assert res.status == "budget_exhausted"
    assert res.stop_reason == "budget_exhausted"
    assert res.final_output == "Partial step 1 answer"

    # Also test empty answer with incomplete step 2 -> status='incomplete'
    partial_no_status = SimpleNamespace(
        task=task_obj,
        steps=[SimpleNamespace(step_number=1, description="Step 1", thoughts=None, reflection=None)],
        reflections=[],
        planstep_outputs=[ps_out1],
        answer="",
    )
    monkeypatch.setattr(
        advanced_prompting.AdvancedPromptEngineer,
        "main",
        lambda self, task_prompt: partial_no_status,
    )
    res_inc = execute_run(
        RunRequest(mode="advanced_pipeline", task_prompt="Incomplete step 2 test")
    )
    assert res_inc.status == "incomplete"
    assert res_inc.stop_reason == "incomplete_plan_steps"


def test_policy_modules_have_no_import_time_openai_construction():
    """Fresh subprocess importing provider_client, conversation_manager, execution_modes stays pure."""
    repo_root = Path(__file__).resolve().parent
    env = dict(os.environ)
    env.pop("OPENAI_API_KEY", None)
    code = (
        "import openai, provider_client, conversation_manager, execution_modes; "
        "assert provider_client.get_active_execution_context() is None; "
        "print('IMPORT_OK')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "IMPORT_OK" in proc.stdout


def test_complexity_measures_seam_respects_policy_and_reraises_errors(monkeypatch):
    """complexity_measures works when client=None at import, uses effective model, and never swallows policy errors."""
    import complexity_measures

    observed_models = []

    def handler(kind, kwargs):
        observed_models.append((kind, kwargs.get("model")))
        if kind == "parse":
            parsed = complexity_measures.Plan(
                steps=[
                    complexity_measures.PlanStep(
                        step_number=1,
                        completed=False,
                        step_name="Step 1",
                        step_description="Do step 1",
                        step_explanation="Explain",
                        step_output="Out",
                        step_full_text="Step 1: Do step 1",
                        subtasks=[],
                    )
                ]
            )
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        message=SimpleNamespace(parsed=parsed, content="Step 1: Do step 1", refusal=None),
                    )
                ]
            )
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="1. Step one\n2. Step two\nMerged PlanStep: Step 1",
                        parsed=None,
                    )
                )
            ]
        )

    fake_client = _OptionsAwareClient(handler)
    monkeypatch.setattr(provider_client, "get_client", lambda: fake_client)
    monkeypatch.setattr(complexity_measures, "client", None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    # 1. Inside gpt-4o-only scope with client=None, complexity_measures uses LazyClient + gpt-4o
    with provider_client.provider_execution_scope(
        default_model="gpt-4o",
        allowed_models=("gpt-4o",),
        max_calls=10,
        timeout_seconds=10.0,
    ):
        assert complexity_measures.generate_plan_legacy("Test plan")
        assert complexity_measures.generate_follow_up_questions("Test plan") == ["Step one", "Step two"]
        plan = complexity_measures.convert_plan("1. Step one")
        assert isinstance(plan, complexity_measures.Plan)

    assert len(observed_models) == 3
    assert all(m == "gpt-4o" for _, m in observed_models)

    # 2. Policy / model errors inside complexity_measures handlers must never be swallowed
    with provider_client.provider_execution_scope(
        default_model="gpt-4o",
        allowed_models=("gpt-4o",),
        max_calls=0,
        timeout_seconds=10.0,
    ):
        with pytest.raises(provider_client.RunBudgetExceededError):
            complexity_measures.generate_plan_legacy("Blocked")
        with pytest.raises(provider_client.RunBudgetExceededError):
            complexity_measures.convert_plan("Blocked")
        with pytest.raises(provider_client.RunBudgetExceededError):
            complexity_measures.generate_plan("Blocked")
        with pytest.raises(provider_client.RunBudgetExceededError):
            complexity_measures.generate_follow_up_questions("Blocked")
        step_obj = complexity_measures.PlanStep(
            step_number=1,
            completed=False,
            step_name="S1",
            step_description="Desc",
            step_explanation="Exp",
            step_output="Out",
            step_full_text="Step 1",
            subtasks=[],
        )
        with pytest.raises(provider_client.RunBudgetExceededError):
            complexity_measures.merge_steps([step_obj], step_obj)

    with provider_client.provider_execution_scope(
        default_model="gpt-4o",
        allowed_models=("gpt-4o",),
        max_calls=5,
        timeout_seconds=10.0,
    ):
        with pytest.raises(provider_client.DisallowedModelError):
            complexity_measures.classify_remaining_text_structured("Leftover", model="gpt-4o-mini")


def test_advanced_pipeline_budget_exhausted_inside_assess_complexity(monkeypatch):
    """max_calls=1 exhausting inside assess_complexity() -> is_complex_llm() surfaces budget_exhausted cleanly."""
    import advanced_prompting
    import complexity_measures

    monkeypatch.setenv("APS_PROVIDER_MAX_CALLS", "1")
    monkeypatch.setenv("APS_DEFAULT_MODEL", "gpt-4o-mini")
    monkeypatch.delenv("APS_ALLOWED_MODELS", raising=False)

    calls = []

    def handler(kind, kwargs):
        calls.append(kind)
        if kind == "parse":
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            parsed=SimpleNamespace(output_type="Simple Concise Answer", file_extension=".txt"),
                            content="{}",
                        )
                    )
                ]
            )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Refined prompt", parsed=None))]
        )

    fake_client = _OptionsAwareClient(handler)
    monkeypatch.setattr(provider_client, "get_client", lambda: fake_client)
    monkeypatch.setattr(complexity_measures, "client", None)
    monkeypatch.setattr(complexity_measures, "get_nlp_spacy", lambda: (lambda text: [1, 2, 3]))
    monkeypatch.setattr(
        advanced_prompting.AdvancedPromptEngineer,
        "task_into_prompt",
        lambda self, task: task,
    )

    # First real provider call happens in output_type_determination -> exhausts max_calls=1;
    # second call inside assess_complexity() -> is_complex_llm() -> generate_plan_legacy()
    # must raise RunBudgetExceededError (never swallowed into complexity=0.0 AssertionError).
    def fake_ot(task):
        advanced_prompting.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": task}],
        )
        return advanced_prompting.OutputType(output_type="Simple Concise Answer", file_extension=".txt")

    monkeypatch.setattr(advanced_prompting, "output_type_determination", fake_ot)

    res = execute_run(
        RunRequest(
            mode="advanced_pipeline",
            task_prompt="Assess complexity with budget 1",
            max_steps=3,
            agents=0,
        )
    )
    assert len(calls) == 1
    assert res.status == "budget_exhausted"
    assert res.stop_reason == "budget_exhausted"
    assert provider_client.get_active_execution_context() is None
