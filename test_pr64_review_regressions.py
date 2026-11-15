"""PR64 review regressions: real policy/controller paths, deterministic SDK only."""
import subprocess
import sys

import pytest

import advanced_prompting as ap
import provider_client as pc
from test_standalone_provider_boundary import direct_provider


@pytest.mark.parametrize("probe", ["budget", "consume"])
@pytest.mark.parametrize("cancel", [False, True])
def test_cancellation_callback_can_read_stats_without_deadlock(probe, cancel):
    # A subprocess bounds the original deadlock and leaves no blocked test worker.
    code = """
import sys
import provider_client as pc
snapshots = []
cancel = sys.argv[2] == 'True'
def cancellation():
    snapshots.append(pc.get_execution_stats()['calls_made'])
    return cancel
with pc.provider_execution_scope(max_calls=2, timeout_seconds=5, cancel_check=cancellation) as ctx:
    try:
        if sys.argv[1] == 'budget':
            pc.check_provider_budget()
        else:
            pc.consume_provider_call()
    except pc.RunCancelledError:
        assert cancel
    else:
        assert not cancel
    assert snapshots == [0]
    assert ctx.calls_made == (1 if sys.argv[1] == 'consume' and not cancel else 0)
    assert pc.get_execution_stats()['termination_reason'] == ('cancelled' if cancel else None)
assert pc.get_active_execution_context() is None
print('CALLBACK_COMPLETE')
"""
    try:
        result = subprocess.run([sys.executable, "-c", code, probe, str(cancel)],
            capture_output=True, text=True, timeout=3)
    except subprocess.TimeoutExpired:
        pytest.fail("Cancellation callback deadlocked while reading execution stats")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CALLBACK_COMPLETE" in result.stdout


@pytest.mark.parametrize("value", [True, False, 1.5, 2.0, "2", -1, float("inf"), float("nan")])
def test_scope_rejects_coerced_allowances_before_replacing_parent(value):
    with pc.provider_execution_scope(max_calls=3, timeout_seconds=5) as parent:
        with pytest.raises(ValueError, match="max_calls"):
            with pc.provider_execution_scope(max_calls=value):
                pytest.fail("Invalid policy was installed")
        assert pc.get_active_execution_context() is parent
        assert parent.calls_made == 0


@pytest.mark.parametrize("value", [None, 0, 300])
def test_scope_preserves_valid_optional_zero_and_trusted_large_allowances(value):
    with pc.provider_execution_scope(max_calls=value, timeout_seconds=5) as ctx:
        assert ctx.max_calls == value
        if value == 0:
            with pytest.raises(pc.RunBudgetExceededError):
                pc.consume_provider_call()
            assert ctx.calls_made == 0


@pytest.mark.parametrize("reserved", [False, True])
def test_stats_reading_cancellation_stops_second_real_sdk_attempt(direct_provider, reserved):
    import contextlib
    snapshots = []
    def cancellation():
        snapshots.append(pc.get_execution_stats()["calls_made"])
        return direct_provider.ledger.total_calls >= 1
    with pc.provider_execution_scope(max_calls=3, timeout_seconds=5,
        cancel_check=cancellation) as ctx:
        scope = pc.reserve_call_scope if reserved else contextlib.nullcontext
        with scope():
            pc.LazyClient().chat.completions.create(model="gpt-4o-mini", messages=[])
        with pytest.raises(pc.RunCancelledError):
            with scope():
                pc.LazyClient().chat.completions.create(model="gpt-4o-mini", messages=[])
        assert ctx.calls_made == direct_provider.ledger.total_calls == 1
        assert pc.get_execution_stats()["termination_reason"] == "cancelled"
        assert 0 in snapshots and 1 in snapshots
    assert pc.get_active_execution_context() is None


def finalization_inputs():
    step = ap.Step(description="Accepted step content", step_number=1, remaining_budget=1)
    plan_step = ap.PlanStep(step_number=1, completed=False, step_name="Deliver summary",
        step_description="Deliver summary", step_explanation="Synthesize accepted content",
        step_output="Accepted summary", step_full_text="Deliver summary", subtasks=[])
    task = ap.Task("Deliver summary", "Deliver summary", .5, [step], [], "", None,
        ap.Plan(steps=[plan_step]), ap.OutputType(output_type="Text File", file_extension=".txt"))
    return step, task, plan_step


@pytest.mark.parametrize("stop,exception", [
    ("budget_exhausted", pc.RunBudgetExceededError),
    ("timeout", pc.RunTimeoutError),
    ("cancelled", pc.RunCancelledError),
])
def test_filename_policy_stop_propagates_before_synthetic_fallback(direct_provider, monkeypatch, stop, exception):
    clock = {"now": 1000.0}
    monkeypatch.setattr(pc.time, "monotonic", lambda: clock["now"])
    def cancellation():
        # Both real SDK calls (component decision + finalization) must finish.
        if direct_provider.ledger.total_calls == 2:
            if stop == "timeout":
                clock["now"] = 1002.0
            return stop == "cancelled"
        return False
    step, task, plan_step = finalization_inputs()
    engineer = ap.AdvancedPromptEngineer(ap.PromptEngineeringConfig())
    with pc.provider_execution_scope(allowed_models=("gpt-4o-mini",),
        max_calls=2 if stop == "budget_exhausted" else 5, timeout_seconds=1,
        cancel_check=cancellation) as ctx:
        with pytest.raises(exception):
            engineer.finalize_planstep_output([step], task, plan_step)
        assert ctx.termination_reason == stop
        assert [row["call_type"] for row in ctx.recorded_calls] == [
            "component_decision", "finalize_planstep_output"]
        assert ctx.calls_made == direct_provider.ledger.total_calls == 2
    assert pc.get_active_execution_context() is None


def test_filename_ordinary_failure_retains_fallback(direct_provider, monkeypatch):
    # Narrow filename fallback unit; no policy failure is injected here.
    step, task, plan_step = finalization_inputs()
    engineer = ap.AdvancedPromptEngineer(ap.PromptEngineeringConfig())
    def ordinary_failure(*args):
        raise RuntimeError("Filename helper unavailable")
    monkeypatch.setattr(engineer, "name_file", ordinary_failure)
    with pc.provider_execution_scope(max_calls=3, timeout_seconds=5):
        result = engineer.finalize_planstep_output([step], task, plan_step)
    assert result.file_name == "planstep_1_output.txt"
    assert result.final_output
