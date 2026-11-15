"""Issue51 direct-run acceptance proofs with fixtures only below orchestration."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading

import pytest

import advanced_prompting as ap
import provider_client as pc
from output_generator import get_active_run_id
from test_standalone_provider_boundary import direct_provider, task, engine
from tools.e2e.provider_fixture import scenario_scope


def team():
    return ap.AdvancedPromptEngineer(ap.PromptEngineeringConfig(agents=2,max_steps=2,initial_budget=2,n=1,
        max_sc_calls=6,max_calls=200))


def test_direct_nested_multistep_cap_is_shared_and_preserves_prior_output(direct_provider,monkeypatch):
    monkeypatch.setenv('APS_PROVIDER_MAX_CALLS','200')
    with scenario_scope('two_step_metadata'):
        result=team().main(task())
    assert result.status=='completed',result
    ledger=direct_provider.ledger
    second=next(entry.call_index for entry,prompt in zip(ledger.entries,ledger.captured_prompts)
        if entry.response_format=='ExtractFormat' and 'SECOND_STEP_OUTPUT' in prompt)
    cap=second+1
    ctx=direct_provider.observed[0]['context']
    assert ctx.calls_made==len(ctx.recorded_calls)==ledger.total_calls
    assert len({id(row['context']) for row in direct_provider.observed})==1
    assert len({row['owner'] for row in direct_provider.observed})==1
    types={entry['call_type'] for entry in ctx.recorded_calls}
    assert {'call_openai','choose_best_response','judge_step_completion','judge_final_answer'}<=types
    assert any('evaluate the following step' in prompt.lower() for prompt in ledger.captured_prompts)
    ledger.reset()
    direct_provider.observed.clear()
    monkeypatch.setenv('APS_PROVIDER_MAX_CALLS',str(cap))
    instance=ap.AdvancedPromptEngineer(ap.PromptEngineeringConfig(agents=2,max_steps=2,initial_budget=2,n=1,max_sc_calls=6))
    with scenario_scope('two_step_metadata'):
        result=instance.main(task())
    assert result.status==result.stop_reason=='budget_exhausted'
    assert ledger.total_calls==cap
    assert result.task.plan.steps[0].completed and not result.task.plan.steps[1].completed
    assert result.planstep_outputs and result.collaboration_results
    assert result.final_reward is None and result.error_result is not None
    assert pc.get_active_execution_context() is None and get_active_run_id() is None


@pytest.mark.parametrize('scenario,expected',[('revision_accept','completed'),('revision_reject','error'),('revision_provider_failure','error')])
def test_direct_nested_revisions_share_one_finite_ledger(direct_provider,monkeypatch,scenario,expected):
    monkeypatch.setenv('APS_PROVIDER_MAX_CALLS','200')
    instance=ap.AdvancedPromptEngineer(ap.PromptEngineeringConfig(agents=2,max_steps=1,initial_budget=1,n=1,max_sc_calls=6))
    with scenario_scope(scenario):
        result=instance.main(task())
    assert result.status==expected,result
    ctx=direct_provider.observed[0]['context']
    assert ctx.calls_made==len(ctx.recorded_calls)==direct_provider.ledger.total_calls
    assert len({id(row['context']) for row in direct_provider.observed})==1
    assert 'final_output_correction' in {entry['call_type'] for entry in ctx.recorded_calls}
    if expected=='completed':
        assert 'ACCEPTED_REVISION' in result.answer
    else:
        assert result.final_reward is None
        assert 'REJECTED_REVISION' not in str(result.to_dict())
    assert pc.get_active_execution_context() is None


def test_simultaneous_direct_runs_have_independent_ledgers_and_owners(direct_provider,monkeypatch):
    barrier=threading.Barrier(2)
    configure=direct_provider.with_options
    def synchronized_options(**kwargs):
        barrier.wait(timeout=5)
        return configure(**kwargs)
    monkeypatch.setattr(direct_provider,'with_options',synchronized_options)
    def run(index):
        result=engine(max_calls=1).main(task()+str(index))
        assert pc.get_active_execution_context() is None and get_active_run_id() is None
        return result
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(run,range(2)))
    assert all(result.status=='budget_exhausted' for result in results)
    assert direct_provider.ledger.total_calls==2
    assert len({id(row['context']) for row in direct_provider.observed})==2
    assert len({row['owner'] for row in direct_provider.observed})==2
    assert all(row['context'].calls_made==1 for row in direct_provider.observed)


def test_literal_readme_programmatic_example_runs_under_guard(direct_provider,capsys):
    readme=(Path(__file__).parent/'README.md').read_text()
    section=readme.split('## Python Programmatic Usage',1)[1]
    code=section.split('```python',1)[1].split('```',1)[0]
    namespace={}
    exec(compile(code,'README.md programmatic example','exec'),namespace)
    result=namespace['interaction']
    assert result.status=='completed',result
    assert result.task.plan.steps
    assert 'Plan Steps:' in capsys.readouterr().out
    assert 0<direct_provider.ledger.total_calls<=40
    assert pc.get_active_execution_context() is None


def retry_engine(**kwargs):
    return ap.AdvancedPromptEngineer(ap.PromptEngineeringConfig(agents=0,max_steps=1,initial_budget=1,n=1,
        max_sc_calls=6,max_retries=2,**kwargs))


def test_explicit_retry_attempts_consume_the_same_ledger(direct_provider,monkeypatch):
    waits=[]
    monkeypatch.setattr(ap.time,'sleep',lambda seconds:waits.append(seconds))
    with scenario_scope('standalone_retry'):
        result=retry_engine().main(task())
    assert result.status=='completed',result
    assert len(waits)==1
    ctx=direct_provider.observed[0]['context']
    assert ctx.calls_made==len(ctx.recorded_calls)==direct_provider.ledger.total_calls
    assert sum(bool(entry.error) for entry in direct_provider.ledger.entries)==1
    assert all(row['context'] is ctx for row in direct_provider.observed)
    assert all(option['max_retries']==0 for option in direct_provider.options)


def test_retry_wait_is_clipped_to_remaining_deadline(direct_provider,monkeypatch):
    clock={'now':1000.0}
    waits=[]
    monkeypatch.setattr(pc.time,'monotonic',lambda:clock['now'])
    def wait(seconds):
        waits.append(seconds)
        clock['now']+=seconds
    monkeypatch.setattr(ap.time,'sleep',wait)
    with scenario_scope('standalone_retry'):
        result=retry_engine(timeout_seconds=.25).main(task())
    assert result.status==result.stop_reason=='timeout'
    assert waits==[.25]
    assert direct_provider.ledger.entries[-1].error=='Deterministic transient failure'
    ctx=direct_provider.observed[0]['context']
    assert ctx.calls_made==direct_provider.ledger.total_calls
    assert pc.get_active_execution_context() is None


def test_cancellation_after_last_provider_result_preserves_accepted_answer(direct_provider):
    assert engine().main(task()).status=='completed'
    count=direct_provider.ledger.total_calls
    direct_provider.ledger.reset()
    with pc.provider_execution_scope(allowed_models=('gpt-4o-mini',),max_calls=count+1,timeout_seconds=120,
        cancel_check=lambda:direct_provider.ledger.total_calls>=count) as parent:
        result=engine().main(task())
        assert result.status==result.stop_reason=='cancelled'
        assert result.answer and result.final_reward is None
        assert parent.calls_made==direct_provider.ledger.total_calls==count
        assert pc.get_active_execution_context() is parent
    assert pc.get_active_execution_context() is None
