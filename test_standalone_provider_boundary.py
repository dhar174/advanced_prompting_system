"""Issue51: direct main -> real orchestration -> deterministic SDK, no paid calls."""
import math
from types import SimpleNamespace

import pytest

import advanced_prompting as ap
import provider_client as pc
from output_generator import get_active_run_id
from tools.e2e.provider_fixture import FakeProviderClient, scenario_scope


class RecordingProvider(FakeProviderClient):
    def __init__(self):
        super().__init__()
        self.observed=[]
        self.options=[]
        for resource,name in ((self.chat.completions,'create'),(self.beta.chat.completions,'parse'),(self.embeddings,'create')):
            original=getattr(resource,name)
            def record(*args,_original=original,**kwargs):
                ctx=pc.get_active_execution_context()
                self.observed.append({'context':ctx,'kwargs':dict(kwargs),'owner':get_active_run_id(),
                    'calls_made':ctx.calls_made if ctx else None})
                return _original(*args,**kwargs)
            setattr(resource,name,record)

    def with_options(self,**kwargs):
        self.options.append(dict(kwargs))
        return self


@pytest.fixture
def direct_provider(monkeypatch):
    for key in ('APS_DEFAULT_MODEL','APS_ALLOWED_MODELS','APS_DEFAULT_TEMPERATURE',
                'APS_PROVIDER_TIMEOUT_SECONDS','APS_PROVIDER_MAX_CALLS'):
        monkeypatch.delenv(key,raising=False)
    provider=RecordingProvider()
    monkeypatch.setattr(pc,'get_client',lambda:provider)
    return provider


def engine(**kwargs):
    return ap.AdvancedPromptEngineer(ap.PromptEngineeringConfig(agents=0,max_steps=1,initial_budget=1,n=1,max_retries=1,max_sc_calls=6,**kwargs))


def task():
    return 'Design a verification strategy and risk analysis'


@pytest.mark.parametrize('cap',[0,1])
def test_direct_main_establishes_policy_before_first_provider_call(direct_provider,cap):
    result=engine(max_calls=cap).main(task())
    assert result.status==result.stop_reason=='budget_exhausted'
    assert result.final_reward is None and not result.answer
    assert direct_provider.ledger.total_calls==cap
    if cap:
        ctx=direct_provider.observed[0]['context']
        assert ctx is not None and ctx.calls_made==cap and ctx.max_calls==cap
    assert pc.get_active_execution_context() is None and get_active_run_id() is None


def test_direct_main_omitted_model_inherits_environment(direct_provider,monkeypatch):
    monkeypatch.setenv('APS_DEFAULT_MODEL','gpt-4o')
    monkeypatch.setenv('APS_ALLOWED_MODELS','gpt-4o')
    result=engine().main(task())
    assert result.status=='completed',result
    assert direct_provider.observed
    assert {row['kwargs']['model'] for row in direct_provider.observed}=={'gpt-4o'}
    assert len({id(row['context']) for row in direct_provider.observed})==1
    ctx=direct_provider.observed[0]['context']
    assert ctx.calls_made==direct_provider.ledger.total_calls==len(ctx.recorded_calls)
    assert all(option['max_retries']==0 for option in direct_provider.options)
    assert all(0<row['kwargs']['timeout']<=120 for row in direct_provider.observed)


def test_disallowed_explicit_default_string_is_not_treated_as_omission(direct_provider,monkeypatch):
    monkeypatch.setenv('APS_DEFAULT_MODEL','gpt-4o')
    monkeypatch.setenv('APS_ALLOWED_MODELS','gpt-4o')
    with pytest.raises(pc.DisallowedModelError):
        engine(model='gpt-4o-mini').main(task())
    assert direct_provider.ledger.total_calls==0


@pytest.mark.parametrize('kwargs',[{'max_calls':2},{'timeout_seconds':1},{'max_calls':2,'timeout_seconds':1},
                                 {'allowed_models':('gpt-4o-mini',),'max_calls':2}])
def test_incomplete_parent_policy_is_rejected_without_dispatch(direct_provider,kwargs):
    with pc.provider_execution_scope(**kwargs) as parent:
        with pytest.raises(ValueError):
            engine().main(task())
        assert pc.get_active_execution_context() is parent and parent.calls_made==0
    assert direct_provider.ledger.total_calls==0


def test_exact_final_call_can_complete(direct_provider):
    first=engine().main(task())
    assert first.status=='completed'
    count=direct_provider.ledger.total_calls
    direct_provider.ledger.reset()
    direct_provider.observed.clear()
    result=engine(max_calls=count).main(task())
    assert result.status=='completed' and result.final_reward is not None
    ctx=direct_provider.observed[0]['context']
    assert direct_provider.ledger.total_calls==ctx.calls_made==ctx.max_calls==count
    assert ctx.termination_reason is None


def test_embedding_attempt_shares_parent_ledger_without_chat_model_allowlist(direct_provider):
    with pc.provider_execution_scope(allowed_models=('gpt-4o-mini',),max_calls=2,timeout_seconds=5) as parent:
        pc.LazyClient().embeddings.create(model='embedding-only',input='Text')
        result=engine().main(task())
        assert result.status=='budget_exhausted' and parent.calls_made==2
        assert len(parent.recorded_calls)==direct_provider.ledger.total_calls==2
        assert all(row['context'] is parent for row in direct_provider.observed)


def test_mutated_constructor_limits_are_validated_again_before_dispatch(direct_provider):
    instance=engine()
    instance.config.max_calls=math.inf
    with pytest.raises(ValueError):
        instance.main(task())
    assert direct_provider.ledger.total_calls==0


def test_mutated_model_is_an_explicit_request(direct_provider,monkeypatch):
    monkeypatch.setenv('APS_DEFAULT_MODEL','gpt-4o')
    monkeypatch.setenv('APS_ALLOWED_MODELS','gpt-4o')
    instance=engine()
    instance.config.model='gpt-4o-mini'
    with pytest.raises(pc.DisallowedModelError):
        instance.main(task())
    assert direct_provider.ledger.total_calls==0


def test_conflicting_explicit_model_cannot_change_caller_selection(direct_provider):
    with pc.provider_execution_scope(default_model='gpt-4o',allowed_models=('gpt-4o','gpt-4o-mini'),max_calls=10,timeout_seconds=5) as parent:
        with pytest.raises(ValueError,match='conflicts'):
            engine(model='gpt-4o-mini').main(task())
        assert parent.default_model=='gpt-4o' and parent.calls_made==0
    assert direct_provider.ledger.total_calls==0


def test_final_deadline_check_overrides_swallowed_evaluator_error(direct_provider,monkeypatch):
    clock={'now':1000.0}
    monkeypatch.setattr(pc.time,'monotonic',lambda:clock['now'])
    parse=direct_provider.beta.chat.completions.parse
    def expire_at_final_judge(**kwargs):
        if kwargs.get('response_format') is ap.FinalReviewScore:
            clock['now']=1002.0
            raise RuntimeError('SDK timeout sk-standalone-secret /private/standalone-path')
        clock['now']+=.001
        return parse(**kwargs)
    monkeypatch.setattr(direct_provider.beta.chat.completions,'parse',expire_at_final_judge)
    result=engine(timeout_seconds=1).main(task())
    assert result.status==result.stop_reason=='timeout'
    assert result.answer and result.final_reward is None
    assert 'sk-standalone-secret' not in result.error_result.model_dump_json()
    assert pc.get_active_execution_context() is None and get_active_run_id() is None


def test_late_returning_provider_does_not_dispatch_again(direct_provider,monkeypatch):
    import time
    original=direct_provider.chat.completions.create
    def slow(**kwargs):
        time.sleep(.07)
        return original(**kwargs)
    monkeypatch.setattr(direct_provider.chat.completions,'create',slow)
    result=engine(timeout_seconds=.05).main(task())
    assert result.status=='timeout' and direct_provider.ledger.total_calls==1
    assert direct_provider.options[0]['max_retries']==0
    assert 0<direct_provider.options[0]['timeout']<=.05
    assert pc.get_active_execution_context() is None


def test_unexpected_failure_restores_owned_context(direct_provider,monkeypatch):
    # Narrow boundary unit: an unrelated pre-provider dependency raises.
    instance=engine()
    def failure(*args):
        raise RuntimeError('Dependency failed')
    monkeypatch.setattr(instance,'retrieve_information',failure)
    with pytest.raises(RuntimeError,match='Dependency failed'):
        instance.main(task())
    assert pc.get_active_execution_context() is None and get_active_run_id() is None


@pytest.mark.parametrize('scoped',[False,True])
def test_ambient_reservation_cannot_bypass_main_ledger(direct_provider,scoped):
    import contextlib
    scope=pc.provider_execution_scope(allowed_models=('gpt-4o-mini',),max_calls=8,timeout_seconds=120) if scoped else contextlib.nullcontext()
    with scope:
        with pc.reserve_call_scope('gpt-4o-mini'):
            if scoped:
                pc.LazyClient().chat.completions.create(model='gpt-4o-mini',messages=[{'role':'user','content':'Prior reserved SDK call'}])
            reservation=pc._IN_RESERVED_CALL.get()
            result=engine(max_calls=8).main(task())
            assert result.status=='budget_exhausted'
            assert pc._IN_RESERVED_CALL.get() is reservation
    contexts={id(row['context']):row['context'] for row in direct_provider.observed}
    assert len(contexts)==1
    ctx=next(iter(contexts.values()))
    assert direct_provider.ledger.total_calls==ctx.calls_made==len(ctx.recorded_calls)==8
    assert pc.get_active_execution_context() is None


@pytest.mark.parametrize('value',[-1,True,1.5,'1'])
def test_invalid_consumed_count_fails_before_dispatch(direct_provider,value):
    with pc.provider_execution_scope(allowed_models=('gpt-4o-mini',),max_calls=1,timeout_seconds=5) as parent:
        parent.calls_made=value
        with pytest.raises(ValueError):
            engine().main(task())
    assert direct_provider.ledger.total_calls==0


def test_consumed_count_cannot_be_rolled_back_against_existing_ledger(direct_provider):
    with pc.provider_execution_scope(allowed_models=('gpt-4o-mini',),max_calls=2,timeout_seconds=5) as parent:
        pc.LazyClient().chat.completions.create(model='gpt-4o-mini',messages=[{'role':'user','content':'Prior'}])
        parent.calls_made=0
        with pytest.raises(ValueError):
            engine().main(task())
    assert direct_provider.ledger.total_calls==1


@pytest.mark.parametrize('field,value',[('deadline_monotonic','bad'),('deadline_monotonic',10**400),
    ('temperature','bad'),('temperature',10**400),('allowed_models','gpt-4o-mini')])
def test_malformed_parent_fields_raise_clear_value_error(direct_provider,field,value):
    with pc.provider_execution_scope(allowed_models=('gpt-4o-mini',),max_calls=1,timeout_seconds=5) as parent:
        setattr(parent,field,value)
        with pytest.raises(ValueError):
            engine().main(task())
    assert direct_provider.ledger.total_calls==0


def test_valid_parent_wins_without_replenishing_prior_calls(direct_provider):
    with pc.provider_execution_scope(default_model='gpt-4o',allowed_models=('gpt-4o',),max_calls=2,timeout_seconds=5) as parent:
        pc.LazyClient().chat.completions.create(model='gpt-4o',messages=[{'role':'user','content':'Prior call'}])
        deadline=parent.deadline_monotonic
        result=engine(max_calls=200,timeout_seconds=600).main(task())
        assert result.status=='budget_exhausted'
        assert parent.calls_made==2 and parent.deadline_monotonic==deadline
        assert parent.max_calls==2 and pc.get_active_execution_context() is parent
        assert all(row['context'] is parent for row in direct_provider.observed)


@pytest.mark.parametrize('kind',['expired','cancelled'])
def test_early_parent_policy_stop_returns_fresh_interaction(direct_provider,kind):
    kwargs={'deadline_monotonic':pc.time.monotonic()-1} if kind=='expired' else {'timeout_seconds':5,'cancel_check':lambda:True}
    with pc.provider_execution_scope(allowed_models=('gpt-4o-mini',),max_calls=2,**kwargs) as parent:
        instance=engine()
        instance.task_object=SimpleNamespace(description='STALE_PREVIOUS_TASK')
        result=instance.main(task())
        assert result.status==('timeout' if kind=='expired' else 'cancelled')
        assert result.final_reward is None and result.error_result is not None
        assert result.task.description==task() and result.task.plan.steps==[]
        assert pc.get_active_execution_context() is parent
    assert direct_provider.ledger.total_calls==0
