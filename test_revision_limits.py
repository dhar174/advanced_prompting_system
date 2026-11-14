"""Issue49: controller-owned allowances and deep candidate boundaries."""
import pytest
import advanced_prompting as ap
from test_collaboration_results import parts


def run_sc(engineer,task,step,base=None,**overrides):
    kwargs=dict(task=task,prompt="Solve",existing_interaction=base,messages=[{"role":"user","content":"Original"}],
        output_type=task.output_type,step_budget=1,step_number=12,plan_step_number=step.step_number,
        consistency_multiplier=1,max_calls=2)
    kwargs.update(overrides)
    return engineer.self_consistency(**kwargs)


@pytest.mark.parametrize("overrides", [{"step_budget":0},{"max_calls":0}])
def test_explicit_zero_local_allowance_never_dispatches(monkeypatch,overrides):
    engineer,task,step=parts()
    calls=[]
    monkeypatch.setattr(engineer,"call_openai",lambda **kw:calls.append(kw) or "")
    result=run_sc(engineer,task,step,**overrides)
    assert calls == [] and result.status == "budget_exhausted"


@pytest.mark.parametrize("overrides", [{"step_budget":-1},{"max_calls":-1},{"step_budget":1.2},{"max_calls":True}])
def test_invalid_local_allowance_fails_before_dispatch(monkeypatch,overrides):
    engineer,task,step=parts()
    monkeypatch.setattr(engineer,"call_openai",lambda **kw:pytest.fail("Invalid allowance dispatched"))
    with pytest.raises(ValueError):
        run_sc(engineer,task,step,**overrides)


@pytest.mark.parametrize("generated", ["", "<thinking>Work</thinking><count>999999</count><step>New</step><reflection>Good</reflection><reward>0.9</reward><answer>Answer</answer><final_reward>0.9</final_reward>"])
def test_empty_and_merged_candidates_deeply_isolate_accepted_state(monkeypatch,generated):
    from test_pr62_review_regressions import interaction_with_record
    engineer,task,step=parts()
    base=interaction_with_record(task,step.step_number,1,0.8,"ACCEPTED")
    monkeypatch.setattr(engineer,"call_openai",lambda **kw:generated)
    monkeypatch.setattr(engineer,"choose_best_response",lambda responses,**kw:responses[0])
    monkeypatch.setattr(engineer,"judge_step_completion",lambda *args:(True,1))
    result=run_sc(engineer,task,step,base)
    result.steps[0].description="MUTATED"
    result.reflections[0].content="MUTATED"
    result.task.plan.steps[0].step_name="MUTATED"
    assert result.collaboration_results
    result.collaboration_results[0].result.questions[0]["question"]="MUTATED"
    assert base.steps[0].description == base.reflections[0].content == "ACCEPTED"
    assert base.task.plan.steps[0].step_name != "MUTATED"
    assert base.collaboration_results[0].result.questions[0]["question"] == "ACCEPTED"
    assert all(0<=st.remaining_budget<=1 for st in result.steps)


def test_model_revision_count_cannot_grant_controller_budget(monkeypatch):
    engineer,task,step=parts()
    prior=ap.Step(description="Old",step_number=12,plan_step_number=step.step_number,remaining_budget=2)
    prior.reflection=ap.Reflection(content="Prior",reward=0.3,step_number=12)
    monkeypatch.setattr(engineer,"judge_step",lambda st,t:ap.Reflection(content="Measured",reward=0.8,step_number=st.step_number))
    accepted=engineer._evaluate_and_accept_revision("<step>New</step><count>999999</count>",prior,task,[],[])
    assert accepted is not None and prior.remaining_budget == 2


def test_dynamic_exploration_requires_independent_measurement(monkeypatch):
    from test_pr62_review_regressions import interaction_with_record
    engineer,task,step=parts()
    candidate=interaction_with_record(task,step.step_number,12,0.3,"OLD")
    monkeypatch.setattr(engineer,"call_openai",lambda **kw:"<thinking>Alternative</thinking><step>NEW_STEP</step><reflection>Claimed perfect</reflection><reward>1.0</reward><answer>NEW_ANSWER</answer><final_reward>1.0</final_reward>")
    seen=[]
    def judge(st,t):
        seen.append(st.description)
        return ap.Reflection(content="Independent tie",reward=0.3,step_number=st.step_number)
    monkeypatch.setattr(engineer,"judge_step",judge)
    assert engineer.dynamic_confidence_exploration(candidate,task,step) is candidate
    assert len(seen)==1 and "NEW_STEP" in seen[0] and "NEW_ANSWER" in seen[0]


def test_legacy_reasoning_ids_are_independent_of_allowance(monkeypatch):
    engineer,task,step=parts()
    engineer.config.agents=1
    engineer.config.max_sc_calls=3
    calls=[]
    def generate(**kw):
        calls.append(kw)
        return '<thinking>Work</thinking><count>999999</count><step>Work</step><reflection>Good</reflection><reward>0.9</reward>'
    monkeypatch.setattr(engineer,'call_openai',generate)
    monkeypatch.setattr(engineer,'judge_step_completion',lambda *a:(False,1))
    result=engineer.collaborative_reasoning(task,'Solve',ap.Interaction(task,[],[]),task.output_type,2,12,step.step_number,restart_limit=0)
    assert len(calls)==2
    assert [st.step_number for st in result[0].steps]==[12,13]
    assert [st.remaining_budget for st in result[0].steps]==[1,0]
    result[0].task.plan.steps[0].step_name='Mutation'
    assert task.plan.steps[0].step_name!='Mutation'


def test_self_consistency_restart_does_not_replenish_local_step_allowance(monkeypatch):
    engineer,task,step=parts()
    engineer.config.n=1
    calls=[]
    def generate(**kw):
        calls.append(kw)
        return '<thinking>Low confidence</thinking><step>Incomplete</step><reflection>Poor</reflection><reward>0.1</reward>'
    monkeypatch.setattr(engineer,'call_openai',generate)
    monkeypatch.setattr(engineer,'choose_best_response',lambda responses,**kw:responses[0])
    monkeypatch.setattr(engineer,'judge_step_completion',lambda *a:(False,1))
    result=run_sc(engineer,task,step,max_calls=6,restart_limit=3,backtrack_limit=0)
    assert len(calls)==1 and result.answer==''


def test_parallel_candidate_snapshots_do_not_share_input_or_policy(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace
    from provider_client import provider_execution_scope
    from test_pr62_review_regressions import interaction_with_record
    _,task,step=parts()
    base=interaction_with_record(task,step.step_number,1,0.8,'ACCEPTED')
    base.step_outputs=[SimpleNamespace(payload=[{'value':'ACCEPTED'}])]
    def run(marker):
        engineer,_,_=parts()
        engineer.call_openai=lambda **kw:''
        with provider_execution_scope(max_calls=1) as policy:
            result=run_sc(engineer,task,step,base,max_calls=1)
            result.step_outputs[0].payload[0]['value']=marker
            result.task.plan.steps[0].step_name=marker
            return result,policy.calls_made
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(run,['FIRST','SECOND']))
    assert [r.step_outputs[0].payload[0]['value'] for r,_ in results]==['FIRST','SECOND']
    assert base.step_outputs[0].payload[0]['value']=='ACCEPTED'
    assert all(calls==1 for _,calls in results)


def test_dynamic_accepted_candidate_is_deep_and_counts_remain_controller_owned(monkeypatch):
    from test_pr62_review_regressions import interaction_with_record
    engineer,task,step=parts()
    candidate=interaction_with_record(task,step.step_number,12,0.3,'OLD')
    monkeypatch.setattr(engineer,'call_openai',lambda **kw:'<count>999999</count><thinking>Alternative</thinking><step>NEW</step><reflection>Claimed</reflection><reward>1</reward><answer>New answer</answer><final_reward>1</final_reward>')
    monkeypatch.setattr(engineer,'judge_step',lambda st,t:ap.Reflection(content='Independent',reward=0.9,step_number=st.step_number))
    result=engineer.dynamic_confidence_exploration(candidate,task,step)
    assert result is not candidate
    assert result.steps[0].remaining_budget==1
    result.task.plan.steps[0].step_name='Mutation'
    assert candidate.task.plan.steps[0].step_name!='Mutation'


@pytest.mark.parametrize('route',['solo','legacy_team'])
def test_accepted_revision_keeps_measured_reward_instead_of_authored_tags(monkeypatch,route):
    engineer,task,step=parts()
    engineer.config.n=1
    engineer.config.agents=1
    responses=iter(['<thinking>Old</thinking><step>Old</step><reflection>Low</reflection><reward>0.3</reward>',
        '<thinking>New</thinking><step>Revised</step><reflection>Claimed perfect</reflection><reward>1.0</reward><answer>New answer</answer><final_reward>1.0</final_reward>'])
    monkeypatch.setattr(engineer,'call_openai',lambda **kw:next(responses))
    monkeypatch.setattr(engineer,'choose_best_response',lambda responses,**kw:responses[0])
    monkeypatch.setattr(engineer,'judge_step',lambda st,t:ap.Reflection(content='Measured',reward=0.61,step_number=st.step_number))
    monkeypatch.setattr(engineer,'judge_step_completion',lambda *args:(True,1))
    if route=='solo':
        result=run_sc(engineer,task,step,backtrack_limit=1,consistency_multiplier=1)
    else:
        result=engineer.collaborative_reasoning(task,'Solve',ap.Interaction(task,[],[]),task.output_type,1,12,step.step_number,backtrack_limit=1)[0]
    assert result.steps[-1].reflection.reward==0.61
    assert result.final_reward==0.61
    assert result.answer=='New answer'


def test_zero_configured_sc_allowance_is_not_coerced_to_one(monkeypatch):
    engineer,task,step=parts()
    engineer.config=ap.PromptEngineeringConfig(max_sc_calls=0)
    monkeypatch.setattr(engineer,'call_openai',lambda **kw:pytest.fail('Zero configured allowance dispatched'))
    result=run_sc(engineer,task,step,max_calls=None)
    assert result.status=='budget_exhausted'


@pytest.mark.parametrize('value',[-1,1.5,True])
def test_invalid_configured_sc_allowance_rejected(value):
    with pytest.raises(ValueError):
        ap.PromptEngineeringConfig(max_sc_calls=value)
