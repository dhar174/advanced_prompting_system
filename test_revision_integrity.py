"""Issue 49: real revision controllers with scripted SDK replies below orchestration."""
from types import SimpleNamespace

import pytest

import conversation_manager as cm
import advanced_prompting as ap
from provider_client import provider_execution_scope, RunBudgetExceededError
from test_collaboration_results import parts


def analysis(*, met=False, complete=False, requirement="Required behavior", aligned=True):
    return cm.FinalOutputAnalysis(user_intent="Task", expected_output_type="Text",
        requirements=[cm.OutputRequirements(requirement=requirement, met=met)],
        matches_expected_output_type=True, aligns_with_user_intent=aligned,
        changes_needed_to_align=[] if complete else ["Improve required behavior"], is_complete=complete)


def install_replies(monkeypatch, replies):
    calls=[]
    replies=iter(replies)
    def parse(**kwargs):
        calls.append(kwargs)
        value=next(replies)
        if isinstance(value, Exception):
            raise value
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=value, refusal=None))])
    monkeypatch.setattr(cm, "client", SimpleNamespace(beta=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse)))))
    return calls


def output(text="NEW_ACCEPTED", version=999):
    return cm.FinalOutput(final_output=text, output_type=cm.OutputType(output_type="Text", file_extension=".txt"), version=version)


def analyze(**kwargs):
    return cm.analyze_final_output_result("OLD_ACCEPTED", "Mediator", ["Mediator"], "REAL_TASK", "REAL_PROBLEM",
        output().output_type, **kwargs)


def test_legacy_analyzer_at_version_ceiling_performs_no_work(monkeypatch):
    calls=install_replies(monkeypatch, [analysis(complete=True,met=True)])
    assert cm.analyze_final_output("OLD_ACCEPTED", "Mediator", [], "TASK", "PROBLEM", output().output_type, 5) == "OLD_ACCEPTED"
    assert calls == []


def test_correction_uses_real_content_and_controller_version(monkeypatch):
    calls=install_replies(monkeypatch, [analysis(), output(), analysis(met=True,complete=True)])
    with provider_execution_scope(max_calls=3) as policy:
        result=analyze(current_version=4, max_version=5)
    assert result.accepted.final_output == "NEW_ACCEPTED"
    assert result.accepted.version == result.attempt_version == 5
    assert result.stop_reason == "completed" and result.error is None
    real_inputs=[m["content"] for m in calls[1]["messages"] if m["content"].startswith("Final Output:")]
    assert len(real_inputs)==1 and "OLD_ACCEPTED" in real_inputs[0]
    assert policy.calls_made == len(calls) == 3


@pytest.mark.parametrize("candidate_analysis", [analysis(), analysis(aligned=False,met=True), analysis(met=True,requirement="Changed requirement")])
def test_no_regression_rule_rejects_tie_regression_and_changed_checklist(monkeypatch, candidate_analysis):
    calls=install_replies(monkeypatch, [analysis(), output(), candidate_analysis])
    result=analyze(current_version=4,max_version=5)
    assert result.accepted.final_output == "OLD_ACCEPTED" and result.accepted.version == 4
    assert result.attempt_version == 5 and result.error is not None
    assert len(calls)==3


def test_rejected_corrections_consume_monotonic_version_and_shared_calls(monkeypatch):
    calls=install_replies(monkeypatch, [analysis(), output("REJECT_6"), analysis(), output("REJECT_7"), analysis(), output("REJECT_8"), analysis()])
    with provider_execution_scope(max_calls=7) as policy:
        result=analyze(current_version=5,max_version=8)
    assert result.accepted.final_output == "OLD_ACCEPTED"
    assert result.accepted.version == 5 and result.attempt_version == 8
    assert result.stop_reason == "revision_limit"
    assert len(calls)==policy.calls_made==7


@pytest.mark.parametrize("reply", [None, output(""), RuntimeError("sk-test-secret /private/test-path")])
def test_bad_corrections_never_replace_accepted_content(monkeypatch, reply):
    install_replies(monkeypatch, [analysis(),reply])
    result=analyze(current_version=4,max_version=5)
    assert result.accepted.final_output == "OLD_ACCEPTED"
    assert result.error is not None and "sk-test-secret" not in result.error.model_dump_json()
    assert "/private/test-path" not in result.error.model_dump_json()


def test_deadline_or_budget_preserves_prior_output_and_propagates_policy(monkeypatch):
    calls=install_replies(monkeypatch,[analysis(),output()])
    with provider_execution_scope(max_calls=2) as policy:
        with pytest.raises(RunBudgetExceededError):
            analyze(current_version=4,max_version=5)
    assert len(calls)==policy.calls_made==2


@pytest.mark.parametrize("versions", [(0,5),(-1,5),(1,0),(1,2.5),(True,5)])
def test_invalid_versions_rejected_before_dispatch(monkeypatch,versions):
    calls=install_replies(monkeypatch,[])
    with pytest.raises(ValueError):
        analyze(current_version=versions[0],max_version=versions[1])
    assert calls == []


@pytest.mark.parametrize('votes,keep', [([True,True],True),([False,False],False),([True,False],True),([None,None],True)])
def test_failed_revision_retains_partial_and_never_dispatches_writer(conversation_case,monkeypatch,votes,keep):
    values=iter(votes)
    def vote(*args):
        value=next(values)
        return cm.BinaryVote(vote=value) if value is not None else cm.ErrorResult(error_type='VoteError',error_message='Unavailable')
    monkeypatch.setattr(cm,'cast_binary_vote',vote)
    previous_parse=cm.client.beta.chat.completions.parse
    def parse(**kw):
        if kw['response_format'] is cm.MediatorReviseOrDecide:
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=cm.MediatorReviseOrDecide(final_decision='Revised decision',keep_output=keep)))])
        return previous_parse(**kw)
    monkeypatch.setattr(cm.client.beta.chat.completions,'parse',parse)
    def failed(*args):
        return cm.OutputRevisionResult(accepted=output('USEFUL_PARTIAL',version=4),attempt_version=5,stop_reason='revision_limit',
            error=cm.ErrorResult(error_type='RevisionLimit',error_message='Revision allowance exhausted'))
    monkeypatch.setattr(cm,'analyze_final_output_result',failed)
    result=cm.run_conversation_result('Task',[f'A{i}' for i in range(len(votes))],'A0',num_rounds=1)
    assert result.output.kind == 'failure' and result.output.error.error_type == 'RevisionLimit'
    assert any(turn.get('name')=='Partial Output' and turn['content']=='USEFUL_PARTIAL' for turn in result.conversation)
    conversation_case[0].assert_not_called()


def test_policy_stop_carries_last_accepted_revision_without_promoting_success(monkeypatch):
    first=analysis()
    # First correction improves a requirement but completeness remains false.
    improved=analysis(met=True)
    install_replies(monkeypatch,[first,output('ACCEPTED_PARTIAL'),improved])
    with provider_execution_scope(max_calls=3):
        with pytest.raises(RunBudgetExceededError) as caught:
            analyze(current_version=1,max_version=5)
    result=caught.value.revision_result
    assert result.accepted.final_output=='ACCEPTED_PARTIAL'
    assert result.accepted.version==2 and result.stop_reason=='budget_exhausted'
