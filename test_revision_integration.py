"""Issue49 HTTP proofs inject only deterministic provider responses."""
import json
import pytest
from test_graphql_integration import live_graphql_client, RUN_EXECUTION_MUTATION, _post_graphql
from tools.e2e.provider_fixture import scenario_scope


@pytest.mark.parametrize('mode',['collaborative','advanced_pipeline'])
@pytest.mark.parametrize('scenario',['revision_accept','revision_reject','revision_provider_failure'])
def test_http_revision_controller_preserves_actual_selected_content(live_graphql_client,monkeypatch,mode,scenario):
    client,_,ledger,_=live_graphql_client
    monkeypatch.setenv('APS_PROVIDER_MAX_CALLS','200')
    with scenario_scope(scenario):
        response=_post_graphql(client,RUN_EXECUTION_MUTATION,{'input':{'mode':mode,'taskPrompt':'Design a verification strategy and risk analysis',
            'assistantPersonalities':['Strategist','Mediator'],'leadPersonality':'Strategist','numRounds':1,'agents':2,'maxSteps':1,'initialBudget':1}})
    body=response.get_json()
    assert response.status_code==200 and 'errors' not in body,body
    result=body['data']['runExecution']
    serialized=json.dumps(result)
    if scenario=='revision_accept':
        assert result['status']=='completed',result
        assert 'ACCEPTED_REVISION' in serialized
    else:
        assert result['status']!='completed',result
        assert result['errorMessage']
        assert 'REJECTED_REVISION' not in serialized
        assert any(turn['name']=='Partial Output' and 'ORIGINAL_REVISION' in turn['content'] for turn in result['conversation'])
        assert not result['artifacts']
    assert 'sk-revision-secret' not in serialized and '/private/revision-path' not in serialized
    formats=[e.response_format for e in ledger.entries]
    assert 'FinalOutputAnalysis' in formats and 'FinalOutput' in formats
    assert 0<ledger.total_calls<=200


@pytest.mark.parametrize('mode',['collaborative','advanced_pipeline'])
def test_http_policy_stop_retains_last_accepted_correction(live_graphql_client,monkeypatch,mode):
    client,_,ledger,_=live_graphql_client
    payload={'input':{'mode':mode,'taskPrompt':'Design a verification strategy and risk analysis','agents':2,
        'maxSteps':1,'initialBudget':1,'assistantPersonalities':['Strategist','Mediator'],'leadPersonality':'Strategist','numRounds':1}}
    def run():
        with scenario_scope('revision_partial_policy'):
            response=_post_graphql(client,RUN_EXECUTION_MUTATION,payload)
        body=response.get_json()
        assert 'errors' not in body,body
        return body['data']['runExecution']
    monkeypatch.setenv('APS_PROVIDER_MAX_CALLS','200')
    assert run()['status']=='error'
    corrections=[entry.call_index for entry in ledger.entries if entry.response_format=='FinalOutput']
    assert len(corrections)==4
    cap=corrections[1]-1
    ledger.reset()
    monkeypatch.setenv('APS_PROVIDER_MAX_CALLS',str(cap))
    result=run()
    assert result['status']=='budget_exhausted' and result['stopReason']=='budget_exhausted',result
    assert any(turn['name']=='Partial Output' and turn['content']=='ACCEPTED_PARTIAL' for turn in result['conversation'])
    assert ledger.total_calls==cap
    assert not result['artifacts']
