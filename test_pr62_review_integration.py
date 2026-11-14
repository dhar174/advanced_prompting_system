"""Provider-only HTTP proofs for review concerns, not orchestrator replacements."""
import json
from pathlib import Path

import pytest

from output_generator import artifact_run_scope, generate_text_file_output, list_run_artifacts
from test_graphql_integration import live_graphql_client, RUN_EXECUTION_MUTATION, _post_graphql
from tools.e2e.provider_fixture import scenario_scope


def run(client, scenario, *, agents=2, steps=1):
    query=RUN_EXECUTION_MUTATION.replace("questions {", "questions { planStepNumber").replace("conversation {", "conversation { planStepNumber").replace("reflections {", "reflections { stepDescription")
    payload={"mode":"advanced_pipeline", "taskPrompt":"Design a verification strategy and risk analysis", "agents":agents, "maxSteps":steps, "initialBudget":steps}
    with scenario_scope(scenario):
        response=_post_graphql(client, query, {"input":payload})
    body=response.get_json()
    assert response.status_code == 200 and "errors" not in body, body
    return body["data"]["runExecution"]


@pytest.mark.parametrize("score", ["zero", "low"])
def test_http_solo_takeover_excludes_discarded_debate_and_artifacts(live_graphql_client, monkeypatch, score):
    client, provider, ledger, _=live_graphql_client
    monkeypatch.setenv("APS_PROVIDER_MAX_CALLS", "200")
    result=run(client, "review_takeover_"+score)
    assert result["status"] == "completed", result
    assert "SOLO_TAKEOVER_ANSWER" in result["finalOutput"]
    assert "DISCARDED" not in json.dumps(result)
    assert result["questions"] == [] and result["artifacts"] == []
    assert any("SOLO_TAKEOVER_STEP" in reflection["stepDescription"] for reflection in result["diagnostics"]["reflections"])
    owners={entry.correlation_id for entry in ledger.entries}
    assert len(owners) == 1
    assert any(ref.filename == "discarded-debate.txt" for ref in list_run_artifacts(next(iter(owners))))
    assert sum("Take a different approach" in prompt for prompt in ledger.captured_prompts) == 1
    assert 0 < ledger.total_calls <= 200


@pytest.mark.parametrize("scenario", ["two_step_metadata_partial_failure", "two_step_metadata_provider_failure"])
def test_http_fatal_partial_work_has_errors_and_both_steps_diagnostics(live_graphql_client, monkeypatch, scenario):
    client, _, ledger, _=live_graphql_client
    monkeypatch.setenv("APS_PROVIDER_MAX_CALLS", "200")
    result=run(client, scenario, steps=2)
    assert result["status"] == "error" and result["stopReason"] == "conversation_error", result
    assert result["errorMessage"]
    assert any(event["eventType"] == "run_error" for event in result["events"])
    assert result["diagnostics"]["planSteps"][0]["completed"]
    assert not result["diagnostics"]["planSteps"][1]["completed"]
    assert {q["planStepNumber"] for q in result["questions"]} == {1,2}
    assert all(turn["name"] != "Finalizer" for turn in result["conversation"])
    numbers=[turn["planStepNumber"] for turn in result["conversation"] if turn["planStepNumber"] is not None]
    assert numbers == sorted(numbers)
    serialized=json.dumps(result)
    assert "STEP1_HISTORY" in serialized and "STEP2_HISTORY" in serialized
    assert "sk-review-secret" not in serialized and "/private/review-path" not in serialized
    assert 0 < ledger.total_calls < 200


@pytest.mark.parametrize("scenario", ["review_final_judge_range", "review_final_judge_refused"])
def test_http_final_evaluation_failure_is_explicit_and_preserves_answer(live_graphql_client, monkeypatch, scenario):
    client, _, ledger, _=live_graphql_client
    monkeypatch.setenv("APS_PROVIDER_MAX_CALLS", "200")
    result=run(client, scenario, agents=0)
    assert result["status"] == "incomplete" and result["stopReason"] == "evaluation_unavailable", result
    assert result["finalOutput"] and result["diagnostics"]["finalReward"] is None
    assert result["errorMessage"]
    assert any(event["eventType"] == "run_error" for event in result["events"])
    assert all(turn["name"] != "Finalizer" for turn in result["conversation"])
    assert sum(entry.response_format == "FinalReviewScore" for entry in ledger.entries) == 1


@pytest.mark.parametrize("label", ["tenant/a", "tenant:a", "tenant/a"])
def test_service_nested_foreign_read_and_listing_denied_after_owner_success(live_graphql_client, monkeypatch, label):
    client, provider, ledger, _=live_graphql_client
    monkeypatch.setenv("APS_PROVIDER_MAX_CALLS", "200")
    from execution_modes import execute_run, RunRequest
    def service(label):
        with scenario_scope("review_owned_artifact"):
            return execute_run(RunRequest(mode="advanced_pipeline", task_prompt="Design a verification strategy and risk analysis", agents=2, max_steps=1, initial_budget=1, correlation_id=label))
    owned=service("tenant/a")
    owned=owned.model_dump(by_alias=True)
    assert owned["status"] == "completed" and len(owned["artifacts"]) == 1, owned
    ref=list_run_artifacts(owned["artifacts"][0]["run_id"])[0]
    assert Path(ref.path).read_text() == "OWNED_RESULT_BYTES"
    provider.output_filename=ref.path
    foreign=service(label).model_dump(mode="json")
    serialized=json.dumps(foreign)
    assert "OWNED_RESULT_BYTES" not in serialized
    assert ref.path not in serialized
    assert foreign["artifacts"] == []
    assert Path(ref.path).read_text() == "OWNED_RESULT_BYTES"
    assert any(entry.response_format == "ExtractFormat" for entry in ledger.entries)


def test_http_typed_path_text_never_reads_foreign_bytes(live_graphql_client, monkeypatch):
    client, provider, ledger, _=live_graphql_client
    monkeypatch.setenv("APS_PROVIDER_MAX_CALLS", "200")
    with artifact_run_scope("foreign"):
        path=generate_text_file_output("FOREIGN_PRIVATE_CONTENT", "foreign.txt")
    provider.output_filename=path
    result=run(client, "review_path_text")
    assert result["status"] == "completed", result
    assert path in result["finalOutput"]
    assert "FOREIGN_PRIVATE_CONTENT" not in json.dumps(result)
    assert result["artifacts"] == []


def test_http_selected_unevaluated_solo_candidate_stops_without_second_judge(live_graphql_client, monkeypatch):
    client, _, ledger, _=live_graphql_client
    monkeypatch.setenv("APS_PROVIDER_MAX_CALLS", "200")
    result=run(client, "review_sc_evaluation_refused", agents=0)
    assert result["status"] == "incomplete" and result["stopReason"] == "evaluation_unavailable", result
    assert result["finalOutput"] and result["diagnostics"]["finalReward"] is None
    assert result["errorMessage"]
    assert all(turn["name"] != "Finalizer" for turn in result["conversation"])
    # Each independent SC path is judged once; main never retries the selected one.
    attempts=sum(entry.response_format == "FinalReviewScore" for entry in ledger.entries)
    assert attempts == 5
    assert not any("finalizing the final output of a step" in prompt.lower() for prompt in ledger.captured_prompts)


def test_reused_engine_resets_bookkeeping_and_keeps_shared_provider_ledger(live_graphql_client):
    import advanced_prompting as ap
    from provider_client import provider_execution_scope
    _, _, ledger, _=live_graphql_client
    engineer=ap.AdvancedPromptEngineer(ap.PromptEngineeringConfig(agents=2, max_steps=1, initial_budget=1, n=1))
    with provider_execution_scope(max_calls=300) as policy:
        with scenario_scope("review_owned_artifact"):
            first=engineer.main("Design a verification strategy and risk analysis")
        assert first.status == "completed"
        assert engineer.last_conversation_history and engineer.last_questions_asked
        assert engineer._collaboration_artifact_ids
        first_calls=policy.calls_made
        engineer.config.agents=0
        with scenario_scope("default"):
            second=engineer.main("Design a second verification strategy and risk analysis")
        assert second.status == "completed"
        assert engineer.last_conversation_history == [] and engineer.last_questions_asked == []
        assert engineer._collaboration_artifact_ids == set()
        assert policy.calls_made > first_calls and policy.calls_made == ledger.total_calls
