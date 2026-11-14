"""
Unmocked Flask HTTP /graphql -> execution_modes -> real runtime orchestrators integration tests.

Only `provider_client.get_client` is replaced with `FakeProviderClient` + `ProviderCallLedger`
so `schema.py`, `execution_modes.py`, `conversation_manager.py`, `advanced_prompting.py`,
`complexity_measures.py`, and `output_generator.py` run end-to-end without mocks.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict

import pytest

from app import create_app
from output_generator import clear_tracked_artifacts, list_run_artifacts, read_tracked_artifact
from tools.e2e.provider_fixture import (
    FakeProviderClient,
    ProviderCallLedger,
    install_fake_provider,
    scenario_scope,
)

RUN_EXECUTION_MUTATION = """
mutation RunExec($input: RunExecutionInput!) {
  runExecution(input: $input) {
    mode
    effectiveMode
    status
    stopReason
    finalOutput
    errorMessage
    events {
      eventType
      summary
    }
    usageAvailable
    outputType {
      outputType
      fileExtension
    }
    conversation {
      role
      name
      content
    }
    questions {
      assistant
      question
    }
    artifacts {
      artifactId
      runId
      filename
      path
      mimeType
      sizeBytes
      sha256
    }
    diagnostics {
      complexityScore
      initialBudget
      adjustedBudget
      finalReward
      durationMs
      correlationId
      planSteps {
        stepNumber
        stepName
        stepOutput
        completed
        subtasks {
          subtaskNumber
          completed
        }
      }
      reflections {
        stepNumber
        content
        reward
      }
      agentAttributions {
        agentName
        role
        roundOrStep
        contributionSummary
      }
    }
  }
}
"""

RUN_CONVERSATION_MUTATION = """
mutation RunLegacy($conversation: [ConversationInput!]!, $assistantPersonalities: [String!]!, $leadPersonality: String!, $numRounds: Int!) {
  runConversation(
    conversation: $conversation,
    assistantPersonalities: $assistantPersonalities,
    leadPersonality: $leadPersonality,
    numRounds: $numRounds
  ) {
    conversation { role name content }
    questions { assistant question }
    finalOutput
  }
}
"""


@pytest.fixture
def live_graphql_client(monkeypatch, tmp_path):
    """Create a Flask test client wired to a fresh FakeProviderClient & ProviderCallLedger."""
    clear_tracked_artifacts()
    monkeypatch.setenv("APS_LOG_FILE", str(tmp_path / "logs" / "conversation_logs.json"))
    ledger = ProviderCallLedger()
    provider = FakeProviderClient(ledger=ledger)
    install_fake_provider(provider, monkeypatch=monkeypatch)
    dist_dir = Path(__file__).resolve().parent / "collaborative-assistant-frontend" / "dist"
    app = create_app({"TESTING": True, "STATIC_FOLDER": str(dist_dir), "RATE_LIMIT_RPM": 200})
    return app.test_client(), provider, ledger, tmp_path


def _post_graphql(client, query: str, variables: Dict | None = None):
    return client.post(
        "/graphql",
        json={"query": query, "variables": variables or {}},
        content_type="application/json",
    )


def _run_collaborative_mutation(client, mutation_kind, *, model=None):
    """Exercise either public route without replacing resolvers or runtime helpers."""
    prompt = "[FORMAT:TXT] Build an offline verification checklist."
    if mutation_kind == "runConversation":
        query = RUN_CONVERSATION_MUTATION
        variables = {
            "conversation": [{"role": "user", "name": "User", "content": prompt}],
            "assistantPersonalities": ["Strategist", "Mediator"],
            "leadPersonality": "Strategist",
            "numRounds": 1,
        }
    else:
        query = RUN_EXECUTION_MUTATION
        input_payload = {
            "mode": "collaborative",
            "taskPrompt": prompt,
            "assistantPersonalities": ["Strategist", "Mediator"],
            "leadPersonality": "Strategist",
            "numRounds": 1,
        }
        if model is not None:
            input_payload["model"] = model
        variables = {"input": input_payload}
    response = _post_graphql(client, query, variables)
    assert response.status_code == 200
    return response.get_json()


def _assert_mutation_stopped(body, mutation_kind, expected_status):
    if mutation_kind == "runConversation":
        assert body.get("errors"), body
        assert not (body.get("data") or {}).get("runConversation"), body
    else:
        assert "errors" not in body, body
        result = body["data"]["runExecution"]
        assert result["status"] == expected_status, result
        assert result["stopReason"] == expected_status, result


@pytest.mark.parametrize(
    ("format_tag", "expected_ext", "signature"),
    [
        ("[FORMAT:TXT]", ".txt", b"Deterministic collaborative contribution"),
        ("[FORMAT:JSON]", ".json", b"verified"),
        ("[FORMAT:PDF]", ".pdf", b"%PDF-1."),
        ("[FORMAT:HTML]", ".html", b"<html"),
        ("[FORMAT:CSV]", ".csv", b"item,status"),
        ("[FORMAT:PY]", ".py", b"def solve"),
    ],
)
def test_http_graphql_collaborative_real_orchestration_and_artifacts(
    live_graphql_client, format_tag, expected_ext, signature
):
    client, _provider, ledger, tmp_path = live_graphql_client
    resp = _post_graphql(
        client,
        RUN_EXECUTION_MUTATION,
        {
            "input": {
                "mode": "collaborative",
                "taskPrompt": f"{format_tag} Build an offline verification checklist.",
                "conversation": [
                    {
                        "role": "system",
                        "name": "System",
                        "content": "UI BANNER MUST BE EXCLUDED FROM ORIGINAL_PROMPT",
                    },
                    {
                        "role": "user",
                        "name": "User",
                        "content": f"{format_tag} Build an offline verification checklist.",
                    },
                ],
                "assistantPersonalities": ["Strategist", "Mediator"],
                "leadPersonality": "Strategist",
                "numRounds": 1,
            }
        },
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert "errors" not in body, body
    data = body["data"]["runExecution"]

    assert data["mode"] == "collaborative"
    assert data["effectiveMode"] == "collaborative"
    assert data["status"] == "completed"
    assert data["stopReason"] in {"completed", "final_answer"}
    assert data["finalOutput"]
    assert data["outputType"]["fileExtension"] == expected_ext
    assert len(data["questions"]) >= 1
    assert any(m["role"] == "assistant" for m in data["conversation"])
    assert ledger.total_calls > 0
    assert all("UI BANNER MUST BE EXCLUDED" not in p for p in ledger.captured_prompts)

    corr_id = data["diagnostics"]["correlationId"]
    assert corr_id
    assert len(data["artifacts"]) >= 1
    art = data["artifacts"][0]
    assert art["filename"].endswith(expected_ext)
    assert art["path"].startswith(f"artifacts/{art['runId']}/")
    assert art["sizeBytes"] > 0
    assert len(art["sha256"]) == 64

    registered = list_run_artifacts(run_id=art["runId"])
    assert registered
    disk_file = Path(next(ref.path for ref in registered if ref.artifact_id == art["artifactId"]))
    assert disk_file.is_relative_to(tmp_path / ".aps_artifacts")
    assert disk_file.exists()
    raw_bytes = disk_file.read_bytes()
    assert signature.lower() in raw_bytes.lower()
    # Cross-run isolation: unrelated run cannot read tracked artifact
    assert read_tracked_artifact(str(disk_file), run_id="unrelated_run_id") is None


def test_http_graphql_advanced_pipeline_solo_vs_collaborative_differs(live_graphql_client, monkeypatch):
    client, _provider, ledger, _tmp_path = live_graphql_client
    # Nested team deliberation needs more than the conservative 40-call server default.
    # Keep a finite ceiling while the separate one-call regression proves hard enforcement.
    monkeypatch.setenv("APS_PROVIDER_MAX_CALLS", "200")
    task_prompt = (
        "Design a multi-stage verification strategy and risk analysis "
        "for an offline qualification runbook."
    )

    solo_resp = _post_graphql(
        client,
        RUN_EXECUTION_MUTATION,
        {
            "input": {
                "mode": "advanced_pipeline",
                "taskPrompt": task_prompt,
                "agents": 0,
                "maxSteps": 2,
                "initialBudget": 2,
            }
        },
    )
    assert solo_resp.status_code == 200
    solo_body = solo_resp.get_json()
    assert "errors" not in solo_body, solo_body
    solo_data = solo_body["data"]["runExecution"]
    assert solo_data["status"] == "completed"
    assert solo_data["effectiveMode"] == "advanced_pipeline"
    assert len(solo_data["diagnostics"]["planSteps"]) >= 1
    assert len(solo_data["diagnostics"]["reflections"]) >= 1
    assert ledger.total_calls > 0
    solo_calls = ledger.total_calls

    team_resp = _post_graphql(
        client,
        RUN_EXECUTION_MUTATION,
        {
            "input": {
                "mode": "advanced_pipeline",
                "taskPrompt": task_prompt,
                "agents": 2,
                "maxSteps": 2,
                "initialBudget": 2,
            }
        },
    )
    assert team_resp.status_code == 200
    team_body = team_resp.get_json()
    assert "errors" not in team_body, team_body
    team_data = team_body["data"]["runExecution"]
    assert team_data["status"] == "completed"
    team_calls = ledger.total_calls - solo_calls
    assert team_calls > solo_calls
    assert team_data["finalOutput"] != solo_data["finalOutput"]
    assert "Solo pipeline synthesis" not in team_data["finalOutput"]


def test_http_graphql_tree_of_thought_gated_zero_provider_calls(live_graphql_client):
    client, _provider, ledger, _tmp_path = live_graphql_client
    resp = _post_graphql(
        client,
        RUN_EXECUTION_MUTATION,
        {
            "input": {
                "mode": "tree_of_thought",
                "taskPrompt": "Explore tree branches.",
            }
        },
    )
    assert resp.status_code == 200
    data = resp.get_json()["data"]["runExecution"]
    assert data["status"] == "gated"
    assert data["stopReason"] == "gated"
    assert "Issue #36" in (data["errorMessage"] or "")
    assert ledger.total_calls == 0


def test_http_graphql_negative_provider_outage_and_all_error_voting(live_graphql_client):
    client, provider, ledger, _tmp_path = live_graphql_client

    # 1. All-error voting completes safely through mediator fallback without false success
    with scenario_scope("all_error_voting"):
        err_vote_resp = _post_graphql(
            client,
            RUN_EXECUTION_MUTATION,
            {
                "input": {
                    "mode": "collaborative",
                    "taskPrompt": "Handle voting errors safely.",
                    "assistantPersonalities": ["Strategist", "Mediator"],
                    "leadPersonality": "Strategist",
                    "numRounds": 1,
                }
            },
        )
    assert err_vote_resp.status_code == 200
    vote_data = err_vote_resp.get_json()["data"]["runExecution"]
    assert vote_data["status"] in {"completed", "error", "incomplete"}
    assert ledger.total_calls > 0

    # 2. Provider outage sanitizes secret-like substrings and reports honest error status
    with scenario_scope("provider_error"):
        outage_resp = _post_graphql(
            client,
            RUN_EXECUTION_MUTATION,
            {
                "input": {
                    "mode": "advanced_pipeline",
                    "taskPrompt": "Trigger provider outage.",
                    "agents": 0,
                    "maxSteps": 1,
                    "initialBudget": 1,
                }
            },
        )
    assert outage_resp.status_code == 200
    outage_data = outage_resp.get_json()["data"]["runExecution"]
    assert outage_data["status"] == "error"
    assert "sk-secret" not in json.dumps(outage_data)

    # 3. Empty model XML/response never becomes a fabricated completed answer.
    with scenario_scope("empty_model"):
        empty_resp = _post_graphql(
            client,
            RUN_EXECUTION_MUTATION,
            {
                "input": {
                    "mode": "advanced_pipeline",
                    "taskPrompt": "Trigger empty pipeline.",
                    "agents": 0,
                    "maxSteps": 1,
                    "initialBudget": 1,
                }
            },
        )
    assert empty_resp.status_code == 200
    empty_data = empty_resp.get_json()["data"]["runExecution"]
    assert empty_data["status"] in {"error", "incomplete", "budget_exhausted"}
    assert empty_data["finalOutput"] == ""
    assert not any(step["completed"] for step in empty_data["diagnostics"]["planSteps"])


@pytest.mark.parametrize("mutation_kind", ["runConversation", "runExecution"])
def test_http_public_mutations_share_provider_call_ceiling(
    live_graphql_client, monkeypatch, mutation_kind
):
    """The real deliberation's second SDK attempt must not reach the provider."""
    client, _provider, ledger, _tmp_path = live_graphql_client
    monkeypatch.setenv("APS_PROVIDER_MAX_CALLS", "1")

    body = _run_collaborative_mutation(client, mutation_kind)

    _assert_mutation_stopped(body, mutation_kind, "budget_exhausted")
    assert ledger.total_calls == 1
    assert ledger.entries[0].correlation_id


@pytest.mark.parametrize("mutation_kind", ["runConversation", "runExecution"])
def test_http_public_mutations_enforce_nondefault_model_on_every_real_call(
    live_graphql_client, monkeypatch, mutation_kind
):
    client, _provider, ledger, _tmp_path = live_graphql_client
    allowed_model = "offline-allowed-model"
    monkeypatch.setenv("APS_DEFAULT_MODEL", allowed_model)
    monkeypatch.setenv("APS_ALLOWED_MODELS", allowed_model)

    body = _run_collaborative_mutation(client, mutation_kind)

    assert "errors" not in body, body
    result = body["data"][mutation_kind]
    assert result["finalOutput"], result
    if mutation_kind == "runExecution":
        assert result["status"] == "completed", (result, ledger.total_calls)
    assert ledger.total_calls > 1
    assert {entry.model for entry in ledger.entries} == {allowed_model}
    assert len({entry.correlation_id for entry in ledger.entries}) == 1
    assert ledger.entries[0].correlation_id


@pytest.mark.parametrize("mutation_kind", ["runConversation", "runExecution"])
def test_http_public_mutations_reject_disallowed_models_before_provider_dispatch(
    live_graphql_client, monkeypatch, mutation_kind
):
    client, _provider, ledger, _tmp_path = live_graphql_client
    monkeypatch.setenv("APS_ALLOWED_MODELS", "offline-allowed-model")
    # Legacy clients use the configured default; unified clients can also request a model.
    monkeypatch.setenv("APS_DEFAULT_MODEL", "offline-forbidden-model")

    body = _run_collaborative_mutation(
        client, mutation_kind, model="offline-forbidden-model"
    )

    if mutation_kind == "runConversation":
        assert body.get("errors"), body
    else:
        result = body["data"]["runExecution"]
        assert result["status"] == "error", result
        assert result["stopReason"] == "disallowed_model", result
    assert ledger.total_calls == 0


@pytest.mark.parametrize("mutation_kind", ["runConversation", "runExecution"])
def test_http_public_mutations_stop_after_provider_exceeds_shared_deadline(
    live_graphql_client, monkeypatch, mutation_kind
):
    """A slow boundary fixture cannot replenish the deadline in nested helpers."""
    client, provider, ledger, _tmp_path = live_graphql_client
    provider.set_scenario("slow_delay")
    monkeypatch.setenv("APS_PROVIDER_TIMEOUT_SECONDS", "0.05")

    body = _run_collaborative_mutation(client, mutation_kind)

    _assert_mutation_stopped(body, mutation_kind, "timeout")
    assert ledger.total_calls == 1
    assert ledger.entries[0].correlation_id


@pytest.mark.parametrize("mutation_kind", ["runConversation", "runExecution"])
@pytest.mark.parametrize("filename_kind", ["absolute", "traversal"])
def test_http_public_mutations_confine_actual_model_selected_writes(
    live_graphql_client, mutation_kind, filename_kind
):
    """Script only the provider's tool arguments; invoke the real protected writer."""
    client, provider, ledger, tmp_path = live_graphql_client
    outside = tmp_path / "outside-run-sentinel.txt"
    sentinel = "EXISTING_OUTSIDE_FILE_MUST_NOT_BE_OVERWRITTEN"
    outside.write_text(sentinel, encoding="utf-8")
    provider.output_filename = (
        str(outside) if filename_kind == "absolute" else "../../outside-run-sentinel.txt"
    )

    body = _run_collaborative_mutation(client, mutation_kind)

    assert outside.read_text(encoding="utf-8") == sentinel
    assert ledger.total_calls > 1
    assert any("Final Decision:" in prompt for prompt in ledger.captured_prompts)
    if mutation_kind == "runConversation":
        assert body.get("errors"), body
    else:
        result = body["data"]["runExecution"]
        assert result["status"] != "completed", result
        assert result["artifacts"] == [], result
    assert sentinel not in json.dumps(body)


@pytest.mark.parametrize("mutation_kind", ["runConversation", "runExecution"])
def test_http_public_mutations_allocate_distinct_artifact_owners(live_graphql_client, mutation_kind):
    client, _provider, ledger, _tmp_path = live_graphql_client

    first = _run_collaborative_mutation(client, mutation_kind)
    assert "errors" not in first, first
    first_owners = {entry.correlation_id for entry in ledger.entries}
    first_call_count = ledger.total_calls
    second = _run_collaborative_mutation(client, mutation_kind)
    assert "errors" not in second, second
    second_owners = {
        entry.correlation_id for entry in ledger.entries[first_call_count:]
    }

    assert len(first_owners) == len(second_owners) == 1
    assert None not in first_owners | second_owners
    assert first_owners.isdisjoint(second_owners)
    first_owner = next(iter(first_owners))
    second_owner = next(iter(second_owners))
    first_refs = list_run_artifacts(run_id=first_owner)
    second_refs = list_run_artifacts(run_id=second_owner)
    assert first_refs and second_refs
    assert read_tracked_artifact(first_refs[0].path, run_id=first_owner)
    assert read_tracked_artifact(second_refs[0].path, run_id=second_owner)
    assert read_tracked_artifact(first_refs[0].path, run_id=second_owner) is None
    assert read_tracked_artifact(second_refs[0].path, run_id=first_owner) is None
    assert {ref.path for ref in first_refs}.isdisjoint({ref.path for ref in second_refs})
    if mutation_kind == "runExecution":
        first_artifacts = first["data"]["runExecution"]["artifacts"]
        second_artifacts = second["data"]["runExecution"]["artifacts"]
        assert first_artifacts and second_artifacts
        assert {a["runId"] for a in first_artifacts} == first_owners
        assert {a["runId"] for a in second_artifacts} == second_owners


@pytest.mark.parametrize("agents", [0, 2])
@pytest.mark.parametrize("max_steps", [1, 2])
def test_http_pipeline_requires_all_plan_steps_before_whole_task_success(
    live_graphql_client, monkeypatch, agents, max_steps
):
    """A first-step answer must survive as partial work without skipping step two."""
    client, _provider, ledger, _tmp_path = live_graphql_client
    monkeypatch.setenv("APS_PROVIDER_MAX_CALLS", "200")

    with scenario_scope("two_step"):
        response = _post_graphql(
            client,
            RUN_EXECUTION_MUTATION,
            {
                "input": {
                    "mode": "advanced_pipeline",
                    "taskPrompt": "Complete both independently required deliverables.",
                    "agents": agents,
                    "maxSteps": max_steps,
                    "initialBudget": max_steps,
                }
            },
        )

    assert response.status_code == 200
    body = response.get_json()
    assert "errors" not in body, body
    result = body["data"]["runExecution"]
    steps = result["diagnostics"]["planSteps"]
    assert len(steps) == 2, result
    assert ledger.total_calls > 0
    voting_calls = [entry for entry in ledger.entries if entry.response_format == "BinaryVote"]
    assert bool(voting_calls) is (agents > 0)
    assert steps[0]["completed"], result
    assert "FIRST_STEP_OUTPUT" in result["finalOutput"], result
    if max_steps == 1:
        assert result["status"] in {"budget_exhausted", "incomplete"}, result
        assert not steps[1]["completed"], result
        assert "SECOND_STEP_OUTPUT" not in result["finalOutput"], result
    else:
        assert result["status"] == "completed", (
            result,
            ledger.total_calls,
            dict(Counter(entry.response_format or entry.method for entry in ledger.entries)),
        )
        assert all(step["completed"] for step in steps), result
        assert all(sub["completed"] for step in steps for sub in step["subtasks"]), result
        assert "SECOND_STEP_OUTPUT" in result["finalOutput"], result


def test_http_nested_metadata_survives_both_selected_steps(live_graphql_client, monkeypatch):
    client, _provider, ledger, _tmp_path = live_graphql_client
    monkeypatch.setenv("APS_PROVIDER_MAX_CALLS", "200")
    query = RUN_EXECUTION_MUTATION.replace("conversation {", "conversation { planStepNumber").replace("questions {", "questions { planStepNumber")
    with scenario_scope("two_step_metadata"):
        response = _post_graphql(client, query, {"input": {"mode":"advanced_pipeline", "taskPrompt":"Complete two required deliverables", "agents":2, "maxSteps":2, "initialBudget":2}})
    body = response.get_json()
    assert response.status_code == 200 and "errors" not in body, body
    result = body["data"]["runExecution"]
    assert result["status"] == "completed", (result, ledger.total_calls, [(e.call_index,e.response_format) for e in ledger.entries[-4:]])
    questions = result["questions"]
    assert {q["planStepNumber"] for q in questions} == {1, 2}, questions
    for number in (1, 2):
        expected = f"STEP{number}_QUESTION"
        assert any(q["question"] == expected and q["planStepNumber"] == number for q in questions), questions
        assert any(f"STEP{number}_HISTORY" in message["content"] and message["planStepNumber"] == number for message in result["conversation"]), result
    assert 0 < ledger.total_calls <= 200


def test_http_rejected_nested_metadata_never_becomes_public(live_graphql_client, monkeypatch):
    client, _provider, ledger, _tmp_path = live_graphql_client
    monkeypatch.setenv("APS_PROVIDER_MAX_CALLS", "200")
    query = RUN_EXECUTION_MUTATION.replace("questions {", "questions { planStepNumber")
    with scenario_scope("two_step_metadata_rejected"):
        response = _post_graphql(client, query, {"input":{"mode":"advanced_pipeline", "taskPrompt":"Complete two required deliverables", "agents":2, "maxSteps":2, "initialBudget":2}})
    body = response.get_json()
    assert response.status_code == 200 and "errors" not in body, body
    result = body["data"]["runExecution"]
    assert result["status"] == "completed", result
    assert {q["planStepNumber"] for q in result["questions"]} == {1}, result
    assert "STEP1_HISTORY" in json.dumps(result)
    assert "STEP2_HISTORY" not in json.dumps(result)
    assert "STEP2_QUESTION" not in json.dumps(result)
    assert 0 < ledger.total_calls <= 200


def test_http_nested_limit_is_shared_after_first_step(live_graphql_client, monkeypatch):
    client, _provider, ledger, _tmp_path = live_graphql_client
    monkeypatch.setenv("APS_PROVIDER_MAX_CALLS", "200")
    query = RUN_EXECUTION_MUTATION.replace("questions {", "questions { planStepNumber")
    variables = {"input":{"mode":"advanced_pipeline", "taskPrompt":"Complete two required deliverables", "agents":2, "maxSteps":2, "initialBudget":2}}
    with scenario_scope("two_step_metadata"):
        success = _post_graphql(client, query, variables).get_json()["data"]["runExecution"]
    assert success["status"] == "completed", success
    nested_second = next(entry.call_index for entry, prompt in zip(ledger.entries, ledger.captured_prompts)
                         if entry.response_format == "ExtractFormat" and "SECOND_STEP_OUTPUT" in prompt)
    ceiling = nested_second + 1
    assert ceiling < ledger.total_calls
    ledger.reset()
    monkeypatch.setenv("APS_PROVIDER_MAX_CALLS", str(ceiling))
    with scenario_scope("two_step_metadata"):
        response = _post_graphql(client, query, variables)
    result = response.get_json()["data"]["runExecution"]
    assert result["status"] == "budget_exhausted", result
    assert result["stopReason"] == "budget_exhausted", result
    assert ledger.total_calls == ceiling
    assert len({entry.correlation_id for entry in ledger.entries}) == 1
    assert result["diagnostics"]["planSteps"][0]["completed"]
    assert not result["diagnostics"]["planSteps"][1]["completed"]
    assert {q["planStepNumber"] for q in result["questions"]} == {1}


def test_http_nested_failure_preserves_prior_selected_metadata(live_graphql_client, monkeypatch):
    client, _provider, ledger, _tmp_path = live_graphql_client
    monkeypatch.setenv("APS_PROVIDER_MAX_CALLS", "200")
    query = RUN_EXECUTION_MUTATION.replace("questions {", "questions { planStepNumber")
    with scenario_scope("two_step_metadata_failure"):
        response = _post_graphql(client, query, {"input":{"mode":"advanced_pipeline", "taskPrompt":"Complete two required deliverables", "agents":2, "maxSteps":2, "initialBudget":2}})
    body = response.get_json()
    assert response.status_code == 200 and "errors" not in body, body
    result = body["data"]["runExecution"]
    assert result["status"] == "error", result
    assert result["stopReason"] == "conversation_error", result
    assert result["diagnostics"]["planSteps"][0]["completed"]
    assert not result["diagnostics"]["planSteps"][1]["completed"]
    assert {q["planStepNumber"] for q in result["questions"]} == {1}
    assert "STEP1_HISTORY" in json.dumps(result)
    assert result["diagnostics"]["finalReward"] is None
    assert 0 < ledger.total_calls < 200


def test_http_nested_writer_cannot_overwrite_foreign_registered_artifact(live_graphql_client, monkeypatch):
    from output_generator import artifact_run_scope, generate_text_file_output
    import conversation_manager as cm
    client, provider, ledger, tmp_path = live_graphql_client
    monkeypatch.setenv("APS_PROVIDER_MAX_CALLS", "200")
    with artifact_run_scope("foreign-owner"):
        path = generate_text_file_output("FOREIGN_CONTENT_SENTINEL", "foreign.txt")
        output = cm.artifact_conversation_output(path)
        assert output.kind == "artifact" and output.content == "FOREIGN_CONTENT_SENTINEL"
    provider.output_filename = path
    response = _post_graphql(client, RUN_EXECUTION_MUTATION, {"input":{"mode":"advanced_pipeline", "taskPrompt":"[FORMAT:TXT] Design a strategy and analysis", "agents":2, "maxSteps":1, "initialBudget":1}})
    body = response.get_json()
    assert response.status_code == 200 and "errors" not in body, body
    serialized = json.dumps(body)
    assert "FOREIGN_CONTENT_SENTINEL" not in serialized
    assert str(tmp_path) not in serialized
    assert Path(path).read_text() == "FOREIGN_CONTENT_SENTINEL"
    assert any(entry.response_format == "ExtractFormat" for entry in ledger.entries)
    assert any("generate the output in the following format" in prompt for prompt in ledger.captured_prompts)
    assert ledger.total_calls <= 200
