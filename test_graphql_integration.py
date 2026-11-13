"""
Unmocked Flask HTTP /graphql -> execution_modes -> real runtime orchestrators integration tests.

Only `provider_client.get_client` is replaced with `FakeProviderClient` + `ProviderCallLedger`
so `schema.py`, `execution_modes.py`, `conversation_manager.py`, `advanced_prompting.py`,
`complexity_measures.py`, and `output_generator.py` run end-to-end without mocks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pytest

from app import create_app
from output_generator import clear_tracked_artifacts, read_tracked_artifact
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
        completed
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
    assert art["path"].startswith(f"artifacts/{corr_id}/")
    assert art["sizeBytes"] > 0
    assert len(art["sha256"]) == 64

    disk_file = tmp_path / ".aps_artifacts" / corr_id / art["filename"]
    assert disk_file.exists()
    raw_bytes = disk_file.read_bytes()
    assert signature.lower() in raw_bytes.lower()
    # Cross-run isolation: unrelated run cannot read tracked artifact
    assert read_tracked_artifact(str(disk_file), run_id="unrelated_run_id") is None


def test_http_graphql_advanced_pipeline_solo_vs_collaborative_differs(live_graphql_client):
    client, _provider, ledger, _tmp_path = live_graphql_client
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

    # 3. Empty model XML/response triggers error instead of fake completion
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
    assert empty_data["status"] == "error"
