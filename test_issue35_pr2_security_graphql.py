"""PR 2 Security, GraphQL Contract, and Feedback Manager Regression Tests for Issue #35."""

from __future__ import annotations

import json
import os
from unittest.mock import patch

import pytest

import app as app_module
import feedback_manager
from execution_modes import CAPABILITY_REGISTRY, OutputType, PERSONALITY_CATALOG, RunResult
from output_generator import ArtifactRef


def test_app_healthz_and_readyz(tmp_path):
    """Verify /healthz, /readyz, and missing-build 404 SPA fallback without stack traces."""
    flask_app = app_module.create_app({"STATIC_FOLDER": str(tmp_path / "missing_dist")})
    client = flask_app.test_client()

    health_resp = client.get("/healthz")
    assert health_resp.status_code == 200
    assert health_resp.get_json() == {"status": "ok"}
    assert health_resp.headers.get("X-Content-Type-Options") == "nosniff"
    assert health_resp.headers.get("X-Frame-Options") == "DENY"

    ready_resp = client.get("/readyz")
    assert ready_resp.status_code == 200
    ready_payload = ready_resp.get_json()
    assert ready_payload["status"] == "ready"
    assert ready_payload["provider_configured"] is False
    assert ready_payload["static_bundle_available"] is False

    root_resp = client.get("/")
    assert root_resp.status_code == 404
    assert root_resp.get_json()["error"] == "Frontend build is not available."


def test_app_spa_static_serving_and_traversal_blocked(tmp_path):
    """Verify SPA serves index.html, blocks path traversal, and 404s missing assets."""
    dist_dir = tmp_path / "dist"
    assets_dir = dist_dir / "assets"
    assets_dir.mkdir(parents=True)
    (dist_dir / "index.html").write_text("<html><body>SPA</body></html>", encoding="utf-8")
    (assets_dir / "app.js").write_text("console.log('ok');", encoding="utf-8")

    flask_app = app_module.create_app({"STATIC_FOLDER": str(dist_dir)})
    client = flask_app.test_client()

    assert client.get("/").status_code == 200
    assert b"SPA" in client.get("/").data
    assert client.get("/assets/app.js").status_code == 200
    assert client.get("/assets/missing.js").status_code == 404
    # Client-side route falls back cleanly to index.html
    assert client.get("/workspace/session-1").status_code == 200
    # Path traversal outside static folder must fail
    traversal = client.get("/../../secret.txt")
    assert traversal.status_code == 404


def test_app_graphql_security_guards(tmp_path):
    """Verify Content-Type 415, Origin 403, Bearer 401, Max Body 413, Rate Limit 429, GraphiQL 405."""
    flask_app = app_module.create_app(
        {
            "STATIC_FOLDER": str(tmp_path),
            "MAX_CONTENT_LENGTH": 256,
            "RATE_LIMIT_RPM": 2,
            "API_TOKEN": "secret-test-token",
            "ALLOWED_ORIGINS": ["http://localhost:5173"],
        }
    )
    client = flask_app.test_client()

    # Unknown Origin -> 403
    bad_origin = client.post(
        "/graphql",
        data=json.dumps({"query": "{ hello }"}),
        content_type="application/json",
        headers={"Origin": "https://evil.example", "Authorization": "Bearer secret-test-token"},
    )
    assert bad_origin.status_code == 403

    # Missing Bearer Token -> 401
    missing_auth = client.post(
        "/graphql",
        data=json.dumps({"query": "{ hello }"}),
        content_type="application/json",
        headers={"Origin": "http://localhost:5173"},
    )
    assert missing_auth.status_code == 401

    # Invalid Bearer Token -> 401
    wrong_auth = client.post(
        "/graphql",
        data=json.dumps({"query": "{ hello }"}),
        content_type="application/json",
        headers={"Origin": "http://localhost:5173", "Authorization": "Bearer wrong-token"},
    )
    assert wrong_auth.status_code == 401

    # CORS OPTIONS preflight -> 204
    preflight = client.options("/graphql", headers={"Origin": "http://localhost:5173"})
    assert preflight.status_code == 204
    assert preflight.headers.get("Access-Control-Allow-Origin") == "http://localhost:5173"
    assert "POST" in (preflight.headers.get("Access-Control-Allow-Methods") or "")

    # Spoofed Host + external Origin without explicit ALLOWED_ORIGINS -> 403
    default_app = app_module.create_app({"STATIC_FOLDER": str(tmp_path)})
    spoof_resp = default_app.test_client().post(
        "/graphql",
        json={"query": "{ hello }"},
        headers={"Host": "evil.example", "Origin": "http://evil.example"},
    )
    assert spoof_resp.status_code == 403

    # Wrong Content-Type -> 415
    bad_ct = client.post(
        "/graphql",
        data="{ hello }",
        content_type="text/plain",
        headers={"Origin": "http://localhost:5173", "Authorization": "Bearer secret-test-token"},
    )
    assert bad_ct.status_code == 415

    # Payload too large -> 413
    oversized = client.post(
        "/graphql",
        data=json.dumps({"query": "{ hello }", "extra": "x" * 400}),
        content_type="application/json",
        headers={"Origin": "http://localhost:5173", "Authorization": "Bearer secret-test-token"},
    )
    assert oversized.status_code == 413

    # GraphiQL disabled on GET -> 405
    get_gql = client.get(
        "/graphql",
        headers={"Origin": "http://localhost:5173", "Authorization": "Bearer secret-test-token"},
    )
    assert get_gql.status_code == 405

    # Rate limit RPM=2 applies across GET ?query=... and POST requests -> 3rd returns 429
    headers = {"Origin": "http://localhost:5173", "Authorization": "Bearer secret-test-token"}
    r1 = client.post("/graphql", json={"query": "{ hello }"}, headers=headers)
    r2 = client.get("/graphql", query_string={"query": "{ hello }"}, headers=headers)
    r3 = client.post("/graphql", json={"query": "{ hello }"}, headers=headers)
    assert r1.status_code == 200
    assert r1.get_json()["data"] == {"hello": "Hello, World!"}
    assert r1.headers.get("Access-Control-Allow-Origin") == "http://localhost:5173"
    assert r2.status_code == 200
    assert r3.status_code == 429
    assert "Retry-After" in r3.headers


def test_graphql_personalities_and_extended_execution_capabilities():
    """Verify Query.personalities, Query.providerPolicy, and Query.executionModes readiness."""
    query = """
    query Capabilities {
      personalities {
        id
        instructions
      }
      providerPolicy {
        defaultModel
        allowedModels
        defaultTemperature
      }
      executionModes {
        id
        enabled
        ready
        unavailableReason
        supportsCancellation
        supportsResume
      }
    }
    """
    result = app_module.schema.execute_sync(query)
    assert result.errors is None
    assert result.data is not None

    returned_ids = [p["id"] for p in result.data["personalities"]]
    assert returned_ids == list(PERSONALITY_CATALOG.keys())
    assert len(returned_ids) == 26

    policy = result.data["providerPolicy"]
    assert policy["defaultModel"] in policy["allowedModels"]

    modes = {m["id"]: m for m in result.data["executionModes"]}
    assert set(modes.keys()) == set(CAPABILITY_REGISTRY.keys())
    assert modes["tree_of_thought"]["enabled"] is False
    assert modes["tree_of_thought"]["ready"] is False
    # Offline test fixture forbids unconfigured provider -> ready is False with reason
    assert modes["collaborative"]["enabled"] is True
    assert modes["collaborative"]["ready"] is False
    assert "credentials" in (modes["collaborative"]["unavailableReason"] or "").lower()


def test_graphql_run_execution_exposes_artifacts_and_sanitizes_validation_errors():
    """Verify RunExecutionResponse artifacts/stopReason mapping and clean error sanitization."""
    mutation = """
    mutation RunExec($input: RunExecutionInput!) {
      runExecution(input: $input) {
        mode
        effectiveMode
        status
        stopReason
        finalOutput
        errorMessage
        usageAvailable
        artifacts {
          artifactId
          runId
          filename
          path
          mimeType
          sizeBytes
          sha256
        }
      }
    }
    """
    # Disallowed model is returned cleanly as status="error" without stack traces or secret leaks
    bad_result = app_module.schema.execute_sync(
        mutation,
        variable_values={
            "input": {
                "mode": "collaborative",
                "taskPrompt": "Test prompt",
                "model": "forbidden-model-xyz",
            }
        },
    )
    assert bad_result.errors is None
    payload = bad_result.data["runExecution"]
    assert payload["status"] == "error"
    assert payload["stopReason"] == "disallowed_model"
    assert "not permitted" in payload["errorMessage"].lower()

    mocked = RunResult(
        mode="advanced_pipeline",
        effective_mode="advanced_pipeline",
        status="completed",
        stop_reason="final_answer",
        final_output="Generated output",
        output_type=OutputType(output_type="JSON", file_extension=".json"),
        artifacts=[
            ArtifactRef(
                artifact_id="art-1",
                run_id="run-123",
                filename="report.json",
                path="artifacts/run-123/report.json",
                mime_type="application/json",
                size_bytes=42,
                sha256="abc123",
            )
        ],
        usage={"available": False},
    )
    with patch("execution_modes.execute_run", return_value=mocked):
        ok_result = app_module.schema.execute_sync(
            mutation,
            variable_values={"input": {"mode": "advanced_pipeline", "taskPrompt": "Generate report"}},
        )
    assert ok_result.errors is None
    ok_payload = ok_result.data["runExecution"]
    assert ok_payload["effectiveMode"] == "advanced_pipeline"
    assert ok_payload["stopReason"] == "final_answer"
    assert ok_payload["usageAvailable"] is False
    assert len(ok_payload["artifacts"]) == 1
    assert ok_payload["artifacts"][0]["filename"] == "report.json"
    # Relative run-scoped path only (never host filesystem root)
    assert ok_payload["artifacts"][0]["path"] == "artifacts/run-123/report.json"


def test_feedback_manager_lazy_creation_and_rating_validation(tmp_path, monkeypatch):
    """Verify feedback_manager creates log files lazily, validates bounds, and recovers corrupt logs."""
    # Ensure importing feedback_manager never eagerly creates LOG_DIR inside clean cwd
    assert not (tmp_path / "logs").exists()

    log_target = tmp_path / "nested" / "feedback.json"
    monkeypatch.setenv("APS_LOG_FILE", str(log_target))

    submit_query = """
    mutation Submit($conversation: [ConversationInput!]!, $feedback: [FeedbackInput!]!) {
      submitFeedback(conversation: $conversation, feedback: $feedback) {
        success
      }
    }
    """
    # Rating outside 1..5 rejected via GraphQL submitFeedback (above and below bounds)
    for bad_rating in (0, 9):
        invalid_res = app_module.schema.execute_sync(
            submit_query,
            variable_values={
                "conversation": [{"role": "user", "name": "User", "content": "Hi"}],
                "feedback": [{"assistant": "Mediator", "rating": bad_rating}],
            },
        )
        assert invalid_res.errors is not None
        assert "between 1 and 5" in invalid_res.errors[0].message

    # Boolean rating rejected directly in feedback_manager
    with pytest.raises(ValueError, match="between 1 and 5"):
        feedback_manager.log_conversation(
            [{"role": "user", "content": "Hi"}],
            [{"assistant": "Mediator", "rating": True}],
        )

    # Empty conversation and oversized comments rejected
    with pytest.raises(ValueError, match="non-empty"):
        feedback_manager.log_conversation([], [{"assistant": "Mediator", "rating": 5}])
    with pytest.raises(ValueError, match="comments exceed"):
        feedback_manager.log_conversation(
            [{"role": "user", "content": "Hi"}],
            [{"assistant": "Mediator", "rating": 5, "comments": "x" * 5001}],
        )

    # Corrupt JSON on disk recovers cleanly rather than throwing 500
    log_target.parent.mkdir(parents=True, exist_ok=True)
    log_target.write_text("{corrupt-json", encoding="utf-8")

    valid_res = app_module.schema.execute_sync(
        submit_query,
        variable_values={
            "conversation": [{"role": "user", "name": "User", "content": "Hi"}],
            "feedback": [{"assistant": "Mediator", "rating": 5, "comments": "Great summary"}],
        },
    )
    assert valid_res.errors is None
    assert valid_res.data == {"submitFeedback": {"success": "Feedback submitted successfully."}}

    assert os.path.isfile(log_target)
    saved = json.loads(log_target.read_text(encoding="utf-8"))
    assert len(saved) == 1
    assert saved[0]["feedback"][0]["rating"] == 5
    assert saved[0]["feedback"][0]["comments"] == "Great summary"
    assert "recorded_at" in saved[0]
