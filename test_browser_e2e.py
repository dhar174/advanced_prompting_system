"""Offline real-browser Playwright end-to-end verification tests.

Exercises the built React/Vite SPA served by Flask (`OfflineE2EServer`) against
the real Strawberry GraphQL API, real orchestration (`conversation_manager` and
`advanced_prompting`), real filesystem artifact generation, and the deterministic
`FakeProviderClient` with fail-closed loopback network enforcement inside Chromium.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterator
from urllib.parse import urlparse

import pytest
from playwright.sync_api import ConsoleMessage, Page, Route, expect, sync_playwright

from output_generator import read_tracked_artifact
from tools.e2e.server_runner import OfflineE2EServer

_REPO_ROOT = Path(__file__).resolve().parent
_FRONTEND_DIST = _REPO_ROOT / "collaborative-assistant-frontend" / "dist"


def _ensure_frontend_dist() -> None:
    if not (_FRONTEND_DIST / "index.html").is_file():
        pytest.fail(
            f"Built frontend bundle missing at {_FRONTEND_DIST}. "
            "Run ./tools/run-frontend-offline.ps1 -Action build before Playwright E2E tests."
        )


def _is_allowed_browser_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme in {"data", "blob", "about"}:
        return True
    if parsed.scheme in {"http", "https", "ws", "wss"}:
        host = (parsed.hostname or "").lower()
        return host in {"127.0.0.1", "localhost", "::1"}
    return False


def _attach_loopback_network_guard(page: Page, blocked_urls: list[str]) -> None:
    def _handle_route(route: Route) -> None:
        req_url = route.request.url
        if _is_allowed_browser_url(req_url):
            route.continue_()
        else:
            blocked_urls.append(req_url)
            route.abort("blockedbyclient")

    page.route("**/*", _handle_route)


@pytest.fixture
def e2e_server(tmp_path: Path) -> Iterator[OfflineE2EServer]:
    _ensure_frontend_dist()
    with OfflineE2EServer(runtime_dir=tmp_path) as server:
        yield server
        ledger_dir = _REPO_ROOT / ".qualification" / "logs"
        ledger_dir.mkdir(parents=True, exist_ok=True)
        server.ledger.save_json(ledger_dir / "browser_e2e_latest_ledger.json")


@pytest.fixture
def browser_page(e2e_server: OfflineE2EServer):
    blocked_urls: list[str] = []
    page_errors: list[str] = []
    console_errors: list[str] = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1366, "height": 900})
        page = context.new_page()
        _attach_loopback_network_guard(page, blocked_urls)
        page.on("pageerror", lambda exc: page_errors.append(str(exc)))

        def _on_console(msg: ConsoleMessage) -> None:
            if msg.type == "error":
                console_errors.append(msg.text)

        page.on("console", _on_console)
        page.goto(e2e_server.base_url, wait_until="networkidle")
        yield page, e2e_server, blocked_urls, page_errors, console_errors
        context.close()
        browser.close()


def test_browser_e2e_collaborative_pdf_questions_followup_and_feedback(browser_page):
    page, server, blocked_urls, page_errors, console_errors = browser_page

    # Wait for catalog queries (executionModes + personalities) to hydrate UI
    expect(page.locator("#execution-mode")).to_have_value("collaborative")
    strategist_cb = page.locator("#personality-Strategist")
    strategist_cb.wait_for(state="visible", timeout=10000)

    analytical_cb = page.locator('[id="personality-Analytical Thinker"]')
    if analytical_cb.is_checked():
        analytical_cb.uncheck()
    if not strategist_cb.is_checked():
        strategist_cb.check()
    mediator_cb = page.locator("#personality-Mediator")
    if not mediator_cb.is_checked():
        mediator_cb.check()

    page.locator("#lead-personality").select_option("Strategist")
    page.locator("#num-rounds").fill("1")
    page.get_by_role("button", name="Start New Conversation").click()

    prompt_box = page.locator('textarea[name="prompt"]')
    prompt_box.fill("[FORMAT:PDF] Deliver a collaborative offline verification report.")
    page.get_by_test_id("send-message-button").click()

    expect(page.get_by_text("Clarifying Questions Raised (Read-Only):")).to_be_visible(timeout=20000)
    expect(page.get_by_role("heading", name="Final Output:")).to_be_visible(timeout=20000)
    expect(page.get_by_text("Generated Run Artifacts (1)")).to_be_visible(timeout=20000)
    expect(page.get_by_text("deliverable.pdf", exact=True)).to_be_visible()

    assert server.ledger.total_calls > 0
    assert all(
        "Welcome! Configure your assistants" not in prompt
        and "Conversation started with" not in prompt
        for prompt in server.ledger.captured_prompts
    )

    artifact_files = list((server.runtime_dir / ".aps_artifacts").glob("run_*/*.pdf"))
    assert len(artifact_files) == 1
    pdf_bytes = artifact_files[0].read_bytes()
    assert pdf_bytes.startswith(b"%PDF-1.")

    # Post-completion contract: sending without starting a new session prompts user cleanly
    calls_after_turn1 = server.ledger.total_calls
    prompt_box.fill("[FORMAT:TXT] Follow-up without restart.")
    page.get_by_test_id("send-message-button").click()
    expect(page.get_by_text("Please start a new conversation first.")).to_be_visible(timeout=5000)
    assert server.ledger.total_calls == calls_after_turn1

    # Starting a new session runs cleanly and delivers a fresh TXT artifact
    page.get_by_role("button", name="Start New Conversation").click()
    prompt_box.fill("[FORMAT:TXT] Follow-up: add smoke verification steps.")
    page.get_by_test_id("send-message-button").click()
    expect(page.get_by_text("deliverable.txt", exact=True)).to_be_visible(timeout=20000)
    assert server.ledger.total_calls > calls_after_turn1

    # Multi-turn follow-up GraphQL payload propagates Prior Conversation Context to provider
    multi_turn = page.evaluate(
        """async () => {
            const resp = await fetch('/graphql', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query: `mutation RunExec($input: RunExecutionInput!) {
                        runExecution(input: $input) {
                            status
                            outputType { fileExtension }
                            finalOutput
                        }
                    }`,
                    variables: {
                        input: {
                            mode: 'collaborative',
                            taskPrompt: '[FORMAT:TXT] Finalize checklist',
                            conversation: [
                                { role: 'user', name: 'User', content: 'Initial draft scope' },
                                { role: 'assistant', name: 'Mediator', content: 'Drafted checklist v1' },
                                { role: 'user', name: 'User', content: '[FORMAT:TXT] Finalize checklist' }
                            ],
                            assistantPersonalities: ['Strategist', 'Mediator'],
                            leadPersonality: 'Strategist',
                            numRounds: 1
                        }
                    }
                })
            });
            return await resp.json();
        }"""
    )
    assert multi_turn["data"]["runExecution"]["status"] == "completed"
    assert any("Prior Conversation Context" in p for p in server.ledger.captured_prompts)

    # Submit feedback through /graphql from browser context and verify JSON log persistence
    fb_result = page.evaluate(
        """async () => {
            const resp = await fetch('/graphql', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query: `mutation SubmitFeedback($conversation: [ConversationInput!]!, $feedback: [FeedbackInput!]!) {
                        submitFeedback(conversation: $conversation, feedback: $feedback) { success }
                    }`,
                    variables: {
                        conversation: [
                            { role: 'user', name: 'User', content: '[FORMAT:PDF] Deliver verification report' },
                            { role: 'assistant', name: 'Strategist', content: 'Verified consensus deliverable.' }
                        ],
                        feedback: [
                            {
                                assistant: 'Strategist',
                                rating: 5,
                                comments: 'Browser E2E feedback verified'
                            }
                        ]
                    }
                })
            });
            return await resp.json();
        }"""
    )
    assert fb_result["data"]["submitFeedback"]["success"] == "Feedback submitted successfully."
    feedback_log = server.runtime_dir / "logs" / "conversation_logs.json"
    assert feedback_log.is_file()
    saved_entries = json.loads(feedback_log.read_text(encoding="utf-8"))
    assert any(
        any(fb.get("comments") == "Browser E2E feedback verified" for fb in entry.get("feedback", []))
        for entry in saved_entries
    )

    assert blocked_urls == []
    assert page_errors == []
    assert console_errors == []


def test_browser_e2e_advanced_pipeline_solo_vs_collaborative_and_tabs(browser_page):
    page, server, blocked_urls, page_errors, console_errors = browser_page

    mode_select = page.locator("#execution-mode")
    mode_select.wait_for(state="visible", timeout=10000)
    mode_select.select_option("advanced_pipeline")

    page.locator("#max-steps").fill("2")
    page.locator("#agent-count").fill("0")
    page.get_by_role("button", name="Start Pipeline Session").click()

    task_prompt = (
        "Design a multi-stage verification strategy and risk analysis "
        "for an offline qualification runbook."
    )
    page.locator('textarea[name="prompt"]').fill(task_prompt)
    page.get_by_test_id("send-message-button").click()

    expect(page.get_by_role("heading", name="Final Output:")).to_be_visible(timeout=25000)
    expect(page.get_by_text("Solo pipeline synthesis").first).to_be_visible()
    expect(page.get_by_text("Reasoning Diagnostics & Telemetry")).to_be_visible()

    # Exercise every diagnostics tab in the rendered UI
    page.get_by_role("tab", name="Steps & Reflections").click()
    page.get_by_role("tab", name="Agent Attributions").click()
    page.get_by_role("tab", name="Event Stream").click()
    page.get_by_role("tab", name="Execution Plan").click()

    solo_calls = server.ledger.total_calls
    assert solo_calls > 0

    # Collaborative multi-agent step run (agents=2)
    page.locator("#agent-count").fill("2")
    page.get_by_role("button", name="Start Pipeline Session").click()
    page.locator('textarea[name="prompt"]').fill(task_prompt)
    page.get_by_test_id("send-message-button").click()

    expect(page.get_by_role("heading", name="Final Output:")).to_be_visible(timeout=25000)
    expect(page.get_by_text("Solo pipeline synthesis")).to_have_count(0)
    assert server.ledger.total_calls - solo_calls > solo_calls

    assert blocked_urls == []
    assert page_errors == []
    assert console_errors == []


def test_browser_e2e_tree_of_thought_disabled_in_ui_zero_provider_calls(browser_page):
    page, server, blocked_urls, page_errors, console_errors = browser_page

    tot_option = page.locator('#execution-mode option[value="tree_of_thought"]')
    tot_option.wait_for(state="attached", timeout=10000)
    assert tot_option.is_disabled()
    option_text = tot_option.inner_text()
    assert "Disabled" in option_text
    assert "Issue #36" in option_text

    assert server.ledger.total_calls == 0
    assert blocked_urls == []
    assert page_errors == []
    assert console_errors == []


def test_browser_e2e_negative_xss_escaping_and_error_handling(browser_page):
    page, _server, blocked_urls, page_errors, _console_errors = browser_page

    active_scenario = {"value": "xss_output"}

    def _inject_scenario(route: Route) -> None:
        if route.request.method == "POST" and route.request.url.endswith("/graphql"):
            headers = dict(route.request.headers)
            headers["x-aps-e2e-scenario"] = active_scenario["value"]
            route.continue_(headers=headers)
        else:
            route.continue_()

    page.route("**/graphql", _inject_scenario)

    # 1. XSS escaping check: untrusted output must never execute in Chromium DOM
    page.locator("#num-rounds").fill("1")
    page.get_by_role("button", name="Start New Conversation").click()
    page.locator('textarea[name="prompt"]').fill("Return escaped output safely.")
    page.get_by_test_id("send-message-button").click()

    expect(page.get_by_role("heading", name="Final Output:")).to_be_visible(timeout=20000)
    assert page.evaluate("() => window.__xss_executed === true") is False
    assert "Escaped response verified." in page.locator("body").inner_text()

    # 2. Provider outage must sanitize secrets, show failure alert, and NOT render Final Output
    active_scenario["value"] = "provider_error"
    page.locator("#execution-mode").select_option("advanced_pipeline")
    page.locator("#max-steps").fill("1")
    page.locator("#agent-count").fill("0")
    page.get_by_role("button", name="Start Pipeline Session").click()
    page.locator('textarea[name="prompt"]').fill("Trigger upstream provider outage.")
    page.get_by_test_id("send-message-button").click()

    alert = page.get_by_role("alert")
    expect(alert).to_be_visible(timeout=15000)
    expect(alert).to_contain_text("Execution Failed")
    assert "sk-secret" not in page.locator("body").inner_text()
    expect(page.get_by_role("heading", name="Final Output:")).not_to_be_visible()

    # 3. Empty model response triggers honest error instead of fake completion
    active_scenario["value"] = "empty_model"
    page.get_by_role("button", name="Start Pipeline Session").click()
    page.locator('textarea[name="prompt"]').fill("Trigger empty model failure.")
    page.get_by_test_id("send-message-button").click()

    expect(page.get_by_role("alert")).to_be_visible(timeout=15000)
    expect(page.get_by_role("alert")).to_contain_text("Execution Failed")
    expect(page.get_by_role("heading", name="Final Output:")).not_to_be_visible()

    assert blocked_urls == []
    assert page_errors == []


def test_browser_e2e_negative_double_submit_and_abort_wait_isolation(browser_page):
    page, server, blocked_urls, page_errors, _console_errors = browser_page

    graphql_post_count = 0

    def _slow_graphql(route: Route) -> None:
        nonlocal graphql_post_count
        if route.request.method == "POST" and route.request.url.endswith("/graphql"):
            post_body = route.request.post_data or ""
            if "runExecution" in post_body or "RunExecution" in post_body:
                graphql_post_count += 1
            headers = dict(route.request.headers)
            headers["x-aps-e2e-scenario"] = "slow_delay"
            route.continue_(headers=headers)
        else:
            route.continue_()

    page.route("**/graphql", _slow_graphql)

    page.locator("#execution-mode").select_option("advanced_pipeline")
    page.locator("#max-steps").fill("1")
    page.locator("#agent-count").fill("0")
    page.get_by_role("button", name="Start Pipeline Session").click()

    draft_text = "Slow execution test draft must remain recoverable."
    prompt_box = page.locator('textarea[name="prompt"]')
    prompt_box.fill(draft_text)

    # Rapid double-click must send at most 1 in-flight RunExecution GraphQL request
    send_btn = page.get_by_test_id("send-message-button")
    send_btn.click()
    if send_btn.is_enabled():
        send_btn.click(force=True)

    cancel_btn = page.get_by_test_id("cancel-execution-button")
    expect(cancel_btn).to_be_visible(timeout=5000)
    expect(cancel_btn).to_have_text("Abort Wait (Server Continues)")
    cancel_btn.click()

    expect(page.get_by_text("Execution Cancelled")).to_be_visible(timeout=5000)
    expect(prompt_box).to_have_value(draft_text)

    # Even after the slow server request completes, cancelled UI state must stay isolated
    page.wait_for_timeout(1200)
    expect(page.get_by_role("heading", name="Final Output:")).not_to_be_visible()
    expect(page.get_by_text("Execution Cancelled")).to_be_visible()
    assert graphql_post_count == 1
    assert server.ledger.total_calls >= 1

    assert blocked_urls == []
    assert page_errors == []


def test_negative_cors_and_cross_run_artifact_isolation(e2e_server: OfflineE2EServer, tmp_path: Path):
    req = urllib.request.Request(
        f"{e2e_server.base_url}/graphql",
        data=json.dumps({"query": "{ __typename }"}).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Origin": "https://untrusted-external.example",
        },
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req, timeout=5)
    assert exc_info.value.code == 403

    # Cross-run artifact isolation against non-existent/foreign run id
    sample_file = tmp_path / ".aps_artifacts" / "run_owner" / "secret.txt"
    sample_file.parent.mkdir(parents=True, exist_ok=True)
    sample_file.write_text("owner only", encoding="utf-8")
    assert read_tracked_artifact(str(sample_file), run_id="other_run") is None
