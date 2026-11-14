"""
Deterministic offline LLM provider fixture & call ledger for integration and Playwright E2E tests.

Strictly test/qualification tooling — never imported by production modules unless explicitly
installed via `install_fake_provider` or `server_runner`.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import os
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional, Tuple
from output_generator import get_active_run_id

_ACTIVE_SCENARIO: contextvars.ContextVar[str] = contextvars.ContextVar(
    "aps_e2e_provider_scenario", default="default"
)
_ACTIVE_CORRELATION_ID: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "aps_e2e_correlation_id", default=None
)


@dataclass
class ProviderCallEntry:
    call_index: int
    method: str
    model: str
    response_format: Optional[str]
    correlation_id: Optional[str]
    scenario: str
    prompt_preview: str
    error: Optional[str] = None


class ProviderCallLedger:
    """Thread-safe ledger recording every call dispatched through FakeProviderClient."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._entries: List[ProviderCallEntry] = []
        self._captured_prompts: List[str] = []

    def record(
        self,
        *,
        method: str,
        model: str,
        response_format: Optional[str],
        correlation_id: Optional[str],
        scenario: str,
        prompt_preview: str,
        full_prompt: str = "",
        error: Optional[str] = None,
    ) -> ProviderCallEntry:
        with self._lock:
            entry = ProviderCallEntry(
                call_index=len(self._entries) + 1,
                method=method,
                model=model,
                response_format=response_format,
                correlation_id=correlation_id,
                scenario=scenario,
                prompt_preview=prompt_preview[:240],
                error=error,
            )
            self._entries.append(entry)
            if full_prompt:
                self._captured_prompts.append(full_prompt)
            return entry

    @property
    def entries(self) -> List[ProviderCallEntry]:
        with self._lock:
            return list(self._entries)

    @property
    def total_calls(self) -> int:
        with self._lock:
            return len(self._entries)

    @property
    def captured_prompts(self) -> List[str]:
        with self._lock:
            return list(self._captured_prompts)

    def calls_for_correlation(self, correlation_id: str) -> List[ProviderCallEntry]:
        with self._lock:
            return [e for e in self._entries if e.correlation_id == correlation_id]

    def reset(self) -> None:
        with self._lock:
            self._entries.clear()
            self._captured_prompts.clear()

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "total_calls": len(self._entries),
                "entries": [asdict(e) for e in self._entries],
            }

    def save_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return target


def _flatten_messages(messages: Any) -> str:
    if not messages:
        return ""
    parts: List[str] = []
    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("role", "")
            content = msg.get("content", "")
            parts.append(f"[{role}] {content}")
        else:
            parts.append(str(msg))
    return "\n".join(parts)


def _extract_format_marker(text: str) -> str:
    match = re.search(r"\[FORMAT:[A-Za-z]+\]", text or "", re.IGNORECASE)
    return match.group(0).upper() if match else ""


def _detect_requested_output_type(prompt_text: str):
    from conversation_manager import OutputType

    # Prefer explicit [FORMAT:*] tag anywhere, otherwise inspect only user lines
    # so system prompt examples (e.g., Python Script) never override requested format.
    tag = _extract_format_marker(prompt_text)
    if tag == "[FORMAT:PDF]":
        return OutputType(output_type="PDF File", file_extension=".pdf")
    if tag == "[FORMAT:JSON]":
        return OutputType(output_type="JSON File", file_extension=".json")
    if tag == "[FORMAT:HTML]":
        return OutputType(output_type="HTML File", file_extension=".html")
    if tag == "[FORMAT:CSV]":
        return OutputType(output_type="CSV File", file_extension=".csv")
    if tag == "[FORMAT:PY]":
        return OutputType(output_type="Python Script", file_extension=".py")
    if tag == "[FORMAT:CONCISE]":
        return OutputType(output_type="Simple Concise Answer", file_extension=".txt")
    if tag == "[FORMAT:TXT]":
        return OutputType(output_type="Text File", file_extension=".txt")

    user_only = "\n".join(
        line for line in prompt_text.splitlines() if line.startswith("[user]") or "[FORMAT:" in line
    )
    lower = user_only.lower()
    if "pdf report" in lower or "as a pdf" in lower:
        return OutputType(output_type="PDF File", file_extension=".pdf")
    if "json output" in lower or "as json" in lower:
        return OutputType(output_type="JSON File", file_extension=".json")
    if "html report" in lower or "as html" in lower:
        return OutputType(output_type="HTML File", file_extension=".html")
    if "csv file" in lower or "as csv" in lower:
        return OutputType(output_type="CSV File", file_extension=".csv")
    if "python script" in lower:
        return OutputType(output_type="Python Script", file_extension=".py")
    if "concise answer" in lower:
        return OutputType(output_type="Simple Concise Answer", file_extension=".txt")
    return OutputType(output_type="Text File", file_extension=".txt")


def _tool_for_output_type(prompt_text: str, ext: str, scenario: str) -> Tuple[str, str]:
    filename_map = {
        ".pdf": "deliverable.pdf",
        ".json": "deliverable.json",
        ".html": "deliverable.html",
        ".csv": "deliverable.csv",
        ".py": "deliverable.py",
        ".txt": "deliverable.txt",
    }
    tool_map = {
        ".pdf": "generate_pdf_output",
        ".json": "generate_json_output",
        ".html": "generate_html_output",
        ".csv": "generate_csv_output",
        ".py": "generate_python_script_output",
        ".txt": "generate_text_output",
    }
    if "simple concise answer" in prompt_text.lower():
        return "generate_simple_concise_answer_message", ""
    ext_norm = ext if ext.startswith(".") else f".{ext}"
    if scenario == "artifact_write_fail":
        return "generate_text_output", "\x00invalid/unwritable.txt"
    return tool_map.get(ext_norm, "generate_text_output"), filename_map.get(ext_norm, "deliverable.txt")


class FakeProviderClient:
    """
    OpenAI SDK-compatible deterministic provider double recording every call to ProviderCallLedger.
    """

    def __init__(self, ledger: Optional[ProviderCallLedger] = None, scenario: str = "default") -> None:
        self.ledger = ledger or ProviderCallLedger()
        self.output_filename = None
        self.default_scenario = scenario
        self._lock = threading.RLock()
        self._slow_delay_used = False
        self.chat = _ChatNamespace(self)
        self.beta = _BetaNamespace(self)
        self.embeddings = _EmbeddingsNamespace(self)

    def current_scenario(self) -> str:
        ctx_val = _ACTIVE_SCENARIO.get()
        return ctx_val if ctx_val != "default" else self.default_scenario

    def set_scenario(self, scenario: str) -> None:
        with self._lock:
            self.default_scenario = scenario
            self._slow_delay_used = False

    def _current_corr_id(self) -> Optional[str]:
        ctx_corr = _ACTIVE_CORRELATION_ID.get()
        if ctx_corr:
            return ctx_corr
        try:
            from output_generator import get_active_run_id

            return get_active_run_id()
        except Exception:
            return None

    def _maybe_delay(self, scenario: str) -> None:
        if scenario == "slow_delay":
            with self._lock:
                first = not self._slow_delay_used
                self._slow_delay_used = True
            if first:
                time.sleep(0.35)


class _ChatCompletions:
    def __init__(self, parent: FakeProviderClient) -> None:
        self._parent = parent

    def create(self, **kwargs: Any) -> Any:
        messages = kwargs.get("messages") or []
        model = str(kwargs.get("model") or "gpt-4o-mini")
        prompt_text = _flatten_messages(messages)
        scenario = self._parent.current_scenario()
        corr_id = self._parent._current_corr_id()
        self._parent._maybe_delay(scenario)

        if scenario == "provider_error":
            self._parent.ledger.record(
                method="chat.completions.create",
                model=model,
                response_format=None,
                correlation_id=corr_id,
                scenario=scenario,
                prompt_preview=prompt_text,
                full_prompt=prompt_text,
                error="Simulated upstream outage",
            )
            raise RuntimeError("Simulated upstream outage sk-secret-should-not-leak /private/aps-sensitive-path")

        if scenario == "empty_model":
            self._parent.ledger.record(
                method="chat.completions.create",
                model=model,
                response_format=None,
                correlation_id=corr_id,
                scenario=scenario,
                prompt_preview=prompt_text,
                full_prompt=prompt_text,
            )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="", tool_calls=None), finish_reason="stop")]
            )

        self._parent.ledger.record(
            method="chat.completions.create",
            model=model,
            response_format=None,
            correlation_id=corr_id,
            scenario=scenario,
            prompt_preview=prompt_text,
            full_prompt=prompt_text,
        )

        # Tool calling inside conversation_manager.finalize_output
        tools = kwargs.get("tools")
        if tools:
            ext_match = re.search(r"file extension:\s*(\.?[a-zA-Z0-9]+)", prompt_text, re.IGNORECASE)
            ext = ext_match.group(1) if ext_match else ".txt"
            if not ext.startswith("."):
                ext = f".{ext}"
            tool_name, filename = _tool_for_output_type(prompt_text, ext, scenario)
            fd_match = re.search(r"Final Decision:\s*(.*)$", prompt_text, re.DOTALL)
            final_dec = fd_match.group(1).strip() if fd_match else "Deterministic final output"
            if ext.lower() == ".json":
                payload = json.dumps({"summary": final_dec, "verified": True})
            elif ext.lower() == ".csv":
                payload = "item,status\nplan,verified\n"
            elif ext.lower() == ".html":
                payload = f"<h1>Result</h1><p>{final_dec}</p>"
            elif ext.lower() == ".py":
                payload = "def solve():\n    return 'verified'\n"
            else:
                payload = final_dec
            args_dict: Dict[str, Any] = {"final_decision": payload}
            filename = self._parent.output_filename or filename
            if filename:
                args_dict["filename"] = filename
            tool_call = SimpleNamespace(
                id="call_e2e_1",
                function=SimpleNamespace(name=tool_name, arguments=json.dumps(args_dict)),
            )
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=None, tool_calls=[tool_call]),
                        finish_reason="tool_calls",
                    )
                ]
            )

        lower = prompt_text.lower()
        if scenario == "xss_output":
            xss_payload = (
                "<script>window.__xss_executed = true;</script>"
                '<img src=x onerror="window.__xss_executed = true">'
                " Escaped response verified."
            )
            if "<step>" in lower or "enclosing all thoughts within <thinking>" in lower:
                content = (
                    "<count>0</count>"
                    "<thinking>Verify HTML/script escaping safety.</thinking>"
                    f"<step>Respond safely: {xss_payload}</step>"
                    "<reflection>Verified output includes unexecuted payload.</reflection>"
                    "<reward>0.91</reward>"
                    f"<agent_response>{xss_payload}</agent_response>"
                )
            else:
                content = xss_payload
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=None), finish_reason="stop")]
            )

        # Prompting / reasoning branches inside AdvancedPromptEngineer
        if "project_name" in lower or "short, descriptive project name" in lower:
            content = "Offline_E2E_Project"
        elif "rephrase the following task" in lower or "task into a clear, actionable prompt" in lower:
            content = "Complete the requested task clearly and accurately."
        elif "collaboratively" in lower and ("<thinking>" in lower or "<step>" in lower):
            content = (
                "<count>0</count>"
                "<thinking>Collaborative team synthesized an improved multi-perspective solution.</thinking>"
                "<step>Refine deliverable through multi-agent review.</step>"
                "<reflection>Collaborative team improved clarity and completeness.</reflection>"
                "<reward>0.96</reward>"
                "<answer>Collaborative team revision verified: structured multi-agent answer delivered cleanly.</answer>"
                "<final_reward>0.96</final_reward>"
                "<agent_response>Collaborative team revision verified: structured multi-agent answer delivered cleanly.</agent_response>"
            )
        elif "<thinking>" in lower or "<step>" in lower or "<count>" in lower or "problem-solving" in lower:
            if "<reflection>" in lower and "<reward>" in lower and "evaluate the following step" in lower:
                content = (
                    "<reflection>Step is clear, accurate, and aligned with requirements.</reflection>"
                    "<reward>0.92</reward>"
                )
            else:
                content = (
                    "<count>0</count>"
                    "<thinking>Analyzed task constraints and structured solution.</thinking>"
                    "<step>Synthesize structured deliverable for the task.</step>"
                    "<reflection>Verified solution structure and completeness.</reflection>"
                    "<reward>0.90</reward>"
                    "<answer>Solo pipeline synthesis: verified deterministic solution delivered cleanly.</answer>"
                    "<final_reward>0.90</final_reward>"
                    "<agent_response>Solo pipeline synthesis: verified deterministic solution delivered cleanly.</agent_response>"
                )
        elif "problem statement" in lower and "based on the initial prompt" in lower:
            fmt_tag = _extract_format_marker(prompt_text)
            suffix = f" {fmt_tag}" if fmt_tag else ""
            content = f"Refined problem statement aligned with user constraints.{suffix}"
        elif "summarize" in lower:
            fmt_tag = _extract_format_marker(prompt_text)
            suffix = f" {fmt_tag}" if fmt_tag else ""
            content = f"Team summary: participants aligned on a verified consensus deliverable.{suffix}"
        else:
            fmt_tag = _extract_format_marker(prompt_text)
            suffix = f" {fmt_tag}" if fmt_tag else ""
            content = f"Deterministic collaborative contribution aligned with task requirements.{suffix}"

        if scenario.startswith("two_step"):
            markers = re.findall(r"(?:FIRST|SECOND)_STEP_OUTPUT", prompt_text)
            marker = markers[-1] if markers else "FIRST_STEP_OUTPUT"
            content = (content.replace("</answer>", f" {marker}</answer>")
                       if "</answer>" in content else f"{content} {marker}")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=None), finish_reason="stop")]
        )


class _BetaCompletions:
    def __init__(self, parent: FakeProviderClient) -> None:
        self._parent = parent

    def parse(self, **kwargs: Any) -> Any:
        import advanced_prompting as ap
        import conversation_manager as cm
        from complexity_measures import Plan, PlanStep, Subtask

        messages = kwargs.get("messages") or []
        model = str(kwargs.get("model") or "gpt-4o-mini")
        response_format = kwargs.get("response_format")
        fmt_name = getattr(response_format, "__name__", str(response_format)) if response_format else None
        prompt_text = _flatten_messages(messages)
        scenario = self._parent.current_scenario()
        corr_id = self._parent._current_corr_id()
        self._parent._maybe_delay(scenario)

        if scenario == "provider_error":
            self._parent.ledger.record(
                method="beta.chat.completions.parse",
                model=model,
                response_format=fmt_name,
                correlation_id=corr_id,
                scenario=scenario,
                prompt_preview=prompt_text,
                full_prompt=prompt_text,
                error="Simulated upstream outage",
            )
            raise RuntimeError("Simulated upstream outage sk-secret-should-not-leak /private/aps-sensitive-path")

        self._parent.ledger.record(
            method="beta.chat.completions.parse",
            model=model,
            response_format=fmt_name,
            correlation_id=corr_id,
            scenario=scenario,
            prompt_preview=prompt_text,
            full_prompt=prompt_text,
        )

        if response_format is cm.BinaryVote:
            if scenario == "all_error_voting":
                raise RuntimeError("Simulated vote parse error")
            parsed: Any = cm.BinaryVote(vote=True)
        elif response_format is cm.ConfidenceVote:
            if scenario == "all_error_voting":
                raise RuntimeError("Simulated confidence parse error")
            parsed = cm.ConfidenceVote(confidence=0.92)
        elif response_format is cm.OutputType:
            parsed = _detect_requested_output_type(prompt_text)
        elif response_format is cm.ExtractFormat:
            parsed = cm.ExtractFormat(
                Facts=["Task constraints validated offline."],
                Arguments=["Structured approach satisfies requirements."],
                Direct_Replies=cm.DirectReply(
                    assistant_name="Strategist",
                    reply="Are there additional edge-case deployment constraints to prioritize?",
                ),
                Decisions=["Proceed with consensus deliverable."],
                Recommended_Actions=["Review generated summary and export artifacts."],
            )
        elif response_format is cm.TaskList:
            parsed = cm.TaskList(
                tasks=[
                    cm.TaskCompletion(
                        task="Deliver verified solution",
                        completed=True,
                        completion_artifact="Consensus summary",
                    )
                ]
            )
        elif response_format is cm.MediatorReviseOrDecide:
            decision_text = (
                "<script>window.__xss_executed = true;</script> Mediator safe decision"
                if scenario == "xss_output"
                else "Mediator consensus decision verified offline."
            )
            parsed = cm.MediatorReviseOrDecide(final_decision=decision_text, keep_output=True)
        elif response_format is cm.FinalOutput:
            out_type = _detect_requested_output_type(prompt_text)
            final_text = (
                "<script>window.__xss_executed = true;</script> Escaped final deliverable"
                if scenario == "xss_output"
                else "Verified deterministic consensus deliverable for offline execution."
            )
            parsed = cm.FinalOutput(final_output=final_text, output_type=out_type, version=1)
        elif response_format is cm.FinalOutputAnalysis:
            out_type = _detect_requested_output_type(prompt_text)
            parsed = cm.FinalOutputAnalysis(
                user_intent="Produce accurate task solution",
                expected_output_type=out_type.output_type,
                requirements=[cm.OutputRequirements(requirement="Clear complete answer", met=True)],
                matches_expected_output_type=True,
                aligns_with_user_intent=True,
                changes_needed_to_align=[],
                is_complete=True,
            )
        elif response_format is ap.CompletionStatus:
            parsed = ap.CompletionStatus(completion=True)
        elif response_format is ap.FinalReviewScore:
            parsed = ap.FinalReviewScore(score=0.93)
        elif response_format is ap.PromptSuggestions:
            parsed = ap.PromptSuggestions(prompt_modifications=["Keep answer concise and well-structured."])
        elif response_format is ap.response_choices:
            parsed = ap.response_choices(number_of_selected_response=1)
        elif response_format is ap.StepNumber:
            parsed = ap.StepNumber(score=1)
        elif response_format is Plan:
            parsed = Plan(
                steps=[
                    PlanStep(
                        step_number=1,
                        completed=False,
                        step_name="Execute Task",
                        step_description="Design a verification strategy and risk analysis for the deliverable",
                        step_explanation="Produce deterministic offline plan deliverable",
                        step_output="Final structured solution",
                        step_full_text="Step 1: Execute Task",
                        subtasks=[
                            Subtask(
                                subtask_number=1,
                                completed=False,
                                subtask_name="Draft Deliverable",
                                subtask_description="Draft clear response",
                                subtask_explanation="Ensure accuracy",
                                subtask_output="Draft ready",
                                subtask_full_text="1.1 Draft Deliverable",
                                subtasks=[],
                            )
                        ],
                    )
                ]
            )
        else:
            raise AssertionError(f"Unexpected structured response_format in FakeProviderClient: {response_format!r}")

        if response_format is Plan and scenario.startswith("two_step"):
            parsed.steps[0].subtasks = []
            parsed.steps[0].step_description = "Brainstorm strategy FIRST_STEP_OUTPUT"
            parsed.steps[0].description = parsed.steps[0].step_description
            second = parsed.steps[0].model_copy(deep=True)
            second.step_number = 2
            second.step_name = "Second Required Step"
            second.step_description = "Brainstorm strategy SECOND_STEP_OUTPUT"
            second.description = second.step_description
            second.step_full_text = "Step 2: SECOND_STEP_OUTPUT"
            parsed.steps.append(second)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed, content=parsed.model_dump_json(), refusal=None), finish_reason="stop")]
        )


class _ChatNamespace:
    def __init__(self, parent: FakeProviderClient) -> None:
        self.completions = _ChatCompletions(parent)


class _BetaNamespace:
    def __init__(self, parent: FakeProviderClient) -> None:
        self.chat = SimpleNamespace(completions=_BetaCompletions(parent))


class _EmbeddingsNamespace:
    def __init__(self, parent: FakeProviderClient) -> None:
        self._parent = parent

    def create(self, **kwargs: Any) -> Any:
        text = str(kwargs.get("input") or "")
        model = str(kwargs.get("model") or "text-embedding-3-small")
        self._parent.ledger.record(
            method="embeddings.create",
            model=model,
            response_format=None,
            correlation_id=self._parent._current_corr_id(),
            scenario=self._parent.current_scenario(),
            prompt_preview=text,
            full_prompt=text,
        )
        # Deterministic 16-dim unit-norm-ish vector derived from text bytes
        vec = [((sum(text.encode("utf-8")) + i * 31) % 97) / 97.0 for i in range(16)]
        return SimpleNamespace(data=[SimpleNamespace(embedding=vec)])


@contextlib.contextmanager
def scenario_scope(scenario: str, correlation_id: Optional[str] = None) -> Iterator[None]:
    """Bind an E2E provider scenario & correlation ID for the current execution context."""
    tok_s = _ACTIVE_SCENARIO.set(scenario)
    tok_c = _ACTIVE_CORRELATION_ID.set(correlation_id) if correlation_id is not None else None
    try:
        yield
    finally:
        if tok_c is not None:
            _ACTIVE_CORRELATION_ID.reset(tok_c)
        _ACTIVE_SCENARIO.reset(tok_s)


def install_fake_provider(client: FakeProviderClient, monkeypatch: Any = None):
    """
    Install `client` at `provider_client.get_client` cleanly (compatible with
    `provider_client.has_configured_provider()` without requiring OPENAI_API_KEY).
    """
    import provider_client

    def fake_get_client():
        return client

    fake_get_client.__name__ = "fake_get_client"
    if monkeypatch is not None:
        monkeypatch.setattr(provider_client, "get_client", fake_get_client)
        return client
    original = provider_client.get_client
    provider_client.get_client = fake_get_client
    return original
