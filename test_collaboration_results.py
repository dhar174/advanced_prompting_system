"""Issue 48: exercise real ownership/conversion, not mocked existence/open checks."""
from types import SimpleNamespace

import pytest

import advanced_prompting as ap
import conversation_manager as cm
from output_generator import (
    artifact_run_scope, generate_pdf_output, generate_text_file_output,
    get_active_run_id, get_tracked_artifact_ref,
)


def typed_fixture(value, *, artifact=False, output_type=None):
    """Adapt explicit test data at a mapping-test boundary, never production inference."""
    history, questions, content = value
    if isinstance(content, cm.ErrorResult):
        output = cm.failure_output(content.error_type, content.error_message)
    elif artifact:
        output = cm.artifact_conversation_output(content, output_type)
    else:
        output = cm.ConversationOutput(kind="text", content=content, output_type=output_type or cm.OutputType(output_type="Simple Concise Answer", file_extension=".txt"))
    normalized = [{"assistant": q.get("assistant", ""), "question": q.get("question") or q.get("reply", "")} for q in questions]
    return cm.ConversationResult(output=output, conversation=history, questions=normalized)


def parts():
    step = ap.PlanStep(step_number=7, step_name="Design strategy", step_description="Design a strategy and analysis", step_explanation="", step_output="Required", step_full_text="", completed=False, subtasks=[])
    task = ap.Task("Original task", "Original task", 2.0, [], [], "", 0.0, ap.Plan(steps=[step]), cm.OutputType(output_type="Simple Concise Answer", file_extension=".txt"))
    return ap.AdvancedPromptEngineer(ap.PromptEngineeringConfig()), task, step


def provider_reply(monkeypatch, *, content=None, tool=None, filename="result.txt", refusal=None):
    import json
    calls = [] if tool is None else [SimpleNamespace(id="tool1", function=SimpleNamespace(name=tool, arguments=json.dumps({"final_decision": "OWNED_CONTENT", "filename": filename})))]
    message = SimpleNamespace(content=content, tool_calls=calls, refusal=refusal)
    monkeypatch.setattr(cm, "client", SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: SimpleNamespace(choices=[SimpleNamespace(message=message)])))))


def test_converter_requires_registered_active_owner(tmp_path):
    engineer, task, step = parts()
    with artifact_run_scope("owner"):
        owned = generate_text_file_output("OWNED_CONTENT", "owned.txt")
        assert engineer.convert_conversation_output_to_final_step_output(owned, step, task).final_output == "OWNED_CONTENT"
        unregistered = tmp_path / "unregistered.txt"
        unregistered.write_text("FOREIGN_SENTINEL")
        with pytest.raises(cm.ConversationResultError, match="artifact"):
            engineer.convert_conversation_output_to_final_step_output(str(unregistered), step, task)
    with artifact_run_scope("owner"):
        with pytest.raises(cm.ConversationResultError, match="artifact"):
            engineer.convert_conversation_output_to_final_step_output(owned, step, task)


@pytest.mark.parametrize("raw", [None, "", " ", "missing.txt"])
def test_missing_legacy_artifact_is_explicit_failure(raw):
    engineer, task, step = parts()
    with artifact_run_scope("missing"):
        with pytest.raises(cm.ConversationResultError):
            engineer.convert_conversation_output_to_final_step_output(raw, step, task)


def test_typed_text_that_looks_like_a_path_is_not_read(monkeypatch, tmp_path):
    path = tmp_path / "text.txt"
    path.write_text("FOREIGN_SENTINEL")
    provider_reply(monkeypatch, content=str(path))
    result = cm.finalize_output_result("a task", cm.OutputType(output_type="Simple Concise Answer", file_extension=".txt"))
    assert result.kind == "text"
    assert result.content == str(path)
    assert result.artifacts == []


@pytest.mark.parametrize("tool,ext", [("generate_text_output", ".txt"), ("generate_pdf_output", ".pdf")])
def test_typed_writer_returns_only_registered_active_owner(monkeypatch, tool, ext):
    provider_reply(monkeypatch, tool=tool, filename="result" + ext)
    with artifact_run_scope("typed") as owner:
        result = cm.finalize_output_result("task", cm.OutputType(output_type="PDF File" if ext == ".pdf" else "Text File", file_extension=ext))
        assert result.kind == "artifact"
        assert len(result.artifacts) == 1
        ref = result.artifacts[0]
        assert ref.run_id == owner == get_active_run_id()
        assert get_tracked_artifact_ref(ref.path) == ref
        assert ref.path not in result.content
        assert result.content == "OWNED_CONTENT" if ext == ".txt" else result.content == "Generated PDF artifact: result.pdf"


@pytest.mark.parametrize("content,refusal", [(None, None), (" ", None), (None, "Cannot comply")])
def test_empty_or_refused_provider_result_is_failure(monkeypatch, content, refusal):
    provider_reply(monkeypatch, content=content, refusal=refusal)
    result = cm.finalize_output_result("task", cm.OutputType(output_type="Simple Concise Answer", file_extension=".txt"))
    assert result.kind == "failure"
    assert result.content == ""
    assert result.error is not None


def test_deleted_registered_artifact_does_not_become_answer():
    from pathlib import Path
    engineer, task, step = parts()
    with artifact_run_scope("deleted"):
        path = generate_text_file_output("content", "deleted.txt")
        Path(path).unlink()
        with pytest.raises(cm.ConversationResultError):
            engineer.convert_conversation_output_to_final_step_output(path, step, task)
