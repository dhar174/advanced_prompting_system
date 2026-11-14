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


@pytest.mark.parametrize("score", [0.61, 1.0])
def test_nested_candidate_uses_actual_judged_content(monkeypatch, score):
    engineer, task, step = parts()
    outcome = cm.ConversationResult(output=cm.ConversationOutput(kind="text", content="ACTUAL_NESTED_OUTPUT"),
        conversation=[{"role":"assistant", "content":"SELECTED_HISTORY"}],
        questions=[{"assistant":"Strategist", "question":"SELECTED_QUESTION"}])
    monkeypatch.setattr(cm, "run_conversation_result", lambda **kw: outcome)
    def judge(candidate, judged_task):
        assert candidate.description == "ACTUAL_NESTED_OUTPUT"
        return ap.Reflection(content="Actual evaluation", reward=score, step_number=candidate.step_number)
    monkeypatch.setattr(engineer, "judge_step", judge)
    result = engineer.collaborative_reasoning_main(task, step, ap.Interaction(task, [], []), task.output_type, 1, 12)
    assert result.final_reward == score
    assert result.steps[0].step_number == 12
    assert result.reflections[0].content == "Actual evaluation"
    assert result.collaboration_results[0].result.quality_score == score


def test_nested_evaluator_failure_preserves_unmeasured_partial_output(monkeypatch):
    engineer, task, step = parts()
    monkeypatch.setattr(cm, "run_conversation_result", lambda **kw: cm.ConversationResult(output=cm.ConversationOutput(kind="text", content="Useful partial")))
    reflection = ap.Reflection(content="Error", reward=0.0, step_number=12)
    reflection._judge_error = True
    monkeypatch.setattr(engineer, "judge_step", lambda *a: reflection)
    result = engineer.collaborative_reasoning_main(task, step, ap.Interaction(task, [], []), task.output_type, 1, 12)
    assert result.status == "incomplete"
    assert result.stop_reason == "evaluation_unavailable"
    assert result.final_reward is None
    assert result.reflections == []
    assert result.answer == "Useful partial"


def test_nested_failure_cannot_be_a_perfect_answer(monkeypatch):
    engineer, task, step = parts()
    monkeypatch.setattr(cm, "run_conversation_result", lambda **kw: cm.ConversationResult(output=cm.failure_output("EmptyOutput", "No usable output.")))
    result = engineer.collaborative_reasoning_main(task, step, ap.Interaction(task, [], []), task.output_type, 1, 12)
    assert result.status == "error"
    assert result.answer == ""
    assert result.final_reward is None
    assert result.steps == []


def test_committed_metadata_is_copied_and_roundtrips():
    _, task, _ = parts()
    source = ap.Interaction(task, [], [], collaboration_results=[ap.CollaborationRecord(plan_step_number=7, step_number=12,
        result=cm.ConversationResult(output=cm.ConversationOutput(kind="text", content="answer"),
            questions=[{"assistant":"A", "question":"Selected"}]))])
    accepted = ap.Interaction(task, [], [])
    accepted.commit_collaboration_results(source)
    source.collaboration_results[0].result.questions[0]["question"] = "REJECTED_MUTATION"
    assert accepted.collaboration_results[0].result.questions[0]["question"] == "Selected"
    restored = ap.Interaction.from_dict(accepted.to_dict())
    assert restored.collaboration_results == accepted.collaboration_results
    restored.collaboration_results[0].result.questions.clear()
    assert accepted.collaboration_results[0].result.questions
