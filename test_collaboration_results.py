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
    if not isinstance(value, tuple) or len(value) != 3:
        return value  # Deliberately malformed typed-runner fixtures must fail closed.
    history, questions, content = value
    history, questions = history or [], questions or []
    if any(not isinstance(entry, dict) for entry in history):
        output = cm.failure_output("InvalidHistory", "Conversation history contains an invalid message.")
        history = [entry for entry in history if isinstance(entry, dict)]
    elif isinstance(content, cm.ErrorResult):
        output = cm.failure_output(content.error_type, content.error_message)
    elif not isinstance(content, str) or not content.strip():
        output = cm.failure_output("EmptyOutput", "The conversation returned no usable output.")
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


def provider_reply(monkeypatch, *, content=None, tool=None, filename="result.txt", refusal=None, payload="OWNED_CONTENT"):
    import json
    calls = [] if tool is None else [SimpleNamespace(id="tool1", function=SimpleNamespace(name=tool, arguments=json.dumps({"final_decision": payload, "filename": filename})))]
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


@pytest.mark.parametrize("value", [{"kind":"text", "content":""}, {"kind":"artifact"},
    {"kind":"failure"}, {"kind":"failure", "content":"Not an answer", "error":{"error_type":"x", "error_message":"Failed"}}])
def test_output_contract_rejects_contradictory_or_empty_success(value):
    with pytest.raises(ValueError):
        cm.ConversationOutput.model_validate(value)


def test_explicit_foreign_reference_cannot_be_rebound_to_current_owner():
    engineer, task, step = parts()
    with artifact_run_scope("ref-owner"):
        path = generate_text_file_output("Owner succeeds", "ref.txt")
        output = cm.artifact_conversation_output(path)
        assert engineer.convert_conversation_result_to_final_step_output(cm.ConversationResult(output=output), step, task).final_output == "Owner succeeds"
        output.artifacts[0] = output.artifacts[0].model_copy(update={"run_id":"foreign-owner"})
        with pytest.raises(cm.ConversationResultError):
            engineer.convert_conversation_result_to_final_step_output(cm.ConversationResult(output=output), step, task)


def test_artifact_stat_failure_is_an_explicit_safe_failure(monkeypatch):
    with artifact_run_scope("stat"):
        path = generate_text_file_output("owned", "stat.txt")
        def failed_stat(*a):
            raise PermissionError("sk-test-secret /private/sensitive-path")
        monkeypatch.setattr(cm.os.path, "getsize", failed_stat)
        output = cm.artifact_conversation_output(path)
        assert output.kind == "failure"
        assert "sk-test-secret" not in output.model_dump_json()
        assert "/private/sensitive-path" not in output.model_dump_json()


@pytest.mark.parametrize("labels", [("tenant/a", "tenant:a"), ("same", "same")])
def test_nested_foreign_owner_denial_after_registered_success(labels):
    from output_generator import get_artifact_registry, resolve_output_filepath
    engineer, task, step = parts()
    with artifact_run_scope(labels[0]) as owner:
        path = generate_text_file_output("OWNER_SENTINEL", "owned.txt")
        ref = get_tracked_artifact_ref(path)
        assert engineer.convert_conversation_output_to_final_step_output(path, step, task).final_output == "OWNER_SENTINEL"
    with artifact_run_scope(labels[1]) as foreign:
        assert owner != foreign
        assert get_artifact_registry().list_artifacts(owner) == []
        with pytest.raises(PermissionError):
            resolve_output_filepath(path)
        with pytest.raises(cm.ConversationResultError):
            engineer.convert_conversation_output_to_final_step_output(path, step, task)
        from pathlib import Path
        import hashlib
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == ref.sha256


def test_symlink_and_empty_registered_files_are_not_answers(tmp_path):
    from pathlib import Path
    from output_generator import track_generated_artifact
    engineer, task, step = parts()
    with artifact_run_scope("symlink"):
        path = generate_text_file_output("OWNER_CONTENT", "owned.txt")
        link = tmp_path / "link.txt"
        link.symlink_to(path)
        with pytest.raises(cm.ConversationResultError):
            engineer.convert_conversation_output_to_final_step_output(str(link), step, task)
        Path(path).write_text("")
        track_generated_artifact(path)
        with pytest.raises(cm.ConversationResultError):
            engineer.convert_conversation_output_to_final_step_output(path, step, task)


def test_parallel_nested_reads_keep_distinct_registered_owners():
    from concurrent.futures import ThreadPoolExecutor
    import threading
    barrier = threading.Barrier(2)
    def run(label):
        engineer, task, step = parts()
        with artifact_run_scope("reused") as owner:
            path = generate_text_file_output(label, "parallel.txt")
            barrier.wait(timeout=5)
            return owner, engineer.convert_conversation_output_to_final_step_output(path, step, task).final_output
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, ["FIRST_RUN", "SECOND_RUN"]))
    assert results[0][0] != results[1][0]
    assert [r[1] for r in results] == ["FIRST_RUN", "SECOND_RUN"]


def test_nested_candidate_copies_the_accepted_base_for_comparable_scoring(monkeypatch):
    engineer, task, step = parts()
    prior_reflection = ap.Reflection(content="Prior accepted", reward=0.61, step_number=1)
    prior_step = ap.Step(description="Prior output", step_number=1, plan_step_number=1, remaining_budget=0, reflection=prior_reflection)
    accepted = ap.Interaction(task, [prior_step], [prior_reflection])
    monkeypatch.setattr(cm, "run_conversation_result", lambda **kw: cm.ConversationResult(output=cm.ConversationOutput(kind="text", content="Next output")))
    monkeypatch.setattr(engineer, "judge_step", lambda candidate, task: ap.Reflection(content="New evaluation", reward=0.95, step_number=candidate.step_number))
    result = engineer.collaborative_reasoning_main(task, step, accepted, task.output_type, 1, 12)
    assert result.sum_rewards() == pytest.approx(1.56)
    assert [s.step_number for s in result.steps] == [1, 12]
    result.steps[0].description = "Candidate mutation"
    result.reflections[0].content = "Candidate reflection mutation"
    assert accepted.steps[0].description == "Prior output"
    assert accepted.reflections[0].content == "Prior accepted"


@pytest.mark.parametrize("payload", [None, "", " \n "])
def test_empty_pdf_tool_payload_is_failure_before_writing(monkeypatch, payload):
    from output_generator import list_run_artifacts
    provider_reply(monkeypatch, tool="generate_pdf_output", filename="empty.pdf", payload=payload)
    with artifact_run_scope("empty-payload"):
        output = cm.finalize_output_result("A valid task", cm.OutputType(output_type="PDF File", file_extension=".pdf"))
        assert output.kind == "failure"
        assert list_run_artifacts() == []


def test_direct_nested_adapter_keeps_artifact_scope_until_conversion(monkeypatch):
    import provider_client
    from tools.e2e.provider_fixture import FakeProviderClient, install_fake_provider
    fake = FakeProviderClient()
    install_fake_provider(fake, monkeypatch)
    engineer, task, step = parts()
    step.description = step.step_description = "[FORMAT:TXT] Design a strategy and analysis"
    assert get_active_run_id() is None
    with provider_client.provider_execution_scope(default_model="gpt-4o-mini", max_calls=100, timeout_seconds=10):
        result = engineer.invoke_collaborative_reasoning(task, step)
    assert result.final_output.strip()
    assert result.collaboration_result.output.kind == "artifact"
    assert get_active_run_id() is None
    assert 0 < fake.ledger.total_calls <= 100


def test_legacy_conversation_questions_keep_reply_and_attribution(monkeypatch):
    from tools.e2e.provider_fixture import FakeProviderClient, install_fake_provider
    fake = FakeProviderClient()
    install_fake_provider(fake, monkeypatch)
    history, questions, output = cm.run_conversation("Write a concise response", ["Strategist", "Mediator"], "Strategist", num_rounds=1, max_calls=100)
    assert history and questions and output
    assert all({"assistant", "question", "reply", "asking_assistant"} <= set(q) for q in questions)
    assert all(q["reply"] == q["question"] for q in questions)
    assert 0 < fake.ledger.total_calls <= 100


def test_typed_writer_preserves_previous_artifact_versions(monkeypatch):
    from pathlib import Path
    with artifact_run_scope("versions"):
        provider_reply(monkeypatch, tool="generate_text_output", filename="same.txt", payload="ACCEPTED_BYTES")
        first = cm.finalize_output_result("task", cm.OutputType(output_type="Text File", file_extension=".txt"))
        provider_reply(monkeypatch, tool="generate_text_output", filename="same.txt", payload="REJECTED_BYTES")
        second = cm.finalize_output_result("task", cm.OutputType(output_type="Text File", file_extension=".txt"))
        assert first.artifacts[0].path != second.artifacts[0].path
        assert Path(first.artifacts[0].path).read_text() == "ACCEPTED_BYTES"
        assert Path(second.artifacts[0].path).read_text() == "REJECTED_BYTES"


def test_conversation_runtime_failure_is_typed_and_cleans_its_scopes(monkeypatch):
    import provider_client
    def failed_conversation(**kwargs):
        raise cm.LLMCallError("sk-fixture-secret /private/sensitive-path")
    monkeypatch.setattr(cm, "_run_conversation_impl", failed_conversation)
    result = cm.run_conversation_result("task", ["Strategist"], "Strategist", max_calls=1)
    assert result.output.kind == "failure"
    assert "sk-fixture-secret" not in result.model_dump_json()
    assert "/private/sensitive-path" not in result.model_dump_json()
    assert get_active_run_id() is None
    assert provider_client.get_active_execution_context() is None
