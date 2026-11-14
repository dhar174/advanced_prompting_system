"""PR62 review regressions: real state transitions and explicitly scoped mapping tests."""
from types import SimpleNamespace

import pytest

import advanced_prompting as ap
import conversation_manager as cm
import execution_modes as em
from test_collaboration_results import parts, provider_reply


def interaction_with_record(task, plan_number, number, score, marker):
    reflection = ap.Reflection(content=marker, reward=score, step_number=number)
    step = ap.Step(description=marker, step_number=number, plan_step_number=plan_number,
                   remaining_budget=1, reflection=reflection)
    record = ap.CollaborationRecord(plan_step_number=plan_number, step_number=number,
        result=cm.ConversationResult(output=cm.ConversationOutput(kind="text", content=marker),
            conversation=[{"role":"assistant", "content":marker}],
            questions=[{"assistant":"A", "question":marker}], quality_score=score))
    return ap.Interaction(task, [step], [reflection], answer=marker, final_reward=score, collaboration_results=[record])


@pytest.mark.parametrize("old_score", [0.0, 0.3])
def test_exploration_atomically_replaces_equal_length_candidate(monkeypatch, old_score):
    engineer, task, plan_step = parts()
    monkeypatch.setattr(engineer, "judge_step", lambda st, t: ap.Reflection(content="Independent assessment", reward=0.9, step_number=st.step_number))
    base = interaction_with_record(task, 1, 1, 0.8, "PRIOR_ACCEPTED")
    candidate = interaction_with_record(task, plan_step.step_number, 12, old_score, "DISCARDED_DEBATE")
    candidate.steps = base.steps + candidate.steps
    candidate.reflections = base.reflections + candidate.reflections
    candidate.step_outputs = [SimpleNamespace(associated_plan_step=plan_step, final_output="DISCARDED_FILE")]
    response = "<thinking>Alternative</thinking><step>SOLO_STEP</step><reflection>Solo assessment</reflection><reward>0.9</reward><answer>SOLO_ANSWER</answer><final_reward>0.9</final_reward>"
    monkeypatch.setattr(engineer, "call_openai", lambda **kwargs: response)
    result = engineer.dynamic_confidence_exploration(candidate, task, plan_step, accepted_base=base)
    assert result is not candidate
    assert result.answer == "SOLO_ANSWER"
    assert [s.description for s in result.steps] == ["PRIOR_ACCEPTED", "SOLO_STEP"]
    assert [s.step_number for s in result.steps] == [1, 12]
    assert [r.result.output.content for r in result.collaboration_results] == ["PRIOR_ACCEPTED"]
    assert result.step_outputs == []
    assert candidate.answer == "DISCARDED_DEBATE"
    assert candidate.steps[-1].description == "DISCARDED_DEBATE"
    assert base.steps[0].description == "PRIOR_ACCEPTED"


@pytest.mark.parametrize("score", [None, 0.3, 0.2, float("nan"), 1.5])
def test_exploration_rejection_preserves_entire_candidate(monkeypatch, score):
    engineer, task, step = parts()
    monkeypatch.setattr(engineer, "judge_step", lambda st, t: ap.Reflection(content="Independent rejection", reward=score, step_number=st.step_number) if score is not None else SimpleNamespace(reward=None))
    candidate = interaction_with_record(task, step.step_number, 12, 0.3, "ACCEPTED")
    response = f"<thinking>Alternative</thinking><step>Rejected</step><reflection>Rejected</reflection><reward>0.9</reward><answer>Rejected answer</answer><final_reward>{score}</final_reward>"
    monkeypatch.setattr(engineer, "call_openai", lambda **kwargs: response)
    result = engineer.dynamic_confidence_exploration(candidate, task, step)
    assert result is candidate
    assert result.answer == "ACCEPTED"
    assert result.collaboration_results[0].result.questions[0]["question"] == "ACCEPTED"


def test_serialized_error_dict_is_validated_and_copied():
    _, task, _ = parts()
    data={"error_type":"ProviderError", "error_message":"Unavailable"}
    interaction=ap.Interaction(task, [], [], error_result=data)
    data["error_message"]="Caller mutation"
    assert isinstance(interaction.error_result, cm.ErrorResult)
    assert interaction.error_result.error_message == "Unavailable"
    serialized=interaction.to_dict()
    serialized["error_result"]={"error_type":"ProviderError", "error_message":"Round trip"}
    restored=ap.Interaction.from_dict(serialized)
    assert restored.error_result.error_type == "ProviderError"


@pytest.mark.parametrize("score", [-0.1, 1.1, float("nan"), float("inf"), None])
def test_invalid_final_judge_is_unknown_and_preserves_output(monkeypatch, score):
    engineer, task, _ = parts()
    candidate=ap.Interaction(task, [], [], answer="USEFUL_OUTPUT", final_reward=0.7, status="completed")
    parsed=SimpleNamespace(score=score) if score is not None else None
    response=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed, refusal=None))])
    monkeypatch.setattr(ap, "client", SimpleNamespace(beta=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=lambda **kw:response)))))
    result=engineer.judge_final_answer(task, candidate)
    assert result.answer == "USEFUL_OUTPUT"
    assert result.final_reward is None
    assert result.status == "incomplete"
    assert result.error_result.error_type == "EvaluationUnavailable"


@pytest.mark.parametrize("value", [{}, []])
@pytest.mark.parametrize("tool", ["generate_json_output", "generate_csv_output"])
def test_empty_structured_tool_payload_is_explicit_failure(monkeypatch, value, tool):
    provider_reply(monkeypatch, tool=tool, payload=value)
    result=cm.finalize_output_result("task", cm.OutputType(output_type="JSON File", file_extension=".json"))
    assert result.kind == "failure"


def test_artifact_output_requires_exactly_one_reference():
    from output_generator import artifact_run_scope, generate_text_file_output
    with artifact_run_scope("multiple"):
        one=cm.artifact_conversation_output(generate_text_file_output("one", "one.txt"))
        two=cm.artifact_conversation_output(generate_text_file_output("two", "two.txt"))
        with pytest.raises(ValueError):
            cm.ConversationOutput(kind="artifact", content="Outputs", artifacts=one.artifacts+two.artifacts)


def test_error_with_prior_output_always_has_public_error_and_event(monkeypatch):
    _, task, step=parts()
    partial=ap.Interaction(task, [ap.Step(description="PRIOR_OUTPUT", step_number=1, plan_step_number=step.step_number, remaining_budget=0)], [],
        answer="PRIOR_OUTPUT", status="error", stop_reason="conversation_error")
    monkeypatch.setattr(ap.AdvancedPromptEngineer, "main", lambda *args:partial)
    result=em.execute_run(em.RunRequest(mode="advanced_pipeline", task_prompt="task"))
    assert result.status == "error"
    assert result.error is not None
    assert any(event.event_type == "run_error" for event in result.events)
    assert all(turn.get("name") != "Finalizer" for turn in result.conversation)


def test_terminal_failed_candidate_metadata_is_preserved_as_partial():
    engineer, task, step=parts()
    accepted=interaction_with_record(task, 1, 1, 0.9, "ACCEPTED_HISTORY")
    failed=interaction_with_record(task, step.step_number, 12, 0.1, "PARTIAL_FAILED_HISTORY")
    failed.status="error"
    failed.collaboration_results[0].result.output=cm.failure_output("ProviderError", "Unavailable")
    engineer._record_conversation_failure(accepted, [failed], step.step_number)
    assert accepted.error_result.error_type == "ProviderError"
    assert accepted.collaboration_results[0].result.output.content == "ACCEPTED_HISTORY"
    assert accepted.failed_collaboration_results[0].result.conversation[0]["content"] == "PARTIAL_FAILED_HISTORY"


def test_failure_and_artifact_annotations_are_bounded_and_honest():
    from output_generator import artifact_run_scope, generate_text_file_output
    with artifact_run_scope("annotation"):
        output=cm.artifact_conversation_output(generate_text_file_output("LARGE_BODY_"*1000, "large.txt"))
        annotation=cm.conversation_output_annotation(output)
        assert "large.txt" in annotation
        assert "LARGE_BODY_" not in annotation and len(annotation)<300
    failed=cm.conversation_output_annotation(cm.failure_output("ProviderError", "Unavailable"))
    assert failed.startswith("Output failed:")


def test_partial_failure_metadata_and_step_order_survive_public_mapping(monkeypatch):
    engineer, task, step=parts()
    first=step.model_copy(deep=True, update={"step_number":1, "step_name":"Step one", "completed":True, "step_output":"FIRST_SUMMARY"})
    second=step.model_copy(deep=True, update={"step_number":2, "step_name":"Step two", "completed":False})
    task.plan=ap.Plan(steps=[first,second])
    accepted=interaction_with_record(task, 1, 1, 0.9, "FIRST_HISTORY")
    failed=interaction_with_record(task, 2, 2, 0.1, "PARTIAL_SECOND_HISTORY")
    failed.collaboration_results[0].result.output=cm.failure_output("ProviderError", "Unavailable")
    engineer._record_conversation_failure(accepted, [failed], 2)
    accepted.status="error"
    accepted.stop_reason="conversation_error"
    monkeypatch.setattr(ap.AdvancedPromptEngineer, "main", lambda *args:accepted)
    result=em.execute_run(em.RunRequest(mode="advanced_pipeline", task_prompt="task"))
    assert any(turn["content"] == "PARTIAL_SECOND_HISTORY" and turn["plan_step_number"] == 2 for turn in result.conversation)
    assert {q["plan_step_number"] for q in result.questions} == {1,2}
    attributed=[turn["plan_step_number"] for turn in result.conversation if turn.get("plan_step_number") is not None]
    assert attributed == sorted(attributed)
    summary=next(turn for turn in result.conversation if turn["content"] == "FIRST_SUMMARY")
    assert summary["plan_step_number"] == 1
    assert all(turn.get("name") != "Finalizer" for turn in result.conversation)


def test_mid_conversation_provider_failure_retains_accumulated_turns(monkeypatch):
    from tools.e2e.provider_fixture import FakeProviderClient, install_fake_provider, scenario_scope
    fake=FakeProviderClient()
    install_fake_provider(fake, monkeypatch)
    with scenario_scope("partial_conversation_failure"):
        result=cm.run_conversation_result("Discuss a useful response", ["Strategist","Mediator"], "Strategist", num_rounds=1, max_calls=100)
    assert result.output.kind == "failure"
    assert any("PARTIAL_HISTORY" in turn["content"] for turn in result.conversation)
    assert any(q["question"] == "PARTIAL_QUESTION" for q in result.questions)
    assert "sk-review-secret" not in result.model_dump_json()
    assert "/private/review-path" not in result.model_dump_json()


def test_multi_step_exploration_rebases_each_shared_reflection_once(monkeypatch):
    engineer, task, step = parts()
    monkeypatch.setattr(engineer, "judge_step", lambda st, t: ap.Reflection(content="Independent assessment", reward=0.9, step_number=st.step_number))
    candidate = interaction_with_record(task, step.step_number, 12, 0.3, "OLD")
    response = ("<thinking>Alternative</thinking><step>First</step><reflection>First review</reflection><reward>0.9</reward>"
                "<step>Second</step><reflection>Second review</reflection><reward>0.9</reward>"
                "<answer>New answer</answer><final_reward>0.9</final_reward>")
    monkeypatch.setattr(engineer, "call_openai", lambda **kwargs:response)
    result=engineer.dynamic_confidence_exploration(candidate, task, step)
    assert [st.step_number for st in result.steps] == [12,13]
    assert [st.reflection.step_number for st in result.steps] == [12,13]
    assert [reflection.step_number for reflection in result.reflections] == [12,13]

@pytest.mark.parametrize("score", [0.0, 1.0])
def test_final_judge_preserves_valid_endpoints(monkeypatch, score):
    engineer, task, _=parts()
    response=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=SimpleNamespace(score=score), refusal=None))])
    monkeypatch.setattr(ap, "client", SimpleNamespace(beta=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=lambda **kw:response)))))
    result=engineer.judge_final_answer(task, ap.Interaction(task, [], [], answer="Useful"))
    assert result.final_reward == score and result.error_result is None


@pytest.mark.parametrize("exception_name,reason", [("RunBudgetExceededError","budget_exhausted"), ("RunTimeoutError","timeout"), ("RunCancelledError","cancelled")])
def test_final_evaluation_policy_stop_is_attempted_once(monkeypatch, exception_name, reason):
    import provider_client
    engineer, task, _=parts()
    attempts=[]
    def parse(**kwargs):
        attempts.append(kwargs)
        raise getattr(provider_client, exception_name)("sk-review-secret /private/review-path")
    monkeypatch.setattr(ap, "client", SimpleNamespace(beta=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse)))))
    result=engineer._judge_completed_interaction(task, ap.Interaction(task, [], [], answer="Useful"))
    assert len(attempts) == 1
    assert result.answer == "Useful" and result.final_reward is None
    assert result.status == reason and result.stop_reason == reason
    assert "sk-review-secret" not in result.error_result.model_dump_json()


def test_self_consistency_merge_preserves_unknown_quality_and_evaluation_error(monkeypatch):
    engineer, task, step=parts()
    engineer.config.n=1
    engineer.config.max_sc_calls=2
    monkeypatch.setattr(engineer, "call_openai", lambda **kw:"<thinking>Work</thinking><step>New work</step><reflection>Good</reflection><reward>0.9</reward><answer>New answer</answer>")
    monkeypatch.setattr(engineer, "choose_best_response", lambda responses, **kw:responses[0])
    monkeypatch.setattr(engineer, "judge_step_completion", lambda *args:(True,1))
    def failed_judge(task, candidate):
        candidate.final_reward=None
        candidate.status="incomplete"
        candidate.stop_reason="evaluation_unavailable"
        candidate.error_result=cm.ErrorResult(error_type="EvaluationUnavailable", error_message="Unavailable")
        return candidate
    monkeypatch.setattr(engineer, "judge_final_answer", failed_judge)
    base=ap.Interaction(task, [], [], answer="Previous answer", final_reward=0.9)
    result=engineer.self_consistency(task, "Solve", base, [], task.output_type, 1, 12, step.step_number, consistency_multiplier=1)
    assert result.final_reward is None
    assert result.error_result.error_type == "EvaluationUnavailable"
    assert result.answer == "New answer"


@pytest.mark.parametrize("foreign_label", ["tenant:a", "tenant/a"])
def test_actual_collaborative_adapter_denies_foreign_typed_artifact_after_owner_success(monkeypatch, foreign_label):
    from output_generator import artifact_run_scope, generate_text_file_output, get_artifact_registry
    engineer, task, step=parts()
    monkeypatch.setattr(engineer, "judge_step", lambda st, task:ap.Reflection(content="Measured", reward=0.61, step_number=st.step_number))
    with artifact_run_scope("tenant/a") as owner:
        path=generate_text_file_output("PRIVATE_OWNER_BYTES", "owned.txt")
        result=cm.ConversationResult(output=cm.artifact_conversation_output(path))
        monkeypatch.setattr(cm, "run_conversation_result", lambda **kw:result.model_copy(deep=True))
        success=engineer.collaborative_reasoning_main(task, step, ap.Interaction(task,[],[]), task.output_type, 1, 12)
        assert success.answer == "PRIVATE_OWNER_BYTES"
        assert len(get_artifact_registry().list_artifacts(owner)) == 1
        unregistered=result.model_copy(deep=True)
        unregistered.output.artifacts[0]=unregistered.output.artifacts[0].model_copy(update={"artifact_id":"art_unregistered"})
        monkeypatch.setattr(cm, "run_conversation_result", lambda **kw:unregistered.model_copy(deep=True))
        denied=engineer.collaborative_reasoning_main(task, step, ap.Interaction(task,[],[]), task.output_type, 1, 12)
        assert denied.status == "error" and "PRIVATE_OWNER_BYTES" not in str(denied.to_dict())
        monkeypatch.setattr(cm, "run_conversation_result", lambda **kw:result.model_copy(deep=True))
    with artifact_run_scope(foreign_label):
        assert get_artifact_registry().list_artifacts(owner) == []
        denied=engineer.collaborative_reasoning_main(task, step, ap.Interaction(task,[],[]), task.output_type, 1, 12)
        assert denied.status == "error" and denied.answer == ""
        assert "PRIVATE_OWNER_BYTES" not in str(denied.to_dict())


def test_unknown_candidate_quality_does_not_dispatch_exploration(monkeypatch):
    engineer, task, step=parts()
    candidate=interaction_with_record(task, step.step_number, 12, 0.3, "Useful")
    candidate.final_reward=None
    monkeypatch.setattr(engineer, "call_openai", lambda **kw:pytest.fail("Unknown quality must not dispatch exploration"))
    assert engineer.dynamic_confidence_exploration(candidate, task, step) is candidate


def test_final_judge_provider_exception_returns_safe_unknown_quality(monkeypatch):
    engineer, task, _=parts()
    def fail(**kwargs):
        raise RuntimeError("sk-review-secret /private/review-path")
    monkeypatch.setattr(ap, "client", SimpleNamespace(beta=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=fail)))))
    result=engineer.judge_final_answer(task, ap.Interaction(task, [], [], answer="Useful"))
    assert result.answer == "Useful" and result.final_reward is None
    assert result.stop_reason == "evaluation_unavailable"
    assert "sk-review-secret" not in result.error_result.model_dump_json()


def test_defensive_empty_eligible_branch_keeps_structured_error(monkeypatch):
    # Explicit branch unit: ordinary self-consistency has no collaboration records,
    # so the review's partial-solo reproduction does not itself make eligible empty.
    from test_advanced_prompting import TestPR39EngineAndSelfConsistency as Helpers
    step=Helpers._make_plan_step(1, "One", "Required")
    engineer=Helpers._setup_deterministic_main_engineer(ap.Plan(steps=[step]), agents=2)
    _, task, _=parts()
    failed=interaction_with_record(task, 1, 1, 0.3, "Unusable candidate answer")
    failed.status="error"
    failed.collaboration_results[0].result.output=cm.failure_output("RefusedOutput", "Unavailable")
    monkeypatch.setattr(engineer, "collaborative_reasoning_main", lambda **kw:failed)
    monkeypatch.setattr(engineer, "self_consistency", lambda *args, **kw:failed)
    monkeypatch.setattr(cm, "output_type_determination", lambda *args, **kw:task.output_type)
    monkeypatch.setattr(ap, "output_type_determination", lambda *args, **kw:task.output_type)
    result=engineer.main("Solve safely")
    assert result.status == "error" and result.stop_reason == "conversation_error"
    assert result.error_result.error_type == "RefusedOutput"
