from types import SimpleNamespace
from unittest.mock import Mock
import pytest
import conversation_manager as cm
import complexity_measures as complexity
import advanced_prompting as ap


def response(parsed=None, content='Fixture proposal'):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed, content=content))])


@pytest.mark.parametrize('field,entry', [
    ('arguments', 'Legacy argument'), ('arguments', ['One', 'Two']),
    ('to_do_list', ['Run', 'tests']), ('decisions', 'Legacy decision'),
])
def test_memory_legacy_forms_round_trip(field, entry):
    expected = cm.ConversationMemory(3)
    ingest = {'arguments': expected.add_argument, 'to_do_list': expected.add_to_do_item, 'decisions': expected.add_decision}
    ingest[field](entry)
    restored = cm.ConversationMemory.from_dict({field: [entry]})
    assert restored.to_dict()[field] == expected.to_dict()[field]
    assert cm.ConversationMemory.from_dict(restored.to_dict()).to_dict() == restored.to_dict()


def test_partial_majority_returns_selected_definition(monkeypatch):
    monkeypatch.setattr(cm.client.chat.completions, 'create', lambda **kw: response(content='Selected definition'))
    monkeypatch.setattr(cm, 'cast_confidence_vote', lambda *a: cm.ConfidenceVote(confidence=.9))
    votes = iter([True, True, True, False])
    monkeypatch.setattr(cm, 'cast_binary_vote', lambda *a: cm.BinaryVote(vote=next(votes)))
    monkeypatch.setattr(cm, 'get_assistant_response', lambda *a: {'role': 'assistant', 'content': 'Revision: Selected definition'})
    assert cm.define_problem('Original task', dict.fromkeys(['A','B','C','D'], 'prompt')) == 'Selected definition'


@pytest.mark.parametrize('state', [None, False])
@pytest.mark.parametrize('name', ['is_real_words','is_complex_graph','is_complex_recursive','is_complex_ast','is_complex_llm'])
def test_optional_spacy_unavailable_does_not_call_provider(monkeypatch, name, state):
    monkeypatch.setattr(complexity, 'nlp_spacy', state)
    monkeypatch.setattr(complexity.spacy, 'load', Mock(side_effect=OSError('model absent')))
    def no_generation(*args, **kwargs):
        pytest.fail('Unavailable NLP should return before model generation')
    monkeypatch.setattr(complexity, 'generate_plan_legacy', no_generation)
    result = getattr(complexity, name)('Keep original text')
    assert result == ('Keep original text' if name == 'is_real_words' else 0.0)




def test_low_participation_finalizes(conversation_case, monkeypatch):
    votes = iter([True, True, None, None, None])
    def vote(*a):
        value = next(votes)
        return cm.BinaryVote(vote=value) if value is not None else cm.ErrorResult(error_type='fixture', error_message='failed')
    monkeypatch.setattr(cm, 'cast_binary_vote', vote)
    history, questions, output = cm.run_conversation('task', ['A','B','C','D','E'], 'A', num_rounds=1)
    assert output == 'fixture-output.txt'
    conversation_case[0].assert_called_once()
    assert any(m['content'].startswith('Final Output:') for m in history)


def test_all_error_continue_vote_enters_bounded_fallback(conversation_case, monkeypatch):
    def vote(name, messages, prompt):
        if 'continue the conversation' in prompt:
            return cm.ErrorResult(error_type='fixture', error_message='failed')
        return cm.BinaryVote(vote=True)
    monkeypatch.setattr(cm, 'cast_binary_vote', vote)
    history, _, output = cm.run_conversation('task', ['A'], 'A', num_rounds=2)
    assert output == 'fixture-output.txt'
    assert any(m['round'] == 0 and m['content'].startswith('Final Decision:') for m in history)


@pytest.mark.parametrize('votes', [[True,True], [False,False], [True,False], [None,None], [True,True,True,False,False]])
def test_final_vote_branches_execute_production(conversation_case, monkeypatch, votes):
    values = iter(votes)
    def vote(*a):
        value = next(values)
        return cm.BinaryVote(vote=value) if value is not None else cm.ErrorResult(error_type='fixture', error_message='failed')
    monkeypatch.setattr(cm, 'cast_binary_vote', vote)
    history, _, output = cm.run_conversation('task', [f'A{i}' for i in range(len(votes))], 'A0', num_rounds=1)
    assert output == 'fixture-output.txt'
    if votes != [True, True]:
        assert cm.MediatorReviseOrDecide in conversation_case[1]
    assert any(m['role'] == 'user' and m['content'].startswith('Please provide your final decision or solution based on the information available:') for m in history)
    assert any(m['role'] == 'system' and m['content'] == 'Problem Statement: Fixture task' for m in history)


def test_main_calls_real_adapter_with_planstep(monkeypatch):
    engineer = ap.AdvancedPromptEngineer(ap.PromptEngineeringConfig())
    step = ap.PlanStep(step_number=1, completed=False, step_name='Build', step_description='Build output', step_explanation='Build', step_output='output', step_full_text='Build output', subtasks=[])
    plan = complexity.Plan(steps=[step])
    monkeypatch.setattr(engineer, 'retrieve_information', lambda *a: 'context')
    monkeypatch.setattr(engineer, 'task_into_prompt', lambda text: text)
    monkeypatch.setattr(ap, 'output_type_determination', lambda *a: cm.OutputType(output_type='Text File', file_extension='.txt'))
    monkeypatch.setattr(engineer, 'assess_complexity', lambda *a: (.5, plan))
    monkeypatch.setattr(engineer, 'name_project', lambda *a: 'fixture')
    monkeypatch.setattr(engineer, 'adjust_step_budget', lambda *a: 1)
    monkeypatch.setattr(engineer, 'generate_initial_prompt', lambda *a, **kw: ('User task', 'System instructions'))
    monkeypatch.setattr(engineer, 'convert_planstep_to_prompt', lambda *a: 'Build output')
    monkeypatch.setattr(engineer, 'condense_plan', lambda *a: 'Build output')
    class AdapterReached(BaseException):
        pass
    def choose(actual_step, task):
        assert actual_step is step
        assert task.description == 'Build output'
        raise AdapterReached()
    monkeypatch.setattr(engineer, 'should_use_multi_agent_reasoning', choose)
    with pytest.raises(AdapterReached):
        engineer.main('Actual user task')


@pytest.mark.parametrize('votes', [[False,False], [True,False], [None,None]])
def test_dissent_revision_reaches_final_generation(conversation_case, monkeypatch, votes):
    values = iter(votes)
    def vote(*a):
        value = next(values)
        return cm.BinaryVote(vote=value) if value is not None else cm.ErrorResult(error_type='fixture', error_message='failed')
    monkeypatch.setattr(cm, 'cast_binary_vote', vote)
    def parse(**kw):
        if kw['response_format'] is cm.OutputType:
            return response(cm.OutputType(output_type='Text File', file_extension='.txt'))
        return response(cm.MediatorReviseOrDecide(final_decision='Revised fixture decision', keep_output=False))
    monkeypatch.setattr(cm.client.beta.chat.completions, 'parse', parse)
    calls = []
    def assistant(*args, **kw):
        calls.append(str(args))
        return {'role': 'assistant', 'content': 'Fixture output'}
    monkeypatch.setattr(cm, 'get_assistant_response', assistant)
    history, _, output = cm.run_conversation('task', ['A','B'], 'A', num_rounds=1)
    assert output == 'fixture-output.txt'
    assert any('Revised fixture decision' in c for c in calls)
    assert any(m['content'] == 'Final Decision: Revised fixture decision (Revised)' for m in history)
    conversation_case[0].assert_called_once()


@pytest.mark.parametrize('votes', [[False,False], [True,False], [None,None]])
def test_mediator_failure_terminates_with_fallback(conversation_case, monkeypatch, votes):
    values = iter(votes)
    def vote(*a):
        value = next(values)
        return cm.BinaryVote(vote=value) if value is not None else cm.ErrorResult(error_type='fixture', error_message='failed')
    monkeypatch.setattr(cm, 'cast_binary_vote', vote)
    mediator_calls = []
    def parse(**kw):
        if kw['response_format'] is cm.OutputType:
            return response(cm.OutputType(output_type='Text File', file_extension='.txt'))
        mediator_calls.append(kw)
        raise RuntimeError('Fixture mediator failure')
    monkeypatch.setattr(cm.client.beta.chat.completions, 'parse', parse)
    history, _, output = cm.run_conversation('task', ['A','B'], 'A', num_rounds=1)
    assert output == 'fixture-output.txt'
    assert len(mediator_calls) == 1
    conversation_case[0].assert_called_once()
