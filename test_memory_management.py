import pytest
from unittest.mock import MagicMock, patch
from conversation_manager import (
    ConversationMemory,
    ArgumentItem,
    DecisionItem,
    ToDoItem,
    OutputType,
    analyze_to_do_list,
)
from advanced_prompting import AdvancedPromptEngineer, PromptEngineeringConfig, Task, PlanStep


def test_argument_item_model_and_dict_access():
    item = ArgumentItem(round=1, arguments="Strategy A is faster", speaker="Strategist")
    assert item.round == 1
    assert item.arguments == "Strategy A is faster"
    assert item.speaker == "Strategist"
    # Backwards-compatible dict access
    assert item["round"] == 1
    assert item["arguments"] == "Strategy A is faster"
    assert item.get("speaker") == "Strategist"
    assert item.get("nonexistent", "default") == "default"


def test_decision_item_model_and_dict_access():
    decision = DecisionItem(round=2, decision="Use PostgreSQL", decided_by="Mediator")
    assert decision.round == 2
    assert decision.decision == "Use PostgreSQL"
    assert decision.decided_by == "Mediator"
    # Backwards-compatible dict access
    assert decision["decision"] == "Use PostgreSQL"
    assert decision.get("decided_by") == "Mediator"


def test_to_do_item_model_and_str():
    todo = ToDoItem(task="Write integration tests", assigned_to="Tester")
    assert todo.task == "Write integration tests"
    assert todo.status == "pending"
    assert str(todo) == "Write integration tests"
    assert todo["status"] == "pending"
    assert todo.get("assigned_to") == "Tester"


def test_conversation_memory_typing_and_ingestion():
    memory = ConversationMemory(num_rounds=3)

    # Ingest facts
    memory.add_fact("System requires Python 3.12")
    memory.add_fact(["PostgreSQL is selected", "Docker is preferred"])
    # Duplicate fact should be ignored
    memory.add_fact("System requires Python 3.12")
    assert len(memory.facts) == 3

    # Ingest arguments via string, dict, and model
    memory.add_argument("Initial proposal", round_num=1, speaker="Engineer")
    memory.add_argument({"round": 1, "arguments": "Security concern", "speaker": "SecurityLead"})
    memory.add_argument(ArgumentItem(round=2, arguments="Performance optimization", speaker="Architect"))
    assert len(memory.arguments) == 3
    assert all(isinstance(arg, ArgumentItem) for arg in memory.arguments)

    # Ingest decisions
    memory.add_decision("Adopt microservices", round_num=1)
    memory.add_decision({"round": 2, "decision": "Adopt GraphQL", "decided_by": "Mediator"})
    memory.add_decision(DecisionItem(round=2, decision="Finalize schema"))
    assert len(memory.decisions) == 3
    assert all(isinstance(dec, DecisionItem) for dec in memory.decisions)

    # Ingest to-do items
    memory.add_to_do_item("Draft schema")
    memory.add_to_do_item({"task": "Configure Docker", "assigned_to": "DevOps"})
    memory.add_to_do_item(ToDoItem(task="Setup CI/CD"))
    assert len(memory.to_do_list) == 3
    assert all(isinstance(t, ToDoItem) for t in memory.to_do_list)

    # Complete a task
    memory.add_completed_task("Draft schema", "Schema created in schema.graphql")
    assert "Draft schema" in memory.completed_tasks
    matching_todo = next(t for t in memory.to_do_list if t.task == "Draft schema")
    assert matching_todo.status == "completed"


def test_conversation_memory_serialization_roundtrip():
    memory = ConversationMemory(num_rounds=2)
    memory.add_fact("PostgreSQL database")
    memory.add_argument("Argument 1", round_num=1, speaker="Alice")
    memory.add_decision("Decision 1", round_num=1, decided_by="Mediator")
    memory.add_direct_reply("Alice", "I agree with Bob")
    memory.add_recommended_action("Review logs")
    memory.add_to_do_item("Deploy staging")
    memory.add_completed_task("Setup local env", "Done via make setup")
    memory.set_decided_output_type(OutputType(output_type="JSON", file_extension=".json"))

    data = memory.to_dict()
    assert isinstance(data, dict)
    assert data["facts"] == ["PostgreSQL database"]
    assert len(data["arguments"]) == 1
    assert data["arguments"][0]["speaker"] == "Alice"
    assert data["decisions"][0]["decided_by"] == "Mediator"
    assert data["decided_output_type"]["output_type"] == "JSON"

    # Deserialize back
    restored = ConversationMemory.from_dict(data)
    assert restored.facts == memory.facts
    assert len(restored.arguments) == 1
    assert isinstance(restored.arguments[0], ArgumentItem)
    assert restored.arguments[0].speaker == "Alice"
    assert len(restored.decisions) == 1
    assert isinstance(restored.decisions[0], DecisionItem)
    assert restored.decisions[0].decision == "Decision 1"
    assert restored.direct_replies == {"Alice": "I agree with Bob"}
    assert restored.recommended_actions == ["Review logs"]
    assert len(restored.to_do_list) == 1
    assert isinstance(restored.to_do_list[0], ToDoItem)
    assert restored.to_do_list[0].task == "Deploy staging"
    assert restored.completed_tasks == {"Setup local env": "Done via make setup"}
    assert restored.decided_output_type.output_type == "JSON"


def test_conversation_memory_compaction_and_summary_bounding():
    memory = ConversationMemory(num_rounds=5)
    for i in range(10):
        memory.add_fact(f"Fact {i}")
        memory.add_argument(f"Argument {i}", round_num=i // 2 + 1)
        memory.add_recommended_action(f"Action {i}")

    # Summary with max items bounding
    summary = memory.get_memory_summary(max_items_per_category=3)
    assert "latest 3" in summary
    assert "7 earlier omitted" in summary
    assert "Fact 9" in summary
    assert "Fact 0" not in summary

    # Summary with token bounding
    summary_small_tokens = memory.get_memory_summary(max_tokens=10)
    assert "[Summary Compacted due to token limit]" in summary_small_tokens

    # In-place compaction
    memory.compact_memory(max_items_per_category=4)
    assert len(memory.facts) == 4
    assert len(memory.arguments) == 4
    assert len(memory.recommended_actions) == 4
    assert memory.facts[0] == "Fact 6"
    assert memory.facts[-1] == "Fact 9"


def test_analyze_to_do_list_with_typed_items():
    memory = ConversationMemory(num_rounds=2)
    memory.add_to_do_item(ToDoItem(task="Analyze metrics"))
    memory.add_to_do_item("Run benchmark")

    history = [
        {"role": "user", "name": "User", "content": "Start conversation"},
        {"role": "assistant", "name": "Analyst", "content": "I finished analyzing metrics."},
    ]

    # Mock OpenAI client beta.chat.completions.parse call
    mock_task_completion = MagicMock()
    mock_task_completion.task = "Analyze metrics"
    mock_task_completion.completed = True
    mock_task_completion.completion_artifact = "I finished analyzing metrics."

    mock_parsed_result = MagicMock()
    mock_parsed_result.tasks = [mock_task_completion]

    mock_choice = MagicMock()
    mock_choice.message.parsed = mock_parsed_result

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    with patch("conversation_manager.client.beta.chat.completions.parse", return_value=mock_response):
        updated_memory = analyze_to_do_list(memory, history)
        assert updated_memory is memory
        assert "Analyze metrics" in memory.completed_tasks
        assert memory.completed_tasks["Analyze metrics"] == "I finished analyzing metrics."


def test_advanced_prompting_metadata_retention():
    config = PromptEngineeringConfig()
    engineer = AdvancedPromptEngineer(config)
    assert engineer.last_conversation_history is None
    assert engineer.last_questions_asked is None

    plan_step = PlanStep(
        step_number=1,
        completed=False,
        step_name="Design database",
        step_description="Design database",
        step_explanation="Design database schema",
        step_output="Database schema",
        step_full_text="Design database schema",
        subtasks=[],
    )
    task = Task(
        description="Build API",
        refined_description="Build API",
        complexity=3,
        steps=[],
        reflections=[],
        answer="",
        final_reward=1.0,
        plan=None,
        output_type=None,
    )

    sample_history = [{"role": "assistant", "name": "Architect", "content": "Schema drafted"}]
    sample_questions = ["Does the schema support multi-tenancy?"]
    sample_path = "output/step_1.txt"

    with patch("conversation_manager.run_conversation", return_value=(sample_history, sample_questions, sample_path)):
        with patch.object(engineer, "convert_conversation_output_to_final_step_output", return_value=MagicMock()):
            engineer.invoke_collaborative_reasoning(task, plan_step)
            # Verify metadata retention
            assert engineer.last_conversation_history == sample_history
            assert engineer.last_questions_asked == sample_questions


def test_empty_memory_summary():
    memory = ConversationMemory(num_rounds=4)
    summary = memory.get_memory_summary()
    assert summary == "Rounds Left in Conversation to Complete Solution to Problem Statement: 4"
    data = memory.to_dict()
    assert data["rounds_left"] == 4
    assert data["facts"] == []
    assert data["arguments"] == []
    assert data["decisions"] == []
    assert data["to_do_list"] == []
    assert data["decided_output_type"] is None


def test_from_dict_partial_and_empty_payloads():
    # Empty payload
    empty_mem = ConversationMemory.from_dict({})
    assert empty_mem.rounds_left == 3
    assert empty_mem.facts == []
    assert empty_mem.arguments == []
    assert empty_mem.decisions == []
    assert empty_mem.to_do_list == []
    assert empty_mem.decided_output_type is None

    # Partial payload with custom fields and extra unknown keys
    partial_data = {
        "rounds_left": 1,
        "facts": ["Standalone Fact"],
        "direct_replies": {"Lead": "Acknowledged"},
        "unknown_future_field": "Ignored",
    }
    restored = ConversationMemory.from_dict(partial_data)
    assert restored.rounds_left == 1
    assert restored.facts == ["Standalone Fact"]
    assert restored.direct_replies == {"Lead": "Acknowledged"}


def test_memory_edge_case_inputs():
    memory = ConversationMemory(num_rounds=1)

    # None and empty strings should be gracefully ignored
    memory.add_fact(None)
    memory.add_fact("")
    memory.add_fact("   ")
    assert len(memory.facts) == 0

    memory.add_argument(None)
    memory.add_decision(None)
    memory.add_to_do_item(None)
    assert len(memory.arguments) == 0
    assert len(memory.decisions) == 0
    assert len(memory.to_do_list) == 0

    memory.add_direct_reply(None, "reply")
    memory.add_direct_reply("assistant", None)
    memory.add_direct_reply("", "")
    assert len(memory.direct_replies) == 0

    memory.add_recommended_action(None)
    memory.add_recommended_action("")
    assert len(memory.recommended_actions) == 0

    # Decrement rounds past zero
    memory.decrement_rounds_left()
    assert memory.rounds_left == 0
    memory.decrement_rounds_left()
    assert memory.rounds_left == -1


def test_argument_item_complex_arguments_list():
    item = ArgumentItem(
        round=3,
        arguments=["Performance is 2x faster", "Memory usage is lower"],
        speaker="Optimizer",
    )
    assert isinstance(item.arguments, list)
    assert len(item.arguments) == 2
    assert item["arguments"] == ["Performance is 2x faster", "Memory usage is lower"]

    memory = ConversationMemory(num_rounds=2)
    memory.add_argument(item)
    summary = memory.get_memory_summary()
    assert "Performance is 2x faster" in summary
    assert "Memory usage is lower" in summary


def test_to_do_item_status_transitions():
    memory = ConversationMemory(num_rounds=2)
    todo = ToDoItem(task="Setup database schema", status="pending", assigned_to="Architect")
    memory.add_to_do_item(todo)

    summary_before = memory.get_memory_summary()
    assert "To Do List:" in summary_before
    assert "Setup database schema" in summary_before
    assert "Completed Tasks:" not in summary_before

    # Complete task
    memory.add_completed_task("Setup database schema", "Completed in schema.sql")
    assert todo.status == "completed"
    assert "Setup database schema" in memory.completed_tasks

    summary_after = memory.get_memory_summary()
    assert "Completed Tasks:" in summary_after
    assert "Setup database schema: Completed in schema.sql" in summary_after
    assert "To Do List:" not in summary_after


def test_memory_compaction_edge_cases():
    memory = ConversationMemory(num_rounds=5)

    # When fewer than max items exist, compaction is a no-op
    memory.add_fact("Single fact")
    memory.compact_memory(max_items_per_category=5)
    assert len(memory.facts) == 1

    # When completed to-dos exceed 5, compact keeps only the 5 newest
    for i in range(10):
        todo = ToDoItem(task=f"Completed task {i}", status="completed")
        memory.add_to_do_item(todo)

    # And 3 active pending tasks
    for i in range(3):
        todo = ToDoItem(task=f"Pending task {i}", status="pending")
        memory.add_to_do_item(todo)

    assert len(memory.to_do_list) == 13
    memory.compact_memory(max_items_per_category=5)
    # 3 pending + 5 capped completed = 8 items
    assert len(memory.to_do_list) == 8
    pending_tasks = [t.task for t in memory.to_do_list if t.status == "pending"]
    assert pending_tasks == ["Pending task 0", "Pending task 1", "Pending task 2"]


def test_analyze_to_do_list_empty_memory():
    memory = ConversationMemory(num_rounds=2)
    result = analyze_to_do_list(memory, [])
    assert result == "No to-do items were identified in the conversation memory."


def test_analyze_to_do_list_mixed_formats():
    memory = ConversationMemory(num_rounds=2)
    memory.add_to_do_item(ToDoItem(task="Write Dockerfile"))
    memory.add_to_do_item(["Run", "security", "scan"])
    memory.add_to_do_item({"task": "Configure secrets"})
    memory.add_to_do_item("Verify TLS")

    mock_comp1 = MagicMock()
    mock_comp1.task = "Write Dockerfile"
    mock_comp1.completed = True
    mock_comp1.completion_artifact = "Dockerfile created"

    mock_comp2 = MagicMock()
    mock_comp2.task = "Verify TLS"
    mock_comp2.completed = False
    mock_comp2.completion_artifact = "Not yet verified"

    mock_parsed_result = MagicMock()
    mock_parsed_result.tasks = [mock_comp1, mock_comp2]

    mock_choice = MagicMock()
    mock_choice.message.parsed = mock_parsed_result

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    with patch("conversation_manager.client.beta.chat.completions.parse", return_value=mock_response):
        history = [{"role": "assistant", "name": "DevOps", "content": "I finished the Dockerfile."}]
        updated_memory = analyze_to_do_list(memory, history)
        assert updated_memory is memory
        assert "Write Dockerfile" in memory.completed_tasks
        assert "Verify TLS" not in memory.completed_tasks


def test_analyze_to_do_list_exception_handling():
    memory = ConversationMemory(num_rounds=2)
    memory.add_to_do_item("Important task")

    with patch("conversation_manager.client.beta.chat.completions.parse", side_effect=RuntimeError("API failure")):
        with pytest.raises(Exception) as exc_info:
            analyze_to_do_list(memory, [])
        assert "Error in analyze_to_do_list" in str(exc_info.value)


def test_advanced_prompting_invalid_run_conversation_shapes():
    config = PromptEngineeringConfig()
    engineer = AdvancedPromptEngineer(config)
    plan_step = PlanStep(
        step_number=1,
        completed=False,
        step_name="Step 1",
        step_description="Desc",
        step_explanation="Exp",
        step_output="Out",
        step_full_text="Full",
        subtasks=[],
    )
    task = Task(
        description="Task",
        refined_description="Task",
        complexity=1,
        steps=[],
        reflections=[],
        answer="",
        final_reward=1.0,
        plan=None,
        output_type=None,
    )

    invalid_returns = [
        "not a tuple",
        (1, 2),
        (1, 2, 3, 4),
        None,
    ]

    for invalid_val in invalid_returns:
        with patch("conversation_manager.run_conversation", return_value=invalid_val):
            with pytest.raises(ValueError) as exc_info:
                engineer.invoke_collaborative_reasoning(task, plan_step)
            assert "Expected run_conversation to return a 3-tuple" in str(exc_info.value)


def test_advanced_prompting_empty_metadata_retention():
    config = PromptEngineeringConfig()
    engineer = AdvancedPromptEngineer(config)
    plan_step = PlanStep(
        step_number=1,
        completed=False,
        step_name="Step 1",
        step_description="Desc",
        step_explanation="Exp",
        step_output="Out",
        step_full_text="Full",
        subtasks=[],
    )
    task = Task(
        description="Task",
        refined_description="Task",
        complexity=1,
        steps=[],
        reflections=[],
        answer="",
        final_reward=1.0,
        plan=None,
        output_type=None,
    )

    with patch("conversation_manager.run_conversation", return_value=([], [], "output/path.txt")):
        with patch.object(engineer, "convert_conversation_output_to_final_step_output", return_value=MagicMock()):
            engineer.invoke_collaborative_reasoning(task, plan_step)
            assert engineer.last_conversation_history == []
            assert engineer.last_questions_asked == []


def test_determine_output_type_extensions_and_sniffing():
    config = PromptEngineeringConfig()
    engineer = AdvancedPromptEngineer(config)
    task = Task(
        description="Task",
        refined_description="Task",
        complexity=1,
        steps=[],
        reflections=[],
        answer="",
        final_reward=1.0,
        plan=None,
        output_type=None,
    )

    # 1. By file extension
    assert engineer.determine_output_type_from_content("", "script.py", task).output_type == "Python Script"
    assert engineer.determine_output_type_from_content("", "data.json", task).output_type == "JSON"
    assert engineer.determine_output_type_from_content("", "page.html", task).output_type == "HTML"
    assert engineer.determine_output_type_from_content("", "sheet.csv", task).output_type == "CSV"
    assert engineer.determine_output_type_from_content("", "doc.pdf", task).output_type == "PDF"

    # 2. Content sniffing when file_path is None or extensionless
    py_code = "def calculate_sum(a, b):\n    return a + b\n"
    assert engineer.determine_output_type_from_content(py_code, None, task).output_type == "Python Script"

    json_str = '{"status": "ok", "count": 42}'
    assert engineer.determine_output_type_from_content(json_str, "", task).output_type == "JSON"

    html_code = "<html><body><h1>Hello World</h1></body></html>"
    assert engineer.determine_output_type_from_content(html_code, None, task).output_type == "HTML"

    # 3. Default fallback
    assert engineer.determine_output_type_from_content("Just plain text with no markers", None, task).output_type == "Text File"

