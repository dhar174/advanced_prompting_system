import pytest
from conversation_manager import normalize_message, ErrorResult, run_conversation

class MockConversationInput:
    def __init__(self, content, role="assistant", name=""):
        self.content = content
        self.role = role
        self.name = name

def test_normalize_message_dict():
    msg = {"role": "user", "content": "hello", "name": "Alice"}
    normalized = normalize_message(msg)
    assert normalized == {"role": "user", "content": "hello", "name": "Alice"}

def test_normalize_message_string():
    msg = "just a string"
    normalized = normalize_message(msg)
    assert normalized == {"role": "assistant", "content": "just a string", "name": ""}

def test_normalize_message_object():
    msg = MockConversationInput("object content", "system", "Bot")
    normalized = normalize_message(msg)
    assert normalized == {"role": "system", "content": "object content", "name": "Bot"}

def test_normalize_message_error_result():
    msg = ErrorResult(error_type="TestError", error_message="Something failed")
    normalized = normalize_message(msg)
    assert normalized == {"role": "system", "content": "Error: Something failed", "name": "error"}

def test_normalize_message_missing_content():
    msg = {"role": "user"}
    normalized = normalize_message(msg)
    assert normalized["content"] == ""
    assert normalized["role"] == "user"

def test_normalize_message_missing_role():
    msg = {"content": "hello"}
    normalized = normalize_message(msg)
    assert normalized["role"] == "assistant"
    assert normalized["content"] == "hello"

import conversation_manager
from unittest.mock import patch

def test_problem_statement_normalization_string():
    with patch('conversation_manager.define_problem', return_value="def") as spy:
        try:
            conversation_manager.run_conversation("String problem", {"A": "prompt A"}, "A", num_rounds=1)
        except Exception:
            pass
        assert spy.called
        assert spy.call_args[0][0].endswith("String problem")

def test_problem_statement_normalization_list():
    with patch('conversation_manager.define_problem', return_value="def") as spy:
        try:
            conversation_manager.run_conversation([{"content": "List dict problem"}], {"A": "prompt A"}, "A", num_rounds=1)
        except Exception:
            pass
        assert spy.called
        assert spy.call_args[0][0].endswith("List dict problem")

def test_problem_statement_normalization_dict():
    with patch('conversation_manager.define_problem', return_value="def") as spy:
        try:
            conversation_manager.run_conversation({"content": "Dict problem"}, {"A": "prompt A"}, "A", num_rounds=1)
        except Exception:
            pass
        assert spy.called
        assert spy.call_args[0][0].endswith("Dict problem")


from conversation_manager import (
    safe_get_message_field,
    safe_get_feedback_field,
    validate_message_structure,
)


def test_safe_get_message_field_edge_cases():
    # Valid dict
    assert safe_get_message_field({"content": "hello"}, "content") == "hello"
    # Missing key with default
    assert safe_get_message_field({"role": "user"}, "content", "empty") == "empty"
    # Non-dict inputs
    assert safe_get_message_field(None, "content", "default") == "default"
    assert safe_get_message_field("string", "content", "default") == "default"
    assert safe_get_message_field(12345, "content", "default") == "default"
    assert safe_get_message_field([], "content", "default") == "default"


def test_safe_get_feedback_field_edge_cases():
    # Valid feedback dict
    assert safe_get_feedback_field({"feedback": "looks good"}, "feedback") == "looks good"
    # Non-dict inputs
    assert safe_get_feedback_field(None, "feedback", "none") == "none"
    assert safe_get_feedback_field("feedback text", "feedback", "none") == "none"
    assert safe_get_feedback_field(["feedback"], "feedback", "none") == "none"


def test_validate_message_structure_edge_cases():
    # Valid message
    assert validate_message_structure({"role": "assistant", "content": "valid"}) is True
    assert validate_message_structure({"role": "user", "content": "", "name": "Alice"}) is True

    # Missing role or content
    assert validate_message_structure({"role": "assistant"}) is False
    assert validate_message_structure({"content": "no role"}) is False
    assert validate_message_structure({}) is False

    # Non-dict inputs
    assert validate_message_structure(None) is False
    assert validate_message_structure("string message") is False
    assert validate_message_structure([{"role": "user", "content": "nested"}]) is False


def test_normalize_message_numeric_and_boolean_content():
    # Numbers
    assert normalize_message(42)["content"] == "42"
    assert normalize_message(3.1415)["content"] == "3.1415"

    # Booleans
    assert normalize_message(True)["content"] == "True"
    assert normalize_message(False)["content"] == "False"

    # Custom defaults
    norm = normalize_message(None, default_role="system", default_name="Auditor")
    assert norm["role"] == "system"
    assert norm["name"] == "Auditor"
    assert norm["content"] == "None"

