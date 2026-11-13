import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add the project directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from conversation_manager import (
    BinaryVote,
    ErrorResult,
)
import conversation_manager

class TestVotingSemantics(unittest.TestCase):

    def test_cast_binary_vote_yes(self):
        """Test that cast_binary_vote returns parsed BinaryVote(vote=True) on successful affirmative vote."""
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(parsed=BinaryVote(vote=True)))
        ]
        with patch.object(conversation_manager.client.beta.chat.completions, "parse", return_value=mock_response):
            vote = conversation_manager.cast_binary_vote("Strategist", [], "Confirm?")
        self.assertIsInstance(vote, BinaryVote)
        self.assertTrue(vote.vote)

    def test_cast_binary_vote_no(self):
        """Test that cast_binary_vote returns parsed BinaryVote(vote=False) on negative vote."""
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(parsed=BinaryVote(vote=False)))
        ]
        with patch.object(conversation_manager.client.beta.chat.completions, "parse", return_value=mock_response):
            vote = conversation_manager.cast_binary_vote("Strategist", [], "Confirm?")
        self.assertIsInstance(vote, BinaryVote)
        self.assertFalse(vote.vote)

    def test_cast_binary_vote_error_result(self):
        """Test that cast_binary_vote catches exceptions and returns an ErrorResult."""
        with patch.object(conversation_manager.client.beta.chat.completions, "parse", side_effect=RuntimeError("API error")):
            result = conversation_manager.cast_binary_vote("Strategist", [], "Confirm?")
        self.assertIsInstance(result, ErrorResult)
        self.assertEqual(result.error_type, "LLMCallError")
        self.assertIn("API error", result.error_details)

    def test_define_problem_all_yes_votes(self):
        """Test that define_problem reaches consensus without revision when all binary votes are YES."""
        from conversation_manager import ConfidenceVote
        mock_llm_response = MagicMock()
        mock_llm_response.choices = [MagicMock(message=MagicMock(content="Proposed problem statement"))]

        with patch.object(conversation_manager.client.chat.completions, "create", return_value=mock_llm_response), \
             patch("conversation_manager.cast_confidence_vote", return_value=ConfidenceVote(confidence=0.9)), \
             patch("conversation_manager.cast_binary_vote", return_value=BinaryVote(vote=True)), \
             patch("conversation_manager.get_assistant_response") as mock_get_response:
            result = conversation_manager.define_problem("Raw prompt", {"Strategist": "prompt"})
            self.assertEqual(result, "Proposed problem statement")
            # Revision should NOT be called since all votes were YES
            mock_get_response.assert_not_called()

    def test_define_problem_binary_vote_false_truthiness(self):
        """Test that BinaryVote(vote=False) is never counted as YES despite being a truthy object."""
        from conversation_manager import ConfidenceVote
        mock_llm_response = MagicMock()
        mock_llm_response.choices = [MagicMock(message=MagicMock(content="Original problem statement"))]

        false_vote = BinaryVote(vote=False)
        # Ensure it is truthy as a Pydantic object
        self.assertTrue(bool(false_vote))

        revision_reply = {"role": "assistant", "content": "Revision: Better problem statement"}

        with patch.object(conversation_manager.client.chat.completions, "create", return_value=mock_llm_response), \
             patch("conversation_manager.cast_confidence_vote", return_value=ConfidenceVote(confidence=0.9)), \
             patch("conversation_manager.cast_binary_vote", side_effect=[false_vote, BinaryVote(vote=True)]), \
             patch("conversation_manager.get_assistant_response", return_value=revision_reply) as mock_get_response:
            result = conversation_manager.define_problem("Raw prompt", {"Strategist": "prompt"})
            # Verify revision WAS invoked because vote=False was not counted as YES
            self.assertTrue(mock_get_response.called)
            self.assertEqual(result, "Better problem statement")

    def test_define_problem_error_result_not_counted_as_yes(self):
        """Test that ErrorResult is tracked as an error and never counted as YES."""
        from conversation_manager import ConfidenceVote
        mock_llm_response = MagicMock()
        mock_llm_response.choices = [MagicMock(message=MagicMock(content="Original problem statement"))]

        err_vote = ErrorResult(error_type="LLMCallError", error_message="Failed")

        with patch.object(conversation_manager.client.chat.completions, "create", return_value=mock_llm_response), \
             patch("conversation_manager.cast_confidence_vote", return_value=ConfidenceVote(confidence=0.9)), \
             patch("conversation_manager.cast_binary_vote", return_value=err_vote), \
             patch("conversation_manager.get_assistant_response") as mock_get_response:
            # When all votes error out, define_problem detects zero valid votes and raises an exception
            with self.assertRaises(Exception):
                conversation_manager.define_problem("Raw prompt", {"Strategist": "prompt"})
            mock_get_response.assert_not_called()

    # Real final-decision history assertions live in
    # test_pr34_regressions::test_final_vote_branches_execute_production.

if __name__ == '__main__':
    unittest.main()
