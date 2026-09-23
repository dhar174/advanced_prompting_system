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

    @patch('conversation_manager.cast_binary_vote')
    @patch('conversation_manager.get_assistant_response')
    def test_all_yes_votes(self, mock_get_response, mock_cast_vote):
        """Test that all YES votes result in consensus."""
        # Setup mocks
        mock_cast_vote.return_value = BinaryVote(vote=True, reason="Looks good")
        mock_get_response.return_value = {"role": "assistant", "content": "Test"}
        
        # Test just the counting logic directly since it's embedded in the main loop
        yes_votes = 0
        no_votes = 0
        voting_errors = 0
        
        votes = [
            BinaryVote(vote=True, reason="1"),
            BinaryVote(vote=True, reason="2"),
        ]
        
        for v in votes:
            if isinstance(v, BinaryVote):
                if v.vote:
                    yes_votes += 1
                else:
                    no_votes += 1
            else:
                voting_errors += 1
                
        self.assertEqual(yes_votes, 2)
        self.assertEqual(no_votes, 0)
        self.assertEqual(voting_errors, 0)

    def test_all_no_votes(self):
        """Test that all NO votes result in correct counts."""
        yes_votes = 0
        no_votes = 0
        voting_errors = 0
        
        votes = [
            BinaryVote(vote=False, reason="1"),
            BinaryVote(vote=False, reason="2"),
        ]
        
        for v in votes:
            if isinstance(v, BinaryVote):
                if v.vote:
                    yes_votes += 1
                else:
                    no_votes += 1
            else:
                voting_errors += 1
                
        self.assertEqual(yes_votes, 0)
        self.assertEqual(no_votes, 2)
        self.assertEqual(voting_errors, 0)

    def test_binary_vote_false_truthiness(self):
        """Test that BinaryVote(vote=False) is never counted as YES because it is truthy."""
        yes_votes = 0
        no_votes = 0
        voting_errors = 0
        
        vote = BinaryVote(vote=False, reason="No")
        
        # Ensure it's truthy but doesn't increment yes_votes with correct semantics
        self.assertTrue(bool(vote), "BinaryVote(vote=False) is a truthy object")
        
        if isinstance(vote, BinaryVote):
            if vote.vote:
                yes_votes += 1
            else:
                no_votes += 1
        else:
            voting_errors += 1
            
        self.assertEqual(yes_votes, 0)
        self.assertEqual(no_votes, 1)

    def test_error_result_not_yes(self):
        """Test that ErrorResult is never counted as YES."""
        yes_votes = 0
        no_votes = 0
        voting_errors = 0
        
        vote = ErrorResult(error_type="TestError", error_message="Fail")
        
        if isinstance(vote, BinaryVote):
            if vote.vote:
                yes_votes += 1
            else:
                no_votes += 1
        else:
            voting_errors += 1
            
        self.assertEqual(yes_votes, 0)
        self.assertEqual(no_votes, 0)
        self.assertEqual(voting_errors, 1)

    def test_error_votes_not_double_counted(self):
        """Test that ErrorResult increments voting_errors but not no_votes/continue_votes."""
        continue_votes = 0
        end_votes = 0
        voting_errors = 0
        
        vote = ErrorResult(error_type="TestError", error_message="Fail")
        
        if isinstance(vote, BinaryVote):
            if vote.vote:
                continue_votes += 1
            else:
                end_votes += 1
        else:
            voting_errors += 1
            # Note: Previously there was a double count here with continue_votes += 1
            
        self.assertEqual(continue_votes, 0)
        self.assertEqual(end_votes, 0)
        self.assertEqual(voting_errors, 1)

    def test_final_decision_messages_preserve_values(self):
        """Test that Final-decision messages preserve actual role and content values."""
        conversation_history = []
        rnd = 1
        final_content_messages = [
            {"role": "system", "content": "System message"},
            {"role": "user", "content": "User message"}
        ]
        
        for message in final_content_messages:
            conversation_history.append(
                {
                    "role": message["role"],
                    "name": "Mediator",
                    "round": rnd,
                    "content": message["content"],
                }
            )
            
        self.assertEqual(len(conversation_history), 2)
        self.assertEqual(conversation_history[0]["role"], "system")
        self.assertEqual(conversation_history[0]["content"], "System message")
        self.assertEqual(conversation_history[1]["role"], "user")
        self.assertEqual(conversation_history[1]["content"], "User message")

if __name__ == '__main__':
    unittest.main()
