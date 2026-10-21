"""
Contract test for GraphQL runConversation mutation (Issue #32, Issue #14).
Validates that the Strawberry GraphQL schema resolver returns the exact schema shape
expected by the frontend React client.
"""

import unittest
from unittest.mock import patch
import strawberry
from schema import Query, Mutation

class TestGraphQLContract(unittest.TestCase):
    def setUp(self):
        self.schema = strawberry.Schema(query=Query, mutation=Mutation)

    def test_run_conversation_contract(self):
        """Verify GraphQL mutation runConversation matches frontend contract."""
        query = """
        mutation RunConversationTest(
            $conversation: [ConversationInput!]!,
            $assistantPersonalities: [String!]!,
            $leadPersonality: String!,
            $numRounds: Int!
        ) {
            runConversation(
                conversation: $conversation,
                assistantPersonalities: $assistantPersonalities,
                leadPersonality: $leadPersonality,
                numRounds: $numRounds
            ) {
                conversation {
                    role
                    name
                    content
                }
                questions {
                    assistant
                    question
                }
                finalOutput
            }
        }
        """
        variables = {
            "conversation": [
                {"role": "user", "name": "User", "content": "Test prompt"}
            ],
            "assistantPersonalities": ["Strategist", "Mediator"],
            "leadPersonality": "Strategist",
            "numRounds": 1,
        }

        mocked_backend_return = (
            [
                {
                    "role": "assistant",
                    "name": "Strategist",
                    "content": "Test response",
                }
            ],
            [
                {
                    "assistant": "Educator",
                    "reply": "What is the expected complexity?",
                    "asking_assistant": "Strategist",
                }
            ],
            "Test final output",
        )

        with patch("conversation_manager.run_conversation", return_value=mocked_backend_return):
            result = self.schema.execute_sync(query, variable_values=variables)

        self.assertIsNone(result.errors, f"GraphQL execution returned errors: {result.errors}")
        self.assertIsNotNone(result.data)
        
        expected_data = {
            "runConversation": {
                "conversation": [
                    {
                        "role": "assistant",
                        "name": "Strategist",
                        "content": "Test response",
                    }
                ],
                "questions": [
                    {
                        "assistant": "Educator",
                        "question": "What is the expected complexity?",
                    }
                ],
                "finalOutput": "Test final output",
            }
        }
        self.assertEqual(result.data, expected_data)

    def test_submit_feedback_contract(self):
        """Verify GraphQL mutation submitFeedback."""
        query = """
        mutation SubmitFeedbackTest(
            $conversation: [ConversationInput!]!,
            $feedback: [FeedbackInput!]!
        ) {
            submitFeedback(
                conversation: $conversation,
                feedback: $feedback
            ) {
                success
            }
        }
        """
        variables = {
            "conversation": [{"role": "user", "name": "User", "content": "Hello"}],
            "feedback": [{"assistant": "Strategist", "rating": 5}],
        }
        with patch("feedback_manager.log_conversation", return_value=None):
            result = self.schema.execute_sync(query, variable_values=variables)

        self.assertIsNone(result.errors)
        self.assertEqual(result.data, {"submitFeedback": {"success": "Feedback submitted successfully."}})

if __name__ == "__main__":
    unittest.main()
