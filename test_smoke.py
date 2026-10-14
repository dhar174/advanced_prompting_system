"""
Production smoke test suite (Issue #32).
Fast, reliable checks with zero external network or OpenAI calls:
1. conversation_manager.py compiles/imports;
2. output-type fallback returns the configured fallback for an intentionally unrelated type;
3. file generators propagate write failures predictably;
4. run_conversation() contract is represented correctly in mocks/fixtures (3-tuple);
5. AdvancedPromptEngineer.invoke_collaborative_reasoning() consumes the production-shaped tuple;
6. GraphQL runConversation returns the schema shape expected by the frontend.
"""

import pytest
import unittest
from unittest.mock import patch, mock_open, MagicMock
import py_compile
import os
import strawberry

class TestSmokeSuite(unittest.TestCase):

    def test_1_conversation_manager_compiles_and_imports(self):
        """1. Verify conversation_manager.py compiles and imports cleanly."""
        file_path = os.path.join(os.path.dirname(__file__), "conversation_manager.py")
        py_compile.compile(file_path, doraise=True)
        import conversation_manager
        self.assertIsNotNone(conversation_manager.OutputType)
        self.assertIsNotNone(conversation_manager.DEFAULT_OUTPUT_TYPE)

    def test_2_output_type_fallback_unrelated_type(self):
        """2. Verify output-type fallback returns configured default for unrelated type."""
        from conversation_manager import (
            DEFAULT_FALLBACK_OUTPUT_TYPE,
            match_output_type_to_sample,
        )

        unrelated_type = "999999999999999"
        selected_type = match_output_type_to_sample(unrelated_type)
        self.assertEqual(selected_type, DEFAULT_FALLBACK_OUTPUT_TYPE)

    def test_3_file_generators_propagate_write_failures(self):
        """3. File generators propagate write failures predictably."""
        from output_generator import generate_json_output, generate_text_file_output

        with patch("builtins.open", side_effect=OSError("Disk full or permission denied")):
            with self.assertRaises(OSError):
                generate_json_output({"key": "val"}, "fake_output.json")
            with self.assertRaises(OSError):
                generate_text_file_output("Some text", "fake_output.txt")

    @pytest.mark.usefixtures("conversation_case")
    def test_4_run_conversation_contract_mock_shape(self):
        """4. Verify run_conversation contract shape is a 3-tuple when executed with mocked LLM boundaries."""
        import conversation_manager
        from conversation_manager import BinaryVote

        with patch("conversation_manager.define_problem", return_value="Test problem statement"), \
             patch("conversation_manager.cast_binary_vote", return_value=BinaryVote(vote=True, reason="OK")), \
             patch("conversation_manager.get_assistant_response", return_value={"role": "assistant", "content": "Done"}), \
             patch("conversation_manager.analyze_final_output", return_value="Checked output"), \
             patch("conversation_manager.finalize_output", return_value="path/to/final_output.txt"), \
             patch("builtins.open", mock_open()):
            production_tuple = conversation_manager.run_conversation(
                problem_statement="Test statement",
                selected_personalities={"Strategist": "You are a strategist"},
                lead_personality="Strategist",
                num_rounds=1
            )

        self.assertIsInstance(production_tuple, tuple)
        self.assertEqual(len(production_tuple), 3)
        self.assertIsInstance(production_tuple[0], list)
        self.assertIsInstance(production_tuple[1], list)
        self.assertEqual(production_tuple[2], "path/to/final_output.txt")

    def test_5_invoke_collaborative_reasoning_consumes_tuple(self):
        """5. AdvancedPromptEngineer.invoke_collaborative_reasoning consumes 3-tuple properly."""
        from advanced_prompting import (
            AdvancedPromptEngineer,
            PromptEngineeringConfig,
            Task,
            PlanStep,
            FinalStepOutput,
        )

        config = PromptEngineeringConfig()
        engineer = AdvancedPromptEngineer(config)
        task = Task(
            description="Build a web app",
            refined_description="Build a web app",
            complexity=1,
            steps=[],
            reflections=[],
            answer="",
            final_reward=1.0,
            plan=None,
            output_type=None,
        )
        plan_step = PlanStep(
            step_number=1,
            completed=False,
            step_name="Design frontend",
            step_description="Design frontend",
            step_explanation="Design frontend",
            step_output="Frontend structure",
            step_full_text="Design frontend",
            subtasks=[],
        )

        dummy_history = [{"role": "assistant", "content": "Done"}]
        dummy_questions = []
        dummy_path = "output.txt"
        production_tuple = (dummy_history, dummy_questions, dummy_path)

        mock_output_obj = MagicMock()
        with patch("conversation_manager.run_conversation", return_value=production_tuple):
            with patch.object(
                engineer,
                "convert_conversation_output_to_final_step_output",
                return_value=mock_output_obj,
            ) as mock_converter:
                result = engineer.invoke_collaborative_reasoning(
                    task=task,
                    plan_step=plan_step,
                )
                # Verify that ONLY the path (dummy_path) was passed to converter, NOT the tuple!
                mock_converter.assert_called_once()
                self.assertEqual(mock_converter.call_args[0][0], dummy_path)
                self.assertEqual(result, mock_output_obj)

    def test_6_graphql_schema_contract(self):
        """6. GraphQL runConversation returns exact schema shape expected by frontend."""
        from schema import Query, Mutation

        schema = strawberry.Schema(query=Query, mutation=Mutation)
        query = """
        mutation TestSmoke($conversation: [ConversationInput!]!, $assistantPersonalities: [String!]!, $leadPersonality: String!, $numRounds: Int!) {
            runConversation(conversation: $conversation, assistantPersonalities: $assistantPersonalities, leadPersonality: $leadPersonality, numRounds: $numRounds) {
                conversation { role name content }
                questions { assistant question }
                finalOutput
            }
        }
        """
        variables = {
            "conversation": [{"role": "user", "name": "User", "content": "Hi"}],
            "assistantPersonalities": ["Strategist"],
            "leadPersonality": "Strategist",
            "numRounds": 1,
        }
        mock_result = (
            [{"role": "assistant", "name": "Strategist", "content": "Hello"}],
            [],
            "Final output string",
        )
        with patch("conversation_manager.run_conversation", return_value=mock_result):
            res = schema.execute_sync(query, variable_values=variables)

        self.assertIsNone(res.errors)
        self.assertIn("runConversation", res.data)
        rc = res.data["runConversation"]
        self.assertIn("conversation", rc)
        self.assertIn("questions", rc)
        self.assertIn("finalOutput", rc)
        self.assertEqual(rc["finalOutput"], "Final output string")

if __name__ == "__main__":
    unittest.main()
