"""
Contract test for GraphQL runConversation mutation and Track 1 execution modes (Issue #32, Issue #35, Issue #21).
Validates that the Strawberry GraphQL schema resolver returns the exact schema shape
expected by the frontend React client.
"""

import unittest
from unittest.mock import patch
import strawberry
from schema import Query, Mutation
from execution_modes import (
    RunResult,
    RunDiagnostics,
    RunEvent,
    PlanStepDiagnostic,
    SubtaskDiagnostic,
    ReflectionDiagnostic,
    AgentAttribution,
    CAPABILITY_REGISTRY,
)
from conversation_manager import OutputType, ErrorResult

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

    def test_query_execution_modes(self):
        """Verify Query.executionModes returns registered modes and capability flags."""
        query = """
        query GetExecutionModes {
            executionModes {
                id
                label
                description
                enabled
                gatedIssue
                supportsMultiAgent
                supportsStepPlanning
            }
        }
        """
        result = self.schema.execute_sync(query)
        self.assertIsNone(result.errors, f"Errors: {result.errors}")
        self.assertIsNotNone(result.data)
        modes = result.data.get("executionModes")
        self.assertIsInstance(modes, list)
        self.assertEqual(len(modes), 3)

        mode_map = {m["id"]: m for m in modes}
        self.assertIn("collaborative", mode_map)
        self.assertIn("advanced_pipeline", mode_map)
        self.assertIn("tree_of_thought", mode_map)

        self.assertTrue(mode_map["collaborative"]["enabled"])
        self.assertIsNone(mode_map["collaborative"]["gatedIssue"])

        self.assertTrue(mode_map["advanced_pipeline"]["enabled"])
        self.assertTrue(mode_map["advanced_pipeline"]["supportsStepPlanning"])

        self.assertFalse(mode_map["tree_of_thought"]["enabled"])
        self.assertEqual(mode_map["tree_of_thought"]["gatedIssue"], "Issue #36")

    def test_mutation_run_execution_tree_of_thought_gated(self):
        """Verify Mutation.runExecution handles gated tree_of_thought mode cleanly."""
        query = """
        mutation RunExecutionTest($input: RunExecutionInput!) {
            runExecution(input: $input) {
                mode
                status
                finalOutput
                errorMessage
                outputType {
                    outputType
                    fileExtension
                }
                events {
                    eventType
                    mode
                    summary
                }
            }
        }
        """
        variables = {
            "input": {
                "mode": "tree_of_thought",
                "taskPrompt": "Prove P != NP",
            }
        }
        result = self.schema.execute_sync(query, variable_values=variables)
        self.assertIsNone(result.errors, f"Errors: {result.errors}")
        self.assertIsNotNone(result.data)
        run_data = result.data["runExecution"]
        self.assertEqual(run_data["mode"], "tree_of_thought")
        self.assertEqual(run_data["status"], "gated")
        self.assertIn("Issue #36", run_data["errorMessage"])
        self.assertEqual(run_data["outputType"]["outputType"], "Simple Concise Answer")
        self.assertTrue(any(e["eventType"] == "mode_gated" for e in run_data["events"]))

    def test_mutation_run_execution_collaborative_contract(self):
        """Verify Mutation.runExecution maps full collaborative results and telemetry."""
        query = """
        mutation RunExecutionTest($input: RunExecutionInput!) {
            runExecution(input: $input) {
                mode
                status
                finalOutput
                outputType {
                    outputType
                    fileExtension
                }
                conversation {
                    role
                    name
                    content
                }
                questions {
                    assistant
                    question
                }
                diagnostics {
                    complexityScore
                    finalReward
                    agentAttributions {
                        agentName
                        role
                        contributionSummary
                    }
                }
                events {
                    eventType
                    summary
                }
            }
        }
        """
        variables = {
            "input": {
                "mode": "collaborative",
                "taskPrompt": "Collaborate on system architecture",
                "assistantPersonalities": ["Strategist", "Coding Guru"],
                "leadPersonality": "Strategist",
                "numRounds": 2,
            }
        }

        mock_result = RunResult(
            mode="collaborative",
            status="completed",
            final_output="Final Architecture Blueprint",
            output_type=OutputType(output_type="Python Script", file_extension=".py"),
            conversation=[
                {"role": "user", "name": "User", "content": "Collaborate on system architecture"},
                {"role": "assistant", "name": "Strategist", "content": "Strategy proposal"},
            ],
            questions=[{"assistant": "Strategist", "question": "What is the expected QPS?"}],
            diagnostics=RunDiagnostics(
                complexity_score=0.75,
                final_reward=0.9,
                agent_attributions=[
                    AgentAttribution(
                        agent_name="Strategist",
                        role="lead",
                        round_or_step=1,
                        contribution_summary="Proposed modular service mesh",
                    )
                ],
            ),
            events=[
                RunEvent(event_type="mode_selected", mode="collaborative", summary="Selected collaborative mode"),
                RunEvent(event_type="output_finalized", mode="collaborative", summary="Completed deliberation"),
            ],
        )

        with patch("execution_modes.execute_run", return_value=mock_result):
            result = self.schema.execute_sync(query, variable_values=variables)

        self.assertIsNone(result.errors, f"Errors: {result.errors}")
        self.assertIsNotNone(result.data)
        run_data = result.data["runExecution"]
        self.assertEqual(run_data["mode"], "collaborative")
        self.assertEqual(run_data["status"], "completed")
        self.assertEqual(run_data["finalOutput"], "Final Architecture Blueprint")
        self.assertEqual(run_data["outputType"]["outputType"], "Python Script")
        self.assertEqual(run_data["outputType"]["fileExtension"], ".py")
        self.assertEqual(len(run_data["conversation"]), 2)
        self.assertEqual(len(run_data["questions"]), 1)
        self.assertEqual(run_data["diagnostics"]["complexityScore"], 0.75)
        self.assertEqual(len(run_data["diagnostics"]["agentAttributions"]), 1)
        self.assertEqual(len(run_data["events"]), 2)

    def test_mutation_run_advanced_pipeline(self):
        """Verify Mutation.runAdvancedPipeline convenience method."""
        query = """
        mutation RunAdvancedPipelineTest($taskPrompt: String!, $maxSteps: Int) {
            runAdvancedPipeline(taskPrompt: $taskPrompt, maxSteps: $maxSteps) {
                mode
                status
                finalOutput
                diagnostics {
                    planSteps {
                        stepNumber
                        stepName
                        subtasks {
                            subtaskNumber
                            subtaskName
                            completed
                        }
                    }
                    reflections {
                        stepNumber
                        reward
                        content
                    }
                }
            }
        }
        """
        variables = {
            "taskPrompt": "Implement secure authentication middleware",
            "maxSteps": 15,
        }

        mock_result = RunResult(
            mode="advanced_pipeline",
            status="completed",
            final_output="JWT Auth Middleware implemented.",
            output_type=OutputType(output_type="Python Script", file_extension=".py"),
            conversation=[
                {"role": "user", "name": "User", "content": "Implement secure authentication middleware"},
                {"role": "assistant", "name": "AdvancedPromptEngineer", "content": "Pipeline completed."},
            ],
            diagnostics=RunDiagnostics(
                complexity_score=0.82,
                plan_steps=[
                    PlanStepDiagnostic(
                        step_number=1,
                        step_name="Validate Claims",
                        step_description="Verify JWT signature and expiry",
                        subtasks=[
                            SubtaskDiagnostic(
                                subtask_number=1,
                                subtask_name="Check signature",
                                completed=True,
                            )
                        ],
                        completed=True,
                    )
                ],
                reflections=[
                    ReflectionDiagnostic(
                        step_number=1,
                        content="Claims validation succeeded with high confidence",
                        reward=0.95,
                    )
                ],
            ),
        )

        with patch("execution_modes.execute_run", return_value=mock_result):
            result = self.schema.execute_sync(query, variable_values=variables)

        self.assertIsNone(result.errors, f"Errors: {result.errors}")
        self.assertIsNotNone(result.data)
        pipeline_data = result.data["runAdvancedPipeline"]
        self.assertEqual(pipeline_data["mode"], "advanced_pipeline")
        self.assertEqual(pipeline_data["status"], "completed")
        self.assertEqual(pipeline_data["finalOutput"], "JWT Auth Middleware implemented.")
        self.assertEqual(len(pipeline_data["diagnostics"]["planSteps"]), 1)
        self.assertEqual(pipeline_data["diagnostics"]["planSteps"][0]["stepName"], "Validate Claims")
        self.assertEqual(pipeline_data["diagnostics"]["planSteps"][0]["subtasks"][0]["subtaskName"], "Check signature")
        self.assertEqual(pipeline_data["diagnostics"]["reflections"][0]["reward"], 0.95)

if __name__ == "__main__":
    unittest.main()
