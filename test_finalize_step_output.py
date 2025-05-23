import re
import unittest
from unittest.mock import Mock, patch
from advanced_prompting import (
    AdvancedPromptEngineer,
    Step,
    Task,
    PlanStep,
    FinalStepOutput,
    PromptEngineeringConfig,
    OutputType,
)
from complexity_measures import Plan


class TestFinalizeStepOutput(unittest.TestCase):
    @patch("advanced_prompting.openai.chat.completions.create")
    def test_finalize_step_output_success(self, mock_create):
        # Mock the OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Finalized step output"))]
        mock_create.return_value = mock_response

        # Create sample objects
        step = Step(
            description="Define the Function Structure",
            step_number=1,
            remaining_budget=3,
        )
        plan_step = PlanStep(
            step_number=1,
            completed=False,
            step_name="Define the Function Structure",
            step_description="Define the Function Structure",
            step_explanation="Define the Function Structure",
            step_output="Define the Function Structure",
            step_full_text="Define the Function Structure",
            subtasks=[],
        )

        task = Task(
            description="Create a Python function that takes two numbers as input and returns their sum",
            refined_description="Create a Python function that takes two numbers as input and returns their sum",
            complexity=0.5,
            steps=[step],
            reflections=[],
            answer="",
            final_reward=0.0,
            plan=Plan(
                steps=[plan_step],
            ),
            output_type=OutputType(output_type="python", file_extension=".py"),
            project_name="Addition Function",
        )
        previous_steps = []

        # Initialize AdvancedPrompting
        config = PromptEngineeringConfig()

        ap = AdvancedPromptEngineer(config)

        # Call the method
        result = ap.finalize_step_output(step, task, plan_step, previous_steps)

        # Assertions
        self.assertIsInstance(result, FinalStepOutput)
        self.assertEqual(
            result.final_output, "Finalized step output"
        ), f"Final output is {result.final_output}"
        self.assertEqual(
            result.output_type.file_extension, ".py"
        ), f"Output type is {result.output_type.file_extension}"
        self.assertEqual(
            result.component_type, "standalone_file"
        ), f"Component type is {result.component_type}"


if __name__ == "__main__":
    unittest.main()
