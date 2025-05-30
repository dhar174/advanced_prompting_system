from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch, mock_open # Ensure mock_open is imported
import os # Ensure os is imported

from matplotlib.pyplot import step # This import seems unused, might be from old code
from advanced_prompting import (
    AdvancedPromptEngineer,
    FinalPlanStepOutput,
    FinalStepOutput,
    PromptEngineeringConfig,
    Task,
    Step,
    Reflection,
    Interaction,
    OutputType,
    ComponentType # Ensure ComponentType is imported
)
import re
from complexity_measures import Plan, PlanStep
import test_c # These seem to be related to the custom config, might not be needed for new tests
import test_b # These seem to be related to the custom config, might not be needed for new tests
from pydantic import BaseModel


class config_for_test:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.engineer = None
        """
        test_b for test_b.py
        test_c for test_c.py
        """
        self.parser_arg = (
            "default" if kwargs.get("parser_arg") is None else kwargs.get("parser_arg")
        )
        if self.parser_arg != "default":
            self.parser = (
                test_b.parse_response
                if self.parser_arg == "test_b"
                else test_c.parse_response
            )
        else:
            self.parser = None

    def get_engineer(self):
        if self.engineer is None:
            self.config = PromptEngineeringConfig()
            self.engineer = AdvancedPromptEngineer(self.config)
            self.parser = self.engineer.parse_response
        return self.engineer

# Assuming TEST_CONFIG might be defined elsewhere or was part of a larger setup.
# For the new tests, we'll rely on setUp in TestAdvancedPromptingAdapterMethods.
# If TEST_CONFIG is essential for the whole file, this might need adjustment.
# For now, proceeding with the new test class being self-contained.
TEST_CONFIG = config_for_test(parser_arg="default") # Minimal instantiation for old tests to potentially run

class TestParseResponse(unittest.TestCase):

    def setUp(self):
        """Initialize test environment with config, engineer and sample task."""
        self.parser_arg = "default"
        self.config = TEST_CONFIG
        self.parser_arg = TEST_CONFIG.parser_arg
        self.output_type = OutputType(output_type="text", file_extension="txt")
        # self.output_type.output_type = "text" # .output_type is not directly assignable for Pydantic
        # self.output_type.file_extension = "txt"
        if self.config is None:
            self.config = config_for_test(parser_arg=self.parser_arg)
        self.parser = None
        if self.config.parser_arg != "default":
            if self.config.parser_arg == "test_b":
                self.config.parser = test_b.parse_response
            elif self.config.parser_arg == "test_c":
                self.config.parser = test_c.parse_response
        elif self.config.parser_arg == "default":
            self.config.get_engineer() # This will initialize self.config.engineer and self.config.parser
            self.parser = self.config.parser


        if self.parser is None: # Fallback if still None
             # Ensure engineer is created if parser is still None
            if not hasattr(self.config, 'engineer') or self.config.engineer is None:
                self.config.get_engineer()
            self.parser = self.config.parser


        print(f"Parser: {self.parser_arg} \n {self.config.parser.__name__}")

        self.sample_task = Task(
            description="Sample test task description",
            refined_description="Refined test task description",
            complexity=1,
            steps=[],  # Steps are added in the tests
            reflections=[],
            answer="",
            final_reward=0.0,
            plan=Plan(
                steps=[
                    PlanStep(
                        step_number=1,
                        completed=False,
                        step_name="Initialize Test Case",
                        step_description="Create initial test case structure",
                        step_explanation="Set up the basic test framework with required imports and test class",
                        step_output="TestCase class with setUp method",
                        step_full_text="Initialize test case by creating class structure and importing dependencies",
                        subtasks=[],
                    ),
                    PlanStep(
                        step_number=2,
                        completed=False,
                        step_name="Write Test Case",
                        step_description="Write test case for the function",
                        step_explanation="Write test case for the function to be tested",
                        step_output="Test case for the function",
                        step_full_text="Write test case for the function to be tested",
                        subtasks=[],
                    ),
                ]
            ),
            output_type=self.output_type,
        )

    def test_none_response_handling(self):
        response = None
        interaction = self.config.parser(response, self.sample_task)
        print(f"Interaction test_none_response_handling: {interaction} \n")

        self.assertEqual(len(interaction.steps), 0)
        self.assertEqual(len(interaction.reflections), 0)
        self.assertEqual(interaction.answer, "")
        self.assertEqual(interaction.final_reward, 0.0)

    def test_non_string_response_handling(self):
        response = 12345  # Non-string response
        interaction = self.config.parser(response, self.sample_task)
        print(f"Interaction test_non_string_response_handling: {interaction} \n")

        self.assertEqual(len(interaction.steps), 0)
        self.assertEqual(len(interaction.reflections), 0)
        self.assertEqual(interaction.answer, "")
        self.assertEqual(interaction.final_reward, 0.0)

    # ... (rest of TestParseResponse methods remain unchanged) ...
    def test_reflection_missing_reward(self):
        response = """
        <count>5</count>
        <thinking>Thinking...</thinking>
        <step>Test step</step>
        <reflection>Test reflection without reward</reflection>
        """
        interaction = self.config.parser(response, self.sample_task)
        print(f"Interaction test_reflection_missing_reward: {interaction} \n")

        self.assertEqual(len(interaction.reflections), 1)
        self.assertEqual(
            interaction.reflections[0].content, "Test reflection without reward"
        )
        self.assertEqual(interaction.reflections[0].reward, 0.7)

    def test_reflection_with_special_characters(self):
        response = """
        <count>5</count>
        <thinking>Thinking...</thinking>
        <step>Test step</step>
        <reflection>Reflection with special characters: <>&"</reflection>
        <reward>0.8</reward>
        """
        interaction = self.config.parser(response, self.sample_task)
        print(f"Interaction test_reflection_with_special_characters: {interaction} \n")

        self.assertEqual(len(interaction.reflections), 1)
        self.assertEqual(
            interaction.reflections[0].content,
            'Reflection with special characters: <>&"',
        )
        self.assertEqual(interaction.reflections[0].reward, 0.8)

    def test_non_numeric_count_and_reward(self):
        response = """
        <count>abc</count>
        <thinking>Thinking...</thinking>
        <step>Test step</step>
        <reflection>Test reflection</reflection>
        <reward>xyz</reward>
        """
        interaction = self.config.parser(response, self.sample_task)
        print(f"Interaction test_non_numeric_count_and_reward: {interaction} \n")

        self.assertEqual(len(interaction.steps), 1)
        self.assertEqual(interaction.steps[0].remaining_budget, 11) # Default initial budget - 1 if count fails
        self.assertEqual(interaction.reflections[0].reward, 0.7) # Default reward if parsing fails

    def test_steps_and_reflections_objs_provided(self):
        existing_steps = [
            Step(
                description="Existing step",
                step_number=1,
                remaining_budget=4,
                reflection=None, # Will be populated by judge_step if not found
            )
        ]
        existing_reflections = [
            Reflection(content="Existing reflection", reward=0.5, step_number=1)
        ]
        # Link reflection to step
        existing_steps[0].reflection = existing_reflections[0]


        response = """
        <count>4</count>
        <thinking>Thinking...</thinking>
        <step>New step</step>
        <reflection>New reflection</reflection>
        <reward>0.7</reward>
        """

        interaction = self.config.parser(
            response,
            self.sample_task,
            steps_objs=existing_steps,
            reflections_objs=existing_reflections,
            initial_budget=4 # to align with count
        )
        print(f"Interaction test_steps_and_reflections_objs_provided: {interaction} \n")

        self.assertEqual(len(interaction.steps), 2)
        self.assertEqual(interaction.steps[0].description, "Existing step")
        self.assertEqual(interaction.steps[0].reflection.content, "Existing reflection")
        self.assertEqual(interaction.steps[1].description, "New step")
        self.assertEqual(interaction.steps[1].reflection.content, "New reflection")
        self.assertEqual(interaction.steps[1].reflection.reward, 0.7)
        self.assertEqual(interaction.steps[1].remaining_budget, 3) # 4 - 1
        self.assertEqual(interaction.reflections[0].reward, 0.5)
        self.assertEqual(interaction.steps[0].reflection.reward, 0.5)


    def test_overlapping_steps_and_reflections(self):
        existing_steps = [
            Step(
                description="Overlapping step",
                step_number=1,
                remaining_budget=4,
                reflection=Reflection(
                    content="Initial reflection", reward=0.6, step_number=1 # Corrected step_number
                ),
            )
        ]

        response = """
        <count>5</count>
        <thinking>Thinking...</thinking>
        <step>Overlapping step</step>
        <reflection>Updated reflection</reflection>
        <reward>0.8</reward>
        <answer>Final answer</answer>
        <final_reward>0.9</final_reward>
        """

        interaction = self.config.parser(
            response, self.sample_task, steps_objs=existing_steps, initial_budget=5
        )
        print(f"Interaction test_overlapping_steps_and_reflections: {interaction} \n")

        self.assertEqual(len(interaction.steps), 1)
        self.assertEqual(interaction.steps[0].description, "Overlapping step")
        self.assertEqual(interaction.steps[0].reflection.content, "Updated reflection") # Updated
        self.assertEqual(interaction.steps[0].reflection.reward, 0.8) # Updated
        self.assertEqual(interaction.answer, "Final answer")
        self.assertEqual(interaction.final_reward, 0.9)

    def test_unexpected_tags_in_response(self):
        response = """
        <count>5</count>
        <thinking>Thinking...</thinking>
        <unexpected>Unexpected content</unexpected>
        <step>Test step</step>
        <reflection>Test reflection</reflection>
        <reward>0.9</reward>
        """
        interaction = self.config.parser(response, self.sample_task, plan_step_number=1)
        print(f"Interaction test_unexpected_tags_in_response: {interaction} \n")

        self.assertEqual(len(interaction.steps), 1)
        self.assertEqual(interaction.steps[0].description, "Test step")
        self.assertEqual(interaction.steps[0].reflection.content, "Test reflection")
        self.assertEqual(interaction.steps[0].reflection.reward, 0.9)

    def test_multiple_answers_and_rewards(self):
        response = """
        <count>5</count>
        <thinking>Thinking...</thinking>
        <step>First step</step>
        <answer>First answer</answer>
        <final_reward>0.7</final_reward>
        <count>4</count>
        <thinking>Thinking more...</thinking>
        <step>Second step</step>
        <answer>Second answer</answer>
        <final_reward>0.85</final_reward>
        """

        interaction = self.config.parser(response, self.sample_task)
        print(f"Interaction test_multiple_answers_and_rewards: {interaction} \n")
        
        self.assertEqual(len(interaction.steps), 2)
        self.assertEqual(interaction.steps[0].description, "First step")
        self.assertEqual(interaction.steps[1].description, "Second step")
        # The parser should only take the first answer and final_reward
        self.assertEqual(interaction.answer, "First answer") 
        self.assertEqual(interaction.final_reward, 0.7)


    def test_missing_count_tag(self):
        response = """
        <thinking>Thinking...</thinking>
        <step>Test step</step>
        <reflection>Test reflection</reflection>
        <reward>0.8</reward>
        """
        interaction = self.config.parser(response, self.sample_task, initial_budget=5)
        print(f"Interaction test_missing_count_tag: {interaction} \n")

        self.assertEqual(len(interaction.steps), 1)
        self.assertEqual(interaction.steps[0].remaining_budget, 4) # initial_budget - 1

    def test_missing_step_tag(self):
        response = """
        <count>5</count>
        <thinking>Thinking...</thinking>
        <reflection>Test reflection without step</reflection>
        <reward>0.8</reward>
        """
        interaction = self.config.parser(response, self.sample_task)
        print(f"Interaction test_missing_step_tag: {interaction} \n")

        self.assertEqual(len(interaction.steps), 0) # No step tag, so no step object created
        self.assertEqual(len(interaction.reflections), 0) # Reflections are tied to steps

    def test_empty_step_and_reflection_tags(self):
        response = """
        <count>5</count>
        <thinking>Thinking...</thinking>
        <step></step>
        <reflection></reflection>
        <reward>0.8</reward>
        """
        interaction = self.config.parser(response, self.sample_task)
        print(f"Interaction test_empty_step_and_reflection_tags: {interaction} \n")

        self.assertEqual(len(interaction.steps), 1)
        self.assertEqual(interaction.steps[0].description, "")
        self.assertEqual(interaction.steps[0].reflection.content, "")
        self.assertEqual(interaction.steps[0].reflection.reward, 0.8)

    def test_basic_response_parsing(self):
        response = """
        <count>5</count>
        <thinking>Thinking...</thinking>
        <step>First step</step>
        <reflection>Good first step</reflection>
        <reward>0.8</reward>
        <answer>Final answer</answer>
        <final_reward>0.9</final_reward>"""

        interaction = self.config.parser(response, self.sample_task)
        print(f"Interaction test_basic_response_parsing: {interaction} \n")

        self.assertIsInstance(interaction, Interaction)
        self.assertEqual(len(interaction.steps), 1)
        self.assertEqual(interaction.steps[0].description, "First step")
        self.assertEqual(interaction.answer, "Final answer")
        self.assertEqual(interaction.final_reward, 0.9)

    def test_multiple_steps_parsing(self):
        response = """
        <count>5</count>
        <thinking>Thinking...</thinking>
        <step>First step</step>
        <reflection>First reflection</reflection>
        <reward>0.8</reward>
        <count>4</count>
        <thinking>Thinking...</thinking>
        <step>Second step</step>
        <reflection>Second reflection</reflection>
        <reward>0.9</reward>
        <answer>Final answer</answer>
        <final_reward>0.85</final_reward>"""

        interaction = self.config.parser(response, self.sample_task)
        print(f"Interaction test_multiple_steps_parsing: {interaction} \n")

        self.assertEqual(len(interaction.steps), 2)
        self.assertEqual(interaction.steps[0].description, "First step")
        self.assertEqual(interaction.steps[1].description, "Second step")
        self.assertEqual(len(interaction.reflections), 2) # Each step should have a reflection

    def test_missing_reflection_handling(self):
        response = """
        <count>5</count>
        <thinking>Thinking...</thinking>
        <step>Step without reflection</step>
"""

        interaction = self.config.parser(response, self.sample_task)
        print(f"Interaction test_missing_reflection_handling: {interaction} \n")
        self.assertEqual(len(interaction.steps), 1)
        self.assertIsNotNone(interaction.steps[0].reflection) # judge_step should be called
        self.assertEqual(interaction.steps[0].reflection.reward, 0.7) # Default from judge_step if no reward tag

    def test_budget_counting(self):
        response = """
        <count>5</count>
        <thinking>Thinking...</thinking>
        <step>First step</step>
        <count>4</count>
        <thinking>Thinking...</thinking>
        <step>Second step</step>
"""

        interaction = self.config.parser(response, self.sample_task, initial_budget=5)
        print(f"Interaction test_budget_counting: {interaction} \n")

        self.assertEqual(interaction.steps[0].remaining_budget, 4) # 5 - 1
        self.assertEqual(interaction.steps[1].remaining_budget, 3) # 4 - 1


    def test_interaction_merging(self):
        existing_interaction = Interaction(
            task=self.sample_task,
            steps=[
                Step(
                    description="Existing step",
                    step_number=1,
                    remaining_budget=5, # This should be initial_budget - step_number
                    reflection=Reflection(content="Existing reflection", reward=0.75, step_number=1),
                )
            ],
            reflections=[Reflection(content="Existing reflection", reward=0.75, step_number=1)],
            answer="",
            final_reward=0.0,
        )

        response = """
        <count>5</count>
        <thinking>Thinking...</thinking>
        <step>New step</step>
        <reflection>New reflection</reflection>
        <reward>0.9</reward>"""

        interaction = self.config.parser(
            response, self.sample_task, interaction=existing_interaction, initial_budget=5
        )
        print(
            f"Interaction: {interaction} \n Length of steps: {len(interaction.steps)} \n Steps: {interaction.steps}"
        )

        self.assertEqual(len(interaction.steps), 2) # Should be 1 existing + 1 new
        self.assertEqual(interaction.steps[0].description, "Existing step")
        self.assertEqual(interaction.steps[1].description, "New step")
        self.assertEqual(interaction.steps[1].remaining_budget, 4) # 5 - 1

    def test_reward_score_validation(self):
        response = """
        <count>5</count>
        <thinking>Thinking...</thinking>
        <step>Test step</step>
        <reflection>Test reflection</reflection>
        <reward>invalid</reward>""" # Invalid reward

        interaction = self.config.parser(response, self.sample_task)
        print(f"Interaction test_reward_score_validation: {interaction} \n")

        self.assertEqual(interaction.steps[0].reflection.reward, 0.7) # Default if invalid


    def test_empty_response_handling(self):
        response = ""
        interaction = self.config.parser(response, self.sample_task)
        print(f"Interaction test_empty_response_handling: {interaction} \n")

        self.assertEqual(len(interaction.steps), 0)
        self.assertEqual(len(interaction.reflections), 0)
        self.assertEqual(interaction.answer, "")
        self.assertEqual(interaction.final_reward, 0.0)

    def test_malformed_tags_handling(self):
        response = """
        <count>5</count>
        <thinking>Thinking...</thinking>
        <step>Unclosed step
        <reflection>Unclosed reflection
        <reward>0.8</reward>""" # Tags are not properly closed

        interaction = self.config.parser(response, self.sample_task)
        print(f"Interaction test_malformed_tags_handling: {interaction} \n")

        self.assertEqual(len(interaction.steps), 1) # Parser should still find the step
        self.assertEqual(interaction.steps[0].description, "Unclosed step")
        self.assertIsNotNone(interaction.steps[0].reflection)
        self.assertEqual(interaction.steps[0].reflection.content, "Unclosed reflection")
        self.assertEqual(interaction.steps[0].reflection.reward, 0.8)


    def test_initial_budget_handling(self):
        response = """
        <count>10</count>
        <thinking>Thinking...</thinking>
        <step>First step</step>
"""

        interaction = self.config.parser(response, self.sample_task, initial_budget=10) # Explicitly set initial_budget
        print(f"Interaction test_initial_budget_handling: {interaction} \n")

        self.assertEqual(interaction.steps[0].remaining_budget, 9) # 10 - 1

    def test_existing_steps_integration(self):
        # Arrange
        existing_steps = [
            Step(
                description="Existing step (existing step test)",
                step_number=1,
                remaining_budget=9, # initial_budget 10 - 1
                reflection=Reflection(content="Existing reflection", reward=0.7, step_number=1),
            )
        ]

        response = """
        <count>9</count>
        <thinking>Thinking...</thinking>
        <step>New step (existing step test)</step>
        <reflection>New reflection</reflection>
        <reward>0.9</reward>"""

        # Act
        interaction = self.config.parser(
            response, self.sample_task, steps_objs=existing_steps, initial_budget=10
        )
        print(f"Interaction (existing step test): {interaction} \n")

        # Assert
        self.assertEqual(len(interaction.steps), 2)
        self.assertEqual(
            interaction.steps[0].description, "Existing step (existing step test)"
        )
        self.assertEqual(
            interaction.steps[1].description, "New step (existing step test)"
        )
        self.assertEqual(interaction.steps[1].remaining_budget, 8) # 9-1

    def test_reflection_objects_integration(self):
        # Arrange
        existing_reflections = [
            Reflection(content="Previous reflection", reward=0.8, step_number=1)
        ]
        existing_steps = [
             Step(description="Step for reflection", step_number=1, remaining_budget=3, reflection=existing_reflections[0])
        ]


        response = """
        <count>3</count>
        <thinking>Thinking...</thinking>
        <step>New step after reflection</step>
        <reflection>New reflection content</reflection>
        <reward>0.85</reward>"""

        # Act
        interaction = self.config.parser(
            response,
            self.sample_task,
            steps_objs=existing_steps, # Pass steps_objs that already have reflections
            reflections_objs=existing_reflections, # Also pass reflections_objs for parser to use
            initial_budget=4, # Count starts at 4, new step will be at count 3
        )
        print(f"Interaction (reflection object test): {interaction} \n")

        # Assert
        self.assertEqual(len(interaction.steps), 2) # 1 existing + 1 new
        self.assertEqual(interaction.steps[0].reflection.content, "Previous reflection")
        self.assertEqual(interaction.steps[1].reflection.content, "New reflection content")
        self.assertEqual(interaction.steps[1].remaining_budget, 2) # 3 - 1

    def test_duplicate_step_handling(self):
        response = """
        <count>5</count>
        <thinking>Thinking...</thinking>
        <step>First step</step>
        <reflection>Ref1</reflection><reward>0.1</reward>
        <count>4</count>
        <thinking>Thinking...</thinking>
        <step>First step</step> 
        <reflection>Ref2</reflection><reward>0.2</reward>
"""
        # Parser should create two distinct Step objects if descriptions are identical but are part of different "blocks"
        interaction = self.config.parser(response, self.sample_task, initial_budget=5)
        print(f"Interaction (duplicate step test): {interaction} \n")

        self.assertEqual(len(interaction.steps), 2)
        self.assertEqual(interaction.steps[0].description, "First step")
        self.assertEqual(interaction.steps[1].description, "First step")
        self.assertEqual(interaction.steps[0].step_number, 1)
        self.assertEqual(interaction.steps[1].step_number, 2)
        self.assertEqual(interaction.steps[0].reflection.content, "Ref1")
        self.assertEqual(interaction.steps[1].reflection.content, "Ref2")


    def test_reward_range_validation(self):
        response = """
        <count>5</count>
        <thinking>Thinking...</thinking>
        <step>Test step</step>
        <reflection>Test reflection</reflection>
        <reward>1.5</reward>""" # Reward out of 0.0-1.0 range

        interaction = self.config.parser(response, self.sample_task)

        print(f"Interaction (reward range test): {interaction} \n")
        # The parser should clamp or default the reward.
        # Assuming it defaults to 0.0 or a safe value if parsing fails or clamps.
        # Current consolidate_steps does not clamp, judge_step does.
        # If reflection is parsed directly, it takes the value.
        # If judge_step is called, it ensures the range.
        # Given the response provides the reward, it's taken as is.
        # Let's assume the test implies the final stored reward should be valid.
        # The provided parser logic for <reward> uses a regex that matches 0.0-1.0.
        # If it doesn't match, it defaults. Here, "1.5" won't match "0\.\d+|1\.0".
        self.assertEqual(interaction.steps[0].reflection.reward, 0.7) # Default from judge_step

    def print_parser(self):
        print(f"Parser: {self.parser_arg} \n {self.config.parser.__name__}")
        self.assertEqual(self.config.parser.__name__, "parse_response")


class TestAdvancedPromptEngineer(unittest.TestCase):
    def setUp(self):
        """Initialize test environment with config, engineer and sample task."""
        self.config_container = config_for_test(parser_arg="default") # Use the container
        self.engineer = self.config_container.get_engineer() # Get engineer via container

        self.output_type = OutputType(output_type="python", file_extension=".py")
        # self.output_type.output_type = "python" # Not assignable
        # self.output_type.file_extension = ".py" # Not assignable

        self.sample_task = Task(
            description="Create a simple function to calculate factorial",
            refined_description="Write a Python function that calculates factorial recursively",
            complexity=1,
            steps=[],
            reflections=[],
            answer="",
            final_reward=0.0,
            plan=Plan(
                steps=[
                    PlanStep(
                        step_number=1,
                        completed=False,
                        step_name="Define Function",
                        step_description="Create factorial function signature",
                        step_explanation="Define the factorial function with proper parameters",
                        step_output="Function definition",
                        step_full_text="Create factorial function with proper signature and docstring",
                        subtasks=[],
                    )
                ]
            ),
            output_type=self.output_type,
        )

    def test_count_tokens(self):
        text = ["Hello world", "Testing tokens"]
        token_count = self.engineer.count_tokens(text)
        self.assertIsInstance(token_count, int)
        self.assertTrue(token_count > 0)

    def test_generate_initial_prompt(self):
        task_desc = "Write a function to calculate factorial" # Use task_desc for clarity
        retrieved_info = "Factorial is the product of all positive integers up to n"
        step_budget = 5
        complexity = 2

        initial_prompt, system_prompt = self.engineer.generate_initial_prompt(
            task_desc, retrieved_info, step_budget, complexity, self.output_type
        )

        self.assertIsInstance(initial_prompt, str)
        self.assertIsInstance(system_prompt, str)
        self.assertIn("factorial", initial_prompt.lower())
        self.assertIn("<thinking>", system_prompt) # system_prompt contains the rules with tags
        self.assertIn("<step>", system_prompt)

    def test_judge_step(self):
        step = Step(
            description="Define factorial function with parameter n",
            step_number=1,
            remaining_budget=5, # Example budget
            reflection=None, # Reflection will be generated by judge_step
        )
        reflection = self.engineer.judge_step(step, self.sample_task)

        self.assertIsInstance(reflection, Reflection)
        self.assertTrue(0.0 <= reflection.reward <= 1.0)
        self.assertIsInstance(reflection.content, str)
        self.assertEqual(reflection.step_number, 1)

    def test_component_decision(self):
        # from advanced_prompting import ComponentType # Already imported at top

        task = self.sample_task
        plan_step = task.plan.steps[0]

        decision = self.engineer.component_decision(task, plan_step)
        # Ensure decision is one of the valid string values from ComponentType
        self.assertIn(decision, ComponentType.values())


    def test_judge_step_completion(self):
        step_list = [ # judge_step_completion expects a List[Step]
            Step(
                description="def factorial(n):\n    if n == 0: return 1\n    return n * factorial(n-1)",
                step_number=1,
                remaining_budget=4,
                reflection=None, # Will be judged if needed by other parts, not directly by this
            )
        ]
        completed, next_step = self.engineer.judge_step_completion(
            step_list, self.sample_task.plan.steps[0], max_plan_steps=1
        )

        self.assertIsInstance(completed, bool)
        self.assertIsInstance(next_step, int)


    def test_finalize_step_output(self):
        # from advanced_prompting import FinalStepOutput # Already imported

        step = Step(
            description="def factorial(n):\n    return 1 if n == 0 else n * factorial(n-1)",
            step_number=1,
            remaining_budget=4,
            reflection=None, # Will be generated if needed
            plan_step_number=1,
        )

        output = self.engineer.finalize_step_output(
            step, self.sample_task, self.sample_task.plan.steps[0], [] # Pass empty list for previous_steps
        )

        self.assertIsInstance(output, FinalStepOutput)
        self.assertEqual(output.version, 1)
        self.assertEqual(output.associated_plan_step, self.sample_task.plan.steps[0])


    def test_automatic_chain_of_thought(self):
        task_desc = "Calculate factorial of 5"
        thoughts = self.engineer.automatic_chain_of_thought(task_desc)

        self.assertIsInstance(thoughts, str)
        self.assertTrue(len(thoughts) > 0)
        self.assertIn("factorial", thoughts.lower())


    def test_least_to_most(self):
        task_desc = "Calculate factorial of 5"
        sub_tasks = self.engineer.least_to_most(task_desc)

        self.assertIsInstance(sub_tasks, list)
        self.assertTrue(len(sub_tasks) > 0)
        self.assertTrue(all(isinstance(task_item, str) for task_item in sub_tasks)) # Renamed 'task' to 'task_item'


    def test_progressive_hint(self):
        sub_task = "Define factorial function"
        hints = self.engineer.progressive_hint(sub_task)

        self.assertIsInstance(hints, list)
        self.assertTrue(len(hints) > 0)
        self.assertTrue(all(isinstance(hint, str) for hint in hints))

    def test_assess_complexity(self):
        task_desc = "Write a recursive factorial function with error handling"
        complexity, plan = self.engineer.assess_complexity(task_desc)

        self.assertIsInstance(complexity, (int, float)) # Complexity is float
        self.assertGreater(complexity, 0) # is_complex_final returns float > 0
        self.assertIsInstance(plan, Plan)

    def test_adjust_step_budget(self):
        task_desc = "Write a factorial calculator"
        complexity = 3

        result = self.engineer.adjust_step_budget(task_desc, complexity)

        if isinstance(result, tuple): # is_complex_final might be called again
            budget, plan = result
            self.assertIsInstance(budget, int)
            self.assertIsInstance(plan, Plan)
        else: # Original complexity was non-zero
            self.assertIsInstance(result, int)
            self.assertGreater(result, 0)


    def test_dynamic_confidence_exploration(self):
        interaction = Interaction(
            task=self.sample_task,
            steps=[],
            reflections=[],
            answer="def factorial(n): return 1 if n == 0 else n * factorial(n-1)",
            final_reward=0.4, # Low reward to trigger exploration
        )

        # dynamic_confidence_exploration expects a PlanStep as the third argument
        result = self.engineer.dynamic_confidence_exploration(
            interaction, self.sample_task, self.sample_task.plan.steps[0] # Pass task object and a plan_step
        )

        self.assertIsInstance(result, Interaction)
        self.assertTrue(hasattr(result, "final_reward"))


    def test_merge_interactions(self):
        interaction_a = Interaction(
            task=self.sample_task,
            steps=[Step(description="Step 1", step_number=1, remaining_budget=5)], # Added description
            reflections=[],
            answer="Answer A",
            final_reward=0.8,
        )

        interaction_b = Interaction(
            task=self.sample_task, # Should be the same task object for meaningful merge
            steps=[Step(description="Step 2", step_number=2, remaining_budget=4)], # Added description
            reflections=[],
            answer="Answer B",
            final_reward=0.9,
        )

        merged = self.engineer.merge_interactions(interaction_a, interaction_b)

        self.assertIsInstance(merged, Interaction)
        self.assertEqual(len(merged.steps), 2) # 1 from A, 1 from B
        self.assertEqual(merged.final_reward, 0.8) # Takes A's then B's if A is 0

    def test_choose_best_response(self):
        """Test the choose_best_response method"""

        # Create test data
        mock_responses = [
            "First test response",
            "Second test response with more detail",
            "Third detailed test response with specific steps",
        ]

        test_plan_step = PlanStep(
            step_number=1,
            completed=False,
            step_name="Test Step",
            step_description="Test description",
            step_explanation="Test explanation",
            step_output="Expected output",
            step_full_text="Full text of test step",
            subtasks=[],
        )

        test_interaction = Interaction(
            task=self.sample_task,
            steps=[Step(description="Previous step", step_number=1, remaining_budget=5)], # Added description
            reflections=[
                Reflection(content="Test reflection", reward=0.7, step_number=1)
            ],
            answer="",
            final_reward=0.0,
        )

        # Test normal operation
        response = self.engineer.choose_best_response(
            responses=mock_responses,
            plan_step=test_plan_step,
            interaction=test_interaction,
            step_number=2,
        )
        self.assertIsInstance(response, str)
        self.assertIn(response, mock_responses) # Should be one of the inputs

        # Test empty responses list
        empty_response = self.engineer.choose_best_response(
            responses=[],
            plan_step=test_plan_step,
            interaction=test_interaction,
            step_number=2,
        )
        self.assertEqual(empty_response, "")

        # Test single response
        single_response_str = "Single test response" # Renamed to avoid conflict
        result = self.engineer.choose_best_response(
            responses=[single_response_str],
            plan_step=test_plan_step,
            interaction=test_interaction,
            step_number=2,
        )
        self.assertEqual(result, single_response_str)


    def print_parser(self): # This test was specific to the old config structure
        # print(f"Parser: {self.parser_arg} \n {self.config.parser.__name__}")
        # self.assertEqual(self.config.parser.__name__, "parse_response")
        pass


# Import necessary classes from advanced_prompting.py


# Create dummy versions of Step, Task, and PlanStep for testing
# These seem to be for TestFinalizePlanStepOutput, keeping them localized if possible
# class DummyFinalStepOutput: # Not needed if using actual FinalStepOutput
#     def __init__(self, final_output):
#         self.final_output = final_output


# class DummyStep: # Not needed if using actual Step
#     def __init__(self, thoughts, description, reflection, final_step_output=None):
#         self.thoughts = thoughts
#         self.description = description
#         self.reflection = reflection
#         self.final_step_output = final_step_output


# class DummyTask: # Not needed if using actual Task
#     def __init__(self):
#         self.output_type = SimpleNamespace(file_extension=".py", output_type="python")


# from advanced_prompting import OutputType # Already imported


# class DummyPlanStep: # Not needed if using actual PlanStep
#     def __init__(
#         self,
#         step_number,
#         step_output,
#         step_name,
#         step_description,
#         step_explanation,
#         step_full_text,
#     ):
#         self.step_number = step_number
#         self.step_output = OutputType(output_type="python", file_extension=".py")
#         self.step_name = step_name
#         self.step_description = step_description
#         self.step_explanation = step_explanation
#         self.step_full_text = step_full_text


class TestFinalizePlanStepOutput(unittest.TestCase):
    def setUp(self):
        # from advanced_prompting import ( # Already imported
        #     AdvancedPromptEngineer,
        #     ComponentType,
        #     FinalPlanStepOutput,
        # )

        # Create a dummy config with minimal attributes required by AdvancedPromptEngineer.
        dummy_config_obj = PromptEngineeringConfig() # Use actual config
        self.engineer = AdvancedPromptEngineer(dummy_config_obj)
        # Override component_decision to always return a fixed component type.

        self.output_type = OutputType(output_type="python", file_extension=".py")
        # self.output_type.output_type = "python"
        # self.output_type.file_extension = ".py"
        self.sample_task = Task(
            description="Create a simple function to calculate factorial",
            refined_description="Write a Python function that calculates factorial recursively",
            complexity=1,
            steps=[],
            reflections=[],
            answer="",
            final_reward=0.0,
            plan=Plan(
                steps=[
                    PlanStep(
                        step_number=1,
                        completed=False,
                        step_name="Define Function",
                        step_description="Create factorial function signature",
                        step_explanation="Define the factorial function with proper parameters",
                        step_output="Function definition",
                        step_full_text="Create factorial function with proper signature and docstring",
                        subtasks=[],
                    ),
                    PlanStep(
                        step_number=2,
                        completed=False,
                        step_name="Implement Function",
                        step_description="Implement the factorial function",
                        step_explanation="Implement the factorial function using recursion",
                        step_output="Function implementation",
                        step_full_text="Implement the factorial function using recursion",
                        subtasks=[],
                    ),
                ]
            ),
            output_type=self.output_type,
            project_name="Factorial Calculator",
        )

        self.plan_step = self.sample_task.plan.steps[0]
        # Create a list of dummy steps.

        self.steps_list = [ # Renamed to avoid conflict
            Step(
                description="def factorial(n):\n    return 1 if n == 0 else n * factorial(n-1)",
                step_number=1,
                remaining_budget=4,
                reflection=None,
                plan_step_number=1,
            ),
            Step(
                description="def factorial(n):\n    return 1 if n == 0 else n * factorial(n-1)", # Same desc for simplicity
                step_number=2,
                remaining_budget=3,
                reflection=None,
                plan_step_number=1,
            ),
        ]
        print(
            f"Plan step: {self.plan_step} \n Steps: {self.steps_list} type: {type(self.steps_list)} type of plan step: {type(self.plan_step)} type of steps entry 0 {type(self.steps_list[0])}"
        )
        self.steps_list[0].final_step_output = FinalStepOutput( # Use actual FinalStepOutput
            final_output="Step one output",
            version=1,
            output_type=self.sample_task.output_type,
            component_type=ComponentType["standalone_file"],
            associated_plan_step=self.plan_step,
            step=self.steps_list[0],
        )
        self.steps_list[1].final_step_output = FinalStepOutput(
            final_output="Step two output",
            version=1,
            output_type=self.sample_task.output_type,
            component_type=ComponentType["standalone_file"],
            associated_plan_step=self.plan_step,
            step=self.steps_list[1],
        )
        self.engineer.component_decision = lambda task, plan_step: ComponentType[
            "response_to_prompt"
        ]
        # Override name_file to simply return a fixed file name.
        self.engineer.name_file = (
            lambda content, ext: f"planstep_{self.plan_step.step_number}_output{ext}"
        )

    @patch("advanced_prompting.openai.chat.completions.create")
    def test_finalize_planstep_output_success(self, mock_create):
        # Simulate a successful LLM response.
        fake_final_output = "Final combined output from LLM"
        fake_response = MagicMock()
        fake_message = MagicMock()
        fake_message.content = fake_final_output
        fake_response.choices = [SimpleNamespace(message=fake_message)]
        mock_create.return_value = fake_response

        output = self.engineer.finalize_planstep_output(
            self.steps_list, self.sample_task, self.plan_step
        )
        # Assert that the final output contains the fake final output.
        self.assertIsInstance(output, FinalPlanStepOutput)
        self.assertEqual(output.final_output, fake_final_output)
        self.assertEqual(output.output_type, self.sample_task.output_type)
        # The file name is generated using our patched name_file.
        self.assertIn("planstep_1_output", output.file_name)
        print("Success test_finalize_planstep_output_success")

    @patch("advanced_prompting.openai.chat.completions.create")
    def test_finalize_planstep_output_empty_response(self, mock_create):
        # Simulate an LLM response returning empty content.
        fake_response = MagicMock()
        fake_message = MagicMock()
        fake_message.content = "   "  # Empty after stripping.
        fake_response.choices = [SimpleNamespace(message=fake_message)]
        mock_create.return_value = fake_response

        output = self.engineer.finalize_planstep_output(
            self.steps_list, self.sample_task, self.plan_step
        )
        # In this case, the final output should fall back to plan_step.step_output.
        self.assertEqual(output.final_output, self.plan_step.step_output)
        self.assertIn("planstep_1_output", output.file_name)
        print("Success test_finalize_planstep_output_empty_response")

    @patch("advanced_prompting.openai.chat.completions.create")
    def test_finalize_planstep_output_exception_handling(self, mock_create):
        # Simulate an exception during the LLM call.
        mock_create.side_effect = Exception("Simulated API error")

        output = self.engineer.finalize_planstep_output(
            self.steps_list, self.sample_task, self.plan_step
        )
        # In case of exception, output should use plan_step.step_output as final output.
        self.assertEqual(output.final_output, self.plan_step.step_output)
        self.assertIn("planstep_1_output", output.file_name)
        print("Success test_finalize_planstep_output_exception_handling")

    # unittest.main()

# New Test Class for Adapter Methods
class TestAdvancedPromptingAdapterMethods(unittest.TestCase):
    def setUp(self):
        self.config = PromptEngineeringConfig() 
        self.engineer = AdvancedPromptEngineer(self.config)
        
        self.mock_plan_step = PlanStep(
            step_number=1,
            step_name="Test Step",
            step_description="A test plan step.",
            step_explanation="Test explanation.",
            step_output="Test output.",
            step_full_text="Full text for test.",
            completed=False,
            subtasks=[]
        )
        self.mock_task = Task(
            description="Test Task",
            refined_description="Refined Test Task",
            complexity=5,
            steps=[], 
            reflections=[],
            answer="",
            final_reward=0.0,
            plan=Plan(steps=[self.mock_plan_step]), 
            output_type=OutputType(output_type="Text File", file_extension=".txt")
        )

    def test_determine_output_type_from_content(self):
        # Test case 1: Python file by extension
        output = self.engineer.determine_output_type_from_content(
            content="def hello(): print('world')", 
            file_path="script.py", 
            task=self.mock_task
        )
        self.assertEqual(output.output_type, "Python Script")
        self.assertEqual(output.file_extension, ".py")

        # Test case 2: JSON file by extension
        output = self.engineer.determine_output_type_from_content(
            content='{"key": "value"}', 
            file_path="data.json", 
            task=self.mock_task
        )
        self.assertEqual(output.output_type, "JSON")
        self.assertEqual(output.file_extension, ".json")

        # Test case 3: HTML file by extension
        output = self.engineer.determine_output_type_from_content(
            content="<h1>Hello</h1>", 
            file_path="page.html", 
            task=self.mock_task
        )
        self.assertEqual(output.output_type, "HTML")
        self.assertEqual(output.file_extension, ".html")

        # Test case 4: PDF file by extension
        output = self.engineer.determine_output_type_from_content(
            content="%PDF-1.4...", 
            file_path="report.pdf", 
            task=self.mock_task
        )
        self.assertEqual(output.output_type, "PDF")
        self.assertEqual(output.file_extension, ".pdf")
        
        # Test case 5: CSV file by extension
        output = self.engineer.determine_output_type_from_content(
            content="col1,col2\nval1,val2", 
            file_path="data.csv", 
            task=self.mock_task
        )
        self.assertEqual(output.output_type, "CSV")
        self.assertEqual(output.file_extension, ".csv")

        # Test case 6: Fallback to task.output_type for unknown extension
        task_with_txt_output = Task(description="Test", refined_description="", complexity=1, steps=[], reflections=[], answer="", final_reward=0.0, plan=None, output_type=OutputType(output_type="Text File", file_extension=".txt"))
        output = self.engineer.determine_output_type_from_content(
            content="some text", 
            file_path="unknown.doc", 
            task=task_with_txt_output
        )
        self.assertEqual(output.output_type, "Text File")
        self.assertEqual(output.file_extension, ".txt")

        # Test case 7: Fallback to content sniffing (Python)
        output = self.engineer.determine_output_type_from_content(
            content="import os\ndef my_func():\n  pass",
            file_path="another.unknown", # Unknown extension
            task=self.mock_task # mock_task's default is .txt
        )
        self.assertEqual(output.output_type, "Python Script")
        self.assertEqual(output.file_extension, ".py")
        
        # Test case 8: Fallback to content sniffing (JSON)
        output = self.engineer.determine_output_type_from_content(
            content='{ "name": "test" }',
            file_path="another.unknown2",
            task=self.mock_task
        )
        self.assertEqual(output.output_type, "JSON")
        self.assertEqual(output.file_extension, ".json")

        # Test case 9: Fallback to content sniffing (HTML)
        output = self.engineer.determine_output_type_from_content(
            content="<html><head></head><body>Test</body></html>",
            file_path="another.unknown3",
            task=self.mock_task
        )
        self.assertEqual(output.output_type, "HTML")
        self.assertEqual(output.file_extension, ".html")

    def test_select_personalities_for_planstep(self):
        task_for_select = Task(description="Test", refined_description="", complexity=3, steps=[], reflections=[], answer="", final_reward=0.0, plan=None, output_type=OutputType(output_type="Text", file_extension=".txt"))
        
        plan_step1 = PlanStep(step_number=1, step_name="Step 1", description="simple task", step_explanation="", step_output="", step_full_text="", completed=False, subtasks=[])
        personalities1 = self.engineer.select_personalities_for_planstep(plan_step1, task_for_select)
        self.assertEqual(sorted(personalities1), sorted(["Strategist", "Educator"]))

        plan_step2 = PlanStep(step_number=1, step_name="Step 2", description="write some code", step_explanation="", step_output="", step_full_text="", completed=False, subtasks=[])
        personalities2 = self.engineer.select_personalities_for_planstep(plan_step2, task_for_select)
        self.assertEqual(sorted(personalities2), sorted(["Strategist", "Educator", "Software Engineer"]))

        plan_step3 = PlanStep(step_number=1, step_name="Step 3", description="perform data analysis", step_explanation="", step_output="", step_full_text="", completed=False, subtasks=[])
        personalities3 = self.engineer.select_personalities_for_planstep(plan_step3, task_for_select)
        self.assertEqual(sorted(personalities3), sorted(["Strategist", "Educator", "Business Analyst"]))
        
        plan_step4 = PlanStep(step_number=1, step_name="Step 4", description="design a new UI", step_explanation="", step_output="", step_full_text="", completed=False, subtasks=[])
        personalities4 = self.engineer.select_personalities_for_planstep(plan_step4, task_for_select)
        self.assertEqual(sorted(personalities4), sorted(["Strategist", "Educator", "UX Designer"]))

        plan_step5 = PlanStep(step_number=1, step_name="Step 5", description="design and code analysis report for python script", step_explanation="", step_output="", step_full_text="", completed=False, subtasks=[])
        personalities5 = self.engineer.select_personalities_for_planstep(plan_step5, task_for_select)
        self.assertEqual(len(personalities5), 3)
        self.assertIn("Strategist", personalities5)
        self.assertIn("Educator", personalities5)
        # The third one depends on the order of checks in the implementation
        # Based on current implementation: code -> Software Engineer is added first if present.
        self.assertIn("Software Engineer", personalities5)


    def test_should_use_multi_agent_reasoning(self):
        task_simple = Task(description="Test", refined_description="", complexity=3, steps=[], reflections=[], answer="", final_reward=0.0, plan=None, output_type=self.mock_task.output_type)
        task_complex = Task(description="Test", refined_description="", complexity=8, steps=[], reflections=[], answer="", final_reward=0.0, plan=None, output_type=self.mock_task.output_type)

        plan_step_short = PlanStep(step_number=1, step_name="Step 1", description="short task", step_explanation="", step_output="", step_full_text="", completed=False, subtasks=[])
        self.assertFalse(self.engineer.should_use_multi_agent_reasoning(plan_step_short, task_simple))

        long_desc = "a" * 201
        plan_step_long = PlanStep(step_number=1, step_name="Step 2", description=long_desc, step_explanation="", step_output="", step_full_text="", completed=False, subtasks=[])
        # Long description (1) + task_complex (1) = 2 indicators
        self.assertTrue(self.engineer.should_use_multi_agent_reasoning(plan_step_long, task_complex)) 
        # Long description (1) + task_simple (0) = 1 indicator
        self.assertFalse(self.engineer.should_use_multi_agent_reasoning(plan_step_long, task_simple))


        plan_step_design = PlanStep(step_number=1, step_name="Step 3", description="design a solution", step_explanation="", step_output="", step_full_text="", completed=False, subtasks=[])
        # design (1) + task_complex (1) = 2
        self.assertTrue(self.engineer.should_use_multi_agent_reasoning(plan_step_design, task_complex))

        plan_step_strategy_analysis = PlanStep(step_number=1, step_name="Step 5", description="strategic analysis of a complex design", step_explanation="", step_output="", step_full_text="", completed=False, subtasks=[])
        # design (1) + strategy (1) + analysis (1) + task_complex (1) = 4 indicators
        self.assertTrue(self.engineer.should_use_multi_agent_reasoning(plan_step_strategy_analysis, task_complex))
        
        # Only high task complexity
        self.assertFalse(self.engineer.should_use_multi_agent_reasoning(plan_step_short, task_complex)) # complexity > 7 (1)

    @patch("advanced_prompting.os.path.exists")
    @patch("builtins.open", new_callable=mock_open, read_data="dummy file content")
    def test_convert_conversation_output_to_final_step_output_file_exists(self, mock_file_open_method, mock_os_exists_method):
        mock_os_exists_method.return_value = True
        
        mocked_output_type = OutputType(output_type="Processed Text", file_extension=".txt")
        
        # Patch the instance method directly
        with patch.object(self.engineer, 'determine_output_type_from_content', return_value=mocked_output_type) as mock_determine:
            dummy_path = "dummy_output.txt"
            self.mock_task.steps = [] 

            final_step_output = self.engineer.convert_conversation_output_to_final_step_output(
                dummy_path, self.mock_plan_step, self.mock_task
            )

            self.assertEqual(final_step_output.final_output, "dummy file content")
            self.assertEqual(final_step_output.output_type, mocked_output_type)
            self.assertEqual(final_step_output.associated_plan_step, self.mock_plan_step)
            self.assertIsInstance(final_step_output.step, Step)
            self.assertEqual(final_step_output.step.description, f"Converted output from conversation for: {self.mock_plan_step.description}")
            self.assertEqual(final_step_output.step.step_number, 1) 
            self.assertEqual(final_step_output.step.remaining_budget, 0) # As per implementation
            self.assertEqual(final_step_output.step.plan_step_number, self.mock_plan_step.step_number)
            mock_file_open_method.assert_called_once_with(dummy_path, 'r')
            mock_determine.assert_called_once_with("dummy file content", dummy_path, self.mock_task)

    @patch("advanced_prompting.os.path.exists")
    def test_convert_conversation_output_to_final_step_output_file_not_exists(self, mock_os_exists_method):
        mock_os_exists_method.return_value = False
        
        mocked_output_type = OutputType(output_type="Path Reference", file_extension="")
        with patch.object(self.engineer, 'determine_output_type_from_content', return_value=mocked_output_type) as mock_determine:
            dummy_path = "non_existent_output.txt"
            self.mock_task.steps = []

            final_step_output = self.engineer.convert_conversation_output_to_final_step_output(
                dummy_path, self.mock_plan_step, self.mock_task
            )

            self.assertEqual(final_step_output.final_output, f"Output path: {dummy_path}")
            self.assertEqual(final_step_output.output_type, mocked_output_type)
            mock_determine.assert_called_once_with(f"Output path: {dummy_path}", dummy_path, self.mock_task)

    @patch('advanced_prompting.AdvancedPromptEngineer.convert_conversation_output_to_final_step_output')
    @patch('advanced_prompting.run_conversation') 
    @patch('advanced_prompting.AdvancedPromptEngineer.select_personalities_for_planstep')
    def test_invoke_collaborative_reasoning(self, mock_select_personalities, mock_run_conversation, mock_convert_output):
        mock_select_personalities.return_value = ["Strategist", "Educator"]
        mock_run_conversation.return_value = "path/to/dummy_output.txt" # run_conversation returns a path
        
        # Define what the mocked convert_conversation_output_to_final_step_output should return
        expected_final_step_output = FinalStepOutput(
            final_output="converted_output_content",
            output_type=self.mock_task.output_type, # Example OutputType
            version=1,
            component_type="response_to_prompt", # Example component type
            associated_plan_step=self.mock_plan_step,
            step=Step(description="Collaborative reasoning step", step_number=1, remaining_budget=0, plan_step_number=self.mock_plan_step.step_number) 
        )
        mock_convert_output.return_value = expected_final_step_output

        result = self.engineer.invoke_collaborative_reasoning(self.mock_task, self.mock_plan_step)

        mock_select_personalities.assert_called_once_with(self.mock_plan_step, self.mock_task)
        mock_run_conversation.assert_called_once_with(
            problem_statement=[{"content": f"Step {self.mock_plan_step.step_number}: {self.mock_plan_step.description}"}],
            selected_personalities=["Strategist", "Educator"],
            lead_personality="Strategist",
            num_rounds=2
        )
        mock_convert_output.assert_called_once_with("path/to/dummy_output.txt", self.mock_plan_step, self.mock_task)
        self.assertEqual(result, expected_final_step_output)


class TestFinalizePlanStepOutput(unittest.TestCase):
    def setUp(self):
        dummy_config_obj = PromptEngineeringConfig() 
        self.engineer = AdvancedPromptEngineer(dummy_config_obj)
        
        self.output_type = OutputType(output_type="python", file_extension=".py")
        self.sample_task = Task(
            description="Create a simple function to calculate factorial",
            refined_description="Write a Python function that calculates factorial recursively",
            complexity=1,
            steps=[],
            reflections=[],
            answer="",
            final_reward=0.0,
            plan=Plan(
                steps=[
                    PlanStep(
                        step_number=1,
                        completed=False,
                        step_name="Define Function",
                        step_description="Create factorial function signature",
                        step_explanation="Define the factorial function with proper parameters",
                        step_output="Function definition",
                        step_full_text="Create factorial function with proper signature and docstring",
                        subtasks=[],
                    ),
                    PlanStep(
                        step_number=2,
                        completed=False,
                        step_name="Implement Function",
                        step_description="Implement the factorial function",
                        step_explanation="Implement the factorial function using recursion",
                        step_output="Function implementation",
                        step_full_text="Implement the factorial function using recursion",
                        subtasks=[],
                    ),
                ]
            ),
            output_type=self.output_type,
            project_name="Factorial Calculator",
        )

        self.plan_step = self.sample_task.plan.steps[0]
        
        self.steps_list = [ 
            Step(
                description="def factorial(n):\n    return 1 if n == 0 else n * factorial(n-1)",
                step_number=1,
                remaining_budget=4,
                reflection=None,
                plan_step_number=1,
            ),
            Step(
                description="def factorial(n):\n    return 1 if n == 0 else n * factorial(n-1)", 
                step_number=2,
                remaining_budget=3,
                reflection=None,
                plan_step_number=1,
            ),
        ]
        # print( # Commented out print
        #     f"Plan step: {self.plan_step} \n Steps: {self.steps_list} type: {type(self.steps_list)} type of plan step: {type(self.plan_step)} type of steps entry 0 {type(self.steps_list[0])}"
        # )
        self.steps_list[0].final_step_output = FinalStepOutput( 
            final_output="Step one output",
            version=1,
            output_type=self.sample_task.output_type,
            component_type=ComponentType["standalone_file"],
            associated_plan_step=self.plan_step,
            step=self.steps_list[0],
        )
        self.steps_list[1].final_step_output = FinalStepOutput(
            final_output="Step two output",
            version=1,
            output_type=self.sample_task.output_type,
            component_type=ComponentType["standalone_file"],
            associated_plan_step=self.plan_step,
            step=self.steps_list[1],
        )
        self.engineer.component_decision = lambda task, plan_step: ComponentType[
            "response_to_prompt"
        ]
        self.engineer.name_file = (
            lambda content, ext: f"planstep_{self.plan_step.step_number}_output{ext}"
        )

    @patch("advanced_prompting.openai.chat.completions.create")
    def test_finalize_planstep_output_success(self, mock_create):
        fake_final_output = "Final combined output from LLM"
        fake_response = MagicMock()
        fake_message = MagicMock()
        fake_message.content = fake_final_output
        fake_response.choices = [SimpleNamespace(message=fake_message)]
        mock_create.return_value = fake_response

        output = self.engineer.finalize_planstep_output(
            self.steps_list, self.sample_task, self.plan_step
        )
        self.assertIsInstance(output, FinalPlanStepOutput)
        self.assertEqual(output.final_output, fake_final_output)
        self.assertEqual(output.output_type, self.sample_task.output_type)
        self.assertIn("planstep_1_output", output.file_name)
        # print("Success test_finalize_planstep_output_success") # Commented out

    @patch("advanced_prompting.openai.chat.completions.create")
    def test_finalize_planstep_output_empty_response(self, mock_create):
        fake_response = MagicMock()
        fake_message = MagicMock()
        fake_message.content = "   " 
        fake_response.choices = [SimpleNamespace(message=fake_message)]
        mock_create.return_value = fake_response

        output = self.engineer.finalize_planstep_output(
            self.steps_list, self.sample_task, self.plan_step
        )
        self.assertEqual(output.final_output, self.plan_step.step_output)
        self.assertIn("planstep_1_output", output.file_name)
        # print("Success test_finalize_planstep_output_empty_response") # Commented out

    @patch("advanced_prompting.openai.chat.completions.create")
    def test_finalize_planstep_output_exception_handling(self, mock_create):
        mock_create.side_effect = Exception("Simulated API error")

        output = self.engineer.finalize_planstep_output(
            self.steps_list, self.sample_task, self.plan_step
        )
        self.assertEqual(output.final_output, self.plan_step.step_output)
        self.assertIn("planstep_1_output", output.file_name)
        # print("Success test_finalize_planstep_output_exception_handling") # Commented out


if __name__ == "__main__":
    unittest.main(verbosity=2)
