from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from matplotlib.pyplot import step
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
)
import re
from complexity_measures import Plan, PlanStep
import test_c
import test_b
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


class TestParseResponse(unittest.TestCase):

    def setUp(self):
        """Initialize test environment with config, engineer and sample task."""
        self.parser_arg = "default"
        self.config = TEST_CONFIG
        self.parser_arg = TEST_CONFIG.parser_arg
        self.output_type = OutputType(output_type="text", file_extension="txt")
        self.output_type.output_type = "text"
        self.output_type.file_extension = "txt"
        if self.config is None:
            self.config = config_for_test(parser_arg=self.parser_arg)
        self.parser = None
        if self.config.parser_arg != "default":
            if self.config.parser_arg == "test_b":
                self.config.parser = test_b.parse_response
            elif self.config.parser_arg == "test_c":
                self.config.parser = test_c.parse_response
        elif self.config.parser_arg == "default":
            self.config.get_engineer()
            self.parser = self.config.parser

        if self.parser is None:
            self.parser = (
                self.config.parser
                if self.config.parser is not None
                else self.config.get_engineer().parse_response
            )

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
        self.assertEqual(interaction.steps[0].remaining_budget, 11)
        self.assertEqual(interaction.reflections[0].reward, 0.7)

    def test_steps_and_reflections_objs_provided(self):
        existing_steps = [
            Step(
                description="Existing step",
                step_number=1,
                remaining_budget=4,
                reflection=None,
            )
        ]
        existing_reflections = [
            Reflection(content="Existing reflection", reward=0.5, step_number=1)
        ]

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
        )
        print(f"Interaction test_steps_and_reflections_objs_provided: {interaction} \n")

        self.assertEqual(len(interaction.steps), 2)
        self.assertEqual(interaction.steps[0].description, "Existing step")
        self.assertEqual(interaction.steps[0].reflection.content, "Existing reflection")
        self.assertEqual(interaction.steps[1].description, "New step")
        self.assertEqual(interaction.steps[1].reflection.content, "New reflection")
        self.assertEqual(interaction.steps[1].reflection.reward, 0.7)
        self.assertEqual(interaction.steps[1].remaining_budget, 3)
        self.assertEqual(interaction.reflections[0].reward, 0.5)
        self.assertEqual(interaction.steps[0].reflection.reward, 0.5)

    def test_overlapping_steps_and_reflections(self):
        existing_steps = [
            Step(
                description="Overlapping step",
                step_number=1,
                remaining_budget=4,
                reflection=Reflection(
                    content="Initial reflection", reward=0.6, step_number=0
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
        self.assertEqual(interaction.steps[0].reflection.content, "Initial reflection")
        self.assertEqual(interaction.steps[0].reflection.reward, 0.6)
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
        <step>Second step</step>
        <answer>Second answer</answer>
        <final_reward>0.85</final_reward>
        """

        interaction = self.config.parser(response, self.sample_task)
        print(f"Interaction test_multiple_answers_and_rewards: {interaction} \n")

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
        self.assertEqual(interaction.steps[0].remaining_budget, 4)

    def test_missing_step_tag(self):
        response = """
        <count>5</count>
        <thinking>Thinking...</thinking>
        <reflection>Test reflection without step</reflection>
        <reward>0.8</reward>
        """
        interaction = self.config.parser(response, self.sample_task)
        print(f"Interaction test_missing_step_tag: {interaction} \n")

        self.assertEqual(len(interaction.steps), 0)
        self.assertEqual(len(interaction.reflections), 0)
        # self.assertEqual(
        #     interaction.reflections[0].content, "Test reflection without step"
        # )
        # self.assertEqual(interaction.reflections[0].reward, 0.8)

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
        # self.assertEqual(interaction.steps[0].reflection.content, "")
        self.assertEqual(
            interaction.steps[0].reflection.reward, round(0.7, 2)
        ), f"Reward: {interaction.steps[0].reflection.reward}"

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
        self.assertEqual(len(interaction.reflections), 2)

    def test_missing_reflection_handling(self):
        response = """
        <count>5</count>
        <thinking>Thinking...</thinking>
        <step>Step without reflection</step>
"""

        interaction = self.config.parser(response, self.sample_task)
        print(f"Interaction test_missing_reflection_handling: {interaction} \n")
        self.assertEqual(len(interaction.steps), 1)
        self.assertIsNotNone(interaction.steps[0].reflection)

    def test_budget_counting(self):
        response = """
        <count>5</count>
        <thinking>Thinking...</thinking>
        <step>First step</step>
        <count>4</count>
        <thinking>Thinking...</thinking>
        <step>Second step</step>
"""

        interaction = self.config.parser(response, self.sample_task)
        print(f"Interaction test_budget_counting: {interaction} \n")

        self.assertEqual(interaction.steps[0].remaining_budget, 5)
        self.assertEqual(interaction.steps[1].remaining_budget, 4)

    def test_interaction_merging(self):
        existing_interaction = Interaction(
            task=self.sample_task,
            steps=[
                Step(
                    description="Existing step",
                    step_number=1,
                    remaining_budget=5,
                    reflection=None,
                )
            ],
            reflections=[],
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
            response, self.sample_task, interaction=existing_interaction
        )
        print(
            f"Interaction: {interaction} \n Length of steps: {len(interaction.steps)} \n Steps: {interaction.steps}"
        )

        self.assertEqual(len(interaction.steps), 2)
        self.assertEqual(interaction.steps[0].description, "Existing step")
        self.assertEqual(interaction.steps[1].description, "New step")

    def test_reward_score_validation(self):
        response = """
        <count>5</count>
        <thinking>Thinking...</thinking>
        <step>Test step</step>
        <reflection>Test reflection</reflection>
        <reward>invalid</reward>"""

        interaction = self.config.parser(response, self.sample_task)
        print(f"Interaction test_reward_score_validation: {interaction} \n")

        self.assertEqual(interaction.steps[0].reflection.reward, 0.0)

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
        <reward>0.8</reward>"""

        interaction = self.config.parser(response, self.sample_task)
        print(f"Interaction test_malformed_tags_handling: {interaction} \n")

        self.assertEqual(len(interaction.steps), 1)
        self.assertIsNotNone(interaction.steps[0].reflection)

    def test_initial_budget_handling(self):
        response = """
        <count>10</count>
        <thinking>Thinking...</thinking>
        <step>First step</step>
"""

        interaction = self.config.parser(response, self.sample_task, initial_budget=5)
        print(f"Interaction test_initial_budget_handling: {interaction} \n")

        self.assertEqual(interaction.steps[0].remaining_budget, 5)

    def test_existing_steps_integration(self):
        # Arrange
        existing_steps = [
            Step(
                description="Existing step",
                step_number=1,
                remaining_budget=5,
                reflection=None,
            )
        ]

        response = """
        <count>4</count>
        <thinking>Thinking...</thinking>
        <step>New step</step>
        <reflection>New reflection</reflection>
        <reward>0.9</reward>"""

        # Act
        interaction = self.config.parser(
            response, self.sample_task, steps_objs=existing_steps
        )
        print(f"Interaction (existing step test): {interaction} \n")

        # Assert
        self.assertEqual(len(interaction.steps), 2)
        self.assertEqual(interaction.steps[0].description, "Existing step")
        self.assertEqual(interaction.steps[1].description, "New step")
        self.assertEqual(interaction.steps[1].remaining_budget, 4)

    def test_reflection_objects_integration(self):
        # Arrange
        existing_reflections = [
            Reflection(content="Previous reflection", reward=0.8, step_number=0)
        ]

        response = """
        <count>4</count>
        <thinking>Thinking...</thinking>
        <step>New step</step>
        <reflection>New reflection</reflection>
        <reward>0.85</reward>"""

        # Act
        interaction = self.config.parser(
            response, self.sample_task, reflections_objs=existing_reflections
        )
        print(f"Interaction (reflection object test): {interaction} \n")

        # Assert
        self.assertEqual(len(interaction.steps), 1)
        self.assertEqual(interaction.steps[0].reflection.content, "New reflection")
        self.assertEqual(interaction.steps[0].remaining_budget, 4)

    def test_duplicate_step_handling(self):
        response = """
        <count>5</count>
        <thinking>Thinking...</thinking>
        <step>First step</step>
        <count>4</count>
        <thinking>Thinking...</thinking>
        <step>First step</step> 
"""

        interaction = self.config.parser(response, self.sample_task)
        print(f"Interaction (duplicate step test): {interaction} \n")

        self.assertEqual(len(interaction.steps), 2)
        self.assertEqual(interaction.steps[0].step_number, 0)
        self.assertEqual(interaction.steps[1].step_number, 1)

    def test_reward_range_validation(self):
        response = """
        <count>5</count>
        <thinking>Thinking...</thinking>
        <step>Test step</step>
        <reflection>Test reflection</reflection>
        <reward>1.5</reward>"""

        interaction = self.config.parser(response, self.sample_task)

        print(f"Interaction (reward range test): {interaction} \n")

        self.assertLessEqual(interaction.steps[0].reflection.reward, 1.0)
        self.assertGreaterEqual(interaction.steps[0].reflection.reward, 0.0)

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
        print(f"Interaction (basic response test): {interaction} \n")

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
        print(f"Interaction (multiple steps test): {interaction} \n")

        self.assertEqual(len(interaction.steps), 2)
        self.assertEqual(interaction.steps[0].description, "First step")
        self.assertEqual(interaction.steps[1].description, "Second step")
        self.assertEqual(len(interaction.reflections), 2)

    def test_missing_reflection_handling(self):
        response = """
        <count>5</count>
        <thinking>Thinking...</thinking>
        <step>Step without reflection</step>
"""

        interaction = self.config.parser(response, self.sample_task)
        print(f"Interaction (missing reflection test) : {interaction} \n")
        self.assertEqual(len(interaction.steps), 1)
        self.assertIsNotNone(interaction.steps[0].reflection)

    def test_budget_counting(self):
        response = """
        <count>5</count>
        <thinking>Thinking...</thinking>
        <step>First step</step>
        <count>4</count>
        <thinking>Thinking...</thinking>
        <step>Second step</step>
"""

        interaction = self.config.parser(response, self.sample_task)
        print(f"Interaction (budget counting test): {interaction} \n")

        self.assertEqual(interaction.steps[0].remaining_budget, 4)
        self.assertEqual(interaction.steps[1].remaining_budget, 3)

    def test_interaction_merging(self):
        existing_interaction = Interaction(
            task=self.sample_task,
            steps=[
                Step(
                    description="Existing step",
                    step_number=1,
                    remaining_budget=5,
                    reflection=None,
                )
            ],
            reflections=[],
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
            response, self.sample_task, interaction=existing_interaction
        )
        print(
            f"Interaction: {interaction} \n Length of steps: {len(interaction.steps)} \n Steps: {interaction.steps}"
        )

        self.assertEqual(len(interaction.steps), 2)
        self.assertEqual(interaction.steps[0].description, "Existing step")
        self.assertEqual(interaction.steps[1].description, "New step")

    def test_reward_score_validation(self):
        response = """
        <count>5</count>
        <thinking>Thinking...</thinking>
        <step>Test step</step>
        <reflection>Test reflection</reflection>
        <reward>invalid</reward>"""

        interaction = self.config.parser(response, self.sample_task)
        print(f"Interaction (reward score test): {interaction} \n")

        self.assertEqual(interaction.steps[0].reflection.reward, 0.7)

    def test_empty_response_handling(self):
        response = ""
        interaction = self.config.parser(response, self.sample_task)
        print(f"Interaction (empty response test): {interaction} \n")

        self.assertEqual(len(interaction.steps), 0)
        self.assertEqual(len(interaction.reflections), 0)
        self.assertEqual(interaction.answer, "")
        self.assertEqual(interaction.final_reward, 0.0)

    def test_initial_budget_handling(self):
        response = """
        <count>10</count>
        <thinking>Thinking...</thinking>
        <step>First step</step>
"""

        interaction = self.config.parser(response, self.sample_task, initial_budget=5)
        print(f"Interaction (initial budget test): {interaction} \n")

        self.assertEqual(interaction.steps[0].remaining_budget, 9)

    def test_existing_steps_integration(self):
        # Arrange
        existing_steps = [
            Step(
                description="Existing step (existing step test)",
                step_number=1,
                remaining_budget=9,
                reflection=None,
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
            response, self.sample_task, steps_objs=existing_steps
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
        self.assertEqual(interaction.steps[1].remaining_budget, 8)

    def test_reflection_objects_integration(self):
        # Arrange
        existing_reflections = [
            Reflection(content="Previous reflection", reward=0.8, step_number=1)
        ]

        response = """
        <count>4</count>
        <thinking>Thinking...</thinking>
        <step>New step</step>
        <reflection>New reflection</reflection>
        <reward>0.85</reward>"""

        # Act
        interaction = self.config.parser(
            response,
            self.sample_task,
            reflections_objs=existing_reflections,
            initial_budget=4,
        )
        print(f"Interaction (reflection object test): {interaction} \n")

        # Assert
        self.assertEqual(len(interaction.steps), 1)
        self.assertEqual(interaction.steps[0].reflection.content, "Previous reflection")
        self.assertEqual(interaction.steps[0].remaining_budget, 3)

    def test_duplicate_step_handling(self):
        response = """
        <count>5</count>
        <thinking>Thinking...</thinking>
        <step>First step</step>
        <count>4</count>
        <thinking>Thinking...</thinking>
        <step>First step</step> 
"""

        interaction = self.config.parser(response, self.sample_task)
        print(f"Interaction (duplicate step test): {interaction} \n")

        self.assertEqual(len(interaction.steps), 2)
        self.assertEqual(interaction.steps[0].step_number, 1)
        self.assertEqual(interaction.steps[1].step_number, 2)

    def test_reward_range_validation(self):
        response = """
        <count>5</count>
        <thinking>Thinking...</thinking>
        <step>Test step</step>
        <reflection>Test reflection</reflection>
        <reward>1.5</reward>"""

        interaction = self.config.parser(response, self.sample_task)

        print(f"Interaction (reward range test): {interaction} \n")

        self.assertLessEqual(interaction.steps[0].reflection.reward, 1.0)
        self.assertGreaterEqual(interaction.steps[0].reflection.reward, 0.0)

    def print_parser(self):
        print(f"Parser: {self.parser_arg} \n {self.config.parser.__name__}")
        self.assertEqual(self.config.parser.__name__, "parse_response")


class TestAdvancedPromptEngineer(unittest.TestCase):
    def setUp(self):
        """Initialize test environment with config, engineer and sample task."""
        self.config = TEST_CONFIG

        if self.config.parser_arg != "default":
            if self.config.parser_arg == "test_b":
                self.config.parser = test_b.parse_response
            elif self.config.parser_arg == "test_c":
                self.config.parser = test_c.parse_response
        elif self.config.parser_arg == "default":
            self.config.get_engineer()
            self.parser = self.config.parser
        self.parser_arg = self.config.parser_arg
        if self.parser is None:
            self.parser = (
                self.config.parser
                if self.config.parser is not None
                else self.config.get_engineer().parse_response
            )

        print(f"Parser: {self.parser_arg} \n {self.config.parser.__name__}")
        self.engineer = AdvancedPromptEngineer(self.config.config)
        self.output_type = OutputType
        self.output_type.output_type = "python"
        self.output_type.file_extension = ".py"
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
        task = "Write a function to calculate factorial"
        retrieved_info = "Factorial is the product of all positive integers up to n"
        step_budget = 5
        complexity = 2

        initial_prompt, system_prompt = self.engineer.generate_initial_prompt(
            task, retrieved_info, step_budget, complexity, self.output_type
        )

        self.assertIsInstance(initial_prompt, str)
        self.assertIsInstance(system_prompt, str)
        self.assertIn("factorial", initial_prompt.lower())
        self.assertIn("<thinking>", system_prompt)
        self.assertIn("<step>", system_prompt)

    def test_judge_step(self):
        step = Step(
            description="Define factorial function with parameter n",
            step_number=1,
            remaining_budget=5,
            reflection=None,
        )
        reflection = self.engineer.judge_step(step, self.sample_task)

        self.assertIsInstance(reflection, Reflection)
        self.assertTrue(0.0 <= reflection.reward <= 1.0)
        self.assertIsInstance(reflection.content, str)
        self.assertEqual(reflection.step_number, 1)

    def test_component_decision(self):
        from advanced_prompting import ComponentType

        task = self.sample_task
        plan_step = task.plan.steps[0]

        decision = self.engineer.component_decision(task, plan_step)

        self.assertIsInstance(decision, ComponentType)
        self.assertIn(decision, ComponentType)

    def test_judge_step_completion(self):
        step = [
            Step(
                description="def factorial(n):\n    if n == 0: return 1\n    return n * factorial(n-1)",
                step_number=1,
                remaining_budget=4,
                reflection=None,
            )
        ]
        completed, next_step = self.engineer.judge_step_completion(
            step, self.sample_task.plan.steps[0], max_plan_steps=1
        )

        self.assertIsInstance(completed, bool)
        self.assertIsInstance(next_step, int)

    def test_finalize_step_output(self):
        from advanced_prompting import FinalStepOutput

        step = Step(
            description="def factorial(n):\n    return 1 if n == 0 else n * factorial(n-1)",
            step_number=1,
            remaining_budget=4,
            reflection=None,
            plan_step_number=1,
        )

        output = self.engineer.finalize_step_output(
            step, self.sample_task, self.sample_task.plan.steps[0], []
        )

        self.assertIsInstance(output, FinalStepOutput)
        self.assertEqual(output.version, 1)
        self.assertEqual(output.associated_plan_step, self.sample_task.plan.steps[0])

    def test_automatic_chain_of_thought(self):
        task = "Calculate factorial of 5"
        thoughts = self.engineer.automatic_chain_of_thought(task)

        self.assertIsInstance(thoughts, str)
        self.assertTrue(len(thoughts) > 0)
        self.assertIn("factorial", thoughts.lower())

    def test_least_to_most(self):
        task = "Calculate factorial of 5"
        sub_tasks = self.engineer.least_to_most(task)

        self.assertIsInstance(sub_tasks, list)
        self.assertTrue(len(sub_tasks) > 0)
        self.assertTrue(all(isinstance(task, str) for task in sub_tasks))

    def test_progressive_hint(self):
        sub_task = "Define factorial function"
        hints = self.engineer.progressive_hint(sub_task)

        self.assertIsInstance(hints, list)
        self.assertTrue(len(hints) > 0)
        self.assertTrue(all(isinstance(hint, str) for hint in hints))

    def test_assess_complexity(self):
        task = "Write a recursive factorial function with error handling"
        complexity, plan = self.engineer.assess_complexity(task)

        self.assertIsInstance(complexity, (int, float))
        self.assertGreater(complexity, 0)
        self.assertIsInstance(plan, Plan)

    def test_adjust_step_budget(self):
        task = "Write a factorial calculator"
        complexity = 3

        result = self.engineer.adjust_step_budget(task, complexity)

        if isinstance(result, tuple):
            budget, plan = result
            self.assertIsInstance(budget, int)
            self.assertIsInstance(plan, Plan)
        else:
            self.assertIsInstance(result, int)
            self.assertGreater(result, 0)

    def test_dynamic_confidence_exploration(self):
        interaction = Interaction(
            task=self.sample_task,
            steps=[],
            reflections=[],
            answer="def factorial(n): return 1 if n == 0 else n * factorial(n-1)",
            final_reward=0.4,
        )

        result = self.engineer.dynamic_confidence_exploration(
            interaction, "Write a factorial function"
        )

        self.assertIsInstance(result, Interaction)
        self.assertTrue(hasattr(result, "final_reward"))

    def test_merge_interactions(self):
        interaction_a = Interaction(
            task=self.sample_task,
            steps=[Step("Step 1", 1, 5)],
            reflections=[],
            answer="Answer A",
            final_reward=0.8,
        )

        interaction_b = Interaction(
            task=self.sample_task,
            steps=[Step("Step 2", 2, 4)],
            reflections=[],
            answer="Answer B",
            final_reward=0.9,
        )

        merged = self.engineer.merge_interactions(interaction_a, interaction_b)

        self.assertIsInstance(merged, Interaction)
        self.assertEqual(len(merged.steps), 2)
        self.assertEqual(merged.final_reward, 0.9)

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
            steps=[Step("Previous step", 1, 5)],
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
        self.assertGreater(len(response), 0)

        # Test empty responses list
        empty_response = self.engineer.choose_best_response(
            responses=[],
            plan_step=test_plan_step,
            interaction=test_interaction,
            step_number=2,
        )
        self.assertEqual(empty_response, "")

        # Test single response
        single_response = "Single test response"
        result = self.engineer.choose_best_response(
            responses=[single_response],
            plan_step=test_plan_step,
            interaction=test_interaction,
            step_number=2,
        )
        self.assertEqual(result, single_response)

        # Test with invalid step number
        response_invalid_step = self.engineer.choose_best_response(
            responses=mock_responses,
            plan_step=test_plan_step,
            interaction=test_interaction,
            step_number=-1,
        )
        self.assertIsInstance(response_invalid_step, str)

        # Test with None response in list
        responses_with_none = ["Valid response", None, "Another valid response"]
        response_none = self.engineer.choose_best_response(
            responses=responses_with_none,
            plan_step=test_plan_step,
            interaction=test_interaction,
            step_number=2,
        )
        self.assertIsInstance(response_none, str)
        self.assertGreater(len(response_none), 0)

        # Test with empty PlanStep
        empty_plan_step = PlanStep(
            step_number=1,
            completed=False,
            step_name="",
            step_description="",
            step_explanation="",
            step_output="",
            step_full_text="",
            subtasks=[],
        )
        response_empty_plan = self.engineer.choose_best_response(
            responses=mock_responses,
            plan_step=empty_plan_step,
            interaction=test_interaction,
            step_number=2,
        )
        self.assertIsInstance(response_empty_plan, str)
        self.assertGreater(len(response_empty_plan), 0)

        # Test with empty interaction
        empty_interaction = Interaction(
            task=self.sample_task, steps=[], reflections=[], answer="", final_reward=0.0
        )
        response_empty_interaction = self.engineer.choose_best_response(
            responses=mock_responses,
            plan_step=test_plan_step,
            interaction=empty_interaction,
            step_number=1,
        )
        self.assertIsInstance(response_empty_interaction, str)
        self.assertGreater(len(response_empty_interaction), 0)

        # Test error handling with malformed response
        malformed_responses = ["<invalid>xml", "not<proper>format", "<broken>tags"]
        response_malformed = self.engineer.choose_best_response(
            responses=malformed_responses,
            plan_step=test_plan_step,
            interaction=test_interaction,
            step_number=2,
        )
        self.assertIsInstance(response_malformed, str)
        print(f"Response: {response_malformed}")

    def print_parser(self):
        print(f"Parser: {self.parser_arg} \n {self.config.parser.__name__}")
        self.assertEqual(self.config.parser.__name__, "parse_response")


# Import necessary classes from advanced_prompting.py


# Create dummy versions of Step, Task, and PlanStep for testing
class DummyFinalStepOutput:
    def __init__(self, final_output):
        self.final_output = final_output


class DummyStep:
    def __init__(self, thoughts, description, reflection, final_step_output=None):
        self.thoughts = thoughts
        self.description = description
        self.reflection = reflection
        self.final_step_output = final_step_output


class DummyTask:
    def __init__(self):
        self.output_type = SimpleNamespace(file_extension=".py", output_type="python")


from advanced_prompting import OutputType


class DummyPlanStep:
    def __init__(
        self,
        step_number,
        step_output,
        step_name,
        step_description,
        step_explanation,
        step_full_text,
    ):
        self.step_number = step_number
        self.step_output = OutputType(output_type="python", file_extension=".py")
        self.step_name = step_name
        self.step_description = step_description
        self.step_explanation = step_explanation
        self.step_full_text = step_full_text


class TestFinalizePlanStepOutput(unittest.TestCase):
    def setUp(self):
        from advanced_prompting import (
            AdvancedPromptEngineer,
            ComponentType,
            FinalPlanStepOutput,
        )

        # Create a dummy config with minimal attributes required by AdvancedPromptEngineer.
        dummy_config = SimpleNamespace(model="dummy-model")
        self.engineer = AdvancedPromptEngineer(dummy_config)
        # Override component_decision to always return a fixed component type.

        self.output_type = OutputType(output_type="python", file_extension=".py")
        self.output_type.output_type = "python"
        self.output_type.file_extension = ".py"
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

        self.steps = [
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
        print(
            f"Plan step: {self.plan_step} \n Steps: {self.steps} type: {type(self.steps)} type of plan step: {type(self.plan_step)} type of steps entry 0 {type(self.steps[0])}"
        )
        self.steps[0].final_step_output = FinalStepOutput(
            final_output="Step one output",
            version=1,
            output_type=self.sample_task.output_type,
            component_type=ComponentType["standalone_file"],
            associated_plan_step=self.plan_step,
            step=self.steps[0],
        )
        self.steps[1].final_step_output = FinalStepOutput(
            final_output="Step two output",
            version=1,
            output_type=self.sample_task.output_type,
            component_type=ComponentType["standalone_file"],
            associated_plan_step=self.plan_step,
            step=self.steps[1],
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
            self.steps, self.sample_task, self.plan_step
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
            self.steps, self.sample_task, self.plan_step
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
            self.steps, self.sample_task, self.plan_step
        )
        # In case of exception, output should use plan_step.step_output as final output.
        self.assertEqual(output.final_output, self.plan_step.step_output)
        self.assertIn("planstep_1_output", output.file_name)
        print("Success test_finalize_planstep_output_exception_handling")

    # unittest.main()


if __name__ == "__main__":
    # TEST_CONFIG = config_for_test(parser_arg="default")
    # engineer_test = TestAdvancedPromptEngineer()
    # engineer_test.setUp()
    # engineer_test.test_choose_best_response()
    test_finalizeplanstep = TestFinalizePlanStepOutput()
    test_finalizeplanstep.setUp()
    test_finalizeplanstep.test_finalize_planstep_output_success()
    test_finalizeplanstep.test_finalize_planstep_output_empty_response()
    test_finalizeplanstep.test_finalize_planstep_output_exception_handling()
    print("All tests passed")
    # unittest.main()
    # print(f"Parser: {TEST_CONFIG.parser_arg} \n {TEST_CONFIG.config.parser.__name__}")
