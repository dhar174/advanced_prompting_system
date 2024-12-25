import unittest

from matplotlib.pyplot import step
from advanced_prompting import (
    AdvancedPromptEngineer,
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
    def __init__(self, *args, **kwargs):
        self.parser_arg = kwargs.pop("parser_arg", "test_c")
        self.config = kwargs.pop("config", None)
        self.output_type = OutputType
        self.output_type.output_type = "text"
        self.output_type.file_extension = "txt"
        if self.config is None:
            self.config = config_for_test(parser_arg=self.parser_arg)
        self.parser = None
        super(TestParseResponse, self).__init__(*args, **kwargs)

    def setUp(self):
        """Initialize test environment with config, engineer and sample task."""
        if self.config.parser_arg != "default":
            if self.config.parser_arg == "test_b":
                self.config.parser = test_b.parse_response
            elif self.config.parser_arg == "test_c":
                self.config.parser = test_c.parse_response
        elif self.config.parser_arg == "default":
            self.config.get_engineer()
            self.parser = self.config.parser

        if self.parser is None:
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
                        step_name="Initialize Test Case",
                        step_description="Create initial test case structure",
                        step_explanation="Set up the basic test framework with required imports and test class",
                        step_output="TestCase class with setUp method",
                        step_full_text="Initialize test case by creating class structure and importing dependencies",
                        subtasks=[],
                    ),
                    PlanStep(
                        step_number=2,
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
        self.assertEqual(interaction.reflections[0].reward, 0.0)

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
        self.assertEqual(interaction.steps[0].remaining_budget, 0)
        self.assertEqual(interaction.reflections[0].reward, 0.0)

    def test_steps_and_reflections_objs_provided(self):
        existing_steps = [
            Step(
                description="Existing step",
                step_number=0,
                remaining_budget=5,
                reflection=None,
            )
        ]
        existing_reflections = [
            Reflection(content="Existing reflection", reward=0.5, step_number=0)
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

    def test_overlapping_steps_and_reflections(self):
        existing_steps = [
            Step(
                description="Overlapping step",
                step_number=0,
                remaining_budget=5,
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
        self.assertEqual(interaction.steps[0].reflection.content, "Updated reflection")
        self.assertEqual(interaction.steps[0].reflection.reward, 0.8)
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
        interaction = self.config.parser(response, self.sample_task)
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
        self.assertEqual(interaction.steps[0].remaining_budget, 5)

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
        self.assertEqual(len(interaction.reflections), 1)
        self.assertEqual(
            interaction.reflections[0].content, "Test reflection without step"
        )
        self.assertEqual(interaction.reflections[0].reward, 0.8)

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
            interaction.steps[0].reflection.reward, 0.8
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
        print(f"Interaction (reward score test): {interaction} \n")

        self.assertEqual(interaction.steps[0].reflection.reward, 0.0)

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

        self.assertEqual(interaction.steps[0].remaining_budget, 10)

    def test_existing_steps_integration(self):
        # Arrange
        existing_steps = [
            Step(
                description="Existing step",
                step_number=1,
                remaining_budget=10,
                reflection=None,
            )
        ]

        response = """
        <count>9</count>
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


if __name__ == "__main__":
    test_config = config_for_test(parser_arg="test_c")
    test = TestParseResponse(parser_arg="test_c", config=test_config)
    test.setUp()
    unittest.main()
