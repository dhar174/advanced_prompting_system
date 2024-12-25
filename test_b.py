import re
from typing import List, Optional
from weakref import ref

from advanced_prompting import Task, print_saver, Interaction, Reflection, Step

# ===========================
# Define Necessary Classes
# ===========================


# ===========================
# Define the consolidate_steps Function
# ===========================


def consolidate_steps(
    step_objs: List[Step],
    steps: List[str],
    counts: List[int],
    reflections: List[Reflection],
) -> List[Step]:
    """
    Consolidate the steps from step_objs, steps, and counts into a single ordered list of Step objects.

    Args:
        step_objs (List[Step]): Existing list of Step objects.
        steps (List[str]): List of step descriptions.
        counts (List[int]): List of counts representing the budget before each step.

    Returns:
        List[Step]: Consolidated and ordered list of Step objects.
    """
    # Step 1: Create a map from existing step_objs keyed by step_number
    existing_map = {obj.step_number: obj for obj in step_objs}

    # Step 2: Determine total_steps
    if counts:
        # The initial budget from counts[0] does not necessarily equal the number of steps used
        existing_max = max((obj.step_number for obj in step_objs), default=0)
        steps_count = len(steps)
        total_steps = max(existing_max, steps_count)
    else:
        # If no counts, rely on steps and step_objs
        existing_max = max((obj.step_number for obj in step_objs), default=0)
        steps_count = len(steps)
        total_steps = max(existing_max, steps_count)

    # Step 3: Iterate through each step_number from 1 to total_steps
    for i in range(total_steps):
        step_number = i + 1
        description = steps[i] if i < len(steps) else None
        pre_count = counts[i] if i < len(counts) else None
        pre_count = int(pre_count) if pre_count is not None else None
        remaining_budget = pre_count - 1 if pre_count is not None else None
        reflection = reflections[i] if i < len(reflections) else None

        if step_number in existing_map:
            obj = existing_map[step_number]
            # Update description if necessary
            if description is not None and (
                not obj.description or obj.description != description
            ):
                obj.description = description
            # Update remaining_budget if necessary
            if (
                remaining_budget is not None
                and remaining_budget != obj.remaining_budget
            ):
                obj.remaining_budget = remaining_budget
        else:
            # Create a new Step object
            if description is None:
                description = f"Step {step_number}"  # Fallback description
            if remaining_budget is None:
                # Infer remaining_budget if missing
                remaining_budget = (
                    max(counts[0] - step_number, 0)
                    if counts
                    else max(total_steps - step_number, 0)
                )
            new_obj = Step(
                description, step_number, remaining_budget, reflection=reflection
            )
            existing_map[step_number] = new_obj

    # Step 4: Sort the steps by step_number
    step_objs_final = [existing_map[sn] for sn in sorted(existing_map.keys())]
    return step_objs_final


# ===========================
# Define the Process Steps Method
# ===========================


def process_steps(
    steps_objs: Optional[List[Step]],
    steps: List[str],
    counts: List[int],
    reflections: List[Optional[Reflection]],
    rewards: List[float],
    response: str,
    first_count: int,
    print_saver,  # Assuming print_saver is an object with print_and_store method
    interaction: Interaction,  # Assuming Interaction is a defined class
    task,
) -> Interaction:
    """
    Process and consolidate steps, reflections, and rewards into the interaction object.

    Args:
        steps_objs (Optional[List[Step]]): Existing list of Step objects.
        steps (List[str]): List of step descriptions.
        counts (List[int]): List of counts representing the budget before each step.
        reflections (List[Optional[Reflection]]): List of reflections.
        rewards (List[float]): List of rewards.
        response (str): Response string containing XML-like tags.
        first_count (int): The initial count value.
        print_saver: Object with a print_and_store method.
        interaction (Interaction): Object to store steps and reflections.
        task: The current task context.

    Returns:
        Interaction: The updated interaction object.
    """
    # Ensure steps_objs is a list
    steps_objs = steps_objs or []

    # ===========================
    # Step 1: Consolidate Steps
    # ===========================

    # Call the consolidate_steps function to align step_objs with steps and counts
    steps_objs = consolidate_steps(steps_objs, steps, counts, reflections)

    # ===========================
    # Step 2: Update Interaction with Consolidated Steps
    # ===========================

    for step_obj in steps_objs:
        if step_obj not in interaction.steps:
            interaction.steps.append(step_obj)

    # ===========================
    # Step 3: Process Reflections and Rewards
    # ===========================

    for i, step_obj in enumerate(steps_objs):
        # Handle reflections
        reflection = reflections[i] if i < len(reflections) else None
        if reflection is None:
            # If reflection is missing, generate it
            reflection = judge_step(
                step_obj, task
            )  # Assuming judge_step is defined elsewhere
            if reflection is not None:
                reflection.step_number = step_obj.step_number
                step_obj.reflection = reflection
                interaction.reflections.append(reflection)
        else:
            # Assign the reflection and reward to the step
            step_obj.reflection = reflection
            reflection.reward = rewards[i] if i < len(rewards) else 0.0
            if reflection not in interaction.reflections:
                interaction.reflections.append(reflection)

        # Handle rewards if not already set
        if reflection and reflection.reward == 0.0 and i < len(rewards):
            reflection.reward = rewards[i]

    # ===========================
    # Step 4: Extract Answer from Response
    # ===========================

    answer_match = re.search(
        r"<answer>(.*?)(?:</answer>|<final_reward>)", response, re.DOTALL
    )
    if answer_match:
        interaction.answer = answer_match.group(1).strip()

    # ===========================
    # Step 5: Extract Final Reward from Response
    # ===========================

    final_reward_match = re.search(
        r"<final_reward>(0\.\d+?|1\.0)</final_reward>", response, re.DOTALL
    )
    if final_reward_match:
        interaction.final_reward = float(final_reward_match.group(1))

    # ===========================
    # Final Assertion
    # ===========================

    assert isinstance(
        interaction, Interaction
    ), "Interaction object is not properly instantiated."

    return interaction


# ===========================
# Define Auxiliary Functions (Placeholders)
# ===========================


def judge_step(step: Step, task):
    """
    Placeholder function to generate a Reflection object based on a Step and a task.

    Args:
        step (Step): The step for which to generate the reflection.
        task: The current task context.

    Returns:
        Reflection: The generated reflection object.
    """
    # Implement the actual logic to generate a Reflection
    return Reflection(content=f"Reflection for {step.description}")


def cosine_similarity_custom(embedding1, embedding2):
    """
    Placeholder function to calculate cosine similarity between two embeddings.

    Args:
        embedding1: First embedding vector.
        embedding2: Second embedding vector.

    Returns:
        float: Cosine similarity score.
    """
    # Implement the actual cosine similarity logic
    return 1.0  # Placeholder value


def get_embedding(text):
    """
    Placeholder function to get an embedding vector for a given text.

    Args:
        text (str): The text to embed.

    Returns:
        Any: The embedding vector.
    """
    # Implement the actual embedding logic
    return []  # Placeholder value


# ===========================
# Example Usage
# ===========================
def parse_response(
    response: str,
    task: Task,
    steps_objs=None,
    reflections_objs=None,
    current_step_number: int = 0,
    current_remaining_budget: int = 0,
    interaction: Interaction = None,
    initial_budget: int = 0,
) -> Interaction:
    """
    Parses the OpenAI API response to extract steps, reflections, answers, and rewards.
    """
    # TODO: Implement a more robust response parsing mechanism, like using structured output from the model
    if interaction is None:
        interaction = Interaction(
            task=task, steps=[], reflections=[], answer="", final_reward=0.0
        )

    if response is None or not isinstance(response, str):
        return interaction

    # Extract steps
    if steps_objs is None or not isinstance(steps_objs, list):
        steps_objs = []

    if reflections_objs is None or not isinstance(reflections_objs, list):
        reflections_objs = []

    steps = re.findall(
        r"<step>(.*?)<(?:\/step|reflection|reward|step)>", response, re.DOTALL
    )
    # remove empty steps
    print_saver.print_and_store(f"Steps: {steps}")
    first_count = re.search(
        r"<count>(.*?)<(?:\/count|thinking|step|reflection|reward|count)>",
        response,
        re.DOTALL,
    )  # Represents the initial step budget
    if first_count and first_count.group(1).strip().isnumeric():
        first_count = int(first_count.group(1))
    elif steps_objs is not None and steps_objs != []:
        first_count = max([s.remaining_budget for s in steps_objs])
    elif task.plan:
        first_count = len(task.plan.steps) + len(
            [sub_task for step in task.plan.steps for sub_task in step.subtasks]
        )
    if initial_budget != 0 and first_count != 0:
        if first_count != initial_budget:
            print_saver.print_and_store(
                f"Initial budget mismatch. Adjusting initial budget. {first_count} vs initial: {initial_budget}"
            )
            first_count = initial_budget

    elif initial_budget != 0 and first_count == 0:
        first_count = initial_budget
        print_saver.print_and_store(
            f"Initial budget mismatch. Adjusting initial budget. {first_count}"
        )
    elif initial_budget == 0 and first_count == 0:
        initial_budget = 12
        first_count = initial_budget
    elif initial_budget == 0 and first_count != 0 and first_count is not None:
        initial_budget = first_count
    else:
        initial_budget = 12
        first_count = initial_budget

    counts = re.findall(
        r"<count>(.*?)<(?:\/count|thinking|step|reflection|reward|count)>",
        response,
        re.DOTALL,
    )
    # Extract reflections
    # Revert reflections to the original pattern
    reflections = re.findall(
        r"<reflection>(.*?)<(?:\/reflection|thinking|step|count|reward|reflection)>",
        response,
        re.DOTALL,
    )
    reflections = [reflection for reflection in reflections if reflection.strip() != ""]

    # Use the modified pattern for rewards
    rewards = re.findall(
        r"</reflection>\s*.*?<reward>(0\.\d+?|1\.0)<(?:/reward|thinking|step|reflection|count|reward?)>",
        response,
        re.DOTALL,
    )
    print_saver.print_and_store(f"Rewards: {rewards}")
    for i in range(len(rewards)):
        print_saver.print_and_store(f"Step {i + 1} reward: {rewards[i]}")
    i = 0
    interaction = process_steps(
        steps_objs=steps_objs,
        steps=steps,
        counts=counts,
        reflections=reflections,
        rewards=rewards,
        response=response,
        first_count=first_count,
        print_saver=print_saver,
        interaction=interaction,
        task=task,
    )

    return interaction


def main():
    # Initialize print_saver and interaction objects
    class PrintSaver:
        def print_and_store(self, message):
            print(message)  # For simplicity, just print

    print_saver = PrintSaver()
    interaction = Interaction()

    # Example data
    step_objs = [
        Step(description="Initialize project", step_number=1, remaining_budget=9),
        Step(description="Gather requirements", step_number=2, remaining_budget=8),
        Step(description="Design architecture", step_number=4, remaining_budget=6),
    ]

    steps = [
        "Initialize project",
        "Gather requirements",
        "Develop features",
        "Test the application",
        "Deploy to production",
    ]

    counts = [
        10,  # Before Step 1
        9,  # Before Step 2
        8,  # Before Step 3
        7,  # Before Step 4
        6,  # Before Step 5
    ]

    reflections = [
        None,  # Reflection for Step 1 will be generated
        None,  # Reflection for Step 2 will be generated
        None,  # Reflection for Step 3 will be generated
        None,  # Reflection for Step 4 will be generated
        None,  # Reflection for Step 5 will be generated
    ]

    rewards = [
        0.5,  # Reward for Step 1
        0.6,  # Reward for Step 2
        0.7,  # Reward for Step 3
        0.8,  # Reward for Step 4
        0.9,  # Reward for Step 5
    ]

    response = """
    <step>Initialize project</step>
    <count>10</count>
    <step>Gather requirements</step>
    <count>9</count>
    <step>Develop features</step>
    <count>8</count>
    <step>Test the application</step>
    <count>7</count>
    <step>Deploy to production</step>
    <count>6</count>
    <answer>Process completed successfully.</answer>
    <final_reward>1.0</final_reward>
    """

    first_count = 10  # Example initial count

    # Process the steps
    updated_interaction = process_steps(
        steps_objs=step_objs,
        steps=steps,
        counts=counts,
        reflections=reflections,
        rewards=rewards,
        response=response,
        first_count=first_count,
        print_saver=print_saver,
        interaction=interaction,
        task="Example Task",
    )

    # Print the updated interaction
    print(updated_interaction)


if __name__ == "__main__":
    main()
