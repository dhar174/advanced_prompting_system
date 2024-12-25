from calendar import c
from math import e
import random
import re
from turtle import st
from typing import List, Optional
from weakref import ref
import numpy as np
from regex import E, R
from zmq import has
from advanced_prompting import Task, print_saver, Interaction, Step, Reflection

# ===========================
# Define Necessary Classes
# ===========================


# ===========================
# Define Auxiliary Functions
# ===========================


def judge_step(step: Step, task: str) -> Reflection:
    """
    Generates a Reflection object based on a Step and a task.

    Args:
        step (Step): The step for which to generate the reflection.
        task (str): The current task context.

    Returns:
        Reflection: The generated reflection object.
    """
    # Implement the actual logic to generate a Reflection
    return Reflection(
        content=f"Reflection for {step.description}",
        step_number=step.step_number,
        reward=random.uniform(0.0, 1.0),
    )


def cosine_similarity_custom(embedding1, embedding2) -> float:
    """
    Calculates cosine similarity between two embeddings.

    Args:
        embedding1: First embedding vector.
        embedding2: Second embedding vector.

    Returns:
        float: Cosine similarity score.
    """
    if not embedding1 or not embedding2:
        return 0.0
    vec1 = np.array(embedding1)
    vec2 = np.array(embedding2)
    if vec1.size == 0 or vec2.size == 0:
        return 0.0
    cosine_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return float(cosine_sim)


def get_embedding(text: str):
    """
    Retrieves an embedding vector for a given text.

    Args:
        text (str): The text to embed.

    Returns:
        List[float]: The embedding vector.
    """
    # Implement the actual embedding logic, e.g., using a pre-trained model
    # For simplicity, returning a dummy vector
    return [1.0] * 300  # Example: 300-dimensional vector


# ===========================
# Define Helper Functions
# ===========================


def handle_length_mismatch(
    steps_objs: List[Step],
    steps: List[str],
    counts: List[str | int],
    reflections: List[Optional[Reflection | str]],
    rewards: List[float | str],
    print_saver,
    interaction: Interaction,
    task: str,
    first_count: int,
) -> None:
    """
    Adjusts steps_objs and steps to handle length mismatches between them.

    Args:
        steps_objs (List[Step]): Existing list of Step objects.
        steps (List[str]): List of step descriptions.
        counts (counts: List[str | int]): List of counts representing the budget before each step.
        reflections (List[Optional[Reflection | str]]): List of reflections.
        rewards (List[float | str]): List of rewards.
        print_saver: Object with a print_and_store method.
        interaction (Interaction): Interaction object to update.
        task (str): Current task context.
        first_count (int): Initial count value.
    """
    if len(steps_objs) > len(steps):
        print_saver.print_and_store(
            f"Steps and steps_objs length mismatch. Adjusting steps. {len(steps_objs)} vs {len(steps)}"
        )
        # Handle steps_objs larger than steps
        missing_steps = {
            step.description.strip(): step.step_number
            for step in steps_objs
            if step.description.strip() not in [s.strip() for s in steps]
        }
        for description, step_num in missing_steps.items():
            print_saver.print_and_store(f"Missing step: {description}")
            # Insert missing steps
            steps, counts, reflections, rewards, steps_objs = insert_missing_step(
                step_num,
                description,
                counts,
                reflections,
                rewards,
                steps,
                steps_objs,
                print_saver,
                interaction,
                task,
            )
    elif len(steps_objs) < len(steps):
        print_saver.print_and_store("Stepsobj smaller than steps. Adjusting steps.")
        # Handle steps larger than steps_objs
        missing_steps = {
            step.strip(): idx
            for idx, step in enumerate(steps)
            if step.strip() not in [s.description.strip() for s in steps_objs]
        }
        for description, idx in missing_steps.items():
            print_saver.print_and_store(f"Missing step: {description}")
            # Insert missing steps
            steps, counts, reflections, rewards, steps_objs = insert_missing_step(
                idx + 1,
                description,
                counts,
                reflections,
                rewards,
                steps,
                steps_objs,
                print_saver,
                interaction,
                task,
            )


def insert_missing_step(
    step_num: int,
    description: str,
    counts: List[str | int],
    reflections: List[Optional[Reflection | str]],
    rewards: List[float],
    steps: List[str],
    steps_objs: List[Step],
    print_saver,
    interaction: Interaction,
    task: str,
) -> tuple:
    """
    Inserts a missing step into steps_objs, steps, counts, reflections, and rewards.

    Args:
        step_num (int): The step number to insert.
        description (str): Description of the step.
        counts (List[str | int]): List of counts.
        reflections (List[Optional[Reflection | str]]): List of reflections.
        rewards (List[float]): List of rewards.
        steps (List[str]): List of step descriptions.
        steps_objs (List[Step]): Existing list of Step objects.
        print_saver: Object with a print_and_store method.
        interaction (Interaction): Interaction object to update.
        task (str): Current task context.
    """
    # Validate step number is sequential
    if step_num > 1 and not any(s.step_number == step_num - 1 for s in steps_objs):
        raise ValueError(f"Non-sequential step number found: {step_num}")

    # Insert step into steps and counts
    steps.insert(step_num - 1, description)
    if step_num - 2 >= 0 and (step_num - 1) < len(counts):
        counts.insert(step_num - 1, int(counts[step_num - 2]) - 1)
    else:
        counts.append(int(counts[-1]) - 1 if counts else 0)
    print_saver.print_and_store(f"Inserted step: {description}")
    # Handle reflections
    if step_num <= len(reflections):
        reflection = reflections[step_num - 1]
        if reflection is None:
            reflection = judge_step(
                Step(description, step_num, counts[max((step_num - 1), 0)]), task
            )
            reflections.insert(step_num - 1, reflection)
    elif step_num - 1 == len(reflections):
        reflection = judge_step(Step(description, step_num, counts[step_num - 1]), task)
        reflections.append(reflection)
    elif step_num - 1 > len(reflections):
        reflection = judge_step(Step(description, step_num, counts[step_num - 1]), task)
        reflections.insert(step_num - 1, reflection)

    # Handle rewards
    if step_num - 1 < len(rewards):
        rewards.insert(step_num - 1, 0.0)  # Placeholder, will be updated later
    else:
        rewards.append(0.0)
    if not isinstance(reflections[step_num - 1], Reflection):
        reflections[step_num - 1] = Reflection(
            content=str(reflections[step_num - 1]),
            reward=rewards[step_num - 1],
            step_number=step_num,
        )
        reflections[step_num - 1] = reflection
    # Create and append the new Step object
    new_step = Step(description, step_num, counts[step_num - 1], reflection)
    steps_objs.insert(step_num - 1, new_step)
    interaction.steps.append(new_step)
    interaction.reflections.append(reflections[step_num - 1])
    return (steps, counts, reflections, rewards, steps_objs)


def validate_steps(
    steps_objs: List[Step],
    steps: List[str],
    counts: List[str | int],
    reflections: List[Optional[Reflection | str]],
    rewards: List[float],
    print_saver,
    interaction: Interaction,
    task: str,
    first_count: int,
) -> None:
    """
    Validates and adjusts steps, steps_objs, counts, reflections, and rewards.

    Args:
        steps_objs (List[Step]): Existing list of Step objects.
        steps (List[str]): List of step descriptions.
        counts (List[str | int]): List of counts.
        reflections (List[Optional[Reflection | str]]): List of reflections.
        rewards (List[float]): List of rewards.
        print_saver: Object with a print_and_store method.
        interaction (Interaction): Interaction object to update.
        task (str): Current task context.
        first_count (int): Initial count value.
    """
    handle_length_mismatch(
        steps_objs,
        steps,
        counts,
        reflections,
        rewards,
        print_saver,
        interaction,
        task,
        first_count,
    )
    # Additional validation and synchronization can be added here as needed


def ensure_content_similarity(
    steps_objs: List[Step],
    steps: List[str],
    first_count: int,
    print_saver,
    interaction: Interaction,
    task: str,
) -> None:
    """
    Ensures content similarity between steps_objs and steps using cosine similarity.

    Args:
        steps_objs (List[Step]): Existing list of Step objects.
        steps (List[str]): List of step descriptions.
        first_count (int): Initial count value.
        print_saver: Object with a print_and_store method.
        interaction (Interaction): Interaction object to update.
        task (str): Current task context.
    """
    for i, step_desc in enumerate(steps):
        if i >= len(steps_objs):
            break
        step_obj = steps_objs[i]
        similarity = cosine_similarity_custom(
            get_embedding(step_obj.description.strip()),
            get_embedding(step_desc.strip()),
        )
        if similarity < 0.9:
            print_saver.print_and_store(
                f"Step content mismatch at step {i+1}. Updating description."
            )
            step_obj.description = step_desc.strip()
            # Optionally, regenerate reflection if content changes significantly
            step_obj.reflection = judge_step(step_obj, task)
            interaction.reflections.append(step_obj.reflection)


# ===========================
# Define the consolidate_steps Function
# ===========================


def remove_nonnumeric_counts(
    counts: List[str | int], first_count: int
) -> List[str | int]:
    """
    Remove non-numeric values from the counts list.

    Args:
        counts (List[str | int]): List of counts.

    Returns:
        List[str | int]: List of counts with non-numeric values removed.
    """
    new_counts = []
    len_counts = len(counts)
    for count in counts:
        if isinstance(count, (int, float)) or (
            isinstance(count, str) and count.isdigit()
        ):
            new_counts.append(count)
        elif isinstance(count, str):
            new_digits = "".join([c for c in count if c.isdigit()])
            try:
                new_counts.append(int(new_digits))
            except ValueError:
                try:
                    new_count = re.search(r"\d+", count)
                    if new_count:
                        new_counts.append(int(new_count.group(0)))
                except Exception as e:
                    print(e)
                    new_counts.append(0)
        else:
            if count is not None:
                try:
                    new_counts.append(int(count))
                except ValueError:
                    try:
                        new_count = re.search(r"\d+", count)
                        if new_count:
                            new_counts.append(int(new_count.group(0)))
                    except Exception as e:
                        print(e)
                        new_counts.append(0)
    if len(new_counts) < len_counts:
        if len(new_counts) == 0:
            new_counts = [first_count]
        new_counts.extend(
            [new_counts[-1] - i for i in range(len_counts - len(new_counts))]
        )
        print_saver.print_and_store(
            f"Counts length mismatch (less than started with). Adjusting counts. {len(new_counts)} vs {len_counts}"
        )
    elif len(new_counts) > len_counts:
        new_counts = new_counts[:len_counts]
        print_saver.print_and_store(
            f"Counts length mismatch (more than started with). Adjusting counts. {len(new_counts)} vs {len_counts}"
        )
    return new_counts


def consolidate_steps(
    steps_objs: List[Step],
    steps: List[str],
    counts: List[str | int],
    reflections: List[Reflection],
    first_count: int,
) -> List[Step]:
    """
    Consolidate the steps from steps_objs, steps, counts, and reflections into a single ordered list of Step objects.

    This function aligns and merges the provided lists of step descriptions and corresponding Step objects, ensuring consistency in step numbering and handling discrepancies such as duplicates, gaps, and order inversions.

    The consolidation process involves the following logic for updating `step_obj.step_number`:

    1. **Equal Lengths (`len(steps) == len(steps_objs)`):**
        - **Gap Handling:** Update `step_obj.step_number` by identifying gaps between the current step number and its neighbors. Adjust the step number based on the nearest preceding and following Step objects that maintain both the numerical sequence and the original index order.
        - **Order Verification:** Ensure that the sequence order hasn't been disrupted. If the order is inconsistent, apply a different strategy to reorder the Step objects appropriately.

    2. **More Steps Strings than Step Objects (`len(steps) > len(steps_objs)`):**
        - **Gap Correspondence:** Determine if the gap in step numbers corresponds to the number of missing step descriptions in `steps_objs`. Adjust `step_obj.step_number` by considering the number of steps missing on either side of the current step number.
        - **Insertion Logic:** Insert missing steps at positions that reflect the identified gaps, ensuring that the step numbering remains sequential and consistent with the provided descriptions.

    3. **Fewer Steps Strings than Step Objects (`len(steps) < len(steps_objs)`):**
        - **Gap Correspondence:** Similar to the previous case, identify if the gap aligns with the number of missing step descriptions. Update `step_obj.step_number` accordingly by analyzing the surrounding steps.
        - **Adjustment Strategy:** Reassign step numbers to accommodate the discrepancies, ensuring that all Step objects are correctly numbered without overlaps or omissions.

    4. **Order Discrepancies:**
        - **Integrity Check:** Verify that the order of steps has not been compromised. If inconsistencies are detected, implement a reordering mechanism that realigns the Step objects based on both their numerical order and their original positions.
        - **Alternative Approach:** In cases where the standard gap analysis fails due to complex order issues, adopt an alternative strategy to systematically reorder the steps, ensuring logical progression and accurate numbering.

    Args:
        steps_objs (List[Step]): Existing list of Step objects.
        steps (List[str]): List of step descriptions.
        counts (List[str | int]): List of counts representing the budget before each step.

    Returns:
        List[Step]: Consolidated and ordered list of Step objects.
    """
    # Step 1: Create a map from existing steps_objs keyed by step_number
    existing_map = {obj.step_number: obj for obj in steps_objs}

    # Step 2: Determine total_steps
    total_steps = max(len(steps), len(steps_objs))

    # Step 3: Iterate through each step_number from 1 to total_steps
    steps_temp = steps.copy()
    steps_objs_temp = steps_objs.copy()
    for i in range(total_steps):
        step_number = i + 1
        reflection = reflections[i] if i < len(reflections) else None
        if i == 0:
            # Store original sequence
            original_steps = [(obj.step_number, obj.description) for obj in steps_objs]
            original_numbers = [s[0] for s in original_steps]
            # First pass: identify all issues
            step_issues = {
                "duplicates": [
                    n for n in set(original_numbers) if original_numbers.count(n) > 1
                ],
                "gaps": [
                    (a, b)
                    for a, b in zip(original_numbers[:-1], original_numbers[1:])
                    if b - a > 1
                ],
                "inversions": [
                    (a, b)
                    for a, b in zip(original_numbers[:-1], original_numbers[1:])
                    if b <= a
                ],
            }

            # Log initial analysis
            print_saver.print_and_store(f"Original sequence analysis:")
            for issue_type, issues in step_issues.items():
                if issues:
                    print_saver.print_and_store(f"- {issue_type.title()}: {issues}")

            # Analyze sequence patterns
            has_gaps = any(
                b - a > 1 for a, b in zip(original_numbers[:-1], original_numbers[1:])
            )
            is_strictly_increasing = all(
                b > a for a, b in zip(original_numbers[:-1], original_numbers[1:])
            )
            duplicates = [
                n for n in set(original_numbers) if original_numbers.count(n) > 1
            ]

            # Log diagnostics
            print_saver.print_and_store(f"Original sequence: {original_numbers}")
            print_saver.print_and_store(f"Has gaps: {has_gaps}")
            print_saver.print_and_store(
                f"Is strictly increasing: {is_strictly_increasing}"
            )
            if duplicates:
                print_saver.print_and_store(
                    f"Duplicate step numbers found: {duplicates}"
                )

            # Correct step numbers while logging changes
            for idx, step_obj in enumerate(steps_objs_temp, start=1):
                if step_obj.step_number != idx:
                    try:
                        original_idx = original_steps.index(
                            (step_obj.step_number, step_obj.description)
                        )
                        gap = step_obj.step_number - idx  # Calculate the gap

                        print_saver.print_and_store(
                            f"Correcting step number for '{step_obj.description}': {step_obj.step_number} -> {idx}. Gap: {gap}. Original index: {original_idx}."
                        )
                    except ValueError:
                        original_idx = -1  # Indicates not found
                        print_saver.print_and_store(
                            f"Step '{step_obj.description}' with number {step_obj.step_number} not found in original_steps. Assigning new step number {idx}. Gap: {step_obj.step_number - idx}. Original index: {original_idx}. (-1 indicates not found)"
                        )
                    # Update the step number based on several criteria, listed below with logical reasoning for each condition:
                    # If len(steps) == len(steps_objs):
                    #     - Check for gaps in step_number sequence and adjust based on nearest surrounding Step objects.
                    #     - Ensure the order of steps_objs matches the order of steps to maintain consistency.
                    # elif len(steps) > len(steps_objs):
                    #     - Identify missing step descriptions that arent in steps_objs that are in `steps`.
                    #     - Determine if the total gap corresponds to the total number of missing step descriptions, or if the gap corresponds to the difference between the step number of the current step_obj and the step number of the next or previous Step object that is in both lists, and if that gap corresponds to the number of missing steps on either side before reaching a description that is in both lists.
                    #     - Insert new Step objects at appropriate positions based on missing descriptions.
                    #     - Reassign step_number to maintain sequential integrity.
                    #     - steps is the source of truth if it is longer than steps_objs and is equal to total_steps.
                    #     - if gap is equal to the number of missing steps on either side (ie, the gap is due to missing steps_objs entries), adjust step_obj.step_number based on missing steps on either side, as long as the index of the `steps` entries that are missing from `steps_objs` is the same as the missing step number before reaching either the end or the beginning of the list or the previous/next step that is in both lists, in either direction.
                    #     - If the gap is not due to missing steps_objs entries, adjust step_obj.step_number based on the nearest preceding and following Step objects that maintain both the numerical sequence and the original index order.
                    #     - If the order of steps_objs has been compromised, apply a different strategy to reorder the Step objects appropriately.
                    #     - For example, if the idx is less than the step_obj.step_number, lets say idx = 2 and step_obj.step_number = 5, we would expect that the entry at `steps[idx]` would not match the description of the current step_obj so we would need to find the next entry in steps that matches the description of the current step_obj and if the index of that entry plus 1 is equal to the step_obj.step_number, we would know (and should confirm) that if the description of steps_objs[idx] is equal to the description of steps[idx], then we can safely assume that steps[idx:step_obj.step_number] are the missing steps, especially if the description of steps_objs[idx - 1] is equal to the description of steps[idx - 1]. If this is the case, we can safely assume that the missing steps are steps[idx:step_obj.step_number] and we can insert them into steps_objs at the correct indexes.
                    #     - If the idx is greater than the step_obj.step_number, lets say idx = 5 and step_obj.step_number = 2, this might indicate that the order of steps_objs has been compromised and we should check if the description of steps[idx] is equal to the description of the current step_obj. If it is, it likely means the order is messed up, and we need to next determine if each entry by index in steps matches the description of the corresponding entry in steps_objs. If they do, the order is likely correct but the numbers arent, which we can verify by a) checking whether the largest step number in steps_objs is equal to the length of steps_objs (indicating perhaps steps_objs is missing steps from the end of steps, so check if the last few (len(steps) - max(steps_objs.step_number)) steps in steps are missing, otherwise, if the largest step number in steps_objs is equal to the length of `steps`, then we can assume that steps_objs is missing steps from the somewhere in the middle, so we need to verify whether the step numbers are correct by checking if the step number of each step in steps_objs is equal to the index of that step in steps, and if the subset of steps that are in both lists are in the same order.) and b) we can check if the step numbers are correct by checking if the step number of each step in steps_objs is equal to the index of that step in steps, and if not (or in addition to) checking if the subset of steps that are in both lists are in the same order. If they are not, we need to reorder the steps_objs list to match the order of the steps list as long as the length of counts is equal to the length of the steps list and the the original order of counts decreases from on index to the next and never increases or repeats a count. If either the length of counts is not equal to the length of steps or the original order of counts increases from one index to the next, we need to check if the either a) the length of counts is equal to the length of steps_objs and all the counts are in the same order as the original counts or b) the length of counts is equal to the length of steps_objs and the counts are not in the same order as the original counts but the counts correspond exactly to the step numbers of the steps_objs list. If a) is true and b) is not, if the order of step_number values is ascending and sequential, the order is correct and if the number of matching counts to step_numbers subtracted from the length of counts equals the number of missing steps, we should doublecheck that the steps in `steps` that correspond to the indexes of `counts` entries that did not match any step_number values in steps_objs are the missing steps. If this is the case, we can insert them into steps_objs at the correct indexes, using the existing step_number values from the entries in steps_objs that are in the same order as the missing steps in steps. If b) is true, we need to check if the counts correspond to the step numbers of the steps_objs list. If they do and the step_numbers only increase from one index to the next, we can assume the order is correct and the counts are correct and we can assume that the missing steps are the steps in `steps` that correspond to the indexes of `counts` entries that did not match any step_number values in steps_objs if the number of missing string descriptions that are in `steps` but not in `[step_obj.description for step_obj in steps_objs]`. If this is the case, we can insert them into steps_objs at the correct indexes, using the existing step_number values from the entries in steps_objs that are in the same order as the missing steps in steps. If the counts do not correspond to the step numbers of the steps_objs list, we need to check if the counts are in the same order as the original counts and if the length of counts is equal to the length of steps_objs. If both are true, we can assume that the order is correct and the counts are correct and we can assume that the missing steps are the steps in `steps` that correspond to the indexes of `counts` entries that did not match any step_number values in steps_objs. If this is the case, we can insert them into steps_objs at the correct indexes, using the existing step_number values from the entries in steps_objs that are in the same order as the missing steps in steps.
                    # elif len(steps) < len(steps_objs):
                    #     - Identify missing step descriptions that arent in `steps` list that are in `steps_objs`.
                    #     - Determine if the total gap corresponds to the total number of missing step descriptions, or if the gap corresponds to the difference between the step number of the current step_obj and the step number of the next or previous Step object that is in both lists, and if that gap corresponds to the number of missing steps on either side before reaching a description that is in both lists.
                    #     - Insert new Step objects at appropriate positions based on missing descriptions.
                    #     - Reassign step_number to maintain sequential integrity.
                    #     - steps_objs is the source of truth if it is longer than steps and is equal to total_steps.
                    #     - Remembering that steps_objs is longer than the `steps` list, if the gap is equal to the number of missing steps on either side (ie, the gap is due to missing steps entries), if steps_objs is the source of truth because it is longer than the `steps` list, we can assume that the missing steps are steps in `steps_objs` that are not in `steps` and we can insert them into steps at the correct indexes, as long as the step_number values of steps_objs are in ascending order and sequential. If the step_number values are not in ascending order and sequential, we need to check if the counts are in ascending order and sequential and if the counts are in ascending order and sequential, we can assume that the missing steps are the steps in `steps_objs` that are not in `steps` and we can insert them into steps at the correct indexes, using the existing step_number values from the entries in steps_objs that are in the same order as the missing steps in steps. If the counts are not in ascending order and sequential, we need to check if the counts correspond to the step numbers of the steps_objs list. If they do and the step_numbers only increase from one index to the next, we can assume the order is correct and the counts are correct and we can assume that the missing steps are the steps in `steps` that correspond to the indexes of `counts` entries that did not match any step_number values in steps_objs. If this is the case, we can insert them into steps_objs at the correct indexes, using the existing step_number values from the entries in steps_objs that are in the same order as the missing steps in steps. If the counts do not correspond to the step numbers of the steps_objs list, we need to check if the counts are in the same order as the original counts and if the length of counts is equal to the length of steps_objs. If both are true, we can assume that the order is correct and the counts are correct and we can assume that the missing steps are the steps in `steps` that correspond to the indexes of `counts` entries that did not match any step_number values in steps_objs. If this is the case, we can insert them into steps_objs at the correct indexes, using the existing step_number values from the entries in steps_objs that are in the same order as the missing steps in steps, as long as each subset of step descriptions that are in both lists are in the same order as each other. If they are not, based on a) which list is longer and b) whether the step_number values are in ascending order and sequential and c) whether the counts are in ascending order and sequential and d) whether the counts correspond to the step numbers of the steps_objs list, we can determine which list is the source of truth and insert the missing steps into the other list at the correct indexes.
                    #     - If the gap is not due to missing steps entries, adjust step_obj.step_number based on the nearest preceding and following Step objects that maintain both the numerical sequence and the original index order.
                    # Check for order discrepancies and apply alternative logic if the order has been compromised.
                    step_obj.step_number = idx

            # Re-analyze after correction
            corrected_numbers = [obj.step_number for obj in steps_objs_temp]
            has_gaps = any(
                b - a > 1 for a, b in zip(corrected_numbers[:-1], corrected_numbers[1:])
            )
            is_strictly_increasing = all(
                b > a for a, b in zip(corrected_numbers[:-1], corrected_numbers[1:])
            )
            duplicates = [
                n for n in set(corrected_numbers) if corrected_numbers.count(n) > 1
            ]

            # Log post-correction diagnostics
            print_saver.print_and_store(f"Corrected sequence: {corrected_numbers}")
            print_saver.print_and_store(f"Has gaps after correction: {has_gaps}")
            print_saver.print_and_store(
                f"Is strictly increasing after correction: {is_strictly_increasing}"
            )
            if duplicates:
                for duplicate in duplicates:
                    print_saver.print_and_store(
                        f"Handling duplicate step number: {duplicate}"
                    )
                    # Example strategy: Increment step numbers of subsequent steps
                    for step_obj in steps_objs:
                        if step_obj.step_number == duplicate:
                            step_obj.step_number += 1
                            print_saver.print_and_store(
                                f"Incremented step number for '{step_obj.description}' to {step_obj.step_number}."
                            )
        if len(steps) < total_steps and total_steps == len(steps_objs):
            print_saver.print_and_store(
                f"Steps length mismatch (less in steps than in steps_objs, and len of steps_objs is equal to total_steps). Adjusting steps. steps: {len(steps)} vs steps_objs: {len(steps_objs)} and total_steps: {total_steps}"
            )
            missing_objs = [obj for obj in steps_objs if obj.description not in steps]
            for idx, obj in enumerate(missing_objs):
                # Find the next step that appears in both lists
                next_match = None
                obj_index = steps_objs.index(obj)
                for following in steps_objs[obj_index + 1 :]:
                    if following.description in steps:
                        next_match = following.description
                        break
                # If there's a subsequent match, insert right before it; otherwise append
                if next_match:
                    insert_idx = steps.index(next_match)
                else:
                    insert_idx = len(steps)
                steps_temp.insert(insert_idx, obj.description)
        elif len(steps) > total_steps and total_steps == len(steps_objs):
            print_saver.print_and_store(
                f"Steps length mismatch (more in steps than in steps_objs, and len of steps_objs is equal to total_steps). Adjusting steps. steps: {len(steps)} vs steps_objs: {len(steps_objs)} and total_steps: {total_steps}"
            )
            missing_steps = [
                step
                for step in steps
                if step not in [obj.description for obj in steps_objs]
            ]
            for idx, step in enumerate(missing_steps):
                # Find the next step that appears in both lists
                next_match = None
                step_index = steps.index(step)
                for following in steps[step_index + 1 :]:
                    if following in [obj.description for obj in steps_objs]:
                        next_match = following
                        break
                # If there's a subsequent match, insert right before it; otherwise append
                if next_match:
                    insert_idx = [obj.description for obj in steps_objs].index(
                        next_match
                    )
                else:
                    insert_idx = len(steps_objs)
                steps_objs_temp.insert(
                    insert_idx,
                    Step(
                        step,
                        insert_idx + 1,
                        counts[insert_idx] if insert_idx < len(counts) else 0,
                    ),
                )
        elif len(steps) < total_steps and total_steps < len(steps_objs):
            print_saver.print_and_store(
                f"Steps length mismatch (less in steps than in steps_objs, and len of steps_objs is greater than total_steps). Adjusting steps. steps: {len(steps)} vs steps_objs: {len(steps_objs)} and total_steps: {total_steps}"
            )
            missing_objs = [obj for obj in steps_objs if obj.description not in steps]
            for idx, obj in enumerate(missing_objs):
                # Find the next step that appears in both lists
                next_match = None
                obj_index = steps_objs.index(obj)
                for following in steps_objs[obj_index + 1 :]:
                    if following.description in steps:
                        next_match = following.description
                        break
                # If there's a subsequent match, insert right before it; otherwise append
                if next_match:
                    insert_idx = steps.index(next_match)
                else:
                    insert_idx = len(steps)
                steps_temp.insert(insert_idx, obj.description)
        elif len(steps) > total_steps and total_steps < len(steps_objs):
            print_saver.print_and_store(
                f"Steps length mismatch (more in steps than in steps_objs, and len of steps_objs is greater than total_steps). Adjusting steps. steps: {len(steps)} vs steps_objs: {len(steps_objs)} and total_steps: {total_steps}"
            )
            missing_steps = [
                step
                for step in steps
                if step not in [obj.description for obj in steps_objs]
            ]
            for idx, step in enumerate(missing_steps):
                # Find the next step that appears in both lists
                next_match = None
                step_index = steps.index(step)
                for following in steps[step_index + 1 :]:
                    if following in [obj.description for obj in steps_objs]:
                        next_match = following
                        break
                # If there's a subsequent match, insert right before it; otherwise append
                if next_match:
                    insert_idx = [obj.description for obj in steps_objs].index(
                        next_match
                    )
                else:
                    insert_idx = len(steps_objs)
                steps_objs_temp.insert(
                    insert_idx,
                    Step(
                        step,
                        insert_idx + 1,
                        counts[insert_idx] if insert_idx < len(counts) else 0,
                    ),
                )
        elif len(steps) == total_steps and total_steps < len(steps_objs):
            print_saver.print_and_store(
                f"Steps length mismatch (equal in steps and steps_objs, and len of steps_objs is greater than total_steps). Adjusting steps. steps: {len(steps)} vs steps_objs: {len(steps_objs)} and total_steps: {total_steps}"
            )
            missing_objs = [obj for obj in steps_objs if obj.description not in steps]
            for idx, obj in enumerate(missing_objs):
                # Find the next step that appears in both lists
                next_match = None
                obj_index = steps_objs.index(obj)
                for following in steps_objs[obj_index + 1 :]:
                    if following.description in steps:
                        next_match = following.description
                        break
                # If there's a subsequent match, insert right before it; otherwise append
                if next_match:
                    insert_idx = steps.index(next_match)
                else:
                    insert_idx = len(steps)
                steps_temp.insert(insert_idx, obj.description)
        elif len(steps) == total_steps and total_steps > len(steps_objs):
            print_saver.print_and_store(
                f"Steps length mismatch (equal in steps and steps_objs, and len of steps_objs is less than total_steps). Adjusting steps. steps: {len(steps)} vs steps_objs: {len(steps_objs)} and total_steps: {total_steps}"
            )
            missing_steps = [
                step
                for step in steps
                if step not in [obj.description for obj in steps_objs]
            ]
            for idx, step in enumerate(missing_steps):
                # Find the next step that appears in both lists
                next_match = None
                step_index = steps.index(step)
                for following in steps[step_index + 1 :]:
                    if following in [obj.description for obj in steps_objs]:
                        next_match = following
                        break
                # If there's a subsequent match, insert right before it; otherwise append
                if next_match:
                    insert_idx = [obj.description for obj in steps_objs].index(
                        next_match
                    )
                else:
                    insert_idx = len(steps_objs)
                steps_objs_temp.insert(
                    insert_idx,
                    Step(
                        step,
                        insert_idx + 1,
                        counts[insert_idx] if insert_idx < len(counts) else 0,
                    ),
                )
        elif len(steps) < total_steps and total_steps > len(steps_objs):
            # This result indicates that the steps list is shorter than the total steps and the steps_objs list is shorter than the total steps. So we should check if the steps list is shorter than the steps_objs list or vice versa as the total steps is greater than both. Even if they are of the same length, we need to check if either has steps not in the other, as each could have unique steps, and adding them together might result in both lists equaling the total steps, if we're lucky.
            # This result indicates that the steps list is shorter than the total steps and the steps_objs list is also shorter than the total steps.
            # We should check if one list is shorter than the other or if they are the same length, then handle missing items in each accordingly.

            if len(steps) < len(steps_objs):
                print_saver.print_and_store(
                    f"Steps length mismatch (less in steps than in steps_objs, total_steps is higher). Adjusting steps. steps: {len(steps)} vs steps_objs: {len(steps_objs)} and total_steps: {total_steps}"
                )
                missing_objs = [
                    obj for obj in steps_objs if obj.description not in steps
                ]
                for idx, obj in enumerate(missing_objs):
                    next_match = None
                    obj_index = steps_objs.index(obj)
                    for following in steps_objs[obj_index + 1 :]:
                        if following.description in steps:
                            next_match = following.description
                            break
                    if next_match:
                        insert_idx = steps.index(next_match)
                    else:
                        insert_idx = len(steps)
                    steps_temp.insert(insert_idx, obj.description)

            elif len(steps) > len(steps_objs):
                print_saver.print_and_store(
                    f"Steps length mismatch (more in steps than in steps_objs, total_steps is higher). Adjusting steps_objs. steps: {len(steps)} vs steps_objs: {len(steps_objs)} and total_steps: {total_steps}"
                )
                missing_steps = [
                    step
                    for step in steps
                    if step not in [obj.description for obj in steps_objs]
                ]
                for idx, step_value in enumerate(missing_steps):
                    next_match = None
                    step_index = steps.index(step_value)
                    for following in steps[step_index + 1 :]:
                        if following in [obj.description for obj in steps_objs]:
                            next_match = following
                            break
                    if next_match:
                        insert_idx = [obj.description for obj in steps_objs].index(
                            next_match
                        )
                    else:
                        insert_idx = len(steps_objs)
                    steps_objs_temp.insert(
                        insert_idx,
                        Step(
                            step_value,
                            insert_idx + 1,
                            counts[insert_idx] if insert_idx < len(counts) else 0,
                        ),
                    )

            elif len(steps) == len(steps_objs):
                # len(steps) == len(steps_objs), but both are still less than total_steps.
                # We try reconciling any missing content in each list to move toward total_steps.
                print_saver.print_and_store(
                    f"Steps length mismatch (equal in steps and steps_objs, but both less than total_steps). Adjusting. steps: {len(steps)} vs steps_objs: {len(steps_objs)} and total_steps: {total_steps}"
                )
                missing_objs = [
                    obj for obj in steps_objs if obj.description not in steps
                ]
                for obj in missing_objs:
                    obj_index = steps_objs.index(obj)
                    next_match = None
                    for following in steps_objs[obj_index + 1 :]:
                        if following.description in steps:
                            next_match = following.description
                            break
                    if next_match:
                        insert_idx = steps.index(next_match)
                    else:
                        insert_idx = len(steps)
                    steps_temp.insert(insert_idx, obj.description)

                missing_steps = [
                    step_value
                    for step_value in steps
                    if step_value not in [o.description for o in steps_objs]
                ]
                for step_value in missing_steps:
                    step_index = steps.index(step_value)
                    next_match = None
                    for following in steps[step_index + 1 :]:
                        if following in [o.description for o in steps_objs]:
                            next_match = following
                            break
                    if next_match:
                        insert_idx = [o.description for o in steps_objs].index(
                            next_match
                        )
                    else:
                        insert_idx = len(steps_objs)
                    steps_objs_temp.insert(
                        insert_idx,
                        Step(
                            step_value,
                            insert_idx + 1,
                            counts[insert_idx] if insert_idx < len(counts) else 0,
                        ),
                    )

        elif len(steps) > total_steps and total_steps > len(steps_objs):
            # This condition means that the steps list is longer than the total steps and the steps_objs list is also shorter than the total steps, meaning that the steps list is longer than the steps_objs list, but somehow the total steps is greater than both. #
            # This is a tricky situation, but we can handle it by adding the missing steps from the steps list to the steps_objs list.
            print_saver.print_and_store(
                f"Steps length mismatch (more in steps than in steps_objs, and total_steps is in between). Adjusting steps_objs. steps: {len(steps)} vs steps_objs: {len(steps_objs)} and total_steps: {total_steps}"
            )
            missing_steps = [
                step
                for step in steps
                if step not in [obj.description for obj in steps_objs]
            ]
            for idx, step in enumerate(missing_steps):
                next_match = None
                step_index = steps.index(step)
                for following in steps[step_index + 1 :]:
                    if following in [obj.description for obj in steps_objs]:
                        next_match = following
                        break
                if next_match:
                    insert_idx = [obj.description for obj in steps_objs].index(
                        next_match
                    )
                else:
                    insert_idx = len(steps_objs)
                steps_objs_temp.insert(
                    insert_idx,
                    Step(
                        step,
                        insert_idx + 1,
                        counts[insert_idx] if insert_idx < len(counts) else 0,
                    ),
                )
        elif len(steps) == len(steps_objs) and total_steps < max(
            len(steps), len(steps_objs)
        ):
            print_saver.print_and_store(
                f"Steps length mismatch (equal in steps and steps_objs, but both greater than total_steps). Adjusting. steps: {len(steps)} vs steps_objs: {len(steps_objs)} and total_steps: {total_steps}"
            )
            missing_objs = [obj for obj in steps_objs if obj.description not in steps]
            for obj in missing_objs:
                obj_index = steps_objs.index(obj)
                next_match = None
                for following in steps_objs[obj_index + 1 :]:
                    if following.description in steps:
                        next_match = following.description
                        break
                if next_match:
                    insert_idx = steps.index(next_match)
                else:
                    insert_idx = len(steps)
                steps_temp.insert(insert_idx, obj.description)

            missing_steps = [
                step_value
                for step_value in steps
                if step_value not in [o.description for o in steps_objs]
            ]
            for step_value in missing_steps:
                step_index = steps.index(step_value)
                next_match = None
                for following in steps[step_index + 1 :]:
                    if following in [o.description for o in steps_objs]:
                        next_match = following
                        break
                if next_match:
                    insert_idx = [o.description for o in steps_objs].index(next_match)
                else:
                    insert_idx = len(steps_objs)
                steps_objs_temp.insert(
                    insert_idx,
                    Step(
                        step_value,
                        insert_idx + 1,
                        counts[insert_idx] if insert_idx < len(counts) else 0,
                    ),
                )
        elif len(steps) == len(steps_objs) and total_steps > max(
            len(steps), len(steps_objs)
        ):
            print_saver.print_and_store(
                f"Steps length mismatch (equal in steps and steps_objs, but both less than total_steps). Adjusting. steps: {len(steps)} vs steps_objs: {len(steps_objs)} and total_steps: {total_steps}"
            )
            missing_objs = [obj for obj in steps_objs if obj.description not in steps]
            for obj in missing_objs:
                obj_index = steps_objs.index(obj)
                next_match = None
                for following in steps_objs[obj_index + 1 :]:
                    if following.description in steps:
                        next_match = following.description
                        break
                if next_match:
                    insert_idx = steps.index(next_match)
                else:
                    insert_idx = len(steps)
                steps_temp.insert(insert_idx, obj.description)

            missing_steps = [
                step_value
                for step_value in steps
                if step_value not in [o.description for o in steps_objs]
            ]
            for step_value in missing_steps:
                step_index = steps.index(step_value)
                next_match = None
                for following in steps[step_index + 1 :]:
                    if following in [o.description for o in steps_objs]:
                        next_match = following
                        break
                if next_match:
                    insert_idx = [o.description for o in steps_objs].index(next_match)
                else:
                    insert_idx = len(steps_objs)
                steps_objs_temp.insert(
                    insert_idx,
                    Step(
                        step_value,
                        insert_idx + 1,
                        counts[insert_idx] if insert_idx < len(counts) else 0,
                    ),
                )
        else:
            raise ValueError(
                f"Unhandled case. Please check the length of steps, steps_objs, and total_steps. steps: {len(steps)} vs steps_objs: {len(steps_objs)} and total_steps: {total_steps}"
            )
        steps = steps_temp
        steps_objs = steps_objs_temp

        description = steps[i] if i < len(steps) else f"Step {step_number}"
        counts = remove_nonnumeric_counts(counts, first_count)
        pre_count = (
            counts[i] if i < len(counts) else (int(counts[-1]) - 1 if counts else 0)
        )
        pre_count = int(pre_count)
        remaining_budget = pre_count if pre_count is not (None or 0) else 0

        if step_number in existing_map.keys():
            obj = existing_map[step_number]
            # Update description if necessary
            if description and obj.description != description:
                obj.description = description
            # Update remaining_budget if necessary
            if remaining_budget != obj.remaining_budget:
                if remaining_budget < obj.remaining_budget:
                    print_saver.print_and_store(
                        f"Remaining budget mismatch for step {step_number} (shorter than expected). Adjusting remaining budget. {remaining_budget} vs {obj.remaining_budget}"
                    )
                    remaining_budget = obj.remaining_budget
                elif remaining_budget > obj.remaining_budget:
                    print_saver.print_and_store(
                        f"Remaining budget mismatch for step {step_number} (longer than expected). Adjusting remaining budget. {remaining_budget} vs {obj.remaining_budget}"
                    )
                    obj.remaining_budget = remaining_budget
        else:
            # Create a new Step object
            new_obj = Step(description, step_number, remaining_budget, reflection)
            existing_map[step_number] = new_obj

    # Step 4: Sort the steps by step_number
    step_objs_final = [existing_map[sn] for sn in sorted(existing_map.keys())]
    return step_objs_final


# ===========================
# Define the process_steps Function
# ===========================


def process_steps(
    steps_objs: Optional[List[Step]],
    steps: List[str],
    counts: List[str | int],
    reflections: List[Optional[Reflection]],
    rewards: List[float],
    response: str,
    first_count: int,
    print_saver,  # Assuming print_saver is an object with print_and_store method
    interaction: Interaction,  # Assuming Interaction is a defined class
    task: str,
) -> Interaction:
    """
    Process and consolidate steps, reflections, and rewards into the interaction object.

    Args:
        steps_objs (Optional[List[Step]]): Existing list of Step objects.
        steps (List[str]): List of step descriptions.
        counts (List[str | int]): List of counts representing the budget before each step.
        reflections (List[Optional[Reflection | str]]): List of reflections.
        rewards (List[float]): List of rewards.
        response (str): Response string containing XML-like tags.
        first_count (int): The initial count value.
        print_saver: Object with a print_and_store method.
        interaction (Interaction): Object to store steps and reflections.
        task (str): The current task context.

    Returns:
        Interaction: The updated interaction object.
    """
    # Ensure steps_objs is a list
    steps_objs = steps_objs or []

    # ===========================
    # Step 1: Consolidate Steps
    # ===========================

    # Call the consolidate_steps function to align steps_objs with steps and counts
    steps_objs = consolidate_steps(steps_objs, steps, counts, reflections, first_count)

    # ===========================
    # Step 2: Validate and Adjust Steps
    # ===========================

    validate_steps(
        steps_objs,
        steps,
        counts,
        reflections,
        rewards,
        print_saver,
        interaction,
        task,
        first_count,
    )

    # ===========================
    # Step 3: Ensure Content Similarity
    # ===========================

    ensure_content_similarity(
        steps_objs, steps, first_count, print_saver, interaction, task
    )

    # ===========================
    # Step 4: Update Interaction with Consolidated Steps
    # ===========================

    for step_obj in steps_objs:
        if step_obj not in interaction.steps:
            interaction.steps.append(step_obj)

    # ===========================
    # Step 5: Process Reflections and Rewards
    # ===========================

    for i, step_obj in enumerate(steps_objs):
        # Handle reflections
        reflection = reflections[i] if i < len(reflections) else None
        if i == 0 and step_obj.step_number == 0:
            step_obj.step_number = 1
            # Now check steps after the first one to ensure they are sequential
            for j in range(1, len(steps_objs)):
                if steps_objs[j].step_number != step_obj.step_number + 1:
                    steps_objs[j].step_number = step_obj.step_number + 1

        if reflection is None:
            # If reflection is missing, generate it
            reflection = judge_step(step_obj, task)
            if reflection is not None:
                reflection.step_number = step_obj.step_number
                step_obj.reflection = reflection
                interaction.reflections.append(reflection)
        else:
            # Assign the reflection and reward to the step
            step_obj.reflection = (
                reflection
                if isinstance(reflection, Reflection)
                else Reflection(
                    content=str(reflection),
                    reward=0.0,
                    step_number=step_obj.step_number,
                )
            )
            step_obj.reflection.reward = float(rewards[i]) if i < len(rewards) else 0.0
            assert isinstance(
                step_obj.reflection, Reflection
            ), "Reflection object is not properly instantiated."
            assert hasattr(
                step_obj.reflection, "content"
            ), "Reflection object does not have a content attribute."
            print_saver.print_and_store(
                f"Reflection for step {step_obj.step_number}: {step_obj.reflection}"
            )
            print_saver.print_and_store(
                f"Type of reflection: {type(step_obj.reflection)}"
            )
            if (
                interaction.reflections
                and len(interaction.reflections) > 0
                and interaction.reflections != []
            ):
                print_saver.print_and_store(
                    f"Type of reflection in interaction: {type(interaction.reflections)} and type of first item: {type(interaction.reflections[0])}"
                )
            else:
                print_saver.print_and_store(
                    f"Type of reflection in interaction: {type(interaction.reflections)} and type of reflection in step: {type(step_obj.reflection)}"
                )
                assert isinstance(
                    step_obj.reflection, Reflection
                ), "Reflection object is not properly instantiated."

            if step_obj.reflection not in interaction.reflections:
                interaction.reflections.append(step_obj.reflection)

        # Handle rewards if not already set
        if (
            step_obj.reflection
            and step_obj.reflection.reward == 0.0
            and i < len(rewards)
        ):
            step_obj.reflection.reward = float(rewards[i])

    # ===========================
    # Step 6: Extract Answer from Response
    # ===========================

    answer_match = re.search(
        r"<answer>(.*?)(?:</answer>|<final_reward>)", response, re.DOTALL
    )
    if answer_match:
        interaction.answer = answer_match.group(1).strip()

    # ===========================
    # Step 7: Extract Final Reward from Response
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
    # Convert reflections to Reflection objects
    for i, reflection in enumerate(reflections):
        if reflections_objs is not None and i < len(reflections_objs):
            reflections[i] = reflections_objs[i]
        else:
            reflections[i] = Reflection(
                content=reflection,
                step_number=i + 1,
                reward=0.0,
            )

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
    steps_objs = [
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
        "10",  # Before Step 1
        "9",  # Before Step 2
        "8",  # Before Step 3
        "7",  # Before Step 4
        "6",  # Before Step 5
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
        steps_objs=steps_objs,
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
