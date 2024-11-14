from calendar import c
import json
from turtle import st
import openai
import random
import os
import re
import logging
import sys
import time
from typing import Any, Dict, List, Tuple
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Change to DEBUG to see all messages
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)


# Retrieve OpenAI API key from environment variables for security
openai.api_key = os.getenv("OPENAI_API_KEY")

if not openai.api_key:
    logging.error(
        "OpenAI API key not found. Please set the 'OPENAI_API_KEY' environment variable."
    )
    sys.exit(1)


def query_llm(
    prompt: str, max_completion_tokens: int = 1500, retry_count: int = 3
) -> str:
    """
    Sends a prompt to the OpenAI GPT-4 model and retrieves the response with retry logic.

    Args:
        prompt (str): The prompt to send to the model.
        max_completion_tokens (int): Maximum number of tokens in the response.
        retry_count (int): Number of retries for transient errors.

    Returns:
        str: The model's response.
    """
    for attempt in range(retry_count):
        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini",  # Specify the model
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=max_completion_tokens,
                temperature=0.7,
                stop=["<end>", "<|endoftext|>"],
            )
            content = response.choices[0].message.content.strip()
            logging.debug(f"Model Response: {content}")
            return content

        except Exception as e:
            logging.error(f"Unexpected error during API call: {e}")
            return ""
    logging.error("Exceeded maximum retry attempts.")
    return ""


class BaseExtractFormat(BaseModel):
    extracted: str = Field(..., description="The extracted structured information.")


class JsonListFormat(BaseModel):
    extracted: List[str] = Field(
        ..., description="The extracted structured information as a list."
    )


class JsonDictFormat(BaseModel):
    extracted: Dict[str, str] = Field(
        ..., description="The extracted structured information as a dictionary."
    )


class JsonNestedFormat(BaseModel):
    extracted: Dict[str, Any] = Field(
        ..., description="The extracted structured information as a nested dictionary."
    )


class JsonTableFormat(BaseModel):
    extracted: List[Dict[str, str]] = Field(
        ..., description="The extracted structured information as a table."
    )


class JsonGraphFormat(BaseModel):
    extracted: Dict[str, List[str]] = Field(
        ..., description="The extracted structured information as a graph."
    )


class JsonTreeFormat(BaseModel):
    extracted: Dict[str, Any] = Field(
        ..., description="The extracted structured information as a tree."
    )


class JsonCodeFormat(BaseModel):
    extracted: str = Field(
        ..., description="The extracted structured information as code snippets."
    )


class JsonMathFormat(BaseModel):
    extracted: str = Field(
        ...,
        description="The extracted structured information as mathematical equations.",
    )


class JsonSubtaskFormat(BaseModel):
    subtasks: List[str] = Field(
        ..., description="The subtasks extracted from the main task."
    )


def query_llm_with_structured_output(
    prompt: str, output_format: str, retry_count: int = 3, **kwargs
) -> Any:
    """
    Sends a prompt to the OpenAI GPT-4 model and retrieves the response with structured output.

    Args:
        prompt (str): The prompt to send to the model.
        output_format (BaseModel): The Pydantic model to parse the structured output.

    Returns:
        Any: The parsed structured output.
    """
    map_format = {
        "json": BaseExtractFormat,
        "json_list": JsonListFormat,
        "json_dict": JsonDictFormat,
        "json_nested": JsonNestedFormat,
        "json_table": JsonTableFormat,
        "json_graph": JsonGraphFormat,
        "json_tree": JsonTreeFormat,
        "json_code": JsonCodeFormat,
        "json_math": JsonMathFormat,
        "json_custom": BaseExtractFormat,
        "": BaseExtractFormat,
        "default": BaseExtractFormat,
        "list": JsonListFormat,
        "dict": JsonDictFormat,
        "nested": JsonNestedFormat,
        "table": JsonTableFormat,
        "graph": JsonGraphFormat,
        "tree": JsonTreeFormat,
        "code": JsonCodeFormat,
        "math": JsonMathFormat,
        "custom": BaseExtractFormat,
        "subtask": JsonSubtaskFormat,
        "subtasks": JsonSubtaskFormat,
        "tasks": JsonSubtaskFormat,
        "subtask_list": JsonSubtaskFormat,
    }
    # Ask the assistant to cast a confidence vote based on the messages
    messages = [{"role": "system", "content": prompt}]
    for attempt in range(retry_count):
        try:
            response = openai.beta.chat.completions.parse(
                model="gpt-4o-mini",  # Specify the model
                messages=messages,  # Pass the messages directly as a list of dicts
                response_format=map_format[output_format],  # Specify the output format
            )
            # Extract the vote from the response
            output = response.choices[0].message.parsed
            return output

        except Exception as e:
            logging.error(f"Failed to parse structured output: {e}")
            return None
    logging.error("Exceeded maximum retry attempts.")


def self_consistency_evaluation(response: str) -> Tuple[str, float]:
    """
    Evaluates the reasoning process and assigns a confidence score.

    Args:
        response (str): The reasoning or solution to evaluate.

    Returns:
        Tuple[str, float]: The evaluation text and the confidence score.
    """
    logging.info("Performing Self-Consistency Evaluation...")
    reflection_prompt = f"""
Based on the solution provided: {response}
<reflection> Critically evaluate the reasoning process so far. How confident are you in the accuracy? Assign a reward score from 0.0 to 1.0, reflecting confidence in the approach: </reflection>
<reward> Please assign a score now. </reward>
"""
    evaluation = query_llm(reflection_prompt)
    if not evaluation:
        logging.warning("No response received from Self-Consistency Evaluation.")
        return "", 0.0

    # Use regex to extract the score within <reward> tags
    score_match = re.search(
        r"<reward>\s*([\d\.]+)\s*</reward>", evaluation, re.IGNORECASE
    )
    if score_match:
        try:
            score = float(score_match.group(1))
            logging.debug(f"Extracted Score: {score}")
            return evaluation, score
        except ValueError:
            logging.error("Failed to convert extracted score to float.")
            return evaluation, 0.0
    else:
        logging.warning("No score found in Self-Consistency Evaluation response.")
        return evaluation, 0.0


# 1. Automatic Prompt Engineering (APE)
def generate_prompt_variants(
    base_prompt: str, score_threshold: float = 0.8, max_variants: int = 10
) -> str:
    """
    Generates multiple prompt variants by experimenting with tone, clarity, and conciseness.
    Returns the best-performing variant based on the confidence score.

    Args:
        base_prompt (str): The original prompt.
        score_threshold (float): The minimum confidence score required to accept a variant.
        max_variants (int): The maximum number of variants to generate.

    Returns:
        str: The optimized prompt variant.
    """
    logging.info("Starting Automatic Prompt Engineering (APE)...")
    for i in range(max_variants):
        modification = random.choice(["tone", "clarity", "conciseness"])
        variant = f"{base_prompt} [Experiment with: {modification}]"
        logging.debug(f"Generated Variant {i+1}: {variant}")
        result = query_llm(variant)
        if not result:
            continue  # Skip if no response was received
        reflection, score = self_consistency_evaluation(result) if result else ("", 0.0)
        if score > score_threshold and result not in [base_prompt, variant, ""]:
            logging.info(f"Selected Prompt Variant: {variant} (Score: {score})")
            return variant
    logging.warning("No prompt variant met the score threshold. Using base prompt.")
    return base_prompt


def is_complex(input_query: str) -> bool:
    """
    Determines if the problem represented by the input query is complex.

    Args:
        input_query (str): The problem to solve.

    Returns:
        bool: True if the query is complex, False otherwise.
    """
    keywords = ["complex", "complicated", "difficult", "challenging"]
    word_count = len(input_query.split())
    if any(keyword in input_query.lower() for keyword in keywords) or word_count > 50:
        return True
    return False


# 2. Adaptive Complexity Handling
def adjust_complexity(input_query: str) -> str:
    """
    Adjusts the number of steps for problem-solving based on the complexity of the input query.

    Args:
        input_query (str): The problem to solve.

    Returns:
        str: The solution after adjusting complexity.
    """
    logging.info("Adjusting complexity based on input query...")
    word_count = len(input_query.split())
    if word_count > 50 or "complex" in input_query.lower():
        num_steps = 40
        logging.debug(
            f"Input is complex (word count: {word_count}). Setting num_steps to {num_steps}."
        )
    else:
        num_steps = 20
        logging.debug(
            f"Input is simple/moderate (word count: {word_count}). Setting num_steps to {num_steps}."
        )
    adjusted_steps = least_to_most_prompt(
        input_query, num_steps=num_steps
    )  # Use L2M for breakdown
    if not adjusted_steps or adjusted_steps in [input_query, ""]:
        logging.error("Failed to adjust complexity.")
        return ""
    return adjusted_steps


MAX_DEPTH = 2


def process_subtask(subtask, depth=0):
    if depth >= MAX_DEPTH:
        return least_to_most_prompt(subtask)
    if is_complex(subtask):
        # Decompose further
        sub_subtasks = decompose_subtask(subtask)
        results = [process_subtask(sst, depth + 1) for sst in sub_subtasks]
        return combine_results(results)
    else:
        return least_to_most_prompt(subtask)


# 3. Hierarchical Reasoning
def hierarchical_reasoning(input_query: str) -> str:
    """
    Decomposes the main problem into high-level subtasks and solves each iteratively.

    Args:
        input_query (str): The main problem to solve.

    Returns:
        str: The combined solutions to all subtasks.
    """
    # Define the function schema
    # function_schema = {
    #     "name": "decompose_task",
    #     "description": "Decomposes a task into subtasks.",
    #     "parameters": {
    #         "type": "object",
    #         "properties": {
    #             "subtasks": {
    #                 "type": "array",
    #                 "items": {"type": "string"},
    #                 "description": "List of subtasks."
    #             }
    #         },
    #         "required": ["subtasks"]
    #     }
    # }
    subproblems = None
    logging.info("Starting Hierarchical Reasoning...")
    main_task_prompt = (
        f"Break down the main task: '{input_query}' into its high-level subtasks. "
    )
    try:

        # Call the LLM with function calling
        subproblems_response = query_llm_with_structured_output(
            main_task_prompt,
            "subtask_list",
        )

        subproblems = subproblems_response.get("subtasks", [])
        responses = []
        if isinstance(subproblems, JsonListFormat) or isinstance(
            subproblems, JsonSubtaskFormat
        ):
            subproblems = subproblems.extracted
        else:
            logging.error("Invalid subtasks format in Hierarchical Reasoning.")
            raise ValueError("Invalid subtasks format in Hierarchical Reasoning.")
        if isinstance(subproblems, str):
            subproblems = [subproblems.split(",")]
        if isinstance(subproblems, list) and len(subproblems) > 0:
            for idx, subproblem in enumerate(subproblems, 1):
                logging.info(f"Solving Subtask {idx}: {subproblem}")
                response = least_to_most_prompt(subproblem)
                if response:
                    responses.append(f"Subtask {idx} Solution:\n{response}")
                else:
                    logging.warning(f"No response received for Subtask {idx}.")
        elif isinstance(subproblems, dict) and len(subproblems) > 0:
            for idx, (key, value) in enumerate(subproblems.items(), 1):
                logging.info(f"Solving Subtask {idx}: {key}")
                response = least_to_most_prompt(value)
                if response:
                    responses.append(f"Subtask {idx} Solution:\n{response}")
                else:
                    logging.warning(f"No response received for Subtask {idx}.")
        else:
            logging.error(
                "Invalid subtasks format in Hierarchical Reasoning. Trying alternative methods."
            )
            raise ValueError("Invalid subtasks format in Hierarchical Reasoning.")
        if responses and len(responses) > 0:
            hierarchical_solution = "\n\n".join(responses)
            return hierarchical_solution
        else:
            logging.error(
                "No subtasks found in Hierarchical Reasoning. Trying alternative methods."
            )
            raise ValueError(
                "No subtasks found in Hierarchical Reasoning. Trying alternative methods."
            )

    except Exception as e:
        logging.error(f"Failed to retrieve subtasks from Hierarchical Reasoning: {e}")
    main_task_prompt = (
        f"Please break down the following task into clear, high-level subtasks. "
        f"Task: '{input_query}'.\n"
        'Respond with a JSON array of subtasks, for example: ["Subtask 1", "Subtask 2"]'
    )

    subproblems_response = query_llm(main_task_prompt)
    try:
        subproblems = json.loads(subproblems_response)
    except json.JSONDecodeError:
        logging.error("Failed to parse subtasks as JSON.")
    if not subproblems_response or subproblems_response == "" or not subproblems:
        logging.error("Failed to retrieve subtasks from Hierarchical Reasoning.")
        try:
            main_task_prompt = (
                f"Break down the main task: '{input_query}' into its high-level subtasks. "
                "Provide the subtasks as a bulleted list."
            )
            subproblems_response = query_llm(main_task_prompt)
            if not subproblems_response or subproblems_response == "":
                logging.error(
                    "Failed to retrieve subtasks from Hierarchical Reasoning."
                )
                raise ValueError(
                    "Failed to retrieve subtasks from Hierarchical Reasoning. The response was empty."
                )
        except Exception as e:
            logging.error(
                f"Failed to retrieve subtasks from Hierarchical Reasoning: {e}"
            )
            return f"Failed to retrieve subtasks from Hierarchical Reasoning: {e}"

    if not subproblems and subproblems_response:
        # Use regex to extract numbered or bulleted lists
        subproblems = re.findall(
            r"^\s*(?:\d+\.|\-|\*|\•)\s*(.*)", subproblems_response, re.MULTILINE
        )

    if not subproblems:
        # Fallback to splitting by numbered lists if bulleted lists fail
        subproblems = re.findall(
            r"^\s*\d+\.\s*(.*)", subproblems_response, re.MULTILINE
        )

    if not subproblems:
        # The backup fallback is to split by semicolons
        subproblems = [
            part.strip() for part in subproblems_response.split(";") if part.strip()
        ]

    if not subproblems:
        # Fallback to splitting by commas if semicolons fail
        subproblems = [
            part.strip() for part in subproblems_response.split(",") if part.strip()
        ]

    if not subproblems:
        # Fallback to splitting by lines if regex fails
        subproblems = [
            line.strip() for line in subproblems_response.split("\n") if line.strip()
        ]

    if not subproblems:
        # Fallback to splitting by sentences if no lines are found
        subproblems = [
            sentence.strip()
            for sentence in re.split(r"[.!?]", subproblems_response)
            if sentence.strip()
        ]

    if not subproblems:
        logging.error("No subtasks found in Hierarchical Reasoning.")
        return ""

    responses = []
    if isinstance(subproblems, str):
        subproblems = [subproblems]
    if isinstance(subproblems, list) and len(subproblems) > 0:
        for idx, subproblem in enumerate(subproblems, 1):
            logging.info(f"Solving Subtask {idx}: {subproblem}")
            response = least_to_most_prompt(subproblem)
            if response:
                responses.append(f"Subtask {idx} Solution:\n{response}")
            else:
                logging.warning(f"No response received for Subtask {idx}.")
    elif isinstance(subproblems, dict) and len(subproblems) > 0:
        for idx, (key, value) in enumerate(subproblems.items(), 1):
            logging.info(f"Solving Subtask {idx}: {key}")
            response = least_to_most_prompt(value)
            if response:
                responses.append(f"Subtask {idx} Solution:\n{response}")
            else:
                logging.warning(f"No response received for Subtask {idx}.")
    else:
        logging.error("Invalid subtasks format in Hierarchical Reasoning.")
        return ""

    hierarchical_solution = "\n\n".join(responses)
    return hierarchical_solution


# 4. Retrieval-Augmented Generation (RAG)
def fetch_external_knowledge(input_query: str) -> str:
    """
    Simulates external knowledge retrieval. In practice, integrate with real APIs or databases.

    Args:
        input_query (str): The problem to solve.

    Returns:
        str: Retrieved external knowledge.
    """
    logging.info("Fetching external knowledge (simulated)...")
    # Placeholder for external knowledge retrieval
    # Replace this with actual API calls or database queries as needed
    external_knowledge = "Relevant data fetched from external databases."
    return external_knowledge


def retrieval_augmented_query(input_query: str) -> str:
    """
    Uses external knowledge to augment the prompt for solving the problem.

    Args:
        input_query (str): The problem to solve.

    Returns:
        str: The solution incorporating external knowledge.
    """
    logging.info("Starting Retrieval-Augmented Generation (RAG)...")
    external_knowledge = fetch_external_knowledge(input_query)
    prompt = f"Using the following external information: {external_knowledge}, solve the problem: {input_query}"
    rag_response = query_llm(prompt)
    if rag_response:
        logging.info("RAG solution obtained.")
    else:
        logging.warning("No response received from RAG.")
    return rag_response


# 5. Collaborative Multi-Agent Reasoning
def collaborative_reasoning(input_query: str) -> str:
    """
    Utilizes two agents: one to generate a solution and another to critique and refine it.

    Args:
        input_query (str): The problem to solve.

    Returns:
        str: The refined solution after collaborative reasoning.
    """
    logging.info("Starting Collaborative Multi-Agent Reasoning...")

    # Agent A generates a solution
    agent_a_prompt = f"Agent A: Solve the problem '{input_query}'"
    agent_a_response = query_llm(agent_a_prompt)
    if not agent_a_response:
        logging.error("Agent A failed to provide a solution.")
        return ""
    logging.debug(f"Agent A's Response:\n{agent_a_response}\n")

    # Agent B critiques and refines the solution
    agent_b_prompt = f"Agent B: Critique the following solution: {agent_a_response}"
    agent_b_critique = query_llm(agent_b_prompt)
    if not agent_b_critique:
        logging.error("Agent B failed to provide a critique.")
        return agent_a_response  # Return original solution if critique fails
    logging.debug(f"Agent B's Critique:\n{agent_b_critique}\n")

    # Agent A refines based on Agent B's critique
    agent_a_refined_prompt = (
        f"Agent A: Refine the solution based on this critique: {agent_b_critique}"
    )
    refined_response = query_llm(agent_a_refined_prompt)
    if not refined_response:
        logging.error("Agent A failed to refine the solution.")
        return agent_a_response  # Return original solution if refinement fails
    logging.debug(f"Agent A's Refined Response:\n{refined_response}\n")

    return refined_response


# 6. Dynamic Confidence Exploration
def dynamic_exploration(input_query: str) -> str:
    """
    Iteratively explores different reasoning paths to achieve sufficient confidence.

    Args:
        input_query (str): The problem to solve.

    Returns:
        str: The solution with sufficient confidence.
    """
    logging.info("Starting Dynamic Confidence Exploration...")
    response = ""
    for iteration in range(5):
        logging.info(f"Dynamic Exploration Iteration {iteration + 1}...")
        response = auto_cot(input_query)
        if not response:
            logging.warning("No response received from auto_cot.")
            continue
        reflection, score = self_consistency_evaluation(response)
        logging.info(f"Reflection Score: {score}")
        if score > 0.8:
            logging.info("Sufficient confidence achieved.")
            return response
        elif iteration == 4:
            # Last resort: use a different approach
            alternative_prompt = (
                f"Explore a very different strategy for '{input_query}'"
            )
            alternative_response = least_to_most_prompt(alternative_prompt)
            if alternative_response:
                logging.info("Dynamic Exploration Final Attempt Successful.")
                return alternative_response
            else:
                logging.warning("Dynamic Exploration Final Attempt Failed.")
                return response  # Return the last response even if it's low confidence
    logging.warning(
        "Dynamic Confidence Exploration did not achieve sufficient confidence."
    )
    return response  # Return the last response


# Supporting Functions


def auto_cot(input_query: str) -> str:
    """
    Performs initial zero-shot reasoning using Chain-of-Thought (CoT).

    Args:
        input_query (str): The problem to solve.

    Returns:
        str: The reasoning process.
    """
    logging.info("Performing Auto-CoT (Zero-Shot Reasoning)...")
    cot_prompt = f"""
Let's think step by step through the problem: "{input_query}"
<thinking> We will break down the problem logically and explore multiple approaches. </thinking>
"""
    cot_response = query_llm(cot_prompt)
    if cot_response:
        logging.debug(f"Auto-CoT Response:\n{cot_response}\n")
    else:
        logging.warning("No response received from Auto-CoT.")
    return cot_response


def least_to_most_prompt(input_query: str, num_steps: int = 20) -> str:
    """
    Progressively breaks down complex tasks into subproblems using Least-to-Most Prompting.

    Args:
        input_query (str): The problem to solve.
        num_steps (int): The number of steps to initiate the breakdown.

    Returns:
        str: The step-by-step breakdown of the solution.
    """
    logging.info(f"Applying Least-to-Most Prompting with {num_steps} steps...")
    prompt = f"""
Begin by breaking down the solution into steps. Each step should be enclosed in <step> tags. 
Start with {num_steps} steps, and ask for more if needed:

Input problem: {input_query}
<thinking> Break it down step-by-step </thinking>
"""
    steps_response = query_llm(prompt)
    if (
        steps_response
        and steps_response != prompt
        and steps_response != input_query
        and steps_response != ""
        and "<step>" in steps_response
    ):
        logging.debug(f"Least-to-Most Prompting Response:\n{steps_response}\n")
    elif "<step>" not in steps_response:
        logging.warning("No step tags found in Least-to-Most Prompting response.")
    else:
        logging.warning("No response received from Least-to-Most Prompting.")
        return ""
    return steps_response


def handle_low_confidence(response: str, score: float) -> str:
    """
    Handles scenarios where the confidence score is below the threshold by backtracking and exploring alternative approaches.

    Args:
        response (str): The initial solution.
        score (float): The confidence score associated with the solution.

    Returns:
        str: The refined solution after handling low confidence.
    """
    logging.info("Checking if low confidence handling is required...")
    if score < 0.5:
        logging.warning(
            f"Low confidence detected (Score: {score}). Initiating backtracking..."
        )
        backtrack_prompt = f"""
The reward score was low ({score}). Consider backtracking and trying a different approach.
<thinking> Explore alternative approaches and adjust the reasoning strategy. </thinking>
"""
        refined_response = query_llm(backtrack_prompt)
        if refined_response:
            logging.debug(f"Refined Solution after Backtracking:\n{refined_response}\n")
            return refined_response
        else:
            logging.warning("No response received during Low Confidence Handling.")
            return response
    logging.info("Confidence score is adequate. No backtracking needed.")
    return response


def final_synthesis(refined_solution: str, other_solutions: list) -> str:
    """
    Synthesizes the final answer by considering all refined solutions.

    Args:
        refined_solution (str): The primary refined solution.
        other_solutions (list): Other solutions to consider.

    Returns:
        str: The final synthesized answer and reflection.
    """
    logging.info("Starting Final Synthesis and Reflection...")
    combined_solutions = "\n".join(other_solutions + [refined_solution])
    final_prompt = f"""
Based on all the solutions provided:

{combined_solutions}

Synthesize the final answer within <answer> tags, providing a clear, concise summary of the refined solution. 
Conclude with a final reflection on the overall solution, discussing its effectiveness, challenges, and the quality of the approach.
<reflection> Final reflection: </reflection>
<answer> Please provide the final solution now. </answer>
"""
    final_answer = query_llm(final_prompt)
    if final_answer:
        logging.debug(f"Final Answer and Reflection:\n{final_answer}\n")
    else:
        logging.warning("No response received during Final Synthesis.")
    return final_answer


# Master Function: Solve Complex Problem with Comprehensive Prompt Engineering
def solve_complex_problem(input_query: str):
    """
    Orchestrates the entire problem-solving process by integrating all advanced prompt engineering techniques.

    Args:
        input_query (str): The problem to solve.
    """
    logging.info(f"Solving Problem: {input_query}\n{'='*60}")

    # Step 1: Automatic Prompt Engineering (APE)
    optimized_prompt = generate_prompt_variants(input_query)

    # Step 2: Adaptive Complexity Handling
    complex_solution = adjust_complexity(optimized_prompt)
    logging.info(f"Complexity Adjusted Solution:\n{complex_solution}\n{'-'*60}")
    if not complex_solution or complex_solution in [input_query, ""]:
        complex_solution = (
            f"Failed to adjust complexity for the optimized prompt: {optimized_prompt}"
        )

    # Step 3: Hierarchical Reasoning
    hierarchical_solution = hierarchical_reasoning(input_query)
    logging.info(
        f"Hierarchical Reasoning Solutions:\n{hierarchical_solution}\n{'-'*60}"
    )
    if not hierarchical_solution or hierarchical_solution in [input_query, ""]:
        hierarchical_solution = f"Failed to break down the problem into subtasks for hierarchical reasoning for optimized prompt: {optimized_prompt}"

    # Step 4: Retrieval-Augmented Generation (RAG)
    rag_solution = retrieval_augmented_query(input_query)
    logging.info(f"Retrieval-Augmented Generation Solution:\n{rag_solution}\n{'-'*60}")
    if not rag_solution or rag_solution in [input_query, ""]:
        rag_solution = f"Failed to retrieve external knowledge for optimized prompt: {optimized_prompt}"

    # Step 5: Collaborative Multi-Agent Reasoning
    collaborative_solution = collaborative_reasoning(input_query)
    logging.info(
        f"Collaborative Multi-Agent Reasoning Solution:\n{collaborative_solution}\n{'-'*60}"
    )
    if not collaborative_solution or collaborative_solution in [input_query, ""]:
        collaborative_solution = f"Failed to collaborate with multiple agents for optimized prompt: {optimized_prompt}"

    # Step 6: Dynamic Confidence Exploration
    dynamic_solution = dynamic_exploration(input_query)
    logging.info(
        f"Dynamic Confidence Exploration Solution:\n{dynamic_solution}\n{'-'*60}"
    )
    if not dynamic_solution or dynamic_solution in [input_query, ""]:
        dynamic_solution = f"Failed to achieve sufficient confidence for optimized prompt: {optimized_prompt}"

    # Additional Steps from Script A

    # Auto-CoT for initial zero-shot reasoning
    initial_response = auto_cot(input_query)
    logging.info(f"Auto-CoT Initial Response:\n{initial_response}\n{'-'*60}")
    if not initial_response or initial_response in [input_query, ""]:
        initial_response = f"Failed to perform initial zero-shot reasoning for the input query: {input_query}"

    # Least-to-Most breakdown of steps
    step_by_step_solution = least_to_most_prompt(input_query)
    logging.info(f"Step-by-Step Solution:\n{step_by_step_solution}\n{'-'*60}")
    if not step_by_step_solution or step_by_step_solution in [input_query, ""]:
        step_by_step_solution = f"Failed to break down the problem into steps for the input query: {input_query}"
        logging.warning("Failed to break down the problem into steps.")
    else:
        # Self-consistency evaluation of the solution
        reflection, score = self_consistency_evaluation(step_by_step_solution)
        logging.info(f"Self-Consistency Evaluation Reflection:\n{reflection}\n{'-'*60}")
        logging.info(f"Self-Consistency Evaluation Score: {score}\n{'-'*60}")

        # Backtrack and refine approach if needed
        refined_solution = handle_low_confidence(step_by_step_solution, score)
        logging.info(
            f"Refined Solution after Low Confidence Handling:\n{refined_solution}\n{'-'*60}"
        )
        if not refined_solution or refined_solution in [input_query, ""]:
            refined_solution = (
                f"Failed to refine the solution for the input query: {input_query}"
            )

    # Collect all solutions for final synthesis
    all_solutions = [
        complex_solution,
        hierarchical_solution,
        rag_solution,
        collaborative_solution,
        dynamic_solution,
        initial_response,
        step_by_step_solution,
        refined_solution,
    ]

    # Final synthesis and reflection
    final_answer = final_synthesis(refined_solution, all_solutions)
    logging.info(
        f"Final Answer and Reflection:\n{final_answer}\n{'='*60}\nProblem Solving Completed."
    )


# Example Input
if __name__ == "__main__":
    problem = "Natalie has 3 apples and she wants to share them equally with 2 friends. How many apples will each person get?"
    solve_complex_problem(problem)
