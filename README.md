# Advanced Prompting Engine

A sophisticated Python-based prompting engine that leverages multi-agent collaboration, complexity-based step planning, and adaptive reasoning to solve complex tasks using OpenAI's GPT models.

## Features

- **Multi-Agent Collaboration**: Implements a collaborative reasoning system with multiple AI agents working together to solve complex problems
- **Complexity-Based Planning**: Uses 

complexity_measures.py

 to analyze task complexity and generate optimal execution plans
- **Dynamic Step Budgeting**: Automatically adjusts processing steps based on task complexity
- **Self-Consistency Checks**: Validates solutions through multiple reasoning paths
- **Adaptive Prompt Engineering**: Refines prompts based on performance data and feedback
- **Structured Output Management**: Handles various output types via 

OutputType

 class

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/advanced-prompting-engine.git
cd advanced-prompting-engine
```

2. Install required dependencies:
```bash
pip install openai pydantic tqdm regex traitlets
```

3. Set up your OpenAI API key:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

## Quick Start Guide

This guide demonstrates a common use case: solving a complex problem that requires step-by-step reasoning. We'll use the `AdvancedPromptEngineer` to generate a plan for organizing a community event.

```python
from advanced_prompting import AdvancedPromptEngineer, PromptEngineeringConfig
from advanced_prompting.config import OutputType # Assuming OutputType is here

# 1. Initialize PromptEngineeringConfig
# These parameters control how the AdvancedPromptEngineer behaves.
# - model: Specifies the OpenAI model to use (e.g., "gpt-4o-mini", "gpt-4").
# - agents: Number of AI agents that will collaborate on the task. More agents can lead to more robust solutions but increase cost.
# - max_steps: The maximum number of reasoning steps the engineer can take. Prevents runaway processes.
# - initial_budget: Relates to the complexity allowed for the initial plan.
# - output_type: Defines the desired format of the output. Default is a string.
config = PromptEngineeringConfig(
    model="gpt-4o-mini",
    agents=3,
    max_steps=10,
    initial_budget=10, # Corresponds to a moderately complex task
    output_type=OutputType.STRING # Explicitly setting for clarity, though it's the default
)

# 2. Create an AdvancedPromptEngineer instance
# This object orchestrates the problem-solving process.
engineer = AdvancedPromptEngineer(config)

# 3. Define a slightly more complex task
# Let's ask the engineer to outline a plan for a community hackathon.
task = """
Create a comprehensive plan to organize a 2-day community hackathon focused on AI for social good.
The plan should include:
- Pre-event: Theme ideation, speaker outreach, participant registration, sponsorship.
- During event: Schedule, judging criteria, mentor roles, technical support.
- Post-event: Prize distribution, feedback collection, community building.
Provide a timeline for each phase.
"""

# 4. Run the engineer with the task
# The main method executes the multi-agent collaboration and reasoning process.
result = engineer.main(task)

# 5. Print various parts of the result
# The result object contains detailed information about the solution.
print("--- Generated Plan ---")
print(result.answer)
print("\\n--- Confidence and Reasoning ---")
print(f"Final Confidence Score: {result.final_reward}")
print(f"Number of Steps Taken: {len(result.history) if result.history else 'N/A'}") # history might be None or empty
# You can also inspect result.history for detailed step-by-step reasoning if needed.
# For example:
# if result.history:
# for i, step_info in enumerate(result.history):
# print(f"Step {i+1}: Action - {step_info.action}, Output - {step_info.output}")
```

## Usage

```python
from advanced_prompting import AdvancedPromptEngineer, PromptEngineeringConfig

# Initialize configuration
config = PromptEngineeringConfig()

# Create prompt engineer instance
engineer = AdvancedPromptEngineer(config)

# Define your task
task = """
Design a system that...
"""

# Generate solution
result = engineer.main(task)

# Access results
print(result.answer)
print(f"Confidence Score: {result.final_reward}")
```

## Core Concepts

The Advanced Prompting Engine is built around a modular architecture designed to tackle complex tasks by breaking them down into manageable steps and leveraging multiple AI perspectives.

-   **`advanced_prompting.py` (Main Engine)**: This is the heart of the system. It orchestrates the entire process, from receiving the initial task to delivering the final solution. It manages the multi-agent collaboration, directs the iterative reasoning process, and integrates feedback for adaptive learning. The `AdvancedPromptEngineer` class within this module is the primary interface for users. It takes a task and a `PromptEngineeringConfig` object, and then guides the task through stages of planning, execution, and self-correction.

-   **`complexity_measures.py` (Complexity Analysis & Planning)**: Before tackling a task, the engine first assesses its complexity. This module provides tools to analyze the input task and estimate the resources (e.g., number of reasoning steps, model capabilities) required. Based on this analysis, it helps in generating an initial execution plan or "budget." For instance, a highly complex task might be allocated more steps or involve more specialized agents. This proactive planning helps in optimizing resource usage and improving the quality of the output.

-   **`conversation_manager.py` (Output & Interaction Handling)**: This module is responsible for managing the inputs and outputs of the AI agents and the overall system. It defines how results are structured (e.g., using the `OutputType` enum for formats like strings, JSON, lists) and ensures that the conversation flow between agents, and between the system and the user, is coherent. It plays a crucial role in formatting the final answer and presenting any intermediate steps or reasoning paths if requested. For example, it ensures that if a JSON output is specified, the final answer conforms to valid JSON syntax.

**Interaction Flow:**

1.  A user submits a task to the `AdvancedPromptEngineer` (`advanced_prompting.py`).
2.  The engine may use `complexity_measures.py` to analyze the task's complexity and determine an initial plan and resource allocation (e.g. step budget).
3.  The `AdvancedPromptEngineer` then initiates a multi-agent collaborative process. It breaks down the problem and assigns sub-tasks to different AI agents (conceptual agents, actual model calls are managed by the engineer).
4.  Agents (or the engineer itself through structured prompts) perform reasoning steps. `conversation_manager.py` helps in formatting the prompts sent to the LLM and parsing its responses.
5.  The process may involve iterative refinement, self-correction, and dynamic adjustments to the plan based on intermediate results.
6.  Finally, `conversation_manager.py` ensures the final output is presented in the desired format, along with any supporting information like confidence scores or steps taken.

This architecture allows the system to adapt its approach based on the task's nature and complexity, leading to more robust and reliable solutions.

## Configuration

The 

PromptEngineeringConfig

 class supports various parameters:

- `max_steps`: Maximum number of reasoning steps (default: 20)
- `initial_budget`: Initial token/computation budget (default: 20)
- `agents`: Number of collaborative agents (default: 3)
- `temperature`: Sampling temperature (default: 0.7)
- `model`: GPT model to use (default: "gpt-4o-mini")

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
