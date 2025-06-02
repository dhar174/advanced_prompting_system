#!/usr/bin/env python3
# examples/basic_usage.py

"""
This script demonstrates the basic usage of the AdvancedPromptEngineer.
It shows how to:
1. Import necessary classes.
2. Initialize a PromptEngineeringConfig (using defaults).
3. Create an AdvancedPromptEngineer instance.
4. Define a simple task.
5. Run the engineer to get a solution.
6. Print the answer.
"""

from advanced_prompting import AdvancedPromptEngineer, PromptEngineeringConfig

def run_basic_example():
    """
    Runs a basic example of the AdvancedPromptEngineer.
    """
    print("Running basic_usage.py example...\n")

    # Initialize configuration (using default settings)
    # You can customize parameters like model, temperature, max_steps, etc.
    # For example: config = PromptEngineeringConfig(model="gpt-4", temperature=0.5)
    config = PromptEngineeringConfig()
    print(f"Using configuration: model={config.model}, max_steps={config.max_steps}\n")

    # Create prompt engineer instance
    engineer = AdvancedPromptEngineer(config)

    # Define a straightforward task
    task = "Explain the theory of relativity in simple terms for a high school student."
    print(f"Task: {task}\n")

    # Generate solution
    # The main method processes the task and returns a result object.
    try:
        result = engineer.main(task)

        # Access and print the primary answer
        print("--- Result ---")
        if result and hasattr(result, 'answer'):
            print("Answer:")
            print(result.answer)
        else:
            print("No answer received or result format is unexpected.")
            if result:
                print(f"Full result object: {result}")

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure your OpenAI API key is set correctly and dependencies are installed.")
        print("Refer to the main README.md for setup instructions.")

if __name__ == "__main__":
    run_basic_example()
