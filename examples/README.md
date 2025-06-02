# Usage Examples

This directory contains Python scripts that demonstrate how to use the Advanced Prompting Engine for various tasks. These examples are intended to help you quickly understand the core functionalities and get started with your own projects.

## Available Examples

-   **`basic_usage.py`**:
    *   Demonstrates the fundamental workflow of the `AdvancedPromptEngineer`.
    *   Covers initializing the configuration, creating an engineer instance, defining a simple task, and retrieving the answer.
    *   Useful for new users to verify their setup and see a straightforward execution.

-   **`multi_agent_collaboration.py`**:
    *   Shows an example of tackling a more complex task that might benefit from the engine's advanced features, such as multi-agent collaboration (though activation depends on backend logic).
    *   Illustrates how to configure the engineer for more demanding scenarios.
    *   Provides guidance on how to inspect various parts of the result object, including the answer, confidence score, and number of steps taken.
    *   Includes a commented-out section showing how one might access detailed conversation or step history if the `AdvancedPromptEngineer` populates these fields.

## Running the Examples

To run these examples:

1.  Ensure you have followed the installation instructions in the main `README.md` of this repository.
2.  Make sure your `OPENAI_API_KEY` environment variable is set.
3.  Navigate to the `examples` directory in your terminal:
    ```bash
    cd path/to/your/clone/advanced-prompting-engine/examples
    ```
4.  Execute the desired script using Python:
    ```bash
    python3 basic_usage.py
    ```
    or
    ```bash
    python3 multi_agent_collaboration.py
    ```

Refer to the comments within each script for more detailed explanations of the code.
