#!/usr/bin/env python3
# examples/multi_agent_collaboration.py

"""
This script demonstrates tackling a more complex task using the AdvancedPromptEngineer.
While this example code itself doesn't guarantee multi-agent collaboration will be
triggered (as it depends on the backend logic and task complexity assessment),
it shows:
1. How to set up a configuration that might be suitable for complex tasks.
2. How to define a more involved task.
3. What parts of the result object could be inspected for insights into the
   reasoning process, including a placeholder for conversation history which
   would be populated if multi-agent collaboration occurred and was exposed.
"""

from advanced_prompting import AdvancedPromptEngineer, PromptEngineeringConfig
# Assuming OutputType might be relevant for complex tasks, though not strictly used here
# from advanced_prompting.config import OutputType

def run_multi_agent_example():
    """
    Runs an example that is structured for a potentially complex task.
    """
    print("Running multi_agent_collaboration.py example...\n")

    # Initialize configuration
    # For complex tasks, you might consider:
    # - A more capable model (e.g., "gpt-4o" if "gpt-4o-mini" is default)
    # - A higher initial_budget if the system uses it for planning
    # - Adjusting the number of 'agents' if that's a direct config.
    # The specific settings that trigger multi-agent behavior depend on the
    # AdvancedPromptEngineer's internal logic.
    config = PromptEngineeringConfig(
        model="gpt-4o-mini", # Or a more powerful model like "gpt-4" or "gpt-4o"
        agents=3,          # Explicitly setting, matches default but good for demo
        max_steps=15,      # Allowing more steps for a complex task
        initial_budget=15  # Slightly higher budget
    )
    print(f"Using configuration: model={config.model}, agents={config.agents}, max_steps={config.max_steps}, initial_budget={config.initial_budget}\n")

    # Create prompt engineer instance
    engineer = AdvancedPromptEngineer(config)

    # Define a more complex task
    task = """
    Develop a comprehensive three-point plan to significantly reduce plastic waste in urban environments.
    For each point, consider the following aspects:
    1.  Economic feasibility and potential funding sources.
    2.  Social impact and community engagement strategies.
    3.  Technological innovations or requirements.
    Present the plan in a clear, actionable format.
    """
    print(f"Complex Task: {task}\n")

    # Generate solution
    try:
        result = engineer.main(task)

        # Access and print results
        print("--- Result ---")
        if result:
            if hasattr(result, 'answer'):
                print("Answer:")
                print(result.answer)
            else:
                print("Result object does not have an 'answer' attribute.")

            if hasattr(result, 'final_reward'):
                print(f"\nFinal Confidence Score (Reward): {result.final_reward}")

            # 'steps' attribute might not exist, or history might be the better attribute
            # Based on README, result.history contains steps.
            if hasattr(result, 'history') and result.history is not None:
                print(f"Number of Steps Taken: {len(result.history)}")
            elif hasattr(result, 'steps') and result.steps is not None: # Fallback if history isn't there
                 print(f"Number of Steps Taken (from result.steps): {len(result.steps)}")
            else:
                print("Number of steps not available in result.history or result.steps.")

            # Placeholder for accessing conversation history if available
            # The actual attribute name for conversation history might vary (e.g., result.conversation_log, result.agent_interactions)
            # This depends on how AdvancedPromptEngineer exposes it.
            # The `history` attribute usually contains a list of actions and observations.
            if hasattr(result, 'history') and result.history:
                print("\n--- Reasoning Steps (History) ---")
                for i, step_info in enumerate(result.history):
                    action = getattr(step_info, 'action', 'N/A')
                    output = getattr(step_info, 'output', 'N/A')
                    # Some step_info objects might be simple strings or dicts depending on backend
                    if isinstance(action, dict) and 'tool_input' in action: # Example of more detailed action
                        action_detail = action['tool_input']
                        print(f"Step {i+1}: Action - {action_detail}, Output - {output[:100]}...") # Truncate long outputs
                    else:
                        print(f"Step {i+1}: Action - {str(action)[:100]}, Output - {str(output)[:100]}...")

            # Example of how one MIGHT check for a specific multi-agent conversation log if the system populated it.
            # This is speculative as the internal structure of 'result' isn't fully known from the prompt.
            if hasattr(result, 'conversation_history'): # Replace 'conversation_history' with the actual attribute name
                print("\n--- Multi-Agent Conversation History (if available) ---")
                if result.conversation_history:
                    for entry in result.conversation_history:
                        # Assuming 'entry' might be a dict with 'agent_id' and 'message'
                        agent_id = entry.get('agent_id', 'Unknown Agent')
                        message = entry.get('message', 'No message content')
                        print(f"Agent [{agent_id}]: {message}")
                else:
                    print("Conversation history attribute exists but is empty.")
            else:
                print("\n(Note: `result.conversation_history` attribute not found. Multi-agent specific logs depend on backend implementation.)")

        else:
            print("No result received.")

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure your OpenAI API key is set correctly and dependencies are installed.")
        print("Refer to the main README.md for setup instructions.")

if __name__ == "__main__":
    run_multi_agent_example()
