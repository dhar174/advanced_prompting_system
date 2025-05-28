```mermaid
erDiagram
    Task ||--o{ Interaction : "has many"
    Task ||--|| Plan : "has one"
    Task {
        string task_id
        string user_input
        string refined_input
        string current_module
    }

    Plan ||--o{ PlanStep : "has many"
    Plan {
        string plan_id
        string task_id
        string title
        int total_steps
    }

    PlanStep ||--o{ Subtask : "can have many"
    PlanStep ||--o{ FinalPlanStepOutput : "associated with one"
    PlanStep {
        string step_id
        string plan_id
        string description
        int step_number
        bool completed
    }

    Subtask {
        string subtask_id
        string plan_step_id
        string description
        bool completed
    }

    Interaction ||--o{ Step : "has many"
    Interaction {
        string interaction_id
        string task_id
        string timestamp
    }

    Step ||--o{ Reflection : "can have one"
    Step ||--o{ FinalStepOutput : "can have one"
    Step {
        string step_id_interaction  // Renamed to avoid conflict with PlanStep's step_id
        string interaction_id
        string prompt
        string response
        string agent_id
    }

    Reflection {
        string reflection_id
        string step_id_interaction
        string content
        float score
    }

    FinalStepOutput {
        string final_output_id
        string step_id_interaction
        string output_data
        string output_type
    }

    FinalPlanStepOutput ||--o{ Step : "contains many"
    FinalPlanStepOutput {
        string final_plan_step_output_id
        string plan_step_id
        string summary
    }

    ConversationMemory {
        string memory_id
        string conversation_id
        string last_summary
        list history_interactions
    }

    OutputType {
        string type_name
        string description
        string schema  // e.g., JSON schema
    }

    Interaction ||--o{ ConversationMemory : "can use"
    PlanStep ||--o{ OutputType : "defines target"

```
