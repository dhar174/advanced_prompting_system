```mermaid
graph TD
    A[Start run_conversation] --> B[Initial Problem Statement Input];
    B --> C[Consensus on Problem Definition (define_problem)];
    C --> D[Output Type Determination (output_type_determination)];

    D --> E{Main Conversation Loop (Rounds)};
    E -- Next Round --> F[Summarize Previous Round (if applicable)];
    F --> G[Calculate Assistant Priorities];
    G --> H[Mediator Guides Discussion for Current Round];

    H --> I{Loop Through Prioritized Assistants};
    I -- Next Assistant --> J[Get Assistant Response (get_assistant_response)];
    J --> K[Extract Information into ConversationMemory (extract_information)];
    K --> L[Handle Socratic Questions / Direct Replies];
    L --> I; 

    I -- All Assistants Processed for Round --> M[Analyze to_do_list Completion (analyze_to_do_list)];
    M --> N[Mediator Summarizes Round];
    N --> O{Voting: Continue or End Conversation?};
    O -- Continue --> E; 

    O -- End Conversation --> P[Final Decision/Solution Generation by Designated Assistant];
    P --> Q[Final Output Generation using Function Calling (finalize_output)];
    Q --> R[Iterative Refinement of Final Output (analyze_final_output)];
    R --> S{Is Output Sufficient?};
    S -- Yes --> T[End run_conversation];
    S -- No --> Q;

    subgraph "Conversation Round"
        direction LR
        F --> G --> H --> I
        I -- Next Assistant --> J --> K --> L
        L --> I
        I -- All Assistants Processed --> M --> N --> O
    end
```
