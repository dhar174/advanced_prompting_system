import { gql } from '@apollo/client';
import type { ConversationInputType, RunConversationResponseType } from '../graphqlTypes';

export const RUN_CONVERSATION = gql`
  mutation RunConversation(
    $conversation: [ConversationInput!]!,
    $assistantPersonalities: [String!]!,
    $leadPersonality: String!,
    $numRounds: Int!
  ) {
    runConversation(
      conversation: $conversation,
      assistantPersonalities: $assistantPersonalities,
      leadPersonality: $leadPersonality,
      numRounds: $numRounds
    ) {
      conversation {
        role
        name
        content
      }
      questions {
        assistant
        question
      }
      finalOutput
      plan {
        steps {
          step_number
          completed
          step_name
          step_description
          step_explanation
          step_output
          step_full_text
          subtasks {
            subtask_number
            completed
            subtask_description
            subtask_name
            subtask_explanation
            subtask_output
            subtask_full_text
            subtasks {
              # Recursive subtasks, if needed, or define depth
              subtask_number
              completed
              subtask_description
              subtask_name
              subtask_explanation
              subtask_output
              subtask_full_text
            }
          }
        }
      }
      conversation_memory {
        facts
        arguments
        decisions
        direct_replies
        recommended_actions
        to_do_list
        completed_tasks
        rounds_left
        decided_output_type
      }
      agent_collaboration {
        agent_name
        priority_score
        contributions
        votes_cast
        questions_asked
      }
      complexity_metrics {
        overall_score
        reasoning_depth
        solution_complexity
        collaboration_intensity
        confidence_level
      }
      processing_status {
        current_round
        total_rounds
        current_step
        progress_percentage
        estimated_time_remaining
      }
    }
  }
`;

export interface RunConversationVariables {
  conversation: ConversationInputType[];
  assistantPersonalities: string[];
  leadPersonality: string;
  numRounds: number;
}

export interface RunConversationData {
  runConversation: RunConversationResponseType;
}
