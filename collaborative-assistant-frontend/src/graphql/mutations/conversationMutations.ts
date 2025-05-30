import { gql } from '@apollo/client';
import type { ConversationInputType, ConversationType, QuestionType } from '../graphqlTypes';

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
  runConversation: {
    conversation: ConversationType[];
    questions: QuestionType[] | null;
    finalOutput: string;
  };
}
