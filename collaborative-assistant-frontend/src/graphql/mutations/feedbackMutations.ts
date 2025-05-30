import { gql } from '@apollo/client';
import { ConversationInputType, FeedbackInputType, QuestionInputType } from '../graphqlTypes';

export const SUBMIT_FEEDBACK = gql`
  mutation SubmitFeedback(
    $conversation: [ConversationInput!]!,
    $feedback: [FeedbackInput!]!,
    $questions: [QuestionInput!]
  ) {
    submitFeedback(
      conversation: $conversation,
      feedback: $feedback,
      questions: $questions
    ) {
      success
    }
  }
`;

export interface SubmitFeedbackVariables {
  conversation: ConversationInputType[];
  feedback: FeedbackInputType[];
  questions?: QuestionInputType[] | null;
}

export interface SubmitFeedbackData {
  submitFeedback: {
    success: boolean; // Changed from string to boolean to match typical success responses
  };
}
