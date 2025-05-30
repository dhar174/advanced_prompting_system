import { useMutation, type ApolloError } from '@apollo/client';
import { SUBMIT_FEEDBACK, type SubmitFeedbackVariables, type SubmitFeedbackData } from '../graphql/mutations/feedbackMutations';

type SubmitFeedbackMutateFunction = (
  variables: SubmitFeedbackVariables
) => Promise<void>; // Adjusted for consistency

interface UseSubmitFeedbackResult {
  submitFeedback: SubmitFeedbackMutateFunction;
  data: SubmitFeedbackData | undefined | null; // Allow undefined for initial state
  loading: boolean;
  error: ApolloError | undefined;
}

export const useSubmitFeedback = (): UseSubmitFeedbackResult => {
  const [mutateFunction, { data, loading, error }] = useMutation<SubmitFeedbackData, SubmitFeedbackVariables>(SUBMIT_FEEDBACK);

  const submitFeedback: SubmitFeedbackMutateFunction = async (variables) => {
    try {
      await mutateFunction({ variables });
    } catch (e) {
      // Error is already handled by the error object from useMutation
      // console.error("Error in submitFeedback mutation:", e);
    }
  };

  return { submitFeedback, data, loading, error };
};
