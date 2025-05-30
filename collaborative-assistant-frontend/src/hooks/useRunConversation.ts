import { useMutation, ApolloError } from '@apollo/client';
import { RUN_CONVERSATION, RunConversationVariables, RunConversationData } from '../graphql/mutations/conversationMutations';

type RunConversationMutateFunction = (
  variables: RunConversationVariables
) => Promise<void>; // Adjusted to better reflect typical useMutation hook return

interface UseRunConversationResult {
  runConversation: RunConversationMutateFunction;
  data: RunConversationData | undefined | null; // Allow undefined for initial state
  loading: boolean;
  error: ApolloError | undefined;
}

export const useRunConversation = (): UseRunConversationResult => {
  const [mutateFunction, { data, loading, error }] = useMutation<RunConversationData, RunConversationVariables>(RUN_CONVERSATION);

  const runConversation: RunConversationMutateFunction = async (variables) => {
    try {
      await mutateFunction({ variables });
    } catch (e) {
      // Error is already handled by the error object from useMutation
      // console.error("Error in runConversation mutation:", e);
    }
  };
  
  return { runConversation, data, loading, error };
};
