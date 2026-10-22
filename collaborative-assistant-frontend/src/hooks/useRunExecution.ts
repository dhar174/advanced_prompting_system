import { useMutation, useQuery, type ApolloError } from '@apollo/client';
import {
  RUN_EXECUTION,
  GET_EXECUTION_MODES,
  type RunExecutionVariables,
  type RunExecutionData,
  type GetExecutionModesData,
} from '../graphql/executionOperations';
import type { ExecutionModeCapabilityType } from '../graphql/graphqlTypes';

type RunExecutionMutateFunction = (
  variables: RunExecutionVariables
) => Promise<void>;

export interface UseRunExecutionResult {
  runExecution: RunExecutionMutateFunction;
  data: RunExecutionData | undefined | null;
  loading: boolean;
  error: ApolloError | undefined;
  executionModes: ExecutionModeCapabilityType[];
  modesLoading: boolean;
  modesError: ApolloError | undefined;
}

export const useRunExecution = (): UseRunExecutionResult => {
  const [mutateFunction, { data, loading, error }] = useMutation<
    RunExecutionData,
    RunExecutionVariables
  >(RUN_EXECUTION);

  const {
    data: modesData,
    loading: modesLoading,
    error: modesError,
  } = useQuery<GetExecutionModesData>(GET_EXECUTION_MODES, {
    fetchPolicy: 'cache-first',
  });

  const runExecution: RunExecutionMutateFunction = async (variables) => {
    try {
      await mutateFunction({ variables });
    } catch {
      // Error handled by useMutation error state
    }
  };

  const executionModes: ExecutionModeCapabilityType[] =
    modesData?.executionModes || [
      {
        id: 'collaborative',
        label: 'Collaborative Multi-Agent',
        description: 'Multi-agent round deliberation & consensus voting.',
        enabled: true,
        gatedIssue: null,
        supportsMultiAgent: true,
        supportsStepPlanning: false,
      },
      {
        id: 'advanced_pipeline',
        label: 'Standalone Pipeline',
        description: 'Complexity-driven step planning, reflections & reward judging.',
        enabled: true,
        gatedIssue: null,
        supportsMultiAgent: true,
        supportsStepPlanning: true,
      },
      {
        id: 'tree_of_thought',
        label: 'Tree of Thoughts',
        description: 'Deliberate search over reasoning trees.',
        enabled: false,
        gatedIssue: 'Issue #36',
        supportsMultiAgent: false,
        supportsStepPlanning: true,
      },
    ];

  return {
    runExecution,
    data,
    loading,
    error,
    executionModes,
    modesLoading,
    modesError,
  };
};
