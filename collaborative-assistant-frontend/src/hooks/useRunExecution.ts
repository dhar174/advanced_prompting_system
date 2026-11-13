import { useCallback, useRef, useState } from 'react';
import { useMutation, useQuery, type ApolloError } from '@apollo/client';
import {
  RUN_EXECUTION,
  GET_EXECUTION_MODES,
  GET_PERSONALITIES,
  GET_PROVIDER_POLICY,
  type RunExecutionVariables,
  type RunExecutionData,
  type GetExecutionModesData,
  type GetPersonalitiesData,
  type GetProviderPolicyData,
} from '../graphql/executionOperations';
import type {
  ExecutionModeCapabilityType,
  PersonalityCapabilityType,
  ProviderPolicyType,
  RunExecutionResponseType,
} from '../graphql/graphqlTypes';

export const DEFAULT_PERSONALITY_IDS: string[] = [
  'Analytical Thinker',
  'Creative Innovator',
  'Skeptical Critic',
  'Optimistic Advocate',
  'Practical Realist',
  'Empathetic Counselor',
  'Mediator',
  'Socratic Thinker',
  'Data Scientist',
  'Ethical Advisor',
  'Futurist',
  'Storyteller',
  'Strategist',
  'Coding Guru',
  'Legal Expert',
  'Financial Analyst',
  'Healthcare Professional',
  'Marketing Specialist',
  'Educator',
  'Software Engineer',
  'Software Architect',
  'UX Designer',
  'UI Designer',
  'Project Manager',
  'Business Analyst',
  'Code Debugger',
];

export const DEFAULT_EXECUTION_MODES: ExecutionModeCapabilityType[] = [
  {
    id: 'collaborative',
    label: 'Collaborative Multi-Agent',
    description: 'Multi-agent round deliberation & consensus voting.',
    enabled: true,
    ready: true,
    unavailableReason: null,
    gatedIssue: null,
    supportsMultiAgent: true,
    supportsStepPlanning: false,
    supportsCancellation: false,
    supportsResume: false,
  },
  {
    id: 'advanced_pipeline',
    label: 'Standalone Pipeline',
    description: 'Complexity-driven step planning, reflections & reward judging.',
    enabled: true,
    ready: true,
    unavailableReason: null,
    gatedIssue: null,
    supportsMultiAgent: true,
    supportsStepPlanning: true,
    supportsCancellation: false,
    supportsResume: false,
  },
  {
    id: 'tree_of_thought',
    label: 'Tree of Thoughts',
    description: 'Deliberate search over reasoning trees.',
    enabled: false,
    ready: false,
    unavailableReason: 'Gated pending Issue #36.',
    gatedIssue: 'Issue #36',
    supportsMultiAgent: false,
    supportsStepPlanning: true,
    supportsCancellation: false,
    supportsResume: false,
  },
];

export function formatExecutionError(err: unknown): string {
  if (!err) return '';
  if (typeof err === 'string') return err;
  const apolloErr = err as ApolloError & { networkError?: { statusCode?: number } };
  const statusCode = apolloErr?.networkError?.statusCode;
  if (statusCode === 401) {
    return 'Authentication required to execute GraphQL requests.';
  }
  if (statusCode === 403) {
    return 'Request origin rejected by server policy.';
  }
  if (statusCode === 413) {
    return 'Request payload exceeds maximum allowed size.';
  }
  if (statusCode === 415) {
    return 'Unsupported request content type.';
  }
  if (statusCode === 429) {
    return 'Rate limit exceeded. Please wait briefly before retrying.';
  }
  if (apolloErr?.graphQLErrors && apolloErr.graphQLErrors.length > 0) {
    return apolloErr.graphQLErrors[0].message;
  }
  return apolloErr?.message || 'Execution failed due to a network or server error.';
}

export type RunExecutionMutateFunction = (
  variables: RunExecutionVariables
) => Promise<RunExecutionResponseType | null>;

export interface UseRunExecutionResult {
  runExecution: RunExecutionMutateFunction;
  cancelActiveRun: () => void;
  resetExecutionState: () => void;
  data: RunExecutionData | undefined | null;
  lastResult: RunExecutionResponseType | null;
  loading: boolean;
  error: ApolloError | undefined;
  errorMessage: string | null;
  wasCancelled: boolean;
  executionModes: ExecutionModeCapabilityType[];
  personalities: PersonalityCapabilityType[];
  providerPolicy: ProviderPolicyType | null;
  modesLoading: boolean;
  modesError: ApolloError | undefined;
}

export const useRunExecution = (): UseRunExecutionResult => {
  const [mutateFunction, mutationState] = useMutation<
    RunExecutionData,
    RunExecutionVariables
  >(RUN_EXECUTION);
  const { data, loading, error } = mutationState;
  const resetMutation = (mutationState as { reset?: () => void }).reset;

  const {
    data: modesData,
    loading: modesLoading,
    error: modesError,
  } = useQuery<GetExecutionModesData>(GET_EXECUTION_MODES, {
    fetchPolicy: 'cache-first',
  });

  const { data: personalitiesData } = useQuery<GetPersonalitiesData>(GET_PERSONALITIES, {
    fetchPolicy: 'cache-first',
  });

  const { data: policyData } = useQuery<GetProviderPolicyData>(GET_PROVIDER_POLICY, {
    fetchPolicy: 'cache-first',
  });

  const [inFlight, setInFlight] = useState(false);
  const [wasCancelled, setWasCancelled] = useState(false);
  const [clientErrorMessage, setClientErrorMessage] = useState<string | null>(null);
  const [lastResult, setLastResult] = useState<RunExecutionResponseType | null>(null);

  const inFlightRef = useRef(false);
  const requestSeqRef = useRef(0);
  const abortControllerRef = useRef<AbortController | null>(null);

  const cancelActiveRun = useCallback(() => {
    requestSeqRef.current += 1;
    inFlightRef.current = false;
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    resetMutation?.();
    setInFlight(false);
    setWasCancelled(true);
    setClientErrorMessage(null);
  }, [resetMutation]);

  const resetExecutionState = useCallback(() => {
    requestSeqRef.current += 1;
    inFlightRef.current = false;
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    resetMutation?.();
    setInFlight(false);
    setWasCancelled(false);
    setClientErrorMessage(null);
    setLastResult(null);
  }, [resetMutation]);

  const runExecution: RunExecutionMutateFunction = useCallback(
    async (variables) => {
      if (inFlightRef.current) {
        return null;
      }

      const seq = ++requestSeqRef.current;
      const controller = typeof AbortController !== 'undefined' ? new AbortController() : null;
      abortControllerRef.current = controller;
      inFlightRef.current = true;
      setInFlight(true);
      setWasCancelled(false);
      setClientErrorMessage(null);

      try {
        const result = await mutateFunction(
          controller
            ? { variables, context: { fetchOptions: { signal: controller.signal } } }
            : { variables }
        );
        // Ignore stale or cancelled responses
        if (requestSeqRef.current !== seq) {
          return null;
        }
        const payload = result?.data?.runExecution ?? null;
        if (payload) {
          setLastResult(payload);
        }
        return payload;
      } catch (err) {
        if (requestSeqRef.current !== seq) {
          return null;
        }
        const formatted = formatExecutionError(err);
        setClientErrorMessage(formatted || 'Execution request failed.');
        return null;
      } finally {
        if (requestSeqRef.current === seq) {
          abortControllerRef.current = null;
          inFlightRef.current = false;
          setInFlight(false);
        }
      }
    },
    [mutateFunction]
  );

  const executionModes: ExecutionModeCapabilityType[] =
    modesData?.executionModes && modesData.executionModes.length > 0
      ? modesData.executionModes
      : DEFAULT_EXECUTION_MODES;

  const personalities: PersonalityCapabilityType[] =
    personalitiesData?.personalities && personalitiesData.personalities.length > 0
      ? personalitiesData.personalities
      : DEFAULT_PERSONALITY_IDS.map((id) => ({ id, instructions: '' }));

  const providerPolicy: ProviderPolicyType | null = policyData?.providerPolicy ?? null;
  const combinedLoading = Boolean(loading || inFlight);
  const derivedError = wasCancelled
    ? null
    : clientErrorMessage || (error ? formatExecutionError(error) : null);

  return {
    runExecution,
    cancelActiveRun,
    resetExecutionState,
    data: wasCancelled ? null : data,
    lastResult,
    loading: combinedLoading,
    error: wasCancelled ? undefined : error,
    errorMessage: derivedError,
    wasCancelled,
    executionModes,
    personalities,
    providerPolicy,
    modesLoading,
    modesError,
  };
};
