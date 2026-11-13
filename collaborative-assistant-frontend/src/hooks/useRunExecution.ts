import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
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
import {
  clearApiToken,
  getApiToken,
  isAuthError,
  setApiToken,
  subscribeApiToken,
  subscribeAuthRequired,
} from '../auth/apiToken';

export const DEFAULT_MIN_NUM_ROUNDS = 1;
export const DEFAULT_MAX_NUM_ROUNDS = 10;
export const DEFAULT_MAX_STEPS = 50;
export const DEFAULT_MAX_AGENTS = 10;

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

export type CapabilityDiscoveryStatus =
  | 'loading'
  | 'available'
  | 'empty'
  | 'failed'
  | 'auth_required';

export function formatExecutionError(err: unknown): string {
  if (!err) return '';
  if (typeof err === 'string') return err;
  const apolloErr = err as ApolloError & {
    statusCode?: number;
    networkError?: { statusCode?: number; status?: number; response?: { status?: number } };
  };
  const statusCode =
    apolloErr?.networkError?.statusCode ??
    apolloErr?.networkError?.status ??
    apolloErr?.networkError?.response?.status ??
    apolloErr?.statusCode;
  if (statusCode === 401 || isAuthError(err)) {
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
  retryDiscovery: () => Promise<void>;
  applyApiToken: (token: string) => Promise<void>;
  clearStoredApiToken: () => Promise<void>;
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
  discoveryStatus: CapabilityDiscoveryStatus;
  discoveryErrorMessage: string | null;
  capabilitiesReady: boolean;
  authRequired: boolean;
  hasApiToken: boolean;
}

export const useRunExecution = (): UseRunExecutionResult => {
  const [mutateFunction, mutationState] = useMutation<
    RunExecutionData,
    RunExecutionVariables
  >(RUN_EXECUTION);
  const { data, loading, error } = mutationState;
  const resetMutation = (mutationState as { reset?: () => void }).reset;

  const [apiToken, setApiTokenState] = useState<string>(() => getApiToken());
  const [authRequiredOverride, setAuthRequiredOverride] = useState<boolean>(false);
  const [authErrorMessage, setAuthErrorMessage] = useState<string | null>(null);
  const [isRetryingDiscovery, setIsRetryingDiscovery] = useState<boolean>(false);

  const queryContext = useMemo(
    () => (apiToken ? { headers: { Authorization: `Bearer ${apiToken}` } } : undefined),
    [apiToken]
  );

  const {
    data: modesData,
    loading: modesLoading,
    error: modesError,
    refetch: refetchModes,
  } = useQuery<GetExecutionModesData>(GET_EXECUTION_MODES, {
    fetchPolicy: 'cache-first',
    notifyOnNetworkStatusChange: true,
    ...(queryContext ? { context: queryContext } : {}),
  });

  const {
    data: personalitiesData,
    loading: personalitiesLoading,
    error: personalitiesError,
    refetch: refetchPersonalities,
  } = useQuery<GetPersonalitiesData>(GET_PERSONALITIES, {
    fetchPolicy: 'cache-first',
    ...(queryContext ? { context: queryContext } : {}),
  });

  const {
    data: policyData,
    loading: policyLoading,
    error: policyError,
    refetch: refetchPolicy,
  } = useQuery<GetProviderPolicyData>(GET_PROVIDER_POLICY, {
    fetchPolicy: 'cache-first',
    ...(queryContext ? { context: queryContext } : {}),
  });

  const [inFlight, setInFlight] = useState(false);
  const [wasCancelled, setWasCancelled] = useState(false);
  const [clientErrorMessage, setClientErrorMessage] = useState<string | null>(null);
  const [lastResult, setLastResult] = useState<RunExecutionResponseType | null>(null);

  const inFlightRef = useRef(false);
  const requestSeqRef = useRef(0);
  const abortControllerRef = useRef<AbortController | null>(null);

  useEffect(() => {
    const unsubToken = subscribeApiToken((nextToken) => {
      setApiTokenState(nextToken);
    });
    const unsubAuth = subscribeAuthRequired((msg) => {
      setAuthRequiredOverride(true);
      setAuthErrorMessage(msg || 'Authentication required to execute GraphQL requests.');
    });
    return () => {
      unsubToken();
      unsubAuth();
    };
  }, []);

  useEffect(() => {
    const capAuthErr = [modesError, personalitiesError, policyError].find((e) =>
      isAuthError(e)
    );
    if (capAuthErr) {
      clearApiToken();
      setAuthRequiredOverride(true);
      setAuthErrorMessage(formatExecutionError(capAuthErr));
    }
  }, [modesError, personalitiesError, policyError]);

  const validServerModes = useMemo(() => {
    const rawModes = modesData?.executionModes;
    if (!Array.isArray(rawModes)) {
      return [];
    }
    return rawModes.filter(
      (mode): mode is ExecutionModeCapabilityType =>
        Boolean(mode && typeof mode.id === 'string' && typeof mode.enabled === 'boolean')
    );
  }, [modesData]);

  const anyAuthQueryError =
    isAuthError(modesError) ||
    isAuthError(personalitiesError) ||
    isAuthError(policyError);

  const anyCapabilityQueryError = Boolean(
    modesError || personalitiesError || policyError
  );

  const discoveryStatus: CapabilityDiscoveryStatus = useMemo(() => {
    if (
      isRetryingDiscovery ||
      modesLoading ||
      (!modesData && (personalitiesLoading || policyLoading))
    ) {
      return 'loading';
    }
    if (authRequiredOverride || anyAuthQueryError) {
      return 'auth_required';
    }
    if (anyCapabilityQueryError) {
      return 'failed';
    }
    if (!modesData || validServerModes.length === 0) {
      return 'empty';
    }
    return 'available';
  }, [
    isRetryingDiscovery,
    modesLoading,
    modesData,
    personalitiesLoading,
    policyLoading,
    authRequiredOverride,
    anyAuthQueryError,
    anyCapabilityQueryError,
    validServerModes.length,
  ]);

  // Fail closed: never expose fabricated enabled/ready modes when discovery is not confirmed.
  const executionModes: ExecutionModeCapabilityType[] = useMemo(
    () => (discoveryStatus === 'available' ? validServerModes : []),
    [discoveryStatus, validServerModes]
  );

  const personalities: PersonalityCapabilityType[] = useMemo(() => {
    if (discoveryStatus === 'auth_required' || discoveryStatus === 'failed') {
      return [];
    }
    if (personalitiesData?.personalities && personalitiesData.personalities.length > 0) {
      return personalitiesData.personalities;
    }
    if (discoveryStatus === 'available') {
      return DEFAULT_PERSONALITY_IDS.map((id) => ({ id, instructions: '' }));
    }
    return [];
  }, [discoveryStatus, personalitiesData]);

  const providerPolicy: ProviderPolicyType | null = useMemo(() => {
    if (discoveryStatus === 'auth_required' || discoveryStatus === 'failed') {
      return null;
    }
    return policyData?.providerPolicy ?? null;
  }, [discoveryStatus, policyData]);

  const capabilitiesReady = useMemo(
    () =>
      discoveryStatus === 'available' &&
      executionModes.some((mode) => mode.enabled && mode.ready === true),
    [discoveryStatus, executionModes]
  );

  const discoveryErrorMessage = useMemo(() => {
    if (discoveryStatus === 'auth_required') {
      return (
        authErrorMessage ||
        formatExecutionError(modesError || personalitiesError || policyError) ||
        'Authentication required to access GraphQL capabilities.'
      );
    }
    if (discoveryStatus === 'failed') {
      return (
        formatExecutionError(modesError || personalitiesError || policyError) ||
        'Unable to verify server execution capabilities.'
      );
    }
    if (discoveryStatus === 'empty') {
      return 'No execution modes were returned by the server.';
    }
    return null;
  }, [
    discoveryStatus,
    authErrorMessage,
    modesError,
    personalitiesError,
    policyError,
  ]);

  const retryDiscovery = useCallback(async () => {
    setIsRetryingDiscovery(true);
    setAuthRequiredOverride(false);
    setAuthErrorMessage(null);
    setClientErrorMessage(null);
    try {
      const results = await Promise.allSettled([
        refetchModes?.(),
        refetchPersonalities?.(),
        refetchPolicy?.(),
      ]);
      for (const res of results) {
        if (res.status === 'rejected' && isAuthError(res.reason)) {
          clearApiToken();
          setAuthRequiredOverride(true);
          setAuthErrorMessage(formatExecutionError(res.reason));
          break;
        }
      }
    } finally {
      setIsRetryingDiscovery(false);
    }
  }, [refetchModes, refetchPersonalities, refetchPolicy]);

  const applyApiToken = useCallback(
    async (token: string) => {
      const trimmed = String(token || '').trim();
      if (!trimmed) {
        clearApiToken();
        setApiTokenState('');
        setAuthRequiredOverride(false);
        await retryDiscovery();
        return;
      }
      setApiToken(trimmed);
      setApiTokenState(trimmed);
      await retryDiscovery();
    },
    [retryDiscovery]
  );

  const clearStoredApiToken = useCallback(async () => {
    clearApiToken();
    setApiTokenState('');
    setAuthRequiredOverride(false);
    setAuthErrorMessage(null);
    await retryDiscovery();
  }, [retryDiscovery]);

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

      if (discoveryStatus !== 'available') {
        setClientErrorMessage(
          discoveryErrorMessage || 'Server capabilities are not ready for execution.'
        );
        return null;
      }

      const modeCap = executionModes.find((m) => m.id === variables.input.mode);
      if (!modeCap || !modeCap.enabled || modeCap.ready !== true) {
        setClientErrorMessage(
          modeCap?.unavailableReason ||
            `Execution mode '${variables.input.mode}' is not available or ready.`
        );
        return null;
      }

      if (
        variables.input.mode === 'collaborative' &&
        variables.input.numRounds !== undefined
      ) {
        const minRounds = modeCap.minNumRounds ?? DEFAULT_MIN_NUM_ROUNDS;
        const maxRounds = modeCap.maxNumRounds ?? DEFAULT_MAX_NUM_ROUNDS;
        if (
          !Number.isInteger(variables.input.numRounds) ||
          variables.input.numRounds < minRounds ||
          variables.input.numRounds > maxRounds
        ) {
          setClientErrorMessage(
            `Number of rounds must be between ${minRounds} and ${maxRounds}.`
          );
          return null;
        }
      }

      const seq = ++requestSeqRef.current;
      const controller =
        typeof AbortController !== 'undefined' ? new AbortController() : null;
      abortControllerRef.current = controller;
      inFlightRef.current = true;
      setInFlight(true);
      setWasCancelled(false);
      setClientErrorMessage(null);

      const currentToken = getApiToken();
      const mutateContext: Record<string, unknown> = {};
      if (controller) {
        mutateContext.fetchOptions = { signal: controller.signal };
      }
      if (currentToken) {
        mutateContext.headers = { Authorization: `Bearer ${currentToken}` };
      }

      try {
        const result = await mutateFunction(
          Object.keys(mutateContext).length > 0
            ? { variables, context: mutateContext }
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
        if (isAuthError(err)) {
          clearApiToken();
          setApiTokenState('');
          setAuthRequiredOverride(true);
          setAuthErrorMessage(formatted);
        }
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
    [discoveryStatus, discoveryErrorMessage, executionModes, mutateFunction]
  );

  const combinedLoading = Boolean(loading || inFlight);
  const derivedError = wasCancelled
    ? null
    : clientErrorMessage || (error ? formatExecutionError(error) : null);

  return {
    runExecution,
    cancelActiveRun,
    resetExecutionState,
    retryDiscovery,
    applyApiToken,
    clearStoredApiToken,
    data: wasCancelled ? null : data,
    lastResult,
    loading: combinedLoading,
    error: wasCancelled ? undefined : error,
    errorMessage: derivedError,
    wasCancelled,
    executionModes,
    personalities,
    providerPolicy,
    modesLoading: Boolean(modesLoading || isRetryingDiscovery),
    modesError,
    discoveryStatus,
    discoveryErrorMessage,
    capabilitiesReady,
    authRequired: discoveryStatus === 'auth_required',
    hasApiToken: Boolean(apiToken),
  };
};
