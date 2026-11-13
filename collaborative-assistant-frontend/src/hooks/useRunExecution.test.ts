/// <reference types="vitest/globals" />
import { renderHook, act } from '@testing-library/react';
import {
  useRunExecution,
  DEFAULT_PERSONALITY_IDS,
  formatExecutionError,
} from './useRunExecution';
import { useMutation, useQuery } from '@apollo/client';
import {
  RUN_EXECUTION,
  GET_EXECUTION_MODES,
  GET_PERSONALITIES,
  GET_PROVIDER_POLICY,
} from '../graphql/executionOperations';
import { clearApiToken, getApiToken, setApiToken } from '../auth/apiToken';
import type { ExecutionModeCapabilityType } from '../graphql/graphqlTypes';
import { describe, it, expect, vi, beforeEach } from 'vitest';

vi.mock('@apollo/client', async (importOriginal) => {
  const original = await importOriginal<Record<string, unknown>>();
  return {
    ...original,
    useMutation: vi.fn(),
    useQuery: vi.fn(),
  };
});

const mockMutateFunction = vi.fn();

const READY_MODES: ExecutionModeCapabilityType[] = [
  {
    id: 'collaborative',
    label: 'Collaborative Debate',
    description: 'Multi-agent round deliberation.',
    enabled: true,
    ready: true,
    unavailableReason: null,
    gatedIssue: null,
    supportsMultiAgent: true,
    supportsStepPlanning: false,
    supportsCancellation: false,
    supportsResume: false,
    minNumRounds: 1,
    maxNumRounds: 10,
    maxSteps: 50,
    maxAgents: 10,
  },
  {
    id: 'advanced_pipeline',
    label: 'Standalone Pipeline',
    description: 'Complexity-driven step planning.',
    enabled: true,
    ready: true,
    unavailableReason: null,
    gatedIssue: null,
    supportsMultiAgent: true,
    supportsStepPlanning: true,
    supportsCancellation: false,
    supportsResume: false,
    minNumRounds: 1,
    maxNumRounds: 10,
    maxSteps: 50,
    maxAgents: 10,
  },
  {
    id: 'tree_of_thought',
    label: 'Tree of Thoughts',
    description: 'Gated reasoning trees.',
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

function mockCapabilities({
  modes = READY_MODES,
  modesLoading = false,
  modesError = undefined as unknown,
  personalities = DEFAULT_PERSONALITY_IDS.map((id) => ({ id, instructions: '' })),
  personalitiesError = undefined as unknown,
  policy = {
    defaultModel: 'gpt-4o-mini',
    allowedModels: ['gpt-4o-mini', 'gpt-4o'],
    defaultTemperature: 0.7,
  },
  policyError = undefined as unknown,
  refetchModes = vi.fn().mockResolvedValue({ data: { executionModes: modes } }),
  refetchPersonalities = vi.fn().mockResolvedValue({ data: { personalities } }),
  refetchPolicy = vi.fn().mockResolvedValue({ data: { providerPolicy: policy } }),
}: {
  modes?: ExecutionModeCapabilityType[] | null;
  modesLoading?: boolean;
  modesError?: unknown;
  personalities?: Array<{ id: string; instructions: string }> | null;
  personalitiesError?: unknown;
  policy?: { defaultModel: string; allowedModels: string[]; defaultTemperature: number } | null;
  policyError?: unknown;
  refetchModes?: ReturnType<typeof vi.fn>;
  refetchPersonalities?: ReturnType<typeof vi.fn>;
  refetchPolicy?: ReturnType<typeof vi.fn>;
} = {}) {
  (useQuery as unknown as ReturnType<typeof vi.fn>).mockImplementation((query: unknown) => {
    if (query === GET_EXECUTION_MODES) {
      return {
        data: modes !== null ? { executionModes: modes } : null,
        loading: modesLoading,
        error: modesError,
        refetch: refetchModes,
      };
    }
    if (query === GET_PERSONALITIES) {
      return {
        data: personalities !== null ? { personalities } : null,
        loading: false,
        error: personalitiesError,
        refetch: refetchPersonalities,
      };
    }
    if (query === GET_PROVIDER_POLICY) {
      return {
        data: policy !== null ? { providerPolicy: policy } : null,
        loading: false,
        error: policyError,
        refetch: refetchPolicy,
      };
    }
    return { data: null, loading: false, error: undefined };
  });
}

describe('useRunExecution hook', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    window.sessionStorage.clear();
    clearApiToken();
    mockMutateFunction.mockResolvedValue({ data: { runExecution: null } });
    (useMutation as unknown as ReturnType<typeof vi.fn>).mockReturnValue([
      mockMutateFunction,
      { data: null, loading: false, error: null },
    ]);
    mockCapabilities();
  });

  it('fails closed (empty, zero executionModes) when discovery returns null data', async () => {
    mockCapabilities({ modes: null, personalities: null, policy: null });
    const { result } = renderHook(() => useRunExecution());

    expect(result.current.discoveryStatus).toBe('empty');
    expect(result.current.executionModes).toEqual([]);
    expect(result.current.capabilitiesReady).toBe(false);

    let output: unknown;
    await act(async () => {
      output = await result.current.runExecution({
        input: { mode: 'collaborative', taskPrompt: 'Should not run' },
      });
    });

    expect(output).toBeNull();
    expect(mockMutateFunction).not.toHaveBeenCalled();
  });

  it('models loading state fail-closed until discovery resolves', async () => {
    mockCapabilities({ modes: null, modesLoading: true });
    const { result } = renderHook(() => useRunExecution());

    expect(result.current.discoveryStatus).toBe('loading');
    expect(result.current.modesLoading).toBe(true);
    expect(result.current.executionModes).toEqual([]);
    expect(result.current.capabilitiesReady).toBe(false);

    await act(async () => {
      await result.current.runExecution({
        input: { mode: 'collaborative', taskPrompt: 'Wait for loading' },
      });
    });
    expect(mockMutateFunction).not.toHaveBeenCalled();
  });

  it('models network/GraphQL discovery failure fail-closed without fallback ready modes', async () => {
    mockCapabilities({
      modes: null,
      modesError: { message: 'Network connection refused' },
    });
    const { result } = renderHook(() => useRunExecution());

    expect(result.current.discoveryStatus).toBe('failed');
    expect(result.current.executionModes).toEqual([]);
    expect(result.current.capabilitiesReady).toBe(false);
    expect(result.current.discoveryErrorMessage).toMatch(/Network connection refused/i);

    await act(async () => {
      await result.current.runExecution({
        input: { mode: 'advanced_pipeline', taskPrompt: 'Run anyway?' },
      });
    });
    expect(mockMutateFunction).not.toHaveBeenCalled();
  });

  it('models empty capability catalog fail-closed and blocks execution', async () => {
    mockCapabilities({ modes: [] });
    const { result } = renderHook(() => useRunExecution());

    expect(result.current.discoveryStatus).toBe('empty');
    expect(result.current.executionModes).toEqual([]);
    expect(result.current.capabilitiesReady).toBe(false);

    await act(async () => {
      await result.current.runExecution({
        input: { mode: 'collaborative', taskPrompt: 'Empty catalog' },
      });
    });
    expect(mockMutateFunction).not.toHaveBeenCalled();
  });

  it('blocks execution when server reports mode disabled or ready=false', async () => {
    const notReadyModes: ExecutionModeCapabilityType[] = READY_MODES.map((m) =>
      m.id === 'collaborative'
        ? {
            ...m,
            enabled: true,
            ready: false,
            unavailableReason: 'OPENAI_API_KEY missing on server.',
          }
        : m
    );
    mockCapabilities({ modes: notReadyModes });
    const { result } = renderHook(() => useRunExecution());

    expect(result.current.discoveryStatus).toBe('available');
    await act(async () => {
      await result.current.runExecution({
        input: { mode: 'collaborative', taskPrompt: 'Not ready' },
      });
    });
    expect(mockMutateFunction).not.toHaveBeenCalled();
    expect(result.current.errorMessage).toMatch(/OPENAI_API_KEY missing/i);

    await act(async () => {
      await result.current.runExecution({
        input: { mode: 'tree_of_thought', taskPrompt: 'Gated' },
      });
    });
    expect(mockMutateFunction).not.toHaveBeenCalled();
    expect(result.current.errorMessage).toMatch(/Gated pending Issue #36/i);
  });

  it('transitions to auth_required on 401 discovery error, clears invalid token, and restores on applyApiToken rediscovery', async () => {
    setApiToken('stale-token');
    let serverAuthorized = false;
    const refetchModes = vi.fn().mockImplementation(async () => {
      serverAuthorized = getApiToken() === 'valid-token';
      if (!serverAuthorized) {
        throw Object.assign(new Error('Unauthorized'), {
          networkError: { statusCode: 401 },
        });
      }
      return { data: { executionModes: [READY_MODES[0]] } };
    });

    (useQuery as unknown as ReturnType<typeof vi.fn>).mockImplementation((query: unknown) => {
      if (query === GET_EXECUTION_MODES) {
        return serverAuthorized
          ? { data: { executionModes: [READY_MODES[0]] }, loading: false, error: undefined, refetch: refetchModes }
          : {
              data: null,
              loading: false,
              error: { networkError: { statusCode: 401 }, message: 'Unauthorized' },
              refetch: refetchModes,
            };
      }
      return { data: null, loading: false, error: undefined, refetch: vi.fn() };
    });

    const { result, rerender } = renderHook(() => useRunExecution());

    expect(result.current.discoveryStatus).toBe('auth_required');
    expect(result.current.authRequired).toBe(true);
    expect(result.current.executionModes).toEqual([]);
    expect(getApiToken()).toBe('');

    await act(async () => {
      await result.current.runExecution({
        input: { mode: 'collaborative', taskPrompt: 'Blocked by 401' },
      });
    });
    expect(mockMutateFunction).not.toHaveBeenCalled();

    // Supply valid token and trigger rediscovery
    await act(async () => {
      await result.current.applyApiToken('valid-token');
    });
    rerender();

    expect(refetchModes).toHaveBeenCalled();
    expect(getApiToken()).toBe('valid-token');
    expect(result.current.discoveryStatus).toBe('available');
    expect(result.current.authRequired).toBe(false);
    // Restores only capabilities returned by server (1 mode, not 3 fabricated defaults)
    expect(result.current.executionModes).toHaveLength(1);
    expect(result.current.executionModes[0].id).toBe('collaborative');
  });

  it('transitions to auth_required when runExecution mutation returns 401', async () => {
    setApiToken('revoked-token');
    mockMutateFunction.mockRejectedValueOnce({
      networkError: { statusCode: 401 },
      message: 'Unauthorized',
    });

    const { result } = renderHook(() => useRunExecution());
    expect(result.current.discoveryStatus).toBe('available');

    await act(async () => {
      await result.current.runExecution({
        input: { mode: 'collaborative', taskPrompt: 'Trigger 401' },
      });
    });

    expect(result.current.authRequired).toBe(true);
    expect(result.current.discoveryStatus).toBe('auth_required');
    expect(getApiToken()).toBe('');
    expect(result.current.errorMessage).toMatch(/Authentication required/i);
  });

  it('enforces capability-driven numRounds bounds and attaches Bearer header on mutation when token is set', async () => {
    setApiToken('active-bearer-token');
    const customModes = READY_MODES.map((m) =>
      m.id === 'collaborative' ? { ...m, minNumRounds: 1, maxNumRounds: 5 } : m
    );
    mockCapabilities({ modes: customModes });

    const { result } = renderHook(() => useRunExecution());

    await act(async () => {
      await result.current.runExecution({
        input: { mode: 'collaborative', taskPrompt: 'Too many rounds', numRounds: 6 },
      });
    });
    expect(mockMutateFunction).not.toHaveBeenCalled();
    expect(result.current.errorMessage).toMatch(/between 1 and 5/i);

    await act(async () => {
      await result.current.runExecution({
        input: { mode: 'collaborative', taskPrompt: 'Valid rounds', numRounds: 5 },
      });
    });

    expect(useMutation).toHaveBeenCalledWith(RUN_EXECUTION);
    expect(mockMutateFunction).toHaveBeenCalledWith(
      expect.objectContaining({
        variables: { input: { mode: 'collaborative', taskPrompt: 'Valid rounds', numRounds: 5 } },
        context: expect.objectContaining({
          headers: { Authorization: 'Bearer active-bearer-token' },
        }),
      })
    );
    expect(useQuery).toHaveBeenCalledWith(
      GET_EXECUTION_MODES,
      expect.objectContaining({
        fetchPolicy: 'cache-first',
      })
    );
    const modesQueryOptions = (useQuery as unknown as ReturnType<typeof vi.fn>).mock.calls.find(
      (call) => call[0] === GET_EXECUTION_MODES
    )?.[1] as { context?: unknown } | undefined;
    expect(modesQueryOptions?.context).toBeUndefined();
  });

  it('clears stored token BEFORE discovery refetch on clearStoredApiToken and transitions fail-closed on 401', async () => {
    setApiToken('active-bearer-token');
    let tokenSeenDuringRefetch: string | null = null;
    let isCleared = false;

    const refetchModes = vi.fn().mockImplementation(async () => {
      tokenSeenDuringRefetch = getApiToken();
      if (!tokenSeenDuringRefetch) {
        isCleared = true;
        throw Object.assign(new Error('Authentication required.'), {
          networkError: { statusCode: 401 },
        });
      }
      return { data: { executionModes: READY_MODES } };
    });

    (useQuery as unknown as ReturnType<typeof vi.fn>).mockImplementation((query: unknown) => {
      if (query === GET_EXECUTION_MODES) {
        return isCleared
          ? {
              data: null,
              loading: false,
              error: { networkError: { statusCode: 401 }, message: 'Authentication required.' },
              refetch: refetchModes,
            }
          : {
              data: { executionModes: READY_MODES },
              loading: false,
              error: undefined,
              refetch: refetchModes,
            };
      }
      return { data: null, loading: false, error: undefined, refetch: vi.fn() };
    });

    const { result, rerender } = renderHook(() => useRunExecution());
    expect(result.current.discoveryStatus).toBe('available');
    expect(result.current.hasApiToken).toBe(true);

    await act(async () => {
      await result.current.clearStoredApiToken();
    });
    rerender();

    expect(refetchModes).toHaveBeenCalledTimes(1);
    expect(tokenSeenDuringRefetch).toBe('');
    expect(getApiToken()).toBe('');
    expect(result.current.hasApiToken).toBe(false);
    expect(result.current.discoveryStatus).toBe('auth_required');
    expect(result.current.executionModes).toEqual([]);
  });

  it('prevents stale response application when cancelled in flight', async () => {
    let resolvePending: ((v: unknown) => void) | null = null;
    mockMutateFunction.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          resolvePending = resolve;
        })
    );

    const { result } = renderHook(() => useRunExecution());
    let returned: unknown = 'unset';
    let promise: Promise<unknown>;

    act(() => {
      promise = result.current.runExecution({
        input: { mode: 'collaborative', taskPrompt: 'Test cancel' },
      });
    });

    expect(result.current.loading).toBe(true);

    act(() => {
      result.current.cancelActiveRun();
    });

    expect(result.current.wasCancelled).toBe(true);
    expect(result.current.loading).toBe(false);
    expect(result.current.errorMessage).toBeNull();

    await act(async () => {
      resolvePending?.({
        data: {
          runExecution: {
            mode: 'collaborative',
            status: 'completed',
            finalOutput: 'Stale late output',
            conversation: [],
            questions: [],
            outputType: { outputType: 'Simple Concise Answer', fileExtension: '.txt' },
            diagnostics: { planSteps: [], reflections: [], agentAttributions: [] },
            events: [],
          },
        },
      });
      returned = await promise;
    });

    expect(returned).toBeNull();
    expect(result.current.lastResult).toBeNull();
  });

  it('formats HTTP security errors cleanly without leaking internals', () => {
    expect(formatExecutionError({ networkError: { statusCode: 401 }, message: 'Unauthorized' })).toMatch(
      /Authentication required/i
    );
    expect(formatExecutionError({ networkError: { statusCode: 403 }, message: 'Forbidden' })).toMatch(
      /Origin rejected/i
    );
    expect(formatExecutionError({ networkError: { statusCode: 429 }, message: 'Too Many' })).toMatch(
      /Rate limit exceeded/i
    );
  });
});
