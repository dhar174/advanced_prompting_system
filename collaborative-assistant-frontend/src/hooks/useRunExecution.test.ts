/// <reference types="vitest/globals" />
import { renderHook, act } from '@testing-library/react';
import { useRunExecution, DEFAULT_PERSONALITY_IDS, formatExecutionError } from './useRunExecution';
import { useMutation, useQuery } from '@apollo/client';
import {
  RUN_EXECUTION,
  GET_EXECUTION_MODES,
  GET_PERSONALITIES,
  GET_PROVIDER_POLICY,
} from '../graphql/executionOperations';
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

describe('useRunExecution hook', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockMutateFunction.mockResolvedValue({ data: { runExecution: null } });
    (useMutation as unknown as ReturnType<typeof vi.fn>).mockReturnValue([
      mockMutateFunction,
      { data: null, loading: false, error: null },
    ]);
    (useQuery as unknown as ReturnType<typeof vi.fn>).mockReturnValue({
      data: null,
      loading: false,
      error: null,
    });
  });

  it('initializes with default execution modes, 26 backend personalities, and null data', () => {
    const { result } = renderHook(() => useRunExecution());
    expect(typeof result.current.runExecution).toBe('function');
    expect(typeof result.current.cancelActiveRun).toBe('function');
    expect(result.current.data).toBeNull();
    expect(result.current.loading).toBe(false);
    expect(result.current.executionModes.length).toBe(3);
    expect(result.current.executionModes[0].id).toBe('collaborative');
    expect(result.current.executionModes[1].id).toBe('advanced_pipeline');
    expect(result.current.executionModes[2].id).toBe('tree_of_thought');
    expect(result.current.personalities.length).toBe(DEFAULT_PERSONALITY_IDS.length);
    expect(result.current.personalities.map((p) => p.id)).toContain('Analytical Thinker');
    expect(result.current.personalities.map((p) => p.id)).toContain('Mediator');
  });

  it('calls mutate function with provided variables and fetches capability queries', async () => {
    const { result } = renderHook(() => useRunExecution());
    const inputPayload = {
      input: {
        mode: 'advanced_pipeline',
        taskPrompt: 'Analyze performance metrics',
        maxSteps: 10,
      },
    };

    await act(async () => {
      await result.current.runExecution(inputPayload);
    });

    expect(useMutation).toHaveBeenCalledWith(RUN_EXECUTION);
    expect(mockMutateFunction).toHaveBeenCalledWith(
      expect.objectContaining({ variables: inputPayload })
    );
    expect(useQuery).toHaveBeenCalledWith(
      GET_EXECUTION_MODES,
      expect.objectContaining({ fetchPolicy: 'cache-first' })
    );
    expect(useQuery).toHaveBeenCalledWith(
      GET_PERSONALITIES,
      expect.objectContaining({ fetchPolicy: 'cache-first' })
    );
    expect(useQuery).toHaveBeenCalledWith(
      GET_PROVIDER_POLICY,
      expect.objectContaining({ fetchPolicy: 'cache-first' })
    );
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
