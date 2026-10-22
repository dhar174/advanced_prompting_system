/// <reference types="vitest/globals" />
import { renderHook, act } from '@testing-library/react';
import { useRunExecution } from './useRunExecution';
import { useMutation, useQuery } from '@apollo/client';
import { RUN_EXECUTION, GET_EXECUTION_MODES } from '../graphql/executionOperations';
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

  it('should initialize with default execution modes and null data', () => {
    const { result } = renderHook(() => useRunExecution());
    expect(typeof result.current.runExecution).toBe('function');
    expect(result.current.data).toBeNull();
    expect(result.current.loading).toBe(false);
    expect(result.current.executionModes.length).toBe(3);
    expect(result.current.executionModes[0].id).toBe('collaborative');
    expect(result.current.executionModes[1].id).toBe('advanced_pipeline');
    expect(result.current.executionModes[2].id).toBe('tree_of_thought');
  });

  it('should call mutate function with provided variables', async () => {
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
    expect(mockMutateFunction).toHaveBeenCalledWith({ variables: inputPayload });
  });

  it('should query execution modes with cache-first policy', () => {
    renderHook(() => useRunExecution());
    expect(useQuery).toHaveBeenCalledWith(
      GET_EXECUTION_MODES,
      expect.objectContaining({ fetchPolicy: 'cache-first' })
    );
  });
});
