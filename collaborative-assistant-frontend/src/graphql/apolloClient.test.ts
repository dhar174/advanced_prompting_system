/// <reference types="vitest/globals" />
import { ApolloLink, Observable } from '@apollo/client';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { createApolloClient } from './apolloClient';
import { GET_EXECUTION_MODES, RUN_EXECUTION } from './executionOperations';
import { SUBMIT_FEEDBACK } from './mutations/feedbackMutations';
import {
  clearApiToken,
  getApiToken,
  setApiToken,
  subscribeAuthRequired,
} from '../auth/apiToken';

describe('apolloClient auth & 401 handling', () => {
  beforeEach(() => {
    window.sessionStorage.clear();
    clearApiToken();
  });

  it('attaches Authorization: Bearer <token> to queries and mutations when token is set', async () => {
    const capturedHeaders: Array<Record<string, string> | undefined> = [];
    const mockLink = new ApolloLink((operation) => {
      const ctx = operation.getContext();
      capturedHeaders.push(ctx.headers as Record<string, string> | undefined);
      return new Observable((observer) => {
        if (operation.operationName === 'GetExecutionModes') {
          observer.next({ data: { executionModes: [] } });
        } else if (operation.operationName === 'RunExecution') {
          observer.next({
            data: {
              runExecution: {
                mode: 'collaborative',
                effectiveMode: 'collaborative',
                status: 'completed',
                stopReason: 'final_answer',
                finalOutput: 'ok',
                errorMessage: null,
                usageAvailable: false,
                outputType: { outputType: 'Simple Concise Answer', fileExtension: '.txt' },
                conversation: [],
                questions: [],
                artifacts: [],
                diagnostics: {
                  complexityScore: null,
                  initialBudget: null,
                  adjustedBudget: null,
                  finalReward: null,
                  durationMs: null,
                  tokenCount: null,
                  correlationId: null,
                  planSteps: [],
                  reflections: [],
                  agentAttributions: [],
                },
                events: [],
              },
            },
          });
        } else {
          observer.next({ data: { submitFeedback: { success: 'ok' } } });
        }
        observer.complete();
      });
    });

    setApiToken('my-session-token');
    const apollo = createApolloClient(mockLink);

    await apollo.query({ query: GET_EXECUTION_MODES, fetchPolicy: 'network-only' });
    await apollo.mutate({
      mutation: RUN_EXECUTION,
      variables: { input: { mode: 'collaborative', taskPrompt: 'Hello' } },
    });
    await apollo.mutate({
      mutation: SUBMIT_FEEDBACK,
      variables: {
        conversation: [{ role: 'user', content: 'Hi' }],
        feedback: [{ assistant: 'Mediator', rating: 5 }],
      },
    });

    expect(capturedHeaders).toHaveLength(3);
    for (const headers of capturedHeaders) {
      expect(headers?.Authorization).toBe('Bearer my-session-token');
    }
  });

  it('omits Authorization header when no token is set or after clearApiToken', async () => {
    const capturedHeaders: Array<Record<string, string> | undefined> = [];
    const mockLink = new ApolloLink((operation) => {
      capturedHeaders.push(operation.getContext().headers as Record<string, string> | undefined);
      return new Observable((observer) => {
        observer.next({ data: { executionModes: [] } });
        observer.complete();
      });
    });

    const apollo = createApolloClient(mockLink);
    await apollo.query({ query: GET_EXECUTION_MODES, fetchPolicy: 'network-only' });

    setApiToken('temp-token');
    clearApiToken();
    await apollo.query({ query: GET_EXECUTION_MODES, fetchPolicy: 'network-only' });

    expect(capturedHeaders[0]?.Authorization).toBeUndefined();
    expect(capturedHeaders[1]?.Authorization).toBeUndefined();
  });

  it('clears stored token and notifies authRequired listeners on HTTP 401', async () => {
    const onAuthRequired = vi.fn();
    const unsub = subscribeAuthRequired(onAuthRequired);

    const mock401Link = new ApolloLink(() => {
      return new Observable((observer) => {
        const err = Object.assign(new Error('Response not successful: Received status code 401'), {
          statusCode: 401,
          result: { error: 'Invalid bearer token.' },
        });
        observer.error(err);
      });
    });

    setApiToken('expired-token');
    const apollo = createApolloClient(mock401Link);

    await expect(
      apollo.query({ query: GET_EXECUTION_MODES, fetchPolicy: 'network-only' })
    ).rejects.toThrow();

    expect(getApiToken()).toBe('');
    expect(onAuthRequired).toHaveBeenCalledTimes(1);
    unsub();
  });
});
