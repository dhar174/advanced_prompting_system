/// <reference types="vitest/globals" />
import { describe, it, expect } from 'vitest';
import { print } from 'graphql';
import { SUBMIT_FEEDBACK, type SubmitFeedbackData } from './feedbackMutations';
import {
  GET_EXECUTION_MODES,
  GET_PERSONALITIES,
  GET_PROVIDER_POLICY,
  RUN_EXECUTION,
} from '../executionOperations';

describe('GraphQL Operation Contracts', () => {
  it('defines SubmitFeedback mutation matching string success payload', () => {
    const queryStr = print(SUBMIT_FEEDBACK);
    expect(queryStr).toContain('submitFeedback');
    expect(queryStr).toContain('success');
    const sample: SubmitFeedbackData = {
      submitFeedback: { success: 'Feedback submitted successfully.' },
    };
    expect(typeof sample.submitFeedback.success).toBe('string');
  });

  it('includes readiness, personality catalog, provider policy, and artifact selections', () => {
    expect(print(GET_EXECUTION_MODES)).toContain('unavailableReason');
    expect(print(GET_PERSONALITIES)).toContain('personalities');
    expect(print(GET_PROVIDER_POLICY)).toContain('allowedModels');
    const runExecStr = print(RUN_EXECUTION);
    expect(runExecStr).toContain('artifacts');
    expect(runExecStr).toContain('stopReason');
    expect(runExecStr).toContain('effectiveMode');
  });
});
