/// <reference types="vitest/globals" />
import { render, screen, act, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import ConversationPage from './ConversationPage';
import {
  useRunExecution,
  DEFAULT_EXECUTION_MODES,
  DEFAULT_PERSONALITY_IDS,
} from '../hooks/useRunExecution';
import { vi, describe, it, expect, beforeEach } from 'vitest';
import { ApolloProvider } from '@apollo/client';
import { client } from '../graphql/apolloClient';

vi.mock('../hooks/useRunExecution', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../hooks/useRunExecution')>();
  return {
    ...actual,
    useRunExecution: vi.fn(),
  };
});

const { mockToastError, mockToastSuccess } = vi.hoisted(() => ({
  mockToastError: vi.fn(),
  mockToastSuccess: vi.fn(),
}));

vi.mock('react-hot-toast', () => ({
  __esModule: true,
  default: {
    success: mockToastSuccess,
    error: mockToastError,
  },
  Toaster: () => <div data-testid="toaster" />,
}));

const mockRunExecutionFn = vi.fn();
const mockCancelActiveRun = vi.fn();
const mockResetExecutionState = vi.fn();

const buildHookMock = (overrides: Record<string, unknown> = {}) => ({
  runExecution: mockRunExecutionFn,
  cancelActiveRun: mockCancelActiveRun,
  resetExecutionState: mockResetExecutionState,
  data: null,
  lastResult: null,
  loading: false,
  error: undefined,
  errorMessage: null,
  wasCancelled: false,
  executionModes: DEFAULT_EXECUTION_MODES,
  personalities: DEFAULT_PERSONALITY_IDS.map((id) => ({ id, instructions: '' })),
  providerPolicy: {
    defaultModel: 'gpt-4o-mini',
    allowedModels: ['gpt-4o-mini', 'gpt-4o'],
    defaultTemperature: 0.7,
  },
  modesLoading: false,
  modesError: undefined,
  ...overrides,
});

describe('ConversationPage Unified Lifecycle Integration', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockRunExecutionFn.mockResolvedValue(null);
    (useRunExecution as unknown as ReturnType<typeof vi.fn>).mockImplementation(() => buildHookMock());
  });

  const renderConversationPage = () => {
    return render(
      <ApolloProvider client={client}>
        <ConversationPage />
      </ApolloProvider>
    );
  };

  it('starts collaborative session with authoritative backend personalities and routes through runExecution', async () => {
    renderConversationPage();

    const startButton = screen.getByRole('button', { name: /Start New Conversation/i });
    await userEvent.click(startButton);
    expect(
      await screen.findByText(/Conversation started with Mediator leading Analytical Thinker, Mediator for 3 rounds./i)
    ).toBeInTheDocument();

    const textarea = screen.getByPlaceholderText('Type your message...') as HTMLTextAreaElement;
    await userEvent.type(textarea, 'Design a resilient cache');
    await userEvent.click(screen.getByTestId('send-message-button'));

    expect(mockRunExecutionFn).toHaveBeenCalledTimes(1);
    const args = mockRunExecutionFn.mock.calls[0][0];
    expect(args.input.mode).toBe('collaborative');
    expect(args.input.taskPrompt).toBe('Design a resilient cache');
    expect(args.input.conversation).toEqual([
      { role: 'user', name: 'User', content: 'Design a resilient cache' },
    ]);
    expect(args.input.assistantPersonalities).toEqual(['Analytical Thinker', 'Mediator']);
    expect(args.input.leadPersonality).toBe('Mediator');
    expect(args.input.model).toBe('gpt-4o-mini');
  });

  it('displays assistant reply, questions, artifacts, and final output for completed runExecution', async () => {
    const completedPayload = {
      mode: 'collaborative',
      effectiveMode: 'collaborative',
      status: 'completed',
      stopReason: 'final_answer',
      conversation: [
        { role: 'system', name: 'System', content: 'Conversation started...' },
        { role: 'user', name: 'User', content: 'Summarize architecture' },
        { role: 'assistant', name: 'Mediator', content: 'Consensus reached on layered services.' },
      ],
      questions: [{ assistant: 'Analytical Thinker', question: 'Do you need sharding?' }],
      finalOutput: 'Architecture consensus summary.',
      outputType: { outputType: 'Simple Concise Answer', fileExtension: '.txt' },
      artifacts: [
        {
          artifactId: 'art-1',
          runId: 'run-42',
          filename: 'summary.txt',
          path: 'artifacts/run-42/summary.txt',
          mimeType: 'text/plain',
          sizeBytes: 128,
          sha256: 'abcdef1234567890',
        },
      ],
      diagnostics: {
        complexityScore: 0.45,
        initialBudget: 10,
        adjustedBudget: 10,
        finalReward: 0.9,
        durationMs: 420,
        planSteps: [],
        reflections: [],
        agentAttributions: [],
      },
      events: [],
    };
    mockRunExecutionFn.mockResolvedValueOnce(completedPayload);

    renderConversationPage();
    await userEvent.click(screen.getByRole('button', { name: /Start New Conversation/i }));

    const textarea = screen.getByPlaceholderText('Type your message...') as HTMLTextAreaElement;
    await userEvent.type(textarea, 'Summarize architecture');
    await userEvent.click(screen.getByTestId('send-message-button'));

    expect(await screen.findByText('Consensus reached on layered services.')).toBeInTheDocument();
    expect(await screen.findByText('Do you need sharding?')).toBeInTheDocument();
    expect(await screen.findByText('Architecture consensus summary.')).toBeInTheDocument();
    expect(await screen.findByText('summary.txt')).toBeInTheDocument();

    // After completed run, sending again without restarting is blocked cleanly
    await userEvent.type(textarea, 'Follow-up after end');
    await userEvent.click(screen.getByTestId('send-message-button'));
    expect(mockToastError).toHaveBeenCalledWith('Please start a new conversation first.');
    expect(mockRunExecutionFn).toHaveBeenCalledTimes(1);
  });

  it('prevents double-send while a run is in flight and supports Cancel Run', async () => {
    let resolveRun: ((v: unknown) => void) | null = null;
    mockRunExecutionFn.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          resolveRun = resolve;
        })
    );

    renderConversationPage();
    await userEvent.click(screen.getByRole('button', { name: /Start New Conversation/i }));

    const textarea = screen.getByPlaceholderText('Type your message...') as HTMLTextAreaElement;
    await userEvent.type(textarea, 'Long running task');
    await userEvent.click(screen.getByTestId('send-message-button'));

    expect(mockRunExecutionFn).toHaveBeenCalledTimes(1);

    // Rerender in loading state to test Cancel button and disabled send
    (useRunExecution as unknown as ReturnType<typeof vi.fn>).mockImplementation(() =>
      buildHookMock({ loading: true })
    );
    await act(async () => {
      fireEvent.click(screen.getByTestId('send-message-button'));
    });
    expect(mockRunExecutionFn).toHaveBeenCalledTimes(1);

    await act(async () => {
      resolveRun?.(null);
    });
  });

  it('renders honest error / budget_exhausted banners without misleading success state', async () => {
    const partialPayload = {
      mode: 'advanced_pipeline',
      effectiveMode: 'advanced_pipeline',
      status: 'budget_exhausted',
      stopReason: 'budget_exhausted',
      conversation: [{ role: 'user', name: 'User', content: 'Huge proof' }],
      questions: [],
      finalOutput: '',
      errorMessage: 'Step budget exhausted before full convergence.',
      outputType: { outputType: 'Simple Concise Answer', fileExtension: '.txt' },
      artifacts: [],
      diagnostics: { planSteps: [], reflections: [], agentAttributions: [] },
      events: [],
    };
    mockRunExecutionFn.mockResolvedValueOnce(partialPayload);

    renderConversationPage();
    const modeSelect = screen.getByLabelText('Execution Mode:') as HTMLSelectElement;
    await userEvent.selectOptions(modeSelect, 'advanced_pipeline');
    await userEvent.click(screen.getByRole('button', { name: /Start Pipeline Session/i }));

    const textarea = screen.getByPlaceholderText('Type your message...');
    await userEvent.type(textarea, 'Huge proof');
    await userEvent.click(screen.getByTestId('send-message-button'));

    expect(await screen.findByText(/Step Budget Exhausted \(Partial Result\)/i)).toBeInTheDocument();
    expect(screen.getByText(/Step budget exhausted before full convergence./i)).toBeInTheDocument();
  });
});
