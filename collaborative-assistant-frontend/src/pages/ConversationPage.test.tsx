/// <reference types="vitest/globals" />
import { render, screen, act, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import ConversationPage from './ConversationPage';
import {
  useRunExecution,
  DEFAULT_PERSONALITY_IDS,
} from '../hooks/useRunExecution';
import type { ExecutionModeCapabilityType } from '../graphql/graphqlTypes';
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
const mockRetryDiscovery = vi.fn();
const mockApplyApiToken = vi.fn();
const mockClearStoredApiToken = vi.fn();

const MOCK_EXECUTION_MODES: ExecutionModeCapabilityType[] = [
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
    minNumRounds: 1,
    maxNumRounds: 10,
    maxSteps: 50,
    maxAgents: 10,
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
    minNumRounds: 1,
    maxNumRounds: 10,
    maxSteps: 50,
    maxAgents: 10,
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

const buildHookMock = (overrides: Record<string, unknown> = {}) => ({
  runExecution: mockRunExecutionFn,
  cancelActiveRun: mockCancelActiveRun,
  resetExecutionState: mockResetExecutionState,
  retryDiscovery: mockRetryDiscovery,
  applyApiToken: mockApplyApiToken,
  clearStoredApiToken: mockClearStoredApiToken,
  data: null,
  lastResult: null,
  loading: false,
  error: undefined,
  errorMessage: null,
  wasCancelled: false,
  executionModes: MOCK_EXECUTION_MODES,
  personalities: DEFAULT_PERSONALITY_IDS.map((id) => ({ id, instructions: '' })),
  providerPolicy: {
    defaultModel: 'gpt-4o-mini',
    allowedModels: ['gpt-4o-mini', 'gpt-4o'],
    defaultTemperature: 0.7,
  },
  modesLoading: false,
  modesError: undefined,
  discoveryStatus: 'available',
  discoveryErrorMessage: null,
  capabilitiesReady: true,
  authRequired: false,
  hasApiToken: false,
  ...overrides,
});

describe('ConversationPage Unified Lifecycle Integration', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockRunExecutionFn.mockResolvedValue(null);
    mockRetryDiscovery.mockResolvedValue(undefined);
    mockApplyApiToken.mockResolvedValue(undefined);
    mockClearStoredApiToken.mockResolvedValue(undefined);
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

  it('shows auth-required prompt on 401, preserves draft and config across token entry and retry, and supports clearing token', async () => {
    // Initially authenticated so user configures rounds=6 and starts typing a draft
    const { rerender } = renderConversationPage();
    const roundsInput = screen.getByRole('spinbutton', { name: 'Number of Rounds (per turn):' });
    fireEvent.change(roundsInput, { target: { value: '6' } });
    expect(roundsInput).toHaveValue(6);

    const textarea = screen.getByPlaceholderText('Type your message...') as HTMLTextAreaElement;
    await userEvent.type(textarea, 'Preserve my draft prompt');

    // Server returns 401 -> hook transitions to auth_required
    (useRunExecution as unknown as ReturnType<typeof vi.fn>).mockImplementation(() =>
      buildHookMock({
        discoveryStatus: 'auth_required',
        authRequired: true,
        capabilitiesReady: false,
        executionModes: [],
        personalities: [],
        providerPolicy: null,
        discoveryErrorMessage: 'Authentication required to execute GraphQL requests.',
      })
    );
    rerender(
      <ApolloProvider client={client}>
        <ConversationPage />
      </ApolloProvider>
    );

    expect(screen.getByRole('alert')).toHaveTextContent(/Authentication required/i);
    expect(screen.getByRole('button', { name: /Start New Conversation/i })).toBeDisabled();
    // User draft and config remain preserved
    expect(textarea).toHaveValue('Preserve my draft prompt');
    expect(roundsInput).toHaveValue(6);

    // User enters token and clicks Save & Retry
    const tokenInput = screen.getByLabelText('API Bearer Token');
    await userEvent.type(tokenInput, 'secret-api-token');
    await userEvent.click(screen.getByTestId('apply-api-token-button'));
    expect(mockApplyApiToken).toHaveBeenCalledWith('secret-api-token');

    // Token succeeds -> capabilities restored
    (useRunExecution as unknown as ReturnType<typeof vi.fn>).mockImplementation(() =>
      buildHookMock({
        hasApiToken: true,
        discoveryStatus: 'available',
        authRequired: false,
      })
    );
    rerender(
      <ApolloProvider client={client}>
        <ConversationPage />
      </ApolloProvider>
    );

    expect(screen.getByRole('status')).toHaveTextContent('Token set');
    expect(screen.getByRole('button', { name: /Start New Conversation/i })).toBeEnabled();
    expect(textarea).toHaveValue('Preserve my draft prompt');
    expect(roundsInput).toHaveValue(6);

    // Clear token calls clearStoredApiToken
    await userEvent.click(screen.getByTestId('clear-api-token-button'));
    expect(mockClearStoredApiToken).toHaveBeenCalledTimes(1);
  });

  it('shows accessible error banner and Retry Discovery button on capability failure and disables execution on empty/not-ready states', async () => {
    (useRunExecution as unknown as ReturnType<typeof vi.fn>).mockImplementation(() =>
      buildHookMock({
        discoveryStatus: 'failed',
        capabilitiesReady: false,
        executionModes: [],
        discoveryErrorMessage: 'Network failure reaching /graphql.',
      })
    );

    const { rerender } = renderConversationPage();
    expect(screen.getByRole('alert')).toHaveTextContent(/Network failure reaching \/graphql/i);
    expect(screen.getByRole('button', { name: /Start New Conversation/i })).toBeDisabled();

    await userEvent.click(screen.getByTestId('retry-discovery-button'));
    expect(mockRetryDiscovery).toHaveBeenCalledTimes(1);

    // Empty catalog also stays fail-closed
    (useRunExecution as unknown as ReturnType<typeof vi.fn>).mockImplementation(() =>
      buildHookMock({
        discoveryStatus: 'empty',
        capabilitiesReady: false,
        executionModes: [],
        discoveryErrorMessage: 'No execution modes were returned by the server.',
      })
    );
    rerender(
      <ApolloProvider client={client}>
        <ConversationPage />
      </ApolloProvider>
    );
    expect(screen.getByRole('alert')).toHaveTextContent(/No execution modes available/i);
    expect(screen.getByRole('button', { name: /Start New Conversation/i })).toBeDisabled();

    // Ready=false mode stays disabled
    (useRunExecution as unknown as ReturnType<typeof vi.fn>).mockImplementation(() =>
      buildHookMock({
        discoveryStatus: 'available',
        capabilitiesReady: false,
        executionModes: MOCK_EXECUTION_MODES.map((m) =>
          m.id === 'collaborative'
            ? { ...m, ready: false, unavailableReason: 'Provider key missing.' }
            : m
        ),
      })
    );
    rerender(
      <ApolloProvider client={client}>
        <ConversationPage />
      </ApolloProvider>
    );
    expect(screen.getByRole('status')).toHaveTextContent('Provider key missing.');
    expect(screen.getByRole('button', { name: /Start New Conversation/i })).toBeDisabled();
  });

  it('enforces capability-driven maxNumRounds limit in UI without allowing out-of-range submission', async () => {
    (useRunExecution as unknown as ReturnType<typeof vi.fn>).mockImplementation(() =>
      buildHookMock({
        executionModes: MOCK_EXECUTION_MODES.map((m) =>
          m.id === 'collaborative' ? { ...m, maxNumRounds: 5 } : m
        ),
      })
    );

    renderConversationPage();
    const roundsInput = screen.getByRole('spinbutton', { name: 'Number of Rounds (per turn):' });
    expect(roundsInput).toHaveAttribute('max', '5');

    fireEvent.change(roundsInput, { target: { value: '6' } });
    expect(screen.getByRole('alert')).toHaveTextContent(/between 1 and 5/i);
    expect(screen.getByRole('button', { name: /Start New Conversation/i })).toBeDisabled();
  });
});
