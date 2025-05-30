/// <reference types="vitest/globals" />
import { render, screen, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import ConversationPage from './ConversationPage';
import { useRunConversation } from '../hooks/useRunConversation';
import { vi, describe, it, expect, beforeEach } from 'vitest';
import { ApolloProvider } from '@apollo/client';
import { client } from '../graphql/apolloClient';
import toast from 'react-hot-toast';

vi.mock('../hooks/useRunConversation');

const specificToastErrorMessage = "Please start a new conversation first.";
vi.mock('react-hot-toast', () => ({
  __esModule: true,
  default: {
    success: vi.fn(),
    error: vi.fn((message) => {
      if (message === specificToastErrorMessage) {
        throw new Error(`toast.error was called with: ${specificToastErrorMessage}`);
      }
    }),
  },
  Toaster: () => <div data-testid="toaster" />,
}));

const mockRunConversationFn = vi.fn();

describe('ConversationPage Integration Test', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    // Reset mock function to a default implementation for each test
    mockRunConversationFn.mockResolvedValue({});
    (useRunConversation as vi.Mock).mockImplementation(() => ({
      runConversation: mockRunConversationFn,
      data: null,
      loading: false,
      error: null,
    }));
  });

  const renderConversationPage = () => {
    return render(
      <ApolloProvider client={client}>
        <ConversationPage />
      </ApolloProvider>
    );
  };

  it('allows typing a message and sending it, displaying user message', async () => {
    // Initial state is set by beforeEach: data: null, loading: false, error: null
    renderConversationPage();
    const startButton = screen.getByRole('button', { name: /Start New Conversation/i });
    await userEvent.click(startButton);
    expect(await screen.findByText(/Conversation started with/i)).toBeInTheDocument();

    const textarea = screen.getByPlaceholderText('Type your message...') as HTMLTextAreaElement;
    await userEvent.type(textarea, 'Hello Assistant');
    const sendButton = screen.getByTestId('send-message-button');
    expect(sendButton).toBeEnabled();

    await userEvent.click(sendButton); // This will use the mock from beforeEach initially

    expect(screen.getByText('Hello Assistant')).toBeInTheDocument(); // Optimistic update
    expect(mockRunConversationFn).toHaveBeenCalledTimes(1);

    const calledWithArgs = mockRunConversationFn.mock.calls[0][0];
    expect(calledWithArgs.conversation).toEqual(
        expect.arrayContaining([
            expect.objectContaining({ role: 'user', content: 'Hello Assistant' })
        ])
    );
  });

  it('displays assistant response, questions, and final output when data is returned', async () => {
    const { rerender } = renderConversationPage(); // Get rerender function
    const startButton = screen.getByRole('button', { name: /Start New Conversation/i });
    await userEvent.click(startButton);
    expect(await screen.findByText(/Conversation started with/i)).toBeInTheDocument();

    const textarea = screen.getByPlaceholderText('Type your message...') as HTMLTextAreaElement;
    await userEvent.type(textarea, 'User message');
    const sendButton = screen.getByTestId('send-message-button');
    expect(sendButton).toBeEnabled();

    // Simulate the hook reacting to the call and going into a loading state
    await act(async () => {
      (useRunConversation as vi.Mock).mockImplementation(() => ({
        runConversation: mockRunConversationFn,
        data: null, loading: true, error: null,
      }));
      await userEvent.click(sendButton); // This calls mockRunConversationFn
      // Rerender to apply the new mock state
      rerender(
        <ApolloProvider client={client}>
          <ConversationPage />
        </ApolloProvider>
      );
      await Promise.resolve();
    });

    expect(mockRunConversationFn).toHaveBeenCalledTimes(1);
    screen.debug(document.body, 300000); // DEBUG: Check DOM when loading
    expect(await screen.findByText('Assistant is typing...')).toBeInTheDocument();

    const responseData = {
      runConversation: {
        conversation: [
          // Ensure names are not strictly required by MessageBubble if not always present
          { role: 'system', content: 'Conversation started...' },
          { role: 'user', content: 'User message' },
          { role: 'assistant', content: 'Assistant response' },
        ],
        questions: [{ assistant: 'Helpful Assistant', question: 'Was this helpful?' }],
        finalOutput: 'This is the final output.',
      },
    };

    // Simulate data being returned from the hook
    await act(async () => {
      (useRunConversation as vi.Mock).mockImplementation(() => ({
        runConversation: mockRunConversationFn,
        data: responseData,
        loading: false,
        error: null,
      }));
      // Rerender to apply the new mock state with data
      rerender(
        <ApolloProvider client={client}>
          <ConversationPage />
        </ApolloProvider>
      );
      await Promise.resolve();
    });

    screen.debug(document.body, 300000); // DEBUG: Check DOM when data is present
    expect(await screen.findByText('Assistant response')).toBeInTheDocument();
    expect(await screen.findByText('Was this helpful?')).toBeInTheDocument();
    expect(await screen.findByText('This is the final output.')).toBeInTheDocument();
    expect(screen.queryByText('Assistant is typing...')).not.toBeInTheDocument();
  });

  it('shows loading indicator when runConversationLoading is true', async () => {
    const { rerender } = renderConversationPage(); // Initial mock: loading: false
    const startButton = screen.getByRole('button', { name: /Start New Conversation/i });
    await userEvent.click(startButton);
    await screen.findByText(/Conversation started with/i);

    // Simulate the hook now being in a loading state
    await act(async () => {
      (useRunConversation as vi.Mock).mockImplementation(() => ({
        runConversation: mockRunConversationFn,
        data: null, loading: true, error: null,
      }));
      rerender(
        <ApolloProvider client={client}>
          <ConversationPage />
        </ApolloProvider>
      );
      await Promise.resolve();
    });
    // screen.debug(document.body, 300000); // DEBUG: Check DOM for loading indicator
    expect(await screen.findByText('Assistant is typing...')).toBeInTheDocument();
  });

  it('displays error message when runConversationError is present', async () => {
    const { rerender } = renderConversationPage(); // Initial mock: loading: false, error: null
    const startButton = screen.getByRole('button', { name: /Start New Conversation/i });
    await userEvent.click(startButton);
    await screen.findByText(/Conversation started with/i);

    const errorMessage = 'Network Error';
    // Simulate the hook now having an error
    await act(async () => {
      (useRunConversation as vi.Mock).mockImplementation(() => ({
        runConversation: mockRunConversationFn,
        data: null, loading: false, error: new Error(errorMessage),
      }));
      rerender(
        <ApolloProvider client={client}>
          <ConversationPage />
        </ApolloProvider>
      );
      await Promise.resolve();
    });

    screen.debug(document.body, 300000); // DEBUG: Check DOM for error message
    const strongErrorElement = await screen.findByText((content, element) => {
      return element?.tagName.toLowerCase() === 'strong' && content.startsWith('Error:');
    });
    expect(strongErrorElement).toBeInTheDocument();

    const errorContainer = strongErrorElement.parentElement;
    expect(errorContainer).toHaveTextContent(/Network Error. Please try again or adjust settings./i);
    expect(errorContainer).toHaveClass('text-red-700');
  });

  it('handles null data from useRunConversation gracefully', async () => {
    const { rerender } = renderConversationPage();
    const startButton = screen.getByRole('button', { name: /Start New Conversation/i });
    await userEvent.click(startButton);
    await screen.findByText(/Conversation started with/i);

    const textarea = screen.getByPlaceholderText('Type your message...') as HTMLTextAreaElement;
    await userEvent.type(textarea, 'Test message');
    const sendButton = screen.getByTestId('send-message-button');

    // Simulate loading on click
    await act(async () => {
      (useRunConversation as vi.Mock).mockImplementation(() => ({
        runConversation: mockRunConversationFn,
        data: null, loading: true, error: null,
      }));
      await userEvent.click(sendButton);
      rerender(
        <ApolloProvider client={client}>
          <ConversationPage />
        </ApolloProvider>
      );
      await Promise.resolve();
    });
    expect(mockRunConversationFn).toHaveBeenCalledTimes(1);
    // screen.debug(document.body, 300000); // DEBUG: Check loading state
    expect(await screen.findByText('Assistant is typing...')).toBeInTheDocument();

    // Simulate data coming back as null, loading false
    await act(async () => {
      (useRunConversation as vi.Mock).mockImplementation(() => ({
        runConversation: mockRunConversationFn,
        data: null, loading: false, error: null,
      }));
      rerender(
        <ApolloProvider client={client}>
          <ConversationPage />
        </ApolloProvider>
      );
      await Promise.resolve();
    });
    // screen.debug(document.body, 300000); // DEBUG: Check DOM after null data
    expect(screen.queryByText('Assistant is typing...')).not.toBeInTheDocument();
    expect(screen.queryByText('Assistant response')).not.toBeInTheDocument();
    expect(screen.queryByText('Was this helpful?')).not.toBeInTheDocument();
    expect(screen.queryByText('This is the final output.')).not.toBeInTheDocument();
  });

  it('handles runConversation result as null gracefully', async () => {
    renderConversationPage();
    const startButton = screen.getByRole('button', { name: /Start New Conversation/i });
    await userEvent.click(startButton);
    expect(await screen.findByText(/Conversation started with/i)).toBeInTheDocument();

    const textarea = screen.getByPlaceholderText('Type your message...') as HTMLTextAreaElement;
    await userEvent.type(textarea, 'Test message');
    const sendButton = screen.getByTestId('send-message-button');

    // Simulate data directly after click
    (useRunConversation as vi.Mock).mockImplementation(() => ({
      runConversation: mockRunConversationFn.mockResolvedValueOnce({ data: { runConversation: null } }),
      data: { runConversation: null }, loading: false, error: null,
    }));
    await userEvent.click(sendButton);

    expect(mockRunConversationFn).toHaveBeenCalledTimes(1);
    expect(screen.queryByText('Assistant response')).not.toBeInTheDocument();
    expect(screen.queryByText('Was this helpful?')).not.toBeInTheDocument();
    expect(screen.queryByText('This is the final output.')).not.toBeInTheDocument();
    expect(screen.queryByText(/Error:/i)).not.toBeInTheDocument();
  });

  it('handles missing fields in runConversation response gracefully', async () => {
    const { rerender } = renderConversationPage();
    const startButton = screen.getByRole('button', { name: /Start New Conversation/i });
    await userEvent.click(startButton);
    expect(await screen.findByText(/Conversation started with/i)).toBeInTheDocument();

    const textarea = screen.getByPlaceholderText('Type your message...') as HTMLTextAreaElement;
    await userEvent.type(textarea, 'Test message for partial data');
    const sendButton = screen.getByTestId('send-message-button');

    // Simulate loading on click
    await act(async () => {
      (useRunConversation as vi.Mock).mockImplementation(() => ({
        runConversation: mockRunConversationFn,
        data: null, loading: true, error: null,
      }));
      await userEvent.click(sendButton);
      rerender(
        <ApolloProvider client={client}>
          <ConversationPage />
        </ApolloProvider>
      );
      await Promise.resolve();
    });
    expect(mockRunConversationFn).toHaveBeenCalledTimes(1);
    // screen.debug(document.body, 300000); // DEBUG: Check loading state
    expect(await screen.findByText('Assistant is typing...')).toBeInTheDocument();

    // Test case 1: Only conversation is present
    const partialData1 = { runConversation: { conversation: [{ role: 'assistant', content: 'Only conversation here' }]}};
    await act(async () => {
      (useRunConversation as vi.Mock).mockImplementation(() => ({
        runConversation: mockRunConversationFn,
        data: partialData1, loading: false, error: null,
      }));
      rerender(
        <ApolloProvider client={client}>
          <ConversationPage />
        </ApolloProvider>
      );
      await Promise.resolve();
    });
    // screen.debug(document.body, 300000); // DEBUG: Check partial data 1
    expect(await screen.findByText('Only conversation here')).toBeInTheDocument();
    expect(screen.queryByText('Was this helpful?')).not.toBeInTheDocument();
    expect(screen.queryByText('This is the final output.')).not.toBeInTheDocument();
    expect(screen.queryByText('Assistant is typing...')).not.toBeInTheDocument();

    // Test case 2: Only questions are present
    await act(async () => {
      (useRunConversation as vi.Mock).mockImplementation(() => ({
        runConversation: mockRunConversationFn,
        data: null, loading: true, error: null, // Set loading again
      }));
      rerender(
        <ApolloProvider client={client}>
          <ConversationPage />
        </ApolloProvider>
      );
      await Promise.resolve();
    });
    // screen.debug(document.body, 300000); // DEBUG: Check loading for partial data 2
    expect(await screen.findByText('Assistant is typing...')).toBeInTheDocument();

    const partialData2 = { runConversation: { conversation: [], questions: [{ assistant: 'Assistant', question: 'Only a question here' }]}};
    await act(async () => {
      (useRunConversation as vi.Mock).mockImplementation(() => ({
        runConversation: mockRunConversationFn,
        data: partialData2, loading: false, error: null,
      }));
      rerender(
        <ApolloProvider client={client}>
          <ConversationPage />
        </ApolloProvider>
      );
      await Promise.resolve();
    });
    // screen.debug(document.body, 300000); // DEBUG: Check partial data 2
    expect(await screen.findByText('Only a question here')).toBeInTheDocument();
    expect(screen.queryByText('Only conversation here')).not.toBeInTheDocument();
    expect(screen.queryByText('This is the final output.')).not.toBeInTheDocument();
    expect(screen.queryByText('Assistant is typing...')).not.toBeInTheDocument();

    // Test case 3: Only finalOutput is present
    await act(async () => {
      (useRunConversation as vi.Mock).mockImplementation(() => ({
        runConversation: mockRunConversationFn,
        data: null, loading: true, error: null, // Set loading again
      }));
      rerender(
        <ApolloProvider client={client}>
          <ConversationPage />
        </ApolloProvider>
      );
      await Promise.resolve();
    });
    // screen.debug(document.body, 300000); // DEBUG: Check loading for partial data 3
    expect(await screen.findByText('Assistant is typing...')).toBeInTheDocument();

    const partialData3 = { runConversation: { conversation: [], finalOutput: 'Only final output here' }};
    await act(async () => {
      (useRunConversation as vi.Mock).mockImplementation(() => ({
        runConversation: mockRunConversationFn,
        data: partialData3, loading: false, error: null,
      }));
      rerender(
        <ApolloProvider client={client}>
          <ConversationPage />
        </ApolloProvider>
      );
      await Promise.resolve();
    });
    // screen.debug(document.body, 300000); // DEBUG: Check partial data 3
    expect(await screen.findByText('Only final output here')).toBeInTheDocument();
    expect(screen.queryByText('Only a question here')).not.toBeInTheDocument();
    expect(screen.queryByText('Only conversation here')).not.toBeInTheDocument();
    expect(screen.queryByText('Assistant is typing...')).not.toBeInTheDocument();
  });
});
