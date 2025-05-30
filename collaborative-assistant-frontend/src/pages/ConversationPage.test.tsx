/// <reference types="vitest/globals" />
import { render, screen, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import ConversationPage from './ConversationPage'; // Changed to default import
import { useRunConversation } from '../hooks/useRunConversation';
import { vi, describe, it, expect, beforeEach } from 'vitest';
import { ApolloProvider } from '@apollo/client'; // Import ApolloProvider
import { client } from '../graphql/apolloClient'; // Import your Apollo client instance

// Mock the custom hook
vi.mock('../hooks/useRunConversation');

// Mock react-hot-toast
vi.mock('react-hot-toast', () => ({
  __esModule: true,
  default: {
    success: vi.fn(),
    error: vi.fn(),
  },
  Toaster: () => <div data-testid="toaster" />, // Mock Toaster component
}));


const mockRunConversationFn = vi.fn();
let mockLoading = false;
let mockError: Error | null = null;
let mockData: any = null;

// Define a mutable object for the hook's state
let currentHookState: { loading: boolean; error: Error | null; data: any };

describe('ConversationPage Integration Test', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    // Reset the hook's state before each test
    currentHookState = {
      loading: false,
      error: null,
      data: null,
    };
    (useRunConversation as vi.Mock).mockImplementation(() => [
      mockRunConversationFn, // The function to trigger the run
      currentHookState,      // The reactive state of the hook
    ]);
  });

  const renderConversationPage = () => {
    // Wrap with ApolloProvider if your components expect it, even with mocked hooks
    return render(
      <ApolloProvider client={client}>
        <ConversationPage />
      </ApolloProvider>
    );
  };


  it('allows typing a message and sending it, displaying user message', async () => {
    renderConversationPage();

    // Start a conversation
    const startButton = screen.getByRole('button', { name: /Start New Conversation/i });
    await userEvent.click(startButton);

    const textarea = screen.getByPlaceholderText('Type your message...') as HTMLTextAreaElement;
    await userEvent.type(textarea, 'Hello Assistant');

    const sendButton = screen.getByTestId('send-message-button');
    await userEvent.click(sendButton);

    expect(screen.getByText('Hello Assistant')).toBeInTheDocument();
    expect(mockRunConversationFn).toHaveBeenCalledTimes(1);
    // Check some basic args if needed, e.g., if conversation history includes the new message
    const calledWithArgs = mockRunConversationFn.mock.calls[0][0].variables;
    expect(calledWithArgs.conversation).toEqual(
        expect.arrayContaining([
            expect.objectContaining({ role: 'user', content: 'Hello Assistant' })
        ])
    );
  });

  it('displays assistant response, questions, and final output when data is returned', async () => {
    renderConversationPage();

    // Start a conversation
    const startButton = screen.getByRole('button', { name: /Start New Conversation/i });
    await userEvent.click(startButton);

    const textarea = screen.getByPlaceholderText('Type your message...') as HTMLTextAreaElement;
    await userEvent.type(textarea, 'User message');
    const sendButton = screen.getByTestId('send-message-button');

    // Simulate the action of sending a message
    // mockRunConversationFn will be called by the component
    await userEvent.click(sendButton);
    expect(mockRunConversationFn).toHaveBeenCalledTimes(1);

    // Simulate the hook updating to a loading state
    act(() => {
      currentHookState.loading = true;
    });

    // Check for loading indicator (optional, but good for completeness)
    // Assuming 'Assistant is typing...' is your loading text
    expect(screen.getByText('Assistant is typing...')).toBeInTheDocument();

    // Simulate the hook receiving data and updating its state
    const responseData = {
      runConversation: {
        conversation: [
          { role: 'system', name: 'System', content: 'Conversation started...' },
          { role: 'user', name: 'User', content: 'User message' },
          { role: 'assistant', name: 'Helpful Assistant', content: 'Assistant response' },
        ],
        questions: [{ assistant: 'Helpful Assistant', question: 'Was this helpful?' }],
        finalOutput: 'This is the final output.',
      },
    };

    act(() => {
      currentHookState.loading = false;
      currentHookState.data = responseData;
    });

    // Assert that the UI updates with the new data
    // findBy* queries are good for asynchronous updates as they wait for elements to appear.
    expect(await screen.findByText('Assistant response')).toBeInTheDocument();
    expect(await screen.findByText('Was this helpful?')).toBeInTheDocument();
    expect(await screen.findByText('This is the final output.')).toBeInTheDocument();
    // Ensure loading indicator is gone
    expect(screen.queryByText('Assistant is typing...')).not.toBeInTheDocument();
  });

  it('shows loading indicator when runConversationLoading is true', async () => {
    renderConversationPage();
    const startButton = screen.getByRole('button', { name: /Start New Conversation/i });
    await userEvent.click(startButton); // Start conversation

    act(() => {
      currentHookState.loading = true;
    });

    expect(screen.getByText('Assistant is typing...')).toBeInTheDocument();
  });

  it('displays error message when runConversationError is present', async () => {
    renderConversationPage();
    const startButton = screen.getByRole('button', { name: /Start New Conversation/i });
    await userEvent.click(startButton); // Start conversation

    const errorMessage = 'Network Error';
    act(() => {
      currentHookState.error = new Error(errorMessage);
    });

    expect(await screen.findByText((content, element) => {
      const hasText = (node: Element | null): boolean => node?.textContent?.includes(`Error: ${errorMessage}`) || false;
      return hasText(element) && (element?.classList.contains('text-red-700') || false);
    })).toBeInTheDocument();
  });

  it('handles null data from useRunConversation gracefully', async () => {
    renderConversationPage();
    const startButton = screen.getByRole('button', { name: /Start New Conversation/i });
    await userEvent.click(startButton);
    const textarea = screen.getByPlaceholderText('Type your message...') as HTMLTextAreaElement;
    await userEvent.type(textarea, 'Test message');
    const sendButton = screen.getByTestId('send-message-button');
    await userEvent.click(sendButton);

    act(() => {
      currentHookState.loading = true;
    });
    expect(screen.getByText('Assistant is typing...')).toBeInTheDocument();

    act(() => {
      currentHookState.loading = false;
      currentHookState.data = null; // Simulate null data response
    });

    expect(screen.queryByText('Assistant is typing...')).not.toBeInTheDocument();
    // Depending on implementation, either nothing is shown or a specific message.
    // For now, let's assume no specific message for "null data" itself,
    // and the component should not crash.
    // We check that elements expected from a valid response are NOT present.
    expect(screen.queryByText('Assistant response')).not.toBeInTheDocument();
    expect(screen.queryByText('Was this helpful?')).not.toBeInTheDocument();
    expect(screen.queryByText('This is the final output.')).not.toBeInTheDocument();
  });

  it('handles runConversation result as null gracefully', async () => {
    renderConversationPage();
    const startButton = screen.getByRole('button', { name: /Start New Conversation/i });
    await userEvent.click(startButton);
    const textarea = screen.getByPlaceholderText('Type your message...') as HTMLTextAreaElement;
    await userEvent.type(textarea, 'Test message');
    const sendButton = screen.getByTestId('send-message-button');
    await userEvent.click(sendButton);

    act(() => {
      currentHookState.loading = false;
      currentHookState.data = { runConversation: null }; // Simulate runConversation being null
    });

    expect(screen.queryByText('Assistant response')).not.toBeInTheDocument();
    expect(screen.queryByText('Was this helpful?')).not.toBeInTheDocument();
    expect(screen.queryByText('This is the final output.')).not.toBeInTheDocument();
    // It's good practice to also check that no error message is displayed accidentally
    expect(screen.queryByText(/Error:/i)).not.toBeInTheDocument();
  });

  it('handles missing fields in runConversation response gracefully', async () => {
    renderConversationPage();
    const startButton = screen.getByRole('button', { name: /Start New Conversation/i });
    await userEvent.click(startButton);
    const textarea = screen.getByPlaceholderText('Type your message...') as HTMLTextAreaElement;
    await userEvent.type(textarea, 'Test message for partial data');
    const sendButton = screen.getByTestId('send-message-button');
    await userEvent.click(sendButton);

    act(() => {
      currentHookState.loading = true;
    });
     expect(screen.getByText('Assistant is typing...')).toBeInTheDocument();


    // Test case 1: Only conversation is present
    act(() => {
      currentHookState.loading = false;
      currentHookState.data = {
        runConversation: {
          conversation: [{ role: 'assistant', name: 'Assistant', content: 'Only conversation here' }],
          // questions and finalOutput are missing
        },
      };
    });

    expect(await screen.findByText('Only conversation here')).toBeInTheDocument();
    expect(screen.queryByText('Was this helpful?')).not.toBeInTheDocument(); // Example question text
    expect(screen.queryByText('This is the final output.')).not.toBeInTheDocument(); // Example final output text
    expect(screen.queryByText('Assistant is typing...')).not.toBeInTheDocument();

    // Test case 2: Only questions are present
    act(() => {
      currentHookState.loading = true; // Reset to loading before new data
    });
    expect(screen.getByText('Assistant is typing...')).toBeInTheDocument();

    act(() => {
      currentHookState.loading = false;
      currentHookState.data = {
        runConversation: {
          questions: [{ assistant: 'Assistant', question: 'Only a question here' }],
          // conversation and finalOutput are missing
        },
      };
    });
    expect(await screen.findByText('Only a question here')).toBeInTheDocument();
    // Check that previous data is cleared or not displayed if component logic dictates so
    expect(screen.queryByText('Only conversation here')).not.toBeInTheDocument();
    expect(screen.queryByText('This is the final output.')).not.toBeInTheDocument();
    expect(screen.queryByText('Assistant is typing...')).not.toBeInTheDocument();

    // Test case 3: Only finalOutput is present
    act(() => {
      currentHookState.loading = true; // Reset to loading before new data
    });
    expect(screen.getByText('Assistant is typing...')).toBeInTheDocument();

    act(() => {
      currentHookState.loading = false;
      currentHookState.data = {
        runConversation: {
          finalOutput: 'Only final output here',
          // conversation and questions are missing
        },
      };
    });
    expect(await screen.findByText('Only final output here')).toBeInTheDocument();
    expect(screen.queryByText('Only a question here')).not.toBeInTheDocument();
    expect(screen.queryByText('Only conversation here')).not.toBeInTheDocument();
    expect(screen.queryByText('Assistant is typing...')).not.toBeInTheDocument();
  });
});
