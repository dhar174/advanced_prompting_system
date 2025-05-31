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

// Mock react-hot-toast
const mockToastError = vi.fn();
const mockToastSuccess = vi.fn();
vi.mock('react-hot-toast', () => ({
  __esModule: true,
  default: {
    success: mockToastSuccess,
    error: mockToastError,
  },
  Toaster: () => <div data-testid="toaster" />, // Optional: if you want to assert Toaster presence
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
    expect(await screen.findByText('Assistant is typing...')).toBeInTheDocument();

    const responseData = {
      runConversation: {
        conversation: [
          { role: 'system', content: 'Conversation started...' }, // Keep system message for context
          { role: 'user', content: 'User message' }, // Keep user message for context
          { role: 'assistant', content: 'Assistant response' },
        ],
        questions: [{ assistant: 'Helpful Assistant', question: 'Was this helpful?' }],
        finalOutput: 'This is the final output.',
      },
    };

    // Simulate data being returned from the hook
    await act(async () => {
      (useRunConversation as vi.Mock).mockImplementation(() => ({
        runConversation: mockRunConversationFn, // This function was already called
        data: responseData, // Provide the new data
        loading: false, // Set loading to false
        error: null,
      }));
      // Rerender to apply the new mock state with data
      rerender(
        <ApolloProvider client={client}>
          <ConversationPage />
        </ApolloProvider>
      );
      await Promise.resolve(); // Allow promises to resolve, e.g., state updates
    });

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
    expect(await screen.findByText('Assistant is typing...')).toBeInTheDocument();
  });

  it('displays error message when runConversationError is present', async () => {
    const { rerender } = renderConversationPage();
    const startButton = screen.getByRole('button', { name: /Start New Conversation/i });
    await userEvent.click(startButton);
    await screen.findByText(/Conversation started with/i);

    const errorMessage = 'Network Error';
    const errorObj = new Error(errorMessage); // Create an actual error object

    // Simulate the hook now having an error
    await act(async () => {
      (useRunConversation as vi.Mock).mockImplementation(() => ({
        runConversation: mockRunConversationFn,
        data: null,
        loading: false,
        error: errorObj, // Use the error object
      }));
      rerender(
        <ApolloProvider client={client}>
          <ConversationPage />
        </ApolloProvider>
      );
      await Promise.resolve();
    });

    // Check for error message in ConversationDisplay
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
    expect(await screen.findByText('Assistant is typing...')).toBeInTheDocument();

    // Simulate data coming back as null, loading false
    await act(async () => {
      (useRunConversation as vi.Mock).mockImplementation(() => ({
        runConversation: mockRunConversationFn,
        data: null, // Data is null
        loading: false, // Loading finished
        error: null,
      }));
      rerender(
        <ApolloProvider client={client}>
          <ConversationPage />
        </ApolloProvider>
      );
      await Promise.resolve();
    });
    expect(screen.queryByText('Assistant is typing...')).not.toBeInTheDocument();
    // Check that no new content is displayed if data.runConversation is null
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
    expect(await screen.findByText('Assistant is typing...')).toBeInTheDocument();

    // Test case 2: Only questions are present
    const partialData2 = { runConversation: { conversation: [], questions: [{ assistant: 'Assistant', question: 'Only a question here' }]}};
    await act(async () => {
      (useRunConversation as vi.Mock).mockImplementation(() => ({
        runConversation: mockRunConversationFn,
        data: partialData2, // Provide partial data
        loading: false, // Set loading to false
        error: null,
      }));
      rerender(
        <ApolloProvider client={client}>
          <ConversationPage />
        </ApolloProvider>
      );
      await Promise.resolve();
    });
    expect(await screen.findByText('Only a question here')).toBeInTheDocument();
    // Ensure other data is not present if not returned
    expect(screen.queryByText('Only conversation here')).not.toBeInTheDocument(); // From previous partial data test
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
    expect(await screen.findByText('Assistant is typing...')).toBeInTheDocument();

    // Test case 3: Only finalOutput is present
    const partialData3 = { runConversation: { conversation: [], finalOutput: 'Only final output here' }};
    await act(async () => {
      (useRunConversation as vi.Mock).mockImplementation(() => ({
        runConversation: mockRunConversationFn,
        data: partialData3, // Provide partial data
        loading: false, // Set loading to false
        error: null,
      }));
      rerender(
        <ApolloProvider client={client}>
          <ConversationPage />
        </ApolloProvider>
      );
      await Promise.resolve();
    });
    expect(await screen.findByText('Only final output here')).toBeInTheDocument();
    // Ensure other data is not present
    expect(screen.queryByText('Only a question here')).not.toBeInTheDocument(); // From previous partial data test
    expect(screen.queryByText('Only conversation here')).not.toBeInTheDocument(); // From previous partial data test
    expect(screen.queryByText('Assistant is typing...')).not.toBeInTheDocument();
  });

  it('renders initial components correctly', () => {
    renderConversationPage();
    // Check for AssistantConfigPanel elements
    expect(screen.getByText('Configure Assistants')).toBeInTheDocument(); // CardTitle in AssistantConfigPanel
    expect(screen.getByRole('button', { name: /Start New Conversation/i })).toBeInTheDocument();

    // Check for initial message in ConversationDisplay (rendered by ConversationPage initially)
    expect(screen.getByText(/Welcome! Configure your assistants on the left and start the conversation./i)).toBeInTheDocument();

    // Check for ChatInput
    expect(screen.getByPlaceholderText('Type your message...')).toBeInTheDocument();
    expect(screen.getByTestId('send-message-button')).toBeInTheDocument();

    // FeedbackDisplay is initially empty, so no specific text to check unless we add a placeholder
    // For now, ensuring no error and the page structure is sound is enough for initial render.
  });

  it('updates configuration and uses it in runConversation', async () => {
    renderConversationPage();

    // Simulate changing number of rounds (example of config change)
    // AssistantConfigPanel has a label "Number of Rounds" associated with its input
    const roundsInput = screen.getByLabelText('Number of Rounds') as HTMLInputElement;
    await userEvent.clear(roundsInput);
    await userEvent.type(roundsInput, '5');
    expect(roundsInput.value).toBe('5');

    // For personalities and lead, direct selection might be complex if they use custom select components.
    // We'll assume AssistantConfigPanel updates parent state correctly.
    // Here, we focus on numRounds which is a direct input.
    // Other configurations like selectedPersonalities and leadPersonality are defaulted in ConversationPage state.
    // To test those, we would need to either:
    // 1. Simulate clicks on those specific select elements within AssistantConfigPanel (if easily targetable)
    // 2. Mock parts of AssistantConfigPanel if it's too complex to interact with its internals.
    // For this test, we'll rely on the default personalities and lead, but with updated rounds.

    const startButton = screen.getByRole('button', { name: /Start New Conversation/i });
    await userEvent.click(startButton);

    const textarea = screen.getByPlaceholderText('Type your message...') as HTMLTextAreaElement;
    await userEvent.type(textarea, 'Config test message');
    const sendButton = screen.getByTestId('send-message-button');
    await userEvent.click(sendButton);

    expect(mockRunConversationFn).toHaveBeenCalledTimes(1);
    const calledArgs = mockRunConversationFn.mock.calls[0][0];
    expect(calledArgs.numRounds).toBe(5); // Verify updated rounds
    expect(calledArgs.assistantPersonalities).toEqual(['Helpful Assistant']); // Default
    expect(calledArgs.leadPersonality).toBe('Helpful Assistant'); // Default
  });

  it('ends conversation when conditions are met and prevents further messages', async () => {
    const { rerender } = renderConversationPage();

    // Configure for a short conversation: 1 round, 1 personality
    // This means conversation ends after 1 user message + 1 assistant reply (length 2 for 1*1*2)
    // For the condition: conversation.length >= numRounds * selectedPersonalities.length * 2
    // System message + user message + assistant message = 3.
    // If numRounds = 1, selectedPersonalities.length = 1, then 1 * 1 * 2 = 2.
    // So, after 1 exchange (user + assistant), if there's a finalOutput, it should deactivate.
    // Initial system message is at index 0. User message at 1. Assistant at 2. Length = 3.
    // So, conversation.length (3) >= numRounds (1) * selectedPersonalities.length (1) * 2 (2) -> 3 >= 2 is true.

    const roundsInput = screen.getByLabelText('Number of Rounds') as HTMLInputElement;
    await userEvent.clear(roundsInput);
    await userEvent.type(roundsInput, '1');

    // Start conversation
    const startButton = screen.getByRole('button', { name: /Start New Conversation/i });
    await userEvent.click(startButton);
    await screen.findByText(/Conversation started with Helpful Assistant leading Helpful Assistant for 1 rounds./i)

    // Send first user message
    const textarea = screen.getByPlaceholderText('Type your message...') as HTMLTextAreaElement;
    await userEvent.type(textarea, 'First and final user message');
    const sendButton = screen.getByTestId('send-message-button');

    // Mock runConversation to return a response that includes finalOutput
    const responseToEndConversation = {
      runConversation: {
        conversation: [
          { role: 'system', name: 'System', content: 'Conversation started...' },
          { role: 'user', name: 'User', content: 'First and final user message' },
          { role: 'assistant', name: 'Helpful Assistant', content: 'This is the only reply.' },
        ],
        questions: [],
        finalOutput: 'The conversation has now ended.',
      },
    };

    (useRunConversation as vi.Mock).mockImplementation(() => ({
      runConversation: mockRunConversationFn.mockResolvedValueOnce({ data: responseToEndConversation }),
      data: null, // Initially no data
      loading: false,
      error: null,
    }));

    await userEvent.click(sendButton); // This triggers runConversation

    // Simulate data being returned
    await act(async () => {
      (useRunConversation as vi.Mock).mockImplementation(() => ({
        runConversation: mockRunConversationFn, // Already called, just need to provide data
        data: responseToEndConversation,
        loading: false,
        error: null,
      }));
      rerender(
        <ApolloProvider client={client}>
          <ConversationPage />
        </ApolloProvider>
      );
      await Promise.resolve();
    });

    expect(await screen.findByText('This is the only reply.')).toBeInTheDocument();
    expect(await screen.findByText('The conversation has now ended.')).toBeInTheDocument();
    expect(mockRunConversationFn).toHaveBeenCalledTimes(1);


    // Try to send another message
    await userEvent.type(textarea, 'Attempting another message');
    expect(textarea.value).toBe('Attempting another messageFirst and final user message'); // Textarea doesn't clear if send fails early

    // Click send button again
    // Need to ensure the button is re-enabled after the first send if textarea is not empty.
    // The ChatInput enables button if !isLoading && inputValue.trim().
    // Since loading should be false now, and input is not empty, it should be enabled.
    expect(sendButton).toBeEnabled();
    await userEvent.click(sendButton);

    // Assert that runConversation was NOT called again
    expect(mockRunConversationFn).toHaveBeenCalledTimes(1);

    // Assert that the specific toast message was shown
    expect(mockToastError).toHaveBeenCalledWith("Please start a new conversation first.");

    // Optional: Verify the UI reflects the inactive state (e.g., user message not added)
    expect(screen.queryByText('Attempting another message')).not.toBeInTheDocument();
  });
});
