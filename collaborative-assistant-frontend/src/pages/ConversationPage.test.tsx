import { render, screen, act, waitFor } from '@testing-library/react';
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
let mockData: any = null; // Allow 'any' for easier mocking of complex data structures

describe('ConversationPage Integration Test', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockLoading = false;
    mockError = null;
    mockData = null;
    (useRunConversation as vi.Mock).mockReturnValue([
      mockRunConversationFn,
      { loading: mockLoading, error: mockError, data: mockData },
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
    // Use data-testid for the send button
    const sendButton = screen.getByTestId('send-message-button');
    
    // Mock the hook to return data after the next call
    mockData = {
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
    (useRunConversation as vi.Mock).mockReturnValue([
      mockRunConversationFn.mockImplementation(async () => { /* Simulating promise resolution */ }),
      { loading: false, error: null, data: mockData }, // Simulate data being available immediately for this test
    ]);
    
    // Re-render or trigger update if necessary for the hook mock to take effect.
    // For this setup, the next interaction should use the new mock values.
    // However, it's often better to set the mock *before* the action that triggers its use.
    // Let's assume the mock is updated before the component re-renders due to state changes from send.

    await userEvent.click(sendButton); // This call will use the initial mock

    // To simulate the data update, we need to re-render with the new mock state.
    // This is a common challenge in testing hooks that update asynchronously.
    // A more robust way is to have the mock function itself update the data/loading/error states.
    
    // For this test, let's update the mock and re-render the component.
    // This is not ideal but demonstrates the goal.
    // A better approach would involve `act` and managing the hook's state from the mock itself.

    // Simulate receiving data by re-rendering with the new mock state.
    // This simulates the component re-rendering after the hook updates.
    (useRunConversation as vi.Mock).mockReturnValue([
        mockRunConversationFn, // The function itself
        { loading: false, error: null, data: mockData } // The new state
    ]);
    // No explicit re-render call needed here if the hook's state change triggers it.
    // The test needs to wait for the UI to update based on the new `mockData`.

    // We need to wait for the UI to update based on the mocked data.
    // Using findBy queries to wait for elements to appear.
    expect(await screen.findByText('Assistant response')).toBeInTheDocument();
    expect(await screen.findByText('Was this helpful?')).toBeInTheDocument();
    expect(await screen.findByText('This is the final output.')).toBeInTheDocument();
  });

  it('shows loading indicator when runConversationLoading is true', () => {
    (useRunConversation as vi.Mock).mockReturnValue([
      mockRunConversationFn,
      { loading: true, error: null, data: null },
    ]);
    renderConversationPage();
    // Start conversation to enable chat input area fully
    const startButton = screen.getByRole('button', { name: /Start New Conversation/i });
    // Need to use `act` for state updates that might happen during userEvent
    act(() => {
        userEvent.click(startButton);
    });
    expect(screen.getByText('Assistant is typing...')).toBeInTheDocument();
  });

  it('displays error message when runConversationError is present', async () => {
    mockError = new Error('Network Error');
    (useRunConversation as vi.Mock).mockReturnValue([
      mockRunConversationFn,
      { loading: false, error: mockError, data: null },
    ]);
    renderConversationPage();
     // Start conversation to enable chat input area fully
    const startButton = screen.getByRole('button', { name: /Start New Conversation/i });
    await act(async () => {
        await userEvent.click(startButton);
    });

    // Wait for the error message to appear (it might be displayed asynchronously)
    // Using findByText to wait for the element. Check for the main message content.
    expect(await screen.findByText((content, element) => {
      const hasText = (node: Element | null) => node?.textContent?.includes('Error: Network Error') || false;
      return hasText(element) && element?.classList.contains('text-red-700');
    })).toBeInTheDocument();
  });
});
