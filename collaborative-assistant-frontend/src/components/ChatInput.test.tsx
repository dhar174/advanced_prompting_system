import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import ChatInput from './ChatInput';
import { describe, it, expect, vi } from 'vitest';
// import { Send, Loader2 } from 'lucide-react'; // TSC flags as unused due to mock

// Mock lucide-react icons
vi.mock('lucide-react', async (importOriginal) => {
  const original = await importOriginal() as any;
  return {
    ...original,
    Send: (props: any) => <svg data-testid="send-icon" {...props} />,
    Loader2: (props: any) => <svg data-testid="loader-icon" {...props} />,
  };
});

describe('ChatInput component', () => {
  const mockOnSendMessage = vi.fn();

  it('renders textarea and send button', () => {
    render(<ChatInput onSendMessage={mockOnSendMessage} isLoading={false} />);
    expect(screen.getByPlaceholderText('Type your message...')).toBeInTheDocument();
    expect(screen.getByTestId('send-icon')).toBeInTheDocument();
  });

  it('updates input value on change', async () => {
    render(<ChatInput onSendMessage={mockOnSendMessage} isLoading={false} />);
    const textarea = screen.getByPlaceholderText('Type your message...') as HTMLTextAreaElement;
    await userEvent.type(textarea, 'Hello there');
    expect(textarea.value).toBe('Hello there');
  });

  it('calls onSendMessage with the message and clears input on send button click', async () => {
    render(<ChatInput onSendMessage={mockOnSendMessage} isLoading={false} />);
    const textarea = screen.getByPlaceholderText('Type your message...') as HTMLTextAreaElement;
    const sendButton = screen.getByRole('button');

    await userEvent.type(textarea, 'Test message  '); // With trailing spaces
    await userEvent.click(sendButton);

    expect(mockOnSendMessage).toHaveBeenCalledTimes(1);
    expect(mockOnSendMessage).toHaveBeenCalledWith('Test message'); // Test trimming
    expect(textarea.value).toBe('');
  });

  it('calls onSendMessage on Enter key press (not Shift+Enter)', async () => {
    render(<ChatInput onSendMessage={mockOnSendMessage} isLoading={false} />);
    const textarea = screen.getByPlaceholderText('Type your message...') as HTMLTextAreaElement;

    await userEvent.type(textarea, 'Message with enter');
    await userEvent.keyboard('{Enter}');

    expect(mockOnSendMessage).toHaveBeenCalledWith('Message with enter');
    expect(textarea.value).toBe('');
  });

  it('does not call onSendMessage on Shift+Enter key press', async () => {
    mockOnSendMessage.mockClear(); // Clear previous calls
    render(<ChatInput onSendMessage={mockOnSendMessage} isLoading={false} />);
    const textarea = screen.getByPlaceholderText('Type your message...') as HTMLTextAreaElement;

    await userEvent.type(textarea, 'Message with shift enter');
    await userEvent.keyboard('{Shift>}{Enter}{/Shift}'); // Simulate Shift+Enter

    expect(mockOnSendMessage).not.toHaveBeenCalled();
    expect(textarea.value).toBe('Message with shift enter\n'); // Newline should be added
  });


  it('disables textarea and button when isLoading is true', () => {
    render(<ChatInput onSendMessage={mockOnSendMessage} isLoading={true} />);
    expect(screen.getByPlaceholderText('Type your message...')).toBeDisabled();
    expect(screen.getByRole('button')).toBeDisabled();
  });

  it('shows loader icon when isLoading is true', () => {
    render(<ChatInput onSendMessage={mockOnSendMessage} isLoading={true} />);
    expect(screen.getByTestId('loader-icon')).toBeInTheDocument();
    expect(screen.queryByTestId('send-icon')).not.toBeInTheDocument();
  });

  it('does not call onSendMessage if message is empty or only whitespace', async () => {
    mockOnSendMessage.mockClear();
    render(<ChatInput onSendMessage={mockOnSendMessage} isLoading={false} />);
    const sendButton = screen.getByRole('button');
    const textarea = screen.getByPlaceholderText('Type your message...') as HTMLTextAreaElement;

    // Test with empty message
    await userEvent.click(sendButton);
    expect(mockOnSendMessage).not.toHaveBeenCalled();

    // Test with whitespace message
    await userEvent.type(textarea, '   ');
    await userEvent.click(sendButton);
    expect(mockOnSendMessage).not.toHaveBeenCalled();
    expect(textarea.value).toBe('   '); // Should not clear if not sent
  });
   it('send button is disabled if message is empty or only whitespace', async () => {
    render(<ChatInput onSendMessage={mockOnSendMessage} isLoading={false} />);
    const sendButton = screen.getByRole('button');
    const textarea = screen.getByPlaceholderText('Type your message...') as HTMLTextAreaElement;

    expect(sendButton).toBeDisabled(); // Initially disabled

    await userEvent.type(textarea, '   ');
    expect(sendButton).toBeDisabled(); // Disabled with only whitespace

    await userEvent.type(textarea, 'Valid message');
    expect(sendButton).not.toBeDisabled(); // Enabled with valid message
  });
});
