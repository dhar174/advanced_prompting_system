/// <reference types="vitest/globals" />
import React from 'react';
import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import MessageBubble from './MessageBubble'; // Assuming default export
import { Message } from '../types'; // Assuming Message type is exported from types.ts or similar

// Helper to create a message object
const createMessage = (
  role: string,
  content: string,
  name?: string | null
): Message => ({
  role,
  content,
  name,
});

describe('MessageBubble', () => {
  describe('User Message Rendering', () => {
    const userMessage: Message = createMessage('user', 'Hello, this is a user message.');
    const namedUserMessage: Message = createMessage('user', 'Hello, I am TestUser.', 'TestUser');

    it('renders correctly when isUser is true and message.name is not provided', () => {
      render(<MessageBubble message={userMessage} isUser={true} />);
      const bubble = screen.getByText(userMessage.content).closest('div');
      expect(bubble?.parentElement).toHaveClass('ml-auto justify-end'); // bubbleAlignment
      expect(bubble).toHaveClass('bg-user-bubble-bg text-user-bubble-text'); // bubbleColor
      expect(screen.getByText(userMessage.content)).toBeInTheDocument();

      // Sender name line should NOT be present if message.name is null for a user
      // as per component logic: (message.name || message.role !== 'user') -> (null || 'user' !== 'user') -> false
      expect(screen.queryByText(/You \(user\)/i)).not.toBeInTheDocument();
    });

    it('renders correctly when isUser is true and message.name IS provided', () => {
        render(<MessageBubble message={namedUserMessage} isUser={true} />);
        const bubble = screen.getByText(namedUserMessage.content).closest('div');
        expect(bubble?.parentElement).toHaveClass('ml-auto justify-end');
        expect(bubble).toHaveClass('bg-user-bubble-bg text-user-bubble-text');
        expect(screen.getByText(namedUserMessage.content)).toBeInTheDocument();

        // Sender name line IS present if message.name is provided for a user
        // (message.name || message.role !== 'user') -> ('TestUser' || 'user' !== 'user') -> true
        const senderElement = screen.getByText(/TestUser \(user\)/i);
        expect(senderElement).toBeInTheDocument();
        expect(senderElement).toHaveClass('text-user-bubble-text/80');
      });
  });

  describe('Assistant Message Rendering', () => {
    const assistantMessage: Message = createMessage('assistant', 'I am an assistant.');
    const namedAssistantMessage: Message = createMessage(
      'assistant',
      'I am SpecificBot.',
      'SpecificBot'
    );

    it('renders correctly when isUser is false and message.name is not provided', () => {
      render(<MessageBubble message={assistantMessage} isUser={false} />);
      const bubble = screen.getByText(assistantMessage.content).closest('div');
      expect(bubble?.parentElement).toHaveClass('mr-auto justify-start');
      expect(bubble).toHaveClass('bg-assistant-bubble-bg text-text-primary');
      expect(screen.getByText(assistantMessage.content)).toBeInTheDocument();

      // Sender name line IS present for non-user messages
      // (message.name || message.role !== 'user') -> (null || 'assistant' !== 'user') -> true
      // senderName = message.name || (isUser ? 'You' : 'Assistant') -> 'Assistant'
      const senderElement = screen.getByText('Assistant (assistant)');
      expect(senderElement).toBeInTheDocument();
      expect(senderElement).toHaveClass('text-text-secondary');
    });

    it('renders correctly when isUser is false and message.name IS provided', () => {
      render(<MessageBubble message={namedAssistantMessage} isUser={false} />);
      const bubble = screen.getByText(namedAssistantMessage.content).closest('div');
      expect(bubble?.parentElement).toHaveClass('mr-auto justify-start');
      expect(bubble).toHaveClass('bg-assistant-bubble-bg text-text-primary');
      expect(screen.getByText(namedAssistantMessage.content)).toBeInTheDocument();

      // Sender name line IS present
      // senderName = message.name || (isUser ? 'You' : 'Assistant') -> 'SpecificBot'
      const senderElement = screen.getByText('SpecificBot (assistant)');
      expect(senderElement).toBeInTheDocument();
      expect(senderElement).toHaveClass('text-text-secondary');
    });
  });

  describe('System Message Rendering (or other roles)', () => {
    const systemMessage: Message = createMessage('system', 'System update applied.');
    const namedSystemMessage: Message = createMessage(
      'system',
      'System critical alert by SysMonitor.',
      'SysMonitor'
    );

    it('renders system message correctly when message.name is not provided', () => {
      render(<MessageBubble message={systemMessage} isUser={false} />); // isUser is false for non-user roles
      const bubble = screen.getByText(systemMessage.content).closest('div');
      expect(bubble?.parentElement).toHaveClass('mr-auto justify-start'); // Assistant-like alignment
      expect(bubble).toHaveClass('bg-assistant-bubble-bg text-text-primary'); // Assistant-like color
      expect(screen.getByText(systemMessage.content)).toBeInTheDocument();

      // Sender name line IS present for non-user messages
      // (message.name || message.role !== 'user') -> (null || 'system' !== 'user') -> true
      // senderName = message.name || (isUser ? 'You' : 'Assistant') -> 'Assistant'
      // This will render "Assistant (system)" based on current component logic.
      // If specific handling for "system" role's default name is desired, component needs change.
      // For now, testing existing behavior.
      const senderElement = screen.getByText('Assistant (system)');
      expect(senderElement).toBeInTheDocument();
    });

    it('renders system message correctly when message.name IS provided', () => {
      render(<MessageBubble message={namedSystemMessage} isUser={false} />);
      const bubble = screen.getByText(namedSystemMessage.content).closest('div');
      expect(bubble?.parentElement).toHaveClass('mr-auto justify-start');
      expect(bubble).toHaveClass('bg-assistant-bubble-bg text-text-primary');
      expect(screen.getByText(namedSystemMessage.content)).toBeInTheDocument();

      // Sender name line IS present
      // senderName = message.name || (isUser ? 'You' : 'Assistant') -> 'SysMonitor'
      const senderElement = screen.getByText('SysMonitor (system)');
      expect(senderElement).toBeInTheDocument();
    });
  });

  describe('Content Display', () => {
    it('renders message.content accurately', () => {
      const content = 'This is the exact content.';
      const message = createMessage('user', content);
      render(<MessageBubble message={message} isUser={true} />);
      expect(screen.getByText(content)).toBeInTheDocument();
    });

    it('preserves whitespace in message.content', () => {
      const contentWithWhitespace = 'Line 1\n  Line 2\n    Line 3';
      const message = createMessage('assistant', contentWithWhitespace);
      render(<MessageBubble message={message} isUser={false} data-testid="message-bubble-test" />);
      // Find the <p> tag that should have the 'whitespace-pre-wrap' class.
      // This assumes the content is rendered within a <p> tag that has this class.
      // Based on MessageBubble.tsx: <p className="text-sm whitespace-pre-wrap">{message.content}</p>
      const allParagraphs = screen.getAllByText((content, element) => {
        return element?.tagName.toLowerCase() === 'p' && content.trim() !== '' && element.classList.contains('whitespace-pre-wrap');
      });
      // Find the specific paragraph that contains our whitespace content
      const contentElement = allParagraphs.find(el => el.textContent === contentWithWhitespace);
      expect(contentElement).toBeInTheDocument();
      expect(contentElement).toHaveClass('whitespace-pre-wrap');
      expect(contentElement?.textContent).toBe(contentWithWhitespace);
    });
  });

  describe('Sender Name Logic (Detailed)', () => {
    it('sender name line appears for non-user (assistant) messages with default name', () => {
      const message = createMessage('assistant', 'Assistant content.');
      render(<MessageBubble message={message} isUser={false} />);
      expect(screen.getByText('Assistant (assistant)')).toBeInTheDocument();
    });

    it('sender name line appears for non-user (assistant) messages with provided name', () => {
      const message = createMessage('assistant', 'Content here.', 'CustomBot');
      render(<MessageBubble message={message} isUser={false} />);
      expect(screen.getByText('CustomBot (assistant)')).toBeInTheDocument();
    });

    it('sender name line appears for non-user (system) messages with default name', () => {
        const message = createMessage('system', 'System content.');
        render(<MessageBubble message={message} isUser={false} />);
        // Based on current logic: senderName becomes 'Assistant' for non-user, non-named roles
        expect(screen.getByText('Assistant (system)')).toBeInTheDocument();
      });

    it('sender name line appears for non-user (system) messages with provided name', () => {
        const message = createMessage('system', 'Content here.', 'SystemLogger');
        render(<MessageBubble message={message} isUser={false} />);
        expect(screen.getByText('SystemLogger (system)')).toBeInTheDocument();
    });

    it('sender name line appears for user messages IF message.name is provided', () => {
      const message = createMessage('user', 'User content with name.', 'RegisteredUser');
      render(<MessageBubble message={message} isUser={true} />);
      expect(screen.getByText('RegisteredUser (user)')).toBeInTheDocument();
    });

    it('sender name line does NOT appear for user messages if message.name is null/undefined', () => {
      const message = createMessage('user', 'Anonymous user content.');
      render(<MessageBubble message={message} isUser={true} />);
      // Querying for any text that might include "(user)" to ensure the line is absent
      expect(screen.queryByText(/\(user\)/i)).not.toBeInTheDocument();
      // Specifically, "You (user)" should not be there as per component logic
      expect(screen.queryByText('You (user)')).not.toBeInTheDocument();
    });
  });
});

// Definition for Message type (can be moved to a shared types file)
// Assuming it's already defined elsewhere if `../types` was used.
// For this standalone test file, if not imported, it might be:
// interface Message {
//   role: string;
//   name?: string | null;
//   content: string;
// }
