import React, { useEffect, useId, useState } from 'react';
import { Textarea } from './ui/Textarea';
import { Button } from './ui/Button';
import { Send, Loader2 } from 'lucide-react';

export const MAX_PROMPT_CHARS = 12000;

interface ChatInputProps {
  onSendMessage: (message: string) => void;
  isLoading: boolean;
  disabled?: boolean;
  maxLength?: number;
  restoreDraft?: { text: string; key: number } | null;
}

const ChatInput: React.FC<ChatInputProps> = ({
  onSendMessage,
  isLoading,
  disabled = false,
  maxLength = MAX_PROMPT_CHARS,
  restoreDraft = null,
}) => {
  const [inputValue, setInputValue] = useState('');
  const inputId = useId();
  const isBlocked = isLoading || disabled;

  useEffect(() => {
    if (restoreDraft && restoreDraft.text) {
      setInputValue(restoreDraft.text);
    }
  }, [restoreDraft]);

  const handleInputChange = (event: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInputValue(event.target.value);
  };

  const handleSubmit = (event?: React.FormEvent) => {
    event?.preventDefault();
    const trimmed = inputValue.trim();
    if (trimmed && !isBlocked) {
      onSendMessage(trimmed);
      setInputValue('');
    }
  };

  const handleKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.nativeEvent.isComposing) {
      return;
    }
    if (event.key === 'Enter' && !event.shiftKey && !isBlocked) {
      event.preventDefault();
      handleSubmit();
    }
  };

  return (
    <form
      onSubmit={handleSubmit}
      className="flex items-start space-x-2 p-4 border-t border-slate-200 bg-surface-card"
    >
      <label htmlFor={inputId} className="sr-only">
        Prompt message
      </label>
      <Textarea
        id={inputId}
        name="prompt"
        value={inputValue}
        onChange={handleInputChange}
        onKeyDown={handleKeyDown}
        placeholder="Type your message..."
        className="flex-1 resize-none"
        rows={1}
        maxLength={maxLength}
        disabled={isBlocked}
      />
      <Button
        type="submit"
        disabled={isBlocked || !inputValue.trim()}
        size="lg"
        variant="primary"
        className="h-[40px]"
        aria-label={isLoading ? 'Sending message' : 'Send message'}
        data-testid="send-message-button"
      >
        {isLoading ? (
          <Loader2 size={20} className="animate-spin" />
        ) : (
          <Send size={20} />
        )}
      </Button>
    </form>
  );
};

export default ChatInput;
