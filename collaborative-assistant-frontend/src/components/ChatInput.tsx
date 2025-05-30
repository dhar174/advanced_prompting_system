import React, { useState } from 'react';
import { Textarea } from './ui/Textarea';
import { Button } from './ui/Button';
import { Send, Loader2 } from 'lucide-react'; // Using Loader2 for a spinner

interface ChatInputProps {
  onSendMessage: (message: string) => void;
  isLoading: boolean;
}

const ChatInput: React.FC<ChatInputProps> = ({ onSendMessage, isLoading }) => {
  const [inputValue, setInputValue] = useState('');

  const handleInputChange = (event: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInputValue(event.target.value);
  };

  const handleSubmit = () => {
    if (inputValue.trim() && !isLoading) {
      onSendMessage(inputValue.trim());
      setInputValue('');
    }
  };

  const handleKeyPress = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey && !isLoading) {
      event.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="flex items-start space-x-2 p-4 border-t border-slate-200 bg-surface-card">
      <Textarea
        value={inputValue}
        onChange={handleInputChange}
        onKeyDown={handleKeyDown}
        placeholder="Type your message..."
        className="flex-1 resize-none"
        rows={1} // Start with 1 row, will auto-grow with content due to flex
        disabled={isLoading}
      />
      <Button
        onClick={handleSubmit}
        disabled={isLoading || !inputValue.trim()}
        size="lg" // Make button a bit larger
        variant="primary"
        className="h-[40px]" // Match textarea height if rows=1
        data-testid="send-message-button"
      >
        {isLoading ? (
          <Loader2 size={20} className="animate-spin" />
        ) : (
          <Send size={20} />
        )}
      </Button>
    </div>
  );
};

export default ChatInput;
