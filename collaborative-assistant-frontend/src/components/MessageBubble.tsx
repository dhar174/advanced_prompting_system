import React from 'react';
// Using Card for structure, but could be a simple div
// import { Card } from './ui/Card';

interface Message {
  role: string;
  name?: string | null; // name can be null
  content: string;
}

interface MessageBubbleProps {
  message: Message;
  isUser: boolean;
}

const MessageBubble: React.FC<MessageBubbleProps> = ({ message, isUser }) => {
  const bubbleAlignment = isUser ? 'ml-auto justify-end' : 'mr-auto justify-start';
  const bubbleColor = isUser ? 'bg-user-bubble-bg text-user-bubble-text' : 'bg-assistant-bubble-bg text-text-primary';
  const senderName = message.name || (isUser ? 'You' : 'Assistant');

  return (
    <div className={`flex flex-col w-full ${bubbleAlignment} mb-2`}>
      <div className={`max-w-xl md:max-w-2xl p-3 rounded-lg shadow-sm ${bubbleColor}`}>
        {(message.name || message.role !== 'user') && (
            <p className={`text-xs font-medium mb-1 ${isUser ? 'text-user-bubble-text/80' : 'text-text-secondary'}`}>
                {senderName} ({message.role})
            </p>
        )}
        <p className="text-sm whitespace-pre-wrap">{message.content}</p>
      </div>
    </div>
  );
};

export default MessageBubble;
