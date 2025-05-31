import React from 'react';
import MessageBubble from './MessageBubble';
import type { ConversationType } from '../graphql/graphqlTypes';
import type { ApolloError } from '@apollo/client'; // Assuming ApolloError

interface ConversationDisplayProps {
  conversationHistory: ConversationType[];
  runConversationLoading: boolean;
  runConversationError: ApolloError | undefined;
  // messagesEndRef is no longer needed here
}

const ConversationDisplay: React.FC<ConversationDisplayProps> = ({
  conversationHistory,
  runConversationLoading,
  runConversationError,
  // messagesEndRef,
}) => {
  return (
    <>
      {conversationHistory.map((msg, index) => (
        <MessageBubble key={index} message={msg} isUser={msg.role === 'user'} />
      ))}
      {runConversationLoading && (
        <div className="text-center text-sm text-text-secondary italic py-2">Assistant is typing...</div>
      )}
      {runConversationError && !runConversationLoading && (
        <div className="mt-4 p-3 bg-red-100/50 border border-red-300 text-red-700 rounded-md text-sm">
          <strong>Error:</strong> {runConversationError.message}. Please try again or adjust settings.
        </div>
      )}
      {/* The messagesEndRef div is now managed by ConversationPage */}
    </>
  );
};

export default ConversationDisplay;
