import React from 'react';
import MessageBubble from './MessageBubble';
import type { ConversationType } from '../graphql/graphqlTypes';
import type { ApolloError } from '@apollo/client';

interface ConversationDisplayProps {
  conversationHistory: ConversationType[];
  runConversationLoading: boolean;
  runConversationError: ApolloError | undefined;
  errorMessage?: string | null;
}

const ConversationDisplay: React.FC<ConversationDisplayProps> = ({
  conversationHistory,
  runConversationLoading,
  runConversationError,
  errorMessage,
}) => {
  const displayError = errorMessage || runConversationError?.message || null;

  return (
    <>
      {conversationHistory.map((msg, index) => (
        <MessageBubble key={`${msg.role}-${index}`} message={msg} isUser={msg.role === 'user'} />
      ))}
      {runConversationLoading && (
        <div
          role="progressbar"
          aria-label="Running reasoning workflow"
          aria-valuetext="Running reasoning workflow"
          className="text-center text-sm text-text-secondary italic py-2"
        >
          Assistant is working...
        </div>
      )}
      {displayError && !runConversationLoading && (
        <div
          role="alert"
          className="mt-4 p-3 bg-red-100/60 border border-red-300 text-red-800 rounded-md text-sm"
        >
          <strong>Error:</strong> {displayError}. Please try again or adjust settings.
        </div>
      )}
    </>
  );
};

export default ConversationDisplay;
