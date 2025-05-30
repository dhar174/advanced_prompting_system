import React, { useState, useEffect, useRef } from 'react';
import AssistantConfigPanel from '../components/AssistantConfigPanel';
import ChatInput from '../components/ChatInput';
import MessageBubble from '../components/MessageBubble';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/Card';
import { useRunConversation } from '../hooks/useRunConversation';
import type { ConversationInputType, ConversationType, QuestionType } from '../graphql/graphqlTypes';
import toast from 'react-hot-toast'; // Using react-hot-toast for errors

const AVAILABLE_PERSONALITIES = ['Helpful Assistant', 'Sarcastic Assistant', 'Domain Expert', 'Creative Writer', 'Code Generator', 'Summarizer'];
const INITIAL_ROUNDS = 3;

const ConversationPage: React.FC = () => {
  const [conversationHistory, setConversationHistory] = useState<ConversationType[]>([
    { role: 'system', name: 'System', content: 'Welcome! Configure your assistants on the left and start the conversation.' }
  ]);
  const [currentQuestions, setCurrentQuestions] = useState<QuestionType[]>([]);
  const [finalOutput, setFinalOutput] = useState<string | null>(null);

  const [availablePersonalities] = useState<string[]>(AVAILABLE_PERSONALITIES);
  const [selectedPersonalities, setSelectedPersonalities] = useState<string[]>(['Helpful Assistant']);
  const [leadPersonality, setLeadPersonality] = useState<string>('Helpful Assistant');
  const [numRounds, setNumRounds] = useState<number>(INITIAL_ROUNDS);
  const [isConversationActive, setIsConversationActive] = useState<boolean>(false);

  const { runConversation, data: runConversationData, loading: runConversationLoading, error: runConversationError } = useRunConversation();
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [conversationHistory, currentQuestions, finalOutput, runConversationLoading]);

  useEffect(() => {
    if (runConversationData?.runConversation) {
      const { conversation, questions, finalOutput: newFinalOutput } = runConversationData.runConversation;
      setConversationHistory(conversation);
      setCurrentQuestions(questions || []);
      setFinalOutput(newFinalOutput);

      // Logic to deactivate conversation can be more sophisticated, e.g., based on number of turns or if final output is substantial
      if (newFinalOutput && conversation.length >= numRounds * selectedPersonalities.length * 2) { // Example condition
         // setIsConversationActive(false); // Keep it active to allow follow-up or feedback
      }
    }
  }, [runConversationData, numRounds, selectedPersonalities.length]);

  useEffect(() => {
    if (runConversationError) {
      console.error("Error running conversation:", runConversationError);
      toast.error(`Error: ${runConversationError.message}`);
      // setIsConversationActive(false); // Allow user to retry or change config
    }
  }, [runConversationError]);

  const handleStartConversation = () => {
    if (selectedPersonalities.length === 0 || !leadPersonality || numRounds <= 0) {
      toast.error("Please select personalities, a lead, and set rounds.");
      return;
    }
    setIsConversationActive(true);
    setConversationHistory([{ role: 'system', name: 'System', content: `Conversation started with ${leadPersonality} leading ${selectedPersonalities.join(', ')} for ${numRounds} rounds. Send your first message.` }]);
    setCurrentQuestions([]);
    setFinalOutput(null);
  };

  const handleSendMessage = async (messageContent: string) => {
    if (!isConversationActive) {
      toast.error("Please start a new conversation first.");
      return;
    }
    const userMessage: ConversationInputType = {
      role: 'user',
      content: messageContent,
      name: 'User', // Explicitly set user name for input
    };

    const currentHistoryWithUserMessage = [...conversationHistory, userMessage];
    setConversationHistory(currentHistoryWithUserMessage); // Optimistic update
    setCurrentQuestions([]);
    setFinalOutput(null);

    const conversationInput: ConversationInputType[] = currentHistoryWithUserMessage.map(msg => ({
      role: msg.role,
      name: msg.name,
      content: msg.content,
    }));

    try {
       await runConversation({
        // variables are passed directly to the runConversation function from the hook
        conversation: conversationInput,
        assistantPersonalities: selectedPersonalities,
        leadPersonality,
        numRounds,
      });
    } catch (error) {
        // Error is handled by the useEffect for runConversationError
        // If not using toast for errors, you might update UI here
    }
  };

  return (
    <div className="flex h-screen bg-surface-ground text-text-primary">
      <div className="w-1/3 min-w-[320px] max-w-[400px] p-4 border-r border-slate-300 overflow-y-auto bg-slate-50 scrollbar-thin scrollbar-thumb-slate-300 scrollbar-track-slate-100">
        <AssistantConfigPanel
          availablePersonalities={availablePersonalities}
          selectedPersonalities={selectedPersonalities}
          onSelectedPersonalitiesChange={setSelectedPersonalities}
          leadPersonality={leadPersonality}
          onLeadPersonalityChange={setLeadPersonality}
          numRounds={numRounds}
          onNumRoundsChange={setNumRounds}
          onStartConversation={handleStartConversation}
          isConversationActive={isConversationActive && runConversationLoading} // Panel disabled when active AND loading
        />
      </div>

      <div className="flex-1 flex flex-col">
        <Card className="flex-grow flex flex-col m-4 rounded-lg shadow-lg bg-surface-card">
          <CardHeader className="border-b border-slate-200">
            <CardTitle className="text-text-primary">Conversation Thread</CardTitle>
          </CardHeader>
          <CardContent className="flex-grow overflow-y-auto p-6 space-y-4 h-[calc(100vh-180px)] scrollbar-thin scrollbar-thumb-slate-400 scrollbar-track-surface-ground/50">
            {conversationHistory.map((msg, index) => (
              <MessageBubble key={index} message={msg} isUser={msg.role === 'user'} />
            ))}
            {runConversationLoading && (
              <div className="text-center text-sm text-text-secondary italic py-2">Assistant is typing...</div>
            )}
            {currentQuestions.length > 0 && !runConversationLoading && (
              <div className="mt-4 p-4 border-t border-slate-200">
                <h3 className="text-md font-semibold mb-2 text-text-primary">Questions for Feedback:</h3>
                {currentQuestions.map((q, i) => (
                  <div key={i} className="p-3 bg-yellow-100/50 border border-yellow-300 rounded-md mb-2 text-sm text-yellow-800">
                    <p className="font-medium">{q.assistant}:</p>
                    <p>{q.question}</p>
                  </div>
                ))}
              </div>
            )}
            {finalOutput && !runConversationLoading && (
              <div className="mt-4 p-4 border-t border-slate-200">
                <h3 className="text-md font-semibold mb-2 text-text-primary">Final Output:</h3>
                <div className="p-3 bg-green-100/50 border border-green-300 rounded-md text-sm text-green-800 whitespace-pre-wrap">
                  {finalOutput}
                </div>
              </div>
            )}
             {/* Inline error display as an alternative/addition to toast */}
            {runConversationError && !runConversationLoading && (
                <div className="mt-4 p-3 bg-red-100/50 border border-red-300 text-red-700 rounded-md text-sm">
                    <strong>Error:</strong> {runConversationError.message}. Please try again or adjust settings.
                </div>
            )}
            <div ref={messagesEndRef} />
          </CardContent>
          <ChatInput
            onSendMessage={handleSendMessage}
            isLoading={runConversationLoading} // Input disabled only when loading. User can type if conversation is active but not loading.
          />
        </Card>
      </div>
    </div>
  );
};

export default ConversationPage;
