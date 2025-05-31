import React, { useState, useEffect, useRef } from 'react';
import AssistantConfigPanel from '../components/AssistantConfigPanel';
import ChatInput from '../components/ChatInput';
// MessageBubble is no longer directly used here, but ConversationDisplay uses it.
// import MessageBubble from '../components/MessageBubble';
import ConversationDisplay from '../components/ConversationDisplay';
import FeedbackDisplay from '../components/FeedbackDisplay';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/Card';
import { useRunConversation } from '../hooks/useRunConversation';
import type { ConversationInputType, ConversationType, QuestionType } from '../graphql/graphqlTypes';
import toast from 'react-hot-toast';

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

      if (newFinalOutput && conversation.length >= numRounds * selectedPersonalities.length * 2) {
        // Signifies the end of the planned conversation rounds.
        // Set isConversationActive to false to prevent further messages
        // and prompt the user to start a new conversation.
        setIsConversationActive(false);
      }
    }
  }, [runConversationData, numRounds, selectedPersonalities.length]);

  useEffect(() => {
    if (runConversationError) {
      console.error("Error running conversation:", runConversationError);
      toast.error(`Error: ${runConversationError.message}`);
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
      name: 'User',
    };

    const currentHistoryWithUserMessage = [...conversationHistory, userMessage];
    setConversationHistory(currentHistoryWithUserMessage);
    setCurrentQuestions([]);
    setFinalOutput(null);

    const conversationInput: ConversationInputType[] = currentHistoryWithUserMessage.map(msg => ({
      role: msg.role,
      name: msg.name,
      content: msg.content,
    }));

    try {
       await runConversation({
        conversation: conversationInput,
        assistantPersonalities: selectedPersonalities,
        leadPersonality,
        numRounds,
      });
    } catch (error) {
        // Error is handled by the useEffect for runConversationError
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
          isConversationActive={isConversationActive && runConversationLoading}
        />
      </div>

      <div className="flex-1 flex flex-col">
        <Card className="flex-grow flex flex-col m-4 rounded-lg shadow-lg bg-surface-card">
          <CardHeader className="border-b border-slate-200">
            <CardTitle className="text-text-primary">Conversation Thread</CardTitle>
          </CardHeader>
          <CardContent className="flex-grow overflow-y-auto p-6 space-y-4 h-[calc(100vh-180px)] scrollbar-thin scrollbar-thumb-slate-400 scrollbar-track-surface-ground/50">
            <ConversationDisplay
              conversationHistory={conversationHistory}
              runConversationLoading={runConversationLoading}
              runConversationError={runConversationError}
              // messagesEndRef is no longer passed here
            />
            <FeedbackDisplay
              currentQuestions={currentQuestions}
              finalOutput={finalOutput}
              runConversationLoading={runConversationLoading}
            />
            <div ref={messagesEndRef} /> {/* Correct placement of messagesEndRef */}
          </CardContent>
          <ChatInput
            onSendMessage={handleSendMessage}
            isLoading={runConversationLoading}
          />
        </Card>
      </div>
    </div>
  );
};

export default ConversationPage;
