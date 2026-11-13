import React, { useState, useEffect, useRef } from 'react';
import AssistantConfigPanel from '../components/AssistantConfigPanel';
import ChatInput from '../components/ChatInput';
import ConversationDisplay from '../components/ConversationDisplay';
import FeedbackDisplay from '../components/FeedbackDisplay';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/Card';
import { useRunConversation } from '../hooks/useRunConversation';
import { useRunExecution } from '../hooks/useRunExecution';
import type {
  ConversationInputType,
  ConversationType,
  QuestionType,
  ExecutionModeId,
  OutputTypeInfo,
  RunDiagnosticsType,
  RunEventType,
} from '../graphql/graphqlTypes';
import toast from 'react-hot-toast';

const AVAILABLE_PERSONALITIES = [
  'Helpful Assistant',
  'Sarcastic Assistant',
  'Domain Expert',
  'Creative Writer',
  'Code Generator',
  'Summarizer',
];
const INITIAL_ROUNDS = 3;

const ConversationPage: React.FC = () => {
  const [conversationHistory, setConversationHistory] = useState<ConversationType[]>([
    {
      role: 'system',
      name: 'System',
      content: 'Welcome! Configure your assistants on the left and start the conversation.',
    },
  ]);
  const [currentQuestions, setCurrentQuestions] = useState<QuestionType[]>([]);
  const [finalOutput, setFinalOutput] = useState<string | null>(null);
  const [outputType, setOutputType] = useState<OutputTypeInfo | null>(null);
  const [diagnostics, setDiagnostics] = useState<RunDiagnosticsType | null>(null);
  const [events, setEvents] = useState<RunEventType[] | null>(null);

  // Configuration state
  const [availablePersonalities] = useState<string[]>(AVAILABLE_PERSONALITIES);
  const [selectedPersonalities, setSelectedPersonalities] = useState<string[]>([
    'Helpful Assistant',
  ]);
  const [leadPersonality, setLeadPersonality] = useState<string>('Helpful Assistant');
  const [numRounds, setNumRounds] = useState<number>(INITIAL_ROUNDS);
  const [isConversationActive, setIsConversationActive] = useState<boolean>(false);

  // Track 1 Execution Mode additions
  const [executionMode, setExecutionMode] = useState<ExecutionModeId>('collaborative');
  const [maxSteps, setMaxSteps] = useState<number>(20);
  const [agentCount, setAgentCount] = useState<number>(3);

  // Hooks
  const {
    runConversation,
    data: runConversationData,
    loading: runConversationLoading,
    error: runConversationError,
  } = useRunConversation();

  const {
    runExecution,
    data: runExecutionData,
    loading: runExecutionLoading,
    error: runExecutionError,
  } = useRunExecution();

  const isLoading = runConversationLoading || runExecutionLoading;
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [conversationHistory, currentQuestions, finalOutput, isLoading]);

  // Handle collaborative conversation data
  useEffect(() => {
    if (runConversationData?.runConversation) {
      const {
        conversation,
        questions,
        finalOutput: newFinalOutput,
      } = runConversationData.runConversation;
      setConversationHistory(conversation);
      setCurrentQuestions(questions || []);
      setFinalOutput(newFinalOutput);
      setOutputType({ outputType: 'Simple Concise Answer', fileExtension: '.txt' });

      if (
        newFinalOutput &&
        conversation.length >= numRounds * selectedPersonalities.length * 2
      ) {
        setIsConversationActive(false);
      }
    }
  }, [runConversationData, numRounds, selectedPersonalities.length]);

  // Handle unified execution data
  useEffect(() => {
    if (runExecutionData?.runExecution) {
      const res = runExecutionData.runExecution;
      setConversationHistory(res.conversation);
      setCurrentQuestions(res.questions || []);
      setFinalOutput(res.finalOutput);
      setOutputType(res.outputType);
      setDiagnostics(res.diagnostics);
      setEvents(res.events);

      if (res.status === 'completed' || res.status === 'gated' || res.status === 'error') {
        setIsConversationActive(false);
      }
    }
  }, [runExecutionData]);

  // Handle errors
  useEffect(() => {
    if (runConversationError) {
      console.error('Error running conversation:', runConversationError);
      toast.error(`Error: ${runConversationError.message}`);
    }
  }, [runConversationError]);

  useEffect(() => {
    if (runExecutionError) {
      console.error('Error in execution:', runExecutionError);
      toast.error(`Execution Error: ${runExecutionError.message}`);
    }
  }, [runExecutionError]);

  const handleStartConversation = () => {
    if (executionMode === 'collaborative') {
      if (selectedPersonalities.length === 0 || !leadPersonality || numRounds <= 0) {
        toast.error('Please select personalities, a lead, and set rounds.');
        return;
      }
      setIsConversationActive(true);
      setConversationHistory([
        {
          role: 'system',
          name: 'System',
          content: `Conversation started with ${leadPersonality} leading ${selectedPersonalities.join(
            ', '
          )} for ${numRounds} rounds. Send your first message.`,
        },
      ]);
    } else if (executionMode === 'advanced_pipeline') {
      setIsConversationActive(true);
      setConversationHistory([
        {
          role: 'system',
          name: 'System',
          content: `Advanced Prompt Pipeline initialized (Max Steps: ${maxSteps}, Collaborative Agents: ${agentCount}). Enter your task prompt below.`,
        },
      ]);
    }
    setCurrentQuestions([]);
    setFinalOutput(null);
    setDiagnostics(null);
    setEvents(null);
  };

  const handleSendMessage = async (messageContent: string) => {
    if (!isConversationActive) {
      toast.error('Please start a new conversation first.');
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

    const conversationInput: ConversationInputType[] = currentHistoryWithUserMessage.map(
      (msg) => ({
        role: msg.role,
        name: msg.name,
        content: msg.content,
      })
    );

    try {
      if (executionMode === 'collaborative') {
        await runConversation({
          conversation: conversationInput,
          assistantPersonalities: selectedPersonalities,
          leadPersonality,
          numRounds,
        });
      } else {
        await runExecution({
          input: {
            mode: executionMode,
            taskPrompt: messageContent,
            conversation: conversationInput,
            maxSteps,
            agents: agentCount,
          },
        });
      }
    } catch {
      // Error handled by useEffects
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
          isConversationActive={isConversationActive && isLoading}
          executionMode={executionMode}
          onExecutionModeChange={setExecutionMode}
          maxSteps={maxSteps}
          onMaxStepsChange={setMaxSteps}
          agentCount={agentCount}
          onAgentCountChange={setAgentCount}
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
              runConversationLoading={isLoading}
              runConversationError={runConversationError || runExecutionError}
            />
            <FeedbackDisplay
              currentQuestions={currentQuestions}
              finalOutput={finalOutput}
              runConversationLoading={isLoading}
              outputType={outputType}
              diagnostics={diagnostics}
              events={events}
            />
            <div ref={messagesEndRef} />
          </CardContent>
          <ChatInput onSendMessage={handleSendMessage} isLoading={isLoading} />
        </Card>
      </div>
    </div>
  );
};

export default ConversationPage;
