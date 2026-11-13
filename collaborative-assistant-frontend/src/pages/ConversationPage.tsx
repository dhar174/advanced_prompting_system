import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import ApiAuthPanel from '../components/ApiAuthPanel';
import AssistantConfigPanel from '../components/AssistantConfigPanel';
import ChatInput from '../components/ChatInput';
import ConversationDisplay from '../components/ConversationDisplay';
import FeedbackDisplay from '../components/FeedbackDisplay';
import { Card, CardContent, CardHeader } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import {
  DEFAULT_MAX_AGENTS,
  DEFAULT_MAX_NUM_ROUNDS,
  DEFAULT_MAX_STEPS,
  DEFAULT_MIN_NUM_ROUNDS,
  useRunExecution,
} from '../hooks/useRunExecution';
import type {
  ArtifactRefType,
  ConversationInputType,
  ConversationType,
  QuestionType,
  ExecutionModeId,
  OutputTypeInfo,
  RunDiagnosticsType,
  RunEventType,
  RunExecutionResponseType,
  RunStatus,
} from '../graphql/graphqlTypes';
import toast from 'react-hot-toast';

const INITIAL_ROUNDS = 3;
const INITIAL_SELECTED_PERSONALITIES = ['Analytical Thinker', 'Mediator'];
const INITIAL_LEAD_PERSONALITY = 'Mediator';

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
  const [artifacts, setArtifacts] = useState<ArtifactRefType[] | null>(null);
  const [runStatus, setRunStatus] = useState<RunStatus | null>(null);
  const [stopReason, setStopReason] = useState<string | null>(null);
  const [runErrorMessage, setRunErrorMessage] = useState<string | null>(null);
  const [restoreDraft, setRestoreDraft] = useState<{ text: string; key: number } | null>(null);

  // Unified Hook
  const {
    runExecution,
    cancelActiveRun,
    resetExecutionState,
    retryDiscovery,
    applyApiToken,
    clearStoredApiToken,
    data: runExecutionData,
    loading: runExecutionLoading,
    error: runExecutionError,
    errorMessage: hookErrorMessage,
    wasCancelled,
    executionModes,
    personalities,
    providerPolicy,
    discoveryStatus,
    discoveryErrorMessage,
    authRequired,
    hasApiToken,
  } = useRunExecution();

  const availablePersonalities = useMemo(
    () => personalities.map((p) => p.id),
    [personalities]
  );

  // Configuration state (grounded in backend PERSONALITY_CATALOG)
  const [selectedPersonalities, setSelectedPersonalities] = useState<string[]>(
    INITIAL_SELECTED_PERSONALITIES
  );
  const [leadPersonality, setLeadPersonality] = useState<string>(INITIAL_LEAD_PERSONALITY);
  const [numRounds, setNumRounds] = useState<number>(INITIAL_ROUNDS);
  const [isConversationActive, setIsConversationActive] = useState<boolean>(false);

  // Execution Mode & Server-Owned Model Policy
  const [executionMode, setExecutionMode] = useState<ExecutionModeId>('collaborative');
  const [maxSteps, setMaxSteps] = useState<number>(20);
  const [agentCount, setAgentCount] = useState<number>(3);
  const [selectedModel, setSelectedModel] = useState<string>('gpt-4o-mini');

  // Reconcile selected personalities and model whenever backend catalogs change
  useEffect(() => {
    if (availablePersonalities.length > 0) {
      const validSelection = selectedPersonalities.filter((p) =>
        availablePersonalities.includes(p)
      );
      if (validSelection.length !== selectedPersonalities.length) {
        const reconciled =
          validSelection.length > 0 ? validSelection : [availablePersonalities[0]];
        setSelectedPersonalities(reconciled);
        if (!reconciled.includes(leadPersonality)) {
          setLeadPersonality(reconciled[0]);
        }
      } else if (leadPersonality && !validSelection.includes(leadPersonality) && validSelection.length > 0) {
        setLeadPersonality(validSelection[0]);
      }
    }
  }, [availablePersonalities, selectedPersonalities, leadPersonality]);

  useEffect(() => {
    if (providerPolicy?.allowedModels && providerPolicy.allowedModels.length > 0) {
      if (!providerPolicy.allowedModels.includes(selectedModel)) {
        setSelectedModel(providerPolicy.defaultModel || providerPolicy.allowedModels[0]);
      }
    }
  }, [providerPolicy, selectedModel]);

  const activeModeCapability = useMemo(
    () => executionModes.find((m) => m.id === executionMode),
    [executionModes, executionMode]
  );

  const activeMinRounds =
    typeof activeModeCapability?.minNumRounds === 'number' &&
    activeModeCapability.minNumRounds >= 1
      ? activeModeCapability.minNumRounds
      : DEFAULT_MIN_NUM_ROUNDS;

  const activeMaxRounds =
    typeof activeModeCapability?.maxNumRounds === 'number' &&
    activeModeCapability.maxNumRounds >= activeMinRounds
      ? activeModeCapability.maxNumRounds
      : DEFAULT_MAX_NUM_ROUNDS;

  const activeMaxSteps =
    typeof activeModeCapability?.maxSteps === 'number' &&
    activeModeCapability.maxSteps >= 1
      ? activeModeCapability.maxSteps
      : DEFAULT_MAX_STEPS;

  const activeMaxAgents =
    typeof activeModeCapability?.maxAgents === 'number' &&
    activeModeCapability.maxAgents >= 0
      ? activeModeCapability.maxAgents
      : DEFAULT_MAX_AGENTS;

  const isLoading = runExecutionLoading;
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inFlightLockRef = useRef<boolean>(false);
  const activeRunSeqRef = useRef<number>(0);
  const lastSubmittedDraftRef = useRef<string | null>(null);
  const appliedDataRef = useRef<RunExecutionResponseType | null>(null);

  const applyRunResult = useCallback((res: RunExecutionResponseType) => {
    setConversationHistory(res.conversation);
    setCurrentQuestions(res.questions || []);
    setFinalOutput(res.finalOutput || null);
    setOutputType(res.outputType || null);
    setDiagnostics(res.diagnostics || null);
    setEvents(res.events || null);
    setArtifacts(res.artifacts || null);
    setRunStatus(res.status);
    setStopReason(res.stopReason || null);
    setRunErrorMessage(res.errorMessage || null);

    if (res.status === 'error' || res.status === 'gated' || res.status === 'timeout') {
      toast.error(res.errorMessage || `Execution stopped with status: ${res.status}`);
      if (lastSubmittedDraftRef.current) {
        setRestoreDraft({ text: lastSubmittedDraftRef.current, key: Date.now() });
      }
    } else {
      lastSubmittedDraftRef.current = null;
    }

    if (
      res.status === 'completed' ||
      res.status === 'gated' ||
      res.status === 'error' ||
      res.status === 'budget_exhausted' ||
      res.status === 'incomplete' ||
      res.status === 'cancelled' ||
      res.status === 'timeout'
    ) {
      setIsConversationActive(false);
    }
  }, []);

  useEffect(() => {
    if (messagesEndRef.current && typeof messagesEndRef.current.scrollIntoView === 'function') {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [conversationHistory, currentQuestions, finalOutput, isLoading, runStatus]);

  // Synchronize external Apollo/mock updates without re-applying after cancellation
  useEffect(() => {
    const incoming = runExecutionData?.runExecution;
    if (incoming && !wasCancelled && incoming !== appliedDataRef.current) {
      appliedDataRef.current = incoming;
      applyRunResult(incoming);
    }
  }, [runExecutionData, wasCancelled, applyRunResult]);

  useEffect(() => {
    if (wasCancelled) {
      setRunStatus('cancelled');
      setStopReason('user_cancelled');
      setRunErrorMessage('Execution wait aborted by user.');
    }
  }, [wasCancelled]);

  useEffect(() => {
    if (runExecutionError && !wasCancelled) {
      const msg = hookErrorMessage || runExecutionError.message;
      console.error('Error in execution:', runExecutionError);
      setRunStatus('error');
      setRunErrorMessage(msg);
      toast.error(`Execution Error: ${msg}`);
      if (lastSubmittedDraftRef.current) {
        setRestoreDraft({ text: lastSubmittedDraftRef.current, key: Date.now() });
      }
    }
  }, [runExecutionError, hookErrorMessage, wasCancelled]);

  const validateExecutionReadiness = (): string | null => {
    if (discoveryStatus !== 'available') {
      return (
        discoveryErrorMessage ||
        'Server capabilities are not ready. Please retry discovery.'
      );
    }
    if (!activeModeCapability || !activeModeCapability.enabled) {
      return (
        activeModeCapability?.unavailableReason ||
        'Selected execution mode is currently disabled.'
      );
    }
    if (activeModeCapability.ready !== true) {
      return (
        activeModeCapability.unavailableReason ||
        'Selected execution mode is not ready on the server.'
      );
    }
    if (executionMode === 'collaborative') {
      if (selectedPersonalities.length === 0 || !leadPersonality) {
        return 'Please select personalities and a lead personality.';
      }
      if (
        !Number.isInteger(numRounds) ||
        numRounds < activeMinRounds ||
        numRounds > activeMaxRounds
      ) {
        return `Number of rounds must be between ${activeMinRounds} and ${activeMaxRounds}.`;
      }
    } else if (executionMode === 'advanced_pipeline') {
      if (!Number.isInteger(maxSteps) || maxSteps < 1 || maxSteps > activeMaxSteps) {
        return `Max reasoning steps must be between 1 and ${activeMaxSteps}.`;
      }
      if (
        !Number.isInteger(agentCount) ||
        agentCount < 0 ||
        agentCount > activeMaxAgents
      ) {
        return `Collaborative agents must be between 0 and ${activeMaxAgents}.`;
      }
    } else {
      return 'Selected execution mode is currently disabled.';
    }
    return null;
  };

  const handleStartConversation = () => {
    if (isLoading || inFlightLockRef.current) {
      return;
    }

    const readinessError = validateExecutionReadiness();
    if (readinessError) {
      toast.error(readinessError);
      return;
    }

    activeRunSeqRef.current += 1;
    lastSubmittedDraftRef.current = null;
    resetExecutionState?.();

    if (executionMode === 'collaborative') {
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
    setOutputType(null);
    setDiagnostics(null);
    setEvents(null);
    setArtifacts(null);
    setRunStatus(null);
    setStopReason(null);
    setRunErrorMessage(null);
  };

  const handleCancelExecution = () => {
    activeRunSeqRef.current += 1;
    inFlightLockRef.current = false;
    cancelActiveRun?.();
    setIsConversationActive(false);
    setRunStatus('cancelled');
    setStopReason('user_cancelled');
    setRunErrorMessage('Execution wait aborted by user.');
    if (lastSubmittedDraftRef.current) {
      setRestoreDraft({ text: lastSubmittedDraftRef.current, key: Date.now() });
    }
  };

  const handleSendMessage = async (messageContent: string) => {
    if (isLoading || inFlightLockRef.current) {
      return;
    }
    if (!isConversationActive) {
      toast.error('Please start a new conversation first.');
      return;
    }

    const readinessError = validateExecutionReadiness();
    if (readinessError) {
      toast.error(readinessError);
      setRestoreDraft({ text: messageContent, key: Date.now() });
      return;
    }

    const userMessage: ConversationInputType = {
      role: 'user',
      content: messageContent,
      name: 'User',
    };

    lastSubmittedDraftRef.current = messageContent;
    const currentHistoryWithUserMessage = [...conversationHistory, userMessage];
    setConversationHistory(currentHistoryWithUserMessage);
    setCurrentQuestions([]);
    setFinalOutput(null);
    setArtifacts(null);
    setRunStatus(null);
    setStopReason(null);
    setRunErrorMessage(null);

    // Strip UI-only system status banners before sending conversation turns to backend
    const conversationInput: ConversationInputType[] = currentHistoryWithUserMessage
      .filter((msg) => msg.role !== 'system')
      .map((msg) => ({
        role: msg.role,
        name: msg.name,
        content: msg.content,
      }));

    const callSeq = ++activeRunSeqRef.current;
    inFlightLockRef.current = true;
    try {
      const response = await runExecution({
        input: {
          mode: executionMode,
          taskPrompt: messageContent,
          conversation: conversationInput,
          assistantPersonalities: selectedPersonalities,
          leadPersonality,
          numRounds,
          maxSteps,
          agents: agentCount,
          model: selectedModel,
        },
      });
      if (activeRunSeqRef.current !== callSeq) {
        return;
      }
      if (response) {
        appliedDataRef.current = response;
        applyRunResult(response);
      } else if (!inFlightLockRef.current) {
        // Cancelled in flight
      } else {
        // Transport/hook error (including 401) without GraphQL RunExecutionResponse
        setConversationHistory(conversationHistory);
        if (lastSubmittedDraftRef.current) {
          setRestoreDraft({ text: lastSubmittedDraftRef.current, key: Date.now() });
        }
      }
    } catch {
      if (activeRunSeqRef.current === callSeq) {
        setConversationHistory(conversationHistory);
        if (lastSubmittedDraftRef.current) {
          setRestoreDraft({ text: lastSubmittedDraftRef.current, key: Date.now() });
        }
      }
    } finally {
      if (activeRunSeqRef.current === callSeq) {
        inFlightLockRef.current = false;
      }
    }
  };

  const combinedDisplayError = runErrorMessage || hookErrorMessage || null;
  const cancelLabel = activeModeCapability?.supportsCancellation
    ? 'Cancel Run'
    : 'Abort Wait (Server Continues)';

  return (
    <div className="flex h-screen bg-surface-ground text-text-primary">
      <aside
        aria-label="Assistant Configuration"
        className="w-1/3 min-w-[320px] max-w-[400px] p-4 border-r border-slate-300 overflow-y-auto bg-slate-50 scrollbar-thin scrollbar-thumb-slate-300 scrollbar-track-slate-100"
      >
        <ApiAuthPanel
          authRequired={authRequired}
          hasToken={hasApiToken}
          isBusy={isLoading}
          errorMessage={authRequired ? discoveryErrorMessage : null}
          onApplyToken={applyApiToken}
          onClearToken={clearStoredApiToken}
        />

        {discoveryStatus === 'loading' && (
          <div
            role="status"
            className="mb-3 p-2.5 rounded border border-slate-300 bg-slate-100 text-xs text-slate-800"
          >
            Loading server capabilities...
          </div>
        )}

        {(discoveryStatus === 'failed' || discoveryStatus === 'empty') && (
          <div
            role="alert"
            className="mb-3 p-3 rounded border border-rose-300 bg-rose-50 text-xs text-rose-900 space-y-2"
          >
            <p className="font-semibold">
              {discoveryStatus === 'failed'
                ? 'Unable to load server capabilities.'
                : 'No execution modes available from server.'}
            </p>
            <p>
              {discoveryErrorMessage ||
                'Execution is disabled until server readiness is confirmed.'}
            </p>
            <Button
              type="button"
              size="sm"
              variant="outline"
              onClick={() => void retryDiscovery()}
              data-testid="retry-discovery-button"
            >
              Retry Discovery
            </Button>
          </div>
        )}

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
          executionModes={executionModes}
          discoveryStatus={discoveryStatus}
          maxSteps={maxSteps}
          onMaxStepsChange={setMaxSteps}
          agentCount={agentCount}
          onAgentCountChange={setAgentCount}
          allowedModels={providerPolicy?.allowedModels}
          selectedModel={selectedModel}
          onSelectedModelChange={setSelectedModel}
        />
      </aside>

      <main className="flex-1 flex flex-col">
        <Card className="flex-grow flex flex-col m-4 rounded-lg shadow-lg bg-surface-card">
          <CardHeader className="border-b border-slate-200 flex flex-row items-center justify-between">
            <h1 className="text-xl font-semibold leading-none tracking-tight text-text-primary">
              Conversation Thread
            </h1>
            {isLoading && (
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={handleCancelExecution}
                data-testid="cancel-execution-button"
              >
                {cancelLabel}
              </Button>
            )}
          </CardHeader>
          <CardContent className="flex-grow overflow-y-auto p-6 space-y-4 h-[calc(100vh-180px)] scrollbar-thin scrollbar-thumb-slate-400 scrollbar-track-surface-ground/50">
            <ConversationDisplay
              conversationHistory={conversationHistory}
              runConversationLoading={isLoading}
              runConversationError={runExecutionError}
              errorMessage={runExecutionError ? combinedDisplayError : null}
            />
            <FeedbackDisplay
              currentQuestions={currentQuestions}
              finalOutput={finalOutput}
              runConversationLoading={isLoading}
              outputType={outputType}
              diagnostics={diagnostics}
              events={events}
              runStatus={runStatus}
              stopReason={stopReason}
              errorMessage={combinedDisplayError}
              artifacts={artifacts}
            />
            <div ref={messagesEndRef} />
          </CardContent>
          <ChatInput
            onSendMessage={handleSendMessage}
            isLoading={isLoading}
            restoreDraft={restoreDraft}
          />
        </Card>
      </main>
    </div>
  );
};

export default ConversationPage;
