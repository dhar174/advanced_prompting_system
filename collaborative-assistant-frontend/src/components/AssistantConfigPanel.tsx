import React, { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/Card';
import { Button } from './ui/Button';
import { Input } from './ui/Input';
import { Label } from './ui/Label';
import type { ExecutionModeCapabilityType, ExecutionModeId } from '../graphql/graphqlTypes';
import {
  DEFAULT_MAX_AGENTS,
  DEFAULT_MAX_NUM_ROUNDS,
  DEFAULT_MAX_STEPS,
  DEFAULT_MIN_NUM_ROUNDS,
  type CapabilityDiscoveryStatus,
} from '../hooks/useRunExecution';

export interface AssistantConfigPanelProps {
  availablePersonalities: string[];
  selectedPersonalities: string[];
  onSelectedPersonalitiesChange: (personalities: string[]) => void;
  leadPersonality: string;
  onLeadPersonalityChange: (personality: string) => void;
  numRounds: number;
  onNumRoundsChange: (rounds: number) => void;
  onStartConversation: () => void;
  isConversationActive: boolean;
  // Unified Execution Mode additions
  executionMode?: ExecutionModeId;
  onExecutionModeChange?: (mode: ExecutionModeId) => void;
  executionModes?: ExecutionModeCapabilityType[];
  discoveryStatus?: CapabilityDiscoveryStatus;
  maxSteps?: number;
  onMaxStepsChange?: (steps: number) => void;
  agentCount?: number;
  onAgentCountChange?: (agents: number) => void;
  allowedModels?: string[];
  selectedModel?: string;
  onSelectedModelChange?: (model: string) => void;
}

const DISCOVERY_PLACEHOLDER_LABELS: Record<CapabilityDiscoveryStatus, string> = {
  loading: 'Loading server capabilities...',
  auth_required: 'Authentication required to load modes',
  failed: 'Unable to load execution modes',
  empty: 'No execution modes available',
  available: 'No execution modes available',
};

const AssistantConfigPanel: React.FC<AssistantConfigPanelProps> = ({
  availablePersonalities,
  selectedPersonalities,
  onSelectedPersonalitiesChange,
  leadPersonality,
  onLeadPersonalityChange,
  numRounds,
  onNumRoundsChange,
  onStartConversation,
  isConversationActive,
  executionMode = 'collaborative',
  onExecutionModeChange,
  executionModes,
  discoveryStatus,
  maxSteps = 20,
  onMaxStepsChange,
  agentCount = 3,
  onAgentCountChange,
  allowedModels,
  selectedModel,
  onSelectedModelChange,
}) => {
  const hasDiscoveryContract =
    discoveryStatus !== undefined || executionModes !== undefined;
  const modeOptions = executionModes ?? [];
  const activeModeCapability = modeOptions.find((m) => m.id === executionMode);

  const minRounds =
    typeof activeModeCapability?.minNumRounds === 'number' &&
    Number.isInteger(activeModeCapability.minNumRounds) &&
    activeModeCapability.minNumRounds >= 1
      ? activeModeCapability.minNumRounds
      : DEFAULT_MIN_NUM_ROUNDS;

  const maxRounds =
    typeof activeModeCapability?.maxNumRounds === 'number' &&
    Number.isInteger(activeModeCapability.maxNumRounds) &&
    activeModeCapability.maxNumRounds >= minRounds
      ? activeModeCapability.maxNumRounds
      : DEFAULT_MAX_NUM_ROUNDS;

  const maxAllowedSteps =
    typeof activeModeCapability?.maxSteps === 'number' &&
    Number.isInteger(activeModeCapability.maxSteps) &&
    activeModeCapability.maxSteps >= 1
      ? activeModeCapability.maxSteps
      : DEFAULT_MAX_STEPS;

  const maxAllowedAgents =
    typeof activeModeCapability?.maxAgents === 'number' &&
    Number.isInteger(activeModeCapability.maxAgents) &&
    activeModeCapability.maxAgents >= 0
      ? activeModeCapability.maxAgents
      : DEFAULT_MAX_AGENTS;

  const [roundsDraft, setRoundsDraft] = useState<string>(String(numRounds));
  const [stepsDraft, setStepsDraft] = useState<string>(String(maxSteps));
  const [agentsDraft, setAgentsDraft] = useState<string>(String(agentCount));

  useEffect(() => {
    setRoundsDraft(String(numRounds));
  }, [numRounds]);

  useEffect(() => {
    setStepsDraft(String(maxSteps));
  }, [maxSteps]);

  useEffect(() => {
    setAgentsDraft(String(agentCount));
  }, [agentCount]);

  const handlePersonalityToggle = (personality: string) => {
    const newSelection = selectedPersonalities.includes(personality)
      ? selectedPersonalities.filter((p) => p !== personality)
      : [...selectedPersonalities, personality];
    onSelectedPersonalitiesChange(newSelection);

    if (leadPersonality === personality && !newSelection.includes(personality)) {
      onLeadPersonalityChange(newSelection.length > 0 ? newSelection[0] : '');
    }
  };

  const handleLeadPersonalityChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    onLeadPersonalityChange(event.target.value);
  };

  const handleNumRoundsChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const raw = event.target.value;
    setRoundsDraft(raw);
    const trimmed = raw.trim();
    if (!/^\d+$/.test(trimmed)) {
      return;
    }
    const rounds = Number.parseInt(trimmed, 10);
    if (rounds >= minRounds && rounds <= maxRounds) {
      onNumRoundsChange(rounds);
    }
  };

  const handleModeChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    if (onExecutionModeChange && event.target.value) {
      onExecutionModeChange(event.target.value as ExecutionModeId);
    }
  };

  const handleMaxStepsChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const raw = event.target.value;
    setStepsDraft(raw);
    const trimmed = raw.trim();
    if (!/^\d+$/.test(trimmed)) {
      return;
    }
    const steps = Number.parseInt(trimmed, 10);
    if (steps >= 1 && steps <= maxAllowedSteps && onMaxStepsChange) {
      onMaxStepsChange(steps);
    }
  };

  const handleAgentCountChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const raw = event.target.value;
    setAgentsDraft(raw);
    const trimmed = raw.trim();
    if (!/^\d+$/.test(trimmed)) {
      return;
    }
    const count = Number.parseInt(trimmed, 10);
    if (count >= 0 && count <= maxAllowedAgents && onAgentCountChange) {
      onAgentCountChange(count);
    }
  };

  const isPipeline = activeModeCapability
    ? activeModeCapability.supportsStepPlanning
    : executionMode === 'advanced_pipeline';
  const isCollaborative = activeModeCapability
    ? activeModeCapability.supportsMultiAgent && !activeModeCapability.supportsStepPlanning
    : executionMode === 'collaborative';

  const getNumRoundsError = (): string | null => {
    const trimmed = roundsDraft.trim();
    if (!trimmed || !/^\d+$/.test(trimmed)) {
      return `Enter a whole number between ${minRounds} and ${maxRounds}.`;
    }
    const parsed = Number.parseInt(trimmed, 10);
    if (parsed < minRounds || parsed > maxRounds) {
      return `Number of rounds must be between ${minRounds} and ${maxRounds}.`;
    }
    if (!Number.isInteger(numRounds) || numRounds < minRounds || numRounds > maxRounds) {
      return `Number of rounds must be between ${minRounds} and ${maxRounds}.`;
    }
    return null;
  };

  const getMaxStepsError = (): string | null => {
    const trimmed = stepsDraft.trim();
    if (!trimmed || !/^\d+$/.test(trimmed)) {
      return `Enter a whole number between 1 and ${maxAllowedSteps}.`;
    }
    const parsed = Number.parseInt(trimmed, 10);
    if (parsed < 1 || parsed > maxAllowedSteps) {
      return `Max reasoning steps must be between 1 and ${maxAllowedSteps}.`;
    }
    if (!Number.isInteger(maxSteps) || maxSteps < 1 || maxSteps > maxAllowedSteps) {
      return `Max reasoning steps must be between 1 and ${maxAllowedSteps}.`;
    }
    return null;
  };

  const getAgentCountError = (): string | null => {
    const trimmed = agentsDraft.trim();
    if (!trimmed || !/^\d+$/.test(trimmed)) {
      return `Enter a whole number between 0 and ${maxAllowedAgents}.`;
    }
    const parsed = Number.parseInt(trimmed, 10);
    if (parsed < 0 || parsed > maxAllowedAgents) {
      return `Collaborative agents must be between 0 and ${maxAllowedAgents}.`;
    }
    if (!Number.isInteger(agentCount) || agentCount < 0 || agentCount > maxAllowedAgents) {
      return `Collaborative agents must be between 0 and ${maxAllowedAgents}.`;
    }
    return null;
  };

  const numRoundsError = isCollaborative ? getNumRoundsError() : null;
  const maxStepsError = isPipeline ? getMaxStepsError() : null;
  const agentCountError = isPipeline ? getAgentCountError() : null;

  const isDiscoveryReady =
    discoveryStatus !== undefined
      ? discoveryStatus === 'available'
      : executionModes !== undefined
      ? modeOptions.length > 0
      : true;

  const isModeMissing = hasDiscoveryContract && !activeModeCapability;
  const isModeDisabled = activeModeCapability ? !activeModeCapability.enabled : false;
  const isModeNotReady = activeModeCapability ? activeModeCapability.ready !== true : false;

  const isStartDisabled =
    isConversationActive ||
    !isDiscoveryReady ||
    isModeMissing ||
    isModeDisabled ||
    isModeNotReady ||
    (isCollaborative
      ? selectedPersonalities.length === 0 ||
        !leadPersonality ||
        Boolean(numRoundsError)
      : Boolean(maxStepsError) || Boolean(agentCountError));

  const buttonText = isConversationActive
    ? isPipeline
      ? 'Pipeline in Progress...'
      : 'Conversation in Progress...'
    : isPipeline
    ? 'Start Pipeline Session'
    : 'Start New Conversation';

  const modeSelectDisabled =
    isConversationActive || (hasDiscoveryContract && modeOptions.length === 0);

  return (
    <Card className="w-full max-w-md bg-surface-card">
      <CardHeader>
        <CardTitle className="text-text-primary">Configure Assistants</CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Execution Mode Selector */}
        <div>
          <Label htmlFor="execution-mode" className="block text-sm font-medium text-text-primary mb-1">
            Execution Mode:
          </Label>
          <select
            id="execution-mode"
            value={modeOptions.length > 0 ? executionMode : hasDiscoveryContract ? '' : executionMode}
            onChange={handleModeChange}
            disabled={modeSelectDisabled}
            className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-slate-300 bg-surface-card text-text-primary focus:outline-none focus:ring-ring-brand focus:border-ring-brand sm:text-sm rounded-md disabled:bg-slate-100 disabled:text-slate-500 disabled:cursor-not-allowed"
          >
            {modeOptions.length > 0 ? (
              modeOptions.map((mode) => {
                const suffix = !mode.enabled
                  ? ` (Disabled${mode.gatedIssue ? ` - ${mode.gatedIssue}` : ''})`
                  : mode.ready !== true
                  ? ' (Provider not configured)'
                  : '';
                return (
                  <option key={mode.id} value={mode.id} disabled={!mode.enabled}>
                    {mode.label}
                    {suffix}
                  </option>
                );
              })
            ) : hasDiscoveryContract ? (
              <option value="" disabled>
                {DISCOVERY_PLACEHOLDER_LABELS[discoveryStatus ?? 'empty']}
              </option>
            ) : (
              <option value="collaborative">Collaborative Debate</option>
            )}
          </select>
          {activeModeCapability &&
            (!activeModeCapability.enabled || activeModeCapability.ready !== true) &&
            activeModeCapability.unavailableReason && (
              <p role="status" className="mt-1.5 text-xs text-amber-700">
                {activeModeCapability.unavailableReason}
              </p>
            )}
        </div>

        {/* Optional Server-Owned Model Selector */}
        {allowedModels && allowedModels.length > 0 && onSelectedModelChange && (
          <div>
            <Label htmlFor="model-select" className="block text-sm font-medium text-text-primary mb-1">
              Model Policy:
            </Label>
            <select
              id="model-select"
              value={selectedModel && allowedModels.includes(selectedModel) ? selectedModel : allowedModels[0]}
              onChange={(e) => onSelectedModelChange(e.target.value)}
              disabled={isConversationActive}
              className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-slate-300 bg-surface-card text-text-primary focus:outline-none focus:ring-ring-brand focus:border-ring-brand sm:text-sm rounded-md disabled:bg-slate-100 disabled:text-slate-500 disabled:cursor-not-allowed"
            >
              {allowedModels.map((modelName) => (
                <option key={modelName} value={modelName}>
                  {modelName}
                </option>
              ))}
            </select>
          </div>
        )}

        {isCollaborative && (
          <>
            <fieldset>
              <legend className="block text-sm font-medium text-text-primary mb-2">
                Select Personalities:
              </legend>
              <div className="space-y-2 max-h-56 overflow-y-auto pr-1">
                {availablePersonalities.map((personality) => (
                  <div key={personality} className="flex items-center">
                    <input
                      type="checkbox"
                      id={`personality-${personality}`}
                      value={personality}
                      checked={selectedPersonalities.includes(personality)}
                      onChange={() => handlePersonalityToggle(personality)}
                      disabled={isConversationActive}
                      className="h-4 w-4 text-brand-primary border-slate-300 rounded focus:ring-ring-brand"
                    />
                    <Label htmlFor={`personality-${personality}`} className="ml-2 block text-sm text-text-secondary">
                      {personality}
                    </Label>
                  </div>
                ))}
              </div>
            </fieldset>

            <div>
              <Label htmlFor="lead-personality" className="block text-sm font-medium text-text-primary mb-1">
                Lead Personality:
              </Label>
              <select
                id="lead-personality"
                value={leadPersonality}
                onChange={handleLeadPersonalityChange}
                disabled={isConversationActive || selectedPersonalities.length === 0}
                className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-slate-300 bg-surface-card text-text-primary focus:outline-none focus:ring-ring-brand focus:border-ring-brand sm:text-sm rounded-md disabled:bg-slate-100 disabled:text-slate-500 disabled:cursor-not-allowed"
              >
                <option value="" disabled={selectedPersonalities.length > 0}>
                  {selectedPersonalities.length === 0 ? 'Select personalities first' : 'Select a lead'}
                </option>
                {selectedPersonalities.map((personality) => (
                  <option key={personality} value={personality}>
                    {personality}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <Label
                htmlFor="num-rounds"
                className="block text-sm font-medium text-text-primary mb-1"
                title="A 'round' consists of each selected assistant processing the conversation sequence once. The lead personality speaks last in each turn."
              >
                Number of Rounds (per turn):
              </Label>
              <Input
                type="number"
                id="num-rounds"
                value={roundsDraft}
                onChange={handleNumRoundsChange}
                min={minRounds}
                max={maxRounds}
                step={1}
                disabled={isConversationActive}
                aria-invalid={Boolean(numRoundsError)}
                aria-describedby={numRoundsError ? 'num-rounds-error' : 'num-rounds-hint'}
                className="mt-1 block w-full"
              />
              <p id="num-rounds-hint" className="mt-1 text-xs text-text-secondary">
                Allowed range: {minRounds}–{maxRounds} rounds.
              </p>
              {numRoundsError && (
                <p id="num-rounds-error" role="alert" className="mt-1 text-xs text-rose-700">
                  {numRoundsError}
                </p>
              )}
            </div>
          </>
        )}

        {isPipeline && (
          <div className="space-y-4">
            <div>
              <Label htmlFor="max-steps" className="block text-sm font-medium text-text-primary mb-1">
                Max Reasoning Steps:
              </Label>
              <Input
                type="number"
                id="max-steps"
                value={stepsDraft}
                onChange={handleMaxStepsChange}
                min={1}
                max={maxAllowedSteps}
                step={1}
                disabled={isConversationActive}
                aria-invalid={Boolean(maxStepsError)}
                aria-describedby={maxStepsError ? 'max-steps-error' : undefined}
                className="mt-1 block w-full"
              />
              {maxStepsError && (
                <p id="max-steps-error" role="alert" className="mt-1 text-xs text-rose-700">
                  {maxStepsError}
                </p>
              )}
            </div>

            <div>
              <Label htmlFor="agent-count" className="block text-sm font-medium text-text-primary mb-1">
                Collaborative Agents per Step:
              </Label>
              <Input
                type="number"
                id="agent-count"
                value={agentsDraft}
                onChange={handleAgentCountChange}
                min={0}
                max={maxAllowedAgents}
                step={1}
                disabled={isConversationActive}
                aria-invalid={Boolean(agentCountError)}
                aria-describedby={agentCountError ? 'agent-count-error' : undefined}
                className="mt-1 block w-full"
              />
              {agentCountError && (
                <p id="agent-count-error" role="alert" className="mt-1 text-xs text-rose-700">
                  {agentCountError}
                </p>
              )}
            </div>
          </div>
        )}

        <Button
          type="button"
          onClick={onStartConversation}
          disabled={isStartDisabled}
          className="w-full"
          variant="primary"
        >
          {buttonText}
        </Button>
      </CardContent>
    </Card>
  );
};

export default AssistantConfigPanel;
