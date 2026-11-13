import React, { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/Card';
import { Button } from './ui/Button';
import { Input } from './ui/Input';
import { Label } from './ui/Label';
import type { ExecutionModeCapabilityType, ExecutionModeId } from '../graphql/graphqlTypes';

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
  maxSteps?: number;
  onMaxStepsChange?: (steps: number) => void;
  agentCount?: number;
  onAgentCountChange?: (agents: number) => void;
  allowedModels?: string[];
  selectedModel?: string;
  onSelectedModelChange?: (model: string) => void;
}

const FALLBACK_MODES: ExecutionModeCapabilityType[] = [
  {
    id: 'collaborative',
    label: 'Collaborative Debate (Multi-Agent Deliberation)',
    description: 'Multi-agent round deliberation & consensus voting.',
    enabled: true,
    ready: true,
    gatedIssue: null,
    supportsMultiAgent: true,
    supportsStepPlanning: false,
    supportsCancellation: false,
    supportsResume: false,
  },
  {
    id: 'advanced_pipeline',
    label: 'Standalone Pipeline (Complexity & Step Planning)',
    description: 'Complexity-driven step planning, reflections & reward judging.',
    enabled: true,
    ready: true,
    gatedIssue: null,
    supportsMultiAgent: true,
    supportsStepPlanning: true,
    supportsCancellation: false,
    supportsResume: false,
  },
  {
    id: 'tree_of_thought',
    label: 'Tree of Thoughts',
    description: 'Deliberate search over reasoning trees.',
    enabled: false,
    ready: false,
    gatedIssue: 'Issue #36',
    unavailableReason: 'Gated pending Issue #36.',
    supportsMultiAgent: false,
    supportsStepPlanning: true,
    supportsCancellation: false,
    supportsResume: false,
  },
];

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
  maxSteps = 20,
  onMaxStepsChange,
  agentCount = 3,
  onAgentCountChange,
  allowedModels,
  selectedModel,
  onSelectedModelChange,
}) => {
  const modeOptions = executionModes && executionModes.length > 0 ? executionModes : FALLBACK_MODES;
  const activeModeCapability = modeOptions.find((m) => m.id === executionMode);

  const [stepsDraft, setStepsDraft] = useState<string>(String(maxSteps));
  const [agentsDraft, setAgentsDraft] = useState<string>(String(agentCount));

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
    if (event.target.value === '') {
      onNumRoundsChange(1);
      return;
    }
    const rounds = parseInt(event.target.value, 10);
    if (!isNaN(rounds) && rounds >= 1 && rounds <= 20) {
      onNumRoundsChange(rounds);
    }
  };

  const handleModeChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    if (onExecutionModeChange) {
      onExecutionModeChange(event.target.value as ExecutionModeId);
    }
  };

  const handleMaxStepsChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const raw = event.target.value;
    setStepsDraft(raw);
    const steps = parseInt(raw, 10);
    if (!isNaN(steps) && steps >= 1 && steps <= 50 && onMaxStepsChange) {
      onMaxStepsChange(steps);
    }
  };

  const handleMaxStepsBlur = () => {
    const steps = parseInt(stepsDraft, 10);
    const clamped = isNaN(steps) ? 1 : Math.min(Math.max(steps, 1), 50);
    setStepsDraft(String(clamped));
    if (onMaxStepsChange) {
      onMaxStepsChange(clamped);
    }
  };

  const handleAgentCountChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const raw = event.target.value;
    setAgentsDraft(raw);
    const count = parseInt(raw, 10);
    if (!isNaN(count) && count >= 0 && count <= 10 && onAgentCountChange) {
      onAgentCountChange(count);
    }
  };

  const handleAgentCountBlur = () => {
    const count = parseInt(agentsDraft, 10);
    const clamped = isNaN(count) ? 0 : Math.min(Math.max(count, 0), 10);
    setAgentsDraft(String(clamped));
    if (onAgentCountChange) {
      onAgentCountChange(clamped);
    }
  };

  const isPipeline = activeModeCapability
    ? activeModeCapability.supportsStepPlanning
    : executionMode === 'advanced_pipeline';
  const isCollaborative = activeModeCapability
    ? activeModeCapability.supportsMultiAgent && !activeModeCapability.supportsStepPlanning
    : executionMode === 'collaborative';

  const isModeDisabled = activeModeCapability ? !activeModeCapability.enabled : false;
  const isModeNotReady = activeModeCapability ? activeModeCapability.ready === false : false;

  const isStartDisabled =
    isConversationActive ||
    isModeDisabled ||
    isModeNotReady ||
    (isCollaborative
      ? selectedPersonalities.length === 0 || !leadPersonality || numRounds <= 0
      : maxSteps <= 0);

  const buttonText = isConversationActive
    ? isPipeline
      ? 'Pipeline in Progress...'
      : 'Conversation in Progress...'
    : isPipeline
    ? 'Start Pipeline Session'
    : 'Start New Conversation';

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
            value={executionMode}
            onChange={handleModeChange}
            disabled={isConversationActive}
            className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-slate-300 bg-surface-card text-text-primary focus:outline-none focus:ring-ring-brand focus:border-ring-brand sm:text-sm rounded-md disabled:bg-slate-100 disabled:text-slate-500 disabled:cursor-not-allowed"
          >
            {modeOptions.map((mode) => {
              const suffix = !mode.enabled
                ? ` (Disabled${mode.gatedIssue ? ` - ${mode.gatedIssue}` : ''})`
                : mode.ready === false
                ? ' (Provider not configured)'
                : '';
              return (
                <option key={mode.id} value={mode.id} disabled={!mode.enabled}>
                  {mode.label}
                  {suffix}
                </option>
              );
            })}
          </select>
          {activeModeCapability && !activeModeCapability.ready && activeModeCapability.unavailableReason && (
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
                value={numRounds}
                onChange={handleNumRoundsChange}
                min="1"
                max="20"
                disabled={isConversationActive}
                className="mt-1 block w-full"
              />
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
                onBlur={handleMaxStepsBlur}
                min="1"
                max="50"
                disabled={isConversationActive}
                className="mt-1 block w-full"
              />
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
                onBlur={handleAgentCountBlur}
                min="0"
                max="10"
                disabled={isConversationActive}
                className="mt-1 block w-full"
              />
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
