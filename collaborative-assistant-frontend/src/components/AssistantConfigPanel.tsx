import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/Card';
import { Button } from './ui/Button';
import { Input } from './ui/Input';
import { Label } from './ui/Label';
import type { ExecutionModeId } from '../graphql/graphqlTypes';

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
  // Optional Track 1 Execution Mode additions
  executionMode?: ExecutionModeId;
  onExecutionModeChange?: (mode: ExecutionModeId) => void;
  maxSteps?: number;
  onMaxStepsChange?: (steps: number) => void;
  agentCount?: number;
  onAgentCountChange?: (agents: number) => void;
}

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
  maxSteps = 20,
  onMaxStepsChange,
  agentCount = 3,
  onAgentCountChange,
}) => {
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
    const rounds = parseInt(event.target.value, 10);
    if (!isNaN(rounds) && rounds >= 1) {
      onNumRoundsChange(rounds);
    } else if (event.target.value === '') {
      onNumRoundsChange(1);
    }
  };

  const handleModeChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    if (onExecutionModeChange) {
      onExecutionModeChange(event.target.value as ExecutionModeId);
    }
  };

  const handleMaxStepsChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const steps = parseInt(event.target.value, 10);
    if (!isNaN(steps) && steps >= 1 && onMaxStepsChange) {
      onMaxStepsChange(steps);
    }
  };

  const handleAgentCountChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const count = parseInt(event.target.value, 10);
    if (!isNaN(count) && count >= 0 && onAgentCountChange) {
      onAgentCountChange(count);
    }
  };

  const isCollaborative = executionMode === 'collaborative';
  const isPipeline = executionMode === 'advanced_pipeline';

  const isStartDisabled = isConversationActive || (
    isCollaborative
      ? selectedPersonalities.length === 0 || !leadPersonality || numRounds <= 0
      : maxSteps <= 0
  );

  const buttonText = isConversationActive
    ? (isPipeline ? 'Pipeline in Progress...' : 'Conversation in Progress...')
    : (isPipeline ? 'Start Pipeline Session' : 'Start New Conversation');

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
            <option value="collaborative">Collaborative Debate (Multi-Agent Deliberation)</option>
            <option value="advanced_pipeline">Standalone Pipeline (Complexity & Step Planning)</option>
            <option value="tree_of_thought" disabled>
              Tree of Thoughts (Disabled - Issue #36)
            </option>
          </select>
        </div>

        {isCollaborative && (
          <>
            <div>
              <Label className="block text-sm font-medium text-text-primary mb-2">Select Personalities:</Label>
              <div className="space-y-2">
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
            </div>

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
                  {selectedPersonalities.length === 0 ? "Select personalities first" : "Select a lead"}
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
                value={maxSteps}
                onChange={handleMaxStepsChange}
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
                value={agentCount}
                onChange={handleAgentCountChange}
                min="0"
                max="10"
                disabled={isConversationActive}
                className="mt-1 block w-full"
              />
            </div>
          </div>
        )}

        <Button
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
