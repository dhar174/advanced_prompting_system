import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/Card';
import { Button } from './ui/Button';
import { Input } from './ui/Input';
import { Label } from './ui/Label';

interface AssistantConfigPanelProps {
  availablePersonalities: string[];
  selectedPersonalities: string[];
  onSelectedPersonalitiesChange: (personalities: string[]) => void;
  leadPersonality: string;
  onLeadPersonalityChange: (personality: string) => void;
  numRounds: number;
  onNumRoundsChange: (rounds: number) => void;
  onStartConversation: () => void;
  isConversationActive: boolean;
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
    if (!isNaN(rounds) && rounds >= 1) { // Ensure rounds are positive
      onNumRoundsChange(rounds);
    } else if (event.target.value === '') {
        onNumRoundsChange(1); // Or some default / allow empty temporarily
    }
  };

  return (
    <Card className="w-full max-w-md bg-surface-card">
      <CardHeader>
        <CardTitle className="text-text-primary">Configure Assistants</CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
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

        <Button
          onClick={onStartConversation}
          disabled={isConversationActive || selectedPersonalities.length === 0 || !leadPersonality || numRounds <= 0}
          className="w-full"
          variant="primary"
        >
          {isConversationActive ? 'Conversation in Progress...' : 'Start New Conversation'}
        </Button>
      </CardContent>
    </Card>
  );
};

export default AssistantConfigPanel;
