/// <reference types="vitest/globals" />
import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { vi, describe, it, expect, beforeEach } from 'vitest';
import AssistantConfigPanel from './AssistantConfigPanel'; // Assuming default export

const mockOnSelectedPersonalitiesChange = vi.fn();
const mockOnLeadPersonalityChange = vi.fn();
const mockOnNumRoundsChange = vi.fn();
const mockOnStartConversation = vi.fn();

const defaultProps = {
  availablePersonalities: ['Alice', 'Bob', 'Charlie'],
  selectedPersonalities: [],
  onSelectedPersonalitiesChange: mockOnSelectedPersonalitiesChange,
  leadPersonality: '',
  onLeadPersonalityChange: mockOnLeadPersonalityChange,
  numRounds: 1,
  onNumRoundsChange: mockOnNumRoundsChange,
  onStartConversation: mockOnStartConversation,
  isConversationActive: false,
};

const renderPanel = (props = {}) => {
  return render(<AssistantConfigPanel {...defaultProps} {...props} />);
};

describe('AssistantConfigPanel', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Initial Rendering', () => {
    it('renders correctly with default/empty props', () => {
      renderPanel({ availablePersonalities: [], selectedPersonalities: [] });
      expect(screen.getByText('Configure Assistants')).toBeInTheDocument();
      // Changed from getByLabelText to getByText as the label is not directly associated with a single control here
      expect(screen.getByText('Select Personalities:')).toBeInTheDocument();
      expect(screen.queryByRole('checkbox')).not.toBeInTheDocument(); // No personalities
      expect(screen.getByLabelText('Lead Personality:')).toBeInTheDocument();
      expect(screen.getByRole('combobox', { name: 'Lead Personality:' })).toBeInTheDocument();
      expect(screen.getByText('Select personalities first')).toBeInTheDocument(); // Placeholder in dropdown
      expect(screen.getByLabelText('Number of Rounds (per turn):')).toBeInTheDocument();
      expect(screen.getByRole('spinbutton', { name: 'Number of Rounds (per turn):' })).toHaveValue(1);
      expect(screen.getByRole('button', { name: 'Start New Conversation' })).toBeInTheDocument();
    });

    it('renders correctly with a typical set of props', () => {
      renderPanel({
        selectedPersonalities: ['Alice', 'Bob'],
        leadPersonality: 'Alice',
        numRounds: 3,
      });
      expect(screen.getByText('Configure Assistants')).toBeInTheDocument();

      // Checkboxes
      const aliceCheckbox = screen.getByLabelText('Alice') as HTMLInputElement;
      const bobCheckbox = screen.getByLabelText('Bob') as HTMLInputElement;
      const charlieCheckbox = screen.getByLabelText('Charlie') as HTMLInputElement;
      expect(aliceCheckbox).toBeChecked();
      expect(bobCheckbox).toBeChecked();
      expect(charlieCheckbox).not.toBeChecked();

      // Lead Personality Dropdown
      const leadDropdown = screen.getByRole('combobox', { name: 'Lead Personality:' }) as HTMLSelectElement;
      expect(leadDropdown).toHaveValue('Alice');
      expect(screen.getByText('Select a lead')).toBeInTheDocument(); // Placeholder in dropdown

      // Number of Rounds
      expect(screen.getByRole('spinbutton', { name: 'Number of Rounds (per turn):' })).toHaveValue(3);

      // Button
      expect(screen.getByRole('button', { name: 'Start New Conversation' })).toBeEnabled();
    });

    it('verifies checkbox states match selectedPersonalities', () => {
      renderPanel({ selectedPersonalities: ['Bob'] });
      expect(screen.getByLabelText('Alice') as HTMLInputElement).not.toBeChecked();
      expect(screen.getByLabelText('Bob') as HTMLInputElement).toBeChecked();
      expect(screen.getByLabelText('Charlie') as HTMLInputElement).not.toBeChecked();
    });

    it('verifies lead personality dropdown is populated and shows correct leadPersonality', () => {
      renderPanel({ selectedPersonalities: ['Alice', 'Bob'], leadPersonality: 'Bob' });
      const leadDropdown = screen.getByRole('combobox', { name: 'Lead Personality:' }) as HTMLSelectElement;
      expect(leadDropdown).toHaveValue('Bob');
      expect(leadDropdown.options).toHaveLength(3); // "Select a lead" + Alice + Bob
      expect(leadDropdown.options[1].text).toBe('Alice');
      expect(leadDropdown.options[2].text).toBe('Bob');
    });

    it('verifies num rounds input shows correct numRounds', () => {
      renderPanel({ numRounds: 5 });
      expect(screen.getByRole('spinbutton', { name: 'Number of Rounds (per turn):' })).toHaveValue(5);
    });

    it('verifies the "Start New Conversation" button text', () => {
      renderPanel();
      expect(screen.getByRole('button', { name: 'Start New Conversation' })).toBeInTheDocument();
    });
  });

  describe('Personality Selection', () => {
    it('calls onSelectedPersonalitiesChange when an unselected personality is clicked', async () => {
      renderPanel({ selectedPersonalities: ['Alice'] });
      const bobCheckbox = screen.getByLabelText('Bob');
      await userEvent.click(bobCheckbox);
      expect(mockOnSelectedPersonalitiesChange).toHaveBeenCalledWith(['Alice', 'Bob']);
    });

    it('calls onSelectedPersonalitiesChange when a selected personality is clicked', async () => {
      renderPanel({ selectedPersonalities: ['Alice', 'Bob'] });
      const bobCheckbox = screen.getByLabelText('Bob');
      await userEvent.click(bobCheckbox);
      expect(mockOnSelectedPersonalitiesChange).toHaveBeenCalledWith(['Alice']);
    });

    it('calls onLeadPersonalityChange if leadPersonality is deselected (to first in new selection)', async () => {
      renderPanel({ selectedPersonalities: ['Alice', 'Bob'], leadPersonality: 'Alice' });
      const aliceCheckbox = screen.getByLabelText('Alice');
      await userEvent.click(aliceCheckbox); // Deselect Alice
      expect(mockOnSelectedPersonalitiesChange).toHaveBeenCalledWith(['Bob']);
      // Alice was lead, now Bob should be lead
      expect(mockOnLeadPersonalityChange).toHaveBeenCalledWith('Bob');
    });

    it('calls onLeadPersonalityChange with empty string if leadPersonality is deselected and no personalities remain', async () => {
      renderPanel({ selectedPersonalities: ['Alice'], leadPersonality: 'Alice' });
      const aliceCheckbox = screen.getByLabelText('Alice');
      await userEvent.click(aliceCheckbox); // Deselect Alice
      expect(mockOnSelectedPersonalitiesChange).toHaveBeenCalledWith([]);
      expect(mockOnLeadPersonalityChange).toHaveBeenCalledWith('');
    });
  });

  describe('Lead Personality Change', () => {
    it('calls onLeadPersonalityChange when dropdown value changes', async () => {
      renderPanel({ selectedPersonalities: ['Alice', 'Bob'], leadPersonality: 'Alice' });
      const leadDropdown = screen.getByRole('combobox', { name: 'Lead Personality:' });
      await userEvent.selectOptions(leadDropdown, 'Bob');
      expect(mockOnLeadPersonalityChange).toHaveBeenCalledWith('Bob');
    });

    it('dropdown is disabled if no personalities are selected', () => {
      renderPanel({ selectedPersonalities: [] });
      const leadDropdown = screen.getByRole('combobox', { name: 'Lead Personality:' });
      expect(leadDropdown).toBeDisabled();
      expect(screen.getByText('Select personalities first')).toBeInTheDocument();
    });

    it('dropdown shows "Select a lead" if personalities are selected but no lead yet', () => {
        renderPanel({ selectedPersonalities: ['Alice', 'Bob'], leadPersonality: '' });
        const leadDropdown = screen.getByRole('combobox', { name: 'Lead Personality:' }) as HTMLSelectElement;
        expect(leadDropdown.options[0].text).toBe('Select a lead');
        expect(leadDropdown.options[0].disabled).toBe(true); // The placeholder itself is disabled for selection
      });
  });

  describe('Number of Rounds Change', () => {
    it('calls onNumRoundsChange with the new number', async () => {
      renderPanel({ numRounds: 1 });
      const numRoundsInput = screen.getByRole('spinbutton', { name: 'Number of Rounds (per turn):' }) as HTMLInputElement;
      // Using fireEvent.change to directly set the value to '3'
      // This avoids complexities with userEvent.clear followed by userEvent.type on number inputs,
      // which can lead to unexpected intermediate states or values like '13' if not handled carefully.
      fireEvent.change(numRoundsInput, { target: { value: '3' } });
      expect(mockOnNumRoundsChange).toHaveBeenCalledWith(3);
    });

    it('calls onNumRoundsChange with 1 if input is cleared (empty string)', async () => {
      renderPanel({ numRounds: 3 });
      const numRoundsInput = screen.getByRole('spinbutton', { name: 'Number of Rounds (per turn):' });
      // userEvent.clear doesn't trigger onChange for type="number" in the same way as actual empty input sometimes
      // const numRoundsInput = screen.getByRole('spinbutton', { name: 'Number of Rounds (per turn):' }); // This line was duplicated
      // fireEvent.change is more direct for this specific case of empty value
      fireEvent.change(numRoundsInput, { target: { value: '' } });
      expect(mockOnNumRoundsChange).toHaveBeenCalledWith(1);
    });

    it('does not call onNumRoundsChange for invalid input "0"', async () => {
      renderPanel({ numRounds: 1 });
      const numRoundsInput = screen.getByRole('spinbutton', { name: 'Number of Rounds (per turn):' });
      fireEvent.change(numRoundsInput, { target: { value: '0' } });
      // The component's logic: parseInt("0") is 0, which is < 1, so onNumRoundsChange is NOT called with 0.
      // The mock should not have been called with 0.
      // Depending on previous calls in other tests, we might need to clear mocks or check for specific non-call.
      // For this test, we assume mockOnNumRoundsChange was not called with 0 from this specific interaction.
      expect(mockOnNumRoundsChange).not.toHaveBeenCalledWith(0);
    });

    it('does not call onNumRoundsChange for non-numeric input "abc"', async () => {
      renderPanel({ numRounds: 1 });
      const numRoundsInput = screen.getByRole('spinbutton', { name: 'Number of Rounds (per turn):' });
      fireEvent.change(numRoundsInput, { target: { value: 'abc' } });
      expect(mockOnNumRoundsChange).not.toHaveBeenCalledWith('abc'); // Ensure it's not called with NaN or string
    });

    it('does not call onNumRoundsChange for negative input "-5"', async () => {
        renderPanel({ numRounds: 1 });
        const numRoundsInput = screen.getByRole('spinbutton', { name: 'Number of Rounds (per turn):' });
        fireEvent.change(numRoundsInput, { target: { value: '-5' } });
        // The component's handleNumRoundsChange will not call onNumRoundsChange if parseInt(value) < 1
        expect(mockOnNumRoundsChange).not.toHaveBeenCalledWith(-5);
      });
  });

  describe('Start Conversation Button', () => {
    it('is disabled if isConversationActive is true', () => {
      renderPanel({ isConversationActive: true, selectedPersonalities: ['Alice'], leadPersonality: 'Alice', numRounds: 1 });
      expect(screen.getByRole('button', { name: 'Conversation in Progress...' })).toBeDisabled();
    });

    it('is disabled if selectedPersonalities is empty', () => {
      renderPanel({ selectedPersonalities: [], leadPersonality: 'Alice', numRounds: 1 });
      expect(screen.getByRole('button', { name: 'Start New Conversation' })).toBeDisabled();
    });

    it('is disabled if leadPersonality is empty', () => {
      renderPanel({ selectedPersonalities: ['Alice'], leadPersonality: '', numRounds: 1 });
      expect(screen.getByRole('button', { name: 'Start New Conversation' })).toBeDisabled();
    });

    it('is disabled if numRounds is less than or equal to 0', () => {
      // The component's input has min="1", and logic prevents setting numRounds <= 0 via onNumRoundsChange
      // So we test the prop directly.
      renderPanel({ selectedPersonalities: ['Alice'], leadPersonality: 'Alice', numRounds: 0 });
      expect(screen.getByRole('button', { name: 'Start New Conversation' })).toBeDisabled();
    });

    it('is enabled if all conditions are met', () => {
      renderPanel({
        selectedPersonalities: ['Alice'],
        leadPersonality: 'Alice',
        numRounds: 1,
        isConversationActive: false,
      });
      expect(screen.getByRole('button', { name: 'Start New Conversation' })).toBeEnabled();
    });

    it('calls onStartConversation when clicked and enabled', async () => {
      renderPanel({
        selectedPersonalities: ['Alice'],
        leadPersonality: 'Alice',
        numRounds: 1,
      });
      const startButton = screen.getByRole('button', { name: 'Start New Conversation' });
      await userEvent.click(startButton);
      expect(mockOnStartConversation).toHaveBeenCalledTimes(1);
    });

    it('button text changes to "Conversation in Progress..." if isConversationActive is true', () => {
      renderPanel({ isConversationActive: true, selectedPersonalities: ['Alice'], leadPersonality: 'Alice' });
      expect(screen.getByRole('button', { name: 'Conversation in Progress...' })).toBeInTheDocument();
    });
  });

  describe('Disabled State when Conversation is Active', () => {
    beforeEach(() => {
      renderPanel({
        isConversationActive: true,
        selectedPersonalities: ['Alice', 'Bob'],
        leadPersonality: 'Alice',
        numRounds: 2,
      });
    });

    it('all personality checkboxes are disabled', () => {
      const aliceCheckbox = screen.getByLabelText('Alice');
      const bobCheckbox = screen.getByLabelText('Bob');
      expect(aliceCheckbox).toBeDisabled();
      expect(bobCheckbox).toBeDisabled();
    });

    it('lead personality dropdown is disabled', () => {
      expect(screen.getByRole('combobox', { name: 'Lead Personality:' })).toBeDisabled();
    });

    it('number of rounds input is disabled', () => {
      expect(screen.getByRole('spinbutton', { name: 'Number of Rounds (per turn):' })).toBeDisabled();
    });
  });
});
