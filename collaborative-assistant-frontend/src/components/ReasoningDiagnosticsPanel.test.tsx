/// <reference types="vitest/globals" />
import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import ReasoningDiagnosticsPanel from './ReasoningDiagnosticsPanel';
import type { RunDiagnosticsType, RunEventType } from '../graphql/graphqlTypes';

describe('ReasoningDiagnosticsPanel', () => {
  const sampleDiagnostics: RunDiagnosticsType = {
    complexityScore: 0.82,
    initialBudget: 20,
    adjustedBudget: 15,
    finalReward: 0.88,
    durationMs: 1240,
    planSteps: [
      {
        stepNumber: 1,
        stepName: 'Architect Service',
        stepDescription: 'Design modular contracts',
        stepExplanation: 'Encapsulate domain models',
        stepOutput: 'Service diagram complete',
        completed: true,
        assignedAgents: ['Strategist', 'Coding Guru'],
        subtasks: [
          {
            subtaskNumber: 1,
            subtaskName: 'Draft interfaces',
            subtaskDescription: 'Create TypeScript types',
            subtaskOutput: 'graphqlTypes.ts',
            completed: true,
          },
        ],
      },
    ],
    reflections: [
      {
        stepNumber: 1,
        stepDescription: 'Service design check',
        thoughts: 'Checked coupling and cohesion',
        content: 'Design passes high-modularity criteria',
        reward: 0.9,
        remainingBudget: 14,
      },
    ],
    agentAttributions: [
      {
        agentName: 'Strategist',
        role: 'lead',
        roundOrStep: 1,
        contributionSummary: 'Proposed decoupling schemas',
      },
    ],
  };

  const sampleEvents: RunEventType[] = [
    {
      eventId: 'evt_1',
      eventType: 'mode_selected',
      timestamp: '2026-10-22T10:00:00Z',
      mode: 'advanced_pipeline',
      summary: 'Pipeline initialized',
    },
  ];

  it('renders nothing when diagnostics and events are absent', () => {
    const { container } = render(<ReasoningDiagnosticsPanel diagnostics={null} events={null} />);
    expect(container.firstChild).toBeNull();
  });

  it('renders complexity badge, reward score, and plan steps', () => {
    render(<ReasoningDiagnosticsPanel diagnostics={sampleDiagnostics} events={sampleEvents} />);
    expect(screen.getByText(/Reasoning Diagnostics & Telemetry/i)).toBeInTheDocument();
    expect(screen.getByText(/High Complexity \(0.82\)/i)).toBeInTheDocument();
    expect(screen.getByText(/Reward: 0.88/i)).toBeInTheDocument();
    expect(screen.getByText(/Step 1: Architect Service/i)).toBeInTheDocument();
    expect(screen.getByText(/Draft interfaces/i)).toBeInTheDocument();
  });

  it('allows switching to Reflections tab and displays thoughts', () => {
    render(<ReasoningDiagnosticsPanel diagnostics={sampleDiagnostics} events={sampleEvents} />);
    const reflectionsTab = screen.getByRole('button', { name: /Steps & Reflections/i });
    fireEvent.click(reflectionsTab);
    expect(screen.getByText(/Step 1 Reflection/i)).toBeInTheDocument();
    expect(screen.getByText(/Checked coupling and cohesion/i)).toBeInTheDocument();
    expect(screen.getByText(/Design passes high-modularity criteria/i)).toBeInTheDocument();
  });

  it('allows switching to Agent Attributions tab and Event Stream tab', () => {
    render(<ReasoningDiagnosticsPanel diagnostics={sampleDiagnostics} events={sampleEvents} />);
    const agentsTab = screen.getByRole('button', { name: /Agent Attributions/i });
    fireEvent.click(agentsTab);
    expect(screen.getByText('Strategist')).toBeInTheDocument();
    expect(screen.getByText(/Proposed decoupling schemas/i)).toBeInTheDocument();

    const eventsTab = screen.getByRole('button', { name: /Event Stream/i });
    fireEvent.click(eventsTab);
    expect(screen.getByText('mode_selected')).toBeInTheDocument();
    expect(screen.getByText(/Pipeline initialized/i)).toBeInTheDocument();
  });

  it('toggles expand/collapse on header button click', () => {
    render(<ReasoningDiagnosticsPanel diagnostics={sampleDiagnostics} events={sampleEvents} />);
    const toggleBtn = screen.getByRole('button', { name: /Reasoning Diagnostics & Telemetry/i });
    expect(screen.getByRole('button', { name: /Execution Plan/i })).toBeInTheDocument();

    fireEvent.click(toggleBtn);
    expect(screen.queryByRole('button', { name: /Execution Plan/i })).not.toBeInTheDocument();

    fireEvent.click(toggleBtn);
    expect(screen.getByRole('button', { name: /Execution Plan/i })).toBeInTheDocument();
  });
});
