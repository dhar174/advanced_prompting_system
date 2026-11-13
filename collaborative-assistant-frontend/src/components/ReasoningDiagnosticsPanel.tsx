import React, { useState } from 'react';
import type { RunDiagnosticsType, RunEventType } from '../graphql/graphqlTypes';

export interface ReasoningDiagnosticsPanelProps {
  diagnostics?: RunDiagnosticsType | null;
  events?: RunEventType[] | null;
}

export const ReasoningDiagnosticsPanel: React.FC<ReasoningDiagnosticsPanelProps> = ({
  diagnostics,
  events,
}) => {
  const [activeTab, setActiveTab] = useState<'plan' | 'reflections' | 'agents' | 'events'>('plan');
  const [isExpanded, setIsExpanded] = useState(true);

  if (!diagnostics && (!events || events.length === 0)) {
    return null;
  }

  const complexityScore = diagnostics?.complexityScore;
  const complexityBadge =
    complexityScore !== null && complexityScore !== undefined
      ? complexityScore < 0.4
        ? { text: 'Low Complexity', color: 'bg-emerald-100 text-emerald-800' }
        : complexityScore < 0.7
        ? { text: 'Moderate Complexity', color: 'bg-amber-100 text-amber-800' }
        : { text: 'High Complexity', color: 'bg-rose-100 text-rose-800' }
      : null;

  return (
    <div className="mt-4 border border-slate-200 rounded-lg overflow-hidden bg-white shadow-sm">
      {/* Header bar */}
      <div className="px-4 py-3 bg-slate-50 border-b border-slate-200 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <button
            type="button"
            onClick={() => setIsExpanded(!isExpanded)}
            className="text-sm font-semibold text-slate-800 flex items-center space-x-1.5 hover:text-blue-600"
          >
            <span>{isExpanded ? '▼' : '▶'}</span>
            <span>Reasoning Diagnostics & Telemetry</span>
          </button>
          {complexityBadge && (
            <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${complexityBadge.color}`}>
              {complexityBadge.text} ({complexityScore?.toFixed(2)})
            </span>
          )}
          {diagnostics?.finalReward !== null && diagnostics?.finalReward !== undefined && (
            <span className="text-xs px-2 py-0.5 rounded-full bg-blue-100 text-blue-800 font-medium">
              Reward: {diagnostics.finalReward.toFixed(2)}
            </span>
          )}
          {diagnostics?.durationMs && (
            <span className="text-xs text-slate-500">
              {diagnostics.durationMs.toFixed(0)}ms
            </span>
          )}
        </div>
        <div className="text-xs text-slate-500">
          {diagnostics?.initialBudget && (
            <span>Budget: {diagnostics.adjustedBudget ?? diagnostics.initialBudget}</span>
          )}
        </div>
      </div>

      {isExpanded && (
        <div className="p-4 space-y-4">
          {/* Tabs */}
          <div className="flex border-b border-slate-200 space-x-4 text-xs font-medium">
            <button
              type="button"
              onClick={() => setActiveTab('plan')}
              className={`pb-2 border-b-2 transition-colors ${
                activeTab === 'plan'
                  ? 'border-blue-600 text-blue-600 font-semibold'
                  : 'border-transparent text-slate-500 hover:text-slate-700'
              }`}
            >
              Execution Plan ({diagnostics?.planSteps?.length ?? 0})
            </button>
            <button
              type="button"
              onClick={() => setActiveTab('reflections')}
              className={`pb-2 border-b-2 transition-colors ${
                activeTab === 'reflections'
                  ? 'border-blue-600 text-blue-600 font-semibold'
                  : 'border-transparent text-slate-500 hover:text-slate-700'
              }`}
            >
              Steps & Reflections ({diagnostics?.reflections?.length ?? 0})
            </button>
            <button
              type="button"
              onClick={() => setActiveTab('agents')}
              className={`pb-2 border-b-2 transition-colors ${
                activeTab === 'agents'
                  ? 'border-blue-600 text-blue-600 font-semibold'
                  : 'border-transparent text-slate-500 hover:text-slate-700'
              }`}
            >
              Agent Attributions ({diagnostics?.agentAttributions?.length ?? 0})
            </button>
            <button
              type="button"
              onClick={() => setActiveTab('events')}
              className={`pb-2 border-b-2 transition-colors ${
                activeTab === 'events'
                  ? 'border-blue-600 text-blue-600 font-semibold'
                  : 'border-transparent text-slate-500 hover:text-slate-700'
              }`}
            >
              Event Stream ({events?.length ?? 0})
            </button>
          </div>

          {/* Tab 1: Execution Plan */}
          {activeTab === 'plan' && (
            <div className="space-y-3">
              {diagnostics?.planSteps && diagnostics.planSteps.length > 0 ? (
                diagnostics.planSteps.map((step) => (
                  <div
                    key={step.stepNumber}
                    className="p-3 border border-slate-200 rounded-md bg-slate-50 text-xs space-y-1.5"
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-2">
                        <span className="font-bold text-slate-800">
                          Step {step.stepNumber}: {step.stepName}
                        </span>
                        <span
                          className={`px-1.5 py-0.5 rounded text-[10px] font-semibold ${
                            step.completed
                              ? 'bg-emerald-100 text-emerald-800'
                              : 'bg-amber-100 text-amber-800'
                          }`}
                        >
                          {step.completed ? 'Completed' : 'In Progress'}
                        </span>
                      </div>
                      {step.assignedAgents.length > 0 && (
                        <div className="flex items-center space-x-1">
                          {step.assignedAgents.map((agent) => (
                            <span
                              key={agent}
                              className="px-1.5 py-0.5 bg-blue-50 text-blue-700 rounded text-[10px] border border-blue-200"
                            >
                              {agent}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                    <p className="text-slate-600">{step.stepDescription}</p>
                    {step.stepOutput && (
                      <div className="mt-1 p-2 bg-white rounded border border-slate-200 text-slate-700 font-mono text-[11px]">
                        {step.stepOutput}
                      </div>
                    )}
                    {step.subtasks && step.subtasks.length > 0 && (
                      <div className="pl-3 mt-2 border-l-2 border-slate-300 space-y-1">
                        <span className="font-semibold text-slate-700">Subtasks:</span>
                        {step.subtasks.map((sub) => (
                          <div key={sub.subtaskNumber} className="flex items-center space-x-2">
                            <span className={sub.completed ? 'text-emerald-600' : 'text-slate-400'}>
                              {sub.completed ? '✓' : '○'}
                            </span>
                            <span className={sub.completed ? 'line-through text-slate-500' : 'text-slate-700'}>
                              {sub.subtaskName}
                            </span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                ))
              ) : (
                <p className="text-xs text-slate-500 italic">No structured plan generated for this run.</p>
              )}
            </div>
          )}

          {/* Tab 2: Reflections */}
          {activeTab === 'reflections' && (
            <div className="space-y-3">
              {diagnostics?.reflections && diagnostics.reflections.length > 0 ? (
                diagnostics.reflections.map((refl, idx) => (
                  <div
                    key={idx}
                    className="p-3 border border-slate-200 rounded-md bg-slate-50 text-xs space-y-1.5"
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-bold text-slate-800">
                        Step {refl.stepNumber} Reflection
                      </span>
                      <span
                        className={`px-1.5 py-0.5 rounded text-[10px] font-semibold ${
                          refl.reward >= 0.7
                            ? 'bg-emerald-100 text-emerald-800'
                            : refl.reward >= 0.4
                            ? 'bg-amber-100 text-amber-800'
                            : 'bg-rose-100 text-rose-800'
                        }`}
                      >
                        Reward: {refl.reward.toFixed(2)}
                      </span>
                    </div>
                    {refl.thoughts && (
                      <div className="p-2 bg-amber-50 text-amber-900 rounded border border-amber-200 text-[11px] font-mono whitespace-pre-wrap">
                        {refl.thoughts}
                      </div>
                    )}
                    <p className="text-slate-700">{refl.content}</p>
                    <div className="text-[10px] text-slate-400 text-right">
                      Remaining budget: {refl.remainingBudget}
                    </div>
                  </div>
                ))
              ) : (
                <p className="text-xs text-slate-500 italic">No reflections recorded for this run.</p>
              )}
            </div>
          )}

          {/* Tab 3: Agent Attributions */}
          {activeTab === 'agents' && (
            <div className="space-y-2">
              {diagnostics?.agentAttributions && diagnostics.agentAttributions.length > 0 ? (
                diagnostics.agentAttributions.map((attr, idx) => (
                  <div
                    key={idx}
                    className="p-2.5 border border-slate-200 rounded-md bg-slate-50 text-xs flex items-start justify-between"
                  >
                    <div className="space-y-0.5">
                      <div className="flex items-center space-x-2">
                        <span className="font-semibold text-slate-800">{attr.agentName}</span>
                        <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-200 text-slate-600">
                          {attr.role}
                        </span>
                      </div>
                      <p className="text-slate-600">{attr.contributionSummary}</p>
                    </div>
                    <span className="text-[10px] text-slate-400">Round/Step {attr.roundOrStep}</span>
                  </div>
                ))
              ) : (
                <p className="text-xs text-slate-500 italic">No agent attributions recorded.</p>
              )}
            </div>
          )}

          {/* Tab 4: Event Stream */}
          {activeTab === 'events' && (
            <div className="space-y-1.5">
              {events && events.length > 0 ? (
                events.map((evt) => (
                  <div
                    key={evt.eventId}
                    className="p-2 border border-slate-200 rounded bg-white text-xs flex items-center justify-between"
                  >
                    <div className="flex items-center space-x-2">
                      <span className="font-mono text-[10px] text-slate-400">
                        {new Date(evt.timestamp).toLocaleTimeString()}
                      </span>
                      <span className="font-semibold text-slate-700">{evt.eventType}</span>
                      <span className="text-slate-500">— {evt.summary}</span>
                    </div>
                    <span className="text-[10px] bg-slate-100 text-slate-600 px-1 rounded">
                      {evt.mode}
                    </span>
                  </div>
                ))
              ) : (
                <p className="text-xs text-slate-500 italic">No telemetry events emitted.</p>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default ReasoningDiagnosticsPanel;
