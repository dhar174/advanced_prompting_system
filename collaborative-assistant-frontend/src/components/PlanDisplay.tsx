import React from 'react';
import { PlanType, PlanStepType, SubtaskType } from '../graphql/graphqlTypes';
import { Card } from './ui/Card'; // Assuming a Card component exists

interface PlanDisplayProps {
  plan: PlanType | null | undefined;
}

const SubtaskItem: React.FC<{ subtask: SubtaskType }> = ({ subtask }) => (
  <div className="ml-4 p-2 border-l border-gray-300">
    <p className="font-semibold">{subtask.subtask_name} (Step {subtask.subtask_number})</p>
    <p className="text-sm text-gray-600">{subtask.subtask_description}</p>
    {subtask.subtask_explanation && <p className="text-xs italic text-gray-500 mt-1">Explanation: {subtask.subtask_explanation}</p>}
    {subtask.subtask_output && <p className="text-xs text-green-700 mt-1">Output: {subtask.subtask_output}</p>}
    <p className={`text-xs ${subtask.completed ? 'text-green-500' : 'text-red-500'}`}>
      {subtask.completed ? 'Completed' : 'Pending'}
    </p>
    {subtask.subtasks && subtask.subtasks.length > 0 && (
      <div className="mt-2">
        <p className="text-xs font-medium">Sub-tasks:</p>
        {subtask.subtasks.map(st => <SubtaskItem key={st.subtask_number} subtask={st} />)}
      </div>
    )}
  </div>
);

const PlanStepItem: React.FC<{ step: PlanStepType }> = ({ step }) => (
  <Card className="mb-4 p-4">
    <h3 className="text-lg font-semibold">{step.step_name} (Step {step.step_number})</h3>
    <p className="text-sm text-gray-700">{step.step_description}</p>
    {step.step_explanation && <p className="text-xs italic text-gray-500 mt-1">Explanation: {step.step_explanation}</p>}
    {step.step_output && <p className="text-sm text-green-700 mt-1">Output: {step.step_output}</p>}
    <p className={`mt-2 font-medium ${step.completed ? 'text-green-600' : 'text-red-600'}`}>
      Status: {step.completed ? 'Completed' : 'Pending'}
    </p>
    {step.subtasks && step.subtasks.length > 0 && (
      <div className="mt-3">
        <h4 className="text-md font-medium">Subtasks:</h4>
        {step.subtasks.map(subtask => (
          <SubtaskItem key={subtask.subtask_number} subtask={subtask} />
        ))}
      </div>
    )}
  </Card>
);

export const PlanDisplay: React.FC<PlanDisplayProps> = ({ plan }) => {
  if (!plan || !plan.steps || plan.steps.length === 0) {
    return <p className="text-gray-500">No plan available.</p>;
  }

  return (
    <div className="my-4">
      <h2 className="text-xl font-bold mb-3">AI Execution Plan</h2>
      {plan.steps.map(step => (
        <PlanStepItem key={step.step_number} step={step} />
      ))}
    </div>
  );
};
