import React from 'react';
import { ConversationMemoryType } from '../graphql/graphqlTypes';
import { Card } from './ui/Card';

interface ConversationMemoryDisplayProps {
  memory: ConversationMemoryType | null | undefined;
}

export const ConversationMemoryDisplay: React.FC<ConversationMemoryDisplayProps> = ({ memory }) => {
  if (!memory) {
    return <p className="text-gray-500">No conversation memory available.</p>;
  }

  return (
    <Card className="my-4 p-4">
      <h2 className="text-xl font-bold mb-3">Conversation Memory</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <h3 className="text-lg font-semibold">Facts</h3>
          {memory.facts.length > 0 ? (
            <ul className="list-disc pl-5">
              {memory.facts.map((fact, index) => <li key={index} className="text-sm">{fact}</li>)}
            </ul>
          ) : <p className="text-sm text-gray-500">No facts recorded.</p>}
        </div>
        <div>
          <h3 className="text-lg font-semibold">Arguments</h3>
          {memory.arguments.length > 0 ? (
            <ul className="list-disc pl-5">
              {memory.arguments.map((arg, index) => <li key={index} className="text-sm">{arg}</li>)}
            </ul>
          ) : <p className="text-sm text-gray-500">No arguments recorded.</p>}
        </div>
        <div>
          <h3 className="text-lg font-semibold">Decisions</h3>
          {memory.decisions.length > 0 ? (
            <ul className="list-disc pl-5">
              {memory.decisions.map((decision, index) => <li key={index} className="text-sm">{decision}</li>)}
            </ul>
          ) : <p className="text-sm text-gray-500">No decisions recorded.</p>}
        </div>
        <div>
          <h3 className="text-lg font-semibold">To-Do List</h3>
          {memory.to_do_list.length > 0 ? (
            <ul className="list-disc pl-5">
              {memory.to_do_list.map((todo, index) => <li key={index} className="text-sm">{todo}</li>)}
            </ul>
          ) : <p className="text-sm text-gray-500">To-do list is empty.</p>}
        </div>
        <div>
          <h3 className="text-lg font-semibold">Completed Tasks</h3>
          {memory.completed_tasks.length > 0 ? (
            <ul className="list-disc pl-5">
              {memory.completed_tasks.map((task, index) => <li key={index} className="text-sm">{task}</li>)}
            </ul>
          ) : <p className="text-sm text-gray-500">No tasks completed yet.</p>}
        </div>
        <div>
          <h3 className="text-lg font-semibold">Recommended Actions</h3>
          {memory.recommended_actions.length > 0 ? (
            <ul className="list-disc pl-5">
              {memory.recommended_actions.map((action, index) => <li key={index} className="text-sm">{action}</li>)}
            </ul>
          ) : <p className="text-sm text-gray-500">No recommended actions.</p>}
        </div>
      </div>
      <div className="mt-4">
        <p className="text-sm">Rounds Left: <span className="font-semibold">{memory.rounds_left}</span></p>
        {memory.decided_output_type && <p className="text-sm">Decided Output Type: <span className="font-semibold">{memory.decided_output_type}</span></p>}
      </div>
    </Card>
  );
};
