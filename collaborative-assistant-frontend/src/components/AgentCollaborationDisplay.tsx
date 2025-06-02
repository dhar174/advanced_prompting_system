import React from 'react';
import { AgentCollaborationType } from '../graphql/graphqlTypes';
import { Card } from './ui/Card';

interface AgentCollaborationDisplayProps {
  collaboration: AgentCollaborationType[];
}

export const AgentCollaborationDisplay: React.FC<AgentCollaborationDisplayProps> = ({ collaboration }) => {
  if (!collaboration || collaboration.length === 0) {
    return <p className="text-gray-500">No agent collaboration data available.</p>;
  }

  return (
    <Card className="my-4 p-4">
      <h2 className="text-xl font-bold mb-3">Agent Collaboration</h2>
      <div className="space-y-4">
        {collaboration.map((agent, index) => (
          <div key={index} className="p-3 border rounded-md">
            <h3 className="text-lg font-semibold">{agent.agent_name}</h3>
            <p className="text-sm">Priority Score: {agent.priority_score.toFixed(2)}</p>
            {agent.contributions.length > 0 && (
              <div>
                <h4 className="text-md font-medium mt-1">Contributions:</h4>
                <ul className="list-disc pl-5 text-sm">
                  {agent.contributions.map((contrib, i) => <li key={i}>{contrib}</li>)}
                </ul>
              </div>
            )}
            {agent.votes_cast.length > 0 && (
              <div>
                <h4 className="text-md font-medium mt-1">Votes Cast:</h4>
                <ul className="list-disc pl-5 text-sm">
                  {agent.votes_cast.map((vote, i) => <li key={i}>{vote}</li>)}
                </ul>
              </div>
            )}
            {agent.questions_asked.length > 0 && (
              <div>
                <h4 className="text-md font-medium mt-1">Questions Asked:</h4>
                <ul className="list-disc pl-5 text-sm">
                  {agent.questions_asked.map((question, i) => <li key={i}>{question}</li>)}
                </ul>
              </div>
            )}
          </div>
        ))}
      </div>
    </Card>
  );
};
