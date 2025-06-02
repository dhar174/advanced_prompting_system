import React from 'react';
import { ProcessingStatusType } from '../graphql/graphqlTypes';
import { Card } from './ui/Card';

interface ProcessingStatusDisplayProps {
  status: ProcessingStatusType | null | undefined;
}

export const ProcessingStatusDisplay: React.FC<ProcessingStatusDisplayProps> = ({ status }) => {
  if (!status) {
    return <p className="text-gray-500">No processing status available.</p>;
  }

  return (
    <Card className="my-4 p-4">
      <h2 className="text-xl font-bold mb-3">Processing Status</h2>
      <div className="space-y-2">
        <p className="text-sm">Current Round: <span className="font-semibold">{status.current_round} / {status.total_rounds}</span></p>
        <p className="text-sm">Current Step: <span className="font-semibold">{status.current_step}</span></p>
        <div>
          <p className="text-sm">Progress: <span className="font-semibold">{status.progress_percentage.toFixed(1)}%</span></p>
          <div className="w-full bg-gray-200 rounded-full h-2.5 dark:bg-gray-700 mt-1">
            <div className="bg-blue-600 h-2.5 rounded-full" style={{ width: `${status.progress_percentage}%` }}></div>
          </div>
        </div>
        {status.estimated_time_remaining !== null && status.estimated_time_remaining !== undefined && (
          <p className="text-sm">Estimated Time Remaining: <span className="font-semibold">{status.estimated_time_remaining} seconds</span></p>
        )}
      </div>
    </Card>
  );
};
