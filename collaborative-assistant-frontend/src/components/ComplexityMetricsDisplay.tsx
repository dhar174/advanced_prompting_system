import React from 'react';
import { ComplexityMetricsType } from '../graphql/graphqlTypes';
import { Card } from './ui/Card';

interface ComplexityMetricsDisplayProps {
  metrics: ComplexityMetricsType | null | undefined;
}

export const ComplexityMetricsDisplay: React.FC<ComplexityMetricsDisplayProps> = ({ metrics }) => {
  if (!metrics) {
    return <p className="text-gray-500">No complexity metrics available.</p>;
  }

  return (
    <Card className="my-4 p-4">
      <h2 className="text-xl font-bold mb-3">Complexity Metrics</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <p className="text-sm">Overall Score: <span className="font-semibold">{metrics.overall_score.toFixed(2)}</span></p>
        <p className="text-sm">Reasoning Depth: <span className="font-semibold">{metrics.reasoning_depth.toFixed(2)}</span></p>
        <p className="text-sm">Solution Complexity: <span className="font-semibold">{metrics.solution_complexity.toFixed(2)}</span></p>
        <p className="text-sm">Collaboration Intensity: <span className="font-semibold">{metrics.collaboration_intensity.toFixed(2)}</span></p>
        <p className="text-sm">Confidence Level: <span className="font-semibold">{(metrics.confidence_level * 100).toFixed(1)}%</span></p>
      </div>
    </Card>
  );
};
