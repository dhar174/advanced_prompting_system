import React from 'react';
import type {
  QuestionType,
  OutputTypeInfo,
  RunDiagnosticsType,
  RunEventType,
} from '../graphql/graphqlTypes';
import StructuredOutputRenderer from './StructuredOutputRenderer';
import ReasoningDiagnosticsPanel from './ReasoningDiagnosticsPanel';

export interface FeedbackDisplayProps {
  currentQuestions: QuestionType[];
  finalOutput: string | null;
  runConversationLoading: boolean;
  outputType?: OutputTypeInfo | null;
  diagnostics?: RunDiagnosticsType | null;
  events?: RunEventType[] | null;
}

const FeedbackDisplay: React.FC<FeedbackDisplayProps> = ({
  currentQuestions,
  finalOutput,
  runConversationLoading,
  outputType,
  diagnostics,
  events,
}) => {
  return (
    <>
      {currentQuestions.length > 0 && !runConversationLoading && (
        <div className="mt-4 p-4 border-t border-slate-200">
          <h3 className="text-md font-semibold mb-2 text-text-primary">Questions for Feedback:</h3>
          {currentQuestions.map((q, i) => (
            <div key={i} className="p-3 bg-yellow-100/50 border border-yellow-300 rounded-md mb-2 text-sm text-yellow-800">
              <p className="font-medium">{q.assistant}:</p>
              <p>{q.question}</p>
            </div>
          ))}
        </div>
      )}
      {finalOutput && !runConversationLoading && (
        <div className="mt-4 p-4 border-t border-slate-200">
          <h3 className="text-md font-semibold mb-2 text-text-primary">Final Output:</h3>
          <StructuredOutputRenderer content={finalOutput} outputType={outputType} />
        </div>
      )}
      {!runConversationLoading && (
        <ReasoningDiagnosticsPanel diagnostics={diagnostics} events={events} />
      )}
    </>
  );
};

export default FeedbackDisplay;
