import React from 'react';
import type { QuestionType } from '../graphql/graphqlTypes';

interface FeedbackDisplayProps {
  currentQuestions: QuestionType[];
  finalOutput: string | null;
  runConversationLoading: boolean;
}

const FeedbackDisplay: React.FC<FeedbackDisplayProps> = ({
  currentQuestions,
  finalOutput,
  runConversationLoading,
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
          <div className="p-3 bg-green-100/50 border border-green-300 rounded-md text-sm text-green-800 whitespace-pre-wrap">
            {finalOutput}
          </div>
        </div>
      )}
    </>
  );
};

export default FeedbackDisplay;
