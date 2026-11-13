import React from 'react';
import type {
  ArtifactRefType,
  QuestionType,
  OutputTypeInfo,
  RunDiagnosticsType,
  RunEventType,
  RunStatus,
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
  runStatus?: RunStatus | null;
  stopReason?: string | null;
  errorMessage?: string | null;
  artifacts?: ArtifactRefType[] | null;
}

const NON_COMPLETED_BANNERS: Record<
  Exclude<RunStatus, 'completed'>,
  { title: string; style: string; role: 'alert' | 'status' }
> = {
  gated: {
    title: 'Execution Mode Gated',
    style: 'bg-amber-50 border-amber-300 text-amber-900',
    role: 'alert',
  },
  error: {
    title: 'Execution Failed',
    style: 'bg-rose-50 border-rose-300 text-rose-900',
    role: 'alert',
  },
  timeout: {
    title: 'Execution Timed Out',
    style: 'bg-rose-50 border-rose-300 text-rose-900',
    role: 'alert',
  },
  budget_exhausted: {
    title: 'Step Budget Exhausted (Partial Result)',
    style: 'bg-amber-50 border-amber-300 text-amber-900',
    role: 'status',
  },
  incomplete: {
    title: 'Execution Incomplete',
    style: 'bg-amber-50 border-amber-300 text-amber-900',
    role: 'status',
  },
  cancelled: {
    title: 'Execution Cancelled',
    style: 'bg-slate-100 border-slate-300 text-slate-800',
    role: 'status',
  },
};

const FeedbackDisplay: React.FC<FeedbackDisplayProps> = ({
  currentQuestions,
  finalOutput,
  runConversationLoading,
  outputType,
  diagnostics,
  events,
  runStatus,
  stopReason,
  errorMessage,
  artifacts,
}) => {
  const nonCompletedBanner =
    runStatus && runStatus !== 'completed' ? NON_COMPLETED_BANNERS[runStatus] : null;

  const isSuccessOutput =
    Boolean(finalOutput) &&
    (runStatus === 'completed' || (runStatus == null && !errorMessage));

  const isPartialOutput =
    Boolean(finalOutput) &&
    (runStatus === 'budget_exhausted' || runStatus === 'incomplete' || runStatus === 'timeout') &&
    finalOutput !== errorMessage;

  return (
    <>
      {nonCompletedBanner && !runConversationLoading && (
        <div
          role={nonCompletedBanner.role}
          className={`mt-4 p-3 border rounded-md text-sm space-y-1 ${nonCompletedBanner.style}`}
        >
          <div className="flex items-center justify-between">
            <strong className="font-semibold">{nonCompletedBanner.title}</strong>
            {stopReason && (
              <span className="text-xs font-mono opacity-80">reason: {stopReason}</span>
            )}
          </div>
          {errorMessage && <p className="text-xs">{errorMessage}</p>}
        </div>
      )}

      {currentQuestions.length > 0 && !runConversationLoading && (
        <div className="mt-4 p-4 border-t border-slate-200">
          <h3 className="text-md font-semibold text-text-primary">
            Clarifying Questions Raised (Read-Only):
          </h3>
          <p className="text-xs text-text-secondary mb-2">
            Mid-run pause/resume is not currently supported; start a follow-up turn or session to address these questions.
          </p>
          {currentQuestions.map((q, i) => (
            <div key={i} className="p-3 bg-yellow-100/50 border border-yellow-300 rounded-md mb-2 text-sm text-yellow-800">
              <p className="font-medium">{q.assistant}:</p>
              <p>{q.question}</p>
            </div>
          ))}
        </div>
      )}

      {isSuccessOutput && finalOutput && !runConversationLoading && (
        <div className="mt-4 p-4 border-t border-slate-200">
          <h3 className="text-md font-semibold mb-2 text-text-primary">Final Output:</h3>
          <StructuredOutputRenderer content={finalOutput} outputType={outputType} />
        </div>
      )}

      {isPartialOutput && finalOutput && !runConversationLoading && (
        <div className="mt-4 p-4 border border-amber-300 bg-amber-50/70 rounded-md">
          <h3 className="text-md font-semibold mb-2 text-amber-900">Partial Output (Unverified / Incomplete):</h3>
          <pre className="text-sm whitespace-pre-wrap text-amber-950 font-sans">{finalOutput}</pre>
        </div>
      )}

      {artifacts && artifacts.length > 0 && !runConversationLoading && (
        <div className="mt-4 p-4 border border-slate-200 rounded-lg bg-slate-50">
          <h3 className="text-sm font-semibold mb-2 text-text-primary">
            Generated Run Artifacts ({artifacts.length})
          </h3>
          <ul className="space-y-2 text-xs">
            {artifacts.map((art) => (
              <li
                key={art.artifactId}
                className="p-2 bg-white border border-slate-200 rounded flex items-center justify-between gap-2"
              >
                <div>
                  <span className="font-semibold text-slate-800">{art.filename}</span>
                  <span className="ml-2 text-slate-500">({art.mimeType}, {art.sizeBytes} bytes)</span>
                  <div className="font-mono text-[11px] text-slate-500">{art.path}</div>
                </div>
                <span className="font-mono text-[10px] text-slate-400">
                  sha256: {art.sha256.slice(0, 12)}...
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {!runConversationLoading && (
        <ReasoningDiagnosticsPanel
          diagnostics={diagnostics}
          events={events}
          stopReason={stopReason}
          artifacts={artifacts}
          runStatus={runStatus}
        />
      )}
    </>
  );
};

export default FeedbackDisplay;
