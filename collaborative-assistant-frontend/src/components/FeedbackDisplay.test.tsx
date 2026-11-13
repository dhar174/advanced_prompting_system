/// <reference types="vitest/globals" />
import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import FeedbackDisplay from './FeedbackDisplay';

describe('FeedbackDisplay', () => {
  it('renders non-completed run banners honestly with alert/status roles and suppresses green success box', () => {
    render(
      <FeedbackDisplay
        currentQuestions={[]}
        finalOutput="Tree of Thoughts execution mode is currently gated pending Issue #36."
        runConversationLoading={false}
        runStatus="gated"
        stopReason="gated"
        errorMessage="Tree of Thoughts execution mode is currently gated pending Issue #36."
      />
    );
    const alert = screen.getByRole('alert');
    expect(alert).toHaveTextContent(/Execution Mode Gated/i);
    expect(alert).toHaveTextContent(/Issue #36/i);
    expect(screen.queryByText('Final Output:')).not.toBeInTheDocument();
  });

  it('renders Partial Output card for budget_exhausted status instead of green Final Output', () => {
    render(
      <FeedbackDisplay
        currentQuestions={[]}
        finalOutput="Draft partial summary before step budget limit."
        runConversationLoading={false}
        runStatus="budget_exhausted"
        stopReason="budget_exhausted"
        errorMessage="Budget exhausted."
      />
    );
    expect(screen.getByText(/Step Budget Exhausted \(Partial Result\)/i)).toBeInTheDocument();
    expect(screen.getByText(/Partial Output \(Unverified \/ Incomplete\):/i)).toBeInTheDocument();
    expect(screen.queryByText('Final Output:')).not.toBeInTheDocument();
  });

  it('renders generated run artifacts with path and checksum metadata for completed runs', () => {
    render(
      <FeedbackDisplay
        currentQuestions={[]}
        finalOutput="Artifact generated."
        runConversationLoading={false}
        runStatus="completed"
        stopReason="final_answer"
        artifacts={[
          {
            artifactId: 'art-1',
            runId: 'run-77',
            filename: 'plan.json',
            path: 'artifacts/run-77/plan.json',
            mimeType: 'application/json',
            sizeBytes: 256,
            sha256: 'deadbeef1234567890',
          },
        ]}
      />
    );
    expect(screen.getByText('Final Output:')).toBeInTheDocument();
    expect(screen.getByText(/Generated Run Artifacts \(1\)/i)).toBeInTheDocument();
    expect(screen.getByText('plan.json')).toBeInTheDocument();
    expect(screen.getByText(/artifacts\/run-77\/plan\.json/i)).toBeInTheDocument();
    expect(screen.getByText(/deadbeef1234/i)).toBeInTheDocument();
  });
});
