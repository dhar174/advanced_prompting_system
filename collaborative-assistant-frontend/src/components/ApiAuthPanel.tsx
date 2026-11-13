import React, { useState } from 'react';
import { Button } from './ui/Button';
import { Input } from './ui/Input';
import { Label } from './ui/Label';

export interface ApiAuthPanelProps {
  authRequired: boolean;
  hasToken: boolean;
  isBusy?: boolean;
  errorMessage?: string | null;
  onApplyToken: (token: string) => Promise<void> | void;
  onClearToken: () => Promise<void> | void;
}

const ApiAuthPanel: React.FC<ApiAuthPanelProps> = ({
  authRequired,
  hasToken,
  isBusy = false,
  errorMessage,
  onApplyToken,
  onClearToken,
}) => {
  const [tokenDraft, setTokenDraft] = useState('');
  const [showToken, setShowToken] = useState(false);

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    const trimmed = tokenDraft.trim();
    if (!trimmed || isBusy) {
      return;
    }
    setTokenDraft('');
    setShowToken(false);
    await onApplyToken(trimmed);
  };

  const handleClear = async () => {
    setTokenDraft('');
    setShowToken(false);
    await onClearToken();
  };

  const statusTextId = 'api-token-help';
  const alertId = 'api-auth-alert';
  const describedBy = authRequired ? `${alertId} ${statusTextId}` : statusTextId;

  return (
    <section
      aria-labelledby="api-auth-heading"
      className={`mb-3 p-3 rounded-lg border ${
        authRequired
          ? 'border-rose-300 bg-rose-50'
          : 'border-slate-200 bg-surface-card'
      }`}
    >
      <div className="flex items-center justify-between mb-2">
        <h2 id="api-auth-heading" className="text-sm font-semibold text-text-primary">
          API Authentication
        </h2>
        {hasToken && !authRequired && (
          <span
            role="status"
            className="text-xs font-medium text-emerald-800 bg-emerald-100 px-2 py-0.5 rounded"
          >
            Token set
          </span>
        )}
      </div>

      {authRequired && (
        <div
          id={alertId}
          role="alert"
          className="mb-2 p-2 rounded border border-rose-300 bg-white text-xs text-rose-900"
        >
          {errorMessage ||
            'Authentication required. Enter your API bearer token and retry.'}
        </div>
      )}

      <form onSubmit={handleSubmit} className="space-y-2">
        <div>
          <Label
            htmlFor="api-token-input"
            className="block text-xs font-medium text-text-primary mb-1"
          >
            API Bearer Token
          </Label>
          <div className="flex items-center gap-1.5">
            <Input
              id="api-token-input"
              type={showToken ? 'text' : 'password'}
              value={tokenDraft}
              onChange={(e) => setTokenDraft(e.target.value)}
              placeholder={hasToken ? 'Token configured (enter to replace)' : 'Enter bearer token'}
              autoComplete="off"
              disabled={isBusy}
              aria-invalid={authRequired ? true : undefined}
              aria-describedby={describedBy}
              className="h-9 text-xs"
            />
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => setShowToken((prev) => !prev)}
              aria-label={showToken ? 'Hide API token' : 'Show API token'}
              aria-pressed={showToken}
              disabled={isBusy}
              className="shrink-0"
            >
              {showToken ? 'Hide' : 'Show'}
            </Button>
          </div>
          <p id={statusTextId} className="mt-1 text-[11px] text-text-secondary">
            Stored in session storage only for Authorization: Bearer requests.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <Button
            type="submit"
            size="sm"
            variant="primary"
            disabled={isBusy || !tokenDraft.trim()}
            data-testid="apply-api-token-button"
          >
            {authRequired ? 'Save & Retry' : 'Apply Token'}
          </Button>
          <Button
            type="button"
            size="sm"
            variant="outline"
            onClick={handleClear}
            disabled={isBusy || (!hasToken && !tokenDraft)}
            data-testid="clear-api-token-button"
          >
            Clear Token
          </Button>
        </div>
      </form>
    </section>
  );
};

export default ApiAuthPanel;
