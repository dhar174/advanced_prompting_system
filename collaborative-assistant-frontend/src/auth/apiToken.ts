export const API_TOKEN_STORAGE_KEY = 'aps_api_token';

type TokenListener = (token: string) => void;
type AuthRequiredListener = (message?: string) => void;

let inMemoryToken = '';
const tokenListeners = new Set<TokenListener>();
const authRequiredListeners = new Set<AuthRequiredListener>();

function readSessionStorageToken(): string | null {
  try {
    if (typeof window !== 'undefined' && window.sessionStorage) {
      return window.sessionStorage.getItem(API_TOKEN_STORAGE_KEY);
    }
  } catch {
    // Ignore storage access errors in restricted browser contexts.
  }
  return null;
}

function writeSessionStorageToken(token: string): void {
  try {
    if (typeof window !== 'undefined' && window.sessionStorage) {
      if (token) {
        window.sessionStorage.setItem(API_TOKEN_STORAGE_KEY, token);
      } else {
        window.sessionStorage.removeItem(API_TOKEN_STORAGE_KEY);
      }
    }
  } catch {
    // Fall back to in-memory token state only.
  }
}

export function getApiToken(): string {
  const stored = readSessionStorageToken();
  if (stored !== null) {
    const trimmed = stored.trim();
    inMemoryToken = trimmed;
    return trimmed;
  }
  return inMemoryToken;
}

export function setApiToken(token: string): void {
  const trimmed = String(token || '').trim();
  if (!trimmed) {
    clearApiToken();
    return;
  }
  const changed = getApiToken() !== trimmed;
  inMemoryToken = trimmed;
  writeSessionStorageToken(trimmed);
  if (changed) {
    tokenListeners.forEach((listener) => listener(trimmed));
  }
}

export function clearApiToken(): void {
  const hadToken = getApiToken() !== '';
  inMemoryToken = '';
  writeSessionStorageToken('');
  if (hadToken) {
    tokenListeners.forEach((listener) => listener(''));
  }
}

export function subscribeApiToken(listener: TokenListener): () => void {
  tokenListeners.add(listener);
  return () => {
    tokenListeners.delete(listener);
  };
}

export function notifyAuthRequired(message?: string): void {
  authRequiredListeners.forEach((listener) => listener(message));
}

export function subscribeAuthRequired(listener: AuthRequiredListener): () => void {
  authRequiredListeners.add(listener);
  return () => {
    authRequiredListeners.delete(listener);
  };
}

export function isAuthError(err: unknown): boolean {
  if (!err || typeof err !== 'object') {
    return false;
  }
  const record = err as {
    statusCode?: unknown;
    status?: unknown;
    message?: unknown;
    networkError?: {
      statusCode?: unknown;
      status?: unknown;
      response?: { status?: unknown };
      result?: { error?: unknown };
    };
    graphQLErrors?: ReadonlyArray<{
      message?: unknown;
      extensions?: { code?: unknown; statusCode?: unknown; http?: { status?: unknown } };
    }>;
  };

  if (
    record.statusCode === 401 ||
    record.status === 401 ||
    record.networkError?.statusCode === 401 ||
    record.networkError?.status === 401 ||
    record.networkError?.response?.status === 401
  ) {
    return true;
  }

  if (Array.isArray(record.graphQLErrors)) {
    for (const gqlErr of record.graphQLErrors) {
      const code = gqlErr?.extensions?.code;
      const extStatus =
        gqlErr?.extensions?.statusCode ?? gqlErr?.extensions?.http?.status;
      if (code === 'UNAUTHENTICATED' || extStatus === 401) {
        return true;
      }
    }
  }

  const netErrMsg =
    typeof record.networkError?.result?.error === 'string'
      ? record.networkError.result.error
      : '';
  const msg = typeof record.message === 'string' ? record.message : '';
  const combined = `${msg} ${netErrMsg}`.trim();
  return /\b401\b|Authentication required|Invalid bearer token/i.test(combined);
}
