import { ApolloClient, InMemoryCache, HttpLink, from, type ApolloLink } from '@apollo/client';
import { setContext } from '@apollo/client/link/context';
import { onError } from '@apollo/client/link/error';
import { clearApiToken, getApiToken, isAuthError, notifyAuthRequired } from '../auth/apiToken';

export function createAuthLink(): ApolloLink {
  return setContext((_, previousContext) => {
    const existingHeaders =
      (previousContext?.headers as Record<string, string> | undefined) ?? {};
    const token = getApiToken();
    if (!token) {
      return { headers: existingHeaders };
    }
    return {
      headers: {
        ...existingHeaders,
        Authorization: `Bearer ${token}`,
      },
    };
  });
}

export function createAuthErrorLink(): ApolloLink {
  return onError(({ networkError, graphQLErrors }) => {
    if (isAuthError({ networkError, graphQLErrors })) {
      clearApiToken();
      notifyAuthRequired('Authentication required to execute GraphQL requests.');
    }
  });
}

export function createApolloClient(terminatingLink?: ApolloLink) {
  const httpLink =
    terminatingLink ??
    new HttpLink({
      uri: '/graphql', // Flask GraphQL endpoint
    });

  return new ApolloClient({
    link: from([createAuthErrorLink(), createAuthLink(), httpLink]),
    cache: new InMemoryCache(),
  });
}

export const client = createApolloClient();
