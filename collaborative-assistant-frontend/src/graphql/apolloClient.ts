import { ApolloClient, InMemoryCache, HttpLink } from '@apollo/client';

const httpLink = new HttpLink({
  uri: '/graphql', // Flask GraphQL endpoint
});

export const client = new ApolloClient({
  link: httpLink,
  cache: new InMemoryCache(),
});
