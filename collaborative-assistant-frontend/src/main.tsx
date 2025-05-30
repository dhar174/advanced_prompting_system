import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App.tsx';
import './index.css'; // Tailwind CSS
import { ApolloProvider } from '@apollo/client';
import { client } from './graphql/apolloClient.ts';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <ApolloProvider client={client}>
      <App />
    </ApolloProvider>
  </React.StrictMode>,
);
