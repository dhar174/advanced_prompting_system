import ConversationPage from './pages/ConversationPage';
import ErrorBoundary from './components/ErrorBoundary';
import { Toaster } from 'react-hot-toast';

function App() {
  return (
    <ErrorBoundary>
      <ConversationPage />
      <Toaster position="top-right" />
    </ErrorBoundary>
  );
}

export default App;
