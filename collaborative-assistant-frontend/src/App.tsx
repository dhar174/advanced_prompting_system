// import './App.css'; // Removed as it's not used with Tailwind typically
import ConversationPage from './pages/ConversationPage';
import { Toaster } from 'react-hot-toast';

function App() {
  return (
    <>
      <ConversationPage />
      {/* Optional: Toaster for error notifications from react-hot-toast */}
      <Toaster position="top-right" />
    </>
  );
}

export default App;
