import React, { useState, useEffect, useRef } from 'react';
import { FiSave, FiEdit2, FiUser, FiThumbsUp, FiThumbsDown } from 'react-icons/fi';
import { motion } from 'framer-motion';
import html2pdf from 'html2pdf.js';
import './App.css';
import { BrowserRouter as Router, useNavigate } from 'react-router-dom';
import { GoogleOAuthProvider } from '@react-oauth/google';
import AuthModal from './AuthModal';

const ChatMessage = ({ message, isUser, onEdit, onFeedback }) => (
  <motion.div
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ duration: 0.3 }}
    className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4 group`}
  >
    <motion.div
      whileHover={{ scale: 1.02 }}
      className={`max-w-3/4 p-3 rounded-lg ${isUser ? 'bg-gray-200 text-black' : 'bg-white text-black border border-gray-300'} relative cursor-pointer`}
    >
      {message.text}
      {!isUser && (
        <div className="mt-2 flex justify-end space-x-2">
          <motion.button
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
            onClick={() => onFeedback(true)}
            className="text-gray-500 hover:text-green-500 transition-colors duration-200"
            aria-label="Provide positive feedback"
          >
            <FiThumbsUp size={16} />
          </motion.button>
          <motion.button
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
            onClick={() => onFeedback(false)}
            className="text-gray-500 hover:text-red-500 transition-colors duration-200"
            aria-label="Provide negative feedback"
          >
            <FiThumbsDown size={16} />
          </motion.button>
        </div>
      )}
      {isUser && (
        <motion.button
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.9 }}
          onClick={() => onEdit(message.text)}
          className="absolute -top-2 -right-2 bg-white text-gray-600 rounded-full p-1 shadow-md opacity-0 group-hover:opacity-100 transition-opacity"
        >
          <FiEdit2 size={12} />
        </motion.button>
      )}
    </motion.div>
  </motion.div>
);

const ChatApp = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [user, setUser] = useState(null);
  const [showAuthModal, setShowAuthModal] = useState(false);
  const messagesEndRef = useRef(null);
  const navigate = useNavigate();
  const [queryCount, setQueryCount] = useState(0);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [authError, setAuthError] = useState(null);
  const [authSuccess, setAuthSuccess] = useState(false);

  const apiUrl = 'https://podai-8fe41.web.app';
  //const apiUrl = 'http://localhost:5000';

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(scrollToBottom, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    setMessages(prev => [...prev, { text: input, isUser: true, feedback: null }]);
    setInput('');
    setIsLoading(true);

    try {
      const token = localStorage.getItem('token');
      const response = await fetch(`${apiUrl}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': token ? `Bearer ${token}` : '',
        },
        body: JSON.stringify({ message: input }),
      });

      if (!response.ok) {
        if (response.status === 401) {
          setShowAuthModal(true);
          throw new Error('Please login or register to continue using the chatbot.');
        }
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setMessages(prev => [...prev, { text: data.response, isUser: false, feedback: null }]);
      
      if (!token) {
        setQueryCount(prev => prev + 1);
        if (queryCount >= 2) {
          setShowAuthModal(true);
        }
      }
    } catch (error) {
      console.error('Error:', error);
      setMessages(prev => [...prev, { text: error.message || "Sorry, an error occurred. Please try again later.", isUser: false, feedback: null }]);
    } finally {
      setIsLoading(false);
    }
  };

  const onFeedback = async (messageIndex, isPositive) => {
    try {
      const token = localStorage.getItem('token');
      if (!token) {
        setShowAuthModal(true);
        return;
      }

      const response = await fetch(`${apiUrl}/feedback`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
        },
        body: JSON.stringify({
          message: messages[messageIndex].text,
          is_positive: isPositive,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const updatedMessages = [...messages];
      updatedMessages[messageIndex] = {
        ...updatedMessages[messageIndex],
        feedback: isPositive ? 'positive' : 'negative',
      };
      setMessages(updatedMessages);
    } catch (error) {
      console.error('Error submitting feedback:', error);
    }
  };

  const onEdit = (originalMessage) => {
    const updatedMessage = prompt('Edit your message:', originalMessage);
    if (updatedMessage !== null && updatedMessage !== originalMessage) {
      setMessages(prev => prev.map(msg => 
        msg.text === originalMessage && msg.isUser ? { ...msg, text: updatedMessage } : msg
      ));
    }
  };

  const handleSaveConversation = () => {
    const element = document.createElement('div');
    element.innerHTML = `
      <h1 style="text-align: center; color: #333;">Lex Fridman Podcast Chat</h1>
      ${messages.map(msg => `
        <div style="margin-bottom: 10px; ${msg.isUser ? 'text-align: right;' : ''}">
          <strong>${msg.isUser ? 'You' : 'AI'}:</strong>
          <p style="margin: 5px 0; padding: 10px; border-radius: 10px; display: inline-block; max-width: 70%; ${
            msg.isUser ? 'background-color: #e0e0e0;' : 'background-color: #f0f0f0; border: 1px solid #d0d0d0;'
          }">${msg.text}</p>
        </div>
      `).join('')}
    `;

    const opt = {
      margin:       10,
      filename:     'lex-fridman-chat.pdf',
      image:        { type: 'jpeg', quality: 0.98 },
      html2canvas:  { scale: 2 },
      jsPDF:        { unit: 'mm', format: 'a4', orientation: 'portrait' }
    };

    html2pdf().from(element).set(opt).save();
  };

  const handleAuth = async (email, password, name, isLogin) => {
    try {
      const endpoint = isLogin ? '/api/login' : '/api/register';
      const response = await fetch(`${apiUrl}${endpoint}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email, password, name }),
      });
  
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
      }
  
      const data = await response.json();
      localStorage.setItem('token', data.token);
      setUser({ name: data.name, email: email });
      setIsAuthenticated(true);
      setAuthSuccess(true);
      setShowAuthModal(false);
      setTimeout(() => setAuthSuccess(false), 3000);
    } catch (error) {
      console.error('Authentication error:', error.message);
      setAuthError(error.message);
    }
  };

  const handleLogout = async () => {
    try {
      const token = localStorage.getItem('token');
      await fetch(`${apiUrl}/api/logout`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });
      localStorage.removeItem('token');
      setUser(null);
    } catch (error) {
      console.error('Logout error:', error);
    }
  };

  return (
    <div className="chat-app">
      {authSuccess && (
        <motion.div
          initial={{ opacity: 0, y: -50 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -50 }}
          className="fixed top-0 left-0 right-0 bg-gray-800 text-white text-center py-2 z-50"
        >
          Success! You're now logged in.
        </motion.div>
      )}
      
      <motion.header
        initial={{ y: -50 }}
        animate={{ y: 0 }}
        transition={{ type: "spring", stiffness: 100 }}
        className="header"
      >
        <div className="max-w-6xl mx-auto flex justify-between items-center">
          <h1 className="text-2xl font-semibold text-black">Lex Fridman Podcast Chat 🎙️</h1>
          {user ? (
            <div className="flex items-center space-x-4">
              <span className="text-gray-600">{user.name}</span>
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={handleLogout}
                className="text-black hover:text-gray-600"
              >
                Logout
              </motion.button>
            </div>
          ) : (
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => setShowAuthModal(true)}
              className="flex items-center space-x-2 text-black hover:text-gray-600"
            >
              <FiUser />
              <span>Login / Register</span>
            </motion.button>
          )}
        </div>
      </motion.header>

      <div className="chat-container">
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5 }}
          className="chat-window"
        >
          <div className="bg-gray-50 p-6 border-b border-gray-200">
            <h2 className="text-xl font-light text-black">Explore insights from Lex Fridman's conversations</h2>
          </div>
          <div className="h-[calc(100vh-400px)] overflow-y-auto p-6">
            {messages.map((msg, index) => (
              <ChatMessage 
                key={index} 
                message={msg} 
                isUser={msg.isUser} 
                onEdit={onEdit} 
                onFeedback={(isPositive) => onFeedback(index, isPositive)}
              />
            ))}
            <div ref={messagesEndRef} />
          </div>
          <form onSubmit={handleSubmit} className="p-4 bg-white border-t border-gray-200">
            <div className="flex items-center space-x-2">
              <motion.input
                whileFocus={{ scale: 1.02 }}
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask about Lex Fridman's podcasts..."
                className="flex-grow rounded-full border border-gray-300 focus:border-gray-500 focus:ring focus:ring-gray-200 focus:ring-opacity-50 py-2 px-4 text-sm"
                disabled={isLoading}
              />
              <motion.button 
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                type="submit" 
                disabled={isLoading}
                className="rounded-full bg-black hover:bg-gray-800 text-white font-medium py-2 px-6 text-sm transition-colors duration-300"
              >
                {isLoading ? 'Thinking...' : 'Ask'}
              </motion.button>
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                type="button"
                onClick={handleSaveConversation}
                className="rounded-full bg-gray-100 hover:bg-gray-200 text-black p-2 transition-colors duration-300"
              >
                <FiSave size={20} />
              </motion.button>
            </div>
          </form>
        </motion.div>
      </div>
      <AuthModal
        isOpen={showAuthModal}
        onClose={() => setShowAuthModal(false)}
        onAuth={handleAuth}
        authError={authError}
      />
    </div>
  );
};

const App = () => (
  <GoogleOAuthProvider clientId="YOUR_GOOGLE_CLIENT_ID">
    <Router>
      <ChatApp />
    </Router>
  </GoogleOAuthProvider>
);

export default App;