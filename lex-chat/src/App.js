import React, { useState, useEffect, useRef } from 'react';
import { FiSave, FiEdit2, FiUser, FiThumbsUp, FiThumbsDown } from 'react-icons/fi';
import { motion } from 'framer-motion';
import { handleFeedback, handleEdit } from './chatUtils';
import html2pdf from 'html2pdf.js';

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

  // Use environment variable for API URL
  const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:5000';

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
      const response = await fetch(`${apiUrl}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: input }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setMessages(prev => [...prev, { text: data.response, isUser: false, feedback: null }]);
    } catch (error) {
      console.error('Error:', error);
      setMessages(prev => [...prev, { text: "Sorry, an error occurred. Please try again later.", isUser: false, feedback: null }]);
    } finally {
      setIsLoading(false);
    }
  };

  const onFeedback = async (messageIndex, isPositive) => {
    const updatedMessage = await handleFeedback(messageIndex, isPositive, messages, apiUrl);
    setMessages(prevMessages => 
      prevMessages.map((msg, index) => 
        index === messageIndex ? updatedMessage : msg
      )
    );
  };

  const onEdit = (originalMessage) => {
    const updatedMessage = prompt('Edit your message:', originalMessage);
    const newMessageText = handleEdit(originalMessage, updatedMessage);
    setMessages(prev => prev.map(msg => 
      msg.text === originalMessage && msg.isUser ? { ...msg, text: newMessageText } : msg
    ));
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

  const handleAuth = (action) => {
    // Implement actual authentication logic here
    setUser({ name: 'John Doe' });
    setShowAuthModal(false);
  };

  return (
    <div className="flex flex-col min-h-screen bg-white">
      <motion.header
        initial={{ y: -50 }}
        animate={{ y: 0 }}
        transition={{ type: "spring", stiffness: 100 }}
        className="bg-gray-100 shadow-sm p-4"
      >
        <div className="max-w-6xl mx-auto flex justify-between items-center">
          <h1 className="text-2xl font-semibold text-black">Lex Fridman Podcast Chat 🎙️</h1>
          {user ? (
            <div className="flex items-center space-x-4">
              <span className="text-gray-600">{user.name}</span>
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => setUser(null)}
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

      <div className="flex-grow flex items-center justify-center p-4">
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5 }}
          className="w-full max-w-4xl bg-white rounded-lg shadow-lg overflow-hidden"
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

      {showAuthModal && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center"
        >
          <motion.div
            initial={{ y: -50, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            exit={{ y: 50, opacity: 0 }}
            className="bg-white p-8 rounded-lg shadow-xl relative"
          >
            <button
              onClick={() => setShowAuthModal(false)}
              className="absolute top-2 right-2 text-gray-500 hover:text-gray-700"
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
            <h2 className="text-2xl font-semibold mb-4">Login / Register</h2>
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => handleAuth('login')}
              className="w-full bg-black text-white py-2 rounded mb-2"
            >
              Login
            </motion.button>
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => handleAuth('register')}
              className="w-full bg-gray-100 text-black py-2 rounded"
            >
              Register
            </motion.button>
          </motion.div>
        </motion.div>
      )}
    </div>
  );
};

export default ChatApp;