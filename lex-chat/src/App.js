import React, { useState, useEffect, useRef } from 'react';

const ChatMessage = ({ message, isUser }) => (
  <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}>
    <div className={`max-w-3/4 p-3 rounded-2xl ${isUser ? 'bg-blue-500 text-white' : 'bg-gray-100 text-gray-800'}`}>
      {message}
    </div>
  </div>
);

const ChatApp = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  // Use environment variable for API URL
  const apiUrl = process.env.REACT_APP_API_URL || 'https://kitchan98--lex-fridman-podcast-rag-web-app.modal.run';

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(scrollToBottom, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    setMessages(prev => [...prev, { text: input, isUser: true }]);
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
      setMessages(prev => [...prev, { text: data.response, isUser: false }]);
    } catch (error) {
      console.error('Error:', error);
      setMessages(prev => [...prev, { text: "Sorry, an error occurred. Please try again later.", isUser: false }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-white p-4">
      <div className="w-full max-w-3xl mx-auto rounded-3xl shadow-lg overflow-hidden bg-white">
        <div className="bg-gray-50 p-6">
          <h1 className="text-3xl font-light text-gray-800 mb-2">Lex Fridman Podcast Chat</h1>
          <p className="text-sm text-gray-500">Explore insights from Lex Fridman's conversations</p>
        </div>
        <div className="h-[calc(100vh-300px)] overflow-y-auto p-6">
          {messages.map((msg, index) => (
            <ChatMessage key={index} message={msg.text} isUser={msg.isUser} />
          ))}
          <div ref={messagesEndRef} />
        </div>
        <form onSubmit={handleSubmit} className="p-6 bg-white border-t border-gray-200">
          <div className="flex items-center">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask about Lex Fridman's podcasts..."
              className="flex-grow mr-2 rounded-full border border-gray-300 focus:border-blue-500 focus:ring focus:ring-blue-200 focus:ring-opacity-50 p-2"
              disabled={isLoading}
            />
            <button 
              type="submit" 
              disabled={isLoading}
              className="rounded-full bg-blue-500 hover:bg-blue-600 text-white font-medium py-2 px-6 transition-colors duration-300"
            >
              {isLoading ? 'Thinking...' : 'Ask'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default ChatApp;