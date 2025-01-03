// chatUtils.js

export const handleFeedback = async (messageIndex, isPositive, messages, apiUrl) => {
    const message = messages[messageIndex];
    try {
      const token = localStorage.getItem('token');
      const response = await fetch(`${apiUrl}/feedback`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
        },
        body: JSON.stringify({
          messageId: messageIndex,
          isPositive: isPositive,
          messageContent: message.text,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to submit feedback');
      }

      return { ...message, feedback: isPositive ? 'positive' : 'negative' };
    } catch (error) {
      console.error('Error submitting feedback:', error);
      return message;
    }
  };
  
  export const handleEdit = (originalMessage, newMessage) => {
    if (newMessage && newMessage !== originalMessage) {
      return newMessage;
    }
    return originalMessage;
  };