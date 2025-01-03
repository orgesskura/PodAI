// chatUtils.js

export const handleFeedback = async (messageIndex, isPositive, messages, apiUrl) => {
    console.log(`Feedback for message ${messageIndex}: ${isPositive ? 'positive' : 'negative'}`);
    
    try {
      const response = await fetch(`${apiUrl}/feedback`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          messageIndex, 
          isPositive, 
          messageContent: messages[messageIndex].text 
        }),
      });
  
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
  
      const data = await response.json();
      console.log('Feedback sent successfully:', data);
      
      // Return the updated message with feedback
      return { ...messages[messageIndex], feedback: isPositive };
    } catch (error) {
      console.error('Error sending feedback:', error);
      // Return the original message if there's an error
      return messages[messageIndex];
    }
  };
  
  export const handleEdit = (originalMessage, newMessage) => {
    if (newMessage && newMessage !== originalMessage) {
      return newMessage;
    }
    return originalMessage;
  };