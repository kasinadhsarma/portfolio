// VisionAI Chat Interface JavaScript

// Function to handle sending messages
function sendMessage() {
    console.log('sendMessage function triggered');
    const input = document.getElementById('botInput');
    const message = input.value.trim();

    if (message) {
        displayMessage(message, 'user-message');
        input.value = '';
        // Reset textarea height
        input.style.height = 'auto';
        getBotResponse(message);
    }
}

// Function to display messages in the chat
function displayMessage(message, className) {
    console.log(`Displaying message: ${message} with class ${className}`);
    const messageContainer = document.createElement('div');
    messageContainer.classList.add('message', className);
    messageContainer.textContent = message;
    document.getElementById('botMessages').appendChild(messageContainer);
    messageContainer.scrollIntoView({ behavior: 'smooth' });
}

// Function to get bot response from API
async function getBotResponse(message) {
    console.log(`Sending message to API: ${message}`);
    showTypingIndicator();

    try {
        const response = await fetch('http://localhost:8000/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: message,
                context: 'portfolio',
                user_info: {
                    name: 'User',
                    interests: ['AI', 'Development', 'Technology']
                }
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        console.log('API Response:', data);

        hideTypingIndicator();

        if (data.error) {
            throw new Error(data.error);
        }

        displayMessage(data.response, 'bot-message');
        expressEmotion(data.emotion || 'neutral');
    } catch (error) {
        console.error('Error:', error);
        hideTypingIndicator();
        displayMessage('I apologize, but I encountered an error. As VisionAI, I aim to provide better assistance. Please try again.', 'bot-message');
        expressEmotion('sad');
    }
}

// Function to show typing indicator
function showTypingIndicator() {
    const indicator = document.createElement('div');
    indicator.classList.add('typing-indicator');
    indicator.innerHTML = '<span></span><span></span><span></span>';
    document.getElementById('botMessages').appendChild(indicator);
    indicator.scrollIntoView({ behavior: 'smooth' });
}

// Function to hide typing indicator
function hideTypingIndicator() {
    const indicator = document.querySelector('.typing-indicator');
    if (indicator) {
        indicator.remove();
    }
}

// Function to express emotions through emoji
function expressEmotion(emotion) {
    const emojiMap = {
        'happy': './assets/images/happy-emoji.png',
        'sad': './assets/images/sad-emoji.png',
        'angry': './assets/images/angry-emoji.png',
        'neutral': './assets/images/neutral-emoji.png'
    };

    const emojiSrc = emojiMap[emotion.toLowerCase()] || emojiMap['neutral'];
    const chatLogo = document.querySelector('.chat-logo');
    const toggleLogo = document.querySelector('.toggle-logo');

    if (chatLogo) chatLogo.src = emojiSrc;
    if (toggleLogo) toggleLogo.src = emojiSrc;
}

// Function to toggle the bot interface
function toggleBot() {
    const chatContainer = document.querySelector('.visionai-chat-container');
    if (chatContainer) {
        const isHidden = chatContainer.style.transform === 'translateY(120%)';
        chatContainer.style.transform = isHidden ? 'translateY(0)' : 'translateY(120%)';
        // Reset textarea height when hiding chat
        if (!isHidden) {
            document.getElementById('botInput').style.height = 'auto';
        }
    }
}

// Function to handle textarea auto-resize
function initTextareaHandlers() {
    const textarea = document.getElementById('botInput');
    if (!textarea) return;

    textarea.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
    });

    // Reset height on focus
    textarea.addEventListener('focus', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
    });
}

// Initialize the chat interface
document.addEventListener('DOMContentLoaded', function() {
    initTextareaHandlers();
    // Add click handler for toggle button
    document.querySelector('.visionai-toggle')?.addEventListener('click', toggleBot);
    // Display welcome message
    displayMessage('Hello! I\'m VisionAI, your AI assistant. I can help you with questions about AI and development. What would you like to know?', 'bot-message');
});
