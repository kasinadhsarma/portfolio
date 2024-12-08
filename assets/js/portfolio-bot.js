// Portfolio Bot JavaScript

// Function to handle sending messages
function sendMessage() {
    console.log('sendMessage function triggered');
    const input = document.getElementById('botInput');
    const message = input.value.trim();
    if (message) {
        displayMessage(message, 'user');
        input.value = '';
        getBotResponse(message);
    }
}

// Function to display messages in the chat
function displayMessage(message, sender) {
    console.log(`Displaying message: ${message} from ${sender}`);
    const messageContainer = document.createElement('div');
    messageContainer.classList.add('message', sender);
    messageContainer.textContent = message;
    document.getElementById('botMessages').appendChild(messageContainer);
    messageContainer.scrollIntoView();
}

// Function to get bot response from API
async function getBotResponse(message) {
    console.log(`Sending message to API: ${message}`);
    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: message })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        console.log('API Response:', data);

        displayMessage(data.response, 'bot');
        expressEmotion(data.emotion);
    } catch (error) {
        console.error('Error:', error);
        displayMessage('I apologize, but I encountered an error. As VisionAI, I aim to provide better assistance. Please try again.', 'bot');
        expressEmotion('sad');
    }
}

// Function to express emotions
function expressEmotion(emotion) {
    const emojiMap = {
        'happy': './assets/images/happy-emoji.png',
        'sad': './assets/images/sad-emoji.png',
        'angry': './assets/images/angry-emoji.png',
        'neutral': './assets/images/neutral-emoji.png'
    };

    const emojiSrc = emojiMap[emotion.toLowerCase()] || emojiMap['neutral'];
    const botToggleCircle = document.querySelector('.bot-toggle-circle');
    botToggleCircle.innerHTML = `<img src="${emojiSrc}" alt="${emotion} Emoji" class="emoji-icon">`;
}

// Function to toggle the bot interface
function toggleBot() {
    const botContainer = document.querySelector('.bot-container');
    const botToggleCircle = document.querySelector('.bot-toggle-circle');
    if (botContainer.style.display === 'none' || botContainer.style.display === '') {
        botContainer.style.display = 'block';
        botToggleCircle.innerHTML = ''; // Clear the circle content
    } else {
        botContainer.style.display = 'none';
        botToggleCircle.innerHTML = '<img src="./assets/images/neutral-emoji.png" alt="Neutral Emoji" class="emoji-icon">'; // Add neutral emoji back to the circle
    }
}
