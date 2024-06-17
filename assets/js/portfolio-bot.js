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

// Function to get bot response based on user input
function getBotResponse(message) {
    const responses = {
        'hello': { text: 'Hi there! How can I help you today?', emotion: 'happy' },
        'hi': { text: 'Hi! How can I assist you today?', emotion: 'happy' },
        'help': { text: 'Sure, I am here to help! What do you need assistance with?', emotion: 'neutral' },
        'portfolio': { text: 'This is my portfolio. You can find information about my projects, skills, and contact details here.', emotion: 'neutral' },
        'projects': { text: 'I have worked on various projects including AI models, cybersecurity applications, and more.', emotion: 'neutral' },
        'web development': { text: 'I am fully immersed in the world of web development, refining my skills in Python and JavaScript for backend development. My journey involves crafting my own website, integrating cutting-edge technology, and embracing the full spectrum of full-stack web development.', emotion: 'happy' },
        'contact': { text: 'You can contact me via email at kasinadhsarma@gmail.com or through the contact form on this page.', emotion: 'neutral' },
        'skills': { text: 'I have skills in web development, AI, cybersecurity, and more. Feel free to ask about any specific skill!', emotion: 'neutral' },
        'experience': { text: 'I have experience working on various projects in web development, AI, and cybersecurity. Check out my projects section for more details.', emotion: 'neutral' },
        'education': { text: 'I have a background in computer science and have completed various courses and certifications in web development, AI, and cybersecurity.', emotion: 'neutral' },
        'default': { text: 'I am not sure how to respond to that. Can you please rephrase your question?', emotion: 'neutral' }
    };

    const keywords = Object.keys(responses);
    let response = responses['default'];

    console.log(`User message: ${message}`);
    for (let keyword of keywords) {
        if (message.toLowerCase().includes(keyword)) {
            response = responses[keyword];
            console.log(`Matched keyword: ${keyword}`);
            console.log(`Selected response: ${response.text}`);
            break;
        }
    }

    setTimeout(() => {
        displayMessage(response.text, 'bot');
        expressEmotion(response.emotion);
    }, 1000);
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
