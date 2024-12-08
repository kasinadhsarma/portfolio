// Portfolio Bot Chat Interface

// Extend the existing VisionAI chat interface functionality
document.addEventListener('DOMContentLoaded', function() {
    // Initialize specific portfolio bot features
    initPortfolioBot();
});

function initPortfolioBot() {
    // Add any portfolio-specific initialization logic
    const welcomeMessage = `
        Hello! I'm your Portfolio AI Assistant. 
        I can help you with:
        - Analyzing your investment portfolio
        - Providing financial insights
        - Answering questions about your investments
    `;
    displayMessage(welcomeMessage, 'bot-message');

    // Add portfolio-specific event listeners or setup
    setupPortfolioInputHandlers();
}

function setupPortfolioInputHandlers() {
    const input = document.getElementById('botInput');
    
    // Add special handling for portfolio-related queries
    input.addEventListener('keydown', function(event) {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault(); // Prevent default newline
            const message = this.value.trim();
            
            // Check for specific portfolio commands
            if (isPortfolioCommand(message)) {
                processPortfolioCommand(message);
            } else {
                // Use existing sendMessage logic for general queries
                sendMessage();
            }
        }
    });
}

function isPortfolioCommand(message) {
    // Define special portfolio-related command patterns
    const portfolioCommands = [
        /^analyze portfolio/i,
        /^investment breakdown/i,
        /^portfolio performance/i,
        /^risk assessment/i
    ];

    return portfolioCommands.some(command => command.test(message));
}

function processPortfolioCommand(message) {
    // Special handling for portfolio-specific commands
    console.log(`Processing portfolio command: ${message}`);
    
    // Add custom logic for portfolio analysis
    fetch('/portfolio-analysis', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
            command: message,
            timestamp: new Date().toISOString()
        })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Portfolio analysis failed');
        }
        return response.json();
    })
    .then(data => {
        // Display portfolio analysis results
        displayMessage(data.analysis, 'bot-message');
        
        // Optionally visualize data
        if (data.visualizations) {
            renderPortfolioVisualizations(data.visualizations);
        }
        
        // Express emotion based on portfolio performance
        expressEmotion(data.emotion || 'neutral');
    })
    .catch(error => {
        console.error('Portfolio Analysis Error:', error);
        displayMessage('Sorry, I couldn\'t complete the portfolio analysis. Please try again.', 'bot-message');
        expressEmotion('sad');
    });
}

function renderPortfolioVisualizations(visualizations) {
    // Create and append visualization elements
    const visualizationContainer = document.createElement('div');
    visualizationContainer.classList.add('portfolio-visualizations');
    
    visualizations.forEach(viz => {
        const vizElement = document.createElement('div');
        vizElement.innerHTML = viz.html || `<img src="${viz.imageUrl}" alt="Portfolio Visualization">`;
        vizElement.classList.add('visualization');
        visualizationContainer.appendChild(vizElement);
    });

    document.getElementById('botMessages').appendChild(visualizationContainer);
}

// Override or extend existing VisionAI functions if needed
function getBotResponse(message) {
    // Check if it's a portfolio-specific query first
    if (isPortfolioCommand(message)) {
        processPortfolioCommand(message);
    } else {
        // Fall back to original method for general queries
        originalGetBotResponse(message);
    }
}

// Keep a reference to the original method
const originalGetBotResponse = getBotResponse;