const titleElement = document.querySelector("title");
const titleText = "Intel + Community Labs = ML + AO = ";
const displayLength = 23;
let currentIndex = 0;

function rotateTitle() {
    let rotatedText = titleText + titleText;
    let displayText = rotatedText.substring(currentIndex, currentIndex + displayLength);
    titleElement.textContent = displayText;
    currentIndex = (currentIndex + 1) % titleText.length;
}

setInterval(rotateTitle, 500);

const serverURL = "http://127.0.0.1:3000";

const chatForm = document.getElementById('chat-form');
const chatHistory = document.getElementById('chat-history');
const userPromptInput = document.getElementById('user-prompt');

function addMessage(content, isUser = false) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${isUser ? 'user-message' : 'assistant-message'}`;
    messageDiv.textContent = content;
    
    const welcomeMessage = chatHistory.querySelector('.welcome-message');
    if (welcomeMessage) {
        welcomeMessage.remove();
    }
    
    chatHistory.appendChild(messageDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight;
    
    return messageDiv;
}

chatForm.addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const userMessage = userPromptInput.value.trim();
    if (!userMessage) return;
    
    addMessage(userMessage, true);
    userPromptInput.value = '';
    
    const processingMessage = addMessage('Processing your request...', false);
    
    try {
        const response = await fetch(`${serverURL}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: userMessage })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        processingMessage.remove();
        addMessage(data.response, false);
        
    } catch (error) {
        console.error('Error:', error);
        processingMessage.textContent = 'Error: Could not connect to server';
    }
});