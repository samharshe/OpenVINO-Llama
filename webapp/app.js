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