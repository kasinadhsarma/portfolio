const { createCanvas } = require('canvas');
const fs = require('fs');
const path = require('path');

// Create directory if it doesn't exist
const assetsDir = path.join(__dirname, '..', 'assets', 'images');
if (!fs.existsSync(assetsDir)) {
    fs.mkdirSync(assetsDir, { recursive: true });
}

function createEmoji(type) {
    const canvas = createCanvas(100, 100);
    const ctx = canvas.getContext('2d');

    // Clear canvas
    ctx.clearRect(0, 0, 100, 100);

    // Draw face circle
    ctx.beginPath();
    ctx.arc(50, 50, 40, 0, Math.PI * 2);
    ctx.fillStyle = '#FFD700';  // Golden yellow
    ctx.fill();
    ctx.strokeStyle = '#000000';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Draw eyes
    ctx.beginPath();
    ctx.arc(35, 40, 5, 0, Math.PI * 2);
    ctx.arc(65, 40, 5, 0, Math.PI * 2);
    ctx.fillStyle = '#000000';
    ctx.fill();

    // Draw expression based on type
    switch (type) {
        case 'happy':
            // Smile
            ctx.beginPath();
            ctx.arc(50, 50, 25, 0, Math.PI);
            ctx.strokeStyle = '#000000';
            ctx.lineWidth = 2;
            ctx.stroke();
            break;

        case 'sad':
            // Frown
            ctx.beginPath();
            ctx.arc(50, 70, 25, Math.PI, Math.PI * 2);
            ctx.strokeStyle = '#000000';
            ctx.lineWidth = 2;
            ctx.stroke();
            break;

        case 'angry':
            // Angry eyebrows and frown
            ctx.beginPath();
            ctx.moveTo(25, 30);
            ctx.lineTo(45, 35);
            ctx.moveTo(75, 30);
            ctx.lineTo(55, 35);
            ctx.strokeStyle = '#000000';
            ctx.lineWidth = 2;
            ctx.stroke();

            // Angry mouth
            ctx.beginPath();
            ctx.arc(50, 70, 20, Math.PI, Math.PI * 2);
            ctx.strokeStyle = '#000000';
            ctx.lineWidth = 2;
            ctx.stroke();
            break;

        case 'neutral':
            // Straight line for mouth
            ctx.beginPath();
            ctx.moveTo(35, 60);
            ctx.lineTo(65, 60);
            ctx.strokeStyle = '#000000';
            ctx.lineWidth = 2;
            ctx.stroke();
            break;
    }

    // Save the emoji
    const buffer = canvas.toBuffer('image/png');
    fs.writeFileSync(path.join(assetsDir, `${type}-emoji.png`), buffer);
}

// Generate all emojis
['happy', 'sad', 'angry', 'neutral'].forEach(createEmoji);
