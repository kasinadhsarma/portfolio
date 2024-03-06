// Set launch date
const launchDate = new Date("May 1, 2024 00:00:00").getTime();

// Update countdown every second
const countDownTimer = setInterval(function() {
    const now = new Date().getTime();
    const distance = launchDate - now;

    const days = Math.floor(distance / (1000 * 60 * 60 * 24));
    const hours = Math.floor((distance % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
    const minutes = Math.floor((distance % (1000 * 60 * 60)) / (1000 * 60));
    const seconds = Math.floor((distance % (1000 * 60)) / 1000);

    document.querySelector(".days .number").textContent = days < 10 ? `0${days}` : days;
    document.querySelector(".hours .number").textContent = hours < 10 ? `0${hours}` : hours;
    document.querySelector(".minutes .number").textContent = minutes < 10 ? `0${minutes}` : minutes;
    document.querySelector(".seconds .number").textContent = seconds < 10 ? `0${seconds}` : seconds;

    if (distance < 0) {
        clearInterval(countDownTimer);
        // Do something on launch day!
    }
}, 1000);
