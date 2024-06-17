document.addEventListener('DOMContentLoaded', function() {
    const toggleSwitch = document.getElementById('mode-toggle');
    const currentTheme = localStorage.getItem('theme') ? localStorage.getItem('theme') : null;
    const tdnn = document.querySelector('.tdnn');

    if (currentTheme) {
        document.documentElement.setAttribute('data-theme', currentTheme);

        if (currentTheme === 'bright') {
            toggleSwitch.checked = true;
            tdnn.classList.add('day');
        }
    }

    toggleSwitch.addEventListener('change', function(e) {
        if (e.target.checked) {
            document.documentElement.setAttribute('data-theme', 'bright');
            localStorage.setItem('theme', 'bright');
            tdnn.classList.add('day');
        } else {
            document.documentElement.setAttribute('data-theme', 'dark');
            localStorage.setItem('theme', 'dark');
            tdnn.classList.remove('day');
        }
    });

    const sunIcon = document.querySelector('.sun-icon');
    const moonIcon = document.querySelector('.moon-icon');

    if (sunIcon && moonIcon) {
        sunIcon.addEventListener('click', function() {
            document.documentElement.setAttribute('data-theme', 'bright');
            localStorage.setItem('theme', 'bright');
            toggleSwitch.checked = true;
            tdnn.classList.add('day');
        });

        moonIcon.addEventListener('click', function() {
            document.documentElement.setAttribute('data-theme', 'dark');
            localStorage.setItem('theme', 'dark');
            toggleSwitch.checked = false;
            tdnn.classList.remove('day');
        });
    } else {
        console.error('Sun or Moon icon not found in the DOM.');
    }
});
