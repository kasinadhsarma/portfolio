const toggleBtns = document.querySelectorAll('.toggle-btn');

toggleBtns.forEach(btn => {
  btn.addEventListener('click', () => {
    const childContainer = btn.nextElementSibling;
    childContainer.classList.toggle('hidden');
    btn.textContent = childContainer.classList.contains('hidden') ? '&#9658;' : '&#9660;';
  });
});