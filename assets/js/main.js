'use strict';

// Function to toggle the 'active' class on an element
const elementToggleFunc = (element) => {
  if (element) {
    element.classList.toggle('active');
  }
};

// Sidebar toggle functionality
const sidebar = document.querySelector('[data-sidebar]');
const sidebarBtn = document.querySelector('[data-sidebar-btn]');

if (sidebar && sidebarBtn) {
  sidebarBtn.addEventListener('click', () => {
    elementToggleFunc(sidebar);
  });
}

// Testimonials modal functionality
const testimonialsItem = document.querySelectorAll('[data-testimonials-item]');
const modalContainer = document.querySelector('[data-modal-container]');
const modalCloseBtn = document.querySelector('[data-modal-close-btn]');
const overlay = document.querySelector('[data-overlay]');
const modalImg = document.querySelector('[data-modal-img]');
const modalTitle = document.querySelector('[data-modal-title]');
const modalText = document.querySelector('[data-modal-text]');

const testimonialsModalFunc = () => {
  if (modalContainer && overlay) {
    modalContainer.classList.toggle('active');
    overlay.classList.toggle('active');
  }
};

testimonialsItem.forEach(item => {
  item.addEventListener('click', () => {
    if (modalImg && modalTitle && modalText) {
      modalImg.src = item.querySelector('[data-testimonials-avatar]').src;
      modalImg.alt = item.querySelector('[data-testimonials-avatar]').alt;
      modalTitle.textContent = item.querySelector('[data-testimonials-title]').textContent;
      modalText.textContent = item.querySelector('[data-testimonials-text]').textContent;
      testimonialsModalFunc();
    }
  });
});

if (modalCloseBtn && overlay) {
  modalCloseBtn.addEventListener('click', testimonialsModalFunc);
  overlay.addEventListener('click', testimonialsModalFunc);
}

// Dropdown filter functionality
const select = document.querySelector('[data-select]');
const selectItems = document.querySelectorAll('[data-select-item]');
const selectValue = document.querySelector('[data-select-value]');
const filterBtn = document.querySelectorAll('[data-filter-btn]');

if (select) {
  select.addEventListener('click', () => {
    elementToggleFunc(select);
  });
}

selectItems.forEach(item => {
  item.addEventListener('click', () => {
    const selectedValue = item.innerText.toLowerCase();
    if (selectValue) {
      selectValue.textContent = item.innerText;
    }
    elementToggleFunc(select);
    filterFunc(selectedValue);
  });
});

// Filter items based on category
const filterItems = document.querySelectorAll('[data-filter-item]');

const filterFunc = (category) => {
  filterItems.forEach(item => {
    if (category === 'all') {
      item.classList.add('active');
    } else if (category === item.dataset.category) {
      item.classList.add('active');
    } else {
      item.classList.remove('active');
    }
  });
};

let lastClickedBtn = filterBtn[0];

filterBtn.forEach(btn => {
  btn.addEventListener('click', () => {
    const selectedValue = btn.innerText.toLowerCase();
    if (selectValue) {
      selectValue.textContent = btn.innerText;
    }
    filterFunc(selectedValue);
    if (lastClickedBtn) {
      lastClickedBtn.classList.remove('active');
    }
    btn.classList.add('active');
    lastClickedBtn = btn;
  });
});

// Form validation functionality
const form = document.querySelector('[data-form]');
const formInputs = document.querySelectorAll('[data-form-input]');
const formBtn = document.querySelector('[data-form-btn]');

formInputs.forEach(input => {
  input.addEventListener('input', () => {
    if (form && formBtn) {
      form.checkValidity() ? formBtn.removeAttribute('disabled') : formBtn.setAttribute('disabled', '');
    }
  });
});

// Navigation functionality
const navigationLinks = document.querySelectorAll('[data-nav-link]');
const pages = document.querySelectorAll('[data-page]');

navigationLinks.forEach(link => {
  link.addEventListener('click', () => {
    pages.forEach(page => {
      if (link.textContent.toLowerCase() === page.dataset.page) {
        page.classList.add('active');
        link.classList.add('active');
        window.scrollTo(0, 0);
      } else {
        page.classList.remove('active');
        link.classList.remove('active');
      }
    });
  });
});

// Contacts section toggle functionality
const contactsSection = document.querySelector('.sidebar-info_more');
const contactsBtn = document.querySelector('.info_more-btn');

if (contactsSection && contactsBtn) {
  contactsBtn.addEventListener('click', () => {
    elementToggleFunc(contactsSection);
  });
}
