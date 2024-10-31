// JavaScript functionality for interactivity

function learnMore() {
  alert("Discover more about Data Doppler and our data solutions!");
}

// Form submission event
document.getElementById('contactForm').addEventListener('submit', function(event) {
  event.preventDefault();
  alert('Thank you for reaching out. We will get back to you soon!');
});
