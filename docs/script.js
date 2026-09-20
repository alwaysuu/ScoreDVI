const navToggle = document.querySelector('.nav-toggle');
const siteNav = document.querySelector('.site-nav');

navToggle?.addEventListener('click', () => {
  const isOpen = siteNav.classList.toggle('is-open');
  navToggle.setAttribute('aria-expanded', String(isOpen));
  document.body.classList.toggle('no-scroll', isOpen);
});

siteNav?.querySelectorAll('a').forEach((link) => link.addEventListener('click', () => {
  siteNav.classList.remove('is-open');
  navToggle?.setAttribute('aria-expanded', 'false');
  document.body.classList.remove('no-scroll');
}));

const progress = document.querySelector('.scroll-progress span');
const updateProgress = () => {
  const available = document.documentElement.scrollHeight - window.innerHeight;
  const percent = available > 0 ? (window.scrollY / available) * 100 : 0;
  if (progress) progress.style.width = `${Math.min(100, percent)}%`;
};
window.addEventListener('scroll', updateProgress, { passive: true });
updateProgress();

const revealObserver = new IntersectionObserver((entries) => {
  entries.forEach((entry) => {
    if (entry.isIntersecting) {
      entry.target.classList.add('is-visible');
      revealObserver.unobserve(entry.target);
    }
  });
}, { threshold: 0.08, rootMargin: '0px 0px -30px' });
document.querySelectorAll('.reveal').forEach((element) => revealObserver.observe(element));

const filters = document.querySelectorAll('.filter');
const resultCards = document.querySelectorAll('.result-card');
filters.forEach((button) => button.addEventListener('click', () => {
  filters.forEach((item) => item.classList.remove('is-active'));
  button.classList.add('is-active');
  const selected = button.dataset.filter;
  resultCards.forEach((card) => card.classList.toggle('is-hidden', selected !== 'all' && card.dataset.category !== selected));
}));

const lightbox = document.querySelector('#lightbox');
const lightboxImage = lightbox?.querySelector('img');
const lightboxCaption = lightbox?.querySelector('p');
document.querySelectorAll('.image-button').forEach((button) => button.addEventListener('click', () => {
  if (!lightbox || !lightboxImage || !lightboxCaption) return;
  lightboxImage.src = button.dataset.image;
  lightboxImage.alt = button.querySelector('img')?.alt || '';
  lightboxCaption.textContent = button.dataset.caption || '';
  lightbox.showModal();
  document.body.classList.add('no-scroll');
}));

const closeLightbox = () => { lightbox?.close(); document.body.classList.remove('no-scroll'); };
lightbox?.querySelector('.lightbox-close')?.addEventListener('click', closeLightbox);
lightbox?.addEventListener('click', (event) => { if (event.target === lightbox) closeLightbox(); });
lightbox?.addEventListener('cancel', () => document.body.classList.remove('no-scroll'));

const copyButton = document.querySelector('#copy-bibtex');
copyButton?.addEventListener('click', async () => {
  const text = document.querySelector('#bibtex')?.innerText || '';
  try {
    await navigator.clipboard.writeText(text);
    copyButton.textContent = 'Copied';
    window.setTimeout(() => { copyButton.textContent = 'Copy'; }, 1600);
  } catch { copyButton.textContent = 'Select text'; }
});

const year = document.querySelector('#year');
if (year) year.textContent = String(new Date().getFullYear());
