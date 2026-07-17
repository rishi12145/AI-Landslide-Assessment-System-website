/**
 * landing.js - Animations for the landing page
 */

document.addEventListener('DOMContentLoaded', () => {
    // Generate background particles
    const particleLayer = document.getElementById('particles');
    
    // Create 30 random particles
    for (let i = 0; i < 30; i++) {
        createParticle(particleLayer);
    }
    
    // Add subtle mouse move effect to hero
    const hero = document.querySelector('.hero-content');
    document.addEventListener('mousemove', (e) => {
        const x = (e.clientX / window.innerWidth - 0.5) * 20;
        const y = (e.clientY / window.innerHeight - 0.5) * 20;
        
        hero.style.transform = `translate(${x}px, ${y}px)`;
    });
});

function createParticle(container) {
    const particle = document.createElement('div');
    particle.className = 'particle';
    
    // Randomize properties
    const size = Math.random() * 4 + 2; // 2px to 6px
    const left = Math.random() * 100; // 0% to 100%
    const duration = Math.random() * 15 + 10; // 10s to 25s
    const delay = Math.random() * 10; // 0s to 10s
    
    particle.style.width = `${size}px`;
    particle.style.height = `${size}px`;
    particle.style.left = `${left}%`;
    particle.style.top = '100%';
    particle.style.animationDuration = `${duration}s`;
    particle.style.animationDelay = `${delay}s`;
    
    container.appendChild(particle);
}
