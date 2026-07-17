/**
 * nav.js - Slide navigation controller for dashboard.html
 */

document.addEventListener('DOMContentLoaded', () => {
    const slidesContainer = document.getElementById('slides-container');
    const prevBtn = document.getElementById('prev-slide');
    const nextBtn = document.getElementById('next-slide');
    const dotsContainer = document.getElementById('slide-dots');
    
    let currentSlide = 0;
    let slides = [];
    
    function initSlides() {
        slides = Array.from(document.querySelectorAll('.slide'));
        updateUI();
    }
    
    function updateUI() {
        // Update transforms
        slides.forEach((slide, index) => {
            slide.style.transform = `translateX(${(index - currentSlide) * 100}%)`;
            if (index === currentSlide) {
                slide.classList.add('active');
            } else {
                slide.classList.remove('active');
            }
        });
        
        // Update buttons
        prevBtn.disabled = currentSlide === 0;
        nextBtn.disabled = currentSlide === slides.length - 1;
        
        // Update dots
        dotsContainer.innerHTML = '';
        slides.forEach((_, index) => {
            const dot = document.createElement('div');
            dot.className = `slide-dot ${index === currentSlide ? 'active' : ''}`;
            dot.addEventListener('click', () => {
                currentSlide = index;
                updateUI();
            });
            dotsContainer.appendChild(dot);
        });
    }
    
    function nextSlide() {
        if (currentSlide < slides.length - 1) {
            currentSlide++;
            updateUI();
        }
    }
    
    function prevSlide() {
        if (currentSlide > 0) {
            currentSlide--;
            updateUI();
        }
    }
    
    // Event listeners
    prevBtn.addEventListener('click', prevSlide);
    nextBtn.addEventListener('click', nextSlide);
    
    document.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowRight' || e.key === 'ArrowDown') nextSlide();
        if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') prevSlide();
    });
    
    // Mouse wheel (with debounce)
    let wheelTimeout;
    document.addEventListener('wheel', (e) => {
        // Prevent sliding if scrolling inside a scrollable element (like chat or report)
        const isScrollable = e.target.closest('.chat-log') || e.target.closest('.report-scroll');
        if (isScrollable) return;
        
        if (wheelTimeout) return;
        
        if (e.deltaY > 50) {
            nextSlide();
            wheelTimeout = setTimeout(() => wheelTimeout = null, 800);
        } else if (e.deltaY < -50) {
            prevSlide();
            wheelTimeout = setTimeout(() => wheelTimeout = null, 800);
        }
    });

    // Observer to handle dynamically injected sections from dashboard.js
    const dashboardContainer = document.getElementById('dashboard-container');
    const slidePlaceholder = document.getElementById('slide-dashboard');
    
    const observer = new MutationObserver((mutations) => {
        for (let mutation of mutations) {
            if (mutation.addedNodes.length > 0) {
                const analysisContainer = dashboardContainer.querySelector('.analysis-dashboard');
                if (analysisContainer) {
                    const sections = Array.from(analysisContainer.querySelectorAll('.dash-section'));
                    if (sections.length > 0) {
                        // We have data! Let's convert them to slides.
                        let insertAfter = slidePlaceholder;
                        
                        const themeClasses = [
                            'slide-bg-metadata',
                            'slide-bg-coherence',
                            'slide-bg-phase',
                            'slide-bg-segmentation',
                            'slide-bg-shape',
                            'slide-bg-confidence',
                            'slide-bg-severity'
                        ];
                        sections.forEach((section, idx) => {
                            const newSlide = document.createElement('section');
                            newSlide.className = `slide ${themeClasses[idx] || 'slide-bg-metadata'}`;
                            
                            // Give it the same card structure
                            newSlide.innerHTML = `
                                <div class="slide-content">
                                    <div class="card glass-card fill-height">
                                        <div class="card-header flex-header">
                                            <h2>Analysis Section</h2>
                                            <span class="source-badge">Source: Analysis Engine JSON</span>
                                        </div>
                                        <div class="dashboard-scroll">
                                            <!-- Section content goes here -->
                                        </div>
                                    </div>
                                </div>
                            `;
                            
                            // Move the section into the new slide
                            newSlide.querySelector('.dashboard-scroll').appendChild(section);
                            
                            // Update the header title based on the section's h3
                            const title = section.querySelector('h3');
                            if (title) {
                                newSlide.querySelector('h2').textContent = title.textContent;
                                // Hide the original header since it's now in the card header
                                section.querySelector('.dash-section-header').style.display = 'none';
                            }
                            
                            slidesContainer.insertBefore(newSlide, insertAfter.nextSibling);
                            insertAfter = newSlide;
                        });
                        
                        // Remove the placeholder slide
                        slidePlaceholder.remove();
                        
                        // Re-initialize slides
                        initSlides();
                        
                        // Go to the first new slide
                        currentSlide = slides.indexOf(insertAfter) - sections.length + 1;
                        updateUI();
                        
                        observer.disconnect();
                    }
                }
            }
        }
    });
    
    observer.observe(dashboardContainer, { childList: true, subtree: true });

    // Initial setup
    initSlides();
});
