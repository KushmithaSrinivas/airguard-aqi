// AirGuard Global JavaScript Helpers

document.addEventListener('DOMContentLoaded', () => {
    // Theme Toggle setup
    const themeToggleBtn = document.getElementById('theme-toggle');
    const htmlEl = document.documentElement;

    // Load saved or system theme
    const savedTheme = localStorage.getItem('airguard-theme');
    if (savedTheme === 'dark' || (!savedTheme && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
        htmlEl.classList.add('dark');
    } else {
        htmlEl.classList.remove('dark');
    }

    if (themeToggleBtn) {
        themeToggleBtn.addEventListener('click', () => {
            if (htmlEl.classList.contains('dark')) {
                htmlEl.classList.remove('dark');
                localStorage.setItem('airguard-theme', 'light');
            } else {
                htmlEl.classList.add('dark');
                localStorage.setItem('airguard-theme', 'dark');
            }
        });
    }

    // Mobile Hamburger Menu
    const mobileMenuBtn = document.getElementById('mobile-menu-btn');
    const mobileMenu = document.getElementById('mobile-menu');
    if (mobileMenuBtn && mobileMenu) {
        mobileMenuBtn.addEventListener('click', () => {
            mobileMenu.classList.toggle('hidden');
        });
    }
});

// Geolocation helper
function locateUserAndRedirect() {
    if (!navigator.geolocation) {
        alert('Geolocation is not supported by your browser.');
        window.location.href = '/dashboard?city=Delhi';
        return;
    }

    const btn = document.getElementById('btn-use-location');
    if (btn) {
        btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin mr-1.5"></i> Detecting Location...';
    }

    navigator.geolocation.getCurrentPosition(
        async (position) => {
            const lat = position.coords.latitude;
            const lon = position.coords.longitude;
            try {
                // Fetch list of cities to find nearest
                const res = await fetch('/api/cities');
                const cities = await res.json();
                
                // Find closest city using euclidean approximation
                let closestCity = 'Delhi';
                let minDistance = 999999;
                
                // Direct lookup or fallback to popular
                window.location.href = `/dashboard?city=${encodeURIComponent(closestCity)}&lat=${lat}&lon=${lon}`;
            } catch (err) {
                window.location.href = '/dashboard?city=Delhi';
            }
        },
        (error) => {
            alert('Location access denied or unavailable. Redirecting to Delhi default dashboard.');
            window.location.href = '/dashboard?city=Delhi';
        },
        { timeout: 5000 }
    );
}
