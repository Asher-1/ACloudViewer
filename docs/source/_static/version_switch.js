/* Version switcher for ACloudViewer documentation */
// This file provides version selection functionality for documentation pages
// Mode B: /documentation/ = stable, /documentation/dev/ = development, /documentation/vX.X.X/ = version archives

(function() {
    'use strict';

    // Version data - Mode B structure
    // - stable: /documentation/ (latest stable release)
    // - dev: /documentation/dev/ (development from main branch)
    // - vX.X.X: /documentation/vX.X.X/ (version archives)
    const VERSIONS = [
        { value: 'stable', display: 'Latest Stable', url: '/ACloudViewer/documentation/' },
        { value: 'dev', display: 'Development (main)', url: '/ACloudViewer/documentation/dev/' },
        // Additional versioned releases will be dynamically loaded
    ];

    // Get current version from URL path
    function getCurrentVersion() {
        const path = window.location.pathname;
        
        // Check for /documentation/dev/
        if (path.includes('/documentation/dev/') || path.endsWith('/documentation/dev')) {
            return 'dev';
        }
        
        // Check for /documentation/vX.X.X/
        const versionMatch = path.match(/\/documentation\/(v[\d.]+(?:-[\w]+)?)\//);
        if (versionMatch) {
            return versionMatch[1];
        }
        
        // Default to stable (/documentation/)
        return 'stable';
    }

    // Get base URL for documentation
    function getBaseUrl() {
        const path = window.location.pathname;
        const baseMatch = path.match(/^(\/ACloudViewer\/documentation)/);
        if (baseMatch) {
            return baseMatch[1];
        }
        return '/ACloudViewer/documentation';
    }

    // Get relative path within documentation (e.g., python_api/index.html)
    function getRelativePath() {
        const path = window.location.pathname;
        // Remove base documentation path and version prefix
        let relativePath = path.replace(/^.*\/documentation\/(?:dev\/|v[\d.]+(?:-[\w]+)?\/)?/, '');
        // Ensure we have a path
        if (!relativePath || relativePath === '') {
            relativePath = 'index.html';
        }
        return relativePath;
    }

    // Switch to a different version
    function switchVersion(version) {
        const baseUrl = getBaseUrl();
        const relativePath = getRelativePath();
        let newUrl;
        
        if (version === 'stable') {
            // /documentation/ (root = stable)
            newUrl = `${baseUrl}/${relativePath}`;
        } else if (version === 'dev') {
            // /documentation/dev/
            newUrl = `${baseUrl}/dev/${relativePath}`;
        } else {
            // /documentation/vX.X.X/
            newUrl = `${baseUrl}/${version}/${relativePath}`;
        }
        
        window.location.href = newUrl;
    }

    // Load versions from downloads_data.json
    async function loadVersions() {
        try {
            const response = await fetch('/ACloudViewer/downloads_data.json');
            if (!response.ok) {
                console.warn('Failed to load downloads_data.json, using default versions');
                return;
            }
            
            const data = await response.json();
            if (data.version_metadata && Array.isArray(data.version_metadata)) {
                // Keep stable and dev, add versioned releases
                const baseVersions = VERSIONS.slice(0, 2);
                VERSIONS.length = 0;
                VERSIONS.push(...baseVersions);
                
                // Add versions from metadata - ONLY show versions that have documentation
                data.version_metadata.forEach(version => {
                    // Skip main-devel (shown as "dev")
                    // Skip versions without documentation (has_documentation: false)
                    if (version.value && 
                        version.value !== 'main-devel' && 
                        version.has_documentation === true) {
                        // Convert version tag to path format (e.g., v3.9.3 -> v3.9.3)
                        const versionPath = version.value.startsWith('v') ? version.value : `v${version.value}`;
                        VERSIONS.push({
                            value: versionPath,
                            display: version.display_name || version.value,
                            url: `${getBaseUrl()}/${versionPath}/`
                        });
                    }
                });
                
                // Sort versions (stable first, dev second, then by version number descending)
                VERSIONS.sort((a, b) => {
                    if (a.value === 'stable') return -1;
                    if (b.value === 'stable') return 1;
                    if (a.value === 'dev') return -1;
                    if (b.value === 'dev') return 1;
                    // Compare version numbers
                    return b.value.localeCompare(a.value, undefined, { numeric: true });
                });
            }
        } catch (error) {
            console.warn('Error loading versions:', error);
        }
    }

    // Initialize version switcher
    function init() {
        // Wait for DOM to be ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', init);
            return;
        }

        // Load versions asynchronously
        loadVersions().then(() => {
            // Don't create fixed position selector - only use sidebar selector
            // The sidebar version selector is already present in brand.html template
            
            // Dispatch event to notify sidebar version selector
            const event = new CustomEvent('versionsLoaded', { 
                detail: { versions: VERSIONS, currentVersion: getCurrentVersion() }
            });
            document.dispatchEvent(event);
        });
    }

    // Update sidebar version selector
    function updateSidebarSelector() {
        const select = document.getElementById('docs-version-select-sidebar');
        if (select) {
            const currentVersion = getCurrentVersion();
            select.innerHTML = '';
            VERSIONS.forEach(v => {
                const option = document.createElement('option');
                option.value = v.value;
                option.textContent = v.display;
                if (v.value === currentVersion) {
                    option.selected = true;
                }
                select.appendChild(option);
            });
        }
    }

    // Export functions to global scope
    window.ACloudViewerVersionSwitcher = {
        switchVersion: switchVersion,
        getCurrentVersion: getCurrentVersion,
        getVersions: () => VERSIONS,
        updateSidebarSelector: updateSidebarSelector,
        init: init
    };

    // Auto-initialize
    init();
    
    // Listen for versions loaded event to update sidebar
    document.addEventListener('versionsLoaded', updateSidebarSelector);
})();
