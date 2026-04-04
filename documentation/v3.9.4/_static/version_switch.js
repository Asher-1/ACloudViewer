/* Version switcher for ACloudViewer documentation */
// This file provides version selection functionality for documentation pages
// Similar to Open3D's version management system

(function() {
    'use strict';

    // Version data - will be populated from downloads_data.json or GitHub API
    const VERSIONS = [
        { value: 'latest', display: 'Latest (main)', url: '/ACloudViewer/documentation/' },
        // Additional versions will be dynamically loaded
    ];

    // Get current version from URL path
    function getCurrentVersion() {
        const path = window.location.pathname;
        const match = path.match(/\/documentation\/(v[\d.]+(?:-[\w]+)?)\//);
        if (match) {
            return match[1];
        }
        return 'latest';
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

    // Switch to a different version
    function switchVersion(version) {
        const currentPath = window.location.pathname;
        const baseUrl = getBaseUrl();
        
        if (version === 'latest') {
            // Remove version from path
            const newPath = currentPath.replace(/\/documentation\/v[\d.]+(?:-[\w]+)?\//, '/documentation/');
            window.location.href = newPath;
        } else {
            // Add or replace version in path
            const versionPath = `${baseUrl}/${version}/`;
            const relativePath = currentPath.replace(/^.*\/documentation\/(?:v[\d.]+(?:-[\w]+)?\/)?/, '');
            window.location.href = versionPath + relativePath;
        }
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
                // Clear default versions
                VERSIONS.length = 1; // Keep 'latest'
                
                // Add versions from metadata - ONLY show versions that have documentation
                data.version_metadata.forEach(version => {
                    // Skip main-devel (it's shown as "latest")
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
                
                // Sort versions (latest first, then by version number descending)
                VERSIONS.sort((a, b) => {
                    if (a.value === 'latest') return -1;
                    if (b.value === 'latest') return 1;
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
            select.innerHTML = '';
            VERSIONS.forEach(v => {
                const option = document.createElement('option');
                option.value = v.value;
                option.textContent = v.display;
                if (v.value === getCurrentVersion() || (getCurrentVersion() === 'latest' && v.value === 'latest')) {
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

