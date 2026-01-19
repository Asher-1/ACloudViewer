// ACloudViewer Documentation Version Switcher
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
                
                // Add versions from metadata
                data.version_metadata.forEach(version => {
                    if (version.value && version.value !== 'main-devel') {
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

    // Create version selector HTML
    function createVersionSelector() {
        const currentVersion = getCurrentVersion();
        const baseUrl = getBaseUrl();
        
        const selector = document.createElement('div');
        selector.id = 'version-selector';
        selector.className = 'version-selector';
        selector.innerHTML = `
            <style>
                .version-selector {
                    position: fixed;
                    top: 20px;
                    right: 20px;
                    z-index: 10000;
                    background: #fff;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    padding: 10px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                    font-size: 14px;
                }
                .version-selector label {
                    display: block;
                    margin-bottom: 5px;
                    font-weight: 600;
                    color: #333;
                }
                .version-selector select {
                    width: 100%;
                    padding: 6px 8px;
                    border: 1px solid #ccc;
                    border-radius: 3px;
                    background: #fff;
                    font-size: 14px;
                    cursor: pointer;
                }
                .version-selector select:hover {
                    border-color: #999;
                }
                .version-selector select:focus {
                    outline: none;
                    border-color: #0066cc;
                    box-shadow: 0 0 0 2px rgba(0, 102, 204, 0.2);
                }
                .version-table {
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 10px;
                }
                .version-table td {
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }
                .version-table td a {
                    color: #0066cc;
                    text-decoration: none;
                }
                .version-table td a:hover {
                    text-decoration: underline;
                }
                .version-table tr:hover {
                    background: #f5f5f5;
                }
            </style>
            <label for="version-select">Documentation Version:</label>
            <select id="version-select" onchange="ACloudViewerVersionSwitcher.switchVersion(this.value)">
                ${VERSIONS.map(v => 
                    `<option value="${v.value}" ${v.value === currentVersion || (currentVersion === 'latest' && v.value === 'latest') ? 'selected' : ''}>
                        ${v.display}
                    </option>`
                ).join('')}
            </select>
        `;
        
        return selector;
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
            // Create and insert version selector
            const selector = createVersionSelector();
            document.body.appendChild(selector);
        });
    }

    // Export functions to global scope
    window.ACloudViewerVersionSwitcher = {
        switchVersion: switchVersion,
        getCurrentVersion: getCurrentVersion,
        getVersions: () => VERSIONS,
        init: init
    };

    // Auto-initialize
    init();
})();

