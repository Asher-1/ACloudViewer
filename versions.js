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

    // Cache for documentation existence checks (to avoid repeated requests)
    const DOC_EXISTENCE_CACHE = new Map();

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

    // Check if documentation exists for a version
    async function checkDocumentationExists(versionPath) {
        try {
            const baseUrl = getBaseUrl();
            const docUrl = `${baseUrl}/${versionPath}/index.html`;
            const response = await fetch(docUrl, { method: 'HEAD' });
            return response.ok;
        } catch (error) {
            return false;
        }
    }

    // Parse version string to compare (e.g., "v3.9.3" -> [3, 9, 3])
    function parseVersion(versionStr) {
        const match = versionStr.match(/v?(\d+)\.(\d+)\.(\d+)/);
        if (match) {
            return [parseInt(match[1]), parseInt(match[2]), parseInt(match[3])];
        }
        return [0, 0, 0];
    }

    // Compare versions (returns true if v1 >= v2)
    function isVersionGreaterOrEqual(v1, v2) {
        const [major1, minor1, patch1] = parseVersion(v1);
        const [major2, minor2, patch2] = parseVersion(v2);
        
        if (major1 !== major2) return major1 > major2;
        if (minor1 !== minor2) return minor1 > minor2;
        return patch1 >= patch2;
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
                
                // Documentation is available from v3.9.3 onwards
                const MIN_DOC_VERSION = 'v3.9.3';
                
                // Collect version candidates
                const versionCandidates = [];
                
                data.version_metadata.forEach(version => {
                    if (version.value && version.value !== 'main-devel') {
                        // Convert version tag to path format (e.g., v3.9.3 -> v3.9.3)
                        const versionPath = version.value.startsWith('v') ? version.value : `v${version.value}`;
                        
                        // Check if version has documentation
                        let hasDocumentation = false;
                        
                        // Method 1: Use has_documentation field if available (preferred)
                        if (version.hasOwnProperty('has_documentation')) {
                            hasDocumentation = version.has_documentation;
                        } 
                        // Method 2: Fallback to version comparison
                        else if (isVersionGreaterOrEqual(versionPath, MIN_DOC_VERSION)) {
                            hasDocumentation = true;
                        }
                        
                        if (hasDocumentation) {
                            versionCandidates.push({
                                value: versionPath,
                                display: version.display_name || version.value,
                                url: `${getBaseUrl()}/${versionPath}/`,
                                needsVerification: !version.hasOwnProperty('has_documentation')
                            });
                        }
                    }
                });
                
                // For versions without explicit has_documentation flag, verify existence
                const versionsNeedingVerification = versionCandidates.filter(v => v.needsVerification);
                
                if (versionsNeedingVerification.length > 0) {
                    console.info(`Verifying documentation existence for ${versionsNeedingVerification.length} version(s)...`);
                    
                    const existenceChecks = await Promise.all(
                        versionsNeedingVerification.map(async (v) => {
                            const exists = await checkDocumentationExists(v.value);
                            return { version: v, exists: exists };
                        })
                    );
                    
                    // Add only verified versions
                    existenceChecks.forEach(({ version, exists }) => {
                        if (exists) {
                            VERSIONS.push(version);
                        } else {
                            console.info(`Skipping ${version.value}: documentation not found`);
                        }
                    });
                    
                    // Add versions that don't need verification
                    versionCandidates.filter(v => !v.needsVerification).forEach(v => {
                        VERSIONS.push(v);
                    });
                } else {
                    // All versions have explicit documentation flags, no need to verify
                    versionCandidates.forEach(v => {
                        VERSIONS.push(v);
                    });
                }
                
                // Sort versions (latest first, then by version number descending)
                VERSIONS.sort((a, b) => {
                    if (a.value === 'latest') return -1;
                    if (b.value === 'latest') return 1;
                    // Compare version numbers
                    return b.value.localeCompare(a.value, undefined, { numeric: true });
                });
                
                console.info(`Loaded ${VERSIONS.length} documentation versions`);
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

