/*
 * Version switcher for ACloudViewer documentation.
 *
 * SINGLE SOURCE OF TRUTH
 * ----------------------
 * The deployment layout is the truth, not any hand-maintained list:
 *
 *   /documentation/            redirect -> newest release (or /dev if none)
 *   /documentation/dev/        development build from main
 *   /documentation/vX.Y.Z/     permanent version archives
 *   /documentation/versions.json   generated at deploy time from the
 *                                  directories that actually exist
 *
 * Every version UI on the site reads versions.json. There is deliberately no
 * hardcoded version array here: a list baked into JS silently rots the moment
 * a release is published or an archive fails to restore, and it cannot know
 * whether a version directory is actually reachable.
 *
 * Link rewriting is path-preserving: switching versions keeps the reader on
 * the page they were on (`.../v3.9.4/python_api/index.html` -> the same page
 * under another version). If that page does not exist in the target version
 * we fall back to that version's root rather than serving a 404.
 */
(function () {
    'use strict';

    // Resolve the documentation root from the current URL so the switcher
    // works under any deploy prefix (project pages, custom domain, local
    // preview) instead of assuming /ACloudViewer/documentation. The root is
    // the path up to and including the first `documentation` segment, which
    // covers every page shape: the bare root, `<root>/index.html`, and any
    // depth beneath a version directory.
    var DOC_ROOT = (function () {
        var m = window.location.pathname.match(/^(.*?\/documentation)(?:\/|$)/);
        if (m) {
            return m[1];
        }
        // Nothing documentation-like in the path: fall back to the canonical
        // production prefix rather than guessing.
        return '/ACloudViewer/documentation';
    })();

    var VERSIONS_JSON = DOC_ROOT + '/versions.json';

    // Last-resort list used only when versions.json cannot be fetched. It is
    // intentionally minimal: enough to leave the page, never a fake catalogue.
    var FALLBACK_VERSIONS = [
        { id: 'dev', label: 'Development (main)', kind: 'dev' }
    ];

    var state = {
        versions: FALLBACK_VERSIONS.slice(),
        latest: '',
        current: 'stable',
        loaded: false
    };

    // ------------------------------------------------------------------
    // URL <-> version helpers
    // ------------------------------------------------------------------

    function getCurrentVersion() {
        var path = window.location.pathname;
        var archive = path.match(/\/documentation\/(v\d+\.\d+\.\d+)\//);
        if (archive) {
            return archive[1];
        }
        if (/\/documentation\/dev\/?$/.test(path) || path.indexOf('/documentation/dev/') !== -1) {
            return 'dev';
        }
        // /documentation/ itself redirects to the newest release.
        return 'stable';
    }

    // The page path below the version segment, e.g. "python_api/index.html".
    // Returns "" when the reader is at a version root.
    function getRelativePath() {
        var path = window.location.pathname;
        var rest = path.replace(/^.*\/documentation\/(?:dev\/|v\d+\.\d+\.\d+\/)?/, '');
        rest = rest.replace(/^\/+/, '');
        return rest === 'index.html' ? '' : rest;
    }

    function versionUrl(versionId, relativePath) {
        var base;
        if (versionId === 'stable') {
            // `stable` is the redirect at the documentation root.
            base = DOC_ROOT + '/';
        } else if (versionId === 'dev') {
            base = DOC_ROOT + '/dev/';
        } else {
            base = DOC_ROOT + '/' + versionId + '/';
        }
        return relativePath ? base + relativePath : base;
    }

    // ------------------------------------------------------------------
    // Existence probe. HEAD is cheap and avoids downloading the page just to
    // learn whether the reader's current page exists in the target version.
    // A same-origin HEAD that fails (404, or a server that rejects HEAD)
    // resolves false and we fall back to the version root.
    // ------------------------------------------------------------------

    function exists(url) {
        return fetch(url, { method: 'HEAD', redirect: 'follow' })
            .then(function (r) { return r.ok; })
            .catch(function () { return false; });
    }

    function switchVersion(versionId) {
        var relativePath = getRelativePath();
        var position = window.location.search + window.location.hash;
        var target = versionUrl(versionId, relativePath) + position;

        if (!relativePath) {
            window.location.href = target;
            return;
        }

        exists(target).then(function (ok) {
            window.location.href = ok ? target : versionUrl(versionId, '');
        });
    }

    // ------------------------------------------------------------------
    // Load versions.json
    // ------------------------------------------------------------------

    function loadVersions() {
        return fetch(VERSIONS_JSON, { cache: 'no-cache' })
            .then(function (r) {
                if (!r.ok) {
                    throw new Error('versions.json HTTP ' + r.status);
                }
                return r.json();
            })
            .then(function (data) {
                if (!data || !Array.isArray(data.versions) || data.versions.length === 0) {
                    throw new Error('versions.json has no versions');
                }
                state.versions = data.versions;
                state.latest = data.latest || '';
                state.loaded = true;
            })
            .catch(function (err) {
                // Degrade to the fallback list; never leave a dead control.
                console.warn('[docs] version list unavailable:', err.message);
                state.versions = FALLBACK_VERSIONS.slice();
                state.loaded = false;
            });
    }

    // ------------------------------------------------------------------
    // Rendering. The sidebar table and the landing-page dropdown are both
    // filled from one state, so they cannot disagree.
    // ------------------------------------------------------------------

    // Label for a version, matching the Open3D table convention: the release
    // number, annotated when it is the newest or the development build.
    function versionLabel(v) {
        if (v.id === 'dev') {
            return v.label || 'Development (main)';
        }
        if (v.kind === 'latest') {
            return (v.label || v.id) + ' (latest stable)';
        }
        return v.label || v.id;
    }

    // `stable` is a redirect, not a directory: versions.json never lists it.
    // Resolve it to the version the redirect actually serves.
    function currentSelection() {
        var current = getCurrentVersion();
        return current === 'stable' ? (state.latest || 'dev') : current;
    }

    // Directory that holds a version's HTML. `stable` resolves to the docs
    // root, which redirects to the newest release.
    function versionDir(v) {
        if (v.id === 'stable') {
            return DOC_ROOT + '/';
        }
        return DOC_ROOT + '/' + v.id + '/';
    }

    // Sidebar table, mirroring the Open3D docs site: one row per version with
    // the docs entry point and the C++ API entry point side by side.
    //
    // The C++ entry point is `cpp_api.html`, not `cpp_api/`: the Sphinx build
    // emits cpp_api.html plus a cpp_api/ directory of sub-pages, and only the
    // former is a usable landing page.
    function renderVersionTable() {
        var tables = document.querySelectorAll('[data-docs-version-table]');
        if (!tables.length) {
            return;
        }
        var selected = currentSelection();

        Array.prototype.forEach.call(tables, function (container) {
            container.innerHTML = '';

            var table = document.createElement('table');
            table.className = 'version-table';

            state.versions.forEach(function (v) {
                var dir = versionDir(v);
                var row = document.createElement('tr');
                if (v.id === selected) {
                    row.className = 'is-current';
                }

                var docCell = document.createElement('td');
                docCell.className = 'version-cell';
                var docLink = document.createElement('a');
                docLink.href = versionUrl(v.id, getRelativePath()) +
                    window.location.search + window.location.hash;
                docLink.addEventListener('click', function (event) {
                    // Keep normal links usable for new tabs and copy-link.
                    if (event.button || event.ctrlKey || event.metaKey ||
                            event.shiftKey || event.altKey) { return; }
                    event.preventDefault();
                    switchVersion(v.id);
                });
                docLink.textContent = versionLabel(v);
                docCell.appendChild(docLink);

                var cppCell = document.createElement('td');
                cppCell.className = 'cpp-cell';
                var cppLink = document.createElement('a');
                cppLink.href = dir + 'cpp_api.html';
                cppLink.textContent = v.id === 'dev'
                    ? 'C++ API (main)'
                    : 'C++ API';
                cppCell.appendChild(cppLink);

                row.appendChild(docCell);
                row.appendChild(cppCell);
                table.appendChild(row);
            });

            container.appendChild(table);
        });
    }

    // Landing-page dropdown. Kept for the version picker embedded in the
    // index page; the sidebar uses the table above.
    function renderSelectors() {
        var selected = currentSelection();
        // Immutable release HTML predates data-docs-version and calls the
        // legacy API from inline scripts. Its IDs remain part of our contract.
        var selects = document.querySelectorAll(
            'select[data-docs-version], #docs-version-select-sidebar, #docs-version-select-main');

        Array.prototype.forEach.call(selects, function (select) {
            select.innerHTML = '';

            state.versions.forEach(function (v) {
                var option = document.createElement('option');
                option.value = v.id;
                option.textContent = versionLabel(v);
                if (v.id === selected) {
                    option.selected = true;
                }
                select.appendChild(option);
            });

            select.disabled = false;
        });
    }

    // Fill every version UI on the page.
    function render() {
        renderVersionTable();
        renderSelectors();
    }

    // One delegated listener for the whole document. Templates therefore only
    // need to mark a <select data-docs-version>; they do not have to repeat an
    // inline onchange, and a select that forgets one still works. Markup is
    // not where this behaviour should be duplicated.
    function bindSelectors() {
        document.addEventListener('change', function (event) {
            var select = event.target;
            if (!select || !select.matches || !select.matches('select[data-docs-version]')) {
                return;
            }
            // Archived templates already navigate in inline onchange. Do not
            // issue a second HEAD probe/navigation if they also carry a marker.
            if (select.value && !select.hasAttribute('onchange')) {
                switchVersion(select.value);
            }
        });
    }

    function init() {
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', function () {
                init();
            });
            return;
        }
        bindSelectors();
        loadVersions().then(function () {
            state.current = getCurrentVersion();
            render();
            document.dispatchEvent(new CustomEvent('docs:versionsready', {
                detail: {
                    versions: state.versions,
                    latest: state.latest,
                    current: state.current
                }
            }));
            // v3.9.4 inline initializers listen for this historical event.
            document.dispatchEvent(new CustomEvent('versionsLoaded'));
        });
    }

    window.ACloudViewerVersionSwitcher = {
        switchVersion: switchVersion,
        getCurrentVersion: getCurrentVersion,
        getVersions: function () {
            // Preserve the canonical id/label/kind fields and supply the old
            // template's value/display aliases only at the runtime boundary.
            // versions.json itself keeps its single canonical schema.
            return state.versions.map(function (v) {
                return Object.assign({}, v, { value: v.id, display: versionLabel(v) });
            });
        },
        getLatest: function () { return state.latest; },
        render: render,
        renderVersionTable: renderVersionTable,
        renderSelectors: renderSelectors,
        updateSidebarSelector: renderSelectors,
        init: init
    };

    window.addEventListener('hashchange', renderVersionTable);
    init();
})();
