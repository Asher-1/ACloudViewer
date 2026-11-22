# ACloudViewer Documentation Automation

This directory contains automation scripts, build guides, and deployment documentation for ACloudViewer documentation and website management.

## 📁 Directory Structure

```
docs/automation/
├── README.md                      # This file - automation overview
├── BUILDING_DOCS.md               # Documentation build guide
├── DEPLOYMENT.md                  # GitHub Pages deployment guide
└── scripts/
    ├── scan_releases.py           # Scans GitHub Releases, generates downloads_data.json
    ├── restore_release_archives.sh # Restores known assets through validated staging
    ├── deploy_unified_docs.sh     # Merges site + docs, derives versions.json/sitemap/404
    ├── test_deploy_unified_docs.sh # Contract tests for the deploy script (no CI needed)
    ├── verify_docs_site.py        # Browser e2e: switcher, deep links, 404 rewriting
    ├── check_workflow_limits.py   # Fails CI on workflow files GitHub would reject
    └── README.md                  # Detailed script documentation
```

## 📖 Documentation Guides

### [BUILDING_DOCS.md](BUILDING_DOCS.md)

Complete guide for building ACloudViewer documentation locally and in CI/CD.

**Topics covered**:
- Prerequisites and dependencies
- Build methods (ci_utils.sh, make_docs.py, Docker)
- Documentation structure and components
- Verification and preview
- Troubleshooting common issues
- Tips and best practices

**Quick start**:
```bash
export CI=true NPROC=4 CMAKE_BUILD_PARALLEL_LEVEL=1
source util/ci_utils.sh
build_docs ON
cd docs/_out/html && python3 -m http.server 8080
```

This executes the headless notebook stage and the GUI/WebRTC/ML API stage. When Doxygen is requested, its successful exit must also produce `cpp_api/api/index.html`; a process exit code without that entry page is not a complete C++ API build.

### [DEPLOYMENT.md](DEPLOYMENT.md)

Complete guide for deploying documentation and website to GitHub Pages.

**Topics covered**:
- Deployment architecture
- Automatic and manual deployment
- Documentation versioning (latest vs releases)
- GitHub Actions workflows
- Verification and monitoring
- Troubleshooting deployment issues

**Publication boundary**: Pages is deployed from one workflow artifact, not by committing generated pages to `gh-pages`. A feature-branch build uploads a review artifact and skips production Pages. Publishing requires explicit authorization for a reviewed main-branch change or release event; do not merge or push to main merely to validate a branch. After publication, verify the exact deployment SHA and the real github.io pages, not only a green build.

## 🚀 Scripts Overview

### `scan_releases.py`

**Purpose**: Automatically scans GitHub Releases and generates `downloads_data.json` for the website download selector and documentation version management.

**Usage**:
```bash
# Run manually
python3 docs/automation/scripts/scan_releases.py

# Output
docs/downloads_data.json
```

**Triggered by**:
- `.github/workflows/update-downloads.yml` - Automatically runs after releases
- `.github/workflows/documentation.yml` - Runs during documentation builds

**What it does**:
1. Fetches releases up to the configured inventory limit using explicit API pages
2. Parses asset names to detect platforms, Python versions, architectures
3. Records download metadata; `has_documentation` is a legacy hint, not deployment evidence
4. Generates `downloads_data.json` with:
   - Version metadata (for version selector)
   - Download links (for download page)
   - Platform/Python version support info

**Key Features**:
- ✅ Automatically detects Windows, macOS, Linux packages
- ✅ Parses Python wheel packages (manylinux, macosx, win)
- ✅ Preserves release inclusion; actual documentation versions come from deployed `versions.json`
- ✅ Supports both stable releases and main-devel (Beta)

## 📊 Generated Data Structure

### `downloads_data.json`

```json
{
  "generated_at": "2026-02-01T20:17:49",
  "version_metadata": [
    {
      "value": "v3.9.4",
      "display_name": "v3.9.4",
      "release_date": "2026-02-01",
      "python_versions": ["3.10", "3.11", "3.12", "3.13"],
      "ubuntu_versions": ["ubuntu20.04", "ubuntu22.04", "ubuntu24.04"],
      "has_documentation": true,
      "is_default": false
    },
    {
      "value": "main-devel",
      "display_name": "Beta",
      "has_documentation": true,
      "is_default": true
    }
  ],
  "download_data": {
    "v3.9.4": {
      "windows": { "app": {...}, "wheel": {...} },
      "macos": { "app": {...}, "wheel": {...} },
      "linux": { "ubuntu20.04": {...}, "ubuntu22.04": {...} }
    }
  }
}
```

## 🔧 Configuration

### Minimum Documentation Version

The scanner retains the historical download-metadata hint below. It does not decide which documentation versions exist; `deploy_unified_docs.sh` derives `/documentation/versions.json` from validated directories:

```python
# Only versions >= MIN_DOC_VERSION will have has_documentation=true
MIN_DOC_VERSION = (3, 9, 4)  # Documentation versioning started from v3.9.4
```

### Release Limit

API/transport/JSON failures abort generation rather than publish a partial list. `release_date` is UTC `YYYY-MM-DD` (or null when unknown). Manual cleanup keeps unknown dates conservatively; `max_versions` sorts known dates newest first and filters the metadata array and `download_data` together. These options affect download metadata, not documentation archive retention.

```python
# Change how many releases to scan
releases = fetch_releases(limit=DEFAULT_RELEASE_LIMIT)  # 100 total, explicitly paginated
```

## 🔄 Integration with CI/CD

### Update Downloads Workflow

**File**: `.github/workflows/update-downloads.yml`

**Triggers**:
- After build workflows complete (Windows, macOS, Linux)
- When a release is published or deleted
- Manual trigger with options
- No scheduled trigger is configured

**What it does**:
1. Runs `scan_releases.py` to regenerate `downloads_data.json`
2. Uploads the generated file to the `main-devel` release asset (no git commit)
3. Triggers Documentation on main for automatic events, or when explicitly requested by a manual run; dry runs never upload or deploy

### Documentation Workflow

**File**: `.github/workflows/documentation.yml`

**Integration**:
- Fetches `downloads_data.json` from the `main-devel` release asset
- Falls back to generating from API if not available
- Uses the data for:
  - Download page on main website
- The documentation selector reads `/documentation/versions.json`, generated from restored version directories, independently of download metadata
- Stable documentation archives live on their releases; dev archives named with source SHA, workflow run ID and attempt live on `main-devel`. A main build uploads a uniquely named archive without `--clobber`, confirms the asset exists, then prunes older docs backups. Re-running the same workflow attempt cannot target or delete its predecessor.
- `restore_release_archives.sh` owns archive restoration. It distinguishes a genuinely absent asset from a failed inventory/download and requires a regular root `index.html` before publishing stable or dev trees. If several dev archives coexist, it selects the uniquely newest valid `createdAt`; missing/invalid/tied timestamps or a broken newest archive stop restoration instead of silently falling back to older content
- Each merge uses fresh staging. Report logical content bytes, the exact uncompressed tar stream (including headers/padding), and a gzip transfer estimate separately. The selected project policy blocks tar size **at or above 10,000,000,000 bytes (10 GB decimal)**. Above GitHub Pages' officially supported 1 GB range, emit a risk warning but continue to real deployment; this is not a guarantee of service or deployment success. Gzip size is telemetry, not an independent 1 GB rejection gate. Preserve every historical version, reject invalid archive entries and propagate actual upload/deployment/live-verification failures. The deployment timeout remains enforced; upload success alone never means the live site passed.

### New-build source copies

New Sphinx builds set `html_copy_source = False` alongside the already-hidden source link. This omits the additional `_sources/*.txt` copies; rendered pages, Doxygen header-source HTML, downloadable notebooks with outputs, images and search remain. It does not rewrite historical archives. Raw `_sources` URLs are not generated for new builds, so this is a reading/download-preserving optimization rather than preservation of every raw-file URL. A frozen-output audit measured 97,871,116 bytes saved on the November 2026 dev baseline; the complete multi-version site still requires capacity work.

The same frozen-output comparison measured a deterministic GNU tar stream with fixed metadata and `gzip -n -6`: 507,571,781 bytes with source copies and 440,685,312 bytes without them, saving 66,886,469 bytes (63.79 MiB) in that gzip representation. This is a local transfer estimate, not an Actions ZIP size or a Pages deployment receipt.

### Doxygen publication and worker budget

New builds retain rendered C++ pages, header-source HTML, SVG graphs and their clickable links. The integration copy omits only confirmed root-level Graphviz `.map`/`.md5` sidecars with paired SVG graphs and matching cache signatures; these remain in the producer tree for incremental builds. It preserves JS source maps, downloads, PNG-mode maps and nested user files. The frozen dev API integration omitted 22,483 cache files totaling 17,444,422 bytes, with every retained file byte-identical. Historical archives are unchanged.

Set a positive `NPROC` for bounded local builds. Sphinx workers and generated Doxygen `DOT_NUM_THREADS` both honor it; Doxygen retains a smaller explicit graph-worker setting. A missing template, invalid limit or template I/O failure stops the requested Doxygen build instead of reusing a stale configuration. Account for other concurrent work when choosing the limit.

### Resource completeness

Doxygen's HTML generation can return success after a LaTeX formula failure. The build loads `amsmath` for the metric equations and validates referenced local images before replacing the published C++ API tree; formula PNGs are decoded with the existing Pillow dependency. Plugin READMEs containing raw HTML image tags are also published at their rendered relative paths: MyST's ordinary Markdown image handling does not copy those dependencies automatically. Missing source assets fail the build, and stale output images must not conceal them.

Historical repairs are missing-file overlays, not replacements of release archives. The v3.9.4 mainpage pictures come from its own release source tag; TensorBoard illustrations use the matched, attributed PyTorch 2.7 assets; its formula pictures are bound to the SHA256 of its original `_formulas.tex`. Existing historical HTML and images are not overwritten by these repairs. The workflow fetches only required release tags for source recovery, rather than the complete git history.

### Original QCustomPlot illustrations

Known missing QCustomPlot 1.x illustrations are supplied by 29 newly authored API examples, visibly labelled “Original example”. They use deterministic original data and layouts, not copied upstream documentation artwork. The shared `install_qcp_doc_images.py` publisher validates the reviewed asset hashes and installs only referenced missing images for both new builds and historical overlays. It does not rewrite API HTML or replace a release's existing images. Source, reproduction commands, per-image hashes and the separate generator/image license notices are retained under `assets/qcustomplot-original/` and attribution is distributed with repaired API trees.

## 📝 Maintenance

### When to Update

- **Routine release**: The workflows assemble archives and Pages automatically, but maintainers must confirm archive upload, historical restoration, capacity checks and live browser acceptance
- **Manual review**: Required for changed version filtering/platforms, missing or invalid archive assets, dependency updates, or a production deployment that passes upload but fails live verification

### Testing Changes

```bash
# 1. Make changes to scan_releases.py
vim docs/automation/scripts/scan_releases.py

# 2. Test locally
python3 docs/automation/scripts/scan_releases.py

# 3. Verify output
cat docs/downloads_data.json | jq '.version_metadata'

# 4. Commit changes
git add docs/automation/scripts/scan_releases.py
git commit -m "feat(automation): update release scanning logic"
```

## 🐛 Troubleshooting

### Issue: No versions showing in selector

**Check**:
```bash
# Verify downloads_data.json exists
ls -lh docs/downloads_data.json

# Check version metadata
jq '.version_metadata[] | select(.has_documentation == true)' docs/downloads_data.json
```

**Solution**: For downloads, verify metadata `value` entries match `download_data` keys. For documentation, inspect `/documentation/versions.json` and require an actual deployed version entry page; the legacy `has_documentation` hint is insufficient.

### Issue: Platform downloads missing

**Check**:
```bash
# View parsed assets for a version
jq '.download_data["v3.9.4"]' docs/downloads_data.json
```

**Solution**: Verify asset naming follows expected patterns in `scan_releases.py`.

## 📚 Related Documentation

- **[Building Documentation](BUILDING_DOCS.md)** - How to build documentation locally and in CI/CD
- **[Deployment Guide](DEPLOYMENT.md)** - How to deploy to GitHub Pages
- **[Version Management](../guides/VERSION_MANAGEMENT.md)** - Documentation version management
- **[Scripts Documentation](scripts/README.md)** - Detailed documentation for automation scripts
- **[GitHub Actions Workflows](https://github.com/Asher-1/ACloudViewer/tree/main/.github/workflows)** - CI/CD workflow configurations

## 🎯 Summary

The automation keeps separate, verifiable contracts:
- Download metadata uses the canonical `version_metadata` array and `download_data` object
- Documentation versions derive from validated archive trees, not the download metadata hint
- Build, archive upload, Pages publication and live browser acceptance are distinct stages; unfinished or failed stages remain incomplete
- Archive integrity, deployment size and dependency changes still require maintenance review

---

**Last Updated**: November 2026
**Maintainer**: ACloudViewer Team  
**Status**: Automation is active; successful production publication and live acceptance must be verified

