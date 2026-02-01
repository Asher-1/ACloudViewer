# ACloudViewer Documentation Automation

This directory contains automation scripts, build guides, and deployment documentation for ACloudViewer documentation and website management.

## 📁 Directory Structure

```
docs/automation/
├── README.md               # This file - automation overview
├── BUILDING_DOCS.md        # Documentation build guide
├── DEPLOYMENT.md           # GitHub Pages deployment guide
└── scripts/
    ├── scan_releases.py    # Scans GitHub Releases and generates downloads_data.json
    └── README.md           # Detailed script documentation
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
source util/ci_utils.sh
build_docs ON
cd docs/_out/html && python3 -m http.server 8080
```

### [DEPLOYMENT.md](DEPLOYMENT.md)

Complete guide for deploying documentation and website to GitHub Pages.

**Topics covered**:
- Deployment architecture
- Automatic and manual deployment
- Documentation versioning (latest vs releases)
- GitHub Actions workflows
- Verification and monitoring
- Troubleshooting deployment issues

**Quick start**:
```bash
git add docs/
git commit -m "docs: update documentation"
git push origin main  # Automatically triggers deployment
```

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
1. Fetches all releases from GitHub API
2. Parses asset names to detect platforms, Python versions, architectures
3. Determines which versions have documentation (`has_documentation` flag)
4. Generates `downloads_data.json` with:
   - Version metadata (for version selector)
   - Download links (for download page)
   - Platform/Python version support info

**Key Features**:
- ✅ Automatically detects Windows, macOS, Linux packages
- ✅ Parses Python wheel packages (manylinux, macosx, win)
- ✅ Filters versions by documentation availability (>= v3.9.4)
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

Edit `scan_releases.py` to change which versions show in the documentation version selector:

```python
# Only versions >= MIN_DOC_VERSION will have has_documentation=true
MIN_DOC_VERSION = (3, 9, 4)  # Documentation versioning started from v3.9.4
```

### Release Limit

```python
# Change how many releases to scan
releases = fetch_releases(limit=5)  # Default: 5
```

## 🔄 Integration with CI/CD

### Update Downloads Workflow

**File**: `.github/workflows/update-downloads.yml`

**Triggers**:
- After build workflows complete (Windows, macOS, Linux)
- When a release is published or deleted
- Manual trigger with options
- Scheduled daily at 00:00 UTC

**What it does**:
1. Runs `scan_releases.py` to regenerate `downloads_data.json`
2. Commits changes to `gh-pages` branch
3. GitHub Pages automatically deploys the updated data

### Documentation Workflow

**File**: `.github/workflows/documentation.yml`

**Integration**:
- Fetches `downloads_data.json` from `gh-pages` branch
- Falls back to generating from API if not available
- Uses the data for:
  - Version selector in documentation sidebar
  - Download page on main website

## 📝 Maintenance

### When to Update

- ✅ **Automatic**: No action needed for normal releases
- ⚠️ **Manual**: Only if changing version filtering logic or adding new platforms

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

**Solution**: Ensure `MIN_DOC_VERSION` is set correctly and versions >= that version exist.

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
- **[GitHub Actions Workflows](../../.github/workflows/)** - CI/CD workflow configurations

## 🎯 Summary

This automation system ensures:
- ✅ Website download links stay up-to-date automatically
- ✅ Documentation version selector only shows available versions
- ✅ Zero manual maintenance required
- ✅ Consistent data across website and documentation

---

**Last Updated**: February 2026  
**Maintainer**: ACloudViewer Team  
**Status**: ✅ Active and Automated

