# ACloudViewer Documentation Deployment Guide

Complete guide for deploying ACloudViewer documentation and website to GitHub Pages.

## 📋 Overview

ACloudViewer uses **GitHub Pages** to host:
- **Main website**: Homepage, downloads, donation page
- **API documentation**: Sphinx (Python) + Doxygen (C++)
- **Download metadata**: `downloads_data.json` (auto-generated)

All content is deployed to the `gh-pages` branch and served from:  
**https://asher-1.github.io/ACloudViewer/**

## 🏗️ Deployment Architecture

```
GitHub Repository (main branch)
├── docs/
│   ├── index.html              # Main website
│   ├── styles.css
│   ├── script.js
│   ├── images/
│   └── source/                 # Sphinx documentation source
│
└── .github/workflows/
    ├── documentation.yml       # Documentation build & deploy
    └── update-downloads.yml    # downloads_data.json update

                ↓ GitHub Actions

gh-pages Branch
├── index.html                  # Main website (root)
├── downloads_data.json         # Download links (auto-updated)
├── images/, scripts/, styles/
└── documentation/
    ├── latest/                 # Latest docs (main branch)
    │   ├── index.html
    │   ├── python_api/
    │   ├── cpp_api/
    │   └── tutorial/
    └── v3.9.4/                 # Version-specific docs (tags)
        ├── index.html
        └── ...
```

## 🚀 Quick Deployment

### Automatic Deployment (Recommended)

**For website updates**:
```bash
# 1. Edit files in docs/
vim docs/index.html

# 2. Commit and push
git add docs/
git commit -m "docs: update website"
git push origin main
```

**For documentation updates**:
```bash
# 1. Edit source files or code
vim docs/source/intro.rst
# or modify Python docstrings

# 2. Commit and push
git add .
git commit -m "docs: update API documentation"
git push origin main
```

GitHub Actions automatically:
1. Builds documentation in Docker (if `documentation.yml` triggered)
2. Deploys website to `gh-pages` root
3. Deploys docs to `gh-pages/documentation/latest/`
4. Preserves version-specific docs (e.g., `v3.9.4/`)

### Manual Workflow Trigger

Manually trigger documentation build:

1. Go to: https://github.com/Asher-1/ACloudViewer/actions
2. Select **"Documentation"** workflow
3. Click **"Run workflow"**
4. Choose branch (usually `main`)
5. Click **"Run workflow"** button

## 📦 Documentation Versioning

### Version Management

ACloudViewer maintains multiple documentation versions:

- **`latest/`**: Built from `main` branch (development docs)
  - Title: "ACloudViewer main-{commit} documentation"
  - Updated on every push to main
  
- **`v{X.Y.Z}/`**: Built from release tags (stable docs)
  - Title: "ACloudViewer {version} documentation"
  - Created when a release is published
  - Example: `v3.9.4/`, `v3.9.5/`

### Version Selector (Mode B)

The documentation includes a version selector that:
- Shows "Latest Stable" for the current stable release at `/documentation/`
- Shows "Development (main)" for development docs at `/documentation/dev/`
- Shows "v{X.Y.Z}" for historical release versions at `/documentation/vX.Y.Z/`
- Only displays versions with `has_documentation: true`
- Filters out versions older than v3.9.4

**Configuration**:
- `docs/automation/scripts/scan_releases.py` generates version metadata
- `docs/downloads_data.json` stores version list
- `docs/source/_static/version_switch.js` implements the UI

See [VERSION_MANAGEMENT.md](../guides/VERSION_MANAGEMENT.md) for details.

## ⚙️ GitHub Pages Configuration

### Initial Setup

**Step 1: Access Settings**

1. Go to: https://github.com/Asher-1/ACloudViewer/settings/pages
2. Or navigate: **Settings** → **Pages**

**Step 2: Configure Source**

Under **Build and deployment**:
1. **Source**: `Deploy from a branch`
2. **Branch**: `gh-pages`
3. **Folder**: `/ (root)`
4. Click **Save**

**Step 3: Verify**

Wait 1-2 minutes, then visit:
- Main site: https://asher-1.github.io/ACloudViewer/
- Latest docs: https://asher-1.github.io/ACloudViewer/documentation/latest/
- Release docs: https://asher-1.github.io/ACloudViewer/documentation/v3.9.4/

## 🤖 Automated Deployment Workflows

### 1. Documentation Workflow

**File**: `.github/workflows/documentation.yml`

**Triggers**:
- Push to `main` branch
- Pull request (build only, no deploy)
- Release publication
- Manual workflow dispatch

**Process**:
1. Set environment variables (`DEVELOPER_BUILD`, `IS_RELEASE`, `DOC_VERSION`)
2. Fetch `downloads_data.json` for version selector
3. Build Docker image (`docker/Dockerfile.docs`)
4. Build documentation inside Docker
5. Extract documentation archive (`.tar.gz`)
6. Upload as workflow artifact
7. **Deploy to GitHub Pages** (main/release only):
   - Main branch → `documentation/latest/`
   - Release tag → `documentation/v{X.Y.Z}/`
   - Website files → root (if updated)

**Key Configuration**:
```yaml
- name: Deploy to GitHub Pages
  uses: peaceiris/actions-gh-pages@v3
  with:
    github_token: ${{ secrets.GITHUB_TOKEN }}
    publish_dir: ./html_extracted
    destination_dir: documentation/${{ env.DOC_VERSION }}
    keep_files: true  # Preserve other versions
```

### 2. Downloads Update Workflow

**File**: `.github/workflows/update-downloads.yml`

**Triggers**:
- Release publication
- Manual workflow dispatch
- Scheduled (weekly)

**Process**:
1. Run `docs/automation/scripts/scan_releases.py`
2. Generate `downloads_data.json` with:
   - Release metadata
   - Download links
   - Documentation availability flags
3. Deploy to `gh-pages` root

## 🔧 Manual Deployment

### Deploy Website Only

```bash
# Install gh-pages CLI tool
npm install -g gh-pages

# Deploy website files
cd /path/to/ACloudViewer
gh-pages -d docs -b gh-pages
```

### Deploy Documentation Only

```bash
# Build documentation locally
cd docs
python make_docs.py --sphinx --doxygen

# Deploy to documentation/test/
gh-pages -d _out/html -b gh-pages -e documentation/test
```

### Deploy Specific Version

```bash
# Build for specific version
export CLOUDVIEWER_VERSION=3.9.4
cd docs
python make_docs.py --sphinx --doxygen --is_release

# Deploy to versioned directory
gh-pages -d _out/html -b gh-pages -e documentation/v3.9.4
```

## 🔍 Verification

### Check Deployment Status

```bash
# Check if site is accessible
curl -I https://asher-1.github.io/ACloudViewer/

# Check latest documentation
curl -I https://asher-1.github.io/ACloudViewer/documentation/latest/

# Check specific version
curl -I https://asher-1.github.io/ACloudViewer/documentation/v3.9.4/

# View gh-pages branch commits
git fetch origin gh-pages
git log origin/gh-pages --oneline --graph -10
```

### Verify Features

- ✅ Homepage loads correctly
- ✅ Images and styles display properly
- ✅ Navigation works (internal links)
- ✅ Download links functional
- ✅ Version selector works
- ✅ Latest documentation accessible
- ✅ Version-specific docs accessible
- ✅ Mobile responsive
- ✅ Search functionality works

### Test Locally

```bash
# Clone gh-pages branch
git fetch origin gh-pages
git checkout gh-pages

# Start local server
python3 -m http.server 8765

# Visit in browser
# http://localhost:8765
# http://localhost:8765/documentation/latest/
```

## 🐛 Troubleshooting

### Problem: 404 Error

**Symptoms**: Site shows "404 Not Found"

**Causes**:
- GitHub Pages not enabled
- Wrong branch/folder configured
- Deployment in progress

**Solutions**:
```bash
# 1. Verify GitHub Pages settings
# Visit: Settings → Pages → Check source is "gh-pages / (root)"

# 2. Check gh-pages branch exists
git ls-remote origin gh-pages

# 3. Wait 2-3 minutes for deployment
# GitHub Pages needs time to build

# 4. Clear browser cache
# Hard refresh: Ctrl+F5 (Windows/Linux) or Cmd+Shift+R (macOS)
```

### Problem: Documentation Not Updated

**Symptoms**: Documentation shows old content

**Causes**:
- GitHub Actions failed
- Wrong workflow triggered
- Browser cache

**Solutions**:
```bash
# 1. Check GitHub Actions status
# Visit: https://github.com/Asher-1/ACloudViewer/actions

# 2. View workflow logs
# Click on workflow run → View job details

# 3. Re-run workflow
# Click "Re-run all jobs" button

# 4. Force refresh browser
# Ctrl+Shift+R or Cmd+Shift+R

# 5. Check if correct version deployed
curl -I https://asher-1.github.io/ACloudViewer/documentation/latest/
# Look for Last-Modified header
```

### Problem: Version Selector Not Working

**Symptoms**: Version dropdown is empty or outdated

**Causes**:
- `downloads_data.json` not updated
- JavaScript error in `version_switch.js`

**Solutions**:
```bash
# 1. Check downloads_data.json exists
curl https://asher-1.github.io/ACloudViewer/downloads_data.json

# 2. Validate JSON format
curl https://asher-1.github.io/ACloudViewer/downloads_data.json | python -m json.tool

# 3. Trigger update-downloads workflow
# Go to Actions → update-downloads → Run workflow

# 4. Check browser console for errors
# F12 → Console tab → Look for JavaScript errors
```

### Problem: Permission Errors

**Error**: `Resource not accessible by integration`

**Solution**:
```yaml
# Ensure workflow has correct permissions in .github/workflows/documentation.yml
permissions:
  contents: write
  pages: write
  id-token: write
```

## 📊 Monitoring

### Deployment History

```bash
# View recent deployments
git log origin/gh-pages --oneline --graph -20

# Check last deployment date
git log origin/gh-pages -1 --format="%ai %s"

# List files in gh-pages
git ls-tree -r --name-only origin/gh-pages | head -30

# Check documentation directories
git ls-tree -r --name-only origin/gh-pages | grep documentation/
```

### File Structure

```bash
# View gh-pages structure
git fetch origin gh-pages
git ls-tree -d -r origin/gh-pages

# Example output:
# documentation/
# documentation/latest/
# documentation/v3.9.4/
# images/
# scripts/
# styles/
```

## 🔒 Security Best Practices

### Important Files

**`.nojekyll`**
```
# Empty file - tells GitHub Pages to skip Jekyll processing
# Required for Sphinx docs (files starting with underscore)
```

**`robots.txt`**
```
User-agent: *
Allow: /

Sitemap: https://asher-1.github.io/ACloudViewer/sitemap.xml
```

### Best Practices

- ✅ Never commit secrets or API keys
- ✅ Use `keep_files: true` in deployment to preserve other versions
- ✅ Review changes before pushing to main
- ✅ Test documentation builds locally before committing
- ✅ Monitor deployment logs for errors
- ✅ Keep documentation versions synchronized with releases

## 📚 Related Documentation

- **Build Documentation**: [BUILDING_DOCS.md](BUILDING_DOCS.md)
- **Version Management**: [../guides/VERSION_MANAGEMENT.md](../guides/VERSION_MANAGEMENT.md)
- **Automation Scripts**: [scripts/README.md](scripts/README.md)
- **Automation Overview**: [README.md](README.md)

## ✅ Deployment Checklist

Before deploying:

- [ ] Test documentation build locally
- [ ] Verify all internal links work
- [ ] Check version selector displays correctly
- [ ] Test mobile responsiveness
- [ ] Validate HTML/CSS (no console errors)
- [ ] Test download links
- [ ] Review GitHub Actions logs
- [ ] Confirm gh-pages branch updated
- [ ] Wait 2-3 minutes for GitHub Pages rebuild
- [ ] Verify live site in browser
- [ ] Test version switching
- [ ] Check that old versions are preserved

---

**Last Updated**: February 2026  
**Maintainer**: ACloudViewer Team  
**Status**: ✅ Production Ready

