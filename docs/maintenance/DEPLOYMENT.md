# ACloudViewer Website Deployment Guide

Complete guide for deploying the ACloudViewer website to GitHub Pages.

## 📋 Overview

ACloudViewer uses **GitHub Pages** to host:
- Main website (homepage, downloads)
- API documentation (Sphinx/Doxygen)
- Download data (JSON)

All content is deployed to the `gh-pages` branch and served from `https://asher-1.github.io/ACloudViewer/`.

## 🏗️ Deployment Architecture

```
GitHub Repository (main branch)
├── docs/
│   ├── index.html          # Main website
│   ├── styles.css
│   ├── script.js
│   └── images/
│
└── .github/workflows/
    └── documentation.yml   # Automated deployment

                ↓ GitHub Actions

gh-pages Branch
├── index.html              # Main website (root)
├── downloads_data.json     # Download links
├── images/
├── scripts/
├── styles/
└── documentation/          # API docs (subdirectory)
    ├── index.html
    ├── python_api/
    ├── cpp_api/
    └── tutorial/
```

## 🚀 Quick Deployment

### Prerequisites

- Repository write access
- Git configured
- GitHub Pages enabled in repository settings

### Deploy to GitHub Pages

```bash
# 1. Navigate to project root
cd /path/to/ACloudViewer

# 2. Add changes
git add docs/

# 3. Commit changes
git commit -m "docs: update website"

# 4. Push to GitHub
git push origin main
```

GitHub Actions will automatically:
1. Build documentation (if `documentation.yml` is triggered)
2. Deploy main website to gh-pages root
3. Deploy API docs to gh-pages/documentation/
4. Publish to https://asher-1.github.io/ACloudViewer/

## ⚙️ GitHub Pages Configuration

### Step 1: Access Settings

1. Go to: https://github.com/Asher-1/ACloudViewer/settings/pages
2. Or click: **Settings** → **Pages** in repository

### Step 2: Configure Source

Under **Build and deployment**:

1. **Source**: `Deploy from a branch`
2. **Branch**: `gh-pages`
3. **Folder**: `/ (root)`
4. Click **Save**

### Step 3: Verify

Wait 1-2 minutes, then visit:
- Main site: https://asher-1.github.io/ACloudViewer/
- API docs: https://asher-1.github.io/ACloudViewer/documentation/

## 📦 File Structure

### Source Files (main branch)

```
docs/
├── index.html              # Main website source
├── styles.css              # Styles
├── script.js               # JavaScript
├── 404.html                # Error page
├── robots.txt              # SEO config
├── sitemap.xml             # Site map
├── .nojekyll              # Disable Jekyll
├── images/                 # Images
└── source/                 # Sphinx source (for API docs)
```

### Deployed Files (gh-pages branch)

```
gh-pages/
├── index.html              # Main website
├── downloads_data.json     # Download data (auto-generated)
├── images/                 # Website images
├── scripts/                # Website scripts
├── styles/                 # Website styles
└── documentation/          # API documentation (auto-generated)
    ├── index.html
    ├── _static/
    ├── python_api/
    ├── cpp_api/
    └── tutorial/
```

## 🤖 Automated Deployment

### GitHub Actions Workflow

**File**: `.github/workflows/documentation.yml`

**Triggers**:
- Push to `main` branch
- Pull request (build only)
- Manual workflow dispatch

**Steps**:
1. Build documentation in Docker
2. Extract HTML files
3. Deploy main website to gh-pages root
4. Deploy API docs to gh-pages/documentation/
5. Update downloads_data.json

**Key Configuration**:
```yaml
- name: Deploy to GitHub Pages
  uses: peaceiris/actions-gh-pages@v3
  with:
    github_token: ${{ secrets.GITHUB_TOKEN }}
    publish_dir: ./docs/_out/html
    destination_dir: documentation
    keep_files: true  # Preserve existing files
```

### Manual Trigger

1. Go to: https://github.com/Asher-1/ACloudViewer/actions
2. Select "Documentation" workflow
3. Click "Run workflow"
4. Choose branch (usually `main`)
5. Click "Run workflow" button

## 🔧 Manual Deployment

### Deploy Main Website Only

```bash
# Build locally (if needed)
cd docs
# Make changes to index.html, styles.css, etc.

# Deploy using gh-pages CLI tool
npm install -g gh-pages

# Deploy
gh-pages -d docs -b gh-pages
```

### Deploy API Documentation Only

```bash
# Build documentation
cd docs
python make_docs.py --sphinx --doxygen

# Deploy
gh-pages -d _out/html -b gh-pages -e documentation
```

## 🔍 Verification

### Check Deployment Status

```bash
# Check if site is accessible
curl -I https://asher-1.github.io/ACloudViewer/

# Check documentation
curl -I https://asher-1.github.io/ACloudViewer/documentation/

# View gh-pages branch
git fetch origin gh-pages
git log origin/gh-pages --oneline -10
```

### Verify Features

- ✅ Homepage loads correctly
- ✅ Images display properly
- ✅ Navigation works
- ✅ Download links functional
- ✅ API documentation accessible
- ✅ Mobile responsive
- ✅ Search engines can crawl (robots.txt)

## 🐛 Troubleshooting

### Problem: 404 Error

**Causes**:
- GitHub Pages not enabled
- Wrong branch/folder configured
- Deployment in progress

**Solutions**:
```bash
# 1. Check GitHub Pages settings
# Visit: Settings → Pages → Verify configuration

# 2. Check gh-pages branch exists
git ls-remote origin gh-pages

# 3. Wait 2-3 minutes for deployment
# GitHub Pages needs time to build and deploy

# 4. Clear browser cache
# Hard refresh: Ctrl+F5 (Windows/Linux) or Cmd+Shift+R (macOS)
```

### Problem: Broken Links

**Causes**:
- Incorrect relative paths
- Missing files
- Case sensitivity issues

**Solutions**:
```bash
# Check gh-pages branch structure
git fetch origin gh-pages
git ls-tree -r --name-only origin/gh-pages

# Test links locally
cd docs
python3 -m http.server 8080
# Visit http://localhost:8080
```

### Problem: Documentation Not Updated

**Causes**:
- GitHub Actions failed
- Wrong workflow triggered
- Cache issues

**Solutions**:
```bash
# 1. Check GitHub Actions status
# Visit: https://github.com/Asher-1/ACloudViewer/actions

# 2. View workflow logs
# Click on failed workflow → View details

# 3. Re-run workflow
# Click "Re-run jobs" button

# 4. Clear GitHub Pages cache
# Settings → Pages → "Change branch" and back
```

### Problem: Permission Errors

**Error**: `Resource not accessible by integration`

**Solution**:
```yaml
# Ensure workflow has correct permissions
permissions:
  contents: write
  pages: write
  id-token: write
```

## 📊 Monitoring

### Deployment Statistics

```bash
# View deployment history
git log origin/gh-pages --oneline -20

# Check last deployment date
git log origin/gh-pages -1 --format="%ai"

# Count files in deployment
git ls-tree -r --name-only origin/gh-pages | wc -l
```

### Analytics

GitHub Pages doesn't provide built-in analytics. To add analytics:

1. **Google Analytics**: Add tracking code to `docs/index.html`
2. **Plausible**: Lightweight, privacy-friendly alternative
3. **GitHub Traffic**: Repository Insights → Traffic

## 🔒 Security

### Important Files

**`.nojekyll`**
```
# Empty file - tells GitHub Pages to skip Jekyll processing
# Required for files starting with underscore
```

**`robots.txt`**
```
User-agent: *
Allow: /

Sitemap: https://asher-1.github.io/ACloudViewer/sitemap.xml
```

**`404.html`**
```html
<!-- Custom 404 error page -->
<!-- Redirects users to homepage or shows helpful message -->
```

### Best Practices

- ✅ Never commit secrets to gh-pages branch
- ✅ Use `keep_files: true` to prevent data loss
- ✅ Review changes before deployment
- ✅ Test locally before pushing
- ✅ Monitor deployment logs

## 📚 Related Documentation

- [GitHub Pages Setup Guide](../guides/GITHUB_PAGES_SETUP.md)
- [Build Documentation Guide](../automation/BUILD_DOCUMENTATION.md)
- [Download Automation Guide](../automation/README.md)
- [GitHub Pages Official Docs](https://docs.github.com/en/pages)
- [peaceiris/actions-gh-pages](https://github.com/peaceiris/actions-gh-pages)

## ✅ Deployment Checklist

Before deploying:

- [ ] Test changes locally
- [ ] Verify all links work
- [ ] Check mobile responsiveness
- [ ] Validate HTML/CSS
- [ ] Test download links
- [ ] Review GitHub Actions logs
- [ ] Confirm gh-pages branch updated
- [ ] Wait 2-3 minutes for GitHub Pages to rebuild
- [ ] Verify live site
- [ ] Check search engine accessibility

---

**Last Updated**: 2026-01-13  
**Maintained by**: ACloudViewer Team
