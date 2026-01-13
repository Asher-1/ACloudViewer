# GitHub Pages Setup Guide

## 🎯 Deployment Architecture

ACloudViewer uses a unified `gh-pages` branch to deploy all static content:

```
gh-pages branch structure:
├── index.html              # Main website homepage
├── downloads_data.json     # Download data
├── images/                 # Main website images
├── scripts/                # Main website scripts
├── styles/                 # Main website styles
└── documentation/          # API documentation (auto-generated)
    ├── index.html          # API documentation homepage
    ├── _static/            # Sphinx static files
    └── ...                 # Other documentation files
```

### 📍 Access URLs

- **Main Website**: https://asher-1.github.io/ACloudViewer/
- **API Documentation**: https://asher-1.github.io/ACloudViewer/documentation/

---

## ⚙️ GitHub Pages Setup Steps

### Step 1: Access Repository Settings

1. Open your browser and visit: https://github.com/Asher-1/ACloudViewer/settings/pages
2. Or navigate in GitHub repository: Click **Settings** → **Pages**

### Step 2: Configure Deployment Source

In the **Build and deployment** section:

1. **Source**: Select **Deploy from a branch**

2. **Branch**: 
   - Branch dropdown: Select **`gh-pages`**
   - Folder dropdown: Select **`/ (root)`**

3. Click **Save** button to save configuration

### Step 3: Wait for Deployment

- GitHub will automatically start deployment (usually 1-3 minutes)
- After deployment completes, a green banner will appear at the top:
  ```
  ✓ Your site is live at https://asher-1.github.io/ACloudViewer/
  ```

### Step 4: Verify Deployment

Visit the following URLs to confirm successful deployment:

```bash
# Main website
curl -I https://asher-1.github.io/ACloudViewer/

# API documentation
curl -I https://asher-1.github.io/ACloudViewer/documentation/
```

---

## 🚀 Automated Deployment Process

### Workflow Trigger Conditions

The `documentation.yml` workflow triggers on:

1. **Push to main**: When code is pushed to the `main` branch
2. **Pull Request**: When a PR is created or updated (build only, no deployment)
3. **Manual trigger**: Manually run the workflow from Actions tab

### Deployment Steps

When code is pushed to the `main` branch, the workflow will:

1. **Build API Documentation**
   - Use Docker to build documentation
   - Extract generated HTML files

2. **Upload to Release**
   - Package documentation and upload to `main-devel` release

3. **Deploy Main Website**
   - Copy main website files from `docs/` to `gh-pages` branch root directory

4. **Deploy API Documentation**
   - Deploy generated documentation to `documentation/` directory in `gh-pages` branch

5. **Update GitHub Pages**
   - GitHub automatically detects `gh-pages` branch update
   - Republish website (usually 1-2 minutes)

---

## 🔧 Troubleshooting

### Problem 1: Page Shows 404

**Cause**: GitHub Pages might not be enabled or configured incorrectly

**Solutions**:
1. Confirm GitHub Pages is enabled (see configuration steps above)
2. Confirm source branch is set to `gh-pages` not `main`
3. Wait 2-3 minutes for deployment to take effect
4. Clear browser cache and retry

### Problem 2: API Documentation Links Broken

**Cause**: Documentation path has changed

**Check**:
```bash
# Check gh-pages branch structure
git fetch origin gh-pages
git ls-tree -r --name-only origin/gh-pages | grep documentation
```

**Solutions**:
- Ensure `destination_dir: documentation` is configured correctly in workflow
- Ensure main website documentation links point to `/documentation/`

### Problem 3: Main Website and Documentation Overwrite Each Other

**Cause**: `keep_files` setting is incorrect

**Solutions**:
- Ensure both deployment steps have `keep_files: true` set
- Ensure `force_orphan: true` is not used (it deletes all historical files)

### Problem 4: Workflow Permission Errors

**Error Message**: `Resource not accessible by integration`

**Solutions**:
Ensure workflow has sufficient permissions:
```yaml
permissions:
  contents: write        # Required: upload artifacts and releases
  pages: write           # Required: deploy to GitHub Pages
  id-token: write        # Required: GitHub Pages deployment authentication
  pull-requests: write   # Required: comment on PRs
  issues: write          # Required: create issue comments
```

---

## 📚 Technical Details

### peaceiris/actions-gh-pages How It Works

1. **Clone `gh-pages` branch** (create if it doesn't exist)
2. **Copy files** to specified directory (`destination_dir`)
3. **Commit changes** to `gh-pages` branch
4. **Push to remote** triggering GitHub Pages redeployment

### keep_files Parameter Usage

```yaml
keep_files: true   # Keep existing files, only update/add new files
keep_files: false  # Delete all files, completely replace (default)
```

### Deployment Order Importance

Workflow must deploy main website first, then API documentation:

```yaml
# 1. Deploy main website first (root directory)
- uses: peaceiris/actions-gh-pages@v3
  with:
    publish_dir: ./docs
    # No destination_dir, deploys to root directory

# 2. Then deploy API documentation (subdirectory)
- uses: peaceiris/actions-gh-pages@v3
  with:
    publish_dir: ./docs-output
    destination_dir: documentation  # Deploy to subdirectory
```

This ensures:
- ✅ Main website files in root directory
- ✅ API documentation in `/documentation/` subdirectory
- ✅ Both coexist peacefully with `keep_files: true`

---

## ✅ Configuration Checklist

Before deployment, confirm:

- [ ] GitHub Pages is enabled
- [ ] Source is set to `gh-pages` branch
- [ ] Folder is set to `/ (root)`
- [ ] Workflow permissions are configured completely
- [ ] `keep_files: true` is set in both deployment steps
- [ ] Main website links point to `/documentation/`

---

## 📖 Related Documentation

- [GitHub Pages Official Documentation](https://docs.github.com/en/pages)
- [peaceiris/actions-gh-pages](https://github.com/peaceiris/actions-gh-pages)
- [CI Documentation Build Guide](../automation/BUILD_DOCUMENTATION.md)

---

**Last Updated**: 2026-01-13  
**Maintainer**: ACloudViewer Team
