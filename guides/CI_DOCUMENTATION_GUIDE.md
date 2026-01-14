# CI Documentation Build & Deployment Guide

> **Status**: Production-Ready  
> **Last Updated**: 2026-01-11

## 📋 Overview

This guide describes the automated documentation build and deployment system for ACloudViewer.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    GitHub Event Trigger                      │
│  • Push to main                                              │
│  • Pull Request                                              │
│  • Manual Dispatch                                           │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│              GitHub Actions Workflow                         │
│  .github/workflows/documentation.yml                         │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│                 Docker Build                                 │
│  docker/Dockerfile.docs                                      │
│  • Install system dependencies                               │
│  • Install Python packages                                   │
│  • Install Doxygen                                           │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│            Documentation Generation                          │
│  • Doxygen: Generate C++ API XML                            │
│  • Sphinx: Build HTML documentation                          │
│  • Create tarball                                            │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ├───────────────┬─────────────────┐
                   ▼               ▼                 ▼
        ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
        │   Artifact   │  │   Release    │  │ GitHub Pages │
        │   Upload     │  │   Upload     │  │  Deployment  │
        │              │  │ (main-devel) │  │  (main only) │
        └──────────────┘  └──────────────┘  └──────────────┘
```

## 📁 File Structure

```
ACloudViewer/
├── .github/
│   └── workflows/
│       └── documentation.yml          # CI workflow
├── docker/
│   └── Dockerfile.docs                # Documentation build container
└── docs/
    ├── requirements.txt               # Python dependencies
    ├── Doxyfile                       # Doxygen configuration
    ├── build_docs.sh                  # Local build script
    ├── source/
    │   ├── conf.py                    # Sphinx configuration
    │   ├── index.rst                  # Documentation homepage
    │   ├── _static/                   # Static assets
    │   └── _templates/                # HTML templates
    └── guides/
        └── CI_DOCUMENTATION_GUIDE.md  # This file
```

## 🚀 Workflow Triggers

### 1. Push to Main Branch

```yaml
on:
  push:
    branches:
      - main
```

**What happens**:
- Documentation is built
- Uploaded to GitHub Artifacts
- Uploaded to `main-devel` release
- **Deployed to GitHub Pages** at `https://asher-1.github.io/ACloudViewer/documentation/`

### 2. Pull Request

```yaml
on:
  pull_request:
    types: [opened, reopened, synchronize]
```

**What happens**:
- Documentation is built
- Uploaded to GitHub Artifacts
- **Bot comments on PR** with download link
- NOT deployed to GitHub Pages

### 3. Manual Dispatch

```yaml
on:
  workflow_dispatch:
    inputs:
      developer_build:
        description: 'Set to OFF for Release documentation'
        required: false
        default: 'ON'
```

**What happens**:
- Can trigger from GitHub Actions tab
- Can choose `DEVELOPER_BUILD` option
- Same as push to main

## 🐳 Docker Build Process

### Dockerfile.docs

```dockerfile
FROM ubuntu:22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip doxygen graphviz

# Install Python documentation tools
COPY docs/requirements.txt /root/ACloudViewer/docs/
RUN pip3 install -r docs/requirements.txt

# Copy repository
COPY . /root/ACloudViewer/

# Build documentation
RUN cd docs && \
    doxygen Doxyfile && \
    make html && \
    tar -czf acloudviewer-docs.tar.gz html/
```

### Key Features

1. **Isolated Environment**: No dependency conflicts
2. **Reproducible Builds**: Same environment every time
3. **Version Pinning**: Exact package versions
4. **Multi-stage Capable**: Can optimize further if needed

## 📦 Build Steps

### Step 1: Environment Setup

```yaml
- name: Checkout source code
  uses: actions/checkout@v4
  with:
    fetch-depth: 0  # Full git history
```

### Step 2: Docker Build

```yaml
- name: Build documentation Docker image
  run: |
    docker build \
      --build-arg BASE_IMAGE=ubuntu:22.04 \
      --build-arg DEVELOPER_BUILD=${DEVELOPER_BUILD} \
      -t acloudviewer-ci:docs \
      -f docker/Dockerfile.docs .
```

### Step 3: Extract Documentation

```yaml
- name: Extract documentation from Docker
  run: |
    docker run \
      -v "${GITHUB_WORKSPACE}:/opt/mount" \
      --rm acloudviewer-ci:docs \
      bash -c "cp /root/ACloudViewer/acloudviewer-*-docs.tar.gz /opt/mount/"
    
    # Rename to use GitHub SHA
    mv acloudviewer-*-docs.tar.gz acloudviewer-${GITHUB_SHA}-docs.tar.gz
```

### Step 4: Upload Artifact

```yaml
- name: Upload documentation artifact
  uses: actions/upload-artifact@v4
  with:
    name: acloudviewer-${{ github.sha }}-docs
    path: acloudviewer-${{ github.sha }}-docs.tar.gz
    compression-level: 0  # Already compressed
```

### Step 5: Update Release (Main branch only)

```yaml
- name: Update main-devel release
  if: github.ref == 'refs/heads/main'
  run: |
    gh release upload main-devel \
      acloudviewer-${{ github.sha }}-docs.tar.gz \
      --clobber
```

### Step 6: Deploy to GitHub Pages (Main branch only)

```yaml
- name: Deploy to GitHub Pages
  if: github.ref == 'refs/heads/main'
  uses: peaceiris/actions-gh-pages@v3
  with:
    github_token: ${{ secrets.GITHUB_TOKEN }}
    publish_dir: ./docs-output
    destination_dir: documentation
    keep_files: true
```

## 🌐 GitHub Pages Deployment

### Configuration

1. **Repository Settings** → **Pages**:
   - Source: Deploy from a branch
   - Branch: `gh-pages`
   - Folder: `/` (root)

2. **URL Structure**:
   ```
   https://asher-1.github.io/ACloudViewer/
   ├── index.html                    # Main website
   ├── downloads_data.json           # Download data
   └── docs/
       └── html/
           └── index.html            # Documentation
   ```

3. **Access Points**:
   - Main Website: `https://asher-1.github.io/ACloudViewer/`
   - Documentation: `https://asher-1.github.io/ACloudViewer/documentation/`
   - Downloads: `https://asher-1.github.io/ACloudViewer/#download`

### Deployment Strategy

```yaml
destination_dir: documentation   # Deploy to subdirectory
keep_files: true                 # Don't delete existing files
```

**Why `keep_files: true`?**
- Preserves `downloads_data.json`
- Preserves main website (`index.html`)
- Only updates `documentation/` directory

## 🧪 Local Testing

### Prerequisites

```bash
# Activate environment
conda activate cloudViewer

# Install dependencies
cd docs
pip install -r requirements.txt

# Install Doxygen
brew install doxygen graphviz  # macOS
# sudo apt-get install doxygen graphviz  # Ubuntu
```

### Build Documentation

```bash
cd docs
./scripts/build_docs.sh
```

### Test with HTTP Server

```bash
python3 -m http.server 8080 --directory _build/html
# Open http://localhost:8080
```

### Test Docker Build (Locally)

```bash
# Build Docker image
docker build \
  --build-arg BASE_IMAGE=ubuntu:22.04 \
  --build-arg DEVELOPER_BUILD=ON \
  -t acloudviewer-ci:docs \
  -f docker/Dockerfile.docs .

# Extract documentation
docker run \
  -v "${PWD}:/opt/mount" \
  --rm acloudviewer-ci:docs \
  bash -c "cp /root/ACloudViewer/acloudviewer-*-docs.tar.gz /opt/mount/"

# Verify
tar -tzf acloudviewer-*-docs.tar.gz | head -20
```

## 🔧 Configuration Options

### Developer vs Release Build

```yaml
DEVELOPER_BUILD: ON   # Default, includes development features
DEVELOPER_BUILD: OFF  # Release mode, clean documentation
```

**Differences**:
- `ON`: Shows all APIs, including experimental
- `OFF`: Only stable, documented APIs

### Sphinx Configuration

Key settings in `source/conf.py`:

```python
# Theme
html_theme = 'furo'  # Modern, clean

# Extensions
extensions = [
    'sphinx.ext.autodoc',      # Python API docs
    'breathe',                 # C++ API bridge
    'exhale',                  # Auto C++ docs
    'sphinx_copybutton',       # Copy code buttons
    'sphinx_tabs.tabs',        # Tabbed content
    'myst_parser',             # Markdown support
]

# Breathe (C++ API)
breathe_projects = {
    "ACloudViewer": "../xml/"
}
```

### Doxygen Configuration

Key settings in `Doxyfile`:

```ini
# Input source code
INPUT = ../libs/CloudViewer \
        ../libs/PCLEngine \
        ../libs/Python/pybind

# Output XML for Sphinx
GENERATE_XML = YES
XML_OUTPUT = xml

# Generate diagrams
HAVE_DOT = YES
DOT_IMAGE_FORMAT = png
```

## 📊 Monitoring

### Check Workflow Status

```bash
# View latest runs
gh run list --workflow=documentation.yml

# View specific run
gh run view <run-id>

# Download artifacts
gh run download <run-id>
```

### Check GitHub Pages

```bash
# Check deployment status
curl -I https://asher-1.github.io/ACloudViewer/documentation/

# Verify content
curl -s https://asher-1.github.io/ACloudViewer/documentation/ | grep '<title>'
```

### Verify Release Upload

```bash
# Check main-devel release
gh release view main-devel

# List assets
gh release view main-devel --json assets
```

## 🐛 Troubleshooting

### Problem: Docker build fails

**Check**:
```bash
# Test locally
docker build -f docker/Dockerfile.docs .

# Check logs
docker build --progress=plain -f docker/Dockerfile.docs .
```

**Common Issues**:
- Missing dependencies in `requirements.txt`
- Doxygen configuration errors
- Source code path issues

### Problem: Sphinx build fails

**Check**:
```bash
cd docs
sphinx-build -b html source _build/html -v
```

**Common Issues**:
- Missing Python packages
- RST syntax errors
- Missing C++ API XML

### Problem: GitHub Pages not updating

**Check**:
1. GitHub Actions succeeded?
2. `gh-pages` branch exists?
3. Pages enabled in settings?
4. Correct `destination_dir`?

**Force rebuild**:
```bash
# Trigger workflow manually
gh workflow run documentation.yml
```

### Problem: Documentation looks broken

**Check**:
- CSS/JS files loaded?
- Relative paths correct?
- Images accessible?

**Debug**:
```bash
# Check browser console
# Check network tab
# Verify file structure in gh-pages branch
```

## 📚 Best Practices

1. **Version Control**:
   - Never commit `_build/` or `xml/` directories
   - Use `.gitignore` for generated files

2. **Documentation Quality**:
   - Write clear docstrings
   - Include code examples
   - Add diagrams where helpful
   - Keep documentation up-to-date

3. **CI/CD**:
   - Test locally before pushing
   - Check build logs on PR
   - Monitor GitHub Pages deployment

4. **Performance**:
   - Use Docker build cache
   - Minimize Docker image size
   - Optimize image assets

5. **Maintenance**:
   - Update dependencies regularly
   - Review and fix warnings
   - Clean up old artifacts

## 🔗 References

- [Open3D Documentation](https://github.com/isl-org/Open3D/tree/main/docs)
- [Open3D Docker Setup](https://github.com/isl-org/Open3D/blob/main/docker/Dockerfile.docs)
- [Open3D CI Workflow](https://github.com/isl-org/Open3D/blob/main/.github/workflows/documentation.yml)
- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [Doxygen Manual](https://www.doxygen.nl/manual/)
- [Furo Theme](https://pradyunsg.me/furo/)
- [GitHub Actions](https://docs.github.com/en/actions)
- [GitHub Pages](https://docs.github.com/en/pages)

## ✅ Checklist for New Setup

- [ ] Install local dependencies (`requirements.txt`)
- [ ] Install Doxygen and Graphviz
- [ ] Run `build_docs.sh` locally
- [ ] Verify documentation builds correctly
- [ ] Create Docker image (`docker build`)
- [ ] Test Docker build
- [ ] Push to GitHub
- [ ] Check GitHub Actions workflow
- [ ] Verify artifact upload
- [ ] Verify GitHub Pages deployment
- [ ] Test live documentation URL
- [ ] Update main website link

---
