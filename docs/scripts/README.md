# Documentation Testing Scripts

This directory contains testing and validation scripts for the ACloudViewer documentation system.

## 📋 Available Scripts

### 🧪 test_doc_structure.sh

**Purpose**: Validates the documentation structure and required files

**Usage**:
```bash
cd /path/to/ACloudViewer
./docs/scripts/test_doc_structure.sh
```

**What it checks**:
- ✅ Required files exist (`Makefile`, `make_docs.py`, `conf.py`, etc.)
- ✅ Directory structure is correct
- ✅ Configuration files are valid
- ✅ Dependencies are available

### 🌐 test_github_pages_locally.sh

**Purpose**: Tests the GitHub Pages dual-layer deployment locally

**Usage**:
```bash
cd /path/to/ACloudViewer
./docs/scripts/test_github_pages_locally.sh [method]
```

**Methods**:
1. `docker` - Full Docker build (default, most complete)
2. `local` - Use existing docs/_out/html
3. `simple` - Quick preview without building

**What it tests**:
- ✅ Main website deployment structure
- ✅ `/documentation/` sub-page deployment
- ✅ File conflicts and overwrites
- ✅ Navigation and links
- ✅ Local preview server

**Example**:
```bash
# Full Docker test
./docs/scripts/test_github_pages_locally.sh docker

# Quick preview of existing docs
./docs/scripts/test_github_pages_locally.sh simple
```

## 🚀 Building Documentation

The scripts in this directory are for **testing only**. To build documentation, use:

### Local Build (Recommended)

```bash
cd docs
make docs
```

This will:
1. Build Python module (if needed)
2. Generate C++ API docs (Doxygen)
3. Generate Python API docs (Sphinx autodoc)
4. Build tutorials (Jupyter notebooks)
5. Create final HTML output in `docs/_out/html/`

### CI/CD Build

The CI/CD system uses `util/ci_utils.sh::build_docs`:

```bash
source util/ci_utils.sh
build_docs OFF  # or ON for developer mode
```

### Docker Build

```bash
docker build --network=host \
    -t acloudviewer-ci:docs \
    -f docker/Dockerfile.docs .
```

## 📁 Related Files

- **Build System**:
  - `docs/Makefile` - Main build orchestration
  - `docs/make_docs.py` - Python build script
  - `util/ci_utils.sh` - CI/CD build functions
  - `docker/Dockerfile.docs` - Docker build definition

- **Configuration**:
  - `docs/source/conf.py` - Sphinx configuration
  - `docs/Doxyfile.in` - Doxygen configuration
  - `docs/requirements.txt` - Python dependencies

- **Testing**:
  - `.github/workflows/documentation.yml` - CI/CD workflow
  - `docs/scripts/` - This directory

## 🔄 Workflow Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Documentation Build                      │
└─────────────────────────────────────────────────────────────┘

Local Development:
  cd docs && make docs → docs/_out/html/

CI/CD (GitHub Actions):
  util/ci_utils.sh::build_docs → docs/_out/html/
  
Docker:
  docker build ... Dockerfile.docs → /root/ACloudViewer/docs/_out/html/

Deployment:
  GitHub Pages:
    - Main website → /
    - API docs → /documentation/
```

## 📚 Documentation

For more information, see:
- [docs/README.md](../README.md) - Main documentation guide
- [util/ci_utils.sh](../../util/ci_utils.sh) - Build functions

## 🧹 Maintenance

These scripts are kept minimal and focused:
- ✅ Structure validation
- ✅ GitHub Pages testing
- ❌ No redundant build scripts (use `make docs` instead)
- ❌ No deprecated test scripts

---

**Last Updated**: 2026-01-13  
**Maintained By**: ACloudViewer Team
