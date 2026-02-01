# Building ACloudViewer Documentation

Complete guide for building ACloudViewer documentation locally and in CI/CD.

## 🚀 Quick Start

```bash
# Recommended: Use ci_utils.sh (same as CI/CD)
source util/ci_utils.sh
build_docs ON

# Preview
cd docs/_out/html && python3 -m http.server 8080
# Open http://localhost:8080
```

## 📋 Prerequisites

### System Requirements
- **Python**: 3.10-3.13
- **CMake**: 3.18+
- **Doxygen**: For C++ API documentation
- **Pandoc**: For Jupyter notebook conversion

### Install Dependencies

```bash
# Ubuntu/Debian
sudo apt-get update && sudo apt-get install -y \
    build-essential cmake git \
    doxygen pandoc \
    python3-dev python3-pip

# macOS
brew install cmake doxygen pandoc python@3.11

# Python dependencies
pip install -r docs/requirements.txt
```

## 🛠️ Build Methods

### Method 1: Using ci_utils.sh (Recommended)

✅ **Best for**: Most users, matches CI/CD behavior

```bash
cd /path/to/ACloudViewer

# Source the utilities
source util/ci_utils.sh

# Build documentation
build_docs ON   # Development build (faster, uses git commit)
# or
build_docs OFF  # Release build (uses version numbers)

# Output: docs/_out/html/
```

**What it does**:
1. Checks if Python module exists (reuses if found)
2. Builds Python module if needed
3. Generates Python API docs
4. Generates C++ API docs (Doxygen)
5. Copies and processes Jupyter notebooks
6. Builds HTML with Sphinx
7. Shows statistics

### Method 2: Using make_docs.py

✅ **Best for**: Fine-grained control, debugging

```bash
cd docs

# Build everything
python make_docs.py --sphinx --doxygen

# Build only Python API + tutorials
python make_docs.py --sphinx

# Build only C++ API
python make_docs.py --doxygen

# Release build (use version numbers instead of git hash)
python make_docs.py --sphinx --doxygen --is_release

# Clean build
python make_docs.py --clean --sphinx --doxygen

# Parallel build (faster)
python make_docs.py --sphinx --parallel

# Output: docs/_out/html/
```

**Options**:
- `--sphinx` - Build Sphinx documentation (Python API + tutorials)
- `--doxygen` - Build Doxygen documentation (C++ API)
- `--is_release` - Use version numbers (e.g., "3.9.4") instead of "main-{commit}"
- `--clean` - Remove previous build artifacts
- `--parallel` - Use parallel processing (faster)

### Method 3: Docker Build

✅ **Best for**: Isolated environment, CI/CD testing

```bash
# Build documentation in Docker
docker build \
  --build-arg DEVELOPER_BUILD=ON \
  -t acloudviewer-ci:docs \
  -f docker/Dockerfile.docs \
  .

# Extract documentation archive
docker run \
  -v $(pwd):/opt/mount \
  --rm acloudviewer-ci:docs \
  bash -c "cp /root/ACloudViewer/acloudviewer-*-docs.tar.gz /opt/mount/"

# Unpack and preview
mkdir -p docs-preview
tar -xzf acloudviewer-*-docs.tar.gz -C docs-preview/
cd docs-preview && python3 -m http.server 8080
```

## 📚 Documentation Structure

### Generated Output

```
docs/_out/html/
├── index.html              # Main documentation landing page
├── python_api/             # Python API reference (auto-generated)
│   ├── index.html
│   ├── cloudViewer.camera.html
│   ├── cloudViewer.geometry.html
│   ├── cloudViewer.geometry.PointCloud.html
│   └── ...
├── cpp_api/                # C++ API reference (Doxygen)
│   ├── index.html
│   ├── annotated.html
│   └── ...
└── tutorial/               # Jupyter notebook tutorials
    ├── geometry/
    ├── pipelines/
    └── visualization/
```

### Source Files

- **Python API**: Auto-generated from Python docstrings
- **C++ API**: Generated from C++ Doxygen comments in `libs/cloudViewer/`
- **Tutorials**: Jupyter notebooks in `docs/jupyter/`
- **Guides**: ReStructuredText/Markdown files in `docs/source/`

## 🔍 Verification

### Check Build Success

```bash
# Verify output exists
ls -lh docs/_out/html/index.html

# Count generated files
find docs/_out/html -type f | wc -l
find docs/_out/html -name '*.html' | wc -l

# Check size
du -sh docs/_out/html/
```

### Test Python Module

```bash
# Find Python module
cd build_app/lib/Release/Python/cuda  # or cpu

# Test import
python3 -c "import pybind as cloudViewer; print(dir(cloudViewer))"
```

## 🌐 Preview Documentation

### Local HTTP Server

```bash
# Start server
cd docs/_out/html
python3 -m http.server 8080

# Open in browser
# http://localhost:8080
```

### Live Reload (Development)

```bash
# Install sphinx-autobuild
pip install sphinx-autobuild

# Start live server (auto-rebuilds on changes)
cd docs
sphinx-autobuild source _out/html --open-browser

# Or use Makefile
make livehtml
```

## 🐛 Troubleshooting

### Python Module Not Found

**Error**: `ImportError: No module named 'cloudViewer'`

**Solution**:
```bash
# Build Python module
cd build_app
cmake .. -DBUILD_PYTHON_MODULE=ON
make cloudViewer_pybind -j$(nproc)
```

### Doxygen Not Installed

**Error**: `doxygen: command not found`

**Solution**:
```bash
sudo apt-get install doxygen  # Ubuntu/Debian
brew install doxygen          # macOS
```

### Pandoc Not Installed

**Error**: `nbsphinx requires pandoc`

**Solution**:
```bash
sudo apt-get install pandoc  # Ubuntu/Debian
brew install pandoc          # macOS
pip install pypandoc
```

### Sphinx Build Warnings

**Too many warnings**:
```bash
# Warnings are expected during development
# To focus on errors only:
cd docs
python make_docs.py --sphinx 2>&1 | grep -i error
```

### Clean Build

**If build is broken**:
```bash
# Remove all build artifacts
cd docs
rm -rf _out/ source/python_api/

# Rebuild from scratch
python make_docs.py --clean --sphinx --doxygen
```

## 💡 Tips and Best Practices

### 1. Incremental Builds

For faster iteration:
```bash
# Skip Doxygen (slower) if only editing Python/tutorial docs
python make_docs.py --sphinx

# Use live reload for instant preview
make livehtml
```

### 2. Python Module Reuse

`build_docs` automatically reuses existing Python modules:
```bash
# First run: builds Python module + docs (slow)
build_docs ON

# Subsequent runs: reuses module, only rebuilds docs (fast)
build_docs ON
```

### 3. Development vs Release Builds

```bash
# Development: Uses "main-{commit_hash}" in title
python make_docs.py --sphinx

# Release: Uses version number "3.9.4" in title
python make_docs.py --sphinx --is_release
```

### 4. Parallel Builds

Speed up Sphinx builds on multi-core systems:
```bash
python make_docs.py --sphinx --parallel
```

### 5. Documentation Quality

- **Python**: Use Google-style docstrings with type hints
- **C++**: Use Doxygen comments (`///` or `/** */`)
- **Examples**: Include code examples in docstrings
- **Links**: Use Sphinx cross-references (`:class:`, `:func:`)

## 🚀 CI/CD Integration

### GitHub Actions Workflow

**File**: `.github/workflows/documentation.yml`

**Triggers**:
- Push to `main` branch
- Pull requests (for testing)
- Manual dispatch
- Release publication

**Process**:
1. Build Docker image with documentation dependencies
2. Run `build_docs OFF` (release mode)
3. Extract documentation archive
4. Upload as workflow artifact
5. Deploy to GitHub Pages (main branch only)

**Deployment**:
- Main website: `https://asher-1.github.io/ACloudViewer/`
- Documentation: `https://asher-1.github.io/ACloudViewer/documentation/`
- Version archive: `https://asher-1.github.io/ACloudViewer/documentation/{version}/`

### Local CI Testing

Test the same Docker build locally:
```bash
# Build Docker image (same as CI)
docker build \
  --build-arg DEVELOPER_BUILD=OFF \
  -t acloudviewer-ci:docs \
  -f docker/Dockerfile.docs \
  .

# Extract and verify
docker run -v $(pwd):/opt/mount --rm acloudviewer-ci:docs \
  bash -c "cp /root/ACloudViewer/acloudviewer-*-docs.tar.gz /opt/mount/"

tar -tzf acloudviewer-*-docs.tar.gz | head -20
```

## 📖 Documentation Components

### Python API (Sphinx + autodoc)

- **Source**: 41+ modules in `cloudViewer` package
- **Generation**: Sphinx `autodoc` introspects Python modules
- **Coverage**: Core, Geometry, IO, Pipelines, ML, Tensor, GUI

### C++ API (Doxygen)

- **Source**: Header files in `libs/cloudViewer/`
- **Generation**: Doxygen parses C++ comments
- **Output**: Class hierarchy, function reference

### Tutorials (nbsphinx)

- **Source**: Jupyter notebooks in `docs/jupyter/`
- **Generation**: nbsphinx converts `.ipynb` to HTML
- **Categories**: Geometry, Pipelines, Tensor, Visualization

### Guides (RST/Markdown)

- **Source**: `.rst` and `.md` files in `docs/source/`
- **Generation**: Sphinx/MyST parser
- **Content**: Getting started, build guides, API guides

## 📚 Quick Reference

| Task | Command |
|------|---------|
| **Build all docs** | `source util/ci_utils.sh && build_docs ON` |
| **Build Python API only** | `cd docs && python make_docs.py --sphinx` |
| **Build C++ API only** | `cd docs && python make_docs.py --doxygen` |
| **Preview docs** | `cd docs/_out/html && python3 -m http.server 8080` |
| **Live reload** | `cd docs && make livehtml` |
| **Clean build** | `cd docs && python make_docs.py --clean --sphinx --doxygen` |
| **Docker build** | `docker build -t acloudviewer-ci:docs -f docker/Dockerfile.docs .` |
| **Release build** | `python make_docs.py --sphinx --doxygen --is_release` |

## 🔗 Related Documentation

- **Version Management**: [VERSION_MANAGEMENT.md](../guides/VERSION_MANAGEMENT.md)
- **Deployment Guide**: [DEPLOYMENT.md](DEPLOYMENT.md)
- **Automation Scripts**: [scripts/README.md](scripts/README.md)
- **Automation Overview**: [README.md](README.md)

---

**Last Updated**: February 2026  
**Maintainer**: ACloudViewer Team  
**Status**: ✅ Production Ready
