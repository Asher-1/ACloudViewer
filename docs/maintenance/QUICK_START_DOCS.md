# Quick Start: Building ACloudViewer Documentation

## 🚀 TL;DR

```bash
# Build documentation locally (recommended)
source util/ci_utils.sh
build_docs ON

# Preview
cd docs/_out/html && python3 -m http.server 8080
# Open http://localhost:8080
```

---

## 📋 Prerequisites

### System Requirements
- Ubuntu 22.04 (or compatible)
- Python 3.8+
- CMake 3.18+
- Doxygen
- Pandoc

### Install Dependencies

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y \
    build-essential cmake git \
    doxygen pandoc \
    python3-dev python3-pip

# Install Python dependencies
pip install -r docs/requirements.txt
```

---

## 🛠️ Build Methods

### Method 1: Using ci_utils.sh (Recommended)

**Advantages**: Same logic as CI/CD, automatic Python module detection

```bash
cd /path/to/ACloudViewer

# Source the utilities
source util/ci_utils.sh

# Build documentation
# ON = development build (faster)
# OFF = release build (uses version numbers)
build_docs ON

# Output: docs/_out/html/
```

**What it does**:
1. Checks if Python module exists
2. Builds Python module if needed (minimal config)
3. Generates Python API docs (41 modules)
4. Generates C++ API docs (Doxygen)
5. Copies Jupyter notebooks (34 notebooks)
6. Builds HTML with Sphinx
7. Verifies output and shows statistics

### Method 2: Using make_docs.py Directly

**Advantages**: More control over build options

```bash
cd docs

# Build both Sphinx and Doxygen
python make_docs.py --sphinx --doxygen

# Build only Python API docs
python make_docs.py --sphinx

# Build only C++ API docs
python make_docs.py --doxygen

# Release build (use version numbers)
python make_docs.py --sphinx --doxygen --is_release

# Clean and rebuild
python make_docs.py --clean --sphinx --doxygen

# Output: _out/html/
```

### Method 3: Using Makefile

**Advantages**: Simple commands

```bash
cd docs

# Build all documentation
make docs

# Build only Sphinx (Python API + tutorials)
make html

# Build only Doxygen (C++ API)
make doxygen

# Clean build artifacts
make clean

# Live rebuild with auto-refresh (requires sphinx-autobuild)
make livehtml
```

---

## 🐳 Docker Build

### Build Docker Image

```bash
# Development build
docker build \
  --build-arg DEVELOPER_BUILD=ON \
  -t acloudviewer-ci:docs \
  -f docker/Dockerfile.docs .

# Release build
docker build \
  --build-arg DEVELOPER_BUILD=OFF \
  -t acloudviewer-ci:docs \
  -f docker/Dockerfile.docs .
```

### Extract Documentation

```bash
# Extract documentation archive
docker run \
  -v $(pwd):/opt/mount \
  --rm acloudviewer-ci:docs \
  bash -c "cp /root/ACloudViewer/acloudviewer-*-docs.tar.gz /opt/mount/"

# Unpack
mkdir -p docs-preview
tar -xzf acloudviewer-*-docs.tar.gz -C ./docs-preview/

# Preview
cd docs-preview && python3 -m http.server 8080
```

---

## 📚 Output Structure

```
docs/_out/html/
├── index.html                          # Main documentation page
├── python_api/                         # Python API documentation
│   ├── cloudViewer.camera.html
│   ├── cloudViewer.core.html
│   ├── cloudViewer.geometry.html
│   ├── cloudViewer.geometry.PointCloud.html
│   ├── cloudViewer.io.html
│   └── ... (41 modules, 100+ classes, 200+ functions)
├── cpp_api/                            # C++ API documentation (Doxygen)
│   ├── index.html
│   ├── annotated.html
│   ├── classes.html
│   └── ...
└── tutorial/                           # Jupyter notebook tutorials
    ├── geometry/
    │   ├── pointcloud.html
    │   ├── mesh.html
    │   └── ...
    ├── pipelines/
    │   ├── icp_registration.html
    │   └── ...
    └── visualization/
        └── ...
```

---

## 🔍 Verification

### Check Build Success

```bash
# Check if output exists
ls -lh docs/_out/html/index.html

# Count generated files
echo "Total files: $(find docs/_out/html -type f | wc -l)"
echo "HTML files: $(find docs/_out/html -name '*.html' | wc -l)"

# Check size
du -sh docs/_out/html/
```

### Test Python Module Import

```bash
# Test if cloudViewer module can be imported
cd build_app/lib/Release/Python/cuda
python3 -c "import pybind as cloudViewer; print(dir(cloudViewer))"
```

### Run Verification Script

```bash
# Run comprehensive verification
bash docs/test_doc_generation.sh
```

---

## 🌐 Preview Documentation

### Local HTTP Server

```bash
# Python 3
cd docs/_out/html
python3 -m http.server 8080

# Open browser
# http://localhost:8080
```

### Live Reload (Development)

```bash
# Install sphinx-autobuild
pip install sphinx-autobuild

# Start live server
cd docs
make livehtml

# Opens browser automatically at http://localhost:8000
# Documentation auto-rebuilds on file changes
```

---

## 🐛 Troubleshooting

### Issue 1: Python Module Not Found

**Error**: `ImportError: No module named 'cloudViewer'`

**Solution**:
```bash
# Build Python module first
cd build_app
cmake .. -DBUILD_PYTHON_MODULE=ON
make cloudViewer_pybind -j$(nproc)

# Verify
cd lib/Release/Python/cuda
python3 -c "import pybind; print(dir(pybind))"
```

### Issue 2: Doxygen Not Installed

**Error**: `doxygen: command not found`

**Solution**:
```bash
sudo apt-get install doxygen
```

### Issue 3: Pandoc Not Installed

**Error**: `nbsphinx requires pandoc`

**Solution**:
```bash
sudo apt-get install pandoc
pip install pypandoc
```

### Issue 4: Sphinx Build Fails

**Error**: `sphinx-build: command not found`

**Solution**:
```bash
pip install -r docs/requirements.txt
```

### Issue 5: Missing Dependencies

**Error**: Various import errors

**Solution**:
```bash
# Install all documentation dependencies
source util/ci_utils.sh
install_docs_dependencies "${CLOUDVIEWER_ML_ROOT}"
```

---

## 📖 Documentation Components

### Python API Documentation

**Source**: 41 modules defined in `docs/documented_modules.txt`

**Generation Process**:
1. Import `cloudViewer` module
2. Introspect classes and functions
3. Generate `.rst` files with Sphinx autodoc directives
4. Build HTML with Sphinx

**Coverage**:
- Core modules (camera, core, geometry, io)
- ML modules (ml, ml.contrib)
- Pipeline modules (registration, odometry, integration)
- Reconstruction modules (ACloudViewer-specific)
- Tensor modules (t.geometry, t.pipelines)
- Visualization modules (gui, rendering, app)

### C++ API Documentation

**Source**: C++ header files in `libs/cloudViewer/`

**Generation Process**:
1. Doxygen reads `docs/Doxyfile`
2. Parses C++ comments (`///` or `/** */`)
3. Generates HTML documentation
4. Copies to `_out/html/cpp_api/`

### Jupyter Notebook Tutorials

**Source**: 34 notebooks in `docs/jupyter/`

**Generation Process**:
1. Copy notebooks to `source/tutorial/`
2. nbsphinx converts notebooks to HTML
3. Integrated into Sphinx documentation

**Categories**:
- Core (2): tensor, hashmap
- Geometry (15): mesh, pointcloud, kdtree, etc.
- Pipelines (8): registration, odometry, etc.
- Tensor (2): t_geometry, t_pipelines
- Visualization (4): GUI, plotly, 3D Gaussian splatting

---

## 🚀 CI/CD Integration

### GitHub Actions Workflow

**File**: `.github/workflows/documentation.yml`

**Trigger**:
- Push to `main` branch
- Pull requests
- Manual dispatch

**Steps**:
1. Build Docker image (`Dockerfile.docs`)
2. Extract documentation (`.tar.gz`)
3. Upload artifact
4. Deploy to GitHub Pages (main branch only)

**Deployment URLs**:
- Main Website: `https://asher-1.github.io/ACloudViewer/`
- API Documentation: `https://asher-1.github.io/ACloudViewer/documentation/`
- Python API: `https://asher-1.github.io/ACloudViewer/documentation/python_api/`
- C++ API: `https://asher-1.github.io/ACloudViewer/documentation/cpp_api/`

---

## 📝 Tips and Best Practices

### 1. Incremental Builds

For faster iteration during development:

```bash
# Build only Sphinx (skip Doxygen)
cd docs
python make_docs.py --sphinx

# Use live reload
make livehtml
```

### 2. Python Module Reuse

The `build_docs` function automatically detects existing Python modules:

```bash
# First build (compiles module)
source util/ci_utils.sh
build_docs ON

# Subsequent builds (reuses module, much faster)
build_docs ON
```

### 3. Clean Builds

If you encounter issues:

```bash
# Clean all build artifacts
cd docs
make clean

# Rebuild from scratch
python make_docs.py --clean --sphinx --doxygen
```

### 4. Parallel Builds

Speed up Sphinx builds:

```bash
# Use all CPU cores
python make_docs.py --sphinx --parallel
```

### 5. Documentation Quality

- Use Google-style docstrings in Python code
- Use Doxygen comments in C++ code (`///` or `/** */`)
- Include examples in docstrings
- Add type hints to Python functions

---

## 📚 Additional Resources

- **Detailed Analysis**: `docs/DOC_GENERATION_ANALYSIS.md`
- **Verification Script**: `docs/test_doc_generation.sh`
- **Sphinx Documentation**: https://www.sphinx-doc.org/
- **Doxygen Documentation**: https://www.doxygen.nl/
- **nbsphinx Documentation**: https://nbsphinx.readthedocs.io/

---

## 💡 Quick Reference

| Task | Command |
|------|---------|
| Build all docs | `source util/ci_utils.sh && build_docs ON` |
| Build Python API only | `cd docs && python make_docs.py --sphinx` |
| Build C++ API only | `cd docs && python make_docs.py --doxygen` |
| Preview docs | `cd docs/_out/html && python3 -m http.server 8080` |
| Live reload | `cd docs && make livehtml` |
| Clean build | `cd docs && make clean` |
| Docker build | `docker build -t acloudviewer-ci:docs -f docker/Dockerfile.docs .` |
| Verify build | `bash docs/test_doc_generation.sh` |

---

**Last Updated**: 2026-01-14  
**Status**: ✅ Production Ready
