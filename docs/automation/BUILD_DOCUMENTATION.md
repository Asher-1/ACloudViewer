# Building ACloudViewer Documentation

This guide explains how to build comprehensive API documentation for ACloudViewer using the **Open3D-style** documentation system.

## 📋 Overview

ACloudViewer documentation system includes:

* **Doxygen**: Generates standalone C++ API HTML documentation
* **Sphinx**: Generates Python API and tutorials HTML documentation  
* **make_docs.py**: Orchestrates the entire build process
* **autodoc**: Automatically documents Python modules
* **nbsphinx**: Integrates Jupyter notebooks

## 🏗️ Documentation Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Documentation Build                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Doxygen (Independent C++ API)                          │
│     ├─> Parse C++ source code                             │
│     ├─> Generate standalone HTML                          │
│     └─> Output: docs/doxygen/html/                        │
│                                                             │
│  2. Sphinx (Python API + Tutorials)                        │
│     ├─> Generate Python API .rst files (autodoc)          │
│     ├─> Copy Jupyter notebooks                            │
│     ├─> Build HTML documentation                          │
│     └─> Output: docs/_out/html/                           │
│                                                             │
│  3. Integration                                             │
│     ├─> Copy Doxygen HTML → docs/_out/html/cpp_api/      │
│     └─> Unified documentation accessible via links        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Key Design Principles:**
- **Doxygen runs independently** - generates complete C++ API HTML
- **Sphinx handles Python** - uses autodoc for Python API, nbsphinx for tutorials
- **Simple file system integration** - Doxygen HTML copied to Sphinx output directory
- **No forced coupling** - Each tool runs in its domain, linked via filesystem

Reference: [Open3D Documentation System](https://github.com/isl-org/Open3D/tree/main/docs)

## 🎯 Documentation Structure

```
docs/
├── make_docs.py               # Main orchestration script
├── Doxyfile                   # Doxygen configuration
├── Doxyfile.in                # Doxygen template (for CMake)
├── CMakeLists.txt             # CMake documentation targets
├── requirements.txt           # Python dependencies
│
├── source/                    # Sphinx source files
│   ├── conf.py               # Sphinx configuration
│   ├── index.rst             # Main documentation page
│   ├── getting_started/      # Getting started guides
│   ├── tutorial/             # Jupyter notebook tutorials
│   │   ├── geometry/         # Geometry tutorials
│   │   ├── visualization/    # Visualization tutorials
│   │   └── ...
│   ├── python_api/           # Python API (auto-generated)
│   ├── cpp_api/              # C++ API (links to Doxygen)
│   └── _static/              # Static assets
│
├── doxygen/                  # Doxygen output (generated, temporary)
│   ├── html/                 # Standalone C++ API HTML
│   └── xml/                  # XML (optional, for Breathe)
│
└── _out/                     # Final documentation (generated)
    └── html/                 # Unified HTML output
        ├── index.html        # Main entry point
        ├── python_api/       # Python API docs
        ├── tutorial/         # Tutorial docs
        └── cpp_api/          # C++ API docs (copied from Doxygen)
```

## 🔧 Prerequisites

### System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y \
    doxygen \
    graphviz \
    python3-pip \
    python3-dev \
    pandoc
```

**macOS:**
```bash
brew install doxygen graphviz pandoc
```

**Windows:**
- Install Doxygen from https://www.doxygen.nl/download.html
- Install Graphviz from https://graphviz.org/download/
- Add both to PATH

### Python Dependencies

```bash
# Install documentation dependencies
pip install -r docs/requirements.txt
```

## 🚀 Building Documentation

### Method 1: Using make_docs.py (Recommended)

This is the **Open3D-style** orchestration script that handles the entire build process.

```bash
cd docs

# Build both Sphinx and Doxygen documentation
python make_docs.py --sphinx --doxygen

# Build only Sphinx (Python API + tutorials)
python make_docs.py --sphinx

# Build only Doxygen (C++ API)
python make_docs.py --doxygen

# Release mode (cleaner output)
python make_docs.py --sphinx --doxygen --is_release

# View documentation
cd _out/html
python3 -m http.server 8000
# Open http://localhost:8000
```

**What `make_docs.py` does:**
1. Generates Python API `.rst` files using autodoc
2. Copies Jupyter notebooks to source directory
3. Runs Doxygen to generate C++ API HTML
4. Runs Sphinx to build main documentation
5. Copies Doxygen HTML to `_out/html/cpp_api/`
6. Creates unified documentation output

### Method 2: Using CMake Targets

```bash
# From build directory
cd build

# Build C++ API documentation only
make doxygen

# Build Python API + tutorials only
make sphinx-html

# View documentation
python3 -m http.server --directory docs/_out/html 8080
```

### Method 3: Using ci_utils.sh (CI/CD)

```bash
# Source the utilities
source util/ci_utils.sh

# Build documentation (includes Python module build if needed)
build_docs ON  # Developer mode
# or
build_docs OFF # Release mode
```

**What `build_docs` does:**
1. Checks if Python module exists (skips build if found)
2. Builds Python module if needed (minimal dependencies)
3. Calls `make_docs.py --sphinx --doxygen`
4. Displays build statistics and preview commands

### Method 4: Docker Build (Isolated Environment)

```bash
# Build documentation Docker image
docker build \
  --network=host \
  -t acloudviewer-ci:docs \
  -f docker/Dockerfile.docs \
  .

# Extract documentation
docker run \
  -v "${PWD}:/opt/mount" \
  --rm acloudviewer-ci:docs \
  bash -c "cp /root/ACloudViewer/acloudviewer-*-docs.tar.gz /opt/mount/"

# Extract and preview
mkdir -p docs-output
tar -xzf acloudviewer-*-docs.tar.gz -C docs-output/
cd docs-output && python3 -m http.server 8080
```

## 📝 Documentation Configuration

### Sphinx Configuration (`docs/source/conf.py`)

Key settings for Open3D-style documentation:

```python
# Project information
project = 'ACloudViewer'
release = '3.9.3'

# Theme (modern, clean)
html_theme = 'furo'

# Extensions
extensions = [
    'sphinx.ext.autodoc',      # Python API docs
    'sphinx.ext.autosummary',  # API summary tables
    'sphinx.ext.napoleon',     # Google/NumPy docstrings
    'sphinx.ext.viewcode',     # Source code links
    'nbsphinx',               # Jupyter notebooks
    'myst_parser',            # Markdown support
    'sphinx_copybutton',      # Copy code buttons
]

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}

# nbsphinx settings
nbsphinx_execute = 'never'  # Don't execute notebooks during build
nbsphinx_allow_errors = True
nbsphinx_codecell_lexer = 'python3'  # Fix ipython3 lexer warnings

# Register ipython3 lexer alias (fix Pygments warnings)
from pygments.lexers import Python3Lexer
from sphinx.highlighting import lexers
lexers['ipython3'] = Python3Lexer()

# Suppress known harmless warnings
suppress_warnings = [
    'nbsphinx',
    'misc.highlighting_failure',
    'ref.duplicate',
    'toc.not_readable',
]
```

### Doxygen Configuration (`docs/Doxyfile`)

Key settings for C++ API documentation:

```ini
# Project information
PROJECT_NAME = "ACloudViewer"
PROJECT_BRIEF = "3D Point Cloud Processing Library"

# Input source code
INPUT = ../libs/CloudViewer \
        ../libs/PCLEngine \
        ../libs/Python/pybind \
        ../plugins/core

# Output formats
GENERATE_HTML = YES
GENERATE_XML = YES  # Optional, for Breathe integration
GENERATE_LATEX = NO

# HTML output
HTML_OUTPUT = html
HTML_COLORSTYLE_HUE = 220
HTML_COLORSTYLE_SAT = 100
HTML_COLORSTYLE_GAMMA = 80

# Generate diagrams
HAVE_DOT = YES
DOT_IMAGE_FORMAT = svg
CALL_GRAPH = YES
CALLER_GRAPH = YES

# Exclude patterns (avoid build artifacts)
EXCLUDE_PATTERNS = */test/* \
                   */tests/* \
                   */build/* \
                   */build_*/* \
                   */3rdparty/* \
                   *_autogen/* \
                   moc_*.cpp \
                   ui_*.h
```

## 🐛 Troubleshooting

### Problem: Doxygen not found

**Solution:**
```bash
# Check Doxygen installation
doxygen --version

# If not installed:
# Ubuntu: sudo apt-get install doxygen
# macOS: brew install doxygen
# Windows: Download from https://www.doxygen.nl/
```

### Problem: Sphinx build warnings

**Check:**
```bash
# Build with verbose output
cd docs
sphinx-build -W -b html source _out/html -v
```

**Common issues:**
- Missing Python packages: `pip install -r requirements.txt`
- RST syntax errors: Check `.rst` files for formatting issues
- Missing references: Ensure all `:doc:` paths are correct

### Problem: Pygments lexer warnings

**Symptom:** `WARNING: Pygments lexer name 'ipython3' is not known`

**Solution:** This is fixed in `conf.py` by registering the lexer alias:
```python
from pygments.lexers import Python3Lexer
from sphinx.highlighting import lexers
lexers['ipython3'] = Python3Lexer()
```

### Problem: Python module not found

**Solution:**
```bash
# Build Python module first
cd build
make python-package -j$(nproc)

# Or use ci_utils.sh which handles this automatically
source ../util/ci_utils.sh
build_docs ON
```

### Problem: Notebooks not rendering

**Check:**
```bash
# Ensure nbsphinx is installed
pip install nbsphinx ipython

# Check notebook format
jupyter nbconvert --to notebook --execute your_notebook.ipynb
```

## 📚 Best Practices

### 1. Documentation Workflow

```bash
# 1. Update source code with docstrings
# 2. Update Jupyter notebooks if needed
# 3. Build documentation locally
cd docs
python make_docs.py --sphinx --doxygen

# 4. Check for warnings
grep -i warning build.log

# 5. Preview locally
python3 -m http.server --directory _out/html 8080

# 6. Commit changes
git add source/
git commit -m "docs: update API documentation"
```

### 2. Writing Good Documentation

**C++ (Doxygen style):**
```cpp
/// \brief Voxel downsampling.
///
/// Creates a uniformly downsampled point cloud by averaging all points
/// within a voxel.
///
/// \param voxel_size Size of the voxel in meters.
/// \return Downsampled point cloud.
std::shared_ptr<PointCloud> VoxelDownSample(double voxel_size) const;
```

**Python (Google style):**
```python
def voxel_down_sample(self, voxel_size: float) -> PointCloud:
    """Voxel downsampling.
    
    Creates a uniformly downsampled point cloud by averaging all points
    within a voxel.
    
    Args:
        voxel_size (float): Size of the voxel in meters.
    
    Returns:
        PointCloud: Downsampled point cloud.
    
    Example:
        >>> pcd_down = pcd.voxel_down_sample(voxel_size=0.05)
    """
    pass
```

### 3. Jupyter Notebooks

All notebooks must have a **Markdown title** in the first cell:

```markdown
# Your Tutorial Title

Brief description of what this tutorial covers.
```

**Why?** nbsphinx requires the first cell to be a Markdown title to generate proper navigation.

## 🔗 References

- [Open3D Documentation](https://github.com/isl-org/Open3D/tree/main/docs)
- [Open3D make_docs.py](https://github.com/isl-org/Open3D/blob/main/docs/make_docs.py)
- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [Doxygen Manual](https://www.doxygen.nl/manual/)
- [nbsphinx Documentation](https://nbsphinx.readthedocs.io/)
- [Furo Theme](https://pradyunsg.me/furo/)

## ✅ Quick Reference

| Task | Command |
|------|---------|
| Build all docs | `python make_docs.py --sphinx --doxygen` |
| Build Sphinx only | `python make_docs.py --sphinx` |
| Build Doxygen only | `python make_docs.py --doxygen` |
| Preview docs | `python3 -m http.server --directory _out/html 8080` |
| Build with CMake | `make doxygen && make sphinx-html` |
| Build with CI script | `source util/ci_utils.sh && build_docs ON` |
| Docker build | `docker build -t acloudviewer-ci:docs -f docker/Dockerfile.docs .` |

---

**Last Updated**: 2026-01-13  
**Maintained by**: ACloudViewer Team  
**Documentation Style**: Open3D-inspired
