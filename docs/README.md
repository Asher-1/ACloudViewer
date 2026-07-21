# ACloudViewer Documentation

> **Complete documentation system for ACloudViewer**, including website, tutorials, Python API, and C++ API.

> 🌐 **Live Site**: https://asher-1.github.io/ACloudViewer/

## 📁 Directory Structure

```
docs/
├── index.html              # 🌐 Main website homepage
├── styles.css              # Website styles
├── script.js               # Website scripts
├── .nojekyll              # GitHub Pages configuration
├── 404.html               # Custom 404 page
├── robots.txt             # Search engine configuration
├── sitemap.xml            # Site map for SEO
│
├── images/                # 🖼️  Website images
│   ├── ACloudViewer_logo_horizontal.png
│   ├── Annotation.png
│   └── ...
│
├── gifs/                  # 🎬 Animated assets
│   ├── visualizer_predictions.gif
│   └── ...
│
├── automation/            # 🤖 Automation system
│   ├── README.md          # Automation guide
│   ├── SUMMARY.md         # System summary
│   └── scripts/           # Update scripts
│       ├── update_download_links.py
│       └── requirements.txt
│
├── guides/                # 📚 User guides
│   ├── QUICKSTART.md      # Quick start guide
│   ├── cloudviewer-dependency.md
│   └── building/          # Platform-specific build guides
│       ├── compiling-cloudviewer-linux.md
│       ├── compiling-cloudviewer-macos.md
│       └── compiling-cloudviewer-windows.md
│
├── maintenance/           # 🔧 Maintenance documentation
│   ├── WEBSITE_GUIDE.md   # Website maintenance
│   ├── DEPLOYMENT.md      # Deployment guide
│   └── ...
│
├── scripts/               # 🧪 Testing scripts
│   ├── test_doc_structure.sh       # Validate structure
│   ├── test_github_pages_locally.sh # Test deployment
│   └── README.md                    # Script documentation
│
├── Makefile              # 🔨 Main build orchestration
├── make_docs.py          # 📝 Python build script
├── requirements.txt      # 🐍 Python dependencies
│
├── Doxyfile.in           # ⚙️  Doxygen configuration (C++ API)
├── Doxyfile.modules      # Module definitions
│
├── source/               # 📄 Sphinx documentation source
│   ├── conf.py           # Sphinx configuration
│   ├── index.rst         # Documentation entry point
│   ├── tutorial/         # Tutorial notebooks
│   ├── python_api/       # Python API docs
│   └── cpp_api/          # C++ API docs
│
├── jupyter/              # 📓 Original Jupyter notebooks
│   └── ...
│
└── _out/                 # 📦 Generated documentation
    └── html/             # Final HTML output
```

## 🚀 Quick Start

### 📖 Build Documentation Locally

```bash
# Navigate to docs directory
cd docs

# Build all documentation (Python API, C++ API, tutorials)
make docs

# Preview locally
python3 -m http.server 8000 --directory _out/html
```

Then visit http://localhost:8000

### 🌐 Preview Main Website Only

```bash
cd docs
python3 -m http.server 8080
```

Then visit http://localhost:8080

## 📚 Documentation System

ACloudViewer uses a **comprehensive documentation system** combining:

### 1. Main Website (`index.html`)
- Project overview and introduction
- Download links (automated)
- Feature highlights
- Gallery and examples

### 2. Python API Documentation (Sphinx + autodoc)
- 47+ modules documented
- Auto-generated from Python bindings
- Interactive examples
- Search functionality

### 3. C++ API Documentation (Doxygen)
- Complete C++ API reference
- Class hierarchies
- Code examples
- Module organization

### 4. Tutorials (Jupyter Notebooks)
- 30+ interactive tutorials
- Visualization examples
- Point cloud processing
- 3D reconstruction

## 🔨 Building Documentation

### Prerequisites

```bash
# System dependencies (Ubuntu/Debian)
sudo apt-get install doxygen graphviz pandoc

# System dependencies (macOS)
brew install doxygen graphviz pandoc

# Python dependencies
pip install -r requirements.txt
```

### Build Commands

```bash
# 1. Full documentation build (recommended)
cd docs
make docs

# 2. Build specific components
make clean          # Clean generated files
make doxygen        # C++ API only
make sphinx         # Python API + tutorials only

# 3. Using CI/CD build function
source ../util/ci_utils.sh
build_docs OFF      # Release mode
# or
build_docs ON       # Developer mode
```

### Build Options

- **Developer mode** (`DEVELOPER_BUILD=ON`):
  - Faster builds
  - Skip some checks
  - Use for local testing

- **Release mode** (`DEVELOPER_BUILD=OFF`):
  - Complete build
  - All checks enabled
  - Use for production

## 🐳 Docker Build

Build documentation in a clean Docker environment:

```bash
# Build Docker image with documentation (runs in background)
nohup docker build --network=host --build-arg BASE_IMAGE=ubuntu:22.04 --build-arg DEVELOPER_BUILD=OFF -t acloudviewer-ci:docs -f docker/Dockerfile.docs . > docker_docs_build.log 2>&1 &

# Check build progress
tail -f docker_docs_build.log

# Or check if build is still running
ps aux | grep "docker build"

# Extract documentation
docker run -v $(pwd):/opt/mount --rm acloudviewer-ci:docs bash -c "cp /root/ACloudViewer/acloudviewer-*-docs.tar.gz /opt/mount/ && chown $(id -u):$(id -g) /opt/mount/acloudviewer-*-docs.tar.gz"

# Extract and preview
mkdir -p docs-output
tar -xzf acloudviewer-*-docs.tar.gz -C ./docs-output/
cd docs-output && python3 -m http.server 8080
```

## 🧪 Testing

### Test Documentation Structure

```bash
./scripts/test_doc_structure.sh
```

Validates:
- Required files exist
- Directory structure is correct
- Configuration files are valid

### Test GitHub Pages Deployment

```bash
# Full Docker test (recommended)
./scripts/test_github_pages_locally.sh docker

# Use existing build
./scripts/test_github_pages_locally.sh local

# Quick preview
./scripts/test_github_pages_locally.sh simple
```

Tests:
- Main website → `/`
- API documentation → `/documentation/`
- No file conflicts
- Correct navigation

## 🤖 Automation

### Automated Download Links

The website automatically updates download links when new releases are published:

```bash
# Manual trigger (if needed)
python3 automation/scripts/update_download_links.py
```

### CI/CD Documentation Build

Documentation is automatically built and deployed via GitHub Actions:

- **Trigger**: Push to `main` branch
- **Workflow**: `.github/workflows/documentation.yml`
- **Deploy to**:
  - Main website → `https://asher-1.github.io/ACloudViewer/`
  - API docs → `https://asher-1.github.io/ACloudViewer/documentation/`

## 📖 Documentation Sections

### For Users

- [Quick Start Guide](guides/QUICKSTART.md)
- [Build from Source](guides/building/)
- [Dependencies](guides/cloudviewer-dependency.md)
- **AI Plugins (AICore)**
  - [Overview](guides/plugins/README.md)
  - [qDA3 — Depth Anything V3](guides/plugins/qDA3.md)
  - [qFreeSplatter — 3D Gaussian Splatting](guides/plugins/qFreeSplatter.md)
- **Python API** → `/documentation/python_api/index.html`
- **C++ API** → `/documentation/cpp_api/index.html`
- **Tutorials** → `/documentation/tutorial/index.html`

### For Developers

- [Automation System](automation/README.md)
- [Building Documentation](automation/BUILDING_DOCS.md)
- [Deployment Guide](automation/DEPLOYMENT.md)
- [Version Management](guides/VERSION_MANAGEMENT.md)
- [Agent Integration](../agent-integration/README.md) — AI agent control via JSON-RPC, MCP, CLI
- **Build Functions** → `util/ci_utils.sh`

### For Maintainers

- [Website Maintenance](maintenance/WEBSITE_GUIDE.md)
- [Deployment Guide](maintenance/DEPLOYMENT.md)
- [Download Link Management](maintenance/DOWNLOAD_LINKS.md)

## 🔧 Configuration Files

### Sphinx Configuration (`source/conf.py`)

Key settings:
- `add_module_names = False` - Short names in navigation
- `nbsphinx_codecell_lexer = 'python3'` - Jupyter syntax highlighting
- Python path configuration for autodoc
- Theme configuration (Furo)

### Doxygen Configuration (`Doxyfile.in`)

Key settings:
- `HAVE_DOT = NO` - No Graphviz diagrams (following Open3D)
- Modular input paths
- XML output for Breathe integration
- C++ standard: C++17

### Makefile

Orchestrates the entire build process:
```makefile
docs: doxygen sphinx  # Build everything
doxygen:              # Build C++ API
sphinx:               # Build Python API + tutorials
clean:                # Clean generated files
```

## 📊 Statistics

Current documentation includes:
- **Python modules**: 47 (cloudViewer.core, geometry, io, pipelines, etc.)
- **C++ modules**: 11 (core, 3rdparty, libs, plugins, app)
- **Tutorials**: 30+ interactive Jupyter notebooks
- **Pages**: 500+ HTML pages
- **Build time**: ~30-40 minutes (full build)

## 🔄 Workflow

```
Development:
  Edit source → cd docs && make docs → Preview locally

CI/CD:
  Push to main → GitHub Actions → Deploy to GitHub Pages

Docker:
  docker build → Extract tarball → Deploy
```

## 📝 Contributing

### Updating Website Content

1. Edit `index.html`, `styles.css`, or `script.js`
2. Test locally: `python3 -m http.server 8080`
3. Commit and push

### Adding Tutorials

1. Create Jupyter notebook in `jupyter/`
2. Add to `source/tutorial/` index
3. Rebuild: `make docs`

### Updating API Docs

API documentation is **auto-generated** from code:
- Python API: From Python bindings via Sphinx autodoc
- C++ API: From C++ headers via Doxygen

To update: modify the source code and rebuild.

## 🤖 AI Agent Integration

ACloudViewer provides an agent integration module for AI-powered automation:

- **JSON-RPC Plugin** — 32 RPC methods over WebSocket for real-time GUI control
- **MCP Server** — 23 tools for OpenClaw, Cursor IDE, and Claude Code
- **CLI Harness** — 15 processing commands for headless batch processing

Documentation:
- [Agent Integration README](../agent-integration/README.md)
- [MCP Server Reference](../agent-integration/mcp/README.md)
- [CLI Command Reference](../agent-integration/cli/README.md)
- [JSON-RPC API Reference](../agent-integration/docs/JSON-RPC-API.md)
- [Testing Guide](../agent-integration/docs/TESTING.md)
- [CLI-Anything Hub](https://asher-1.github.io/CLI-Anything/) — browse and install agent tools

## 🔗 Related Links

- **Main Repository**: https://github.com/Asher-1/ACloudViewer
- **Releases**: https://github.com/Asher-1/ACloudViewer/releases
- **Issues**: https://github.com/Asher-1/ACloudViewer/issues
- **GitHub Actions**: https://github.com/Asher-1/ACloudViewer/actions
- **CLI-Anything Hub**: https://asher-1.github.io/CLI-Anything/

## 📄 License

This documentation follows the ACloudViewer project license.

---

**Maintained by**: ACloudViewer Team  
**Last Updated**: 2026-01-13  
**Build System**: Sphinx + Doxygen + Makefile  
**Automation**: ✅ Fully Automated
