# ACloudViewer Documentation Build System

This documentation system is based on [Open3D's documentation build system](https://github.com/isl-org/Open3D), which provides a robust and scalable approach to building technical documentation.

## Overview

The documentation build system supports:
- **Sphinx**: Main documentation and Python API (HTML)
- **Doxygen**: C++ API documentation (HTML + XML)
- **reStructuredText & Markdown**: Source file formats
- **Jupyter Notebooks**: Interactive tutorials (planned)

## Quick Start

### Local Development

```bash
# Navigate to docs directory
cd docs

# Build all documentation (Sphinx + Doxygen)
python3 make_docs.py --sphinx --doxygen

# Or use the convenient Makefile
make docs

# View the documentation
python3 -m http.server 8080 --directory _out/html
# Open http://localhost:8080 in your browser
```

### Using Docker

```bash
# Build documentation Docker image
docker build -t acloudviewer-ci:docs -f docker/Dockerfile.docs .

# Extract documentation tarball
docker run -v "${PWD}:/opt/mount" --rm acloudviewer-ci:docs \
    bash -c "cp /root/ACloudViewer/acloudviewer-*-docs.tar.gz /opt/mount"

# Extract and view
tar -xzf acloudviewer-*-docs.tar.gz -C docs-output/
python3 -m http.server 8080 --directory docs-output/
```

## Build System Components

### 1. `make_docs.py` (Main Build Script)

Python-based build orchestrator that:
- Manages Sphinx and Doxygen builds
- Handles parallel building
- Provides clean error messages
- Supports release and development builds

```bash
# Usage examples
python3 make_docs.py --sphinx --doxygen           # Build everything
python3 make_docs.py --sphinx                     # Sphinx only
python3 make_docs.py --doxygen                    # Doxygen only
python3 make_docs.py --sphinx --doxygen --clean   # Clean rebuild
python3 make_docs.py --is_release --sphinx        # Release build
python3 make_docs.py --sphinx --parallel          # Parallel build
```

### 2. `Makefile` (Convenience Wrapper)

Simple Makefile that wraps `make_docs.py`:

```bash
make docs       # Build both Sphinx and Doxygen
make html       # Build Sphinx only
make doxygen    # Build Doxygen only
make clean      # Clean build artifacts
make livehtml   # Live rebuild with auto-refresh
```

### 3. `build_docs.sh` (Interactive Script)

User-friendly bash script with:
- Dependency checking
- Interactive prompts
- Color-coded output
- Environment validation

```bash
./build_docs.sh [DEVELOPER_BUILD] [CLEAN_BUILD]
# Examples:
./build_docs.sh ON NO     # Developer build, incremental
./build_docs.sh OFF YES   # Release build, clean
```

### 4. `Doxyfile` (C++ API Configuration)

Doxygen configuration for C++ API documentation:
- Extracts from `core/`, `libs/eCV_db/`, etc.
- Generates HTML and XML output
- Creates class diagrams with GraphViz
- Integrates with Sphinx via Breathe

### 5. Configuration Files

- `source/conf.py`: Sphinx configuration
- `requirements.txt`: Python dependencies
- `source/_static/`: Custom CSS and JavaScript
- `source/_templates/`: Custom HTML templates

## Directory Structure

```
docs/
├── source/              # Sphinx source files
│   ├── conf.py         # Sphinx configuration
│   ├── index.rst       # Main documentation index
│   ├── _static/        # CSS, JavaScript, images
│   └── _templates/     # Custom HTML templates
├── make_docs.py        # Main build script (like Open3D)
├── Makefile            # Convenience wrapper
├── build_docs.sh       # Interactive build script
├── Doxyfile            # Doxygen configuration
├── requirements.txt    # Python dependencies
└── _out/               # Build output (generated)
    └── html/           # Final HTML documentation
```

## CI/CD Integration

### GitHub Actions Workflow

The documentation is automatically built and deployed via `.github/workflows/documentation.yml`:

1. **Build**: Creates Docker image with all dependencies
2. **Generate**: Runs `make_docs.py` to build documentation
3. **Package**: Creates tarball of HTML output
4. **Upload**: Uploads to GitHub Releases (main-devel)
5. **Deploy**: Deploys to GitHub Pages (on main branch)

### Workflow Triggers

- Push to `main` branch
- Pull request events (opened, reopened, synchronize)
- Manual workflow dispatch

## Key Features (Based on Open3D)

### 1. **Python-First Approach**
- Uses Python script (`make_docs.py`) as the primary build tool
- Easier to extend and maintain than pure Makefile
- Better error handling and logging

### 2. **Modular Architecture**
- Separate builders for Sphinx and Doxygen
- Can build independently or together
- Easy to add new documentation types

### 3. **Docker-Based CI**
- Reproducible builds
- Isolated environment
- Consistent across all platforms

### 4. **Developer-Friendly**
- Multiple build methods (Python, Make, Bash)
- Clear error messages
- Fast incremental builds

## Comparison with Traditional Approach

| Aspect | Traditional Makefile | Open3D-Style (Current) |
|--------|---------------------|------------------------|
| **Primary Tool** | Makefile + sphinx-build | Python script + Sphinx |
| **Extensibility** | Limited (Make syntax) | High (Python) |
| **Error Handling** | Basic | Comprehensive |
| **Parallel Build** | Manual | Built-in |
| **Jupyter Support** | None | Native |
| **CI Integration** | Complex | Streamlined |

## Dependencies

### System Requirements
- Python 3.8+
- Doxygen 1.9+
- GraphViz (for diagrams)
- Git (for version detection)

### Python Packages
See `requirements.txt` for complete list:
- `sphinx>=7.1.2`: Documentation generator
- `furo`: Modern theme
- `breathe>=4.35.0`: C++ API bridge
- `myst-parser`: Markdown support
- `nbsphinx`: Jupyter notebook support

## Troubleshooting

### Common Issues

**1. "No module named 'sphinx'"**
```bash
pip3 install -r docs/requirements.txt
```

**2. "doxygen: command not found"**
```bash
# Ubuntu/Debian
sudo apt-get install doxygen graphviz

# macOS
brew install doxygen graphviz
```

**3. "Sphinx build failed"**
```bash
# Clean rebuild
cd docs
python3 make_docs.py --clean --sphinx --doxygen
```

**4. "Docker build fails"**
```bash
# Check Docker has enough resources (4GB+ RAM recommended)
# Clean Docker cache
docker system prune -a
```

## Contributing

When modifying the documentation system:

1. **Test locally first**:
   ```bash
   cd docs
   python3 make_docs.py --sphinx --doxygen
   ```

2. **Test with Docker**:
   ```bash
   docker build -t test-docs -f docker/Dockerfile.docs .
   ```

3. **Update this guide** if adding new features

4. **Follow Open3D patterns** for consistency

## References

- [Open3D Documentation](https://github.com/isl-org/Open3D/tree/main/docs)
- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [Doxygen Manual](https://www.doxygen.nl/manual/)
- [Breathe Documentation](https://breathe.readthedocs.io/)

## License

This documentation system follows the same license as ACloudViewer (MIT).
Based on Open3D's documentation system (MIT licensed).

