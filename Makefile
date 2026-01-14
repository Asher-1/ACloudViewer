# Makefile for ACloudViewer documentation
# This is a convenience wrapper around make_docs.py (based on Open3D's approach)
#
# Usage:
#   make html        - Build Sphinx documentation only
#   make doxygen     - Build Doxygen (C++ API) documentation only
#   make docs        - Build both Sphinx and Doxygen documentation
#   make clean       - Clean build artifacts
#   make livehtml    - Live rebuild with auto-refresh (requires sphinx-autobuild)

PYTHON       ?= python3
MAKE_DOCS    = $(PYTHON) make_docs.py
BUILDDIR     = _out
SOURCEDIR    = source

.PHONY: help clean html doxygen docs livehtml

# Default target
help:
	@echo "ACloudViewer Documentation Build System"
	@echo "========================================"
	@echo ""
	@echo "Available targets:"
	@echo "  make docs      - Build both Sphinx and Doxygen documentation (recommended)"
	@echo "  make html      - Build Sphinx documentation only"
	@echo "  make doxygen   - Build Doxygen (C++ API) documentation only"
	@echo "  make clean     - Clean build artifacts"
	@echo "  make livehtml  - Live rebuild with auto-refresh browser"
	@echo ""
	@echo "For advanced options, use make_docs.py directly:"
	@echo "  $(PYTHON) make_docs.py --help"

# Build both Sphinx and Doxygen documentation
docs:
	$(MAKE_DOCS) --sphinx --doxygen

# Build Sphinx documentation only
html:
	$(MAKE_DOCS) --sphinx

# Build Doxygen documentation only
doxygen:
	$(MAKE_DOCS) --doxygen

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	@rm -rf $(BUILDDIR) html doxygen
	@echo "Clean complete."

# Live rebuild with auto-refresh (for development)
livehtml:
	@echo "Starting live documentation server..."
	@echo "Documentation will auto-rebuild on file changes"
	@echo "Open http://localhost:8000 in your browser"
	@sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)/html" \
		--open-browser \
		--port 8000 \
		--watch .

# Alias targets for compatibility
.PHONY: sphinx
sphinx: html

