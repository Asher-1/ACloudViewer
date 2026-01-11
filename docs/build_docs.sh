#!/bin/bash
# Build ACloudViewer documentation
# Based on Open3D's documentation build system

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  ACloudViewer Documentation Build System${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo ""

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}❌ Error: requirements.txt not found${NC}"
    echo "Please run this script from the docs/ directory"
    exit 1
fi

# Parse arguments
DEVELOPER_BUILD=${1:-ON}
CLEAN_BUILD=${2:-NO}

echo -e "${GREEN}Configuration:${NC}"
echo "  • Developer Build: ${DEVELOPER_BUILD}"
echo "  • Clean Build: ${CLEAN_BUILD}"
echo ""

# Check for virtual environment
if [ -z "$CONDA_DEFAULT_ENV" ] && [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${YELLOW}⚠️  Warning: No Python virtual environment detected${NC}"
    echo "It's recommended to activate cloudViewer environment first:"
    echo "  conda activate cloudViewer"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[yY]$ ]]; then
        echo "Build cancelled."
        exit 1
    fi
else
    if [ -n "$CONDA_DEFAULT_ENV" ]; then
        echo -e "${GREEN}✅ Conda environment: $CONDA_DEFAULT_ENV${NC}"
    elif [ -n "$VIRTUAL_ENV" ]; then
        echo -e "${GREEN}✅ Virtual environment: $(basename $VIRTUAL_ENV)${NC}"
    fi
fi
echo ""

# Step 1: Check Python dependencies
echo -e "${BLUE}📦 Step 1/5: Checking Python dependencies${NC}"
if ! python3 -c "import sphinx" 2>/dev/null; then
    echo -e "${YELLOW}⚠️  Sphinx not found. Installing dependencies...${NC}"
    pip3 install -r requirements.txt
else
    echo -e "${GREEN}✅ Python dependencies OK${NC}"
fi
echo ""

# Step 2: Check Doxygen
echo -e "${BLUE}📦 Step 2/5: Checking Doxygen${NC}"
if ! command -v doxygen &> /dev/null; then
    echo -e "${RED}❌ Doxygen not found${NC}"
    echo "Please install Doxygen:"
    echo "  macOS: brew install doxygen graphviz"
    echo "  Ubuntu: sudo apt-get install doxygen graphviz"
    exit 1
else
    DOXYGEN_VERSION=$(doxygen --version)
    echo -e "${GREEN}✅ Doxygen ${DOXYGEN_VERSION}${NC}"
fi
echo ""

# Step 3: Check if Sphinx is initialized
echo -e "${BLUE}📦 Step 3/5: Checking Sphinx configuration${NC}"
if [ ! -f "source/conf.py" ]; then
    echo -e "${YELLOW}⚠️  Sphinx not initialized${NC}"
    echo "Running initial setup..."
    
    # Create source directory
    mkdir -p source source/_static source/_templates
    
    # Create basic conf.py
    cat > source/conf.py << 'EOF'
# Configuration file for the Sphinx documentation builder.
# Based on Open3D's documentation system

import os
import sys
from datetime import datetime

# -- Project information -----------------------------------------------------
project = 'ACloudViewer'
copyright = f'{datetime.now().year}, ACloudViewer Team'
author = 'ACloudViewer Team'
release = '3.9.3'
version = '3.9'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'breathe',  # C++ API documentation
    'exhale',   # Auto-generate C++ API docs
    'sphinx_copybutton',
    'sphinx_tabs.tabs',
    'myst_parser',  # Markdown support
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'furo'  # Modern, clean theme like Open3D
html_static_path = ['_static']
html_title = f"{project} {version} documentation"

# Furo theme options
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#0078d4",
        "color-brand-content": "#0078d4",
    },
}

# -- Options for Breathe (C++ API) -------------------------------------------
breathe_projects = {
    "ACloudViewer": "../xml/"
}
breathe_default_project = "ACloudViewer"

# -- Options for Exhale (auto C++ docs) --------------------------------------
exhale_args = {
    "containmentFolder": "./api",
    "rootFileName": "library_root.rst",
    "rootFileTitle": "C++ API",
    "doxygenStripFromPath": "..",
    "createTreeView": True,
    "exhaleExecutesDoxygen": False,  # We run doxygen separately
}

# -- Intersphinx configuration -----------------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}

# -- MyST Markdown configuration ---------------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "deflist",
]
EOF

    # Create basic index.rst
    cat > source/index.rst << 'EOF'
.. ACloudViewer documentation master file

Welcome to ACloudViewer's documentation!
=========================================

ACloudViewer is an advanced 3D data processing and visualization platform.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   introduction
   getting_started
   api/library_root

Introduction
============

ACloudViewer provides comprehensive tools for 3D point cloud processing, 
visualization, and analysis.

Getting Started
===============

Installation instructions and quick start guide.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
EOF

    # Create introduction page
    cat > source/introduction.rst << 'EOF'
Introduction
============

ACloudViewer is a powerful 3D data processing platform designed for:

* Point cloud processing
* 3D reconstruction
* Semantic segmentation
* Interactive visualization

Features
--------

* Real-time 3D visualization
* GPU-accelerated processing
* Python and C++ APIs
* Cross-platform support
EOF

    # Create getting started page
    cat > source/getting_started.rst << 'EOF'
Getting Started
===============

Installation
------------

Python Package
^^^^^^^^^^^^^^

Download the wheel file from GitHub Releases and install:

.. code-block:: bash

   pip install cloudviewer-*.whl

From Source
^^^^^^^^^^^

.. code-block:: bash

   git clone https://github.com/Asher-1/ACloudViewer.git
   cd ACloudViewer
   mkdir build && cd build
   cmake ..
   make -j$(nproc)

Quick Start
-----------

Python Example
^^^^^^^^^^^^^^

.. code-block:: python

   import cloudViewer as cv3d
   
   # Load a point cloud
   pcd = cv3d.io.read_point_cloud("pointcloud.ply")
   
   # Visualize
   cv3d.visualization.draw_geometries([pcd])

C++ Example
^^^^^^^^^^^

.. code-block:: cpp

   #include <cloudViewer/CloudViewer.h>
   
   int main() {
       auto pcd = cv::io::ReadPointCloud("pointcloud.ply");
       cv::visualization::DrawGeometries({pcd});
       return 0;
   }
EOF

    echo -e "${GREEN}✅ Sphinx initialized${NC}"
else
    echo -e "${GREEN}✅ Sphinx configuration found${NC}"
fi
echo ""

# Step 4: Generate Doxygen XML
echo -e "${BLUE}📦 Step 4/5: Generating Doxygen XML${NC}"
if [ -f "Doxyfile" ]; then
    echo "Running Doxygen..."
    doxygen Doxyfile > /dev/null 2>&1
    echo -e "${GREEN}✅ Doxygen XML generated${NC}"
else
    echo -e "${YELLOW}⚠️  Doxyfile not found, skipping C++ API docs${NC}"
fi
echo ""

# Step 5: Build documentation
echo -e "${BLUE}📦 Step 5/5: Building documentation${NC}"

if [ "$CLEAN_BUILD" = "YES" ]; then
    echo "Cleaning previous build..."
    rm -rf _out html doxygen
fi

# Build HTML documentation using make_docs.py (like Open3D)
if [ -f "make_docs.py" ]; then
    echo "Using make_docs.py to build documentation..."
    python3 make_docs.py --sphinx --doxygen
elif [ -f "Makefile" ]; then
    echo "Using Makefile to build documentation..."
    make docs
elif [ -f "source/conf.py" ]; then
    echo "Using sphinx-build directly..."
    sphinx-build -b html source _out/html
else
    echo -e "${RED}❌ No build system found${NC}"
    exit 1
fi

# Create symlink for easier access
if [ -d "_out/html" ] && [ ! -L "html" ]; then
    ln -sf _out/html html
fi

echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ Documentation built successfully!${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "📂 Output directory: ${BLUE}$(pwd)/_out/html${NC}"
echo ""
echo "To view the documentation:"
echo -e "  ${BLUE}python3 -m http.server 8080 --directory _out/html${NC}"
echo "  Then open: http://localhost:8080"
echo ""

