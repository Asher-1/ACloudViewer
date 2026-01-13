#!/bin/bash
# Test script to validate documentation structure
# This script checks if all required files and directories are in place

set -e

echo "================================================================================"
echo "    ACloudViewer Documentation Structure Validation"
echo "================================================================================"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ERRORS=0

# Function to check file existence
check_file() {
    if [ -f "$1" ]; then
        echo "✅ Found: $1"
    else
        echo "❌ Missing: $1"
        ERRORS=$((ERRORS + 1))
    fi
}

# Function to check directory existence
check_dir() {
    if [ -d "$1" ]; then
        echo "✅ Found directory: $1"
    else
        echo "❌ Missing directory: $1"
        ERRORS=$((ERRORS + 1))
    fi
}

echo "📋 Checking core configuration files..."
echo "─────────────────────────────────────────────────────────────────────────────"
check_file "source/conf.py"
check_file "Doxyfile"
check_file "make_docs.py"
check_file "Makefile"
check_file "requirements.txt"
echo ""

echo "📁 Checking documentation structure..."
echo "─────────────────────────────────────────────────────────────────────────────"
check_file "source/index.rst"
check_dir "source/getting_started"
check_dir "source/tutorial"
check_dir "source/python_api"
check_dir "source/cpp_api"
echo ""

echo "📚 Checking Tutorial structure..."
echo "─────────────────────────────────────────────────────────────────────────────"
check_file "source/tutorial/index.rst"
check_dir "source/tutorial/geometry"
check_dir "source/tutorial/visualization"
check_dir "source/tutorial/pipelines"
check_dir "source/tutorial/reconstruction"
check_dir "source/tutorial/ml"
check_dir "source/tutorial/advanced"
echo ""

echo "🐍 Checking Python API structure..."
echo "─────────────────────────────────────────────────────────────────────────────"
check_file "source/python_api/index.rst"
check_file "source/python_api/cloudviewer.rst"
check_file "source/python_api/cloudviewer.geometry.rst"
check_file "source/python_api/cloudviewer.io.rst"
echo ""

echo "⚙️  Checking C++ API structure..."
echo "─────────────────────────────────────────────────────────────────────────────"
check_file "source/cpp_api/index.rst"
check_file "source/cpp_api/core.rst"
check_file "source/cpp_api/geometry.rst"
check_file "source/cpp_api/io.rst"
echo ""

echo "🔧 Checking build scripts..."
echo "─────────────────────────────────────────────────────────────────────────────"
check_file "test_build_docs.sh"
echo "ℹ️  Note: Dependencies are installed via util/ci_utils.sh (for CI/Docker)"
echo ""

echo "📊 Counting RST files..."
echo "─────────────────────────────────────────────────────────────────────────────"
RST_COUNT=$(find source -name "*.rst" -type f | wc -l)
echo "Found $RST_COUNT RST files"
if [ "$RST_COUNT" -ge 50 ]; then
    echo "✅ RST file count looks good (>= 50)"
else
    echo "⚠️  RST file count is low (< 50), expected ~64 files"
fi
echo ""

echo "================================================================================"
if [ $ERRORS -eq 0 ]; then
    echo "✅ All checks passed! Documentation structure is valid."
    echo ""
    echo "📚 Structure Summary:"
    echo "   - Tutorial: $(find source/tutorial -name "*.rst" | wc -l) files"
    echo "   - Python API: $(find source/python_api -name "*.rst" | wc -l) files"
    echo "   - C++ API: $(find source/cpp_api -name "*.rst" | wc -l) files"
    echo "   - Total RST: $RST_COUNT files"
    echo ""
    echo "🚀 Ready to build documentation!"
    echo ""
    echo "Next steps:"
    echo "  1. Docker build:  docker build -f ../docker/Dockerfile.docs -t acloudviewer-docs ."
    echo "  2. Or use Makefile: make docs (requires dependencies)"
    echo "  3. Or use CMake:    cmake -DBUILD_DOCUMENTATION=ON .."
    exit 0
else
    echo "❌ Found $ERRORS error(s) in documentation structure"
    exit 1
fi

