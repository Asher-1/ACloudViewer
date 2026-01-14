#!/bin/bash
# Test script to verify documentation generation pipeline
# This script checks all critical components of the documentation build system

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📚 ACloudViewer Documentation Generation Pipeline Test"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Repository: ${REPO_ROOT}"
echo "Test started: $(date)"
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

test_passed=0
test_failed=0

# Test function
run_test() {
    local test_name="$1"
    local test_command="$2"
    
    echo -n "Testing: ${test_name}... "
    
    if eval "${test_command}" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASS${NC}"
        ((test_passed++))
        return 0
    else
        echo -e "${RED}✗ FAIL${NC}"
        ((test_failed++))
        return 1
    fi
}

# Test function with output
run_test_verbose() {
    local test_name="$1"
    local test_command="$2"
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Test: ${test_name}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    if eval "${test_command}"; then
        echo -e "${GREEN}✓ PASS: ${test_name}${NC}"
        echo ""
        ((test_passed++))
        return 0
    else
        echo -e "${RED}✗ FAIL: ${test_name}${NC}"
        echo ""
        ((test_failed++))
        return 1
    fi
}

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1. File Existence Tests"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

run_test "make_docs.py exists" "test -f ${REPO_ROOT}/docs/make_docs.py"
run_test "make_docs.py is executable" "test -x ${REPO_ROOT}/docs/make_docs.py"
run_test "Dockerfile.docs exists" "test -f ${REPO_ROOT}/docker/Dockerfile.docs"
run_test "ci_utils.sh exists" "test -f ${REPO_ROOT}/util/ci_utils.sh"
run_test "documented_modules.txt exists" "test -f ${REPO_ROOT}/docs/documented_modules.txt"
run_test "conf.py exists" "test -f ${REPO_ROOT}/docs/source/conf.py"
run_test "Doxyfile exists" "test -f ${REPO_ROOT}/docs/Doxyfile"
run_test "Makefile exists" "test -f ${REPO_ROOT}/docs/Makefile"
run_test "requirements.txt exists" "test -f ${REPO_ROOT}/docs/requirements.txt"
run_test "jupyter directory exists" "test -d ${REPO_ROOT}/docs/jupyter"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "2. Python Module Tests"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check for Python module in multiple possible locations
PYTHON_MODULE_FOUND=false
PYTHON_MODULE_PATH=""

# Location 1: build_app/lib/Release/Python/cuda/
if [ -f "${REPO_ROOT}/build_app/lib/Release/Python/cuda/pybind.cpython-"*".so" ]; then
    PYTHON_MODULE_FOUND=true
    PYTHON_MODULE_PATH="${REPO_ROOT}/build_app/lib/Release/Python/cuda/"
    echo -e "${GREEN}✓ PASS: Python module found at build_app/lib/Release/Python/cuda/${NC}"
    ls -lh "${PYTHON_MODULE_PATH}"/pybind.cpython-*.so
    ((test_passed++))
# Location 2: build/lib/python_package/
elif [ -f "${REPO_ROOT}/build/lib/python_package/cloudViewer.cpython-"*".so" ]; then
    PYTHON_MODULE_FOUND=true
    PYTHON_MODULE_PATH="${REPO_ROOT}/build/lib/python_package/"
    echo -e "${GREEN}✓ PASS: Python module found at build/lib/python_package/${NC}"
    ls -lh "${PYTHON_MODULE_PATH}"/cloudViewer.cpython-*.so
    ((test_passed++))
else
    echo -e "${YELLOW}⚠ WARNING: Python module not found (will be built by build_docs)${NC}"
    echo "  Expected locations:"
    echo "    - ${REPO_ROOT}/build_app/lib/Release/Python/cuda/pybind.cpython-*.so"
    echo "    - ${REPO_ROOT}/build/lib/python_package/cloudViewer.cpython-*.so"
    ((test_failed++))
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "3. Content Validation Tests"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Count documented modules
MODULE_COUNT=$(grep -c "^cloudViewer\." "${REPO_ROOT}/docs/documented_modules.txt" || echo 0)
if [ "${MODULE_COUNT}" -gt 0 ]; then
    echo -e "${GREEN}✓ PASS: documented_modules.txt contains ${MODULE_COUNT} modules${NC}"
    ((test_passed++))
else
    echo -e "${RED}✗ FAIL: documented_modules.txt is empty or invalid${NC}"
    ((test_failed++))
fi

# Count Jupyter notebooks
NOTEBOOK_COUNT=$(find "${REPO_ROOT}/docs/jupyter" -name "*.ipynb" 2>/dev/null | wc -l)
if [ "${NOTEBOOK_COUNT}" -gt 0 ]; then
    echo -e "${GREEN}✓ PASS: Found ${NOTEBOOK_COUNT} Jupyter notebooks${NC}"
    ((test_passed++))
else
    echo -e "${RED}✗ FAIL: No Jupyter notebooks found${NC}"
    ((test_failed++))
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "4. Script Integration Tests"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check if build_docs function exists in ci_utils.sh
if grep -q "^build_docs()" "${REPO_ROOT}/util/ci_utils.sh"; then
    echo -e "${GREEN}✓ PASS: build_docs() function exists in ci_utils.sh${NC}"
    ((test_passed++))
else
    echo -e "${RED}✗ FAIL: build_docs() function not found in ci_utils.sh${NC}"
    ((test_failed++))
fi

# Check if Dockerfile.docs calls build_docs
if grep -q "build_docs" "${REPO_ROOT}/docker/Dockerfile.docs"; then
    echo -e "${GREEN}✓ PASS: Dockerfile.docs calls build_docs${NC}"
    ((test_passed++))
else
    echo -e "${RED}✗ FAIL: Dockerfile.docs does not call build_docs${NC}"
    ((test_failed++))
fi

# Check if make_docs.py has correct import paths
if grep -q "build_app/lib/Release/Python/cuda" "${REPO_ROOT}/docs/make_docs.py"; then
    echo -e "${GREEN}✓ PASS: make_docs.py has correct Python module path${NC}"
    ((test_passed++))
else
    echo -e "${YELLOW}⚠ WARNING: make_docs.py may have outdated Python module path${NC}"
    ((test_failed++))
fi

# Check if conf.py has correct import paths
if grep -q "build_app/lib/Release/Python/cuda" "${REPO_ROOT}/docs/source/conf.py"; then
    echo -e "${GREEN}✓ PASS: conf.py has correct Python module path${NC}"
    ((test_passed++))
else
    echo -e "${YELLOW}⚠ WARNING: conf.py may have outdated Python module path${NC}"
    ((test_failed++))
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "5. Python Syntax Tests"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Test make_docs.py syntax
if python3 -m py_compile "${REPO_ROOT}/docs/make_docs.py" 2>/dev/null; then
    echo -e "${GREEN}✓ PASS: make_docs.py has valid Python syntax${NC}"
    ((test_passed++))
else
    echo -e "${RED}✗ FAIL: make_docs.py has syntax errors${NC}"
    ((test_failed++))
fi

# Test conf.py syntax
if python3 -m py_compile "${REPO_ROOT}/docs/source/conf.py" 2>/dev/null; then
    echo -e "${GREEN}✓ PASS: conf.py has valid Python syntax${NC}"
    ((test_passed++))
else
    echo -e "${RED}✗ FAIL: conf.py has syntax errors${NC}"
    ((test_failed++))
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "6. GitHub Actions Workflow Tests"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

WORKFLOW_FILE="${REPO_ROOT}/.github/workflows/documentation.yml"

if [ -f "${WORKFLOW_FILE}" ]; then
    echo -e "${GREEN}✓ PASS: documentation.yml workflow exists${NC}"
    ((test_passed++))
    
    # Check for key steps
    if grep -q "Dockerfile.docs" "${WORKFLOW_FILE}"; then
        echo -e "${GREEN}✓ PASS: Workflow builds Dockerfile.docs${NC}"
        ((test_passed++))
    else
        echo -e "${RED}✗ FAIL: Workflow does not build Dockerfile.docs${NC}"
        ((test_failed++))
    fi
    
    if grep -q "Extract documentation" "${WORKFLOW_FILE}"; then
        echo -e "${GREEN}✓ PASS: Workflow extracts documentation${NC}"
        ((test_passed++))
    else
        echo -e "${RED}✗ FAIL: Workflow does not extract documentation${NC}"
        ((test_failed++))
    fi
    
    if grep -q "Deploy to GitHub Pages" "${WORKFLOW_FILE}"; then
        echo -e "${GREEN}✓ PASS: Workflow deploys to GitHub Pages${NC}"
        ((test_passed++))
    else
        echo -e "${RED}✗ FAIL: Workflow does not deploy to GitHub Pages${NC}"
        ((test_failed++))
    fi
else
    echo -e "${RED}✗ FAIL: documentation.yml workflow not found${NC}"
    ((test_failed++))
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "7. Documentation Build Chain Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "📋 Build Chain Flow:"
echo ""
echo "  GitHub Actions (.github/workflows/documentation.yml)"
echo "    ↓"
echo "  Docker Build (docker/Dockerfile.docs)"
echo "    ↓"
echo "  ci_utils.sh::build_docs()"
echo "    ↓"
echo "  make_docs.py --sphinx --doxygen"
echo "    ↓"
echo "  ┌─────────────────────┬─────────────────────┐"
echo "  │                     │                     │"
echo "  │  PyAPIDocsBuilder   │  DoxygenDocsBuilder │"
echo "  │  (Python API)       │  (C++ API)          │"
echo "  │                     │                     │"
echo "  │  • Import module    │  • Run Doxygen      │"
echo "  │  • Generate .rst    │  • Generate HTML    │"
echo "  │  • autodoc          │  • Copy to output   │"
echo "  │                     │                     │"
echo "  └─────────────────────┴─────────────────────┘"
echo "    ↓                     ↓"
echo "  JupyterDocsBuilder    SphinxDocsBuilder"
echo "  (Copy notebooks)      (Build HTML)"
echo "    ↓                     ↓"
echo "  docs/_out/html/"
echo "    ↓"
echo "  Package as .tar.gz"
echo "    ↓"
echo "  Deploy to GitHub Pages"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "8. Test Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

total_tests=$((test_passed + test_failed))
pass_rate=$((test_passed * 100 / total_tests))

echo "Total tests: ${total_tests}"
echo -e "Passed:      ${GREEN}${test_passed}${NC}"
echo -e "Failed:      ${RED}${test_failed}${NC}"
echo "Pass rate:   ${pass_rate}%"
echo ""

if [ "${test_failed}" -eq 0 ]; then
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}✅ ALL TESTS PASSED - Documentation pipeline is ready!${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    exit 0
else
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${RED}⚠️  SOME TESTS FAILED - Please review the errors above${NC}"
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    exit 1
fi
