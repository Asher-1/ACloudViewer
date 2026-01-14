#!/bin/bash
# =============================================================================
# Docstring Quality Audit Script
# =============================================================================
# This script audits C++ pybind11 docstrings and generates reports on common
# issues that cause Sphinx warnings.
#
# Usage:
#   ./audit_docstrings.sh [module_name]
#
# Examples:
#   ./audit_docstrings.sh                 # Audit all modules
#   ./audit_docstrings.sh geometry        # Audit only geometry module
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYBIND_DIR="${REPO_ROOT}/libs/Python/pybind"
OUTPUT_DIR="/tmp/acloudviewer_docstring_audit"
MODULE_FILTER="${1:-}"

mkdir -p "$OUTPUT_DIR"

echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         ACloudViewer Docstring Quality Audit                 ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# =============================================================================
# Function: Check for common docstring issues
# =============================================================================
check_docstring_issues() {
    local cpp_file="$1"
    local module_name=$(basename $(dirname "$cpp_file"))
    
    # Issue 1: Unescaped asterisks (inline emphasis)
    grep -n '\.def.*"\([^"]*\*[^"\\]\)\+"' "$cpp_file" 2>/dev/null || true
    
    # Issue 2: Missing indentation in Args/Returns
    grep -n '\.def.*"Args:\\n[A-Za-z]' "$cpp_file" 2>/dev/null || true
    
    # Issue 3: Unclosed code blocks
    grep -n '\.def.*"Example:[^:]' "$cpp_file" 2>/dev/null || true
}

# =============================================================================
# Function: Count functions with/without docstrings
# =============================================================================
count_docstrings() {
    local cpp_file="$1"
    
    # Count .def() calls
    local total_defs=$(grep -c '\.def(' "$cpp_file" 2>/dev/null || echo "0")
    total_defs=$(echo "$total_defs" | tr -d '[:space:]')
    total_defs=${total_defs:-0}
    
    # Count .def() with docstring (has quoted string after method name)
    local with_docs=$(grep '\.def(.*, "' "$cpp_file" 2>/dev/null | wc -l || echo "0")
    with_docs=$(echo "$with_docs" | tr -d '[:space:]')
    with_docs=${with_docs:-0}
    
    # Count .def() without docstring
    local without_docs=$((total_defs - with_docs))
    
    echo "$total_defs $with_docs $without_docs"
}

# =============================================================================
# Function: Extract module statistics
# =============================================================================
analyze_module() {
    local module_dir="$1"
    local module_name=$(basename "$module_dir")
    
    echo -e "${YELLOW}Analyzing module: ${module_name}${NC}"
    
    local total_files=$(find "$module_dir" -name "*.cpp" | wc -l)
    local total_functions=0
    local functions_with_docs=0
    local functions_without_docs=0
    
    # Analyze each CPP file
    while IFS= read -r cpp_file; do
        read -r total with without < <(count_docstrings "$cpp_file")
        total_functions=$((total_functions + total))
        functions_with_docs=$((functions_with_docs + with))
        functions_without_docs=$((functions_without_docs + without))
    done < <(find "$module_dir" -name "*.cpp")
    
    # Calculate percentage
    local percentage=0
    if [ $total_functions -gt 0 ]; then
        percentage=$((100 * functions_with_docs / total_functions))
    fi
    
    echo "  Files: $total_files"
    echo "  Total functions: $total_functions"
    echo "  With docstrings: $functions_with_docs ($percentage%)"
    echo "  Without docstrings: $functions_without_docs"
    echo ""
    
    # Write to report
    echo "$module_name,$total_files,$total_functions,$functions_with_docs,$functions_without_docs,$percentage" \
        >> "$OUTPUT_DIR/module_stats.csv"
}

# =============================================================================
# Function: Build documentation and capture warnings
# =============================================================================
capture_sphinx_warnings() {
    echo -e "${YELLOW}Building documentation to capture warnings...${NC}"
    
    cd "${REPO_ROOT}/build_app" 2>/dev/null || {
        echo -e "${RED}Error: build_app directory not found${NC}"
        echo "Please build the project first with:"
        echo "  mkdir build_app && cd build_app"
        echo "  cmake .. -DBUILD_PYTHON_MODULE=ON"
        echo "  make -j\$(nproc)"
        return 1
    }
    
    # Build and capture warnings
    make sphinx-html 2>&1 | tee "$OUTPUT_DIR/sphinx_build.log"
    
    # Extract and categorize warnings
    echo ""
    echo -e "${YELLOW}Categorizing warnings...${NC}"
    
    grep "^WARNING:" "$OUTPUT_DIR/sphinx_build.log" | \
        cut -d: -f3- | \
        sort | uniq -c | sort -rn > "$OUTPUT_DIR/warning_summary.txt"
    
    grep "^ERROR:" "$OUTPUT_DIR/sphinx_build.log" | \
        cut -d: -f3- | \
        sort | uniq -c | sort -rn > "$OUTPUT_DIR/error_summary.txt"
    
    # Count by module
    for module in geometry io utility pipelines visualization ml reconstruction; do
        count=$(grep "python_api/cloudViewer\.${module}" "$OUTPUT_DIR/sphinx_build.log" | \
                grep -E "^(WARNING|ERROR):" | wc -l || echo "0")
        echo "$module: $count issues" >> "$OUTPUT_DIR/module_warnings.txt"
    done
    
    echo ""
    echo -e "${GREEN}Warnings captured to: $OUTPUT_DIR/${NC}"
}

# =============================================================================
# Function: Generate report
# =============================================================================
generate_report() {
    echo ""
    echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║                    Audit Report Summary                      ║${NC}"
    echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    if [ -f "$OUTPUT_DIR/warning_summary.txt" ]; then
        echo -e "${YELLOW}Top 10 Most Common Warnings:${NC}"
        head -10 "$OUTPUT_DIR/warning_summary.txt"
        echo ""
    fi
    
    if [ -f "$OUTPUT_DIR/error_summary.txt" ]; then
        echo -e "${RED}Top 10 Most Common Errors:${NC}"
        head -10 "$OUTPUT_DIR/error_summary.txt"
        echo ""
    fi
    
    if [ -f "$OUTPUT_DIR/module_warnings.txt" ]; then
        echo -e "${YELLOW}Warnings by Module:${NC}"
        cat "$OUTPUT_DIR/module_warnings.txt"
        echo ""
    fi
    
    if [ -f "$OUTPUT_DIR/module_stats.csv" ]; then
        echo -e "${YELLOW}Module Documentation Coverage:${NC}"
        echo "Module,Files,Functions,Documented,Undocumented,Coverage%"
        cat "$OUTPUT_DIR/module_stats.csv"
        echo ""
    fi
    
    echo -e "${GREEN}Full reports saved to: $OUTPUT_DIR${NC}"
    echo ""
    echo -e "${BLUE}Next Steps:${NC}"
    echo "1. Review warning_summary.txt for most common issues"
    echo "2. Check module_warnings.txt to prioritize which modules to fix"
    echo "3. See DOCSTRING_STYLE_GUIDE.md for fixing guidelines"
    echo "4. Run ./fix_docstrings.sh <module> to auto-fix common issues"
}

# =============================================================================
# Main execution
# =============================================================================
main() {
    # Initialize CSV header
    echo "Module,Files,Functions,Documented,Undocumented,Coverage%" \
        > "$OUTPUT_DIR/module_stats.csv"
    
    # Analyze pybind modules
    if [ -z "$MODULE_FILTER" ]; then
        # Analyze all modules
        find "$PYBIND_DIR" -mindepth 1 -maxdepth 1 -type d | while read -r module_dir; do
            analyze_module "$module_dir"
        done
    else
        # Analyze specific module
        module_dir="$PYBIND_DIR/$MODULE_FILTER"
        if [ -d "$module_dir" ]; then
            analyze_module "$module_dir"
        else
            echo -e "${RED}Error: Module '$MODULE_FILTER' not found in $PYBIND_DIR${NC}"
            exit 1
        fi
    fi
    
    # Capture Sphinx warnings
    if [ -z "$MODULE_FILTER" ]; then
        capture_sphinx_warnings || echo -e "${YELLOW}Skipping Sphinx warning capture${NC}"
    fi
    
    # Generate final report
    generate_report
}

# Run main
main
