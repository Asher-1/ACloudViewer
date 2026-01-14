#!/bin/bash
# =============================================================================
# Quick Documentation Warnings Counter
# =============================================================================
# This script quickly counts Sphinx warnings by category without full rebuild.
#
# Usage:
#   ./count_warnings.sh [log_file]
#
# If no log file is provided, it will use the last sphinx build log or build now.

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${REPO_ROOT}/build_app"
LOG_FILE="${1:-${BUILD_DIR}/sphinx_warnings.log}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         Documentation Warnings Counter                       ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if we need to build
if [ ! -f "$LOG_FILE" ] || [ "$1" = "--rebuild" ]; then
    echo -e "${YELLOW}Building documentation...${NC}"
    cd "$BUILD_DIR" 2>/dev/null || {
        echo -e "${RED}Error: build_app directory not found${NC}"
        echo "Please run: mkdir -p build_app && cd build_app && cmake .. && make"
        exit 1
    }
    
    make sphinx-html 2>&1 | tee "$LOG_FILE"
    echo ""
fi

# Count warnings
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Warning Statistics${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"

total_warnings=$(grep -c "^WARNING:" "$LOG_FILE" 2>/dev/null || echo "0")
total_errors=$(grep -c "^ERROR:" "$LOG_FILE" 2>/dev/null || echo "0")

echo -e "${YELLOW}Total Warnings:${NC} $total_warnings"
echo -e "${RED}Total Errors:${NC} $total_errors"
echo ""

# Top 10 warning types
echo -e "${BLUE}Top 10 Most Common Warnings:${NC}"
grep "^WARNING:" "$LOG_FILE" 2>/dev/null | \
    cut -d: -f3- | \
    sort | uniq -c | sort -rn | head -10 | \
    awk '{printf "  %4d  %s\n", $1, substr($0, index($0,$2))}'
echo ""

# Warnings by module
echo -e "${BLUE}Warnings by Module:${NC}"
for module in geometry io utility pipelines visualization ml reconstruction camera core data t; do
    count=$(grep "python_api/cloudViewer\.${module}" "$LOG_FILE" 2>/dev/null | \
            grep -E "^(WARNING|ERROR):" | wc -l || echo "0")
    if [ "$count" -gt 0 ]; then
        printf "  %-15s: %4d issues\n" "$module" "$count"
    fi
done
echo ""

# Top problematic files
echo -e "${BLUE}Top 10 Files with Most Issues:${NC}"
grep -E "^(WARNING|ERROR):" "$LOG_FILE" 2>/dev/null | \
    grep "python_api/cloudViewer" | \
    sed 's/.*python_api\/\([^:]*\):.*/\1/' | \
    sort | uniq -c | sort -rn | head -10 | \
    awk '{printf "  %4d  %s\n", $1, $2}'
echo ""

echo -e "${GREEN}Log file: $LOG_FILE${NC}"
echo -e "${BLUE}For detailed analysis, run: ./audit_docstrings.sh${NC}"
