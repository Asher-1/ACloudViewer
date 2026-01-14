#!/bin/bash
# ============================================================================
# ACloudViewer Translation Update Script
# ============================================================================
# This script updates the translation file by extracting all translatable
# strings from source code and UI files.
#
# Usage:
#   cd eCV/translations
#   bash update_translations.sh
# ============================================================================

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ECV_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$ECV_DIR/.." && pwd)"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔄 ACloudViewer Translation Update"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📂 Directories:"
echo "   Script:  $SCRIPT_DIR"
echo "   eCV:     $ECV_DIR"
echo "   Project: $PROJECT_ROOT"
echo ""

# Find lupdate
LUPDATE=""
if command -v lupdate-qt5 &> /dev/null; then
    LUPDATE="lupdate-qt5"
elif command -v lupdate &> /dev/null; then
    LUPDATE="lupdate"
elif [ -f "/opt/qt515/bin/lupdate" ]; then
    LUPDATE="/opt/qt515/bin/lupdate"
else
    echo "❌ Error: lupdate not found!"
    echo "   Please install Qt development tools"
    exit 1
fi

echo "✓ Found lupdate: $LUPDATE"
echo ""

# Backup existing translation file
TS_FILE="$SCRIPT_DIR/ACloudViewer_zh.ts"
if [ -f "$TS_FILE" ]; then
    BACKUP_FILE="$TS_FILE.backup.$(date +%Y%m%d_%H%M%S)"
    echo "💾 Backing up existing translation file..."
    cp "$TS_FILE" "$BACKUP_FILE"
    echo "   Backup: $BACKUP_FILE"
    echo ""
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔍 Step 1: Collecting source files..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

cd "$ECV_DIR"

# Collect all source files
SOURCE_FILES=$(find . -name "*.cpp" -o -name "*.h" -o -name "*.ui" | grep -v "build" | grep -v "\.moc" | sort)
FILE_COUNT=$(echo "$SOURCE_FILES" | wc -l)

echo "Found $FILE_COUNT source files:"
echo "$SOURCE_FILES" | head -20
if [ $FILE_COUNT -gt 20 ]; then
    echo "... and $((FILE_COUNT - 20)) more files"
fi
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔄 Step 2: Running lupdate..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Run lupdate
$LUPDATE $SOURCE_FILES -ts "$TS_FILE" -no-obsolete

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 Step 3: Analyzing translation status..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Count translations
TOTAL_MESSAGES=$(grep -c "<message>" "$TS_FILE" || echo "0")
UNFINISHED=$(grep -c 'type="unfinished"' "$TS_FILE" || echo "0")
TRANSLATED=$((TOTAL_MESSAGES - UNFINISHED))
PERCENTAGE=0
if [ $TOTAL_MESSAGES -gt 0 ]; then
    PERCENTAGE=$((TRANSLATED * 100 / TOTAL_MESSAGES))
fi

echo "Translation Statistics:"
echo "  Total messages:    $TOTAL_MESSAGES"
echo "  Translated:        $TRANSLATED ($PERCENTAGE%)"
echo "  Untranslated:      $UNFINISHED"
echo ""

if [ $UNFINISHED -gt 0 ]; then
    echo "⚠️  Warning: $UNFINISHED messages need translation"
    echo ""
    echo "Next steps:"
    echo "  1. Open translation file in Qt Linguist:"
    echo "     linguist $TS_FILE"
    echo ""
    echo "  2. Or use batch translation scripts:"
    echo "     cd scripts"
    echo "     python3 final_batch.py"
    echo ""
    echo "  3. After translating, compile the .qm file:"
    echo "     cd ../../build_app"
    echo "     make translations"
    echo ""
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Translation file updated successfully!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Updated file: $TS_FILE"
if [ -f "$BACKUP_FILE" ]; then
    echo "Backup file:  $BACKUP_FILE"
fi
echo ""
