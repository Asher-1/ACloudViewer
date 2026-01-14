# ACloudViewer Documentation Quality Tools

This directory contains tools for auditing and improving docstring quality in ACloudViewer's C++ pybind11 bindings.

## Tools Overview

### 1. `audit_docstrings.sh`

Comprehensive audit tool that analyzes docstring quality and generates reports.

**Features:**
- Counts functions with/without docstrings
- Captures and categorizes Sphinx warnings
- Generates module-by-module statistics
- Identifies common formatting issues

**Usage:**

```bash
# Audit all modules
./audit_docstrings.sh

# Audit specific module
./audit_docstrings.sh geometry

# View results
cat /tmp/acloudviewer_docstring_audit/warning_summary.txt
cat /tmp/acloudviewer_docstring_audit/module_stats.csv
```

**Output Files:**
- `warning_summary.txt` - Top warnings by frequency
- `error_summary.txt` - Top errors by frequency
- `module_warnings.txt` - Warning counts per module
- `module_stats.csv` - Documentation coverage statistics
- `sphinx_build.log` - Full Sphinx build log

### 2. `fix_docstrings.py`

Automated fix tool for common docstring formatting issues.

**Fixes Applied:**
- Escapes unescaped asterisks (`*` → `\*`)
- Adds proper indentation to Args/Returns sections
- Fixes Example blocks (adds `::`)
- Converts inline code quotes (`'True'` → `` `True` ``)
- Escapes broken emphasis markers

**Usage:**

```bash
# Preview fixes without applying (dry-run)
python fix_docstrings.py libs/Python/pybind/geometry/ --dry-run

# Apply fixes with backup
python fix_docstrings.py libs/Python/pybind/geometry/ --backup

# Fix single file
python fix_docstrings.py libs/Python/pybind/geometry/pointcloud.cpp
```

**Options:**
- `--dry-run` - Preview changes without modifying files
- `--backup` - Create `.bak` backup files before modifying

## Workflow: Fixing Documentation Warnings

### Step 1: Audit Current State

```bash
cd /path/to/ACloudViewer
./docs/scripts/audit_docstrings.sh
```

This generates a complete report showing:
- Which modules have the most warnings
- What types of issues are most common
- Current documentation coverage

### Step 2: Prioritize Modules

Based on the audit report, choose which module to fix first. We recommend:

1. **Start with** `utility` (smallest, good for testing)
2. **Then** `io` (important, moderate size)
3. **Then** `geometry` (core module, larger)
4. **Then** `pipelines`, `visualization`, etc.

### Step 3: Auto-Fix Common Issues

```bash
# Preview fixes for a module
python docs/scripts/fix_docstrings.py libs/Python/pybind/geometry/ --dry-run

# Apply fixes with backup
python docs/scripts/fix_docstrings.py libs/Python/pybind/geometry/ --backup
```

### Step 4: Manual Review and Fix

Some issues require manual fixes:

1. **Complex RST formatting** - Follow examples in `DOCSTRING_STYLE_GUIDE.md`
2. **Missing docstrings** - Add proper documentation
3. **Duplicate definitions** - Add `:noindex:` where needed

### Step 5: Verify Fixes

```bash
cd build_app
make sphinx-html 2>&1 | grep -E "python_api/cloudViewer.geometry" | grep -E "^(WARNING|ERROR):"
```

Compare warning count before and after fixes.

### Step 6: Commit Changes

```bash
git add libs/Python/pybind/geometry/
git commit -m "docs: fix docstring format issues in geometry module

- Fixed unescaped asterisks causing RST emphasis warnings
- Added proper indentation to Args/Returns sections
- Fixed Example blocks to use proper RST code block syntax
- Reduced Sphinx warnings from 80 to 10

Refs: docs/DOCSTRING_STYLE_GUIDE.md"
```

## Examples

### Example 1: Quick Audit of Single Module

```bash
$ ./audit_docstrings.sh utility

Analyzing module: utility
  Files: 5
  Total functions: 45
  With docstrings: 38 (84%)
  Without docstrings: 7
```

### Example 2: Auto-Fix with Preview

```bash
$ python fix_docstrings.py libs/Python/pybind/utility/ --dry-run

Processing: libs/Python/pybind/utility/helper.cpp
  [DRY RUN] Would modify file
  
Processing: libs/Python/pybind/utility/eigen.cpp
  ○ No changes needed

Summary
================================
Files modified: 1
Total fixes applied: 15

This was a dry run. No files were actually modified.
```

### Example 3: Complete Module Fix Workflow

```bash
# 1. Audit before
./audit_docstrings.sh geometry > /tmp/before.txt

# 2. Auto-fix common issues
python fix_docstrings.py libs/Python/pybind/geometry/ --backup

# 3. Build docs and check warnings
cd build_app
make sphinx-html 2>&1 | grep geometry | grep WARNING | wc -l
# Output: 25 (reduced from 80)

# 4. Manual fixes for remaining issues
# Edit files based on DOCSTRING_STYLE_GUIDE.md

# 5. Rebuild and verify
make sphinx-html 2>&1 | grep geometry | grep WARNING | wc -l
# Output: 5 (much better!)

# 6. Commit
git commit -am "docs: fix geometry module docstrings"
```

## Integration with CI/CD

### Pre-commit Hook

Add to `.git/hooks/pre-commit`:

```bash
#!/bin/bash
# Check for docstring issues in changed files

changed_cpp=$(git diff --cached --name-only --diff-filter=ACM | grep '\.cpp$')

if [ -n "$changed_cpp" ]; then
    echo "Checking docstrings in modified C++ files..."
    for file in $changed_cpp; do
        python docs/scripts/fix_docstrings.py "$file" --dry-run > /dev/null
        if [ $? -ne 0 ]; then
            echo "Warning: $file may have docstring issues"
            echo "Run: python docs/scripts/fix_docstrings.py $file"
        fi
    done
fi
```

### GitHub Actions Check

```yaml
- name: Check Documentation Warnings
  run: |
    cd build
    make sphinx-html 2>&1 | tee sphinx.log
    
    # Count warnings
    warnings=$(grep -c "^WARNING:" sphinx.log || true)
    echo "Found $warnings Sphinx warnings"
    
    # Fail if warnings increased
    if [ $warnings -gt 200 ]; then
        echo "::error::Too many documentation warnings ($warnings)"
        exit 1
    fi
```

## Best Practices

### 1. Work Module-by-Module

Don't try to fix everything at once. Focus on one module and do it well.

### 2. Always Test After Fixing

```bash
cd build_app
make sphinx-html
# Check that docs still build and render correctly
```

### 3. Use Backup Option

Always use `--backup` when running auto-fix on production code:

```bash
python fix_docstrings.py libs/Python/pybind/ --backup
```

### 4. Review Auto-Fixes

Auto-fixes are not perfect. Always review changes before committing:

```bash
git diff libs/Python/pybind/geometry/
```

### 5. Follow the Style Guide

For manual fixes, always refer to `docs/DOCSTRING_STYLE_GUIDE.md`.

## Troubleshooting

### Issue: Script Not Found

```bash
# Make sure you're in the repository root
cd /path/to/ACloudViewer

# Make scripts executable
chmod +x docs/scripts/*.sh docs/scripts/*.py
```

### Issue: Python Import Errors

```bash
# Ensure you're using Python 3
python3 --version

# Script has no external dependencies, should work with standard library
```

### Issue: Build Directory Not Found

```bash
# audit_docstrings.sh needs build_app directory
mkdir -p build_app && cd build_app
cmake .. -DBUILD_PYTHON_MODULE=ON
make -j$(nproc)
```

## Additional Resources

- **Style Guide**: `docs/DOCSTRING_STYLE_GUIDE.md`
- **Warning Explanations**: `docs/DOCUMENTATION_WARNINGS.md`
- **Build Guide**: `docs/automation/BUILD_DOCUMENTATION.md`
- **Open3D Examples**: `/home/ludahai/develop/code/github/Open3D/cpp/pybind/`

## Contributing

Found a bug or have a suggestion? Please open an issue on GitHub with the `documentation` label.

---

**Maintained by**: ACloudViewer Documentation Team  
**Last Updated**: January 2026
