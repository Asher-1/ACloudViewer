# Translation Tool Scripts Directory

This directory contains the translation scripts and tools for ACloudViewer Chinese translations.

## 📂 File Structure

### Core Update Script

#### `update_translations.sh`
**Primary Translation Update Tool**

Automatically extracts all translatable strings from source code and updates the translation file.

**Features:**
- Scans all `.cpp`, `.h`, `.ui` files in the app directory
- Extracts translatable strings using Qt's `lupdate` tool
- Updates `ACloudViewer_zh.ts` with new strings
- Preserves existing translations
- Creates timestamped backups automatically
- Provides translation statistics

**Usage:**
```bash
cd <repo-root>/app/translations/scripts
bash update_translations.sh
```

**When to use:**
- After adding new UI elements or strings to source code
- When source files are modified with new translatable content
- To refresh the translation file with latest source strings

---

### Translation Scripts (7 files)

Sequential scripts used to translate all 866 remaining entries from 76% to 100%:

#### `translate_to_100.py`
**Round 1: 76% → 90.6% (+525 entries)**

- Comprehensive term translation covering all basic UI elements
- Handles simple terms, common operations, and standard phrases
- Includes point cloud, mesh, registration, and filter operations
- Features: 500+ translation entries in comprehensive dictionary

**Usage:**
```bash
python3 translate_to_100.py
```

#### `translate_round2.py`
**Round 2: 90.6% → 92.6% (+75 entries)**

- Complete sentences and parameter messages
- Progress messages, error messages with parameters
- Status messages with formatted output
- Technical parameter descriptions

**Usage:**
```bash
python3 translate_round2.py
```

#### `translate_round3.py`
**Round 3: 92.6% → 95.7% (+110 entries)**

- Error messages and console output
- Command-line parameter messages
- Missing parameter warnings
- Component-specific error messages

**Usage:**
```bash
python3 translate_round3.py
```

#### `translate_final100.py`
**Round 4: 95.7% → 97.0% (+47 entries)**

- Parameter messages and warnings
- Specialized dialog messages
- Technical formulas and expressions
- Algorithm-specific terminology

**Usage:**
```bash
python3 translate_final100.py
```

#### `translate_last_mile.py`
**Round 5: 97.0% → 98.2% (+45 entries)**

- Long-form technical descriptions
- Multi-paragraph help texts
- Algorithm explanations
- Complex dialog content

**Usage:**
```bash
python3 translate_last_mile.py
```

#### `translate_absolute_final.py`
**Round 6: 98.2% → 98.8% (+22 entries)**

- Near-final entries with complex formatting
- HTML-aware translations
- Multi-line technical specifications
- Edge case messages

**Usage:**
```bash
python3 translate_absolute_final.py
```

#### `translate_100_percent.py`
**Round 7: 98.8% → 99.8% (+36 entries)**

- Final push entries
- Specialized plugin content
- Complex multi-line descriptions
- Last batch of technical content

**Usage:**
```bash
python3 translate_100_percent.py
```

**Note**: Round 8 (final 6 entries to 100%) was handled with an inline script.

---

### Quality Assurance Tools

#### `quality_review.py`
**Translation Quality Audit Tool**

Performs automated quality checks on translations:
- Terminology consistency verification
- Expression quality assessment (信达雅 standards)
- Ambiguity detection
- Readability evaluation
- Generates detailed quality audit report

**Usage:**
```bash
python3 quality_review.py
```

**Output:** `QUALITY_AUDIT_REPORT.md`

---

### Reference Files

#### `ACloudViewer_zh.ts.backup`
**Original Translation Backup**

- Backup of the translation file before optimization (76% coverage)
- Preserved for reference and rollback if needed
- Size: ~844KB
- Entries: 2,746 translated, 866 untranslated

---

## 🚀 Translation Workflow

### Complete Translation Process

To reproduce the entire translation from 76% to 100%:

```bash
# Navigate to scripts directory
cd <repo-root>/app/translations/scripts

# Step 0: Update translation file from source code (if needed)
bash update_translations.sh

# Round 1: Basic terms and common operations
python3 translate_to_100.py

# Round 2: Complete sentences and messages
python3 translate_round2.py

# Round 3: Error messages and console output
python3 translate_round3.py

# Round 4: Parameter messages
python3 translate_final100.py

# Round 5: Long technical descriptions
python3 translate_last_mile.py

# Round 6: Near-final entries
python3 translate_absolute_final.py

# Round 7: Final push to near 100%
python3 translate_100_percent.py

# Round 8: Final 6 entries (handled inline)
# See main conversation for final inline script

# Quality check
python3 quality_review.py

# Compile translation
cd ../../build_app
lrelease ../app/translations/ACloudViewer_zh.ts -qm ../app/translations/ACloudViewer_zh.qm

# Test application
LANG=zh_CN.UTF-8 ./bin/ACloudViewer
```

---

## 📊 Translation Statistics

### Coverage Progression

| Round | Script | Entries Added | Total Translated | Coverage |
|-------|--------|---------------|------------------|----------|
| Initial | - | - | 2,746 | 76.0% |
| Round 1 | translate_to_100.py | +525 | 3,271 | 90.6% |
| Round 2 | translate_round2.py | +75 | 3,346 | 92.6% |
| Round 3 | translate_round3.py | +110 | 3,456 | 95.7% |
| Round 4 | translate_final100.py | +47 | 3,503 | 97.0% |
| Round 5 | translate_last_mile.py | +45 | 3,548 | 98.2% |
| Round 6 | translate_absolute_final.py | +22 | 3,570 | 98.8% |
| Round 7 | translate_100_percent.py | +36 | 3,606 | 99.8% |
| Round 8 | Inline script | +6 | 3,612 | **100.0%** ✅ |

### Total Achievement
- **Total Messages**: 3,612
- **Total Translated**: 3,612 (100.00%)
- **New Translations**: +866
- **Quality Score**: 96/100 (★★★★★)

---

## 🎯 Script Design Philosophy

### Incremental Approach
Each script targets a specific type of content:
1. **Simple terms** → Common UI elements
2. **Complete sentences** → User-facing messages
3. **Error messages** → Console and log output
4. **Parameters** → Technical specifications
5. **Long descriptions** → Help and documentation
6. **Edge cases** → Special formatting and HTML
7. **Final entries** → Remaining specialized content

### Benefits
- ✅ **Manageable**: Each round handles a reasonable amount
- ✅ **Testable**: Progress can be verified after each round
- ✅ **Traceable**: Clear history of what was translated when
- ✅ **Maintainable**: Easy to understand and modify
- ✅ **Reproducible**: Process is well-documented

---

## 🛠️ Customization Guide

### Updating Translation File from Source

When source code is updated with new translatable strings:

```bash
cd <repo-root>/app/translations/scripts
bash update_translations.sh
```

This will:
1. Scan all source files for translatable strings
2. Extract new strings using `lupdate`
3. Update `ACloudViewer_zh.ts` 
4. Create automatic backup
5. Show translation statistics

### Adding New Translations

If new content is added to ACloudViewer that needs translation:

1. **Analyze untranslated content**:
   ```python
   import xml.etree.ElementTree as ET
   tree = ET.parse('ACloudViewer_zh.ts')
   # Extract untranslated entries...
   ```

2. **Create targeted translation dictionary**:
   ```python
   NEW_TRANSLATIONS = {
       "English Text": "中文翻译",
       # Add more entries...
   }
   ```

3. **Apply translations**:
   Use any of the existing scripts as a template, modify the dictionary, and run.

4. **Verify quality**:
   ```bash
   python3 quality_review.py
   ```

### Modifying Translation Standards

Edit `quality_review.py` to update:
- `TERMINOLOGY_STANDARDS`: Standard terminology mapping
- `CONSISTENCY_CHECK`: Terms that must be consistent
- `EXPRESSION_IMPROVEMENTS`: Expression quality rules

---

## 📘 Translation Standards

### Core Principles (信达雅)

1. **信 (Faithfulness)** - Accurate & complete
   - Precisely convey original meaning
   - No omissions or additions
   - Maintain technical accuracy

2. **达 (Expressiveness)** - Natural expression
   - Follow Chinese language conventions
   - Avoid literal translations
   - Clear and understandable

3. **雅 (Elegance)** - Professional quality
   - Use standard professional terms
   - Consistent style throughout
   - Avoid colloquialisms

4. **统一 (Consistency)** - Unified terminology
   - Same concept uses same translation
   - Follow established glossary
   - Maintain overall consistency

5. **无歧义 (Clarity)** - Clear & unambiguous
   - Avoid vague expressions
   - One meaning per term
   - Context-appropriate

### Key Terminology

| English | Chinese | Notes |
|---------|---------|-------|
| Point Cloud | 点云 | Standard term |
| Mesh | 网格 | Not "格网" |
| Normal | 法线 | Not "法向量" |
| Scalar Field | 标量场 | Not "标量域" |
| Registration | 配准 | Not "注册" or "对齐" |
| Bounding Box | 包围盒 | Not "边界框" |
| Filter | 滤波 | Standard term |
| Segment | 分割 | Standard term |

See script source code for comprehensive terminology lists (500+ terms).

---

## 🔍 Quality Assurance

### Automated Checks
- ✅ Terminology consistency
- ✅ Expression quality
- ✅ Ambiguity detection
- ✅ Completeness verification

### Manual Review Areas
Priority items for manual review:
- Technical algorithm descriptions
- Error messages seen by users
- Help documentation
- Plugin-specific content

### Testing Checklist
- [ ] Compile .ts to .qm without errors
- [ ] Launch application with Chinese locale
- [ ] Verify main menus display correctly
- [ ] Check toolbar tooltips
- [ ] Test common dialogs
- [ ] Verify error messages
- [ ] Check specialized features

---

## 📞 Support

### Issues
If you encounter translation issues:
1. Check this documentation
2. Review script comments
3. Run quality_review.py
4. Consult the main README.md

### Contributing
To improve translations:
1. Use Qt Linguist for precision work
2. Or extend script dictionaries
3. Run quality checks
4. Test in application
5. Submit changes

### Resources
- [Qt Linguist Manual](https://doc.qt.io/qt-5/linguist-manual.html)
- [Qt i18n Best Practices](https://doc.qt.io/qt-5/i18n-source-translation.html)
- Main README: `../README.md`

---

## ✅ Cleanup Status

**This directory has been cleaned and optimized** (2026-01-14):

**Removed Files (17 total)**:
- 8 obsolete early scripts (superseded by final scripts)
- 7 intermediate analysis files (temporary results)
- 2 reference files (no longer needed)

**Retained Files (12 total)**:
- 1 update script (`update_translations.sh` - primary tool)
- 7 final translation scripts (sequential, production-ready)
- 1 quality review tool
- 2 documentation files (this file + parent README)
- 1 backup file (original state)

**Space Saved**: ~350KB

**Benefits**:
- ✅ Clear and organized structure
- ✅ Only production-ready scripts
- ✅ Easy to understand workflow
- ✅ Maintainable for future updates

---

**Last Updated**: 2026-01-14  
**Maintainer**: ACloudViewer Translation Team  
**Status**: ✅ Production Ready (100% Coverage)
