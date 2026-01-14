# ACloudViewer Chinese Translation

This directory contains the Chinese translation files and related tools for ACloudViewer.

## 📊 Current Status

- **Translation File**: `ACloudViewer_zh.ts`
- **Total Messages**: 3,612
- **Translated**: 2,746 (76.0%)
- **Untranslated**: 866 (24.0%)

## 📂 Directory Structure

```
eCV/translations/
├── ACloudViewer_zh.ts          # Main translation file (only one, edit this)
├── CMakeLists.txt               # Build configuration
├── README.md                    # This file
│
└── scripts/                     # Tool scripts folder
    ├── README.md                # Script documentation
    ├── *.py                     # 7 automation scripts
    ├── ACloudViewer_zh.ts.backup # Original backup
    ├── ACloudViewer_zh_review.txt # Review list
    └── ACloudViewer_zh.xlsx     # Excel reference
```

## 🚀 Quick Start

### Method 1: Using Qt Linguist (Recommended)

```bash
/opt/qt515/bin/linguist ACloudViewer_zh.ts
```

Qt Linguist provides:
- Visual translation interface
- Context preview
- Translation suggestions
- Batch operations
- Spell checking

### Method 2: Batch Auto-Translation

```bash
cd scripts
python3 final_batch.py          # Most recommended: for remaining content
python3 batch_translate.py      # Basic: common terms
python3 enhance_coverage.py     # Enhanced: extended dictionary
```

### Method 3: Quality Review

```bash
cd scripts
python3 quality_review.py       # Review translation quality
```

## 📘 Translation Standards

### Core Principles: Faithfulness, Expressiveness, Elegance

1. **信 (Faithfulness)** - Faithful to original
   - Accurately convey original meaning
   - No omission or addition
   - Maintain professional terminology accuracy

2. **达 (Expressiveness)** - Fluent expression
   - Follow Chinese language conventions
   - Avoid Europeanized sentence patterns
   - Clear and easy to understand

3. **雅 (Elegance)** - Elegant writing
   - Use professional standard expressions
   - Accurate terminology with unified style
   - Avoid colloquialism

4. **统一 (Consistency)** - Terminology consistency
   - Same concept uses same translation
   - Follow terminology reference table
   - Unified overall style

5. **无歧义 (Clarity)** - Clear and unambiguous
   - Avoid vague expressions
   - One word, one meaning
   - Clear contextual semantics

### Core Terminology Reference

| English | Chinese | Notes |
|---------|---------|-------|
| Point Cloud | 点云 | Use "点云" consistently |
| Mesh | 网格 | Use "网格" consistently |
| Normal | 法线 | Not "法向量" |
| Scalar Field | 标量场 | Not "标量域" |
| Registration | 配准 | Not "注册" or "对齐" |
| Bounding Box | 包围盒 | Not "边界框" |
| Filter | 滤波 | |
| Segment | 分割 | |
| Transform | 变换 | |
| Translation | 平移 | Not "移动" |
| Rotation | 旋转 | |
| Scale | 缩放 | |

For more terms, refer to script comments in the `scripts/` directory.

## 🔧 Testing Translation

### 1. Compile Translation File

```bash
cd ../../build_app
lrelease ../eCV/translations/ACloudViewer_zh.ts \
    -qm ../eCV/translations/ACloudViewer_zh.qm
```

### 2. Run Application Test

```bash
LANG=zh_CN.UTF-8 ./bin/ACloudViewer
```

### 3. Verify Key Areas

- ✅ Main menu bar
- ✅ Toolbar tooltips
- ✅ Dialog titles and buttons
- ✅ Error and warning messages
- ✅ Status bar tips

## 📋 Remaining Work

### Priority 1 - Core Interface (422 entries)
- **MainWindow** (185) - Main menu and toolbar
- **QObject** (237) - Core function messages

### Priority 2 - Professional Tools (106 entries)
- **ccCompass** (72) - Geological compass tool
- **qFacets** (34) - Facet analysis tool

### Priority 3 - Plugin Functions (338 entries)
- RasterizeToolDialog (30)
- qCanupoPlugin (17)
- Various other professional plugins

**Target**: Reach 85%+ coverage

## 🛠️ Tool Scripts Description

For detailed instructions, see [`scripts/README.md`](scripts/README.md)

### Main Scripts

1. **quality_review.py** - Translation quality review
   - Check terminology consistency
   - Detect expression issues
   - Generate review report

2. **batch_translate.py** - Batch translation tool
   - Auto-translate based on dictionary
   - Generate review list

3. **final_batch.py** - Final batch translation
   - Target remaining content
   - Professional terminology translation

4. **enhance_coverage.py** - Coverage enhancement
   - Extended dictionary
   - Improve coverage

## 📈 Improvement History

| Stage | Coverage | Added | Description |
|-------|----------|-------|-------------|
| Initial | 72.2% | - | Original state |
| Round 1 | 75.6% | +126 | Basic translation improvement |
| Quality Opt | 75.6% | +25 fixes | Terminology unification |
| Final Batch | 76.0% | +15 | Professional terms supplement |
| **Total** | **76.0%** | **+141** | **+3.8% improvement** |

## 🎯 Quality Assurance

### Completed
✅ Established translation standards (Faithfulness, Expressiveness, Elegance)  
✅ 200+ standardized terms  
✅ Terminology consistency check mechanism  
✅ Eliminated high-priority terminology issues  
✅ File structure organized  
✅ Automated tool chain  

### Quality Review Results
- Issues found: 412
- Auto-fixed: 25 terminology inconsistencies
- Manual review needed: 387 ambiguous expressions

## 🤝 Contribution Guidelines

### Adding Translations

1. **Recommended Method**: Use Qt Linguist
   ```bash
   /opt/qt515/bin/linguist ACloudViewer_zh.ts
   ```

2. **Batch Method**: Extend script dictionary
   - Edit `scripts/final_batch.py`
   - Add new entries in `FINAL_TRANSLATIONS` dictionary
   - Run script

3. **Validation**: Run quality review
   ```bash
   python3 scripts/quality_review.py
   ```

### Pre-submission Checklist

- [ ] Terminology follows standards
- [ ] Expression follows Chinese conventions
- [ ] No grammatical errors
- [ ] Tested in application
- [ ] Quality review script passed
- [ ] All tests passed

## 📞 Support

- **Script Usage**: See `scripts/README.md`
- **Translation Standards**: Refer to this document and script comments
- **Issue Reporting**: Submit Issue to GitHub
- **Pull Request**: Contributions welcome

## 📚 Reference Resources

- [Qt Linguist Manual](https://doc.qt.io/qt-5/linguist-manual.html)
- [Qt Translation Best Practices](https://doc.qt.io/qt-5/i18n-source-translation.html)
- [scripts/README.md](scripts/README.md) - Detailed script usage guide

---

**Last Updated**: 2026-01-14  
**Maintainer**: ACloudViewer Translation Team  
**Coverage**: 76.0% (2,746/3,612)
