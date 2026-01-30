# Download System Automation Guide

> **Automated download selector system for ACloudViewer website**  
> Automatically scans GitHub Releases and generates download data for the interactive selector

---

## 📋 Table of Contents

1. [System Overview](#system-overview)
2. [How It Works](#how-it-works)
3. [File Structure](#file-structure)
4. [Usage](#usage)
5. [Automatic Discovery](#automatic-discovery)
6. [File Naming Conventions](#file-naming-conventions)
7. [Supported Configurations](#supported-configurations)
8. [Maintenance](#maintenance)

---

## 🎯 System Overview

The download automation system provides:

✅ **Automatic Platform Detection**  
- Windows, macOS, Linux (Ubuntu variants)
- x86_64 and ARM64 architectures
- CPU-only and CUDA versions

✅ **Automatic Python Version Discovery**  
- Scans all `.whl` files in releases
- Detects supported Python versions (3.6-3.13)
- Different versions may support different Python ranges

✅ **Smart Configuration Management**  
- App packages organized by OS version (Ubuntu 18.04/20.04/22.04/24.04)
- Wheel packages use manylinux (universal for Linux)
- Automatic version metadata generation

✅ **Zero Manual Maintenance**  
- Triggered by GitHub Actions on new releases
- No hardcoded version lists
- Self-updating based on actual files

---

## 🔄 How It Works

```
┌──────────────────┐
│ GitHub Releases  │
│  main-devel      │◄── New release published
│  v3.9.3         │
│  v3.9.2         │
└────────┬─────────┘
         │
         ▼
┌─────────────────────┐
│ GitHub Actions      │
│ .github/workflows/  │
│ update-website.yml  │
└────────┬────────────┘
         │
         ▼
┌─────────────────────────────┐
│ scan_releases.py            │
│                             │
│ 1. Fetch all releases       │
│ 2. Parse asset names        │
│ 3. Detect platforms/Python  │
│ 4. Build data structure     │
│ 5. Generate JSON            │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────┐
│ downloads_data.json │◄── Used by website
│                     │
│ {                   │
│   "version_metadata",│
│   "download_data"   │
│ }                   │
└─────────┬───────────┘
         │
         ▼
┌─────────────────────┐
│ Website (index.html)│
│ Interactive Selector│
│ - Version dropdown  │
│ - Platform selector │
│ - Python selector   │
│ - Smart filtering   │
└─────────────────────┘
```

---

## 📁 File Structure

```
docs/
├── automation/
│   ├── scripts/
│   │   └── scan_releases.py          # Main scanner script
│   └── DOWNLOAD_AUTOMATION_GUIDE.md  # This file
├── downloads_data.json                # Generated data (auto-updated)
├── index.html                         # Main website (uses JSON)
├── script.js                          # Download selector logic
└── styles.css                         # Styling

.github/
└── workflows/
    └── update-website-downloads.yml   # Auto-trigger on releases
```

---

## 🚀 Usage

### Manual Run (Local Testing)

```bash
cd /path/to/ACloudViewer
python3 docs/automation/scripts/scan_releases.py
```

**Output:**
- `docs/downloads_data.json` - Generated download data
- Console output with summary and warnings

### Automatic Run (GitHub Actions)

The system runs automatically when:
1. **New release is published** (any tag)
2. **Push to `main` branch**
3. **Daily at 00:00 UTC** (scheduled)
4. **Manual trigger** from Actions tab

No manual intervention required! 🎉

---

## 🔍 Automatic Discovery

The scanner automatically detects:

### 1. **Supported Python Versions**

Scans all wheel files in a release:
```
cloudviewer-3.9.3-cp38-cp38-win_amd64.whl   → Python 3.8
cloudviewer-3.9.3-cp39-cp39-win_amd64.whl   → Python 3.9
cloudviewer-3.9.3-cp310-cp310-win_amd64.whl → Python 3.10
...
```

**Result:**  
Version metadata automatically includes: `["3.8", "3.9", "3.10", ...]`

### 2. **Supported Ubuntu Versions**

Scans all `.run` files:
```
ACloudViewer-3.9.3-ubuntu18.04-cpu-amd64.run → Ubuntu 18.04
ACloudViewer-3.9.3-ubuntu20.04-cpu-amd64.run → Ubuntu 20.04
ACloudViewer-3.9.3-ubuntu22.04-cuda-amd64.run → Ubuntu 22.04
...
```

**Note:** Wheel packages don't need Ubuntu version distinction (manylinux is universal)

### 3. **Available Platforms**

Detects from file names:
- **Windows**: `.exe` files
- **macOS**: `.dmg` files
- **Linux**: `.run` files

### 4. **Architecture Support**

- `amd64`, `x64`, `x86` → x86_64
- `ARM64`, `arm64` → ARM64

### 5. **CUDA Support**

- Files with `cpu` → CPU-only
- Files with `cuda` → CUDA-enabled

---

## 📝 File Naming Conventions

For automatic detection to work, files must follow these patterns:

### Application Packages

```
ACloudViewer-{VERSION}-{PLATFORM}-{CUDA}-{ARCH}.{EXT}
```

**Examples:**
```
✅ ACloudViewer-3.9.3+d236e24-win-cpu-amd64.exe
✅ ACloudViewer-3.9.3+d236e24-mac-cpu-ARM64.dmg
✅ ACloudViewer-3.9.3+d236e24-ubuntu20.04-cpu-amd64.run
✅ ACloudViewer-3.9.3+d236e24-ubuntu20.04-cuda-amd64.run
```

### Python Wheel Packages

```
cloudviewer[-_]cpu?-{VERSION}-cp{PYMAJOR}{PYMINOR}-{ABI}-{PLATFORM}_{ARCH}.whl
```

**Examples:**
```
✅ cloudviewer-3.9.3+d236e24-cp310-cp310-win_amd64.whl
✅ cloudviewer_cpu-3.9.3+d236e24-cp311-cp311-win_amd64.whl
✅ cloudviewer-3.9.3+d236e24-cp312-cp312-macosx_14_0_arm64.whl
✅ cloudviewer-3.9.3+d236e24-cp38-cp38-manylinux_2_27_x86_64.whl
```

**Components:**
- `cloudviewer` or `cloudviewer_cpu` - CPU-only variant
- `cp310` - Python 3.10
- `win_amd64` - Windows x86_64
- `macosx_14_0_arm64` - macOS ARM64
- `manylinux_2_27_x86_64` - Linux x86_64 (universal)

---

## ✅ Supported Configurations

### Current Status (Auto-detected from [GitHub Releases](https://github.com/Asher-1/ACloudViewer/releases))

| Version | Python Versions | Ubuntu Versions (Apps) | Platforms |
|---------|----------------|------------------------|-----------|
| **Beta** (main-devel) | 3.10, 3.11, 3.12, 3.13 | 20.04, 22.04, 24.04 | Win, macOS, Linux |
| v3.9.3 | 3.8, 3.9, 3.10, 3.11, 3.12 | 18.04, 20.04, 22.04 | Win, macOS, Linux |
| v3.9.2 | 3.8, 3.9, 3.10, 3.11, 3.12 | 18.04, 20.04, 22.04 | Win, macOS, Linux |
| v3.9.1 | 3.8, 3.9, 3.10, 3.11, 3.12 | 18.04, 20.04,       | Win, macOS, Linux |
| v3.8.0 | 3.6, 3.7, 3.8 | (Legacy format) | Win, Linux |

**Notes:**
- Wheel packages (`.whl`) don't distinguish Ubuntu versions (manylinux is universal)
- Beta version doesn't support Ubuntu 18.04
- Python version ranges vary by release
- Some older releases may use different file naming conventions

---

## 🔧 Maintenance

### Adding Support for New File Patterns

If files are not being parsed (see warnings in script output), update the regex patterns in `scan_releases.py`:

```python
APP_NAME_PATTERN = re.compile(
    r"ACloudViewer-(?P<version>[\d.]+(?:[+-][\w.]+)?)-"
    r"(?P<platform>win|mac|ubuntu[\d.]+|linux)-"
    r"(?P<cuda>cpu|cuda[\d.]*)-"
    r"(?P<arch>amd64|ARM64|x64|x86)\.(?P<ext>exe|dmg|pkg|run|deb|AppImage)",
    re.IGNORECASE
)
```

### Handling Legacy Releases

Older releases (v3.9.2, v3.9.1) use date-based naming:
```
❌ ACloudViewer-3.9.2-2024-12-24-win-cpu-amd64.exe  # Not parsed yet
```

To support these, add additional regex patterns or normalize file names in releases.

### Testing Changes

```bash
# Test locally
python3 docs/automation/scripts/scan_releases.py

# Check output
cat docs/downloads_data.json

# Verify on website
cd docs && python3 -m http.server 8080
# Visit http://localhost:8080
```

---

## 🎨 Website Integration

The generated `downloads_data.json` is used by the website:

```javascript
// In script.js
fetch('downloads_data.json')
  .then(response => response.json())
  .then(data => {
    // Populate version selector
    renderVersionButtons(data.version_metadata);
    
    // Load download data
    loadDownloadData(data.download_data);
    
    // Initialize selector logic
    initializeDownloadSelector();
  });
```

---

## 📊 Data Structure

### Version Metadata

```json
{
  "value": "v3.9.3",
  "display_name": "v3.9.3",
  "python_versions": ["3.8", "3.9", "3.10", "3.11", "3.12"],
  "ubuntu_versions": ["ubuntu18.04", "ubuntu20.04", "ubuntu22.04"],
  "is_default": false
}
```

### Download Data

```json
{
  "v3.9.3": {
    "windows": {
      "app": {
        "cpu": {
          "amd64": {
            "url": "https://github.com/.../ACloudViewer-3.9.3-win-cpu-amd64.exe",
            "size": "223.2 MB"
          }
        }
      },
      "wheel": {
        "cpu": {
          "amd64": {
            "3.8": { "url": "...", "size": "..." },
            "3.9": { "url": "...", "size": "..." }
          }
        }
      }
    }
  }
}
```

---

## 🚨 Troubleshooting

### Issue: "Not available" for many combinations

**Cause:** Files are not being parsed correctly  
**Solution:**
1. Check console output for warnings
2. Verify file naming follows conventions
3. Update regex patterns if needed
4. Re-run scanner

### Issue: Wheel packages show Ubuntu version selector

**Cause:** Logic error in website code  
**Solution:** Wheel packages should NOT show Ubuntu selector (manylinux is universal)

### Issue: Wrong Python versions showing

**Cause:** Hardcoded version lists  
**Solution:** Remove hardcoded lists, use auto-generated metadata

---

## 📚 Related Files

- [Main Website](../../index.html) - Uses the generated data
- [Update Script](./scripts/scan_releases.py) - Generates the data
- [GitHub Workflow](../../../.github/workflows/update-website-downloads.yml) - Automates updates
- [Bilingual Guide](../BILINGUAL_GUIDE.md) - Website internationalization

---

## 🎉 Benefits

✅ **No Manual Maintenance** - Fully automated  
✅ **Always Accurate** - Based on actual releases  
✅ **Self-Documenting** - Discovers capabilities automatically  
✅ **Version Flexible** - Handles different Python/Ubuntu ranges  
✅ **Error Resistant** - Handles missing files gracefully  

---

**Last Updated:** 2026-01-10  
**Script Version:** 1.0.0  
**License:** MIT / GPL-2.0

