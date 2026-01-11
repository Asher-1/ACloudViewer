# Download Data Architecture

> **Updated**: 2026-01-11  
> **Status**: Simplified & Optimized

## 🏗️ Architecture Overview

```
GitHub Release
     ↓
GitHub Actions (on release/schedule)
     ↓
scan_releases.py
     ↓
downloads_data.json (generated, in .gitignore)
     ↓
GitHub Pages (served with website)
     ↓
Browser (loads downloads_data.json)
     ↓
Download Selector UI
```

## ✨ Key Features

1. **No Git Pollution**: `downloads_data.json` is in `.gitignore`
2. **Auto-Generated**: GitHub Actions generates it automatically
3. **Simple Client**: Browser just loads the JSON file
4. **No API Calls**: No GitHub API rate limits
5. **Fast Loading**: Pre-processed data, instant display

## 📁 Files

### Generated File (Not in Git)

```
docs/downloads_data.json
```

- ❌ **Not committed to git** (in `.gitignore`)
- ✅ **Auto-generated** by GitHub Actions
- ✅ **Deployed** with GitHub Pages
- ✅ **Loaded** by browser

### Processing Script

```
docs/automation/scripts/scan_releases.py
```

- Fetches releases from GitHub API
- Parses asset filenames
- Generates `downloads_data.json`
- Runs in GitHub Actions

### Website Script

```
docs/script.js
```

- Loads `downloads_data.json`
- Populates download selector
- No client-side parsing needed

## 🔄 Data Flow

### 1. GitHub Actions Trigger

Runs when:
- New release is published
- Release is edited
- Daily schedule (00:00 UTC)
- Manual workflow dispatch

### 2. Generate Data

```python
python3 docs/automation/scripts/scan_releases.py
```

Creates `docs/downloads_data.json` with:
- Version metadata
- Download links
- Python version support
- Ubuntu version support
- CUDA availability

### 3. GitHub Pages Deployment

GitHub Pages automatically serves:
- `docs/index.html`
- `docs/script.js`
- `docs/downloads_data.json` ← Included even though in .gitignore!

### 4. Browser Loads Data

```javascript
const response = await fetch('downloads_data.json');
const data = await response.json();
downloadsData = data.download_data;
versionMetadata = data.version_metadata;
```

## 🎯 Advantages

### Before (Client-side API calls)

❌ GitHub API rate limits  
❌ Slow initial load  
❌ Complex client-side parsing  
❌ Browser compatibility issues  
❌ Network dependency

### After (Pre-generated JSON)

✅ No rate limits  
✅ Instant load  
✅ Simple client code  
✅ Better compatibility  
✅ Works offline (cached)

## 🔍 Data Structure

### downloads_data.json

```json
{
  "generated_at": "2026-01-11T10:00:00Z",
  "version_metadata": [
    {
      "value": "main-devel",
      "display_name": "Beta",
      "python_versions": ["3.10", "3.11", "3.12", "3.13"],
      "ubuntu_versions": ["ubuntu20.04", "ubuntu22.04", "ubuntu24.04"],
      "is_default": true
    }
  ],
  "download_data": {
    "main-devel": {
      "windows": { ... },
      "macos": { ... },
      "linux": {
        "ubuntu20.04": { ... },
        "ubuntu22.04": { ... },
        "ubuntu24.04": { ... },
        "wheel": { ... }
      }
    }
  }
}
```

## 🚀 Deployment

### Automatic (GitHub Actions)

```yaml
# .github/workflows/update-website-downloads.yml

- name: Scan releases and generate download data
  run: python3 docs/automation/scripts/scan_releases.py

# File is generated but NOT committed (in .gitignore)
# GitHub Pages will serve it automatically
```

### Manual (Local Testing)

```bash
# 1. Activate environment
conda activate cloudViewer

# 2. Generate data
cd docs
python3 automation/scripts/scan_releases.py

# 3. Verify
ls -lh downloads_data.json
cat downloads_data.json | jq '.version_metadata[].display_name'

# 4. Test locally
python3 -m http.server 8080
# Visit http://localhost:8080
```

## 🧪 Testing

### Verify Data Generation

```bash
python3 docs/automation/scripts/scan_releases.py

# Check output
ls -lh docs/downloads_data.json

# Verify structure
cat docs/downloads_data.json | jq '.version_metadata | length'
cat docs/downloads_data.json | jq '.version_metadata[].display_name'
```

### Test Website

```bash
cd docs
python3 -m http.server 8080
```

Open `http://localhost:8080` and check:
- ✅ Versions load correctly
- ✅ Python versions display
- ✅ Ubuntu versions display (for Linux)
- ✅ Download links work
- ✅ No "Not available" errors

### Check Browser Console

Expected output:
```
📡 Loading downloads data from downloads_data.json...
📥 Response status: 200
✅ Data loaded successfully!
   Generated at: 2026-01-11T10:00:00.000Z
   Versions: 5
📊 Version metadata: (5) [{…}, {…}, {…}, {…}, {…}]
```

## ⚙️ Configuration

### .gitignore

```gitignore
# Documentation auto-generated files
docs/downloads_data.json
```

**Why**: 
- Prevents git history pollution
- File is auto-generated
- GitHub Pages serves it anyway

### GitHub Actions Permissions

```yaml
permissions:
  contents: write  # Not actually needed since we don't commit
```

Can be changed to:
```yaml
permissions:
  contents: read
```

## 🐛 Troubleshooting

### Problem: downloads_data.json not found (404)

**Cause**: File not generated

**Solution**:
```bash
python3 docs/automation/scripts/scan_releases.py
```

### Problem: Python versions not showing

**Cause**: Data structure issue

**Solution**: Check console for errors, verify JSON structure

### Problem: Ubuntu versions missing for old releases

**Cause**: Old releases may not have Ubuntu-specific builds

**Expected**: This is normal, selector adapts automatically

## 📊 Monitoring

### GitHub Actions

Check workflow runs:
```
https://github.com/Asher-1/ACloudViewer/actions/workflows/update-website-downloads.yml
```

### File Status

```bash
# Check if file exists
curl -I https://asher-1.github.io/ACloudViewer/downloads_data.json

# View content
curl https://asher-1.github.io/ACloudViewer/downloads_data.json | jq '.generated_at'
```

## 🔒 Security

- No secrets exposed in JSON
- Public data only (release information)
- No user data collected
- Static file, no backend

## 📈 Performance

- **File Size**: ~35 KB (compressed: ~5 KB)
- **Load Time**: < 100ms
- **Parse Time**: < 10ms
- **Total**: Very fast!

## 🎓 Best Practices

1. **Always use .gitignore** for generated files
2. **Keep client code simple** - let server-side handle complexity
3. **Cache appropriately** - GitHub Pages handles this
4. **Monitor workflow** - check Actions tab regularly
5. **Test locally** - generate and test before pushing

## 🔗 Related Files

- `.gitignore` - Excludes downloads_data.json
- `.github/workflows/update-website-downloads.yml` - Generates file
- `docs/automation/scripts/scan_releases.py` - Processing logic
- `docs/script.js` - Client-side loading

---

**Summary**: Simple, fast, and no git pollution! 🎉
