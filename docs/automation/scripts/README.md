# Release Scanner Script Documentation

## Overview

`scan_releases.py` is an automated script that scans GitHub Releases and generates `downloads_data.json` for the ACloudViewer website and documentation.

## Purpose

This script serves two main purposes:

1. **Website Download Selector**: Provides data for the download page dropdown selectors
2. **Documentation Version Selector**: Determines which versions appear in the documentation version switcher

## How It Works

```
GitHub Releases API
        ↓
Fetch all releases
        ↓
Parse asset filenames
        ↓
Detect platforms, Python versions, architectures
        ↓
Determine documentation availability
        ↓
Generate downloads_data.json
```

## Asset Naming Patterns

### Application Packages

```
ACloudViewer-{version}-{platform}-{cuda}-{arch}.{ext}

Examples:
- ACloudViewer-3.9.4-win-cpu-amd64.exe
- ACloudViewer-3.9.4-mac-cpu-ARM64.dmg
- ACloudViewer-3.9.4-ubuntu20.04-cpu-amd64.run
```

### Python Wheels

```
cloudviewer-{version}-cp{py_ver}-{abi}-{platform_arch}.whl

Examples:
- cloudviewer-3.9.4-cp310-cp310-win_amd64.whl
- cloudviewer-3.9.4-cp311-cp311-macosx_14_0_arm64.whl
- cloudviewer-3.9.4-cp312-cp312-manylinux_2_35_x86_64.whl
```

## Platform Detection

| Platform | Extensions | Example |
|----------|-----------|---------|
| **Windows** | `.exe` | `ACloudViewer-3.9.4-win-cpu-amd64.exe` |
| **macOS** | `.dmg`, `.pkg` | `ACloudViewer-3.9.4-mac-cpu-ARM64.dmg` |
| **Linux** | `.run`, `.deb`, `.AppImage` | `ACloudViewer-3.9.4-ubuntu20.04-cpu-amd64.run` |

## Python Wheel Mapping

### manylinux to Ubuntu Version

```python
manylinux_2_27 → ubuntu18.04  # CentOS 7 / RHEL 7
manylinux_2_31 → ubuntu20.04  # Ubuntu 20.04
manylinux_2_35 → ubuntu22.04  # Ubuntu 22.04
manylinux_2_39 → ubuntu24.04  # Ubuntu 24.04
```

## Documentation Version Filtering

### Version Availability Logic

```python
MIN_DOC_VERSION = (3, 9, 4)  # Documentation versioning started from v3.9.4

# Versions with documentation:
- main-devel (Beta)           → has_documentation: true (always)
- v3.9.4 and newer            → has_documentation: true
- v3.9.3 and older            → has_documentation: false (filtered out)
```

**Why v3.9.4?**
- Documentation versioning was implemented starting from v3.9.4
- Earlier versions don't have archived documentation on GitHub Pages
- Showing them would result in 404 errors

## Usage

### Run Locally

```bash
# Basic usage
cd /Users/asher/develop/code/github/ACloudViewer
python3 docs/automation/scripts/scan_releases.py

# With GitHub token (to avoid rate limits)
export GITHUB_TOKEN=your_token_here
python3 docs/automation/scripts/scan_releases.py
```

### Output

```bash
docs/downloads_data.json
```

### Expected Output

```
================================================================================
🚀 GitHub Releases Scanner
================================================================================
📡 Fetching releases from https://api.github.com/repos/Asher-1/ACloudViewer/releases...
✅ Found 17 releases

🔍 Analyzing releases...

📦 Processing v3.9.4 (41 assets)
  ✓ Platforms: linux, macos, windows
  ✓ Python: 3.10, 3.11, 3.12, 3.13
  ✓ Ubuntu: ubuntu20.04, ubuntu22.04, ubuntu24.04
  ✓ Parsed 41 / 41 assets

📦 Processing Beta (1 assets)
  ✓ Platforms: 
  ✓ Python: N/A
  ✓ Parsed 0 / 1 assets

================================================================================
✅ Successfully generated docs/downloads_data.json
📊 Summary:
   • Versions: 5
   • Total configurations: 215
================================================================================
```

## Configuration

### Change Minimum Documentation Version

```python
# In scan_releases.py, line ~372
MIN_DOC_VERSION = (3, 9, 5)  # Change to your desired version
```

### Change Number of Releases to Scan

```python
# In main(), line ~420
releases = fetch_releases(limit=10)  # Default: 5
```

### Add New Platform Pattern

```python
# In parse_app_asset(), add to APP_NAME_PATTERN
APP_NAME_PATTERN = re.compile(
    r"ACloudViewer-(?P<version>[\d.]+(?:[+-][\w.]+)?)-"
    r"(?P<platform>win|mac|ubuntu[\d.]+|linux|android)-"  # Add 'android'
    r"(?P<cuda>cpu|cuda[\d.]*)-"
    r"(?P<arch>amd64|ARM64|x64|x86)\.(?P<ext>exe|dmg|pkg|run|deb|AppImage|apk)",  # Add 'apk'
    re.IGNORECASE
)
```

## Generated Data Structure

### Version Metadata

Used by documentation version selector:

```json
{
  "value": "v3.9.4",
  "display_name": "v3.9.4",
  "python_versions": ["3.10", "3.11", "3.12", "3.13"],
  "ubuntu_versions": ["ubuntu20.04", "ubuntu22.04", "ubuntu24.04"],
  "has_documentation": true,
  "is_default": false
}
```

### Download Data

Used by website download page:

```json
{
  "v3.9.4": {
    "windows": {
      "app": {
        "cpu": {
          "amd64": {
            "url": "https://github.com/.../ACloudViewer-3.9.4-win-cpu-amd64.exe",
            "size": "223.2 MB"
          }
        }
      },
      "wheel": {
        "cpu": {
          "amd64": {
            "3.10": { "url": "...", "size": "..." },
            "3.11": { "url": "...", "size": "..." }
          }
        }
      }
    }
  }
}
```

## Integration with CI/CD

### Automatic Execution

The script is automatically run by:

1. **`.github/workflows/update-downloads.yml`**
   - Triggered after build workflows complete
   - Triggered when releases are published/deleted
   - Runs daily at 00:00 UTC
   - Can be manually triggered

2. **`.github/workflows/documentation.yml`**
   - Fetches `downloads_data.json` from `gh-pages` branch
   - Falls back to running `scan_releases.py` if not available

### Workflow Integration

```yaml
# In .github/workflows/update-downloads.yml
- name: Generate downloads data
  run: |
    python3 docs/automation/scripts/scan_releases.py
    
- name: Commit to gh-pages
  run: |
    git add docs/downloads_data.json
    git commit -m "chore: update downloads data"
    git push origin gh-pages
```

## Troubleshooting

### Issue: API Rate Limit Exceeded

**Error**: `API rate limit exceeded`

**Solution**:
```bash
# Set GitHub token
export GITHUB_TOKEN=your_personal_access_token
python3 docs/automation/scripts/scan_releases.py
```

**Note**: GitHub Actions automatically use `GITHUB_TOKEN`, so this only affects local testing.

### Issue: Asset Not Parsed

**Symptom**: Warning message `⚠️  Could not parse: filename.ext`

**Cause**: Filename doesn't match expected patterns

**Solution**:
1. Check the filename format
2. Update `APP_NAME_PATTERN` or `WHEEL_NAME_PATTERN` in the script
3. Re-run the script

### Issue: Version Not Showing in Selector

**Check**:
```bash
# Verify has_documentation flag
jq '.version_metadata[] | select(.value == "v3.9.3")' docs/downloads_data.json
```

**Expected**:
```json
{
  "value": "v3.9.3",
  "has_documentation": false  // Will be filtered out
}
```

**Solution**: Versions < v3.9.4 are intentionally filtered. Update `MIN_DOC_VERSION` if needed.

### Issue: SSL Certificate Error

**Error**: `SSL: CERTIFICATE_VERIFY_FAILED`

**Solution**:
```bash
# macOS
/Applications/Python\ 3.x/Install\ Certificates.command

# The script has built-in fallback using ssl._create_unverified_context()
```

## Testing

### Verify Output

```bash
# Check JSON is valid
python3 -m json.tool docs/downloads_data.json > /dev/null && echo "✅ Valid JSON"

# Count versions with documentation
jq '[.version_metadata[] | select(.has_documentation == true)] | length' docs/downloads_data.json

# List all platforms for a version
jq '.download_data["v3.9.4"] | keys' docs/downloads_data.json

# Check Python versions
jq '.version_metadata[] | select(.value == "v3.9.4") | .python_versions' docs/downloads_data.json
```

### Local Preview

```bash
# Start local server
cd docs
python3 -m http.server 8080

# Visit http://localhost:8080
# Check version selector and download links
```

## Best Practices

1. **Always test locally** before pushing changes
2. **Verify JSON validity** after modifications
3. **Check CI logs** after releases to ensure automation works
4. **Keep patterns up-to-date** with new release naming conventions
5. **Document any custom patterns** you add

## Dependencies

- **Python**: 3.7+
- **Standard Library Only**: No external dependencies required
  - `json` - JSON parsing
  - `re` - Regular expressions
  - `urllib` - HTTP requests
  - `ssl` - SSL context
  - `datetime` - Timestamps

## Security

- ✅ Uses GitHub-provided `GITHUB_TOKEN` in CI/CD
- ✅ Tokens auto-expire, no manual management needed
- ✅ Read-only API access (no write operations)
- ✅ SSL certificate verification (with fallback)

## Maintenance

### Regular Tasks

- [ ] Review asset naming patterns quarterly
- [ ] Update `MIN_DOC_VERSION` when documentation versioning changes
- [ ] Test after major releases
- [ ] Monitor CI/CD logs for parsing warnings

### When to Update

- ✅ **New platform added** → Update `APP_NAME_PATTERN`
- ✅ **New wheel format** → Update `WHEEL_NAME_PATTERN`
- ✅ **Documentation policy changes** → Update `MIN_DOC_VERSION`
- ✅ **Release naming changes** → Update regex patterns

---

**Last Updated**: February 2026  
**Maintainer**: ACloudViewer Team  
**Status**: ✅ Active and Production-Ready
