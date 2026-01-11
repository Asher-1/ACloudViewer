# ACloudViewer Website Automation System - Complete Guide

## 🎯 Overview

> This is a **fully automated, zero-maintenance** website download link update system. When you publish a new version on GitHub, the website automatically updates download links without any manual intervention.

## ✨ Key Features

- ✅ **Fully Automated**: No manual website updates required
- ✅ **Real-time Sync**: Automatically triggered after release publication
- ✅ **Smart Detection**: Automatically distinguishes Beta and stable versions
- ✅ **Platform Recognition**: Automatically identifies Windows, macOS, Linux packages
- ✅ **Scheduled Checks**: Daily automatic version checks and synchronization
- ✅ **Zero Dependencies**: Uses Python standard library, no extra packages needed

## 📁 File Structure

```
ACloudViewer/
├── .github/
│   └── workflows/
│       └── update-website-downloads.yml    # GitHub Actions workflow
├── docs/
│   ├── automation/
│   │   └── scripts/
│   │       ├── update_download_links.py    # Auto-update script
│   │       ├── requirements.txt            # Python dependencies (optional)
│   │       └── README.md                   # Detailed script documentation
│   └── index.html                          # Website homepage (auto-updated)
└── docs/automation/README.md               # This document
```

## 🚀 Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  1. Developer publishes new version                         │
│     └─> GitHub Release (main-devel or v3.x.x)             │
│                                                             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  2. Automatically trigger GitHub Actions                    │
│     ├─> Listen for Release publish event                   │
│     ├─> Scheduled task (daily at UTC 0:00)                │
│     └─> Manual trigger (optional)                          │
│                                                             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  3. Python script executes                                  │
│     ├─> Call GitHub API to fetch Releases data            │
│     ├─> Identify Beta version (main-devel tag)            │
│     ├─> Identify stable versions (v3.9.3, v3.4.0, etc.)   │
│     ├─> Match platform-specific package files             │
│     └─> Generate new HTML content                         │
│                                                             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  4. Update and commit                                       │
│     ├─> Update docs/index.html file                        │
│     ├─> Git commit changes                                 │
│     └─> Auto-push to repository                           │
│                                                             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  5. GitHub Pages auto-deploys                              │
│     └─> Website update complete!                           │
│         https://asher-1.github.io/ACloudViewer/docs       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 🔧 System Components

### 1. GitHub Actions Workflow

**File**: `.github/workflows/update-website-downloads.yml`

**Trigger Conditions**:
- 📦 **Release Published**: When a new Release is published or edited
- ⏰ **Scheduled Task**: Daily at UTC 0:00 (8:00 AM Beijing Time)
- 🖱️ **Manual Trigger**: Manually run from GitHub Actions page

**Workflow Steps**:
```yaml
1. Checkout code
2. Install Python 3.11
3. Run update script
4. Check for changes
5. Auto-commit and push (if changes exist)
```

### 2. Python Update Script

**File**: `docs/automation/scripts/update_download_links.py`

**Core Functions**:

#### Version Identification

```python
# Beta version identification
def get_beta_release(releases):
    """
    Identification criteria:
    - Tag name = 'main-devel'
    - Marked as pre-release
    """
    
# Stable version identification  
def get_stable_releases(releases, limit=3):
    """
    Identification criteria:
    - Not a pre-release
    - Tag format: v3.9.3, v3.4.0, etc.
    - Exclude main-devel
    - Get latest 3 versions
    """
```

#### Platform Recognition

> Automatically recognizes packages for the following platforms:

| Platform | Match Pattern | Example Filename |
|----------|--------------|------------------|
| **Windows** | `*.exe` | `ACloudViewer-3.9.3+d236e24-win-cpu-amd64.exe` |
| **macOS** | `*.dmg`, `*.pkg` | `ACloudViewer-3.9.3+d236e24-mac-cpu-ARM64.dmg` |
| **Linux** | `*.run`, `*.deb`, `*.rpm`, `*.appimage` | `ACloudViewer-3.9.3+d236e24-ubuntu20.04-cpu-amd64.run` |
| **Ubuntu** | `*.deb` | `ACloudViewer-ubuntu-20.04.deb` |

#### HTML Generation

```python
# Generate Beta version section
def generate_beta_section(beta_release):
    """
    Generates:
    - Version name
    - Release date
    - Commit SHA
    - Download links for each platform
    """

# Generate stable version section
def generate_stable_section(stable_releases):
    """
    Generates:
    - Version switching tabs
    - Download links for each version
    - Supports up to 3 historical versions
    """
```

## 📋 Usage Guide

### For Developers: Publishing New Versions

#### Publishing Beta Version

```bash
# 1. Develop new features on main branch
git checkout main
git add .
git commit -m "feat: add new feature"
git push origin main

# 2. GitHub Actions will automatically:
#    - Build and publish to main-devel tag
#    - Trigger website update workflow
#    - Auto-update website download links
```

#### Publishing Stable Version

```bash
# 1. Create new release tag
git tag -a v3.10.0 -m "Release v3.10.0"
git push origin v3.10.0

# 2. Create Release on GitHub:
#    - Visit: https://github.com/Asher-1/ACloudViewer/releases/new
#    - Select tag: v3.10.0
#    - Fill in Release notes
#    - Upload compiled packages:
#      * Windows: *.exe
#      * macOS: *.dmg
#      * Linux: *.run
#    - Click "Publish release"

# 3. System automatically:
#    - Triggers update workflow
#    - Identifies new version
#    - Updates website download links
#    - Deploys to GitHub Pages
```

### For Maintainers: Monitoring and Management

#### View Automation Run Status

1. Visit Actions page: https://github.com/Asher-1/ACloudViewer/actions
2. Find "Update Website Download Links" workflow
3. Check recent run records

#### Manually Trigger Update

1. Visit Actions page
2. Select "Update Website Download Links"
3. Click "Run workflow"
4. Select branch (usually main)
5. Click "Run workflow" button

#### Local Testing

```bash
# 1. Enter project directory
cd /Users/asher/develop/code/github/ACloudViewer

# 2. Run update script
python3 docs/automation/scripts/update_download_links.py

# 3. View changes
git diff docs/index.html

# 4. Local preview
cd docs
python3 -m http.server 8080
# Visit http://localhost:8080
```

## 🛠️ Configuration and Customization

### Modify Number of Stable Versions Retrieved

Edit `docs/automation/scripts/update_download_links.py`:

```python
# Find this line and modify the number
stable_releases = get_stable_releases(releases, limit=5)  # Default is 3
```

### Add New Platform Recognition

Edit `docs/automation/scripts/update_download_links.py`, add to `PLATFORM_PATTERNS`:

```python
PLATFORM_PATTERNS = {
    'windows': {...},
    'macos': {...},
    'linux': {...},
    # Add new platform
    'android': {
        'patterns': [r'android.*\.(apk|aab)$', r'\.(apk|aab)$'],
        'display_name': 'Android'
    }
}
```

### Modify Scheduled Task Frequency

Edit `.github/workflows/update-website-downloads.yml`:

```yaml
schedule:
  # Run every 6 hours
  - cron: '0 */6 * * *'
  
  # Run every Monday
  - cron: '0 0 * * 1'
  
  # Run on 1st of each month
  - cron: '0 0 1 * *'
```

## 🔍 Troubleshooting

### Issue 1: Website Not Updated

**Possible Causes**:
- GitHub Actions run failed
- No suitable package files found
- Git commit permission issue

**Solutions**:
```bash
# 1. Check Actions run logs
Visit: https://github.com/Asher-1/ACloudViewer/actions

# 2. View failure reason
Click failed workflow run -> View detailed logs

# 3. Reproduce issue locally
python3 docs/automation/scripts/update_download_links.py
```

### Issue 2: Platform Download Link Not Found

**Possible Causes**:
- Platform package not uploaded in Release
- Filename doesn't match recognition rules

**Solutions**:
```bash
# 1. Check Release assets
Visit: https://github.com/Asher-1/ACloudViewer/releases/tag/main-devel

# 2. Confirm filename format
Windows: *.exe
macOS: *.dmg or *.pkg
Linux: *.run or *.deb or *.rpm

# 3. If filename is special, modify PLATFORM_PATTERNS
Edit docs/automation/scripts/update_download_links.py to add new match patterns
```

### Issue 3: API Rate Limit

**Error Message**: `API rate limit exceeded`

**Solutions**:
```yaml
# GitHub Actions automatically uses GITHUB_TOKEN
# For local testing, set token:
export GITHUB_TOKEN=your_personal_access_token
python3 docs/automation/scripts/update_download_links.py
```

### Issue 4: SSL Certificate Error

**Error Message**: `SSL: CERTIFICATE_VERIFY_FAILED`

**Solutions**:
```bash
# macOS
/Applications/Python\ 3.x/Install\ Certificates.command

# Or use built-in fallback mechanism in script (already implemented)
```

## 📊 Monitoring Metrics

### Success Indicators

- ✅ GitHub Actions run successful (green checkmark)
- ✅ Website displays latest version number
- ✅ Download links work normally
- ✅ Each platform has corresponding links

### Checklist

> After each new version release, verify:

```
□ Beta version number updated
□ Beta version Commit SHA correct
□ Beta version release date correct
□ Windows download link valid
□ macOS download link valid
□ Linux download link valid
□ Stable version tabs correct
□ Historical versions preserved (latest 3)
□ Download works normally when clicked
```

## 🔒 Security

### Token Security
- ✅ Uses GitHub-provided `GITHUB_TOKEN`
- ✅ Token auto-expires, no manual management needed
- ✅ Minimum permissions principle: only requests `contents: write`

### Code Review
- ✅ All changes produce Git commits
- ✅ Can review all changes through Git history
- ✅ Supports rollback to any version

### Prevent Infinite Loop
- ✅ Commit message includes `[skip ci]`
- ✅ Only commits when actual changes exist
- ✅ Avoids triggering chain reactions

## 📚 Related Resources

### Documentation
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [GitHub Releases API](https://docs.github.com/en/rest/releases)
- [GitHub Pages Configuration](https://docs.github.com/en/pages)

### Project Links
- [ACloudViewer Repository](https://github.com/Asher-1/ACloudViewer)
- [Releases Page](https://github.com/Asher-1/ACloudViewer/releases)
- [Actions Page](https://github.com/Asher-1/ACloudViewer/actions)
- [Website URL](https://asher-1.github.io/ACloudViewer/docs)

### Script Documentation
- [Detailed Script Documentation](scripts/README.md)
- [Python Script Source Code](scripts/update_download_links.py)
- [Workflow Configuration](.github/workflows/update-website-downloads.yml)

## 🎉 Summary

> Congratulations! You now have a **fully automated, zero-maintenance** website update system!

### Core Advantages

1. **Zero Manual Maintenance**: Everything auto-completes after version release
2. **Real-time Sync**: Website updates immediately after Release publication
3. **Smart Recognition**: Auto-identifies version types and platforms
4. **Stable and Reliable**: Scheduled checks ensure synchronization
5. **Easy to Extend**: Supports adding new platforms and custom configuration

### Next Steps

- 📦 Publish new version to test system
- 🔍 Monitor first automatic update
- 📝 Adjust configuration as needed
- 🎯 Enjoy the convenience of automation!

---

> **Last Updated**: 2026-01-10  
> **Author**: ACloudViewer Team  
> **Version**: 1.0.0  
> **Maintenance**: Automated by GitHub Actions ⚡️
