# Automated Download Links Update System

## Overview

> This automated system uses GitHub Actions and Python scripts to automatically fetch the latest version information from GitHub Releases and update download links on the website. **Completely maintenance-free**.

## Workflow

```
GitHub Release Published
        ↓
Trigger GitHub Action
        ↓
Fetch Releases API Data
        ↓
Parse Version and Download Links
        ↓
Update docs/index.html
        ↓
Auto-commit and Push
        ↓
GitHub Pages Auto-deploy
```

## Trigger Conditions

> The automation workflow is triggered in the following scenarios:

1. **New Version Release**: Automatically triggered when a new Release is published or edited
2. **Scheduled Task**: Daily automatic check at UTC 0:00
3. **Manual Trigger**: Can be manually run from GitHub Actions page

## Version Identification Rules

### Beta Version (Development)
- **Tag Name**: `main-devel`
- **Characteristic**: Marked as pre-release
- **Purpose**: Contains latest features and experimental functionality
- **Auto-fetch**: First 7 characters of Commit SHA, release date

### Stable Version
- **Tag Format**: `v3.9.3`, `v3.4.0`, etc.
- **Characteristic**: Not pre-release
- **Fetch Count**: Latest 3 stable versions
- **Exclude**: `main-devel` tag

## Platform Recognition Rules

> The script automatically identifies installation packages for the following platforms:

| Platform | File Extensions/Patterns | Example |
|----------|-------------------------|---------|
| Windows | `.exe` | `ACloudViewer-windows-x64.exe` |
| macOS | `.dmg`, `.pkg` | `ACloudViewer-macos.dmg` |
| Linux | `.run`, `.deb`, `.rpm`, `.appimage` | `ACloudViewer-linux.run` |
| Ubuntu/Debian | `.deb` | `ACloudViewer-ubuntu-20.04.deb` |

## File Structure

```
.github/workflows/
└── update-website-downloads.yml    # GitHub Actions workflow configuration

docs/automation/scripts/
├── README.md                       # This document
├── update_download_links.py        # Update script
└── requirements.txt                # Python dependencies

docs/
└── index.html                      # Website homepage (auto-updated)
```

## Usage

### 1. Automatic Execution (Recommended)

> No action required, the system will automatically:
- Monitor Release publish events
- Daily scheduled check for updates
- Auto-commit changes to repository

### 2. Local Testing

```bash
# Install dependencies
pip install requests jinja2

# Set GitHub Token (optional, to avoid API limits)
export GITHUB_TOKEN=your_github_token

# Run script
cd /Users/asher/develop/code/github/ACloudViewer
python3 docs/automation/scripts/update_download_links.py
```

### 3. Manual Trigger GitHub Action

> Steps to manually trigger:

1. Visit GitHub repository
2. Click **Actions** tab
3. Select **Update Website Download Links**
4. Click **Run workflow**
5. Select branch and click **Run workflow**

## Script Function Details

### `update_download_links.py`

#### Main Functions

1. **Fetch Releases Data**
   ```python
   def fetch_releases() -> List[Dict]:
       """Fetch all releases from GitHub API"""
   ```

2. **Identify Beta Version**
   ```python
   def get_beta_release(releases: List[Dict]) -> Optional[Dict]:
       """Get Beta version (main-devel tag)"""
   ```

3. **Identify Stable Versions**
   ```python
   def get_stable_releases(releases: List[Dict], limit: int = 3) -> List[Dict]:
       """Get stable versions (non-pre-release, exclude main-devel)"""
   ```

4. **Platform Matching**
   ```python
   def find_asset_for_platform(assets: List[Dict], platform: str) -> Optional[Dict]:
       """Find matching asset for specified platform"""
   ```

5. **Generate HTML**
   ```python
   def generate_download_section(beta_release, stable_releases) -> str:
       """Generate download section HTML"""
   ```

6. **Update File**
   ```python
   def update_html_file(beta_release, stable_releases):
       """Update HTML file"""
   ```

## Configuration

### GitHub Actions Configuration

> File: `.github/workflows/update-website-downloads.yml`

```yaml
# Trigger conditions
on:
  release:
    types: [published, edited]  # On release publish
  workflow_dispatch:             # Manual trigger
  schedule:
    - cron: '0 0 * * *'          # Daily scheduled

# Permissions
permissions:
  contents: write  # Write permission needed for commits
```

### Environment Variables

- `GITHUB_TOKEN`: Automatically provided, used for GitHub API access and code commits

## HTML Update Mechanism

### Updated HTML Sections

1. **Version Tabs**
   ```html
   <div class="version-tabs">
       <button class="version-tab active" data-version="beta">Beta Development</button>
       <button class="version-tab" data-version="3.9.3">v3.9.3</button>
       ...
   </div>
   ```

2. **Download Content Section**
   ```html
   <div class="version-sections">
       <!-- Beta Version -->
       <div class="version-content active" id="version-beta">
           ...
       </div>
       <!-- Stable Versions -->
       <div class="version-content" id="version-3.9.3">
           ...
       </div>
   </div>
   ```

### Regular Expression Matching

> The script uses regex to precisely locate HTML sections to update:

```python
# Version tabs section
version_tabs_pattern = r'(<div class="version-tabs">\s*)(.*?)(\s*</div>)'

# Download content section
version_sections_pattern = r'(<div class="version-sections">\s*)(.*?)(\s*</div>\s*</div>\s*</section>)'
```

## Troubleshooting

### 1. Script Execution Failed

> **Issue**: Python script error

**Check**:
```bash
# View GitHub Actions logs
# GitHub repo -> Actions -> select failed workflow run

# Local testing
python3 docs/automation/scripts/update_download_links.py
```

### 2. Platform File Not Found

> **Issue**: Download link missing for a platform

**Cause**: Release doesn't have file uploaded for that platform, or filename doesn't match recognition rules

**Solution**: 
- Check file names in Release assets
- Update `PLATFORM_PATTERNS` dictionary to add new matching patterns

### 3. API Rate Limit

> **Issue**: GitHub API returns 403 error

**Cause**: Unauthenticated API requests have rate limits (60/hour)

**Solution**: GitHub Actions automatically use `GITHUB_TOKEN`, usually won't encounter this issue

### 4. Commit Failed

> **Issue**: Git push failed

**Cause**: Insufficient permissions or branch protection rules

**Solution**: 
- Check workflow has `contents: write` permission
- Check branch protection rules allow GitHub Actions to push

## Custom Configuration

### Modify Number of Stable Versions Fetched

> Edit `docs/automation/scripts/update_download_links.py`:

```python
# Change 3 to your desired number
stable_releases = get_stable_releases(releases, limit=5)
```

### Add New Platform Recognition

> Edit `PLATFORM_PATTERNS`:

```python
PLATFORM_PATTERNS = {
    'windows': {...},
    'macos': {...},
    'linux': {...},
    'new_platform': {
        'patterns': [r'pattern1', r'pattern2'],
        'display_name': 'Display Name'
    }
}
```

### Modify Scheduled Task Frequency

> Edit `.github/workflows/update-website-downloads.yml`:

```yaml
schedule:
  - cron: '0 */6 * * *'  # Run every 6 hours
  - cron: '0 0 * * 1'    # Run every Monday
```

## Security

1. **Token Security**: Uses GitHub-provided `GITHUB_TOKEN`, auto-expires
2. **Minimal Permissions**: Only requests necessary `contents: write` permission
3. **Code Review**: All changes create Git commits, can be reviewed and rolled back
4. **Skip CI**: Commit message includes `[skip ci]` to avoid infinite loop triggers

## Maintenance Guide

### Monthly Checks
- [ ] Review GitHub Actions run logs
- [ ] Verify website download links are valid
- [ ] Check if new platforms need support

### On Version Release
- [ ] Confirm Release assets are uploaded
- [ ] Verify file names match recognition rules
- [ ] Check if website auto-updated
- [ ] Test download links are valid

### System Updates
- [ ] Regularly update Python dependencies
- [ ] Upgrade GitHub Actions versions
- [ ] Check GitHub API changes

## Contributing

> If you find bugs or have improvement suggestions:

1. Submit Issue: https://github.com/Asher-1/ACloudViewer/issues
2. Submit Pull Request
3. Describe problem and solution in Issue

## License

> This script follows the ACloudViewer project license.

---

**Last Updated**: 2026-01-10  
**Author**: ACloudViewer Team  
**Maintenance**: Automated by GitHub Actions
