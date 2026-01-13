# GitHub Actions Workflows

## Automation Workflows

### 📦 Update Website Download Links

**File**: `update-website-downloads.yml`

**Purpose**: Automatically fetch latest version information from GitHub Releases and update website download links

**Trigger Conditions**:
- 🚀 On release published or edited
- ⏰ Scheduled daily at 00:00 UTC
- 🖱️ Manual workflow dispatch

**Workflow Steps**:
1. Checkout repository
2. Setup Python 3.11
3. Run `scripts/update_download_links.py`
4. Check for changes
5. Auto-commit and push (if changes detected)

**View Run Status**:
https://github.com/Asher-1/ACloudViewer/actions/workflows/update-website-downloads.yml

**Manual Run**:
1. Visit [Actions](https://github.com/Asher-1/ACloudViewer/actions)
2. Select "Update Website Download Links"
3. Click "Run workflow"

**Related Documentation**:
- [Complete Automation Guide](../../docs/automation/README.md)
- [Script Documentation](../../docs/automation/scripts/README.md)
- [Documentation Index](../../docs/README.md)

---

**Maintained by**: GitHub Actions (Automated)
