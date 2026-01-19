# Documentation Version Management

## Overview

ACloudViewer now supports multi-version documentation management, similar to Open3D's implementation. This allows users to access documentation for any released version of ACloudViewer.

## Features

- ✅ **Automatic Version Archiving**: When a release is published, documentation is automatically archived to a version-specific directory
- ✅ **Version Switcher UI**: Users can switch between different documentation versions using a dropdown selector
- ✅ **Preserved History**: All historical documentation versions are preserved on GitHub Pages
- ✅ **Latest Documentation**: The latest (main branch) documentation is always available at `/documentation/`

## URL Structure

### Latest Documentation (Development)
- **Main**: `https://asher-1.github.io/ACloudViewer/documentation/`
- **Python API**: `https://asher-1.github.io/ACloudViewer/documentation/python_api/`
- **C++ API**: `https://asher-1.github.io/ACloudViewer/documentation/cpp_api/`
- **Tutorials**: `https://asher-1.github.io/ACloudViewer/documentation/tutorial/`

### Version-Specific Documentation
- **Main**: `https://asher-1.github.io/ACloudViewer/documentation/v3.9.3/`
- **Python API**: `https://asher-1.github.io/ACloudViewer/documentation/v3.9.3/python_api/`
- **C++ API**: `https://asher-1.github.io/ACloudViewer/documentation/v3.9.3/cpp_api/`
- **Tutorials**: `https://asher-1.github.io/ACloudViewer/documentation/v3.9.3/tutorial/`

## How It Works

### 1. Release Event Trigger

When a GitHub release is published or edited, the `documentation.yml` workflow automatically:

1. **Detects Release**: Identifies the release tag (e.g., `v3.9.3`)
2. **Builds Documentation**: Builds documentation with `DEVELOPER_BUILD=OFF` for production
3. **Archives to Version Directory**: Deploys documentation to `/documentation/v3.9.3/`
4. **Preserves History**: Uses `keep_files: true` to preserve existing versions

### 2. Version Switcher

The version switcher (`docs/source/_static/version_switch.js`) provides:

- **Dynamic Version Loading**: Automatically loads available versions from `downloads_data.json`
- **Visual Selector**: Fixed dropdown in the top-right corner of documentation pages
- **URL Navigation**: Automatically updates URLs when switching versions
- **Current Version Detection**: Highlights the currently viewed version

### 3. Deployment Structure

```
gh-pages branch:
├── index.html (main website)
├── downloads_data.json
├── documentation/ (latest)
│   ├── index.html
│   ├── python_api/
│   ├── cpp_api/
│   └── tutorial/
└── documentation/
    ├── v3.9.3/ (version archive)
    │   ├── index.html
    │   ├── python_api/
    │   ├── cpp_api/
    │   └── tutorial/
    ├── v3.9.2/ (version archive)
    │   └── ...
    └── v3.8.0/ (version archive)
        └── ...
```

## Usage

### For Users

1. **Access Latest Documentation**: Visit `/documentation/` for the most recent documentation
2. **Switch Versions**: Use the version selector dropdown in the top-right corner
3. **Direct Links**: Use version-specific URLs for permanent links to specific versions

### For Maintainers

#### Publishing a New Release

1. **Create Release Tag**:
   ```bash
   git tag -a v3.9.4 -m "Release v3.9.4"
   git push origin v3.9.4
   ```

2. **Create GitHub Release**:
   - Go to https://github.com/Asher-1/ACloudViewer/releases/new
   - Select tag: `v3.9.4`
   - Fill in release notes
   - Click "Publish release"

3. **Automatic Documentation Archive**:
   - The `documentation.yml` workflow will automatically trigger
   - Documentation will be built and archived to `/documentation/v3.9.4/`
   - The version switcher will automatically include the new version

#### Manual Documentation Build

To manually trigger documentation build for a release:

1. Go to Actions → Documentation workflow
2. Click "Run workflow"
3. Set `developer_build` to `OFF` for release documentation
4. Click "Run workflow"

## Implementation Details

### Workflow Configuration

The `documentation.yml` workflow has been updated to:

- **Support Release Events**: Triggers on `release: types: [published, edited]`
- **Version Detection**: Extracts version from `github.event.release.tag_name`
- **Conditional Deployment**: 
  - Main branch → Deploy to `/documentation/` (latest)
  - Release event → Archive to `/documentation/v{version}/`
- **Preserve History**: Uses `keep_files: true` instead of `force_orphan: true`

### Version Switcher JavaScript

The version switcher (`docs/source/_static/version_switch.js`):

- Loads versions from `downloads_data.json`
- Creates a fixed dropdown selector
- Handles URL navigation between versions
- Automatically detects current version from URL

### Sphinx Integration

The version switcher is automatically included in all documentation pages via:

```python
# docs/source/conf.py
html_js_files = [
    'version_switch.js',
]
```

## Troubleshooting

### Version Not Appearing in Selector

1. **Check Release**: Ensure the release was published (not draft)
2. **Check Workflow**: Verify the documentation workflow completed successfully
3. **Check downloads_data.json**: The version should appear in `version_metadata`
4. **Clear Cache**: Browser cache may need to be cleared

### Documentation Not Archived

1. **Check Workflow Logs**: Review the `documentation.yml` workflow run
2. **Verify Release Event**: Ensure the release event triggered the workflow
3. **Check Permissions**: Verify GitHub Pages write permissions

### Version Switcher Not Showing

1. **Check JavaScript**: Ensure `version_switch.js` is in `_static/` directory
2. **Check Sphinx Config**: Verify `html_js_files` includes `version_switch.js`
3. **Check Browser Console**: Look for JavaScript errors

## Future Enhancements

- [ ] Version comparison view
- [ ] Automatic redirect from old URLs
- [ ] Version-specific search
- [ ] Download links for archived documentation

## References

- Open3D Documentation: https://www.open3d.org/docs/
- GitHub Actions Documentation: https://docs.github.com/en/actions
- Sphinx Documentation: https://www.sphinx-doc.org/

