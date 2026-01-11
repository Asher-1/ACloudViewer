# ACloudViewer Website Documentation

> This directory contains the official website and related documentation for ACloudViewer.

> 🌐 **Live Site**: https://asher-1.github.io/ACloudViewer/docs

## 📁 Directory Structure

```
docs/
├── index.html              # Main homepage
├── styles.css              # Website styles
├── script.js               # Website scripts
├── .nojekyll              # GitHub Pages config
├── 404.html               # 404 error page
├── robots.txt             # Search engine config
├── sitemap.xml            # Site map
│
├── images/                # Image assets
│   ├── ACloudViewer_logo_horizontal.png
│   ├── Annotaion.png
│   ├── SemanticAnnotation.png
│   └── ...
│
├── gifs/                  # Animation assets
│   ├── visualizer_predictions.gif
│   └── ...
│
├── automation/            # 🤖 Automation system
│   ├── README.md          # Complete automation guide
│   ├── SUMMARY.md         # Automation system summary
│   └── scripts/           # Automation scripts
│       ├── update_download_links.py  # Download link updater
│       ├── requirements.txt          # Python dependencies
│       └── README.md                 # Script documentation
│
├── guides/                # 📚 User guides
│   ├── QUICKSTART.md      # Quick start guide
│   ├── DOCUMENTATION_SETUP.md  # 📚 API documentation setup guide
│   ├── cloudviewer-dependency.md  # Dependency documentation
│   └── building/          # Build guides
│       ├── compiling-cloudviewer-linux.md
│       ├── compiling-cloudviewer-macos.md
│       └── compiling-cloudviewer-windows.md
│
├── maintenance/           # 🔧 Maintenance docs
│   ├── WEBSITE_GUIDE.md   # Website maintenance guide
│   ├── DEPLOYMENT.md      # Deployment documentation
│   ├── DOWNLOAD_LINKS.md  # Download link management
│   ├── GALLERY_UPDATE.md  # Gallery update log
│   └── GALLERY_ANNOTATION_UPDATE.md  # Gallery annotation update log
│
├── build_docs.sh          # 🚀 Documentation build script
├── Makefile              # Documentation build commands (generated)
├── Doxyfile              # Doxygen configuration
├── source/               # Sphinx documentation source (generated)
└── html/                 # Generated API documentation (Sphinx output)
```

## 🚀 Quick Start

### Preview Website Locally

```bash
cd docs
python3 -m http.server 8080
```

> **Tip**: Then visit http://localhost:8080 to preview the website

### 📚 Setup API Documentation System (NEW!)

**Quick Start (Recommended - Use cloudViewer Environment)**:

```bash
# Step 1: Activate cloudViewer environment
conda activate cloudViewer
# or if using venv:
# source /path/to/cloudViewer/bin/activate

# Step 2: Navigate to docs
cd docs

# Step 3: Install documentation dependencies
pip install -r requirements.txt

# Step 4: Install Doxygen
brew install doxygen graphviz  # macOS
# or: sudo apt-get install doxygen graphviz  # Linux

# Step 5: Build documentation
./build_docs.sh

# Step 6: Preview documentation
python3 -m http.server 8000 --directory html
```

> **📖 Documentation**:
> - **Quick Start**: [DOCUMENTATION_QUICK_START.md](DOCUMENTATION_QUICK_START.md) - 5-minute guide
> - **Complete Guide**: [guides/DOCUMENTATION_SETUP.md](guides/DOCUMENTATION_SETUP.md) - Full instructions

### Run Automation Update

```bash
cd /Users/asher/develop/code/github/ACloudViewer
python3 docs/automation/scripts/update_download_links.py
```

> **Note**: This script automatically fetches the latest version info from GitHub Releases and updates the website

## 📖 Documentation Navigation

### For Users

> If you're a user of ACloudViewer, start with these guides:

- **[Quick Start](guides/QUICKSTART.md)** - Get started with ACloudViewer quickly
- **[API Documentation Setup](guides/DOCUMENTATION_SETUP.md)** - 📚 Set up Sphinx documentation system
- **[Build Guide](guides/building/)** - Compile from source code
- **[Dependencies](guides/cloudviewer-dependency.md)** - Understand project dependencies

### For Developers

> If you want to understand or improve the automation system:

- **[Automation System](automation/README.md)** - Learn about the automated website update system
- **[Script Documentation](automation/scripts/README.md)** - Detailed script documentation

### For Maintainers

> If you're responsible for website maintenance and deployment:

- **[Website Maintenance](maintenance/WEBSITE_GUIDE.md)** - Website management and maintenance
- **[Deployment Guide](maintenance/DEPLOYMENT.md)** - Website deployment instructions
- **[Download Link Management](maintenance/DOWNLOAD_LINKS.md)** - Manage download links

## 📚 API Documentation System (NEW!)

> ACloudViewer now supports **automatic API documentation generation**, similar to Open3D:
>
> - ✅ **Sphinx-based**: Industry-standard documentation tool
> - ✅ **Auto-generation**: From C++ (Doxygen) and Python code
> - ✅ **Multi-version**: Separate docs for each release
> - ✅ **Beautiful UI**: Read the Docs theme
> - ✅ **Searchable**: Full-text search support
> - ✅ **CI/CD Ready**: GitHub Actions integration

See [Documentation Setup Guide](guides/DOCUMENTATION_SETUP.md) for complete instructions

### Quick Setup

```bash
cd docs
./build_docs.sh  # Build documentation (includes setup and build)
```

## 🤖 Automation System

> This website uses a **fully automated** update system:
> 
> - ✅ **Auto-triggered**: Updates automatically when releases are published
> - ✅ **Scheduled checks**: Daily automatic version checks
> - ✅ **Smart detection**: Automatically identifies Beta and stable versions
> - ✅ **Zero maintenance**: No manual intervention required

See [Automation System Documentation](automation/README.md) for details

## 🔧 Maintenance

### Update Website Content

> Updating the website is a three-step process:

1. Edit `index.html`, `styles.css`, or `script.js`
2. Commit and push to GitHub
3. GitHub Pages will deploy automatically

### Add New Images

> Steps to add image assets:

1. Place images in `images/` or `gifs/` directory
2. Reference with relative path in HTML: `images/your-image.png`
3. Commit and push

### Update Automation Scripts

> When modifying automation scripts:

1. Edit `automation/scripts/update_download_links.py`
2. Test locally: `python3 docs/automation/scripts/update_download_links.py`
3. Commit and push after verification

## 📝 Contributing

> Contributions are welcome! Follow these guidelines:

- **Website improvements**: Edit HTML/CSS/JS files
- **Documentation updates**: Edit Markdown files in `guides/` or `maintenance/`
- **Automation enhancements**: Improve scripts in `automation/scripts/`

> **Before submitting a PR, please**:
> 1. Test all changes locally
> 2. Ensure all links are correct
> 3. Verify automation scripts run properly

## 🔗 Related Links

> Important project links:

- **GitHub Repository**: https://github.com/Asher-1/ACloudViewer
- **Releases**: https://github.com/Asher-1/ACloudViewer/releases
- **Issues**: https://github.com/Asher-1/ACloudViewer/issues
- **Actions**: https://github.com/Asher-1/ACloudViewer/actions

## 📄 License

> This documentation follows the ACloudViewer project license.

---

> **Maintained by**: ACloudViewer Team  
> **Last Updated**: 2026-01-10  
> **Automation**: ✅ Fully Automated
