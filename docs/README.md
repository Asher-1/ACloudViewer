# ACloudViewer 网站文档

这个目录包含了 ACloudViewer 的官方网站和相关文档。

🌐 **在线访问**: https://asher-1.github.io/ACloudViewer/

## 📁 目录结构

```
docs/
├── index.html              # 网站主页
├── styles.css              # 网站样式
├── script.js               # 网站脚本
├── .nojekyll              # GitHub Pages 配置
├── 404.html               # 404 错误页面
├── robots.txt             # 搜索引擎配置
├── sitemap.xml            # 网站地图
│
├── images/                # 图片资源
│   ├── ACloudViewer_logo_horizontal.png
│   ├── ACloudViewerMainUI.png
│   ├── SemanticAnnotation.png
│   └── ...
│
├── gifs/                  # 动画资源
│   ├── visualizer_predictions.gif
│   └── ...
│
├── automation/            # 🤖 自动化系统
│   ├── README.md          # 自动化完整指南
│   ├── SUMMARY.md         # 自动化系统总结
│   └── scripts/           # 自动化脚本
│       ├── update_download_links.py  # 下载链接更新脚本
│       ├── requirements.txt          # Python 依赖
│       └── README.md                 # 脚本文档
│
├── guides/                # 📚 用户指南
│   ├── QUICKSTART.md      # 快速开始
│   ├── cloudviewer-dependency.md  # 依赖说明
│   └── building/          # 编译指南
│       ├── compiling-cloudviewer-linux.md
│       ├── compiling-cloudviewer-macos.md
│       └── compiling-cloudviewer-windows.md
│
└── maintenance/           # 🔧 维护文档
    ├── WEBSITE_GUIDE.md   # 网站维护指南
    ├── DEPLOYMENT.md      # 部署文档
    ├── DOWNLOAD_LINKS.md  # 下载链接管理
    ├── GALLERY_UPDATE.md  # 图库更新日志
    └── GALLERY_ANNOTATION_UPDATE.md  # 图库标注更新日志
```

## 🚀 快速开始

### 本地预览网站

```bash
cd doc
python3 -m http.server 8080
```

然后访问 http://localhost:8080

### 运行自动化更新

```bash
cd /Users/asher/develop/code/github/ACloudViewer
python3 docs/automation/scripts/update_download_links.py
```

## 📖 文档导航

### 对于用户

- **[快速开始](guides/QUICKSTART.md)** - 快速上手 ACloudViewer
- **[编译指南](guides/building/)** - 从源码编译
- **[依赖说明](guides/cloudviewer-dependency.md)** - 了解项目依赖

### 对于开发者

- **[自动化系统](automation/README.md)** - 了解网站自动化更新系统
- **[脚本文档](automation/scripts/README.md)** - 自动化脚本详细说明

### 对于维护者

- **[网站维护](maintenance/WEBSITE_GUIDE.md)** - 网站管理和维护
- **[部署指南](maintenance/DEPLOYMENT.md)** - 网站部署说明
- **[下载链接管理](maintenance/DOWNLOAD_LINKS.md)** - 管理下载链接

## 🤖 自动化系统

本网站采用**完全自动化**的更新系统：

- ✅ **自动触发**：Release 发布时自动更新
- ✅ **定时检查**：每天自动检查新版本
- ✅ **智能识别**：自动识别 Beta 和稳定版本
- ✅ **零维护**：无需人工干预

详情请查看 [自动化系统文档](automation/README.md)

## 🔧 维护

### 更新网站内容

1. 编辑 `index.html`、`styles.css` 或 `script.js`
2. 提交并推送到 GitHub
3. GitHub Pages 会自动部署

### 添加新图片

1. 将图片放到 `images/` 或 `gifs/` 目录
2. 在 HTML 中使用相对路径引用：`images/your-image.png`
3. 提交并推送

### 更新自动化脚本

1. 编辑 `automation/scripts/update_download_links.py`
2. 本地测试：`python3 docs/automation/scripts/update_download_links.py`
3. 提交并推送

## 📝 贡献

欢迎贡献！请参考以下指南：

- **网站改进**：编辑 HTML/CSS/JS 文件
- **文档更新**：编辑 `guides/` 或 `maintenance/` 中的 Markdown 文件
- **自动化优化**：改进 `automation/scripts/` 中的脚本

提交 PR 前请：
1. 本地测试所有更改
2. 确保链接正确
3. 验证自动化脚本运行正常

## 🔗 相关链接

- **GitHub 仓库**: https://github.com/Asher-1/ACloudViewer
- **Releases**: https://github.com/Asher-1/ACloudViewer/releases
- **Issues**: https://github.com/Asher-1/ACloudViewer/issues
- **Actions**: https://github.com/Asher-1/ACloudViewer/actions

## 📄 许可证

本文档遵循 ACloudViewer 项目的许可证。

---

**维护**: ACloudViewer Team  
**最后更新**: 2026-01-10  
**自动化**: ✅ Fully Automated
