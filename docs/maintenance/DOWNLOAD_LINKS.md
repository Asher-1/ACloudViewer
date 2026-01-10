# 下载链接管理指南

本文档说明如何管理网站上的下载链接。

## 📁 下载链接位置

所有下载链接都在 `doc/index.html` 文件的下载区域。

## 🔗 链接格式

### Beta 版本链接格式：
```
https://github.com/Asher-1/ACloudViewer/releases/download/v3.9.4-Beta/文件名
```

### 稳定版本链接格式：
```
https://github.com/Asher-1/ACloudViewer/releases/download/v3.9.3/文件名
```

## 📦 标准文件命名规范

为了让下载链接正常工作，建议在 GitHub Releases 中使用以下文件命名：

### Windows 文件
- `ACloudViewer-vX.X.X-Win64-setup.exe` - 安装程序
- `ACloudViewer-vX.X.X-Win64-portable.zip` - 便携版（可选）

### Linux 文件
- `ACloudViewer-vX.X.X-Linux-x86_64.AppImage` - AppImage 格式
- `ACloudViewer-vX.X.X-Linux-x86_64.tar.gz` - 压缩包格式（可选）

### macOS 文件
- `ACloudViewer-vX.X.X-macOS-x86_64.dmg` - Intel Mac
- `ACloudViewer-vX.X.X-macOS-arm64.dmg` - Apple Silicon (M1/M2/M3/M4)
- `ACloudViewer-vX.X.X-macOS.dmg` - 通用版本（如果是Universal Binary）

## 🔄 发布新版本时更新链接

### 步骤 1：发布 GitHub Release

1. 在 GitHub 创建新的 Release（如 `v3.9.5-Beta` 或 `v3.9.4`）
2. 按照上述命名规范上传文件
3. 记下完整的下载链接

### 步骤 2：更新网站

编辑 `doc/index.html`：

#### A. 更新 Beta 版本

找到 Beta 版本区域（大约在第166行）：

```html
<div class="version-header">
    <span class="version-badge beta">🚀 Beta 版本: v3.9.4-Beta</span>
    <span class="release-date">发布日期: 2025年2月12日</span>
</div>
```

更新版本号和日期。

然后更新下载链接：

```html
<a href="https://github.com/Asher-1/ACloudViewer/releases/download/v新版本号/新文件名" 
   class="btn btn-download-small">
```

#### B. 添加新的稳定版本

1. 在版本标签区域添加新按钮：

```html
<div class="version-tabs">
    <button class="version-tab active" data-version="3.9.4">v3.9.4</button>
    <button class="version-tab" data-version="3.9.3">v3.9.3</button>
    ...
</div>
```

2. 添加对应的下载区域：

```html
<!-- v3.9.4 Downloads -->
<div class="version-content active" id="version-3.9.4">
    <div class="download-grid">
        <div class="download-card compact">
            <div class="os-icon-small"><i class="fab fa-windows"></i></div>
            <h4>Windows 10/11</h4>
            <div class="download-buttons">
                <a href="https://github.com/Asher-1/ACloudViewer/releases/download/v3.9.4/ACloudViewer-v3.9.4-Win64-setup.exe" 
                   class="btn btn-download-small">
                    <i class="fas fa-download"></i> 下载
                </a>
            </div>
        </div>
        <!-- Linux 和 macOS 类似 -->
    </div>
</div>
```

3. 将之前的 `active` 类从旧版本移除，添加到新版本。

### 步骤 3：测试链接

```bash
# 测试下载链接是否有效
curl -I https://github.com/Asher-1/ACloudViewer/releases/download/v3.9.4/ACloudViewer-v3.9.4-Win64-setup.exe
```

应该返回 `302 Found` 或 `200 OK`。

### 步骤 4：部署

```bash
git add doc/
git commit -m "Update download links for version X.X.X"
git push origin master
```

## 🎯 快速更新脚本

如果您经常需要更新链接，可以使用以下脚本（需要修改）：

```bash
#!/bin/bash
# update_version.sh

VERSION=$1  # 例如: 3.9.5-Beta
DATE=$2     # 例如: 2025年3月1日

if [ -z "$VERSION" ] || [ -z "$DATE" ]; then
    echo "用法: ./update_version.sh <版本号> <日期>"
    echo "示例: ./update_version.sh 3.9.5-Beta '2025年3月1日'"
    exit 1
fi

# 更新 index.html 中的版本号
sed -i.bak "s/v[0-9]\+\.[0-9]\+\.[0-9]\+-Beta/v$VERSION/g" doc/index.html
sed -i.bak "s/发布日期: [^<]*/发布日期: $DATE/g" doc/index.html

echo "版本已更新为: $VERSION"
echo "发布日期: $DATE"
echo "请手动检查并提交更改"
```

## ⚠️ 注意事项

### 1. 文件命名一致性

确保 GitHub Release 中的文件名与网站链接完全匹配，包括：
- 版本号格式
- 大小写
- 后缀名

### 2. 链接验证

发布前务必测试所有下载链接：

```bash
# 测试所有链接
curl -I https://github.com/Asher-1/ACloudViewer/releases/download/v3.9.4-Beta/ACloudViewer-v3.9.4-Beta-Win64-setup.exe
curl -I https://github.com/Asher-1/ACloudViewer/releases/download/v3.9.4-Beta/ACloudViewer-v3.9.4-Beta-Linux-x86_64.AppImage
curl -I https://github.com/Asher-1/ACloudViewer/releases/download/v3.9.4-Beta/ACloudViewer-v3.9.4-Beta-macOS-x86_64.dmg
```

### 3. 保留旧版本

建议至少保留最近 3 个稳定版本的下载链接，以便用户回退。

### 4. 文件大小提示

可以在下载按钮旁边显示文件大小：

```html
<p class="file-info">文件大小: ~250 MB</p>
```

## 🔍 链接失效处理

如果链接失效（404错误）：

1. **检查 GitHub Release**：确认文件是否存在
2. **检查文件名**：是否与链接完全匹配
3. **检查版本号**：Release 标签是否正确
4. **备用方案**：临时指向 `releases/latest` 页面

示例替换：

```html
<!-- 临时方案：跳转到 Releases 页面 -->
<a href="https://github.com/Asher-1/ACloudViewer/releases/tag/v3.9.4" 
   class="btn btn-download-small" target="_blank">
    <i class="fas fa-external-link-alt"></i> 查看发布页
</a>
```

## 📊 下载统计

GitHub 自动提供每个 Release 文件的下载统计。您可以在：

```
https://github.com/Asher-1/ACloudViewer/releases
```

查看每个文件的下载次数。

## 🎨 自定义下载页面

如果需要更多自定义，可以修改：

- `doc/styles.css` - 调整下载卡片样式
- `doc/script.js` - 添加下载前的确认对话框或统计
- `doc/index.html` - 添加更多版本或下载选项

## 💡 最佳实践

1. **版本说明**：在每个版本旁边添加简短说明
2. **系统要求**：明确标注每个版本的系统要求
3. **安装指南**：提供快速安装链接
4. **更新日志**：链接到完整的 CHANGELOG
5. **校验和**：提供文件的 SHA256 校验和（可选）

## 📞 需要帮助？

如有问题，请查看：
- GitHub Releases 文档：https://docs.github.com/en/repositories/releasing-projects-on-github
- 或在项目中提交 Issue

---

**最后更新**: 2025-01-10  
**维护者**: ACloudViewer Team

