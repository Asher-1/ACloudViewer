# ACloudViewer 网站部署指南

## 快速部署步骤

### 1. 准备工作

确保您已经：
- 克隆了 ACloudViewer 仓库
- 有 GitHub 仓库的写权限
- Git 已正确配置

### 2. 部署到 GitHub Pages

#### 步骤 1: 推送代码到 GitHub

```bash
# 进入项目根目录
cd /path/to/ACloudViewer

# 添加所有更改
git add docs/

# 提交更改
git commit -m "Add GitHub Pages website"

# 推送到远程仓库
git push origin master
```

#### 步骤 2: 配置 GitHub Pages

1. 访问 GitHub 仓库页面：`https://github.com/Asher-1/ACloudViewer`

2. 点击 `Settings`（设置）标签

3. 在左侧菜单中找到 `Pages`

4. 在 `Source` 部分：
   - Branch: 选择 `master` (或 `main`)
   - Folder: 选择 `/docs`
   - 点击 `Save` 按钮

5. 等待部署完成（通常需要 2-5 分钟）

6. 部署完成后，您会看到一个绿色的提示：
   ```
   Your site is published at https://asher-1.github.io/ACloudViewer/
   ```

### 3. 验证部署

在浏览器中访问：
```
https://asher-1.github.io/ACloudViewer/
```

如果看到完整的网站页面，说明部署成功！

## 常见问题排查

### 问题 1: 图片无法显示

**原因**: 图片路径不正确

**解决方案**:
- 检查图片是否存在于 `doc/images/` 目录
- 确保 HTML 中的路径是相对路径：`../doc/images/xxx.png`

### 问题 2: CSS 样式未加载

**原因**: 路径问题或缓存

**解决方案**:
```bash
# 清除浏览器缓存，或使用隐私浏览模式
# 检查 index.html 中的 CSS 链接
<link rel="stylesheet" href="styles.css">
```

### 问题 3: 404 错误

**原因**: GitHub Pages 配置不正确

**解决方案**:
- 确认选择的分支和文件夹正确
- 确认 `index.html` 在 `docs` 目录根目录
- 等待几分钟让 GitHub Pages 完成构建

### 问题 4: 下载链接失效

**原因**: Release 版本号不匹配

**解决方案**:
1. 访问 `https://github.com/Asher-1/ACloudViewer/releases`
2. 复制正确的下载链接
3. 更新 `index.html` 中的下载链接

## 更新网站内容

### 更新版本信息

编辑 `docs/index.html`，找到版本信息部分：

```html
<div class="version-info">
    <span class="version-badge">最新版本: vX.X.X</span>
    <span class="release-date">发布日期: YYYY年MM月DD日</span>
</div>
```

更新版本号和日期。

### 更新下载链接

找到下载区域的链接，更新为新版本：

```html
<a href="https://github.com/Asher-1/ACloudViewer/releases/download/vX.X.X/ACloudViewer-X.X.X-Win64.exe" 
   class="btn btn-download">
    <i class="fas fa-download"></i> 下载安装包
</a>
```

### 添加新的功能特性

在特性区域添加新的卡片：

```html
<div class="feature-card">
    <div class="feature-icon">
        <i class="fas fa-YOUR-ICON"></i>
    </div>
    <h3>新功能标题</h3>
    <p>功能描述</p>
</div>
```

### 更新截图

1. 将新截图放到 `doc/images/` 目录
2. 在 Gallery 区域更新图片路径：

```html
<div class="gallery-item">
    <img src="../doc/images/new-screenshot.png" alt="描述">
    <div class="gallery-overlay">
        <h3>标题</h3>
        <p>描述</p>
    </div>
</div>
```

## 自定义域名设置

如果要使用自定义域名（如 `www.acloudviewer.com`）：

### 1. 创建 CNAME 文件

```bash
cd docs/
echo "www.acloudviewer.com" > CNAME
git add CNAME
git commit -m "Add custom domain"
git push
```

### 2. 配置 DNS

在您的域名提供商处添加以下记录：

**方法 A: 使用 CNAME（推荐用于 www 子域名）**
```
类型: CNAME
名称: www
值: asher-1.github.io
```

**方法 B: 使用 A 记录（用于根域名）**
```
类型: A
名称: @
值: 185.199.108.153
值: 185.199.109.153
值: 185.199.110.153
值: 185.199.111.153
```

### 3. 在 GitHub 设置中配置

1. 进入 `Settings` > `Pages`
2. 在 `Custom domain` 输入框中输入您的域名
3. 点击 `Save`
4. 等待 DNS 检查完成（可能需要 24-48 小时）

### 4. 启用 HTTPS

DNS 配置完成后：
1. 在 GitHub Pages 设置中勾选 `Enforce HTTPS`
2. 等待 SSL 证书自动配置完成

## 性能优化建议

### 1. 图片优化

```bash
# 使用 ImageOptim 或在线工具压缩图片
# 推荐尺寸：
# - Hero 背景: 1920x1080
# - 功能图标: 512x512
# - 截图: 1200x800
```

### 2. 启用缓存

GitHub Pages 自动配置缓存，无需额外设置。

### 3. 使用 CDN

字体和图标已使用 CDN（Font Awesome），加载速度更快。

## 监控和分析

### 添加 Google Analytics

在 `index.html` 的 `</head>` 前添加：

```html
<!-- Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=GA_MEASUREMENT_ID"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'GA_MEASUREMENT_ID');
</script>
```

### 查看访问统计

使用 GitHub Insights：
1. 进入仓库主页
2. 点击 `Insights` 标签
3. 查看 `Traffic` 部分

## 备份和回滚

### 创建备份

```bash
# 创建备份分支
git checkout -b website-backup
git push origin website-backup
```

### 回滚到之前的版本

```bash
# 查看历史提交
git log --oneline

# 回滚到指定版本
git reset --hard <commit-hash>
git push origin master --force
```

**注意**: 谨慎使用 `--force`，它会覆盖远程仓库。

## 持续集成（可选）

### 使用 GitHub Actions 自动部署

创建 `.github/workflows/deploy.yml`：

```yaml
name: Deploy Website

on:
  push:
    branches: [ master ]
    paths:
      - 'docs/**'

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Validate HTML
        run: |
          sudo apt-get install -y tidy
          tidy -q -e docs/index.html || true
      
      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./docs
```

## 安全建议

1. **不要提交敏感信息**：API 密钥、密码等
2. **定期更新依赖**：特别是 Font Awesome 等外部资源
3. **使用 HTTPS**：始终启用 HTTPS
4. **检查链接**：定期检查外部链接是否有效

## 技术支持

如遇到问题：

1. 查看 [GitHub Pages 文档](https://docs.github.com/en/pages)
2. 在仓库提交 Issue
3. 发送邮件至：ludahai19@163.com

## 更新日志

记录每次重要更新：

```markdown
### 2025-01-10
- 初始版本上线
- 添加下载功能
- 添加多语言支持

### YYYY-MM-DD
- 更新版本到 vX.X.X
- 添加新功能 XXX
- 修复 BUG XXX
```

## 许可证

网站代码采用与 ACloudViewer 项目相同的许可证。

