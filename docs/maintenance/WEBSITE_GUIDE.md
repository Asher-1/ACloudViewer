# 🌐 ACloudViewer GitHub Pages 网站部署指南

## 📋 概述

我已经为您创建了一个**专业、美观、现代化**的 GitHub Pages 网站，专门用于展示 ACloudViewer 项目。

## ✨ 网站特色

### 🎨 设计特点
- ✅ **现代化界面**：采用渐变色背景、卡片式布局
- ✅ **响应式设计**：完美支持手机、平板、桌面设备
- ✅ **流畅动画**：平滑的滚动效果和元素动画
- ✅ **专业配色**：蓝紫渐变主题，视觉效果出众

### 🚀 功能亮点
- ✅ **多平台下载**：Windows、Linux、macOS 下载链接
- ✅ **Python 安装**：一键复制 pip 安装命令
- ✅ **快速开始**：Python、C++、GUI 三种使用方式
- ✅ **功能展示**：8大核心特性卡片展示
- ✅ **应用截图**：图片灯箱效果查看
- ✅ **学习资源**：GitHub、文档、视频教程链接

### 🛠️ 技术特性
- ✅ **SEO 优化**：sitemap.xml、robots.txt
- ✅ **性能优化**：轻量级设计、快速加载
- ✅ **用户体验**：代码一键复制、平滑导航、返回顶部
- ✅ **移动友好**：汉堡菜单、触摸优化

## 📁 文件结构

```
doc/
├── .nojekyll           # 禁用 Jekyll 处理
├── index.html          # 主页面（核心文件）
├── styles.css          # 样式表（美化界面）
├── script.js           # JavaScript 交互
├── 404.html            # 404 错误页面
├── robots.txt          # SEO 配置
├── sitemap.xml         # 网站地图
├── README.md           # 使用说明
├── DEPLOYMENT.md       # 详细部署指南
├── QUICKSTART.md       # 快速开始（推荐阅读）
└── images/             # 图片资源目录
    ├── ACloudViewer_logo_horizontal.png
    ├── ACloudViewerMainUI.png
    ├── SemanticAnnotation.png
    ├── Reconstruction.png
    └── CloudViewerApp.png
```

## 🚀 快速部署（3步完成）

### 步骤 1️⃣：提交代码到 GitHub

```bash
cd /Users/asher/develop/code/github/ACloudViewer

# 添加新文件
git add doc/

# 提交更改
git commit -m "Add professional GitHub Pages website"

# 推送到 GitHub
git push origin master
```

### 步骤 2️⃣：配置 GitHub Pages

1. 在浏览器中打开：
   ```
   https://github.com/Asher-1/ACloudViewer/settings/pages
   ```

2. 在 **Source** 部分设置：
   - **Branch**: 选择 `master` (或 `main`)
   - **Folder**: 选择 `/doc`
   - 点击 **Save** 按钮

3. 等待 2-5 分钟让 GitHub 构建网站

### 步骤 3️⃣：访问您的网站

打开浏览器访问：
```
https://asher-1.github.io/ACloudViewer/
```

**🎉 恭喜！您的专业网站已上线！**

## 💻 本地预览（可选）

在推送到 GitHub 之前，您可以先在本地预览：

```bash
# 进入 doc 目录
cd doc

# 启动本地服务器（选择一种）
python3 -m http.server 8000          # Python 3
python -m SimpleHTTPServer 8000       # Python 2  
npx http-server                       # Node.js
php -S localhost:8000                 # PHP
```

然后在浏览器访问：`http://localhost:8000`

## 🎯 常见自定义需求

### 1. 更新版本号

编辑 `doc/index.html`，搜索并修改：

```html
<span class="version-badge">最新版本: v3.9.4-Beta</span>
<span class="release-date">发布日期: 2025年2月12日</span>
```

### 2. 更新下载链接

找到下载区域（搜索 `download-card`），修改链接：

```html
<a href="https://github.com/Asher-1/ACloudViewer/releases/download/vX.X.X/ACloudViewer-X.X.X-Win64.exe" 
   class="btn btn-download">
    <i class="fas fa-download"></i> 下载安装包
</a>
```

### 3. 修改主题颜色

编辑 `doc/styles.css` 的顶部：

```css
:root {
    --primary-color: #2196F3;      /* 主色调 */
    --secondary-color: #FFC107;     /* 辅助色 */
    --dark-color: #1a1a2e;          /* 深色 */
    --light-color: #f8f9fa;         /* 浅色 */
}
```

### 4. 添加新截图

1. 将图片放到 `doc/images/` 目录
2. 在 `doc/index.html` 的 Gallery 区域添加：

```html
<div class="gallery-item">
    <img src="../doc/images/new-screenshot.png" alt="描述">
    <div class="gallery-overlay">
        <h3>功能标题</h3>
        <p>功能描述</p>
    </div>
</div>
```

## 📊 网站页面包含的内容

1. **Hero 区域**
   - 大标题和副标题
   - 下载和快速开始按钮
   - 开源、社区、跨平台标签

2. **关于区域**
   - 项目简介
   - 版本和许可证徽章

3. **功能特性**
   - 8个核心功能卡片
   - 图标和详细说明

4. **下载区域**
   - Windows/Linux/macOS 三大平台
   - Python pip 安装说明
   - 查看所有版本链接

5. **快速开始**
   - Python 示例
   - C++ 编译指南
   - GUI 使用说明

6. **应用展示**
   - 4张应用截图
   - 点击查看大图

7. **学习资源**
   - GitHub 仓库
   - 示例代码
   - 文档和教程链接

8. **页脚**
   - 快速链接
   - 社交媒体
   - 版权信息

## 🔧 高级配置

### 添加 Google Analytics

编辑 `doc/index.html`，在 `</head>` 前添加：

```html
<!-- Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=YOUR-GA-ID"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'YOUR-GA-ID');
</script>
```

### 使用自定义域名

1. 创建 CNAME 文件：
   ```bash
   echo "www.your-domain.com" > doc/CNAME
   ```

2. 在域名提供商配置 DNS：
   ```
   CNAME记录: www -> asher-1.github.io
   ```

3. 在 GitHub Pages 设置中输入域名

4. 启用 HTTPS（自动配置）

## 📚 参考文档

- **快速开始**: `doc/QUICKSTART.md` ⭐ 推荐先看这个
- **详细部署**: `doc/DEPLOYMENT.md`
- **使用说明**: `doc/README.md`
- **GitHub Pages 官方文档**: https://docs.github.com/en/pages

## ⚠️ 注意事项

1. **图片路径**：确保图片在 `doc/images/` 目录存在
2. **下载链接**：定期检查 Release 链接是否有效
3. **浏览器缓存**：更新后强制刷新（Ctrl+F5 或 Cmd+Shift+R）
4. **部署时间**：GitHub Pages 构建需要 2-5 分钟

## 🆘 遇到问题？

### 问题 1: 网站没有显示
- 检查 GitHub Pages 设置是否正确
- 等待 5-10 分钟再刷新
- 清除浏览器缓存

### 问题 2: 样式丢失
- 确认 `styles.css` 在 docs 目录
- 检查 `index.html` 中的引用路径

### 问题 3: 图片无法显示
- 检查图片路径：`../doc/images/xxx.png`
- 确认图片文件存在

### 问题 4: 下载链接失效
- 访问 GitHub Releases 页面
- 复制正确的下载链接
- 更新 `index.html` 中的链接

## 📞 获取帮助

- **GitHub Issues**: https://github.com/Asher-1/ACloudViewer/issues
- **Email**: ludahai19@163.com
- **GitHub Pages 文档**: https://docs.github.com/en/pages

## 🎉 完成清单

部署完成后，请检查：

- [ ] 网站可以正常访问
- [ ] 所有图片正常显示
- [ ] 下载链接正确有效
- [ ] 移动端显示正常
- [ ] 导航菜单工作正常
- [ ] 代码复制功能正常
- [ ] 图片灯箱效果正常

## 💡 维护建议

1. **定期更新**：每次发布新版本时更新网站
2. **检查链接**：定期检查所有外部链接
3. **监控流量**：使用 GitHub Insights 查看访问统计
4. **收集反馈**：通过 Issues 收集用户意见
5. **备份代码**：定期备份 docs 目录

## 🌟 网站预览

您的网站将包含以下页面：

```
https://asher-1.github.io/ACloudViewer/
├── #home          (首页)
├── #features      (功能特性)
├── #download      (下载页面)
├── #quickstart    (快速开始)
└── #gallery       (应用展示)
```

---

## 🚀 立即开始

现在就执行上面的**快速部署（3步完成）**，让您的专业网站上线吧！

**祝您使用愉快！如有任何问题，随时联系我。🎊**

---

**创建时间**: 2025年1月10日  
**版本**: 1.0  
**适用于**: ACloudViewer v3.9.4-Beta 及以上版本

