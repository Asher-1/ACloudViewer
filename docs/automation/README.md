# ACloudViewer 网站自动化更新系统完整指南

## 🎯 概述

这是一个**完全自动化、零人工维护**的网站下载链接更新系统。当您在GitHub上发布新版本时，网站会自动更新下载链接，无需任何人工干预。

## ✨ 主要特性

- ✅ **完全自动化**：无需手动更新网站
- ✅ **实时同步**：Release发布后自动触发更新
- ✅ **智能识别**：自动区分Beta版和稳定版
- ✅ **平台识别**：自动识别Windows、macOS、Linux安装包
- ✅ **定时检查**：每天自动检查并同步最新版本
- ✅ **零依赖**：使用Python标准库，无需额外安装包

## 📁 文件结构

```
ACloudViewer/
├── .github/
│   └── workflows/
│       └── update-website-downloads.yml    # GitHub Actions工作流
├── scripts/
│   ├── update_download_links.py            # 自动更新脚本
│   ├── requirements.txt                    # Python依赖（可选）
│   └── README.md                          # 脚本详细文档
├── doc/
│   └── index.html                         # 网站首页（自动更新）
└── AUTOMATION_GUIDE.md                    # 本文档
```

## 🚀 工作流程

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  1. 开发者发布新版本                                        │
│     └─> GitHub Release (main-devel或v3.x.x)               │
│                                                             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  2. 自动触发GitHub Actions                                  │
│     ├─> 监听Release发布事件                                │
│     ├─> 定时任务 (每天UTC 0点)                             │
│     └─> 手动触发 (可选)                                     │
│                                                             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  3. Python脚本执行                                          │
│     ├─> 调用GitHub API获取Releases数据                     │
│     ├─> 识别Beta版本 (main-devel tag)                      │
│     ├─> 识别稳定版本 (v3.9.3, v3.4.0等)                    │
│     ├─> 匹配各平台安装包文件                               │
│     └─> 生成新的HTML内容                                   │
│                                                             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  4. 更新并提交                                              │
│     ├─> 更新doc/index.html文件                             │
│     ├─> Git commit变更                                     │
│     └─> 自动推送到仓库                                      │
│                                                             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  5. GitHub Pages自动部署                                    │
│     └─> 网站更新完成!                                       │
│         https://asher-1.github.io/ACloudViewer/            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 🔧 系统组件

### 1. GitHub Actions工作流

**文件**: `.github/workflows/update-website-downloads.yml`

**触发条件**:
- 📦 **Release发布**: 当有新的Release发布或编辑时
- ⏰ **定时任务**: 每天UTC 0点（北京时间早上8点）
- 🖱️ **手动触发**: 在GitHub Actions页面手动运行

**工作流程**:
```yaml
1. Checkout代码
2. 安装Python 3.11
3. 运行更新脚本
4. 检查是否有变更
5. 自动提交并推送（如果有变更）
```

### 2. Python更新脚本

**文件**: `docs/automation/scripts/update_download_links.py`

**核心功能**:

#### 版本识别

```python
# Beta版本识别
def get_beta_release(releases):
    """
    识别条件:
    - Tag名称 = 'main-devel'
    - 标记为pre-release
    """
    
# 稳定版本识别  
def get_stable_releases(releases, limit=3):
    """
    识别条件:
    - 非pre-release
    - Tag格式: v3.9.3, v3.4.0等
    - 排除main-devel
    - 获取最新3个版本
    """
```

#### 平台识别

自动识别以下平台的安装包：

| 平台 | 匹配模式 | 示例文件名 |
|------|---------|-----------|
| **Windows** | `*.exe` | `ACloudViewer-3.9.3+d236e24-win-cpu-amd64.exe` |
| **macOS** | `*.dmg`, `*.pkg` | `ACloudViewer-3.9.3+d236e24-mac-cpu-ARM64.dmg` |
| **Linux** | `*.run`, `*.deb`, `*.rpm`, `*.appimage` | `ACloudViewer-3.9.3+d236e24-ubuntu20.04-cpu-amd64.run` |
| **Ubuntu** | `*.deb` | `ACloudViewer-ubuntu-20.04.deb` |

#### HTML生成

```python
# 生成Beta版本区域
def generate_beta_section(beta_release):
    """
    生成包含:
    - 版本名称
    - 发布日期
    - Commit SHA
    - 各平台下载链接
    """

# 生成稳定版本区域
def generate_stable_section(stable_releases):
    """
    生成包含:
    - 版本切换标签
    - 每个版本的下载链接
    - 支持最多3个历史版本
    """
```

## 📋 使用指南

### 开发者：发布新版本

#### 发布Beta版本

```bash
# 1. 在main分支上开发新功能
git checkout main
git add .
git commit -m "feat: add new feature"
git push origin main

# 2. GitHub Actions会自动:
#    - 构建并发布到main-devel tag
#    - 触发网站更新workflow
#    - 自动更新网站下载链接
```

#### 发布稳定版本

```bash
# 1. 创建新的release标签
git tag -a v3.10.0 -m "Release v3.10.0"
git push origin v3.10.0

# 2. 在GitHub上创建Release:
#    - 访问: https://github.com/Asher-1/ACloudViewer/releases/new
#    - 选择标签: v3.10.0
#    - 填写Release notes
#    - 上传编译好的安装包:
#      * Windows: *.exe
#      * macOS: *.dmg
#      * Linux: *.run
#    - 点击"Publish release"

# 3. 系统自动:
#    - 触发更新workflow
#    - 识别新版本
#    - 更新网站下载链接
#    - 部署到GitHub Pages
```

### 维护者：监控和管理

#### 查看自动化运行状态

1. 访问 Actions 页面: https://github.com/Asher-1/ACloudViewer/actions
2. 查找 "Update Website Download Links" workflow
3. 检查最近的运行记录

#### 手动触发更新

1. 访问 Actions 页面
2. 选择 "Update Website Download Links"
3. 点击 "Run workflow"
4. 选择分支（通常是main）
5. 点击 "Run workflow" 按钮

#### 本地测试

```bash
# 1. 进入项目目录
cd /Users/asher/develop/code/github/ACloudViewer

# 2. 运行更新脚本
python3 docs/automation/scripts/update_download_links.py

# 3. 查看变更
git diff doc/index.html

# 4. 本地预览
cd doc
python3 -m http.server 8080
# 访问 http://localhost:8080
```

## 🛠️ 配置和定制

### 修改获取的稳定版本数量

编辑 `scripts/update_download_links.py`:

```python
# 找到这一行并修改数字
stable_releases = get_stable_releases(releases, limit=5)  # 默认是3
```

### 添加新的平台识别

编辑 `scripts/update_download_links.py`，在 `PLATFORM_PATTERNS` 中添加：

```python
PLATFORM_PATTERNS = {
    'windows': {...},
    'macos': {...},
    'linux': {...},
    # 添加新平台
    'android': {
        'patterns': [r'android.*\.(apk|aab)$', r'\.(apk|aab)$'],
        'display_name': 'Android'
    }
}
```

### 修改定时任务频率

编辑 `.github/workflows/update-website-downloads.yml`:

```yaml
schedule:
  # 每6小时运行一次
  - cron: '0 */6 * * *'
  
  # 每周一运行一次
  - cron: '0 0 * * 1'
  
  # 每月1号运行一次
  - cron: '0 0 1 * *'
```

## 🔍 故障排查

### 问题1: 网站没有更新

**可能原因**:
- GitHub Actions运行失败
- 没有找到合适的安装包文件
- Git提交权限问题

**解决方法**:
```bash
# 1. 检查Actions运行日志
访问: https://github.com/Asher-1/ACloudViewer/actions

# 2. 查看失败原因
点击失败的workflow run -> 查看详细日志

# 3. 本地复现问题
python3 docs/automation/scripts/update_download_links.py
```

### 问题2: 找不到某个平台的下载链接

**可能原因**:
- Release中没有上传该平台的安装包
- 文件名不符合识别规则

**解决方法**:
```bash
# 1. 检查Release assets
访问: https://github.com/Asher-1/ACloudViewer/releases/tag/main-devel

# 2. 确认文件名格式
Windows: *.exe
macOS: *.dmg 或 *.pkg
Linux: *.run 或 *.deb 或 *.rpm

# 3. 如果文件名特殊，修改PLATFORM_PATTERNS
编辑 scripts/update_download_links.py 添加新的匹配模式
```

### 问题3: API速率限制

**错误信息**: `API rate limit exceeded`

**解决方法**:
```yaml
# GitHub Actions自动使用GITHUB_TOKEN
# 本地测试时设置token:
export GITHUB_TOKEN=your_personal_access_token
python3 docs/automation/scripts/update_download_links.py
```

### 问题4: SSL证书错误

**错误信息**: `SSL: CERTIFICATE_VERIFY_FAILED`

**解决方法**:
```bash
# macOS
/Applications/Python\ 3.x/Install\ Certificates.command

# 或使用脚本内置的fallback机制（已实现）
```

## 📊 监控指标

### 成功指标

- ✅ GitHub Actions运行成功（绿色勾）
- ✅ 网站显示最新版本号
- ✅ 下载链接可以正常点击下载
- ✅ 每个平台的安装包都有对应链接

### 检查清单

每次发布新版本后，请验证：

```
□ Beta版本号是否更新
□ Beta版本的Commit SHA是否正确
□ Beta版本的发布日期是否正确
□ Windows下载链接是否有效
□ macOS下载链接是否有效
□ Linux下载链接是否有效
□ 稳定版本标签是否正确
□ 历史版本是否保留（最新3个）
□ 点击下载能否正常下载文件
```

## 🔒 安全性

### Token安全
- ✅ 使用GitHub自动提供的 `GITHUB_TOKEN`
- ✅ Token自动过期，无需手动管理
- ✅ 最小权限原则：只请求 `contents: write`

### 代码审查
- ✅ 所有更改都产生Git提交
- ✅ 可以通过Git历史审查所有变更
- ✅ 支持回滚到任意版本

### 防止无限循环
- ✅ 提交消息包含 `[skip ci]`
- ✅ 只在有实际变更时才提交
- ✅ 避免触发连锁反应

## 📚 相关资源

### 文档
- [GitHub Actions文档](https://docs.github.com/en/actions)
- [GitHub Releases API](https://docs.github.com/en/rest/releases)
- [GitHub Pages配置](https://docs.github.com/en/pages)

### 项目链接
- [ACloudViewer 仓库](https://github.com/Asher-1/ACloudViewer)
- [Releases页面](https://github.com/Asher-1/ACloudViewer/releases)
- [Actions页面](https://github.com/Asher-1/ACloudViewer/actions)
- [网站地址](https://asher-1.github.io/ACloudViewer/)

### 脚本文档
- [详细脚本文档](scripts/README.md)
- [Python脚本源码](scripts/update_download_links.py)
- [工作流配置](.github/workflows/update-website-downloads.yml)

## 🎉 总结

恭喜！您现在拥有了一个**完全自动化、零维护**的网站更新系统！

### 核心优势

1. **零人工维护**: 发布版本后一切自动完成
2. **实时同步**: Release发布即刻更新网站
3. **智能识别**: 自动识别版本类型和平台
4. **稳定可靠**: 定时检查确保同步
5. **易于扩展**: 支持添加新平台和自定义配置

### 下一步

- 📦 发布新版本测试系统
- 🔍 监控第一次自动更新
- 📝 根据需要调整配置
- 🎯 享受自动化带来的便利！

---

**最后更新**: 2026-01-10  
**作者**: ACloudViewer Team  
**版本**: 1.0.0  
**维护**: Automated by GitHub Actions ⚡️

