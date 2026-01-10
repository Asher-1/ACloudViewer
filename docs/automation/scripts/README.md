# 自动化下载链接更新系统

## 概述

这个自动化系统通过GitHub Actions和Python脚本，自动从GitHub Releases获取最新版本信息，并更新网站上的下载链接。**完全无需人工维护**。

## 工作流程

```
GitHub Release 发布
        ↓
触发 GitHub Action
        ↓
获取 Releases API 数据
        ↓
解析版本和下载链接
        ↓
更新 doc/index.html
        ↓
自动提交并推送
        ↓
GitHub Pages 自动部署
```

## 触发条件

自动化流程在以下情况下触发：

1. **新版本发布时**：当有新的Release发布或编辑时自动触发
2. **定时任务**：每天UTC 0点（北京时间早上8点）自动检查更新
3. **手动触发**：可以在GitHub Actions页面手动运行

## 版本识别规则

### Beta版本（开发版）
- **Tag名称**: `main-devel`
- **特征**: 标记为pre-release
- **用途**: 包含最新功能和实验性特性
- **自动获取**: Commit SHA前7位，发布日期

### 稳定版本（Stable）
- **Tag格式**: `v3.9.3`, `v3.4.0` 等
- **特征**: 非pre-release
- **获取数量**: 最新的3个稳定版本
- **排除**: `main-devel` tag

## 平台识别规则

脚本自动识别以下平台的安装包：

| 平台 | 文件后缀/模式 | 示例 |
|------|--------------|------|
| Windows | `.exe` | `ACloudViewer-windows-x64.exe` |
| macOS | `.dmg`, `.pkg` | `ACloudViewer-macos.dmg` |
| Linux | `.run`, `.deb`, `.rpm`, `.appimage` | `ACloudViewer-linux.run` |
| Ubuntu/Debian | `.deb` | `ACloudViewer-ubuntu-20.04.deb` |

## 文件结构

```
.github/workflows/
└── update-website-downloads.yml    # GitHub Actions工作流配置

scripts/
├── README.md                       # 本文档
└── update_download_links.py        # 更新脚本

doc/
└── index.html                      # 网站首页（自动更新）
```

## 使用方法

### 1. 自动运行（推荐）

无需任何操作，系统会自动：
- 监听Release发布事件
- 每天定时检查更新
- 自动提交更改到仓库

### 2. 本地测试

```bash
# 安装依赖
pip install requests jinja2

# 设置GitHub Token（可选，避免API限制）
export GITHUB_TOKEN=your_github_token

# 运行脚本
cd /Users/asher/develop/code/github/ACloudViewer
python docs/automation/scripts/update_download_links.py
```

### 3. 手动触发GitHub Action

1. 访问 GitHub仓库
2. 点击 **Actions** 标签
3. 选择 **Update Website Download Links**
4. 点击 **Run workflow**
5. 选择分支并点击 **Run workflow**

## 脚本功能详解

### `update_download_links.py`

#### 主要功能

1. **获取Releases数据**
   ```python
   def fetch_releases() -> List[Dict]:
       """从GitHub API获取所有releases"""
   ```

2. **识别Beta版本**
   ```python
   def get_beta_release(releases: List[Dict]) -> Optional[Dict]:
       """获取Beta版本（main-devel tag）"""
   ```

3. **识别稳定版本**
   ```python
   def get_stable_releases(releases: List[Dict], limit: int = 3) -> List[Dict]:
       """获取稳定版本（非pre-release，排除main-devel）"""
   ```

4. **平台匹配**
   ```python
   def find_asset_for_platform(assets: List[Dict], platform: str) -> Optional[Dict]:
       """为指定平台查找匹配的asset"""
   ```

5. **生成HTML**
   ```python
   def generate_download_section(beta_release, stable_releases) -> str:
       """生成下载区域的HTML"""
   ```

6. **更新文件**
   ```python
   def update_html_file(beta_release, stable_releases):
       """更新HTML文件"""
   ```

## 配置说明

### GitHub Actions配置

文件: `.github/workflows/update-website-downloads.yml`

```yaml
# 触发条件
on:
  release:
    types: [published, edited]  # Release发布时
  workflow_dispatch:             # 手动触发
  schedule:
    - cron: '0 0 * * *'          # 每天定时

# 权限
permissions:
  contents: write  # 需要写权限来提交更改
```

### 环境变量

- `GITHUB_TOKEN`: 自动提供，用于访问GitHub API和提交代码

## HTML更新机制

### 更新的HTML区域

1. **版本标签栏**
   ```html
   <div class="version-tabs">
       <button class="version-tab active" data-version="beta">Beta 开发版</button>
       <button class="version-tab" data-version="3.9.3">v3.9.3</button>
       ...
   </div>
   ```

2. **下载内容区**
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

### 正则表达式匹配

脚本使用正则表达式精确定位需要更新的HTML区域：

```python
# 版本标签区域
version_tabs_pattern = r'(<div class="version-tabs">\s*)(.*?)(\s*</div>)'

# 下载内容区域
version_sections_pattern = r'(<div class="version-sections">\s*)(.*?)(\s*</div>\s*</div>\s*</section>)'
```

## 故障排查

### 1. 脚本运行失败

**问题**: Python脚本报错

**检查**:
```bash
# 查看GitHub Actions日志
# GitHub仓库 -> Actions -> 选择失败的workflow run

# 本地测试
python docs/automation/scripts/update_download_links.py
```

### 2. 没有找到平台文件

**问题**: 某个平台的下载链接缺失

**原因**: Release中没有上传该平台的文件，或文件名不符合识别规则

**解决**: 
- 检查Release assets中的文件名
- 更新 `PLATFORM_PATTERNS` 字典，添加新的匹配模式

### 3. API速率限制

**问题**: GitHub API返回403错误

**原因**: 未认证的API请求有速率限制（60次/小时）

**解决**: GitHub Actions自动使用 `GITHUB_TOKEN`，通常不会遇到此问题

### 4. 提交失败

**问题**: Git push失败

**原因**: 权限不足或分支保护规则

**解决**: 
- 检查工作流是否有 `contents: write` 权限
- 检查分支保护规则，确保GitHub Actions可以推送

## 自定义配置

### 修改获取的稳定版本数量

编辑 `scripts/update_download_links.py`:

```python
# 将3改为你想要的数量
stable_releases = get_stable_releases(releases, limit=5)
```

### 添加新的平台识别

编辑 `PLATFORM_PATTERNS`:

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

### 修改定时任务频率

编辑 `.github/workflows/update-website-downloads.yml`:

```yaml
schedule:
  - cron: '0 */6 * * *'  # 每6小时运行一次
  - cron: '0 0 * * 1'    # 每周一运行一次
```

## 安全性

1. **Token安全**: 使用GitHub自动提供的 `GITHUB_TOKEN`，自动过期
2. **权限最小化**: 只请求必要的 `contents: write` 权限
3. **代码审查**: 所有更改都会产生Git提交，可以审查和回滚
4. **Skip CI**: 提交消息包含 `[skip ci]`，避免无限循环触发

## 维护指南

### 定期检查

1. **每月检查一次**: GitHub Actions运行日志
2. **Release后验证**: 确认网站下载链接已更新
3. **更新依赖**: 定期更新Python依赖包

### 版本升级

当有新的GitHub Actions版本时：

```yaml
# 更新actions版本
- uses: actions/checkout@v5      # 从v4升级到v5
- uses: actions/setup-python@v6  # 从v5升级到v6
```

## 贡献

如果你发现bug或有改进建议：

1. 提交Issue: https://github.com/Asher-1/ACloudViewer/issues
2. 提交Pull Request
3. 在Issue中描述问题和解决方案

## 许可证

本脚本遵循ACloudViewer项目的许可证。

---

**最后更新**: 2026-01-10  
**作者**: ACloudViewer Team  
**维护**: Automated by GitHub Actions

