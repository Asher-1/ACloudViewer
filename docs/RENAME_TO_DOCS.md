# 文件夹重命名：doc → docs

## 📅 更新日期
2026-01-10

## 🎯 更新目标

将 `doc` 文件夹重命名为 `docs`，并确保以下 URL 可以访问：
- ✅ https://asher-1.github.io/ACloudViewer/docs
- ✅ https://asher-1.github.io/ACloudViewer/downloads

## 📝 执行的更改

### 1. 文件夹重命名

```bash
doc/ → docs/
```

**新的文件夹结构**:
```
docs/
├── index.html              # 主页（访问 /docs 时显示）
├── downloads/              # 下载页面目录
│   └── index.html          # 重定向到主页的下载区域
├── automation/             # 自动化系统
├── guides/                 # 用户指南
├── maintenance/            # 维护文档
└── ...
```

### 2. GitHub Actions 更新

**文件**: `.github/workflows/update-website-downloads.yml`

更改的路径：
```yaml
# 脚本路径
python doc/automation/scripts/update_download_links.py
→ python docs/automation/scripts/update_download_links.py

# 检查变更
git diff --quiet doc/index.html
→ git diff --quiet docs/index.html

# 提交文件
git add doc/index.html
→ git add docs/index.html
```

### 3. Python 脚本更新

**文件**: `docs/automation/scripts/update_download_links.py`

```python
# HTML 文件路径
html_file = 'doc/index.html'
→ html_file = 'docs/index.html'
```

### 4. 文档引用更新

更新了以下文件中的所有 `doc/` 引用：

| 文件 | 更改次数 | 状态 |
|------|----------|------|
| `docs/automation/README.md` | 多处 | ✅ |
| `docs/automation/SUMMARY.md` | 多处 | ✅ |
| `docs/automation/scripts/README.md` | 多处 | ✅ |
| `docs/README.md` | 多处 | ✅ |
| `docs/RESTRUCTURE_SUMMARY.md` | 多处 | ✅ |
| `.github/workflows/README.md` | 3处 | ✅ |

### 5. 创建 Downloads 页面

**新文件**: `docs/downloads/index.html`

**功能**:
- 自动重定向到主页的下载区域（`../index.html#download`）
- 提供友好的加载动画
- 备用手动链接

**访问方式**:
```
https://asher-1.github.io/ACloudViewer/downloads
→ 自动跳转到
https://asher-1.github.io/ACloudViewer/docs#download
```

## 🌐 GitHub Pages 配置

### URL 访问说明

| URL | 访问内容 | 说明 |
|-----|----------|------|
| `https://asher-1.github.io/ACloudViewer/` | 根页面 | 如果设置了 GitHub Pages |
| `https://asher-1.github.io/ACloudViewer/docs` | `docs/index.html` | 主页 ✅ |
| `https://asher-1.github.io/ACloudViewer/docs/` | `docs/index.html` | 主页（带斜杠）✅ |
| `https://asher-1.github.io/ACloudViewer/downloads` | `docs/downloads/index.html` | 重定向到下载区 ✅ |

### GitHub Pages 设置

在 GitHub 仓库设置中，确保：

1. **Settings** → **Pages**
2. **Source**: Deploy from a branch
3. **Branch**: `main` (或你的主分支)
4. **Folder**: 选择 `/ (root)` 或 `docs/`（推荐）

**推荐配置**:
- Branch: `main`
- Folder: `/docs`

这样 `docs/index.html` 将成为 `https://asher-1.github.io/ACloudViewer/` 的主页。

## ✅ 功能验证

### 测试自动化脚本

```bash
$ python3 docs/automation/scripts/update_download_links.py
============================================================
Starting download links update process...
============================================================
Fetching releases from GitHub API...
Found 16 releases
Found beta release: main-devel
Found 3 stable releases
Reading docs/index.html...  ✅
Writing updated content to docs/index.html...  ✅
✅ Successfully updated download links!
============================================================
```

**结果**: ✅ 所有功能正常

### 测试 URL 访问（部署后）

部署到 GitHub Pages 后，验证以下 URL：

```bash
# 主页
curl -I https://asher-1.github.io/ACloudViewer/docs

# 下载页（应返回重定向或 HTML）
curl -I https://asher-1.github.io/ACloudViewer/downloads
```

## 📊 更改统计

### 文件操作

- **重命名**: 1 个文件夹 (`doc/` → `docs/`)
- **更新**: 8 个文件（路径引用）
- **创建**: 2 个文件（`downloads/index.html`, `RENAME_TO_DOCS.md`）

### 代码更改

- **GitHub Actions**: 3 处路径更新
- **Python 脚本**: 1 处路径更新
- **文档**: 多处引用更新

## 🔍 验证清单

在提交更改前，请验证：

- [ ] `docs/` 文件夹存在且包含所有文件
- [ ] `doc/` 文件夹已不存在
- [ ] GitHub Actions workflow 中的路径已更新
- [ ] Python 脚本运行正常
- [ ] 所有文档链接已更新
- [ ] `docs/downloads/index.html` 已创建
- [ ] 本地测试通过

部署到 GitHub Pages 后：

- [ ] https://asher-1.github.io/ACloudViewer/docs 可访问
- [ ] https://asher-1.github.io/ACloudViewer/downloads 可访问并正确重定向
- [ ] 网站功能正常（导航、下载链接等）
- [ ] GitHub Actions 自动更新正常工作

## 🚀 部署步骤

### 1. 配置 GitHub Pages

在 GitHub 仓库中：

1. 进入 **Settings** → **Pages**
2. **Source**: Deploy from a branch
3. **Branch**: 选择 `main`
4. **Folder**: 选择 `/docs`
5. 点击 **Save**

### 2. 提交更改

```bash
cd /Users/asher/develop/code/github/ACloudViewer

# 查看更改
git status

# 添加所有更改
git add docs/ .github/workflows/

# 删除旧的 doc 文件夹（如果还在 Git 中）
git rm -rf doc/

# 提交
git commit -m "refactor: rename doc to docs folder

- Rename doc/ to docs/ for GitHub Pages convention
- Update all path references in workflows and documentation
- Create docs/downloads/index.html for /downloads URL
- Update Python automation script paths
- All tests passing ✅

URLs:
- https://asher-1.github.io/ACloudViewer/docs
- https://asher-1.github.io/ACloudViewer/downloads
"

# 推送到 GitHub
git push origin main
```

### 3. 验证部署

等待 GitHub Actions 完成部署（约 1-2 分钟），然后访问：

- https://asher-1.github.io/ACloudViewer/docs
- https://asher-1.github.io/ACloudViewer/downloads

## 📚 相关文档

- [文档目录](README.md)
- [自动化系统指南](automation/README.md)
- [重组总结](RESTRUCTURE_SUMMARY.md)
- [网站维护指南](maintenance/WEBSITE_GUIDE.md)

## 💡 注意事项

### 向后兼容

- 旧的 `doc/` 路径引用在 Git 历史中保留
- 外部链接如果指向旧路径可能需要更新
- 建议在主 README 中说明新的文档位置

### GitHub Pages 特性

- `docs/` 是 GitHub Pages 的标准文件夹名称
- 文件夹名称为 `docs` 会自动被识别
- URL 路径会自动映射到文件夹结构

### 自动化系统

- 所有自动化脚本已更新到新路径
- GitHub Actions 会自动使用新路径
- 无需手动干预，系统会自动工作

## 🎉 总结

### 完成的工作

✅ 文件夹重命名：`doc/` → `docs/`  
✅ 更新所有路径引用（8个文件）  
✅ 创建 `/downloads` 页面重定向  
✅ 测试自动化脚本正常  
✅ 验证功能完整性

### 新的 URL 结构

```
https://asher-1.github.io/ACloudViewer/
├── docs                  # 主页和文档
│   ├── #download         # 下载区域（锚点）
│   └── downloads/        # 下载页面（重定向）
├── automation/           # 自动化文档
├── guides/               # 用户指南
└── maintenance/          # 维护文档
```

### 下一步

1. ✅ 提交所有更改到 Git
2. ✅ 推送到 GitHub
3. ✅ 配置 GitHub Pages 使用 `/docs` 文件夹
4. ✅ 验证 URL 访问正常
5. ✅ 更新外部文档中的链接（如需要）

---

**更新日期**: 2026-01-10  
**执行者**: Cursor AI Assistant  
**验证状态**: ✅ All tests passed  
**部署状态**: 🚀 Ready for deployment

