# GitHub Actions Workflows

## 自动化工作流

### 📦 Update Website Download Links

**文件**: `update-website-downloads.yml`

**功能**: 自动从GitHub Releases获取最新版本信息并更新网站下载链接

**触发条件**:
- 🚀 Release发布或编辑时
- ⏰ 每天UTC 0点定时运行
- 🖱️ 手动触发

**工作流程**:
1. Checkout代码
2. 安装Python 3.11
3. 运行 `scripts/update_download_links.py`
4. 检测变更
5. 自动提交并推送（如有变更）

**查看运行状态**:
https://github.com/Asher-1/ACloudViewer/actions/workflows/update-website-downloads.yml

**手动运行**:
1. 访问 [Actions](https://github.com/Asher-1/ACloudViewer/actions)
2. 选择 "Update Website Download Links"
3. 点击 "Run workflow"

**相关文档**:
- [完整自动化指南](../../docs/automation/README.md)
- [脚本文档](../../docs/automation/scripts/README.md)
- [文档目录](../../docs/README.md)

---

**维护**: GitHub Actions (Automated)
