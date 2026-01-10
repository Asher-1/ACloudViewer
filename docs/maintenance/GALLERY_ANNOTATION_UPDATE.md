# 应用展示标注标签页更新说明

## 更新日期
2026-01-10

## 更新内容

### 1. 标签页名称变更
- **原标签页名称**: 点云配准
- **新标签页名称**: 语义标注

### 2. 图片重新分类

#### 语义标注标签页（gallery-annotation）
现包含以下三张图片：

1. **ACloudViewerMainUI.png**
   - 标题：主界面
   - 描述：现代化的用户界面，功能强大且易用
   - 来源：从"用户界面"标签页移动

2. **SemanticAnnotation.png**
   - 标题：语义标注工具
   - 描述：智能的3D语义分割和标注功能
   - 来源：从"用户界面"和"机器学习"标签页移动（去重）

3. **SenceCloud.png**
   - 标题：大规模场景标注
   - 描述：处理海量点云数据的语义标注，支持上亿点渲染
   - 来源：从"3D重建"标签页移动

#### 3D重建标签页（gallery-reconstruction）
新增一张图片：

- **ICP-registration.png**
  - 标题：ICP点云配准
  - 描述：高性能的迭代最近点算法，支持多尺度配准
  - 来源：从原"点云配准"标签页移动

#### 用户界面标签页（gallery-ui）
新增一张图片：

- **AbstractionLayers.png**
  - 标题：系统架构
  - 描述：模块化设计，从底层到应用层的完整抽象
  - 来源：从原"点云配准"标签页移动

### 3. 标签页内容总览

#### 用户界面（2张）
- CloudViewerApp.png
- AbstractionLayers.png

#### 3D重建（3张）
- Reconstruction.png
- real-time-3D-Reconstruction.png
- ICP-registration.png

#### 语义标注（3张）✨ 新
- ACloudViewerMainUI.png
- SemanticAnnotation.png
- SenceCloud.png

#### 机器学习（1张）
- getting_started_ml_visualizer.gif

#### 可视化（2张）
- jupyter_visualizer.png
- CloudViewerApp.png

## 技术说明

### 文件修改
- **doc/index.html**: 更新了gallery标签页结构和图片分类
- HTML中的 `data-gallery="annotation"` 属性已正确设置
- ID `gallery-annotation` 已正确配置

### JavaScript兼容性
- 无需修改 `script.js`
- 现有的gallery标签切换逻辑完全兼容新的标注标签页
- 动态加载机制通过 `data-gallery` 属性自动识别

### CSS样式
- 无需修改 `styles.css`
- 所有样式类已适配新标签页

## 测试建议

1. **本地测试**：
   ```bash
   cd /Users/asher/develop/code/github/ACloudViewer/doc
   python3 -m http.server 8000
   ```
   访问 http://localhost:8000

2. **检查项目**：
   - ✅ 标签页切换是否正常
   - ✅ 图片是否正确显示
   - ✅ 悬停效果是否正常
   - ✅ 点击图片lightbox是否工作
   - ✅ 响应式布局在移动端是否正常

## 部署

更新后的页面可以直接推送到GitHub，通过以下地址访问：
- 主页: https://asher-1.github.io/ACloudViewer/
- 应用展示: https://asher-1.github.io/ACloudViewer/#gallery

---

**注意**: 此次更新专注于将标注相关的应用场景集中展示，提升了内容的主题性和可读性。

