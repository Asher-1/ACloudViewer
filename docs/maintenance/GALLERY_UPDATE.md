# 应用展示更新说明

## 🎨 新的分类标签页展示

应用展示区域已经改造成**5个分类标签页**，展示不同领域的应用场景。

### 📋 分类标签

#### 1. **用户界面** (默认显示)
- ✅ 主界面 (`ACloudViewerMainUI.png`)
- ✅ CloudViewer应用 (`CloudViewerApp.png`)
- ✅ 语义标注工具 (`SemanticAnnotation.png`)

**展示内容**: 软件的用户界面和交互设计

#### 2. **3D重建**
- ✅ 场景重建 (`Reconstruction.png`)
- ✅ 实时3D重建 (`real-time-3D-Reconstruction.png`)
- ✅ 大规模场景 (`SenceCloud.png`)

**展示内容**: 点云重建、SLAM、大规模场景处理

#### 3. **点云配准**
- ✅ ICP点云配准 (`ICP-registration.png`)
- ✅ 系统架构 (`AbstractionLayers.png`)

**展示内容**: 点云对齐、配准算法、系统架构设计

#### 4. **机器学习**
- ✅ 机器学习可视化 (`getting_started_ml_visualizer.gif`) - 宽版显示
- ✅ 语义分割 (`SemanticAnnotation.png`)

**展示内容**: 3D深度学习、语义分割、模型训练可视化

#### 5. **可视化**
- ✅ Jupyter集成 (`jupyter_visualizer.png`)
- ✅ PBR渲染 (`CloudViewerApp.png`)

**展示内容**: 多平台可视化、Web可视化、高级渲染

## 🎯 设计特点

### 1. **标签页导航**
```
[用户界面] [3D重建] [点云配准] [机器学习] [可视化]
     ↑默认
```

- 鼠标悬停效果：边框变蓝色，向上移动
- 选中状态：渐变蓝色背景，白色文字

### 2. **响应式布局**
- **桌面**: 每行自适应 2-3 个卡片
- **平板**: 每行 2 个卡片
- **手机**: 每行 1 个卡片，标签垂直排列

### 3. **特殊布局**
- **宽版卡片** (`.gallery-item-wide`): 占据 2 列宽度
  - 用于 GIF 动画等宽屏内容
  - 在移动端自动变为单列

### 4. **图片尺寸**
- 统一高度: 300px
- 自适应裁剪: `object-fit: cover`
- 保持图片比例

### 5. **悬停效果**
- 卡片向上移动 10px
- 阴影加深
- 显示半透明遮罩和文字说明

## 📊 图片使用统计

| 图片文件 | 大小 | 使用次数 | 分类标签 |
|---------|------|---------|---------|
| ACloudViewerMainUI.png | - | 1 | 用户界面 |
| CloudViewerApp.png | - | 2 | 用户界面, 可视化 |
| SemanticAnnotation.png | - | 2 | 用户界面, 机器学习 |
| Reconstruction.png | - | 1 | 3D重建 |
| real-time-3D-Reconstruction.png | 897KB | 1 | 3D重建 |
| SenceCloud.png | 3.8MB | 1 | 3D重建 |
| ICP-registration.png | 1.5MB | 1 | 点云配准 |
| AbstractionLayers.png | 1.1MB | 1 | 点云配准 |
| getting_started_ml_visualizer.gif | 1.8MB | 1 | 机器学习 (宽版) |
| jupyter_visualizer.png | 999KB | 1 | 可视化 |

**总计**: 10 个图片文件，12 次展示

## 🎨 CSS 样式要点

### 标签样式
```css
.gallery-tab {
    /* 默认白色背景，灰色边框 */
}

.gallery-tab:hover {
    /* 蓝色边框，向上移动 */
}

.gallery-tab.active {
    /* 渐变蓝色背景，白色文字 */
}
```

### 宽版卡片
```css
.gallery-item-wide {
    grid-column: span 2;  /* 占据2列 */
}
```

### 响应式
```css
@media (max-width: 768px) {
    .gallery-grid {
        grid-template-columns: 1fr;  /* 单列 */
    }
    .gallery-item-wide {
        grid-column: span 1;  /* 宽版也变单列 */
    }
}
```

## 🔧 JavaScript 功能

### 标签切换逻辑
```javascript
galleryTabs.forEach(tab => {
    tab.addEventListener('click', () => {
        // 1. 获取标签对应的画廊ID
        const gallery = tab.getAttribute('data-gallery');
        
        // 2. 移除所有active类
        galleryTabs.forEach(t => t.classList.remove('active'));
        galleryContents.forEach(c => c.classList.remove('active'));
        
        // 3. 添加active类到点击的标签和对应内容
        tab.classList.add('active');
        document.getElementById(`gallery-${gallery}`).classList.add('active');
    });
});
```

## 🧪 测试清单

- [ ] 点击每个标签，确认内容正确切换
- [ ] 检查所有图片是否正常显示
- [ ] 测试悬停效果（卡片和标签）
- [ ] 测试图片灯箱效果（点击放大）
- [ ] 在不同屏幕尺寸测试响应式布局
- [ ] 确认 GIF 动画正常播放
- [ ] 测试移动端标签垂直排列

## 📝 未来扩展建议

### 1. 添加过滤器
```html
<div class="gallery-filters">
    <button data-filter="all">全部</button>
    <button data-filter="pointcloud">点云</button>
    <button data-filter="mesh">网格</button>
    <button data-filter="rendering">渲染</button>
</div>
```

### 2. 添加搜索功能
```html
<input type="text" class="gallery-search" placeholder="搜索应用...">
```

### 3. 添加更多应用领域
- **工业检测**: 工件扫描、质量控制
- **建筑建模**: BIM、建筑测量
- **自动驾驶**: 车载激光雷达、路径规划
- **医疗影像**: CT/MRI 3D重建、手术规划
- **文物保护**: 文物数字化、虚拟博物馆

### 4. 添加视频演示
```html
<div class="gallery-item">
    <video autoplay loop muted>
        <source src="videos/demo.mp4" type="video/mp4">
    </video>
</div>
```

### 5. 添加案例详情页
每个应用卡片可以链接到详细案例页面，包含：
- 详细描述
- 技术参数
- 使用步骤
- 下载示例数据

## 🌐 访问体验

### 桌面端
1. 访问主页，滚动到"应用展示"区域
2. 看到5个分类标签，默认显示"用户界面"
3. 点击其他标签查看不同领域应用
4. 鼠标悬停卡片查看详细说明
5. 点击卡片放大查看

### 移动端
1. 标签自动变为垂直排列
2. 卡片自动单列显示
3. 点击切换标签
4. 滑动查看更多卡片

## 💡 维护提示

### 添加新图片
1. 将图片放到 `doc/images/` 目录
2. 编辑 `doc/index.html`，在对应分类中添加卡片
3. 更新 `doc/script.js` 的预加载列表

### 添加新分类
1. 在 `gallery-tabs` 中添加新标签按钮
2. 创建对应的 `gallery-content` 区域
3. 添加图片卡片

### 修改分类名称
1. 修改标签按钮文本
2. 保持 `data-gallery` 属性与内容 ID 一致

---

**更新时间**: 2025-01-10  
**版本**: 2.0 (分类标签版)  
**作者**: ACloudViewer Team

