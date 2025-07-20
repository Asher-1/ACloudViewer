# CMake 脚本工具集

本目录包含用于管理和优化 ACloudViewer 项目中 CMake 配置的各种脚本工具。

## 脚本列表

### 1. 版本管理脚本

#### `update_cmake_versions.py`
- **功能**: 批量检查并添加缺失的 `cmake_minimum_required` 声明
- **用法**: `python3 update_cmake_versions.py`
- **说明**: 扫描所有 CMakeLists.txt 文件，确保每个文件都有正确的 CMake 最低版本声明

#### `migrate_to_version_system.py`
- **功能**: 将硬编码的 CMake 版本迁移到新的版本管理系统
- **用法**: `python3 migrate_to_version_system.py`
- **说明**: 将 `cmake_minimum_required(VERSION 3.xx)` 替换为使用全局变量的版本

#### `replace_hardcoded_versions.py`
- **功能**: 替换硬编码的 CMake 版本为变量引用
- **用法**: `python3 replace_hardcoded_versions.py`
- **说明**: 将硬编码的版本号替换为 `CMakeVersionConfig.cmake` 中定义的变量

### 2. 格式优化脚本

#### `fix_cmake_minimum_required_position.py`
- **功能**: 修复 `cmake_minimum_required` 的位置
- **用法**: `python3 fix_cmake_minimum_required_position.py`
- **说明**: 确保 `cmake_minimum_required` 是 CMakeLists.txt 文件中的第一个命令

#### `optimize_cmake_includes.py`
- **功能**: 优化 CMake 包含语句
- **用法**: `python3 optimize_cmake_includes.py`
- **说明**: 移除不必要的 `include(cmake/CMakeVersionConfig.cmake)` 语句

#### `cleanup_empty_lines.py`
- **功能**: 清理多余的空行
- **用法**: `python3 cleanup_empty_lines.py`
- **说明**: 移除 CMake 文件中过多的空行，保持代码整洁

### 3. 第三方库修复脚本

#### `fix_externalproject_cmake_version.py`
- **功能**: 修复 ExternalProject_Add 的 CMake 版本问题
- **用法**: `python3 fix_externalproject_cmake_version.py`
- **说明**: 为 ExternalProject_Add 调用添加 `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` 参数

#### `adjust_cmake_policy_position.py`
- **功能**: 调整 CMake 策略版本参数的位置
- **用法**: `python3 adjust_cmake_policy_position.py`
- **说明**: 将 `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` 移动到 `CMAKE_ARGS` 后的第一行

### 4. 字体优化脚本

#### `test_font_optimization.py`
- **功能**: 测试跨平台字体大小优化
- **用法**: `python3 test_font_optimization.py`
- **说明**: 验证不同平台和分辨率下的字体大小优化效果

#### `test_scalebar_font_optimization.py`
- **功能**: 测试ScaleBar字体大小优化
- **用法**: `python3 test_scalebar_font_optimization.py`
- **说明**: 验证ScaleBar在不同平台和分辨率下的字体大小优化效果

#### `test_scalebar_position.py`
- **功能**: 测试ScaleBar位置调整
- **用法**: `python3 test_scalebar_position.py`
- **说明**: 验证ScaleBar在窗口底部居中的位置调整效果

## 跨平台字体大小优化

### 概述

为了解决在不同平台（Windows、macOS、Linux）和不同分辨率显示器上字体显示过大的问题，我们实现了跨平台的字体大小优化系统。

### 核心函数

#### `GetOptimizedFontSize(int baseFontSize = 12)`
- **功能**: 根据平台、DPI缩放和屏幕分辨率优化字体大小
- **参数**: `baseFontSize` - 基础字体大小（默认12）
- **返回**: 优化后的字体大小

#### `GetPlatformAwareDPIScale()`
- **功能**: 获取平台感知的DPI缩放值
- **返回**: 调整后的DPI缩放值（0.5-2.0范围内）

### 平台特定优化

#### macOS
- **Retina显示器**: 使用较小的字体避免过度放大
- **公式**: `fontSize = max(8, baseSize - (dpiScale - 1) * 2)`
- **DPI缩放**: `adjustedScale = 1.0 + (dpiScale - 1.0) * 0.5`

#### Windows
- **高DPI显示器** (DPI > 120): 减小字体大小
- **低DPI显示器** (DPI < 96): 增加字体大小
- **DPI缩放限制**: 最大1.5倍

#### Linux
- **高分辨率** (≥1920x1080): 减小字体大小
- **低分辨率** (<1366): 增加字体大小
- **超高分辨率** (≥2560x1440): 进一步减小缩放

### 分辨率适配

- **2K及以上分辨率**: 减小字体大小
- **1080p分辨率**: 保持适中字体大小
- **低分辨率**: 增加字体大小

### 使用示例

```cpp
// 在 HotZone 构造函数中
if (win) {
    font = win->font();
    double dpiScale = GetPlatformAwareDPIScale();
    int fontSize = GetOptimizedFontSize(12);
    font.setPointSize(fontSize);
    margin *= dpiScale;
    iconSize *= dpiScale;
    font.setBold(true);
}

// 在屏幕中心消息显示中
QFont newFont(s_tools.instance->m_font);
int fontSize = GetOptimizedFontSize(12);
newFont.setPointSize(fontSize);

// 在 ScaleBar 中
textActor = vtkSmartPointer<vtkTextActor>::New();
textActor->SetInput("1 m");
int optimizedFontSize = getOptimizedFontSize(18);
textActor->GetTextProperty()->SetFontSize(optimizedFontSize);
```

### 测试结果

运行 `test_font_optimization.py` 可以查看当前系统的优化效果：

```
操作系统: Darwin
分辨率: Resolution: 3456 x 2234 Retina

DPI缩放: 2.0
  基础大小 12 -> 优化后 10
  基础大小 14 -> 优化后 12
  基础大小 16 -> 优化后 14

DPI缩放处理:
原始DPI缩放:  2.0 -> 调整后:  1.5
```

运行 `test_scalebar_font_optimization.py` 可以查看ScaleBar的优化效果：

```
操作系统: Darwin
分辨率: Resolution: 3456 x 2234 Retina

DPI缩放: 2.0
  原始字体大小: 36
  优化后字体大小: 15
  原始DPI缩放:  2.0
  调整后DPI缩放:  1.6
  优化效果: +21 像素
```

运行 `test_scalebar_position.py` 可以查看ScaleBar的位置调整效果：

```
窗口大小: 1920 x 1080
DPI缩放: 2.0
  比例尺长度: 640 像素
  底部边距: 100.0 像素
  线条位置: (640.0, 100.0) -> (1280.0, 100.0)
  文本位置: (960.0, 60.0)
  居中偏移: 0.0 像素
```

## 使用建议

1. **定期运行**: 建议在修改 CMake 文件后运行相关脚本
2. **版本控制**: 在提交代码前运行 `cleanup_empty_lines.py`
3. **新项目**: 使用 `update_cmake_versions.py` 确保版本声明正确
4. **第三方库**: 使用 `fix_externalproject_cmake_version.py` 修复兼容性问题
5. **字体优化**: 使用 `test_font_optimization.py` 验证字体显示效果

## 注意事项

- 所有脚本都会备份原始文件
- 建议在运行脚本前提交当前更改
- 某些脚本可能需要管理员权限
- 字体优化函数需要 Qt 5.6+ 支持 