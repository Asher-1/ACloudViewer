# CMake版本管理说明

## 概述

为了避免在多个CMakeLists.txt文件中硬编码CMake版本号，我们提供了统一的版本管理方案。

## 文件结构

```
cmake/
├── CMakeVersionConfig.cmake    # 版本配置文件
└── README_CMakeVersionManagement.md  # 本文档

scripts/cmake/
├── update_cmake_versions.py    # 批量更新脚本
├── migrate_to_version_system.py # 迁移到版本管理系统脚本
├── fix_cmake_minimum_required_position.py # 修复cmake_minimum_required位置脚本
├── optimize_cmake_includes.py  # 优化CMake文件，移除不必要的include
├── replace_hardcoded_versions.py # 替换硬编码版本号为变量
├── cleanup_empty_lines.py      # 清理多余空行
├── fix_externalproject_cmake_version.py # 修复ExternalProject版本兼容性
└── adjust_cmake_policy_position.py # 调整版本参数位置
```

## 版本配置

### 1. 版本变量

在 `cmake/CMakeVersionConfig.cmake` 中定义了以下变量：

- `CLOUDVIEWER_CMAKE_MINIMUM_VERSION`: 主项目最低版本 (3.19)
- `CLOUDVIEWER_SUBPROJECT_CMAKE_MINIMUM_VERSION`: 子项目最低版本 (3.10)
- `CLOUDVIEWER_PLUGIN_CMAKE_MINIMUM_VERSION`: 插件最低版本 (3.10)
- `CLOUDVIEWER_THIRDPARTY_CMAKE_MINIMUM_VERSION`: 第三方库最低版本 (3.10)

### 2. 使用方式

#### 方式1：使用函数（推荐）

```cmake
# 在子项目的CMakeLists.txt中
include(cmake/CMakeVersionConfig.cmake)
set_subproject_cmake_minimum_required()

project(MySubProject)
# ... 其他内容
```

```cmake
# 在插件的CMakeLists.txt中
include(cmake/CMakeVersionConfig.cmake)
set_plugin_cmake_minimum_required()

project(MyPlugin)
# ... 其他内容
```

#### 方式2：使用变量

```cmake
# 在CMakeLists.txt中
include(cmake/CMakeVersionConfig.cmake)
cmake_minimum_required(VERSION ${CLOUDVIEWER_SUBPROJECT_CMAKE_MINIMUM_VERSION})

project(MyProject)
# ... 其他内容
```

## 批量更新脚本

所有CMake相关的脚本都位于 `scripts/cmake/` 目录下，详细说明请参考 [scripts/cmake/README.md](scripts/cmake/README.md)。

### 1. 版本更新脚本 (update_cmake_versions.py)

```bash
# 检查哪些文件缺少cmake_minimum_required
python scripts/cmake/update_cmake_versions.py check

# 为缺少cmake_minimum_required的文件添加版本声明
python scripts/cmake/update_cmake_versions.py add 3.10

# 更新所有文件的cmake_minimum_required版本
python scripts/cmake/update_cmake_versions.py update 3.10
```

**功能:**
- **check**: 检查哪些文件包含 `project()` 但缺少 `cmake_minimum_required`
- **add**: 为缺少的文件添加 `cmake_minimum_required`
- **update**: 更新现有文件的版本号

### 2. 版本系统迁移脚本 (migrate_to_version_system.py)

```bash
# 检查哪些文件可以迁移到新的版本管理系统
python scripts/cmake/migrate_to_version_system.py check

# 将硬编码的cmake_minimum_required迁移到新的版本管理系统
python scripts/cmake/migrate_to_version_system.py migrate
```

**功能:**
- **check**: 检查哪些文件有硬编码的 `cmake_minimum_required` 但未使用新的版本管理系统
- **migrate**: 将硬编码的版本声明替换为新的版本管理系统调用

### 3. 位置修复脚本 (fix_cmake_minimum_required_position.py)

```bash
# 检查哪些文件的cmake_minimum_required位置不正确
python scripts/cmake/fix_cmake_minimum_required_position.py check

# 修复cmake_minimum_required的位置
python scripts/cmake/fix_cmake_minimum_required_position.py fix
```

**功能:**
- **check**: 检查哪些文件的 `cmake_minimum_required` 不在文件开头
- **fix**: 将 `cmake_minimum_required` 移动到文件开头（在注释之后）

### 4. 优化脚本 (optimize_cmake_includes.py)

```bash
# 检查哪些文件可以优化
python scripts/cmake/optimize_cmake_includes.py check

# 优化文件，移除不必要的include和函数调用
python scripts/cmake/optimize_cmake_includes.py optimize
```

**功能:**
- **check**: 检查哪些文件有不必要的include或函数调用
- **optimize**: 移除不必要的include和函数调用，使用全局变量

### 5. 版本替换脚本 (replace_hardcoded_versions.py)

```bash
# 检查哪些文件有硬编码版本号
python scripts/cmake/replace_hardcoded_versions.py check

# 替换硬编码版本号为变量
python scripts/cmake/replace_hardcoded_versions.py replace
```

**功能:**
- **check**: 检查哪些文件有硬编码的版本号
- **replace**: 将硬编码版本号替换为对应的变量

### 6. 空行清理脚本 (cleanup_empty_lines.py)

```bash
# 检查多余空行
python scripts/cmake/cleanup_empty_lines.py check

# 清理多余空行
python scripts/cmake/cleanup_empty_lines.py cleanup
```

**功能:**
- **check**: 检查哪些文件有多余的空行
- **cleanup**: 清理多余的空行，保持代码整洁

### 7. ExternalProject版本修复脚本 (fix_externalproject_cmake_version.py)

```bash
# 检查需要修复的文件
python scripts/cmake/fix_externalproject_cmake_version.py check

# 修复版本兼容性问题
python scripts/cmake/fix_externalproject_cmake_version.py fix
```

**功能:**
- **check**: 检查哪些ExternalProject_Add需要添加版本参数
- **fix**: 为ExternalProject_Add添加 `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` 参数

### 8. 参数位置调整脚本 (adjust_cmake_policy_position.py)

```bash
# 检查需要调整位置的文件
python scripts/cmake/adjust_cmake_policy_position.py check

# 调整参数位置到CMAKE_ARGS第一行
python scripts/cmake/adjust_cmake_policy_position.py adjust
```

**功能:**
- **check**: 检查哪些文件的版本参数不在第一行
- **adjust**: 将版本参数移动到CMAKE_ARGS的第一行

## 最佳实践

### 1. 版本选择原则

- **主项目**: 使用最新的稳定版本 (3.19)
- **子项目**: 使用兼容的最低版本 (3.10)
- **插件**: 使用兼容的最低版本 (3.10)
- **第三方库**: 使用兼容的最低版本 (3.10)

### 2. 升级流程

1. 更新 `cmake/CMakeVersionConfig.cmake` 中的版本变量
2. 运行批量更新脚本：
   ```bash
   python scripts/update_cmake_versions.py update <new_version>
   ```
3. 测试构建是否正常

### 3. 迁移流程

如果您有硬编码的 `cmake_minimum_required` 文件，可以迁移到新的版本管理系统：

1. 检查可迁移的文件：
   ```bash
   python scripts/cmake/migrate_to_version_system.py check
   ```
2. 执行迁移：
   ```bash
   python scripts/cmake/migrate_to_version_system.py migrate
   ```
3. 验证迁移结果

### 4. 新文件创建

创建新的CMakeLists.txt文件时，建议使用变量方式：

```cmake
cmake_minimum_required(VERSION ${CLOUDVIEWER_SUBPROJECT_CMAKE_MINIMUM_VERSION})

project(NewProject)
# ... 项目内容
```

**注意**: 由于变量是全局可见的，不需要在每个文件中都包含 `cmake/CMakeVersionConfig.cmake`。

## 优势

1. **统一管理**: 所有版本号集中在一个文件中
2. **易于维护**: 升级时只需修改配置文件
3. **类型区分**: 不同用途的项目可以使用不同的最低版本
4. **自动化**: 提供脚本进行批量操作
5. **向后兼容**: 保持与现有CMake版本的兼容性
6. **全局可见**: 变量在顶层CMakeLists.txt中定义后，所有子项目都可以直接使用
7. **减少冗余**: 不需要在每个文件中都包含版本配置文件

## 重要注意事项

### 1. cmake_minimum_required 位置要求

**⚠️ 重要**: CMake要求 `cmake_minimum_required` 必须在文件的最开始，不能在任何其他命令（如 `include`）之后。

**正确格式:**
```cmake
cmake_minimum_required(VERSION 3.10)

# 其他命令
include(cmake/CMakeVersionConfig.cmake)
project(MyProject)
```

**错误格式:**
```cmake
include(cmake/CMakeVersionConfig.cmake)  # ❌ 错误！
cmake_minimum_required(VERSION 3.10)     # 必须在开头
```

### 2. 其他注意事项

1. 确保在包含版本配置之前不要调用 `cmake_minimum_required`
2. 版本号应该遵循语义化版本规范
3. 升级版本时要充分测试所有构建场景
4. 第三方库的版本要求可能需要单独考虑
5. 使用位置修复脚本确保所有文件都符合CMake规范 