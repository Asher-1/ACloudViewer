---
name: acloudviewer-aicore-plugin
description: ACloudViewer AICore 插件集成指南。为新任务添加 C-API、实现 ggml 推理、对接 CMake 构建系统、编写 contract 测试的完整规范。适用于新建 AICore 推理任务或集成新插件。
---

# ACloudViewer AICore Plugin Integration Guide

本 skill 聚合了 `core/AICore/` 接入 ACloudViewer 插件生态的全部规范。当需要为 AICore 添加新推理任务（task）或集成新的 AICore 依赖插件时，依次参考以下章节。

底层规则参考：`.agents/rules/acloudviewer-ggml-aicore.mdc`（ggml 构建）、`.agents/rules/acloudviewer-plugin-dev.mdc`（插件架构）。

---

## 1. AICore C-API 设计规范

### 参数封装

函数入参超过 6 个必须封装为 struct。参考本次重构的案例：

| 重构前（长参数列表） | 重构后（struct 封装） |
|---|---|
| `aicore_depth_depth_dense(ctx, path, *out_w, *out_h, **out_depth, **out_conf, **out_sky, *ext, *intr, *is_metric)` | `aicore_depth_depth_dense(ctx, path, &aicore_depth_dense_result)` |
| `aicore_gaussian_tree_overlap(pairs, n, geom, opacity, block, overlap, max_levels, spacing, cap, **out, *n_out, *nodes_out)` | `aicore_gaussian_tree_overlap(pairs, n, geom, opacity, &merge_opts, ...)` |

规则：
- 输出数据统一放入 result struct，附带 `_result_free` 释放函数
- 配置/选项参数统一放入 options struct，builder 模式设值
- 头文件只暴露不透明句柄（`struct aicore_<task>_ctx`），实现细节永远隐藏在 `.cpp` 中

### ABI 版本管理

每个任务必须定义 `aicore_<task>_abi_version(void)`：

```c
AICORE_CAPI int aicore_<task>_abi_version(void);  // bump on breaking ABI change
```

破坏性变更（参数签名变化、struct 字段变化、删除函数）必须递增版本号。返回值在 contract 测试中验证。

### 内存所有权约定

统一释放入口：

```c
AICORE_CAPI void aicore_<task>_free_buffer(void* p);  // 唯一释放函数
```

禁止导出多个不同命名的释放函数（如 `free_string`、`free_floats`、`free_bytes` 等）。每一个返回 malloc'd 内存的 API 都必须在 doc 中注明"free with aicore_<task>_free_buffer"。

### 错误处理

```c
// ctx 中存储错误信息
AICORE_CAPI const char* aicore_<task>_last_error(const aicore_<task>_ctx* ctx);
// C-API 返回 int：0=成功, -1=错误
AICORE_CAPI int aicore_<task>_do_something(aicore_<task>_ctx* ctx, ...);
```

内部实现：`ctx->last_error = "reason";`，C-API 返回 -1。禁止直接 `fprintf` 或 `printf` 错误信息到 stderr（见日志规则）。

### 头文件输出函数名唯一性

`check_capi_coverage.py` 通过正则 `\b(aicore_[a-z0-9_]+)\s*\(` 匹配所有公开 API。命名风格：`aicore_<task>_<verb>_<noun>`。

---

## 2. 配置显式性规则（禁止 getenv/setenv 逻辑控制）

**所有流程控制开关必须通过 options struct 显式传参，禁止在 pipeline 中用 getenv/setenv 读取环境变量控制逻辑。**

### 为什么

环境变量是全局隐式状态：

- 流程不清晰：调用方看不到推理链被什么开关影响（调试开关藏在环境里）
- 测试不可复现：同一二进制在不同 shell 环境行为不同
- 并发不安全：一个线程 setenv 影响所有线程
- 迁移遗留：上游仓库常用 env 藏调试开关，直接移植会把隐式状态带进 AICore

### AICore 已有先例

- depth：历史 `DA_FUSED` / `DA3_FORCE_JOINT_MV` / `DA_PROFILE` 环境变量 → `aicore_depth_options_set_fused_graph` / `_set_force_joint_multiview` / `_set_profile_logging`。ABI 注释明确 "AICore reads no environment variables for logic control"。
- rmbg：上游 `RMBG_VULKAN_MODE` / `RMBG_STRICT_MATH` / `RMBG_VULKAN_*` → `aicore_rmbg_options_set_math_profile` 等 setter（头文件注明 "Replaces the ... environment variables of the upstream port"）。

移植上游代码时，遇到 `getenv("XXX")` 必须改为 options 字段 + setter，并在头文件注释中记录"此 setter 取代上游 XXX 环境变量"。

### 例外（仅两处允许，新代码禁止第三种）

1. `ggml_env_bridge.cpp`：ggml 库在 init 时 snapshot 环境变量（如 `CUDA_VISIBLE_DEVICES`），AICore 必须把 options 翻译成 ggml 期望的 env。这是 ggml 的硬性接口要求，且集中在唯一桥接文件。
2. `CLOUDVIEWER_DATA_ROOT`：跨插件共享的数据根目录（路径配置，非流程开关）。

---

## 3. 资源管理规则

### shutdown 实现

`aicore_<task>_shutdown()` 必须执行真实清理，不得为空实现。yolo/rmbg 的修正先例：

```cpp
AICORE_CAPI void aicore_yolo_shutdown(void) {
    aicore::runtime::purge_inactive_backend_leases();
}
```

`purge_inactive_backend_leases()` 在 `ggml_backend_registry` 中清理过期 backend lease。

### 权重 Host 副本管理

设计模式：支持惰性释放 + 按需重读（参考 yolo P2b）：

```cpp
int aicore_yolo_release_host_weights(ctx);  // 释放 host 副本（device 权重不受影响）
int aicore_yolo_ensure_host_weights(ctx);   // 从 GGUF 文件按偏移 fseek+fread 重读
```

实现要点：
- `HostTensor` 记录 `file_type` / `file_offset`（GGUF 文件中的原始类型和偏移）
- `prepare_host_weights` 提取为可重入函数（Vulkan Q8→F16、CUDA F32→F16 等转换幂等）
- `build_run_plan` 中 `if (!s->wbuf)` 保证权重只上传一次，重建图复用 wbuf 不触碰 host 数据
- **推理结束可以安全释放 host 权重，不释放 device 权重**，大幅降低宿主内存峰值

### 图分配器缓冲控制

对于实时场景（video frames），提供 `keep_graph_buffers` 选项（参考 depth P2c）：

```cpp
void aicore_<task>_options_set_keep_graph_buffers(opts, int enabled);
// ON: 保持图内缓冲复用（高水位 VRAM）
// OFF: 每次推理后释放图缓冲（VRAM 峰值 = 单图，适合多视图/一次性任务）
```

---

## 4. 依赖引入规则（复用仓库已有能力）

### 核心原则

集成上游代码时，**先检查仓库是否已有同能力模块，有则复用，禁止重复引入**。引入一个新依赖 = 解决一个具名缺口（不是"上游就是这么写的"）。

### 已确认可复用的能力

| 需求 | 仓库已有能力 | 禁止引入 |
|---|---|---|
| 图像解码（JPEG/PNG 等） | Qt QImage（内置 codecs） | stb_image / 直调 libjpeg |
| 图像编码/保存 | QImage / QPainter | stb_image_write |
| 图像缩放/裁剪 | QImage scaled / copy | stb_image_resize |
| 推理运行时 | ggml（3rdparty ExternalProject） | 第二套 ggml / ONNX Runtime |
| 线性代数 | Eigen（3rdparty） | 自研矩阵库 |
| JSON | jsoncpp（3rdparty） | 自研 parser |
| 日志 | AICORE_LOG_* → CVLog | 私有日志体系 |
| 模型下载 | ecvModelDownloader（CVPluginAPI） | 私有下载器 |
| 摄像头/视频 | video_base（插件共享库） | 私有时钟/解码 |

### stb 案例（真实教训）

face-detect.cpp 上游依赖 stb_image 解码；AICore facedetect 移植时改为 Qt QImage（`image_io.cpp`），注释同步修正。验证命令：

```bash
rg -n "stb_image|STB_IMAGE_IMPLEMENTATION" core/AICore/src plugins/  # qSIBR 除外
```

必须零命中。

### 引入新依赖前的检查清单

1. 仓库（含 `3rdparty/`）是否已有功能等价模块？
2. 已有模块缺什么（格式支持？性能？）——缺什么补什么，而不是整体换一套
3. 新依赖的许可证与 ACloudViewer（根项目 GPL-2.0-or-later、AICore MIT）是否兼容？
4. 新依赖是否引入构建负担（CMAKE 配置、跨平台 patch）？

---

## 5. 编码风格与命名规范（与 AICore 一致）

### 命名

| 类别 | 风格 | 示例 |
|---|---|---|
| 函数/变量 | snake_case | `prepare_host_weights`、`run_dense_impl`、`use_direct_conv` |
| 类型 | PascalCase | `HostTensor`、`ModelDef`、`EngineOptions` |
| 常量 | k 前缀或 UPPER_SNAKE | `kQuantCount`、`QK8_0` |
| C API | `aicore_<task>_<verb>_<noun>` | `aicore_yolo_set_detect_thresholds` |
| 宏 | AICORE_ 前缀 | `AICORE_LOG_WARN`、`AICORE_CAPI` |

### include 使用绝对路径（从模块根开始）

从 `core/AICore/` 根开始的全路径，**禁止 `../` 相对路径和裸文件名**：

```cpp
// 正确
#include "aicore/depth_capi.h"     // 公开头：include/ 根
#include "common/capi_utils.hpp"   // src 内部：src/ 根
#include "tasks/yolo/yolo_common.hpp"

// 错误
#include "../tasks/yolo/yolo_common.hpp"
#include "yolo_common.hpp"
```

### 其他

- C++17，RAII，`std::unique_ptr` + custom deleter，不可复制 session
- 原始指针只表示 non-owning view 或 C ABI opaque handle；所有权在类型/注释中明确
- 不在 C boundary 暴露异常、STL、Qt、OpenCV、ggml 类型
- 用小的 task-specific 类拆分 loader/graph/postprocess，不写几千行单文件
- 不做与周边代码无关的风格重写；clang-format 匹配现有格式

---

## 6. 性能与内存规则（端到端链路）

### 零拷贝原则

- 输入借用：C ABI 接收带 stride 的只读 view，调用期间有效，不取得所有权
- 输出复用：热循环避免临时 vector 分配，用 session scratch 复用容量；小结果 shrink logical size，不每帧 `shrink_to_fit`
- 避免中间物化：不先构造 packed RGB 再做第二次 CHW 转换；热路径无 JSON serialize/parse；不把 DB 图片临时落盘再重解码

### 省内存

- 权重只上传一次（`build_run_plan` 中 `if (!s->wbuf)` 复用 wbuf），推理结束可释放 host 副本（`session_release_host_weights` / `aicore_<task>_release_host_weights`）
- 大 buffer（mask/depth）用 immutable result handle + borrowed view，禁止 queued signal 深拷贝
- mask/depth 按需物化：不每帧生成 N × source_width × source_height 数据

### 省显存

- 多视图/一次性任务：`keep_graph_buffers=OFF`（单图峰值）
- 实时视频：`keep_graph_buffers=ON`（缓冲复用）
- 必要时调用 `aicore_<task>_release_gpu_working_memory` 在视图间释放图缓冲

### 端到端加速

- 视频实时：单 job + latest-wins，队列深度 ≤ 2（running + pending），结果绑定 source frame + generation
- preprocess 直接从 stride-aware view 写持久 CHW staging，一次 upload
- detect 只回读 decoder 所需 tensor；segment 只为 selected detections 物化 mask
- 日志含 model/task/device/stage，不含用户敏感路径

---

## 7. 日志集成规则

所有 AICore 输出必须走 `AICORE_LOG_*` 宏体系，最终经 CVLog 进入 ACloudViewer Console。

### 日志级别

```c
// src/common/aicore_log.hpp 定义的公共级别常量
#define AICORE_LOG_LEVEL_DEBUG 0
#define AICORE_LOG_LEVEL_INFO  1
#define AICORE_LOG_LEVEL_WARN  2
#define AICORE_LOG_LEVEL_ERROR 3

// 使用方式
AICORE_LOG_DEBUG("yolo", "preprocess took %.2f ms", ms);
AICORE_LOG_INFO("depth", "loaded model: %s", name);
AICORE_LOG_WARN("rmbg", "fallback to CPU");
AICORE_LOG_ERROR("gaussian", "OOM during inference");
```

### 禁止行为

- **禁止直打 `fprintf(stderr, ...)`**（`ggml_env_bridge.cpp` 已有先例修正为 `AICORE_LOG_WARN`）
- **禁止在 task 内定义私有日志级别枚举**——必须用公共层 `aicore_set_log_level` / `aicore_log_at`

### 线程局部日志级别

```cpp
thread_local int tls_log_level = AICORE_LOG_LEVEL_INFO;
void aicore_set_log_level(int level);   // 设置当前线程的最低输出级别
int aicore_get_log_level(void);         // 查询当前线程级别
```

yolo 的 `logf` 已委托公共层，新 task 直接使用 `AICORE_LOG_*` 宏即可。

### 底层实现

[`aicore_log.cpp`](../../../core/AICore/src/common/aicore_log.cpp) 在 `AICore_HAS_CVLOG` 条件下走 `CVLog::Print*` 进入 Console；否则 fallback 到 `fprintf(stderr, ...)`。

---

## 8. ggml 修改规则

**绝对禁止直接修改构建目录中的 ggml 源码。** 完整规则见 `acloudviewer-ggml-aicore.mdc`。以下为要点浓缩：

### ggml 版本锁定（v0.18.1）

- ACloudViewer 的 ggml 固定在 **v0.18.1**（`3rdparty/ggml/ggml.cmake` 的 ExternalProject tarball）。**禁止升级或降级 ggml 版本**——所有已有 patch（`rmbg_merged` / `aliked_merged` / `metal_merged` / `msvc_vulkan` / `cpu_all_variants`）都基于该版本生成，版本漂移会让全部 AI 插件一起失效。
- 新增任务所需的最小 patch 必须与现有 patch 链**语义去重**（rmbg/aliked 已提供的功能不得重复引入），按文件/算子拆分为：公共 ggml API、CPU、CUDA、Vulkan、build dependency。

### patch 兼容性要求（不影响其他模块推理）

1. 新 patch 必须在**"现有 manifest 全部应用后"的树**上生成（先完整重放 `manifest.yaml` 再 diff），禁止在原始 tarball 上生成。
2. 三遍 replay 验证：forward replay → reverse replay → 第二次 idempotent replay 必须全部成功：
   ```bash
   rm -f build_app/ggml/src/ext_ggml-stamp/ext_ggml-{install,done}
   cmake --build build_app --target ext_ggml -j4
   ```
3. 回归验证：patch 应用后，**所有其他 AI 模块的 contract 测试必须全绿**（`test_rmbg_capi_contract`、`test_aliked_capi_contract`、`test_depth_capi_contract`、`test_gaussian_capi_contract` 等），证明共享 ggml 未被破坏。
4. 精度约束：新增算子不得改变已有模块的数值路径——只增加新 op 或新 backend 分支，不改变既有 op 的默认行为。
5. 禁止从上游原样追加 patch（如 ultralytics-ggml 的 `0001-yolo-ggml-backend-integration.patch`）——两边 patch 已独立演进，必须语义合并。

### 修改流程

```
1. 在 build_app/ggml/... 中临时修改并验证（仅作试验场）
2. diff -ruN orig/ modified/ > 3rdparty/ggml/patches/<subdir>/0001-描述.patch
3. 在 3rdparty/ggml/patches/manifest.yaml 中注册（顺序重要）
4. rm -f build_app/ggml/src/ext_ggml-stamp/ext_ggml-{install,done}
   cmake --build build_app --target ext_ggml -j4
5. 仅提交 patch + manifest.yaml + 胶水代码，不提交 build*/ggml/ 下的源码
```

### 精度约束

ggml 修改**不可影响推理数值精度**。contract 测试必须包含推理结果验证（不仅验证 ABI）：

- 使用 Q8_0 / F16 量化路径的模型，与全 F32 推理结果比较 PSNR，阈值参考上游仓库
- Vulkan coopmat / CUDA F16 GEMM 等加速路径不得改变数值分布的基本统计量

---

## 9. CMake 集成清单

### AICore 构建开关

```bash
-DAICore_ENABLED=ON              # 主开关，自动启用 GGML_ENABLED
-DAICore_USE_VULKAN=ON/OFF       # Vulkan（Linux/Windows，macOS 强制 OFF）
-DAICore_BUILD_TESTS=ON          # 构建 contract 测试
```

### 插件构建开关

```bash
-DPLUGIN_STANDARD_Q<task>=ON     # 标准插件开关（大写 + _PLUGIN 后缀）
```

CMake 目标命名：`Q<task>_PLUGIN` 全大写（如 `QDA3_PLUGIN`、`QYOLO_PLUGIN`）。

### 常用链接

插件 CMakeLists.txt 模板：

```cmake
AddPlugin(NAME Q<NAME> ...)
target_link_libraries(Q<NAME>_PLUGIN PRIVATE
    CVPluginAPI
    CVPluginStub
    CVCoreLib
)
```

新 AICore 依赖 task：在 `core/AICore/CMakeLists.txt` 注册 `add_subdirectory(src/tasks/<task>)` + `target_sources` + 链接 ggml 库。

---

## 10. 测试规范

### Contract 测试（必须）

每个 C-API 函数至少有 contract 测试，验证：
- ABI 版本返回值匹配
- NULL 入参安全性（不崩溃）
- 创建（load/options_new）→ 使用 → 释放（free）完整生命周期

测试文件命名：`tests/<task>/test_<task>_capi_contract.cpp`

### 覆盖率要求

`python3 core/AICore/tests/check_capi_coverage.py` 确保覆盖率 >= 95%。

该脚本扫描 `include/aicore/*_capi.h` 中所有 `aicore_*` 函数，检查是否被消费者（tests/plugins/libs）调用。新增 API 必须至少被一个 contract 测试引用。

### 推理精度验证

```cpp
// 伪代码：固定输入 → 输出与基线比较
load_model(ctx, "model.gguf");
float* output = run_inference(ctx, fixed_input);
float psnr = compute_psnr(output, reference_output, size);
assert(psnr > 35.0f);  // 阈值因模型而异
free_buffer(output);
```

### 推理性能验证

```cpp
// 计时模板
auto start = now();
for (int i = 0; i < N; ++i) run_inference(ctx, test_input);
double ms = elapsed_ms(start) / N;
assert(ms < baseline_ms * 1.1);  // 不超过基线 1.1x
```

---

## 11. 插件集成代码模板

### C-API 头文件模板

```c
// include/aicore/<task>_capi.h
#pragma once
#include <stddef.h>
#include <stdint.h>
#include "aicore/export.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct aicore_<task>_ctx aicore_<task>_ctx;
typedef struct aicore_<task>_options aicore_<task>_options;

AICORE_CAPI int aicore_<task>_abi_version(void);

// Options builder
AICORE_CAPI aicore_<task>_options* aicore_<task>_options_new(void);
AICORE_CAPI void aicore_<task>_options_free(aicore_<task>_options* opts);
AICORE_CAPI void aicore_<task>_options_set_device(opts, const char* device);
AICORE_CAPI void aicore_<task>_options_set_threads(opts, int n_threads);

// Lifecycle
AICORE_CAPI aicore_<task>_ctx* aicore_<task>_load_opts(const char* gguf, const opts*);
AICORE_CAPI void aicore_<task>_free(aicore_<task>_ctx* ctx);
AICORE_CAPI int aicore_<task>_is_ready(const aicore_<task>_ctx* ctx);
AICORE_CAPI const char* aicore_<task>_last_error(const aicore_<task>_ctx* ctx);
AICORE_CAPI void aicore_<task>_free_buffer(void* p);

// Inference entry points (use struct parameters for complexity > 6 params)
// ...

#ifdef __cplusplus
}
#endif
```

### Worker 类模板

```cpp
// plugins/core/Standard/q<Worker>/src/q<Worker>.cpp
#include "aicore/<task>_capi.h"

class Q<Worker> : public QObject, public ccStdPluginInterface {
    Q_OBJECT
    ...
private:
    aicore_<task>_ctx* m_ctx = nullptr;
    
    bool loadModel(const QString& gguf_path) {
        auto* opts = aicore_<task>_options_new();
        aicore_<task>_options_set_device(opts, "auto");
        m_ctx = aicore_<task>_load_opts(gguf_path.toUtf8().data(), opts);
        aicore_<task>_options_free(opts);
        return m_ctx != nullptr;
    }
    
    void runInference() {
        // ... 调用推理 API ...
        // 资源释放
        aicore_<task>_free_buffer(some_output);
        aicore_<task>_free(m_ctx);
    }
};
```

---

## 12. 测试数据集成规范（use test data 按钮）

新插件必须提供 "Try sample data" 一键测试入口，**复用共享组件 `ecvTestDataRepository`（libs/CVPluginAPI），禁止自研下载器**。

### 数据集选择

| 插件类型 | Dataset | 内容 |
|---|---|---|
| AI 推理（qYOLO/qRFDetr/qRMBG/qDeepLSD） | `ObjectsDetection` | 共享图片/视频 |
| 重建（qDA3） | `Monstree` | 多视图图片 |
| 人脸（qFaceDetect） | `FriendsFaces` | 人脸视频 |

### 按钮规范（跨插件统一外观）

```cpp
m_useTestDataBtn =
        new QPushButton(QStringLiteral("\U0001f9ea  Try sample data"));
m_useTestDataBtn->setToolTip(
        "Load sample images for inference.\n"
        "Downloads on first use, then cached locally.");
m_useTestDataBtn->setStyleSheet(
        "QPushButton { background: #00897b; color: white; font-weight: bold;"
        " border: none; border-radius: 4px; padding: 5px 12px; }"
        "QPushButton:hover { background: #00796b; }"
        "QPushButton:pressed { background: #00695c; }"
        "QPushButton:disabled { background: #b2dfdb; color: #e0f2f1; }");
```

teal 主题（`#00897b`）是所有插件（qDA3/qFaceDetect/qFreeSplatter/qYOLO）的统一外观，禁止自定义颜色。

### 点击流程（onUseTestData 三态）

1. **已提取**：extract 目录存在且 `findDatasetFile` / `getMonstreeImages` 命中 → 直接填充组件并返回
2. **zip 已缓存**：`verifyZipIntegrity(zipPath, expectedMd5, expectedSize)` 通过 → 进度条 + `extractDataset` → 提取完成后填充
3. **无缓存**：`startDownload(kind)` + 信号链（`downloadFinished` → `extractDataset` → 填充）

```cpp
void onUseTestData() {
    using TestDataset = ecvTestDataRepository::Dataset;
    auto& repo = ecvTestDataRepository::instance();
    const TestDataset kind = TestDataset::ObjectsDetection;

    // 1. 已提取：直接填充
    const QString path = ecvTestDataRepository::findDatasetFile(kind, kTestImage);
    if (!path.isEmpty()) { fillComponents(path); return; }

    // 2. zip 已缓存：提取
    const auto info = ecvTestDataRepository::getDatasetInfo(kind);
    if (ecvTestDataRepository::verifyZipIntegrity(
            ecvTestDataRepository::zipPath(kind), info.expectedMd5,
            info.expectedSize)) {
        setTestDataControlsEnabled(false);
        m_progress->setVisible(true);
        repo.extractDataset(kind);
        return;
    }

    // 3. 无缓存：下载（信号链回调中填充）
    m_progress->setRange(0, 100);
    m_progress->setVisible(true);
    repo.startDownload(kind);
}
```

### 自动填充组件

```cpp
// 图片列表 → 路径输入框（分号分隔）
m_inputPath->setText(images.join(";"));
// 视频 → video 输入源
m_liveWidget->setInputSource(YOLOLiveWidget::InputSource::VideoFile);
m_liveWidget->setVideoFilePath(path, false);
```

### 缓存与完整性

- 目录：`~/cloudViewer_data/download/`（zip）+ `~/cloudViewer_data/extract/`（解压后）
- 下载完整性：MD5 + size 校验（`verifyZipIntegrity`），缓存命中不重复下载
- 下载/提取期间禁用按钮 + 进度条/状态标签可见；失败恢复按钮并在日志说明原因

--

## 13. 插件 UI 设计规范（强制）

所有 AICore 插件对话框**必须**复用共享 UI 工具模块 `ecvAICoreUiHelper.h`（`libs/CVPluginAPI/include/`，命名空间 `ecvAICoreUi`），禁止各插件自建重复的本地 helper / 样式表 / 像素常量。这是 8 个现有插件（qDA3/qDeepLSD/qFaceDetect/qLightGlue/qFreeSplatter/qRFDetr/qRMBG/qYOLO）统一优化后的唯一标准，新插件直接照此实现即可达到商业级外观。

### 13.1 共享工具速查

| 能力 | API | 说明 |
|---|---|---|
| DPI 缩放 | `ecvAICoreUi::dpiScaled(px)` | 96-dpi 名义像素 → 当前屏幕 DPI 实际像素；**所有硬编码像素必须经它转换** |
| 间距/边距 | `tabMargins()`, `vSpacing()`, `hSpacing()`, `tightVSpacing()` | 统一紧凑间距，禁止手写 magic number |
| 尺寸常量 | `previewSize()` (96), `slotPreviewSize()` (88), `dbListMaxHeight()` (140), `filePoolMaxHeight()` (120) | 缩略图 / 列表高度，DPI 感知 |
| 标签工厂 | `makeLabel(text)`, `makeHintLabel(text)` | 表单标签（左对齐）+ 灰色提示文字 |
| 按钮工厂 | `makeSampleDataBtn()`, `makeBrowseBtn(text)` | teal 主题样本按钮 + 固定宽度浏览按钮 |
| SpinBox | `setCompactDoubleSpin()`, `setCompactSpin()` | 紧凑固定宽度数值框 |
| 布局工具 | `setupTabLayout()`, `setupFormGrid()`, `tightenGroupBox()`, `styleTabWidget()` | 页面 / 表单网格 / 分组框 / Tab 统一风格 |
| 段落构建器 | `makeRuntimeRow(device, threads)`, `makeDbSection()`, `connectDbToggle()`, `setupProgressSection()`, `makeActionRow()` | Device/Threads 行、DB 折叠区、进度区、按钮行 |

基本模板（`setupUi()` 开头）：

```cpp
#include "ecvAICoreUiHelper.h"

void MyDialog::setupUi() {
    auto* root = new QVBoxLayout(this);
    ecvAICoreUi::setupTabLayout(root);
    root->setSizeConstraint(QLayout::SetNoConstraint);  // 见 13.4
    auto* tabs = new QTabWidget(this);
    ecvAICoreUi::styleTabWidget(tabs);
    // 表单网格：label 列宽 92（两列 label|field 结构）
    auto* grid = new QGridLayout;
    ecvAICoreUi::setupFormGrid(grid, 92);
    // 运行时参数：Device/Threads 一行
    root->addWidget(ecvAICoreUi::makeRuntimeRow(m_deviceCombo, m_threads));
    // DB 折叠区：
    auto* dbToggle = ecvAICoreUi::makeDbSection(nullptr);
    ecvAICoreUi::connectDbToggle(dbToggle, m_dbContentWidget);
    // 进度区（label + progress，默认隐藏）：
    ecvAICoreUi::setupProgressSection(root, m_downloadLabel, m_progress);
    // 按钮行：
    auto* row = ecvAICoreUi::makeActionRow(m_runBtn, m_cancelBtn);
}
```

### 13.2 布局准则

1. **DPI 感知**：`setMinimumSize` / 缩略图尺寸 / 列表高度 / 按钮宽度全部用 `ecvAICoreUi::dpiScaled()`；禁止裸像素。
2. **紧凑**：页面 layout 用 `setupTabLayout()`（margin 4px / spacing 4px）；分组框用 `tightenGroupBox()`（QSizePolicy::Maximum + 紧凑 margin），使分组框贴合内容、不撑大。
3. **表单**：QGridLayout 一律 `setupFormGrid()`（label 列固定宽度、field 列 stretch=1），标签用 `makeLabel()` 保证左对齐垂直居中。
4. **统一外观**：样本数据按钮必须 `makeSampleDataBtn()`（teal `#00897b`）；浏览按钮必须 `makeBrowseBtn()`；数值框必须 `setCompactSpin*()`。禁止自定义色板。
5. **DB 输入区**：用 `makeDbSection()` + `connectDbToggle()` 做折叠区；列表高度限 `[dpiScaled(60), dbListMaxHeight()]`，内部滚动，**展开不得撑大对话框**（见 13.4）。
6. **进度区**：用 `setupProgressSection()`；注意其创建的进度条**默认隐藏**——下载/推理开始处必须显式 `m_progress->setVisible(true)`（qFaceDetect/qLightGlue 的既有模式），否则进度条永远不可见。

### 13.3 输入预览与点击放大（强制）

- preview 控件用 `ecvClickableImageLabel`；**必须调用 `setPreviewImage(img, size)`（或 `setPreviewPixmap`）而非裸 `setPixmap`**——点击放大依赖内部 `m_fullImage`，裸 `setPixmap` 时点击无反应。
- **DB entity 输入（`db://EntityName`）必须同样支持放大**：`setDbImages()` 时把完整分辨率图像存入 item role（`Qt::UserRole + 1`），`updateImagePreview()` 遇到 `db://` 前缀时从 role 取图再 `setPreviewImage`。参考 qDeepLSD / qLightGlue 的既有实现；qDA3（QComboBox）用 `setItemData(idx, img, role)` 存图。
- 目录 / 多文件输入：取第一张图做预览。

```cpp
// setDbImages 中：
item->setData(Qt::UserRole, e.name);
item->setData(kDbFullImageRole, e.preview);  // 完整分辨率图

// updateImagePreview 中：
if (path.startsWith(QLatin1String("db://"))) {
    // 从列表 item 的 kDbFullImageRole 取图 → setPreviewImage
} else {
    img = QImage(path);
}
```

### 13.4 对话框尺寸与自适应（防放大 / 防松散）

1. **主 layout 设 `QLayout::SetNoConstraint`**：QDialog 默认 minimum-size 约束会让窗口跟随内容 minimumSizeHint 自动变大——DB 折叠区展开、tab 切换、状态文本变化都会撑大对话框、显得松散。设 NoConstraint 后窗口尺寸由首次 sizeHint 决定一次，内容变化不再撑大窗口。
2. **首次显示固定尺寸**：`showEvent` 中（`m_firstShow` 标志）同步 `adjustSize()`——Qt 在发 Show 事件前已完成布局，此时 sizeHint 纯净。
3. **Tab 高度管理**（多 tab 且含视频/长表单的对话框，参考 qFaceDetect / qFreeSplatter）：
   - 首次 `showEvent` **同步**测量 `m_baseChrome = height() - tabWidget->height()`（用 minimumSizeHint 差值或延迟 singleShot 都会算错/输给 X11 窗口映射）；
   - tab 切换时 `resize(width, qBound(min, baseChrome + tabContentSizeHint, available-20))`，targetHeight = tabBar + 内容 sizeHint；
   - `ScreenChangeInternal`（跨屏 DPI）重新测量 chrome；
   - 公式严禁混用 minimum-based 与 sizeHint-based 数值（历史教训：每次切 tab 膨胀 230~600px）。
4. **视频预览必须有高度上限**（见 13.5 陷阱 1），否则"自适应"会变成"无限放大"。

### 13.5 常见陷阱（真实 Bug 沉淀）

1. **视频预览无上限 → UI 无限放大（qFreeSplatter 回归）**：
   - 反馈环：每帧 `setPixmap` → QLabel `sizeHint`=pixmap 尺寸 → `QScrollArea(widgetResizable)` 按 sizeHint 增长 widget → preview 实际变高 → 下一帧 pixmap 按更大 label 缩放 → 循环。纯内部收敛，但**外部窗口扰动（WM 微调/DPI/远程桌面）一旦放大就被 1:1 永久保留、永不回落**，表现为"播放视频时整个 UI 不断放大"。
   - 修复：`updatePreviewHeightCap()`（video_base）adaptive 分支必须 `setMaximumHeight(dpiScaled(560))`；`m_faceCaptureScroll->setMaximumHeight(dpiScaled(560))`；`adaptTabWidgetHeight()` 的 contentHeight 再 `min(..., dpiScaled(560))`。上限要 > 16:9 minimum 给 stretch 留弹性，否则重新引入"大空白"（见陷阱 3）。
2. **QBoxLayout 压缩 + setGeometry clamp 导致控件重叠**：外层容器高度被 `setFixedHeight` 锁死时，内容增长（如 videoControlsRow 从隐藏变可见）后布局按比例压缩控件；若某控件有显式 minimumHeight，`setGeometry` 会 clamp 回 minimum，导致后续控件 y 偏移重叠。排查：先检查"布局计算几何"与"setGeometry 后实际几何"是否一致；修复：用 `setMinimumHeight` + Expanding 替代 `setFixedHeight`，内容可见性变化时宿主重新测量。
3. **preview 设 maximumHeight → 大空白**：QVBoxLayout 会把被 cap 截断的剩余空间分给其它 Preferred 控件（input 行/statusLabel 被拉高），出现"大空白"。必须让唯一 stretch 的 preview 独占剩余空间：不设 max（或 max 很大），其他控件保持自然高度。
4. **DB 组件撑大首页 tab**：list 高度无边或对话框跟随 minimumSizeHint 自动变大。修复：list 限高 + 主 layout `SetNoConstraint`（见 13.4）。
5. **preview 点击无放大**：见 13.3——裸 `setPixmap` 或 DB 输入未存 full image。

---

## 14. 检查清单（新任务接入用）

新增一个 AICore task 并配套插件时，逐项确认：

- [ ] `include/aicore/<task>_capi.h` 定义了 `aicore_<task>_abi_version`
- [ ] 所有输出内存统一用 `aicore_<task>_free_buffer` 释放
- [ ] 长参数列表（>6 个入参）封装为 struct
- [ ] options 使用 builder 模式，所有 setter 支持 NULL no-op
- [ ] shutdown 函数执行真实清理（调用 `purge_inactive_backend_leases`）
- [ ] 所有日志走 `AICORE_LOG_*` 宏（无 `fprintf`）
- [ ] ggml 保持 v0.18.1，patch 在现有 manifest 重放后生成，其他模块 contract 测试全绿
- [ ] 提供 "Try sample data" 按钮（teal 样式），复用 ecvTestDataRepository（ObjectsDetection/Monstree/FriendsFaces）
- [ ] **UI 复用 `ecvAICoreUiHelper.h`**（setupTabLayout/setupFormGrid/makeLabel/makeSampleDataBtn/makeBrowseBtn/makeRuntimeRow/makeDbSection/setupProgressSection），无本地重复 helper 与魔法像素
- [ ] **对话框主 layout 设 `SetNoConstraint` + showEvent 首次 adjustSize**（DB 展开/tab 切换不撑大窗口）
- [ ] **preview 用 `ecvClickableImageLabel` + `setPreviewImage`**，DB 输入（`db://`）存 full image 到 item role 并支持点击放大
- [ ] **视频预览有高度上限**（`updatePreviewHeightCap` adaptive 分支 max + scroll max + tab 高度钳制，防反馈环无限放大）
- [ ] 所有像素尺寸经 `ecvAICoreUi::dpiScaled()`，无裸硬编码
- [ ] 无 getenv/setenv 逻辑控制（仅 ggml_env_bridge / CLOUDVIEWER_DATA_ROOT 两处例外）
- [ ] 未引入仓库已有能力覆盖的第三方模块（stb 验证：`rg -n "stb_image" core/AICore/src plugins/` 零命中）
- [ ] include 使用模块根绝对路径（无 `../`、无裸文件名）
- [ ] 命名与 AICore 一致（snake_case 函数、PascalCase 类型）
- [ ] 零拷贝设计：输入借用、缓冲复用、无中间物化
- [ ] 权重只上传一次，host 副本可释放；实时场景 `keep_graph_buffers` 按需设置
- [ ] CMakeLists.txt 在 `core/AICore/CMakeLists.txt` 注册 task
- [ ] `tests/<task>/test_<task>_capi_contract.cpp` 覆盖所有公开 API
- [ ] `python3 core/AICore/tests/check_capi_coverage.py` >= 95%
- [ ] 如果修改 ggml 源码，patch 文件放入 `3rdparty/ggml/patches/` 并注册 `manifest.yaml`
- [ ] 非 ABI 兼容变更递增 `aicore_<task>_abi_version`