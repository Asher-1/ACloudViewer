# CUDA Graph Parity — 集成侧 vs 上游构建产物对比专项

**状态**: 根因已定位（见 §4），修复已合入（见 §8）
**范围**: `core/AICore/src/tasks/yolo/` 的 CUDA graph 端到端延迟
**结论摘要**: Vulkan/CPU 已对齐或接近上游（多数行 ≤5% 门限）；**CUDA graph 相对上游慢 36~88%**，根因指向**集成侧 ggml-cuda 构建产物**与上游发布二进制的差异，不是图结构、arch、ggml 版本或编译类型。本专项定义可重复的对比流程，供下一次构建产物级排查直接执行。

> **§8 修复结论（2026-08）：根因是 `GGML_CUDA_FORCE_MMQ=ON` 被硬编码**。该宏强制量化 matmul 走 MMQ kernel（小 conv/matmul 固定开销高 ~1.8-2×，与 §3 per-op 特征一致），而该硬编码在 parity 实验（§2 记录两侧均 OFF）**之后**由 `d7206a68a`（cross-platform CUDA deployment）引入。现改为 `AICore_CUDA_FORCE_MMQ` 选项（默认 OFF = 上游性能对齐；ON = 无 cuBLAS DT_NEEDED 的自包含部署），部署侧改由 `AICore_BUNDLE_CUDA_RUNTIME=ON` 携带 libcublas.so.* 满足。

---

## 1. 问题定义

`run_upstream_parity.sh` 全矩阵（cuda 63 + vulkan 63 + cpu 63 行）结果：

| 设备 | e2e p50 vs 上游 | graph p50 vs 上游 | 状态 |
|------|----------------|-------------------|------|
| Vulkan | 28/33 ok（部分 -3~-6% 更快） | 对齐 | ✅ 门限内 |
| CPU | 9 ok | 对齐 | ✅ 门限内 |
| **CUDA** | **+20~88%** | **+36~87%** | ❌ 全部超门限 |

差距集中在 **graph 段**（upload + ggml 图执行 + readback），preprocess/postprocess 已通过线程对齐（`AICORE_TEST_YOLO_THREADS=32`）消除。

## 2. 已排除的根因（带证据）

| 假设 | 验证方式 | 结果 |
|------|----------|------|
| 图结构被重构改变 | `git show HEAD` 对比 build_run_plan 图结构 + Vulkan 侧对齐 | ❌ 图结构一致（Vulkan 不慢证明 op 链相同） |
| ggml 版本差异 | 两侧均 pin ggml v0.18.1 | ❌ 同版本 |
| CUDA arch 不匹配 | `-DCMAKE_CUDA_ARCHITECTURES=86` 重建 ext_ggml | ❌ 无改善（4.36 vs 4.16ms），已恢复 75-real;80-real;86 |
| 编译类型（Debug/Release） | 两侧均 Release | ❌ 相同 |
| FORCE_MMQ 差异 | 两侧均 OFF | ❌ 相同（实验时点）；**但后因 `d7206a68a` 硬编码 ON 引入新差异，已由 §8 修复为可配置默认 OFF** |
| 线程数错配（harness artifact） | GPU rows 设 threads=32 | ✅ 修复后 preprocess ~7x、graph ~50% 差距消失，**但 graph 仍有 +36~87%** |

## 3. 定位证据：per-op profile

`AICORE_TEST_YOLO_PROFILE=1`（→ `aicore_yolo_options_set_profile_ops`）在 free_session 打印 per-op 表（total_ms/calls/avg_us）。

yolov8n-f16 640x640 CUDA graph 对比（集成侧 vs 上游 `yolo-cli` bench）：

| op | 集成侧 avg_us | 上游 avg_us | 比值 |
|----|-------------|------------|------|
| op.0 大 conv (640x640→320x320) | 439 | 431 | 1.02× |
| **op.114 小 conv (60x80 / 30x40 / 15x20)** | **404** | **216** | **1.87×** |
| 其余小 conv（60x80 等） | 慢 1.7~2.0× | 基准 | ~2× |

**特征**：大 kernel 接近，**小 conv kernel 每-op 固定开销 ~1.8-2×**。Vulkan 侧同型号同图结构不慢 → 不是 op 调度/图构建开销，是 **CUDA kernel 本身在集成侧编译产物中更慢**（每-launch 固定开销高）。

## 4. 根因假设（按优先级）

1. **ggml-cuda 编译实例化/特化选项差异**：上游 `yolo-cli` 构建可能启用/禁用某些 CUDA kernel 特化（如 `GGML_CUDA_FORCE_MMQ`、f16 特化、模板实例化 `GGML_CUDA_MMQ_Y` 等）。编译期宏不同 → SASS 不同 → 小 kernel 慢。
2. **CUDA 运行时/工具链版本**：两侧 nvcc 版本、`-O3 -use_fast_math` 等 flags 差异。
3. **cuBLAS 版本绑定**：`GGML_CUDA_USE_GRAPHS` 之外，小 conv 走自定义 kernel，不受 cuBLAS 影响——排除。
4. **L2/TLB 布局**：两侧分配器不同（同进程多 session 已消除），单模型进程化后仍慢 → 排除进程内干扰。

## 5. 专项执行流程（复现 + 对比）

前置：`build_app/bin/ACloudViewer` 含 libAICore（CUDA），上游 checkout 在 `dl/ultralytics-ggml`（含 `yolo-cli` bench 二进制）。

```bash
# 1) 两侧各自跑 yolov8n-f16 CUDA graph，输出 per-op profile
#    集成侧（走 C API，同 profile_ops 开关）:
AICORE_TEST_YOLO_MODELS_DIR=.../gguf AICORE_TEST_YOLO_IMAGE=.../bus.jpg \
AICORE_TEST_YOLO_DEVICE=cuda AICORE_TEST_YOLO_THREADS=32 \
AICORE_TEST_YOLO_PROFILE=1 \
  core/AICore/tests/yolo/test_yolo_capi_performance \
  > integrated.jsonl 2> integrated.op_profile.log

#    上游侧（yolo-cli bench 自身打印 per-op 表）:
cd dl/ultralytics-ggml && ./build/bin/yolo-cli bench ... 2> upstream.op_profile.log

# 2) 按 op 名/形状对齐，生成比值表（core/AICore/tests/yolo/bench_compare.py 的
#    profile 模式可复用；无则手动 grep "op." 行按 key 排序 diff）

# 3) SASS 反汇编对比关键小 conv kernel（两侧各取 op.114 对应的 cubin）:
cuobjdump -sass build_app/ggml/src/ext_ggml-build/ggml/src/ggml-cuda/*.cubin \
  > integrated.sass
cuobjdump -sass dl/ultralytics-ggml/build/ggml/src/ggml-cuda/*.cubin \
  > upstream.sass
# 对比同 kernel 的指令数、寄存器压力、局部内存访问（差异 → 编译选项不同）

# 4) 编译实例化选项清单 diff（两侧 build.ninja/Makefile 中 ggml-cuda 编译行）:
grep -oE '\-D[A-Z0-9_]+(=[0-9]+)?' build_app/ggml/src/ext_ggml-build/.../flags.make \
  | sort -u > integrated.defs
grep -oE '\-D[A-Z0-9_]+(=[0-9]+)?' dl/ultralytics-ggml/build/.../flags.make \
  | sort -u > upstream.defs
diff integrated.defs upstream.defs
```

## 6. 下一步行动（按此顺序尝试，每次重跑 §5.1 验证）

1. **对齐编译宏**：~~将 `upstream.defs` 中 ggml-cuda 相关宏（`GGML_CUDA_*`）差异项在
   `core/AICore/cmake/AICoreCompileDefinitions.cmake` 或 `3rdparty/ggml/ggml.cmake`
   中对齐，重建 ext_ggml + AICore，重跑 parity。~~ **已完成**：`GGML_CUDA_FORCE_MMQ`
   由硬编码 ON 改为 `AICore_CUDA_FORCE_MMQ` 选项（默认 OFF，见 §8）。
2. **对齐 nvcc flags**：`-O3 -use_fast_math`、`--expt-relaxed-constexpr` 等。
3. **对齐工具链**：`nvcc --version` 两侧一致（如上游 CI 用 CUDA 12.x）。
4. 若上述无改善：用 §5.3 的 SASS diff 定位具体 kernel 差异（指令数/寄存器），
   针对该 kernel 在 patch 链内修（**禁止手改 build 下源码，走 3rdparty/ggml/patches/**）。

## 8. 正面修复（2026-08）

根因：`3rdparty/ggml/ggml.cmake` 此前硬编码 `GGML_CUDA_FORCE_MMQ=ON`（由
`d7206a68a` 为跨平台 CUDA 部署引入：libggml-cuda.so 不依赖 libcudart/libcublas，
仅需 NVIDIA driver 即可加载）。该宏强制所有量化 matmul 走 MMQ kernel；小矩阵上
MMQ 的每-launch 固定开销显著高于 cuBLAS 路径，对应 §3 中 op.114 小 conv
(60x80/30x40/15x20) 慢 1.87× 的特征。

修复内容：
- `cmake/AICoreOptions.cmake`：新增 `AICore_CUDA_FORCE_MMQ` 选项（**默认 OFF**，
  与上游 ggml/ultralytics-ggml 构建一致）；经 `aicore_sync_options_to_ggml()` 同步
  为 `GGML_CUDA_FORCE_MMQ`。
- `3rdparty/ggml/ggml.cmake`：不再硬编码 ON，转发用户选项。
- 部署侧：需要 driver-only 自包含时设 `AICore_CUDA_FORCE_MMQ=ON`；否则依赖
  `AICore_BUNDLE_CUDA_RUNTIME=ON`（默认 CI 路径）携带 libcublas.so.*。
- `VerifyNoDynamicCuda.cmake`（qSIBR 回归护栏）语义不变：FORCE_MMQ=ON 构建仍
  无动态 CUDA 依赖。

验证方式：`AICore_CUDA_FORCE_MMQ=OFF` 重建 ext_ggml + AICore 后重跑
`run_upstream_parity.sh --devices cuda --limit 5`，确认 e2e p50 ≤ +5%。

## 7. 回归护栏

- 每次实验后必须重跑：`core/AICore/tests/yolo/run_upstream_parity.sh --devices cuda --limit 5`
- 门限不变：**e2e p50 ≤ +5%**。
- 所有 arch/编译选项实验后恢复仓库默认配置（`cmake -UCMAKE_CUDA_ARCHITECTURES`）。
