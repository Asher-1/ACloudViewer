# Metal 推理性能优化经验（ggml）

本文档汇总了在 Apple Silicon (M2 Max) 上优化 ggml Metal 后端的推理性能时所积累的**可复用模式**与**陷阱清单**。适用场景：接入新模型（YOLO 检测/分割、RF-DETR、depth 等）时，若首次在 Metal 上推理偏慢，可参照本文档系统性地定位并消除瓶颈。

> **核心结论（第一性原理）**：Apple Silicon 的 GPU 算力通过**专用矩阵单元**（simdgroup matrix multiply）释放。如果 kernel 没有走矩阵单元路径（例如标量逐元素循环、每线程元素过少导致 threadgroup 利用率不足），性能会差 10–100 倍。所有优化都围绕这一约束展开。

---

## 1. 定位方法论：用 per-op Profile 找到真瓶颈

### 1.1 不要猜，要测

ggml-metal 自带 **GGML_METAL_OP_PROFILE** 环境变量（需在 `ggml-metal-context.m` 中启用），可逐 op 获取 GPU 执行时间：

```objc
// 在 ggml_metal_graph_compute 的 @autoreleasepool 开头插入 env 门控分支
if (getenv("GGML_METAL_OP_PROFILE") != NULL) {
    static int s_profile_after = 3;  // 第 3 次 compute 后开始（确保 pipeline 缓存）
    if (s_profile_after > 0 && --s_profile_after == 0) {
        id<MTLCommandBuffer> cb = [queue commandBufferWithUnretainedReferences];
        // op_init 9 参数签名：dev, cb, gf, gi, gi+1, false, false, false, 0, 0
        ggml_metal_op_t op = ggml_metal_op_init(ctx->dev, cb, gf, gi, gi + 1,
                                                 false, false, false, 0, 0);
        ggml_metal_op_encode(op, 0);
        ggml_metal_op_free(op);      // 先 free（内部 endEncoding）再 commit
        [cb commit];
        [cb waitUntilCompleted];
        fprintf(stderr, "[op-profile] idx=... src=[%s,%s] k=%lld ... gpu=...us\n", ...);
        return GGML_STATUS_SUCCESS;
    }
}
```

> **Attention**: profile 分支**不要入库**（它是 env 门控的本地调试工具）。patch 中不应包含它。

### 1.2 瓶颈模式识别

从 profile 输出中识别三类根因：

| 模式 | 特征 | 根因 |
|------|------|------|
| **threadgroup 过碎** | op 平均耗时短（<10μs）但数量极大（数百个同类 op），总和大 | 每 threadgroup 处理的元素太少，GPU 核利用率低 |
| **标量路径** | K<64 的 mul_mat 显著慢（0.1 TFLOPS 级） | Apple Matrix Unit 要求 K≥64 才能激活矩阵路径；K 过小时退化为逐元素标量 kernel |
| **带宽饱和** | UNARY / CPY 类 op 达到 ~310 GB/s 时 | kernel 已到硬件带宽极限，进一步优化需减少中间张量流量（图级 fusion） |

---

## 2. 六大可复用 Kernel 优化模式

以下每种模式配有适用条件与实测收益量级（基于 M2 Max 64GB 上的 BiRefNet-Swin-L / YOLOv8 / RF-DETR）。

### 模式 A：Flat-grid 一维化 + 除法消除

**根因**：IM2COL 默认按 `(OC, OH, OW)` 三维排布，当 `[IC, KH, KW]` 很大而 `OC` 很小时（如 conv 的 OC=1），每 threadgroup 只处理 1×9 个元素，GPU 执行单元严重空闲。

**解法**：将多维 grid 塌缩为一维 `total/thread_elements`，线程内用**hw 进位递增**（hw-carry chain）代替逐元素取模除法：

```metal
// 每线程处理 16 个通道（4×float4）
const int total = IC * OH * OW;        // CHW
const int total4 = total >> 2;
const int i0 = idx * 16;               // 4 float4 per thread
for (int j = 0; j < 16; j += 4) {
    int c = (i0 + j) >> 2;             // 进位进位：h*w 递增，c 进位
    int h = c / OW;
    int w = c - h * OW;
    // ... im2col gather
}
```

**适用条件**：`OC * N == 1` 且 `CHW` 是 4 的倍数（float4 对齐）时，收益最大。`CHW` 很大且 16 对齐时更优。

**收益量级**：IM2COL 从 1382ms → 142ms（**-90%**），是本次优化中单 kernel 收益最大的项。

### 模式 B：每线程多元素 + float4 向量化

**根因**：一条 threadgroup 调度（dispatch）有固定开销；逐 1 元素的 kernel 让 GPU 花费大量时间在调度和 threadgroup 管理上。

**解法**：每线程处理 4 个或 16 个元素，用 `float4` / `float16` 向量类型一次加载/存储，grid 缩小到 `1/4` 或 `1/16`。

```metal
// 每线程处理 4 个 float 4（共 16 元素）
const int total = args.ne * ...;
const int idx = tgpig.x * ntg * ... + tiitg;
const int i0 = idx * 16;
// 读取 4 个 float4
float4 v0 = src[i0/4 + 0];
float4 v1 = src[i0/4 + 1];
float4 v2 = src[i0/4 + 2];
float4 v3 = src[i0/4 + 3];
```

**适用条件**：`total` 大（>10000 元素）、带宽受限的 kernel（UNARY、BIN、GET_ROWS）。不适用于计算密集型 kernel（矩阵乘），后者线程数已饱和。

**收益量级**：BIN_OP 从 ~146ms 减少 60%+；GET_ROWS 从 7.2ms → ~2ms（**-72%**）。

### 模式 C：广播 / 行特例消除取模

**根因**：以矩阵按行广播（`ne0 == ne10`）时，逐元素 `i0 % args.ne10` 取模指令 GPU 周期长。

**解法**：提取**广播特例**（`args.ne10 <= 1`：全局标量）和**行对齐特例**（`args.ne10 == args.ne0`：每行相同值），用分支无条件赋值消除除法：

```metal
const bool b_scalar = args.ne10 <= 1;
const bool b_row = args.ne10 == args.ne0;
if (FC_bin_op == 0) {       // ADD
    for (int j = 0; j < 4; ++j) {
        const int i0j = i0 + j;
        const int i10 = b_scalar ? 0 : (b_row ? i0j : i0j % args.ne10);
        d[i0j] = a[i0j] + b[i10];
    }
}
```

**适用条件**：`M=N`（或 `ne0 == ne10`）的二元运算。Metal 的 threadgroup uniform 分支开销接近零（同一 warp 所有线程走同一路径时）。

**收益量级**：BIN_OP 每行减少 1 个整数除法和 1 个取模，综合 -20~50%（与 per-thread 多元素联合使用）。

### 模式 D：K<64 走 F16 矩阵乘门槛放宽

**根因**：Metal 的 simdgroup 矩阵乘要求 K≥64 才激活矩阵单元路径（`kernel_mul_mm`）；K<64 时退化到 matrix-vector 标量 kernel（`kernel_mul_mv_f32_f32`），性能约 **0.1 TFLOPS** 级（矩阵路径的 1/100）。

**解法**：对 `f16×f16` 组合将门槛从 64 放宽到 16：

```cpp
// 在 ggml-metal-ops.cpp 的 get_extra_buffers_mul_mat 或 dispatch 分支中
props_dev->has_simdgroup_mm &&
    ((ne00 >= 16 && op->src[0]->type == GGML_TYPE_F16 && op->src[1]->type == GGML_TYPE_F16)
     || (ne00 >= 64))
    && ne11 > ne11_mm_min
```

> **极其重要的约束**：**只对 f16×f16 放宽**。f32 时 K<64 走矩阵路径会灾难性地慢（从标量路径 38ms 倒退到 2487ms），因为 f32 矩阵路径的 dequantize_f32 开销在小 K 时不可接受。

**适用条件**：模型中存在多个 `[M, K]×[K, N]` 形状且 K 在 [16, 63] 的 f16 权重 GEMM（典型场景：conv 的 K=27 个输出通道、MLP 的 K=32 等）。f16 模型（推荐推理格式）直接受益。

**收益量级**：conv K=27 从 38.4ms → 1.6ms；整个 YOLO v8 推理链从 51.4ms → 14.5ms（**-72%**，包含 IM2COL 加速的累积收益）。

### 模式 E：小 K GEMM 的 F16 权重化（矩阵单元路径锁定）

**根因**：f32 权重在 Metal 中走 `kernel_mul_mm_f32_f32` 时，权重加载经过 `dequantize_f32` 标量路径（逐 4 字节 load 而不是 64 字节 cacheline burst），带宽利用率低，尤其对小 K 的 conv/MLP GEMM 影响大。

**解法**：在 AICore 图构建阶段，对 `use_metal` 路径将 conv 权重和 QKV/Proj 线性层权重指定为 F16：

```cpp
// rmbg_graph.cpp 中
ggml_tensor *w16 = weight_f16(prefix + "weight");  // F16 权重
ggml_tensor *col16 = ggml_im2col(ctx, w16, ..., GGML_TYPE_F16);
// F16 im2col + F16 weight → kernel_mul_mm_f16_f16 矩阵单元路径
```

同时将 F16 im2col 的输出设为 F16（减少写带宽），结合 Metal 图构建的 `metal_f16_gemm` 选项控制。

**适用条件**：模型中有大量小 K（≤256）的 GEMM（conv 的 IC→OC、MLP 的 hidden→4×hidden 等）。大 K GEMM（≥3072）时矩阵路径**已饱和**（~10 TFLOPS），f16 化无收益。

**收益量级**：conv 链 F16 化减少 ~50% 以上（依赖 K 大小）；全模型从 2534ms → 933ms（**-63%**，综合所有优化）。

### 模式 F：图级 Op 融合

**根因**：ggml 的计算图中有大量 reshape/cont/permute 拷贝操作（RMBG 原始图 6461 节点，其中 RESHAPE 1266、CONT 551、PERMUTE 359）。这些操作不增计算量，但搬运大张量数据（[3072, 1024] 级别），累计耗时可观。

**解法**：用自定义 kernel 替代常见多 op 链：

| 融合场景 | 替换前 | 替换后 | 收益 |
|---------|--------|--------|------|
| **SWIN_QKV Layout** | add + 3×(cont + permute + cont) | 1 个 kernel（bias 融合 + 重排写） | -179ms |
| **Flash Attention** | QK matmul + scale + softmax + AV matmul | 1 个 kernel（kernel_flash_attn_ext_f32） | -828ms |
| **Conv + Bias** | im2col + matmul + add + bias | 1 个 kernel（GPU 通用，需自定义 op） | 视情况 |

**适用条件**：图构建阶段可识别（图结构固定时），且新 kernel 的标志重排逻辑与原始拷贝链**比特级等价**（用 output_hash 验证）。

**收益量级**：flash attention 总是有收益（-800~1000ms 级，QKV 重排次数正比）。qkv layout fusion 在有多个 attention block 时每个 block 省 ~3 个 cont+permute。

---

## 3. 图构建决策模式（AICore 层）

### 3.1 Metal 后端检测

**MTL0 陷阱**：`ggml_backend_name()` 返回的是设备名（如 `"MTL0"`），**不是**固定的 `"Metal"` 字符串。必须同时匹配：

```cpp
use_metal = name && (std::strstr(name, "Metal") || std::strstr(name, "MTL"));
```

### 3.2 F16 权重化的条件门控

在 `GraphOptions` 中为 Metal 独立配置：

```cpp
struct GraphOptions {
    bool cuda_f16_gemm = false;       // CUDA 默认 OFF（用户显式 opt-in）
    bool metal_f16_gemm = true;       // Metal 默认 ON（F16 是唯一矩阵单元路径）
};
```

`use_f16_gemm` 的推导：

```cpp
use_f16_gemm =
    (use_cuda_custom && !strict_math && cuda_f16_gemm) ||
    (use_metal && !strict_math && metal_f16_gemm);
```

`strict_math` 模式下禁止 f16 化，保持纯 FP32 精度。

### 3.3 三量化一致性验证

> 在灰度发布前，验证 f32/f16/q8**三量化**的精度一致性（f32/f16 output_hash 应完全一致；q8 因反量化路径不同，hash 不同但 contract 通过）。

```bash
# RMBG contract 测试
AICORE_TEST_RMBG_MODEL=<f16.gguf> test_rmbg_capi_contract
AICORE_TEST_RMBG_MODEL=<f32.gguf> test_rmbg_capi_contract
AICORE_TEST_RMBG_MODEL=<q8.gguf> test_rmbg_capi_contract
# 三量化性能应一致（±1%）——Metal 瓶颈在激活路径，权重量化不影响性能
```

三量化性能一致（±1%）是 Metal 的独特特征（CUDA 上 q8 会更快），原因：瓶颈在**激活路径**（矩阵乘运算本身），而非权重带宽。

---

## 4. 陷阱清单（必须避免）

### 4.1 MSL / C struct 布局一致性（SIGKILL）

**现象**：运行时 SIGKILL (Kill: 9)，无任何错误输出。

**根因**：MSL kernel 侧的 `constant struct kargs_im2col` 包含字段 `OH`/`OW`，但 C 侧（`impl.h`）的对应 struct 不包含这些字段→**MSL 读到的 args.OH/args.OW 是垃圾指针**→GPU 越界访问→GPU reset→OS 发送 SIGKILL。

**修复**：C 侧 `kargs_im2col` 必须**严格匹配** MSL 侧的字段顺序与类型。`sed` 展开模板时也会偏移，用 `#include "impl.h"` 统一声明的结构体。

**预防**：每次在 MSL 侧新增 kernel args 时，同步更新 `impl.h` 的 C 侧 struct。编译 metallib 后立即跑一遍 contract，确认不 SIGKILL。

### 4.2 ExternalProject 增量构建 mtime 陷阱

**现象**：修改某个 patch 文件并 clean install stamp 后重建，`metallib` 没有重新编译。

**根因**：Python 脚本写入 .metal 文件的 mtime 与之前 .o 编译时间同秒 → make 认为源文件 up-to-date → 跳过 metallib 重编。

**修复**：每次 patch 链变化后，必须同时清除 build + install + done stamp：

```bash
rm -f build_app/ggml/src/ext_ggml-stamp/ext_ggml-{build,install,done}
# 极端情况：直接清整个源码目录让 ExternalProject 重新解压
rm -rf build_app/ggml/src/ext_ggml
cmake --build build_app --target ext_ggml -j4
```

### 4.3 ggml_view_4d stride 继承陷阱

**现象**：Flash attention 在大输入（1024²）上输出退化（alpha 值全在 [253,255]）。

**根因**：`ggml_view_4d(src, ne0, ne1, ne2, ne3)` 没有重新计算 nb 数组，而是**继承父张量的 strides**。当用 view 做维度拆分（如将 [3C] 拆成 [hd, heads]）时，继承的 strides 会导致 kernel 读到错误地址。

**预防**：涉及维度拆分的 view 操作需要手动验证 nb 指针。对于 Flash Attention 等 kernel，不要用 strided view 传入，而是先做 `ggml_cont` 确保连续再传。

### 4.4 profile 分支不入库

profile 分支（见 §1.1）是**纯本地调试工具**，包含 `#import "ggml-metal-impl.h"` 等内部头文件，**不要提交到 patch 中**。patch 是面向生产环境的稳定优化，不应包含调试代码。

### 4.5 K<64 全局门槛灾难

**绝对不要**无条件将 K<64 的 mul_mat 全部走矩阵路径。f32 大 M 时 K<64 走矩阵路径会产生灾难性倒退：

```cpp
// 错误的做法：
// (ne00 >= 16 && op->src[0]->type == GGML_TYPE_F32)  // ← 灾难！
// 正确的做法（仅 f16×f16）：
(ne00 >= 16 && op->src[0]->type == GGML_TYPE_F16 && op->src[1]->type == GGML_TYPE_F16)
```

实测：f32 时走 mul_mm 从 2487ms 倒退到 38ms（**倒退 65×**）。根本原因是 f32 矩阵路径的 dequantize_f32 负载在小 K 时极端低效。

---

## 5. 新模型加速检查表

当接入一个新模型（如新的 YOLO 变体、depth 模型、MLP-only 模型）且 Metal 推理偏慢时，按以下顺序检查：

1. **跑 per-op profile** → 识别热点 op 类型
   - MUL_MAT 主导 → 检查 K 分布，确认 K≥64 的走矩阵路径；K<64 的按模式 D 处理
   - IM2COL 主导 → 按模式 A flat-grid 优化（CHW 对齐时最佳）
   - BIN_OP / UNARY / GET_ROWS 主导 → 按模式 B + C 向量化
   - CONT / CPY / PERMUTE 为主 → 考虑图级融合（模式 F）

2. **检查权重类型** → 若是 f32，评估 f16 化收益（模式 E）
   - 小 K GEMM 多 → f16 化大收益
   - 大 K GEMM 多（≥3072）→ 已饱和，仅检查 K<64 门槛

3. **检查 attention block** → 若有，启用 Flash Attention（模式 F）

4. **三量化验证** → f32/f16/q8 各跑 contract，确认精度一致

5. **回归**：全量 contract 通过后，检查 CUDA/Vulkan/CPU 基线是否退化（ggml-metal 代码隔离，不应退化）

---

## 6. 参考

- 合并后的 Metal 优化 patch：`3rdparty/ggml/patches/metal_merged/0001-metal-optimizations.patch`
- AICore 图构建：`core/AICore/src/tasks/rmbg/rmbg_graph.cpp`（Metal 分支参考实现）
- ggml 代码修改规则：`.agents/rules/acloudviewer-ggml-aicore.mdc`