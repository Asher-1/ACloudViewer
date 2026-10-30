# focal 链全确定性化 — 详细设计（方案 C）

状态：**已实施（2026-10-29；落地与验证数据、e2e hash 门禁偏差见 [CUDA_PORTING_NOTES.md](CUDA_PORTING_NOTES.md) 方案 C 落地记录一节）**。目标：消除 kitti_voxelrcnn_focal_multimodal 在
CUDA 后端 0.9978 的精度残差（对上游 C++ 基线），使 focal 链在 CPU/CUDA/
Vulkan 三后端**逐位一致**（超越上游自身 cuda-vs-cpu 的 0.9997）。

## 第一性原理

稀疏子流形 conv 的数学本质：`out[v,o] = bias[o] + Σ_{k<K} Σ_{c<Cin}
feat[idx[v,k], c] · W[k·Cin+c, o]`。ggml mul_mat 把它当 GEMM，归约结合序由
tile/split-K 分派决定（跨实现不同→~2e-7 噪声）→ focal 的 conv_imp sigmoid
过 0.5 硬阈值选 voxel 子集，2945 个位点距阈值 <3e-5 → 个别位点翻转 → 下游
拓扑漂移。**确定性 = 固定加法结合序 + 固定舍入语义**：同一份顺序归约实现，
CPU/GPU 逐位一致（IEEE754 保证：同序 + 同舍入 + 同输入 = 同结果）。

关键约束（来自前轮审计的两条教训）：
1. 仅 conv_imp 确定化无效——每层 main conv 的 mul_mat 输出就是下一层
   imp_features 的输入，噪声逐层传播 → **整条 focal 链（3 层 FocalSparseConv
   的 conv_imp + main conv）都走确定性路径**。
2. fmaf 陷阱：GPU `fmaf`（单舍入）≠ CPU `a*b+c`（双舍入），逐位会差。
   sigmoid 的 `expf` 同理（CUDA ~2ULP vs glibc 0.5ULP）。两侧必须共享同一份
   数学实现。

## 设计

### 1. 确定性稀疏 conv（共享单实现）

邻居索引表（`sparse_indice_map` 展开为 `[n_out, K]` 的行索引，-1=无邻居）
为纯数据移动，无数值影响；权重布局保持 GGUF 的 `[K·Cin, C_out]`（BN 已折
叠进权重+bias）。数值核心只有一个函数，CPU/GPU 共用（模板化 device/host）：

```cpp
// deterministic_sparse_conv.h（CPU/GPU 共享 inline，保证同序同舍入）
template <typename FMA>   // CPU: std::fma; GPU: fmaf —— 均为"单舍入乘加"
inline void det_sparse_conv_rows(const float* feat, int C_in,
                                 const int32_t* idx,        // [n_out, K]
                                 const float* W,            // [K*C_in, C_out]
                                 const float* bias,
                                 float* out, int n_out, int C_out, int K,
                                 FMA fma) {
    for (int v = 0; v < n_out; ++v)
        for (int o = 0; o < C_out; ++o) {
            float acc = bias[o];
            for (int k = 0; k < K; ++k) {              // 归约序：k 外层
                const int r = idx[v * K + k];
                if (r < 0) continue;                   // 零填充：跳过（不引入 -0 差异）
                const float* row = feat + (size_t)r * C_in;
                const float* w   = W + (size_t)(k * C_in) * C_out + o;
                for (int c = 0; c < C_in; ++c)         // c 内层：固定结合序
                    acc = fma(row[c], w[c * C_out], acc);  // 单舍入，两侧一致
            }
            out[(size_t)v * C_out + o] = acc;
        }
}
```

GPU 入口（`openpcdet_cuda_ext.cu` 追加，单编译单元约束不变）：

```cuda
__global__ void det_sparse_conv_kernel(
        const float* __restrict__ feat, int C_in,
        const int32_t* __restrict__ idx,               // [n_out, K]
        const float* __restrict__ W, const float* __restrict__ bias,
        float* __restrict__ out, int n_out, int C_out, int K) {
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    const int o = blockIdx.y * blockDim.y + threadIdx.y;
    if (v >= n_out || o >= C_out) return;
    float acc = bias[o];
    for (int k = 0; k < K; ++k) {
        const int r = idx[v * K + k];
        if (r < 0) continue;
        const float* row = feat + (size_t)r * C_in;
        const float* w   = W + (size_t)(k * C_in) * C_out + o;
        for (int c = 0; c < C_in; ++c)
            acc = fmaf(row[c], w[c * C_out], acc);     // 与 CPU std::fma 同为单舍入
    }
    out[(size_t)v * C_out + o] = acc;
}
// launch: grid=(ceil(n_out/128), ceil(C_out/8)), block=(128,8) —— 线程 (v,o)
// 一一对应，线程内 K*C_in 串行归约；无 shared/warp 归约，无原子操作。
```

`focal_split_voxels` 的 sigmoid/阈值：抽出共享的
`det_sigmoid(x) = 1/(1 + det_expf(-x))`，`det_expf` 用同一份多项式实现
（host/device 同源，禁用 `__expf`）——CPU/GPU 同输入同输出。

### 2. 接线与开关

`run_focal_conv_core`（voxelrcnn.cpp）中 `run_sparse_conv` 的两处调用
（conv_imp、main conv）按 `hp.focal_deterministic`（hparams 私有字段，
gguf_loader 对 focal 模型默认置 true；不新增 public ABI 选项）切换：
- true  → gather 邻居表 → `det_sparse_conv_rows`（CPU 直接调用 / GPU 经
  kernel + D2D 拷回）→ 现有 split/combine 流程原样复用；
- false → 现有 ggml mul_mat 路径原样保留（对照模式，供 A/B 与回归）。
其他 17 模型不读该字段，零影响。确定性路径绕过 ggml 图（直接 kernel 调用），
无 graph-capture 兼容问题。

### 3. 性能预算

KITTI 000008（~2e4 活动 voxel）最重层 conv3：C_out=64、K·Cin=27·64=1728 →
2e4×64 线程 × 1728 串行 FMA ≈ 6ms/层（RTX 3060 估算），3 层 + conv_imp 约
+15–25ms。focal 端到端基线 ~0.8s（上游 REPORT：focal_backbone 占 85%），增
幅 <3%，可接受。focal 非实时场景模型。

### 4. 验证门禁

1. 单元：固定伪随机输入下 `det_sparse_conv_rows` CPU 路径与 GPU kernel
   输出**逐位一致**（memcmp）；与 ggml mul_mat 路径差 ≤2e-6（对照合理性）。
2. 端到端：focal f32 三后端两两 output_hash 一致（CPU=CUDA=Vulkan）；
   vs 上游 cpp_cpu_f32 基线 8/8 matched（mean_iou 预期 ~0.998±，与上游跨实
   现 floor 同量级——确定化消除的是**我方内部**后端间差异，vs 上游的跨实现
   差异由两侧图结构决定，不会消失）。
3. 回归：kitti_pointpillar / kitti_voxelrcnn_car 三后端 output_hash 与修复
   前一致（不受影响）；`test_openpcdet_capi_contract` 通过。

## 改动文件清单

| 文件 | 改动 |
|---|---|
| `core/AICore/src/tasks/openpcdet/extensions/deterministic_sparse_conv.h` | 新增：共享 `det_sparse_conv_rows` 模板 + `det_expf/det_sigmoid` |
| `core/AICore/src/tasks/openpcdet/extensions/deterministic_sparse_conv.cpp` | 新增：CPU 邻居表展开 + host 分发（std::fma 实例化） |
| `core/AICore/src/tasks/openpcdet/extensions/openpcdet_cuda_ext.cu` | 追加：`det_sparse_conv_kernel` + launch 包装（保持单编译单元） |
| `core/AICore/src/tasks/openpcdet/models/voxelrcnn.cpp` | `run_focal_conv_core` 按 `hp.focal_deterministic` 切换路径；邻居表展开复用现有 indice map |
| `core/AICore/src/tasks/openpcdet/models/voxelrcnn.h` | 确定性路径声明 |
| `core/AICore/src/tasks/openpcdet/gguf_loader.h/.cpp` | `hparams::focal_deterministic`（focal 模型默认 true） |
| `core/AICore/CMakeLists.txt` | 新源文件注册 |
| `core/AICore/tests/`（task 测试） | 逐位一致性 + 端到端回归用例 |
| `CUDA_PORTING_NOTES.md` | 方案 C 落地记录与验证数据 |

明确不做：不动 ggml patch（确定性 kernel 全部任务侧）；不加 public ABI
选项；不改其他 17 模型的任何路径。

## 已知权衡（诚实声明）

- 确定化后 vs **上游**的 mean_iou 不会提升到 1.0（跨实现图结构差异仍在，
  预期 ~0.998±，与上游自declared floor 同量级）；收益是**我方三后端逐位
  一致**（可复现性、调试、回归判定的确定性证据），并消除检测级下最接近
  阈值位点的漂移风险。
- ±15–25ms focal 端到端延迟（<3%）。
