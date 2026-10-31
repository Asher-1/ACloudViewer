# Full focal-chain determinism — detailed design (Option C)

Status: **implemented (2026-10-29; implementation evidence and the unmet e2e
hash gate are recorded in the Option C section of
[CUDA_PORTING_NOTES.md](CUDA_PORTING_NOTES.md))**. Goal: eliminate the
kitti_voxelrcnn_focal_multimodal CUDA residual of 0.9978 versus upstream C++
and make the focal chain **bit-identical** across CPU/CUDA/Vulkan, beyond
upstream's own cuda-vs-cpu 0.9997.

## First principles

Sparse submanifold convolution is `out[v,o] = bias[o] + Σ_{k<K} Σ_{c<Cin}
feat[idx[v,k], c] · W[k·Cin+c, o]`. ggml mul_mat treats it as GEMM; tile/split-K
dispatch determines reduction order. Cross-implementation differences produce
~2e-7 noise → focal conv_imp sigmoid selects voxels at a hard 0.5 threshold →
2945 sites within <3e-5 can flip → downstream topology drifts.
**Determinism = fixed addition order + fixed rounding semantics**. The shared
ordered reduction gives CPU/GPU bit identity: IEEE754 guarantees identical
results for identical order, rounding and input.

Constraints learned from the earlier audit:
1. Determinizing only conv_imp is insufficient: main-conv mul_mat output feeds
   the next layer's imp_features. **The whole focal chain (conv_imp + main conv
   in 3 FocalSparseConv layers) must use the deterministic path**.
2. GPU `fmaf` (single rounding) ≠ CPU `a*b+c` (double rounding).
   sigmoid `expf` also differs: CUDA ~2ULP vs glibc 0.5ULP. Both sides must
   share the same mathematical implementation.

## Design

### 1. Deterministic sparse convolution (one shared implementation)

Expanding `sparse_indice_map` into `[n_out, K]` row indices (-1=no neighbor)
is pure data movement. Preserve GGUF `[K·Cin, C_out]` weights with BN folded
into weights+bias. One device/host-templated numeric function serves CPU/GPU:

```cpp
// deterministic_sparse_conv.h: shared CPU/GPU inline, same order and rounding
template <typename FMA>   // CPU: std::fma; GPU: fmaf — both single-rounding multiply-add
inline void det_sparse_conv_rows(const float* feat, int C_in,
                                 const int32_t* idx,        // [n_out, K]
                                 const float* W,            // [K*C_in, C_out]
                                 const float* bias,
                                 float* out, int n_out, int C_out, int K,
                                 FMA fma) {
    for (int v = 0; v < n_out; ++v)
        for (int o = 0; o < C_out; ++o) {
            float acc = bias[o];
            for (int k = 0; k < K; ++k) {              // Reduction order: k outer
                const int r = idx[v * K + k];
                if (r < 0) continue;                   // Skip absent neighbors without introducing -0 differences
                const float* row = feat + (size_t)r * C_in;
                const float* w   = W + (size_t)(k * C_in) * C_out + o;
                for (int c = 0; c < C_in; ++c)         // c inner: fixed association order
                    acc = fma(row[c], w[c * C_out], acc);  // Single rounding on both sides
            }
            out[(size_t)v * C_out + o] = acc;
        }
}
```

GPU entry, appended to `openpcdet_cuda_ext.cu` to preserve the single-translation-unit constraint:

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
            acc = fmaf(row[c], w[c * C_out], acc);     // Single rounding, like CPU std::fma
    }
    out[(size_t)v * C_out + o] = acc;
}
// launch: grid=(ceil(n_out/128), ceil(C_out/8)), block=(128,8); one thread per
// (v,o), serial K*C_in reduction; no shared/warp reduction or atomics.
```

For `focal_split_voxels` sigmoid/threshold, share
`det_sigmoid(x) = 1/(1 + det_expf(-x))`; det_expf uses one host/device polynomial,
not `__expf`, so identical inputs yield identical CPU/GPU outputs.

### 2. Wiring and selection

Both `run_sparse_conv` calls (conv_imp and main conv) in `run_focal_conv_core`
(voxelrcnn.cpp) select by private `hp.focal_deterministic`. gguf_loader defaults
focal models to true; no public ABI option is added.
- true → gather neighbor indices → `det_sparse_conv_rows` (direct CPU call or
  GPU kernel + D2D copy-back) → unchanged existing split/combine flow.
- false → unchanged ggml mul_mat control for A/B and regressions.
The other 17 models never read this field. Direct kernel calls bypass the
ggml graph, avoiding graph-capture compatibility issues.

### 3. Performance budget

KITTI 000008 (~2e4 active voxels), heaviest conv3: C_out=64,
K·Cin=27·64=1728 → 2e4×64 threads × 1728 serial FMA ≈ 6ms/layer
(RTX 3060 estimate). 3 layers + conv_imp add about +15–25ms.
The focal e2e baseline is ~0.8s (upstream REPORT: focal_backbone is 85%);
<3% growth is acceptable for this non-real-time scene model.

### 4. Validation gates

1. Unit: fixed pseudorandom input yields **bit-identical** (memcmp) CPU
   `det_sparse_conv_rows` and GPU kernel output; ggml mul_mat difference ≤2e-6.
2. E2e: focal f32 output_hash agrees pairwise across CPU=CUDA=Vulkan;
   upstream cpp_cpu_f32 comparison matches 8/8, expected mean_iou ~0.998±,
   at the upstream cross-implementation floor. Determinism removes **AICore's
   internal** backend differences, not graph-structure differences vs upstream.
3. Regression: kitti_pointpillar / kitti_voxelrcnn_car output_hash remains
   unchanged on three backends; `test_openpcdet_capi_contract` passes.

## Changed files

| File | Change |
|---|---|
| `core/AICore/src/tasks/openpcdet/extensions/deterministic_sparse_conv.h` | Add shared `det_sparse_conv_rows` template + `det_expf/det_sigmoid` |
| `core/AICore/src/tasks/openpcdet/extensions/deterministic_sparse_conv.cpp` | Add CPU neighbor expansion + host dispatch (std::fma instantiation) |
| `core/AICore/src/tasks/openpcdet/extensions/openpcdet_cuda_ext.cu` | Append `det_sparse_conv_kernel` + launch wrapper (single translation unit) |
| `core/AICore/src/tasks/openpcdet/models/voxelrcnn.cpp` | Select `run_focal_conv_core` by `hp.focal_deterministic`; reuse existing indice maps |
| `core/AICore/src/tasks/openpcdet/models/voxelrcnn.h` | Deterministic-path declaration |
| `core/AICore/src/tasks/openpcdet/gguf_loader.h/.cpp` | `hparams::focal_deterministic` (focal models default true) |
| `core/AICore/CMakeLists.txt` | Register new sources |
| `core/AICore/tests/` (task tests) | Bit-identity + e2e regression cases |
| `CUDA_PORTING_NOTES.md` | Option C implementation and validation evidence |

Explicit exclusions: no ggml patch changes (deterministic kernels are task-side),
no public ABI option, and no path changes for the other 17 models.

## Known trade-offs

- Determinism does not raise mean_iou vs **upstream** to 1.0; graph-structure
  differences remain, expected ~0.998± at upstream's declared floor.
  The intended benefit is **AICore three-backend bit identity**, supporting
  reproducibility, debugging and regression decisions and removing drift risk
  at the closest detection-threshold sites.
- ±15–25ms focal e2e latency (<3%).
