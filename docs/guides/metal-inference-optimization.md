# Metal Inference Performance Optimization Experience (ggml)

This document summarizes the **reusable patterns** and **pitfall checklist** accumulated while optimizing ggml Metal backend inference performance on Apple Silicon (M2 Max). Applicable scenario: when integrating a new model (YOLO detection/segmentation, RF-DETR, depth, etc.), if first-time Metal inference is slow, use this document to systematically locate and eliminate bottlenecks.

> **Core conclusion (first principles)**: Apple Silicon's GPU compute is released through **dedicated matrix units** (simdgroup matrix multiply). If a kernel does not take the matrix-unit path (e.g., scalar element-wise loops, too few elements per thread causing low threadgroup utilization), performance can differ by 10–100×. All optimizations revolve around this constraint.

---

## 1. Profiling Methodology: Find the Real Bottleneck with per-op Profile

### 1.1 Don't guess, measure

ggml-metal ships the **GGML_METAL_OP_PROFILE** environment variable (must be enabled in `ggml-metal-context.m`), which gives per-op GPU execution time:

```objc
// Insert an env-gated branch at the start of the @autoreleasepool in ggml_metal_graph_compute
if (getenv("GGML_METAL_OP_PROFILE") != NULL) {
    static int s_profile_after = 3;  // start after the 3rd compute (ensure pipeline cache)
    if (s_profile_after > 0 && --s_profile_after == 0) {
        id<MTLCommandBuffer> cb = [queue commandBufferWithUnretainedReferences];
        // op_init 9-arg signature: dev, cb, gf, gi, gi+1, false, false, false, 0, 0
        ggml_metal_op_t op = ggml_metal_op_init(ctx->dev, cb, gf, gi, gi + 1,
                                                 false, false, false, 0, 0);
        ggml_metal_op_encode(op, 0);
        ggml_metal_op_free(op);      // free first (internal endEncoding) then commit
        [cb commit];
        [cb waitUntilCompleted];
        fprintf(stderr, "[op-profile] idx=... src=[%s,%s] k=%lld ... gpu=...us\n", ...);
        return GGML_STATUS_SUCCESS;
    }
}
```

> **Attention**: do **not commit** the profile branch (it is an env-gated local debugging tool). It must not be included in patches.

### 1.2 Bottleneck Pattern Recognition

Identify three root-cause classes from profile output:

| Pattern | Signature | Root cause |
|---------|-----------|------------|
| **Fragmented threadgroups** | op avg duration short (<10μs) but huge count (hundreds of same-class ops), large total | too few elements per threadgroup, low GPU core utilization |
| **Scalar path** | mul_mat with K<64 significantly slow (0.1 TFLOPS class) | Apple Matrix Unit requires K≥64 to activate the matrix path; when K is too small it degrades to element-wise scalar kernels |
| **Bandwidth saturation** | UNARY / CPY-class ops reach ~310 GB/s | kernel is at the hardware bandwidth limit; further optimization requires reducing intermediate tensor traffic (graph-level fusion) |

---

## 2. Six Reusable Kernel Optimization Patterns

Each pattern below includes applicable conditions and measured benefit magnitude (based on BiRefNet-Swin-L / YOLOv8 / RF-DETR on M2 Max 64GB).

### Pattern A: Flat-grid flattening + division elimination

**Root cause**: IM2COL defaults to a 3D `(OC, OH, OW)` layout; when `[IC, KH, KW]` is large while `OC` is small (e.g., conv with OC=1), each threadgroup processes only 1×9 elements, leaving the GPU execution units severely idle.

**Solution**: collapse the multi-dimensional grid into a 1D `total/thread_elements`, using an **hw-carry increment** (hw-carry chain) inside the thread instead of per-element modulo division:

```metal
// each thread processes 16 channels (4×float4)
const int total = IC * OH * OW;        // CHW
const int total4 = total >> 2;
const int i0 = idx * 16;               // 4 float4 per thread
for (int j = 0; j < 16; j += 4) {
    int c = (i0 + j) >> 2;             // carry: h*w increments, c carries
    int h = c / OW;
    int w = c - h * OW;
    // ... im2col gather
}
```

**Applicable conditions**: most beneficial when `OC * N == 1` and `CHW` is a multiple of 4 (float4 alignment). Even better when `CHW` is large and 16-aligned.

**Benefit magnitude**: IM2COL from 1382ms → 142ms (**-90%**), the single largest kernel win in this optimization pass.

### Pattern B: Multi-element per thread + float4 vectorization

**Root cause**: each threadgroup dispatch has fixed overhead; kernels processing 1 element per thread make the GPU spend most time on dispatch and threadgroup management.

**Solution**: each thread processes 4 or 16 elements, loading/storing once with `float4` / `float16` vector types; the grid shrinks to `1/4` or `1/16`.

```metal
// each thread processes 4 float4s (16 elements total)
const int total = args.ne * ...;
const int idx = tgpig.x * ntg * ... + tiitg;
const int i0 = idx * 16;
// read 4 float4s
float4 v0 = src[i0/4 + 0];
float4 v1 = src[i0/4 + 1];
float4 v2 = src[i0/4 + 2];
float4 v3 = src[i0/4 + 3];
```

**Applicable conditions**: large `total` (>10000 elements), bandwidth-bound kernels (UNARY, BIN, GET_ROWS). Not suitable for compute-intensive kernels (matrix multiply), where thread count is already saturated.

**Benefit magnitude**: BIN_OP from ~146ms down 60%+; GET_ROWS from 7.2ms → ~2ms (**-72%**).

### Pattern C: Broadcast / row-special-case modulo elimination

**Root cause**: when a matrix is broadcast by row (`ne0 == ne10`), the per-element `i0 % args.ne10` modulo instruction costs many GPU cycles.

**Solution**: extract the **broadcast special case** (`args.ne10 <= 1`: global scalar) and the **row-aligned special case** (`args.ne10 == args.ne0`: same value per row), using branchless unconditional assignment to eliminate division:

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

**Applicable conditions**: binary ops with `M=N` (or `ne0 == ne10`). Metal's threadgroup uniform branch overhead is near zero when all threads in a warp take the same path.

**Benefit magnitude**: BIN_OP saves 1 integer division and 1 modulo per row; combined -20~50% (when combined with per-thread multi-element).

### Pattern D: K<64 F16 matrix multiplication threshold relaxation

**Root cause**: Metal's simdgroup matrix multiply requires K≥64 to activate the matrix-unit path (`kernel_mul_mm`); with K<64 it degrades to the matrix-vector scalar kernel (`kernel_mul_mv_f32_f32`) at roughly **0.1 TFLOPS** (1/100 of the matrix path).

**Solution**: relax the threshold from 64 to 16 for `f16×f16` combinations:

```cpp
// in ggml-metal-ops.cpp, in get_extra_buffers_mul_mat or the dispatch branch
props_dev->has_simdgroup_mm &&
    ((ne00 >= 16 && op->src[0]->type == GGML_TYPE_F16 && op->src[1]->type == GGML_TYPE_F16)
     || (ne00 >= 64))
    && ne11 > ne11_mm_min
```

> **Extremely important constraint**: **relax only for f16×f16**. For f32, taking the matrix path with K<64 is catastrophically slow (regresses from 38ms on the scalar path to 2487ms), because the f32 matrix path's dequantize_f32 overhead is unacceptable at small K.

**Applicable conditions**: the model has multiple `[M, K]×[K, N]` shapes with f16 weights where K is in [16, 63] (typical: conv K=27 output channels, MLP K=32, etc.). f16 models (recommended inference format) benefit directly.

**Benefit magnitude**: conv K=27 from 38.4ms → 1.6ms; the whole YOLO v8 inference chain from 51.4ms → 14.5ms (**-72%**, including accumulated IM2COL speedup).

### Pattern E: Small-K GEMM f16 weight conversion (matrix-unit path lock-in)

**Root cause**: f32 weights go through `kernel_mul_mm_f32_f32` in Metal, where weight loading uses the `dequantize_f32` scalar path (4-byte loads instead of 64-byte cacheline bursts), giving poor bandwidth utilization — especially impactful for small-K conv/MLP GEMMs.

**Solution**: in the AICore graph construction phase, specify conv weights and QKV/Proj linear-layer weights as F16 for the `use_metal` path:

```cpp
// in rmbg_graph.cpp
ggml_tensor *w16 = weight_f16(prefix + "weight");  // F16 weights
ggml_tensor *col16 = ggml_im2col(ctx, w16, ..., GGML_TYPE_F16);
// F16 im2col + F16 weight → kernel_mul_mm_f16_f16 matrix-unit path
```

Also set the F16 im2col output to F16 (reduces write bandwidth), controlled together with the `metal_f16_gemm` option in Metal graph construction.

**Applicable conditions**: the model has many small-K (≤256) GEMMs (conv IC→OC, MLP hidden→4×hidden, etc.). For large-K GEMMs (≥3072) the matrix path is **already saturated** (~10 TFLOPS), and f16 conversion brings no benefit.

**Benefit magnitude**: f16 conversion of the conv chain reduces ~50%+ (K-size dependent); full model from 2534ms → 933ms (**-63%**, combining all optimizations).

### Pattern F: Graph-level Op Fusion

**Root cause**: ggml computation graphs contain many reshape/cont/permute copy operations (RMBG original graph: 6461 nodes, including RESHAPE 1266, CONT 551, PERMUTE 359). These ops add no compute but move large tensors ([3072, 1024] class), accumulating significant time.

**Solution**: replace common multi-op chains with custom kernels:

| Fusion scenario | Before | After | Benefit |
|-----------------|--------|-------|---------|
| **SWIN_QKV Layout** | add + 3×(cont + permute + cont) | 1 kernel (bias fusion + reorder write) | -179ms |
| **Flash Attention** | QK matmul + scale + softmax + AV matmul | 1 kernel (kernel_flash_attn_ext_f32) | -828ms |
| **Conv + Bias** | im2col + matmul + add + bias | 1 kernel (GPU-generic, needs custom op) | depends |

**Applicable conditions**: identifiable at graph-construction time (fixed graph structure), and the new kernel's flag-reorder logic must be **bitwise-equivalent** to the original copy chain (verified with output_hash).

**Benefit magnitude**: flash attention always pays off (-800~1000ms class, proportional to QKV reorder count). qkv layout fusion saves ~3 cont+permute per block when multiple attention blocks exist.

---

## 3. Graph Construction Decision Patterns (AICore layer)

### 3.1 Metal backend detection

**MTL0 trap**: `ggml_backend_name()` returns the device name (e.g., `"MTL0"`), **not** the fixed `"Metal"` string. Must match both:

```cpp
use_metal = name && (std::strstr(name, "Metal") || std::strstr(name, "MTL"));
```

### 3.2 Conditional gating for f16 weight conversion

Configure Metal independently in `GraphOptions`:

```cpp
struct GraphOptions {
    bool cuda_f16_gemm = false;       // CUDA default OFF (explicit user opt-in)
    bool metal_f16_gemm = true;       // Metal default ON (F16 is the only matrix-unit path)
};
```

Derivation of `use_f16_gemm`:

```cpp
use_f16_gemm =
    (use_cuda_custom && !strict_math && cuda_f16_gemm) ||
    (use_metal && !strict_math && metal_f16_gemm);
```

In `strict_math` mode, f16 conversion is forbidden to keep pure FP32 precision.

### 3.3 Three-quantization consistency verification

> Before grayscale release, verify precision consistency across f32/f16/q8 **three quantizations** (f32/f16 output_hash should be exactly identical; q8 differs in hash due to the dequantization path, but contract passes).

```bash
# RMBG contract tests
AICORE_TEST_RMBG_MODEL=<f16.gguf> test_rmbg_capi_contract
AICORE_TEST_RMBG_MODEL=<f32.gguf> test_rmbg_capi_contract
AICORE_TEST_RMBG_MODEL=<q8.gguf> test_rmbg_capi_contract
# Performance should be consistent across the three quantizations (±1%) — Metal bottleneck is the activation path, weight quantization does not affect performance
```

Three-quantization performance consistency (±1%) is a unique Metal characteristic (q8 is faster on CUDA); the reason: the bottleneck is in the **activation path** (the matrix multiply itself), not weight bandwidth.

---

## 4. Pitfall Checklist (must avoid)

### 4.1 MSL / C struct layout consistency (SIGKILL)

**Symptom**: runtime SIGKILL (Kill: 9) with no error output.

**Root cause**: the MSL kernel-side `constant struct kargs_im2col` contains fields `OH`/`OW`, but the corresponding C-side struct (`impl.h`) does not → **the MSL-read args.OH/args.OW are garbage pointers** → GPU out-of-bounds access → GPU reset → OS sends SIGKILL.

**Fix**: the C-side `kargs_im2col` must **strictly match** the MSL-side field order and types. `sed` template expansion also shifts offsets — declare the struct uniformly with `#include "impl.h"`.

**Prevention**: every time new kernel args are added on the MSL side, synchronously update the C-side struct in `impl.h`. Run contract immediately after compiling the metallib to confirm no SIGKILL.

### 4.2 ExternalProject incremental build mtime trap

**Symptom**: after modifying a patch file and cleaning the install stamp, `metallib` is not recompiled.

**Root cause**: the mtime written by the Python script to the .metal file lands in the same second as the previous .o compile time → make considers the source up-to-date → skips metallib recompilation.

**Fix**: after every patch-chain change, clear the build + install + done stamps together:

```bash
rm -f build_app/ggml/src/ext_ggml-stamp/ext_ggml-{build,install,done}
# extreme case: wipe the whole source dir so ExternalProject re-extracts
rm -rf build_app/ggml/src/ext_ggml
cmake --build build_app --target ext_ggml -j4
```

### 4.3 ggml_view_4d stride inheritance trap

**Symptom**: Flash attention output degenerates on large inputs (1024²) (alpha values all in [253,255]).

**Root cause**: `ggml_view_4d(src, ne0, ne1, ne2, ne3)` does not recompute the nb array — it **inherits the parent tensor's strides**. When a view is used for dimension splitting (e.g., splitting [3C] into [hd, heads]), the inherited strides make the kernel read wrong addresses.

**Prevention**: view operations involving dimension splits require manual verification of the nb pointers. For kernels like Flash Attention, do not pass strided views; do `ggml_cont` first to ensure contiguity, then pass.

### 4.4 Profile branch must not be committed

The profile branch (see §1.1) is a **purely local debugging tool** containing internal headers like `#import "ggml-metal-impl.h"` — **do not commit it into patches**. Patches are production-oriented stable optimizations and must not contain debug code.

### 4.5 K<64 global threshold disaster

**Never** unconditionally route all K<64 mul_mats onto the matrix path. For f32 with large M, K<64 on the matrix path produces catastrophic regression:

```cpp
// Wrong:
// (ne00 >= 16 && op->src[0]->type == GGML_TYPE_F32)  // ← disaster!
// Correct (f16×f16 only):
(ne00 >= 16 && op->src[0]->type == GGML_TYPE_F16 && op->src[1]->type == GGML_TYPE_F16)
```

Measured: f32 on the mul_mm path regresses from 38ms to 2487ms (**65× regression**). The root cause is that the f32 matrix path's dequantize_f32 load is extremely inefficient at small K.

---

## 5. New-Model Acceleration Checklist

When integrating a new model (e.g., a new YOLO variant, depth model, MLP-only model) with slow Metal inference, check in this order:

1. **Run per-op profile** → identify hot op types
   - MUL_MAT dominated → check K distribution, confirm K≥64 takes the matrix path; handle K<64 per Pattern D
   - IM2COL dominated → optimize per Pattern A flat-grid (best when CHW is aligned)
   - BIN_OP / UNARY / GET_ROWS dominated → vectorize per Patterns B + C
   - CONT / CPY / PERMUTE dominated → consider graph-level fusion (Pattern F)

2. **Check weight type** → if f32, evaluate f16 conversion benefit (Pattern E)
   - Many small-K GEMMs → large f16 benefit
   - Many large-K GEMMs (≥3072) → already saturated, only check the K<64 threshold

3. **Check attention blocks** → if present, enable Flash Attention (Pattern F)

4. **Three-quantization verification** → run contract for f32/f16/q8 each, confirm precision consistency

5. **Regression**: after all contracts pass, check whether CUDA/Vulkan/CPU baselines regressed (ggml-metal code is isolated and should not regress)

---

## 6. References

- Merged Metal optimization patch: `3rdparty/ggml/patches/metal_merged/0001-metal-optimizations.patch`
- AICore graph construction: `core/AICore/src/tasks/rmbg/rmbg_graph.cpp` (Metal-branch reference implementation)
- ggml code modification rules: `.agents/rules/acloudviewer-ggml-aicore.mdc`
