# CUDA Graph Parity — Integration-Side vs Upstream Build Artifact Comparison

**Status**: root cause identified (see §4), fix merged (see §8)
**Scope**: CUDA graph end-to-end latency for `core/AICore/src/tasks/yolo/`
**Conclusion summary**: Vulkan/CPU are aligned or close to upstream (most rows within the ≤5% threshold); **CUDA graph is 36~88% slower than upstream**, with the root cause pointing to the **integration-side ggml-cuda build artifacts** differing from the upstream release binary — not graph structure, arch, ggml version, or build type. This special project defines a repeatable comparison flow to be executed directly for the next build-artifact-level investigation.

> **§8 Fix conclusion (2026-08): the root cause is `GGML_CUDA_FORCE_MMQ=ON` being hardcoded**. This macro forces quantized matmul onto the MMQ kernel (small conv/matmul fixed overhead ~1.8-2×, consistent with the §3 per-op signature), and this hardcoding was introduced **after** the parity experiments (§2 records both sides OFF) by `d7206a68a` (cross-platform CUDA deployment). It is now converted to the `AICore_CUDA_FORCE_MMQ` option (default OFF = upstream performance parity; ON = self-contained deployment without cuBLAS DT_NEEDED), while the deployment side uses `AICore_BUNDLE_CUDA_RUNTIME=ON` to carry libcublas.so.* instead.

---

## 1. Problem Definition

`run_upstream_parity.sh` full-matrix results (cuda 63 + vulkan 63 + cpu 63 rows):

| Device | e2e p50 vs upstream | graph p50 vs upstream | Status |
|--------|---------------------|-----------------------|--------|
| Vulkan | 28/33 ok (some -3~-6% faster) | aligned | ✅ within threshold |
| CPU | 9 ok | aligned | ✅ within threshold |
| **CUDA** | **+20~88%** | **+36~87%** | ❌ all over threshold |

The gap is concentrated in the **graph segment** (upload + ggml graph execution + readback); preprocess/postprocess have been eliminated via thread alignment (`AICORE_TEST_YOLO_THREADS=32`).

## 2. Ruled-Out Root Causes (with evidence)

| Hypothesis | Verification method | Result |
|------------|---------------------|--------|
| Graph structure changed by refactor | `git show HEAD` compare build_run_plan graph structure + Vulkan side aligned | ❌ graph structure identical (Vulkan not slow proves the op chain is the same) |
| ggml version difference | both sides pin ggml v0.18.1 | ❌ same version |
| CUDA arch mismatch | rebuild ext_ggml with `-DCMAKE_CUDA_ARCHITECTURES=86` | ❌ no improvement (4.36 vs 4.16ms), reverted to 75-real;80-real;86 |
| Build type (Debug/Release) | both sides Release | ❌ same |
| FORCE_MMQ difference | both sides OFF | ❌ same (at experiment time); **but later `d7206a68a` hardcoded it ON introducing a new difference, fixed by §8 to be configurable with default OFF** |
| Thread count mismatch (harness artifact) | GPU rows set threads=32 | ✅ after fix preprocess ~7x and ~50% of graph gap disappeared, **but graph still +36~87%** |

## 3. Localization Evidence: per-op profile

`AICORE_TEST_YOLO_PROFILE=1` (→ `aicore_yolo_options_set_profile_ops`) prints a per-op table (total_ms/calls/avg_us) at free_session.

yolov8n-f16 640x640 CUDA graph comparison (integration side vs upstream `yolo-cli` bench):

| op | integration avg_us | upstream avg_us | ratio |
|----|--------------------|-----------------|-------|
| op.0 large conv (640x640→320x320) | 439 | 431 | 1.02× |
| **op.114 small conv (60x80 / 30x40 / 15x20)** | **404** | **216** | **1.87×** |
| other small convs (60x80 etc.) | 1.7~2.0× slower | baseline | ~2× |

**Signature**: large kernels are close; **small conv kernels have ~1.8-2× per-op fixed overhead**. The Vulkan side with the same model and same graph structure is not slow → this is not op dispatch / graph construction overhead, it is the **CUDA kernel itself being slower in the integration-side build artifact** (higher per-launch fixed overhead).

## 4. Root Cause Hypotheses (by priority)

1. **ggml-cuda compile instantiation/specialization option differences**: the upstream `yolo-cli` build may enable/disable certain CUDA kernel specializations (e.g., `GGML_CUDA_FORCE_MMQ`, f16 specialization, template instantiation `GGML_CUDA_MMQ_Y`, etc.). Different compile-time macros → different SASS → slow small kernels.
2. **CUDA runtime/toolchain versions**: nvcc version differences between the two sides, `-O3 -use_fast_math` flag differences, etc.
3. **cuBLAS version binding**: besides `GGML_CUDA_USE_GRAPHS`, small convs use custom kernels and are not affected by cuBLAS — ruled out.
4. **L2/TLB layout**: the two sides use different allocators (same-process multi-session already eliminated); still slow after single-model process isolation → in-process interference ruled out.

## 5. Special Project Execution Flow (reproduce + compare)

Prerequisites: `build_app/bin/ACloudViewer` contains libAICore (CUDA); the upstream checkout is at `dl/ultralytics-ggml` (contains the `yolo-cli` bench binary).

```bash
# 1) Run yolov8n-f16 CUDA graph on both sides, output per-op profile
#    Integration side (via C API, same profile_ops switch):
AICORE_TEST_YOLO_MODELS_DIR=.../gguf AICORE_TEST_YOLO_IMAGE=.../bus.jpg \
AICORE_TEST_YOLO_DEVICE=cuda AICORE_TEST_YOLO_THREADS=32 \
AICORE_TEST_YOLO_PROFILE=1 \
  core/AICore/tests/yolo/test_yolo_capi_performance \
  > integrated.jsonl 2> integrated.op_profile.log

#    Upstream side (yolo-cli bench prints its own per-op table):
cd dl/ultralytics-ggml && ./build/bin/yolo-cli bench ... 2> upstream.op_profile.log

# 2) Align by op name/shape and generate a ratio table (the profile mode of
#    core/AICore/tests/yolo/bench_compare.py is reusable; otherwise manually
#    grep "op." lines, sort by key, and diff)

# 3) SASS disassembly comparison for key small conv kernels (grab the cubin
#    corresponding to op.114 from each side):
cuobjdump -sass build_app/ggml/src/ext_ggml-build/ggml/src/ggml-cuda/*.cubin \
  > integrated.sass
cuobjdump -sass dl/ultralytics-ggml/build/ggml/src/ggml-cuda/*.cubin \
  > upstream.sass
# Compare instruction count, register pressure, local memory access for the same kernel (differences → different compile options)

# 4) Compile instantiation option list diff (ggml-cuda compile lines in both sides' build.ninja/Makefile):
grep -oE '\-D[A-Z0-9_]+(=[0-9]+)?' build_app/ggml/src/ext_ggml-build/.../flags.make \
  | sort -u > integrated.defs
grep -oE '\-D[A-Z0-9_]+(=[0-9]+)?' dl/ultralytics-ggml/build/.../flags.make \
  | sort -u > upstream.defs
diff integrated.defs upstream.defs
```

## 6. Next Actions (try in this order, re-run §5.1 verification each time)

1. **Align compile macros**: ~~align the ggml-cuda macro (`GGML_CUDA_*`) differences from `upstream.defs` in
   `core/AICore/cmake/AICoreCompileDefinitions.cmake` or `3rdparty/ggml/ggml.cmake`,
   rebuild ext_ggml + AICore, re-run parity.~~ **Done**: `GGML_CUDA_FORCE_MMQ`
   changed from hardcoded ON to the `AICore_CUDA_FORCE_MMQ` option (default OFF, see §8).
2. **Align nvcc flags**: `-O3 -use_fast_math`, `--expt-relaxed-constexpr`, etc.
3. **Align toolchain**: `nvcc --version` identical on both sides (e.g., upstream CI uses CUDA 12.x).
4. If no improvement above: use the §5.3 SASS diff to locate the specific kernel difference (instruction count/registers),
   then fix that kernel inside the patch chain (**do NOT hand-edit source under build/, use 3rdparty/ggml/patches/**).

## 8. Positive Fix (2026-08)

Root cause: `3rdparty/ggml/ggml.cmake` previously hardcoded `GGML_CUDA_FORCE_MMQ=ON` (introduced
by `d7206a68a` for cross-platform CUDA deployment: libggml-cuda.so does not depend on libcudart/libcublas
and only needs the NVIDIA driver to load). This macro forces all quantized matmuls onto the MMQ kernel; on
small matrices, MMQ's per-launch fixed overhead is significantly higher than the cuBLAS path, matching the
§3 signature of op.114 small convs (60x80/30x40/15x20) being 1.87× slower.

Fix content:
- `cmake/AICoreOptions.cmake`: added the `AICore_CUDA_FORCE_MMQ` option (**default OFF**,
  consistent with the upstream ggml/ultralytics-ggml build); synced to
  `GGML_CUDA_FORCE_MMQ` via `aicore_sync_options_to_ggml()`.
- `3rdparty/ggml/ggml.cmake`: no longer hardcodes ON; forwards the user option.
- Deployment side: set `AICore_CUDA_FORCE_MMQ=ON` when a driver-only self-contained build is needed; otherwise rely on
  `AICore_BUNDLE_CUDA_RUNTIME=ON` (default CI path) to carry libcublas.so.*.
- `VerifyNoDynamicCuda.cmake` (qSIBR regression guard) semantics unchanged: FORCE_MMQ=ON builds still have
  no dynamic CUDA dependencies.

Verification: rebuild ext_ggml + AICore with `AICore_CUDA_FORCE_MMQ=OFF`, re-run
`run_upstream_parity.sh --devices cuda --limit 5`, confirm e2e p50 ≤ +5%.

## 7. Regression Guard

- Must re-run after every experiment: `core/AICore/tests/yolo/run_upstream_parity.sh --devices cuda --limit 5`
- Threshold unchanged: **e2e p50 ≤ +5%**.
- After all arch/compile-option experiments, restore the repository default configuration (`cmake -UCMAKE_CUDA_ARCHITECTURES`).
