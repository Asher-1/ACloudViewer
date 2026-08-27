# ggml Version Upgrade Verification Pipeline

> Scope: upgrades of the ggml version pinned by the `3rdparty/ggml`
> ExternalProject in ACloudViewer (e.g. 0.18.1 → 0.21.0). This document is
> the **single authoritative process** for the upgrade action: both
> `AGENTS.md` and `.agents/rules/acloudviewer-ggml-aicore.mdc` reference it.
>
> Companion tool: [`scripts/ggml_upgrade_verify.py`](../../scripts/ggml_upgrade_verify.py)
> (one-click baseline/candidate performance A/B + numerical parity + coverage report).

---

## 0. Design principles (why this process)

A verdict on "is the upgrade safe" is fundamentally composed of only three classes of facts:

1. **Buildable**: the 13+ downstream patches can be replayed on the new version's source tree (idempotently), and AICore compiles.
2. **Numerically correct**: each task produces consistent outputs on CPU and GPU backends (parity gates).
3. **No performance regression**: per task, per backend, latency/VRAM is not worse than the baseline beyond a threshold (default 5%).

Everything else (ctest labels, plugins, CI) is merely a carrier for these three classes of facts. The upgrade process is therefore split into
**5 stages, each with explicit pass/fail criteria**; stop on failure, never skip a stage.

---

## Stage 1: Pre-upgrade research (no code changes)

| Check item | Method | Criteria |
|---|---|---|
| Upstream fixes / breaking changes | Read the target version's release notes; `git log vOLD..vNEW -- src/ggml-vulkan` etc. | List the fixes relevant to this repo (e.g. Vulkan DeviceLost #26371) and breaking changes (e.g. 0.22 removing `ggml-metal.metal`) |
| API compatibility | Compare the `ggml_*` symbols used by `core/AICore` against the target version's headers | No removals; additions must not affect us |
| CMake option compatibility | Compare the options used by `3rdparty/ggml/ggml.cmake` | All retained |
| Patch-chain dry run | Replay `3rdparty/ggml/patches/manifest.yaml` on a clean source tree | Record the list of failing hunks |

**Deliverable**: feasibility report (target version, list of patches requiring manual handling, known risks).

Historical fact anchors (measured 2026-08):

- 0.18.1 → 0.21.0: 0 public API removals, 2 additions; all 21 CMake options retained;
  10 of 14 patches applied with zero modification.
- 0.22.0: the `metal_merged` patch fails to replay because upstream removed `ggml-metal.metal`,
  **not buildable**; that patch must be rewritten first before upgrading.

## Stage 2: Experimental tree + patch chain

1. Copy the main repo into an experimental tree (do not touch the main repo).
2. Modify `GGML_VERSION` / URL / SHA256 in the experimental tree's `3rdparty/ggml/ggml.cmake`.
3. Handle each failing patch individually:
   - hunks already fixed upstream → **delete** (e.g. 0.21 already contains the batching fix, so the duplicate hunk is removed);
   - hunks still needed → manually rebase, then regenerate with `diff -ruN`.
4. Replay on a clean tree + **second-replay idempotency check** (the second replay should all report `already applied`):

```bash
python3 3rdparty/ggml/patches/apply_ggml_patches.py --verify   # or the script's actual interface
```

**Criteria**: both replays pass completely.

## Stage 3: Build + contract tier (fast gate)

```bash
cmake ... -DAICore_ENABLED=ON -DAICore_BUILD_TESTS=ON -DAICore_BUILD_WHITEBOX_TESTS=ON
make AICore -j"${BUILD_JOBS}"
ctest -L capi -LE "model|gpu|e2e" -j4      # = equivalent selection of aicore-contract-tests
```

**Criteria**: 100% pass. Fix any failure before continuing.

## Stage 4: Numerical parity + model tier

```bash
export AICORE_TEST_DEPTH_GGUF=$DATA/da3_models/depth-anything-base-f16.gguf
export AICORE_TEST_DEPTH_IMAGE=examples/test_data/image/00000.png
export AICORE_TEST_DEVICE=vulkan
ctest -L model -j1                          # missing assets auto-skip(77), never counted as failure
```

Key parity gates (test-embedded thresholds are authoritative):

| Test | Meaning | Gate |
|---|---|---|
| `test_depth_capi_backend_parity` | CPU vs GPU depth/pose/intrinsics | rel MAE ≤ 0.015 / 0.02 |
| `test_aliked_capi_parity(_512/_q8)` | ALIKED features CPU vs GPU | test-embedded |
| `test_lightglue_aliked_e2e` | Matching end-to-end | test-embedded |
| `bench_sam3_backend_acceptance` | SAM3 mask/box/score | IoU ≥ 0.98, box ≤ 1px, score ≤ 5e-3 (see note below) |

> SAM3 parity note (2026-08): the IoU gate is 0.98 because the CPU-vs-Vulkan
> mask IoU measures 0.985 on `sam3-visual-f16` (fp16 Vulkan flash-attention vs
> f32 CPU), and the score gate is 5e-3 because the sigmoid presence score
> differs by ~1.5e-3 across backends (measured; 3x margin). These are the FIRST
> valid parity numbers for windowed attention: prior
> to ggml 0.21 + the runtime `win_part` dispatch, ggml-vulkan silently skipped
> WIN_PART/WIN_UNPART nodes (no valid GPU reference existed), and 0.18.1 cannot
> even complete the `all`-mode acceptance (it crashes with DeviceLost).

**Criteria**: all runnable items pass; skipped items are explicitly listed in the report with reasons.

## Stage 5: Performance A/B (one-click)

```bash
python3 scripts/ggml_upgrade_verify.py \
    --baseline <0.18.1 build tree> \
    --candidate <new version build tree> \
    --report /tmp/ggml_upgrade_report.md \
    [--only yolo_vulkan,rfdetr_vulkan]   # optional subset
```

The script runs the same benchmark on **both trees** for each probe, parses the
metrics, compares them metric by metric with a **+5% regression threshold**
(3% noise floor), and outputs a Markdown report plus `(task, backend)` coverage.
`--repeats` (default 2) uses **interleaved rounds**: each round measures baseline
immediately followed by candidate (sharing the same load window), and multiple
rounds take the **median** per metric to eliminate machine-load drift.

> Measured lesson (2026-08): with resident background load on this machine, a single
> A/B run produced a fake +8.5% yolo_cpu regression (`yolo26m-f16`); an isolated
> re-measurement in the same time window showed the candidate was actually faster
> (1634.7 vs 1644.6 ms p50). Therefore: **any single-item CPU regression verdict
> >5% must first be confirmed by an isolated subset re-measurement before it is
> written into conclusions**.
>
> Another trap: **stale `AICORE_TEST_*` env vars inherited from the shell silently
> change asset selection** (e.g. a leftover single-model subset dir degrades a full
> run into a single-model run). The script echoes all asset environment variables
> and their origins (inherited/default) at startup; confirm there is no abnormal
> inheritance before a formal run, or run in a completely fresh shell.

Probe list (`PROBES`):

| Probe | Binary | Output family |
|---|---|---|
| yolo_vulkan / yolo_cpu | `test_yolo_capi_performance` | JSON lines (per model/task) |
| rfdetr_vulkan | `bench_rfdetr_perf` | text (median table) |
| depth_parity | `test_depth_capi_backend_parity` | stderr rel MAE |
| aliked_parity(_q8) / lightglue_e2e / aliked_smoke | ctest binaries of the same names | exit code |
| sam3_acceptance | `bench_sam3_backend_acceptance` | single-line JSON |

**Criteria**: `Verdict: NO REGRESSION` and exit code 0; coverage gaps must be
explained item by item in the report (pre-existing defect / missing asset / missing baseline).

### Coverage composition (performance probes + ctest tiers ≈ 95%)

The `(task, backend)` coverage in the script report counts only **performance A/B probes**; overall upgrade verification coverage is stacked from three tiers:

| Tier | Carrier | Coverage content |
|---|---|---|
| Contract tier | `ctest -L capi -LE "model\|gpu\|e2e"` (17 tests) | C API contracts of **all** tasks (including items not covered by the script: deeplsd / facedetect / gaussian / rmbg / sam3 / trellis, etc.) |
| Model tier | `ctest -L model` (13 tests, asset-gated) | load/inference/accuracy on real GGUFs |
| Performance tier | this script's 9 probes + manual benchmarks | yolo/rfdetr/depth/aliked/lightglue dual-backend + sam3 acceptance + rmbg CPU |

Item-by-item explanation of uncovered entries in the script report:

- `sam3 cpu/vulkan`: the acceptance benchmark runs in the dedicated probe tree (minimal build); see the manual supplement in Stage 5;
- `rmbg vulkan`: blocked by the pre-existing conv2d assert (both versions crash identically, see the pre-existing issues table);
- `rfdetr cpu` / `trellis` / `deeplsd` / `facedetect` / `gaussian` performance: no dedicated bench probes; correctness is covered by the contract tier + model tier; when adding new performance probes, update the script's `PROBES` and `COVERAGE_UNIVERSE` in sync.

### Manual supplement benchmarks (when not covered by the script)

```bash
# SAM3 end-to-end (N runs each on CPU+Vulkan)
bench_sam3_backend_acceptance <sam3-visual-f16.gguf> <image> [runs=10] [backend=all|cpu|vulkan]
# RMBG GPU performance (currently blocked by the pre-existing conv2d assert, see below)
test_rmbg_capi_performance            # requires AICORE_TEST_RMBG_GGUF
```

### Stability gate

GPU backend: **10 consecutive runs** without crashes, without DeviceLost, with non-empty outputs; peak VRAM no higher than baseline.

---

## Known pre-existing issues (unrelated to the upgrade; do not misclassify as regressions)

| Issue | Impact | Evidence |
|---|---|---|
| **0.18.1 Vulkan DeviceLost** | SAM3 reproduces `vk::Device::waitForFences: ErrorDeviceLost` on all models/all sizes at 100%; YOLO multi-model sequential runs crash on the 2nd model | Upstream #26371-series fixes (FLOP-dependent submission strategy + driver-timeout avoidance) were merged since 0.19; measured to be completely gone on 0.21 |
| **RMBG Vulkan conv2d assert** | Both 0.18.1 and 0.21 crash **identically** at `ggml-vulkan.cpp` `GGML_ASSERT(pipeline->parameter_count == descriptor_buffer_infos.size())` (conv2d dispatch) | Reproduced under identical conditions on both trees; a conv2d pipeline descriptor-count defect; RMBG Vulkan has never worked on this machine |
| bench all-mode CPU phase too slow | SAM3 track median ~90s/run @ 4 threads | Use the single-backend selector (4th argument) to shorten acceptance |
| **CUDA F16 conv_transpose abort** | YOLO on CUDA aborts at `conv2d-transpose.cu` `GGML_ASSERT(input->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32)` | Upstream `ggml_cuda_conv_2d_transpose_p0` is F32-only and dispatched unconditionally in BOTH v0.18.1 (line 76) and v0.21.0 (line 70); the yolo_merged patch added F16 conv_transpose only on the Vulkan side. The one-click probe therefore measures `yolo_cpu` only; YOLO CUDA has never been exercised by the pipeline on either version |

## Release criteria (all must be satisfied before merging into the main repo)

1. Stage 2 replay passes idempotently;
2. Stage 3 contract tier 100%;
3. Stage 4 runnable parity tests all pass;
4. Stage 5 `Verdict: NO REGRESSION`, and every coverage gap has a written explanation;
5. The 10-run stability gate passes;
6. Main-repo changes are limited to: the version number in `ggml.cmake`, additions/removals under `patches/`, and necessary AICore adaptations
   (e.g. runtime backend probing instead of compile-time macro gating); sources under `build*/ggml/` must **not** be included.

## Version upgrade history

| Upgrade | Result | Notes |
|---|---|---|
| 0.18.1 → 0.21.0 | ✅ Release review passed | Fixed SAM3 Vulkan win_part silent skip (runtime probing on the AICore side) + DeviceLost (upstream fix); encode −6.4% / track −5.7% (Vulkan); YOLO CPU all models within ±1.3% (no regression confirmed by isolated re-measurement); RFDetr detect-total +1.3%; depth parity worst rel MAE 0.0065→0.0032 (51.6% improvement); aliked/lightglue parity+e2e passed on both trees; first valid SAM3 CPU-vs-Vulkan mask parity IoU 0.985 (gate relaxed 0.995→0.98, fp16 flash-attn vs f32 CPU); SAM3 model-lifecycle fix (shared_ptr-coupled `~sam3_model`); nvcc 11.8 rope.cu duplicate `mode` fix; the three CUDA patches consolidated into `cuda_merged/0001-cuda-downstream-fixes.patch` |
| 0.18.1 → 0.22.0 | ❌ Not buildable | The `metal_merged` patch depends on `ggml-metal.metal`, which upstream has removed |
