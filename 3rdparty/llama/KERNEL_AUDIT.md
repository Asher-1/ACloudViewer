# Embedded ggml operator audit

Audit date: 2026-10-21. This compares the **pristine pinned source**, not a
generated already-patched header, with the CUDA/Vulkan migration candidate.
It is a source-contract review; it does not certify backend throughput.

Source identity is owned by [`llama_cpp.cmake`](llama_cpp.cmake):
llama.cpp `707616feb5f379e99ff7442680a55b4a1704c7c9`, embedded ggml 0.27.0,
archive SHA-256
`a532d0601f65cee41446cfbdc6872b98e689aa727eb366c6b6a61a70e9965759`.
The candidate adds five operation enum values and seventeen public ggml helper
functions compared with that archive. These remain private implementation
dependencies of AICore; they are not additions to the public AICore C ABI.

## Official alternatives and decisions

| Downstream operation/API | Official alternative in this revision | Contract difference and decision |
|---|---|---|
| `ggml_pow` / `GGML_OP_POW` | No equivalent op | Relation model uses a general power operation. Rewriting it as `exp(log(x) * p)` changes zero/negative-input behavior and rounding. Retain. |
| `ggml_grid_sample` / `GGML_OP_GRID_SAMPLE` | Upscaling/resizing ops | Relation model supplies arbitrary sampling coordinates and requires its border/interpolation semantics. A regular resize is insufficient. Retain. |
| `ggml_sam3d_f16_masked_attn` | `ggml_flash_attn_ext` and sparse K/V mask helpers | SAM3D requires specific F16 intermediate rounding and padding/mask behavior. No numerical equivalence has been established. Keep the operator and the ported official attention helper linkage. |
| `ggml_sam3d_sparse_conv_f16` | Dense direct/depthwise convolution | Sparse coordinates and sparse neighborhood aggregation are different contracts. No equivalent sparse convolution op exists in the pinned enum. Retain. |
| `ggml_sam3d_sparse_scatter_mean` | General scatter/reduction ops | Sparse indexed accumulation, counts and F16 rounding need an explicit comparison; a similar name is insufficient. Retain. |
| `ggml_conv_2d_direct_bias_silu` | `ggml_conv_2d_direct`, followed by add and SiLU | YOLO's extension fuses bias/activation into direct convolution. The official constructor has two inputs and no fused activation contract. A decomposition adds intermediates and launches; it requires numeric and performance A/B before replacement. Retain. |
| `ggml_mul_mat_cast` | `ggml_mul_mat`, native CUDA GEMM | The official constructor allocates F32 output; SAM3 also requires F16 output. The candidate already uses the official native GEMM implementation, with a necessary output-stride correction. Retain the output-type extension. |
| `ggml_rope_custom_freqs`, `ggml_rope_custom_freqs_cast`, `ggml_rope_custom_qkv` | `ggml_rope_ext` and frequency scales | The custom SAM3 paths consume precomputed complex frequency tensors and optionally packed F16 QKV. Official RoPE consumes integer positions plus frequency scales. Retain until inputs, layouts and rounding are independently matched. |
| `ggml_conv_transpose_2d_p0_set_prec` | `ggml_prec_set_acc` | The new precision dispatcher does not accept transpose convolution. OCR explicitly requires F32 accumulation for this operation. Retain. |
| `ggml_mul_mat_get_hint`, `ggml_elementwise_set_hint`, `ggml_l2_norm_set_hint`, `ggml_norm_set_hint`, `ggml_soft_max_set_hint`, `ggml_sum_rows_set_hint` | `ggml_prec_set_acc`, `ggml_prec_set_src` | These hints also control full-precision quantized inputs, TF32 exclusion, ordered accumulation and selected fused paths. The new APIs do not establish all those backend behaviors. Retain; replacing individual hints needs task-specific source-output and controlled timing evidence. |

The first five rows account for all five additional enum values. The table
also accounts for all seventeen additional helper functions. Official direct
and transpose convolution support must be checked per backend, tensor type and
layout: the pinned Vulkan direct/transpose paths require contiguous F32 inputs
and outputs, with F32/F16 weights. API presence does not prove support for
quantized weights or channel-contiguous tensors.

This API inventory does not cover kernels implemented under `GGML_OP_CUSTOM`,
private Vulkan entry points, or changed dispatch of existing operations. Those
patch families need the following additional review and regression surfaces:

| Patch family | Official coverage / remaining reason for downstream code | Required consumer check |
|---|---|---|
| ALIKED Vulkan | The pristine source has no ALIKED feature extraction/DCN/SDDH/DKD entry or shaders. Ordinary conv/resize/top-k do not replace the full fused pipeline. | Keypoint/descriptor source parity, empty results, fence ownership and repeated extraction |
| RMBG CUDA/Vulkan custom nodes | Official normalization, affine and attention ops offer possible decompositions; no pristine deformable-im2col or fused Swin/patch pipeline matches these private nodes directly. No decomposition speedup has been measured. | Matte source outputs, typed/image layouts, strict/fast profiles and peak memory |
| YOLO direct/IGEMM and SAM3/LingBot transpose paths | Official direct/transpose convolution exists; downstream code also changes fused activation, native output type, batch handling, cache generation and strict accumulation. Removing the extension requires the same type/layout/session cases, not one successful convolution. | Detection/segmentation heads, multi-batch outputs, destroy/recreate sessions and controlled timings |
| MapAnything attention and SAM3D precise twins | Official flash attention exists. Fixed-reference kernel partitioning and precise arithmetic affect output reduction order; the existing task-owned routes remain until the independent reference gates pass. | Every declared sequence/window profile, F16/Q8 variants and independent upstream stage outputs |
| RAM, relation and OCR precision dispatch | Default matmul/softmax/reductions exist. These families additionally select source-matching input quantization, TF32 policy, accumulator/output type and reduction order. | Original probabilities/ranks, all raw OCR class/DFL heads and both GPU backends |
| Quantized copies/Q8_1 and CUDA graph/cache fixes | Standard copy/quantization/capture exists. Bit-preserving copies, type-specific activation-sum semantics and owner/session-sensitive host work require their own cases. | Quantized block copies, Q4_1 versus K-quants, graph rebuilds and live-session isolation |
| Vulkan allocation/pipeline and build compatibility | These are resource ownership, memory fallback, shader/compiler and extension-dependency fixes rather than alternative math ops. An added official math API does not supersede them. | Small-BAR/fallback behavior, requested backend availability, shader compilation and loader checks |

Old batch backports are already superseded in the candidate: pristine 0.27.0
CPU transpose convolution loops over all batches in `src/ggml-cpu/ops.cpp`,
and Metal `src/ggml-metal/kernels/conv.metal` derives `batch` from the third
grid coordinate, offsets the input by that batch, and writes through `nb3`.
The v0.21 Metal batch-index change in `map_anything_merged` must therefore be
dropped when rebasing; it is not a remaining correctness fix. The separate
Metal input-channel parallel tree reduction in `metal_merged` is still a
performance change, with a different accumulation order. It is now ported in
`llama-patches/0007-acv-metal-operators.patch`, but has not been measured
on macOS. The port chooses a power-of-two reduction width and strides over all
input channels, including non-power-of-two counts above 256. Sixteen independent
CPU scatter-oracle cases cover F32/F16, batch 1/2 and 3/48/192/257 input
channels with exactly equal outputs. That checks the probe and official CPU
kernel; it does not certify the unexecuted Metal kernel. The candidate CPU
patch adds the precision API but does not replace the upstream batched CPU
kernel.

## Existing official-kernel reuse and necessary patches

The migration keeps upstream CUDA native GEMM rather than restoring the old
implementation wholesale. Its F16 destination exposed a stride bug:
`nb1 / sizeof(float)` gives the wrong cuBLAS leading dimension. Ordered patch
[`0006-acv-cuda-native-output-stride.patch`](llama-patches/0006-acv-cuda-native-output-stride.patch)
uses `nb1 / ggml_element_size(dst)`. The retained negative control fails with
the old module; the corrected module passes all six scalar-reference
single/batch/broadcast F16/F32 cases. Real SAM3 F16 repeats pass the focused
numerical check. See the unchanged historical
[`cuda-native-stride-evidence.json`](validation/cuda-native-stride-evidence.json).
This evidence justifies keeping that patch; it is not an all-model speed claim.

The precision-port patches also forward the **official** sparse attention
helper correctly through the precision namespaces. A clean shared-module link
alone did not detect the original unresolved helper symbols; retain loader
relocation checks alongside compilation.

## Reproducible refactor checks and outstanding gates

`build_llama_config_refactor_glj5ibw7/` contains this audit's pristine source
copies and fresh refactor checks. Before/after generated configure commands,
source URL/SHA and patch signatures match for baseline CPU, candidate CPU,
candidate static and candidate CUDA+Vulkan configurations. All twelve archived
JSON receipts retain their original SHA-256 values. The fresh CPU
ExternalProject downloads/verifies, patches, compiles and installs, then passes
both CTests; its configure-contract suite has fourteen tests. Missing requested
Vulkan dependencies now reject the candidate before download, rather than
silently creating a CPU-only build. The baseline behavior is preserved.

None of these checks qualify OCR for publication. Original-head quantization
failures remain independently reproducible in Paddle, even where both native
GPUs exactly match the stored-weight reference. RF-DETR's corrected full
44-model matrix has baseline-identical numeric fingerprints, but still fails
timing comparisons for six CUDA and twenty-two Vulkan models. The serial
22-task full-tier queue remains in progress, with per-process attempts kept
immediately after completion. Do not delete the old production source entry,
remove precision paths or publish a speedup until the required complete
numerical, stability and controlled performance comparisons pass.

The 2026-10-22 developer-profile rerun now matches CPU ISA builds on both
sides (`GGML_CPU_ALL_VARIANTS=ON`, Haswell/AVX2 module actually loaded).
RF-DETR's 44 model fingerprints still match on each GPU backend, while 14
CUDA and 44 Vulkan model rows flag timing regressions. This shorter sampling
protocol differs from the historical warmup A/B above; both are retained.
Matching CPU dispatch has not eliminated the regression, so it cannot be
reported as the root cause. The current frozen install independently passes
36 public ABI/lifecycle contract CTests with verified library loading; this
does not clear any failed model or performance gate.

### 2026-10-22 Vulkan optimizer investigation

The retained four-condition developer experiment
`build_llama_dual_gpu_3nl01j4j/stride-fix/rfdetr-vulkan-four-way-ab-20261022b/summary.json`
uses four interleaved processes, five warmups and thirty measured forwards per
condition. RF-DETR base Q8 forward/e2e deltas are +7.07%/+9.26% with graph
optimization enabled and +2.80%/+2.00% with it disabled on both owners; every
condition has the same output hash. This localizes a substantial part of that
experiment's regression to the optimizer path, without proving which optimizer
change is responsible. Desktop activity prevents release qualification.

Candidate patch `llama-patches/0009-acv-vulkan-pattern-short-circuit.patch`
returns immediately on a failed fusion-pattern element. It changes neither the
match result nor reordering, allocation dependencies, or supported fusions. The
new upstream view-alias dependencies remain necessary for correctness; see
[upstream fix #27812](https://github.com/ggml-org/llama.cpp/pull/27812).
Both Vulkan manifests include the patch. The pinned archive SHA, complete
forward replay, second replay, independent compile/link and nineteen CMake
contracts pass. A separate module directory preserves the active matrix's
frozen libraries. The first two patch A/B experiments were invalid: ggml's
explicit executable-directory loader selected the unpatched module despite
`LD_LIBRARY_PATH`. Each directory now contains `measurement-validity.json`;
neither experiment establishes a patch performance result.

The replacement `stride-fix/vulkan-exact-module-ab-sgg8s2dc/` copies the probe
into each private module directory and checks the actual loaded Vulkan path
and SHA-256 after every process. Ordered patch
`0010-acv-vulkan-dependency-metadata.patch` precomputes source/view-base metadata
within one optimizer call. The original direct and view-alias dependencies
are preserved: 1,000 generated graphs cover 3,062,201 dependency pairs without
a mismatch. The complete ten-patch chain replays from the pinned archive and
is idempotent; its Vulkan source matches the independently compiled module.

Four interleaved processes, five warmups and thirty measured forwards produce
the following medians on this machine. These are developer measurements,
without release GPU-idle qualification.

| RF-DETR base | Old forward / e2e, ms | Unpatched candidate | Patched candidate | Patched versus old |
|---|---:|---:|---:|---:|
| F32 | 35.2228 / 36.3438 | 36.5567 / 37.9775 | 35.6872 / 37.0494 | +1.32% / +1.94% |
| Q8_0 | 33.4813 / 35.1424 | 36.2222 / 38.0596 | 34.7304 / 36.5607 | +3.73% / +4.04% |

All raw output hashes match, and the patched/unpatched candidate optimizer
orders match. This establishes a focused improvement, not an all-model pass.
The isolated complete 44-model check and ALIKED/AnyCalib repeats are at
`stride-fix/vulkan-fixed-full-validation-9aqib7e8/`; they use the fixed module
in both executable search directories and leave the original matrix untouched.
The fixed-module RF-DETR Vulkan retry now passes all 44 models, and AnyCalib
passes all three. These runs retain the default +5% threshold and 3 ms absolute
noise floor; a PASS does not mean every relative delta is below 5%. The ALIKED
retry stopped before inference because its F32 file was absent from the shared
cache. The common catalog download/SHA helper subsequently restored that exact
asset; this preflight failure is not numerical or performance evidence.

The separate actual-optimizer host checker extracts both complete optimizer
implementations and fusion helpers, links the real ggml graph/fusion APIs,
and compares full node orders and ordered allocation-dependency callbacks.
Patch 0009 versus 0010 passes 2,000 generated graphs, including 666 injected
QSA sequences and 154 allocation callbacks. The optional 0011 QSA-presence
experiment passes the same comparison against 0010 but remains outside the
active patch manifests. Its four-process comparison improves base F32 forward
by 4.00% but changes base Q8_0 by +0.95%, without a stable cross-model benefit;
the experiment stays in the build artifact directory and is not promoted into
the source patch set. See `vulkan-qsa-presence-92w2cliw/ab-summary.json`.
The separate LightGlue diagnostic compile subsequently failed at link time;
that failure does not qualify its unexecuted typed-output checks. Host shell
stubs do not establish Vulkan device correctness; the 44-model check above
is separate device evidence for 0010.

The fresh current-source 22-task full-model CUDA/Vulkan matrix is at
`validation-current/candidate/Testing/complete-task-matrix-current-20261022T1135/`.
All twenty Depth rows execute on each backend with passing task numeric checks,
but old/new byte fingerprints differ and several timings fail. Those failures
remain open until actual depth/confidence/pose/intrinsics arrays are compared;
CPU/backend MAE checks alone cannot establish old/new equivalence. External
source-reference rows and macOS Metal device validation are still required
before retiring the production ggml entry.

Public typed-array diagnostics now distinguish the two Depth resolutions.
All ten 504-target Vulkan device fingerprints match; the corresponding CPU
fingerprints differ, so the combined CPU/device fingerprint still fails. A
base-F16 typed 504 capture confirms identical Vulkan depth, confidence, pose
and intrinsics. The 224-target capture with the probe's default thread option
does differ: CUDA depth maximum/mean absolute differences are
`4.95434e-4 / 5.29439e-5`; Vulkan's are `8.86500e-3 / 5.86345e-4`.
Both GPU poses, intrinsics and shapes match. The 224 confidence differences
remain recorded separately. See `depth224-comparison.json` in the isolated
full-check directory. These bounded diagnostic numbers do not waive the
unchanged exact-fingerprint failures or establish original-framework parity.

The Gaussian CUDA group was invalidated by a manifest edit during execution.
Its separate frozen-manifest rerun passes all nine rows, as does the original
Vulkan group. Keep the original invalid report; the replacement receipt is
`stride-fix/gaussian-cuda-manifest-retry-20261022/gaussian-cuda.json`.

The subsequent ALIKED Vulkan retry passes three rows and the GKD CUDA retry
passes four, with three process repeats, five warmups and ten measured
forwards. Their backend logs resolve to `validation-current/candidate/bin`,
not the isolated 0010 module directory: these are current unpatched-module
retry results, not additional device validation for patch 0010. The longer
FaceDetect retry includes the previously uncovered unsuffixed SCRFD file,
executes ten rows per backend, and still fails byte fingerprints; CUDA also
retains one landmark timing failure. External compute processes observed on
the same GPU prevent treating these developer timings as controlled release
measurements. See `stride-fix/typed-precision-retry-3hpwhc75/`.

Depth intermediate captures identify graph 1, node 19 (`MUL_MAT`) as the first
different Vulkan operation at target 224. Its actual contiguous F16
`[48, 6912]` image-column operand and F32 `[768, 6912]` weight operand match
bit for bit across owners; backbone and camera graphs match. An independent
double-precision multiplication of these operands gives mean absolute errors
of `0.00394950409` for the old output and `0.00252681234` for the new output.
The candidate is closer to this operation's mathematical truth. This does
not establish final-task framework parity or waive the current gate. Traced
final public outputs match each owner's untraced outputs exactly. Receipts:
`stride-fix/depth-matmul-inputs-bf_2nrhc/independent-matmul-truth.json` and
`graph-comparison.json`. Fusion, graph-optimizer and SAM3 transpose controls
do not explain this first divergence. The short-M split-K guard is being
tested separately as a one-variable causal control; it is not a promoted
compatibility patch.

The six synthetic LoMa matcher retry rows in the typed-precision directory
have an invalid baseline runtime. The copied current probe retains a
`DT_RPATH` into the candidate tree, which takes precedence over
`LD_LIBRARY_PATH`: `LD_TRACE_LOADED_OBJECTS` confirms new `libAICore.so` with
old `libggml.so.0` and `libggml-base.so.0`. Their "no usable device" messages
are a mixed-runtime harness error, not evidence of a LoMa or driver failure.
The unchanged published-model baseline probe loads the consistent old tree
and succeeds. A portable probe with no absolute RPATH is staged at
`stride-fix/portable-probe-retry-7jv_nxnu/`; only its completed, library-identity
verified results may replace those invalid matcher rows. Original invalid
reports remain available. The full-matrix baseline uses its retained original
probe, so this copied-probe error is specific to that follow-up experiment.

The completed portable-probe retry confirms six selected LoMa CUDA rows pass.
All selected matcher numeric fingerprints also match on Vulkan, which retains
one G-model published-probe process-wall regression of 8.77%. Its exact loaded
AICore identities and probe SHA are in `portable-probe-retry-7jv_nxnu/`.

The one-variable short-M split-K control is now complete. Restoring only
`m >= pipeline->wg_denoms[0]` in the guesser makes all five public base-F16
Depth arrays at target 224 bit-exact to the old runtime, with all 619 captured
backbone/DPT/camera tensors matching. Traced outputs also match the untraced
control. LightGlue's six score fingerprints remain different under this
control, so its cause is independent. See
`stride-fix/vulkan-short-m-split-control-1gm8kbeg/`.

The unregistered `0011-acv-vulkan-depth-short-conv-split-k.patch` limits that
compatibility routing to short F16-im2col/F32-weight DPT convolutions under the
checkpoint's `head.resize.` / `head.scratch.` namespaces. Other matmuls retain
upstream split-K selection. Forward/reverse patch application succeeds on the
ten-patch source, but device routing controls and performance qualification
are still pending in `stride-fix/vulkan-depth-selective-9rb13swh/`. It is not
yet part of either experimental manifest or the production patch chain.

Device qualification update: the selective DPT module restores all five final
base-F16 outputs and all 619 captured nodes at target 224. The actual captured
`48 x 768 x 6912` multiplication follows the old guard only with both the
im2col structure and DPT weight namespace. With unrelated weight names, or
without that structure, its output remains bit-exact to the new upstream
route. All three module identities are printed from process maps and match
the intended module files. This closes the positive/negative routing controls;
full Depth model coverage and controlled performance are still required before
registering patch 0011. The downstream release-sentinel command error in the
experiment's `state.json` does not invalidate these completed device controls.

LightGlue's first changed operation is now isolated independently: graph 0,
node 1604 is the F32 `[256, 1]` matchability weight multiplied by F32
`[256, 64]` descriptors, with identical input bytes and `GGML_PREC_F32` in
both owners. The new Vulkan dispatcher exchanges inputs for a one-row output
and uses GEMV. Against an independent double-precision product, ALIKED's
maximum error decreases from `0.00159187293` to `7.83533316e-7`; SIFT's
decreases from `0.00115371115` to `3.88425246e-7`. All 64 match indices remain
unchanged, and each traced index/score result equals its own untraced run.
Thus the changed fingerprint is not evidence of a less accurate operation.
Keep the exact-gate failure open and establish task-level upstream truth;
do not disable the new upstream GEMV merely to reproduce the older error.
Receipts: `stride-fix/lightglue-matchability-inputs-isd2ac1z/`, especially
`independent-matmul-truth.json`, operand metadata and final-output invariance.

An additional independent reference imports the official LightGlue Python
implementation at `eb42fee2d71449efb0aa5c10549752b5d75384d8`. It executes all nine
transformer blocks and the assignment head from the common, bit-exact captured
input projection, loading every parameter from the actual F16 GGUF. Both old
and new Vulkan outputs retain all 64 official match indices for ALIKED and
SIFT. ALIKED score MAE changes from `0.00045569235` to `0.00044652590`, while
maximum error changes from `0.00716078269` to `0.00718051169`; SIFT MAE changes
from `0.00039310337` to `0.00038479071`, while maximum error changes from
`0.00316187697` to `0.00318214297`. Thus not every error metric improves.
This stored-weight CPU framework diagnostic is not a real-image or original
checkpoint accuracy gate and does not close the exact-fingerprint failures.
Source SHA, model SHA, parameter mapping, arrays and comparisons are retained
in `stride-fix/lightglue-official-graph-o_7qdf7x/`.

### Current full-matrix and independent retry scope

The current full queue has completed the first 23 backend/task groups,
including SAM3's 39-model CUDA group. The table distinguishes valid full-task
developer retries from narrow diagnostic selections; none is a controlled
release qualification. A retry result belongs to its recorded library/module
SHA and cannot qualify a different candidate automatically.

| Task | Rows per backend | CUDA evidence | Vulkan evidence | Remaining acceptance |
|---|---:|---|---|---|
| ALIKED | 3 | PASS, main matrix | PASS, typed retry | Controlled release |
| AnyCalib | 3 | PASS, main matrix | PASS, fixed-module retry | Controlled release |
| DeepLSD | 6 | PASS, main matrix | PASS, main matrix | Controlled release |
| Depth | 20 | FAIL, fingerprints | FAIL, fingerprints; selective DPT routing verified on one model | Full model/source/performance qualification |
| FaceDetect | 10 in current retry | FAIL, five fingerprints and one timing | FAIL, four fingerprints | Typed upstream numeric diagnosis and controlled timing |
| FreeSplatter | 9 | PASS, unchanged-manifest replacement | PASS, main matrix | Controlled release |
| GKD | 4 | PASS, typed retry | PASS, main matrix | Controlled release |
| LightGlue | 6 | PASS, main matrix | FAIL, score fingerprints | Full source-checkpoint/real-input numeric qualification |
| LoMa | 27 | FAIL, full task; six selected retry rows PASS | FAIL, full task; six selected retry rows retain one timing failure | Full-task diagnosis; subset is not complete coverage |
| RF-DETR | 44 | PASS, portable full retry | PASS, fixed-module full retry | Controlled release |
| RMBG | 6 | FAIL, fingerprints | FAIL, fingerprints and timings | Numeric diagnosis and controlled timing |
| SAM3 | 39 | FAIL, 26 timing checks; no numeric/stability failures; interrupted base-plus F32 row separately rechecked PASS | Pending | Timing diagnosis, complete Vulkan group and controlled release |
| SAM3D / TRELLIS / YOLO / MapAnything / LingBot / ReID / OCR / Table / Tagging / Relation | Manifest-expanded | Queued | Queued | Completion, source references, failures and controlled release |

The main reports are under
`validation-current/candidate/Testing/complete-task-matrix-current-20261022T1135/`.
Replacement directories are `typed-precision-retry-3hpwhc75/`,
`portable-probe-retry-7jv_nxnu/`, `vulkan-fixed-full-validation-9aqib7e8/`,
`gaussian-cuda-manifest-retry-20261022/` and
`priority-diagnostic-batch-kpicsy1c/` under `stride-fix/`.
`current-reference-followup-buca6r5k/` binds independent references to the
current candidate after this queue, including pinned MapAnything source builds.
Queued work is not validation evidence. Metal device validation remains open;
the old production source entry must remain until the required matrix passes.

SAM3 CUDA's completed report contains 39 rows and 26 failed timing comparisons,
covering load, encode and PVS median/p95 measurements. It reports no numerical
or repeatability failures. These developer samples are not controlled release
evidence and do not establish whether each timing difference is reproducible.
The earlier interrupted process remains separately documented; its selected
replacement cannot replace the other failed timing comparisons.

Private FaceDetect/RMBG copies export exactly the byte stream used by their
existing fingerprints. All 64 independent processes completed, producing
32 baseline/candidate comparisons. Every dump matches its printed FNV hash
and its historical report; all numeric values are finite. The recorded owner
libraries are separate, and the candidate uses the pre-0010 Vulkan module
`fdb307c4…`, not the fixed `15970710…` module.

| Diagnostic output | CUDA maximum difference | Vulkan maximum difference |
|---|---:|---:|
| Changed FaceDetect embedding packs | `1.65781e-4`; cosine distance `8.39415e-7` | `4.63415e-4`; cosine distance up to `6.23550e-6` |
| SCRFD F32/default detection boxes | `0.00338745` pixels; score `2.27690e-5` | Exact in these rows |
| Dense landmarks, including detection fields | `0.0417213` | `0.0790558` |
| RMBG 1024x1024 alpha, optimized/strict | 8–20 changed pixels, each one gray level | 4–33 changed pixels, each one gray level |

These values quantify changed fingerprints; they do not establish upstream
accuracy or waive the exact gate. Keep numerical diagnosis and timing failures
open. Commands, source/module identities, all byte dumps, hash checks and
comparisons are in `stride-fix/face-rmbg-typed-s5tmpnlj/`, especially
`origins.json`, `state.json` and `comparison.json`.

### 2026-10-22 user-approved scope and pending dispatch fixes

The later user-approved large-model waiver supersedes the queued scope in the
historical table above. `validation/regression-waiver-policy.json` selects
344 catalog rows (326 distinct files) across 18 tasks and waives 212 catalog
rows (200 distinct files). Waived rows are compatibility assumptions, never
measured accuracy or speed passes. OCR has no waiver and retains every original
class and per-edge DFL argmax comparison. The common runner expands the selected
files into 306 scenarios per backend, with no missing or uncovered input assets.

The new serial CUDA/Vulkan run is under
`build_llama_dual_gpu_3nl01j4j/scoped-regression-20261022/`.
Its preflight exposed 18 stale YOLO26 OBB/semantic-segmentation SHA entries;
these now match the existing official GitHub Release asset digests. Both the
candidate AICore and the separately compiled catalog dumper include those
updates. This repairs asset verification, not an inference accuracy result.
The active run uses CPU all-variants builds on both owners and preserves the
baseline install. Pending scenarios and successful process exits are not PASS.

Two isolated dispatch fixes remain outside the active patch manifests:

- `0011-acv-vulkan-depth-short-conv-split-k.patch` preserves the previous
  short-M split-K rule only for DPT head im2col contractions. One base-F16
  224-target capture restores all 619 tensors and all five public outputs.
  Positive DPT and unrelated-name/plain negative controls pass. The remaining
  model and performance checks are still required.
- `0012-acv-cuda-facedetect-thin-im2col.patch` preserves cuBLAS dispatch for
  FaceDetect-owned im2col contractions newly routed through half2 MMVF.
  Forward/reverse patch replay and private compilation/linking pass. Device
  output and model performance qualification have not yet completed.

These patches must be tested with verified actual loaded modules in private
directories. Existing probes contain absolute DT_RPATH entries, so changing
LD_LIBRARY_PATH alone cannot prove an isolated library was exercised.

### RMBG intermediate-capture correction

The earlier RMBG `face-rmbg-firstops-wmucmig0` IM2COL attribution is invalid.
RMBG explicitly calls `ggml_gallocr_reserve` before `alloc_graph`; the diagnostic
marked retained outputs only at `alloc_graph`. The allocator then reused its
already valid allocation plan, allowing later nodes to overwrite captured
intermediates. Unchanged final public outputs alone did not detect this error.
The final alpha-byte comparisons above remain valid; the overwritten buffers
cannot identify the first differing operation.

A host-only three-square control reproduces the issue independently with both
runtime versions: marking the first output after reserve reads 256 instead of
4, while marking it before reserve reads 4; the final output is 256 in every
case. The corrected private interposer marks retention before reserve and is
queued after the selected matrices and dispatch qualification. It captures
only the first twelve nodes, checks actual loaded modules and requires final
output invariance again. Receipts and pending state are under
`stride-fix/rmbg-reserve-capture-r3xrm4u1/`. This correction is diagnostic work,
not a backend fix or a passing model result.

Primary source references:

- [Pinned public ggml API](https://github.com/ggml-org/llama.cpp/blob/707616feb5f379e99ff7442680a55b4a1704c7c9/ggml/include/ggml.h)
- [Official tensor constructors and precision dispatch](https://raw.githubusercontent.com/ggml-org/llama.cpp/707616feb5f379e99ff7442680a55b4a1704c7c9/ggml/src/ggml.c)
- [Official CUDA operation dispatch](https://raw.githubusercontent.com/ggml-org/llama.cpp/707616feb5f379e99ff7442680a55b4a1704c7c9/ggml/src/ggml-cuda/ggml-cuda.cu)
- [Official Vulkan implementation](https://github.com/ggml-org/llama.cpp/blob/707616feb5f379e99ff7442680a55b4a1704c7c9/ggml/src/ggml-vulkan/ggml-vulkan.cpp)
