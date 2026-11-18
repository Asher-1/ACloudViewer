# RelateAnything upstream incremental synchronization

## 2026-10-29: current-source completeness check

The current local upstream HEAD is `537f48f39742600d0dc7ab1b2fcccbc8767b073e`.
Its tree is `1ca78894256a5f2c80e9a2fc8af9e52d10da0615`, identical to the tree
of `14c9744e3cceb14912b1bfc5851343741f3a96e5`; `git diff 14c9744 HEAD`
is empty. The commit histories differ, so a different HEAD alone does not
establish an additional source change. No remote fetch was performed.

The upstream ggml submodule remains at
`8599e0ea3756c4bac4ef813af2241cb1a8bbfb0b` and is dirty. Its actual diff and
`cpp_ggml/third_party/patches/0001-relateanything-ggml.patch` have the same stable
patch-id `5ac875218ebb3f62e297f45dd2b31c2b950d72c5`; reverse-apply checking succeeds.
This identifies the local patch application, not a clean pristine runtime.

| Surface | Current evidence | Remaining boundary |
|---|---|---|
| CUDA/Vulkan DW7 | Registered in applicable manifests; current `build_app` source has both routes, and `nm -C` finds the CUDA kernel plus both Vulkan shader blobs/selector in the backend modules | Symbol presence is not execution or release evidence |
| Relation host | `UPSTREAM.json` still pins `984833746e8a2aa04e7a70fc6e281dcc3bb7db54`; current `full_model.cpp` keeps separate DW and ADD and has no `ggml_elementwise_set_epilogue_hint` calls | Upstream ConvNeXt direct-bias, ConvNeXt GELU hint and decoder FFN GELU hint have not been ported |
| CUDA ADD/GELU fusion | Upstream patch explicitly calls `ggml_cuda_op_add_gelu_erf` for a flagged ADD/GELU_ERF pair; current generated CUDA source has no corresponding route | Local CUDA retains separate computation; merely adding host hints would not integrate this upstream optimization |
| Profiling | Upstream host diff also adds `frameA`/`frameB` diagnostic timing output | Not carried as local inference or common-timing evidence |
| Acceptance | Historical DW7 operator/consumer receipts below remain retained | Consumer performance violations, missing large-F32 and original release gates remain open |

Therefore the approved DW7 increment is integrated, but the complete upstream
host/backend change set is not. The task-owned options adaptation and exclusion
of upstream reporting/gallery policy remain intentional; neither is replaced
with process-global state or upstream acceptance thresholds. This check does
not authorize a new host port, change the source pin or rewrite any old receipt.

### Subsequent user-approved bounded follow-up

#### Actual RED and minimal-implementation progress

The original OCR has fully exited; the stale wait description below is kept as
a historical boundary. The first CUDA numeric startup failure was CUDA 11.8
modules mistakenly loading the system 11.1 toolkit, missing
`cudaLaunchKernelExC`; the startup failures that wrongly took
`CMAKE_CUDA_COMPILER` from the parent cache, and where an old nsys printed a
valid help yet exited 1, are retained as well. The launcher now obtains the
toolkit from the real owner cache under `GGML_BUILD_DIR` and pre-checks with
`ldd -r`; nsys still rejects missing reports and true errors without lowering
the trace/numeric bar.

`build_app/Testing/relate-add-gelu-red-x1p4mzp_/receipt.json` records the
256-case numeric comparison per backend, all rc=0;
`routes/run-1fobgx5r/receipt.json` fully executed 7 CUDA traces: only the hint
mode expected fusion yet actually still ran ADD + GELU as two kernels — a
valid RED. Original runtime/probe identity unchanged. This is operator/route
evidence, not checkpoint or performance acceptance.

The approved minimal implementation is written: `full_model.cpp` wires the
ConvNeXt DW direct-bias plus the pwconv1 and decoder-FFN GELU epilogue hints;
`0027-acv-cuda-add-gelu-epilogue.patch` is registered in the CUDA and combined
ordered chains. The kernel reuses the original `op_gelu_erf`, keeping the
fusion switch, the complete single-consumer domain, actual pointer/layout
checks, and the dangerous-alias fallback. Gallery/materiality policy was not
copied; public ABI, source reference pin, models, original thresholds,
baseline, and old FAIL receipts were not modified.

`build_app/Testing/build_relate_gelu_candidate_v1.py` performs pristine
forward/reverse/idempotent replay in a unique directory; it accepts only a
byte-identical original build source or exactly the three patch-source
differences, then compiles the two CUDA TUs with matched original flags while
every other object stays read-only and identity-checked.

#### Candidate compilation, routing, and real-consumer results

`build_app/Testing/relate-gelu-candidate-cvos3vzk/receipt.json` records the
four-chain forward/reverse/idempotent pass; two CUDA TU compilations,
independent module linking, the loader, the 256-case numeric comparison per
the three backends, and the seven routes are all rc=0. The hint actually
executes `add_gelu_erf_bias_kernel`, the other six routes stay separate,
failures=[], operator_route_green=true; candidate module SHA-256 is
`405d8e1f074299cfbf36571b749b4876057553c6289543f84d62fef6f15aa323`.
Original object/source identity unchanged; host_compiled=false and
release_qualified=false remain truthfully retained. The pure-replay
`relate-gelu-candidate-nif_o1d3` is a separate historical receipt and cannot
substitute the compiled result.

The host side additionally recompiled `full_model.cpp` with matching
`build_app` flags, replacing only that object in the standalone link command,
producing a new monolithic `libAICore`.
`relate-host-consumer-potqw5g7/receipt.json` records the host compile/link and
loader pass, with byte-identical RAFO across before -> candidate -> before for
24 sample-0 model/backend combinations. The serial guard then stopped adding
samples when an external compilation appeared; the continuation reused the
same snapshot bytes and completed the remaining 4 combinations in the new
directory `relate-host-consumer-k30cubhk`, without overwriting old receipts.
The two receipts cover 28 combinations (11 ConvNeXt plus the ViT control),
all with real backend paths and bitwise output verification; the original
library, inputs, candidate, and old receipts kept their identities.

The four original Vulkan p95 1.05-gate FAILs are retained: tiny-F16 1.078470,
small-F16 1.063624, small-Q8 1.050964, ViT-F16 control 1.091177. The original
protocol is still threads=4, warmup=2, GPU repeat=10/CPU repeat=3; this is a
single-bracket local comparison, not a complete controlled release-profile
A/B. large-F32 is still missing; three-source independent checkpoints, the
original frozen performance gates, and native platforms remain open.
host_compiled=true does not equal release_gate_closed=true.

The test-queue description below is an earlier observation; it no longer means
the OCR still holds the lock.

The user subsequently approved the separate host/direct-bias/GELU-hint and
CUDA ADD/GELU fusion increment, with original models, source reference gates,
performance thresholds and frozen baseline unchanged. Upstream gallery and
materiality policy remain excluded.

Only test changes are currently written: the existing elementwise probe gains
seven isolated trace modes and 256 plain/hint numeric cases including an
observable intervening node. `experimental/test_cuda_add_gelu_routes.py`
requires real CUDA execution, positive kernel instance counts, the expected
fusion or fallback and unchanged input/backend digests. Its CMake test is
registered without silently omitting a missing nsys dependency.

`build_app/Testing/run_relate_add_gelu_red_v1.py` compiles a new test object and
executable in a unique build-local directory using the historical driver's
matching compile/link arguments. It does not rebuild or overwrite the old
runtime, probe or receipts. The queue in `relate-add-gelu-red-queue-v1.log`
waits on the original lock, at least 18000 MiB available RAM, at least 2048 MiB
free GPU memory and no active build/compiler/linker. Compilation is one process;
the three backend numeric checks and CUDA traces run serially.

The original OCR matrix still owns the lock. No real route RED, new production
patch/host edit or GREEN has occurred yet; source/helper checks are not route
execution. Observe the queued result before implementing the approved increment.

## 2026-10-28: 14c9744 DW7 increment

Status: the approved CUDA/Vulkan DW7 patches are implemented and operator/consumer output checks passed; performance qualification remains open. None of the following closes the original frozen release gate.

### Exact source and applicable scope

- Read-only upstream repository: `../dl/RelateAnything-GGML`.
- Commit: `14c9744e3cceb14912b1bfc5851343741f3a96e5` (`fix issues`); parent: `b1bea71f811206a9df8207e142eb7a24513845cc`.
- The exact diff covers nine files, 721 additions and 102 deletions. It does not modify `full_model.cpp`.
- The core increment is CWHN 7x7 depthwise specialization and operator regression in the unified upstream ggml patch; registered here as `llama-patches/0024-acv-cuda-depthwise7.patch` and `llama-patches/0024-acv-vulkan-depthwise7.patch`.
- CUDA, Vulkan and combined ordered manifests register their applicable patches; the CPU-only chain adds no GPU specialization.
- Upstream `run_ggml.py`/launcher gallery tooling, report scripts, README/RELEASE_STATUS and cleanup records have no matching local inference owner. As approved, this increment does not copy the independent gallery toolchain, perform cleanup/deletion or use upstream completion records as local evidence.
- Materiality/bootstrap classification and status-name corrections in upstream `write_acceptance_status.py` are its reporting policy; they do not replace the original local accuracy/performance gates.
- `core/AICore/src/tasks/relation/full_model.cpp` and `UPSTREAM.json` were unchanged. The old direct-bias/epilogue host integration remains independently open; engineering references, models, thresholds and the frozen baseline were not rewritten.

### Minimal implementation and local invariants

Each thread/invocation computes four channels and seven adjacent output columns, reusing input rows. With 49 fixed taps, complexity remains O(output elements), without an extra full-image tensor. Preserve ascending ky/kx accumulation per output, skipped boundary taps and separate bias ADD after convolution write-out.

- CUDA: `conv2d_dw_7x7_cwhn_kernel` requires F32 input/weights/output, CWHN, 7x7, stride=dilation=1, channels divisible by four and actual input/weight/output addresses aligned to 16 B. Other cases retain the original kernel. Scalar bias loads support local 4 B-offset views; `__fadd_rn` preserves separate ADD. F16 weights retain the original generic kernel.
- Vulkan: F32 input/output and F32 or F16 weights, with the same shape conditions and 16 B-aligned actual input/output buffer/view offsets. Preserve all four descriptors (kernel/input/output/optional bias) and `a_offset/b_offset/c_offset/d_offset`. DW7 dispatch counts tile invocations through the original workgroup splitting path; ineligible cases fall back. Retain the existing `DW_F16` rounding path.
- Specializations belong to the existing shared runtime owner; plugins and the public C ABI add no ggml interface, model table or second inference implementation.

### Regression and controlled comparison

New regressions in `experimental/` are registered by the existing CMake driver: `depthwise7_test.cpp`, `test_cuda_dw7_routes.py`, `vulkan_dw7_routes_test.cpp` and `test_vulkan_dw7_routes.py`.

- RED: eight eligible cases out of 32 in the old real Vulkan selector did not specialize; the old real CUDA trace ran the generic DW kernel for an eligible request. An initial empty trace caused by an unsupported nsys report name is not valid RED evidence. Scripts support `gpukernsum`/`cuda_gpu_kern_sum` and reject `ERROR:` in report bodies.
- Numerics: nine shapes x F32/F16 weights x bias on/off x prefix 0/1/4, totaling 108 cases/backend. Cover batches, boundaries, tail tiles, inputs smaller than the kernel, padding variants, non-fourfold channels, stride/dilation and WHCN fallback. Verify independent-window `std::fma` oracle error <=1e-5, complete prefix/suffix sentinels, finite values and bitwise stability across three repeats.
- Routing: real CUDA kernel traces verify eligible and unaligned fallback paths; Vulkan tests compile the actual backend selector and cover independent input/output offsets. Selector tests do not replace shader numeric checks.
- All four pinned pristine chains passed forward application, complete reverse application and second-replay idempotency.
- Isolated candidate: `build_relate14_dw7_20261028`, Linux Release, CUDA 11.8, architecture 86, RTX 3060/driver 550.144.03; complete CUDA+Vulkan combined build. Default one build at `-j4`, serializing performance sampling and compilation. Inherited warnings remain; this is not a warning-free-build claim.
- Additional comparison: `build_relate14_no_dw7_20261028`, replaying the current formal combined chain from the same pinned archive with only the two DW7 patches excluded. The full ggml trees differ in only five DW7 source files; headers are identical. Unchanged CUDA objects are reused only after normalized compiler flags/include arguments match; DW files, Vulkan and the remaining runtime are rebuilt. This directory is not the original frozen production baseline.

Verification commands and recorded results:

```sh
ctest --test-dir build_relate14_dw7_20261028 --output-on-failure -j1
python3 3rdparty/llama/experimental/test_cmake_contract.py
python3 core/AICore/tests/relation/test_accuracy.py
python3 core/AICore/tests/relation/test_upstream_probe.py
python3 core/AICore/tests/check_plugin_aicore_boundaries.py --root . --strict
```

The candidate passed 25/25 CTests, including five DW7-specific checks and existing depthwise-bias/epilogue/pool/im2col/accumulation/stride/debug-clone/runtime regressions. Independent Python contracts passed 23/23, accuracy-policy unit tests 9/9, upstream identity/policy unit tests 21/21 and plugin-boundary checks passed. Policy unit tests are not complete checkpoint accuracy qualification.

The same probe running against matching runtimes passed 108 oracle cases each on CPU/CUDA/Vulkan, with identical before/after raw bytes. All three dumps have SHA256 `6cc4bdc21aa9248de829416ac83478cef0192f59571b184ff9d05b129f13696e`.

Public relation consumers retain the same monolithic `libAICore` and catalog sample 0 (`ra4m_50/images/9282.jpg`, SHA256 `6f68010af1c8cc8b59aaaf3a5acbeeb1c8b775db78fb06db50ad54023e8617f0`). Verify original model-catalog sizes/SHA values; threads=4, warmup=2, GPU repeat=10/CPU repeat=3, before->candidate->before. Eleven local ConvNeXt models plus the `vits16plus-f16` control cover 28 model/backend combinations, with byte-identical RAFO outputs. CPU covers only tiny and the ViT control; other families cover CUDA/Vulkan. `convnext-large-f32` is missing; no model was downloaded or replaced. Sample 0 and same-weight byte comparison do not replace a complete dataset or independent-checkpoint correctness.

### Observed verification pitfalls and open performance items

- Old `build_llama_dw_bias_20261026` uses `GGML_MAX_SRC=10`/`GGML_MAX_OP_PARAMS=64`, versus candidate 30/256, with different `ggml_tensor` layouts. A probe compiled with candidate headers against the old runtime produced 108/108 sentinel failures; this does not establish DW7 arithmetic regression. A probe compiled with old headers against old CPU exposed 21 further F32/CWHN/bias failures; retain them outside this increment. New operators are appended to the enum; existing DW op IDs remain unchanged.
- Public probe `DT_RPATH=$ORIGIN/..` takes precedence over `LD_LIBRARY_PATH`; copying only libAICore does not change the actual owner. The first consumer attempt loaded `build_app` backends, was interrupted and marked `evidence_valid=false`. Copy the identical executable into isolated `aicore_tests/`, assert AICore/ggml paths in ldd before execution and all loaded backend paths afterward. Original executables/shared libraries were not overwritten.
- Operator A/B uses warmup=20, iterations=100 per stage, three alternating process rounds and synchronization per iteration. Four-stage CUDA/Vulkan p50 ratios are 0.298-0.358 / 0.257-0.489; these are local operator comparisons, not release speedup claims.
- Initial CUDA 14x14 p95: 0.112020->0.183214 ms (ratio 1.63555, FAIL retained). Five-round CUDA remeasurement with the same protocol: 0.111539->0.038763 ms (ratio 0.34753), without reproducing the first violation or overwriting its receipt.
- The first valid public-consumer run still exceeded 1.05 in forward p50/p95 for tiny-F32/CPU and ViT-F16/CUDA. CPU tiny-F32 p50 increased 6.59%, with baseline-bracket drift of 16.43%. Three-round CPU/CUDA subset remeasurement for both models still contains individual violations, including the ViT CPU control. Preserve all results without changing thresholds, dropping samples or treating unstable brackets as no-regression evidence.
- Consumer performance violations, missing large-F32 and the unclosed original frozen release matrix keep `release_gate_closed=false`. Operator passes do not close other platforms/native packages, complete UI, original OCR/LLM/YOLO/AnyLabeling failures or pending work.

### Local evidence locations (relative to repository root)

- Four-chain replay: `build_relate14_sync_20261028/replay-nglegf_d/receipt.json`.
- Isolated comparison source/build: `build_relate14_no_dw7_20261028/receipt.json` and `build.log`.
- Candidate 25 tests: `build_relate14_dw7_20261028/Testing/Temporary/LastTest.log`; CUDA/Vulkan route receipts are in independent run directories under `Testing/dw7-*-routes/`.
- Three-backend oracle/three-round A/B (including original FAIL): `build_relate14_sync_20261028/operator-ab-x43bkkgq/receipt.json`.
- CUDA five-round remeasurement: `build_relate14_sync_20261028/operator-ab-h3qon14k/receipt.json`.
- Valid public consumers: `build_relate14_sync_20261028/relation-consumer-jflnun3p/receipt.json`.
- Two-model three-round subset remeasurement: `build_relate14_sync_20261028/relation-consumer-avojaaoq/receipt.json`.
- Retained invalid RPATH diagnostic: `build_relate14_sync_20261028/relation-consumer-clyp_nqp/receipt.json`.

## 2026-10-25 historical design (not current acceptance policy)

The original text below retains the historical epilogue/direct-bias design only. Its old whitelist, queued status and accuracy inference cannot replace the measurements above or current original gates.

Status on 2026-10-25: **design frozen, implementation queued**. This document
is the executable plan for syncing the RelateAnything-GGML upstream commits
(`0ca96f3` "G2 gate closure", HEAD `49207da`) into this repository.

## 1. What upstream changed

- `cpp_ggml/third_party/patches/0001-relateanything-ggml.patch` (+752 lines):
  a new **elementwise epilogue mechanism** — `ggml_elementwise_set_epilogue_hint(t,
  GGML_EPILOGUE_GELU)` marks an ADD so the backend applies `gelu_erf` at the
  add's write-out and skips the standalone gelu node; `ggml_vk_op_f32` gains a
  `dst_out` parameter for the fused write-out; plus acceptance-probe tooling
  and a cuda-f32/vulkan deviation whitelist.
- `cpp_ggml/src/full_model.cpp` (+31 lines): ConvNeXt blocks now (a) pass the
  depthwise bias to the convolution itself (`ggml_conv_2d_dw_direct_bias`)
  instead of a broadcast add, and (b) flag the pwconv1 bias-add with the GELU
  epilogue hint.

## 2. Why it maps onto our existing track

Our 0.27 chain already has the direct-bias precedent
(`ggml_conv_2d_direct_bias_silu` in `0001-acv-cpu-operators.patch` /
`0005-acv-vulkan-operators.patch`), and the F16 Vulkan OCR detector needs a
bias epilogue of the same family. The RelateAnything sync and the OCR F16
work therefore share ONE patch: `0013-acv-elementwise-epilogue.patch`
(working title) providing

1. `ggml_elementwise_set_epilogue_hint()` + `GGML_EPILOGUE_GELU` in ggml.h,
2. CPU / CUDA / Vulkan execution of the hinted epilogue (Vulkan: apply gelu_erf
   at the add's write-out via `dst_out`; CPU: fused elementwise pass; CUDA:
   same, or fall back to separate nodes under the graphs compatibility check),
3. `ggml_conv_2d_dw_direct_bias()` in the direct-conv family (all backends),
followed by the relation `full_model.cpp` host-side port (semantic diff
preserved at `/tmp/relate-sem.diff`, 807 normalized lines; the semantic core is
the two epilogue hints and the dw-bias call).

## 3. Verification gates

- relation exact-file scenarios re-run on CUDA and Vulkan: outputs must stay
  within the upstream G2 deviation whitelist; the checkpoint-accuracy row
  (96.68%) is unaffected by construction (same graph math, fused scheduling).
- OCR detector: the same epilogue patch enables the det bias fusion —
  the F16 Vulkan speed gate (11.49 vs 6.9 ms) is the acceptance metric.
- Full contract tests (17) + selected matrices re-run.

## 4. Sequencing

The patch is a prerequisite for both R3c (F16 Vulkan det gate) and R1
(RelateAnything sync); implement it once, verify both consumers.
