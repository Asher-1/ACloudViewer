# qOCR — PP-OCR Text Recognition

PP-OCRv4 text detection and recognition over the AICore `ocr` task, built on
the first-party Paddle inference → GGUF conversion
(`core/AICore/tools/convert_paddle_ocr_gguf.py`).

## Workspace

The **OCR Workspace** action opens the plugin dialog (a pure view in the
shared AICore plugin shape, same split as qRMBG):

- **Recognition language / Quantization** — picks the det/rec GGUF pair from
  the published catalog (det is shared and script-agnostic).
- **Device / Threads** — the shared runtime row. Device entries come from
  `aicore_device_count()` ("auto" follows the platform order: CUDA → Vulkan →
  CPU on Linux/Windows, Metal → CPU on macOS); the request is passed through
  `aicore_ocr_options.device` and resolved by the AICore runtime lease
  registry. GPU-unavailable requests fall back to CPU with a console note.
- **Try sample data** — one-click official PaddleOCR demo image through the
  shared `ecvTestDataRepository` (Dataset::Ppocr).
- **Download missing models** — the unified `ecvModelDownloader` pipeline with
  pinned SHA-256 verification and the shared progress bar.
- **Run OCR / Cancel** — the run executes on an `OCRWorker` thread: the
  device task queue is acquired first (serialized against every other AICore
  task), the context is created on the worker thread and released on the main
  thread, and cooperative cancellation is bound for the run. Progress is
  stage-based with the busy (indeterminate) bar — the aligned fallback for
  backends without per-node progress — plus a 10 s heartbeat in the console.
- **Add result to DB** — exports an annotated image (boxes + transcriptions
  drawn over the source) to the DB tree with metadata (transcription list,
  det/rec models, device, stage timings, source path); named through
  `ecvPluginDbNaming` so repeated runs stay separable.

All log output goes through the shared application console
(`aicore_inference_log` → `CVLog`); the plugin embeds no private log widget.

## Models

Published at the [PP-OCR release](https://github.com/Asher-1/cloudViewer_downloads/releases/tag/PP-OCR)
and pinned in `core/AICore/include/aicore/asset_digests.h`; every entry is
exposed by `test_catalog_dump_urls --json` for the shared download/cache
pipeline. Recognition dictionaries are embedded in the rec GGUFs
(`ocr.dictionary`), so a model file is self-contained.

## 2026-10 端到端基线（当前权威记录）

完整、可复现的矩阵、可视化、差异分类和已否决的优化实验见
[`core/AICore/src/tasks/ocr/README.md`](../../../../core/AICore/src/tasks/ocr/README.md)。下面的历史性能叙述仅保留排障背景；若与本节冲突，以本节和任务 README 为准。

同一 RTX 4090、同一 PNG、PaddleOCR 3.0.1 / paddlepaddle-gpu 3.0.0、3 warmups +
10 timed runs；范围覆盖 det `{f32,f16,q8_0}` × rec `{f32,f16,q8_0}` 的九种组合，CUDA 与
Vulkan 共 54 个端到端结果，全部低于 Paddle GPU p50：

| 图像 | Paddle GPU p50 | CUDA p50 | Vulkan p50 |
|---|---:|---:|---:|
| ppocrv4，2566×1312 | 175.40 ms | 62.43–65.35 ms | 71.83–75.77 ms |
| medal_table | 99.87 ms | 34.19–35.07 ms | 43.59–45.47 ms |
| layout_doc_test | 125.96 ms | 53.44–55.02 ms | 68.28–92.35 ms |

Paddle 字符串逐字相同率不是无标签图像的准确率：F32 图级数值/argmax 门验证的是
转换和执行，而端到端差异还包括 box、crop 和两个 OCR pipeline 的 CTC 决策；layout
样本中 Paddle 本身会漏句尾标点、错误切词，AICore 也存在真实错字。因此不得将这些
相同率作为“与 Paddle 一致”或语义准确率发布宣称；需要逐框标签后才能计算 CER/WER。

带官方 GT 的发布矩阵由
[`run_ocr_gt_matrix.py`](../../../../core/AICore/scripts/run_ocr_gt_matrix.py) 和
[`ocr_gt_eval.py`](../../../../core/AICore/scripts/ocr_gt_eval.py) 执行：它要求固定
GT SHA-256、数据集 revision，并覆盖中文/英文 rec 的 3×3 quantization × CUDA/Vulkan。
ICDAR2015 的官方下载需要注册；在已许可的 GT 包落入测试资产缓存前，该门不会伪装成
“已通过”。Vulkan 性能运行可要求 `performance` governor，以记录/拒绝宿主状态而非由
插件修改系统功耗策略。

## Runtime notes
- **CUDA conv-kernel study (CORRECTED adjudication, <100 ms reached)** —
  the previous session's "direct conv breaks transcription parity" verdict
  was an artifact: every 67-88/98 measurement was taken while the
  experimental "balanced" 48/50 chunking was still in the tree, and that
  chunking alone (two cached builds with different N) made the decodes
  non-deterministic and wrong. Per-pixel det-map comparison shows the
  direct conv (naive AND TF32-WMMA igemm) produces a det map bitwise
  identical to im2col+GEMM on the production geometry. Subsequently the
  REAL root cause of every historical flaky measurement was found and
  fixed: `aicore_ocr_result::textStorage` was a `std::vector<std::string>`
  whose push_back reallocations dangled every previously handed-out
  `box.text` c_str pointer (SSO bodies move with the vector); it is now a
  `std::deque`, which never invalidates element pointers. With that fix
  the threaded per-row CTC decode (previously "racing", 85-95/98) and the
  TF32 combination (98<->97 single-box flap) are BOTH deterministic, and
  the mixed-N cached builds are bit-stable too. Final measured stack
  (`AICORE_OCR_CONV_DIRECT=1` + `GGML_CUDA_IGEMM_F32_TF32=1`): det forward
  60 -> 14.6 ms, rec chunk 68 -> ~15 ms (dw via the direct depthwise op +
  the igemm plan-shape fix), parallel CTC, and a fold-pool device mirror
  (`OcrWeights::realizeFoldPoolToDevice`) that uploads each build's folded
  constants ONCE instead of letting the sched re-copy ~11 MB per compute
  (measured ~12 ms per rec chunk, more than all rec kernels combined) —
  **e2e 98.4-99.4 ms** (p95 ~102, idle GPU; 100.1 ms at 34% util) vs the
  im2col+GEMM default at ~245 ms, with 98/98 valid decodes whose text set
  is bit-identical to the long-standing baseline hash. The igemm plan
  cache was also hardened: plans are validated by the full
  (input, dst, kernel) geometry instead of byte sizes, and the
  `op_params[8..9]` owner_id read (a constant 0 for constructors that
  never write it) was dropped from the reuse/chained decisions.
  `runGraphCached` now takes a pointer+count so the chunk upload reads
  each chunk's contiguous row range directly out of one pad-complete batch
  buffer (unique slot per box; no per-chunk staging copy).
- **Vulkan single-backend route (`AICORE_OCR_VK_SINGLE=1`, this session)** —
  the VK session's default is still the CPU fallback below, but an
  opt-in single-backend route now runs the WHOLE graph on the Vulkan
  device through a plain gallocr (no sched, no CPU secondary, no
  mixed-backend buffers), which sidesteps the sched subbuffer fault
  entirely instead of waiting on a pin-side fix. Three VK-specific
  hazards were diagnosed and fixed on the way: (1) the fold-pool device
  mirror needed per-tensor ALIGNED slots (the raw pool copy kept ggml's
  16-byte alignment, below minStorageBufferOffsetAlignment, and the
  misaligned weight bindings faulted the VK conv2d dispatch);
  (2) views created during the build captured the fold tensors' HOST
  data pointers and had to be re-derived after the mirror (the sched
  path re-inits views at sched_alloc_graph, the plain gallocr does not);
  (3) the graph's named weight leafs are bound to the HOST originals
  (BN folding needs host dereference) and must be repointed at their
  device copies (`OcrWeights::retargetGraphToDevice`, a full src-chain
  walk — the weight leafs live in the meta pool, not the graph ctx).
  Measured (medal_table, idle GPU): **e2e 183-186 ms** — a 17x cut from
  the CPU fallback's 3118 ms — with 98/98 valid decodes, run-to-run
  deterministic, and text IDENTICAL to the CUDA route on all 89 shared
  box coordinates. The conv router keeps VK det graphs on im2col+GEMM
  (the pin's direct conv_2d dispatch segfaults in the NVIDIA driver at
  the 736x1184 production geometry; rec's 320x48 shapes run clean on the
  direct route).

- **Vulkan follow-up (latest session): 120 ms e2e** — three further
  findings cut the single-backend route from 295 ms to **119.8-120.9 ms
  (inf 54.3: det 10.5 + rec 44; post 57)**:
  1. **Batched rec un-blocked for VK.** The batch gate excluded Vulkan by
     name with a stale justification ("a VK batch would execute on the
     CPU secondary anyway") that predates the single-backend route; the
     expiry condition fired and VK sessions now run the chunked batch.
     The chunk cap is per backend: CUDA keeps 48 (im2col VRAM cliff),
     Vulkan keeps 112 — the direct-conv route has no im2col
     materialization, so its workspace scales linearly with N and one
     chunk covers a dense image (3 computes -> 1; the per-chunk fixed
     cost is amortized away).
  2. **NV_coopmat2 is a 3x NEGATIVE optimization for this workload** —
     `GGML_VK_DISABLE_COOPMAT2=1` drops the rec 104-row chunk from
     181.5 ms to 60.3 ms of pure GPU time AND makes the outputs an order
     of magnitude closer to the numpy reference (max_diff 0.00354 ->
     0.000384). nsys shows host-side submit cost is negligible
     (~1630 vkCmdBindPipeline per frame, 0.11 ms total; GPU time is all
     in vkWaitForFences), so the gap is kernel quality, not dispatch
     count — the coopmat2 f32 conv/matmul shaders lose to the cm1/naive
     paths on this driver. llama.cpp tracks similar Vulkan/coopmat2
     regressions upstream. The env is pin-official; making the pin
     default cm2-off needs a cross-task impact sweep first, so it stays
     an explicit opt-out in the deployment notes for now.
  3. **The probe's backend selector was mislabeled** — init_by_name
     matches device names ("Vulkan0") exactly, so every "backend=Vulkan
     device=CUDA0" probe row ever printed measured CUDA. The probe now
     resolves through `ggml_common::find_gpu_backend` (with an explicit
     CPU bypass); true-VK numbers above supersede the old probe rows.
  The recognize-phase chunk upload also had a latent out-of-bounds read
  (global-crop offset over a single-chunk buffer from the second chunk
  on) — fixed alongside the per-backend cap. VK vs CUDA text is
  unchanged-in-content: every box differs by at most 1 px of quad
  geometry (threshold-edge connected-component jitter), no text drift.
  Remaining gap to CUDA's 88 ms is the rec attention/linear section
  (~25 ms over the conv share) and the post stage (~57 ms CTC+crop).

- **Post-stage measurement (current instrumentation)** —
  `aicore_ocr_post_timings()` now reports DB boxes, crop/resize/pack, CTC
  decode, result assembly, and graph input/output transfers without changing
  the common pipeline-timing ABI. On the 98-box `medal_table` F32 run
  (RTX 4090, Vulkan single-backend, coopmat2 disabled, 2 warmups/5 samples),
  p50 was 114.81 ms e2e / 48.12 ms post: DB 2.64 ms, crop 3.31 ms, CTC
  2.82 ms, output readback 10.30 ms, input upload 2.71 ms. These stages do
  not yet sum to the common post wall time; the remainder is explicitly left
  unclaimed pending a graph-call orchestration breakdown. The same protocol
  measured CUDA at 91.00 ms e2e / 36.24 ms post (F32 direct+TF32 opt-ins),
  with 98 boxes in both runs. This is a diagnostic baseline, not a Paddle
  comparison or a justification to change global coopmat2 defaults.

- **Vulkan session routing (this session)** — with the sched_gpu_tagging
  gate closed, a VK session runs every node on the CPU secondary, so its
  graphs are built with the CPU conv paths (winograd/island/batched GEMM)
  and computed with plain `graph_compute` on the CPU lease: 2.49 s vs the
  2.46 s plain-CPU session (the old mixed sched allocation crashed with
  SIGBUS because activations landed in VK memory the CPU compute could
  not read, and the sched itself added ~1.7 s of split overhead to a
  single-backend graph). VK GPU routing stays blocked on the subbuffer
  fault diagnosed earlier.
- **Fixed-48 chunking is load-bearing** — a "balanced" 48/50 chunking
  experiment (two cached builds with different N) made the end-to-end
  decodes non-deterministic (94/92/94 valid across identical runs) while
  every fixture gate stayed green; the fixed cap keeps ONE cached build
  per geometry and the decodes are bit-stable. Reverted to the fixed cap.

- **Official-alignment status (measured against paddlepaddle-gpu 3.0.0 +
  paddleocr 3.0.1 on the same host/image, medal_table.png)** — CLOSED for
  practical purposes: with the official det preprocessing ("min" 736/4000
  round-half-even 32), the official-semantics DB post-process (8-connected
  components, hull min-area rect, box_score_fast over the raw rect region,
  min_size 3, Clipper-equivalent unclip 1.5), the official rec crop
  pipeline (image-space quads, black... rather NORMALIZED-ZERO rec pad per
  paddlex resize_norm_img, BGR order, row-interleaved fixed-320 packing),
  the end-to-end text comparison is **98/98 valid decodes, 77/98 (79%)
  string-identical to the official** with a near-1:1 box set (98 vs 93).
  Most residual diffs are punctuation-width variants (（USA） vs (USA))
  or cases where OUR decode is the sensible one (the rank column 3..14
  decodes correctly here while the official rec returns garbage on those
  tiny crops). The historical 2/86 "common text" disaster was rooted in
  TWO compounding bugs, both fixed: (1) the rec input packer appended the
  width-pad AFTER a (3,h,recW) pack, so any recW<320 crop re-read as
  (3,48,320) interleaved channels/rows into garbage — the pack now writes
  the (3,48,320) layout directly with the pad inside every row; (2) an
  earlier "black pad" misread of resize_norm_img (np.zeros is in the
  NORMALIZED domain = gray 127.5) had put an out-of-distribution -1 fill
  into every short crop. The fixture probes could not see either bug (the
  numpy replay mirrors this runtime, a self-consistency gate); the
  paddle-truth per-op bisect was prepared (PIR backward-chain fetch) but
  became unnecessary once the paddle-cross-check on our own intermediates
  isolated the input pipeline.
- **Same-image speed** (98-box frame, shared GPU): CUDA e2e ~238 ms
  (pre 8 + det+rec inference ~170 at the 4.4x-larger official resolution +
  post ~56) vs the official pipeline's 71.8 ms; CPU ~2447 ms (the big det
  is FLOP-bound on CPU). The remaining GPU gap is structural: ggml runs
  every conv as im2col+F32 GEMM with per-op dispatch, while paddle uses
  cuDNN-style direct convs; the convergence path is a direct-conv CUDA
  kernel (GGML_OP_CONV_2D_DIRECT is declared but unimplemented in this
  pin) or per-op fusion, tracked separately.


- **Text decoding** — the CTC decoder handles both published container
  conventions: vocab = dict+2 (raw `ppocr_keys_v1` vocabulary, trailing space
  class appended at export) and vocab = dict+1 (the table already carries its
  own trailing space entry). The trailing space class is real recognized
  text and is decoded, not skipped.
- **Accuracy fixtures** — the probe fixtures (`ocr_fixtures/*.npy`, rec ch
  + en / det / layout) are regenerated deterministically by
  `core/AICore/scripts/gen_ocr_fixtures.py`, a numpy replay of the audited
  Paddle graph read straight from the F32 GGUF; the historical release .npy
  assets are gone. Measured stock-CPU parity: rec-ch max|diff| 6.5e-3
  (isolated post-softmax near-ties, argmax sequence exact), rec-en 7.8e-4,
  det 3.6e-3, layout 1.2e-2 (raw GFL logits); Vulkan parity is bit-identical
  to CPU stock on all three graphs.
- **Export visualization** — the annotated image follows the official
  PaddleOCR `draw_ocr` style: red 2-px closed polygon outlines (boxes below
  `drop_score` 0.5 are skipped), the image resized to a 600-px long side,
  and a white right-hand text panel ("index: text score", CJK-capable font,
  wrapping and horizontal panel continuation exactly as `text_visual`).
- **CPU blocked island** — on AVX-512 hosts all three graphs (rec/det/
  layout) route their conv/affine/hard_swish/SE chains through the shared
  blocked-layout island (`core/AICore/src/common/blocked_conv.{hpp,cpp}`);
  fixture max|diff| rec 3e-4 / det 3.6e-3 / layout 6.3e-3 (stock: 6.5e-3 /
  3.6e-3 / 1.2e-2). The island consumes each op's OWN input var, so the
  branched FPN topology is handled correctly.
- **Stage timings** — `aicore_ocr_run` reports the common timing struct with
  `valid_fields` set: inference covers det + rec + layout graph compute,
  postprocess excludes it, so the dialog's ms numbers are truthful.
- **GPU dispatch-root-cause fix (WEIGHTS buffer tagging)** — every task
  loader that allocated weights with `ggml_backend_alloc_ctx_tensors*()`
  never called `ggml_backend_buffer_set_usage(..., WEIGHTS)`. The sched's
  pass-1 weight-following / offload routing requires that usage tag, so
  with an "active" CUDA backend every node was silently routed to the CPU
  secondary (nsys proof: zero CUDA kernels, only host copies) and the GPU
  numbers were CPU numbers plus copy overhead. Tagging the buffers
  (OCR, facedetect, yolo, rmbg, gaussian, lightglue, sam3, map_anything,
  depth, trellis, gkd, lingbot, loma, rfdetr — sam3d's loader already had
  it) plus the OCR folded-constant pools routes weight-consuming ops to
  the GPU. Measured single-forward on the idle GPU: rec 40.5 -> 5.3 ms,
  det 204 -> 24 ms, layout 233 -> 18 ms (5-13x). CUDA graphs are also now
  compiled in (`GGML_CUDA_GRAPHS=ON` with the custom-op-exclusion patch
  `patches/cuda_graphs/0001`, which keeps sam3d host-orchestrated ops
  out of capture). Dense-image e2e on an idle GPU: ~2.7 s -> ~0.18 s (batched rec);
  numbers measured while another GPU process saturates the card are
  contention-dominated and much worse.
  Vulkan keeps the legacy all-CPU routing for now. The vk subbuffer fault
  is fully diagnosed: with tagging on, the sched's offload threshold
  (batch < 32) parks the small SE-block ADD on the CPU secondary, the
  downstream CLAMP has no weights so the sched picks Vulkan, and its
  view input inherits the CPU buffer through gallocr's
  `ggml_backend_view_init` — `ggml_vk_tensor_subbuffer` then treats the
  host pool as device memory and faults (SIGBUS; nsys/backtrace
  verified). `GGML_OP_OFFLOAD_MIN_BATCH=1` removes the CPU island and
  the run completes cleanly (1.20 s, 3x over the all-CPU VK path), which
  confirms the mechanism; a durable fix (per-backend min-batch default,
  or dropping the host-buffered view at the AICore build site) is
  tracked separately before the tagging gate opens for VK.
- **Benchmark vs official** — same-host A/B against the official PaddleOCR
  pipeline (tools in `core/AICore/tools/bench_ocr.cpp` +
  `core/AICore/scripts/bench_paddle_official.py`, protocol 3 warmups + 10
  timed end-to-end runs, PP-OCRv4 mobile, dense 74-box image):

  | pipeline | device | e2e p50 |
  |---|---|---|
  | official PaddleOCR | GPU (CUDA) | 112 ms |
  | official PaddleOCR | CPU (paddle, no mkldnn) | 6461 ms |
  | AICore ocr | CPU | 3117 ms (**2.1x faster than official CPU**) |
  | AICore ocr | CUDA (official min/736 det) | ~238-292 ms |
  | AICore ocr | CUDA (GPU shared with a saturating process) | ~3.0 s |

  **Controlled re-measurement (supersedes the table for a current winner
  claim):** on the same RTX 4090 and `medal_table.png`, with F32 models,
  3 warmups and 10 timed end-to-end calls, official PaddleOCR GPU measured
  **68.63 ms p50 / 74.63 ms p95** (93 text rows). AICore CUDA with the
  direct+TF32 candidate measured **105.72 ms p50 / 115.43 ms p95** (98
  boxes); AICore Vulkan single-backend/direct-conv with coopmat2 disabled
  measured **107.87 ms p50 / 111.11 ms p95** (98 boxes). Neither AICore
  backend is presently faster than Paddle under this controlled timing
  protocol. The 93-vs-98 output count also means this is not yet a text and
  spatial parity proof; no "faster than Paddle" release claim is permitted
  until a per-box semantic gate and a repeated same-protocol A/B pass.

  Remaining CUDA gap vs official: (1) the rec per-crop loop still issues
  74 sequential graph computes (batching blocked on the 4-D attention
  re-encode below), (2) box postprocess is single-threaded (~110 ms),
  (3) det/activation node count could shrink via graph fusion. Our
  F16/Q8_0 GGUF source tensors stay in the host-readable GGUF metadata, but
  `OcrWeights::load()` materializes F32 compute tensors because the current
  graph uses F32-only binary/im2col/mul_mat operands. Therefore these files
  are not native low-precision inference and must not be claimed as a
  quantized speedup. A native F16 dispatch requires an official text/space
  accuracy gate and a same-protocol p50 improvement before it can replace
  F32. Batched rec is the next leverage point, but it is BLOCKED on a
  structural constraint: the rec
  graph is batch-safe in Paddle semantics (numpy replay: N=2 output
  halves match N=1 within 1.7e-6), yet the SVTR attention head reshape
  (N,T,3,heads,dim) is rank-5 while ggml tensors are 4-D — N=1 folds the
  size-1 batch axis away, N>1 cannot (mul_mat aborts at build). Batched
  rec requires re-encoding the attention head in 4D first (e.g. fold
  (3,heads) -> one axis with per-(n,head) slice views, or a ggml 5-D
  patch).
- **Batched rec (automatic on CUDA sessions)** — the 4-D attention
  re-encode in buildGraph: each of the rec graph's TWO serial attention
  blocks bypasses its rank-5 reshape chain with strided (d,t,h,n) views
  of the qkv projection (materialized contiguous before mul_mat — the
  CUDA kernel does not preserve strided src0/src1 element order the way
  the CPU kernel does) + batched mul_mat (the llama.cpp attention
  layout), and matmul_v2 places the weight on src0 so ggml's batch
  broadcast rules hold for N>1. Depthwise convs run as one batched GEMM
  for ANY N: the contiguous {W,H,C,N} input reshapes to {W,H,1,C*N}
  (i = c + n*C), and the per-channel filters are ggml_repeat-tiled to
  the full C*N batch — ggml's mul_mat broadcast maps src0's batch as
  i12/r2 (c slowest), which scrambles the c-fastest reshape order, so
  the tiny filter tensor is tiled instead of relying on broadcast (also
  kills the ~30k-node per-channel expansion whose non-contiguous views
  the CUDA IM2COL rejects, and the N=1 channel-0 weight view that did
  the same). Crop collection is parallel (phase A); phase B runs fixed
  48-row chunks (tail pad-filled; the cache key never varies). 48 is
  the measured linear-scaling cap: 10/30/84 ms per forward at
  N=4/16/48 on an idle RTX 4090, collapsing at N=96 (15.5 s) when the
  im2col workspace pushes the allocator against VRAM. Numeric evidence:
  per-slab bitwise self-consistency at every measured N, and the
  N=4/16/48/96 CUDA outputs match the N=1 fixture reference within the
  stock-vs-fixture noise (max diff ~6e-3, p50 ~1e-8, argmax identical).
  Dense-image e2e on the shared GPU: ~2.7 s (per-crop) → ~0.18 s
  (batched). CPU sessions keep per-crop N=1 (the island wins there);
  Vulkan sessions keep per-crop until the vk subbuffer fault that keeps
  sched_gpu_tagging off is fixed.
