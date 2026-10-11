# qRAM — Recognize Anything, RAM++, and Tag2Text

qRAM runs the upstream Recognize Anything GGUF models through the shared
`libAICore` runtime. The image workflow returns all class logits, sigmoid
scores, model thresholds, selected tags, and (for Tag2Text) generated captions.
It supports image batches, decoded database images, camera/video streams,
device selection, and JSON/CSV/text export.

## Architecture

```
qRAM dialog / worker ──► aicore_tagging_* C ABI ──► ggml RAM model graph
         │                         │
         ├── shared test-data repo  ├── image preprocessing / tag scores
         ├── shared model downloader└── CUDA / Vulkan / CPU / Metal
         └── video_base (consumer-driven frames)
```

The C++ graph, tokenizer, and caption search are adapted from the pinned
`recognize-anything-ggml/cpp_ggml` upstream implementation. Only the C ABI is
public to plugins. Published weights are SHA-256 checked and stored in the
shared `ram_models` cache.

## Build

```bash
cmake -B build_app -DBUILD_GUI=ON -DAICore_ENABLED=ON \
  -DPLUGIN_STANDARD_QRAM=ON
cmake --build build_app --target QRAM_PLUGIN ACloudViewer -j4
```

The plugin is listed as **Recognize Anything (RAM++)** and its icon carries
the RAM++ label. Open it from **Plugins → Recognize Anything (RAM++)**.

## Use

1. Choose RAM, RAM++, or Tag2Text and one of its F32, F16, or Q8_0 GGUF files.
   The default is Tag2Text F16, which generates a sentence; RAM and RAM++
   return tags. **Download model** verifies the catalog SHA-256.
2. Select Auto, CUDA, Vulkan, Metal, or CPU when that backend is present.
   Auto resolves in CUDA → Vulkan → CPU order.
3. Click **Use test data** to fetch the shared upstream `ram_datasets.zip`
   cache. The input list is filled with official `demo/demo1.jpg` through
   `demo/demo4.jpg`, with demo1 selected first.
4. Add files, a directory, or selected database images and click **Run**.
   Batch progress, timings, searchable tag scores, and selected tags appear in
   the Results tab. Tag2Text additionally supports captions and specified
   caption tags; beam and caption-length options apply only to Tag2Text.
5. The final Video / camera tab processes frames as the consumer advances.
   **Load sample video** selects the verified `demo/ram_demo1_to_4.avi` member
   from the official RAM archive. Results can be exported as JSON, CSV, or
   text, or added to the database as a labeled image.

The Options tab accepts upstream-compatible custom label lists and thresholds
(text), and label embeddings (headerless float32 `.f32`, with the upstream
class/description dimensions). With no overrides, the model's embedded class list and
per-class thresholds are used. A single global threshold can replace those
thresholds for an inference request.
Tag2Text additionally exposes the upstream threshold margin (`threshold-epsilon`)
for both tag selection and caption conditioning; the default is zero. Exported
tag rows retain the effective selection threshold.

**CPU precision** defaults to Auto. For RAM Q8 it retains floating activations
in the first two high-resolution Swin QKV projections, caching approximately
4 MiB of dequantized weights once per session. This avoids additional CPU
activation quantization at the entrance to attention. The remaining graph,
published weights and selection thresholds are unchanged. **Upstream arithmetic**
selects the original CPU activation-quantized operations for reproducibility.
Other model families, floating models and GPU backends keep their upstream
arithmetic in both modes. Changing precision recreates the worker session.

Models and samples share `CLOUDVIEWER_DATA_ROOT`, defaulting to
`~/cloudViewer_data`; archives remain in `download/` and consumed assets in
`extract/`. A missing or corrupt demo is repaired from a verified cached ZIP
before requesting a download.

## Published models

The combo box and `aicore_tagging_model_at()` are backed by the complete
Asher-1/RAM_GGUF catalog. URLs, exact sizes, and SHA-256 values live in the
AICore catalog; model bytes are cached below `ram_models/`.

| Family | Quantizations |
|---|---|
| RAM++ / Swin-L | F32, F16, Q8_0 |
| RAM / Swin-L | F32, F16, Q8_0 |
| Tag2Text / Swin-B | F32, F16, Q8_0 |

## Validation

The qRAM integration is complete. Run the following reproducible checks when
changing the tagging task, plugin workflow, models, or supported backends.

```bash
cmake --build build_app --target test_tagging_capi_contract test_tagging_probe -j4
ctest --test-dir build_app -R test_tagging_capi_contract --output-on-failure
python3 core/AICore/scripts/validate_all.py --build build_app \
  --backend cuda --tasks tagging --output build_app/Testing/ram-cuda.json
AICORE_TEST_RAM_UPSTREAM=/path/to/recognize-anything-ggml \
python3 core/AICore/scripts/validate_all.py --build build_app \
  --backend cuda --tasks tagging --full --output build_app/Testing/ram-cuda-full.json
QT_QPA_PLATFORM=offscreen build_app/bin/plugin_test/test_ram_workflow --real --backend cuda \
  --output-dir build_app/Testing/ram-ui
```

The default tier probes all three Q8_0 families on demo1. `--full` expands to
21 rows: nine demo1 numeric checks, nine complete 50-frame dataset checks, and
three 50-frame caption checks. The demo probe compares every class logit and
sigmoid score with pinned official PyTorch outputs using each GGUF's own
dequantized weights. CPU Q8 additionally checks all raw logits of the explicit
compatibility mode against the pinned upstream CPU activation-quantized graph.
Auto RAM Q8 independently checks all-class probabilities against the official
PyTorch graph with the same dequantized GGUF and CPU activation arithmetic;
weight-only PyTorch probability gates remain mandatory. References are generated under the system temporary
directory (`/tmp/aicore-tagging-reference-cache` on Linux) from the pinned upstream revision;
no generated truth files are stored in the source tree.
The dataset probe uses original official checkpoints,
scores all four upstream label spaces with mAP/CP/CR and micro/macro metrics,
and compares original GGML with an ABBA execution order, matching model,
backend, threads, warmups, repeats, and input bytes. Caption rows compare
exact sentences and word F1 with both references. Each scenario preserves
its raw outputs independently; p50/p95 and repeatability are recorded.
Tag2Text's rare/ImageNet/HICO label spaces have no supplied Tag2Text ground
truth. Their zero-target mAP/CP/CR values are diagnostics, matching upstream;
the report identifies them through `annotation_role` and `annotation_targets`.
Tag set and caption agreement with both engines remains mandatory on those
images. Caption agreement measures decoder parity rather than human-caption
accuracy. Measured throughput and all-class upstream logit errors accompany
the dataset latency and probability metrics.
Dataset p50/p95 describe the distribution of per-image repeat means across
the two processes; the demo probe additionally measures individual forwards.
Overall latency means weight each dataset by its image count, and overall
percentiles use all retained per-image samples. Score summaries and the
5-percent latency gate use the worst dataset; per-label-space rows remain
available in each report.
Quantized CPU arithmetic can change tags near their thresholds relative to
the original PyTorch checkpoint. The compatibility gate checks raw upstream logits
(maximum error `1e-4`), probabilities (`1e-5`), and selected tags, and also
requires the default mode's independent original-checkpoint F1 at least `0.99`,
probability MAE at most `0.002` and mAP/CP/CR deltas at most `0.01`.
Compatibility mode's per-class probability errors, annotation errors and F1
must not worsen beyond the same GGUF's upstream error (`1e-6` tolerance).
Auto mode is also checked against its independent PyTorch activation-arithmetic
reference (probability maximum error `0.02`, MAE `0.002`). The report retains
Auto mode's original upstream differences, including individual worsened errors;
improved aggregate accuracy does not imply every class improved.
The report retains both engines' errors against PyTorch; quantization error
does not excuse added integration error.
On NVIDIA GPUs the dataset probe reuses the upstream GPU overlap audit and
reserves a visible idle CUDA context between samples. Concurrent compute
processes cause the sample to be retained as an `.overlap-*.log` and retried;
persistent interference fails the row. An explicitly verified idle,
long-lived context can be permitted with the test-only
`AICORE_TEST_RAM_ALLOWED_GPU_PIDS` comma-separated PID list. Active workloads
must never be put in that list.
`AICORE_TEST_RAM_WAIT_PROCESS_PIDS` can name another benchmark coordinator
whose completion must precede sampling, including gaps between its GPU jobs.
The audit also samples graphics processes through `nvidia-smi pmon`; desktop
work above 5% SM utilization invalidates the sample even when the compute-only
process list is empty or that PID is permitted as an idle context. All timing
samples must pass this audit.
Waiting for GPU idle is bounded to 60 seconds per reservation/sample by default
(`--gpu-idle-timeout`); a busy GPU fails the row with the retained activity audit.

The full probe requires the pinned upstream checkout with its official `.pth`
files in `cpp_ggml/models/pytorch`, prepared `cpp_ggml/benchmarks/label_spaces`,
and `cpp_ggml/build-<backend>/ram-ggml`. These references are test-only and do
not enter the production runtime. Follow the upstream benchmark instructions
to prepare them. Full validation fails if a reference or scored label space
is missing. The common runner verifies/downloads all GGUFs and every pinned
consumed archive member. Use `--profile release` with a same-host passing
baseline for release evidence, as specified in the AICore engineering skill.
Official PyTorch references default to CUDA. On a CPU-only host, set the
test-only `AICORE_TEST_RAM_REFERENCE_DEVICE=cpu` (or `--reference-device cpu`
for the dataset probe). CPU and CUDA references use separate cache keys, and
reports identify the reference device.
The test adapter binds the upstream batch helper's device to the evaluator's
explicit device option. This prevents its import-time CUDA default from moving
Tag2Text inputs away from a CPU model; the official model, image transform,
forward operations, and score selection remain unchanged.
For a comparison with matching CPU compiler flags, keep the original upstream
build and create a separate upstream build:

```bash
python3 core/AICore/tests/tagging/prepare_cpu_upstream.py \
  --upstream /path/to/recognize-anything-ggml --candidate-build build_app --jobs 4
```

The helper clones the pinned source and ggml commits into a unique temporary
directory, copies the candidate CPU compiler settings, and retains the exact
loader adapter patch and build provenance. It changes CPU module discovery and
thread setup; graph operations, preprocessing, and tag selection stay upstream.
Set the printed test-only
`AICORE_TEST_RAM_UPSTREAM_BUILD` to it (or pass `--upstream-build` directly to
the dataset probe). The chosen binary, library root, compiled source path,
source hash, and compiler/backend settings are recorded.
An upstream `-march=native` build and AICore's portable Haswell backend are
different compiler configurations and must be identified as such in results.

The real Qt workflow test requires the cached RAM++ Q8_0 model and official
sample; missing assets fail. It checks batch session reuse, all-class/search
views, JSON/CSV/text exports, the database boundary, close/cancel/reopen, and
offline ZIP extraction/repair. Add `--video <fixture.avi>` to test consumer
advancement, stop and restart. The video check also restarts an in-flight job
and verifies that the new stream
holds its first frame until cancellation completes, without draining frames.
Backend switching can be exercised by rerunning
with `--backend vulkan` or `--backend cpu`, or by adding `--switch-backend vulkan`
to switch an already loaded CUDA session. The default CTest test exercises
controls and failure/cancellation completion without loading a model.
Add `--caption` to exercise the real Tag2Text Q8 caption display/export path.
For a first-download test, provide an empty `CLOUDVIEWER_DATA_ROOT` and pass
`--download-sample` without `--real`; the test fails if demo1 was already cached.
Add `--plugin-module <libQRAM_PLUGIN>` to verify the built module, action and icon
through `QPluginLoader` (the name avoids Qt's reserved `--plugin` argument).
Use `--timeout-ms 600000` when a real workflow includes a first GGUF download.
Add `--upstream-binary <build-<backend>/ram-ggml> --label-space-dir
<cpp_ggml/benchmarks/label_spaces>` to drive the Options tab end to end: a real
label list / thresholds / embedding file, a global threshold, Tag2Text batch
beams, and specified tags with sequential beams, shorter length and a threshold
margin. Each case exports the UI result and runs the original `ram-ggml` with
the same model, image and arguments. CPU all-class logits must agree within
`--upstream-logit-tolerance` (default `0.0001`). GPU has no separate upstream-
arithmetic compatibility mode, so its finite raw logit difference is recorded
but is not presented as parity; independent PyTorch and annotation gates own GPU
correctness. On every backend selected tags and captions must be identical, and
the exported effective thresholds must come from the chosen file / value /
margin. The evidence JSON records whether logit parity was required, plus the
image, upstream binary, `libAICore`, plugin and label-space digests.

See [models/MODEL_CARD.md](models/MODEL_CARD.md) for model and sample provenance.
