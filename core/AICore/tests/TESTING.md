# AICore Model Test Matrix

Model tests use the application's native cache automatically: Linux/macOS use
`$HOME/cloudViewer_data/extract`, Windows uses
`%USERPROFILE%/cloudViewer_data/extract`, and `CLOUDVIEWER_DATA_ROOT` changes
that base exactly as it does at runtime. `AICORE_TEST_ASSET_ROOT` overrides it.
Model downloading is opt-in. After enabling `AICore_BUILD_TESTS`, pass
`-DAICORE_TEST_AUTO_DOWNLOAD=ON` only for an explicit test run; then
CTest's asset fixture downloads missing fixture models, the fixed LightGlue
image pair, and `friends_faces` data immediately before an asset-dependent
test runs. Normal configure and normal application builds never access the
network. With the option disabled, direct `ctest` runs keep the existing `77`
skip behavior when assets are absent.

Use one portable root rather than exporting a collection of individual paths:

```bash
cmake -S . -B build_app \
  -DAICore_ENABLED=ON -DAICore_BUILD_TESTS=ON \
  -DAICORE_TEST_AUTO_DOWNLOAD=ON \
  -DAICORE_TEST_ASSET_ROOT=/path/to/cloudViewer_data/extract \
  -DAICORE_TEST_DEVICE=cpu
ctest --test-dir build_app --output-on-failure -L model
```

The root contains the task model directories referenced by
`scripts/validation_manifest.json` (including depth, gaussian, feature,
detection, segmentation, background-removal, SAM3, TRELLIS, and YOLO assets)
plus their fixed test images. Individual `AICORE_TEST_*` variables still
override this convention when a specialized fixture is required.

Use `cuda` or `vulkan` only on a self-hosted Linux/Windows GPU runner, and
`metal` only on macOS. Run the strict backend checks separately:

```bash
ctest --test-dir build_app --output-on-failure -L parity
ctest --test-dir build_app --output-on-failure -L e2e
```

For a reproducible ALIKED 1024 graph profile, use the strict parity fixture;
it reports wall time for upload, backbone subgraphs, DKD, and SDDH without
changing the input resolution or acceptance thresholds (the historical
`AICORE_ALIKED_STAGE_BENCH` env gate was removed in the env cleanup; the
probe is dormant until an explicit API re-enables it):

```bash
ctest --test-dir build_app -R '^test_aliked_capi_parity$' -V
```

Performance changes are accepted only after that test keeps its keypoint and
descriptor parity gates. The `1 s` target is a self-hosted GPU gate, not a
GitHub-hosted CI assertion.

GitHub-hosted runners default `AICORE_TEST_ENABLE_MODEL_TESTS=OFF` and
`AICORE_TEST_AUTO_DOWNLOAD=OFF`; their `aicore-contract-tests` target excludes
`model`, `gpu`, and `e2e` labels. This is intentional: a missing GPU must not
be recorded as a parity pass. Cache the downloaded model bundle by a manifest
checksum on self-hosted GPU runners, enable both options, and inject its path
as `CLOUDVIEWER_AICORE_TEST_ASSETS` or `AICORE_TEST_ASSET_ROOT`. The GPU matrix
is Linux CUDA, Linux/Windows Vulkan, and macOS Metal. It is a required
protected-branch check only where the corresponding hardware label exists.

## Complete model/task regression gate

`core/AICore/scripts/validate_all.py` is the one-click gate to run after changing or
adding a ggml operation. Before expanding the task matrix, it enumerates every
supported GGUF from the built AICore catalogs, verifies the pinned SHA-256, and
downloads missing or corrupt models into the corresponding directory under
`~/cloudViewer_data/extract`. Downloads use temporary files and an atomic rename,
so an interrupted transfer never becomes a cached model. By default, a catalog
mismatch or download failure stops the gate before inference; it is never
converted to a skip. The runner then executes each model or model bundle serially and writes
JSON plus Markdown reports. A missing backend, exit 77, an uncovered new model,
an accuracy failure, or unstable repeated output also makes a complete run fail.

```bash
# Defaults: two process repeats; probes use two warmups and ten timed forwards
# where their API supports repeated inference.
python3 core/AICore/scripts/validate_all.py \
  --build build_app --backend cuda \
  --output build_app/Testing/aicore_validation.json

# Equivalent CMake entry after configuring AICORE_TEST_DEVICE.
cmake --build build_app --target aicore-validate-all -j1
```

For an optimization gate, retain the before and after build directories and
run the preferred controlled A/B form below. It alternates the old and new
probe order on every repeat, so changes in machine load are less likely to be
misclassified as an operator regression. Use the same assets and backend for
both builds.

```bash
python3 core/AICore/scripts/validate_all.py \
  --build build-after --backend cuda \
  --baseline-build build-before \
  --output build-after/Testing/aicore_validation.json
```

The default performance gate requires both a `+5%` relative increase and an
absolute increase above `3` metric units (`ms` for latency, `MiB` for memory).
Task probes own their numeric accuracy gates. Deterministic probes require an
exact cross-build output fingerprint; ALIKED, SAM3, and YOLO instead run
explicit CPU/backend numeric parity gates and use fingerprints for same-build
stability only. A previously captured report can still be supplied with
`--baseline report.json` for archival comparisons, but that mode cannot
control for load changes between the two runs.

YOLO text catalog models are separate scenarios: CLIP and M-CLIP are paired
with a YOLO-World detector, while MobileCLIP is paired with YOLOE. Their first
load/text-encoding latency and exact end-to-end output hash are both gated;
the main YOLO matrix rejects missing or incompatible role-specific text
towers instead of silently skipping those models.

Use `--tasks rmbg,yolo` for a focused development loop and `--list` to populate
and audit that subset's cache without inference. `--offline` performs the same
complete SHA-256 cache audit but fails instead of downloading. `--allow-incomplete`
is local-diagnosis only: downloads are still attempted, but unavailable models,
their dependent scenarios, and exit-77 probes are recorded and skipped while
the remaining rows continue. Its verdict is `INCOMPLETE`, and it must not be
used for a regression-free claim. Combining `--offline --allow-incomplete`
checks only the models already available in the local cache.

The published catalog is the mandatory model set. TRELLIS uses the AICore
runtime catalog as its single source of truth for the complete Hugging Face
`Asher-1/Trellis2-models` release: every f16, q8, and published f32 GGUF has a
resolver URL, exact LFS size, and SHA-256. The default gate downloads missing
TRELLIS pipeline files into `~/cloudViewer_data/extract/trellis_models`,
validates them, and runs f16, q8, and f32 pipeline scenarios. RMBG is a shared
Trellis dependency: its files are downloaded and exercised once from
`~/cloudViewer_data/extract/rmbg_models`. qTrellis consumes that same catalog;
it must not introduce a private URL, size, or digest table.

## YOLO upstream-parity benchmark

`tests/yolo/run_upstream_parity.sh` runs `test_yolo_capi_performance` on one or
more devices (cuda/vulkan/cpu) against the canonical upstream checkout
(`dl/ultralytics-ggml`, models in `cpp_ggml/models/gguf`, image
`ultralytics/assets/bus.jpg`) and gates the result with `tests/yolo/bench_compare.py`:

```bash
core/AICore/tests/yolo/run_upstream_parity.sh          # full matrix
core/AICore/tests/yolo/run_upstream_parity.sh --devices cuda --limit 5
```

- The join key is `(model, task, dtype, backend)`; upstream `device`
  (`cuda|vulkan|cpu`) is matched against the AICore resolved device name
  (`CUDA0|Vulkan0|cpu`), and model names are normalized (`-f16/-f32/-q8_0`
  suffixes stripped).
- The gate is **e2e p50 regression ≤ 5%** (integration-plan §12.5); exit code 1
  lists every failing row.
- Thread parity matters: the upstream matrix used 32 threads for cuda/vulkan
  and 8 for cpu. The script sets `AICORE_TEST_YOLO_THREADS` per device
  (`YOLO_GPU_THREADS`/`YOLO_CPU_THREADS` overridable); running GPU rows with
the default 1 thread regresses preprocess ~7× and the CUDA graph ~50% purely
as a harness artifact.

## Controlled CUDA build comparison

`tests/yolo/cuda_build_compare.sh` collects per-op profiles, compile-definition
differences, and optional SASS dumps for an integrated/upstream comparison:

```bash
core/AICore/tests/yolo/cuda_build_compare.sh --upstream dl/ultralytics-ggml
```

The former small-kernel regression was traced to forcing
`GGML_CUDA_FORCE_MMQ`; the public `AICore_CUDA_FORCE_MMQ` option now defaults
OFF for upstream path parity. Driver-only deployments may enable it explicitly,
but that is a different benchmark configuration and must be reported as such.

Each experiment must re-run `run_upstream_parity.sh --devices cuda --limit 5`
and keep the +5% e2e gate. Do not hand-edit build-tree ggml sources; durable
changes go through `3rdparty/ggml/patches/`.

## Evidence semantics

- Contract tests prove ABI, ownership, NULL safety, stride handling, timing
  semantics, and lifecycle; they do not prove inference accuracy.
- CPU/GPU parity proves agreement between backends; it does not prove agreement
  with the originating framework.
- Stable hashes prove determinism, not semantic correctness.
- Graph time is not plugin end-to-end time. Keep preprocess, inference,
  postprocess, compatibility serialization, queue, decode, and render scopes
  separate.
- A complete claim requires every requested pipeline, model, quantization,
  backend, and platform row. Exit 77 and missing assets remain incomplete rows.
- Percentage speed claims require a clean controlled A/B on the same hardware,
  model, input, backend, thread count, warmups, and iteration protocol.

## Env hygiene guard

AICore reads/writes process environment variables only in two sanctioned
files: `src/common/data_root_util.cpp` (deployment `CLOUDVIEWER_DATA_ROOT`
convention) and `src/common/ggml_env_bridge.cpp` (the single writer of
ggml-side variables, translating explicit options into env before a backend
instance is created). `tests/check_no_env_getenv.sh` enforces this whitelist
and runs in CTest as `test_no_env_getenv` (label `capi`); a `getenv`/`setenv`
in any other source file fails the guard.
