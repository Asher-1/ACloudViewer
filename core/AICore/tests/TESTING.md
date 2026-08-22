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

The root contains `da3_models`, `freesplatter_models`, `lightglue_models`,
`deeplsd_models`, `facedetect_models`, `lightglue_test_images`, and
`friends_faces`. Individual `AICORE_TEST_*` variables still override this
convention when a specialized fixture is required.

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

## CUDA graph build-artifact parity (open item)

CUDA rows still exceed the +5% gate (e2e +20~88%, graph +36~87%) while
Vulkan/CPU are aligned — the gap is isolated to small conv kernels
(per-op ~1.8-2× slower, big kernels equal) and points at the integrated
`ggml-cuda` build artifact, not the graph structure, arch, ggml version, or
build type. The dedicated follow-up lives in
`core/AICore/docs/cuda_graph_parity.md`; evidence collection runs via
`tests/yolo/cuda_build_compare.sh` (per-op profiles of both sides,
compile-definition diff, optional SASS dump):

```bash
core/AICore/tests/yolo/cuda_build_compare.sh --upstream dl/ultralytics-ggml
```

Each experiment must re-run `run_upstream_parity.sh --devices cuda --limit 5`
and keep the +5% e2e gate. Do not hand-edit build-tree ggml sources during
this work — changes go through `3rdparty/ggml/patches/` per the repo rules.

## Env hygiene guard

AICore reads/writes process environment variables only in two sanctioned
files: `src/common/data_root_util.cpp` (deployment `CLOUDVIEWER_DATA_ROOT`
convention) and `src/common/ggml_env_bridge.cpp` (the single writer of
ggml-side variables, translating explicit options into env before a backend
instance is created). `tests/check_no_env_getenv.sh` enforces this whitelist
and runs in CTest as `test_no_env_getenv` (label `capi`); a `getenv`/`setenv`
in any other source file fails the guard.
