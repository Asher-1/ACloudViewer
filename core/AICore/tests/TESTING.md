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
changing the input resolution or acceptance thresholds:

```bash
AICORE_ALIKED_STAGE_BENCH=1 \
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
