# AICore

Unified ggml inference core for CloudViewer. **One public include tree, one naming scheme.**

## Public API (`include/aicore/`)

| Header | Module | C API | C++ namespace |
|--------|--------|-------|---------------|
| `aicore/aicore.h` | Umbrella + extension contract | — | — |
| `aicore/export.h` | Export macros | `AICORE_CAPI` / `AICORE_CXX_API` | — |
| `aicore/depth_capi.h` | Multi-view depth | `aicore_depth_*` | `aicore::depth` |
| `aicore/gaussian_capi.h` | Image → 3D Gaussians | `aicore_gaussian_*` | `aicore::gaussian` |
| `aicore/depth_image.h` | Qt `QImage` helpers | — | `aicore::depth::ImageDepth` |
| `aicore/depth_gguf_keys.h` | GGUF metadata keys | `AICORE_DEPTH_KV_*` | — |

No legacy aliases (`da_capi_*`, `fs_capi_*`, `DA3ImageDepth`, …).

## Layout

```
core/AICore/
  include/aicore/     ← sole public headers
  src/common/         aicore::locate_data_root, extract_model_dir
  src/depth/          aicore::depth
  src/gaussian/       aicore::gaussian
  tests/common/       shared data-path tests
  tests/depth/
  tests/gaussian/
```

## Tests

```bash
cmake -DAICore_ENABLED=ON -DAICore_BUILD_TESTS=ON ...
cmake --build build --target test_data_root test_depth_capi_contract test_gaussian_capi_contract
ctest -LE model    # fast tier (no GGUF)
ctest -L model     # needs AICORE_TEST_DEPTH_GGUF / AICORE_TEST_GAUSSIAN_GGUF
```

### Coverage matrix

| Area | Test | Tier |
|------|------|------|
| `data_root_util` | `test_data_root` | fast |
| depth GGUF keys | `test_gguf_keys` | fast |
| depth path / compute | `test_path_util`, `test_compute_mode` | fast |
| depth C API contract | `test_depth_capi_contract` | capi |
| depth Qt helper | `test_image_depth` | capi |
| depth load + JSON | `test_depth_capi_load` | model |
| gaussian loader | `test_loader` | fast |
| gaussian ingest / pose / graph | `test_image`, `test_pose`, `test_graph_blocks` | fast |
| gaussian path | `test_gaussian_path_util` | fast |
| gaussian C API contract | `test_gaussian_capi_contract` | capi |
| gaussian load + JSON | `test_gaussian_capi_load` | model |
| gaussian parity | `test_parity` | model |
| no legacy exports | `test_no_legacy_symbols` | capi |

## Extending

See `include/aicore/aicore.h`. New capability = `src/<cap>/` + `include/aicore/<cap>_capi.h` + `tests/<cap>/`.

Regenerate depth GGUF keys:

```bash
python3 plugins/core/Standard/qDA3/scripts/gen_gguf_keys_header.py
```
