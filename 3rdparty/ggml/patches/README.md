# ggml patches (ACloudViewer)

Upstream ggml is fetched by CMake `ExternalProject_Add(ext_ggml)`. **Do not edit
the extracted tree by hand.** All changes must live here as reviewable artifacts.

## Adding a patch

1. Check out the pinned ggml version locally (see `GGML_URL` / `GGML_SHA256` in
   `3rdparty/ggml/ggml.cmake`).
2. Make the minimal change in that tree.
3. Export a unified diff from the ggml repo root:

   ```bash
   git diff > 3rdparty/ggml/patches/<topic>/0001-short-name.patch
   ```

4. Register the patch in `manifest.yaml` (order matters).
5. Reconfigure/build; `apply_ggml_patches.py` runs in `PATCH_COMMAND`.

## Apply order at configure time

1. `apply_ggml_patches.py` — **manifest `*.patch` files (required for new work)**
2. Legacy idempotent Python mutators (Metal FA, conv_transpose, CPU variants)

Migrate legacy scripts to `.patch` files when touching the same code paths.

## ALIKED Vulkan (AICore / qLightGlue)

Merged patches (functional modules, 2026-07-31):

| Patch | Purpose |
|-------|---------|
| `aliked_merged/0001-vulkan-aliked-core.patch` | Shaders, dense-copy, DCN, C API, proc address, CMake header install (legacy 0001–0003) |
| `aliked_merged/0002-vulkan-aliked-gpu-followup.patch` | SDDH + DKD parity/speed fixes (legacy 0004–0014) |

Legacy incremental patches under `aliked/0001`–`0014` are kept for archaeology;
0007+ had malformed multi-hunk lines — regenerate merged patches with:

```bash
3rdparty/ggml/patches/export_merged_aliked_patches.sh [path/to/LightGlue-GGML/third_party/ggml]
```

Single-file export (core only, legacy):

```bash
3rdparty/ggml/patches/export_aliked_patch.sh [path/to/LightGlue-GGML/third_party/ggml]
```

Enable in AICore: `AICore_USE_VULKAN=ON` → defines `AICORE_VULKAN_ALIKED` when the patch is applied.
