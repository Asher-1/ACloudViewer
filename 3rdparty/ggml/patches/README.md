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

Merged patch (2026-08-03):

| Patch | Purpose |
|-------|---------|
| `aliked_merged/0001-vulkan-aliked.patch` | Complete ALIKED Vulkan extraction path: shaders, dense-copy, DCN, SDDH, DKD, C API, proc address, and header install |

The old 14-patch incremental chain was removed. Several later patches contained
malformed multi-hunk context and duplicate merged numbering. Regenerate the
single patch from a dirty ggml v0.18.1 reference tree with:

```bash
3rdparty/ggml/patches/export_merged_aliked_patches.sh [path/to/LightGlue-GGML/third_party/ggml]
```

Enable in AICore: `AICore_USE_VULKAN=ON` → defines `AICORE_VULKAN_ALIKED` when the patch is applied.
