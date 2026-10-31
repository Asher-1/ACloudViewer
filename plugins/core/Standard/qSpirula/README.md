# qSpirula — Spirula Studio integration

Process-boundary integration of [Spirula Studio](https://github.com/harry7557558/spirula-studio),
the upstream 3D Gaussian Splatting trainer (photos/video → splat → textured mesh, GPLv3).

## What the plugin does

- **Launch Spirula Studio** — starts the official upstream GUI binary
  (detached). Training, editing, rendering, meshing and device selection all
  run in the official UI, so upstream UI evolution arrives with the next tag.
- **Import Result to DB Tree…** — reads a trained output and adds it to the
  ACloudViewer DB tree:
  - `splat.ply` (or a run directory / `step-*.ckpt`, resolved through their
    `splat.ply` like upstream `--resume`) → colored `ccPointCloud` with
    `Opacity`/`Scale` scalar fields (SH0 → RGB, low-opacity gaussians pruned);
  - `mesh.ply` → regular PLY load (`ccMesh`).
- **Run Headless Training…** — drives `spirula train <dataset>` headlessly:
  dataset folder, run name, device selector, optional verbatim extra args,
  live stdout log, graceful cancel (SIGTERM → upstream checkpoints on exit),
  and a one-click import hand-off of the finished run directory.

Over a **running GUI session** the same operations are exposed as JSON-RPC
methods (`spirula.version/devices/launch/cli/import`) — see
`agent-integration/docs/COMMAND-MAPPING.md`.

## Roadmap: true gaussian-splat rendering (planned)

Today imported gaussians display as a colored point cloud. The planned
upgrade renders them as actual gaussian splats through the existing VTK
point-gaussian mapper pipeline (`VtkVis.cpp` `s_pgPresets` /
`SetSplatShaderCode`):

1. Tag imported clouds with metadata (`Spirula/GaussianModel`) and keep
   `Scale`/`Opacity` scalar fields plus a per-point rotation representation
   (smallest-scale-axis normal, the same convention qFreeSplatter uses).
2. Add a splat preset whose shader code derives point size and alpha from
   those scalar fields instead of a fixed radius.
3. Gate via the display-options machinery so the fallback point rendering
   stays available.

This is a VtkEngine change with its own GPU validation cycle, kept out of the
integration scope on purpose.

## Build

```bash
-DBUILD_SPIRULA=ON            # builds the standalone spirula executable
-DPLUGIN_STANDARD_QSPIRULA=ON # installs the plugin (requires ext_spirula)
```

`3rdparty/spirula/spirula.cmake` builds the executable via ExternalProject.
Its CMake args mirror the upstream release CI verbatim
(`-DSS_BACKEND=vulkan -DSS_ENABLE_PATENTED=ON`); every other switch keeps the
upstream default, so release parity is structural. Release packages carry the
executable **next to the main application** — `bin/ACloudViewer/spirula` on
Linux/Windows, inside the `.app` bundle's `Contents/MacOS/` on macOS — so
SpirulaLocator's application-directory probe finds it on every platform; the
plugin-scan directories stay purely Qt-plugin modules.

Local-only CUDA variant (not for distribution — dynamic cudart + host SASS):

```bash
-DSPIRULA_BACKEND=cuda
```

## Upstream sync

1. Bump `SPIRULA_TAG` (see the tags of the upstream repository).
2. Reconfigure + rebuild (`BUILD_SPIRULA` re-fetches the tag archive).
3. Run the smoke tests: `ctest -R spirula_ -v` (version + `train --help`).
4. If upstream changed the logo, re-copy `assets/icon_128.png` from the new
   source into `images/`.

The upstream source tree is fetched and built unmodified — no patches. The
`spirula` binary is a separate GPLv3 program aggregated with ACloudViewer;
the executable carries its own license (see `LICENSE.spirula`).

## Notes

- Pretrained weights: core 3DGS training needs none (initialization comes
  from SfM). The optional masking/geometry models (SAM2/SAM3, Grounding DINO,
  BiRefNet, …) are downloaded on first use by the upstream GUI/CLI and cached
  locally; nothing is bundled.
- `SS_ENABLE_PATENTED=ON` builds the GPU video decoder (and, per upstream
  constraint, the SAM/nn stack it sits on). On macOS Vulkan Video is
  unavailable and upstream falls back to an external `ffmpeg` at runtime.
- The first build is large (the full trainer + SfM + SAM/nn stack); slangc,
  GLFW, Dear ImGui and (on macOS) MoltenVK are fetched once by the upstream
  CMake modules. Pass `-DSPIRULA_SLANGC=<path>` for offline builds.
  Ubuntu 20.04 note: the default slang release asset requires glibc ≥ 2.33;
  download the `slang-<version>-linux-x86_64-glibc-2.28.tar.gz` asset from the
  same slang release instead and point `SPIRULA_SLANGC` at its `bin/slangc`.
