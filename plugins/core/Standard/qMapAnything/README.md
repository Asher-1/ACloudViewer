# qMapAnything — MapAnything Multi-View Reconstruction

Universal feed-forward metric 3D reconstruction from multi-view image sets,
powered by the in-tree ggml port (`core/AICore/src/tasks/map_anything/`,
upstream map-anything-ggml `cpp_ggml`). One monolithic libAICore task serves
**six model architectures**, routed by the GGUF `general.architecture` key:

| Variant (tab order) | Architecture key | Camera outputs | Notes |
|---|---|---|---|
| VGGT-Omega 1B 512 *(first tab, accuracy pick)* | `vggt_omega` | pose_enc 9 (absT + quat + FoV) | best pointmap/depth/pose accuracy |
| VGGT-Omega 1B 416 (reproduce) | `vggt_omega` | pose_enc 9 | paper-reproduction 416 px checkpoint |
| VGGT-Omega 1B 256 (text) | `vggt_omega` | pose_enc 9 + text embedding | language-aligned head |
| MapAnything | `mapanything` | c2w 4x4 + mask + metric scale | metric scale & poses pick |
| Pi3X | `pi3x` | c2w 4x4 | balanced paper-pinned alternative |
| Pi3 | `pi3` | c2w 4x4 | camera-frame depth convention |
| VGGT-1B | `vggt` | pose_enc 9 + view0-frame points | original VGGT point head |
| DUSt3R | `dust3r` | none (pairwise view1 frame) | throughput / small devices |

Each variant page offers its four published quantizations (F16 / Q8_0 /
Q5_K / F32; Q8_0 marked *recommended*) plus a custom-GGUF path.

## Build

```bash
-DPLUGIN_STANDARD_QMAPANYTHING=ON   # requires AICore_ENABLED=ON
```

## Dialog layout (qYOLO contract)

- Global **Device / Threads** row rendered once above the model list.
- Left-hand **model list** (one item per variant, first = vggt-omega-512)
  driving a stacked widget: **one page per model variant**, each with its
  checkpoint combo, custom path, image size / resize mode and output
  toggles.
- All pages **share** the common components: the multi-view input block
  (add files / import from DB / preview, ≤ 16 views), the Run/Cancel row,
  the *Use test data* controls, the progress bar and the log.

## Use test data

The *Use test data* button downloads the official MapAnything ETH3D
benchmark scenes from
<https://github.com/Asher-1/cloudViewer_downloads/releases/tag/map_anything_vggt_test_data>
into the shared cache root:

```
~/cloudViewer_data/download/<scene>.zip          (streamed SHA-256 verified)
~/cloudViewer_data/extract/map_anything_test_data/<scene>/   (extracted)
```

The first run downloads (scenes are 224 MB – 1.6 GB); later runs load the
cached scene instantly. After loading, the dialog fills the multi-view
input list (capped at 16 views) and `Run` starts the reconstruction.

## Model cache

Published GGUFs: <https://huggingface.co/Asher-1/map_anything_vggt>
(32 checkpoints, SHA-256 pinned in `aicore/asset_digests.h`, catalog
exposed through `test_catalog_dump_urls --json`). Cached under
`~/cloudViewer_data/extract/map_anything_models/`; the dialog downloads a
missing checkpoint through the shared `ecvModelDownloader` on first Run.

## Outputs

- **Merged world-frame point cloud** (RGB colored, added to the DB tree,
  `MapAnything_<model>_<device>`): composed per architecture from the typed
  inference results — world/common-frame points when the model publishes
  them, local points lifted by the c2w pose (pi3 family), or depth
  unprojected through the pose_enc FoV intrinsics (vggt family).
- **Per-view depth rasters** (turbo colorized `ccImage`) when enabled.
- A normalized confidence floor filters unstable points (sigmoid for the
  pi3 family's raw logits, conf−1 for the 1+exp families).

## Notes

- Inference cost grows quadratically with the view count (full inter-frame
  attention) — keep batches modest; the dialog enforces a 16-view cap.
- License notes per family live in the catalog rows (`quant_note` /
  `license_note`): the MapAnything/Pi3 lines are Apache-2.0, the VGGT and
  DUSt3R lineages carry upstream non-commercial terms — verify before
  commercial use.
