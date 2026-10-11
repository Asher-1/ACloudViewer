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

Each variant page offers its published quantizations (F16 / Q8_0 / F32 for
every variant, plus Q5_K and/or Q6_K depending on the family; Q8_0 marked
*recommended*) plus a custom-GGUF path.

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
(36 checkpoints, SHA-256 pinned in `aicore/asset_digests.h`, catalog
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
- Per-architecture official filter rule (shared GUI/headless
  `MapAnythingReconstruction::filter_scene`): mapanything keeps
  `conf > torch.quantile(conf[view], p)` per view; pi3/pi3x keep
  `sigmoid(logit) > t`; the 1+exp families (vggt-omega, vggt-1b, dust3r)
  keep `conf >= max(2, p20)` (upstream `e2e_full_scene.py`) and then the
  normalized `(c-1)/c >= min_conf` gate. `vggt-1b` defaults to 0.5, i.e.
  exactly the `c >= 2` floor: its confidence range depends on the batch
  (torch VGGT-1B itself gives q50 = 1.02 on courtyard frames 8..15), so the
  old absolute `conf >= 5` (0.8) default emptied whole scenes.

## Scene driver and device memory

GUI worker and `-MAP_ANYTHING_RECONSTRUCT` run one Qt-free driver
(`MapAnythingReconstruction::run`): up to `max frames per pass` views (64 by
default, 16 with the dialog's responsive toggle) go through one forward;
larger sets stream overlap-2 windows whose shared frames fit a dense
per-pixel Sim3 into chunk 0's frame.

Scene composition is not a bit-exact upstream contract (only the forward
is). Upstream `scripts/e2e_full_scene.py` defaults to `--align gt`, which
anchors each chunk to benchmark GT camera centers an application does not
have; its legacy `--align chain` fits chunk i to chunk i-1's *unaligned*
points and never composes the transforms, so chunks >= 2 land in the wrong
frame. The driver instead fits each chunk to the already-aligned copy of its
shared frames (same robust Horn Sim3, median + 3 MAD, 2 iterations), i.e. the
chain composed into chunk 0. The filter population also differs by
construction: upstream thresholds on the torch reference confidences and
counts overlap frames once per chunk; the driver uses its own confidences
and each view once.

Before the first compute AICore checks that the gallocr graph buffer was
allocated and that free device memory covers a per-node upper bound of the
CUDA op-scratch pool (F16/Q8_1 GEMM operand conversions, flash-attention
F16 Q/K/V and split-K partials) plus 512 MiB fixed. A refused geometry
returns `-3` from `aicore_map_anything_run_images` instead of aborting
inside the cuBLAS pool; the driver then halves the pass and re-plans the
whole scene (nothing has run or been emitted yet). `frames_per_pass` /
`memory_downshifts` in the RESULT_JSON and the dialog log record it.

`tests/check_gui_headless_parity.py` closes the loop: same images, model,
device and budget through the real `MapAnythingWorker` and the headless
command; a row passes only with byte-identical PLYs, equal effective pass
size, finite Sim3 per streamed chunk and `kept/input >= 1%`.

## Upstream parity contract

The reference is the external map-anything-ggml `cpp_ggml` CLI built on
ggml v0.24.0 plus upstream's own committed
`third_party/ggml-patches/0001-cuda-conv-transpose-ks-fast-path.patch` (CUDA
only; part A of `map_anything_merged/0001` is its port). The validation row
pins the upstream revision, the ggml revision and the sha256 of the ggml
working-tree diff, and fails when a reference build is older than any of its
sources. Two gates, both run through
`core/AICore/scripts/compare_map_anything_parity.py`:

| Profile | Shared input | Expectation | Measured (vggt-omega-1b-512 q8_0, CUDA, 2 views) |
|---|---|---|---|
| `--profile frames` | identical f32 CHW tensor (`aicore_map_anything_run_frames` ↔ `cpp_ggml --bin`) | pose / depth / depth_conf and all 73 stage tensors **bit-identical** | 76/76 exact |
| `--profile images` | identical **lossless** image files (`aicore_image_view` ↔ `cpp_ggml --images`) | pose ≤ 2e-3, depth ≤ 3e-2, conf ≤ 1.0, c2w center ≤ 5e-4, axes ≤ 0.02° | 1.95e-3 / 2.28e-2 / 0.844 / 2.4e-4 / 0.0102° |

The `frames` gate holds because the CUDA backend keeps pristine kernel
selection for every node named `mapany_*` (ggml patch
`map_anything_merged/0001`), opting this task out of the F32-direct F16 GEMM,
TF32 MMF and conv_transpose fast-path overrides other tasks rely on, and
because `AICore_CUDA_FAST_MATH` defaults to ON like upstream. The Vulkan
backend applies the same name gate to the three downstream overrides a
manifest-prefix bisect attributes the Vulkan gap to (RMBG F32-K/V
flash-attention routing, SAM3 2x2/stride-2 conv_transpose fast path, lingbot
F32-weight conv_transpose scalar routing). (The SAM 3D task,
whose upstream is built without fast math, keeps its own numerics inside the same
library through the per-translation-unit kernel twins of ggml patch
`sam3d_merged/0005`, so neither contract forces the other's compiler flag.)

Release matrix: `validate_all.py --full --tasks map_anything --reference
map_anything_ggml=<map-anything-ggml checkout>` runs the `frames` gate for
every published GGUF (36 rows) on the selected backend. Measured on RTX 4090
(driver 580.95.05): CPU, CUDA and Vulkan each 36/36 with max |diff| 0 on every
typed output. Metal is not measured (no macOS host) and bit-parity is **not**
claimed: statically, part C only adds the batch index to the downstream
`metal_merged/0001` conv_transpose_2d kernel, whose IC-parallel tree reduction
sums in a different order than pristine v0.24.0 (`kernels/conv.metal`: IC
sequential per (kh, kw) thread, then serial across threads), and the Metal
backend itself differs between v0.21.0 and v0.24.0.

The `images` residual is entirely pre-model: AICore decodes with Qt and
resamples with its own separable Catmull-Rom, upstream uses stb_image +
stb_image_resize2; the two agree except for ~1e-4 of the samples that round
to a different 8-bit LSB. JPEG inputs are outside the gate — stb's built-in
JPEG decoder and libjpeg differ by up to ±3 LSB on a few percent of pixels
(measured: conf 2.97 on the `DSC_028x` set, which is JPEG despite its `.png`
extension).

Stage dumps need both `-DMAP_ANYTHING_ENABLE_DUMP=ON` and a non-empty
`stage_dump_dir`; production builds compile them out.

## Notes

- Inference cost grows quadratically with the view count (full inter-frame
  attention); see "Scene driver and device memory" for how large sets are
  split.
- License notes per family live in the catalog rows (`quant_note` /
  `license_note`): the MapAnything/Pi3 lines are Apache-2.0, the VGGT and
  DUSt3R lineages carry upstream non-commercial terms — verify before
  commercial use.
