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
input list (capped at 64 views) and `Run` starts the reconstruction.

## Output frame

No reference file is needed. Images alone determine the relative camera
poses and the scene shape; they cannot determine where the world origin is
or which way it faces. A scale-free model cannot determine the scale
either (VGGT authors, facebookresearch/vggt#56: normalised training by
design). Every family therefore outputs the frame its official release
outputs, and AICore is bit-exact with upstream cpp_ggml there:

| Family | Origin / orientation (official) | Unit |
|---|---|---|
| vggt, vggt-1b, vggt-omega | first camera = identity (VGGT paper, GT normalisation) | normalised (training GT scaled to mean point distance 1) |
| mapanything | view 0 (`model.py`: "poses in the frame of the reference view 0") | metric (`metric_scaling_factor`) |
| dust3r | view 1 | scale-free |
| pi3 | reference-free: poses in a learned frame, view 0 is NOT identity (official README; the first-view re-expression in `demo_gradio.py` is commented out) | scale-free |
| pi3x | as pi3 | approximate metric (official README) |

Chunked runs express every chunk in chunk 0's frame. Measured on the
courtyard (38 views, CUDA, budget 64; mapanything / pi3x / vggt-1b ran as
19-view chunks), as the Sim3 scale needed to reach the metric GT frame:
vggt-omega-1b-512-f16 11.34, vggt-1b-q8_0 8.64 (normalised units),
mapanything-q8_0 0.915 and pi3x-q8_0 1.42 (metric models; 1.0 would be
exact). Between chunks the boundary Sim3 scale is 1.03 for mapanything but
2.37 for pi3x, so on this scene pi3x's "approximate metric" is per pass, not consistent
across passes. pi3x camera 0 on the sacre_coeur pair is a general rotation
(R00 = -0.64), as the reference-free design predicts.

Placing the result in an external world (geodetic, site survey, a
benchmark GT) needs external correspondences, the same as COLMAP's
`model_aligner` (>= 3 known camera centers, Sim3). The plugin reads two such
sources, both through `MapAnythingReconstruction::anchor_to_reference`: the
WAI `scene_meta.json` of the benchmark scenes (below), and your own camera
positions in the `model_aligner` text format (see *Your own camera
positions*). The benchmark file comes first because the upstream
map-anything-ggml e2e point clouds are expressed in it:

The upstream e2e point clouds
(`benchmarks/results/*/e2e_*/*.ply`, `scripts/e2e_full_scene.py` default
`--align gt`) are NOT in the model frame: they are moved by one robust Horn Sim3
from the predicted camera centers to the benchmark GT centers of
`scene_meta.json` (courtyard: scale 11.34, metres). Overlaying an upstream PLY
on a model-frame result therefore shows a rotated, ~11x smaller copy with
identical inference underneath.

When every input sits in one `<scene>/images/` directory next to
`<scene>/scene_meta.json` (the WAI layout of the test scenes) and *Anchor to
the benchmark frame* is checked (default), the dialog applies that same Sim3
(`MapAnythingReconstruction::anchor_to_reference`, numpy-median robust Horn,
2 iterations) to the cloud and the cameras, which also enables the 2 cm
conf-priority voxel. Headless runs take the file explicitly (`GT_META`);
inputs without a `scene_meta.json` stay in the official model frame above
(with the 2 cm voxel only for the metric families). Measured
(courtyard, 38 views, vggt-omega-1b-512-f16, CUDA): Sim3 scale 11.3405
(numpy upstream `robust_sim3` on the same cameras: 11.340496), 4,794,314
points vs upstream `cpp_f16single_global_38f.ply` 4,818,539, extent
25.81 x 23.48 x 18.66 vs 25.79 x 23.48 x 18.68 m, nearest-neighbour distance to
the upstream cloud median 1.15 cm / p95 2.29 cm (2 cm voxel; upstream's own
cpp-vs-torch median is 1.54 cm). Without the anchor: extent 1.9 x 1.9 x
2.3, median distance 0.25 m.

All 13 *Use test data* zips ship `<scene>/scene_meta.json` beside
`<scene>/images/` (checked against every release zip's central directory;
each zip is SHA-256 pinned in `aicore/asset_digests.h`), so the benchmark
anchor works on any machine that downloads them. Inputs without the file are
not an error: they keep the official model frame.

### Your own camera positions

To put your own images in a real-world frame (GPS, a total-station survey,
UTM / national-grid coordinates, a lab frame), give the known camera
positions as the text file COLMAP's `model_aligner --ref_images_path` takes
(a file prepared for COLMAP geo-registration works unchanged):

```text
# image_name X Y Z            (lines starting with '#' are skipped)
DSC_0286.JPG 3.2501 -1.7002 1.5234
seq1/DSC_0287.JPG 3.9102 -1.6511 1.5310
```

- GUI: *Reference cameras* (file + `X Y Z` / `GPS lat lon alt`); it wins
  over a benchmark `scene_meta.json`. Headless: `REF_IMAGES_PATH` +
  `REF_IS_GPS 0|1` (required: COLMAP defaults `--ref_is_gps` to 1, so no
  value is assumed; exclusive with `GT_META`; relative paths resolve against
  `OUT_DIR` like `GT_META`). JSON-RPC: `ref_images_path`, `ref_is_gps`.
- `image_name` is matched against the input file path: the file name or a
  trailing part of it. Lines naming no input are ignored (as in COLMAP); a
  name matching two inputs, an input named twice, a malformed line or fewer
  than 3 matched inputs fail the run before the model loads. DB-imported
  images are staged under random names and are not matched.
- `GPS lat lon alt`: WGS84 degrees / metres, converted to local
  East-North-Up metres about the file's first line, exactly
  `model_aligner --ref_is_gps 1 --alignment_type enu`.
- Large cartesian coordinates (any axis of the first line >= 1e4) are
  stored relative to that value truncated to hundreds
  (`ecvGlobalShiftManager::BestShift`); the DB clouds carry it as their
  global shift and the headless PLY switches to double xyz, so nothing is
  lost to float (whose spacing at x = 500000 / y = 4100000 is 3.1 cm /
  25 cm).
- The Sim3 is refused when the centers are (nearly) collinear: their spread
  across the main axis must exceed the median fit residual, otherwise the
  rotation about a straight camera path is noise. Benchmark scenes have
  0.29-2.6 m of such spread.

Measured (meadow, 15 views, vggt-omega-1b-512-q8_0, CUDA; references taken
from the scene's GT centers):

| Reference file | Result |
|---|---|
| `X Y Z` = GT centers | PLY byte-identical to the `GT_META` run |
| `X Y Z` + (500000.12, 4100000.46, 250) | per-view clouds = `GT_META` + shift within 3.4e-5 m; double PLY |
| GPS lat/lon/alt of the same centers | per-view clouds = `GT_META` - first center within 2.9e-6 m |
| all three, GUI worker vs headless | byte-identical |
| COLMAP 4.3 `model_aligner` on the same model-frame cameras (`custom` and `--ref_is_gps 1 --alignment_type enu`) | camera centers within 1.29 cm median / 3.66 cm max of ours; residual to the references 5.37 cm (COLMAP) vs 5.28 cm (ours) |

The fused cloud's 2 cm voxel grid is anchored at the frame origin, so a
frame moved by a non-multiple of 2 cm (the ENU origin above) fuses slightly
different survivors (1,567,409 vs 1,566,693 points); the hundreds-truncated
offset keeps the grid aligned with your own frame.

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
anchors each chunk independently to benchmark GT camera centers (available
only for benchmark scenes; the driver anchors the composed scene once, see
*Output frame*); its legacy `--align chain` fits chunk i to chunk i-1's *unaligned*
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
typed output (CUDA / Vulkan re-measured on revision 601806861b). Metal is not measured (no macOS host) and bit-parity is **not**
claimed: statically, part C only adds the batch index to the downstream
`metal_merged/0001` conv_transpose_2d kernel, whose IC-parallel tree reduction
sums in a different order than pristine v0.24.0 (`kernels/conv.metal`: IC
sequential per (kh, kw) thread, then serial across threads), and the Metal
backend itself differs between v0.21.0 and v0.24.0.

The gate runs S = 3 views. At larger S (`--views 16`, 10 models) CUDA and
Vulkan are bit-exact too. Before `map_anything_merged/0001` part E, CUDA
pi3x diverged from S = 7 on (S <= 6 exact; S = 8 max_diff 3.40e-3, S = 16
4.37e-4: metric scale 0.3055448 vs 0.3055266, inherited by every depth /
point / translation). Stage dumps put the first difference at
`mtdec.blocks.0_sdpa` with bit-identical q / roped k inputs: the metric
branch's single-query cross attention (`flash_attn_ext_f16<64,64,8,1>`,
KV = 967*S), launched stream-k with grid `min(blocks_per_sm * nsm, ntiles)`.
The v0.21 kernel compiles to 127 registers / 20608 B shared (4 blocks/SM),
the v0.24 one (K/V smem swizzle) to 130 / 18560 B (3 blocks/SM), so the KV
split and the fixup summation order differ (nsys, S = 7: grid 424 vs 384).
That is a property of the kernel binary, so part E vendors the reference's
own v0.24.0 flash-attention sources verbatim (namespace
`ggml_cuda_mapany`, `mapany_*` FA nodes only, see the manifest note).
Measured with it: twin instance 130 registers, pi3x S = 8 / 16 max_diff 0,
CUDA full matrix 36/36 and S = 16 10/10 at 0; performance A/B (same host,
interleaved, 3 rounds, p50) within +-1% at S = 2 and S = 16 for
mapanything / pi3x / vggt-1b / vggt-omega. Upgrading ggml to 0.24.0 aborts
configure until part E is dropped; a reference that moves its ggml fails the
CUDA row until the twin is re-vendored (`docs/guides/ggml_upgrade_pipeline.md`).

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
