# qLingbotMap — LingBot-Map Streaming 3D Reconstruction

Streaming RGB-D 3D reconstruction from an ordered image sequence, powered by
the [LingBot-Map](https://technology.robbyant.com/lingbot-map) Geometric
Context Transformer (GCT) running natively on ggml (CPU / CUDA / Vulkan /
Metal — device auto-pick follows the AICore runtime order).

The plugin feeds every frame of a folder through the official crop
preprocessing (aspect-preserving bicubic resize to a patch-grid-snapped
width, default 518), streams them through the persistent-KV-cache GCT graph,
and writes the reconstruction into the DB tree:

- `LingbotMap_<model>_<device>` group with one colored point cloud per frame
  (depth back-projected with the frame intrinsics, camera-to-world pose
  applied, visibility-confidence filtered, optionally sky-filtered),
- COLMAP-style camera frustum per frame (`LingbotMap_cam_*`),
- `LingbotMap_trajectory` camera-center polyline points.

## Requirements

- `AICore_ENABLED=ON` (the engine lives in `core/AICore/src/tasks/lingbot/`
  and is exposed through `aicore/lingbot_capi.h`).
- Optional OpenCV (`BUILD_OPENCV=ON`) for the video-file input, which is
  decoded through the shared `video_base` module (`OpenCVFrameSource`,
  same backend as qFaceDetect/qFreeSplatter; `HAS_OPENCV_FACE_CAPTURE` is
  propagated by that library). Without it the image-folder input is
  unaffected and the video option is hidden; CMake prints a warning
  pointing at `BUILD_OPENCV`.
- A LingBot-Map GGUF from
  [Asher-1/lingbot-map-gguf](https://huggingface.co/Asher-1/lingbot-map-gguf).
  The dialog defaults to the **f16** model — the upstream GUI default and
  full-alignment format (pose 1.72e-04 / depth 4.72e-04 vs the official
  fp32 checkpoint over 286 frames); q8 is the memory-saving option for
  small-VRAM tiers, f32 the exact reference, q4 experimental. Missing
  files auto-download with pinned SHA-256 ingestion. The optional native
  sky-segmentation GGUF (`lingbot-map-skyseg-*`) is downloaded the same
  small-VRAM tiers, f32 the exact reference, q4 experimental, and the
  `long-*` variants serve long sequences. Missing files auto-download with
  pinned SHA-256 ingestion (`lingbot_models/` under the shared data root).
  The optional native sky-segmentation GGUF (`lingbot-map-skyseg-*`) is
  downloaded the same way.

## Usage

1. Plugin → `LingBot-Map Reconstruction`.
2. Pick the model (or point to a custom GGUF) and the input — an ordered
   image folder or a video file sampled at a chosen fps (`Input source`
   combo; `Max frames` limits the stream length in both modes).
3. Optional: enable native sky segmentation (outdoor scenes) and pick the
   SkySeg GGUF variant (f16 recommended / q8_0 / f32 — missing files
   auto-download with pinned SHA-256 ingestion).
4. Run. Progress, the resolved backend, and per-frame streaming are logged;
   cancellation stops after the current frame.

### Test data scenes

The `Test data` combo lists the official demo sequences (cached download,
extract, and image-folder auto-fill via `Try sample data`):

- `Courthouse (outdoor)`, `Oxford Spires (outdoor)`, `University (outdoor)` —
  selecting one automatically enables sky masking (the paired cached mask
  bundle for Oxford/University when already downloaded, native skyseg
  otherwise; the skyseg GGUF itself auto-downloads on Run).
- `Loop (indoor loop closure)` — indoor office walkthrough, no sky: sky
  masking resets to `None` automatically.
- `Custom (own image folder)` — no automatic toggle; a reminder shows while
  sky masking is off (outdoor custom data should enable it manually). A
  folder picked through `Browse…` is treated as custom data too.

An explicit sky-masking choice in the combo always wins over the automatic
scene default. With `Cached test masks` the `Mask folder` row is
auto-filled by the test-data flow and can be customized through its
`Browse…` button (masks are `<frame-stem>.png`, 255 = keep, scaled to the
processed resolution).

### Advanced options (upstream parity)

The `Advanced (KV cache / sampling)` group starts expanded and mirrors the
upstream `ggml_demo.py` engine flags. Every option ships pre-filled with
the upstream default, stays adjustable per run, persists across sessions,
and keeps applying its (default or customized) value while the group is
collapsed:

| Plugin option | Upstream flag | Default | Meaning |
|---------------|---------------|---------|---------|
| KV cache scale | `--kv_cache_scale` | VRAM-auto | Persistent scale frames of the GCT KV cache |
| KV cache window | `--kv_cache_window` | VRAM-auto | Sliding-window frames of the persistent cache |
| Frame stride | `--stride` | 1 | Process every Nth frame (applied after `Max frames`) |
| Image extensions | `--image_ext` | .jpg,.png,.jpeg,.bmp,.tif,.tiff | Case-insensitive extension filter for the folder input |
| Rotate frames 90° clockwise | `--rotate_clockwise_90` | off | For portrait phone sequences stored sideways |
| Keyframe interval | `--keyframe_interval` | Auto | Every N-th streaming frame persists its KV; Auto = ceil(N/320), 1 when the stream is shorter. Windowed mode always runs per-window interval 1. |

On the **first run** (no persisted choice yet) the KV-cache profile AND the
model format are auto-profiled to the detected GPU memory via
`aicore_lingbot_device_total_memory`: ≥22 GiB cards get the official
release profile (8/64) with the **f16** model (the upstream GUI default,
full-alignment format); 12-GB-class and smaller cards keep the official
memory-saving **q8** model with a downscaled profile (4/32, then 2/16); a
CPU-only run follows the upstream default (f16, 8/64) on host memory. An
info hint in the Advanced group states the detected VRAM and the applied
tier; any manual change persists and the auto tier never fires again.

### Video input (upstream `--video_path` / `--fps` parity)

With `Input source → Video file`, frames are sampled exactly like the
upstream `extract_video_frames`: `interval = round(source fps / fps)`
(source fps falls back to 30 when the container does not report it),
converted BGR→RGB, then `Max frames` truncation and `Frame stride` sampling
apply in the same order as upstream. Requires `BUILD_OPENCV=ON`.

### Deliberate non-parities

- `--height` / `--width`: upstream marks them as aspect-ratio-distorting
  parity-debug overrides (depth confidence drops 6.39 → 5.22 on courthouse);
  the plugin keeps the official crop rule only.
- `--kv-f16` mode switch and a separate `--num_scale_frames`: the C API
  pins the official release profile (strict F16 KV cache, scale = scale
  frames), which is also the upstream default; the flash fast path is
  excluded by the numerical contract in `models/MODEL_CARD.md`.
- `--downsample_factor`, `--point_size`, `--streaming_view`,
  `--stream_stride`, `--depth_stride`: upstream viser-viewer display knobs;
  results render through the ACloudViewer DB tree instead.
- `--sky_mask_visualization_dir` / `--export_preprocessed`: debug dumps.
## Long sequences (windowed mode)

The release KV cache grows with the stream, so long sequences use the
official windowed pipeline (`GCTStream.inference_windowed` at the
`keyframe_interval=1` windowed defaults): the sequence is split into
overlapping windows (window size 64, overlap 16, official scale pass 8),
every window runs the streaming primitive over a fresh KV cache, and
consecutive windows are similarity-aligned on the overlap and stitched into
the first window's frame. Pick `Mode → Windowed (long sequences)` and the
`long-*` GGUF variants; window size / overlap expose the official defaults.
The numeric core (`LingbotWindowStitcher`) is gated against the official
Python pipeline by `test_lingbot_window_stitch` (build with
`QLINGBOTMAP_BUILD_TESTS=ON`).

## Long sequences (streaming + official keyframe policy)

Alternatively, keep `Streaming` mode with a `long-*` model: the engine now
implements the official `--keyframe_interval` policy (`Keyframe interval`
in Advanced; `Auto` = ceil(N/320) per demo.py's streaming rule, e.g. Drive
→ 4, Lingbo World → 3). Non-keyframes attend to
`[cache | own KV]` but do not persist their KV, bounding the cache for
arbitrarily long runs without the window stitching step (upstream
`verify_keyframe.sh` parity: interval 2/4 land pose/depth RMSE
1.02e-04/1.31e-04 vs the official mirror — the same 1e-4 class as
interval=1).

## Dataset → model guidance

| Dataset | Frames | Recommended model | Pipeline |
|---|---|---|---|
| Drive (long, car-mounted) | ~1050 @ 10 fps | `long-f16` (or `long-q8`) | Streaming, Auto keyframe |
| Lingbo World (long, walkthrough) | ~667 @ 10 fps | `long-f16` (or `long-q8`) | Streaming, Auto keyframe |
| Courthouse / Oxford / University (short) | ~286 | `f16` (or `q8`) | Streaming |
| Loop (indoor, short) | ~300 | `f16` (or `q8`) | Streaming |

The `Test data` combo carries these scenes with per-item tooltips, and a
hint label below it repeats the recommendation for the selected scene.
Selecting a long dataset auto-fills the video input (10 fps sampling) and
suggests the matching `Max frames`; the model choice always stays yours.

## Live preview and playback

While the engine streams, a transient `LingbotMap_Online_*` group grows in
the render window: one stride-subsampled point cloud plus one camera frustum
per completed frame (windowed runs keep per-window subgroups until the final
aligned result replaces them). After the run, the dialog's playback controls
cycle the final per-frame clouds at 1–60 FPS in `3D (all frames)` or
`4D (current frame)` mode — the same loop the official viewer plays.

## Memory notes

The release KV-cache profile (scale=8, window=64) needs ~21 GiB of GPU
memory at 518×294 with the q8 model (the 8-frame scale pass dominates the
compute workspace). On smaller GPUs (e.g. 12 GB), lower **KV cache
scale/window** in the Advanced group (e.g. 4/32 or 2/16), reduce the
**Processing width**, or switch **Device** to Vulkan — its
system-memory-fallback degrades to host memory on overflow instead of
failing the run (CUDA aborts). The plugin also sizes the resident-KV
special segment from the **actual** stream length (folder walk / video
container metadata) rather than the raw `Max frames` cap, so the cap can
stay high without preallocating unused capacity. Keep long sequences
bounded with `Max frames` and prefer the q8 deployment format on 12–24 GiB
GPUs. In **Windowed mode** the per-window KV cache is sized to the window
instead (the official per-window `clean_kv_cache` semantics), so memory
follows the window size; if the backend reports a graph allocation
failure, lower the window size / processing width or pick another device
(Vulkan and CPU tolerate memory pressure differently than CUDA).

## License

Apache-2.0 (LingBot-Map upstream; GGUF deployment). See
`models/MODEL_CARD.md` for the full asset table.
