# qLingbotMap — LingBot-Map Streaming 3D Reconstruction

Streaming RGB-D 3D reconstruction from an ordered image sequence, powered by
the [LingBot-Map](https://technology.robbyant.com/lingbot-map) Geometric
Context Transformer (GCT) running natively on ggml (CPU / CUDA / Vulkan /
Metal — device auto-pick follows the AICore runtime order).

During the run the live preview uses the SAME official structure at
streaming pace: an online group (`LingbotMap_Online_<model>`) holding the
growing `LingbotMap_frames/` per-frame clouds (one immutable cloud per
frame, appended as each frame completes) plus `LingbotMap_cameras/`
frustums, refreshed by a measured adaptive redraw loop. Because a frame's
cloud never changes after creation, the VTK engine's per-entity draw gate
(`VtkDisplayTools::checkEntityNeedUpdate`) skips every stable cloud and
the refresh stays O(new points) — the main-thread apply+redraw path is
measured per refresh and reported at task end (acceptance budget: well
under 50 ms; interval clamped to a 30 fps floor). Windowed runs build the
cross-window aligned map the same way (`LingbotMap_aligned_frames/`, one
immutable cloud per window batch).

For the materialized result the plugin feeds every frame of a folder
through the official crop preprocessing (aspect-preserving bicubic resize
to a patch-grid-snapped width, default 518), streams them through the
persistent-KV-cache GCT graph, and writes the reconstruction into the DB
tree with the SAME structure and defaults as the official fused viewer
(`ggml_demo.py` + PointCloudViewer):

- `LingbotMap_<model>_<device>` group holding `LingbotMap_frames/` with
  **one point cloud per frame** (`LingbotMap_frame_%06d`) — the official
  viewer keeps per-frame clouds and has no merged/fused entity;
- each cloud filtered exactly like `parse_pc_data`: finite points with
  confidence strictly above the threshold (default 1.5), then the official
  sequential downsample (`--downsample_factor`, default 10 — adjustable in
  the dialog); no extra world-bounds filtering;
- a dedicated `LingbotMap_cameras` subfolder holding the COLMAP-style
  camera frustum per frame (`LingbotMap_cam_*`) — never mixed with the
  point clouds;
- `LingbotMap_trajectory` camera-center polyline points.

4D playback mirrors the official replay: the current frame's point cloud
and frustum only.

## Sky masking (native skyseg) — upstream parity notes

The native skyseg path is aligned with the upstream CLI end-to-end, verified
on the 286-frame courthouse stream (q8, 518x294, scale=8/window=64):

- keep-masks match the upstream CLI's mask cache pixel-wise at 99.99%+
  (the residual 3-7 px/frame are the cv2-bitwise-sensitive edge flips the
  upstream skyseg.cpp itself documents);
- the conf-zeroing semantics (`--mask_sky`) are identical: sky pixels get
  zero depth confidence before delivery;
- the GCT stream outputs are NOT perturbed by the skyseg pass in ACV (the
  skyseg model shares the engine's backend lease). The upstream CLI runs
  skyseg on a second backend instance, which measurably shifts its own GCT
  outputs (frame-0 fov offset ~+0.098 -> ~29 px focal error); ACV does not
  reproduce that defect, so with sky enabled the ACV fused cloud is the
  geometrically self-consistent one.

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
  The dialog's first-run default follows the upstream CLI default — the
  **q8** model on a 24-GB-class GPU (f16 needs a ≥30-GiB card: q8 + the
  release profile measures ~21 GiB, f16 adds 1.1 GiB of weights and OOMs
  on window 6/6 of a 24-GiB 4090); q8 is the memory-saving option for
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

All LingBot-Map test data caches into ONE grouped folder —
`~/cloudViewer_data/extract/lingbot_map_test_data/` (`courthouse/`, `loop/`,
`oxford/`, `university/`, the `*_sky_masks/` bundles and the long-model
`lingbot_map_long/` videos) — instead of littering the extract root shared
with the model catalogs (`lingbot_models/`, `da3_models/`, …).

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
`aicore_lingbot_device_total_memory`: ≥30 GiB cards get the **f16** model
with the official release profile (8/64); 24-GB-class and smaller cards
follow the upstream CLI default (**q8**, ~21 GiB with the release profile —
f16's extra 1.1 GiB of weights OOMs on a 24-GiB card that also runs the
render context) with a downscaled profile (4/32, then 2/16); a CPU-only
run follows the upstream default (f16, 8/64) on host memory. An info hint
in the Advanced group states the detected VRAM and the applied tier; any
manual change persists and the auto tier never fires again.

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
The worker also upgrades a streaming run whose length exceeds one window
(> 320 frames) to windowed automatically — short scenes always keep the
zero-drift single-KV path, exactly like the upstream `--mode streaming`
default.
The numeric core (`LingbotWindowStitcher`) is gated against the official
Python pipeline by `test_lingbot_window_stitch` (build with
`QLINGBOTMAP_BUILD_TESTS=ON`). The engine itself is gated against the
upstream C++ reference by the `lingbot-stream-parity` validation scenario
(`core/AICore/tests/lingbot/test_lingbot_stream_parity`): the pinned
courthouse fixtures stream through the official preprocessing and the
engine's pose/depth/conf must stay within the calibrated per-backend
tolerance of the captured upstream `lingbot-map-cli` goldens (captured by
core/AICore/tests/lingbot/gen_stream_parity_golden.py); unknown backends
skip.

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
the render window. Streaming runs add one immutable per-frame point cloud
plus one camera frustum per completed frame. Windowed runs accumulate the
**aligned** map instead: every finished window contributes its
deduplicated frames after the official similarity warp as one immutable
`LingbotMap_aligned_frames/LingbotMap_aligned_w*` cloud per window batch,
with one aligned camera frustum per frame — never a window-local
(unaligned) intermediate. All online camera frustums live in the dedicated
`LingbotMap_cameras` subfolder, never mixed with the point clouds. After
the run, the dialog's replay transport cycles the final per-frame cameras
at 1–60 FPS in `3D (incremental replay)` or `4D (current frame)` mode —
the same loop the official viewer plays.

The **Live camera** option (official StreamingViewer parity) controls how
the render window's view follows the reconstruction: `Global view (free
camera)` never touches the view; `Fit latest frame (official)` reproduces
the official `_fit_view` — mean center of the newest frame's points, fixed
diagonal eye, re-fitted at frames 1/10/50/120; `Follow latest camera`
rides the newest camera frustum every frame (OpenCV convention: +Z
forward, -Y up). Default is Fit (the official automatic behavior). The
mode is HOT-SWAPPABLE mid-run: changing it while the reconstruction
streams applies immediately from the cached latest frame (logged).

### Run control (pause / resume / image mirror)

- **Pause / Resume**: pauses the reconstruction IN PLACE — both the
  preprocessing loop and the engine's frame callback block on the same
  pause gate, so Pause works from the very first progress tick (not
  only after streaming starts); Resume continues the same stream at the
  exact paused position (cancellation still works while paused).
- **Input frame preview**: the dialog mirrors the processed input image
  of the frame just reconstructed (official viewer's frame panel), and
  the REPLAY transport drives the same preview — scrubbing the timeline
  or playing back shows the matching input frame, for image folders and
  video files alike (both inputs decode into the same per-frame stream).
- **Try sample data in Video mode**: switching `Input source` to `Video`
  retargets the sample-data scene to the long-model mp4 datasets (`Drive`
  / `Lingbo World`); clicking `Try sample data` downloads (if needed) and
  fills the video path automatically. Switching back retargets the image
  scenes.
- **Logging** goes through the shared AICore route
  (`aicore_inference_log` → the MAIN-WINDOW console + log file); the
  dialog hosts no private log widget.
- **Replay transport** (official viewer `Playing` semantics as an
  explicit state machine): `Play` starts the loop (or resumes a pause),
  `Pause` FREEZES on the current frame exactly like the official
  "Playing unchecked" behavior (visibility is not restored — scrub the
  timeline from there), `Stop`/`Reset replay` restore every frame's
  cloud and camera and rewind to frame 0. The timeline slider and the
  `<` / `>` step buttons (official `gui_timestep` / next/prev-frame
  controls) are available while stopped or paused and disabled while
  playing. The `3D (incremental replay)` mode follows the official
  `update_frame_visibility` semantics (`i <= current`), `4D (current
  frame)` shows only the current frame. `Auto-play after
  reconstruction` starts the replay as soon as a result lands.

### Result display (official viewer display sliders, live over the result)

Once a result is materialized, the `Result display` group reproduces the
official PointCloudViewer's display controls against the LAST result
(disabled until then; point/camera sizes and camera stride persist):

- **Visibility threshold** (official `Visibility Threshold` slider,
  1.0-5.0): re-materializes the whole result group with the new
  threshold — the exact official `_regenerate_point_clouds` semantics
  (conf filter AND arange downsample re-run from the raw engine data).
  The slider bottoms at the materialization threshold: points below it
  were never materialized, so the live adjustment can only TIGHTEN the
  display (loosening requires a new Run with a lower Confidence
  threshold). Applied on slider release; the re-materialization is
  session-scoped (it consumes the last run's engine output) and is
  logged.
- **Point size** (official `Point Size` slider, 1-16 px): VTK
  actor-property path — no geometry rebuild.
- **Camera size %** (official `Camera Size` slider): rescales every
  frustum from the auto-computed baseline (25-400 %).
- **Show cameras** + **Camera stride** (official `Show Camera` checkbox
  and `Camera Downsample Factor` slider): frustum visibility, one every
  N frames.
- **Go to first camera / Fit all** (official `First Camera` and
  `Overview` view buttons): jump to the frame-0 camera pose or refit the
  view on the whole reconstruction.

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
