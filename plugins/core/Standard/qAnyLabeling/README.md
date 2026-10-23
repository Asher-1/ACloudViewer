# qAnyLabeling

See [UPSTREAM_PARITY_AUDIT.md](UPSTREAM_PARITY_AUDIT.md) for the full upstream
parity audit, model release status, quantization gates, VLM/KIE, point-cloud
visual parity, and cross-platform package coverage.

Native Qt image annotation workspace inspired by X-AnyLabeling. Enable with
`-DAICore_ENABLED=ON -DPLUGIN_STANDARD_QANYLABELING=ON` and open **Image
annotation** from the Plugins menu.

Upstream visual references (UI overview, navigator, label/shape managers,
settings, compare view, brush and magic-wand interaction loops) live in
[docs/images](docs/images), copied from the X-AnyLabeling sources as the
baseline for layout parity checks.

The current implementation supports local image/folder navigation, X-AnyLabeling
JSON sidecars, object/label lists, template-driven image-flag checking, checked
state, undo/redo,
current-image close (`Ctrl+W`), label-only Save As (`Ctrl+Shift+S`), and
confirmed label-file deletion (`Ctrl+Delete`),
cursor-anchored zoom, panning, object renaming/locking/visibility, canvas
Ctrl-click and list multi-selection, group/ungroup, rectangle/polygon union,
and creation
of polygon, rectangle, rotation, quadrilateral, circle, line, point, line
strip, and cuboid geometry. Rotation has a drag handle; rectangle corner edits
preserve right angles, and cuboid JSON includes the upstream `cuboid3d` depth
metadata. The magic wand selects a seed-connected color region and converts it
to a polygon. Brush editing supports Paint/Erase/Invert, target class
selection, Ctrl-held erase, `R` paint/erase switching, right-click exit,
one-pixel wheel radius steps, stroke undo/redo, commit, and Escape
cancellation. With
no selected instance, a non-empty target class creates a new polygon from the
first committed stroke. For semantic masks, painting a target class removes
overlapping pixels from other editable classes. Erasing restores class 0 only
when it is named `background`; otherwise it leaves pixels unassigned rather
than guessing a replacement class. Exact alpha masks
preserve holes and disconnected areas.
The View menu's Settings action (`Ctrl+0`) groups shortcuts, general behavior,
shape brush controls and canvas display controls. Brush radius and operation,
mask opacity, shape/mask/label visibility, and keep-previous behavior persist
between sessions. The shortcut table reflects the active Qt actions.
Unknown per-shape and top-level JSON fields are retained on save.

The plugin contributes one host action, **X-AnyLabeling Annotation**. Image,
video, point-cloud and AI tools remain modes inside the opened workspace, so
the host menu does not receive duplicate actions for each mode. The workspace
File, Upload and Export menus follow the source action order and use the
official qAnyLabeling icons. `Upload > Image Flags` imports a UTF-8 text
template with one flag per line; the Flags dock is shown only after a template
is loaded, and checked values are persisted in the image sidecar.
The Label Change Manager (`Alt+L`) can rename, remove, hide, or recolor a
category across an inclusive image range. Category color changes are stored in
the compatible `acloudviewer_color` shape extension as RGBA values. Existing
mask pixels change RGB only: per-pixel alpha, holes, and disconnected regions
remain intact. Files with no actual changes are not rewritten.
The image-tags strip below the canvas edits the ordered top-level `tags` array.
It supports add (with the official `tag`, `tag1`, ... defaults), inline edit,
Escape cancel, focus-out commit, multi-select/delete, batch-delete mode with
selected-tag strike-through, copy-all, drag ordering, and `Ctrl+Shift+T`
visibility. Opening a sidecar with tags shows the strip automatically; edits
enter the same atomic save flow as object annotations. Invalid, blank, and
duplicate values are normalized before they are written.
The File menu imports YOLO horizontal boxes, instance segments, oriented boxes,
YOLO Pose, VOC detection/segmentation, and DOTA oriented boxes from a label folder,
plus COCO detection/instance segmentation/keypoints from a JSON file. COCO
uncompressed and compressed RLE instance masks are decoded losslessly; native
exact masks are exported as RLE so holes and disconnected components survive a
round-trip.
Mask import reads RGB or grayscale PNG/JPG masks with the source JSON color map.
MOT tracking import/export uses frame numbers in image names and persistent
group IDs; export writes the source `gt.txt`, `det.txt`, and `seqinfo.ini` set.
MOTS polygon export writes `custom_gt.txt` and `seqinfo.ini`.
ODVG detection import/export reads JSONL records and writes `od.json` plus
`label_map.json`.
VLM-R1 OVD imports JSONL conversations with valid, fenced, `<answer>`-wrapped,
and common JSON5-style box answers and exports the same conversation structure
with an optional class filter and image prefix. Structurally invalid JSONL rows
remain errors so a damaged dataset is not silently accepted.
`Ctrl+4` opens a dedicated PaddleOCR document workspace with the upstream
1560×920 default window, 1200×760 minimum and 240-pixel sidebar. It supports
image/PDF upload to the official asynchronous PaddleOCR API, recent files,
favorites, search/status/type filtering, newest/oldest sorting, linked image
blocks, text corrections, JSON export and Markdown ZIP bundles with cropped
images and layout previews. The default service model is `PaddleOCR-VL-1.6`;
API settings and document records use `~/xanylabeling_data/paddleocr/`.
The saved custom API URL is used by subsequent submissions.
Local text annotation inference remains in the AICore-backed model picker.
Table blocks support inline source token/HTML/TSV editing, merged cells,
formatting, clipboard operations and undo/redo, with source light/dark toolbar
styles and viewport-limited height. Save/Cancel and stale-document guards keep
edits scoped to their source record.
Native PDF rendering, formula/rich-text editors and exact
window visuals are still open. ZIP export
requires locally available page images; its layout rendering is not a pixel
parity claim against the upstream Pillow renderer.

PPOCR Rec/KIE imports and exports the source tab-separated JSON annotation
records, preserving transcription, quadrilateral/polygon geometry, difficulty,
labels, group IDs, and KIE linking metadata. The native export intentionally
creates the source-compatible `crop_img` directory, perspective-corrected
recognition crops, `Label.txt`, and `rec_gt.txt` atomically. KIE keeps its
source annotation stream and linking metadata.
Imports validate the complete batch before replacing any existing image sidecars.
It exports a loaded image folder to YOLO horizontal boxes, instance
segments, or oriented boxes using the selected classes file, plus VOC detection,
VOC segmentation, DOTA oriented boxes, COCO detection/instance segmentation/
keypoints, and YOLO Pose using the source pose-classes YAML layout. Pose export
preserves group IDs, visibility flags, and missing-keypoint padding. RGB and
grayscale mask export reads the source JSON color map, applies label priorities,
and can include empty images or limit output to checked images.
Optional image
copying, skipping empty labels, and keeping out-of-bounds oriented boxes match
the source export workflow. Existing output folders are merged without deletion.

The model panel consumes the AICore DA3, DeepLSD, ALIKED, YOLO, RF-DETR, SAM,
RMBG, GKD, FaceDetect and LoMa catalogs for depth estimation, line and
local-keypoint proposals, detection,
instance and semantic segmentation, pose, oriented boxes, classification, and interactive
segmentation. It is a plain section embedded between the instruction bar and
the canvas — no dock chrome, floating, or title bar — and the left toolbar's
**Auto Labeling** action (`Ctrl+A`) toggles it. SAM point mode uses left-click
positive and right-click negative
prompts; box mode uses a drag rectangle. Full SAM 3 models also accept text
prompts in the model bar. Prompts remain transient, and SAM results are editable
polygons. The panel exposes each catalog quantization and available runtime
backend, uses the shared
downloader and each catalog's cache path, and reuses a worker-thread context.
Selecting one or more visible detection rectangles with no manual prompt runs
the same SAM3 box request once per rectangle and keeps each returned mask linked
to its source class and group.
DA3 depth results are written as sibling `x-anylabeling-depth` PNG files.
Nested AnyView Q8_0/Q4_K selections verify and load the separate metric GGUF.
Typed segmentation masks retain exact alpha pixels in native JSON; a largest
contour polygon remains available for polygon-only interchange. Pose becomes grouped boxes and
points; classification updates image-level flags. RF-DETR masks are generated
at model resolution and mapped to source-image coordinates. RMBG-2.0 alpha
mattes become editable foreground polygons.
YOLO26 semantic class maps become one editable exact mask per present class,
including class 0 and disconnected regions; each shape retains its numeric
`class_id`. These are category regions, not tracked object instances.
GKD accepts comma-separated keypoint names and an optional prompt box. Select
one or more existing rectangles to run top-down keypoint detection in their
ROIs; multiple selections use AICore's shared-prompt batch. Accepted points
keep the source rectangle's group ID and confidence score.
For 1-shot visual prompting, select visible point annotations on an example
image and press **Use selected points**. The support image and ordered points
remain available when browsing target images. Clear the text field for
visual-only inference, or provide one comma-separated name per support point
for multimodal inference; the same support also works with batch and multi-ROI
runs. Press **Clear visual support** to return to text-only prompting.
FaceDetect boxes and five landmarks share a group ID. Selecting a catalog
landmark model enables the paired dense mode: each face additionally emits
the 106 2D and 68 3D points as editable grouped point annotations, with the
3D depth stored in the point's `z` attribute. The detector and landmark files
are downloaded and verified through the shared AICore catalog/cache. YOLO-World and YOLOE
text models expose the published text-encoder quantizations alongside the main
weight quantization. YOLOE prompt-free checkpoints run without a text encoder.
YOLOE checkpoints that carry AICore SAVPE weights also accept one or more
selected visible rectangle annotations as image prompts. The plugin preserves
their source-image pixel bounds, skips the text encoder for that run, and
labels the returned masks as `object0`, `object1`, and so on. A checkpoint
without `yolo.savpe=1` is rejected with an explicit capability error.

RAM, RAM++ and Tag2Text appear in the model picker from the public AICore
catalog. Runs replace the sidecar's top-level `tags` array using the embedded
per-class thresholds; the generic confidence control is disabled. RAM Options
supports English/Chinese output, English delete-tags and keep-only lists. Delete
has priority over keep, and filtering happens before translation. Saved tags
are trimmed and deduplicated like upstream; unchanged tags preserve reviewed
state. Single-image and folder Auto Run preserve existing object shapes.
RAM/RAM++ selects the source OpenCV INTER_LINEAR preprocessing through the v2
public tagging request. The original API and qRAM retain official Pillow
bilinear/antialias preprocessing. Tag2Text additionally saves its caption as the
image description. Exact source ONNX checkpoint equality is not established.
YOLO-World's Object Descriptions selector composes the same RAM catalog with
integer, half-open detection crops. Tag Options also applies to these per-box
descriptions, including Chinese output and filtering; it does not replace the
image-level tags. The source's dedicated composite-model picker/configuration
and exact detector/RAM checkpoint pair remain separate parity items.

OCR uses the public AICore OCR implementation. The picker exposes published
PP-OCRv4 Chinese/English recognition and task-owned recipes for the exact source
Chinese PP-OCRv4, Japanese PP-OCRv3, PP-OCRv5 and PP-OCRv6
Tiny/Small/Medium detector/recognizer pairs. The six new
recipes currently accept local F32 GGUFs; these weights and the shared direction
classifier are not yet published in the downloadable catalog. Missing weights
report the expected shared-cache path. Conversion and source SHA-256 pins are
documented in [the OCR runtime guide](../../../../core/AICore/src/tasks/ocr/README.md).
**Text orientation** is on by default, as upstream: 0/180-degree classification
runs after perspective cropping, rotates only above confidence 0.9, and also
applies to Skip Detection. Turning it off retains recognition without a classifier.
Recipes own the source DB thresholds, candidate limits and batch defaults.
Layout is explicitly off by default; choose a published CDLA layout model to add
regions with their real class labels and reading order. Detection results
replace the current shapes. Skip Detection updates only the descriptions of
selected, visible, unlocked rectangles or polygons with at least three points,
preserving their geometry and metadata, autosave and undo/redo. The shared OCR
task applies the source integer/minimum-area normalization and reading order;
border-crossing crops use replicated pixels, and results retain input ownership.
The next ordinary Run returns to detection. CPU/CUDA/Vulkan window checks cover these paths. PP-OCR KIE annotation
import/export preserves entity labels, directed links, integer-truncated points
and explicit null group IDs. This matches the source annotation workflow;
automatic SER/RE inference is not an active source model and remains reserved
in AICore. All six active source OCR presets have local F32 recipes; new-model
publication/quantization, complete window visuals and dataset accuracy gates
remain open.

Save Visualization Video exports the folder's image annotations to MP4 with
source defaults: 25 FPS (1..120), explicit shape-type selection, five overlay
options and Skip Empty Files off. The default destination is
`../visualizations/<folder>_visualization.mp4`, relative to the first frame's
folder. It streams frames with bounded memory and
preserves an existing destination if cancelled or failed. All frames must have
the same size; MP4 encoding requires even width and height. The workflow is
available in builds with OpenCV video support.

RelateAnything appears as the `relsgg-*` families. Objects come from either of
two sources. With two or more rectangles selected they are the relation objects
and no detector runs. Otherwise the official detector stage of
RelateAnything-GGML runs. The detector is chosen in the Options popup from the
same catalog as the standalone plugin (YOLOE prompt-free, YOLOE, YOLO-World,
YOLO segment, RF-DETR, YOLO detect; F16 and Q8_0 rows). The default is the
official YOLOE prompt-free segmenter; YOLO-World and YOLOE with a class list
bake the official 497 object names through their text encoder, and each
family keeps its official confidence (prompt-free 0.25, YOLO-World 0.15).
Missing detector files are downloaded and verified on the first run. The stage
keeps the best objects after the official class-agnostic
dedupe at IoU 0.7 and the near-full-frame filter, and adds them to the canvas as
ordinary rectangles. Those rectangles are real annotations: they are saved with
the sidecar like any other, and a re-run does not remove them.
The **Options...** button (shown for RelateAnything only) opens a popup with the
two official operating points (video reel: floors 0.45 / 0.42, 8 edges of which
3 spatial, two-graph decode; image demo: one graph, floor 0.5, 10 edges), every
threshold, the detector confidence / NMS IoU / largest-object fraction /
maximum objects, and a predicate vocabulary (official video list, official
image-demo list, the model's full list, or custom text). Custom predicates are
encoded by the relation model's own text student and CLIP tokenizer, so no
retraining is needed; they must already be in the model cache. Editing a number
switches the operating point to Custom. The choices persist between sessions.
Edges use the calibrated score and the shared selection logic of the standalone
plugin (`RelateAnythingDecode.h`). They are drawn as a transient overlay of
curved arrows with predicate chips and listed in the Relation Filter panel
(type, minimum score, text search, CSV export; clicking a row highlights the
edge and selects its subject). The overlay refers to shape indices, so it is
cleared on any shape edit or image change and is never written to the sidecar;
the detector-stage rectangles themselves are saved. Relation Filter rows export
to CSV atomically, with RFC 4180 quoting and a leading apostrophe on cells that
start with `=`, `+`, `-` or `@`. RelateAnything is refused in Auto Run with a
message (the official demos are single-image, video or camera), and the
text-student and tokenizer assets are not listed as models. Video smoothing
stays in the standalone qRelateAnything plugin. `test_qanylabeling_run_path`
drives RAM, the selected-rectangle flow, the detector stage (YOLOE, RF-DETR
and YOLO-World choices) and custom predicates through the real Run button with cached models (skipped when the
models are absent).

The **SAM3 video tracking** action reuses the qSAM3 AICore tracker and shared
video frame reader. It supports text, box, positive/negative point prompts,
model quantization/device/backend selection (the standalone qAnyLabeling
window exposes its own device/backend row), playback, step/seek/reset,
per-instance masks and track IDs, timeline review, refinement, and DB export.
`Track and export video...` runs the consumer-driven tracker from the current
frame through the end and writes one PNG mask per instance plus an atomic
`manifest.json` containing frame numbers, IDs, scores and boxes. Cancelled or
failed runs retain completed frames and are marked `complete:false`. When
exporting from a later frame, the range dialog can export only the remaining
frames or reset the tracker to frame 0. Reset clears manual instance prompts;
add them on the first frame and start export again for a full clip. The manifest's
`complete` field covers the selected start frame through the end; `full_video`
is true only when the entire clip was exported from frame 0. Each mask filename
also contains a detection index so repeated instance IDs cannot overwrite data.
The Import menu accepts this `manifest.json` format and converts every archived
frame into the normal image sidecar format. Stable instance IDs become
`group_id` values, the PNG mask is retained as an exact editable mask (including
holes), and an external outline is generated for polygon tools and MOTS export.
**Open SAM3 archive workspace** extracts the archived source video into an
ordered frame folder beside the archive and opens the resulting editable image
workspace. It checks source path when still available, the source SHA-256 in
new archives, video frame count and dimensions, archived masks, and every decoded frame. Frames and sidecars are
published together only after successful extraction; masks are converted and
written one frame at a time, and cancellation removes the temporary work. A
moved source video can be selected when its old path is no longer available.
Older manifests without `source_sha256` remain readable, while new manifests
reject a same-size, same-frame-count video with different content.
An archive with no detected instances still opens all source frames for manual
labeling; importing it into an existing image folder reports that there are no
instance masks to add.
The separate **SAM3 tracking archive** import adds annotations to an already
extracted folder. That folder must contain every video frame in order from
frame 0, at the archive's dimensions. Partial archives map their recorded
frame indices into that list. The importer checks dimensions and mask paths
before writing sidecars.
The Video Classifier opens the official `video_classification` sidecar format
through the shared video playback component. It supports label colors, a
background-decoded thumbnail timeline, range creation, frame-snapped segment
move and boundary handles, overlap
prevention, segment time and description editing, label settings with UTF-8
text import, playback controls and atomic save. The image and video workflows
refuse to overwrite each other's same-name JSON sidecars. A selected segment
can be exported as an AVI/MP4 clip or sequential JPEG frames using its
frame-snapped start and end times; cancellation leaves no partial output.
The Dataset action exports every labeled segment from the current editor state
into per-label `videos/` MP4 clips and/or `rawframes/` JPEG sequences. It
supports raw-frame FPS selection, `label_map.txt`, `metadata.json`, and an
optional ZIP archive. The exporter validates normalized label and segment
names before writing, accepts Unicode labels, and publishes the dataset only
after all segment outputs and the ZIP are complete. One settings dialog
configures the output and formats. Clips use FFmpeg stream copy by default for
fast keyframe-aligned cuts. Frame-accurate re-encoding uses the source
`libx264`, `veryfast`, CRF 23 settings when FFmpeg is available. OpenCV remains
the fallback for either failed FFmpeg path; metadata records the requested
mode and separate fallback counts.

The **Point cloud annotation** action opens a native Qt/OpenGL point-cloud
workspace. It reads source `.bin` XYZI and ASCII/binary PLY files, discovers
adjacent or `labels/` `.label` files, and writes source-compatible little-endian
32-bit label streams atomically. The low 16 bits store semantic class IDs and
the high 16 bits store instance IDs. The workspace has orbit/pan/zoom,
top/front/side views, semantic/intensity/RGB/instance coloring, point and brush
size controls, polygon and through-selection, semantic/instance assignment,
frame navigation, and undo/redo. Class and instance visibility lists filter the
rendered points without changing labels. Version-1 source class definition JSON
files can be loaded and saved, including semantic names and `#RRGGBB` colors.
Classes can also be created, edited, and removed; removing a class clears its
current-frame semantic and instance labels through the document undo history.

Validated locally with offscreen Qt canvas/plugin/mask-contour tests and a real
image/JSON sidecar round-trip. Two consecutive YOLO `yolo26n-q8_0` detections
passed on CPU, CUDA, and Vulkan. Additional two-forward checks passed for
`yolov8n-seg-q8_0` on CPU/CUDA with `bus.jpg`, `yolo26s-pose-q8_0` on CPU/CUDA
with `bus.jpg`, `yolo26n-obb-f16` on CUDA/Vulkan with `aerial_airport.jpg`,
and `yolo26n-cls-f32` on CUDA/Vulkan with `bus.jpg`. RF-DETR Nano Q8 detection
passed two forwards on CPU/CUDA; Seg-Small Q8 segmentation passed two on CUDA,
both with `bus.jpg`. SAM 2.1 Tiny Q4 point, box, and mixed positive/negative
prompts each passed two forwards on CUDA with `bus.jpg`; point prompts also
passed on CPU and Vulkan. SAM 3 Q8 text prompt passed two forwards on CUDA.
GKD Q4_K returned editable points twice on CPU; FaceDetect buffalo_sc returned
six face groups twice on CPU. YOLO-World Q8_0 with CLIP F16, YOLOE prompt-free
Q8_0, and YOLOE text F16 with MobileCLIP F32 each passed two CUDA forwards.
The real `yoloe-26n-seg-q8_0.gguf` SAVPE checkpoint also returned one editable
visual-prompt mask on two CPU forwards using the `bus.jpg` fixture.
DA3 Base Q4_K depth returned a valid depth image on two CPU forwards using
`SampleRedwoodRGBDImages/color/00000.jpg`; SAM3 Visual Q4_0 returned an
editable instance mask on two CPU box-prompt forwards using `cats_on_sofa.jpg`.
The optional `SAM3PrecisionContract.TrackDistinctVideoFrames` real-video test
passes the first three distinct frames on CPU with a SAM3 Visual Q4_0
checkpoint: the prompted instance keeps its ID and produces non-empty masks on
both later frames. Full-clip mode processes and archives every decoded frame,
including legitimate occlusion/lost-association frames, and checks monotonic
tracker advancement. After adding or refining
an instance, the video worker replays the conditioning frame before advancing
the decoder, matching the AICore tracker frame-index contract. Run the test
with `AICORE_SAM3_TEST_VIDEO=<clip>` and `AICORE_SAM3_TEST_MODEL=<gguf>` set
on `build_app/bin/test_sam3_contract`.
The LoMa bridge automatically uses the shared pinned Sacre-Coeur image pair in
`test_qanylabeling_loma_bridge`; missing/corrupt fixtures or F16 models fail the
test. CPU/CUDA/Vulkan return 578/581/578 reviewed correspondences on this host.
Use `QANY_LOMA_BACKEND=cpu|cuda|vulkan` to select the test backend;
`QANY_LOMA_IMAGE_A/B` optionally override the images, and
`QANY_LOMA_TEST_STALE=1` also tests replacing an image during inference.
Input/model changes invalidate old matches before they can be accepted. This
The paired-image bridge now runs ALIKED proposals through the separate
LightGlue matcher as well as LoMa. CPU/CUDA/Vulkan bridge tests cover the
shared pair and stale-input invalidation; SIFT is intentionally not exposed as
an alternate upstream backend.
The 250-frame CUDA video contract completes
with monotonic frame advancement and 248 non-empty tracked frames. CPU full
clip execution exceeded the 30-minute test budget, so it is recorded as a
performance limitation rather than a pass. The repository CUDA catalog report
and the retained CPU/Vulkan per-row probes are integration evidence; the
expanded cross-backend model matrix remains a release gate and must be
resumed to produce a complete report before release.

For a repeatable real-model smoke probe, build `test_qanylabeling_inference`
and pass explicit paths:
`build_app/bin/plugin_test/test_qanylabeling_inference MODEL IMAGE [DEVICE] [TASK] [PROMPT_OR_TEXT_MODEL]`.
The probe performs two forwards and validates typed geometry, group ownership,
backend selection, and prompt generations. It is intentionally a manual probe
because it requires a downloaded checkpoint and fixture image.

The annotation workspace includes a native Navigator dock and Compare View
(File menu) with shared canvas transforms and same-size reference-image
matching. The View menu follows the source order: Navigator, Image Tags,
Fill Drawing Polygon, Loop Through/Select Labels, Zoom In/Out/Original Size,
Keep Previous Scale/Brightness/Contrast, mutually exclusive Fit Window/Fit
Width (kept enabled only while an image is loaded), and the show/hide
visibility group. `Keep Previous Scale` reuses the previous zoom when loading
the next image. Image-dependent commands (save, navigation, Delete File,
Mark as Checked, Auto Run, zoom and fit actions) start disabled and follow the
image, file-list, sidecar and model-panel lifecycle. The Edit menu includes
persistent digit-key mappings for drawing mode
and label, with a canvas-scoped 0-9 shortcut. Source-compatible `Auto Run`
(`Ctrl+B`) opens an inclusive image-range
dialog and cancellable progress view for supported catalog models; each image
is replaced and saved atomically before the next inference reuses the worker
context. Image files can be removed with the source-compatible
`Ctrl+Shift+Delete` action; the image and its sidecar are moved to a sibling
`_delete_` directory after confirmation. The Shape Manager (`Alt+S`) batches
annotation deletion, image moves, removal of matching shapes and addition of
selected shapes over an inclusive range. It honors the output directory,
preserves locked shapes during removal, rejects out-of-bounds additions and
avoids duplicates. Batch image moves use the parent folder's `_delete_`
directory, as upstream does; existing destination files are preserved and
reported as errors. Complete OCR visuals, advanced VQA/chatbot workflows,
training and additional source-specific formats remain open.
Point-cloud cuboid geometry, sidecars, orthographic views and detection-mode
controls are implemented and tested. Top/Side/Front editors match the source
handle hit order and hover cursors, Shift rotation in 15-degree steps, final
release position and initiating-button ownership. Leaving the view, moving
outside it or losing that button commits the last valid edit once. A source
comparison executes the upstream gesture methods for 87 cases; cross-view
edit previews update the main GL view, other side views and camera without
changing the document until release. Escape restores the original geometry.
Full orthographic visuals remain open. The standalone `PointCloudCamera` utility
parses the upstream calibration schema, projects points with depth sorting, and
clips cuboid edges for `none`/`opencv5` distortion; its KITTI fixture comparison
has a maximum coordinate delta of `1.14e-13` over three synthetic/KITTI cases
with 1,018 cuboid segments, and identical depth-palette RGB values. The
point-cloud window supports multiple camera sources, configuration validation,
hover navigation arrows, basename frame matching and point/cuboid toggles.
The camera image floats at the upper right, with a lower-left resize handle,
cursor-anchored wheel zoom, hand drag and double-click fit. Manual zoom/pan
survives label/cuboid edits; four decoded images are retained in an LRU cache.
Its settings menu persists depth/point-cloud color, point size and opacity, with
independent resets and a depth-percentile legend. Point-cloud colors follow the
main view's semantic, intensity, RGB or instance mode, including hidden instances.
The camera also projects the live creation preview and removes it on cancel.
Fifteen source-camera scene comparisons match every pixel across image scales,
point styles, opacity, selected/occluded cuboids, creation and edit previews; window
chrome and fonts are outside this comparison. The core `PointCloudDatasetExporter` writes the
upstream Datumaro, KITTI Raw and Sly Point Cloud ZIP contracts, including PCD
payloads, cuboid metadata and Sly key maps, and is covered by cancellation and
ZIP-integrity regression. `PointCloudDatasetImporter` safely parses those three
archive contracts, validates frame/class/cuboid records and supports
cancellation. Its prepare/apply API uniquely matches frames, merges classes
while reserving IDs used by existing cuboids, previews sidecar/config changes
and applies them with staged writes and rollback. It preserves segmentation
configuration and rejects changes made after the preview; cancelled/failed
exports preserve the previous ZIP. The actual upstream modules validate five
archive cases and the JSON import plan. All three dataset dialogs now apply
those transactions to the live window, reject stale/reentrant workspace edits,
and preserve unsaved segmentation. Detection/segmentation class tables are
separate, including grouped version-1 save/load and legacy migration. Import
applies annotations to the open local sequence, following upstream; archive
PCD decoding is not required by that contract. The main camera follows the
upstream Z-up basis, default perspective and scale limits, with orthographic
switching, pan/orbit/zoom, cuboid focus and oriented ray-box selection.
Main-view creation previews the nearest visible point along the upstream
one-metre ray cylinder; double-click commits a one-metre cuboid and Esc cancels.
GPU surface selection reads an RGBA8 point-owner buffer with depth, preserving
occlusion, equal-depth ties and point footprints. Brush strokes cover the full
segment between mouse events; Through includes hidden layers. Camera
configuration has a debounced source preview with an eight-thumbnail cache.
Renderer tile caching now reuses a 16-pixel projection index across brush
events, while the GPU owner/depth path remains the source of visible-pixel
truth. The source loads complete frames asynchronously; chunked streaming is
an optional optimization. Focus animation is implemented; exact side-view/handle/dialog
visuals and complete screenshot/gesture parity remain open.
See [TODO.md](TODO.md) for separate lists of absent AICore models and
still-pending supported workflows. The pose config reader currently
accepts the source example's indented schema but not every valid YAML construct.
Image cuboid editing includes the source front-corner and edge-center resize
controls, visible rear-endpoint height controls, rear-center depth control,
and left/right/back face drags. It preserves the front/back vertex order,
minimum dimensions, locked state, undo/redo and `cuboid3d` metadata; controls
and whole cuboids can move outside the image. The canvas and visualization
export draw separate front/back faces with dashed rear edges and an orientation
edge. The canvas now has the source-style bottom-left Canvas Display
panel with collapsible Opacity, Brightness and Contrast controls. Brightness and
contrast rebuild a display-only cache (including the source 16-bit grayscale
formula); opacity applies to masks and geometry while labels remain readable.
The original image, prompts, masks and sidecars are not modified by these
controls. The magic wand uses a local CIELAB conversion and the largest
connected external contour; exact source-region parity and full theme/menu
fixture parity remain open. Performance relative to X-AnyLabeling has not been
measured.

The catalog-to-workflow and validation state is summarized in
[MODEL_MATRIX.md](MODEL_MATRIX.md); the runtime AICore catalog remains the
single source of truth for the selectable model files.

The 2026-10-16 follow-up uses local X-AnyLabeling-GGML v4.1.3
(`745f6c355f94cddb06f6965b3a4118f82033e935`) for interaction comparison.
Run `test_qanylabeling_cuboid` for the focused mouse/geometry regression and
`test_qanylabeling_plugin` for the real-window save/undo/redo workflow. To
compare geometry with the actual upstream Python methods, use its Python
environment (including PyQt6) from the ACloudViewer repository root:

```bash
python plugins/core/Standard/qAnyLabeling/tests/compare_cuboid_upstream.py \
  --upstream /path/to/X-AnyLabeling-GGML \
  --probe build_app/bin/plugin_test/test_qanylabeling_cuboid \
  --output build_app/Testing/qanylabeling-cuboid-upstream.json
```

The script checks the deterministic image-cuboid scenarios, including both
depth directions, rotated faces, non-uniform back offsets, degenerate edges
and size limits. Generated references stay in the build tree and include the
upstream revision and Python source SHA-256 values. The point-cloud cuboid
probe is a separate `compare_pointcloud_cuboid_upstream.py` check; its core
geometry and window workflow are landed. Compare the calibration/projection
core against the upstream KITTI fixture with:

```bash
python plugins/core/Standard/qAnyLabeling/tests/compare_pointcloud_camera_upstream.py \
  --upstream /path/to/X-AnyLabeling-GGML \
  --probe build_app/bin/plugin_test/test_qanylabeling_pointcloud_camera \
  --output build_app/Testing/qanylabeling-pointcloud-camera-upstream.json
```

The real-window regression covers dataset dialogs, confirmation/cancellation,
class migration, segmentation preservation, multi-camera configuration and
overlay pixels, GPU surface selection and main-view cuboid creation/cancel.
With PyQt6 and NumPy available in the reference Python environment, compare
camera scene pixels using the executed source widgets and overlay/color methods:

```bash
QT_QPA_PLATFORM=offscreen python plugins/core/Standard/qAnyLabeling/tests/compare_pointcloud_camera_render_upstream.py \
  --upstream /path/to/X-AnyLabeling-GGML \
  --probe build_app/bin/plugin_test/test_qanylabeling_pointcloud_ui \
  --output build_app/Testing/qanylabeling-camera-render-upstream.json
QT_QPA_PLATFORM=offscreen python plugins/core/Standard/qAnyLabeling/tests/compare_pointcloud_gestures_upstream.py \
  --upstream /path/to/X-AnyLabeling-GGML \
  --probe build_app/bin/plugin_test/test_qanylabeling_pointcloud_ui \
  --output build_app/Testing/qanylabeling-cuboid-gestures-upstream.json
```

The second comparison dispatches real Qt events to all three orthographic
editors and executes the source event handlers in a fixed-camera shell. It
compares box geometry, commit counts and cursors over 87 cases, including
positive/negative half-step rotation snapping. Both reports record the probe,
comparison script and executed source fingerprints.

The view comparison includes 18,216 creation-ray queries (maximum center delta
`7.11e-15`) and 5,208 oriented ray-box queries. Remaining renderer/streaming and complete visual/interaction
parity stay tracked in [TODO.md](TODO.md). Run the view-math comparison with
`compare_pointcloud_view_upstream.py` using the same `--upstream`, `--probe`
and `--output` arguments as the camera comparison above. The core archive and
atomic sidecar/config transaction probes can be run with:

```bash
build_app/bin/plugin_test/test_qanylabeling_pointcloud_dataset
build_app/bin/plugin_test/test_qanylabeling_pointcloud_dataset_importer
python plugins/core/Standard/qAnyLabeling/tests/compare_pointcloud_dataset_upstream.py \
  --upstream /path/to/X-AnyLabeling-GGML \
  --probe build_app/bin/plugin_test/test_qanylabeling_pointcloud_dataset_importer \
  --output build_app/Testing/qanylabeling-pointcloud-dataset-upstream.json
```

Opening or dropping annotation JSON now resolves relative/absolute image paths,
the local basename fallback and embedded-image precedence using the upstream
label-file rules. Image-first loading explicitly overrides stale `imagePath`.
Decoding finishes before replacing the active document, and saves retain the
opened JSON identity and write image paths relative to the destination JSON.
The default file-list checkbox setting is false, matching upstream.

The controlled point-cloud GL geometry comparison has exact framebuffer parity
in six Top/Front/Side/perspective cases; full-window screenshot and streaming
parity remain open.

The orthographic cuboid views now share the main viewport's filtered point
indices and active semantic/instance/RGB/intensity point colors. Point size
uses the upstream device-pixel rule. The depth-crop action is enabled by
default, applies to Side/Front with the 0.1 m margin, and is bypassed for Top
and cuboid creation. Each view has a hover expand/restore button; expansion
fills the center viewport area by hiding the perspective view, and restoration
returns the three-view splitter to its saved sizes. `test_qanylabeling_pointcloud_ui`
regresses rendered point masks, color/size pixels and real window controls.
Handle-icon, panel/dialog and full-window visual parity remain open.

Two additional comparisons execute the actual upstream Python methods:

```bash
python plugins/core/Standard/qAnyLabeling/tests/compare_document_upstream.py \
  --upstream /path/to/X-AnyLabeling-GGML \
  --probe build_app/bin/plugin_test/test_qanylabeling_document \
  --output build_app/Testing/qanylabeling-document-upstream.json
python plugins/core/Standard/qAnyLabeling/tests/compare_shape_batch_upstream.py \
  --upstream /path/to/X-AnyLabeling-GGML \
  --probe build_app/bin/plugin_test/test_qanylabeling_shape_batch \
  --output build_app/Testing/qanylabeling-shape-batch-upstream.json
```

Each checks eight cases and records the upstream commit and source hashes.
The batch comparison covers all four operations with default and alternate
output directories. Run the plugin regressions with
`ctest --test-dir build_app -R '^test_qanylabeling_' --output-on-failure`.

The 2026-10-17 default/interaction comparison uses the same v4.1.3 revision.
`AnnotationDefaults.inc` records the source configuration and its SHA-256;
`AnnotationConfig.h` supplies those defaults to controls and canvas settings.
Saved user preferences continue to override defaults. `move_mode` selects the
label popup's cursor/screen-center position. The main model picker reads its
confidence and IoU defaults through the public AICore catalogs. IoU changes
reach the existing worker/context, including the valid endpoints 0 and 1.

Run the cold-start controls and real upstream drawing comparisons using the
upstream Python environment with PyQt6, PyYAML, NumPy and Pillow installed:

```bash
python plugins/core/Standard/qAnyLabeling/tests/compare_config_upstream.py \
  --upstream /path/to/X-AnyLabeling-GGML \
  --probe build_app/bin/plugin_test/test_qanylabeling_config \
  --plugin build_app/bin/plugins/libQANYLABELING_PLUGIN.so \
  --output build_app/Testing/qanylabeling-config-upstream.json
python plugins/core/Standard/qAnyLabeling/tests/compare_render_upstream.py \
  --upstream /path/to/X-AnyLabeling-GGML \
  --probe build_app/bin/plugin_test/probe_qanylabeling_render \
  --output build_app/Testing/qanylabeling-render-upstream
```

The current control comparison covers 80 shortcut groups, 35 exact model
threshold defaults, 13 checkable actions and five docks. The rendering probe
checks 768 palette indices and 36 rectangle/polygon/circle/line/point/rotation
scenarios, including labels, scores, descriptions, attributes, difficult
annotations and mask visibility. Its 18 geometry-only cases require exact
pixels. Text cases permit one-pixel Qt 5/6 font-metric differences, with both
whole-image and cropped error gates; reports retain reference, C++ and diff
images. These checks do not establish full-window/theme or all-state parity.

Image Classifier (`Ctrl+3`) now edits the active document's image flags. It
supports MultiClass/MultiLabel, category search/editing, A/D navigation,
Ctrl+A/Ctrl+D navigation to unclassified images, numeric category shortcuts,
statistics and class-directory export. AI Assistant and AutoRun use the shared
Chat/VQA service configuration for arbitrary user-category VLM classification,
with the source MultiClass/MultiLabel prompt and editable instructions. Local
model > Run/AutoRun retains the AICore classification catalog and shared
worker/download/batch flow for trained categories. Both paths preserve object
geometry and unrelated metadata. Closing, canceling or changing the source
image invalidates pending results; malformed batch sidecars are reported and
preserved.
`test_qanylabeling_plugin` exercises manual flag changes/navigation/export;
`test_qanylabeling_classifier_run_path` exercises the real YOLO26n q8 model,
1000 categories, two-image batch, metadata preservation and stale-result
rejection. The model and image fixtures come from the shared cache.

The initial RAM/OCR consumer checkpoint (2026-10-17) reused the merged OCR
runtime. Its selected full-quantization scenarios passed on CPU, CUDA and Vulkan:
15 published-weight OCR rows per backend (12 catalog models) and 15 tagging rows per backend
(nine official-preprocess models plus six RAM/RAM++ OpenCV-preprocess rows).
These use the developer profile, two process repeats, one warmup and two
measured forwards. They do not establish release performance, the separate RAM
dataset/caption matrix, or equality to X-AnyLabeling's original ONNX weights.

```bash
python plugins/core/Standard/qAnyLabeling/tests/compare_tagging_upstream.py \
  --upstream /path/to/X-AnyLabeling-GGML \
  --probe build_app/bin/plugin_test/test_qanylabeling_tagging \
  --output build_app/Testing/qanylabeling-tagging-upstream.json
AICORE_TEST_RAM_UPSTREAM=/path/to/recognize-anything-ggml \
python core/AICore/scripts/validate_all.py --build build_app \
  --tasks tagging --scenarios ram-upstream-demo1,ram-anylabeling-preprocess \
  --full --backend cpu --threads 4 \
  --reference anylabeling=/path/to/X-AnyLabeling-GGML \
  --output build_app/Testing/qanylabeling-tagging-full-cpu.json
python core/AICore/scripts/validate_all.py --build build_app \
  --tasks ocr --full --backend cpu --threads 4 \
  --reference xanylabeling=/path/to/X-AnyLabeling-GGML \
  --output build_app/Testing/qanylabeling-ocr-core-full-cpu.json
ctest --test-dir build_app -R '^test_qanylabeling_' --output-on-failure
```

Repeat the two model commands with `--backend cuda` and `--backend vulkan`
and distinct output names. The actual-window OCR test also accepts
`test_qanylabeling_run_path <absolute-plugin-path> --ocr <backend>`; the
YOLO-World/RAM workflow accepts `--tagged <backend>`. These verify autosave,
replacement (including empty results), metadata, undo/redo and stale results.
Runtime library paths must match the configured build; leave Qt's offscreen
selection to CTest so its OpenGL point-cloud fixture can use Xvfb.
The original exact-source OCR extension covered 30 F32 graph, orientation,
page and selected-polygon scenarios. It now includes F16/Q8_0 candidates and
labelled-page gates; the per-file state and local artifact paths are recorded
in [OCR_RELEASE_GATES.md](OCR_RELEASE_GATES.md). These candidates require the
named source checkout and its pinned ONNX assets and remain unpublished.

All files in `images/` are copied unchanged from X-AnyLabeling-GGML commit
`926017d09b22564a78c55eb2f368c7b20faa0466` and remain under that
project's GPL-3.0-only license. See `UPSTREAM-LICENSE`. `AnnotationCuboid.h/.cpp`
adapt the upstream image-cuboid interaction rules and carry the same license
and source attribution. The configuration snapshot, label palette and reused
upstream translations also retain their source attribution and GPL-3.0-only
provenance. The paired RAM presentation vocabularies retain their source
revision, hashes and license in `resources/ram/README.md`. The source classifier prompt in `AnnotationClassifierPrompt.h` and window
stylesheet in `resources/theme/` carry the same GPL-3.0-only attribution.
`PpocrRichTextEditor.h/.cpp` also adapt the rich-text block editor in upstream
`anylabeling/views/labeling/ppocr/editors.py` under GPL-3.0-only. Its source
serialization and heading metrics are checked against that file by
`tests/compare_ppocr_rich_upstream.py`.
Other C++ code is independently written against Qt and AICore.


Remote annotation services and embedded resources (2026-10-17):

- Chatbot (`Ctrl+1`), VQA (`Ctrl+2`) and Image Classifier AI Assistant/AutoRun
  share a Qt Network OpenAI-compatible service client. Choose a provider,
  address, API key and model in the dialog; refresh queries `/models`. The
  initial provider is Ollama at `http://localhost:11434/v1`, temperature 1.0,
  with no model or max-token override. Settings persist across dialogs and
  provider switches. The eight source provider defaults and configuration JSON
  import/export preserve provider/model fields. No model is downloaded by the
  service transport.
- Chat supports streaming UTF-8 SSE, image references, multi-turn messages,
  message editing, deletion, copying, regeneration and resending,
  stop, per-image history reload, history JSON import/export and directory-wide
  ShareGPT multimodal dataset import/ZIP export (messages, image paths and
  original images), and folder AutoRun with bounded concurrency. AutoRun
  preserves unrelated sidecar fields and cancels on source changes. Dataset
  import creates/replaces source-format annotation documents for the selected
  images; malformed records and path escapes fail before writes.
  Its sidecar
  stores upstream `chat_history` role/content/image entries; base64 is sent
  only in requests. VQA supports all four source component types, manual
  values, editable component schemas, AI preview/apply, references and
  folder-wide JSONL export with selected/renamed fields. Component JSON
  import/export preserves the upstream schema; component ordering, title and
  option migration and removal update current-task annotations. Malformed
  sidecars block migration before any writes; each output is saved atomically.
  VQA includes the 11 source-default prompt templates and user-template
  add/edit/delete/selection. Source-compatible component/template files are
  loaded from `~/xanylabeling_data/vqa/{components,prompts}.json` (or the
  configured `qAnyLabeling/workDirectory`). Invalid files are preserved;
  explicit component import can replace an invalid component configuration.
  Classification accepts configured JSON boolean
  flags, ignores unknown categories and preserves object geometry/metadata.
  Bad JSON and canceled/stale requests do not change annotations.
- This is remote service support. Local Florence2/VLM graphs, the full source
  Chat/VQA layout and third-party model
  accuracy remain open in TODO. VQA prompt references read the actual sidecar
  and preserve Python value formatting and replacement order. Dedicated
  prompt/loading/result dialogs include cancellation and stale-source guards. Native Anthropic Messages
  is not a separate protocol here; configured endpoints must implement the
  OpenAI-compatible contract.
- RAM GGUFs embed English tags and thresholds, not Chinese presentation tags.
  The paired resource files provide English-to-Chinese display lookup only.
  RAM labels, themes, icons and compiled translations are embedded in the
  plugin with Qt RCC and travel inside its installed binary. No resource
  source-folder or working-directory lookup exists. The current Linux
  generated install rule, RPATH removal, stripping and Unicode-path
  relocation pass real plugin loading; complete installers and native
  Windows/macOS execution remain open. See `resources/ram/README.md`.

`test_qanylabeling_chat` runs an in-process HTTP server and the actual plugin
windows to check service requests, streaming, timeout/error/cancellation,
VQA component IO and preview/apply, history round trips and custom-category
classification single/batch/cancel/invalid-sidecar behavior. No live service
credentials are required. `compare_config_upstream.py` additionally compares
six source classifier prompts, 48 palette roles, eight panel styles and
light/dark canvas pixels. The complete regression remains:

```bash
ctest --test-dir build_app -R '^test_qanylabeling_' --output-on-failure -j1
```

Evidence is under `build_app/Testing/qanylabeling-service-*`,
`qanylabeling-resource-install.{json,log}` and `qanylabeling-vlm-chat.log`.
The full-window dark snapshots (`qanylabeling-window-dark.png` and
`qanylabeling-upstream-window-dark.png`) retain visible layout/font differences;
passing style/default gates are not a full-window parity verdict.
