# qAnyLabeling

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
The LoMa bridge has a real pinned two-image integration test and reports the
accepted correspondence count. The 250-frame CUDA video contract completes
with monotonic frame advancement and 248 non-empty tracked frames. CPU full
clip execution exceeded the 30-minute test budget, so it is recorded as a
performance limitation rather than a pass. The repository CUDA catalog report
and the retained CPU/Vulkan per-row probes are integration evidence; the
expanded cross-backend model matrix remains a release gate and must be
resumed to produce a complete report before release.

For a repeatable real-model smoke probe, build `test_qanylabeling_inference`
and pass explicit paths:
`build_app/bin/test_qanylabeling_inference MODEL IMAGE [DEVICE] [TASK] [PROMPT_OR_TEXT_MODEL]`.
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
`_delete_` directory after confirmation. Exact point-cloud depth picking, OCR,
VQA, chatbot, training and additional source-specific formats remain open.
See [TODO.md](TODO.md) for separate lists of absent AICore models and
still-pending supported workflows. The pose config reader currently
accepts the source example's indented schema but not every valid YAML construct.
Cuboid editing does not yet provide the source project's face and
depth handles. The canvas now has the source-style bottom-left Canvas Display
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

All files in `images/` are copied unchanged from X-AnyLabeling-GGML commit
`926017d09b22564a78c55eb2f368c7b20faa0466` and remain under that
project's GPL-3.0-only license. See `UPSTREAM-LICENSE`. The C++ implementation
is independently written against the Qt and AICore APIs.
