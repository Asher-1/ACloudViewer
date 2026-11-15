#!/usr/bin/env python3
# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
"""Assemble the qAnyLabeling "Use Test Data" archive.

The archive bundles the official X-AnyLabeling-GGML demo assets into one
deterministic zip so the plugin's File > Use Test Data action can fill every
annotation mode with a working example:

  image_modes/   one image + one sidecar JSON per annotation mode
                 (17 units; the file-name prefix is the mode guide order)
  formats/       official import/export format samples
                 (COCO / VOC / DOTA / YOLO / MOT / ODVG / VLM / masks /
                 attributes / labels)
  ppocr/         official PaddleOCR detection / KIE / recognition samples
  compare_view/  official LLVIP infrared + visible pair (Compare View)
  video/         official demo video (video tracking)
  pointcloud/    official KITTI 3D detection sample (point-cloud cuboids)

Data provenance: 12 of the 17 image_modes units are verbatim upstream files
(only the sidecar "imagePath" field is rewritten to the renamed image). The
rotation unit is mechanically converted from the official DOTA
labelTxt/demo_obb.txt quads. The line/linestrip, cuboid, circle and
quadrilateral units are the only curated sidecars: hand-placed shapes in the
official format on official demo images, flagged "curated" in MANIFEST.txt.

Determinism: every zip entry gets a fixed timestamp and permission, entries
are added in sorted order, so the same upstream revision always produces the
same SHA-256. The digest is pinned in libs/CVPluginAPI/src/
ecvTestDataRepository.cpp and core/AICore/scripts/validation_manifest.json.

Usage:
    python3 build_test_data.py --upstream <X-AnyLabeling-GGML checkout> \
        --output anylabeling_test_data.zip

Publishing (manual, after the digest is pinned in the repository):
    gh release create anylabeling-test-data \
        anylabeling_test_data.zip \
        --repo Asher-1/cloudViewer_downloads \
        --title "qAnyLabeling official test data" \
        --notes "Official X-AnyLabeling-GGML demo assets for the qAnyLabeling
        Use Test Data action; see MANIFEST.txt inside the archive."

Requires Pillow (for the curated sidecars' image-size sanity checks).
"""

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

# Fixed zip timestamp (the zip epoch) keeps the archive byte-reproducible.
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
ARCHIVE_ROOT = "anylabeling_test_data"

UPSTREAM_DEFAULT = Path(
    "/home/asher/develop/code/dl/X-AnyLabeling-GGML")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_size(path: Path):
    from PIL import Image  # deferred: only the curated units need it
    with Image.open(path) as image:
        return image.size


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def dump_json(path: Path, payload) -> None:
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def rewrite_image_path(sidecar: dict, image_name: str) -> dict:
    sidecar = json.loads(json.dumps(sidecar))  # deep copy
    sidecar["imagePath"] = image_name
    sidecar.setdefault("imageData", None)
    return sidecar


def empty_shape(label, points, shape_type):
    return {
        "label": label,
        "score": None,
        "points": points,
        "group_id": None,
        "description": None,
        "difficult": False,
        "shape_type": shape_type,
        "flags": {},
        "attributes": {},
        "kie_linking": [],
    }


def document_for(image_path: Path, shapes) -> dict:
    width, height = image_size(image_path)
    return {
        "version": "3.0.0",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_path.name,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
        "text": "",
    }


def rotation_shapes_from_dota(dota_txt: Path, limit: int = 10):
    """Mechanically convert official DOTA quads to rotation shapes.

    A DOTA row is "x1 y1 x2 y2 x3 y3 x4 y4 label difficult"; the quad corners
    are the rotated rectangle's corners, which is exactly the plugin's
    4-point "rotation" shape. No coordinates are invented.
    """
    shapes = []
    for line in dota_txt.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) < 9:
            continue
        coords = [float(value) for value in fields[:8]]
        points = [[coords[i], coords[i + 1]] for i in range(0, 8, 2)]
        shapes.append(
            empty_shape(fields[8], points, "rotation"))
        if len(shapes) >= limit:
            break
    return shapes


def line_shapes_for_lane():
    """Curated demo shapes on the official demo_lane.jpg (1280x720).

    Two linestrips follow the visible road markings (the solid yellow edge
    line and the dashed white lane boundary) plus one straight "line" shape
    across a single dash. Coordinates are hand-placed approximations.
    """
    yellow_edge = [
        [402, 719], [436, 648], [478, 561], [517, 478], [553, 405],
        [585, 344], [612, 300], [634, 272], [650, 258],
    ]
    white_dashed = [
        [818, 514], [870, 468], [930, 424], [990, 382], [1045, 349],
        [1090, 325],
    ]
    dash_segment = [[1040, 545], [1170, 500]]
    return [
        empty_shape("lane", yellow_edge, "linestrip"),
        empty_shape("lane", white_dashed, "linestrip"),
        empty_shape("lane", dash_segment, "line"),
    ]


def cuboid_shapes_for_street():
    """Curated demo cuboids on the official street.jpg (2592x1944).

    Vertex order matches annotationCuboid: front TL/TR/BR/BL then back
    TL/TR/BR/BL. The two boxes sit on the clearly visible SUV and truck at
    the right of the frame; coordinates are hand-placed approximations.
    The shape descriptions demonstrate the caption editor's data field.
    """
    suv = {
        "front": [[1420, 468], [1818, 448], [1856, 646], [1448, 682]],
        "back": [[1325, 408], [1723, 388], [1761, 586], [1353, 622]],
    }
    truck = {
        "front": [[2065, 258], [2570, 238], [2585, 468], [2085, 512]],
        "back": [[1985, 188], [2490, 168], [2505, 398], [2005, 442]],
    }
    shapes = []
    for label, box, description in (
            ("car", suv, "Silver SUV waiting at the intersection"),
            ("truck", truck, "White delivery truck on the right side")):
        points = box["front"] + box["back"]
        shape = empty_shape(label, points, "cuboid")
        shape["description"] = description
        shapes.append(shape)
    return shapes


def circle_shapes_for_pandas():
    """Curated demo circles on the official demo_pandas.jpg (900x600).

    A "circle" shape stores two points: the center and one edge point
    (the radius is their distance). Three hand-placed face circles.
    """
    faces = [([225, 352], 55), ([392, 325], 55), ([625, 395], 52)]
    return [empty_shape("panda", [center, [center[0] + radius, center[1]]],
                        "circle")
            for center, radius in faces]


def quadrilateral_shapes_for_car():
    """Curated demo quadrilaterals on the official demo_florence2.jpg
    (640x480): the two side windows of the vintage car, whose slight
    perspective skew is what the quadrilateral tool exists for.
    """
    return [
        empty_shape("window",
                    [[188, 184], [259, 180], [261, 240], [186, 242]],
                    "quadrilateral"),
        empty_shape("window",
                    [[263, 180], [330, 178], [331, 236], [264, 238]],
                    "quadrilateral"),
    ]


# (upstream-relative source, archive-relative destination)
VERBATIM_FILES = [
    # Import/export format samples (official).
    ("assets/annotations/coco_detection.json",
     "formats/coco_detection.json"),
    ("assets/annotations/coco_instance_segmentation.json",
     "formats/coco_instance_segmentation.json"),
    ("assets/annotations/coco_keypoints.json",
     "formats/coco_keypoints.json"),
    ("assets/Annotations/demo.xml", "formats/voc_demo.xml"),
    ("assets/labelTxt/demo_obb.txt", "formats/dota_demo_obb.txt"),
    ("assets/classes.txt", "formats/yolo_classes.txt"),
    ("assets/labels/demo.txt", "formats/yolo_labels_demo.txt"),
    ("assets/attributes.json", "formats/attributes.json"),
    ("assets/attributes/person_attributes.json",
     "formats/attributes_person.json"),
    ("assets/attributes/vehicle_attributes.json",
     "formats/attributes_vehicle.json"),
    ("assets/mask_color_map.json", "formats/mask_color_map.json"),
    ("assets/mask_grayscale_map.json", "formats/mask_grayscale_map.json"),
    ("assets/masks/color_mask.png", "formats/masks/color_mask.png"),
    ("assets/masks/gray_mask.png", "formats/masks/gray_mask.png"),
    ("assets/ODVG/od.json", "formats/odvg/od.json"),
    ("assets/ODVG/label_map.json", "formats/odvg/label_map.json"),
    ("assets/vlm_r1_ovd.jsonl", "formats/vlm_r1_ovd.jsonl"),
    ("assets/MOT/det.txt", "formats/mot/det.txt"),
    ("assets/MOT/gt.txt", "formats/mot/gt.txt"),
    ("assets/MOT/seqinfo.ini", "formats/mot/seqinfo.ini"),
    # Official PaddleOCR detection / KIE / recognition samples.
    ("assets/ppocr/images/zh_val_0.jpg", "ppocr/zh_val_0.jpg"),
    ("assets/ppocr/images/zh_val_0.json", "ppocr/zh_val_0.json"),
    ("assets/ppocr/ppocr-kie/class_list.txt", "ppocr/kie/class_list.txt"),
    ("assets/ppocr/ppocr-kie/ppocr_kie.json", "ppocr/kie/ppocr_kie.json"),
    ("examples/optical_character_recognition/key_information_extraction/"
     "sources/zh_train_0.jpg", "ppocr/kie/zh_train_0.jpg"),
    ("examples/optical_character_recognition/key_information_extraction/"
     "sources/zh_train_0.json", "ppocr/kie/zh_train_0.json"),
    ("assets/ppocr/ppocr-rec/Label.txt", "ppocr/rec/Label.txt"),
    ("assets/ppocr/ppocr-rec/rec_gt.txt", "ppocr/rec/rec_gt.txt"),
    ("assets/ppocr/ppocr-rec/crop_img/zh_val_0_crop_0.jpg",
     "ppocr/rec/crop_img/zh_val_0_crop_0.jpg"),
    ("assets/ppocr/ppocr-rec/crop_img/zh_val_0_crop_1.jpg",
     "ppocr/rec/crop_img/zh_val_0_crop_1.jpg"),
    ("assets/ppocr/ppocr-rec/crop_img/zh_val_0_crop_2.jpg",
     "ppocr/rec/crop_img/zh_val_0_crop_2.jpg"),
    # Compare View pair, video tracking, point-cloud cuboids.
    ("assets/compare_view/infrared/LLVIP-test-190001.jpg",
     "compare_view/infrared/LLVIP-test-190001.jpg"),
    ("assets/compare_view/visible/LLVIP-test-190001.jpg",
     "compare_view/visible/LLVIP-test-190001.jpg"),
    ("assets/demo_video.mp4", "video/demo_video.mp4"),
    ("assets/pointcloud/detection/KITTI_3D_det/calibration.json",
     "pointcloud/KITTI_3D_det/calibration.json"),
    ("assets/pointcloud/detection/KITTI_3D_det/image_02/000008.png",
     "pointcloud/KITTI_3D_det/image_02/000008.png"),
    ("assets/pointcloud/detection/KITTI_3D_det/labels/000008.cuboids.json",
     "pointcloud/KITTI_3D_det/labels/000008.cuboids.json"),
    ("assets/pointcloud/detection/KITTI_3D_det/velodyne/000008.bin",
     "pointcloud/KITTI_3D_det/velodyne/000008.bin"),
    ("assets/pointcloud/detection/KITTI_3D_det/source/calib/000008.txt",
     "pointcloud/KITTI_3D_det/source/calib/000008.txt"),
    ("assets/pointcloud/detection/KITTI_3D_det/source/label_2/000008.txt",
     "pointcloud/KITTI_3D_det/source/label_2/000008.txt"),
]

# (upstream image, upstream sidecar or None, archive image name, curated
#  shape builder or None). Most units are self-contained image + JSON pairs;
# an empty sidecar field copies the image only (the compare-view visible
# half is matched to the compare_view/ infrared half by base name).
IMAGE_MODES = [
    ("assets/demo.jpg", "assets/demo.json",
     "01_rectangle_demo.jpg", None),
    ("examples/segmentation/instance_segmentation/sources/cat_dog.webp",
     "examples/segmentation/instance_segmentation/sources/cat_dog.json",
     "02_polygon_cat_dog.webp", None),
    ("assets/demo/demo_obb.png", None,
     "03_rotation_harbor.png", "rotation"),
    ("assets/demo/demo_lane.jpg", None,
     "04_line_lane.jpg", "lane"),
    ("examples/estimation/pose_estimation/sources/human-pose.jpeg",
     "examples/estimation/pose_estimation/sources/human-pose.json",
     "05_point_pose.jpg", None),
    ("examples/classification/shape-level/sources/multi-label/worker.jpg",
     "examples/classification/shape-level/sources/multi-label/worker.json",
     "06_flags_shape_worker.jpg", None),
    ("examples/classification/shape-level/sources/multi-task/"
     "elon-musk-tesla.webp",
     "examples/classification/shape-level/sources/multi-task/"
     "elon-musk-tesla.json",
     "07_flags_shape_multi.webp", None),
    ("examples/classification/image-level/sources/multi-class/apple.jpg",
     "examples/classification/image-level/sources/multi-class/apple.json",
     "08_flags_image_apple.jpg", None),
    ("examples/classification/image-level/sources/multi-label/fruit.png",
     "examples/classification/image-level/sources/multi-label/fruit.json",
     "09_flags_image_multi.png", None),
    ("examples/description/tagging/sources/image-level/street.jpg",
     "examples/description/tagging/sources/image-level/street.json",
     "10_tags_image_street.jpg", None),
    ("examples/optical_character_recognition/text_recognition/sources/"
     "demo_ocr.png",
     "examples/optical_character_recognition/text_recognition/sources/"
     "demo_ocr.json",
     "11_ocr_document.png", None),
    ("examples/optical_character_recognition/text_recognition/sources/"
     "japan_1.jpg",
     "examples/optical_character_recognition/text_recognition/sources/"
     "japan_1.json",
     "12_ocr_japan.jpg", None),
    ("examples/multiple_object_tracking/sources/Bangkok/Bangkok-0.jpg",
     "examples/multiple_object_tracking/sources/Bangkok/Bangkok-0.json",
     "13_mot_group_bangkok.jpg", None),
    ("examples/description/tagging/sources/image-level/street.jpg", None,
     "14_cuboid_street.jpg", "cuboid"),
    ("examples/optical_character_recognition/key_information_extraction/"
     "sources/zh_train_0.jpg",
     "examples/optical_character_recognition/key_information_extraction/"
     "sources/zh_train_0.json",
     "15_kie_linking.jpg", None),
    ("assets/demo/demo_pandas.jpg", None,
     "16_circle_pandas.jpg", "pandas"),
    ("assets/demo/demo_florence2.jpg", None,
     "17_quadrilateral_car.jpg", "car"),
]

MODE_GUIDE = [
    ("01_rectangle_demo.jpg",
     "Rectangle (HBB) — the upstream canonical demo: 51 person/vehicle "
     "rectangles; draw with W/R, edit, group, and review the label dialog."),
    ("02_polygon_cat_dog.webp",
     "Polygon (instance segmentation) — official cat/dog outlines; create "
     "with Ctrl+N, add/delete vertices, and try the brush edit tools."),
    ("03_rotation_harbor.png",
     "Rotation (OBB) — harbor scene with oriented ships/small-vehicles "
     "converted from the official DOTA sample; rotate via Z/X/C/V, adjust "
     "degrees in the shape list."),
    ("04_line_lane.jpg",
     "Line + LineStrip — road-lane markings; draw line (N) and linestrip "
     "(G) shapes along the yellow edge line and the dashed lane boundary."),
    ("05_point_pose.jpg",
     "Point (pose keypoints) — official human-pose sample with a body box "
     "and 16 labeled keypoints; draw points with Q/P."),
    ("06_flags_shape_worker.jpg",
     "Shape-level flags — official worker sample: per-shape flag editing "
     "(label flags dialog) on the annotated shapes."),
    ("07_flags_shape_multi.webp",
     "Shape-level multi-task flags — official sample combining label and "
     "attribute flags on one shape."),
    ("08_flags_image_apple.jpg",
     "Image-level classification flags — official single-class sample; "
     "toggle image flags in the Flags panel (D key focus)."),
    ("09_flags_image_multi.png",
     "Image-level multi-label flags — official multi-label fruit sample "
     "with several concurrent image flags."),
    ("10_tags_image_street.jpg",
     "Image tags — official street scene; edit tags in the Tags panel "
     "(Ctrl+T) and try the tag batch mode."),
    ("11_ocr_document.png",
     "OCR transcription — official document sample with 54 text "
     "quadrilaterals carrying transcriptions."),
    ("12_ocr_japan.jpg",
     "OCR (CJK text) — official Japanese text sample with a transcription "
     "polygon; edit texts via the shape dialog."),
    ("13_mot_group_bangkok.jpg",
     "Group IDs (MOT tracks) — official tracking frame with group_id per "
     "instance; group/ungroup with Ctrl+G/Ctrl+Shift+G."),
    ("14_cuboid_street.jpg",
     "Cuboid (image) — street scene with car/truck cuboids carrying shape "
     "descriptions; drag the front/back faces and edge centers to feel the "
     "3D controls."),
    ("15_kie_linking.jpg",
     "KIE linking (relations) — official key-information-extraction page "
     "with kie_linking edges; click a relation row to highlight the linked "
     "boxes."),
    ("16_circle_pandas.jpg",
     "Circle — panda faces; the circle stores a center and an edge point, "
     "draw with the Circle tool and adjust the radius handles."),
    ("17_quadrilateral_car.jpg",
     "Quadrilateral — the vintage car's slightly skewed side windows, the "
     "classic use case for 4-point perspective boxes."),
]

README_TEMPLATE = """# qAnyLabeling official test data

Bundled from the official X-AnyLabeling-GGML demo assets for the plugin's
**File > Use Test Data** action. Per-file provenance (including the two
curated sidecars) is pinned in `MANIFEST.txt`.

## image_modes/ — one click fills this folder

Open any entry; every unit is an image plus a ready-made sidecar JSON in the
official X-AnyLabeling format:

{mode_rows}

## Other folders

- `formats/` — official import/export samples: COCO (detection, instance
  segmentation, keypoints), VOC XML, DOTA OBB, YOLO classes/labels,
  attributes templates, mask maps + mask PNGs, ODVG, VLM jsonl, MOT
  (det/gt/seqinfo). Use them through File > Import/Export Utilities.
- `ppocr/` — official PaddleOCR detection page, KIE (key information
  extraction) bundle and recognition crops/labels.
- `compare_view/` — official LLVIP infrared/visible pair for the Compare
  View slider.
- `video/` — official demo video for video tracking (Open Video).
- `pointcloud/` — official KITTI 3D detection sample (calibration, velodyne
  bin, camera image, cuboid labels) for the point-cloud annotation window.

## Restoring pristine demo data

Edits are saved into the copied session directory under
`cloudViewer_data/anylabeling_demo`, never into the extracted archive cache.
Delete that directory to restore the pristine demo on the next run.
"""


def upstream_commit(upstream: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=upstream, check=True,
            capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def stage_archive(upstream: Path, staging: Path) -> list:
    """Copy/derive every archive member; return [(relative path, source)]."""
    provenance = []

    def emit(relative: str, source: str) -> Path:
        target = staging / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        provenance.append((relative, source))
        return target

    for source_rel, dest_rel in VERBATIM_FILES:
        source = upstream / source_rel
        if not source.is_file():
            raise SystemExit(f"missing upstream file: {source_rel}")
        target = emit(dest_rel, f"upstream:{source_rel}")
        target.write_bytes(source.read_bytes())

    for image_rel, sidecar_rel, image_name, curated in IMAGE_MODES:
        image_source = upstream / image_rel
        if not image_source.is_file():
            raise SystemExit(f"missing upstream file: {image_rel}")
        emit(f"image_modes/{image_name}", f"upstream:{image_rel}")
        target_image = staging / "image_modes" / image_name
        target_image.write_bytes(image_source.read_bytes())
        if sidecar_rel == "":
            # Image-only unit (compare-view visible half): no sidecar.
            continue
        sidecar_path = staging / "image_modes" / (
                Path(image_name).with_suffix(".json").name)
        if curated is None:
            sidecar_source = upstream / sidecar_rel
            if not sidecar_source.is_file():
                raise SystemExit(f"missing upstream file: {sidecar_rel}")
            sidecar = rewrite_image_path(load_json(sidecar_source),
                                         image_name)
            provenance.append((f"image_modes/{Path(image_name).with_suffix('.json').name}",
                               f"upstream:{sidecar_rel} (imagePath rewritten)"))
        else:
            if curated == "rotation":
                shapes = rotation_shapes_from_dota(
                        upstream / "assets/labelTxt/demo_obb.txt")
                source_note = ("curated: converted from upstream "
                               "assets/labelTxt/demo_obb.txt (DOTA quads)")
            elif curated == "lane":
                shapes = line_shapes_for_lane()
                source_note = ("curated: hand-placed shapes on upstream "
                               "assets/demo/demo_lane.jpg")
            elif curated == "pandas":
                shapes = circle_shapes_for_pandas()
                source_note = ("curated: hand-placed shapes on upstream "
                               "assets/demo/demo_pandas.jpg")
            elif curated == "car":
                shapes = quadrilateral_shapes_for_car()
                source_note = ("curated: hand-placed shapes on upstream "
                               "assets/demo/demo_florence2.jpg")
            elif curated == "cuboid":
                shapes = cuboid_shapes_for_street()
                source_note = ("curated: hand-placed shapes on upstream "
                               "examples/description/tagging/sources/"
                               "image-level/street.jpg")
            else:
                raise SystemExit(f"unknown curated unit: {curated}")
            sidecar = document_for(target_image, shapes)
            provenance.append(
                (f"image_modes/{Path(image_name).with_suffix('.json').name}",
                 source_note))
        dump_json(sidecar_path, sidecar)

    return provenance


def write_readme(staging: Path) -> None:
    rows = "\n".join(
        f"- `{name}` — {description}" for name, description in MODE_GUIDE)
    (staging / "README.md").write_text(
            README_TEMPLATE.format(mode_rows=rows), encoding="utf-8")


def write_manifest(staging: Path, provenance, commit: str) -> None:
    lines = [
        "anylabeling_test_data — per-file provenance",
        f"upstream: Asher-1/X-AnyLabeling-GGML @ {commit}",
        "provenance values: upstream:<repo path> (verbatim copy), curated "
        "(hand-placed shapes in the official format on an official image)",
        "",
    ]
    for relative, source in sorted(provenance):
        path = staging / relative
        lines.append(f"{sha256_of(path)}  {path.stat().st_size:>9}  "
                     f"{source}  {relative}")
    lines.append("")
    (staging / "MANIFEST.txt").write_text("\n".join(lines), encoding="utf-8")


def write_zip(staging: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    members = sorted(path for path in staging.rglob("*") if path.is_file())
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED,
                         compresslevel=9) as archive:
        for member in members:
            info = zipfile.ZipInfo(
                    f"{ARCHIVE_ROOT}/{member.relative_to(staging).as_posix()}",
                    date_time=ZIP_TIMESTAMP)
            info.external_attr = 0o644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, member.read_bytes(),
                             compresslevel=9)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, default=UPSTREAM_DEFAULT,
                        help="X-AnyLabeling-GGML checkout to bundle")
    parser.add_argument("--output", type=Path, required=True,
                        help="destination zip path")
    args = parser.parse_args()

    if not (args.upstream / "assets" / "demo.json").is_file():
        raise SystemExit(f"not an X-AnyLabeling-GGML checkout: {args.upstream}")

    import tempfile
    with tempfile.TemporaryDirectory(prefix="anylabeling_test_data_") as tmp:
        staging = Path(tmp) / ARCHIVE_ROOT
        staging.mkdir()
        provenance = stage_archive(args.upstream, staging)
        # README/MANIFEST are generated after staging so their own digests
        # are not part of the manifest (they document the members).
        write_readme(staging)
        write_manifest(staging, provenance, upstream_commit(args.upstream))
        write_zip(staging, args.output)

    digest = sha256_of(args.output)
    size = args.output.stat().st_size
    print(f"archive: {args.output}")
    print(f"sha256:  {digest}")
    print(f"size:    {size}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
