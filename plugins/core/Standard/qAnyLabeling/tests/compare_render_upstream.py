"""Compare real upstream Canvas rendering with the C++ visualization export.

Use the upstream Python environment. All references and screenshots are
generated under --output. Text/attribute regions and shape interiors are
checked with an explicit one-pixel font-metric tolerance across Qt major
versions. Geometry-only images must match exactly.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ["QT_FONT_DPI"] = "96"
    sys.path.insert(0, str(args.upstream.resolve()))
    from PyQt6 import QtCore, QtGui, QtWidgets
    from anylabeling.views.labeling.shape import Shape
    from anylabeling.views.labeling.widgets.canvas import Canvas
    from anylabeling.views.labeling.utils.colormap import label_colormap
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    args.output.mkdir(parents=True, exist_ok=True)
    shapes = [
        ("rectangle", [[60, 100], [200, 100], [200, 220], [60, 220]]),
        ("polygon", [[60, 100], [200, 120], [160, 220], [60, 200]]),
        ("circle", [[140, 160], [180, 160]]),
        ("line", [[60, 100], [200, 220]]),
        ("point", [[100, 160]]),
        ("rotation", [[60, 100], [200, 100], [200, 220], [60, 220]]),
    ]
    cases = []
    for kind, points in shapes:
        for profile in ("label", "description", "attributes", "no_label",
                        "difficult-no_label", "no_masks-no_label"):
            extra = {"label": "car", "shape_type": kind, "points": points,
                     "group_id": 4, "score": 0.87, "flags": {}}
            if profile == "description": extra["description"] = "parked vehicle"
            if profile == "attributes": extra["attributes"] = {"color": "red", "occluded": True}
            if profile == "difficult-no_label": extra["difficult"] = True
            cases.append({"name": f"{kind}-{profile}", "shapes": [extra],
                          "show_labels": not profile.endswith("no_label"),
                          "show_masks": profile != "no_masks-no_label"})
    # These cases catch the first point mistaken for the anchor, clipped edge
    # labels, ineffective toggles and leaders drawn over labels.
    # The original 36 cases and all pixel/font tolerances are unchanged, and the
    # reference still performs the real upstream drawing.
    for name, kind, points, overrides in [
        ("polygon-top-anchor", "polygon", [[200, 220], [60, 200], [60, 100], [200, 120]], {}),
        ("point-right-edge", "point", [[620, 140]], {}),
        ("point-top-edge", "point", [[80, 8]], {}),
        ("point-bottom-edge", "point", [[620, 475]], {}),
        ("rectangle-description-only", "rectangle", shapes[0][1],
         {"show_labels": False, "description": "parked vehicle"}),
        ("rectangle-scores-off", "rectangle", shapes[0][1], {"show_scores": False}),
        ("point-prompt-no-caption", "point", [[100, 160]], {"label": "AUTOLABEL_OBJECT"}),
    ]:
        data = {"label": overrides.get("label", "car"), "shape_type": kind,
                "points": points, "group_id": 4, "score": 0.87, "flags": {}}
        if "description" in overrides:
            data["description"] = overrides["description"]
        cases.append({"name": name, "shapes": [data], "show_masks": False,
                      "show_labels": overrides.get("show_labels", True),
                      "show_scores": overrides.get("show_scores", True)})
    cases.append({"name": "point-leaders-below-labels", "show_masks": False,
                  "show_labels": True, "shapes": [
                      {"label": "car", "shape_type": "point", "points": [point],
                       "group_id": 4, "score": 0.87, "flags": {}}
                      for point in ([300, 220], [330, 190])]})
    request = args.output / "cases.json"
    request.write_text(json.dumps({"cases": cases}, indent=2))
    cpp = args.output / "cpp"
    subprocess.run([str(args.probe.resolve()), str(request.resolve()), str(cpp.resolve())], check=True)
    palette = label_colormap()
    actual_palette = json.loads((cpp / "palette.json").read_text())
    errors = []
    if actual_palette != [palette[i % 256].tolist() for i in range(-256, 512)]:
        errors.append("label palette mismatch")
    rows = []
    for case in cases:
        canvas = Canvas(parent=None, label_font_size=8)
        canvas.show_attributes = True
        canvas.show_degrees = False
        source_shapes = []
        for data in case["shapes"]:
            shape = Shape(label=data["label"], shape_type=data["shape_type"],
                          score=data.get("score"), group_id=data.get("group_id"),
                          difficult=data.get("difficult", False),
                          description=data.get("description"), attributes=data.get("attributes", {}))
            shape.points = [QtCore.QPointF(*point) for point in data["points"]]
            shape.line_color = QtGui.QColor(*palette[1].tolist(), 255)
            shape.fill_color = QtGui.QColor(*palette[1].tolist(), 128)
            shape.vertex_fill_color = shape.line_color
            shape.line_width = 4.0
            shape.point_size = 10
            if data["shape_type"] in ("polygon", "rectangle", "rotation"):
                shape.close()
            source_shapes.append(shape)
        image = QtGui.QPixmap(640, 480)
        image.fill(QtCore.Qt.GlobalColor.white)
        reference = canvas.render_visualization(image, source_shapes,
                                                 show_labels=case["show_labels"],
                                                 show_scores=case.get("show_scores", True),
                                                 show_texts=case.get("show_texts", True),
                                                 show_masks=case["show_masks"])
        reference_path = args.output / (case["name"] + ".png")
        reference.save(str(reference_path))
        expected = np.asarray(Image.open(reference_path).convert("RGB"), dtype=np.int16)
        actual = np.asarray(Image.open(cpp / (case["name"] + ".png")).convert("RGB"), dtype=np.int16)
        delta = np.abs(expected - actual)
        Image.fromarray(np.uint8(np.minimum(delta * 4, 255))).save(args.output / (case["name"] + "-diff.png"))
        ink = np.any(expected != 255, axis=2) | np.any(actual != 255, axis=2)
        yy, xx = np.where(ink)
        # Offscreen Qt 5 and Qt 6 Arial metrics differ by one pixel in
        # height/ascent. Permit only that spatial difference for text; every
        # geometry-only image is held to an exact-pixel gate.
        crop = (slice(max(0, yy.min()-2), min(480, yy.max()+3)),
                slice(max(0, xx.min()-2), min(640, xx.max()+3)))
        a, b = expected[crop], actual[crop]
        def nearest_error(first, second):
            padded = np.pad(second, ((1, 1), (1, 1), (0, 0)), mode="edge")
            return np.minimum.reduce([
                np.abs(first - padded[y:y+first.shape[0], x:x+first.shape[1]]).mean(axis=2)
                for y in range(3) for x in range(3)]).mean()
        tolerant_mae = float(max(nearest_error(a, b), nearest_error(b, a)))
        geometry_only = case["name"].endswith("-no_label")
        if geometry_only and np.any(delta):
            errors.append(f"{case['name']}: geometry pixels differ")
        if not geometry_only and (tolerant_mae > 1.0 or delta.mean() > 0.25):
            errors.append(f"{case['name']}: text/color/position mismatch ({tolerant_mae})")
        rows.append({"name": case["name"], "full_image_mae": float(delta.mean()),
                     "ink_region_mae_1px": tolerant_mae,
                     "geometry_exact": bool(not np.any(delta)) if geometry_only else None})
        canvas.deleteLater()
    report = {"revision": subprocess.check_output(["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip(),
              "source_hashes": {str(path.relative_to(args.upstream)): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in [args.upstream / "anylabeling/views/labeling/widgets/canvas.py",
                                             args.upstream / "anylabeling/views/labeling/shape.py",
                                             args.upstream / "anylabeling/views/labeling/utils/colormap.py"]},
              "palette_cases": len(actual_palette), "cases": rows, "errors": errors,
              "native_runtime": json.loads((cpp / "qt-runtime.json").read_text()),
              "reference_qt_version": QtCore.qVersion(),
              "probe_sha256": hashlib.sha256(args.probe.read_bytes()).hexdigest(),
              "verdict": "PASS" if not errors else "FAIL"}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
