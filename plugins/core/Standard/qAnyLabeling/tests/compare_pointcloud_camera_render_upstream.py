#!/usr/bin/env python3
"""Compare camera image/point/cuboid pixels with executed upstream Qt code."""
import argparse
import ast
import hashlib
import json
import shutil
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from compare_pointcloud_camera_upstream import calibration, upstream_functions


def load_source(root):
    folder = root / "anylabeling/views/labeling"
    camera = folder / "pointcloud/camera.py"
    tree = ast.parse(camera.read_text())
    namespace, Cuboid = upstream_functions(root)
    namespace.update(QtCore=QtCore, QtGui=QtGui, QtWidgets=QtWidgets,
                     get_icon=lambda *args: QtGui.QIcon())
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef)
             and n.name in ("_CameraCuboidsItem", "CameraView")]
    panel = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                 and n.name == "CameraPanel")
    nodes += [n for n in panel.body if isinstance(n, ast.FunctionDef)
              and n.name == "refresh"]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(camera), "exec"), namespace)
    dialog = folder / "widgets/pointcloud_dialog.py"
    methods = [n for c in ast.parse(dialog.read_text()).body if isinstance(c, ast.ClassDef)
               for n in c.body if isinstance(n, ast.FunctionDef)
               and n.name in ("_class_palette", "_intensity_colors", "_display_colors")]
    assert len(methods) == 3 and len(nodes) == 3
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(dialog), "exec"), namespace)
    detection = folder / "pointcloud/detection.py"
    preview = [n for c in ast.parse(detection.read_text()).body if isinstance(c, ast.ClassDef)
               for n in c.body if isinstance(n, ast.FunctionDef) and n.name == "preview"]
    assert len(preview) == 1
    exec(compile(ast.Module(body=preview, type_ignores=[]), str(detection), "exec"), namespace)
    namespace["detection_preview"] = namespace["preview"]
    return namespace, Cuboid, [camera, dialog, detection]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    app = QtWidgets.QApplication([])
    source, Cuboid, paths = load_source(args.upstream)
    points = np.array([[0, 0, 5, 0], [0, 0, 10, 1], [1, 0, 5, .2],
                       [-1, 0, 5, .7], [0, 1, 5, .5], [2, 1, 7, .9]], np.float32)
    labels = np.array([1, 2, (9 << 16) | 1, 2, (3 << 16) | 2, 0], np.uint32)
    rgb = np.array([[241, 13, 25], [3, 217, 60], [10, 45, 203],
                    [89, 23, 77], [13, 67, 211], [230, 240, 250]], np.uint8)
    palette = {0: "#808080", 1: "#039be5", 2: "#feab01"}
    config = calibration("opencv5")
    boxes = [dict(id=1, class_id=1, center=[0, 0, 5], size=[2, 2, 2],
                  rotation=[.1, -.2, .4], occluded=True, locked=False),
             dict(id=2, class_id=2, center=[2, 0, 6], size=[2, 1, 2],
                  rotation=[0, 0, -.2], occluded=False, locked=False)]
    preview = dict(id=3, class_id=1, center=[-1, 0, 5], size=[1, 1, 1],
                   rotation=[0, 0, .15], occluded=False, locked=False)
    rows = []
    artifacts = args.output.with_suffix("")
    artifacts.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="qany-camera-render-") as temporary:
        temporary = Path(temporary)
        image_path, calibration_path = temporary / "frame.png", temporary / "calibration.json"
        calibration_path.write_text(json.dumps(config))
        image = np.empty((80, 100, 4), np.uint8)
        y, x = np.indices(image.shape[:2])
        image[:, :, 0] = x * 13 % 256
        image[:, :, 1] = y * 17 % 256
        image[:, :, 2] = (x + y) * 11 % 256
        image[:, :, 3] = 255
        photo = QtGui.QImage(image.data, 100, 80, image.strides[0],
                            QtGui.QImage.Format.Format_RGBA8888).copy()
        assert photo.save(str(image_path))
        fixture = dict(points=points.tolist(), labels=labels.tolist(), rgb=rgb.tolist(),
                       palette=palette, image=str(image_path), calibration=str(calibration_path),
                       cuboids=boxes, selected=2, preview=preview)
        cases = []
        for mode in range(4):
            for scale in (1, 1.75):
                cases.append(dict(fixture, mode=mode, scale=scale, size=3, opacity=65,
                                  hidden_instances=[0, 9]))
        for size, opacity in ((1, 100), (3, 35), (7, 65)):
            cases.append(dict(fixture, mode=0, scale=2, size=size, opacity=opacity,
                              depth=True, hidden_instances=[]))
        for kind, changed in [
                ("move", dict(boxes[0], center=[1, .5, 5])),
                ("resize", dict(boxes[0], size=[3, 1, 2])),
                ("rotate", dict(boxes[0], rotation=[.4, -.2, .8])),
                ("cancel", dict(boxes[0], center=[1, .5, 5]))]:
            cases.append(dict(fixture, mode=0, scale=1.75, size=3, opacity=65,
                              depth=False, hidden_instances=[], edit_preview=changed,
                              clear_edit=kind == "cancel", edit_kind=kind))
        for index, case in enumerate(cases):
            case["output"] = str(temporary / f"native-{index}.png")
        input_path = temporary / "input.json"
        input_path.write_text(json.dumps(cases))
        subprocess.run([str(args.probe.resolve()), "--camera-render-probe", str(input_path)], check=True)
        parameters = source["load_calibration"](calibration_path)
        indices, pixels = source["project_points"](points, parameters, 100, 80)
        for index, case in enumerate(cases):
            frame = SimpleNamespace(path=Path("frame.bin"), points=points, rgb=rgb)
            document = SimpleNamespace(frame=frame, labels=labels,
                                       semantic_view=(labels & 65535).astype(np.uint16),
                                       instance_view=labels >> 16)
            reference = SimpleNamespace(document=document, _palette_classes=None,
                _intensity_range=None, tr=lambda text: text, _set_render_legend=lambda *args: None,
                color_mode=SimpleNamespace(currentData=lambda: ("semantic", "intensity", "rgb", "instance")[case["mode"]]),
                class_definitions={"segmentation": [SimpleNamespace(id=k, color=v) for k, v in palette.items()]})
            for name in ("_class_palette", "_intensity_colors", "_display_colors"):
                setattr(reference, name, source[name].__get__(reference))
            colors = reference._display_colors(np.arange(len(points)), None)
            visible = np.ones(len(points), bool)
            for instance in case["hidden_instances"]:
                if instance:
                    visible &= (labels >> 16) != instance
            view = source["CameraView"]()
            view.setSceneRect(QtCore.QRectF(photo.rect()))
            view.image_item.setPixmap(QtGui.QPixmap.fromImage(photo))
            view.overlay_item.setOpacity(case["opacity"] / 100)
            frame_boxes = tuple(Cuboid(**b) for b in boxes)
            if "edit_preview" in case:
                # Execute the source controller's replacement workflow, then
                # render its resulting boxes through the actual camera item.
                proxy = SimpleNamespace(selected_cuboid=None, orthographic_view=None)
                proxy.set_cuboids = lambda values, selected: setattr(proxy, "cuboids", values)
                controller = SimpleNamespace(document=SimpleNamespace(cuboids=frame_boxes),
                    enabled=True, hidden_ids=set(), views=[proxy], selected_id=2,
                    window=SimpleNamespace(class_definitions={"detection": [
                        SimpleNamespace(id=k, color=v, name=str(k)) for k, v in palette.items()]}))
                source["detection_preview"](controller, Cuboid(**case["edit_preview"]))
                if case["clear_edit"]:
                    source["detection_preview"](controller, None)
                frame_boxes = proxy.cuboids
            view.cuboid_item.set_cuboids(frame_boxes, 2, palette,
                                        parameters, 100, 80, Cuboid(**preview))
            overlay = SimpleNamespace(isHidden=lambda: False,
                workspace=SimpleNamespace(document=document, _visible=visible,
                                          viewport=SimpleNamespace(_colors=colors)),
                _key=document, _image=photo, _indices=indices, _pixels=pixels,
                calibration=parameters, view=view, refresh_cuboids=lambda: None,
                overlay_action=SimpleNamespace(isEnabled=lambda: True, isChecked=lambda: True),
                overlay_color_buttons={"depth": SimpleNamespace(isChecked=lambda: case.get("depth", False))},
                overlay_size=SimpleNamespace(value=lambda: case["size"]), tr=lambda text: text,
                depth_legend=SimpleNamespace(setText=lambda *args: None, setToolTip=lambda *args: None))
            source["_update_depth_colors"](overlay, points)
            source["refresh"](overlay)
            size = QtCore.QSize(int(100 * case["scale"]), int(80 * case["scale"]))
            rendered = QtGui.QImage(size, QtGui.QImage.Format.Format_ARGB32)
            rendered.fill(QtCore.Qt.GlobalColor.black)
            painter = QtGui.QPainter(rendered)
            view.scene().render(painter, QtCore.QRectF(QtCore.QPointF(), QtCore.QSizeF(size)),
                                QtCore.QRectF(photo.rect()))
            painter.end()
            expected_path = temporary / f"source-{index}.png"
            assert rendered.save(str(expected_path))
            # Decoding both saved files avoids QImage byte-order/stride assumptions.
            def pixels_from(path):
                loaded = QtGui.QImage(str(path)).convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
                return np.frombuffer(loaded.constBits().asstring(loaded.sizeInBytes()), np.uint8).reshape(loaded.height(), loaded.width(), 4).copy()
            actual, expected = pixels_from(case["output"]), pixels_from(expected_path)
            delta = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
            shutil.copy2(case["output"], artifacts / f"native-{index}.png")
            shutil.copy2(expected_path, artifacts / f"source-{index}.png")
            rows.append(dict(case=index, mode=case["mode"], scale=case["scale"],
                             edit_kind=case.get("edit_kind"),
                             depth=case.get("depth", False), maximum_delta=int(delta.max()),
                             differing_channels=int(np.count_nonzero(delta))))
            view.deleteLater()
    report = dict(verdict="PASS" if all(r["differing_channels"] == 0 for r in rows) else "FAIL",
                  scope="Camera image, point overlay and projected cuboid/preview scene pixels; excludes window chrome/fonts",
                  reference_execution="Actual upstream CameraView, CameraPanel.refresh, cuboid item and display-color functions",
                  upstream_revision=subprocess.check_output(["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip(),
                  probe_sha256=hashlib.sha256(args.probe.read_bytes()).hexdigest(),
                  comparison_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  source_sha256={str(p.relative_to(args.upstream)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}, rows=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report["verdict"] != "PASS"


if __name__ == "__main__":
    raise SystemExit(main())
