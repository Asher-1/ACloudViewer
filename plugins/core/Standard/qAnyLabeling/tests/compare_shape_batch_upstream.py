"""Compare batch edits with the actual upstream ShapeModifyDialog methods."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    upstream = args.upstream.resolve()
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    sys.path.insert(0, str(upstream))
    from PyQt6.QtGui import QImage, QColor
    from PyQt6.QtWidgets import QApplication, QWidget, QListWidget
    from anylabeling.views.labeling.shape import Shape
    from anylabeling.views.labeling.widgets import shape_dialog

    app = QApplication.instance() or QApplication([])

    class NoPopup:
        def __init__(self, *args, **kwargs):
            pass

        def show_popup(self, *args, **kwargs):
            pass

    # Keep the real batch methods, replacing only the transient success toast.
    shape_dialog.Popup = NoPopup
    methods = ["delete_annotations_in_range", "delete_images_in_range",
               "remove_selected_shapes_in_range", "add_selected_shapes_to_range"]

    def shape(label="box", dx=0, **extra):
        return Shape().load_from_dict({"label": label,
            "shape_type": "rectangle", "points": [[dx, 0], [80 + dx, 60]],
            "custom": "preserve", **extra}, close=False).to_dict()

    cases = []
    with tempfile.TemporaryDirectory(prefix="qany-shape-batch-") as temporary:
        for operation in range(4):
            for alternate in (False, True):
                roots = [Path(temporary) / f"{operation}-{alternate}-{side}"
                         for side in ("python", "cpp")]
                selected = [shape(), shape("boundary"), shape("outside", dx=2)]
                for root in roots:
                    (root / "images").mkdir(parents=True)
                    if alternate:
                        (root / "labels").mkdir()
                    for i in range(4):
                        image = QImage(80, 60, QImage.Format.Format_RGB32)
                        image.fill(QColor("white"))
                        assert image.save(str(root / "images" / f"{i}.png"))
                        shapes = [shape(dx=0.9), shape(locked=True),
                                  shape("other"), shape(dx=1.01)]
                        if i == 1:
                            shapes = [shape()]
                        elif i == 2:
                            shapes = []
                        if i < 3:
                            directory = root / ("labels" if alternate else "images")
                            (directory / f"{i}.json").write_text(json.dumps({
                                "imagePath": f"{i}.png", "shapes": shapes,
                                "custom_document": 29}), encoding="utf-8")
                parent = QWidget()
                parent.filename = str(roots[0] / "images/0.png")
                parent.fn_to_index = {parent.filename: 0}
                parent.output_dir = str(roots[0] / "labels") if alternate else None
                parent.canvas = SimpleNamespace(selected_shapes=[
                    Shape().load_from_dict(value, close=False) for value in selected])
                parent.file_list_widget = QListWidget(parent)
                image_files = [str(roots[0] / "images" / f"{i}.png") for i in range(4)]
                parent.file_list_widget.addItems(image_files)
                parent.reset_state = lambda: None
                parent.import_image_folder = lambda path: None
                parent.image_list = []
                dialog = shape_dialog.ShapeModifyDialog(parent)
                # Delete only the middle frames; add also covers a new sidecar.
                first, last = (2, 3) if operation < 2 else (1, 4)
                getattr(dialog, methods[operation])(first, last)
                request = roots[1] / "request.json"
                request.write_text(json.dumps({
                    "images": [str(roots[1] / "images" / f"{i}.png") for i in range(4)],
                    "output": str(roots[1] / "labels") if alternate else "",
                    "first": first, "last": last, "operation": operation,
                    "selected": selected}), encoding="utf-8")
                result = json.loads(subprocess.check_output(
                    [str(args.probe.resolve()), str(request)], text=True))

                def snapshot(root):
                    result = {}
                    for directory in ("images", "labels", "_delete_"):
                        for path in sorted((root / directory).glob("*")):
                            relative = str(path.relative_to(root))
                            result[relative] = (json.loads(path.read_text(encoding="utf-8"))
                                if path.suffix == ".json" else hashlib.sha256(path.read_bytes()).hexdigest())
                    return result

                expected, actual = snapshot(roots[0]), snapshot(roots[1])
                assert actual == expected, (methods[operation], alternate, expected, actual)
                cases.append({"operation": methods[operation], "alternate_output": alternate,
                              "result": result, "verdict": "PASS"})
                dialog.close()
                parent.close()
                app.processEvents()
    sources = ["anylabeling/views/labeling/widgets/shape_dialog.py",
               "anylabeling/views/labeling/shape.py"]
    report = {"upstream_revision": subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {source: hashlib.sha256((upstream / source).read_bytes()).hexdigest()
                          for source in sources},
        "cases": cases, "verdict": "PASS"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Actual upstream ShapeModifyDialog: {len(cases)} cases PASS; evidence: {args.output}")


if __name__ == "__main__":
    main()
