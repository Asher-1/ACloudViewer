"""Run the real upstream cuboid methods and compare the C++ geometry probe.

Run with upstream's Python environment (PyQt6 and the source dependencies).
References are generated under --output; no upstream outputs are vendored.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    upstream = args.upstream.resolve()
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    sys.path.insert(0, str(upstream))
    from PyQt6 import QtCore, QtWidgets
    from anylabeling.views.labeling.shape import Shape
    from anylabeling.views.labeling.widgets.canvas import Canvas

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    canvas = Canvas(parent=None)
    rng = random.Random(20261016)

    def coordinates(shape):
        return [[p.x(), p.y()] for p in shape.points]

    cases = []
    for case_index in range(120):
        width, height = rng.uniform(5, 180), rng.uniform(5, 200)
        angle = rng.uniform(-0.5, 0.5) if case_index % 3 == 0 else 0
        u = QtCore.QPointF(math.cos(angle) * width, math.sin(angle) * width)
        v = QtCore.QPointF(-math.sin(angle) * height, math.cos(angle) * height)
        origin = QtCore.QPointF(rng.uniform(-80, 200), rng.uniform(-80, 200))
        front = [origin, origin + u, origin + u + v, origin + v]
        dx = rng.uniform(5, 80) * (-1 if case_index % 2 else 1)
        depth = QtCore.QPointF(dx, rng.uniform(-70, 70))
        shape = Shape(label="car", shape_type="cuboid")
        shape.points = front + [p + depth for p in front]
        # Non-uniform rear offsets also occur after a source face edit.
        if case_index % 4 == 0:
            shape.points[5] += QtCore.QPointF(8, -3)
            shape.points[6] += QtCore.QPointF(8, -3)
        if case_index == 0:
            shape.points[1] = QtCore.QPointF(shape.points[0])
        shape.sync_cuboid_depth_vector()
        canvas.cuboid_min_depth = [0.0, 5.0, 30.0][case_index % 3]
        case = {"points": coordinates(shape),
                "min_depth": canvas.cuboid_min_depth, "steps": []}
        # Every control (including the hidden ones) and each draggable face
        # runs before randomized sequential edits on the evolving geometry.
        operations = list(range(14)) + ["left", "right", "back"]
        operations += [rng.choice(list(range(14)) + ["left", "right", "back"])
                       for _ in range(20)]
        for control in operations:
            if isinstance(control, int):
                position = QtCore.QPointF(shape.get_cuboid_control_point(control))
                position += QtCore.QPointF(rng.uniform(-200, 200),
                                           rng.uniform(-200, 200))
                step = {"control": control,
                        "position": [position.x(), position.y()]}
                canvas.move_cuboid_control(shape, control, position)
            else:
                offset = QtCore.QPointF(rng.uniform(-100, 100),
                                        rng.uniform(-100, 100))
                step = {"face": control, "position": [offset.x(), offset.y()]}
                canvas.move_cuboid_face_by(shape, control, offset)
            step.update(expected=coordinates(shape),
                        controls=shape.get_cuboid_visible_control_indices())
            case["steps"].append(step)
        cases.append(case)

    revision = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    sources = ["anylabeling/views/labeling/shape.py",
               "anylabeling/views/labeling/widgets/canvas.py"]
    hashes = {name: hashlib.sha256((upstream / name).read_bytes()).hexdigest()
              for name in sources}
    payload = {"upstream_revision": revision, "source_sha256": hashes,
               "seed": 20261016, "cases": cases}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"Upstream revision: {revision}", flush=True)
    print(f"Reference: {args.output} sha256="
          f"{hashlib.sha256(args.output.read_bytes()).hexdigest()}", flush=True)
    subprocess.run([str(args.probe.resolve()), "--upstream-reference",
                    str(args.output.resolve())], check=True)
    canvas.close()
    app.processEvents()


if __name__ == "__main__":
    main()
