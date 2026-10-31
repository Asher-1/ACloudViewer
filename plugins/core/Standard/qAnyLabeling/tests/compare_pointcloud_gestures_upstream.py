#!/usr/bin/env python3
"""Compare real Qt cuboid edit events with executed upstream event handlers.

The reference replaces only the GUI/GL shell. Hit testing, handle geometry,
gesture state, modifiers, release and leave handling execute upstream methods.
The fixture fixes the camera to the selected cuboid's local orthographic axes.
"""
import argparse
import ast
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace

import numpy as np
from PyQt6 import QtCore, QtWidgets

from compare_pointcloud_camera_upstream import upstream_functions


class ReferenceShell:
    def width(self):
        return 600

    def height(self):
        return 400

    def rect(self):
        return QtCore.QRect(0, 0, self.width(), self.height())

    def project(self, points):
        local = (np.asarray(points) - self.origin) @ self.orientation
        return np.column_stack((300 + local[:, self.axes[0]] * self.pixels,
                                200 - local[:, self.axes[1]] * self.pixels))

    @property
    def selected_cuboid(self):
        return self.current

    def _update(self, **kwargs):
        pass

    def _mouse_move(self, event):
        pass

    def _mouse_release(self, event):
        pass

    def _set_creation_preview(self, box):
        pass

    def eventFilter(self, watched, event):
        return False


def source_class(root):
    path = root / "anylabeling/views/labeling/pointcloud/cuboid_viewport.py"
    tree = ast.parse(path.read_text())
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                 and n.name == "CuboidViewport")
    names = {"_basis", "unproject", "_handles", "_hit_handle", "_hit_box", "_mouse_press",
             "_mouse_move", "_mouse_release", "_finish_edit", "_update_handle_hover",
             "eventFilter"}
    methods = [n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in methods} == names
    # Preserve zero-argument super() by compiling a class, rather than detached
    # functions. The base supplies only inert GUI/navigation methods.
    owner.bases = [ast.Name(id="ReferenceShell", ctx=ast.Load())]
    owner.body = methods
    namespace = dict(np=np, math=math, replace=replace, QtCore=QtCore,
                     QtWidgets=QtWidgets, ReferenceShell=ReferenceShell,
                     VIEW_AXES={"top": (0, 1), "side": (0, 2), "front": (1, 2)})
    exec(compile(ast.fix_missing_locations(ast.Module(body=[owner], type_ignores=[])),
                 str(path), "exec"), namespace)
    return namespace["CuboidViewport"], path


def reference(cls, Cuboid, case):
    ref = cls()
    box = Cuboid(**case["box"])
    ref.current = box
    ref.cuboids = (box,)
    ref.axes = case["h"], case["v"]
    ref.origin, ref.orientation = np.array(box.center), box.matrix
    ref._orientation = ref.orientation
    ref.pixels = min(500 / box.size[case["h"]], 300 / box.size[case["v"]])
    ref._center = ref.origin
    ref._scale = 200 / ref.pixels
    ref._extent = 100
    ref._perspective = False
    ref.orthographic_view = {(0, 1): "top", (0, 2): "side", (1, 2): "front"}[ref.axes]
    ref.detection_enabled, ref.creating, ref._error = True, False, ""
    ref._tool = "browse"
    ref._points = np.zeros((0 if case.get("empty") else 1, 3))
    ref._gesture = ref._gesture_button = ref._hover_handle = None
    ref._creation_start = ref._navigation_button = None
    ref._focus_animation = SimpleNamespace(stop=lambda: None)
    ref.cursor = 0
    ref._gl = SimpleNamespace(setFocus=lambda: None,
                              setCursor=lambda shape: setattr(ref, "cursor", shape.value))
    ref.selection_started = SimpleNamespace(emit=lambda: None)
    ref.cuboid_selected = SimpleNamespace(emit=lambda value: None)
    ref.cuboid_preview = SimpleNamespace(emit=lambda value: setattr(ref, "current", value))
    ref.edits = []

    def commit(value):
        if value != ref.cuboids[0]:
            ref.edits.append(asdict(value))
            ref.cuboids = (value,)
        # The real owner refreshes the other views after a commit.
        ref._update_handle_hover(None)

    ref.cuboid_edited = SimpleNamespace(emit=commit)
    return ref


def event(kind, point=(300, 200), button=0, buttons=0, shift=False):
    return dict(kind=kind, point=[float(p) for p in point], button=button,
                buttons=buttons, shift=shift)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, path = source_class(args.upstream)
    _, Cuboid = upstream_functions(args.upstream)
    cases = []
    for h, v in [(0, 1), (0, 2), (1, 2)]:
        seed = dict(h=h, v=v, box=dict(id=1, class_id=1, center=[2, -1, 4],
                    size=[4, 3, 2], rotation=[.2, -.3, .4], locked=False, occluded=False))
        ref = reference(source, Cuboid, seed)
        handles, rotation = ref._handles(ref.current)
        prefix = f"{h}-{v}"

        def add(name, events, **extra):
            cases.append(dict(seed, case=f"{prefix}/{name}", events=events, **extra))

        for index, (_, point) in enumerate(handles):
            target = point + np.array([12, -8])
            add(f"handle-{index}", [event("move", point), event("press", point, 1, 1),
                                   event("release", target, 1)])
        for angle in [22, 23, -22, -23, 0, 7.5, 22.5, 37.5, -7.5, -22.5, -37.5]:
            offset = rotation - np.array([300, 200])
            theta = math.radians(angle)
            target = np.array([300, 200]) + np.array([
                math.cos(theta) * offset[0] + math.sin(theta) * offset[1],
                -math.sin(theta) * offset[0] + math.cos(theta) * offset[1]])
            # A zero-degree Shift click checks a no-op without an undo entry.
            add(f"rotation-shift-{angle}", [event("move", rotation),
                event("press", rotation, 1, 1), event("release", target, 1, shift=True)])
        target = np.array([328, 216])
        add("release-without-move", [event("press", button=1, buttons=1),
                                     event("release", target, 1)])
        add("release-newer-than-move", [event("press", button=1, buttons=1),
            event("move", [308, 202], buttons=1), event("release", target, 1)])
        add("other-button-release", [event("press", button=1, buttons=1),
            event("move", [308, 202], buttons=1), event("release", [310, 208], 2, 1),
            event("release", target, 1)])
        add("other-button-press", [event("press", button=1, buttons=1),
            event("press", [310, 208], 2, 3), event("release", target, 1)])
        add("lost-button", [event("press", button=1, buttons=1),
            event("move", [308, 202], buttons=1), event("move", target),
            event("release", [350, 200], 1)])
        add("outside-move", [event("press", button=1, buttons=1),
            event("move", [308, 202], buttons=1), event("move", [620, 200], buttons=1),
            event("release", target, 1)])
        add("outside-release", [event("press", button=1, buttons=1),
            event("move", [308, 202], buttons=1), event("release", [620, 200], 1)])
        add("leave", [event("press", button=2, buttons=2),
            event("move", [308, 202], buttons=2), event("leave"),
            event("release", target, 2)])
        add("empty-cloud", [event("press", button=1, buttons=1),
                              event("release", target, 1)], empty=True)
        locked = dict(seed["box"], locked=True)
        cases.append(dict(seed, case=f"{prefix}/locked", box=locked,
                          events=[event("move", handles[0][1]),
                                  event("press", button=1, buttons=1), event("release", target, 1)]))

    with tempfile.TemporaryDirectory(prefix="qany-cuboid-gestures-") as temporary:
        input_path = Path(temporary) / "cases.json"
        input_path.write_text(json.dumps(cases))
        actual = json.loads(subprocess.check_output(
            [str(args.probe.resolve()), "--cuboid-gesture-probe", str(input_path)], text=True))
    rows = []
    for case, got in zip(cases, actual, strict=True):
        ref = reference(source, Cuboid, case)
        cursors = []
        for item in case["events"]:
            if item["kind"] == "leave":
                ref.eventFilter(None, QtCore.QEvent(QtCore.QEvent.Type.Leave))
            else:
                kind = item["kind"]
                kind_type = {"press": QtCore.QEvent.Type.MouseButtonPress,
                             "move": QtCore.QEvent.Type.MouseMove,
                             "release": QtCore.QEvent.Type.MouseButtonRelease}[kind]
                e = SimpleNamespace(position=lambda: QtCore.QPointF(*item["point"]),
                    type=lambda: kind_type,
                    button=lambda: QtCore.Qt.MouseButton(item["button"]),
                    buttons=lambda: QtCore.Qt.MouseButton(item["buttons"]),
                    modifiers=lambda: QtCore.Qt.KeyboardModifier.ShiftModifier if item["shift"]
                                      else QtCore.Qt.KeyboardModifier.NoModifier)
                getattr(ref, "_mouse_" + kind)(e)
            cursors.append(ref.cursor)
        assert got["case"] == case["case"]
        assert got["cursors"] == cursors, (case["case"], got["cursors"], cursors)
        assert len(got["edits"]) == len(ref.edits), (case["case"], got, ref.edits)
        maximum = 0.0
        for changed, expected in zip(got["edits"], ref.edits, strict=True):
            for key in ("center", "size", "rotation"):
                np.testing.assert_allclose(changed[key], expected[key], atol=1e-10, rtol=0,
                                           err_msg=f"{case['case']}: {key}")
                maximum = max(maximum, float(np.max(np.abs(np.array(changed[key]) - expected[key]))))
        rows.append(dict(case=case["case"], commits=len(ref.edits), maximum_delta=maximum))
    report = dict(verdict="PASS", rows=rows, coordinate_tolerance=1e-10,
        scope="Orthographic handle hit/hover and edit gesture events; excludes full-window visuals and live cross-view previews",
        reference_execution="Actual upstream CuboidViewport methods and Cuboid geometry in a fixed-camera GUI shell",
        upstream_revision=subprocess.check_output(["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip(),
        source_sha256={str(p.relative_to(args.upstream)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (path, path.parent / "cuboid.py")},
        probe_sha256=hashlib.sha256(args.probe.read_bytes()).hexdigest(),
        comparison_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS: {len(rows)} upstream gesture cases")


if __name__ == "__main__":
    main()
