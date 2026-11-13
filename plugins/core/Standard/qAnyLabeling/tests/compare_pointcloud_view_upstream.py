#!/usr/bin/env python3
"""Exercise actual upstream viewport math without constructing its GL widget."""
import argparse
import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import numpy as np


class Rect:
    def __init__(self, size):
        self.size = size

    def contains(self, point):
        return 0 <= point[0] <= self.size[0] and 0 <= point[1] <= self.size[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.upstream / "anylabeling/views/labeling/pointcloud/viewport.py"
    tree = ast.parse(source.read_text(), filename=str(source))
    viewport = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PointCloudViewport")
    names = {"_fit_indices", "_set_scale", "_basis", "_matrix", "set_view", "_mouse_move", "_wheel"}
    methods = [n for n in viewport.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in methods} == names
    qt = SimpleNamespace(MouseButton=SimpleNamespace(LeftButton=1, RightButton=2, MiddleButton=4),
                         KeyboardModifier=SimpleNamespace(ControlModifier=1))
    namespace = dict(np=np, math=math, QtCore=SimpleNamespace(Qt=qt, QRectF=Rect, QPointF=lambda *p: p,
                     QAbstractAnimation=SimpleNamespace(State=SimpleNamespace(Running=1))))
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), "exec"), namespace)
    cuboid_source = source.with_name("cuboid.py")
    spec = importlib.util.spec_from_file_location("qany_view_cuboid", cuboid_source)
    cuboid = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = cuboid
    spec.loader.exec_module(cuboid)
    detection_source = source.with_name("cuboid_viewport.py")
    detection_class = next(n for n in ast.parse(detection_source.read_text()).body
                           if isinstance(n, ast.ClassDef) and n.name == "CuboidViewport")
    names.update({"unproject", "align_cuboid", "_hit_box", "_update_creation_preview", "move_camera"})
    detection_methods = [n for n in detection_class.body if isinstance(n, ast.FunctionDef)
                         and n.name in {"unproject", "align_cuboid", "_hit_box", "_update_creation_preview", "move_camera"}]
    assert len(detection_methods) == 5
    namespace["Cuboid"] = cuboid.Cuboid
    exec(compile(ast.Module(body=detection_methods, type_ignores=[]), str(detection_source), "exec"), namespace)
    cloud = [[-12, -8, -2], [15, 20, 4], [5, 3, 0]]
    cases = []
    for perspective in (False, True):
        for size in ([900, 600], [300, 900], [1, 1]):
            for view in (None, 0, 1, 2):
                item = dict(points=cloud, size=size, perspective=perspective)
                if view is not None:
                    item["view"] = view
                cases.extend([item, dict(item, pan=[43, -29], orbit=[61, 173], wheel=240)])
    cases.extend([dict(points=[[0, 0, 0]], size=[700, 400]),
                  dict(points=[], size=[700, 400]),
                  dict(points=[[1e7, -1e7, 1e7], [1e7 + 2, -1e7 + 4, 1e7 - 8]],
                       size=[1200, 800], perspective=True),
                  dict(points=cloud, size=[900, 600], orbit=[-20, -1000], wheel=-24000),
                  dict(points=cloud, size=[900, 600], orbit=[20, 1000], wheel=24000)])
    for points in (cloud, [[-200, -150, -3], [200, 150, 8]], [[0, 0, 0]]):
        for wheel in (0, 1200, -1200):
            cases.append(dict(points=points, size=[900, 600], perspective=True, wheel=wheel,
                              limits=[.3 * math.tan(math.radians(25)), 100 * math.tan(math.radians(25))]))
    boxes = [dict(id=i, class_id=1, center=center, size=[4, 2, 2], rotation=[.2, -.3, .7], occluded=False, locked=False)
             for i, center in ((3, [0, 0, 0]), (1, [0, 0, 0]), (2, [4, 4, -4]))]
    for perspective in (False, True):
        for view in (None, 0, 1, 2):
            item = dict(points=cloud, size=[900, 600], perspective=perspective, boxes=boxes, focus=1,
                        hits=[[x, y] for x in range(0, 901, 30) for y in range(0, 601, 30)])
            if view is not None:
                item["view"] = view
            cases.append(item)
    rng = np.random.default_rng(317)
    preview_cloud = rng.uniform(-5, 5, (400, 3)).astype(np.float32).tolist()
    for perspective in (False, True):
        for view in (None, 0, 1, 2):
            item = dict(points=preview_cloud, size=[900, 600], perspective=perspective,
                        previews=[[x, y] for x in range(-30, 932, 30) for y in range(-30, 632, 30)])
            if view is not None:
                item["view"] = view
            cases.extend([item, dict(item, visible=list(range(0, len(preview_cloud), 3))), dict(item, visible=[])])
    # Cover repeated moves, scale clamps and both near-pole pitch clamps.
    keys = ["U", "I", "O", "J", "K", "L", "Up", "Left", "Down", "Right"]
    for perspective in (False, True):
        for view in (None, 0, 1, 2):
            for commands in ([key] for key in keys):
                cases.append(dict(points=cloud, size=[900, 600], perspective=perspective,
                                  **({"view": view} if view is not None else {}), camera_keys=commands))
            for commands in (["Up"] * 30, ["Down"] * 30, ["I"] * 40, ["K"] * 40,
                             rng.choice(keys, size=50).tolist()):
                cases.append(dict(points=cloud, size=[900, 600], perspective=perspective,
                                  **({"view": view} if view is not None else {}), camera_keys=commands))
    expected = []
    for item in cases:
        width, height = item["size"]
        obj = SimpleNamespace(_points=np.asarray(item["points"], dtype=np.float32).reshape(-1, 3),
                              _center=np.zeros(3), _scale=10., _extent=10., _yaw=-45., _pitch=35.,
                              _scale_limits=item.get("limits", (1e-5, 1e12)), _perspective=item.get("perspective", False),
                              _show_axes=False, _gl=None, _tool="browse", _polygon=[], _drag_button=None,
                              orthographic_view=None, _focus_animation=SimpleNamespace(stop=lambda: None, state=lambda: 0),
                              selection_active=False, _animate_camera=lambda *a, **kw: None, setFocus=lambda: None,
                              cuboids=[cuboid.Cuboid(**box) for box in item.get("boxes", [])],
                              _last_mouse=None, _update=lambda **kwargs: None, cancel_selection=lambda: None,
                              creating=True, _error=None, _creation_ray_cache=None,
                              _visible=np.asarray([i in item.get("visible", range(len(item["points"]))) for i in range(len(item["points"]))], dtype=bool),
                              rect=lambda: (width, height),
                              width=lambda: width, height=lambda: height,
                              _physical_size=lambda: (max(1, width), max(1, height)),
                              camera_view_changed=SimpleNamespace(emit=lambda *_: None))
        for name in names:
            setattr(obj, name, namespace[name].__get__(obj))
        obj._fit_indices(None)
        if "view" in item:
            obj.set_view(("top", "front", "side")[item["view"]])
        for action, button in (("pan", 2), ("orbit", 1)):
            if action in item:
                dx, dy = item[action]
                obj._drag_button, obj._last_mouse = button, (0, 0)
                obj._mouse_move(SimpleNamespace(position=lambda: SimpleNamespace(x=lambda: dx, y=lambda: dy)))
        if "wheel" in item:
            obj._wheel(SimpleNamespace(angleDelta=lambda: SimpleNamespace(y=lambda: item["wheel"]), accept=lambda: None))
        for key in item.get("camera_keys", []):
            obj.move_camera(key)
        for box in obj.cuboids:
            if box.id == item.get("focus"):
                obj.align_cuboid(box, fit=True)
        hits = [obj._hit_box(p) or 0 for p in item.get("hits", [])]
        previews = []
        for p in item.get("previews", []):
            result = []
            obj._set_creation_preview = lambda box: result.append(None if box is None else list(box.center))
            obj._update_creation_preview(p)
            previews.append(result[0] if result else None)
        expected.append(dict(matrix=obj._matrix().tolist(), center=obj._center.tolist(), scale=obj._scale,
                             extent=obj._extent, yaw=obj._yaw, pitch=obj._pitch, hits=hits, previews=previews))
    with tempfile.TemporaryDirectory(prefix="qany-view-") as temporary:
        path = Path(temporary) / "cases.json"
        path.write_text(json.dumps(cases))
        actual = json.loads(subprocess.check_output([str(args.probe.resolve()), "--viewport-probe", str(path)], text=True))
    assert len(actual) == len(expected)
    matrix_max = 0.
    state_max = 0.
    preview_max = 0.
    for index, (left, right) in enumerate(zip(actual, expected)):
        for name in right:
            if name == "previews":
                assert len(left[name]) == len(right[name])
                for query, (a, b) in enumerate(zip(left[name], right[name])):
                    assert (a is None) == (b is None), (index, query, a, b)
                    if a is not None:
                        delta = float(np.max(np.abs(np.asarray(a) - b)))
                        assert delta <= 1e-10, (index, query, delta)
                        preview_max = max(preview_max, delta)
                continue
            if name == "hits":
                assert left[name] == right[name], (index, "ray-box intersection", [(cases[index]["hits"][i], a, b) for i, (a, b) in enumerate(zip(left[name], right[name])) if a != b][:8])
                continue
            delta = float(np.max(np.abs(np.asarray(left[name]) - np.asarray(right[name]))))
            # Exact float32 matrix rounding can vary at the final translation
            # dot-product. One ULP is the limit, including large coordinates.
            if name == "matrix":
                tolerance = np.maximum(np.abs(np.spacing(np.asarray(right[name], dtype=np.float32))).astype(float), 1e-7)
                assert np.all(np.abs(np.asarray(left[name]) - right[name]) <= tolerance), (index, name, delta)
                matrix_max = max(matrix_max, delta)
            else:
                assert np.allclose(left[name], right[name], rtol=1e-12, atol=1e-10), (index, name, delta)
                state_max = max(state_max, delta)
    report = dict(verdict="PASS", cases=len(cases), maximum_matrix_delta=matrix_max,
                  camera_motion_commands=sum(len(c.get("camera_keys", [])) for c in cases),
                  ray_box_queries=sum(len(c.get("hits", [])) for c in cases),
                  creation_ray_queries=sum(len(c.get("previews", [])) for c in cases),
                  maximum_creation_center_delta=preview_max,
                  maximum_state_delta=state_max,
                  upstream_revision=subprocess.check_output(["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip(),
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  cuboid_viewport_sha256=hashlib.sha256(detection_source.read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
