#!/usr/bin/env python3
"""End-to-end field-by-field tracker parity: official ultralytics runtime vs
the qYOLO C++ tracker, on the SAME real detector output.

Pipeline:
  1. Deterministic synthetic frames (moving boxes on a noise field).
  2. Official YOLO detector (yolo26n.pt, predict mode) -> real per-frame
     detections (xywh center rows, row-order idx), identical bytes on both
     sides.
  3. Track those detections twice: through the official trackers
     (ultralytics 8.4.x TRACKER_MAP, installed package) and through the C++
     qYOLO tracker via track_parity_harness.
  4. Field-by-field diff: id/cls/idx must match EXACTLY (rows aligned by
     track id per frame); x1/y1/w/h within TOL_COORD px; score within
     TOL_SCORE.

Usage: python3 track_parity_check.py [--harness <bin>] [--frames N]
Not a ctest — needs the ultralytics package and a detector checkpoint.
"""
import argparse
import io
import subprocess
import sys

import numpy as np

TOL_COORD = 5e-3  # px; float32 storage + Kalman float64 vs float32 rounding
TOL_SCORE = 5e-4

TYPES = ("bytetrack", "botsort", "ocsort", "deepocsort", "fasttrack",
         "tracktrack")


def make_frames(n, w=640, h=480):
    """Deterministic frames: real-object sprites (crops taken from bus.jpg,
    detected once) pasted at moving positions over a seeded noise field.
    Real patches keep the detector confident; the motion script exercises
    Lost/Refind/occlusion/flicker paths."""
    import os
    import cv2
    from PIL import Image
    bus = os.path.join(os.path.dirname(__file__), "../../../..")
    bus = os.path.abspath(os.path.join(
        bus, "../dl/ultralytics-ggml/ultralytics/assets/bus.jpg"))
    if not os.path.exists(bus):
        bus = "/home/ludahai/develop/code/github/dl/ultralytics-ggml/"
        bus += "ultralytics/assets/bus.jpg"
    src = np.asarray(Image.open(bus).convert("RGB"))

    from ultralytics import YOLO
    det_model = YOLO("yolo26n.pt")
    r = det_model.predict(src, conf=0.4, verbose=False, device="cpu")[0]
    d = r.boxes.cpu().numpy()
    order = np.argsort(-d.conf)[:2]
    sprites = []
    for i in order:
        x1, y1, x2, y2 = [int(v) for v in d.xyxy[i]]
        sprite = src[y1:y2, x1:x2]
        # Normalise sprite size so the paste always fits the canvas.
        scale = 110.0 / max(sprite.shape[:2])
        if scale < 1.0:
            sprite = cv2.resize(sprite, (max(1, int(sprite.shape[1] * scale)),
                                         max(1, int(sprite.shape[0] * scale))))
        sprites.append((int(d.cls[i]), sprite))
    assert len(sprites) == 2, "expected 2 detected sprites on bus.jpg"

    rng = np.random.default_rng(42)
    frames = []
    for f in range(n):
        img = rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)
        # sprite A: horizontal drive, gap at f%7==5 (Lost/Refind)
        if f % 7 != 5:
            sa = sprites[0][1]
            x = min(int(30 + f * 8.0), w - sa.shape[1] - 1)
            img[40:40 + sa.shape[0], x:x + sa.shape[1]] = sa
        # sprite B: vertical drive with occlusion window f=13..18
        if f < 12 or f > 18:
            sb = sprites[1][1]
            y = min(int(80 + f * 5.0), h - sb.shape[0] - 1)
            img[y:y + sb.shape[0], 420:420 + sb.shape[1]] = sb
        frames.append(np.ascontiguousarray(img))
    return frames


def detect_rows(model, frames):
    """Official detector rows: (cx, cy, w, h, conf, cls, idx) per frame."""
    out = []
    for img in frames:
        r = model.predict(img, conf=0.25, iou=0.7, verbose=False,
                          device="cpu")[0]
        det = r.boxes.cpu().numpy()
        xywh = det.xywh if not hasattr(det, "xywhr") else det.xywhr
        rows = [(float(xywh[i, 0]), float(xywh[i, 1]), float(xywh[i, 2]),
                 float(xywh[i, 3]), float(det.conf[i]), int(det.cls[i]), i)
                for i in range(len(xywh))]
        out.append(rows)
    return out


def official_track(type_name, rows_per_frame):
    from ultralytics.trackers.track import TRACKER_MAP
    from ultralytics.utils import IterableSimpleNamespace, YAML
    yaml = (f"/home/ludahai/.pyenv/versions/3.11/lib/python3.11/site-packages/"
            f"ultralytics/cfg/trackers/{type_name}.yaml")
    cfg = IterableSimpleNamespace(**YAML.load(yaml))
    # The official runtime starts each tracker in a fresh process; the id
    # counter is class-global and TRACKTRACK (not a BYTETracker subclass)
    # never resets it in its ctor — normalise to a fresh process here.
    from ultralytics.trackers.basetrack import BaseTrack
    BaseTrack._count = 0
    tracker = TRACKER_MAP[type_name](args=cfg)

    # DetSet exposes exactly the attribute/boolean-index surface the official
    # trackers use (xywh/conf/cls + results[mask]); rows keep the row-order
    # idx in the last column so the rebuild preserves full-set indices.
    class DetSet:
        def __init__(self, rows):
            a = np.asarray(rows, dtype=np.float64).reshape(-1, 7)
            self.xywh, self.conf = a[:, :4], a[:, 4]
            self.cls, self.idx = a[:, 5].astype(int), a[:, 6].astype(int)
            self._rows = [tuple(r) for r in a]
        def __len__(self): return len(self._rows)
        def __getitem__(self, m):
            return DetSet([self._rows[i] for i in np.flatnonzero(m)])

    cfg = IterableSimpleNamespace(**YAML.load(yaml))
    out = []
    for f, rows in enumerate(rows_per_frame):
        det = DetSet(rows)
        res = tracker.update(det, None)
        res = np.asarray(res).reshape(-1, 8)
        if len(res):
            # Official rows are [x1,y1,x2,y2,...]; normalise to tlwh so both
            # sides speak the same field order.
            res = res.copy()
            res[:, 2] -= res[:, 0]
            res[:, 3] -= res[:, 1]
        out.append(res)
    return out


def cpp_track(harness, type_name, rows_per_frame):
    lines = [f"TRACK {type_name}"]
    for f, rows in enumerate(rows_per_frame):
        lines.append(f"F {f} {len(rows)} " + " ".join(
            f"{r[0]:.6f} {r[1]:.6f} {r[2]:.6f} {r[3]:.6f} {r[4]:.6f} {r[5]} {r[6]}"
            for r in rows))
    proc = subprocess.run([harness], input="\n".join(lines) + "\n",
                          capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        raise RuntimeError(f"harness failed: {proc.stderr[-300:]}")
    out = {}
    for line in proc.stdout.splitlines():
        if not line.startswith("F"):
            continue
        head, _, rest = line.partition(" ")
        rows = []
        for tok in rest.split("] ["):
            tok = tok.strip("[] ")
            if tok:
                v = tok.split(",")
                rows.append([float(v[0]), float(v[1]), float(v[2]),
                             float(v[3]), int(v[4]), float(v[5]), int(v[6]),
                             int(v[7])])
        out[int(head[1:])] = rows
    return out


def diff_frames(official, cpp):
    """Field-by-field comparison; rows aligned by track id per frame."""
    problems = []
    for f, (off, cpp_rows) in enumerate(zip(official, cpp)):
        off_by_id = {int(r[4]): r for r in off}
        cpp_by_id = {int(r[4]): r for r in cpp_rows}
        if set(off_by_id) != set(cpp_by_id):
            problems.append(f"F{f}: id sets differ "
                            f"{sorted(off_by_id)} vs {sorted(cpp_by_id)}")
            continue
        for tid in sorted(off_by_id):
            o, c = off_by_id[tid], cpp_by_id[tid]
            if int(o[6]) != int(c[6]) or int(o[7]) != int(c[7]):
                problems.append(f"F{f} id{tid}: cls/idx {o[6]},{o[7]} vs "
                                f"{c[6]},{c[7]}")
            for i, name in enumerate(("x1", "y1", "w", "h")):
                if abs(o[i] - c[i]) > TOL_COORD:
                    problems.append(f"F{f} id{tid}: {name} {o[i]:.4f} vs "
                                    f"{c[i]:.4f}")
            if abs(o[5] - c[5]) > TOL_SCORE:
                problems.append(f"F{f} id{tid}: score {o[5]:.5f} vs {c[5]:.5f}")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness", default="build_app/bin/track_parity_harness")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--detector", default="yolo26n.pt")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.detector)
    frames = make_frames(args.frames)
    rows = detect_rows(model, frames)
    print(f"detector rows: {[len(r) for r in rows]}")

    failed = 0
    for t in TYPES:
        official = official_track(t, rows)
        cpp = cpp_track(args.harness, t, rows)
        problems = diff_frames(official, [cpp.get(f, [])
                                          for f in range(len(official))])
        n_rows = sum(len(o) for o in official)
        if problems:
            failed += 1
            print(f"{t:11s} FAIL ({len(problems)} field diffs over "
                  f"{n_rows} rows); first: {problems[0]}")
        else:
            print(f"{t:11s} PASS ({n_rows} rows, field-by-field identical "
                  f"within tol)")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
