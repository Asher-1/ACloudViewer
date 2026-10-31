#!/usr/bin/env python3
"""Numeric parity of the C++ tracker / relation-belief port against upstream.

Generates one deterministic scenario (moving objects, a dropout, a label flip,
a near-duplicate region, a near-full-frame box, a low-confidence dip and a
relation that flips sign), runs it through upstream deploy/render_video.py
(Tracker, EdgeBook, decode) and through the C++ replay binary, and diffs the
per-frame state: track boxes / alpha / hits / label, every edge's score,
shown value, on-state and run counters, and the visible edge list.

    AICORE_TEST_RELATE_UPSTREAM=/path/to/RelateAnything-GGML \
        compare_temporal_upstream.py --replay build_app/bin/plugin_test/test_relate_temporal_replay

Exit 77 (skip) when the upstream checkout or its Python deps are missing.
"""
import argparse
import math
import os
import subprocess
import sys
import tempfile
import types

import numpy as np

PREDICATES = [("riding", 0), ("holding", 0), ("wearing", 0),
              ("on", 1), ("beside", 1), ("near", 1)]
CAL_A, CAL_B = 0.5651, -1.9623
W, H, FRAMES = 640, 480, 48


def scenario():
    rng = np.random.default_rng(7)
    frames = []
    for f in range(FRAMES):
        dets = []  # (key, box, conf, label)
        a = [40 + 4 * f, 120, 160 + 4 * f, 400]
        dets.append(("A", a, 0.90, "man" if f % 7 == 3 else "person"))
        dets.append(("B", [200 + rng.normal(0, 1.5), 180, 520, 420], 0.85,
                     "horse"))
        if not 14 <= f <= 26:
            dets.append(("C", [a[0] + 20, a[1] - 30, a[0] + 90, a[1] + 20],
                         0.60, "hat"))
        if f >= 8:
            dets.append(("D", [500, 60 + f, 600, 140 + f],
                         0.15 if 20 <= f <= 24 else 0.70, "car"))
        if 5 <= f <= 12:  # same pixels as A under another name
            dets.append(("dup", [a[0] + 2, a[1] + 1, a[2] + 2, a[3] + 1], 0.55,
                         "person"))
        if 5 <= f <= 12:  # IoU ~0.6 with A: between the two merge thresholds
            dets.append(("mid", [a[0] + 30, a[1], a[2] + 30, a[3]], 0.50,
                         "person"))
        if 10 <= f <= 15:  # near-full-frame region (> 0.85 of the image)
            dets.append(("big", [2, 2, W - 2, H - 2], 0.50, "field"))
        n = len(dets)
        rows = []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                pred = rng.normal(-1.0, 0.8, len(PREDICATES))
                if dets[i][0] == "A" and dets[j][0] == "B":
                    pred[0] = -8.0 if 26 <= f <= 47 else 5.0  # riding flips
                if dets[i][0] == "A" and dets[j][0] == "C":
                    pred[2] = 3.0  # wearing
                if dets[i][0] == "B" and dets[j][0] == "D":
                    pred[4] = 2.5  # beside (spatial)
                rows.append((i, j, 1, float(rng.normal(1.0, 0.3)), pred))
        rows.append((0, 0, 0, 0.0, np.zeros(len(PREDICATES))))  # padding slots
        rows.append((1, 1, 0, 0.0, np.zeros(len(PREDICATES))))
        frames.append((dets, rows))
    return frames


def write_scenario(path, frames):
    with open(path, "w") as fh:
        fh.write("PRED %d\n" % len(PREDICATES))
        for name, sp in PREDICATES:
            fh.write("%s|%d\n" % (name, sp))
        fh.write("CAL %r %r\nSIZE %d %d\nFRAMES %d\n" % (CAL_A, CAL_B, W, H,
                                                         len(frames)))
        for f, (dets, rows) in enumerate(frames):
            fh.write("FRAME %d %d\n" % (f, len(dets)))
            for _, b, c, label in dets:
                fh.write("D %r %r %r %r %r %s\n" % (*[float(x) for x in b], c,
                                                    label))
            fh.write("R %d\n" % len(rows))
            for i, j, v, pair, pred in rows:
                fh.write("P %d %d %d %r %s\n" % (
                    i, j, v, pair, " ".join(repr(float(x)) for x in pred)))


def import_upstream(root):
    sys.path.insert(0, root)
    # render_video only needs ffmpeg for video I/O, which this never touches.
    if "imageio_ffmpeg" not in sys.modules:
        try:
            import imageio_ffmpeg  # noqa: F401
        except Exception:
            sys.modules["imageio_ffmpeg"] = types.ModuleType("imageio_ffmpeg")
    import deploy.render_video as rv
    from deploy.postprocess import ThresholdConfig, decode
    return rv, ThresholdConfig, decode


def run_upstream(root, frames):
    rv, ThresholdConfig, decode = import_upstream(root)
    names = [p for p, _ in PREDICATES]
    is_sp = {p: bool(s) for p, s in PREDICATES}
    vidx = {p: i for i, p in enumerate(names)}
    # Exactly the flags render_video.py passes (defaults of its argparse).
    cfg = ThresholdConfig(threshold=0.0, topk=24, pair_weight=1.0,
                          calib_a=CAL_A, calib_b=CAL_B, max_per_pair=2)
    tracker = rv.Tracker(iou_thr=0.3, merge_iou=0.75, max_age=8, min_hits=2,
                         high_conf=0.25, q=1.0, r=10.0, stride=1)
    book = rv.EdgeBook(score_ema=0.35, on_thr=0.45, off_thr=min(0.38, 0.35),
                       on_frames=2, off_frames=6, max_edges=8, max_spatial=3,
                       spatial_on_thr=0.42, belief=True, kf_q=0.05, kf_r=1.0)
    out = []
    for f, (dets, rows) in enumerate(frames):
        boxes = np.array([d[1] for d in dets], np.float32).reshape(-1, 4)
        confs = np.array([d[2] for d in dets], np.float32)
        labels = [d[3] for d in dets]
        trips, raw = [], None
        if len(boxes) >= 2:
            pred = np.array([r[4] for r in rows], np.float32)
            pair = np.array([r[3] for r in rows], np.float32)
            sub = np.array([r[0] for r in rows], np.int64)
            obj = np.array([r[1] for r in rows], np.int64)
            valid = np.array([r[2] for r in rows], bool)
            trips = decode(pred, pair, sub, obj, valid, names, cfg,
                           boxes_xyxy=boxes, box_scores=confs,
                           box_labels=labels)
            kof = {(int(a), int(b)): k
                   for k, (a, b) in enumerate(zip(sub, obj)) if valid[k]}
            raw = (pred, pair, kof, cfg.contract(), vidx)
        det2track = tracker.observe(boxes, labels, confs, masks=None,
                                    frame_area=float(W * H), max_frac=0.85)
        book.observe(trips, det2track, is_sp, raw)
        tracker.step(W, H)
        book.step({t.tid for t in tracker.tracks})
        lines = ["FRAME %d" % f]
        for t in sorted(tracker.tracks, key=lambda t: t.tid):
            b = t.box
            lines.append("T %d %.3f %.3f %.3f %.3f %.3f %d %d %s" % (
                t.tid, b[0], b[1], b[2], b[3], t.alpha, t.hits, t.misses,
                t.label))
        for k in sorted(book.edges, key=lambda k: (k[0], k[1], vidx[k[2]])):
            e = book.edges[k]
            lines.append("E %d %d %d %.3f %.3f %d %.3f %d %d" % (
                e.sub, e.obj, vidx[e.pred], e.score, e.shown, int(e.on),
                e.alpha, e.on_run, e.off_run))
        for e in book.visible():
            lines.append("V %d %d %d" % (e.sub, e.obj, vidx[e.pred]))
        out.extend(lines)
    return out


def tokens_match(a, b, tol):
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        try:
            fx, fy = float(x), float(y)
        except ValueError:
            if x != y:
                return False
            continue
        if "." in x or "." in y:
            if not math.isclose(fx, fy, abs_tol=tol):
                return False
        elif fx != fy:
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", required=True)
    ap.add_argument("--upstream",
                    default=os.environ.get("AICORE_TEST_RELATE_UPSTREAM", ""))
    ap.add_argument("--tol", type=float, default=0.02,
                    help="absolute tolerance on printed decimals "
                         "(float32 vs float64, 3 decimals)")
    ap.add_argument("--keep", default="")
    args = ap.parse_args()
    if not args.upstream or not os.path.isdir(
            os.path.join(args.upstream, "deploy")):
        print("SKIP: set AICORE_TEST_RELATE_UPSTREAM to a RelateAnything "
              "checkout")
        return 77
    frames = scenario()
    try:
        expected = run_upstream(args.upstream, frames)
    except ImportError as e:
        print("SKIP: upstream dependency missing: %s" % e)
        return 77
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "scenario.txt")
        write_scenario(path, frames)
        actual = subprocess.run([args.replay, path], check=True,
                                capture_output=True,
                                text=True).stdout.splitlines()
        if args.keep:
            os.makedirs(args.keep, exist_ok=True)
            open(os.path.join(args.keep, "upstream.txt"), "w").write(
                "\n".join(expected) + "\n")
            open(os.path.join(args.keep, "cpp.txt"), "w").write(
                "\n".join(actual) + "\n")
    bad, frame = 0, -1
    for i in range(max(len(expected), len(actual))):
        e = expected[i] if i < len(expected) else "<missing>"
        a = actual[i] if i < len(actual) else "<missing>"
        if e.startswith("FRAME"):
            frame = int(e.split()[1])
        if not tokens_match(e.split(), a.split(), args.tol):
            bad += 1
            if bad <= 12:
                print("frame %d line %d\n  upstream: %s\n  cpp:      %s" %
                      (frame, i, e, a))
    kinds = {k: sum(1 for l in expected if l.startswith(k + " "))
             for k in "TEV"}
    print("compared %d lines (%d tracks, %d edges, %d visible); %d mismatches"
          % (len(expected), kinds["T"], kinds["E"], kinds["V"], bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
