#!/usr/bin/env python3
"""Parity gate for src/glb_export.cpp against the reference GLB geometry.

Builds a small deterministic synthetic (depth, conf, K, ext, image) set, computes
the EXPECTED aligned point cloud + colors by replicating the reference functions
from depth_anything_3/utils/export/glb.py in numpy (the reference module imports
`trimesh` at import time and trimesh is not installed, so we replicate
`_depths_to_world_points_with_colors`, `get_conf_thresh`, and
`_compute_alignment_transform_first_cam_glTF_center_by_points` exactly --
`trimesh.transform_points(P, A)` is exactly `(A @ [P;1])[:3]`).

Then it feeds the same inputs to the C++ `glb_parity_dump` harness, parses the
produced .glb POINTS primitive, and asserts the sorted point/color sets match.
"""
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/tmp/da3-src/src")  # reference package location (per task)

# ---------------------------------------------------------------------------
# Reference math (faithful re-implementation; see glb.py)
# ---------------------------------------------------------------------------

def get_conf_thresh(conf, conf_thresh=1.05, conf_thresh_percentile=40.0,
                    ensure_thresh_percentile=90.0):
    lower = np.percentile(conf, conf_thresh_percentile)
    upper = np.percentile(conf, ensure_thresh_percentile)
    return float(min(max(conf_thresh, lower), upper))


def transform_points(points, A):
    # trimesh.transform_points for an affine 4x4: (A @ [X;1])[:3]
    if points.shape[0] == 0:
        return points
    homo = np.hstack([points, np.ones((points.shape[0], 1))])
    return (A @ homo.T)[:3].T


def depths_to_world_points_with_colors(depth, K, ext_w2c, images_u8, conf, conf_thr):
    N, H, W = depth.shape
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    ones = np.ones_like(us)
    pix = np.stack([us, vs, ones], axis=-1).reshape(-1, 3)
    pts_all, col_all = [], []
    for i in range(N):
        d = depth[i]
        valid = np.isfinite(d) & (d > 0)
        valid &= conf[i] >= conf_thr
        if not np.any(valid):
            continue
        d_flat = d.reshape(-1)
        vidx = np.flatnonzero(valid.reshape(-1))
        K_inv = np.linalg.inv(K[i])
        c2w = np.linalg.inv(ext_w2c[i])
        rays = K_inv @ pix[vidx].T
        Xc = rays * d_flat[vidx][None, :]
        Xc_h = np.vstack([Xc, np.ones((1, Xc.shape[1]))])
        Xw = (c2w @ Xc_h)[:3].T.astype(np.float32)
        cols = images_u8[i].reshape(-1, 3)[vidx].astype(np.uint8)
        pts_all.append(Xw)
        col_all.append(cols)
    if not pts_all:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)
    return np.concatenate(pts_all, 0), np.concatenate(col_all, 0)


def compute_alignment(ext_w2c0, points_world):
    w2c0 = ext_w2c0.astype(np.float64)
    M = np.eye(4, dtype=np.float64)
    M[1, 1] = -1.0
    M[2, 2] = -1.0
    A_no_center = M @ w2c0
    if points_world.shape[0] > 0:
        pts_tmp = transform_points(points_world, A_no_center)
        center = np.median(pts_tmp, axis=0)
    else:
        center = np.zeros(3, dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = -center
    return T @ A_no_center


# ---------------------------------------------------------------------------
# Synthetic deterministic input
# ---------------------------------------------------------------------------

def make_synthetic():
    rng = np.random.default_rng(1234)
    N, H, W = 2, 12, 16
    depth = rng.uniform(0.5, 5.0, size=(N, H, W)).astype(np.float32)
    conf = rng.uniform(1.0, 2.0, size=(N, H, W)).astype(np.float32)
    images = rng.integers(0, 256, size=(N, H, W, 3), dtype=np.uint8)

    def K_of(fx, fy, cx, cy):
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    K = np.stack([K_of(20.0, 21.0, 8.0, 6.0), K_of(22.0, 19.5, 7.5, 6.5)]).astype(np.float32)

    def rot(ax, ay, az):
        cx, sx = np.cos(ax), np.sin(ax)
        cy, sy = np.cos(ay), np.sin(ay)
        cz, sz = np.cos(az), np.sin(az)
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        return Rz @ Ry @ Rx

    def ext_of(R, t):
        E = np.eye(4)
        E[:3, :3] = R
        E[:3, 3] = t
        return E.astype(np.float32)

    ext = np.stack([
        ext_of(rot(0.05, -0.1, 0.02), [0.1, -0.2, 0.3]),
        ext_of(rot(-0.08, 0.15, -0.03), [-0.15, 0.25, 0.5]),
    ]).astype(np.float32)
    return N, H, W, depth, conf, K, ext, images


def write_blob(path, N, H, W, depth, conf, K, ext, images):
    with open(path, "wb") as f:
        f.write(struct.pack("<iii", N, H, W))
        f.write(depth.astype("<f4").tobytes())
        f.write(conf.astype("<f4").tobytes())
        f.write(K.reshape(N, 9).astype("<f4").tobytes())
        f.write(ext.reshape(N, 16).astype("<f4").tobytes())
        f.write(images.reshape(N, H * W * 3).astype("<u1").tobytes())


# ---------------------------------------------------------------------------
# Minimal .glb parser -> POINTS primitive POSITION + COLOR_0
# ---------------------------------------------------------------------------

import json as _json

CT = {5120: ("b", 1), 5121: ("B", 1), 5122: ("h", 2), 5123: ("H", 2),
      5125: ("I", 4), 5126: ("f", 4)}
NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def parse_glb(path):
    with open(path, "rb") as f:
        data = f.read()
    magic, version, length = struct.unpack_from("<III", data, 0)
    assert magic == 0x46546C67, "bad magic"
    assert version == 2, "bad version"
    assert length == len(data), "length mismatch"
    off = 12
    json_chunk = bin_chunk = None
    while off < len(data):
        clen, ctype = struct.unpack_from("<II", data, off)
        off += 8
        chunk = data[off:off + clen]
        off += clen
        if ctype == 0x4E4F534A:
            json_chunk = chunk
        elif ctype == 0x004E4942:
            bin_chunk = chunk
    gltf = _json.loads(json_chunk.decode("utf-8"))

    def read_accessor(idx):
        acc = gltf["accessors"][idx]
        bv = gltf["bufferViews"][acc["bufferView"]]
        base = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
        fmt, sz = CT[acc["componentType"]]
        nc = NCOMP[acc["type"]]
        count = acc["count"]
        out = np.empty((count, nc), dtype=np.float64 if fmt == "f" else np.int64)
        for i in range(count):
            for c in range(nc):
                (val,) = struct.unpack_from("<" + fmt, bin_chunk, base + (i * nc + c) * sz)
                out[i, c] = val
        return out, acc

    # find the POINTS primitive (mode==0)
    prim = None
    for m in gltf["meshes"]:
        for p in m["primitives"]:
            if p.get("mode", 4) == 0:
                prim = p
                break
        if prim:
            break
    assert prim is not None, "no POINTS primitive"
    pos, _ = read_accessor(prim["attributes"]["POSITION"])
    col, cacc = read_accessor(prim["attributes"]["COLOR_0"])
    if cacc.get("normalized"):
        pass  # we keep raw u8 values for exact color compare
    return pos.astype(np.float32), col.astype(np.int64)


def sort_key(pos, col):
    # lexsort by (x,y,z,r,g,b)
    keys = (col[:, 3], col[:, 2], col[:, 1], col[:, 0],
            pos[:, 2], pos[:, 1], pos[:, 0])
    return np.lexsort(keys)


def main():
    N, H, W, depth, conf, K, ext, images = make_synthetic()

    conf_thr = get_conf_thresh(conf.reshape(-1), 1.05, 40.0, 90.0)
    pts, cols = depths_to_world_points_with_colors(depth, K, ext, images, conf, conf_thr)
    A = compute_alignment(ext[0], pts)
    exp_pos = transform_points(pts, A).astype(np.float32)
    exp_col = np.hstack([cols.astype(np.int64),
                         np.full((cols.shape[0], 1), 255, np.int64)])

    tmp = tempfile.mkdtemp(prefix="parity_glb_")
    blob = os.path.join(tmp, "in.bin")
    out_glb = os.path.join(tmp, "out.glb")
    write_blob(blob, N, H, W, depth, conf, K, ext, images)

    dump = None
    for cand in ("build/tests/glb_parity_dump", "build/bin/glb_parity_dump"):
        p = os.path.join(ROOT, cand)
        if os.path.exists(p):
            dump = p
            break
    if dump is None:
        # search
        for r, _d, fnames in os.walk(os.path.join(ROOT, "build")):
            if "glb_parity_dump" in fnames:
                dump = os.path.join(r, "glb_parity_dump")
                break
    assert dump, "glb_parity_dump binary not found under build/"

    subprocess.run([dump, blob, out_glb, "1"], check=True)
    act_pos, act_col = parse_glb(out_glb)

    print(f"conf_thr={conf_thr:.6f}  expected_points={exp_pos.shape[0]}  "
          f"glb_points={act_pos.shape[0]}")

    ok = True
    if exp_pos.shape[0] != act_pos.shape[0]:
        print(f"PARITY GLB: FAIL (count mismatch {exp_pos.shape[0]} vs {act_pos.shape[0]})")
        sys.exit(1)

    ei = sort_key(exp_pos, exp_col)
    ai = sort_key(act_pos, act_col)
    ep, ec = exp_pos[ei], exp_col[ei]
    ap, ac = act_pos[ai], act_col[ai]

    dxyz = float(np.max(np.abs(ep - ap))) if ep.shape[0] else 0.0
    dcol = int(np.max(np.abs(ec - ac))) if ec.shape[0] else 0
    print(f"max|dxyz|={dxyz:.3e}  max|dcolor|={dcol}")

    if dxyz >= 1e-4:
        ok = False
    if dcol != 0:
        ok = False

    if ok:
        print("PARITY GLB: PASS")
        sys.exit(0)
    else:
        print("PARITY GLB: FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
