#!/usr/bin/env python3
"""Parity gate for src/colmap_export.cpp against the reference COLMAP export.

Builds a small deterministic synthetic (depth, conf, K, ext, image) set, computes
the EXPECTED COLMAP model fields by replicating the reference exporter
`depth_anything_3/utils/export/colmap.py` in numpy WITHOUT pycolmap:
  - conf_thr = np.percentile(conf, 40)
  - back-project (reference `_depths_to_world_points_with_colors` if importable,
    else replicated in numpy: K_inv @ [u,v,1] * d then c2w @ [Xc;1])
  - cameras: PINHOLE, rescaled intrinsics, orig width/height
  - images: qvec = reference `rotmat2qvec`, tvec
  - points3D: world xyz + rgb

It feeds the same inputs to the C++ `colmap_parity_dump` harness, reads the
produced .bin model with read_write_model.py's reference readers, and asserts the
fields match (counts equal, values within tolerance, rgb exact).
"""
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/tmp/da3-src/src")  # reference package location (per task)

from depth_anything_3.utils.read_write_model import (  # noqa: E402
    read_cameras_binary,
    read_images_binary,
    read_points3D_binary,
    rotmat2qvec,
)

# Try to use the reference back-projection; fall back to a numpy replica.
try:
    from depth_anything_3.utils.export.glb import (  # noqa: E402
        _depths_to_world_points_with_colors as _ref_bp,
    )
    HAVE_REF_BP = True
except Exception:
    HAVE_REF_BP = False


# ---------------------------------------------------------------------------
# Reference back-projection replica (matches _depths_to_world_points_with_colors)
# ---------------------------------------------------------------------------
def depths_to_world_points_with_colors(depth, K, ext_w2c, images_u8, conf, conf_thr):
    if HAVE_REF_BP:
        return _ref_bp(depth, K, ext_w2c, images_u8, conf, conf_thr)
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


# ---------------------------------------------------------------------------
# Synthetic deterministic input (same KIND as parity_glb.py)
# ---------------------------------------------------------------------------
def make_synthetic():
    rng = np.random.default_rng(4321)
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
    # distinct original sizes per frame
    orig_wh = [(W * 3, H * 3), (W * 2, H * 2)]
    names = ["frame_000.png", "frame_001.png"]
    return N, H, W, depth, conf, K, ext, images, orig_wh, names


def write_blob(path, N, H, W, depth, conf, K, ext, images, orig_wh, names):
    with open(path, "wb") as f:
        f.write(struct.pack("<iii", N, H, W))
        f.write(depth.astype("<f4").tobytes())
        f.write(conf.astype("<f4").tobytes())
        f.write(K.reshape(N, 9).astype("<f4").tobytes())
        f.write(ext.reshape(N, 16).astype("<f4").tobytes())
        f.write(images.reshape(N, H * W * 3).astype("<u1").tobytes())
        for (ow, oh) in orig_wh:
            f.write(struct.pack("<ii", int(ow), int(oh)))
        for nm in names:
            b = nm.encode("utf-8")
            f.write(struct.pack("<i", len(b)))
            f.write(b)


def main():
    N, H, W, depth, conf, K, ext, images, orig_wh, names = make_synthetic()

    conf_thr = float(np.percentile(conf, 40.0))
    pts, cols = depths_to_world_points_with_colors(depth, K, ext, images, conf, conf_thr)
    num_points = len(pts)

    # Expected cameras (rescaled intrinsics).
    exp_cams = {}
    exp_qvec = {}
    exp_tvec = {}
    for fidx in range(N):
        ow, oh = orig_wh[fidx]
        intr = K[fidx].copy()
        intr[:1] *= ow / W
        intr[1:2] *= oh / H
        exp_cams[fidx + 1] = {
            "width": ow, "height": oh,
            "params": np.array([intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]],
                               dtype=np.float64),
        }
        R = ext[fidx][:3, :3]
        exp_qvec[fidx + 1] = rotmat2qvec(R).astype(np.float64)
        exp_tvec[fidx + 1] = ext[fidx][:3, 3].astype(np.float64)

    # Run our writer.
    tmp = tempfile.mkdtemp(prefix="parity_colmap_")
    blob = os.path.join(tmp, "in.bin")
    out_dir = os.path.join(tmp, "model")
    os.makedirs(out_dir, exist_ok=True)
    write_blob(blob, N, H, W, depth, conf, K, ext, images, orig_wh, names)

    dump = None
    for cand in ("build/tests/colmap_parity_dump", "build/bin/colmap_parity_dump"):
        p = os.path.join(ROOT, cand)
        if os.path.exists(p):
            dump = p
            break
    if dump is None:
        for r, _d, fnames in os.walk(os.path.join(ROOT, "build")):
            if "colmap_parity_dump" in fnames:
                dump = os.path.join(r, "colmap_parity_dump")
                break
    assert dump, "colmap_parity_dump binary not found under build/"

    subprocess.run([dump, blob, out_dir, "1"], check=True)

    # Read our output with the reference binary readers.
    cams = read_cameras_binary(os.path.join(out_dir, "cameras.bin"))
    imgs = read_images_binary(os.path.join(out_dir, "images.bin"))
    p3d = read_points3D_binary(os.path.join(out_dir, "points3D.bin"))

    print(f"conf_thr={conf_thr:.6f}  expected_points={num_points}  "
          f"colmap_points={len(p3d)}  cameras={len(cams)}  images={len(imgs)}")

    ok = True

    # cameras
    if len(cams) != N:
        print(f"camera count mismatch {len(cams)} vs {N}"); ok = False
    cam_dparam = 0.0
    for cid, ec in exp_cams.items():
        c = cams.get(cid)
        if c is None:
            print(f"missing camera {cid}"); ok = False; continue
        if c.model != "PINHOLE":
            print(f"camera {cid} model {c.model} != PINHOLE"); ok = False
        if c.width != ec["width"] or c.height != ec["height"]:
            print(f"camera {cid} size {(c.width, c.height)} != "
                  f"{(ec['width'], ec['height'])}"); ok = False
        cam_dparam = max(cam_dparam,
                         float(np.max(np.abs(np.asarray(c.params, np.float64) - ec["params"]))))
    if cam_dparam >= 1e-4:
        ok = False

    # images
    img_dq = 0.0
    img_dt = 0.0
    if len(imgs) != N:
        print(f"image count mismatch {len(imgs)} vs {N}"); ok = False
    for iid in exp_qvec:
        im = imgs.get(iid)
        if im is None:
            print(f"missing image {iid}"); ok = False; continue
        q = np.asarray(im.qvec, np.float64)
        e = exp_qvec[iid]
        # q and -q are the same rotation: allow a global per-quaternion sign flip.
        dq = min(float(np.max(np.abs(q - e))), float(np.max(np.abs(q + e))))
        img_dq = max(img_dq, dq)
        img_dt = max(img_dt, float(np.max(np.abs(np.asarray(im.tvec, np.float64) -
                                                 exp_tvec[iid]))))
    if img_dq >= 1e-4 or img_dt >= 1e-4:
        ok = False

    # points3D
    pts_dxyz = 0.0
    pts_dcol = 0
    if len(p3d) != num_points:
        print(f"points3D count mismatch {len(p3d)} vs {num_points}"); ok = False
    else:
        act_xyz = np.array([p3d[k].xyz for k in p3d], np.float64)
        act_rgb = np.array([p3d[k].rgb for k in p3d], np.int64)
        exp_xyz = pts.astype(np.float64)
        exp_rgb = cols.astype(np.int64)

        def lex(xyz, rgb):
            keys = (rgb[:, 2], rgb[:, 1], rgb[:, 0],
                    xyz[:, 2], xyz[:, 1], xyz[:, 0])
            return np.lexsort(keys)

        ai = lex(act_xyz, act_rgb)
        ei = lex(exp_xyz, exp_rgb)
        if num_points:
            pts_dxyz = float(np.max(np.abs(act_xyz[ai] - exp_xyz[ei])))
            pts_dcol = int(np.max(np.abs(act_rgb[ai] - exp_rgb[ei])))
    if pts_dxyz >= 1e-4 or pts_dcol != 0:
        ok = False

    print(f"cam max|dparams|={cam_dparam:.3e}  img max|dqvec|={img_dq:.3e}  "
          f"img max|dtvec|={img_dt:.3e}  pts max|dxyz|={pts_dxyz:.3e}  "
          f"pts max|dcolor|={pts_dcol}")

    # Optional pycolmap cross-check (do not fail parity if absent).
    try:
        import pycolmap  # noqa: F401
        rec = pycolmap.Reconstruction(out_dir)
        print(f"pycolmap sanity: loaded {rec.num_cameras()} cameras, "
              f"{rec.num_images()} images, {rec.num_points3D()} points")
    except Exception as e:
        print(f"pycolmap cross-check skipped: {type(e).__name__}: {e}")

    if ok:
        print("PARITY COLMAP: PASS")
        sys.exit(0)
    print("PARITY COLMAP: FAIL")
    sys.exit(1)


if __name__ == "__main__":
    main()
