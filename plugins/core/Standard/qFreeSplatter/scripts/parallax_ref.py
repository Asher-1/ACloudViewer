#!/usr/bin/env python3
"""Independent (model-free) parallax reference for a frame pair.

Dev-time validation ONLY (never shipped; runs in the nix devShell, which provides
cv2+numpy per flake.nix). The C++ engine's after-inference `--parallax` measures
depth conditioning from the model's OWN recovered geometry; this measures it from
raw 2D image evidence (feature matches -> epipolar geometry), so the two can be
cross-checked. Agreement on a known-good-depth scene validates the engine metric;
a large gap (geometric angle tiny, model angle confident) flags the model
hallucinating depth.

Two outputs:
  * R_H  -- ORB-SLAM homography-vs-fundamental score ratio. ~>0.45 => a homography
            explains the motion => pure-rotation / planar => no usable parallax.
            Calibration-free.
  * median triangulation angle (deg) -- COLMAP's depth-conditioning number; below
            ~1-2 deg depth is ill-posed. Uses an assumed/supplied focal.

Images are center-cropped square and resized to 512 to mirror the engine's
preprocessing, so a focal in 512-px units matches the model's estimated focal.

  nix develop -c python3 scripts/parallax_ref.py IMG0 IMG1 [--focal-px F]
"""
import argparse, sys
import numpy as np
import cv2


def preprocess(path, size=512):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        sys.exit(f"cannot read {path}")
    h, w = img.shape[:2]
    s = min(h, w)
    img = img[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def match(img0, img1):
    g0, g1 = (cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in (img0, img1))
    try:
        det = cv2.SIFT_create(nfeatures=4000)
        norm = cv2.NORM_L2
    except AttributeError:
        det = cv2.ORB_create(nfeatures=4000)
        norm = cv2.NORM_HAMMING
    k0, d0 = det.detectAndCompute(g0, None)
    k1, d1 = det.detectAndCompute(g1, None)
    if d0 is None or d1 is None or len(k0) < 8 or len(k1) < 8:
        sys.exit("too few features")
    bf = cv2.BFMatcher(norm)
    good = [m for m, n in bf.knnMatch(d0, d1, k=2) if m.distance < 0.75 * n.distance]
    p0 = np.float32([k0[m.queryIdx].pt for m in good])
    p1 = np.float32([k1[m.trainIdx].pt for m in good])
    return p0, p1


def score_homography(H, p0, p1, sigma=1.0):
    if H is None:
        return 0.0
    th, inv = 5.991, 1.0 / sigma ** 2
    Hi = np.linalg.inv(H)
    def transfer(M, a, b):
        ah = np.c_[a, np.ones(len(a))]
        q = (M @ ah.T).T
        q = q[:, :2] / q[:, 2:3]
        chi = np.sum((q - b) ** 2, axis=1) * inv
        return np.where(chi < th, th - chi, 0.0)
    return float(np.sum(transfer(H, p0, p1) + transfer(Hi, p1, p0)))


def score_fundamental(F, p0, p1, sigma=1.0):
    if F is None:
        return 0.0
    thF, thScore, inv = 3.841, 5.991, 1.0 / sigma ** 2
    a0 = np.c_[p0, np.ones(len(p0))]
    a1 = np.c_[p1, np.ones(len(p1))]
    def epi(F, src, dst):                             # point-line chi in dst image
        l = (F @ src.T).T                             # lines in dst
        num = np.sum(l[:, :2] * dst, axis=1) + l[:, 2]
        d2 = num ** 2 / (l[:, 0] ** 2 + l[:, 1] ** 2 + 1e-12)
        chi = d2 * inv
        return np.where(chi < thF, thScore - chi, 0.0)
    s01 = epi(F, a0, p1)
    s10 = epi(F.T, a1, p0)
    return float(np.sum(s01 + s10))


def triangulation_angle(p0, p1, f, cx=256.0, cy=256.0):
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], np.float64)
    E, _ = cv2.findEssentialMat(p0, p1, K, cv2.RANSAC, 0.999, 1.0)
    if E is None or E.shape != (3, 3):
        return None, 0
    _, R, t, mP = cv2.recoverPose(E, p0, p1, K)
    inl = (mP.ravel() > 0)
    if inl.sum() < 8:
        return None, int(inl.sum())
    P0 = K @ np.eye(3, 4)
    P1 = K @ np.hstack([R, t])
    X4 = cv2.triangulatePoints(P0, P1, p0[inl].T, p1[inl].T)
    X = (X4[:3] / X4[3]).T
    C1 = (-R.T @ t).ravel()
    r0 = X - 0.0
    r1 = X - C1
    n0 = np.linalg.norm(r0, axis=1)
    n1 = np.linalg.norm(r1, axis=1)
    ok = (n0 > 1e-9) & (n1 > 1e-9) & (X[:, 2] > 0)
    cosang = np.sum(r0[ok] * r1[ok], axis=1) / (n0[ok] * n1[ok])
    ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
    if len(ang) == 0:
        return None, int(inl.sum())
    return float(np.median(ang)), int(ok.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("img0"); ap.add_argument("img1")
    ap.add_argument("--focal-px", type=float, default=None,
                    help="focal in 512-px units; default from --fov-deg")
    ap.add_argument("--fov-deg", type=float, default=60.0)
    a = ap.parse_args()
    f = a.focal_px if a.focal_px else 0.5 * 512 / np.tan(np.radians(a.fov_deg) / 2)

    i0, i1 = preprocess(a.img0), preprocess(a.img1)
    p0, p1 = match(i0, i1)
    H, _ = cv2.findHomography(p0, p1, cv2.RANSAC, 3.0)
    F, _ = cv2.findFundamentalMat(p0, p1, cv2.FM_RANSAC, 3.0, 0.99)
    sH, sF = score_homography(H, p0, p1), score_fundamental(F, p0, p1)
    rH = sH / (sH + sF) if (sH + sF) > 0 else float("nan")
    tri, ntri = triangulation_angle(p0, p1, f)
    tri_s = f"{tri:.3f}" if tri is not None else "n/a"
    print(f"parallax_ref: matches={len(p0)}  R_H={rH:.3f} (>0.45 => degenerate)  "
          f"tri_angle={tri_s} deg (npts={ntri})  focal={f:.1f}px")


if __name__ == "__main__":
    main()
