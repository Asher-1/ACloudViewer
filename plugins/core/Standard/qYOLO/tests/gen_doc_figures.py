#!/usr/bin/env python3
"""Generate the qYOLO doc effect figures from real model runs.

yolo-track.jpg : frames from the moving-sprite sequence with the actual
                 tracker output (boxes + stable ids + scores) rendered.
yolo-reid.jpg  : per-identity embedding similarity matrix from the real
                 reid encoder graph (reid_diag on the published f32 GGUF),
                 same-identity pairs stay bright across occlusion gaps.

Not a ctest — doc asset generation (needs ultralytics + a detector ckpt).
"""
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, "plugins/core/Standard/qYOLO/tests")
import track_parity_check as tpc  # reuse the deterministic frame/det chain

IMAGES_DIR = "plugins/core/Standard/qYOLO/images"
PALETTE = ((66, 186, 66), (214, 112, 32), (66, 66, 214), (180, 66, 214))


def render_strip(frames, rows_per_frame, out_path, picks=(0, 10, 17, 25)):
    tiles = []
    for f in picks:
        img = frames[f].copy()
        for r in rows_per_frame[f]:
            x1, y1, w, h, tid, score, cls = (int(r[0]), int(r[1]), int(r[2]),
                                             int(r[3]), int(r[4]), r[5], r[6])
            color = PALETTE[tid % len(PALETTE)]
            cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), color, 2)
            cv2.putText(img, f"id{tid} {score:.2f}", (x1, max(14, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        cv2.putText(img, f"frame {f}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(img)
    strip = np.concatenate(tiles, axis=1)
    cv2.imwrite(out_path, cv2.cvtColor(strip, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("wrote", out_path, strip.shape)


def reid_matrix(frames, rows_per_frame, gguf, out_path):
    """Real reid-encoder similarity: prep each tracked row's crop exactly
    like tasks/reid/capi.cpp (save_one_box + stretch) and run the published
    f32 GGUF through reid_diag (same graph the plugin uses)."""

    # Track first (bytetrack rows carry stable ids), then sample one crop
    # per (id, frame) across time: same identity across a gap must stay
    # similar; the two identities must stay dissimilar.
    tracked = {}
    for f, rows in enumerate(rows_per_frame):
        for r in rows:
            tracked.setdefault(int(r[4]), []).append((f, r))
    ids = sorted(tracked)[:2]
    crops, labels = [], []
    for tid in ids:
        entries = tracked[tid]
        picks = [entries[0], entries[len(entries) // 2], entries[-1]]
        for f, r in picks:
            crops.append((tid, f, r))
            labels.append(f"id{tid}@f{f}")

    # Batch the CHW prep in torch-free numpy; embed via reid_diag per crop.
    embs = []
    for tid, f, r in crops:
        x1, y1 = int(r[0]), int(r[1])
        x2, y2 = x1 + int(r[2]), y1 + int(r[3])
        # The frames were drawn in RGB; keep bytes identical to capi input.
        crop = frames[f][max(0, y1):y2, max(0, x1):x2]
        chw = np.ascontiguousarray(
            cv2.resize(crop, (224, 224),
                       interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)[0:3]
        ).astype(np.float32) / 255.0
        chw.tofile("/tmp/reid/fig.chw.bin")
        out = "/tmp/reid/fig"
        subprocess.run(["build_app/bin/aicore_tests/reid_diag", "--gguf", gguf,
                        "--chw", "/tmp/reid/fig.chw.bin", "--device", "cuda",
                        "--dump", out], check=True, capture_output=True)
        embs.append(np.fromfile(out + ".embed.bin", dtype=np.float32))
    E = np.stack(embs)
    S = E @ E.T / (np.linalg.norm(E, axis=1)[:, None]
                   * np.linalg.norm(E, axis=1)[None, :] + 1e-12)

    cell = 64
    n = len(labels)
    heat = np.zeros((n * cell, n * cell, 3), dtype=np.uint8)
    for i in range(n):
        for j in range(n):
            v = float(np.clip((S[i, j] - 0.2) / 0.8, 0, 1))
            block = (np.array([[v]])) .repeat(cell, 0).repeat(cell, 1)
            col = (np.stack([block * 255,
                             (1 - abs(block - 0.5) * 2) * 200,
                             (1 - block) * 255], axis=2)).astype(np.uint8)
            heat[i * cell:(i + 1) * cell, j * cell:(j + 1) * cell] = col
            txt = f"{S[i, j]:.2f}"
            cv2.putText(heat, txt, (j * cell + 4, i * cell + cell - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 30, 30), 1,
                        cv2.LINE_AA)
    for i, lab in enumerate(labels):
        cv2.putText(heat, lab, (6, i * cell + 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(out_path, heat, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("wrote", out_path, heat.shape, "| min same-id cos:",
          round(float(min(S[i, i] for i in range(n))), 4))


def main():
    from ultralytics import YOLO
    model = YOLO("yolo26n.pt")
    frames = tpc.make_frames(30)
    rows = tpc.detect_rows(model, frames)

    # C++ tracker output via the parity harness (the plugin tracker).
    harness = "build_app/bin/track_parity_harness"
    lines = ["TRACK bytetrack"]
    for f, rs in enumerate(rows):
        lines.append(f"F {f} {len(rs)} " + " ".join(
            f"{r[0]:.6f} {r[1]:.6f} {r[2]:.6f} {r[3]:.6f} {r[4]:.6f} {r[5]} "
            f"{r[6]}" for r in rs))
    proc = subprocess.run([harness], input="\n".join(lines) + "\n",
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-300:]
    tracked = []
    for line in proc.stdout.splitlines():
        rows_out = []
        _, _, rest = line.partition(" ")
        for tok in rest.split("] ["):
            tok = tok.strip("[] ")
            if tok:
                v = [float(x) for x in tok.split(",")]
                rows_out.append(v)  # x1 y1 w h id score cls idx
        tracked.append(rows_out)

    render_strip(frames, tracked, f"{IMAGES_DIR}/yolo-track.jpg")
    reid_matrix(frames, tracked,
                "/home/ludahai/cloudViewer_data/extract/yolo_models/"
                "reid-yolo26n-cls-f32.gguf",
                f"{IMAGES_DIR}/yolo-reid.jpg")


if __name__ == "__main__":
    main()
