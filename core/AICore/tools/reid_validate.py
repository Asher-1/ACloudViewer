#!/usr/bin/env python3
# ReID encoder validation: accuracy and latency of the AICore ReID C API
# against same-weight PyTorch and ONNX Runtime baselines.
#
# Usage:
#   python3 core/AICore/tools/reid_validate.py \
#       --gguf ~/cloudViewer_data/extract/yolo_models/yolo26n-cls-f16.gguf \
#       --pt  /path/to/yolo26n-cls.pt \
#       --device cuda            # cuda | vulkan | cpu
#       --quant f16              # label for the report row
#
# The accuracy baseline is the PyTorch Classify-head pooled feature (the
# same node the GGML graph exports); the latency baseline is ONNX Runtime
# on a matching embed model exported on the fly. Rows are appended to the
# JSON report consumed by the ReID validation record.

import argparse
import json
import statistics
import time

import numpy as np


def build_probe_inputs(imgsz, boxes):
    """Deterministic synthetic image (same pixel pattern as reid_probe.cpp)
    so the C++ probe and the python baselines see identical geometry."""
    h, w = 480, 640
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.stack(
        [xx * 255.0 / w, yy * 255.0 / h, (xx + yy) * 255.0 / (w + h)],
        axis=2,
    ).astype(np.uint8)
    return np.ascontiguousarray(img), boxes


def torch_baseline(pt_path, img, boxes, imgsz):
    """Official ReID-class extraction: Classify head conv → pool → flatten
    (1280-dim), matching the GGML graph's embed export point."""
    import torch
    from ultralytics import YOLO

    yolo = YOLO(pt_path)
    model = yolo.model
    model.eval()
    model.fuse()
    seq = model.model
    head = seq[-1]  # Classify head

    def prep(b):
        w, hh = b[2] - b[0], b[3] - b[1]
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        nw, nh = w * 1.02 + 20, hh * 1.02 + 20  # pad=10 per side, total +20
        x1 = int(np.clip(int(cx - nw / 2), 0, img.shape[1]))
        y1 = int(np.clip(int(cy - nh / 2), 0, img.shape[0]))
        x2 = int(np.clip(int(cx + nw / 2), 0, img.shape[1]))
        y2 = int(np.clip(int(cy + nh / 2), 0, img.shape[0]))
        crop = img[y1:y2, x1:x2]
        t = crop.transpose(2, 0, 1)[None].astype(np.float32) / 255
        if t.shape[2] != imgsz or t.shape[3] != imgsz:
            t = torch.nn.functional.interpolate(
                torch.from_numpy(t), size=(imgsz, imgsz), mode="bilinear",
                align_corners=False).numpy()
        return torch.from_numpy(np.ascontiguousarray(t[0]))

    crops = [prep(b) for b in boxes]
    with torch.no_grad():
        # Extract from Classify head: conv → pool → flatten (1280-dim)
        x = torch.stack(crops)
        for layer in seq[:-1]:
            x = layer(x)
        x = head.conv(x)
        x = head.pool(x)
        feats = x.flatten(1)
    rows = [f.cpu().float().numpy().reshape(-1) for f in feats]
    return np.asarray(rows, dtype=np.float32)


def ort_baseline(pt_path, img, boxes, imgsz):
    """Export an embed-head ONNX once and run it with ONNX Runtime."""
    import io
    import torch
    from ultralytics import YOLO

    m = YOLO(pt_path).model
    m.eval()
    m.fuse()
    seq = m.model
    head = seq[-1]

    class EmbedHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.body = seq[:-1]
            self.conv = head.conv
            self.pool = head.pool

        def forward(self, x):
            for layer in self.body:
                x = layer(x)
            x = self.conv(x)
            x = self.pool(x)
            return x.flatten(1)

    wrapper = EmbedHead().eval()
    buf = io.BytesIO()
    torch.onnx.export(
        wrapper,
        torch.zeros(1, 3, imgsz, imgsz),
        buf,
        opset_version=13,
        input_names=["images"],
        output_names=["embedding"],
        dynamic_axes={"images": {0: "batch"}},
    )
    import onnxruntime as ort

    sess = ort.InferenceSession(
        buf.getvalue(), providers=["CPUExecutionProvider"]
    )

    rows = []
    for b in boxes:
        w, hh = b[2] - b[0], b[3] - b[1]
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        nw, nh = w * 1.02 + 20, hh * 1.02 + 20
        x1 = int(np.clip(int(cx - nw / 2), 0, img.shape[1]))
        y1 = int(np.clip(int(cy - nh / 2), 0, img.shape[0]))
        x2 = int(np.clip(int(cx + nw / 2), 0, img.shape[1]))
        y2 = int(np.clip(int(cy + nh / 2), 0, img.shape[0]))
        crop = img[y1:y2, x1:x2]
        t = crop.transpose(2, 0, 1)[None].astype(np.float32) / 255
        if t.shape[2] != imgsz or t.shape[3] != imgsz:
            import torch
            t = torch.nn.functional.interpolate(
                torch.from_numpy(t), size=(imgsz, imgsz), mode="bilinear",
                align_corners=False).numpy()
        out = sess.run(None, {"images": np.ascontiguousarray(t)})[0]
        rows.append(out[0])
    return np.asarray(rows, dtype=np.float32)


def aicore_embed(probe, gguf, device, img, boxes, imgsz):
    """Run the C++ reid_probe in its own process (numpy and libAICore must
    not share one process) and return the embeddings plus latency stats."""
    import json
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out_path = f.name
    res = subprocess.run(
        [probe, "--gguf", gguf, "--device", device, "--output", out_path],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        raise RuntimeError(f"reid_probe failed: {res.stderr[-400:]}")
    with open(out_path) as f:
        report = json.load(f)
    rows = np.asarray(report["embeddings"], dtype=np.float32)
    return rows, {
        "p50_ms": report["batch_mean_ms"],
        "min_ms": report["batch_best_ms"],
        "batch": report["count"],
        "dim": report["dim"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--pt", default="/tmp/reid/yolo26n-cls.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--quant", default="f16")
    ap.add_argument("--probe", default="build_app/bin/reid_probe")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    boxes = np.asarray(
        [
            [40, 60, 140, 220],
            [300, 100, 460, 300],
            [180, 200, 260, 330],
        ],
        dtype=np.float32,
    )
    img, _ = build_probe_inputs(224, boxes)

    gg_rows, gg_lat = aicore_embed(args.probe, args.gguf, args.device, img,
                                   boxes, 224)
    pt_rows = torch_baseline(args.pt, img, boxes, 224)

    cos = [
        float(
            np.dot(gg_rows[i], pt_rows[i])
            / (np.linalg.norm(gg_rows[i]) * np.linalg.norm(pt_rows[i]) + 1e-12)
        )
        for i in range(len(boxes))
    ]

    try:
        ort_rows = ort_baseline(args.pt, img, boxes, 224)
        ort_cos = [
            float(
                np.dot(gg_rows[i], ort_rows[i])
                / (np.linalg.norm(gg_rows[i]) * np.linalg.norm(ort_rows[i]) + 1e-12)
            )
            for i in range(len(boxes))
        ]
        # ONNX Runtime latency on the same geometry.
        sess_t0 = time.perf_counter()
        ort_baseline(args.pt, img, boxes, 224)
        ort_ms = (time.perf_counter() - sess_t0) * 1000 / len(boxes) / 5
    except Exception as e:  # noqa: BLE001
        ort_cos, ort_ms = None, None
        print("ort baseline unavailable:", e)

    report = {
        "task": "reid",
        "quant": args.quant,
        "device": args.device,
        "gguf": args.gguf,
        "dim": int(gg_rows.shape[1]),
        "cos_vs_torch": cos,
        "min_cos_vs_torch": min(cos),
        "ort_cos_vs_ggml": ort_cos,
        "ggml_batch_ms": gg_lat,
        "ort_batch_ms_p50_proxy": ort_ms,
        "verdict": "PASS" if min(cos) > 0.99 else "FAIL",
    }
    print(json.dumps(report, indent=2))
    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
