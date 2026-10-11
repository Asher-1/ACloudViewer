#!/usr/bin/env python3
"""Export the PP-OCR det graph-level parity fixture for the AICore `ocr` task.

The fixture is one real document image preprocessed exactly like PaddleOCR
DetResizeForTest (limit_type=max, limit_side_len=960, sides rounded to
multiples of 32, cv2 INTER_LINEAR) + NormalizeImage (BGR, ImageNet mean/std,
x / 255) + ToCHW, run through the Paddle Inference predictor of the archive
the det GGUF is converted from (ocr.source.sha256; see
scripts/paddle_source_models.py):

    det_ch_input.npy      (1, 3, H, W) float32 network input
    det_ch_reference.npy  (1, 1, H, W) float32 sigmoid probability map

A real page matters: a random input drives the DB head to ~0 everywhere, so a
max|diff| gate on it cannot fail (the former synthetic fixture peaked at
0.0036 under a 0.01 threshold).

usage: export_det_fixture.py <det_infer_dir> <out_dir> <image>
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def preprocess(image_path: Path, limit: int):
    import cv2
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"cannot decode {image_path}")
    h, w = img.shape[:2]
    ratio = float(limit) / max(h, w) if max(h, w) > limit else 1.0
    rh = max(int(round(h * ratio / 32) * 32), 32)
    rw = max(int(round(w * ratio / 32) * 32), 32)
    img = cv2.resize(img, (rw, rh)).astype(np.float32)
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    img = (img / 255.0 - mean) / std
    return img.transpose((2, 0, 1))[None].astype(np.float32), (w, h)


def run_paddle(model_dir: Path, x: np.ndarray) -> np.ndarray:
    from paddle.inference import Config, create_predictor
    config = Config(str(model_dir / "inference.pdmodel"),
                    str(model_dir / "inference.pdiparams"))
    config.disable_gpu()
    config.switch_ir_optim(False)
    for name in ("disable_onednn", "disable_mkldnn"):
        try:
            getattr(config, name)()
            break
        except Exception:
            continue
    predictor = create_predictor(config)
    handle = predictor.get_input_handle(predictor.get_input_names()[0])
    handle.reshape(list(x.shape))
    handle.copy_from_cpu(x)
    predictor.run()
    return predictor.get_output_handle(
        predictor.get_output_names()[0]).copy_to_cpu().astype(np.float32)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path,
                        help="det inference dir (inference.pdmodel/pdiparams)")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("image", type=Path)
    parser.add_argument("--limit-side-len", type=int, default=960)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    x, size = preprocess(args.image, args.limit_side_len)
    y = run_paddle(args.model_dir, x)
    np.save(args.out_dir / "det_ch_input.npy", x)
    np.save(args.out_dir / "det_ch_reference.npy", y)
    row = {"image": args.image.name, "size": list(size),
           "input_shape": list(x.shape),
           "params_sha256": sha256(args.model_dir / "inference.pdiparams"),
           "prob_max": float(y.max()), "prob_mean": float(y.mean()),
           "text_fraction": float((y > 0.3).mean())}
    for stem in ("det_ch_input", "det_ch_reference"):
        path = args.out_dir / f"{stem}.npy"
        row[stem] = {"sha256": sha256(path), "size_bytes": path.stat().st_size}
    (args.out_dir / "det_fixture.json").write_text(
        json.dumps(row, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(row))


if __name__ == "__main__":
    main()
