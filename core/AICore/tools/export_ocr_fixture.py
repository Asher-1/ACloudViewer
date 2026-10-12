#!/usr/bin/env python3
"""Export the PP-OCR det/rec graph-level parity fixtures for the AICore `ocr`
task from real images.

Each fixture is one official PaddleOCR test image preprocessed exactly like
the official pipeline and run through the Paddle Inference predictor of the
archive the GGUF is converted from (ocr.source.sha256; see
scripts/paddle_source_models.py):

    det  DetResizeForTest (limit_type=max, limit_side_len=960, sides rounded
         to multiples of 32, cv2 INTER_LINEAR) + NormalizeImage (BGR,
         ImageNet mean/std, x / 255) + ToCHW
         -> <stem>_input.npy (1, 3, H, W), <stem>_reference.npy (1, 1, H, W)
            sigmoid probability map
    rec  OCRReisizeNormImg (h=48, resized_w = ceil(48 * w / h), x / 255,
         (x - 0.5) / 0.5, zero pad to the 320 rec_image_shape width)
         -> <stem>_input.npy (1, 3, 48, 320), <stem>_reference.npy
            (1, T, vocab) per-step softmax

Real inputs matter: a uniform random input drives the DB head to ~0
everywhere and the CTC head to the blank class at every step, so the former
synthetic fixtures could not fail (det peaked at 0.0036 under a 0.01 max|diff|
gate; rec decoded 40/40 blank steps under an "argmax sequence match" gate).

usage: export_ocr_fixture.py {det,rec} <infer_dir> <out_dir> <image> [--stem S]
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def preprocess_det(image_path: Path, limit: int):
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


def preprocess_rec(image_path: Path, width: int):
    import cv2
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"cannot decode {image_path}")
    h, w = img.shape[:2]
    resized_w = math.ceil(48 * w / float(h))
    if resized_w > width:
        raise SystemExit(f"{image_path}: resized width {resized_w} exceeds "
                         f"the {width} px fixture width")
    resized = cv2.resize(img, (resized_w, 48)).astype(np.float32)
    resized = (resized.transpose((2, 0, 1)) / 255.0 - 0.5) / 0.5
    x = np.zeros((1, 3, 48, width), np.float32)
    x[0, :, :, :resized_w] = resized
    return x, (w, h)


def run_paddle(model_dir: Path, x: np.ndarray) -> np.ndarray:
    from paddle.inference import Config, create_predictor
    # Legacy archives ship inference.pdmodel; PaddleX 3 dirs whose
    # pdiparams already match the GGUF source ship the PIR inference.json.
    program = model_dir / "inference.pdmodel"
    if not program.exists():
        program = model_dir / "inference.json"
    config = Config(str(program), str(model_dir / "inference.pdiparams"))
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
    parser.add_argument("role", choices=("det", "rec"))
    parser.add_argument("model_dir", type=Path,
                        help="inference dir (inference.pdmodel/pdiparams)")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("image", type=Path)
    parser.add_argument("--stem", default=None,
                        help="file-name stem (default det_ch / rec_ch)")
    parser.add_argument("--limit-side-len", type=int, default=960)
    parser.add_argument("--rec-width", type=int, default=320)
    args = parser.parse_args()
    stem = args.stem or f"{args.role}_ch"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.role == "det":
        x, size = preprocess_det(args.image, args.limit_side_len)
    else:
        x, size = preprocess_rec(args.image, args.rec_width)
    y = run_paddle(args.model_dir, x)
    np.save(args.out_dir / f"{stem}_input.npy", x)
    np.save(args.out_dir / f"{stem}_reference.npy", y)
    row = {"role": args.role, "image": args.image.name, "size": list(size),
           "input_shape": list(x.shape), "reference_shape": list(y.shape),
           "params_sha256": sha256(args.model_dir / "inference.pdiparams")}
    if args.role == "det":
        row.update(prob_max=float(y.max()), prob_mean=float(y.mean()),
                   text_fraction=float((y > 0.3).mean()))
    else:
        steps = y.reshape(-1, y.shape[-1]).argmax(-1)
        row.update(non_blank_steps=int((steps != 0).sum()),
                   steps=int(steps.size))
    for name in (f"{stem}_input", f"{stem}_reference"):
        path = args.out_dir / f"{name}.npy"
        row[name] = {"sha256": sha256(path), "size_bytes": path.stat().st_size}
    (args.out_dir / f"{stem}_fixture.json").write_text(
        json.dumps(row, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(row))


if __name__ == "__main__":
    main()
