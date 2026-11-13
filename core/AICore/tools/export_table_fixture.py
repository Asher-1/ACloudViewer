#!/usr/bin/env python3
"""Export SLANet table-structure parity fixtures for the AICore `table` task.

Each fixture is a real table image preprocessed exactly like PaddleX
TableStructureRecognition (cv2 BGR decode, ResizeByLong(max_len) with
INTER_LINEAR and Python round, per-channel Normalize as x * alpha + beta in
float32, zero pad to pad_size, CHW) and run through the Paddle inference
predictor of the same archive. The outputs are the exported program's two
fetches: per-step structure softmax rows and sigmoid cell-location rows.

    table_<k>_input.npy  (1, 3, pad, pad) float32 network input
    table_<k>_probs.npy  (steps, vocab)    float32 structure probabilities
    table_<k>_loc.npy    (steps, 8)        float32 cell polygons (sigmoid)

The program appends one zero row after the last decoded step (the while
loop's slice end is the post-increment step counter); it is removed so
`steps` equals the decoded length including the eos step.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def preprocess(image_path: Path, max_len: int, pad: int):
    import cv2
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"cannot decode {image_path}")
    h, w = img.shape[:2]
    scale = max_len / max(h, w)
    rh, rw = round(h * scale), round(w * scale)
    img = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_LINEAR)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    planes = list(cv2.split(img))
    for c in range(3):
        planes[c] = planes[c].astype(np.float32)
        planes[c] *= (1.0 / 255.0) / std[c]
        planes[c] += -mean[c] / std[c]
    img = cv2.merge(planes)
    img = cv2.copyMakeBorder(img, 0, pad - rh, 0, pad - rw,
                             cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return img.transpose((2, 0, 1))[None].astype(np.float32), (w, h)


def run_paddle(model_dir: Path, x: np.ndarray):
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
    outs = [predictor.get_output_handle(n).copy_to_cpu()
            for n in predictor.get_output_names()]
    probs = next(o for o in outs if o.shape[-1] != 8)[0]
    loc = next(o for o in outs if o.shape[-1] == 8)[0]
    return probs, loc


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path,
                        help="SLANet inference dir (inference.pdmodel)")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("images", type=Path, nargs="+")
    parser.add_argument("--max-len", type=int, default=488)
    parser.add_argument("--pad", type=int, default=488)
    parser.add_argument("--eos", type=int, default=49)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for k, image in enumerate(args.images):
        x, size = preprocess(image, args.max_len, args.pad)
        probs, loc = run_paddle(args.model_dir, x)
        tokens = probs.argmax(axis=1)
        eos = np.nonzero(tokens == args.eos)[0]
        steps = int(eos[0]) + 1 if eos.size else probs.shape[0]
        if eos.size == 0 or steps == probs.shape[0]:
            raise SystemExit(f"{image}: decoder did not terminate before "
                             "the padding row")
        probs, loc = probs[:steps], loc[:steps]
        stem = args.out_dir / f"table_{k}"
        np.save(f"{stem}_input.npy", x)
        np.save(f"{stem}_probs.npy", probs.astype(np.float32))
        np.save(f"{stem}_loc.npy", loc.astype(np.float32))
        row = {"index": k, "image": image.name, "size": list(size),
               "steps": steps}
        for suffix in ("input", "probs", "loc"):
            path = Path(f"{stem}_{suffix}.npy")
            row[suffix] = {"file": path.name, "sha256": sha256(path),
                           "size_bytes": path.stat().st_size}
        manifest.append(row)
        print(f"{image.name}: {size[0]}x{size[1]} steps={steps}")
    (args.out_dir / "table_fixtures.json").write_text(
        json.dumps(manifest, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
