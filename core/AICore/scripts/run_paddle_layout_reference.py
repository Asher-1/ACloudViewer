#!/usr/bin/env python3
"""Official PP-Structure layout regions for the AICore layout parity gate.

Runs the upstream PicoDet CDLA layout model
(``picodet_lcnet_x1_0_fgd_layout_cdla_infer``, the source of
``pplcnet-layout-*.gguf``) with Paddle Inference and the PP-Structure
``LayoutPredictor`` contract (PaddleOCR release/2.7
``ppstructure/layout/predict_layout.py``):

    cv2 BGR frame -> Resize to 608x800 (cv2 INTER_LINEAR)
    -> NormalizeImage(ImageNet mean/std, scale 1/255, hwc) -> CHW
    -> PicoDetPostProcess(score 0.5, nms 0.5)

The post-process is the upstream ``ppstructure/layout/picodet_postprocess.py``
file itself (``--postprocess-py``), not a re-implementation, so the reference
is independent of the AICore port. The output is one JSON document

    {"model_sha256": ..., "images": [{"image": name, "sha256": ...,
      "width": w, "height": h,
      "regions": [{"label", "class_id", "bbox": [x1,y1,x2,y2]}]}]}

(the upstream result carries no score, so neither does the reference)

consumed by ``check_layout_regions.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np

CDLA_LABELS = ("text", "title", "figure", "figure_caption", "table",
               "table_caption", "header", "footer", "reference", "equation")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preprocess(image: np.ndarray) -> np.ndarray:
    resized = cv2.resize(image, (608, 800))
    mean = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3)).astype("float32")
    std = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3)).astype("float32")
    normalized = (resized.astype("float32") * np.float32(1.0 / 255.0) - mean) / std
    return np.expand_dims(normalized.transpose((2, 0, 1)), 0).copy()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--postprocess-py", type=Path, required=True,
                        help="upstream ppstructure/layout/picodet_postprocess.py")
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--nms-threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu", choices=("cpu", "gpu"))
    parser.add_argument("images", type=Path, nargs="+")
    args = parser.parse_args()

    from paddle import inference  # deferred: only the reference host needs it

    spec = importlib.util.spec_from_file_location("picodet_postprocess",
                                                  args.postprocess_py)
    assert spec and spec.loader
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as labels:
        labels.write("\n".join(CDLA_LABELS) + "\n")
    post = upstream.PicoDetPostProcess(labels.name,
                                       score_threshold=args.score_threshold,
                                       nms_threshold=args.nms_threshold)

    model = args.model_dir / "model.pdmodel"
    params = args.model_dir / "model.pdiparams"
    config = inference.Config(str(model), str(params))
    if args.device == "gpu":
        config.enable_use_gpu(500, 0)
    else:
        config.disable_gpu()
    config.switch_ir_optim(False)  # graph passes must not alter the reference
    config.disable_glog_info()
    predictor = inference.create_predictor(config)
    input_handle = predictor.get_input_handle(predictor.get_input_names()[0])
    outputs = predictor.get_output_names()
    half = len(outputs) // 2

    report = {"model": model.parent.name, "model_sha256": sha256_file(params),
              "postprocess_sha256": sha256_file(args.postprocess_py),
              "score_threshold": args.score_threshold,
              "nms_threshold": args.nms_threshold, "images": []}
    for path in args.images:
        image = cv2.imread(str(path))
        if image is None:
            raise SystemExit(f"cannot decode {path}")
        tensor = preprocess(image)
        input_handle.copy_from_cpu(tensor)
        predictor.run()
        scores = [predictor.get_output_handle(outputs[i]).copy_to_cpu()
                  for i in range(half)]
        boxes = [predictor.get_output_handle(outputs[i + half]).copy_to_cpu()
                 for i in range(half)]
        regions = []
        for result in post(image, tensor, {"boxes": scores, "boxes_num": boxes}):
            regions.append({"label": result["label"],
                            "class_id": CDLA_LABELS.index(result["label"]),
                            "bbox": [round(float(v), 3) for v in result["bbox"]]})
        report["images"].append({"image": path.name, "sha256": sha256_file(path),
                                 "width": int(image.shape[1]),
                                 "height": int(image.shape[0]),
                                 "regions": regions})
        print(f"{path.name}: {len(regions)} regions", flush=True)
    args.output.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
