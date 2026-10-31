#!/usr/bin/env python3
"""Export scored CDLA predictions from the pinned original Paddle model.

External assets are provisioned by the caller. This accuracy-only process
executes PaddleOCR's original postprocessor and observes NMS output scores;
it does not duplicate decoding, alter predictions or measure performance.
The caller must require a normal process exit before consuming the export.
"""

import argparse
import gc
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from layout_gt_eval import CDLA_LABELS, digest, verified_path
from run_paddle_layout_reference import preprocess

POSTPROCESS_SHA256 = "36f2de7434c74c4c11d2a6b06b0e514bb412c720f9bec2c785f37d7cabffeb3d"
POSTPROCESS_SOURCE = ("https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/"
                      "v2.7.0/ppocr/postprocess/picodet_postprocess.py")


def scored_regions(module, processor, image, tensor, heads):
    """Observe scores discarded by the original final dict construction.

    hard_nms returns Nx5 boxes+scores in the same class-major sequence used
    by PicoDetPostProcess. Its original arrays, order and results are left
    untouched. A fresh source process owns this non-concurrent observation.
    """
    original, scores = module.hard_nms, []

    def observe(*args, **kwargs):
        selected = original(*args, **kwargs)
        scores.extend(float(value) for value in selected[:, 4])
        return selected

    module.hard_nms = observe
    try:
        results = processor(image, tensor, heads)
    finally:
        module.hard_nms = original
    if len(scores) != len(results):
        raise ValueError("official NMS/result score coverage mismatch")
    labels = [label.lower().replace(" ", "_") for label in CDLA_LABELS]
    return [dict(class_id=labels.index(row["label"]), score=score,
                 bbox=[float(v) for v in row["bbox"]])
            for row, score in zip(results, scores)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("image-root", "pins", "source-root", "source-pins", "model",
                 "postprocess-py", "report"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1 or digest(args.postprocess_py) != POSTPROCESS_SHA256:
        parser.error("invalid threads or original postprocessor SHA-256")
    distribution = "paddlepaddle-gpu" if args.device == "cuda" else "paddlepaddle"
    if importlib.metadata.version(distribution) != "3.0.0":
        parser.error("independent source requires Paddle 3.0.0")
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    import cv2
    import paddle
    from paddle.inference import Config, create_predictor
    from run_ocr_paddle_pipeline import verified_source, source_digests, reference_identity

    pins = json.loads(args.pins.read_text())
    if pins["labels"] != list(CDLA_LABELS) or not pins["pages"]:
        raise ValueError("invalid CDLA validation selection/labels")
    names = [page["image"]["path"] for page in pins["pages"]]
    if len(set(names)) != len(names):
        raise ValueError("duplicate CDLA page")
    files = verified_source(args.source_root,
        json.loads(args.source_pins.read_text()), "layout", args.model)
    dependencies = {str(path): digest(path) for path in (
        Path(__file__), args.pins, args.source_pins, args.postprocess_py,
        args.model, *files.values(), Path(digest.__code__.co_filename),
        Path(preprocess.__code__.co_filename), Path(verified_source.__code__.co_filename))}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("official_picodet", args.postprocess_py)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with tempfile.TemporaryDirectory(prefix="cdla-labels-") as directory:
        labels = Path(directory) / "labels.txt"
        labels.write_text("\n".join(label.lower().replace(" ", "_")
                                   for label in CDLA_LABELS) + "\n")
        processor = module.PicoDetPostProcess(str(labels),
            score_threshold=.5, nms_threshold=.5)
    config = Config(str(files["program"]), str(files["parameters"]))
    if args.device == "cuda":
        config.enable_use_gpu(256, 0)
    else:
        config.disable_gpu()
    config.switch_ir_optim(False)
    config.disable_mkldnn()
    config.disable_glog_info()
    config.set_cpu_math_library_num_threads(args.threads)
    predictor = create_predictor(config)
    inputs, outputs = predictor.get_input_names(), predictor.get_output_names()
    if len(inputs) != 1 or len(outputs) != 8:
        raise ValueError("unexpected original layout graph inputs/outputs")
    handle = predictor.get_input_handle(inputs[0])
    report = dict(device=args.device, source_files=source_digests(dict(layout=files)),
        **reference_identity(), pins_sha256=digest(args.pins), dependencies=dependencies,
        postprocess_source=POSTPROCESS_SOURCE, postprocess_sha256=POSTPROCESS_SHA256,
        score_threshold=.5, nms_threshold=.5, images=[],
        scope="Original graph page accuracy only; CPU is diagnostic; no release timing claim")
    for index, page in enumerate(pins["pages"]):
        path = verified_path(args.image_root, page["image"])
        verified_path(args.image_root, page["annotation"])
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"cannot decode {path.name}")
        tensor = preprocess(image)
        handle.reshape(list(tensor.shape))
        handle.copy_from_cpu(tensor)
        predictor.run()
        heads = [predictor.get_output_handle(name).copy_to_cpu() for name in outputs]
        if any(value.dtype != np.float32 or not np.isfinite(value).all() for value in heads):
            raise ValueError("invalid original graph output")
        regions = scored_regions(module, processor, image, tensor,
            dict(boxes=heads[:4], boxes_num=heads[4:]))
        report["images"].append(dict(image=path.name, sha256=page["image"]["sha256"],
            width=int(image.shape[1]), height=int(image.shape[0]), regions=regions))
        if (index + 1) % 50 == 0:
            print(json.dumps(dict(pages=index + 1, planned=len(pins["pages"]))), flush=True)
    del handle, predictor, config
    gc.collect()
    if any(digest(Path(path)) != sha for path, sha in dependencies.items()):
        raise ValueError("layout reference dependency changed during export")
    args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
