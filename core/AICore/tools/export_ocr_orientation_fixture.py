#!/usr/bin/env python3
"""Export real upright/rotated word crops and independent classifier truth.

Conversion/test dependencies only. Source ONNX digest is mandatory. Outputs
belong under a build/temp directory, never the source tree.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from export_onnx_ocr_fixture import preprocess, reference_session


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("words", type=Path, help="PaddleOCR words_en directory")
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--rec-source", type=Path)
    parser.add_argument("--rec-source-sha256")
    parser.add_argument("--rec-weights", type=Path,
                        help="额外诊断同 GGUF 存储权重，不替换独立 source 文本真值")
    parser.add_argument("--dictionary", type=Path)
    parser.add_argument("--dictionary-sha256")
    parser.add_argument("--provider", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1: parser.error("thread count must be positive")
    if hashlib.sha256(args.model.read_bytes()).hexdigest() != args.source_sha256:
        raise SystemExit("classifier source digest mismatch")
    has_rec = bool(args.rec_source)
    if args.rec_weights and not has_rec:
        parser.error("存储诊断必须同时提供 pinned recognizer source 与 dictionary")
    if any((args.rec_source, args.rec_source_sha256, args.dictionary,
            args.dictionary_sha256)) and not all((args.rec_source,
            args.rec_source_sha256, args.dictionary, args.dictionary_sha256)):
        parser.error("recognizer truth requires source/dictionary and both digests")
    if has_rec:
        if hashlib.sha256(args.rec_source.read_bytes()).hexdigest() != args.rec_source_sha256:
            raise SystemExit("recognizer source digest mismatch")
        if hashlib.sha256(args.dictionary.read_bytes()).hexdigest() != args.dictionary_sha256:
            raise SystemExit("recognizer dictionary digest mismatch")
    session = reference_session(str(args.model), args.provider, args.threads)
    stored = None
    if has_rec:
        recognizer = reference_session(str(args.rec_source), args.provider, args.threads)
        characters = [""] + args.dictionary.read_text(encoding="utf-8").splitlines() + [" "]
        if args.rec_weights:
            from onnx_ocr_reference import stored_weight_reference
            graph, storage = stored_weight_reference(args.rec_source, args.rec_weights)
            stored = reference_session(graph.SerializeToString(), args.provider, args.threads)
            stored_identity = {
                "container_sha256": hashlib.sha256(args.rec_weights.read_bytes()).hexdigest(),
                "storage_elements": storage, "provider": stored.get_providers(),
                "provider_options": stored.get_provider_options()}
            del graph
    crops, provenance = [], []
    for i in range(1, 6):
        path = args.words / f"word_{i}.png"
        image = cv2.imread(str(path))
        if image is None: raise SystemExit(f"cannot decode {path}")
        provenance.append({"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        crops.extend([image, image[::-1, ::-1]])
    width = max(crop.shape[1] for crop in crops) + 16
    height = sum(crop.shape[0] + 6 for crop in crops)
    board = np.full((height, width, 3), 255, np.uint8)
    rows, y = [], 0
    for index, crop in enumerate(crops):
        h, w = crop.shape[:2]
        board[y:y+h, :w] = crop
        # Width/height destination corners make the perspective crop an
        # identity sample, including BORDER_REPLICATE at its unused edge.
        quad = [[0, y], [w, y], [w, y+h], [0, y+h]]
        probability = session.run(None, {session.get_inputs()[0].name: preprocess(crop, "cls")})[0][0]
        row = {"quad": quad, "angle": int(probability.argmax()) * 180,
               "score": float(probability.max())}
        rows.append(row)
        y += h + 6
    if not has_rec:
        # Compatibility for the earlier v5/v6 classifier-only fixture protocol.
        # New manifest rows always supply independent recognizer truth.
        for index in (2, 3): rows[index]["text"] = "yourself"
    else:
        # Independent source CTC truth for this recognizer, after the classifier's
        # strict .9 rotation gate. Language-specific models need their own truth.
        corrected = [crop[::-1, ::-1] if row["angle"] == 180 and row["score"] > .9
                     else crop for crop, row in zip(crops, rows)]
        order = np.argsort([crop.shape[1] / crop.shape[0] for crop in corrected])
        stored_texts = [""] * len(corrected)
        stored_max_diff, matching_steps, total_steps = 0.0, 0, 0
        for at in range(0, len(order), 6):
            indices = order[at:at + 6]
            inputs = [preprocess(corrected[i], "rec") for i in indices]
            width = max(x.shape[3] for x in inputs)
            batch = np.zeros((len(inputs), 3, 48, width), np.float32)
            for n, x in enumerate(inputs): batch[n, :, :, :x.shape[3]] = x[0]
            predictions = recognizer.run(None, {recognizer.get_inputs()[0].name: batch})[0]
            outputs = [(predictions, None)]
            if stored is not None:
                rounded = stored.run(None, {stored.get_inputs()[0].name: batch})[0]
                if (rounded.shape != predictions.shape or not np.isfinite(rounded).all()
                        or not np.isfinite(predictions).all()):
                    raise SystemExit("stored-weight recognizer has invalid output")
                # 逐批诊断，不积存 raw head；tiny 六条 40×6906 输出及 float64 差值峰值约 40 MiB。
                stored_max_diff = max(stored_max_diff, float(
                    np.abs(rounded.astype(np.float64) - predictions).max()))
                source_ids, stored_ids = predictions.argmax(-1), rounded.argmax(-1)
                matching_steps += int(np.count_nonzero(source_ids == stored_ids))
                total_steps += source_ids.size
                outputs.append((rounded, stored_texts))
            for values, texts in outputs:
                for index, prediction in zip(indices, values):
                    ids = prediction.argmax(-1)
                    keep = np.ones(len(ids), dtype=bool)
                    keep[1:] = ids[1:] != ids[:-1]
                    keep &= ids != 0
                    text = "".join(characters[i] for i in ids[keep])
                    if texts is None:
                        rows[index]["text"] = text
                    else:
                        texts[index] = text
    args.output.mkdir(parents=True, exist_ok=True)
    image_path = args.output / "orientation.png"
    if not cv2.imwrite(str(image_path), board): raise SystemExit("image write failed")
    reference = {"source_sha256": args.source_sha256,
                 "rec_source_sha256": args.rec_source_sha256,
                 "dictionary_sha256": args.dictionary_sha256,
                 "onnxruntime": ort.__version__, "sources": provenance,
                 "providers": session.get_providers(), "provider_options": session.get_provider_options(),
                 "recognizer_providers": recognizer.get_providers() if has_rec else None,
                 "threads": args.threads,
                 "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(), "rows": rows}
    if stored is not None:
        reference["rec_stored_weights"] = {**stored_identity, "texts": stored_texts,
            "source_max_diff": stored_max_diff,
            "source_argmax_match": matching_steps / total_steps,
            "source_text_mismatch_indices": [i for i, (row, text) in enumerate(zip(rows, stored_texts))
                                             if row["text"] != text],
            "scope": "存储误差诊断；原 rows 仍为独立 source 真值，不替代原文本门"}
    (args.output / "orientation.json").write_text(json.dumps(reference, indent=2) + "\n")


if __name__ == "__main__":
    main()
