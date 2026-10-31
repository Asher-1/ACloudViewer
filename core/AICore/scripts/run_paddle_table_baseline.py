#!/usr/bin/env python3
"""Official PaddleX table-recognition baseline for the AICore `table` GT matrix.

Runs the PaddleX ``table_recognition`` (v1) pipeline with the same models the
AICore GGUFs are converted from - SLANet_plus structure, PP-OCRv4 mobile
det/rec - and the same pipeline options the AICore row uses: no document
preprocessor, no layout detection (each GT image is one table), text
detection ``limit_type=max`` / ``limit_side_len=960`` as in the official
pipeline config. One JSON line per image:

    {"image": "<file name>", "html": "<html>...", "ms": <predict wall>}

Composed pages (``--layout-reference``): the page layout is the official
PP-Structure CDLA output pinned by ``run_paddle_layout_reference.py`` (the
model the AICore layout GGUF is converted from; PaddleX ships no CDLA
layout module). It is handed to the pipeline as ``layout_det_res`` with
``use_layout_detection=False``, so PaddleX runs exactly its layout branch:
full-page OCR, one ``CropByBoxes`` crop per ``table`` region, structure on
the crop, OCR boxes matched inside the region. Every table is emitted:

    {"image": ..., "html": <first table>, "tables": [{"box": [...],
     "html": ...}], "ms": ...}

usage: run_paddle_table_baseline.py --image-root <dir> --list <images.txt>
                                    --output <paddle.jsonl>
                                    --det-gguf <ppocr-det-ch-f32.gguf>
                                    --rec-gguf <ppocr-rec-ch-f32.gguf>
                                    [--device gpu|cpu]

The det/rec weights are the exact files the given GGUFs were converted
from (``paddle_source_models.py``: digest-verified against
``ocr.source.sha256``). The PaddleX default ``PP-OCRv4_mobile_det`` is a
different det checkpoint (its 16 neck SE convolutions differ), so a
baseline on it would not be a same-model reference.
Requires: paddlepaddle(-gpu)==3.0.0, paddlex==3.0.1 (paddleocr==3.0.1).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

SOURCE_CACHE = Path.home() / ".cache" / "aicore_paddle_sources"


def source_dirs(det_gguf: Path, rec_gguf: Path) -> tuple[str, str]:
    from paddle_source_models import source_model_dir
    return (str(source_model_dir(det_gguf, SOURCE_CACHE)),
            str(source_model_dir(rec_gguf, SOURCE_CACHE)))


def pipeline_config(det_model_dir: str | None = None,
                    rec_model_dir: str | None = None) -> dict:
    return {
        "pipeline_name": "table_recognition",
        "use_doc_preprocessor": False,
        "use_layout_detection": False,
        "use_ocr_model": True,
        "SubModules": {
            "TableStructureRecognition": {
                "module_name": "table_structure_recognition",
                "model_name": "SLANet_plus",
                "model_dir": None,
            },
        },
        "SubPipelines": {
            "GeneralOCR": {
                "pipeline_name": "OCR",
                "text_type": "general",
                "use_doc_preprocessor": False,
                "use_textline_orientation": False,
                "SubModules": {
                    "TextDetection": {
                        "module_name": "text_detection",
                        "model_name": "PP-OCRv4_mobile_det",
                        "model_dir": det_model_dir,
                        "limit_side_len": 960,
                        "limit_type": "max",
                        "max_side_limit": 4000,
                        "thresh": 0.3,
                        "box_thresh": 0.6,
                        "unclip_ratio": 1.5,
                    },
                    "TextRecognition": {
                        "module_name": "text_recognition",
                        "model_name": "PP-OCRv4_mobile_rec",
                        "model_dir": rec_model_dir,
                        "batch_size": 1,
                        "score_thresh": 0,
                    },
                },
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--list", type=Path, required=True,
                        help="image paths relative to --image-root, one per line")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="gpu")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--layout-reference", type=Path,
                        help="layout_regions.json-format official layout per page")
    parser.add_argument("--det-gguf", type=Path, required=True,
                        help="det GGUF whose source weights the baseline runs")
    parser.add_argument("--rec-gguf", type=Path, required=True,
                        help="rec GGUF whose source weights the baseline runs")
    args = parser.parse_args()

    layouts: dict[str, dict] = {}
    if args.layout_reference:
        reference = json.loads(args.layout_reference.read_text(encoding="utf-8"))
        for page in reference["images"]:
            layouts[page["image"]] = {"boxes": [
                {"cls_id": r["class_id"], "label": r["label"], "score": 1.0,
                 "coordinate": r["bbox"]} for r in page["regions"]]}

    from paddlex import create_pipeline
    pipeline = create_pipeline(config=pipeline_config(*source_dirs(args.det_gguf,
                                                                   args.rec_gguf)),
                               device=args.device)
    images = [line.strip() for line in args.list.read_text(encoding="utf-8").splitlines()
              if line.strip()]
    def predict(path: Path):
        layout = layouts.get(path.name) if layouts else None
        if layouts and layout is None:
            raise SystemExit(f"{path.name}: no official layout in --layout-reference")
        return next(iter(pipeline.predict(
            str(path), use_doc_orientation_classify=False, use_doc_unwarping=False,
            use_layout_detection=False, use_ocr_model=True, layout_det_res=layout)))
    for _ in range(args.warmups if images else 0):
        predict(args.image_root / images[0])
    with args.output.open("w", encoding="utf-8") as out:
        for name in images:
            t0 = time.perf_counter()
            result = predict(args.image_root / name)
            ms = (time.perf_counter() - t0) * 1000.0
            tables = result["table_res_list"]
            html = tables[0]["pred_html"] if tables else ""
            record = {"image": name, "html": html, "ms": round(ms, 2)}
            if layouts:
                # The pipeline emits one result per layout box labelled
                # "table", in layout order (the result carries no box).
                boxes = [b["coordinate"] for b in layouts[Path(name).name]["boxes"]
                         if b["label"].lower() == "table"]
                if len(boxes) != len(tables):
                    raise SystemExit(f"{name}: {len(tables)} tables for "
                                     f"{len(boxes)} layout table boxes")
                record["tables"] = [{"box": box, "html": t["pred_html"]}
                                    for box, t in zip(boxes, tables)]
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()


if __name__ == "__main__":
    main()
