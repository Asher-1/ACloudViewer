#!/usr/bin/env python3
"""Official PaddleOCR end-to-end benchmark (the comparison baseline for
core/AICore/tools/bench_ocr.cpp).

Runs the official PP-OCRv4 mobile det+rec pipeline on the exact weights the
AICore ocr GGUFs are converted from (``--det-gguf`` / ``--rec-gguf``,
resolved and digest-verified by paddle_source_models.py; the PaddleX default
PP-OCRv4_mobile_det is a different det checkpoint) on the same
host/image/protocol: 3 warmups + 10 timed predicts, wall-clock per call,
p50/p95 reported.

usage: bench_paddle_official.py <image> [warmups] [iters] [--cpu]
                                [--language ch|en] [--recognize-full-image]
                                --det-gguf <det.gguf> --rec-gguf <rec.gguf>
Requires: pip install paddlepaddle-gpu==3.0.0 paddleocr==3.0.1
"""

import json
import math
import statistics
import sys
import time
from pathlib import Path

SOURCE_CACHE = Path.home() / ".cache" / "aicore_paddle_sources"


def emit_recognition_row(result):
    """Emit one full-image quad for cropped-word GT recognition datasets."""
    image = result["input_img"]
    height, width = image.shape[:2]
    text = result.get("rec_text", "")
    score = float(result.get("rec_score", 0.0))
    print(f"TEXT 0 {width / 2:.0f} {height / 2:.0f} {width}x{height} {text}")
    print(f"OCRBOX 0 {score:.6f} 0,0 {width - 1},0 {width - 1},{height - 1} 0,{height - 1}\t{text}")


def emit_text_rows(results):
    """Print the last official result using the bench_ocr TEXT protocol."""
    row = 0
    for result in results:
        texts = result.get("rec_texts", [])
        polygons = result.get("rec_polys", result.get("dt_polys", []))
        for index, text in enumerate(texts):
            if index >= len(polygons) or len(polygons[index]) < 4:
                continue
            quad = polygons[index]
            cx = sum(float(point[0]) for point in quad[:4]) / 4.0
            cy = sum(float(point[1]) for point in quad[:4]) / 4.0
            width = math.hypot(float(quad[1][0]) - float(quad[0][0]),
                               float(quad[1][1]) - float(quad[0][1]))
            height = math.hypot(float(quad[3][0]) - float(quad[0][0]),
                                float(quad[3][1]) - float(quad[0][1]))
            print(f"TEXT {row} {cx:.0f} {cy:.0f} {width:.0f}x{height:.0f} {text}")
            quad_text = " ".join(
                f"{float(point[0]):.3f},{float(point[1]):.3f}"
                for point in quad[:4])
            print(f"OCRBOX {row} 0.000000 {quad_text}\t{text}")
            row += 1


def main():
    image = sys.argv[1]
    warmups = int(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith("-") else 3
    iters = int(sys.argv[3]) if len(sys.argv) > 3 and not sys.argv[3].startswith("-") else 10
    use_cpu = "--cpu" in sys.argv
    recognize_full_image = "--recognize-full-image" in sys.argv
    language = "ch"
    if "--language" in sys.argv:
        index = sys.argv.index("--language")
        if index + 1 >= len(sys.argv) or sys.argv[index + 1] not in {"ch", "en"}:
            raise SystemExit("--language expects ch or en")
        language = sys.argv[index + 1]

    def flag_path(name):
        if name not in sys.argv or sys.argv.index(name) + 1 >= len(sys.argv):
            raise SystemExit(f"{name} <gguf> is required (same-weight baseline)")
        return Path(sys.argv[sys.argv.index(name) + 1])

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from paddle_source_models import source_model_dir
    det_dir = str(source_model_dir(flag_path("--det-gguf"), SOURCE_CACHE))
    rec_dir = str(source_model_dir(flag_path("--rec-gguf"), SOURCE_CACHE))

    from paddleocr import PaddleOCR

    kwargs = dict(
        text_detection_model_name="PP-OCRv4_mobile_det",
        text_detection_model_dir=det_dir,
        text_recognition_model_name=("PP-OCRv4_mobile_rec" if language == "ch"
                                     else "en_PP-OCRv4_mobile_rec"),
        text_recognition_model_dir=rec_dir,
        lang=language,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        enable_mkldnn=False,
    )
    if use_cpu:
        kwargs["device"] = "cpu"
    if recognize_full_image:
        from paddleocr import TextRecognition
        recognizer = TextRecognition(
            model_name=("PP-OCRv4_mobile_rec" if language == "ch"
                        else "en_PP-OCRv4_mobile_rec"),
            model_dir=rec_dir)
        predict = lambda: list(recognizer.predict(image))
    else:
        ocr = PaddleOCR(**kwargs)
        predict = lambda: ocr.predict(image)

    # warmups (first call builds engines and downloads models)
    for _ in range(warmups):
        predict()

    timings = []
    n_text = 0
    last_results = []
    for _ in range(iters):
        t0 = time.perf_counter()
        results = predict()
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000.0)
        last_results = results
        if recognize_full_image:
            n_text = 1 if results else 0
        else:
            for res in results:
                texts = res.get("rec_texts", [])
                n_text = max(n_text, len(texts))

    timings.sort()
    p50 = timings[len(timings) // 2]
    p95 = timings[int(0.95 * (len(timings) - 1))]
    if recognize_full_image:
        if last_results:
            emit_recognition_row(last_results[0])
    else:
        emit_text_rows(last_results)
    print(json.dumps({
        "pipeline": "paddleocr-official",
        "device": "cpu" if use_cpu else "gpu",
        "language": language,
        "recognize_full_image": recognize_full_image,
        "image": image,
        "e2e_p50_ms": round(p50, 2),
        "e2e_p95_ms": round(p95, 2),
        "mean_ms": round(statistics.mean(timings), 2),
        "texts": n_text,
        "warmups": warmups,
        "iters": iters,
    }))


if __name__ == "__main__":
    main()
