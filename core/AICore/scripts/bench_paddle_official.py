#!/usr/bin/env python3
"""Official PaddleOCR end-to-end benchmark (the comparison baseline for
core/AICore/tools/bench_ocr.cpp).

Runs the official PP-OCRv4 mobile det+rec pipeline (the same upstream models
the AICore ocr GGUFs are converted from) on the same host/image/protocol:
3 warmups + 10 timed predicts, wall-clock per call, p50/p95 reported.

usage: bench_paddle_official.py <image> [warmups] [iters] [--cpu]
Requires: pip install paddlepaddle-gpu==3.0.0 paddleocr==3.0.1
"""

import json
import math
import statistics
import sys
import time


def emit_text_rows(results):
    """按 bench_ocr 的 TEXT 协议输出最后一次的官方结果。"""
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
            row += 1


def main():
    image = sys.argv[1]
    warmups = int(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith("-") else 3
    iters = int(sys.argv[3]) if len(sys.argv) > 3 and not sys.argv[3].startswith("-") else 10
    use_cpu = "--cpu" in sys.argv

    from paddleocr import PaddleOCR

    kwargs = dict(
        text_detection_model_name="PP-OCRv4_mobile_det",
        text_recognition_model_name="PP-OCRv4_mobile_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        enable_mkldnn=False,
    )
    if use_cpu:
        kwargs["device"] = "cpu"
    ocr = PaddleOCR(**kwargs)

    # warmups (first call builds engines and downloads models)
    for _ in range(warmups):
        ocr.predict(image)

    timings = []
    n_text = 0
    last_results = []
    for _ in range(iters):
        t0 = time.perf_counter()
        results = ocr.predict(image)
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000.0)
        last_results = results
        for res in results:
            texts = res.get("rec_texts", [])
            n_text = max(n_text, len(texts))

    timings.sort()
    p50 = timings[len(timings) // 2]
    p95 = timings[int(0.95 * (len(timings) - 1))]
    emit_text_rows(last_results)
    print(json.dumps({
        "pipeline": "paddleocr-official",
        "device": "cpu" if use_cpu else "gpu",
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
