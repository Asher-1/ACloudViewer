#!/usr/bin/env python3
"""Compare AICore and PaddleOCR TEXT benchmark outputs.

The inputs are the stdout captures of bench_ocr and bench_paddle_official.py.
Rows are paired one-to-one by nearest center within --max-center-distance; the
comparison deliberately consumes the final OCR text protocol rather than an
internal tensor probe.

usage:
    compare_ocr_text.py <aicore-output> <paddle-output>
                        [--max-center-distance 30] [--require-identical-rate R]
"""

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path


TEXT_ROW = re.compile(
    r"^TEXT\s+(?P<index>\d+)\s+(?P<cx>-?\d+(?:\.\d+)?)\s+"
    r"(?P<cy>-?\d+(?:\.\d+)?)\s+(?P<width>\d+(?:\.\d+)?)x"
    r"(?P<height>\d+(?:\.\d+)?)(?:\s(?P<text>.*))?$"
)


@dataclass(frozen=True)
class TextRow:
    index: int
    cx: float
    cy: float
    width: float
    height: float
    text: str


def read_text_rows_from_output(output: str, source: str = "output") -> list[TextRow]:
    rows = []
    for line_number, raw in enumerate(output.splitlines(), 1):
        if not raw.startswith("TEXT "):
            continue
        match = TEXT_ROW.match(raw)
        if not match:
            raise ValueError(f"{source}:{line_number}: malformed TEXT row: {raw!r}")
        fields = match.groupdict()
        rows.append(TextRow(
            index=int(fields["index"]),
            cx=float(fields["cx"]), cy=float(fields["cy"]),
            width=float(fields["width"]), height=float(fields["height"]),
            text=fields["text"] or ""))
    return rows


def read_text_rows(path: Path) -> list[TextRow]:
    return read_text_rows_from_output(path.read_text(encoding="utf-8"), str(path))


def match_rows(candidate: list[TextRow], reference: list[TextRow], max_distance: float):
    """Return one-to-one nearest-center pairs, ordered by increasing distance.

    O(candidate * reference) is intentional: benchmark outputs contain roughly
    one hundred text rows, so exhaustive candidate edges avoid an extra index
    dependency and make ties deterministic.
    """
    candidates = []
    for candidate_index, ours in enumerate(candidate):
        for reference_index, upstream in enumerate(reference):
            distance = math.hypot(ours.cx - upstream.cx, ours.cy - upstream.cy)
            if distance <= max_distance:
                candidates.append((distance, candidate_index, reference_index))
    candidates.sort(key=lambda edge: (edge[0], edge[1], edge[2]))

    used_candidate = set()
    used_reference = set()
    pairs = []
    for distance, candidate_index, reference_index in candidates:
        if candidate_index in used_candidate or reference_index in used_reference:
            continue
        used_candidate.add(candidate_index)
        used_reference.add(reference_index)
        pairs.append((candidate[candidate_index], reference[reference_index], distance))
    return pairs


def compare(candidate: list[TextRow], reference: list[TextRow], max_distance: float):
    pairs = match_rows(candidate, reference, max_distance)
    identical = sum(ours.text == upstream.text for ours, upstream, _ in pairs)
    nonempty = sum(bool(ours.text) for ours in candidate)
    matched = len(pairs)
    return {
        "protocol": "ocr-text-nearest-center-v1",
        "max_center_distance_px": max_distance,
        "aicore_rows": len(candidate),
        "paddle_rows": len(reference),
        "matched_rows": matched,
        "aicore_unmatched_rows": len(candidate) - matched,
        "paddle_unmatched_rows": len(reference) - matched,
        "aicore_nonempty_text_rate": nonempty / len(candidate) if candidate else 0.0,
        "spatial_precision": matched / len(candidate) if candidate else 0.0,
        "spatial_recall": matched / len(reference) if reference else 0.0,
        "identical_text_rows": identical,
        "identical_text_rate_on_matches": identical / matched if matched else 0.0,
        "pairs": [
            {"aicore_index": ours.index, "paddle_index": upstream.index,
             "center_distance_px": round(distance, 3),
             "identical": ours.text == upstream.text,
             "aicore_text": ours.text, "paddle_text": upstream.text}
            for ours, upstream, distance in pairs
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aicore_output", type=Path)
    parser.add_argument("paddle_output", type=Path)
    parser.add_argument("--max-center-distance", type=float, default=30.0)
    parser.add_argument("--require-identical-rate", type=float, default=None,
                        help="fail if matched-row exact-text rate is below this value")
    args = parser.parse_args()
    if args.max_center_distance <= 0:
        parser.error("--max-center-distance must be positive")
    if args.require_identical_rate is not None and not 0 <= args.require_identical_rate <= 1:
        parser.error("--require-identical-rate must be in [0, 1]")

    report = compare(read_text_rows(args.aicore_output),
                     read_text_rows(args.paddle_output),
                     args.max_center_distance)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    if (args.require_identical_rate is not None and
            report["identical_text_rate_on_matches"] < args.require_identical_rate):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
