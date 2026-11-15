#!/usr/bin/env python3
# YOLO upstream-vs-AICore bench regression gate.
#
# Joins the upstream ultralytics-ggml bench matrix
# (cpp_ggml/benchmarks/bench.jsonl: backend/model/task/dtype rows with
# preprocess/graph/post/e2e ms stats) against the AICore-side JSONL emitted by
# tests/yolo/test_yolo_capi_performance.cpp, then reports the p50 regression of
# graph_ms and e2e_ms per (model, task, dtype, backend) cell.
#
# Gate (ultralytics-ggml-integration-plan.md §12.5): a cell FAILS when its
# e2e_ms p50 regression exceeds --limit (default 5%). Exit code 1 on any FAIL,
# 0 when everything is within the gate, 2 on usage/coverage errors.
#
# Field contract (kept 1:1 with upstream cpp_ggml/src/cli.cpp bench):
#   upstream row: {"backend","model","task","dtype","imgsz","threads",...,
#                  "graph_ms":{"p50"},"e2e_ms":{"p50"}}
#   aicore row:   {"file","task","dtype","device","threads",...,
#                  "graph_ms":{"p50"},"e2e_ms":{"p50"}}
import argparse
import json
import sys

DEFAULT_LIMIT_PCT = 5.0


def norm_backend(name):
    n = (name or "").lower()
    if "cuda" in n:
        return "cuda"
    if "vulkan" in n:
        return "vulkan"
    if "metal" in n:
        return "metal"
    if n in ("cpu", ""):
        return "cpu"
    return n


def model_of_upstream(row):
    return row.get("model", "")


def model_of_aicore(row):
    stem = row.get("file", "").rsplit(".", 1)[0]
    for suffix in ("-f16", "-f32", "-q8_0"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"warning: skipping malformed JSONL line: {exc}",
                      file=sys.stderr)
    return rows


def p50(row, stage):
    stats = row.get(stage) or {}
    if "p50" in stats:
        return float(stats["p50"])
    if "mean" in stats:
        return float(stats["mean"])
    return None


def main():
    ap = argparse.ArgumentParser(
            description="Upstream-vs-AICore YOLO bench regression gate")
    ap.add_argument("--upstream", required=True,
                    help="upstream bench.jsonl (ultralytics-ggml cpp_ggml)")
    ap.add_argument("--aicore", required=True,
                    help="AICore JSONL from test_yolo_capi_performance")
    ap.add_argument("--limit", type=float, default=DEFAULT_LIMIT_PCT,
                    help="max allowed e2e p50 regression in percent "
                         "(default: %(default)s)")
    ap.add_argument("--graph-limit", type=float, default=None,
                    help="optional separate gate for graph_ms p50 (default: "
                         "same as --limit)")
    args = ap.parse_args()
    graph_limit = args.graph_limit if args.graph_limit is not None \
        else args.limit

    upstream = {}
    for row in load_jsonl(args.upstream):
        key = (model_of_upstream(row), row.get("task"), row.get("dtype"),
               norm_backend(row.get("backend")))
        upstream[key] = row

    aicore = {}
    for row in load_jsonl(args.aicore):
        key = (model_of_aicore(row), row.get("task"), row.get("dtype"),
               norm_backend(row.get("device")))
        aicore[key] = row

    missing = sorted(set(upstream) - set(aicore))
    extra = sorted(set(aicore) - set(upstream))

    failures = []
    print(f"{'model':<18}{'task':<9}{'dtype':<6}{'backend':<8}"
          f"{'up e2e':>9}{'ai e2e':>9}{'e2e d%':>8}"
          f"{'up gr':>8}{'ai gr':>8}{'gr d%':>8}  verdict")
    for key in sorted(set(upstream) & set(aicore)):
        model, task, dtype, backend = key
        up, ai = upstream[key], aicore[key]
        up_e2e, ai_e2e = p50(up, "e2e_ms"), p50(ai, "e2e_ms")
        up_gr, ai_gr = p50(up, "graph_ms"), p50(ai, "graph_ms")

        def delta(up_v, ai_v):
            if up_v is None or ai_v is None or up_v <= 0:
                return None
            return (ai_v - up_v) / up_v * 100.0

        d_e2e = delta(up_e2e, ai_e2e)
        d_gr = delta(up_gr, ai_gr)
        verdict = "ok"
        if d_e2e is None:
            verdict = "n/a"
        else:
            if d_e2e > args.limit:
                verdict = "FAIL"
                failures.append((key, "e2e_ms", d_e2e))
            elif d_gr is not None and d_gr > graph_limit:
                verdict = "FAIL(gr)"
                failures.append((key, "graph_ms", d_gr))

        def fmt(v, w):
            return f"{v:>{w}.2f}" if isinstance(v, (int, float)) \
                else f"{'-':>{w}}"

        print(f"{model:<18}{task:<9}{dtype:<6}{backend:<8}"
              f"{fmt(up_e2e, 9)}{fmt(ai_e2e, 9)}{fmt(d_e2e, 8)}"
              f"{fmt(up_gr, 8)}{fmt(ai_gr, 8)}{fmt(d_gr, 8)}  {verdict}")

    if missing:
        print(f"\n[coverage] {len(missing)} upstream cells missing from "
              f"AICore run (not gated):")
        for key in missing:
            print(f"  {key[0]} {key[1]} {key[2]} {key[3]}")
    if extra:
        print(f"\n[coverage] {len(extra)} AICore-only cells (informational):")
        for key in extra:
            print(f"  {key[0]} {key[1]} {key[2]} {key[3]}")

    if failures:
        print(f"\nGATE FAILED: {len(failures)} cell(s) exceed the "
              f"{args.limit}% p50 regression limit:")
        for (key, stage, d) in failures:
            print(f"  {key[0]} {key[1]} {key[2]} {key[3]}: {stage} "
                  f"regression +{d:.1f}%")
        return 1
    print(f"\nGATE PASSED: all shared cells within {args.limit}% "
          f"(e2e) / {graph_limit}% (graph) p50 regression.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
