#!/usr/bin/env python3
"""Convert a PaddleOCR inference model into an AICore GGUF container.

First-party conversion path for the AICore `ocr` task. Like the LoMa
converter, this program lives outside the product build: it needs
``paddlepaddle``, ``numpy`` and ``gguf`` only on the controlled conversion
host, and the ACloudViewer runtime loads the resulting GGUF through ggml
without linking Paddle or ONNX Runtime.

Tensor names are retained verbatim behind the ``ocr.`` namespace so the ggml
graph can be checked against the audited source graph rather than a second,
hand-maintained weight naming scheme. Recognition dictionaries are embedded
as GGUF string arrays (`ocr.dictionary`) so the CTC decoder needs no side
files. Each run emits F32 plus optionally F16 and Q8_0 siblings; tensors
whose last dimension is not a multiple of 32 stay F16 in the Q8_0 file (ggml
block quantization requires that alignment) and the decision is recorded in
the `ocr.quant.reshaped` metadata.
"""

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_paddle_tensors(model_dir: Path):
    """Return {tensor_name: float32 ndarray} from a Paddle inference dir.

    Delegates to the dependency-free ``paddle_ocr_extract`` parser; the
    paddle runtime is deliberately not imported (its Python bindings have
    been observed to crash on this conversion host).
    """
    sys.path.insert(0, str(Path(__file__).parent))
    from paddle_ocr_extract import load_ocr_tensors
    named, feed_names, fetch_names = load_ocr_tensors(model_dir)
    tensors = {name: array for name, array in named}
    if not tensors:
        raise SystemExit(f"no tensors extracted from {model_dir}")
    return tensors, feed_names, fetch_names


def gguf_tensor_name(name: str) -> str:
    return "ocr." + name


def quantized(name: str, tensor: np.ndarray, kind: str,
              reshaped, keep_f32=()):
    """Return (array, ggml_type) for the requested container quant."""
    import gguf
    # Matmul weights sit on the B side of ggml_mul_mat; the CPU kernel only
    # converts an F32 src1 and quantization is honored on A only.
    if keep_f32:
        return tensor, gguf.GGMLQuantizationType.F32
    if kind == "f32":
        return tensor, gguf.GGMLQuantizationType.F32
    # F16 storage caps at 65504 and Q8_0's per-block scale is an F16, so a
    # tensor with values beyond that range (the det head BN's frozen
    # variance reaches 1e7+) would turn into Inf and silently zero the
    # affected layer. Keep such tensors in F32 inside the quantized
    # container; GGUF tensor types are per tensor.
    magnitude = (float(np.abs(tensor.astype(np.float64)).max())
                 if tensor.size else 0.0)
    if magnitude > 60000.0 or not np.isfinite(tensor).all():
        return tensor, gguf.GGMLQuantizationType.F32
    if kind == "f16":
        return tensor.astype(np.float16), gguf.GGMLQuantizationType.F16
    if kind == "q8_0":
        if tensor.ndim >= 2 and tensor.size:
            # Standard ggml conv storage: fold trailing kernel dims into
            # the input dimension so block quantization sees an aligned
            # axis; the graph builder reshapes back to the layer shape.
            folded = int(np.prod(tensor.shape[1:]))
            if folded % 32 == 0:
                reshaped.append(name + ":" +
                                ",".join(str(d) for d in tensor.shape))
                return (gguf.quants.Q8_0.quantize(
                            tensor.reshape(tensor.shape[0], folded)),
                        gguf.GGMLQuantizationType.Q8_0)
        return tensor.astype(np.float16), gguf.GGMLQuantizationType.F16
    raise SystemExit(f"unknown quant {kind}")


def write_gguf(tensors, metadata, output: Path, kind: str,
               reshaped, keep_f32=()):
    import gguf
    writer = gguf.GGUFWriter(str(output), "ocr")
    for key, value in metadata.items():
        if isinstance(value, list):
            if value and isinstance(value[0], str):
                writer.add_array(key, value)
            else:
                writer.add_array(key, [np.int32(v) for v in value])
        elif isinstance(value, float):
            writer.add_float32(key, value)
        elif isinstance(value, int):
            writer.add_uint32(key, value)
        else:
            writer.add_string(key, value)
    # Quantize first so the reshape registry is complete before the
    # metadata block is written (the ocr.quant.reshaped array records
    # every folded q8_0 kernel and must be present in the file).
    # A native 2-D weight (paddle 1x1 conv) carries no kernel dims in the
    # GGUF shape; register it so the builder can restore the 4-D layer
    # shape regardless of storage type.
    for name, tensor in tensors.items():
        if tensor.ndim == 2:
            reshaped.append(name + ":" +
                            ",".join(str(d) for d in tensor.shape))
    quantized_tensors = {
        name: quantized(name, tensor, kind, reshaped,
                        keep_f32=name in keep_f32)
        for name, tensor in tensors.items()
    }
    if reshaped:
        writer.add_array("ocr.quant.reshaped", reshaped)
    for name, (array, ggml_type) in quantized_tensors.items():
        writer.add_tensor(gguf_tensor_name(name), array,
                          raw_dtype=ggml_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def main() -> None:
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path,
                        help="Paddle inference dir containing "
                             "inference.pdmodel/.pdiparams")
    parser.add_argument("output", type=Path, help="output F32 GGUF stem; "
                        "siblings get -f16/-q8_0 suffixes")
    parser.add_argument("--role", required=True,
                        choices=["det", "rec", "layout"])
    parser.add_argument("--arch", required=True,
                        help="source architecture id, e.g. lcnetv3-db, "
                             "svtr-lcnet")
    parser.add_argument("--source-name", required=True,
                        help="upstream archive name for provenance")
    parser.add_argument("--dictionary", type=Path, default=None,
                        help="recognition dictionary (one token per line); "
                             "embedded as ocr.dictionary")
    parser.add_argument("--labels", type=Path, default=None,
                        help="layout class labels (one per line); embedded "
                             "as ocr.labels")
    parser.add_argument("--quants", default="f32,f16,q8_0")
    parser.add_argument("--graph-json", type=Path, default=None,
                        help="op topology JSON (export_ocr_graph.py) "
                             "embedded as ocr.graph.ops so the runtime "
                             "executes the audited source graph")
    parser.add_argument("--no-emit-siblings", action="store_true",
                        help="write only the F32 container at `output`")
    args = parser.parse_args()

    params = args.model_dir / "inference.pdiparams"
    if not params.exists():
        candidates = sorted(args.model_dir.glob("*.pdiparams"))
        if not candidates:
            raise SystemExit(f"no .pdiparams under {args.model_dir}")
        params = candidates[0]
    source_digest = sha256(params)

    tensors, feed_names, fetch_names = load_paddle_tensors(args.model_dir)

    dictionary = None
    if args.dictionary:
        dictionary = [line.rstrip("\n")
                      for line in args.dictionary.read_text(
                              encoding="utf-8").splitlines() if line]
        # PP-OCR CTC decodes blank at index 0 and appends a trailing space
        # class; the runtime reconstructs that ordering, so only the raw
        # vocabulary is stored.
        if not dictionary:
            raise SystemExit(f"empty dictionary {args.dictionary}")

    metadata = {
        "general.name": f"ppocr-{args.role}-{args.arch}",
        "ocr.source.format": "paddle-inference",
        "ocr.source.name": args.source_name,
        "ocr.source.sha256": source_digest,
        "ocr.role": args.role,
        "ocr.arch": args.arch,
        "ocr.graph.inputs": json.dumps(feed_names),
        "ocr.graph.outputs": json.dumps(fetch_names),
    }
    if args.labels is not None:
        metadata["ocr.labels"] = [
                line.rstrip("\n") for line in args.labels.read_text(
                        encoding="utf-8").splitlines() if line.strip()]
        print(f"labels: {len(metadata['ocr.labels'])} classes")
    if dictionary is not None:
        metadata["ocr.dictionary"] = dictionary
        metadata["ocr.dictionary.size"] = len(dictionary)
    matmul_y_weights = set()
    if args.graph_json:
        graph = json.loads(args.graph_json.read_text(encoding="utf-8"))
        metadata["ocr.graph.ops"] = json.dumps(
            graph, separators=(",", ":"))
        ops = graph["ops"] if isinstance(graph, dict) else graph
        metadata["ocr.graph.op_count"] = len(ops)
        # Matmul weights sit on the B side of ggml_mul_mat; the CPU kernel
        # requires an F32 src1 (quantization is honored on A only).
        for op in ops:
            if op.get("type") != "matmul_v2":
                continue
            for io in op.get("inputs", []):
                if io.get("param") == "Y":
                    for arg in io.get("args", []):
                        if arg:
                            # tensor dict keys carry the bare Paddle name
                            matmul_y_weights.add(arg)

    kinds = [k.strip() for k in args.quants.split(",") if k.strip()]
    outputs = []
    if args.no_emit_siblings:
        targets = [(args.output, "f32")]
    else:
        stem = args.output.with_suffix("") if args.output.suffix else args.output
        targets = [(Path(f"{stem}-{k}.gguf"), k) for k in kinds]
    reshaped = []
    for path, kind in targets:
        write_gguf(tensors, metadata, path, kind, reshaped, keep_f32=matmul_y_weights)
        outputs.append(path)
        print(f"wrote {path} ({kind}, {len(tensors)} tensors)")


if __name__ == "__main__":
    main()
