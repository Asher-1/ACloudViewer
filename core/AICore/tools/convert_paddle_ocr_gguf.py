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
the `ocr.quant.reshaped` metadata. Tensors listed in `Q8_0_KEEP_F16` for the
source archive also stay F16 in the Q8_0 file (measured sensitivity, see
there).
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


# Q8_0 candidates kept F16 per source archive. Selection rule: quantize each
# candidate alone (all other tensors F16) and keep it F16 when that single
# tensor moves the per-step softmax on the official rec fixture by more than
# 10x the F16 container's own error. en_PP-OCRv4_rec_infer (F16 container
# 5.7e-4): conv2d_139/141/143 (the backbone's first three aligned 1x1
# convs) move it by 0.32/0.016/0.0079, every other candidate by <= 0.0027;
# all 11 quantized gave 0.44 and changed 28% of ICDAR2015 word texts,
# keeping these three gives 0.0039 for +13 KiB. A dequantized-to-F32 copy
# reproduces the 0.44 exactly, so it is weight rounding, not a kernel.
# Additional labelled-page regression (2026-10-19, developer candidates):
# - ch det conv2d_397: the public Q8_0 scene pipeline exceeded the 0.005 CER
#   degradation gate. Keeping this one tensor F16 passes both languages on
#   50 scenes and Chinese SROIE; the English receipt recognizer is separate.
# - en rec conv2d_161: with that detector candidate, public Q8_0 English
#   SROIE fixed-region / pipeline WER deltas were +0.006834 / +0.008936.
#   Keeping this tensor F16 gives -0.005432 / -0.000350, and the CPU scene
#   pipeline CER delta is +0.003530. The other seven single-tensor candidates
#   fail the SROIE gate. These sets selected the policy; they are regression
#   evidence, not an untouched holdout or full backend/performance release.
# Exact files, parameter storage proportions and source identities remain in
# the qAnyLabeling per-file ledger; do not publish by filename alone.
# ch_PP-OCRv4_rec_infer needs none (all 11 quantized: 5.3e-4).
#
# Composed-page sets (2026-10-14, table-page-gt composed TEDS attribution).
# The composed chain (layout crop -> det cell boxes -> rec text -> SLANet
# structure) is knife-edge sensitive: on 7/100 pinned pages a sub-pixel
# layout-region delta crosses the int-truncation crop boundary and the SLANet
# decode flips (e.g. 0.3 px -> 13x3 -> 13x4 cols, -0.22 TEDS on one page), so
# the all-q8_0 row landed 5e-6 below the Paddle-0.005 gate while each stage
# alone passed. Partial convergence re-rolls which gate fails (layout-f32 +
# table/ocr q8_0 fails TEDS-S), so the fix converges every decode-visible
# stage to its F32 container's decisions:
# - picodet layout: the four FPN head convs produce cls/dis; keeping them F16
#   makes region boxes track F32 (all-q8_0 composed per-page structure trees
#   then match the F32 chain on the previously flipping pages).
# - ch det: the DB head path (level inner convs + concat 1x1 + upsample)
#   produces the sigmoid map; q8_0 weight rounding moves transition pixels
#   (fixture max_diff 0.476) and shifts cell boxes -> rec text.
# - SLANet_plus: the GRU decoder weight and its projection linears perturb
#   both token logits and cell polygons (fixture loc max_diff 0.020 with
#   token_match 1.0); the encoder stays quantized. All kept tensors already
#   exist as F16 in the proven F16 containers of the same models.
Q8_0_KEEP_F16 = {
    "en_PP-OCRv4_rec_infer": {"conv2d_139.w_0", "conv2d_141.w_0",
                              "conv2d_143.w_0", "conv2d_161.w_0"},
    # ch rec: the same three early backbone 1x1 convs as en (same SVTR-mobile
    # architecture, shapes (32,64)/(64,64)/(64,128)). Their q8_0 rounding is
    # invisible on the rec fixture (all 11 quantized: 5.3e-4) but flips greedy
    # CTC text on knife-edge composed-page crops (5/100 pages, 0.0025 TEDS).
    "ch_PP-OCRv4_rec_infer": {"conv2d_160.w_0", "conv2d_162.w_0",
                              "conv2d_164.w_0"},
    "picodet_lcnet_x1_0_fgd_layout_cdla_infer": {
        "conv2d_100.w_0", "conv2d_101.w_0",
        "conv2d_102.w_0", "conv2d_103.w_0"},
    "ch_PP-OCRv4_det_infer": {
        "conv2d_397.w_0",
        "conv2d_132.w_0", "conv2d_133.w_0", "conv2d_134.w_0",
        "conv2d_136.w_0", "conv2d_138.w_0", "conv2d_142.w_0",
        "conv2d_144.w_0", "conv2d_148.w_0", "conv2d_150.w_0",
        "conv2d_154.w_0", "conv2d_156.w_0", "conv2d_159.w_0",
        "conv2d_transpose_0.w_0"},
    "SLANet_plus_infer.tar": {
        "gru_cell_0.w_1", "linear_0.w_0", "linear_1.w_0",
        "linear_3.w_0", "linear_5.w_0"},
}


def gguf_tensor_name(name: str) -> str:
    return "ocr." + name


def table_metadata(model_dir: Path):
    """SLANet structure vocabulary and decode constants from inference.yml.

    The yml is the archive's own pre/post-process contract, so the runtime
    reads the exact class order the logits were trained against instead of
    a second hand-copied table: TableLabelDecode appends `<td></td>` and
    drops `<td>` when merge_no_span_structure is set, then wraps the list in
    `sos` / `eos`.
    """
    import yaml
    config = yaml.safe_load((model_dir / "inference.yml").read_text(
            encoding="utf-8"))
    post = config["PostProcess"]
    tokens = list(post["character_dict"])
    if post.get("merge_no_span_structure", False):
        if "<td></td>" not in tokens:
            tokens.append("<td></td>")
        if "<td>" in tokens:
            tokens.remove("<td>")
    tokens = ["sos"] + tokens + ["eos"]
    ops = {next(iter(op)): next(iter(op.values())) or {}
           for op in config["PreProcess"]["transform_ops"]}
    encode = ops["TableLabelEncode"]
    return {
        "ocr.table.tokens": tokens,
        # The exported while loop runs max_text_length + 1 steps.
        "ocr.table.max_steps": int(encode["max_text_length"]) + 1,
        "ocr.table.loc_reg_num": int(encode["loc_reg_num"]),
        "ocr.table.max_len": int(ops["ResizeTableImage"]["max_len"]),
        "ocr.table.pad_size": int(ops["PaddingTableImage"]["size"][0]),
    }


def quantized(name: str, tensor: np.ndarray, kind: str,
              reshaped, keep_f32=(), keep_f16=False):
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
    if kind == "f16" or (kind == "q8_0" and keep_f16):
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
                values = tensor.reshape(tensor.shape[0], folded)
                blocks = values.reshape(-1, 32)
                maxima = np.abs(blocks).max(axis=1)
                # A subnormal F32 scale can overflow the reference
                # quantizer's reciprocal. Its stored F16 scale is zero,
                # so the block necessarily dequantizes to zero. Canonical
                # zero blocks avoid Inf/NaN -> int8 conversions without
                # changing any representable reconstructed weight.
                tiny = (maxima > 0) & (maxima < 127 / np.finfo(np.float32).max)
                if tiny.any():
                    values = values.copy()
                    values.reshape(-1, 32)[tiny] = 0
                return (gguf.quants.Q8_0.quantize(
                            values),
                        gguf.GGMLQuantizationType.Q8_0)
        return tensor.astype(np.float16), gguf.GGMLQuantizationType.F16
    raise SystemExit(f"unknown quant {kind}")


def write_gguf(tensors, metadata, output: Path, kind: str,
               reshaped, keep_f32=(), keep_f16=(), storage_overrides=None):
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
                        keep_f32=name in keep_f32,
                        keep_f16=name in keep_f16)
        for name, tensor in tensors.items()
    }
    if storage_overrides:
        if kind != "q8_0" or storage_overrides.keys() - tensors.keys():
            raise ValueError("invalid Q8_0 storage override ownership")
        for name, (array, storage) in storage_overrides.items():
            if storage != gguf.GGMLQuantizationType.Q8_0:
                raise ValueError("calibrated override must contain Q8_0 storage")
            shape = tensors[name].shape
            if array.dtype != np.uint8 or array.size != tensors[name].size // 32 * 34:
                raise ValueError("invalid calibrated Q8_0 storage size")
            quantized_tensors[name] = (array, storage)
            reshaped.append(name + ":" + ",".join(str(d) for d in shape))
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
                        choices=["det", "rec", "cls", "layout", "table"])
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
    if args.role == "table":
        metadata.update(table_metadata(args.model_dir))
        metadata["general.name"] = f"slanet-table-{args.arch}"
        print(f"table tokens: {len(metadata['ocr.table.tokens'])}")
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
        write_gguf(tensors, metadata, path, kind, reshaped,
                   keep_f32=matmul_y_weights,
                   keep_f16=Q8_0_KEEP_F16.get(args.source_name, ()))
        outputs.append(path)
        print(f"wrote {path} ({kind}, {len(tensors)} tensors)")


if __name__ == "__main__":
    main()
