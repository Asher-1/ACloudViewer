#!/usr/bin/env python3
"""Append SAVPE (visual-prompt encoder) weights to a non-prompt-free YOLOE GGUF.

The upstream ultralytics-ggml converter (cpp_ggml/scripts/convert_yolo_to_gguf.py)
emits the YOLOE text-prompt graph (reprta + world_detect/world_segment) but not
the head's SAVPE module — the Small Adoptable Visual Prompt Encoder that turns
example boxes/masks on the image into [Q, 512] class embeddings (the official
YOLOE visual-prompt mode). This script post-processes a converted GGUF:

  1. copies every KV and tensor byte-exactly,
  2. adds KV "yolo.savpe" = 1,
  3. adds the 28 savpe conv weights/biases from the original .pt checkpoint
     (BN folded, exactly like the Conv wrappers the upstream converter walks),
     with the same dtype policy as the source file (f32 / f16 / q8_0).

The resulting file keeps loading unchanged in every text/prompt-free path; the
AICore yolo backend only consumes the savpe weights when the caller passes
visual prompts (aicore_yolo_options_set_visual_prompts).

Tensor naming (consumed by core/AICore/src/tasks/yolo/yolo_graph.cpp, the
GraphBuilder w(prefix, suffix) = "prefix.suffix" convention):
  savpe.cv1_<i>_0.w/b   level-i 3x3 Conv (semantic branch, conv 1 of 2)
  savpe.cv1_<i>_1.w/b   level-i 3x3 Conv (semantic branch, conv 2 of 2)
  savpe.cv2_<i>.w/b     level-i 1x1 Conv (activation branch)
  savpe.cv3.w/b         plain Conv2d(3*c3 -> embed, 1x1)
  savpe.cv4.w/b         plain Conv2d(3*c3 -> 16, 3x3, pad 1)
  savpe.cv5.w/b         plain Conv2d(1 -> 16, 3x3, pad 1)  (mask branch)
  savpe.cv6_0.w/b       Conv(2*16 -> 16, 3x3) wrapper (BN+SiLU)
  savpe.cv6_1.w/b       plain Conv2d(16 -> 16, 3x3, pad 1)
where <i> is the FPN level index (0 = P3). Weights are stored in torch layout
[out, in, kh, kw] like every other conv "w" tensor the converter writes.

Usage:
  python3 core/AICore/src/tasks/yolo/tools/convert_yoloe_savpe_gguf.py \
      --gguf yoloe-26s-seg-f16.gguf --model yoloe-26s-seg.pt \
      [--output yoloe-26s-seg-f16.gguf]   # default: overwrite --gguf in place
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path

import numpy as np

Q8_BLOCK = 32

# savpe tensor names, in the layout above (weights + biases).
SAVPE_TENSOR_NAMES = (
    [f"savpe.cv1_{i}_{j}.{k}" for i in range(3) for j in range(2) for k in ("w", "b")]
    + [f"savpe.cv2_{i}.{k}" for i in range(3) for k in ("w", "b")]
    + [f"savpe.{m}.{k}" for m in ("cv3", "cv4", "cv5", "cv6_0", "cv6_1") for k in ("w", "b")]
)


# ---------------------------------------------------------------------------
# GGUF copy helpers (gguf-py reads back typed KV and raw tensor blocks).
# ---------------------------------------------------------------------------

_ELEM_DTYPE = {
    "UINT8": np.uint8, "INT8": np.int8, "UINT16": np.uint16, "INT16": np.int16,
    "UINT32": np.uint32, "INT32": np.int32, "UINT64": np.uint64, "INT64": np.int64,
    "FLOAT32": np.float32, "FLOAT64": np.float64, "BOOL": np.bool_,
}


def _field_value(field):
    """Decode one ReaderField into a python value (scalar / str / list).

    gguf-py stores every element as its own part: field.data lists the part
    index of each element (verified on a real yoloe-26n-seg GGUF: STRING
    arrays list one part per string, scalar arrays one (1,)-shaped part per
    element). Values decode to python natives — gguf-py's type inference
    rejects numpy scalars."""
    vtype = field.types[0]
    values = [field.parts[i] for i in field.data]
    if vtype.name == "STRING":
        decoded = [bytes(p).rstrip(b"\x00").decode("utf-8") for p in values]
        return decoded[0] if len(decoded) == 1 else decoded
    if vtype.name == "ARRAY":
        etype = field.types[1].name
        if etype == "STRING":
            return [bytes(p).rstrip(b"\x00").decode("utf-8") for p in values]
        dtype = _ELEM_DTYPE[etype]
        return [np.asarray(p, dtype=dtype).item() for p in values]
    if vtype.name == "BOOL":
        return bool(values[0][0])
    return values[0][0].item()


def copy_gguf(reader, writer):
    """Byte-exact copy of every KV and tensor from a GGUFReader.

    Numeric arrays are re-added as homogeneous python lists (not numpy
    arrays): gguf-py 0.19's _pack_val requires collections.abc.Sequence and
    ndarray fails that check, while its GGUFValueType.get_type maps python
    int -> INT32 and float -> FLOAT32 — exactly the element types the
    upstream converter writes (op.inputs/aparams = INT32, yolo.strides =
    FLOAT32). Scalar KVs go through the typed add_* calls so UINT32 vs
    INT32 vs FLOAT32 is preserved exactly."""
    for name, field in reader.fields.items():
        if name.startswith("GGUF."):  # header bookkeeping (version/counts)
            continue
        if name == "general.architecture":  # pre-added by GGUFWriter(arch)
            continue
        vtype = field.types[0]
        if vtype.name == "ARRAY":
            etype = field.types[1].name
            writer.add_array(name, _field_value(field))
        elif vtype.name == "STRING":
            writer.add_string(name, _field_value(field))
        elif vtype.name == "BOOL":
            writer.add_bool(name, _field_value(field))
        elif vtype.name in ("UINT8", "UINT16", "UINT32", "UINT64"):
            getattr(writer, "add_" + vtype.name.lower())(name, _field_value(field))
        elif vtype.name in ("INT8", "INT16", "INT32", "INT64"):
            getattr(writer, "add_" + vtype.name.lower())(name, _field_value(field))
        elif vtype.name in ("FLOAT32", "FLOAT64"):
            getattr(writer, "add_" + vtype.name.lower())(name, _field_value(field))
        else:
            raise ValueError(f"unsupported KV type {vtype.name} for {name}")

    for t in reader.tensors:
        writer.add_tensor(t.name, t.data, raw_dtype=t.tensor_type)


# ---------------------------------------------------------------------------
# savpe weight extraction (needs torch + ultralytics)
# ---------------------------------------------------------------------------

def _conv_pair(mod):
    """Return (weight [out,in,kh,kw], bias|None) of a fused Conv wrapper or a
    bare nn.Conv2d, BN already folded by model.fuse()."""
    conv = mod.conv if hasattr(mod, "conv") else mod
    w = conv.weight.detach().cpu().float().numpy()
    b = None
    if conv.bias is not None:
        b = conv.bias.detach().cpu().float().numpy()
    return w, b


def load_savpe_weights(model_path):
    """Extract the head's SAVPE conv weights from a non-pf YOLOE .pt."""
    try:
        import torch
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            f"error: loading {model_path} needs torch + ultralytics ({exc}); "
            "install them or run this step on the conversion host"
        )

    model = YOLO(model_path).model
    model.eval()
    model.fuse()  # fold Conv+BN (same preprocessing as the upstream converter)
    head = model.model[-1]
    if hasattr(head, "lrpc"):
        raise SystemExit(
            "error: this checkpoint is already prompt-free (LRPC head); it "
            "needs no savpe weights and rejects visual prompts"
        )
    if not hasattr(head, "savpe"):
        raise SystemExit("error: checkpoint head carries no savpe module")
    savpe = head.savpe

    out = {}
    for i, seq in enumerate(savpe.cv1):
        # Sequential(Conv(c_in -> c3, 3), Conv(c3 -> c3, 3), Upsample|Identity)
        out[f"savpe.cv1_{i}_0.w"], out[f"savpe.cv1_{i}_0.b"] = _conv_pair(seq[0])
        out[f"savpe.cv1_{i}_1.w"], out[f"savpe.cv1_{i}_1.b"] = _conv_pair(seq[1])
    for i, seq in enumerate(savpe.cv2):
        # Sequential(Conv(c_in -> c3, 1), Upsample|Identity)
        out[f"savpe.cv2_{i}.w"], out[f"savpe.cv2_{i}.b"] = _conv_pair(seq[0])
    out["savpe.cv3.w"], out["savpe.cv3.b"] = _conv_pair(savpe.cv3)
    out["savpe.cv4.w"], out["savpe.cv4.b"] = _conv_pair(savpe.cv4)
    out["savpe.cv5.w"], out["savpe.cv5.b"] = _conv_pair(savpe.cv5)
    # Sequential(Conv(2c -> c, 3) wrapper, bare Conv2d(c -> c, 3))
    out["savpe.cv6_0.w"], out["savpe.cv6_0.b"] = _conv_pair(savpe.cv6[0])
    out["savpe.cv6_1.w"], out["savpe.cv6_1.b"] = _conv_pair(savpe.cv6[1])
    return out


def pack_savpe_tensor(name, arr, dtype):
    """Apply the upstream write_gguf dtype policy to one savpe tensor."""
    is_weight = name.endswith(".w")
    ndim = arr.ndim
    if dtype == "f32":
        return arr.astype(np.float32), "F32"
    if is_weight and ndim == 4:
        if dtype == "q8_0":
            k = arr.shape[1] * arr.shape[2] * arr.shape[3]
            # Mirror should_quantize: 32-aligned K, no depthwise (in == 1).
            if arr.shape[1] > 1 and k % Q8_BLOCK == 0:
                flat = np.ascontiguousarray(arr.reshape(arr.shape[0], -1))
                import gguf

                q = gguf.quants.quantize(flat, gguf.GGMLQuantizationType.Q8_0)
                return q, "Q8_0"
        return arr.astype(np.float16), "F16"
    if is_weight and ndim == 2:
        # (kept for symmetry: bare matmul weights would follow the f16 rule)
        return arr.astype(np.float16 if dtype in ("f16", "q8_0") else np.float32), "F16" if dtype in ("f16", "q8_0") else "F32"
    # biases and 1-D tensors stay F32 (upstream parity)
    return arr.astype(np.float32), "F32"


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--gguf", required=True, help="non-pf YOLOE GGUF produced by convert_yolo_to_gguf.py")
    ap.add_argument("--model", help="matching yoloe-*-seg.pt checkpoint (savpe weight source)")
    ap.add_argument("--output", help="output GGUF (default: overwrite --gguf in place)")
    args = ap.parse_args()

    import gguf

    src = Path(args.gguf)
    dst = Path(args.output) if args.output else src
    if not src.exists():
        raise SystemExit(f"error: {src} not found")
    if dst.exists() and dst.resolve() == src.resolve() and args.output is None:
        # in-place: write to a sibling then atomically replace
        dst = src.with_suffix(".gguf.savpe.tmp")

    reader = gguf.GGUFReader(str(src))
    dtype = "f16"
    for name, field in reader.fields.items():
        if name == "yolo.dtype":
            dtype = _field_value(field)
            break

    old_tensors = {t.name for t in reader.tensors}
    overlap = old_tensors.intersection(SAVPE_TENSOR_NAMES)
    if overlap:
        raise SystemExit(f"error: source GGUF already carries savpe tensors: {sorted(overlap)[:3]} ...")

    writer = gguf.GGUFWriter(str(dst), "yolo")
    copy_gguf(reader, writer)
    writer.add_uint32("yolo.savpe", 1)

    if args.model:
        weights = load_savpe_weights(args.model)
        missing = [n for n in SAVPE_TENSOR_NAMES if n not in weights]
        if missing:
            raise SystemExit(f"error: checkpoint savpe extraction missing {missing[:3]}")
    else:
        raise SystemExit("error: --model is required (savpe weights come from the .pt checkpoint)")

    n_quant = n_f16 = n_f32 = 0
    for name in SAVPE_TENSOR_NAMES:
        arr, kind = pack_savpe_tensor(name, weights[name], dtype)
        if kind == "Q8_0":
            writer.add_tensor(name, arr, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
            n_quant += 1
        elif kind == "F16":
            writer.add_tensor(name, arr, raw_dtype=gguf.GGMLQuantizationType.F16)
            n_f16 += 1
        else:
            writer.add_tensor(name, arr, raw_dtype=gguf.GGMLQuantizationType.F32)
            n_f32 += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    # Integrity report: byte size + SHA-256 (the ecvAssetDigests registry pins
    # released model assets by SHA-256, so a re-published GGUF needs a new
    # digest entry — print it here for the release checklist).
    data = dst.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    print(f"[write] {dst}: +{len(SAVPE_TENSOR_NAMES)} savpe tensors "
          f"(q8_0={n_quant} f16={n_f16} f32={n_f32}), {len(data)} bytes")
    print(f"[sha256] {digest}  {dst.name}")

    if dst.suffix == ".tmp":
        os.replace(dst, src)
        print(f"[in-place] replaced {src}")


if __name__ == "__main__":
    sys.exit(main())
