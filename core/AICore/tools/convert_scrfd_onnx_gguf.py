#!/usr/bin/env python3
"""Convert a pinned insightface SCRFD ONNX detector into a facedetect GGUF.

The facedetect task replays SCRFD detector topologies metadata-driven
(scrfd_graph.cpp routes any GGUF carrying `facedetect.detector.graph`
through run_onnx_graph_multi), so the conversion is a faithful emission of
the ONNX graph as node specs plus raw initializers - the same contract the
published buffalo_m/buffalo_s/antelopev2 packs use. No AICore C++ changes
are needed for a new SCRFD checkpoint.

Node spec contract (antispoof_graph.cpp parse_node, '|' separators):
    Op;name;in1|in2|...;key=value|key=value
Weights resolve as `det.<initializer name>`; the ggml ne is the reversed
ONNX shape over the same row-major bytes.

Supported SCRFD export ops: Conv, Relu, Sigmoid, Add, Mul, MaxPool,
AveragePool, Resize (nearest 2x), Transpose, Reshape.
"""

import argparse
import hashlib
import re
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sanitize(name: str, seen) -> str:
    """GGML tensor names avoid ':', '/', '(' etc.; the published packs use
    pure identifiers. Map every ONNX name to a unique flat one."""
    out = re.sub(r"[^A-Za-z0-9_.\-]", "_", name)
    base = out
    i = 1
    while out in seen and seen[out] != name:
        out = f"{base}.{i}"
        i += 1
    seen[out] = name
    return out


def attrs_dict(node):
    from onnx import AttributeProto

    out = {}
    for a in node.attribute:
        if a.type == AttributeProto.INT:
            out[a.name] = a.ints or a.i
        elif a.type == AttributeProto.INTS:
            out[a.name] = list(a.ints)
        elif a.type == AttributeProto.FLOAT:
            out[a.name] = a.f
        else:
            out[a.name] = None
    return out


def spec_for(node, initializers, rename):
    """Emit the embedded node spec for one ONNX node."""
    op = node.op_type
    # Nodes are keyed by their FIRST OUTPUT tensor name: downstream nodes
    # reference outputs, not node names (matches the published packs).
    name = rename(node.output[0] or node.name)
    ins = [rename(i) for i in node.input]
    attrs = attrs_dict(node)
    if op == "Conv":
        group = int(attrs.get("group", 1) or 1)
        kernel = int(attrs["kernel_shape"][0])
        pads = list(attrs.get("pads", [0, 0, 0, 0]) or [0, 0, 0, 0])
        strides = list(attrs.get("strides", [1, 1]) or [1, 1])
        a = f"g={group}|k={kernel}|p={pads[0]}|s={strides[0]}"
        return f"Conv;{name};{'|'.join(ins)};{a}"
    if op in ("Relu", "Sigmoid"):
        return f"{op};{name};{ins[0]};"
    if op in ("Add", "Mul"):
        return f"{op};{name};{ins[0]}|{ins[1]};"
    if op in ("MaxPool", "AveragePool"):
        kernel = int(attrs["kernel_shape"][0])
        strides = int(list(attrs.get("strides", [kernel]) or [kernel])[0])
        pads = int(list(attrs.get("pads", [0, 0]) or [0, 0])[0])
        return f"{op};{name};{ins[0]};k={kernel}|s={strides}|p={pads}"
    if op == "Resize":
        # The interpreter is a nearest 2x upsample; the SCRFD export only
        # uses it for the FPN top-down 2x path.
        return f"Resize;{name};{ins[0]};"
    if op == "Transpose":
        perm = list(attrs["perm"])
        return f"Transpose;{name};{ins[0]};pm={'/'.join(str(p) for p in perm)}"
    if op == "Reshape":
        shape = np.array(initializers[ins[1]])
        rc = int(shape[-1])
        return f"Reshape;{name};{ins[0]};rc={rc}"
    raise SystemExit(f"unsupported op {op} ({name}); extend the converter")


def main() -> None:
    import gguf  # conversion-host dependency
    import onnx
    from onnx import numpy_helper

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("onnx", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--input-size", type=int, default=640)
    ap.add_argument("--strides", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--num-anchors", type=int, default=2)
    ap.add_argument("--score-thresh", type=float, default=0.5)
    ap.add_argument("--nms-thresh", type=float, default=0.4)
    ap.add_argument("--source-name", default="")
    ap.add_argument("--quant", choices=["f32", "f16", "q8_0"], default="f32",
                    help="weight storage for all tensors (the published "
                         "buffalo packs ship F16; q8_0 uses per-block "
                         "scales)")
    args = ap.parse_args()

    model = onnx.load(str(args.onnx))
    graph = model.graph
    seen = {}
    rename = lambda name: sanitize(name, seen)
    initializers = {
        rename(init.name): numpy_helper.to_array(init)
        for init in graph.initializer
    }

    writer = gguf.GGUFWriter(str(args.output), "facedetect")
    writer.add_string("facedetect.arch", "scrfd")
    writer.add_string("facedetect.detector.kind", "scrfd")
    writer.add_uint32("facedetect.detector.input_size", args.input_size)
    # gguf-py maps python ints to INT32 arrays - the same type the
    # published buffalo packs carry for kv_u32_arr to read.
    writer.add_array("facedetect.detector.strides", args.strides)
    writer.add_uint32("facedetect.detector.num_anchors", args.num_anchors)
    writer.add_float32("facedetect.detector.score_thresh", args.score_thresh)
    writer.add_float32("facedetect.detector.nms_thresh", args.nms_thresh)
    writer.add_string("facedetect.detector.input", graph.input[0].name)
    writer.add_string("facedetect.detector.source_name",
                      args.source_name or args.onnx.stem)
    writer.add_string("facedetect.detector.source_sha256", sha256(args.onnx))

    specs = [spec_for(node, initializers, rename) for node in graph.node]
    writer.add_array("facedetect.detector.graph", specs)
    writer.add_array("facedetect.detector.outputs",
                     [rename(o.name) for o in graph.output])

    # Weight tensors: gguf-py stores dims reversed (GGUF lists the innermost
    # dim first), so passing the ONNX array as-is lands the ggml ne as
    # [KW,KH,Cin,Cout] - the same file layout the buffalo packs ship. The
    # row-major bytes are unchanged and the dtype is widened to F32.
    for init_name, arr in initializers.items():
        flat = np.ascontiguousarray(arr, dtype=np.float32)
        if flat.ndim == 0:
            flat = flat.reshape(1)
        if args.quant == "f16":
            # Store 4-D conv weights as half (the gguf loader dequantizes on
            # read exactly like the F16 buffalo packs). 1-D biases and the
            # GFL scales stay F32: they feed binary ops against f32
            # activations, and the CPU kernels reject f32+f16 operands.
            if flat.ndim == 4:
                writer.add_tensor(
                    "det." + init_name, flat.astype(np.float16),
                    raw_dtype=gguf.GGMLQuantizationType.F16)
            else:
                writer.add_tensor(
                    "det." + init_name, flat,
                    raw_dtype=gguf.GGMLQuantizationType.F32)
        elif args.quant == "q8_0":
            # Fold to 2-D (out, in*kh*kw) for %32 block alignment, then
            # record the reshaped name so a runtime can restore the layer
            # shape if needed (same convention as the ocr converter).
            import gguf.quants
            two_d = flat.reshape(flat.shape[0], -1) if flat.ndim > 1 \
                else flat.reshape(1, -1)
            q = gguf.quants.Q8_0.quantize(two_d)
            writer.add_tensor("det." + init_name, q,
                              raw_dtype=gguf.GGMLQuantizationType.Q8_0)
        else:
            writer.add_tensor("det." + init_name, flat,
                              raw_dtype=gguf.GGMLQuantizationType.F32)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {args.output} with {len(specs)} nodes and "
          f"{len(graph.initializer)} tensors")


if __name__ == "__main__":
    main()
