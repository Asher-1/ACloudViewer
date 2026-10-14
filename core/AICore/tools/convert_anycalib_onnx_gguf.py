#!/usr/bin/env python3
"""Convert the AnyCalib ONNX feed-forward network into an AICore GGUF container.

First-party conversion path for the AICore `anycalib` task (single-image
intrinsics). Like the PaddleOCR converter, this program lives outside the
product build: it needs ``onnx``, ``onnxruntime``, ``numpy`` and ``gguf`` only
on the controlled conversion host, and the ACloudViewer runtime loads the
resulting GGUF through ggml without linking ONNX Runtime.

Source of truth: the pinned upstream release asset
``anycalib_gen_v1.0.0_322x322.onnx``
(SHA256 a3afb0d236bb5021c6c34dda6dcf5377687d39c541d06b779d420aca2e28e0ff,
recorded in libs/Reconstruction/src/calibration/resources.h). The graph is a
traced ``AnyCalibFeedForward`` module (DINOv2 ViT-L/14 backbone + light DPT
decoder + convex tangent head, opset 18) with a static 322x322 input.

Tensor names are retained verbatim behind the ``anycalib.`` namespace so the
ggml graph can be checked against the audited source graph rather than a
second, hand-maintained weight naming scheme. The traced graph stores the big
fused matmul weights (attn.qkv / attn.proj / mlp.fc1 / mlp.fc2) as anonymous
``val_*`` initializers next to their named biases; this converter resolves
each anonymous weight to its bias' name (``.bias`` -> ``.weight``) by walking
the node sequence, and fails loudly if any anonymous fp32 weight stays
unresolved. The input-size-dependent positional-embedding subgraph (canvas
37x37 bicubic-resampled to 23x23 and concatenated with cls+register rows) is
precomputed here with ONNX Runtime into ``anycalib.backbone.pos_embed`` so
the runtime graph needs no Resize/Gather/Slice ops; the canvas tensor is kept
as ``anycalib.backbone.pos_embed_canvas`` for provenance.

Each run emits an F32 and an F16 sibling. Quantized (Q8_0) tiers belong to
the validation manifest work and are intentionally not produced here yet.
A fixed-seed reference input plus the ONNX Runtime ``rays``/``tangent_coords``
outputs are written next to the GGUF as ``*.ref.npz`` for the C++ parity
gate.
"""

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

SOURCE_SHA256 = "a3afb0d236bb5021c6c34dda6dcf5377687d39c541d06b779d420aca2e28e0ff"

# Fixed square network input (round(sqrt(102400) / 14) * 14).
IMAGE_SIZE = 322
PATCH_SIZE = 14
GRID = IMAGE_SIZE // PATCH_SIZE  # 23
# DINOv2 without registers: 1 cls token prepended to the 23x23 patches.
NUM_PREFIX = 1
EMBED_DIM = 1024
DEPTH = 24
NUM_HEADS = 16
MLP_HIDDEN = 4096


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def to_numpy(tensor: onnx.TensorProto) -> np.ndarray:
    return onnx.numpy_helper.to_array(tensor)


def resolve_weight_names(graph: onnx.GraphProto) -> dict[str, str]:
    """Map anonymous ``val_*`` matmul weights to their named bias' weight name.

    The traced sequence per matmul is deterministic:
    ``MatMul(x, val_W) -> Add(val_y, <scope>.<op>.bias)`` where the Add bias
    is a named initializer. ``val_W`` therefore belongs to
    ``<scope>.<op>.weight``. Any anonymous fp32 initializer that cannot be
    resolved this way aborts the conversion: a silently dropped weight would
    surface as a wrong-model runtime, not a conversion error.
    """
    initializers = {tensor.name: tensor for tensor in graph.initializer}
    nodes = list(graph.node)
    resolved: dict[str, str] = {}
    unresolved: set[str] = set()
    for index, node in enumerate(nodes):
        if node.op_type != "MatMul":
            continue
        for weight_name in node.input[1:]:
            if weight_name not in initializers:
                continue
            if not weight_name.startswith("val_"):
                continue
            weight = initializers[weight_name]
            if len(weight.dims) != 2:
                unresolved.add(weight_name)
                continue
            bias_name = None
            for follower in nodes[index + 1:index + 4]:
                if follower.op_type != "Add":
                    continue
                for candidate in follower.input[1:]:
                    if candidate in initializers and candidate.endswith(".bias"):
                        bias_name = candidate
                        break
                if bias_name is not None:
                    break
            if bias_name is None:
                unresolved.add(weight_name)
                continue
            resolved[weight_name] = bias_name[:-len(".bias")] + ".weight"
    if unresolved:
        raise SystemExit(
            "unresolved anonymous weights: " + ", ".join(sorted(unresolved)))
    return resolved


def precompute_pos_embed(model: onnx.ModelProto) -> np.ndarray:
    """Read the resolved pos-embed [1, 1+23^2, 1024] out of the full graph.

    The traced graph resamples the 37x37 canvas to the 23x23 grid at runtime
    with a fixed input size, so the embedding is a constant for this export.
    The full graph validates and runs in ONNX Runtime as-is (verified), so
    the subgraph is not rebuilt: the concatenation output that feeds the
    backbone tokens is appended as an extra graph output and read from one
    reference run. Exactness comes from construction -- ORT executes the
    traced ops themselves.
    """
    graph = model.graph
    pos_embed_name = "backbone.model.pos_embed"
    sub_nodes = []
    reached_concat = False
    for node in graph.node:
        if pos_embed_name in node.input:
            sub_nodes.append(node)
            continue
        if sub_nodes and not reached_concat:
            sub_nodes.append(node)
            if node.op_type == "Concat" and len(node.output) == 1:
                reached_concat = True
                break
    if not reached_concat:
        raise SystemExit("pos-embed subgraph not found in the traced graph")
    resolved_name = sub_nodes[-1].output[0]

    extended = onnx.ModelProto()
    extended.CopyFrom(model)
    extra_output = extended.graph.output.add()
    extra_output.CopyFrom(onnx.helper.make_tensor_value_info(
        resolved_name, onnx.TensorProto.FLOAT, None))

    session = ort.InferenceSession(
        extended.SerializeToString(), providers=["CPUExecutionProvider"])
    image = np.zeros((1, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    feed = {graph.input[0].name: image}
    outputs = session.run([resolved_name], feed)[0]
    expected = NUM_PREFIX + GRID * GRID
    if outputs.shape != (1, expected, EMBED_DIM):
        raise SystemExit(
            f"pos-embed precompute produced {outputs.shape}, "
            f"expected (1, {expected}, {EMBED_DIM})")
    return outputs


def clean_name(name: str) -> str:
    """Keep audited source-graph names, minus redundant ``model.`` segments."""
    return name.replace("backbone.model.", "backbone.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="Output path base; writes <base>.f32.gguf and "
                             "<base>.f16.gguf plus <base>.ref.npz")
    parser.add_argument("--skip-golden", action="store_true",
                        help="Skip the ONNX Runtime reference run")
    args = parser.parse_args()

    digest = sha256(args.onnx)
    if digest != SOURCE_SHA256:
        raise SystemExit(
            f"source sha256 mismatch: {digest} != {SOURCE_SHA256}")

    model = onnx.load(str(args.onnx))
    graph = model.graph

    weights: dict[str, np.ndarray] = {}
    constant_keeps = {"backbone.image_mean", "backbone.image_std"}
    for tensor in graph.initializer:
        name = tensor.name
        if tensor.data_type != onnx.TensorProto.FLOAT:
            continue  # int64 shape constants are replaced by GGUF metadata
        if name.startswith("val_"):
            continue  # resolved separately or dropped with the traced graph
        if (name not in constant_keeps and
                not name.startswith(("backbone.model.blocks.",
                                     "backbone.model.patch_embed.",
                                     "decoder.", "head."))):
            continue  # traced-graph constants (expand, scalars, pos_embed)
        array = to_numpy(tensor).astype(np.float32)
        weights[clean_name(name)] = array

    for anonymous, resolved_name in resolve_weight_names(graph).items():
        tensor = next(t for t in graph.initializer if t.name == anonymous)
        weights[clean_name(resolved_name)] = to_numpy(tensor).astype(np.float32)

    # The canvas positional embedding stays for provenance; the runtime
    # graph consumes the precomputed, input-size-resolved tensor.
    pos_embed = precompute_pos_embed(model)
    canvas = to_numpy(
        next(t for t in graph.initializer
             if t.name == "backbone.model.pos_embed")).astype(np.float32)
    weights["backbone.pos_embed"] = pos_embed
    weights["backbone.pos_embed_canvas"] = canvas

    mean = weights["backbone.image_mean"].reshape(3)
    std = weights["backbone.image_std"].reshape(3)

    golden = None
    if not args.skip_golden:
        session = ort.InferenceSession(str(args.onnx),
                                       providers=["CPUExecutionProvider"])
        rng = np.random.default_rng(20261014)
        image = rng.random((1, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        rays, tangent = session.run(None, {"image": image})
        golden = {"image": image, "rays": rays, "tangent_coords": tangent}

    import gguf

    metadata = {
        "anycalib.arch": "anycalib",
        "anycalib.version": "1.0.0",
        "anycalib.source_sha256": digest,
        "anycalib.source_opset": 18,
        "anycalib.patch_size": PATCH_SIZE,
        "anycalib.image_size": IMAGE_SIZE,
        "anycalib.vit.embed_dim": EMBED_DIM,
        "anycalib.vit.depth": DEPTH,
        "anycalib.vit.num_heads": NUM_HEADS,
        "anycalib.vit.mlp_hidden": MLP_HIDDEN,
        "anycalib.vit.num_prefix_tokens": NUM_PREFIX,
        "anycalib.vit.pos_embed_canvas_rows": 37,
        "anycalib.img.mean": mean.tolist(),
        "anycalib.img.std": std.tolist(),
        "anycalib.head.rays_dim": 3,
        "anycalib.head.tangent_dim": 2,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    for precision, dtype in (("f32", np.float32), ("f16", np.float16)):
        path = args.output.parent / f"{args.output.name}.{precision}.gguf"
        writer = gguf.GGUFWriter(str(path), "anycalib")
        for key, value in metadata.items():
            if isinstance(value, list):
                writer.add_array(key, value)
            elif isinstance(value, int):
                writer.add_uint32(key, value)
            else:
                writer.add_string(key, value)
        for name in sorted(weights):
            writer.add_tensor(name, weights[name].astype(dtype))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        print(f"wrote {path} ({path.stat().st_size / (1 << 20):.1f} MiB, "
              f"{len(weights)} tensors)")

    if golden is not None:
        golden_path = args.output.parent / f"{args.output.name}.ref.npz"
        np.savez_compressed(golden_path, **golden)
        print(f"wrote {golden_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
