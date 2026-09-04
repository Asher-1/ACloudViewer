#!/usr/bin/env python3
"""Convert a pinned COLMAP LoMa ONNX model into an AICore GGUF container.

This conversion program is intentionally outside the product build. It needs
``onnx``, ``numpy``, ``torch`` (DeDoDe-G only), and ``gguf`` only on the controlled conversion host; the
ACloudViewer runtime loads the resulting GGUF through ggml and never links
ONNX Runtime. Tensor names and shapes are retained verbatim so the ggml LoMa
graph can be checked against the audited source graph rather than a second,
hand-maintained weight naming scheme.
"""

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path


SOURCES_PATH = Path(__file__).with_name("loma_sources.json")


def load_sources():
    with SOURCES_PATH.open(encoding="utf-8") as source:
        return json.load(source)["models"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shape(value_info):
    return [dim.dim_value or dim.dim_param
            for dim in value_info.type.tensor_type.shape.dim]


def gguf_tensor_name(name: str) -> str:
    # GGUF accepts arbitrary UTF-8 names, but a stable ASCII namespace makes
    # diagnostics and cross-platform tooling predictable.
    return "loma." + re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def attribute_value(attribute):
    """Return a JSON-safe, lossless-enough ONNX attribute description.

    LoMa lowering must be driven by the exact exported graph rather than a
    hand-maintained operator sequence. Tensor-valued attributes are not used
    by the pinned LoMa models; reject them here so a future exporter change
    cannot silently produce a GGUF graph that the ggml runtime misreads.
    """
    from onnx import AttributeProto

    if attribute.type == AttributeProto.INT:
        return attribute.i
    if attribute.type == AttributeProto.FLOAT:
        return attribute.f
    if attribute.type == AttributeProto.STRING:
        return attribute.s.decode("utf-8")
    if attribute.type == AttributeProto.INTS:
        return list(attribute.ints)
    if attribute.type == AttributeProto.FLOATS:
        return list(attribute.floats)
    if attribute.type == AttributeProto.STRINGS:
        return [value.decode("utf-8") for value in attribute.strings]
    raise SystemExit(
        f"unsupported ONNX attribute type {attribute.type} for {attribute.name}")


def graph_node(node):
    return json.dumps(
        {
            "op": node.op_type,
            "name": node.name,
            "inputs": list(node.input),
            "outputs": list(node.output),
            "attributes": {
                attribute.name: attribute_value(attribute)
                for attribute in node.attribute
            },
        },
        separators=(",", ":"),
        sort_keys=True,
    )


def matcher_hparams(model, variant):
    """Return the explicit LoMa matcher contract for the ggml runner.

    These values are derived from the pinned ONNX graph and are deliberately
    emitted into GGUF instead of being inferred from a tensor-name convention
    at load time.  Detector and descriptor graphs have different contracts.
    """
    if not variant.startswith("matcher_"):
        return {}
    initializers = {item.name: item for item in model.graph.initializer}
    qkv = initializers.get("model.transformers.0.self_attn.Wqkv.weight")
    if qkv is None or len(qkv.dims) != 2 or qkv.dims[0] != 3 * qkv.dims[1]:
        raise SystemExit("LoMa matcher has an invalid first self-attention QKV weight")
    descriptor_dimension = qkv.dims[1]
    input_projection = initializers.get("model.input_proj.weight")
    input_dimension = (input_projection.dims[1]
                       if input_projection is not None else descriptor_dimension)
    if input_projection is not None and list(input_projection.dims) != [
            descriptor_dimension, input_dimension]:
        raise SystemExit("LoMa matcher has an invalid input projection")
    frequencies = [item for item in model.graph.initializer
                   if list(item.dims) == [2, 32]]
    if len(frequencies) != 1:
        raise SystemExit("LoMa matcher must have exactly one [2, 32] RoPE frequency")
    frequency = frequencies[0]
    head_dimension = 2 * frequency.dims[1]
    if descriptor_dimension % head_dimension != 0:
        raise SystemExit("LoMa matcher descriptor dimension is not divisible by RoPE head size")
    head_count = descriptor_dimension // head_dimension
    blocks = {
        int(match.group(1))
        for item in model.graph.initializer
        for match in [re.match(r"model\.transformers\.(\d+)\.", item.name)]
        if match
    }
    if not blocks or blocks != set(range(len(blocks))):
        raise SystemExit("LoMa matcher transformer blocks are not contiguous")
    projection = initializers.get(
        f"model.log_assignment.{len(blocks) - 1}.final_proj.weight")
    if projection is None or list(projection.dims) != [descriptor_dimension,
                                                        descriptor_dimension]:
        raise SystemExit("LoMa matcher has an invalid final projection")
    return {
        "loma.matcher.input_dimension": input_dimension,
        "loma.matcher.descriptor_dimension": descriptor_dimension,
        "loma.matcher.attention.head_count": head_count,
        "loma.matcher.block_count": len(blocks),
        "loma.matcher.layer_norm_epsilon": 1e-5,
        "loma.matcher.filter_threshold": 0.1,
        "loma.matcher.rope_frequency_tensor": gguf_tensor_name(frequency.name),
    }


def descriptor_g_hparams(model, variant):
    """Expose the DINOv2 ViT-L lowering contract without duplicating tensors.

    The exported DeDoDe-G ONNX graph gives the VGG and decoder parameters
    stable module names, but the four MatMul weights in each DINO block are
    anonymous ``val_*`` initializers.  Those names are exporter details, so
    record their graph-derived mapping in GGUF metadata.  The production ggml
    runner never parses ONNX and can reject a changed export deterministically.
    """
    if variant != "descriptor_dedode_g":
        return {}
    blocks = []
    nodes = model.graph.node
    for index, node in enumerate(nodes):
        if node.op_type != "LayerNormalization" or len(node.input) < 3:
            continue
        match = re.fullmatch(
                r"desc\.encoder\.frozen_dinov2\.dinov2_vitl14\.blocks\.(\d+)"
                r"\.norm1\.weight", node.input[1])
        if not match:
            continue
        block = int(match.group(1))
        # Each exported pre-norm DINO block is fixed: norm1, qkv projection,
        # attention, output projection, norm2, fc1, GELU, fc2.
        if index + 30 >= len(nodes):
            raise SystemExit("truncated DeDoDe-G DINOv2 block")
        qkv = nodes[index + 1]
        proj = nodes[index + 18]
        norm2 = nodes[index + 22]
        fc1 = nodes[index + 23]
        fc2 = nodes[index + 30]
        if (qkv.op_type != "MatMul" or proj.op_type != "MatMul" or
                norm2.op_type != "LayerNormalization" or
                fc1.op_type != "MatMul" or fc2.op_type != "MatMul" or
                len(qkv.input) != 2 or len(proj.input) != 2 or
                len(fc1.input) != 2 or len(fc2.input) != 2):
            raise SystemExit(
                    f"unsupported DeDoDe-G DINOv2 block topology at {block}")
        expected_norm2 = (
                "desc.encoder.frozen_dinov2.dinov2_vitl14.blocks."
                f"{block}.norm2.weight")
        if norm2.input[1] != expected_norm2:
            raise SystemExit(f"unexpected DeDoDe-G norm2 tensor at block {block}")
        blocks.append((block, qkv.input[1], proj.input[1],
                       fc1.input[1], fc2.input[1]))
    if [block[0] for block in blocks] != list(range(24)):
        raise SystemExit("DeDoDe-G must expose contiguous ViT-L blocks [0, 24)")
    return {
        "loma.descriptor_g.embedding_dimension": 1024,
        "loma.descriptor_g.attention.head_count": 16,
        "loma.descriptor_g.block_count": len(blocks),
        "loma.descriptor_g.patch_size": 14,
        "loma.descriptor_g.layer_norm_epsilon": 1e-6,
        "loma.descriptor_g.qkv_weights": [gguf_tensor_name(block[1])
                                            for block in blocks],
        "loma.descriptor_g.attention_output_weights": [
            gguf_tensor_name(block[2]) for block in blocks],
        "loma.descriptor_g.mlp_fc1_weights": [gguf_tensor_name(block[3])
                                                for block in blocks],
        "loma.descriptor_g.mlp_fc2_weights": [gguf_tensor_name(block[4])
                                                for block in blocks],
    }


def descriptor_g_position_embedding(initializers):
    """Precompute DINOv2's fixed 784px positional grid exactly once.

    The source DINOv2 implementation requests bicubic interpolation with the
    scale factor ``56.1 / 37`` to avoid a floating-point output-size edge
    case. ggml's resize operator derives its scale from integral source and
    destination dimensions (56 / 37), which is observably different. LoMa's
    current DeDoDe-G contract accepts only 784x784 images, so serializing the
    exact 56x56 grid is both smaller in runtime work and faithful to COLMAP's
    source graph.
    """
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "DeDoDe-G conversion needs torch on the controlled conversion host: "
            f"{exc}") from exc
    from onnx import numpy_helper

    initializer = initializers.get(
        "desc.encoder.frozen_dinov2.dinov2_vitl14.pos_embed")
    if initializer is None:
        raise SystemExit("DeDoDe-G is missing DINOv2 positional embeddings")
    position = numpy_helper.to_array(initializer)
    if position.shape != (1, 1370, 1024):
        raise SystemExit(
            f"unexpected DeDoDe-G positional embedding shape: {position.shape}")
    patches = torch.from_numpy(position[:, 1:].copy()).reshape(1, 37, 37, 1024)
    patches = patches.permute(0, 3, 1, 2)
    scale = 56.1 / 37.0
    grid = torch.nn.functional.interpolate(
        patches, scale_factor=(scale, scale), mode="bicubic")
    if tuple(grid.shape) != (1, 1024, 56, 56):
        raise SystemExit(f"unexpected DeDoDe-G positional grid shape: {tuple(grid.shape)}")
    # ggml tensors are feature-major here: after reshape_2d(1024, 56*56),
    # each feature plane must be contiguous in the same order as Conv output.
    # PyTorch's [C,H,W] array therefore needs an explicit [H,W,C] transpose
    # before GGUF serialization (GGUF reverses dimensions when exposing ne[]).
    return grid[0].permute(1, 2, 0).contiguous().numpy()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sha256",
                        help="Override only for a newly audited upstream asset")
    parser.add_argument("--variant", required=True,
                        help="COLMAP variant, e.g. matcher_B or detector")
    args = parser.parse_args()

    sources = load_sources()
    source = sources.get(args.variant)
    if source is None and args.sha256 is None:
        raise SystemExit(
            f"unknown LoMa variant {args.variant!r}; add its COLMAP-pinned "
            "source to loma_sources.json")
    expected_digest = args.sha256 or source["sha256"]
    if source is not None and args.sha256 is not None and \
            args.sha256.lower() != source["sha256"].lower():
        raise SystemExit(
            f"{args.variant} SHA-256 conflicts with the COLMAP-pinned source")

    actual_digest = sha256(args.model)
    if actual_digest.lower() != expected_digest.lower():
        raise SystemExit(
            f"digest mismatch: expected {expected_digest}, got {actual_digest}")

    try:
        import numpy as np
        import onnx
        from onnx import numpy_helper
        import gguf
    except ImportError as exc:
        raise SystemExit(
            "conversion needs development-only packages onnx, numpy, and gguf: "
            f"{exc}") from exc

    model = onnx.load(str(args.model), load_external_data=False)
    initializers = {initializer.name: initializer for initializer in model.graph.initializer}
    operators = Counter(node.op_type for node in model.graph.node)
    inputs = {initializer.name for initializer in model.graph.initializer}
    graph_inputs = [value for value in model.graph.input if value.name not in inputs]

    writer = gguf.GGUFWriter(str(args.output), "loma")
    writer.add_string("general.name", f"colmap-loma-{args.variant}")
    writer.add_string("loma.source.format", "onnx")
    writer.add_string("loma.source.sha256", actual_digest)
    if source is not None:
        writer.add_string("loma.source.filename", source["file"])
    writer.add_string("loma.variant", args.variant)
    writer.add_uint32("loma.onnx.ir_version", model.ir_version)
    writer.add_array("loma.onnx.opsets", [
        f"{item.domain or 'ai.onnx'}:{item.version}"
        for item in model.opset_import
    ])
    writer.add_array("loma.graph.inputs", [
        json.dumps({"name": value.name, "shape": shape(value)}, separators=(",", ":"))
        for value in graph_inputs
    ])
    writer.add_array("loma.graph.outputs", [
        json.dumps({"name": value.name, "shape": shape(value)}, separators=(",", ":"))
        for value in model.graph.output
    ])
    writer.add_array("loma.graph.operators", [
        f"{name}:{count}" for name, count in sorted(operators.items())
    ])
    # Store the source topology alongside the weights. The runtime never opens
    # ONNX: it reads this checked representation and lowers it to ggml. Keeping
    # the individual nodes makes a model-export change a visible, reviewable
    # contract change rather than an opaque new GGUF blob.
    writer.add_array("loma.graph.nodes", [graph_node(node)
                                           for node in model.graph.node])

    hparams = matcher_hparams(model, args.variant)
    hparams.update(descriptor_g_hparams(model, args.variant))
    for key, value in hparams.items():
        if isinstance(value, int):
            writer.add_uint32(key, value)
        elif isinstance(value, float):
            writer.add_float32(key, value)
        elif isinstance(value, list):
            writer.add_array(key, value)
        else:
            writer.add_string(key, value)

    # ggml's GGUF tensor dimensions are reversed relative to NumPy/ONNX. The
    # source MatMul weights therefore need an explicit transpose so ggml sees
    # [input, output] for mul_mat (the serialized shape is [output, input]).
    dino_linear_names = set()
    precomputed_tensors = {}
    if args.variant == "descriptor_dedode_g":
        for key in ("loma.descriptor_g.qkv_weights",
                    "loma.descriptor_g.attention_output_weights",
                    "loma.descriptor_g.mlp_fc1_weights",
                    "loma.descriptor_g.mlp_fc2_weights"):
            dino_linear_names.update(name.removeprefix("loma.")
                                     for name in hparams[key])
        precomputed_tensors["loma.descriptor_g.pos_embed_56x56"] = (
            descriptor_g_position_embedding(initializers))

    for name, tensor in precomputed_tensors.items():
        writer.add_tensor(name, tensor)

    for initializer in model.graph.initializer:
        tensor = numpy_helper.to_array(initializer)
        if tensor.dtype == np.float64:
            tensor = tensor.astype(np.float32)
        if tensor.dtype not in (np.float32, np.float16, np.int64, np.int32,
                                np.int8, np.uint8, np.bool_):
            raise SystemExit(
                f"unsupported initializer type {tensor.dtype} for {initializer.name}")
        if initializer.name in dino_linear_names:
            if tensor.ndim != 2:
                raise SystemExit(
                    f"DeDoDe-G MatMul weight {initializer.name} is not rank 2")
            tensor = tensor.T.copy()
        writer.add_tensor(gguf_tensor_name(initializer.name), tensor)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    print(f"wrote {args.output} from {args.model}")
    print(f"sha256={actual_digest} initializers={len(model.graph.initializer)} "
          f"nodes={len(model.graph.node)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
