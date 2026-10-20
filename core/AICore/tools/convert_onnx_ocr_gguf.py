#!/usr/bin/env python3
"""Convert audited PP-OCR ONNX graphs into the existing AICore OCR runtime.

ONNX is a conversion dependency only. No ONNX Runtime dependency or inference
engine is added to AICore. Unknown operators/attributes fail conversion; source
SHA-256, logical ranks, integer shape constants and exact topology travel in
the GGUF. F32 is the reference container; sibling quantizations use the shared
Paddle converter's storage rules and need independent numerical acceptance.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

from convert_paddle_ocr_gguf import quantized, write_gguf


def validate_source_identity(digest, role, dictionary_digest, assets):
    for asset in assets:
        if asset["sha256"] != digest:
            continue
        if asset.get("role", role) != role:
            raise ValueError("source model role mismatch")
        expected = asset.get("dictionary", {}).get("sha256")
        if expected and dictionary_digest != expected:
            raise ValueError("source recognizer dictionary SHA-256 mismatch")


def op(kind, inputs, output, **attrs):
    return {"type": kind,
            "inputs": [{"param": key, "args": value if isinstance(value, list)
                        else [value]} for key, value in inputs.items()],
            "outputs": [{"param": "Out", "args": [output]}],
            "attrs": [{"name": key, "value": value} for key, value in attrs.items()]}


def lower(model):
    allowed = {
        "Conv": {"auto_pad", "kernel_shape", "strides", "dilations", "pads", "group"},
        "ConvTranspose": {"auto_pad", "kernel_shape", "strides", "dilations", "pads", "group", "output_padding"},
        "BatchNormalization": {"epsilon", "momentum", "training_mode"},
        "HardSigmoid": {"alpha", "beta"}, "Softmax": {"axis"},
        "MaxPool": {"auto_pad", "kernel_shape", "strides", "pads", "ceil_mode"},
        "AveragePool": {"auto_pad", "kernel_shape", "strides", "pads", "ceil_mode", "count_include_pad"},
        "Resize": {"mode", "coordinate_transformation_mode", "nearest_mode"},
        "LayerNormalization": {"axis", "epsilon", "stash_type"},
        "ReduceMean": {"axes", "keepdims"}, "Cast": {"to"},
        "Transpose": {"perm"}, "Squeeze": {"axes"}, "Unsqueeze": {"axes"},
        "Reshape": {"allowzero"}, "Concat": {"axis"}, "Constant": {"value"},
    }
    for node in model.graph.node:
        unknown = {a.name for a in node.attribute} - allowed.get(node.op_type, set())
        if node.domain or unknown:
            raise ValueError(f"unsupported ONNX domain/attributes: {node.op_type} {node.domain} {unknown}")
        a = {item.name: helper.get_attribute_value(item) for item in node.attribute}
        # Validate before constant folding too: an unsupported attribute must
        # not disappear when all the node's operands happen to be constants.
        if node.op_type == "Reshape" and a.get("allowzero", 0):
            raise ValueError("allowzero Reshape is unsupported")
        if node.op_type in ("MaxPool", "AveragePool") and a.get("auto_pad", b"NOTSET") not in (b"NOTSET", b"VALID", b"SAME_UPPER"):
            raise ValueError("unsupported pooling auto_pad")
        if node.op_type == "LayerNormalization" and a.get("stash_type", 1) != 1:
            raise ValueError("only F32 LayerNormalization stash is audited")
    # Some published v5 intermediate annotations describe pre-optimization
    # activation sizes on folded bias constants. Recompute intermediate
    # shapes from operators/initializers rather than trusting stale hints.
    del model.graph.value_info[:]
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    constants = {t.name: numpy_helper.to_array(t).copy()
                 for t in model.graph.initializer}
    ranks, shapes = {}, {}
    for value in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        dims = [int(d.dim_value) if d.HasField("dim_value") else -1
                for d in value.type.tensor_type.shape.dim]
        ranks[value.name], shapes[value.name] = len(dims), dims
    nodes = list(model.graph.node)
    producers = {name: node for node in nodes for name in node.output}
    users = {}
    for node in nodes:
        for name in node.input:
            users.setdefault(name, []).append(node)

    def attrs(node):
        return {a.name: helper.get_attribute_value(a) for a in node.attribute}

    # Constant folding also resolves the many v6 Identity aliases and the
    # v5 conv-bias Reshape nodes. Never fold model activations.
    for node in nodes:
        a, ins = attrs(node), list(node.input)
        if node.op_type == "Constant":
            constants[node.output[0]] = numpy_helper.to_array(a["value"]).copy()
        elif ins and all(name in constants for name in ins if name):
            x = [constants[name] for name in ins if name]
            kind = node.op_type
            value = None
            if kind in ("Identity", "Cast"):
                value = x[0] if kind == "Identity" else x[0].astype(helper.tensor_dtype_to_np_dtype(a["to"]))
            elif kind == "Reshape":
                dims = x[1].astype(np.int64).tolist()
                dims = [x[0].shape[i] if d == 0 else d for i, d in enumerate(dims)]
                value = x[0].reshape(dims)
            elif kind == "Unsqueeze":
                value = np.expand_dims(x[0], tuple(a.get("axes", x[1] if len(x) > 1 else [])))
            elif kind == "Squeeze":
                value = np.squeeze(x[0], tuple(a.get("axes", x[1] if len(x) > 1 else [])))
            elif kind == "Concat":
                value = np.concatenate(x, axis=a["axis"])
            elif kind in ("Add", "Mul", "Div", "Sub", "Pow"):
                value = {"Add": np.add, "Mul": np.multiply, "Div": np.divide,
                         "Sub": np.subtract, "Pow": np.power}[kind](*x)
            if value is not None:
                constants[node.output[0]] = np.asarray(value).copy()

    # ONNX's generic shape inference loses logical ranks in the v6 dynamic
    # Shape/Slice/Concat chains. Propagate ranks from operator semantics;
    # unknown dimensions stay unknown, rather than treating an unknown
    # TensorShapeProto as a scalar.
    for name, value in constants.items():
        ranks[name], shapes[name] = value.ndim, list(value.shape)
    for node in nodes:
        kind, ins, out, a = node.op_type, list(node.input), node.output[0], attrs(node)
        if out in constants: continue
        r = ranks.get(ins[0], 0) if ins else 0
        if kind in ("Conv", "ConvTranspose", "GlobalAveragePool", "MaxPool", "AveragePool", "Resize"):
            r = 4
        elif kind in ("Add", "Mul", "Sub", "Div", "Pow", "MatMul"):
            r = max(ranks.get(name, 0) for name in ins)
        elif kind == "Transpose": r = len(a["perm"])
        elif kind in ("Squeeze", "Unsqueeze"):
            axes = a.get("axes", constants[ins[1]].tolist() if len(ins) > 1 else [])
            r += len(axes) * (1 if kind == "Unsqueeze" else -1)
            if kind == "Unsqueeze" and r == 1: shapes[out] = [1]
        elif kind == "Shape":
            shapes[out], r = [r], 1
        elif kind == "Reshape":
            r = constants[ins[1]].size if ins[1] in constants else shapes[ins[1]][0]
        elif kind == "Slice" and r == 1 and ins[0] in shapes:
            size = shapes[ins[0]][0]
            start, end = int(constants[ins[1]][0]), int(constants[ins[2]][0])
            shapes[out] = [len(range(size)[slice(start, end)])] if size >= 0 else [-1]
        elif kind == "Concat" and r == 1:
            lengths = [shapes.get(name, [-1])[0] for name in ins]
            shapes[out] = [sum(lengths)] if all(x >= 0 for x in lengths) else [-1]
        ranks[out] = int(r)

    def scalar(name, expected):
        return name in constants and constants[name].size == 1 and np.isclose(
            float(constants[name].reshape(-1)[0]), expected, rtol=1e-7, atol=1e-7)

    def binary(node, kind, constant):
        if node is None or node.op_type != kind:
            return None
        for i in range(2):
            if scalar(node.input[i], constant):
                return node.input[1 - i]
        return None

    # Exact erf GELU, verified structurally and numerically against ONNX.
    # ggml_gelu_erf avoids inventing a new custom ggml operator.
    gelus, skipped = {}, set()
    for node in nodes:
        body = binary(node, "Mul", .5)
        mul = producers.get(body)
        if mul is None or mul.op_type != "Mul":
            continue
        for i in range(2):
            x, add_name = mul.input[i], mul.input[1 - i]
            add = producers.get(add_name)
            erf_name = binary(add, "Add", 1)
            erf = producers.get(erf_name)
            div = producers.get(erf.input[0]) if erf is not None and erf.op_type == "Erf" else None
            if div is None or div.op_type != "Div" or div.input[0] != x or not scalar(div.input[1], np.sqrt(2)):
                continue
            chain = [mul, add, erf, div]
            if any(len(users.get(n.output[0], [])) != 1 for n in chain):
                continue
            gelus[node.output[0]] = x
            skipped.update(n.output[0] for n in chain)

    ops = []
    keep_f32 = set()
    for node in nodes:
        kind, ins, out, a = node.op_type, list(node.input), node.output[0], attrs(node)
        if out in constants or out in skipped:
            continue
        if out in gelus:
            ops.append(op("onnx_gelu", {"X": gelus[out]}, out))
            continue
        if kind in ("Conv", "ConvTranspose"):
            kernel = constants[ins[1]].shape[-2:]
            strides, dilations = a.get("strides", [1, 1]), a.get("dilations", [1, 1])
            pads = a.get("pads", [0, 0, 0, 0])
            auto = a.get("auto_pad", b"NOTSET")
            if auto == b"SAME_UPPER":
                if strides != [1, 1]:
                    raise ValueError("dynamic SAME convolution requires stride one")
                total = [(k - 1) * d for k, d in zip(kernel, dilations)]
                pads = [x // 2 for x in total] + [x - x // 2 for x in total]
            elif auto not in (b"NOTSET", b"VALID"):
                raise ValueError(f"unsupported auto_pad {auto}")
            if pads[:2] != pads[2:]:
                padded = out + "__pad"
                ops.append(op("onnx_pad", {"X": ins[0]}, padded, pads=pads))
                ranks[padded] = 4
                ins[0], pads = padded, [0, 0, 0, 0]
            groups = a.get("group", 1)
            mapped = "conv2d_transpose" if kind == "ConvTranspose" else (
                "depthwise_conv2d" if groups > 1 else "conv2d")
            if groups > 1 and (constants[ins[1]].shape[1] != 1 or kind == "ConvTranspose"):
                raise ValueError("only audited depthwise groups are supported")
            if kind == "ConvTranspose" and (any(pads) or any(a.get("output_padding", []))):
                raise ValueError("unsupported transposed convolution padding")
            params = {"Input": ins[0], "Filter": ins[1]}
            if len(ins) > 2: params["Bias"] = ins[2]
            ops.append(op(mapped, params, out, strides=strides, dilations=dilations,
                          paddings=pads[:2], groups=groups))
        elif kind == "BatchNormalization":
            if a.get("training_mode", 0): raise ValueError("training BatchNormalization is unsupported")
            ops.append(op("batch_norm", dict(zip(["X", "Scale", "Bias", "Mean", "Variance"], ins)),
                          out, epsilon=float(a.get("epsilon", 1e-5))))
        elif kind in ("Relu", "Sigmoid", "HardSwish", "HardSigmoid", "Softmax"):
            mapped = {"Relu": "relu", "Sigmoid": "sigmoid", "HardSwish": "hard_swish",
                      "HardSigmoid": "hard_sigmoid", "Softmax": "softmax"}[kind]
            extra = {}
            if kind == "HardSigmoid": extra = {"slope": float(a.get("alpha", .2)), "offset": float(a.get("beta", .5))}
            if kind == "Softmax" and a.get("axis", -1) not in (-1, ranks[ins[0]] - 1):
                raise ValueError("only final-axis softmax is supported")
            ops.append(op(mapped, {"X": ins[0]}, out, **extra))
        elif kind in ("GlobalAveragePool", "AveragePool", "MaxPool"):
            k = a.get("kernel_shape", [1, 1]); pads = a.get("pads", [0] * 4)
            if a.get("auto_pad") == b"SAME_UPPER":
                if a.get("strides", [1, 1]) != [1, 1]:
                    raise ValueError("unsupported SAME pool")
                pads = [(x - 1) // 2 for x in k] + [x // 2 for x in k]
            if kind == "MaxPool" and pads == [0, 0, 1, 1]:
                padded = out + "__pad"
                ops.append(op("onnx_pad", {"X": ins[0]}, padded, pads=pads, edge=True))
                ranks[padded] = 4
                ins[0], pads = padded, [0] * 4
            if a.get("ceil_mode", 0) or pads[:2] != pads[2:] or (kind == "AveragePool" and any(pads)):
                raise ValueError("unsupported pooling rounding/padding")
            ops.append(op("pool2d", {"X": ins[0]}, out, ksize=k,
                          strides=a.get("strides", [1, 1]), paddings=pads[:2],
                          global_pooling=kind == "GlobalAveragePool",
                          pooling_type="max" if kind == "MaxPool" else "avg"))
        elif kind == "Resize":
            if a.get("mode") != b"nearest" or a.get("coordinate_transformation_mode") != b"asymmetric" or a.get("nearest_mode") != b"floor":
                raise ValueError("unsupported Resize semantics")
            scales = constants[ins[2]].tolist()
            if scales[:2] != [1, 1] or any(x != int(x) or x < 1 for x in scales):
                raise ValueError("unsupported Resize scales")
            ops.append(op("onnx_resize", {"X": ins[0]}, out, scales=scales))
        elif kind == "LayerNormalization":
            if a.get("axis", -1) not in (-1, ranks[ins[0]] - 1):
                raise ValueError("unsupported LayerNormalization axis")
            ops.append(op("layer_norm", dict(zip(["X", "Scale", "Bias"], ins)), out,
                          epsilon=float(a.get("epsilon", 1e-5))))
        elif kind == "Clip":
            limits = [float(constants[name].reshape(-1)[0]) for name in ins[1:]
                      if name in constants and constants[name].size == 1]
            if len(limits) != 2 or not np.isfinite(limits).all() or limits[0] > limits[1]:
                raise ValueError("Clip requires two finite constant limits")
            ops.append(op("onnx_clip", {"X": ins[0]}, out, minimum=limits[0], maximum=limits[1]))
        elif kind == "ReduceMean":
            axes = a.get("axes", constants[ins[1]].tolist() if len(ins) > 1 else [])
            if not a.get("keepdims", 1): raise ValueError("ReduceMean requires keepdims")
            if axes == [2, 3] and ranks[ins[0]] == 4:
                ops.append(op("pool2d", {"X": ins[0]}, out, global_pooling=True,
                              pooling_type="avg", ksize=[1, 1], strides=[1, 1], paddings=[0, 0]))
            elif axes in ([-1], [ranks[ins[0]] - 1]):
                ops.append(op("onnx_mean", {"X": ins[0]}, out))
            else:
                raise ValueError(f"unsupported ReduceMean axes {axes}")
        elif kind == "MatMul":
            if ins[1] in constants: keep_f32.add(ins[1])
            ops.append(op("matmul_v2", {"X": ins[0], "Y": ins[1]}, out))
        elif kind in ("Add", "Mul", "Sub", "Div", "Pow"):
            # The existing SVTR attention bypass reads this explicit scale.
            if kind == "Mul" and any(scalar(name, 1 / np.sqrt(d)) for name in ins for d in (15, 24)):
                scale_name = next(name for name in ins if any(scalar(name, 1 / np.sqrt(d)) for d in (15, 24)))
                name = next(name for name in ins if name != scale_name)
                ops.append(op("scale", {"X": name}, out, scale=float(constants[scale_name].reshape(-1)[0])))
            else:
                if kind == "Pow" and not scalar(ins[1], 2): raise ValueError("only square Pow is audited")
                ops.append(op("onnx_" + kind.lower(), {"X": ins[0], "Y": ins[1]}, out))
        elif kind in ("Identity", "Sqrt", "Shape", "Cast"):
            if kind == "Cast" and a["to"] not in (onnx.TensorProto.INT32, onnx.TensorProto.INT64):
                raise ValueError("only integer shape Cast is audited")
            ops.append(op("onnx_" + kind.lower(), {"X": ins[0]}, out))
        elif kind == "Transpose":
            ops.append(op("onnx_transpose", {"X": ins[0]}, out, perm=a["perm"]))
        elif kind in ("Squeeze", "Unsqueeze"):
            axes = a.get("axes", constants[ins[1]].tolist() if len(ins) > 1 else [])
            ops.append(op("onnx_" + kind.lower(), {"X": ins[0]}, out, axes=axes))
        elif kind == "Reshape":
            params, extra = {"X": ins[0], "Shape": ins[1]}, {}
            if a.get("allowzero", 0): raise ValueError("allowzero Reshape is unsupported")
            if ins[1] in constants: extra["shape"] = constants[ins[1]].tolist()
            ops.append(op("onnx_reshape", params, out, **extra))
        elif kind == "Concat":
            ops.append(op("onnx_concat", {"X": ins}, out, axis=a["axis"]))
        elif kind == "Slice":
            if any(name not in constants for name in ins[1:]): raise ValueError("dynamic Slice parameters")
            axes = constants[ins[3]].tolist() if len(ins) > 3 else list(range(constants[ins[1]].size))
            steps = constants[ins[4]].tolist() if len(ins) > 4 else [1] * len(axes)
            if any(x != 1 for x in steps): raise ValueError("only unit Slice steps supported")
            ops.append(op("onnx_slice", {"X": ins[0]}, out,
                          starts=constants[ins[1]].tolist(), ends=constants[ins[2]].tolist(), axes=axes))
        else:
            raise ValueError(f"unsupported ONNX op {kind}: {node.name}")
    inputs = [x.name for x in model.graph.input if x.name not in constants]
    if len(inputs) != 1 or len(model.graph.output) != 1: raise ValueError("expected one image input/output")
    ops.insert(0, op("feed", {}, inputs[0]))
    ops.append(op("fetch", {"X": model.graph.output[0].name}, "__fetch"))
    # Only preserve initializers still consumed after constant folding/fusion.
    used = {name for entry in ops for io in entry["inputs"] for name in io["args"]}
    tensors = {name: np.ascontiguousarray(value.reshape(value.shape[-4:]).astype(np.float32))
               for name, value in constants.items() if name in used and value.dtype.kind == "f"}
    # Normalization variances and affine constants can be tiny. Rounding
    # them to half changes the normalization itself; only convolution
    # kernels are storage-quantized. Dense RHS operands remain F32 because
    # ggml's matmul contract quantizes its left operand only.
    kernels = {entry["inputs"][1]["args"][0] for entry in ops
               if entry["type"] in ("conv2d", "depthwise_conv2d", "conv2d_transpose")}
    keep_f32.update(set(tensors) - kernels)
    integers = {name: value.reshape(-1).astype(np.int64).tolist()
                for name, value in constants.items() if name in used and value.dtype.kind in "iu"}
    for name, value in constants.items():
        ranks[name], shapes[name] = value.ndim, list(value.shape)
    return tensors, {"format": "onnx-lowered-v1", "ops": ops, "ranks": ranks,
                     "shapes": shapes, "constants": integers}, keep_f32


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path, help="GGUF output stem")
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--role", choices=("det", "rec", "cls"), required=True)
    parser.add_argument("--dictionary", type=Path)
    parser.add_argument("--quants", default="f32",
                        help="F32 by default; F16/Q8 candidates require independent numerical acceptance")
    parser.add_argument("--precision-policy", type=Path,
                        help="source-SHA-keyed measured kernel storage exceptions")
    parser.add_argument("--calibration-root", type=Path,
                        help="root of SHA-pinned conversion calibration images, when the policy requires them")
    args = parser.parse_args()
    digest = hashlib.sha256(args.model.read_bytes()).hexdigest()
    if digest != args.source_sha256: raise SystemExit("source SHA-256 mismatch")
    dictionary_bytes = None
    if args.role == "rec":
        if args.dictionary is None: raise SystemExit("recognizer requires exact source dictionary")
        dictionary_bytes = args.dictionary.read_bytes()
    dictionary_digest = hashlib.sha256(dictionary_bytes).hexdigest() if dictionary_bytes is not None else None
    source_pins = json.loads(Path(__file__).with_name("ocr_onnx_sources.json").read_text())
    validate_source_identity(digest, args.role, dictionary_digest, source_pins["assets"])
    tensors, graph, keep_f32 = lower(onnx.load(str(args.model)))
    policy = {}
    if args.precision_policy:
        document = json.loads(args.precision_policy.read_text())
        if document.get("schema") != 1:
            raise SystemExit("unsupported precision policy schema")
        policy = document.get("sources", {}).get(digest, {})
        if not policy:
            raise SystemExit("precision policy does not cover this source SHA-256")
    metadata = {"general.name": args.model.stem, "ocr.role": args.role,
                "ocr.arch": args.model.stem, "ocr.source.format": "onnx",
                "ocr.source.name": args.model.name, "ocr.source.sha256": digest,
                "ocr.graph.ops": json.dumps(graph, separators=(",", ":")),
                "ocr.graph.op_count": len(graph["ops"])}
    if args.role == "rec":
        metadata["ocr.dictionary"] = dictionary_bytes.decode("utf-8").splitlines()
        metadata["ocr.dictionary.sha256"] = dictionary_digest
    quants = args.quants.split(",")
    if set(policy) - {"f32", "f16", "q8_0"}:
        raise SystemExit("unsupported precision policy quantization")
    # Validate the entire request before creating the first output artifact.
    for kind in quants:
        if kind not in ("f32", "f16", "q8_0"): raise SystemExit(f"unsupported quantization {kind}")
        exceptions = policy.get(kind, {})
        unknown = set(exceptions) - {"keep_f32", "keep_f16", "activation_q8"}
        if unknown:
            raise SystemExit(f"unsupported precision policy fields: {unknown}")
        f32 = set(exceptions.get("keep_f32", []))
        f16 = set(exceptions.get("keep_f16", []))
        calibrated = exceptions.get("activation_q8", {}).get("kernels", {})
        if "activation_q8" in exceptions and kind != "q8_0":
            raise SystemExit("activation Q8_0 requires a q8_0 container")
        if calibrated and (kind != "q8_0" or set(calibrated) & (keep_f32 | f32 | f16) or
                           set(calibrated) - tensors.keys()):
            raise SystemExit("conflicting or unknown activation Q8_0 kernels")
        if (f32 | f16) - (set(tensors) - keep_f32) or f32 & f16:
            raise SystemExit("precision policy names unknown, non-kernel or conflicting tensors")
        if f16 and kind != "q8_0":
            raise SystemExit("keep_f16 exceptions require a q8_0 container")
        if kind == "f32" and (f32 or f16):
            raise SystemExit("F32 source containers do not accept precision exceptions")
        if kind != "f32" and not any(
                quantized(name, value, kind, [], keep_f32=name in keep_f32 | f32,
                          keep_f16=name in f16)[1].name == kind.upper()
                for name, value in tensors.items()):
            raise SystemExit(f"precision policy leaves no {kind.upper()} weights in the requested container")
    overrides = {}
    if "q8_0" in quants and policy.get("q8_0", {}).get("activation_q8"):
        from ocr_q8_calibration import calibrated_overrides
        overrides = calibrated_overrides(args.model, tensors, policy["q8_0"]["activation_q8"],
                                         args.calibration_root, args.role)
    for kind in quants:
        exceptions = policy.get(kind, {})
        f32 = set(exceptions.get("keep_f32", []))
        f16 = set(exceptions.get("keep_f16", []))
        target = Path(f"{args.output}-{kind}.gguf")
        target.parent.mkdir(parents=True, exist_ok=True)
        quant_metadata = dict(metadata)
        if exceptions:
            quant_metadata["ocr.quant.policy"] = json.dumps(exceptions, sort_keys=True)
        write_gguf(tensors, quant_metadata, target, kind, [],
                   keep_f32=keep_f32 | f32, keep_f16=f16,
                   storage_overrides=overrides if kind == "q8_0" else None)
        print(f"{target}: sha256={hashlib.sha256(target.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
