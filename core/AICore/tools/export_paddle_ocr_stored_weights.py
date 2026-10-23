#!/usr/bin/env python3
"""Reconstruct GGUF weights for an independent, unchanged Paddle graph.

This is attribution, not release qualification: compare Paddle's original
weights, Paddle with these stored weights, and the native GGUF execution.
No native outputs are used to construct the reference. Parameter descriptors,
LoD headers and ordering are preserved byte for byte; only F32 payloads change.
The original model files are never overwritten.
"""

import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

from paddle_ocr_extract import _parse_tensor_desc, load_ocr_tensors


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def payload_ranges(data):
    """Locate F32 payloads in the serialized Paddle LoDTensor stream, O(bytes)."""
    position = 0
    while position < len(data):
        start = position
        if len(data) - position < 12:
            raise ValueError("truncated LoDTensor header")
        version, lod_count = struct.unpack_from("<IQ", data, position)
        position += 12
        if version != 0 or lod_count != 0:
            raise ValueError("only version-zero, LoD-free OCR parameters are supported")
        if len(data) - position < 8:
            raise ValueError("truncated tensor header")
        tensor_version, descriptor_size = struct.unpack_from("<II", data, position)
        position += 8
        if tensor_version != 0 or descriptor_size > len(data) - position:
            raise ValueError("invalid tensor descriptor")
        dtype, shape = _parse_tensor_desc(data[position:position + descriptor_size])
        position += descriptor_size
        if dtype != 5 or not shape or any(d < 1 for d in shape):
            raise ValueError("expected a nonempty F32 parameter")
        count = 1
        for dimension in shape:
            count *= dimension
        end = position + count * 4
        if end > len(data):
            raise ValueError("truncated tensor payload")
        yield start, position, end, tuple(shape)
        position = end


def replace_payloads(original, named, values):
    """Require exact names, shapes and stream alignment before replacement."""
    names = [name for name, _ in named]
    if len(set(names)) != len(names) or set(names) != set(values):
        raise ValueError("stored/source parameter name coverage differs")
    ranges = list(payload_ranges(original))
    if len(ranges) != len(named):
        raise ValueError("parameter stream count differs")
    output = bytearray(original)  # Two parameter streams in memory; no activations.
    for (name, source), (_, begin, end, shape) in zip(named, ranges):
        if shape != source.shape:
            raise ValueError(f"source parameter stream shape mismatch: {name}")
        source_bytes = np.asarray(source, dtype="<f4").tobytes()
        if original[begin:end] != source_bytes:
            raise ValueError(f"source parameter stream order mismatch: {name}")
        value = values[name]
        if value.dtype != np.float32 or value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"invalid reconstructed parameter: {name}")
        output[begin:end] = np.asarray(value, dtype="<f4").tobytes()
    return output


def stored_values(model, named):
    import gguf

    reader = gguf.GGUFReader(str(model))
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    expected = {"ocr." + name for name, _ in named}
    if len(tensors) != len(reader.tensors) or set(tensors) != expected:
        raise ValueError("GGUF/source parameter name coverage differs")
    values, storage = {}, []
    allowed = {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16,
               gguf.GGMLQuantizationType.Q8_0}
    for name, original in named:
        tensor = tensors["ocr." + name]
        if tensor.tensor_type not in allowed:
            raise ValueError(f"unsupported storage type: {name}")
        value = gguf.quants.dequantize(tensor.data, tensor.tensor_type)
        shapes = {original.shape}
        if tensor.tensor_type == gguf.GGMLQuantizationType.Q8_0 and original.ndim >= 2:
            shapes.add((original.shape[0], original.size // original.shape[0]))
        if value.shape not in shapes:
            raise ValueError(f"GGUF/source parameter shape differs: {name}")
        # The converter only folds trailing kernel dimensions for Q8_0;
        # it never permutes the Paddle parameter payload's C order.
        value = np.asarray(value, dtype=np.float32).reshape(original.shape).copy()
        if not np.isfinite(value).all():
            raise ValueError(f"nonfinite stored parameter: {name}")
        values[name] = value
        storage.append(dict(name=name, shape=list(original.shape),
            storage=tensor.tensor_type.name,
            source_payload_sha256=hashlib.sha256(original.tobytes()).hexdigest(),
            stored_payload_sha256=hashlib.sha256(value.tobytes()).hexdigest(),
            max_diff=float(np.max(np.abs(value.astype(np.float64) - original)))))
    return values, storage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("program", "parameters", "model", "output-parameters", "report"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("program", "parameters", "model"):
        parser.add_argument("--" + name + "-sha256", required=True)
    args = parser.parse_args()
    inputs = [args.program, args.parameters, args.model]
    outputs = [args.output_parameters, args.report]
    if (len({p.resolve() for p in inputs + outputs}) != 5 or
            any(p.exists() for p in outputs)):
        parser.error("outputs must be distinct new files outside the original input paths")
    before = {str(path): digest(path) for path in inputs}
    for name in ("program", "parameters", "model"):
        if before[str(getattr(args, name))] != getattr(args, name + "_sha256"):
            parser.error(f"{name} SHA-256 mismatch")
    # load_ocr_tensors selects the archive's program/parameter files. Reject
    # an ambiguous directory instead of silently binding another pair.
    programs = list(args.program.parent.glob("*.pdmodel"))
    parameters = list(args.program.parent.glob("*.pdiparams"))
    if (programs != [args.program] or parameters != [args.parameters]):
        parser.error("require one original pdmodel/pdiparams pair in the same directory")
    named, _, _ = load_ocr_tensors(args.program.parent)
    values, storage = stored_values(args.model, named)
    result = replace_payloads(args.parameters.read_bytes(), named, values)
    if before != {str(path): digest(path) for path in inputs}:
        raise ValueError("input identities changed during reconstruction")
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_parameters.open("xb") as target:
        target.write(result)
    report = dict(schema=1, inputs=before,
        script_sha256=digest(Path(__file__)),
        extractor_sha256=digest(Path(load_ocr_tensors.__code__.co_filename)),
        output_parameters_sha256=digest(args.output_parameters),
        parameter_count=len(named), parameters=storage,
        scope="Stored-weight Paddle attribution only; original FP32 accuracy gates remain mandatory")
    with args.report.open("x") as target:
        target.write(json.dumps(report, indent=2) + "\n")
    print(f"Stored-weight Paddle parameters: {args.output_parameters}")


if __name__ == "__main__":
    main()
