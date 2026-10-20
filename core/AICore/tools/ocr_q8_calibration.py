"""Conversion-only activation-weighted Q8_0; never part of AICore inference.

Calibration uses pinned images distinct from acceptance fixtures. The output
is ordinary ggml Q8_0 bytes, and still needs independent source/page gates.
"""
import copy
import hashlib
from pathlib import Path

import numpy as np


ALGORITHM = "activation-error-feedback-q8_0-v1"


def activation_q8_0(weight, activation, damping):
    """Quantize aligned pointwise rows using source activation covariance."""
    if (weight.ndim != 4 or tuple(weight.shape[2:]) != (1, 1) or
            weight.shape[1] % 32 or not np.isfinite(weight).all()):
        raise ValueError("activation Q8_0 requires finite aligned pointwise kernels")
    if not np.isfinite(damping) or damping <= 0:
        raise ValueError("activation Q8_0 damping must be positive and finite")
    w = weight.reshape(weight.shape[0], -1).astype(np.float64)
    x = np.asarray(activation, np.float64)
    if (x.ndim != 2 or x.shape[0] != w.shape[1] or x.shape[1] == 0 or
            not np.isfinite(x).all()):
        raise ValueError("activation Q8_0 input covariance shape/value mismatch")
    h = x @ x.T / x.shape[1]
    diagonal_mean = float(np.diag(h).mean())
    if diagonal_mean <= 0:
        raise ValueError("activation Q8_0 covariance has no signal")
    h += np.eye(h.shape[0]) * (diagonal_mean * damping)
    # O(C^3) inversion is conversion-only: cross-channel covariance is what
    # lets each rounding error be compensated in the remaining input channels.
    inverse = np.linalg.cholesky(np.linalg.inv(h)).T
    scales16 = (np.abs(w.reshape(w.shape[0], -1, 32)).max(-1) / 127).astype(np.float16)
    if not np.isfinite(scales16).all():
        raise ValueError("activation Q8_0 scale exceeds F16 range")
    scales = scales16.astype(np.float64)
    adjusted = w.copy()
    integers = np.zeros(w.shape, np.int8)
    for column in range(w.shape[1]):
        scale = scales[:, column // 32]
        safe = np.where(scale == 0, 1, scale)
        q = np.clip(np.rint(adjusted[:, column] / safe), -127, 127).astype(np.int8)
        q[scale == 0] = 0
        integers[:, column] = q
        error = (adjusted[:, column] - q.astype(np.float64) * scale) / inverse[column, column]
        adjusted[:, column:] -= error[:, None] * inverse[column, column:]
    blocks = np.empty((w.shape[0], w.shape[1] // 32, 34), np.uint8)
    blocks[:, :, :2] = scales16.astype("<f2").view(np.uint8).reshape(w.shape[0], -1, 2)
    blocks[:, :, 2:] = integers.view(np.uint8).reshape(w.shape[0], -1, 32)
    return blocks.reshape(w.shape[0], -1)


def calibrated_overrides(source, tensors, config, root, role):
    """Verify calibration provenance before executing the source CPU graph."""
    import cv2
    import gguf
    import onnx
    import onnxruntime as ort
    from export_onnx_ocr_fixture import preprocess, source_crop

    if set(config) != {"algorithm", "images", "kernels"} or config["algorithm"] != ALGORITHM:
        raise ValueError("unsupported activation Q8_0 calibration contract")
    if role != "rec":
        raise ValueError("activation Q8_0 is audited for recognizers only")
    if root is None:
        raise ValueError("activation Q8_0 requires --calibration-root")
    root = Path(root).resolve()
    if not config["images"] or not config["kernels"]:
        raise ValueError("activation Q8_0 needs pinned images and kernels")
    images = []
    identities = set()
    for entry in config["images"]:
        if set(entry) not in ({"relative_path", "sha256"}, {"relative_path", "sha256", "quad"}):
            raise ValueError("invalid activation calibration image contract")
        relative = Path(entry["relative_path"])
        quad = entry.get("quad")
        if quad is not None and (len(quad) != 8 or not np.isfinite(quad).all()):
            raise ValueError("invalid calibration crop quadrilateral")
        identity = (relative, tuple(quad) if quad is not None else None)
        if relative.is_absolute() or ".." in relative.parts or identity in identities:
            raise ValueError("invalid or duplicate calibration relative path")
        identities.add(identity)
        path = root / relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("activation calibration image SHA-256 mismatch")
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError("activation calibration image decode failed")
        if quad is not None:
            image = source_crop(image, quad)
        images.append(preprocess(image, role))
    model = onnx.load(str(source))
    layers = {}
    for node in model.graph.node:
        if node.op_type == "Conv" and node.input[1] in config["kernels"]:
            attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
            if (node.input[1] in layers or attrs.get("group", 1) != 1 or
                    attrs.get("strides", [1, 1]) != [1, 1] or
                    any(attrs.get("pads", [0, 0, 0, 0]))):
                raise ValueError("activation Q8_0 requires one unpadded stride-one Conv consumer")
            layers[node.input[1]] = node.input[0]
    if layers.keys() != config["kernels"].keys():
        raise ValueError("activation Q8_0 kernel missing from source Conv")
    observer = copy.deepcopy(model)
    del observer.graph.output[:]
    for name in layers.values():
        observer.graph.output.append(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None))
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = 4, 1
    options.log_severity_level = 3
    session = ort.InferenceSession(observer.SerializeToString(), options, providers=["CPUExecutionProvider"])
    activations = {name: [] for name in layers}
    for image in images:
        outputs = session.run(list(layers.values()), {session.get_inputs()[0].name: image})
        for name, value in zip(layers, outputs):
            if value.ndim != 4:
                raise ValueError("activation Q8_0 requires NCHW source activations")
            activations[name].append(value.transpose(1, 0, 2, 3).reshape(value.shape[1], -1))
    return {name: (activation_q8_0(tensors[name], np.concatenate(activations[name], axis=1), damping),
                   gguf.GGMLQuantizationType.Q8_0)
            for name, damping in config["kernels"].items()}
