"""Offline QAT forward values for the exact GGUF weight formats.

This module has no inference/runtime dependency. A training framework can use
these frozen forward values with a straight-through gradient for its separate
F32 master weights. Independent original-source gates still own acceptance.
"""

import numpy as np


def storage_forward(weight, storage):
    """Return F32 decoded values after real F32/F16/Q8_0 serialization.

    Q8_0 rounds the integers using the unrounded block scale, then serializes
    the scale as F16. Rounding the scale before computing integers changes the
    format's forward values. Work and temporary storage are O(weight.size).
    """
    value = np.asarray(weight)
    if value.dtype != np.float32 or not value.size or not np.isfinite(value).all():
        raise ValueError("QAT master weights must be finite nonempty F32 arrays")
    if storage == "F32":
        return value.copy()
    if storage == "F16":
        if np.abs(value).max() > np.finfo(np.float16).max:
            raise ValueError("QAT F16 weights exceed finite storage range")
        return value.astype(np.float16).astype(np.float32)
    if storage != "Q8_0":
        raise ValueError("unsupported QAT weight storage")
    if value.ndim < 2 or value.reshape(value.shape[0], -1).shape[1] % 32:
        raise ValueError("QAT Q8_0 rows must contain complete 32-element blocks")
    blocks = value.reshape(-1, 32)
    scale = np.abs(blocks).max(axis=1, keepdims=True) / np.float32(127)
    scale16 = scale.astype(np.float16)
    if not np.isfinite(scale16).all():
        raise ValueError("QAT Q8_0 scales exceed finite F16 storage range")
    inverse = np.zeros_like(scale)
    np.divide(np.float32(1), scale, out=inverse, where=scale != 0)
    normalized = blocks * inverse
    # ggml's reference roundf resolves halfway values away from zero.
    integers = np.copysign(np.floor(np.abs(normalized) + np.float32(.5)), normalized)
    decoded = integers.astype(np.int8).astype(np.float32) * scale16.astype(np.float32)
    return decoded.reshape(value.shape)
