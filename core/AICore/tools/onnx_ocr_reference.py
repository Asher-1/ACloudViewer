"""Independent ONNX execution with the exact stored GGUF weights.

This is validation tooling only. Restoring storage-rounded constants in the
original ONNX topology separates quantization error from native graph error;
it does not replace comparison with the unmodified source checkpoint.
"""

from collections import Counter
import hashlib

import gguf
import numpy as np
import onnx
from onnx import numpy_helper


def stored_weight_reference(model, container):
    """Return a source graph with GGUF constants restored as float32.

    Constant-folded aliases are replaced at their original output name. Dead
    source nodes may remain, but no replacement has two producers. The graph
    operators and the image pre/postprocessing stay owned by the source.
    """
    from convert_onnx_ocr_gguf import lower

    source_bytes = model.read_bytes()
    reference = onnx.load_model_from_string(source_bytes)
    # lower mutates value_info; never mutate the independent reference graph.
    tensors, _, _ = lower(onnx.load_model_from_string(source_bytes))
    reader = gguf.GGUFReader(str(container))
    field = reader.get_field("ocr.source.sha256")
    expected = field.parts[field.data[0]].tobytes().decode() if field else ""
    if hashlib.sha256(source_bytes).hexdigest() != expected:
        raise ValueError("GGUF/source SHA-256 mismatch")
    restored, storage = {}, Counter()
    for tensor in reader.tensors:
        if not tensor.name.startswith("ocr."):
            raise ValueError(f"unexpected GGUF tensor namespace: {tensor.name}")
        name = tensor.name[4:]
        if name not in tensors:
            raise ValueError(f"GGUF constant absent from source: {name}")
        values = gguf.quants.dequantize(tensor.data, tensor.tensor_type)
        restored[name] = np.asarray(values, np.float32).reshape(tensors[name].shape)
        storage[tensor.tensor_type.name] += int(values.size)
    if restored.keys() != tensors.keys():
        raise ValueError("GGUF does not cover the source constants exactly")
    return replace_constants(reference, restored), dict(storage)


def replace_constants(reference, restored):
    """Replace constants at their source names; preserve live operators."""
    original_shapes = {value.name: tuple(value.dims)
                       for value in reference.graph.initializer}
    # Identity and folded bias/shape operators can produce constants used by
    # the lowered graph. Replace their producer, not their inputs: rounding
    # an input before a fold would validate different stored weights.
    nodes = []
    for node in reference.graph.node:
        replaced = set(node.output) & restored.keys()
        if replaced:
            if len(replaced) != len(node.output):
                raise ValueError("partially replaced multi-output source node")
            continue
        nodes.append(node)
    del reference.graph.node[:]
    reference.graph.node.extend(nodes)
    initializers = [value for value in reference.graph.initializer
                    if value.name not in restored]
    initializers.extend(numpy_helper.from_array(
        array.reshape(original_shapes.get(name, array.shape)), name)
        for name, array in restored.items())
    del reference.graph.initializer[:]
    reference.graph.initializer.extend(initializers)
    # Remove only dead source constants after alias replacement. This avoids
    # hundreds of ORT warnings while retaining the live source topology.
    live = {value.name for value in reference.graph.output}
    nodes = []
    for node in reversed(reference.graph.node):
        if live.intersection(node.output):
            nodes.append(node)
            live.update(node.input)
    del reference.graph.node[:]
    reference.graph.node.extend(reversed(nodes))
    initializers = [value for value in reference.graph.initializer if value.name in live]
    del reference.graph.initializer[:]
    reference.graph.initializer.extend(initializers)
    return reference
