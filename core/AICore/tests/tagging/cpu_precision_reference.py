#!/usr/bin/env python3
"""Independent PyTorch reference for the CPU RAM Q8 activation arithmetic.

The official graph and GGUF weights are retained. Q8 linear inputs use the
32-value block scale and nearest-even rounding of GGML's AVX2 CPU dot product;
AUTO leaves the first two high-resolution Swin QKV inputs floating. This is
arithmetic parity evidence, never original-checkpoint accuracy evidence.
"""

import argparse
import importlib.util
from pathlib import Path
import sys


def quantized_input(module, inputs):
    import torch
    value = inputs[0]
    blocks = value.reshape(*value.shape[:-1], -1, 32)
    maximum = blocks.abs().amax(dim=-1, keepdim=True)
    multiplier = torch.where(maximum != 0, 127.0 / maximum, 0.0)
    scale = (maximum / 127.0).half().float()
    restored = (blocks * multiplier).round().clamp(-127, 127) * scale
    return (restored.reshape_as(value), *inputs[1:])


def configure(model, path, precision):
    import gguf
    import torch
    reader = gguf.GGUFReader(path)
    state = model.state_dict()
    quantized = set()
    for tensor in reader.tensors:
        name = tensor.name.removeprefix("ram.")
        if name not in state:
            continue
        data = gguf.dequantize(tensor.data, tensor.tensor_type).copy()
        state[name].copy_(torch.from_numpy(data.reshape(tuple(state[name].shape))))
        if tensor.tensor_type == gguf.GGMLQuantizationType.Q8_0:
            quantized.add(name)
    for name, module in model.named_modules():
        protected = (precision == 0 and
                     name.startswith(("visual_encoder.layers.0.",
                                      "visual_encoder.layers.1.")) and
                     name.endswith(".attn.qkv"))
        if isinstance(module, torch.nn.Linear) and name + ".weight" in quantized and not protected:
            module.register_forward_pre_hook(quantized_input)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--cpu-precision", type=int, choices=(0, 1), default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--demo", action="store_true")
    args, remaining = parser.parse_known_args()
    upstream = args.upstream.resolve()
    sys.path.insert(0, str(upstream))
    sys.path.insert(0, str(upstream / "cpp_ggml/scripts"))
    import torch
    torch.set_num_threads(args.threads)
    relative = ("cpp_ggml/scripts/parity_reference.py" if args.demo else
                "cpp_ggml/benchmarks/python_eval.py")
    spec = importlib.util.spec_from_file_location("ram_cpu_reference", upstream / relative)
    evaluation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluation)
    if args.demo:
        factory = evaluation.ram
        def configured_factory(**kwargs):
            return configure(factory(**kwargs), args.gguf, args.cpu_precision)
        evaluation.ram = configured_factory
        # The upstream demo has no device argument; this reference is CPU only.
        torch.cuda.is_available = lambda: False
    else:
        factory = evaluation.load_model
        def configured_factory(name, checkpoint, device):
            if name != "ram" or device.type != "cpu":
                raise ValueError("CPU activation reference requires RAM on CPU")
            return configure(factory(name, checkpoint, device), args.gguf, args.cpu_precision)
        evaluation.load_model = configured_factory
        import batch_inference
        batch_inference.device = torch.device("cpu")
    sys.argv = [str(spec.origin), *remaining]
    return evaluation.main()


if __name__ == "__main__":
    raise SystemExit(main())
