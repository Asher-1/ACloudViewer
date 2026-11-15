#!/usr/bin/env python3
"""Convert the Recognize Anything Model checkpoint into an AICore GGUF.

First-party conversion path for the image-tagging models consumed by the
AICore `yolo` task's CLASSIFY role (RAM = Swin-L image encoder + tag
attention decoder over a fixed vocabulary). Like the LoMa/PP-OCR converters,
this program runs only on the controlled conversion host (it needs
``torch`` and ``gguf``) and the ACloudViewer runtime reads the GGUF through
ggml without linking PyTorch.

The 4574-entry tag vocabulary is embedded as the `ram.tag_list` GGUF string
array, so a model file is self-contained. Tensor names keep the checkpoint
hierarchy behind the `ram.` namespace. Each run emits F32 plus F16 and Q8_0
siblings; tensors whose last dimension is not a multiple of 32 stay F16 in
the Q8_0 file and are recorded under `ram.quant.reshaped`.
"""

import argparse
import hashlib
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_state_dict(path: Path):
    import torch
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]
    if not isinstance(checkpoint, dict):
        raise SystemExit(f"unsupported checkpoint layout in {path}")
    state = {}
    for key, value in checkpoint.items():
        tensor = value.detach().cpu().numpy()
        if tensor.dtype != np.float32:
            tensor = tensor.astype(np.float32)
        state[key] = tensor
    if not state:
        raise SystemExit(f"no tensors extracted from {path}")
    return state


def quantized(name: str, tensor: np.ndarray, kind: str, reshaped):
    import gguf
    if kind == "f32":
        return tensor, gguf.GGMLQuantizationType.F32
    if kind == "f16":
        return tensor.astype(np.float16), gguf.GGMLQuantizationType.F16
    if kind == "q8_0":
        if tensor.ndim >= 2 and tensor.size:
            # Standard ggml storage: fold trailing kernel dims into the
            # input dimension so block quantization sees an aligned axis;
            # the graph builder reshapes back to the layer shape.
            folded = int(np.prod(tensor.shape[1:]))
            if folded % 32 == 0:
                reshaped.append(name + ":" +
                                ",".join(str(d) for d in tensor.shape))
                return (gguf.quants.Q8_0.quantize(
                            tensor.reshape(tensor.shape[0], folded)),
                        gguf.GGMLQuantizationType.Q8_0)
        return tensor.astype(np.float16), gguf.GGMLQuantizationType.F16
    raise SystemExit(f"unknown quant {kind}")


def write_gguf(tensors, metadata, output: Path, kind: str,
               reshaped):
    import gguf
    writer = gguf.GGUFWriter(str(output), "ram")
    for key, value in metadata.items():
        if isinstance(value, list):
            if value and isinstance(value[0], str):
                writer.add_array(key, value)
            else:
                writer.add_array(key, [np.int32(v) for v in value])
        elif isinstance(value, float):
            writer.add_float32(key, value)
        elif isinstance(value, int):
            writer.add_uint32(key, value)
        else:
            writer.add_string(key, value)
    if reshaped:
        writer.add_array("ram.quant.reshaped", reshaped)
    for name, tensor in tensors.items():
        array, ggml_type = quantized(name, tensor, kind,
                                     reshaped)
        writer.add_tensor("ram." + name, array, raw_dtype=ggml_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path,
                        help="Recognize Anything .pth checkpoint")
    parser.add_argument("output", type=Path,
                        help="output stem; siblings get -f16/-q8_0 suffixes")
    parser.add_argument("--tag-list", type=Path, required=True,
                        help="ram_tag_list.txt (one tag per line)")
    parser.add_argument("--source-name", default="ram_swin_large_14m.pth")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--quants", default="f32,f16,q8_0")
    args = parser.parse_args()

    tags = [line.strip() for line in
            args.tag_list.read_text(encoding="utf-8").splitlines()
            if line.strip()]

    tensors = load_state_dict(args.checkpoint)
    metadata = {
        "general.name": "ram-swin-large-14m",
        "ram.source.format": "torch-checkpoint",
        "ram.source.name": args.source_name,
        "ram.source.sha256": sha256(args.checkpoint),
        "ram.tag_list": tags,
        "ram.tag_list.size": len(tags),
        "ram.image_size": args.image_size,
    }

    stem = (args.output.with_suffix("") if args.output.suffix
            else args.output)
    reshaped = []
    for kind in [k.strip() for k in args.quants.split(",") if k.strip()]:
        path = Path(f"{stem}-{kind}.gguf")
        write_gguf(tensors, metadata, path, kind,
                   reshaped if kind == "q8_0" else [])
        print(f"wrote {path} ({kind}, {len(tensors)} tensors)")


if __name__ == "__main__":
    main()
