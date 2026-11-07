#!/usr/bin/env python3
"""Independent resident official PyTorch Tag2Text caption reference."""
import argparse
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    sys.path.insert(0, str(args.upstream / "cpp_ggml/benchmarks"))
    sys.path.insert(0, str(args.upstream / "cpp_ggml/scripts"))
    from python_eval import load_model
    from model_manifest import MODEL_CHECKPOINTS
    from ram import get_transform
    import torch
    from PIL import Image
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("PyTorch CUDA is unavailable; use --device cpu explicitly")
    model = load_model("tag2text", args.upstream / "cpp_ggml/models/pytorch" /
                       MODEL_CHECKPOINTS["tag2text"], device)
    transform = get_transform(384)
    with args.output.open("w") as output:
        for index, path in enumerate(args.manifest.read_text().splitlines()):
            times = []
            for iteration in range(-args.warmup if index == 0 else 0, args.repeat):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                image = transform(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
                with torch.inference_mode():
                    caption = model.generate(image, max_length=30)[0]
                if device.type == "cuda":
                    torch.cuda.synchronize()
                if iteration >= 0:
                    times.append((time.perf_counter() - start) * 1000)
            output.write(json.dumps({"image": path, "caption": caption,
                                     "latency_ms": sum(times) / len(times),
                                     "device": str(device)}) + "\n")


if __name__ == "__main__":
    main()
