#!/usr/bin/env python3
"""Render PyTorch vs GGML side-by-side comparison figures for DeepLSD README assets."""

from __future__ import annotations

import argparse
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

DEEPLSD_ROOT = Path(__file__).resolve().parents[3].parent / "dl" / "DeepLSD"


def read_deeplsd_output(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = path.read_bytes()
    width, height = struct.unpack_from("<ii", data, 8)
    plane = width * height
    df = np.frombuffer(data, dtype=np.float32, count=plane, offset=16).reshape(height, width)
    angle = np.frombuffer(data, dtype=np.float32, count=plane, offset=16 + plane * 4).reshape(height, width)
    return df, angle


def run_deeplsd_ggml(gguf: Path, gray: np.ndarray, binary: Path, device: str, out: Path) -> None:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        img_path = Path(tmp.name)
    cv2.imwrite(str(img_path), gray)
    subprocess.run(
        [str(binary), "extract", str(gguf), str(img_path), str(out), "--device", device],
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gguf-deeplsd", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(DEEPLSD_ROOT))

    gray_full = cv2.imread(str(args.image), cv2.IMREAD_GRAYSCALE)
    if gray_full is None:
        raise SystemExit(f"cannot read {args.image}")

    gray_dl = cv2.resize(gray_full, (512, 512), interpolation=cv2.INTER_LINEAR)
    dl_out = args.output_dir / "deeplsd_cpp.dlsd"
    run_deeplsd_ggml(
        args.gguf_deeplsd,
        gray_dl,
        DEEPLSD_ROOT / "cpp/build/deeplsd_extract",
        args.device,
        dl_out,
    )
    from deeplsd.models.export_ggml import load_export_model  # type: ignore

    m = load_export_model(str(DEEPLSD_ROOT / "weights/deeplsd_wireframe.tar"), device="cpu")
    with torch.inference_mode():
        tdf, tang = m(torch.from_numpy(gray_dl.astype(np.float32) / 255.0)[None, None])
    gdf, gang = read_deeplsd_output(dl_out)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for r, (name, tg, gg) in enumerate((("df_norm", tdf[0].numpy(), gdf), ("angle", tang[0].numpy(), gang))):
        axes[r, 0].imshow(tg, cmap="magma")
        axes[r, 0].set_title(f"PyTorch {name}")
        axes[r, 0].axis("off")
        axes[r, 1].imshow(gg, cmap="magma")
        axes[r, 1].set_title(f"GGML {name}")
        axes[r, 1].axis("off")
        im = axes[r, 2].imshow(np.abs(tg - gg), cmap="hot")
        axes[r, 2].set_title(f"|diff| {name}")
        axes[r, 2].axis("off")
        fig.colorbar(im, ax=axes[r, 2], fraction=0.046)
    fig.suptitle(f"DeepLSD parity ({args.device}, 512²)")
    fig.tight_layout()
    fig.savefig(args.output_dir / f"deeplsd_parity_{args.device}.png", dpi=160)
    plt.close(fig)
    print("Wrote comparison PNGs to", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
