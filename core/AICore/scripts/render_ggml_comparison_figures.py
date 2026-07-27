#!/usr/bin/env python3
"""Render PyTorch vs GGML side-by-side comparison figures for README assets."""

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

ELOFTR_ROOT = Path(__file__).resolve().parents[3].parent / "MVS" / "EfficientLoFTR"
DEEPLSD_ROOT = Path(__file__).resolve().parents[3].parent / "dl" / "DeepLSD"


def read_eloftr_output(path: Path) -> np.ndarray:
    data = path.read_bytes()
    w, h, c = struct.unpack_from("<iii", data, 8)
    feat = np.frombuffer(data, dtype=np.float32, count=w * h * c, offset=20)
    return feat.reshape(c, h, w)


def read_deeplsd_output(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = path.read_bytes()
    width, height = struct.unpack_from("<ii", data, 8)
    plane = width * height
    df = np.frombuffer(data, dtype=np.float32, count=plane, offset=16).reshape(height, width)
    angle = np.frombuffer(data, dtype=np.float32, count=plane, offset=16 + plane * 4).reshape(height, width)
    return df, angle


def run_eloftr_ggml(gguf: Path, gray: np.ndarray, binary: Path, device: str, out: Path) -> None:
    h, w = gray.shape
    inp = (gray.astype(np.float32) / 255.0).reshape(-1)
    with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(inp.astype("<f4").tobytes())
    subprocess.run(
        [str(binary), str(gguf), str(tmp_path), str(h), str(w), str(out), "--device", device],
        check=True,
    )


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
    parser.add_argument("--gguf-eloftr", type=Path, required=True)
    parser.add_argument("--gguf-deeplsd", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(ELOFTR_ROOT / "scripts"))
    sys.path.insert(0, str(ELOFTR_ROOT / "src"))
    sys.path.insert(0, str(DEEPLSD_ROOT))

    gray_full = cv2.imread(str(args.image), cv2.IMREAD_GRAYSCALE)
    if gray_full is None:
        raise SystemExit(f"cannot read {args.image}")

    # ELoFTR @ 640
    gray_el = cv2.resize(gray_full, (640, 640), interpolation=cv2.INTER_LINEAR)
    el_out = args.output_dir / "eloftr_cpp.elout"
    run_eloftr_ggml(
        args.gguf_eloftr,
        gray_el,
        ELOFTR_ROOT / "cpp/build/eloftr_backbone",
        args.device,
        el_out,
    )
    from verify_eloftr_ggml import load_torch_backbone  # type: ignore

    model = load_torch_backbone(ELOFTR_ROOT / "weights/eloftr_outdoor.ckpt")
    with torch.inference_mode():
        torch_feat = model(torch.from_numpy(gray_el.astype(np.float32) / 255.0)[None, None])["feats_c"][0].numpy()
    ggml_feat = read_eloftr_output(el_out)
    diff = np.abs(torch_feat - ggml_feat).mean(axis=0)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(gray_el, cmap="gray")
    axes[0].set_title("Input (640²)")
    axes[0].axis("off")
    axes[1].imshow(torch_feat.mean(axis=0), cmap="viridis")
    axes[1].set_title("PyTorch mean feat")
    axes[1].axis("off")
    im = axes[2].imshow(diff, cmap="hot")
    axes[2].set_title("GGML |diff| mean")
    axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    fig.tight_layout()
    fig.savefig(args.output_dir / f"eloftr_parity_{args.device}.png", dpi=160)
    plt.close(fig)

    # DeepLSD @ 512
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
