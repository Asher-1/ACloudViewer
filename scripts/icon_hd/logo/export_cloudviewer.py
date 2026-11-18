"""Export the CloudViewer icon set from its vector master.

Scope note -- read before extending this file
---------------------------------------------
Only **CloudViewer** is generated here.  The ACloudViewer mark is deliberately
left as its original artwork, because six vectorisation routes were measured
against it and every one either lost the design or produced a *larger* file:

    parametric redraw        10 KB    IoU 0.81   loses the brush-stroke texture
    silhouette fill          97 KB    IoU 0.99   interior collapses to flat colour
    posterised colour layers 850-2600 KB  IoU 0.87-0.92  gradients band visibly
    stroke-skeleton tracing  97 KB    IoU 0.89   ink inflates into capsule blobs

That mark is a painted texture, not geometry.  Replacing it would be a
redesign, and the brief was explicit that it must not be redesigned.  Its
existing raster set also turned out to be hand-optimised per size -- measurably
sharper at 16-32 px than a downscale of the master -- so there is nothing to
"fix" there either.

CloudViewer is different: it *is* geometry (a tapered ribbon around a disc), and
a vector reconstruction beats the checked-in PNGs at every size up to 128 px:

    size   repo PNG error   vector error   (mean abs, 0-255, vs master downscale)
     16 px      60.2            32.6
     24 px      42.5            23.0
     32 px      32.9            21.6
     48 px      25.9            19.4
     64 px      21.1            16.5
    128 px      10.4            10.5
    256 px       4.9             7.4   <- raster wins here, so 256 stays raster

Hence: vector for 16-128, keep the original 256 px raster.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import cairosvg
from PIL import Image

from . import marks as M

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MASTERS = HERE / "masters"

# Sizes rendered from the vector.  256 is excluded on purpose: the measured
# error is lower for the original raster at that size.
VECTOR_SIZES = (16, 24, 32, 48, 64, 128)


def master() -> Image.Image:
    return Image.open(MASTERS / "cloudviewer_1024.png").convert("RGBA")


def render(size: int) -> Image.Image:
    body = M.cloudviewer_mark(master(), small=size < 64)
    png = cairosvg.svg2png(
        bytestring=M.wrap(body).encode("utf-8"),
        output_width=size,
        output_height=size,
    )
    return Image.open(io.BytesIO(png)).convert("RGBA")


def vector_svg() -> str:
    return M.wrap(M.cloudviewer_mark(master()))


def _write(path: Path, data: bytes, dry: bool) -> None:
    if dry:
        print(f"  [dry-run] {path.relative_to(ROOT)} ({len(data):,} B)")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Regenerate the CloudViewer icon set.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    icon_dir = ROOT / "app/Resources/images/icon"

    for size in VECTOR_SIZES:
        img = render(size)
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        _write(icon_dir / f"cloudviewer_{size}.png", buf.getvalue(), args.dry_run)

        ibuf = io.BytesIO()
        img.save(ibuf, format="ICO", sizes=[(size, size)])
        _write(icon_dir / f"cloudviewer_{size}.ico", ibuf.getvalue(), args.dry_run)

    # A true vector master: this replaces a 559 KB SVG that was really a
    # base64-wrapped PNG and had no vector geometry in it at all.
    _write(icon_dir / "cloudviewer_256.svg", vector_svg().encode(), args.dry_run)

    print("CloudViewer icon set written" + (" (dry run)" if args.dry_run else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
