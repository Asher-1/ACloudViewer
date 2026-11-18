"""Synchronise the docs marks with the real product artwork.

The defect this fixes
---------------------
``docs/images/ACloudViewer.svg`` embedded a *different, simplified* drawing of
the mark: a flat cube with one orange dot, 8,249 bytes, sha1 60cc5cf.  The real
artwork in ``app/Resources/images/icon/logo_256.png`` is the full wireframe cube
with the point burst, 97,980 bytes, sha1 5c114b2.  Measured: 45/255 mean
absolute difference and only 31.7% alpha coverage versus 50.0% -- the docs were
showing a watered-down version of the brand.

Nothing about the design changes here.  The docs get the same artwork the app
ships, re-rendered from the 1024 px master so it stays sharp.

Usage
-----
    python3 -m scripts.icon_hd.logo.docs            # write
    python3 -m scripts.icon_hd.logo.docs --dry-run
"""

from __future__ import annotations

import argparse
import base64
import io
import sys
from pathlib import Path

import cairosvg
from PIL import Image

from . import lockup

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MASTERS = HERE / "masters"

DOCS = ROOT / "docs/images"

# The icon mark at the size the docs header displays it.  Measured trade-off:
#   128 px ->  40 KB    256 px -> 140 KB    384 px -> 288 KB    512 px -> 474 KB
# The docs header draws it under 150 px, so 256 px leaves a 1.7x margin for
# HiDPI without the 474 KB weight of the 512 px variant.
ICON_SIZE = 256

# Lockup icon height.  The PNG is the heaviest consumer (README, docs header,
# Doxygen), so it is rendered at the size those surfaces actually need, while
# the SVG stays resolution-independent for anyone who needs more:
#   256 -> 1014x266    384 -> 1521x399    512 -> 2028x532
# 384 covers a 4K header at 1:1; the SVG covers 8K.
LOCKUP_ICON_HEIGHT = 384


def icon_svg(size: int = ICON_SIZE) -> str:
    """The product mark as a standalone SVG, embedding the master artwork."""
    master = Image.open(MASTERS / "acloudviewer_1024.png").convert("RGBA")
    img = master.resize((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}">\n'
        f'  <image x="0" y="0" width="{size}" height="{size}" '
        f'href="data:image/png;base64,{b64}"/>\n'
        f'</svg>\n'
    )


def write(path: Path, data: bytes, dry: bool) -> None:
    if dry:
        print(f"  [dry-run] {path.relative_to(ROOT)} ({len(data):,} B)")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    print(f"  wrote {path.relative_to(ROOT)} ({len(data):,} B)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Sync docs marks with product artwork.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    # 1. The docs icon: real artwork, not a simplified stand-in.
    write(DOCS / "ACloudViewer.svg", icon_svg().encode(), args.dry_run)

    # 2. The lockup: same design, resolution suited to 4K/8K.
    lock_svg = lockup.build(icon_height=LOCKUP_ICON_HEIGHT)
    write(DOCS / "ACloudViewer_logo_horizontal.svg", lock_svg.encode(), args.dry_run)
    png = cairosvg.svg2png(bytestring=lock_svg.encode("utf-8"))
    write(DOCS / "ACloudViewer_logo_horizontal.png", png, args.dry_run)

    return 0


if __name__ == "__main__":
    sys.exit(main())
