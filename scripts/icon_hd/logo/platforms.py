"""Verify that no icon asset needs a platform chain the repository cannot satisfy.

Why this exists
---------------
Deleting a PNG that only the Windows ``.rc`` mentions, or an ``.icns`` that only
``PostInstall.cmake`` renames, breaks packaging on a platform the developer is
not running.  That failure is invisible locally and only surfaces in CI or at
release time -- the worst possible place to find it.

So the check is mechanical, not memory-based.  It derives the set of assets the
repository *actually ships* from four independent mechanisms and asserts that
every one of them resolves:

    qrc        every <file> in every .qrc resolves on disk
    rc         every ICON/BITMAP path inside a Windows .rc
    cmake      every quoted Resources/images/*.ext in CMake files
    desktop    every Icon= in a .desktop (may be a theme name, not a path)

Usage
-----
    python3 -m scripts.icon_hd.logo.platforms
    python3 -m scripts.icon_hd.logo.platforms --json build/platforms.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
IMAGES = ROOT / "app/Resources/images"

SKIP_DIRS = {"build", "build_app", "build_headless_docs", "3rdparty", ".git", "node_modules"}


def _walk(suffixes):
    for p in ROOT.rglob("*"):
        if p.suffix.lower() not in suffixes or not p.is_file():
            continue
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        yield p


# ---------------------------------------------------------------------------
# Each mechanism returns a list of findings
# ---------------------------------------------------------------------------


def check_qrc() -> list[dict]:
    """Every resource listed in a .qrc must exist next to that .qrc."""
    out = []
    for q in _walk({".qrc"}):
        try:
            text = q.read_text(errors="ignore")
        except OSError:
            continue
        base = q.parent
        for m in re.finditer(r"<file[^>]*>([^<]+)</file>", text):
            rel = m.group(1).strip()
            out.append({
                "mechanism": "qrc",
                "declared_in": str(q.relative_to(ROOT)),
                "asset": rel,
                "resolved": (base / rel).exists(),
                "platform": "all",
            })
    return out


def check_rc() -> list[dict]:
    """Windows resource scripts reference .ico by bare filename, relative to them."""
    out = []
    for rc in _walk({".rc"}):
        try:
            text = rc.read_text(errors="ignore")
        except OSError:
            continue
        for m in re.finditer(r'(?:ICON|BITMAP|RCDATA)\s+\w*\s*(?:,\s*)?DISCARDABLE\s+"([^"]+)"',
                             text, re.I):
            target = m.group(1)
            out.append({
                "mechanism": "rc",
                "declared_in": str(rc.relative_to(ROOT)),
                "asset": target,
                "resolved": (rc.parent / target).exists(),
                "platform": "windows",
            })
    return out


def check_cmake() -> list[dict]:
    """CMake mentions icon paths as quoted strings; resolve each one."""
    out = []
    for cm in _walk({".cmake", ".txt"}):
        if cm.name != "CMakeLists.txt" and cm.suffix != ".cmake":
            continue
        try:
            text = cm.read_text(errors="ignore")
        except OSError:
            continue
        for m in re.finditer(r'"([^"\n]*Resources/images/[^"\n]+\.(?:png|ico|icns|svg|gif|rc))"',
                             text):
            rel = m.group(1)
            cand = (ROOT / rel).resolve()
            # CMake paths are often relative to the project or to the source dir
            resolved = cand.exists() or (ROOT / "app" / rel).exists()
            out.append({
                "mechanism": "cmake",
                "declared_in": str(cm.relative_to(ROOT)),
                "asset": rel,
                "resolved": resolved,
                "platform": "packaging",
            })
    return out


def check_desktop() -> list[dict]:
    """``Icon=`` is either a bare name next to the .desktop or an absolute path.

    A bare name is *not* relative to the repository root -- it is resolved by the
    desktop environment against the icon theme after installation, and the file
    that gets installed sits beside the .desktop.  Treating it as a repo-relative
    path produced two false "broken" findings on ``ACloudViewer.desktop`` and
    ``CloudViewer.desktop`` even though both PNGs are present in those folders.
    """
    out = []
    for d in _walk({".desktop"}):
        try:
            text = d.read_text(errors="ignore")
        except OSError:
            continue
        m = re.search(r"^Icon=(.+)$", text, re.M)
        if not m:
            continue
        icon = m.group(1).strip()
        is_path = "/" in icon or icon.lower().endswith(
            (".png", ".svg", ".ico", ".icns", ".xpm"))
        if not is_path:
            out.append({"mechanism": "desktop", "declared_in": str(d.relative_to(ROOT)),
                        "asset": icon, "resolved": True, "platform": "linux",
                        "note": "主题名，运行时解析"})
            continue
        # A bare filename resolves beside the .desktop; a rooted path against ROOT.
        cand = (d.parent / icon) if "/" not in icon else (ROOT / icon.lstrip("/"))
        out.append({"mechanism": "desktop", "declared_in": str(d.relative_to(ROOT)),
                    "asset": icon, "resolved": cand.exists(), "platform": "linux"})
    return out


# ---------------------------------------------------------------------------
# Product-icon inventory per platform
# ---------------------------------------------------------------------------

PLATFORM_ASSETS = {
    "windows": ["app/Resources/images/icon/logo_256.ico",
                "app/Resources/images/icon/ecv_icon.rc"],
    "macos": ["app/Mac/logo_256.icns"],
    "linux": ["scripts/ACloudViewer/ACloudViewer.desktop"],
}


def product_icons() -> list[dict]:
    """The size ladders that must survive on every platform."""
    out = []
    for p in sorted((IMAGES / "icon").iterdir()):
        if not p.is_file():
            continue
        platform = {".ico": "windows", ".icns": "macos", ".rc": "windows"}.get(
            p.suffix.lower(), "all")
        out.append({"platform": platform, "asset": str(p.relative_to(ROOT)),
                    "bytes": p.stat().st_size})
    for label, paths in PLATFORM_ASSETS.items():
        for rel in paths:
            if not (ROOT / rel).exists():
                out.append({"platform": label, "asset": rel, "bytes": 0,
                            "MISSING": True})
    return out


# ---------------------------------------------------------------------------


def run() -> dict:
    findings = check_qrc() + check_rc() + check_cmake() + check_desktop()
    icons = product_icons()
    broken = [f for f in findings if not f["resolved"]]
    missing_platform = [i for i in icons if i.get("MISSING")]
    return {
        "summary": {
            "references_checked": len(findings),
            "references_broken": len(broken),
            "product_icons": len(icons),
            "platform_assets_missing": len(missing_platform),
        },
        "by_mechanism": {
            m: sum(1 for f in findings if f["mechanism"] == m)
            for m in ("qrc", "rc", "cmake", "desktop")
        },
        "broken": broken,
        "missing_platform": missing_platform,
        "product_icons": icons,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Verify icon availability per platform.")
    ap.add_argument("--json", default=None, help="write the full report here")
    a = ap.parse_args(argv)

    r = run()
    s = r["summary"]
    print("=== 图标平台依赖核验 ===")
    print(f"  引用总数        {s['references_checked']}")
    print(f"  破坏的引用      {s['references_broken']}")
    print(f"  产品图标        {s['product_icons']}")
    print(f"  平台资源缺失    {s['platform_assets_missing']}")
    print("  按机制:", r["by_mechanism"])
    if r["broken"]:
        print("\n  破坏项:")
        for b in r["broken"][:20]:
            print(f"    {b['mechanism']:<8} {b['declared_in']}  ->  {b['asset']}")
    if r["missing_platform"]:
        print("\n  平台缺失:")
        for m in r["missing_platform"]:
            print(f"    {m['platform']:<8} {m['asset']}")

    ok = s["references_broken"] == 0 and s["platform_assets_missing"] == 0
    print("\n" + ("PASS: 所有平台的图标依赖均可解析" if ok else "FAIL: 见上"))
    if a.json:
        out = Path(a.json)
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(r, indent=1, ensure_ascii=False))
        print(f"report -> {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
