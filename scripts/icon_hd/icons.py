#!/usr/bin/env python3
"""One entry point for the whole icon toolchain.

Why this exists
---------------
Before this file there were nineteen modules in ``logo/`` and seven of them had
their own ``main``.  Nothing listed which subcommand existed, nothing checked
that the dependencies were installed, and nothing said which of the four
pipelines -- design, export, verify, document -- a given script belonged to.
That is not "ready to use"; it is a drawer of tools.

So: one command, four verbs, and a dependency check that runs before anything
else and names the pip command that fixes a miss.

    python3 scripts/icon_hd/icons.py doctor
    python3 scripts/icon_hd/icons.py build
    python3 scripts/icon_hd/icons.py audit
    python3 scripts/icon_hd/icons.py verify
    python3 scripts/icon_hd/icons.py logo

Every verb prints what it is about to do, and fails loudly with the actual
error rather than a summary that hides it.
"""

from __future__ import annotations

import argparse
import importlib
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LOGO = HERE / "logo"

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


# ---------------------------------------------------------------------------
# Dependencies, declared once
# ---------------------------------------------------------------------------

CORE = {
    "cairosvg": "cairosvg",          # SVG -> PNG for every measurement
    "PIL": "Pillow",                 # image IO
    "numpy": "numpy",                # ink coverage, diffs
}
DESIGN = {
    "cv2": "opencv-python",          # contour tracing in vectorize
    "skimage": "scikit-image",       # sub-pixel contours
    "scipy": "scipy",                # inpainting in repair
    "fontTools": "fonttools",        # wordmark outlines in lockup
}


def _have(mod: str) -> bool:
    try:
        importlib.import_module(mod)
        return True
    except Exception:
        return False


def doctor(strict: bool = False) -> int:
    """Report what is installed and what each missing package would cost."""
    print("=== 依赖自检 ===\n")
    rows = []
    for group, pkgs in (("核心（所有命令）", CORE), ("设计（build 的追踪/母版）", DESIGN)):
        print(f"{group}:")
        for mod, pipname in pkgs.items():
            ok = _have(mod)
            print(f"  {'OK     ' if ok else 'MISSING'}  {mod:<12} pip install {pipname}")
            rows.append((mod, ok, pipname))
        print()

    ok_core = all(_have(m) for m in CORE)
    ok_design = all(_have(m) for m in DESIGN)

    print("=== 其它 ===")
    print(f"  {'OK     ' if shutil.which('rcc') else 'MISSING'}  rcc"
          "          Qt 资源编译器，verify 用它验证 qrc 可编译")
    print(f"  {'OK     ' if shutil.which('git') else 'MISSING'}  git"
          "          audit 用 git show 读取改动前的旧图标")
    print()

    if ok_core:
        print("核心依赖齐全：build / audit / verify / logo 均可运行。")
        if not ok_design:
            print("设计依赖不全：build 会跳过追踪类图标，其余命令不受影响。")
        return 0

    missing = " ".join(p for m, ok, p in rows if not ok and m in CORE)
    print(f"核心依赖缺失。安装：\n    pip install {missing}")
    return 1 if strict else 0


# ---------------------------------------------------------------------------
# Verbs
# ---------------------------------------------------------------------------


def _run_module(dotted: str, argv: list[str]) -> int:
    """Import a module under logo/ and hand it argv, reporting real tracebacks."""
    print(f"--- {dotted} {' '.join(argv)}")
    try:
        mod = importlib.import_module(dotted)
    except Exception as exc:
        print(f"    无法加载 {dotted}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    try:
        return int(mod.main(argv) or 0)
    except SystemExit as e:
        return int(e.code or 0)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"    {dotted} 失败: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def build(argv: list[str]) -> int:
    """Render every icon from logo/*.py and write the SVGs into the repository."""
    print("=== build: 由设计系统渲染全部工具图标 ===")
    from logo import (fonticons, miscicons, pqicons, pqvariants, rooticons,
                      slicons, toolicons, treeicons)

    images = ROOT / "app/Resources/images"
    mods = (toolicons, treeicons, pqicons, pqvariants,
            rooticons, miscicons, slicons, fonticons)

    written = skipped = 0
    total = 0
    for mod in mods:
        for key, fn in mod.REGISTRY.items():
            hits = list(images.rglob(f"{key.split('/')[-1]}.svg"))
            if not hits:
                skipped += 1
                print(f"  SKIP  {key}  (仓库中无同名文件，不新建路径)")
                continue
            text = fn()
            hits[0].write_text(text)
            written += 1
            total += len(text)
    print(f"\n  写入 {written} 个，跳过 {skipped} 个，共 {total:,} B")
    print("  SVG 已就地更新；调用方的 qrc/ui 路径无需改动。")
    return 0


def audit(argv: list[str]) -> int:
    """Build the before/after review page."""
    print("=== audit: 生成优化前后对照报告 ===")
    return _run_module("logo.audit", argv)


def verify(argv: list[str]) -> int:
    """qrc compiles, SVG parses, no raster disguised as vector, no dangling refs."""
    print("=== verify: 结构完整性 ===")
    rc = 0

    print("\n[1/4] qrc 可编译")
    rcc = shutil.which("rcc")
    if not rcc:
        print("  SKIP  未找到 rcc")
    else:
        fails = 0
        for q in ROOT.rglob("*.qrc"):
            if "3rdparty" in str(q) or "build" in str(q).lower():
                continue
            p = subprocess.run([rcc, "--list", str(q)], capture_output=True, text=True)
            warn = (p.stderr or "").strip()
            if p.returncode != 0 or warn:
                fails += 1
                print(f"  FAIL  {q.relative_to(ROOT)}")
                for line in warn.splitlines()[:4]:
                    print(f"        {line}")
        print(f"  {'PASS' if not fails else 'FAIL'}  {fails} 个 qrc 有问题")
        rc |= 1 if fails else 0

    print("\n[2/4] SVG 可解析")
    import xml.etree.ElementTree as ET
    n = bad = 0
    for p in (ROOT / "app/Resources/images").rglob("*.svg"):
        n += 1
        try:
            ET.parse(p)
        except Exception as e:
            bad += 1
            print(f"  FAIL  {p.relative_to(ROOT)}: {e}")
    print(f"  {'PASS' if not bad else 'FAIL'}  {n} 个 SVG，{bad} 个非法")
    rc |= 1 if bad else 0

    print("\n[3/4] 无位图伪装成矢量")
    fake = []
    for p in (ROOT / "app/Resources/images").rglob("*.svg"):
        t = p.read_text(errors="ignore")
        if "base64" in t or "<image" in t:
            fake.append(p.relative_to(ROOT))
    for f in fake:
        print(f"  WARN  {f} 内嵌位图")
    print(f"  {'PASS' if not fake else 'WARN'}  {len(fake)} 个文件内嵌位图"
          "（产品 logo 允许，工具图标不允许）")

    print("\n[4/4] 平台依赖")
    rc |= _run_module("logo.platforms", [])

    print(f"\n=== verify {'PASS' if rc == 0 else 'FAIL'} ===")
    return rc


def logo(argv: list[str]) -> int:
    """The product-mark pipeline (measure / repair / export)."""
    print("=== logo: 产品标识 ===")
    if not argv:
        print("  用法: logo measure | repair | export")
        print("    measure  打印母版几何（可选 --write 刷新 particles.json）")
        print("    repair   修复 ACloudViewer 母版核心孔洞")
        print("    export   导出 CloudViewer 尺寸阶梯")
        return 0
    verb, rest = argv[0], argv[1:]
    return {
        "measure": lambda: _run_module("logo.measure", rest),
        "repair": lambda: _run_module("logo.repair", rest),
        "export": lambda: _run_module("logo.export_cloudviewer", rest),
        "docs": lambda: _run_module("logo.docs", rest),
    }.get(verb, lambda: (print(f"  未知子命令: {verb}"), 1)[1])()


# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="icons.py",
        description="ACloudViewer 图标工具链（唯一入口）",
        epilog="先跑 doctor 确认依赖，再 build / audit / verify。",
    )
    sub = ap.add_subparsers(dest="verb")

    p = sub.add_parser("doctor", help="检查依赖，指出缺什么、怎么装")
    p.add_argument("--strict", action="store_true", help="缺依赖时返回非零")

    sub.add_parser("build", help="由设计系统渲染并写入全部工具图标")

    p = sub.add_parser("audit", help="生成优化前后对照报告")
    p.add_argument("--out", default=None)
    p.add_argument("--only", default=None)

    sub.add_parser("verify", help="qrc/XML/位图伪装/平台依赖 四项校验")

    p = sub.add_parser("logo", help="产品标识管线")
    p.add_argument("args", nargs=argparse.REMAINDER)

    a = ap.parse_args(argv)

    if a.verb == "doctor":
        return doctor(a.strict)
    if a.verb == "build":
        if not all(_have(m) for m in CORE):
            print("核心依赖缺失，先跑 doctor。", file=sys.stderr)
            return 1
        return build([])
    if a.verb == "audit":
        rest = []
        if a.out:
            rest += ["--out", a.out]
        if a.only:
            rest += ["--only", a.only]
        return audit(rest)
    if a.verb == "verify":
        return verify([])
    if a.verb == "logo":
        return logo(a.args)

    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
