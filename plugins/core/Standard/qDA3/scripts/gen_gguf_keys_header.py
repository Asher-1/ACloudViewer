#!/usr/bin/env python3
"""Generate core/AICore/include/aicore/depth_gguf_keys.h from gguf_keys.py."""
import sys
from pathlib import Path

QDA3 = Path(__file__).resolve().parent.parent
REPO = QDA3.parents[3]
sys.path.insert(0, str(QDA3))
import scripts.gguf_keys as K

CANONICAL = REPO / "core/AICore/include/aicore/depth_gguf_keys.h"


def cident_canonical(short: str) -> str:
    return "AICORE_DEPTH_KV_" + short.replace(".", "_").upper()


def render_canonical() -> str:
    idents = [cident_canonical(s) for s in K.KV]
    assert len(set(idents)) == len(idents), "cident collision in K.KV"
    lines = [
        "// AUTO-GENERATED from plugins/core/Standard/qDA3/scripts/gguf_keys.py",
        "// Canonical GGUF KV strings for the depth module. Do not edit by hand.",
        "#pragma once",
        "",
    ]
    for short, full in K.KV.items():
        lines.append(f'#define {cident_canonical(short)} "{full}"')
    lines.append(f'#define AICORE_DEPTH_ARCH "{K.ARCH}"')
    return "\n".join(lines) + "\n"


def main():
    CANONICAL.parent.mkdir(parents=True, exist_ok=True)
    CANONICAL.write_text(render_canonical())
    print(f"wrote {CANONICAL.relative_to(REPO)}")


if __name__ == "__main__":
    main()
