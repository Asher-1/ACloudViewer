#!/usr/bin/env python3
"""Generate include/da_gguf_keys.h from scripts/gguf_keys.py."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import scripts.gguf_keys as K

ROOT = Path(__file__).resolve().parent.parent
HEADER_PATH = ROOT / "include" / "da_gguf_keys.h"

def cident(s): return "DA_KV_" + s.replace(".", "_").upper()

def render():
    idents = [cident(s) for s in K.KV]
    assert len(set(idents)) == len(idents), "cident collision in K.KV"
    lines = ["// AUTO-GENERATED from scripts/gguf_keys.py - do not edit.", "#pragma once", ""]
    for short, full in K.KV.items():
        lines.append(f'#define {cident(short)} "{full}"')
    lines.append(f'#define DA_ARCH "{K.ARCH}"')
    return "\n".join(lines) + "\n"

def main():
    HEADER_PATH.parent.mkdir(parents=True, exist_ok=True)
    HEADER_PATH.write_text(render())
    print("wrote include/da_gguf_keys.h")

if __name__ == "__main__":
    main()
