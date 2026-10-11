#!/usr/bin/env python3
"""Keep RelateAnything free of the removed PyTorch attention dependency."""

import argparse
from pathlib import Path
import re
import sys
import unittest


FORBIDDEN_RELATION_DEPENDENCIES = (
    "3rdparty/ggml/relation",
    "pytorch_mem_eff",
    "mem_eff_attention",
    "ATen/native/transformers",
    "GGML_SAM3D_MEMEFF_DIR",
)


def relation_files(root: Path):
    roots = (
        root / "core/AICore/src/tasks/relation",
        root / "3rdparty/ggml/patches/relation_merged",
    )
    for directory in roots:
        yield from sorted(
            path for path in directory.rglob("*")
            if path.is_file() and path.suffix in {".c", ".cc", ".cpp", ".h", ".hpp", ".patch"}
        )


def check_dependency_closure(root: Path):
    forbidden = re.compile("|".join(re.escape(item) for item in FORBIDDEN_RELATION_DEPENDENCIES))
    violations = []
    for path in relation_files(root):
        text = path.read_text(encoding="utf-8", errors="replace")
        for line_number, line in enumerate(text.splitlines(), 1):
            if forbidden.search(line):
                violations.append(f"{path.relative_to(root)}:{line_number}: {line.strip()}")

    relation_memeff = root / "3rdparty/ggml/relation/pytorch_mem_eff"
    if relation_memeff.exists() or relation_memeff.is_symlink():
        violations.append(f"unexpected relation dependency directory: {relation_memeff.relative_to(root)}")

    # The SAM3D-only fetch block may mention its own headers in ggml.cmake;
    # the RelateAnything integration must never turn that path into a relation
    # setting or include.
    ggml_cmake = (root / "3rdparty/ggml/ggml.cmake").read_text(
        encoding="utf-8", errors="replace")
    if "GGML_CUDA_STRICT_ATTENTION_ROOT" in ggml_cmake:
        violations.append("ggml.cmake still exposes the removed relation strict-attention root")
    return violations


class RelationDependencyClosureTests(unittest.TestCase):
    root = Path(__file__).resolve().parents[4]

    def test_relation_has_no_pytorch_attention_dependency(self):
        root = Path(self.root)
        self.assertEqual(check_dependency_closure(root), [])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path,
                        default=Path(__file__).resolve().parents[4])
    args, remaining = parser.parse_known_args()
    RelationDependencyClosureTests.root = args.root.resolve()
    return unittest.main(argv=[sys.argv[0], *remaining])


if __name__ == "__main__":
    main()
