#!/usr/bin/env python3
"""Keep RelateAnything inference on ggml and the existing Qt dependency."""

import argparse
from pathlib import Path
import re
import sys
import tempfile
import unittest


FORBIDDEN_RELATION_DEPENDENCIES = (
    "3rdparty/ggml/relation",
    "pytorch_mem_eff",
    "mem_eff_attention",
    "ATen/native/transformers",
    "GGML_SAM3D_MEMEFF_DIR",
)
FORBIDDEN_INCLUDES = re.compile(
    r'^\s*#\s*include\s*[<"](?:ATen|torch|c10|cutlass|unicode)/')
DIRECT_ICU_LINK = re.compile(
    r'target_link_libraries\s*\(\s*(AICore(?:_test)?)\s+[^)]*\bICU::',
    re.IGNORECASE)


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
            # A merged patch carries context from other task owners. Only its
            # additions can introduce a dependency into RelateAnything.
            if path.suffix == ".patch" and (not line.startswith("+") or line.startswith("+++")):
                continue
            source_line = line[1:] if path.suffix == ".patch" else line
            if forbidden.search(line) or FORBIDDEN_INCLUDES.search(source_line):
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
    aicore_cmake = root / "core/AICore/CMakeLists.txt"
    if aicore_cmake.is_file():
        # Qt's transitive Unicode dependencies are outside this check. It
        # protects the runtime and white-box archive's direct link contract.
        text = re.sub(r'#[^\n]*', '', aicore_cmake.read_text(encoding="utf-8"))
        for match in DIRECT_ICU_LINK.finditer(text):
            violations.append(f"{match[1]} has an unnecessary direct ICU link")
    return violations


class RelationDependencyClosureTests(unittest.TestCase):
    root = Path(__file__).resolve().parents[4]

    def test_relation_has_no_pytorch_attention_dependency(self):
        root = Path(self.root)
        self.assertEqual(check_dependency_closure(root), [])

    def test_patch_context_is_allowed_but_new_dependency_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            patches = root / "3rdparty/ggml/patches/relation_merged"
            patches.mkdir(parents=True)
            (root / "3rdparty/ggml/ggml.cmake").write_text("")
            patch = patches / "test.patch"
            patch.write_text(" GGML_SAM3D_MEMEFF_DIR\n-pytorch_mem_eff\n+ggml_flash_attn_ext\n")
            self.assertEqual(check_dependency_closure(root), [])
            patch.write_text("+pytorch_mem_eff\n")
            self.assertEqual(len(check_dependency_closure(root)), 1)

    def test_framework_and_icu_includes_are_rejected_in_sources_and_patch_additions(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            sources = root / "core/AICore/src/tasks/relation"
            patches = root / "3rdparty/ggml/patches/relation_merged"
            sources.mkdir(parents=True)
            patches.mkdir(parents=True)
            (root / "3rdparty/ggml/ggml.cmake").write_text("")
            for dependency in ("ATen", "torch", "c10", "cutlass", "unicode"):
                with self.subTest(dependency=dependency):
                    (sources / "inference.cpp").write_text(
                        f'#include <{dependency}/runtime.h>\n')
                    (patches / "operator.patch").write_text(
                        f'+#include "{dependency}/runtime.h"\n')
                    self.assertEqual(len(check_dependency_closure(root)), 2)

    def test_runtime_and_white_box_direct_icu_links_are_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "3rdparty/ggml").mkdir(parents=True)
            (root / "3rdparty/ggml/ggml.cmake").write_text("")
            (root / "core/AICore").mkdir(parents=True)
            cmake = root / "core/AICore/CMakeLists.txt"
            cmake.write_text(
                "# target_link_libraries(AICore PRIVATE ICU::uc)\n"
                "target_link_libraries(AICore PRIVATE Qt5::Core)\n")
            self.assertEqual(check_dependency_closure(root), [])
            cmake.write_text(
                "target_link_libraries(AICore PRIVATE ICU::uc)\n"
                "target_link_libraries(AICore_test PRIVATE\n ICU::i18n)\n")
            self.assertEqual(len(check_dependency_closure(root)), 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path,
                        default=Path(__file__).resolve().parents[4])
    args, remaining = parser.parse_known_args()
    RelationDependencyClosureTests.root = args.root.resolve()
    return unittest.main(argv=[sys.argv[0], *remaining])


if __name__ == "__main__":
    main()
