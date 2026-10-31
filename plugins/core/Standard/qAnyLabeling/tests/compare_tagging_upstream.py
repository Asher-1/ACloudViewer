#!/usr/bin/env python3
"""Compare the C++ presentation adapter to actual source RAM methods.

No model/truth outputs are vendored. The probe and pinned local source generate
the transient report requested by --output under the build directory.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.upstream / "anylabeling/services/auto_labeling/__base__/ram.py"
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RecognizeAnything")
    methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    normalization = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "normalize_tag_mode")
    module = ast.Module(body=[normalization, methods["postprocess"], methods["get_results"]], type_ignores=[])
    scope = {"np": np, "logger": SimpleNamespace(warning=lambda *_: None)}
    exec(compile(module, str(source), "exec"), scope)
    tags_source = args.upstream / "anylabeling/views/labeling/utils/image_tags.py"
    tag_methods = [node for node in ast.parse(tags_source.read_text()).body if isinstance(node, ast.FunctionDef)]
    exec(compile(ast.Module(body=tag_methods, type_ignores=[]), str(tags_source), "exec"), scope)
    folder = args.upstream / "anylabeling/services/auto_labeling/configs/ram"
    english = (folder / "ram_tag_list.txt").read_text().splitlines()
    chinese = (folder / "ram_tag_list_chinese.txt").read_text().splitlines()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    probe_data = args.output.with_suffix(".probe.json")
    subprocess.run([str(args.probe.resolve()), str(probe_data)], check=True)
    cases = json.loads(probe_data.read_text())
    for case in cases:
        deleted = case["delete_tags"]
        retained = case["filter_tags"]
        indices = ([english.index(label) for label in deleted] if deleted else
                   [index for index, label in enumerate(english) if label not in retained] if retained else [])
        adapter = SimpleNamespace(tag_list=np.array(english), tag_list_chinese=np.array(chinese),
                                  delete_tag_index=indices, tag_mode=scope["normalize_tag_mode"](case["mode"]))
        tags = scope["postprocess"](adapter, [np.ones((1, len(english))), np.array([1])])
        expected = scope["get_results"](adapter, tags)
        if case["tags"] != expected:
            raise AssertionError(f"RAM output mismatch: {case['mode']} / {deleted} / {retained}")
        if case["normalized_tags"] != scope["normalize_image_tags"](expected):
            raise AssertionError(f"Saved tag normalization mismatch: {case['mode']} / {deleted} / {retained}")
    report = {"verdict": "PASS", "label_count": len(english), "scenarios": len(cases),
              "upstream_commit": subprocess.check_output(["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip(),
              "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "normalization_sha256": hashlib.sha256(tags_source.read_bytes()).hexdigest(),
              "vocabulary_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.glob("*.txt")},
              "scope": "raw presentation strings, filter semantics and saved tag normalization; not model accuracy"}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
