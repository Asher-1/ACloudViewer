#!/usr/bin/env python3
"""Execute the upstream color methods independently of the C++ rendering probe."""
import argparse
import ast
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import numpy as np
from PyQt6 import QtGui


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = args.upstream / 'anylabeling/views/labeling/widgets/pointcloud_dialog.py'
    tree = ast.parse(source.read_text())
    methods = [node for cls in tree.body if isinstance(cls, ast.ClassDef)
               for node in cls.body if isinstance(node, ast.FunctionDef)
               and node.name in ('_class_palette', '_intensity_colors', '_display_colors')]
    assert len(methods) == 3
    namespace = dict(np=np, QtGui=QtGui)
    for method in methods:
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(source), 'exec'), namespace)
    @dataclass(frozen=True)
    class Definition:
        id: int
        color: str
    class Reference:
        tr = staticmethod(lambda value: value)
        _set_render_legend = staticmethod(lambda value: None)
    for method in methods:
        setattr(Reference, method.name, namespace[method.name])
    reference = Reference()
    reference._palette_classes = None
    palette = {0: '#808080', 1: '#039be5', 65535: '#feab01'}
    reference.class_definitions = {'segmentation': [Definition(i, c) for i, c in palette.items()]}
    rng = np.random.default_rng(341271)
    labels = np.r_[np.array([0, 1, 65535, 65536, 65537, 0xffffffff], np.uint32), rng.integers(0, 2**32, 2048, dtype=np.uint32)]
    intensity = rng.uniform(-25, 73, len(labels)).astype(np.float32)
    intensity[0], intensity[1] = -25, 73
    points = np.zeros((len(labels), 4), np.float32)
    points[:, 3] = intensity
    indices = np.arange(len(labels))
    rows, expected = [], []
    for mode, name in enumerate(('semantic', 'intensity', 'rgb', 'instance')):
        reference.color_mode = SimpleNamespace(currentData=lambda: name)
        for with_rgb in (False, True):
            rgb = rng.integers(0, 256, (len(labels), 3), dtype=np.uint8) if with_rgb else None
            reference.document = SimpleNamespace(frame=SimpleNamespace(points=points, rgb=rgb),
                labels=labels, semantic_view=(labels & 65535).astype(np.uint16), instance_view=labels >> 16)
            for selected in (False, True):
                reference._intensity_range = None
                reference._palette_classes = None  # Method mutates its temporary semantic colors.
                colors = reference._display_colors(indices, np.ones(len(labels), bool) if selected else None)
                expected.extend(colors)
                for index, label in enumerate(labels):
                    rows.append(dict(mode=mode, label=int(label), intensity=float(intensity[index]),
                        low=-25, high=73, selected=selected, palette=palette,
                        rgb=rgb[index].tolist() if rgb is not None else None))
    with tempfile.TemporaryDirectory(prefix='qany-colors-') as folder:
        input_path, output_path = Path(folder)/'input.json', Path(folder)/'output.json'
        input_path.write_text(json.dumps(rows))
        subprocess.run([str(args.probe.resolve()), '--colors-probe', str(input_path), str(output_path)], check=True)
        actual = np.asarray(json.loads(output_path.read_text()), np.float32)
    expected = np.asarray(expected, np.float32)
    delta = np.abs(actual - expected)
    report = dict(verdict='PASS' if np.array_equal(actual, expected) else 'FAIL', cases=len(rows),
        differing_channels=int(np.count_nonzero(delta)), maximum_delta=float(delta.max()),
        upstream_revision=subprocess.check_output(['git', '-C', str(args.upstream), 'rev-parse', 'HEAD'], text=True).strip(),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))
    return report['verdict'] != 'PASS'


if __name__ == '__main__':
    raise SystemExit(main())
