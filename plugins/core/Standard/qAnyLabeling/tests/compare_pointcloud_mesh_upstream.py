#!/usr/bin/env python3
"""Compare emitted GL mesh with actual upstream CuboidViewport method."""
import argparse
import ast
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
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
    folder = args.upstream / 'anylabeling/views/labeling/pointcloud'
    source = folder / 'cuboid_viewport.py'
    spec = importlib.util.spec_from_file_location('mesh_cuboid', folder / 'cuboid.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    cls = next(n for n in ast.parse(source.read_text()).body
               if isinstance(n, ast.ClassDef) and n.name == 'CuboidViewport')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                  and n.name == '_update_cuboid_mesh')
    namespace = dict(np=np, QtGui=QtGui, FACES=module.FACES, EDGES=module.EDGES)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)
    rng = np.random.default_rng(1432)
    cases, expected = [], []
    for n in range(96):
        boxes = tuple(module.Cuboid(i+1, i+1, tuple(rng.uniform(-20, 20, 3)),
                      tuple(rng.uniform(.01, 5, 3)), tuple(rng.uniform(-np.pi, np.pi, 3)),
                      (n+i) % 2 == 0, n % 3 == 0) for i in range(n % 4))
        preview = module.Cuboid(9, 1, (0., 1., 2.), (1., 1., 1.)) if n % 3 == 0 else None
        selected = n % 4 + 1
        colors = {1: '#039be5', 3: '#feab01'}
        orthographic = n % 2 == 0
        obj = SimpleNamespace(cuboids=boxes, selected_id=selected, _creation_preview=preview,
                              class_colors=colors, orthographic_view='top' if orthographic else None,
                              _gl=None, _update=lambda: None)
        namespace['_update_cuboid_mesh'](obj)
        vertices, face_count = obj._cuboid_mesh
        expected.append((vertices, face_count))
        item = dict(boxes=[asdict(b) for b in boxes], selected=selected,
                    colors=colors, orthographic=orthographic)
        if preview is not None:
            item['preview'] = asdict(preview)
        cases.append(item)
    with tempfile.TemporaryDirectory(prefix='qany-mesh-') as temporary:
        inputs, outputs = Path(temporary) / 'input.json', Path(temporary) / 'output.json'
        inputs.write_text(json.dumps(cases))
        subprocess.run([str(args.probe.resolve()), '--mesh-probe', str(inputs), str(outputs)], check=True)
        actual = json.loads(outputs.read_text())
    assert len(actual) == len(expected)
    maximum, vertex_count = 0., 0
    for index, (got, (vertices, faces)) in enumerate(zip(actual, expected)):
        assert got['face_vertices'] == faces, (index, got['face_vertices'], faces)
        output = np.asarray(got['vertices'], dtype=np.float32).reshape(-1, 7)
        assert output.shape == vertices.shape, (index, output.shape, vertices.shape)
        # Both sides expose the float32 GPU buffer, so no geometric tolerance
        # is needed: this catches winding, metric dashes, alpha and ordering.
        np.testing.assert_array_equal(output, vertices, err_msg=f'case {index}')
        if output.size:
            maximum = max(maximum, float(np.max(np.abs(output - vertices))))
        vertex_count += len(vertices)
    report = dict(verdict='PASS', cases=len(cases), vertices=vertex_count,
                  maximum_float32_delta=maximum,
                  upstream_revision=subprocess.check_output(
                      ['git', '-C', str(args.upstream), 'rev-parse', 'HEAD'], text=True).strip(),
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  geometry_sha256=hashlib.sha256((folder / 'cuboid.py').read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
