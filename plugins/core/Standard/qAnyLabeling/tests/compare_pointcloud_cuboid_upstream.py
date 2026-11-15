#!/usr/bin/env python3
"""Transient numeric truth from the named upstream's actual cuboid.py."""
import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--probe', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    folder = args.upstream / 'anylabeling/views/labeling/pointcloud'
    package = types.ModuleType('parity_pointcloud')
    package.__path__ = [str(folder)]
    sys.modules[package.__name__] = package
    modules = {}
    for name in ('cuboid', 'model', 'io'):
        spec = importlib.util.spec_from_file_location(f'{package.__name__}.{name}', folder / f'{name}.py')
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        modules[name] = mod
    Cuboid = modules['cuboid'].Cuboid
    rng = np.random.default_rng(20261016)
    cases, expected = [], []
    for n in range(160):
        angles = rng.uniform(-np.pi, np.pi, 3)
        if n < 4:
            angles[1] = (-1 if n % 2 else 1) * np.pi / 2
        box = Cuboid(n + 1, n % 5 + 1, tuple(rng.uniform(-10, 10, 3)),
                     tuple(rng.uniform(.01, 8, 3)), tuple(angles), n % 2 == 0, False)
        points = rng.uniform(-12, 12, (30, 3)).astype(np.float32)
        contains = box.contains(points).tolist()
        operations = []
        original = asdict(box)
        for step in range(12):
            if step % 2:
                axis, angle = int(rng.integers(0, 3)), float(rng.uniform(-2, 2))
                operations.append(dict(type='rotate', axis=axis, angle=angle))
                box = box.rotated(axis, angle)
            else:
                signs = rng.integers(-1, 2, 3).tolist()
                delta = rng.uniform(-5, 5, 3)
                operations.append(dict(type='resize', signs=signs, delta=delta.tolist()))
                box = box.resized(range(3), signs, delta)
        cases.append(dict(box=original, points=points.tolist(), operations=operations))
        expected.append(dict(valid=True, box=asdict(box), corners=box.corners().tolist(),
                             contains=contains, fitted=asdict(box.fitted(points))))
    valid = cases[0]['box']
    for key, value in [('id', 0), ('id', True), ('class_id', 65536),
                       ('size', [0, 1, 2]), ('center', [1, 2]), ('locked', 1), ('extra', 0)]:
        invalid = dict(valid)
        invalid[key] = value
        cases.append(dict(box=invalid))
        expected.append(dict(valid=False))
    with tempfile.TemporaryDirectory(prefix='qany-cuboid3d-') as tmp:
        tmp = Path(tmp)
        source = tmp / 'input.json'
        output = tmp / 'output.json'
        source.write_text(json.dumps(cases))
        subprocess.run([str(args.probe.resolve()), '--probe', str(source), str(output)], check=True)
        actual = json.loads(output.read_text())
        assert len(actual) == len(expected)
        maximum = 0.0
        for index, (got, want) in enumerate(zip(actual, expected)):
            assert got['valid'] == want['valid'], (index, got, want)
            if not want['valid']:
                continue
            assert got['contains'] == want['contains'], index
            for name in ('box', 'fitted'):
                for key in ('id', 'class_id', 'occluded', 'locked'):
                    assert got[name][key] == want[name][key], (index, name, key)
                for key in ('center', 'size', 'rotation'):
                    delta = float(np.max(np.abs(np.array(got[name][key]) - want[name][key])))
                    maximum = max(maximum, delta)
                    assert delta < 1e-8, (index, name, key, delta)
            np.testing.assert_allclose(got['corners'], want['corners'], atol=1e-8, rtol=0)
        # Actual upstream loader accepts the plugin's emitted schema and its
        # writer produces the same JSON object, independent of key ordering.
        sidecar = tmp / 'sample.cuboids.json'
        source_path = tmp / 'sample.bin'
        box = Cuboid(**actual[0]['box'])
        modules['io'].save_cuboids(sidecar, (box,), source_path)
        plugin_sidecar = tmp / 'plugin.cuboids.json'
        subprocess.run([str(args.probe.resolve()), '--roundtrip', str(sidecar),
                        str(plugin_sidecar), str(source_path)], check=True)
        assert modules['io'].load_cuboids(plugin_sidecar, source_path) == (box,)
        assert json.loads(plugin_sidecar.read_text()) == json.loads(sidecar.read_text())
    report = dict(verdict='PASS', upstream_revision=subprocess.check_output(
        ['git', '-C', str(args.upstream), 'rev-parse', 'HEAD'], text=True).strip(),
        source_sha256={name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
                       for name in ('cuboid.py', 'io.py', 'model.py')},
        cases=len(cases), valid_scenarios=160, sequential_edits=1920,
        coordinate_tolerance=1e-8, maximum_delta=maximum)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
