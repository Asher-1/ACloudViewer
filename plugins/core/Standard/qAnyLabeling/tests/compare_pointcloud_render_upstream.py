#!/usr/bin/env python3
"""Compare real Qt/OpenGL framebuffers; run under xvfb-run on Linux.

Captures the GL geometry surface only. Text, controls, axes and whole-window
theme parity require separate comparisons and are not claimed by this gate.
"""
import argparse
from dataclasses import asdict
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets


def pixels(image):
    image = image.convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
    data = image.bits()
    data.setsize(image.sizeInBytes())
    return np.frombuffer(data, np.uint8).reshape(image.height(), image.bytesPerLine())[:, :image.width()*4].reshape(image.height(), image.width(), 4).copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    folder = args.upstream / 'anylabeling/views/labeling/pointcloud'
    package = types.ModuleType('render_pointcloud')
    package.__path__ = [str(folder)]
    sys.modules[package.__name__] = package
    source = importlib.import_module(package.__name__ + '.cuboid_viewport')
    app = QtWidgets.QApplication([])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cloud = np.asarray([[x, y, z, .5] for x in np.linspace(-4, 4, 13)
                        for y in np.linspace(-3, 3, 11) for z in (-1., 0., 1.)], np.float32)
    colors = {0: '#808080', 1: '#039be5', 2: '#feab01'}
    cases, expected = [], []
    with tempfile.TemporaryDirectory(prefix='qany-render-') as temporary:
        temporary = Path(temporary)
        classes = temporary / 'classes.json'
        classes.write_text(json.dumps(dict(version=1,
            detection=dict(classes=[dict(id=i, name=str(i), color=colors[i]) for i in (1, 2)]),
            segmentation=dict(classes=[dict(id=0, name='Unlabeled', color=colors[0])]))))
        for index, (direction, selected, occluded) in enumerate(
                (('Top', -1, False), ('Top', 0, False), ('Front', 0, False),
                 ('Side', 1, True), ('', 0, False), ('', 1, True))):
            boxes = (source.Cuboid(1, 1, (0., 0., 0.), (4., 2., 2.), (.1, -.2, .4)),
                     source.Cuboid(2, 2, (1., -1., .5), (2., 2., 2.), (-.2, .1, -.3), occluded))
            path = temporary / f'cloud{index}.bin'
            cloud.tofile(path)
            path.with_suffix('.cuboids.json').write_text(json.dumps(dict(
                version=1, type='pointcloud-cuboids', point_cloud=path.name, cuboids=[asdict(b) for b in boxes])))
            # Source's actual widget and renderer; only suppress QWidget overlay
            # to isolate GL geometry from Qt5/Qt6 font raster differences.
            view = source.CuboidViewport()
            view.resize(640, 480)
            view._show_axes = False
            view.show()
            app.processEvents()
            view.set_cloud(cloud)
            view.set_colors(np.tile(np.array([255, 255, 255], np.uint8), (len(cloud), 1)))
            view.set_point_size(8)
            view.class_colors = colors
            view.detection_enabled = True
            view.set_cuboids(boxes, boxes[selected].id if selected >= 0 else None)
            if direction:
                view.set_view(direction.lower())
            loop = QtCore.QEventLoop()
            QtCore.QTimer.singleShot(150, loop.quit)
            loop.exec()
            view.validate_rendering()
            reference = view._gl.grabFramebuffer()
            expected.append(pixels(reference))
            reference.save(str(args.output.with_name(f'{args.output.stem}-{index}-upstream.png')))
            output = args.output.with_name(f'{args.output.stem}-{index}-plugin.png').resolve()
            item = dict(cloud=str(path), classes=str(classes), selected=selected, output=str(output))
            if direction:
                item['view'] = direction
            cases.append(item)
            view.close()
            view.deleteLater()
            app.processEvents()
        inputs = temporary / 'cases.json'
        inputs.write_text(json.dumps(cases))
        environment = os.environ.copy()
        # The reference runs Qt6; the compiled plugin probe owns its Qt5
        # platform plugin path through RPATH and must not inherit Qt6 plugins.
        environment.pop('QT_PLUGIN_PATH', None)
        environment.pop('QT_QPA_PLATFORM_PLUGIN_PATH', None)
        subprocess.run([str(args.probe.resolve()), '--render-probe', str(inputs)], check=True, env=environment)
    rows = []
    for index, (item, reference) in enumerate(zip(cases, expected)):
        actual = pixels(QtGui.QImage(item['output']))
        assert actual.shape == reference.shape, (actual.shape, reference.shape)
        delta = np.abs(actual.astype(np.int16) - reference.astype(np.int16))
        rows.append(dict(case=index, view=item.get('view', 'perspective'),
                         selected=item['selected'], differing_pixels=int(np.any(delta, axis=2).sum()),
                         mean_absolute_rgb_delta=float(delta[:, :, :3].mean()),
                         maximum_channel_delta=int(delta.max())))
    # Report differences without silently declaring raster parity from mesh
    # equality. Exact pixels are required by this controlled geometry protocol.
    report = dict(verdict='PASS' if all(r['differing_pixels'] == 0 for r in rows) else 'FAIL',
                  protocol='640x480 geometry GL surface, samples=4, point size=8; excludes text, controls and axes',
                  rows=rows, upstream_revision=subprocess.check_output(
                      ['git', '-C', str(args.upstream), 'rev-parse', 'HEAD'], text=True).strip(),
                  source_sha256={name: hashlib.sha256((folder/name).read_bytes()).hexdigest()
                                 for name in ('viewport.py', 'cuboid_viewport.py', 'cuboid.py')})
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))
    return report['verdict'] != 'PASS'


if __name__ == '__main__':
    raise SystemExit(main())
