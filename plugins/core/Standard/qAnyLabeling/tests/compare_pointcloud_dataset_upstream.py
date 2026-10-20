#!/usr/bin/env python3
"""Compare dataset archives and the import plan with actual upstream modules."""

import argparse
import base64
import dataclasses
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import copy
import xml.etree.ElementTree as ET
from zipfile import ZipFile, ZIP_DEFLATED


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.upstream / "anylabeling/views/labeling/pointcloud"
    package_name = "qany_upstream_pointcloud_dataset"
    package = types.ModuleType(package_name)
    package.__path__ = [str(source.resolve())]
    sys.modules[package_name] = package
    importer = importlib.import_module(f"{package_name}.import_dataset")
    io = importlib.import_module(f"{package_name}.io")
    model = importlib.import_module(f"{package_name}.model")
    rows = []
    with tempfile.TemporaryDirectory(prefix="qany-dataset-upstream-") as folder:
        folder = Path(folder)
        subprocess.run([str(args.probe.resolve()), "--fixture", str(folder)], check=True)
        actual = json.loads((folder / "cpp-plan.json").read_text())
        # Restore the exact before-images captured by the C++ preview.
        for change in actual["changes"]:
            path = Path(change["path"])
            if change["existed"]:
                path.write_bytes(base64.b64decode(change["before"]))
            else:
                path.unlink(missing_ok=True)
        target = importer.ImportTarget(folder / "frame_01.bin", folder / "frame_01.label",
                                       folder / "frame_01.cuboids.json")
        classes = (model.ClassDefinition(0, "Unlabeled", "#808080"),
                   model.ClassDefinition(1, "pedestrian", "#FF0000"))
        expected = importer.prepare_import(folder / "cancelled.zip", "datumaro", (target,),
                                           classes, folder / "classes.json")
        assert actual["classes"] == [dataclasses.asdict(item) for item in expected.classes]
        assert actual["previous_objects"] == expected.previous_objects
        assert actual["new_classes"] == expected.new_classes
        assert len(actual["changes"]) == len(expected.changes)
        for got, reference in zip(actual["changes"], expected.changes):
            assert got["path"] == str(reference.path)
            assert got["existed"] == (reference.before is not None)
            assert base64.b64decode(got["before"]) == reference.before
            assert got["after"] == json.loads(reference.after), reference.path
        importer.apply_import(expected)
        boxes = io.load_cuboids(target.cuboid_path, target.path)
        assert boxes == expected.targets[0].cuboids
        def compare_reader(path, format_name):
            definitions, frames = importer.read_dataset(path, format_name)
            cpp = json.loads(subprocess.check_output(
                [str(args.probe.resolve()), "--read", str(path), format_name], text=True))
            assert cpp["classes"] == [dataclasses.asdict(item) for item in definitions]
            assert len(cpp["frames"]) == len(frames)
            for actual_frame, expected_frame in zip(cpp["frames"], frames):
                assert actual_frame["name"] == expected_frame.name
                if expected_frame.name is None:
                    assert actual_frame["index"] == expected_frame.index
                expected_boxes = json.loads(json.dumps([dataclasses.asdict(box) for box in expected_frame.cuboids]))
                assert actual_frame["cuboids"] == expected_boxes
            rows.append({"case": path.stem, "format": format_name, "frames": len(frames),
                         "cuboids": sum(len(frame.cuboids) for frame in frames)})
            return definitions, frames

        for format_name in ("datumaro", "kitti_raw", "sly_pointcloud"):
            definitions, frames = compare_reader(folder / f"{format_name}.zip", format_name)
            assert len(definitions) == 1 and definitions[0].name == "car"
            assert len(frames) == 1 and frames[0].name == "frame_01"
            assert len(frames[0].cuboids) == 1
            box = frames[0].cuboids[0]
            assert box.id == 1 and box.class_id == 1
            assert list(box.center) == [1.0, 2.0, 3.0]
            assert list(box.size) == [4.0, 2.0, 1.5]
            assert list(box.rotation) == [0.1, 0.2, 0.3]
            assert box.occluded and box.locked == (format_name != "kitti_raw")

        with ZipFile(folder / "datumaro.zip") as archive:
            datumaro = json.loads(archive.read("annotations/default.json"))
        # Export annotation IDs are dataset-wide, while imported cuboid IDs
        # restart within each frame. Numeric item IDs are also valid upstream.
        second = copy.deepcopy(datumaro["items"][0])
        second["id"] = 25
        second["annotations"][0]["id"] = 99
        duplicate = copy.deepcopy(second["annotations"][0])
        duplicate["id"] = 100
        second["annotations"].append(duplicate)
        datumaro["items"].append(second)
        datumaro_path = folder / "datumaro-multiple.zip"
        with ZipFile(datumaro_path, "w", ZIP_DEFLATED) as archive:
            archive.writestr("annotations/default.json", json.dumps(datumaro))
        compare_reader(datumaro_path, "datumaro")
        invalid = copy.deepcopy(datumaro)
        invalid["items"][0]["annotations"][0]["type"] = "points"
        invalid_path = folder / "datumaro-unsupported.zip"
        with ZipFile(invalid_path, "w", ZIP_DEFLATED) as archive:
            archive.writestr("annotations/default.json", json.dumps(invalid))
        try:
            importer.read_dataset(invalid_path, "datumaro")
        except ValueError:
            pass
        else:
            raise AssertionError("upstream accepted unsupported annotations")
        rejected = subprocess.run([str(args.probe.resolve()), "--read", str(invalid_path), "datumaro"],
                                  capture_output=True, text=True)
        assert rejected.returncode != 0

        with ZipFile(folder / "kitti_raw.zip") as archive:
            tree = ET.fromstring(archive.read("tracklet_labels.xml"))
            metadata = archive.read("dataset_meta.json")
        track = tree.find("tracklets/item")
        track.find("first_frame").text = "2"
        poses = track.find("poses")
        poses.find("count").text = "3"
        pose = poses.find("item")
        pose.find("occlusion").text = "1"
        pose.find("occlusion_kf").text = "1"
        for _ in range(2):
            inherited = copy.deepcopy(pose)
            inherited.find("occlusion").text = "-1"
            inherited.find("occlusion_kf").text = "0"
            inherited.remove(inherited.find("truncation"))
            for axis in ("rx", "ry", "rz"):
                inherited.remove(inherited.find(axis))
            poses.append(inherited)
        kitti_path = folder / "kitti-inherited-occlusion.zip"
        with ZipFile(kitti_path, "w", ZIP_DEFLATED) as archive:
            archive.writestr("tracklet_labels.xml", ET.tostring(tree))
            archive.writestr("dataset_meta.json", metadata)
        _, frames = compare_reader(kitti_path, "kitti_raw")
        assert all(frame.cuboids[0].occluded for frame in frames)
    report = {
        "verdict": "PASS",
        "upstream_revision": subprocess.check_output(
            ["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                          for name in ("export.py", "importers.py", "import_dataset.py", "io.py")},
        "archive_rows": rows,
        "plan_changes": len(actual["changes"]),
        "new_classes": expected.new_classes,
        "previous_objects": expected.previous_objects,
        "checks": ["actual upstream archive readers", "frame matching", "reserved class ID merge",
                   "before images", "sidecar JSON", "grouped config", "upstream apply/reload",
                   "per-frame cuboid IDs", "numeric frame names", "unsupported annotations rejected",
                   "KITTI inherited occlusion", "KITTI optional pose fields"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
