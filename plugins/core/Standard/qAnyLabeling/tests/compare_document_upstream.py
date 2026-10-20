"""Compare path/pixel precedence with the actual upstream LabelFile loader."""

import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    upstream = args.upstream.resolve()
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    sys.path.insert(0, str(upstream))
    from PIL import Image
    from anylabeling.views.labeling.label_file import LabelFile

    evidence = []
    with tempfile.TemporaryDirectory(prefix="qany-document-") as temporary:
        root = Path(temporary)
        images, labels = root / "images", root / "labels"
        images.mkdir()
        labels.mkdir()
        image = images / "scene.png"
        Image.new("RGB", (37, 19), "red").save(image)
        Image.new("RGB", (23, 17), "green").save(labels / "scene.png")
        embedded = io.BytesIO()
        Image.new("RGB", (11, 7), "blue").save(embedded, format="PNG")
        cases = [
            ("relative", "../images/scene.png", None, False),
            ("absolute", str(image), None, False),
            ("basename-fallback", "/old/machine/scene.png", None, False),
            ("explicit-image-override", "/missing/old.png", None, True),
            ("embedded-precedence", "../images/scene.png",
             base64.b64encode(embedded.getvalue()).decode(), False),
            ("embedded-missing-file", "/missing/old.png",
             base64.b64encode(embedded.getvalue()).decode(), False),
            ("missing-file", "/missing/old.png", None, False),
            ("invalid-embedded", "../images/scene.png", "not an image", False),
        ]
        for name, stored, pixels, override in cases:
            filename = labels / "scene.json"
            filename.write_text(json.dumps({"version": "4.1.3", "shapes": [],
                "imagePath": stored, "imageData": pixels}), encoding="utf-8")
            try:
                loaded = LabelFile(str(filename), image_path=str(image) if override else None)
                expected = {"ok": True,
                    "path": os.path.normpath(loaded.resolved_image_path),
                    "size": list(Image.open(io.BytesIO(loaded.image_data)).size)}
            except Exception as error:
                expected = {"ok": False}
            command = [str(args.probe.resolve()), str(image if override else filename)]
            if override:
                command.append(str(labels))
            actual = json.loads(subprocess.check_output(command, text=True))
            assert actual["ok"] == expected["ok"], (name, expected, actual)
            if expected["ok"]:
                assert actual["path"] == expected["path"], (name, expected, actual)
                assert actual["size"] == expected["size"], (name, expected, actual)
            # Temporary paths are omitted from the retained evidence.
            evidence.append({"case": name, "ok": actual["ok"],
                "size": actual["size"] if actual["ok"] else None, "verdict": "PASS"})
    source = "anylabeling/views/labeling/label_file.py"
    report = {"upstream_revision": subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {source: hashlib.sha256((upstream / source).read_bytes()).hexdigest()},
        "cases": evidence, "verdict": "PASS"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Actual upstream LabelFile: {len(evidence)} cases PASS; evidence: {args.output}")


if __name__ == "__main__":
    main()
