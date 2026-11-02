#!/usr/bin/env python3
"""One independent-source PULC attribute probe, scheduled by validate_all.py.

No download, catalog or matrix ownership here. The common runner supplies
verified assets; generated truth (the onnxruntime reference and the source
identity JSON) stays next to the scenario report in the build directory.
The probe itself is test_attribute_probe.py; this wrapper only regenerates
the independent reference from the pinned source ONNX and bridges the
report metrics to the common runner's generic parser.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
# The public probe lives with the task's other test contracts.
PROBE = SCRIPTS.parent / "tests" / "attribute" / "test_attribute_probe.py"

# Recipe groups copied from core/AICore/src/tasks/attribute/attribute_capi.cpp
# (threshold -1 means argmax over the group; the C++ file owns the values).
PERSON_GROUPS = [
    ("Hat", ["Yes", "No"], 0.5),
    ("Glasses", ["Yes", "No"], 0.3),
    ("Sleeve", ["ShortSleeve", "LongSleeve"], -1.0),
    ("UpperStride", ["Yes", "No"], 0.5),
    ("UpperLogo", ["Yes", "No"], 0.5),
    ("UpperPlaid", ["Yes", "No"], 0.5),
    ("UpperSplice", ["Yes", "No"], 0.5),
    ("LowerStripe", ["Yes", "No"], 0.5),
    ("LowerPattern", ["Yes", "No"], 0.5),
    ("LongCoat", ["Yes", "No"], 0.5),
    ("Trousers", ["Yes", "No"], 0.5),
    ("Shorts", ["Yes", "No"], 0.5),
    ("Skirt&Dress", ["Yes", "No"], 0.5),
    ("Shoe", ["Boots", "No boots"], 0.5),
    ("HandBag&Dress", ["Yes", "No"], 0.5),
    ("ShoulderBag&Dress", ["Yes", "No"], 0.5),
    ("Backpack", ["Yes", "No"], 0.5),
    ("HoldObjectsInFront", ["Yes", "No"], 0.6),
    ("Age", ["AgeLess18", "Age18-60", "AgeOver60"], -1.0),
    ("Gender", ["Female", "Male"], 0.5),
    ("Direction", ["Front", "Side", "Back"], -1.0),
]

VEHICLE_GROUPS = [
    ("Color", ["yellow", "orange", "green", "gray", "red", "blue", "white",
               "golden", "brown", "black"], -1.0),
    ("Type", ["sedan", "suv", "van", "hatchback", "mpv", "pickup", "bus",
              "truck", "estate"], -1.0),
]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def decode(probabilities, groups):
    attributes = {}
    offset = 0
    for name, labels, threshold in groups:
        binary = threshold >= 0
        width = 1 if binary else len(labels)
        if binary:
            selected = 0 if probabilities[offset] > threshold else 1
        else:
            selected = max(range(width),
                           key=lambda i: probabilities[offset + i])
        probability = probabilities[offset + (0 if binary else selected)]
        if binary and selected == 1:
            probability = 1.0 - probability
        attributes[name] = labels[selected]
        offset += width
    if offset != len(probabilities):
        raise ValueError("recipe groups do not consume every probability")
    return attributes


def independent_reference(source: Path, image: Path, family: str,
                          work: Path):
    """Run the pinned source ONNX in onnxruntime and decode its output.

    Preprocessing mirrors the upstream X-AnyLabeling PULC contract and the
    AICore implementation: uint8 BGR->RGB, uint8 INTER_LINEAR resize to the
    model input, x/255 normalized with ImageNet mean/std, NCHW float32.
    """
    try:
        import cv2
        import numpy as np
        import onnxruntime as ort
    except ImportError as error:
        print(f"independent reference environment unavailable: {error}",
              file=sys.stderr)
        sys.exit(77)
    session = ort.InferenceSession(str(source),
                                   providers=["CPUExecutionProvider"])
    input_info = session.get_inputs()[0]
    _, _, height, width = input_info.shape
    pixels = cv2.imread(str(image), cv2.IMREAD_COLOR)
    if pixels is None:
        raise ValueError(f"cannot decode {image}")
    # Upstream contract: BGR -> RGB before the uint8 resize.
    pixels = pixels[:, :, ::-1]
    resized = cv2.resize(pixels, (width, height),
                         interpolation=cv2.INTER_LINEAR)
    normalized = resized.astype(np.float32) * np.float32(1.0 / 255.0)
    normalized = (normalized - np.array([0.485, 0.456, 0.406],
                                        dtype=np.float32)) / np.array(
                                            [0.229, 0.224, 0.225],
                                            dtype=np.float32)
    blob = normalized.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
    (raw,) = session.run(None, {input_info.name: blob})
    probabilities = np.asarray(raw, dtype=np.float32).squeeze()
    groups = PERSON_GROUPS if family == "person" else VEHICLE_GROUPS
    expected = sum(1 if t >= 0 else len(labels) for _, labels, t in groups)
    if probabilities.size != expected:
        raise ValueError(f"{family}: {probabilities.size} probabilities, "
                         f"expected {expected}")
    reference = work / f"pulc-{family}-probabilities.npy"
    np.save(reference, probabilities)
    identity = work / f"pulc-{family}-reference.json"
    identity.write_text(json.dumps({
        "source_sha256": digest(source),
        "source_revision": "r20231111",
        "family": family,
        "probabilities": probabilities.tolist(),
        "attributes": decode(probabilities.tolist(), groups),
    }, indent=1) + "\n")
    return reference, identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--image-sha256", required=True)
    parser.add_argument("--family", choices=("person", "vehicle"),
                        required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-diff", type=float, default=0.001,
                        help="f32 source probability tolerance, owned by the manifest")
    args = parser.parse_args()
    # The public probe imports cv2 in this same interpreter; a missing
    # environment capability is a skip (77), never a gate failure.
    try:
        import cv2  # noqa: F401
        import onnxruntime  # noqa: F401
    except ImportError as error:
        print(f"independent reference environment unavailable: {error}",
              file=sys.stderr)
        sys.exit(77)

    if digest(args.source) != args.source_sha256:
        raise SystemExit("source ONNX SHA-256 mismatch")
    if digest(args.image) != args.image_sha256:
        raise SystemExit("pinned input image SHA-256 mismatch")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
            dir=args.report.parent, prefix="pulc-reference-") as folder:
        reference, identity = independent_reference(
            args.source, args.image, args.family, Path(folder))
        command = [
            sys.executable, str(PROBE),
            "--library", str(args.library.resolve()),
            "--model", str(args.model),
            "--image", str(args.image),
            "--reference", str(reference),
            "--reference-sha256", digest(reference),
            "--source-identity", str(identity),
            "--source-identity-sha256", digest(identity),
            "--backend", args.backend,
            "--quant", "f32",
            "--warmups", str(args.warmups),
            "--repeats", str(args.repeats),
            "--threads", str(args.threads),
            "--json-out", str(args.report),
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        sys.stderr.write(completed.stderr)
    if not args.report.is_file():
        sys.stdout.write(completed.stdout)
        raise SystemExit(completed.returncode or 1)
    report = json.loads(args.report.read_text())
    # Flat numeric metrics for the common runner's generic parser; the raw
    # public-probe report is preserved verbatim as the evidence payload.
    wrapped = dict(report)
    wrapped["maximum_output_error"] = float(report["maximum_output_error"])
    wrapped["attributes_match"] = int(bool(report["attributes_match"]))
    wrapped["stable"] = int(bool(report["stable"]))
    wrapped["pass"] = int(bool(report["pass"]))
    wrapped["source_sha256"] = args.source_sha256
    wrapped["source_onnx_sha256"] = args.source_sha256
    wrapped["family"] = args.family
    args.report.write_text(json.dumps(wrapped, ensure_ascii=False, indent=2)
                           + "\n")
    # The tolerance belongs to the manifest; surface a mismatch in the exit
    # code even when the probe's own threshold happens to agree.
    if (report["maximum_output_error"] >= args.max_diff
            or not report["pass"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
