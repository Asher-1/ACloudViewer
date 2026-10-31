"""Reference identity gates through real scheduler subprocesses; temporary copies preserve originals."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.fixture
def diffusion_capture(tmp_path):
    import importlib.util

    source = os.environ.get("AICORE_TEST_DIFFUSION_REFERENCE")
    captured = os.environ.get("AICORE_TEST_DIFFUSION_CAPTURE")
    if not source or not captured:
        pytest.skip("Requires explicit authenticated independent diffusion reference and real capture")
    script = Path(__file__).resolve().parents[2] / "src/tasks/instantmesh/tools/instantmesh_backend_ab.py"
    spec = importlib.util.spec_from_file_location("instantmesh_diffusion_probe", script)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    root, capture = tmp_path / "reference", tmp_path / "capture"
    root.mkdir()
    capture.mkdir()
    shutil.copy2(Path(source) / "provenance.json", root / "provenance.json")
    # Linked reference payloads are read-only; mutations below replace only the copied receipt.
    for path in Path(source).iterdir():
        if path.is_file() and path.name != "provenance.json":
            (root / path.name).symlink_to(path.resolve())
    for name in ("launch.json", "gui.log", "decode-input.bin", "decode-output.bin"):
        shutil.copy2(Path(captured) / name, capture / name)
    return probe, root, capture


def test_authenticated_real_diffusion_capture_is_diagnostic_only(diffusion_capture):
    probe, root, capture = diffusion_capture
    result = probe.compare_diffusion_capture(root, capture)
    assert result["reference_authenticated"] == 1 and result["reference_members"] == 538
    assert result["accuracy_status"] == "INCOMPLETE" and result["independent_reference_complete"] == 0
    assert all(result[f"view{view}_rgb_rmse_bytes"] >= 0 for view in range(6))
    assert result["final_latents_max_abs"] >= result["final_latents_rmse"] >= 0


@pytest.mark.parametrize("mutation", ["reference_identity", "reference_digest", "reference_inventory",
                                     "capture_identity", "capture_shape", "capture_digest",
                                     "duplicate_capture", "nan_capture"])
def test_real_diffusion_capture_authentication_fails_closed(diffusion_capture, mutation):
    probe, root, capture = diffusion_capture
    if mutation.startswith("reference_"):
        path = root / "provenance.json"
        data = json.loads(path.read_text())
        if mutation == "reference_identity":
            data["checkpoint_revision"] = "invalid"
        elif mutation == "reference_inventory":
            (root / "unauthenticated_alias.bin").symlink_to((root / "ref_latents.bin").resolve())
            for key in ("files_sha256", "files_bytes"):
                data[key]["unauthenticated_alias.bin"] = data[key].pop("ref_latents.bin")
        else:
            data["files_sha256"]["ref_latents.bin"] = "0" * 64
        path.write_text(json.dumps(data))
    elif mutation == "capture_identity":
        path = capture / "launch.json"
        data = json.loads(path.read_text())
        data["input_sha256"] = "0" * 64
        path.write_text(json.dumps(data))
    elif mutation == "capture_shape":
        path = capture / "gui.log"
        lines = path.read_text().splitlines()
        for index, line in enumerate(lines):
            if line.startswith("IM_CAPTURE "):
                data = json.loads(line[len("IM_CAPTURE "):])
                if data["member"] == "decode-input.bin":
                    data["shape"] = [1, 80, 120]
                    lines[index] = "IM_CAPTURE " + json.dumps(data)
        path.write_text("\n".join(lines) + "\n")
    elif mutation == "duplicate_capture":
        path = capture / "gui.log"
        text = path.read_text()
        row = next(line for line in text.splitlines() if line.startswith("IM_CAPTURE "))
        path.write_text(text + row + "\n")
    else:
        path = capture / "decode-input.bin"
        payload = bytearray(path.read_bytes())
        payload[:4] = b"\x00\x00\xc0\x7f" if mutation == "nan_capture" else b"\0" * 4
        path.write_bytes(payload)
        if mutation == "nan_capture":
            log = capture / "gui.log"
            lines = log.read_text().splitlines()
            for index, line in enumerate(lines):
                if line.startswith("IM_CAPTURE "):
                    data = json.loads(line[len("IM_CAPTURE "):])
                    if data["member"] == "decode-input.bin":
                        data["sha256"] = hashlib.sha256(payload).hexdigest()
                        lines[index] = "IM_CAPTURE " + json.dumps(data)
            log.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError):
        probe.compare_diffusion_capture(root, capture)


@pytest.fixture
def reference(tmp_path):
    binary = os.environ.get("AICORE_TEST_SCHEDULER_BINARY")
    source = os.environ.get("AICORE_TEST_SCHEDULER_REFERENCE")
    if not binary or not source:
        pytest.skip("Requires an explicit scheduler binary and freshly generated reference")
    root = tmp_path / "scheduler"
    shutil.copytree(source, root)
    return binary, root


def run(reference):
    binary, root = reference
    return subprocess.run([binary, str(root)], capture_output=True, text=True, timeout=30)


def test_authenticated_scheduler_passes(reference):
    result = run(reference)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["walk75_max_abs"] < 1e-4


def test_missing_provenance_is_incomplete(reference):
    (reference[1] / "provenance.json").unlink()
    result = run(reference)
    assert result.returncode == 77, result.stdout + result.stderr


@pytest.mark.parametrize("key,value", [
    ("upstream_revision", "0" * 40),
    ("generator_sha256", "0" * 64),
    ("seed", 43),
    ("steps", 74),
    ("latent_elements", 1),
])
def test_mismatched_reference_identity_fails(reference, key, value):
    path = reference[1] / "provenance.json"
    data = json.loads(path.read_text())
    data[key] = value
    path.write_text(json.dumps(data))
    result = run(reference)
    assert result.returncode == 1, result.stdout + result.stderr


def test_bad_provenance_json_fails(reference):
    (reference[1] / "provenance.json").write_text("{")
    result = run(reference)
    assert result.returncode == 1, result.stdout + result.stderr


def test_digest_mismatch_fails_even_with_unchanged_numeric_inputs(reference):
    path = reference[1] / "provenance.json"
    data = json.loads(path.read_text())
    data["files_sha256"]["x0.bin"] = "0" * 64
    path.write_text(json.dumps(data))
    result = run(reference)
    assert result.returncode == 1, result.stdout + result.stderr


@pytest.mark.parametrize("mutation,code", [("missing", 77), ("truncated", 1), ("nan", 1)])
def test_missing_or_corrupt_input_never_passes(reference, mutation, code):
    path = reference[1] / "x0.bin"
    if mutation == "missing":
        path.unlink()
    elif mutation == "truncated":
        path.write_bytes(b"\0")
    else:
        payload = bytearray(path.read_bytes())
        payload[:4] = b"\x00\x00\xc0\x7f"
        path.write_bytes(payload)
    result = run(reference)
    assert result.returncode == code, result.stdout + result.stderr


def prepare_component_reference(component, tmp_path):
    binary = os.environ.get("AICORE_TEST_COMPONENTS_BINARY", "")
    source = os.environ.get("AICORE_TEST_COMPONENT_REFERENCE", "")
    if not binary or not source:
        pytest.skip("Requires an explicit component binary and fresh real reference directory")
    root = tmp_path / component
    shutil.copytree(Path(source) / ("clip_vision" if component == "clip" else component), root)
    return binary, component, root


@pytest.fixture(params=["clip", "vae", "unet"])
def component_reference(request, tmp_path):
    return prepare_component_reference(request.param, tmp_path)


@pytest.fixture
def unet_reference(tmp_path):
    return prepare_component_reference("unet", tmp_path)


def run_component(reference):
    binary, component, root = reference
    # Stop after reference validation, before backend creation; negative contracts do no model/GPU work.
    return subprocess.run([binary, component, str(root / "absent.gguf"), str(root), "cpu"],
                          capture_output=True, text=True, timeout=30)


def test_component_valid_reference_reaches_model_check(component_reference):
    result = run_component(component_reference)
    assert result.returncode == 77 and "GGUF" in result.stderr, result.stdout + result.stderr


@pytest.mark.parametrize("key", ["generator_sha256", "checkpoint_revision", "checkpoint_sha256", "seed"])
def test_component_identity_is_authenticated_before_model_load(component_reference, key):
    path = component_reference[2] / "provenance.json"
    data = json.loads(path.read_text())
    data[key] = 123 if key == "seed" else "invalid"
    path.write_text(json.dumps(data))
    result = run_component(component_reference)
    assert result.returncode == 1 and key in result.stderr, result.stdout + result.stderr


@pytest.mark.parametrize("mutation,code", [("missing", 77), ("truncated", 1), ("nan", 1)])
def test_component_numeric_reference_rejected_before_model_load(component_reference, mutation, code):
    root = component_reference[2]
    file = next(root.glob("*.bin"))
    if mutation == "missing":
        file.unlink()
    elif mutation == "truncated":
        file.write_bytes(b"\0")
    else:
        payload = bytearray(file.read_bytes())
        payload[:4] = b"\x00\x00\xc0\x7f"
        file.write_bytes(payload)
        path = root / "provenance.json"
        data = json.loads(path.read_text())
        data["files_sha256"][file.name] = hashlib.sha256(payload).hexdigest()
        path.write_text(json.dumps(data))
    result = run_component(component_reference)
    assert result.returncode == code and "GGUF" not in result.stderr, result.stdout + result.stderr


@pytest.mark.parametrize("key", ["override_revision", "override_sha256", "pipeline_sha256",
                                  "checkpoint_config_sha256", "batch", "height", "width",
                                  "reference_height", "reference_width", "context_length"])
def test_unet_specific_identity_is_authenticated(unet_reference, key):
    path = unet_reference[2] / "provenance.json"
    data = json.loads(path.read_text())
    data[key] = "invalid"
    path.write_text(json.dumps(data))
    result = run_component(unet_reference)
    assert result.returncode == 1 and key in result.stderr, result.stdout + result.stderr


@pytest.fixture
def geometry_reference(tmp_path):
    binary = os.environ.get("AICORE_TEST_COMPONENTS_BINARY", "")
    source = os.environ.get("AICORE_TEST_GEOMETRY_REFERENCE", "")
    if not binary or not source:
        pytest.skip("Requires an explicit component binary and independent four-image reference")
    root = tmp_path / "geometry"
    shutil.copytree(source, root)
    manifest = tmp_path / "manifest.json"
    shutil.copyfile(Path(__file__).resolve().parents[2] / "scripts/validation_manifest.json", manifest)
    return binary, root, manifest


def run_geometry(reference, component="dino"):
    binary, root, manifest = reference
    model = "dino_f32.gguf" if component == "dino" else "lrm_transformer_f32.gguf"
    return subprocess.run([binary, component, str(root / model),
                           str(root), "cpu", "4", str(manifest), str(root / "baseline")],
                          capture_output=True, text=True, timeout=30)


@pytest.mark.parametrize("component", ["dino", "lrm"])
def test_geometry_reference_identity_before_model_load(geometry_reference, component):
    path = geometry_reference[1] / "provenance.json"
    data = json.loads(path.read_text())
    data["checkpoint_sha256"] = "0" * 64
    path.write_text(json.dumps(data))
    result = run_geometry(geometry_reference, component)
    assert result.returncode == 1 and "checkpoint_sha256" in result.stderr, result.stdout + result.stderr


@pytest.mark.parametrize("component", ["dino", "lrm"])
def test_geometry_nonfinite_reference_cannot_reach_model_load(geometry_reference, component):
    root = geometry_reference[1]
    path = root / "blue_cat" / ("image.bin" if component == "dino" else "dino_features.bin")
    payload = bytearray(path.read_bytes())
    payload[:4] = b"\x00\x00\xc0\x7f"
    path.write_bytes(payload)
    receipt = root / "provenance.json"
    data = json.loads(receipt.read_text())
    data["scenarios"]["blue_cat"]["files_sha256"][path.name] = hashlib.sha256(payload).hexdigest()
    receipt.write_text(json.dumps(data))
    result = run_geometry(geometry_reference, component)
    assert result.returncode == 1 and "floating-point" in result.stderr, result.stdout + result.stderr


def test_geometry_preflight_missing_baseline_is_incomplete(geometry_reference):
    env = dict(os.environ)
    env.pop("AICORE_TEST_GEOMETRY_BASELINE", None)
    result = subprocess.run([sys.executable, __file__, "--geometry"],
                            env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 77 and "INCOMPLETE" in result.stdout


@pytest.fixture
def frozen_geometry_reference(geometry_reference):
    binary, root, manifest = geometry_reference
    source = os.environ.get("AICORE_TEST_GEOMETRY_BASELINE", "")
    if not source:
        pytest.skip("Requires authenticated original-upstream component references")
    # Copy only CPU/f32, the sole combination used here; preserve shared references and real models.
    target = root / "baseline"
    target.mkdir()
    for file in Path(source).glob("*-cpu-f32-*.bin"):
        shutil.copyfile(file, target / file.name)
    return binary, root, manifest


@pytest.mark.parametrize("component", ["dino", "lrm"])
def test_geometry_authenticated_reference_reaches_missing_model(frozen_geometry_reference, component):
    result = run_geometry(frozen_geometry_reference, component)
    assert result.returncode == 77 and "GGUF" in result.stderr, result.stdout + result.stderr


@pytest.mark.parametrize("mutation", ["missing_bound", "negative_bound", "wrong_bound",
                                      "missing_case", "cpu_fallback", "missing_baseline", "corrupt_baseline"])
def test_geometry_frozen_gate_failures_before_model_load(frozen_geometry_reference, mutation):
    binary, root, manifest = frozen_geometry_reference
    data = json.loads(manifest.read_text())
    case = data["instantmesh_geometry_component_accuracy"]["cases"]["blue_cat/cpu/f32"]
    code = 1
    if mutation == "missing_bound":
        del case["absolute_bounds"]["dino_view5_max_abs"]
    elif mutation == "negative_bound":
        case["added_error_bounds"]["dino_view3_rmse"] = -1
    elif mutation == "wrong_bound":
        case["absolute_bounds"]["dino_view1_max_abs"] *= 2
    elif mutation == "missing_case":
        del data["instantmesh_geometry_component_accuracy"]["cases"]["robot/cpu/f32"]
        code = 77
    elif mutation == "cpu_fallback":
        case["baseline_files"]["dino"]["actual_backend"] = "Vulkan0"
    else:
        path = root / "baseline" / case["baseline_files"]["dino"]["file"]
        if mutation == "missing_baseline":
            path.unlink()
            code = 77
        else:
            path.write_bytes(b"bad")
    manifest.write_text(json.dumps(data))
    result = run_geometry(frozen_geometry_reference)
    assert result.returncode == code and "GGUF" not in result.stderr, result.stdout + result.stderr


@pytest.fixture
def synth_reference(tmp_path):
    binary = os.environ.get("AICORE_TEST_COMPONENTS_BINARY", "")
    keys = ("AICORE_TEST_SYNTH_REFERENCE", "AICORE_TEST_SYNTH_BASELINE",
            "AICORE_TEST_WEIGHT_REFERENCE", "AICORE_TEST_GEOMETRY_REFERENCE")
    sources = [os.environ.get(key, "") for key in keys]
    if not binary or not all(sources):
        pytest.skip("Requires explicit Synthesizer/quantization/geometry/baseline references")
    roots = []
    for index, source in enumerate(sources):
        root = tmp_path / str(index)
        shutil.copytree(source, root)
        roots.append(root)
    manifest = tmp_path / "manifest.json"
    shutil.copyfile(Path(__file__).resolve().parents[2] / "scripts/validation_manifest.json", manifest)
    return binary, roots, manifest


def run_synth(reference):
    binary, roots, manifest = reference
    return subprocess.run([binary, "synth", str(roots[0] / "synthesizer_f32.gguf"),
                           str(roots[0]), "cpu", "4", str(manifest), str(roots[1]),
                           str(roots[3]), str(roots[2])], capture_output=True, text=True, timeout=30)


def test_synth_valid_reference_reaches_missing_model(synth_reference):
    result = run_synth(synth_reference)
    assert result.returncode == 77 and "GGUF" in result.stderr, result.stdout + result.stderr


@pytest.mark.parametrize("root_index", [0, 1, 2, 3])
def test_synth_corrupt_receipt_is_rejected_before_model(synth_reference, root_index):
    root = synth_reference[1][root_index]
    path = root / ("capture.json" if root_index == 1 else "provenance.json")
    path.write_text("{}")
    result = run_synth(synth_reference)
    assert result.returncode == 1 and "identity" in result.stderr, result.stdout + result.stderr


# Ignoring frozen scene identity would incorrectly accept tampered layout/scale/backend declarations.
@pytest.mark.parametrize("field,value", [
    ("kind", "full-pipeline-accuracy"),
    ("upstream_revision", "0" * 40),
    ("grid_res", 64),
    ("plane_shape", [1, 3, 80, 32, 128]),
    ("scale", 1.0),
    ("backends", ["cuda"]),
])
def test_synth_frozen_scene_identity_before_model_load(synth_reference, field, value):
    path = synth_reference[2]
    data = json.loads(path.read_text())
    data["instantmesh_synthesizer_component_accuracy"][field] = value
    path.write_text(json.dumps(data))
    result = run_synth(synth_reference)
    assert result.returncode == 1 and "identity" in result.stderr, result.stdout + result.stderr


@pytest.mark.parametrize("field", ["absolute_multiplier", "added_multiplier", "epsilon",
                                   "weight_quantization_multiplier", "weight_epsilon", "weight_migration_limit"])
def test_synth_frozen_bounds_cannot_be_relaxed(synth_reference, field):
    path = synth_reference[2]
    data = json.loads(path.read_text())
    data["instantmesh_synthesizer_component_accuracy"][field] *= 2
    path.write_text(json.dumps(data))
    result = run_synth(synth_reference)
    assert result.returncode == 1 and "identity" in result.stderr, result.stdout + result.stderr


@pytest.mark.parametrize("field", ["preserve_original_baseline", "preserve_raw_added_metrics",
                                   "other_outputs_and_dino_lrm_gates_unchanged"])
def test_synth_exception_must_preserve_history_and_other_gates(synth_reference, field):
    path = synth_reference[2]
    data = json.loads(path.read_text())
    data["instantmesh_synthesizer_component_accuracy"]["compatibility_exception"][field] = False
    path.write_text(json.dumps(data))
    result = run_synth(synth_reference)
    assert result.returncode == 1 and "identity" in result.stderr, result.stdout + result.stderr


@pytest.mark.parametrize("mutation,code", [("missing", 77), ("truncated", 1), ("nan", 1)])
def test_synth_bad_output_reference_never_passes(synth_reference, mutation, code):
    path = synth_reference[1][0] / "blue_cat/weight.bin"
    if mutation == "missing":
        path.unlink()
    elif mutation == "truncated":
        path.write_bytes(b"bad")
    else:
        payload = bytearray(path.read_bytes())
        payload[:4] = b"\x00\x00\xc0\x7f"
        path.write_bytes(payload)
    result = run_synth(synth_reference)
    assert result.returncode == code and "GGUF" not in result.stderr, result.stdout + result.stderr


@pytest.fixture
def preprocess_binary():
    binary = os.environ.get("AICORE_TEST_PREPROCESS_BINARY", "")
    if not Path(binary).is_file():
        pytest.skip("Requires the real preprocessing probe")
    return binary


def test_preprocess_missing_reference_is_incomplete(preprocess_binary, tmp_path):
    result = subprocess.run([preprocess_binary, str(tmp_path / "absent"), str(tmp_path / "image.png")],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 77, result.stdout + result.stderr


def test_preprocess_rejects_wrong_reference_identity(preprocess_binary, tmp_path):
    (tmp_path / "provenance.json").write_text(json.dumps({"upstream_revision": "invalid"}))
    result = subprocess.run([preprocess_binary, str(tmp_path), str(tmp_path / "image.png")],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 1 and "identity" in result.stderr, result.stdout + result.stderr


@pytest.mark.parametrize("filter_name", ["bilinear", "bicubic"])
@pytest.mark.parametrize("shape", [(17, 13, 5, 3), (4, 3, 11, 7), (8, 5, 8, 5),
                                   (8, 5, 8, 2), (1, 17, 6, 3)])
def test_preprocess_aa_matches_framework(preprocess_binary, tmp_path, filter_name, shape):
    if os.environ.get("AICORE_TEST_PREPROCESS_FRAMEWORK") != "1":
        pytest.skip("Requires the explicitly selected independent framework environment")
    import numpy as np
    import torch
    from torchvision.transforms.v2 import functional as transform
    from torchvision.transforms import InterpolationMode
    width, height, out_width, out_height = shape
    pixels = np.random.default_rng(42).integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    input_path, output_path = tmp_path / "input.rgb", tmp_path / "output.rgb"
    pixels.tofile(input_path)
    result = subprocess.run([preprocess_binary, "resize", str(input_path), str(width), str(height),
                             str(out_width), str(out_height), filter_name, str(output_path)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    reference = transform.resize(torch.from_numpy(pixels.transpose(2, 0, 1).copy()),
                                 [out_height, out_width], antialias=True,
                                 interpolation=getattr(InterpolationMode, filter_name.upper()))
    expected = reference.permute(1, 2, 0).contiguous().numpy().tobytes()
    assert output_path.read_bytes() == expected


@pytest.mark.parametrize("filter_name", ["bilinear", "bicubic"])
@pytest.mark.parametrize("shape", [(17, 13, 5), (13, 17, 7), (6, 3, 3),
                                   (3, 6, 3), (1023, 769, 224), (769, 1023, 512)])
def test_preprocess_aa_condition_matches_processor(preprocess_binary, tmp_path, filter_name, shape,
                                                  record_testsuite_property):
    if os.environ.get("AICORE_TEST_PREPROCESS_FRAMEWORK") != "1":
        pytest.skip("Requires the explicitly selected independent framework environment")
    import numpy as np
    import torch
    import transformers
    from transformers import CLIPImageProcessor
    record_testsuite_property("torch_version", torch.__version__)
    record_testsuite_property("transformers_version", transformers.__version__)
    width, height, target = shape
    pixels = np.random.default_rng(42).integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    input_path, output_path = tmp_path / "input.rgb", tmp_path / "output.f32"
    pixels.tofile(input_path)
    result = subprocess.run([preprocess_binary, "normalize", str(input_path), str(width), str(height),
                             str(target), filter_name, str(output_path)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    processor = CLIPImageProcessor(size={"shortest_edge": target},
        crop_size={"height": target, "width": target}, resample=2 if filter_name == "bilinear" else 3,
        image_mean=[0.5] * 3 if filter_name == "bilinear" else [0.48145466, 0.4578275, 0.40821073],
        image_std=[0.8] * 3 if filter_name == "bilinear" else [0.26862954, 0.26130258, 0.27577711])
    reference = processor(images=pixels, input_data_format="channels_last", return_tensors="pt").pixel_values.numpy()
    actual = np.fromfile(output_path, dtype="<f4").reshape(1, 3, target, target)
    assert np.max(np.abs(actual - reference)) < 1e-6


@pytest.mark.parametrize("padding", [0, 20])
def test_preprocess_aa_rgba_matches_pil_paste(preprocess_binary, tmp_path, padding):
    if os.environ.get("AICORE_TEST_PREPROCESS_FRAMEWORK") != "1":
        pytest.skip("Requires the explicitly selected independent framework environment")
    import numpy as np
    from PIL import Image
    width, height = 256, 7
    rgba = np.random.default_rng(42).integers(0, 256, size=(height, width, 4), dtype=np.uint8)
    rgba[:, :, 3] = np.arange(256, dtype=np.uint8)
    rows = np.full((height, width * 4 + padding), 19, dtype=np.uint8)
    rows[:, :width * 4] = rgba.reshape(height, width * 4)
    input_path, output_path = tmp_path / "input.rgba", tmp_path / "output.rgb"
    rows.tofile(input_path)
    result = subprocess.run([preprocess_binary, "composite", str(input_path), str(width), str(height),
                             str(rows.shape[1]), str(output_path)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    source = Image.fromarray(rgba)
    reference = Image.new("RGB", source.size, (255, 255, 255))
    reference.paste(source, mask=source.getchannel("A"))
    assert output_path.read_bytes() == reference.tobytes()


@pytest.mark.parametrize("seed", [0, 42, 2**32 + 42])
@pytest.mark.parametrize("sizes", [[16], [38400, 32768, 38400], [38400] + [32768, 38400] * 75])
def test_rng_normal_draws_match_torch_cpu(preprocess_binary, tmp_path, seed, sizes):
    if os.environ.get("AICORE_TEST_PREPROCESS_FRAMEWORK") != "1":
        pytest.skip("Requires the explicitly selected independent framework environment")
    import numpy as np
    import torch
    output = tmp_path / "noise.f32"
    result = subprocess.run([preprocess_binary, "noise", str(seed), ",".join(map(str, sizes)), str(output)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    generator = torch.Generator(device="cpu").manual_seed(seed)
    expected = np.concatenate([torch.randn(size, generator=generator, dtype=torch.float32).numpy() for size in sizes])
    actual = np.fromfile(output, dtype="<f4")
    assert actual.shape == expected.shape
    assert np.max(np.abs(actual - expected)) < 1e-6


def test_rng_authenticated_75_step_reference(preprocess_binary):
    source = os.environ.get("AICORE_TEST_E2E_REFERENCE", "")
    if not source:
        pytest.skip("Requires the explicit authenticated independent 75-step reference")
    result = subprocess.run([preprocess_binary, "noise-reference", source],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    metrics = json.loads(result.stdout)
    assert metrics["reference_authenticated"] == 1
    assert metrics["independent_reference_complete"] == 0
    for prefix in ("initial", "condition", "ancestral"):
        assert metrics[prefix + "_max_abs"] < 1e-6
        assert metrics[prefix + "_rmse"] < 1e-6


if __name__ == "__main__":
    if "--preprocess" in sys.argv:
        if not Path(os.environ.get("AICORE_TEST_PREPROCESS_BINARY", "")).is_file():
            print("INCOMPLETE: missing real preprocessing binary")
            sys.exit(77)
        sys.exit(pytest.main([__file__, "-q", "-k", "preprocess and not preprocess_aa"]))
    if "--synth" in sys.argv:
        keys = ("AICORE_TEST_SYNTH_REFERENCE", "AICORE_TEST_SYNTH_BASELINE",
                "AICORE_TEST_WEIGHT_REFERENCE", "AICORE_TEST_GEOMETRY_REFERENCE")
        if not Path(os.environ.get("AICORE_TEST_COMPONENTS_BINARY", "")).is_file() or not all(
                (Path(os.environ.get(key, "")) / ("capture.json" if "BASELINE" in key else "provenance.json")).is_file()
                for key in keys):
            print("INCOMPLETE: missing Synthesizer component binary or reference receipts")
            sys.exit(77)
        sys.exit(pytest.main([__file__, "-q", "-k", "synth"]))
    if "--geometry" in sys.argv:
        root = Path(os.environ.get("AICORE_TEST_GEOMETRY_REFERENCE", ""))
        binary = Path(os.environ.get("AICORE_TEST_COMPONENTS_BINARY", ""))
        baseline = Path(os.environ.get("AICORE_TEST_GEOMETRY_BASELINE", ""))
        if not binary.is_file() or not (root / "provenance.json").is_file() or not (baseline / "capture.json").is_file():
            print("INCOMPLETE: missing four-image component binary, independent reference or upstream capture receipt")
            sys.exit(77)
        sys.exit(pytest.main([__file__, "-q", "-k", "geometry"]))
    if "--components" in sys.argv:
        root = Path(os.environ.get("AICORE_TEST_COMPONENT_REFERENCE", ""))
        binary = Path(os.environ.get("AICORE_TEST_COMPONENTS_BINARY", ""))
        if not binary.is_file() or not all((root / name / "provenance.json").is_file()
                for name in ("clip_vision", "vae", "unet")):
            print("INCOMPLETE: missing component binary or authenticated references")
            sys.exit(77)
        sys.exit(pytest.main([__file__, "-q", "-k", "component or unet_specific"]))
    source = os.environ.get("AICORE_TEST_SCHEDULER_REFERENCE", "")
    binary = os.environ.get("AICORE_TEST_SCHEDULER_BINARY", "")
    if not source or not binary or not (Path(source) / "provenance.json").is_file() or not Path(binary).is_file():
        print("INCOMPLETE: missing freshly generated scheduler reference or binary")
        sys.exit(77)
    sys.exit(pytest.main([__file__, "-q", "-k", "not component and not unet_specific and not geometry and not synth and not preprocess and not rng"]))
