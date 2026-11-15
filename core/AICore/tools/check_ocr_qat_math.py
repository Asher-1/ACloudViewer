"""Audit untrained storage-graph output against the original CUDA teacher.

Only pinned CDLA train pages are read. This diagnostic does not train weights,
read held-out inputs, change an acceptance threshold or qualify a release.
The caller selects the TF32 policy before process startup; it is recorded here.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ocr_paddle_environment import source_environment

source_env = source_environment(Path(sys.executable))
os.environ.clear()
os.environ.update(source_env)
os.environ["FLAGS_enable_pir_api"] = "0"

import cv2
import numpy as np
import paddle
from paddle.inference import Config, create_predictor

from ocr_qat_graph import head_argmax_counts, storage_training_graph
from ocr_qat_storage import storage_forward
from paddle_ocr_extract import load_ocr_tensors
from run_paddle_layout_reference import preprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--pins", type=Path, required=True)
    parser.add_argument("--pages", type=int, default=4)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.pages < 1 or not paddle.is_compiled_with_cuda():
        raise ValueError("positive page count and CUDA-enabled Paddle required")
    pins = json.loads(args.pins.read_text())
    program, parameters = args.model / "model.pdmodel", args.model / "model.pdiparams"
    if sha(program) != pins["program_sha256"] or sha(parameters) != pins["parameters_sha256"]:
        raise ValueError("original source model identity differs")
    entries = pins["images"][:args.pages]
    if len(entries) != args.pages:
        raise ValueError("not enough pinned training pages")
    examples, tensors = [], {}
    config = Config(str(program), str(parameters))
    config.enable_use_gpu(256, 0)
    if not config.use_gpu():
        raise RuntimeError("original source CUDA teacher is disabled")
    config.switch_ir_optim(False)
    config.disable_mkldnn()
    config.disable_glog_info()
    config.set_cpu_math_library_num_threads(1)
    teacher = create_predictor(config)
    for entry in entries:
        path = Path(entry["path"])
        if not path.stem.startswith("train_") or sha(path) != entry["sha256"]:
            raise ValueError("input training split or identity differs")
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError("training page decode failed")
        x = preprocess(image)
        teacher.get_input_handle(teacher.get_input_names()[0]).copy_from_cpu(x)
        teacher.run()
        heads = [teacher.get_output_handle(name).copy_to_cpu()
                 for name in teacher.get_output_names()]
        examples.append((x, heads))
        tensors.update({f"teacher_{len(examples) - 1}_{i}": value for i, value in enumerate(heads)})
    del teacher, config
    named = dict(load_ocr_tensors(args.model)[0])
    paddle.enable_static()
    place = paddle.CUDAPlace(0)
    rows = []
    for kind in ("F32", "F16"):
        storage = {name: kind if value.ndim >= 2 else "F32"
                   for name, value in named.items()}
        graph, input_name, outputs, masters, stored = storage_training_graph(
            program.read_bytes(), named, storage)
        graph = graph.clone(for_test=True)
        scope = paddle.static.Scope()
        with paddle.static.scope_guard(scope):
            executor = paddle.static.Executor(place)
            for name, value in named.items():
                if name not in stored:
                    scope.var(name).get_tensor().set(value, place)
            for name, var in masters.items():
                scope.var(var.name).get_tensor().set(named[name], place)
            for page, (entry, (x, truth)) in enumerate(zip(entries, examples)):
                feed = {input_name: x}
                for name, var in stored.items():
                    feed[var.name] = storage_forward(named[name], storage[name])
                actual = executor.run(graph, feed=feed,
                                      fetch_list=[graph.global_block().var(name) for name in outputs])
                counts = head_argmax_counts(actual, truth)
                tensors.update({f"{kind.lower()}_{page}_{i}": value for i, value in enumerate(actual)})
                rows.append(dict(storage=kind, image=entry, **counts,
                                 heads=[dict(max_abs=float(np.abs(a.astype(np.float64) - b).max()),
                                             mae=float(np.abs(a.astype(np.float64) - b).mean()))
                                        for a, b in zip(actual, truth)]))
    archive = args.report.with_suffix(".outputs.npz")
    np.savez(archive, **tensors)
    report = dict(scope="untrained CUDA teacher/storage-graph diagnostic; no release acceptance",
                  program_sha256=sha(program), parameters_sha256=sha(parameters),
                  pins_sha256=sha(args.pins), paddle_version=paddle.__version__,
                  math_policy={"NVIDIA_TF32_OVERRIDE": os.environ.get("NVIDIA_TF32_OVERRIDE")},
                  outputs_sha256=sha(archive), rows=rows, release_qualified=False)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"math_policy": report["math_policy"],
                      "rows": [{key: value for key, value in row.items() if key not in ("heads", "image")}
                               for row in rows]}, indent=2))


if __name__ == "__main__":
    main()
