"""Offline layout QAT pilot with actual serialized F16/Q8_0 forward weights.

Training images and recipe are pinned before optimization. This tool never
loads held-out inputs, acceptance outputs or labels, and never qualifies a
model for release. Use a separate process for independent original-source
validation of the serialized final checkpoint.
"""

import argparse
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
from ocr_paddle_environment import source_environment

source_env = source_environment(Path(sys.executable))
# update() alone retains keys deliberately removed by source_environment(),
# including host logging options that can break Paddle's normal destruction.
os.environ.clear()
os.environ.update(source_env)
os.environ["FLAGS_enable_pir_api"] = "0"
# Match bench_paddle_ocr_graph.py before either cuDNN or cuBLAS initializes.
# Otherwise training may optimize against a different FP32 math policy than
# the independent original-source gate. This is offline calibration only.
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"

import cv2
import numpy as np
import paddle
from paddle.inference import Config, create_predictor

from ocr_qat_graph import (head_argmax_counts, head_distillation_loss,
                           set_storage_learning_rates, storage_training_graph,
                           teacher_head_feed)
from ocr_qat_storage import storage_forward
from paddle_ocr_extract import load_ocr_tensors


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def write_npz(path, values):
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez(temporary, **values)
    temporary.replace(path)


def train(args):
    # Keep the same authoritative layout preprocessing as the source probe.
    from run_paddle_layout_reference import preprocess

    if args.epochs <= 0 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("positive finite QAT recipe required")
    if (not math.isfinite(args.q8_learning_rate_multiplier)
            or args.q8_learning_rate_multiplier <= 0):
        raise ValueError("Q8 learning-rate multiplier must be finite and positive")
    if args.storage != "Q8_0" and args.q8_learning_rate_multiplier != 1:
        raise ValueError("Q8 learning-rate scaling requires Q8_0 storage")
    cuda_required = args.teacher_device == "cuda" or args.optimizer_device == "cuda"
    if cuda_required and args.teacher_gpu_id < 0:
        raise ValueError("CUDA GPU id must be non-negative")
    if cuda_required and not paddle.is_compiled_with_cuda():
        raise RuntimeError("CUDA execution requires a CUDA-enabled Paddle build; CPU fallback is forbidden")
    class_ranking_margin = (args.ranking_margin if args.class_ranking_margin is None
                            else args.class_ranking_margin)
    dfl_ranking_margin = (args.ranking_margin if args.dfl_ranking_margin is None
                          else args.dfl_ranking_margin)
    if any(not math.isfinite(value) or value < 0
           for value in (args.ranking_weight, args.ranking_margin,
                         class_ranking_margin, dfl_ranking_margin)):
        raise ValueError("ranking controls must be finite and non-negative")
    pins = json.loads(args.pins.read_text())
    program, parameters = args.model / "model.pdmodel", args.model / "model.pdiparams"
    if digest(program) != pins["program_sha256"] or digest(parameters) != pins["parameters_sha256"]:
        raise ValueError("original QAT model SHA-256 mismatch")
    images = pins["images"]
    if not images or len({entry["sha256"] for entry in images}) != len(images):
        raise ValueError("QAT needs distinct pinned training images")
    for entry in images:
        path = Path(entry["path"])
        if not path.stem.startswith("train_") or digest(path) != entry["sha256"]:
            raise ValueError("QAT training image identity/split mismatch")
    args.output.mkdir(parents=True, exist_ok=False)
    named = dict(load_ocr_tensors(args.model)[0])
    storage = {name: "F32" if value.ndim < 2 else "F16" for name, value in named.items()}
    if args.storage == "Q8_0":
        plan = json.loads(args.storage_plan.read_text())
        if {row["name"] for row in plan["parameters"]} != named.keys():
            raise ValueError("QAT storage-plan coverage differs")
        for row in plan["parameters"]:
            if row["storage"] == "Q8_0":
                storage[row["name"]] = "Q8_0"
        if "Q8_0" not in storage.values():
            raise ValueError("QAT storage plan contains no Q8_0 kernels")
    recipe = dict(
        scope=f"Fixed offline pilot; {len(images)} disjoint training pages, no held-out outputs; not release acceptance",
        program_sha256=digest(program), parameters_sha256=digest(parameters),
        pins_sha256=digest(args.pins), images=images, storage=storage,
        storage_plan_sha256=digest(args.storage_plan) if args.storage == "Q8_0" else None,
        epochs=args.epochs, learning_rate=args.learning_rate, seed=123,
        q8_learning_rate_multiplier=args.q8_learning_rate_multiplier,
        learning_rate_policy="Only Q8_0 masters receive the multiplier; F16 kernels "
                             "and F32 affine/bias masters retain the base rate",
        optimizer="Adam", loss="mean of eight energy-normalized original-head MSE terms "
                           "plus optional per-head raw-logit linear winner-ranking hinge",
        ranking_weight=args.ranking_weight, ranking_margin=args.ranking_margin,
        class_ranking_margin=class_ranking_margin,
        dfl_ranking_margin=dfl_ranking_margin,
        class_ranking_space=args.class_ranking_space,
        ranking_loss="linear hinge; teacher winner vs strongest competitor with a minimum "
                     "positive margin; mean over violating decisions; class units follow "
                     "class_ranking_space, DFL uses logits and four independent 8-bin groups",
        checkpoint_selection="fewest training-page class and DFL argmax flips, then lowest normalized MSE",
        normalization="fixed original F32 running statistics; trainable F32 affine parameters",
        f32_biases="trainable where connected to a live convolution; remain F32 in deployment",
        optimizer_device=args.optimizer_device.upper(),
        optimizer_gpu_id=args.teacher_gpu_id if args.optimizer_device == "cuda" else None,
        math_policy={"NVIDIA_TF32_OVERRIDE": os.environ["NVIDIA_TF32_OVERRIDE"]},
        teacher_device=args.teacher_device.upper(),
        teacher_gpu_id=args.teacher_gpu_id if args.teacher_device == "cuda" else None,
        scripts={name: digest(Path(__file__).with_name(name)) for name in
                 ("train_ocr_storage_qat.py", "ocr_qat_graph.py", "ocr_qat_storage.py")},
        started_at=time.time(), release_qualified=False)
    write_json(args.output / "recipe.json", recipe)

    config = Config(str(program), str(parameters))
    if args.teacher_device == "cuda":
        config.enable_use_gpu(500, args.teacher_gpu_id)
        if not config.use_gpu():
            raise RuntimeError("Paddle did not enable the requested CUDA teacher")
    else:
        config.disable_gpu()
    config.switch_ir_optim(False)
    config.disable_mkldnn()
    config.disable_glog_info()
    config.set_cpu_math_library_num_threads(1)
    teacher = create_predictor(config)
    examples = []
    for entry in images:
        decoded = cv2.imread(entry["path"])
        if decoded is None:
            raise ValueError("QAT training image decode failed")
        x = preprocess(decoded)
        teacher.get_input_handle(teacher.get_input_names()[0]).copy_from_cpu(x)
        teacher.run()
        heads = [teacher.get_output_handle(name).copy_to_cpu()
                 for name in teacher.get_output_names()]
        if not all(np.isfinite(head).all() for head in heads):
            raise ValueError("nonfinite original training output")
        examples.append((x, heads))
    del teacher, config

    paddle.enable_static()
    paddle.seed(123)
    graph, input_name, outputs, masters, stored = storage_training_graph(
        program.read_bytes(), named, storage)
    set_storage_learning_rates(masters, storage, args.q8_learning_rate_multiplier)
    startup = paddle.static.Program()
    with paddle.static.program_guard(graph, startup):
        loss, targets, winner_targets = head_distillation_loss(
            graph, outputs, ranking_weight=args.ranking_weight,
            ranking_margin=args.ranking_margin,
            class_ranking_margin=class_ranking_margin,
            dfl_ranking_margin=dfl_ranking_margin,
            class_ranking_space=args.class_ranking_space)
        eval_graph = graph.clone(for_test=True)
        eval_outputs = [eval_graph.global_block().var(name) for name in outputs]
        optimizer = paddle.optimizer.Adam(learning_rate=args.learning_rate,
                                         parameters=list(masters.values()))
        _, gradients = optimizer.minimize(loss)
    if {var.name for var, grad in gradients} != {var.name for var in masters.values()}:
        raise ValueError("QAT loss does not reach every trainable kernel")
    training_graph_sha256 = hashlib.sha256(graph.desc.serialize_to_string()).hexdigest()
    place = (paddle.CUDAPlace(args.teacher_gpu_id) if args.optimizer_device == "cuda"
             else paddle.CPUPlace())
    scope = paddle.static.Scope()
    history = []
    epoch_argmax = []
    best_masters = None
    best_score = None
    best_epoch = None
    best_epoch_counts = None
    with paddle.static.scope_guard(scope):
        executor = paddle.static.Executor(place)
        executor.run(startup)
        for name, value in named.items():
            if name not in stored:
                scope.var(name).get_tensor().set(value, place)
        for name, var in masters.items():
            scope.var(var.name).get_tensor().set(named[name], place)
        for epoch in range(args.epochs):
            for index, (x, truth) in enumerate(examples):
                feed = {input_name: x}
                for name, var in stored.items():
                    value = np.asarray(scope.find_var(masters[name].name).get_tensor()) if name in masters else named[name]
                    feed[var.name] = storage_forward(value, storage[name])
                feed.update(teacher_head_feed(outputs, truth, targets, winner_targets))
                started = time.monotonic()
                result = executor.run(graph, feed=feed,
                                      fetch_list=[loss])
                measured = float(np.asarray(result[0]).item())
                if not math.isfinite(measured):
                    raise ValueError("nonfinite QAT training loss")
                row = dict(epoch=epoch + 1, image=index + 1, loss=measured,
                           wall_seconds=time.monotonic() - started)
                history.append(row)
                write_json(args.output / "training-progress.json", dict(
                    steps=history, epoch_argmax=epoch_argmax,
                    current_epoch=epoch + 1,
                    release_qualified=False))
                print(json.dumps(row), flush=True)

            # Evaluate one fixed post-epoch checkpoint across every train page.
            # Online counts would mix outputs from continually changing weights.
            epoch_counts = dict(epoch=epoch + 1, class_decisions=0,
                                class_flips=0, dfl_decisions=0, dfl_flips=0)
            normalized_mse = 0.0
            mse_terms = 0
            for x, truth in examples:
                feed = {input_name: x}
                for name, var in stored.items():
                    value = np.asarray(scope.find_var(masters[name].name).get_tensor()) if name in masters else named[name]
                    feed[var.name] = storage_forward(value, storage[name])
                feed.update(teacher_head_feed(outputs, truth, targets, winner_targets))
                predicted = executor.run(eval_graph, feed=feed,
                                         fetch_list=eval_outputs)
                counts = head_argmax_counts(predicted, truth)
                for key, value in counts.items():
                    epoch_counts[key] += value
                for actual_head, truth_head in zip(predicted, truth):
                    error = np.asarray(actual_head, dtype=np.float64) - truth_head
                    reference_energy = float(np.mean(np.square(truth_head, dtype=np.float64)))
                    normalized_mse += float(np.mean(np.square(error))) / max(
                        reference_energy, 1e-12)
                    mse_terms += 1
            normalized_mse /= mse_terms
            epoch_counts["normalized_mse"] = normalized_mse
            epoch_argmax.append(epoch_counts)
            score = (epoch_counts["class_flips"] + epoch_counts["dfl_flips"],
                     normalized_mse, epoch + 1)
            if best_score is None or score < best_score:
                best_score = score
                best_epoch = epoch + 1
                best_epoch_counts = dict(epoch_counts)
                best_masters = {
                    name: np.asarray(scope.find_var(var.name).get_tensor()).copy()
                    for name, var in masters.items()
                }
                write_npz(args.output / "best_masters.npz", best_masters)
            write_json(args.output / "training-progress.json", dict(
                steps=history, epoch_argmax=epoch_argmax,
                current_epoch=epoch + 1, best_epoch=best_epoch,
                best_epoch_argmax=best_epoch_counts,
                release_qualified=False))
        final_masters = {name: np.asarray(scope.find_var(var.name).get_tensor()).copy()
                         for name, var in masters.items()}
        if not all(np.isfinite(value).all() for value in final_masters.values()):
            raise ValueError("nonfinite QAT checkpoint")
        if not all(np.array_equal(value, np.asarray(scope.find_var(name).get_tensor()))
                   for name, value in named.items() if name not in stored):
            raise ValueError("QAT changed fixed nontrainable state")
    if best_masters is None:
        raise ValueError("QAT did not produce a checkpoint")
    # Keep both endpoints for diagnosis, but export the best training-page
    # checkpoint rather than assuming the final epoch is the best one.
    deployed = {name: storage_forward(best_masters.get(name, value), storage[name])
                for name, value in named.items()}
    write_npz(args.output / "masters.npz", best_masters)
    write_npz(args.output / "final-masters.npz", final_masters)
    write_npz(args.output / "stored-weights.npz", deployed)
    result = dict(recipe_sha256=digest(args.output / "recipe.json"),
                  training_graph_sha256=training_graph_sha256, steps=len(history),
                  trainable_tensors=len(masters), stored_forward_tensors=len(stored),
                  tensor_storage_counts=dict(collections.Counter(storage.values())),
                  fixed_nontrainable_state_unchanged=True,
                  masters_sha256=digest(args.output / "masters.npz"),
                  stored_weights_sha256=digest(args.output / "stored-weights.npz"),
                  train_argmax_by_epoch=epoch_argmax, best_epoch=best_epoch,
                  best_epoch_argmax=best_epoch_counts,
                  finished_at=time.time(), release_qualified=False,
                  scope="Completed offline pilot training only; serialized original-source/GPU/page/performance gates pending")
    write_json(args.output / "result.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--pins", required=True, type=Path)
    parser.add_argument("--storage", choices=("F16", "Q8_0"), required=True)
    parser.add_argument("--storage-plan", type=Path)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--q8-learning-rate-multiplier", type=float, default=1.0,
                        help="offline Q8_0 master step multiplier; F16/F32 rates stay unchanged")
    parser.add_argument("--ranking-weight", type=float, default=0.0)
    parser.add_argument("--ranking-margin", type=float, default=1e-6)
    parser.add_argument("--class-ranking-margin", type=float, default=None,
                        help="optional class margin in probability or log-probability units")
    parser.add_argument("--dfl-ranking-margin", type=float, default=None,
                        help="optional DFL margin in raw-logit units")
    parser.add_argument("--class-ranking-space", choices=("probability", "log_probability"),
                        default="probability",
                        help="offline class ranking domain; DFL always uses original logits")
    parser.add_argument("--teacher-device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--optimizer-device", choices=("cuda", "cpu"), default="cpu",
                        help="training executor device, independent of the source teacher")
    parser.add_argument("--teacher-gpu-id", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.storage == "Q8_0" and args.storage_plan is None:
        parser.error("Q8_0 requires an explicit storage plan")
    train(args)


if __name__ == "__main__":
    main()
