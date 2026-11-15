"""Build an offline storage-aware training graph from the original Paddle graph.

The source program remains immutable. Quantized convolution kernels and
already-F32 convolution biases/normalization affine values are trainable;
normalization running statistics stay fixed. Every step feeds deployed storage
values through a straight-through gradient from separate F32 masters.
"""


def head_argmax_counts(actual, expected):
    """Count exact class and per-edge DFL winner changes across all layout heads."""
    import numpy as np

    if len(actual) != 8 or len(expected) != 8:
        raise ValueError("all eight layout heads are required")
    totals = dict(class_decisions=0, class_flips=0, dfl_decisions=0,
                  dfl_flips=0)
    for prediction, reference in zip(actual, expected):
        prediction = np.asarray(prediction)
        reference = np.asarray(reference)
        if (prediction.shape != reference.shape or prediction.ndim != 3 or
                prediction.shape[0] != 1 or prediction.shape[-1] not in (10, 32) or
                not np.isfinite(prediction).all() or
                not np.isfinite(reference).all()):
            raise ValueError("invalid layout head output for argmax accounting")
        if prediction.shape[-1] == 10:
            predicted = np.argmax(prediction, axis=-1)
            expected_winners = np.argmax(reference, axis=-1)
            kind = "class"
        else:
            prediction = prediction.reshape(*prediction.shape[:-1], 4, 8)
            reference = reference.reshape(*reference.shape[:-1], 4, 8)
            predicted = np.argmax(prediction, axis=-1)
            expected_winners = np.argmax(reference, axis=-1)
            kind = "dfl"
        totals[f"{kind}_decisions"] += int(expected_winners.size)
        totals[f"{kind}_flips"] += int(np.count_nonzero(predicted != expected_winners))
    return totals


def teacher_head_feed(outputs, truth, targets, winner_targets):
    """Feed only the targets present in the selected training loss (O(elements))."""
    import numpy as np

    if len(outputs) != len(truth):
        raise ValueError("teacher/output head counts differ")
    output_names = set(outputs)
    if not output_names.issubset(targets) or not set(winner_targets).issubset(output_names):
        raise ValueError("teacher target mappings do not match the selected heads")
    feed = {}
    for name, value in zip(outputs, truth):
        feed[targets[name].name] = value
        if name in winner_targets:
            depth = 10 if value.shape[-1] == 10 else 8
            scores = value if depth == 10 else value.reshape(*value.shape[:-1], 4, 8)
            feed[winner_targets[name].name] = np.argmax(scores, axis=-1).astype(np.int64)
    return feed


def set_storage_learning_rates(masters, storage, q8_multiplier=1.0):
    """Scale only actual Q8_0 master updates, preserving affine/F16 updates.

    Q8 integer-code boundaries and F16/F32 parameter spacing differ. The
    caller pins this offline training choice before reading any held-out
    outputs. This does not change serialization or deployment arithmetic.
    """
    import math

    if not math.isfinite(q8_multiplier) or q8_multiplier <= 0:
        raise ValueError("Q8 learning-rate multiplier must be finite and positive")
    if not set(masters).issubset(storage):
        raise ValueError("master learning-rate storage coverage differs")
    if any(storage[name] not in {"F32", "F16", "Q8_0"} for name in masters):
        raise ValueError("unsupported master learning-rate storage")
    for name, master in masters.items():
        master.optimize_attr["learning_rate"] = (
            q8_multiplier if storage[name] == "Q8_0" else 1.0)


def storage_training_graph(program_bytes, named, storage):
    """Return a private program plus handles; never read acceptance outputs.

    Callers must select Paddle's legacy static mode before importing Paddle.
    Graph work is O(operators + parameters); full activations and gradients,
    rather than weight serialization, dominate training memory.
    """
    import paddle

    if set(storage) != set(named):
        raise ValueError("QAT storage/source parameter coverage differs")
    if any(value not in {"F32", "F16", "Q8_0"} for value in storage.values()):
        raise ValueError("unsupported QAT storage contract")
    program = paddle.static.Program.parse_from_string(program_bytes)
    # Legacy tensor arithmetic uses the program-local generator, whereas
    # layer helpers use the global unique_name guard below. Protect both.
    program._name_generator = paddle.base.unique_name.UniqueNameGenerator("qat_graph_")
    block = program.global_block()
    if len(program.blocks) != 1:
        raise ValueError("QAT currently requires a single-block inference graph")
    inputs = [op.output("Out")[0] for op in block.ops if op.type == "feed"]
    outputs = [op.input("X")[0] for op in block.ops if op.type == "fetch"]
    if len(inputs) != 1 or len(outputs) != 8:
        raise ValueError("QAT requires the original single-input eight-head layout graph")
    def op_inputs(op, name):
        return op.input(name) if name in op.input_names else []

    kernels, running_statistics = set(), set()
    for op in block.ops:
        if op.type in {"conv2d", "depthwise_conv2d"}:
            kernels.update(op.input("Filter"))
        if op.type == "batch_norm":
            if not op.attr("use_global_stats"):
                raise ValueError("QAT requires fixed original normalization statistics")
            # Backward needs training mode; use_global_stats preserves running
            # statistics while allowing gradients through affine parameters.
            op._set_attr("is_test", False)
            running_statistics.update(op_inputs(op, "Mean"))
            running_statistics.update(op_inputs(op, "Variance"))
    if not kernels or not kernels.issubset(named):
        raise ValueError("QAT convolution parameter coverage differs")
    affine = (named.keys() - kernels) - running_statistics
    trainable = kernels | affine
    if any(storage[name] != "F32" for name in named.keys() - kernels):
        raise ValueError("QAT requires F32 storage for biases and normalization parameters")
    # Exported inference programs may retain disconnected prediction branches.
    # They still receive stored forward values but must not enter an optimizer
    # expecting a loss gradient. One reverse pass finds the live ancestors.
    ancestors, live_trainables = set(outputs), set()
    for op in reversed(block.ops):
        if ancestors.intersection(op.output_arg_names):
            # Handle in-place definitions without tracing past an overwritten
            # value; current layout's nonpersistent outputs are distinct.
            ancestors.difference_update(op.output_arg_names)
            ancestors.update(op.input_arg_names)
            if op.type in {"conv2d", "depthwise_conv2d"}:
                live_trainables.update(op.input("Filter"))
            live_trainables.update(set(op.input_arg_names) & affine)
    live_trainables.intersection_update(trainable)
    for i in reversed(range(len(block.ops))):
        if block.ops[i].type in {"feed", "fetch"}:
            block._remove_op(i)
    for var in block.vars.values():
        var.stop_gradient = var.persistable or var.name == inputs[0]
    masters, quantized = {}, {}
    prefix = 0
    for i, name in enumerate(sorted(trainable)):
        source = block.var(name)
        if tuple(source.shape) != tuple(named[name].shape):
            raise ValueError("QAT source kernel shape differs")
        feed = block.create_var(name=f"qat_stored_{i}", shape=source.shape,
                                dtype=source.dtype, stop_gradient=True, is_data=True)
        quantized[name] = feed
        source.persistable = False
        source.is_parameter = False
        # A freshly exported graph can recreate this Python wrapper as a
        # Parameter; demote the wrapper as well as its serialized descriptor.
        source = block.create_var(name=name, shape=source.shape, dtype=source.dtype,
                                  persistable=False, stop_gradient=False)
        if name not in live_trainables:
            block._insert_op(prefix, type="assign", inputs={"X": feed},
                             outputs={"Out": source})
            prefix += 1
            continue
        master = block.create_parameter(name=f"qat_master_{i}", shape=source.shape,
                                        dtype=source.dtype, trainable=True)
        master.stop_gradient = False
        delta = block.create_var(name=f"qat_delta_{i}", shape=source.shape,
                                 dtype=source.dtype, stop_gradient=True)
        source.stop_gradient = False
        block._insert_op(prefix, type="elementwise_sub", inputs={"X": feed, "Y": master},
                         outputs={"Out": delta}, attrs={"axis": -1})
        block._insert_op(prefix + 1, type="elementwise_add", inputs={"X": master, "Y": delta},
                         outputs={"Out": source}, attrs={"axis": -1})
        prefix += 2
        masters[name] = master
    return program, inputs[0], outputs, masters, quantized


def head_distillation_loss(program, outputs, ranking_weight=0.0, ranking_margin=1e-6,
                           class_ranking_space="probability",
                           class_ranking_margin=None, dfl_ranking_margin=None):
    """Use all original heads and positions with separate class/DFL scales.

    ``ranking_weight`` optionally constrains the teacher's top-1 to remain
    ahead of its strongest competitor. ``class_ranking_margin`` and
    ``dfl_ranking_margin`` use the corresponding class score space and DFL
    logit units; omitted values retain the legacy shared ``ranking_margin``.
    The teacher's
    argmax comes from the current training-page tensor, but its original gap
    is not a target: the release gate requires the winner to remain unchanged,
    not its probability margin to be reconstructed. A linear hinge keeps a
    useful gradient for near-tied decisions where a squared hinge nearly
    vanishes. Ranking is averaged over decisions that violate the required
    margin, so a small number of changed winners is not diluted by the many
    already-correct cells in a dense feature map. Class heads are sigmoid
    probabilities, while DFL heads contain logits. Optional log-probability
    ranking preserves class winners and avoids multiplying the ranking
    gradient by a tiny sigmoid probability. Only the MSE term is
    energy-normalized. A zero default leaves ranking disabled.
    """
    import math

    import paddle
    from paddle.base.layer_helper import LayerHelper

    def reduce_sum(value, axis):
        # Paddle 3.3 routes paddle.sum through its eager-only tensor method in
        # legacy static mode; append the equivalent static op explicitly.
        helper = LayerHelper("reduce_sum", input=value)
        result = helper.create_variable_for_type_inference(dtype=value.dtype)
        helper.append_op(
            type="reduce_sum", inputs={"X": value}, outputs={"Out": result},
            attrs={"dim": [axis], "keep_dim": False, "reduce_all": False})
        return result

    def static_log(value):
        # Paddle 3.x routes paddle.log through an eager-only tensor method in
        # legacy static mode. Append the registered static log op directly.
        helper = LayerHelper("log", input=value)
        result = helper.create_variable_for_type_inference(dtype=value.dtype)
        helper.append_op(type="log", inputs={"X": value}, outputs={"Out": result})
        return result

    class_margin = (ranking_margin if class_ranking_margin is None
                    else class_ranking_margin)
    dfl_margin = ranking_margin if dfl_ranking_margin is None else dfl_ranking_margin
    if any(not math.isfinite(value) or value < 0 for value in
           (ranking_weight, ranking_margin, class_margin, dfl_margin)):
        raise ValueError("ranking controls must be finite and non-negative")
    if class_ranking_space not in {"probability", "log_probability"}:
        raise ValueError("unsupported class ranking space")
    targets, winner_targets, losses = {}, {}, []
    block = program.global_block()
    # Loading a ProgramDesc does not advance Paddle's name generator. Without
    # a prefix, loss arithmetic can overwrite original variables such as tmp_0.
    with paddle.utils.unique_name.guard("qat_loss_"):
        for i, name in enumerate(outputs):
            output = block.var(name)
            if output.shape[-1] not in {10, 32}:
                raise ValueError("unexpected original layout head shape")
            target = paddle.static.data(f"qat_teacher_{i}", list(output.shape), "float32")
            targets[name] = target
            # Relative head energy balances the output-distillation terms. It is
            # fixed from each training page, never tuned with held-out outputs,
            # thresholds or acceptance labels.
            scale = paddle.mean(target * target) + 1e-12
            scale.stop_gradient = True
            head_loss = paddle.mean((output - target) ** 2) / scale
            if ranking_weight:
                # This is O(number of head elements): one-hot selects the
                # teacher winner and a masked max finds the runner-up. DFL
                # encodes four independent 8-bin distributions, not 32 classes.
                depth = 10 if output.shape[-1] == 10 else 8
                scores = output if depth == 10 else paddle.reshape(output, [-1, 4, 8])
                if depth == 10 and class_ranking_space == "log_probability":
                    # log is monotone, so all original winners stay targets.
                    # The floor only prevents log(0); representable normal F32
                    # probabilities retain d(log(sigmoid(z)))/dz = 1-p.
                    scores = static_log(paddle.clip(scores, min=1.17549435e-38))
                winner_shape = list(output.shape[:-1])
                if depth == 8:
                    winner_shape.append(4)
                winner_target = paddle.static.data(
                    f"qat_teacher_winner_{i}", winner_shape, "int64")
                winner_targets[name] = winner_target
                winner = paddle.nn.functional.one_hot(
                    winner_target, num_classes=depth)
                winner = paddle.cast(winner, output.dtype)
                if depth == 8:
                    winner = paddle.reshape(winner, [-1, 4, 8])
                predicted_true = reduce_sum(scores * winner, axis=-1)
                # The dynamic bound suppresses only the teacher-winner entry.
                # Stopping its gradient keeps this equivalent to a masked max
                # without materializing a full tensor filled with -inf.
                bound = (paddle.max(scores, axis=-1, keepdim=True)
                         - paddle.min(scores, axis=-1, keepdim=True) + 1.0)
                bound.stop_gradient = True
                predicted_other = paddle.max(scores - winner * bound, axis=-1)
                required = class_margin if depth == 10 else dfl_margin
                violation = paddle.nn.functional.relu(
                    required - (predicted_true - predicted_other))
                flat_violation = paddle.reshape(violation, [-1])
                active = paddle.cast(flat_violation > 0, output.dtype)
                active.stop_gradient = True
                active_count = reduce_sum(active, axis=0)
                active_count.stop_gradient = True
                # A tiny floor avoids 0/0 when every decision is correct;
                # positive integer counts are unchanged at float32 precision.
                ranking_loss = (reduce_sum(flat_violation, axis=0) /
                                (active_count + 1e-12))
                head_loss = head_loss + ranking_weight * ranking_loss
            losses.append(head_loss)
        loss = paddle.add_n(losses) / len(losses)
    return loss, targets, winner_targets
