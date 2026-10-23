"""Run in the isolated Paddle 3.0 legacy-static environment, on CPU only.

This exercises actual forward/backward operators on a small graph, not OCR
accuracy. The real model still needs training and independent GPU acceptance.
"""

import os

os.environ["FLAGS_enable_pir_api"] = "0"

import unittest

import numpy as np
import paddle

from ocr_qat_graph import (head_argmax_counts, head_distillation_loss,
                           set_storage_learning_rates, storage_training_graph,
                           teacher_head_feed)
from ocr_qat_storage import storage_forward


class TrainingGraphTests(unittest.TestCase):
    def test_adam_scales_only_q8_master_updates(self):
        paddle.enable_static()
        graph, startup = paddle.static.Program(), paddle.static.Program()
        storage = {"float": "F32", "half": "F16", "integer": "Q8_0"}
        with paddle.static.program_guard(graph, startup):
            masters = {name: graph.global_block().create_parameter(
                name="lr_" + name, shape=[1], dtype="float32", trainable=True)
                       for name in storage}
            set_storage_learning_rates(masters, storage, 10.0)
            loss = paddle.add_n([paddle.mean(value) for value in masters.values()])
            optimizer = paddle.optimizer.Adam(learning_rate=1e-3,
                                              parameters=list(masters.values()))
            optimizer.minimize(loss)
        scope = paddle.static.Scope()
        place = paddle.CPUPlace()
        with paddle.static.scope_guard(scope):
            executor = paddle.static.Executor(place)
            executor.run(startup)
            for value in masters.values():
                scope.var(value.name).get_tensor().set(np.ones(1, np.float32), place)
            executor.run(graph)
            updates = {name: 1.0 - float(np.asarray(
                scope.find_var(value.name).get_tensor()).item())
                       for name, value in masters.items()}
        self.assertAlmostEqual(updates["float"], 1e-3, places=6)
        self.assertEqual(updates["float"], updates["half"])
        self.assertAlmostEqual(updates["integer"], 1e-2, places=6)

    def run_loss(self, heads, truth, **options):
        paddle.enable_static()
        graph, startup = paddle.static.Program(), paddle.static.Program()
        with paddle.static.program_guard(graph, startup):
            outputs = [paddle.static.data(f"head_{i}", list(value.shape), "float32")
                       for i, value in enumerate(heads)]
            for output in outputs:
                output.stop_gradient = False
            loss, targets, winner_targets = head_distillation_loss(
                graph, [var.name for var in outputs], **options)
            gradients = paddle.static.gradients(loss, outputs)
        feed = {output.name: value for output, value in zip(outputs, heads)}
        feed.update(teacher_head_feed([output.name for output in outputs], truth,
                                     targets, winner_targets))
        scope = paddle.static.Scope()
        with paddle.static.scope_guard(scope):
            executor = paddle.static.Executor(paddle.CPUPlace())
            executor.run(startup)
            measured = executor.run(graph, feed=feed, fetch_list=[loss] + gradients)
        return float(np.asarray(measured[0]).item()), measured[1:]

    def test_zero_ranking_matches_original_mse(self):
        rng = np.random.default_rng(102)
        truth = [rng.normal(size=(1, 2, channels)).astype(np.float32)
                 for channels in [10] * 4 + [32] * 4]
        heads = [value + rng.normal(0, .1, value.shape).astype(np.float32) for value in truth]
        expected = sum(float(np.mean((head - target) ** 2) / (np.mean(target ** 2) + 1e-12))
                       for head, target in zip(heads, truth)) / 8
        measured, gradients = self.run_loss(heads, truth)
        self.assertAlmostEqual(measured, expected, places=7)
        self.assertTrue(all(np.isfinite(gradient).all() for gradient in gradients))

    def test_ranking_uses_each_dfl_distribution(self):
        truth = [np.arange(channels, dtype=np.float32).reshape(1, 1, channels)
                 for channels in [10] * 4 + [32] * 4]
        heads = [value.copy() for value in truth]
        # The global 32-bin winner is unchanged, but side two has flipped.
        heads[4][0, 0, 14:16] = [16, 14]
        base, base_gradients = self.run_loss(heads, truth)
        measured, gradients = self.run_loss(heads, truth, ranking_weight=3.0, ranking_margin=0.0)
        expected_rank = 3.0 * (16.0 - 14.0) / 8
        self.assertAlmostEqual(measured - base, expected_rank, places=7)
        rank_gradient = gradients[4] - base_gradients[4]
        self.assertGreater(rank_gradient[0, 0, 14], 0)
        self.assertLess(rank_gradient[0, 0, 15], 0)
        self.assertTrue(np.isfinite(rank_gradient).all())
        np.testing.assert_array_equal(rank_gradient[0, 0, 24:32], 0)

    def test_ranking_finds_competing_class_without_rebuilding_teacher_gap(self):
        truth = [np.arange(channels, dtype=np.float32).reshape(1, 1, channels)
                 for channels in [10] * 4 + [32] * 4]
        heads = [value.copy() for value in truth]
        heads[0][0, 0, 8:10] = [10, 8]
        base, _ = self.run_loss(heads, truth)
        measured, gradients = self.run_loss(heads, truth, ranking_weight=2.0, ranking_margin=.25)
        self.assertAlmostEqual(measured - base, 2.0 * 2.25 / 8, places=7)
        self.assertGreater(gradients[0][0, 0, 8], 0)
        self.assertLess(gradients[0][0, 0, 9], 0)
        self.assertTrue(all(np.isfinite(gradient).all() for gradient in gradients))

    def test_class_and_dfl_ranking_margins_use_independent_units(self):
        truth = [np.arange(channels, dtype=np.float32).reshape(1, 1, channels)
                 for channels in [10] * 4 + [32] * 4]
        heads = [value.copy() for value in truth]
        heads[0][0, 0, 8:10] = [10, 10]
        heads[4][0, 0, 14:16] = [16, 16]
        base, base_gradients = self.run_loss(heads, truth)
        measured, gradients = self.run_loss(
            heads, truth, ranking_weight=1.0, ranking_margin=0.0,
            class_ranking_margin=.2, dfl_ranking_margin=.8)
        self.assertAlmostEqual(measured - base, (.2 + .8) / 8, places=7)
        class_rank_gradient = gradients[0] - base_gradients[0]
        dfl_rank_gradient = gradients[4] - base_gradients[4]
        self.assertLess(class_rank_gradient[0, 0, 9], 0)
        self.assertGreater(class_rank_gradient[0, 0, 8], 0)
        self.assertLess(dfl_rank_gradient[0, 0, 15], 0)
        self.assertGreater(dfl_rank_gradient[0, 0, 14], 0)

    def test_linear_ranking_gradient_does_not_decay_with_violation_size(self):
        truth = [np.arange(channels, dtype=np.float32).reshape(1, 1, channels)
                 for channels in [10] * 4 + [32] * 4]
        gradients = []
        for competitor in (10.0, 100.0):
            heads = [value.copy() for value in truth]
            heads[0][0, 0, 8:10] = [competitor, 8.0]
            _, base_gradients = self.run_loss(heads, truth)
            _, measured_gradients = self.run_loss(
                heads, truth, ranking_weight=8.0, ranking_margin=.25)
            gradients.append(measured_gradients[0] - base_gradients[0])
        self.assertAlmostEqual(float(gradients[0][0, 0, 8]), 1.0, places=6)
        self.assertAlmostEqual(float(gradients[1][0, 0, 8]), 1.0, places=6)
        self.assertAlmostEqual(float(gradients[0][0, 0, 9]), -1.0, places=6)
        self.assertAlmostEqual(float(gradients[1][0, 0, 9]), -1.0, places=6)

    def test_ranking_is_not_diluted_by_correct_dense_feature_cells(self):
        truth = [np.tile(np.arange(channels, dtype=np.float32), (1, 128, 1))
                 for channels in [10] * 4 + [32] * 4]
        heads = [value.copy() for value in truth]
        heads[0][0, 0, 8:10] = [10, 8]
        base, base_gradients = self.run_loss(heads, truth)
        measured, gradients = self.run_loss(
            heads, truth, ranking_weight=8.0, ranking_margin=0.0)
        self.assertAlmostEqual(measured - base, 2.0, places=6)
        rank_gradient = gradients[0] - base_gradients[0]
        self.assertAlmostEqual(float(rank_gradient[0, 0, 8]), 1.0, places=6)
        self.assertAlmostEqual(float(rank_gradient[0, 0, 9]), -1.0, places=6)
        np.testing.assert_array_equal(rank_gradient[0][1:, :], 0)

    def test_correct_winner_needs_no_teacher_probability_margin(self):
        truth = [np.arange(channels, dtype=np.float32).reshape(1, 1, channels)
                 for channels in [10] * 4 + [32] * 4]
        heads = [value.copy() for value in truth]
        # The original winner (index 9) remains top-1 with a smaller positive
        # gap than the teacher. Only winner flips or a sub-floor gap are penalized.
        heads[0][0, 0, 8:10] = [8.5, 9.0]
        base, _ = self.run_loss(heads, truth)
        measured, gradients = self.run_loss(
            heads, truth, ranking_weight=5.0, ranking_margin=.25)
        self.assertAlmostEqual(measured, base, places=7)
        self.assertTrue(all(np.isfinite(gradient).all() for gradient in gradients))

    def test_log_probability_ranking_preserves_winner_and_low_probability_gradient(self):
        # Model the sigmoid explicitly: the gradient reaching its logits must
        # stay useful even when the class scores are all close to zero.
        for probability in (1e-2, 1e-6):
            truth = [np.tile(np.arange(1, channels + 1, dtype=np.float32), (1, 1, 1))
                     for channels in [10] * 4 + [32] * 4]
            heads = [value.copy() for value in truth]
            for i in range(4):
                truth[i] *= np.float32(probability / 10)
                heads[i] = truth[i].copy()
            heads[0][0, 0, 8:10] = [probability, probability / 2]
            _, base_gradients = self.run_loss(heads, truth)
            measured, gradients = self.run_loss(
                heads, truth, ranking_weight=8.0, ranking_margin=1e-3,
                class_ranking_space="log_probability")
            rank_gradient = gradients[0] - base_gradients[0]
            sigmoid_gradient = heads[0] * (1 - heads[0])
            upstream = rank_gradient * sigmoid_gradient
            self.assertGreater(float(upstream[0, 0, 8]), .98)
            self.assertLess(float(upstream[0, 0, 9]), -.98)
            self.assertTrue(np.isfinite(measured))
            self.assertTrue(all(np.isfinite(gradient).all() for gradient in gradients))

    def test_log_probability_ranking_handles_zero_and_keeps_dfl_logits(self):
        truth = [np.zeros((1, 1, channels), np.float32)
                 for channels in [10] * 4 + [32] * 4]
        for value in truth[:4]:
            value[0, 0, 9] = 1e-5
        heads = [value.copy() for value in truth]
        truth[4][0, 0, 15] = 1
        heads[4][0, 0, 14:16] = [2, -1]
        base, base_gradients = self.run_loss(heads, truth)
        measured, gradients = self.run_loss(
            heads, truth, ranking_weight=8.0, ranking_margin=0,
            class_ranking_space="log_probability")
        self.assertAlmostEqual(measured - base, 3.0, places=6)
        rank_gradient = gradients[4] - base_gradients[4]
        self.assertAlmostEqual(float(rank_gradient[0, 0, 14]), 1.0, places=6)
        self.assertAlmostEqual(float(rank_gradient[0, 0, 15]), -1.0, places=6)
        self.assertTrue(all(np.isfinite(gradient).all() for gradient in gradients))

    def test_invalid_ranking_controls(self):
        for option in ("ranking_weight", "ranking_margin", "class_ranking_margin",
                       "dfl_ranking_margin"):
            for value in (-1.0, float("nan"), float("inf")):
                with self.subTest(option=option, value=value), self.assertRaises(ValueError):
                    head_distillation_loss(None, [], **{option: value})
        with self.assertRaisesRegex(ValueError, "ranking space"):
            head_distillation_loss(None, [], class_ranking_space="unknown")

    def test_stored_forward_live_gradients_and_disconnected_branch(self):
        paddle.enable_static()
        original, startup = paddle.static.Program(), paddle.static.Program()
        with paddle.static.program_guard(original, startup):
            image = paddle.static.data("image", [1, 2, 4, 4], "float32")
            value = paddle.static.nn.conv2d(
                image, 42, 1, param_attr=paddle.ParamAttr(name="kernel"),
                bias_attr=paddle.ParamAttr(name="bias"))
            # Imported ProgramDesc does not restore its name-generator state.
            # A loss expression must not silently overwrite this live tensor.
            original_temp = original.global_block().create_var(name="tmp_0", shape=value.shape,
                                                                dtype="float32")
            value = paddle.assign(value, output=original_temp)
            paddle.static.nn.conv2d(image, 42, 1, bias_attr=False,
                                   param_attr=paddle.ParamAttr(name="dead_kernel"))
            classes, locations = paddle.split(value, [10, 32], axis=1)
            heads = [paddle.transpose(paddle.flatten(head, start_axis=2), [0, 2, 1])
                     for head in [classes] * 4 + [locations] * 4]
        block = original.global_block()
        feed = block.create_var(name="feed", type=paddle.base.core.VarDesc.VarType.FEED_MINIBATCH,
                                persistable=True)
        fetch = block.create_var(name="fetch", type=paddle.base.core.VarDesc.VarType.FETCH_LIST,
                                 persistable=True)
        block._insert_op(0, type="feed", inputs={"X": feed}, outputs={"Out": image}, attrs={"col": 0})
        for i, head in enumerate(heads):
            block.append_op(type="fetch", inputs={"X": head}, outputs={"Out": fetch}, attrs={"col": i})
        source = original.desc.serialize_to_string()
        rng = np.random.default_rng(91)
        named = {
            "kernel": rng.normal(size=(42, 2, 1, 1)).astype(np.float32),
            "bias": rng.normal(size=(42,)).astype(np.float32),
            "dead_kernel": rng.normal(size=(42, 2, 1, 1)).astype(np.float32),
        }
        storage = {"kernel": "F16", "bias": "F32", "dead_kernel": "F16"}
        graph, input_name, outputs, masters, stored = storage_training_graph(
            source, named, storage)
        self.assertEqual(set(masters), {"kernel", "bias"})
        self.assertEqual(set(stored), set(named))
        self.assertEqual(source, original.desc.serialize_to_string())
        optimizer_startup = paddle.static.Program()
        with paddle.static.program_guard(graph, optimizer_startup):
            loss, targets, _ = head_distillation_loss(graph, outputs)
            optimizer = paddle.optimizer.Adam(
                learning_rate=1e-3, parameters=list(masters.values()))
            _, grads = optimizer.minimize(loss)
        self.assertEqual(tuple(graph.global_block().var("tmp_0").shape), (1, 42, 4, 4))
        self.assertEqual(len(grads), 2)
        place = paddle.CPUPlace()
        scope = paddle.static.Scope()
        with paddle.static.scope_guard(scope):
            executor = paddle.static.Executor(place)
            executor.run(optimizer_startup)
            for name, master in masters.items():
                scope.var(master.name).get_tensor().set(named[name], place)
            x = rng.normal(size=(1, 2, 4, 4)).astype(np.float32)
            feed_values = {input_name: x}
            for name, var in stored.items():
                feed_values[var.name] = storage_forward(named[name], storage[name])
            expected = np.einsum("oc,nchw->nohw", storage_forward(named["kernel"], "F16")[:, :, 0, 0], x)
            expected += named["bias"][None, :, None, None]
            expected_heads = [expected[:, :10].reshape(1, 10, 16).transpose(0, 2, 1)] * 4
            expected_heads += [expected[:, 10:].reshape(1, 32, 16).transpose(0, 2, 1)] * 4
            for i, name in enumerate(outputs):
                feed_values[targets[name].name] = expected_heads[i] + np.float32(.01)
            actual = executor.run(graph, feed=feed_values, fetch_list=outputs + [loss, grads[0][1]])
        for head, expected_head in zip(actual[:8], expected_heads):
            np.testing.assert_allclose(head, expected_head, atol=1e-6, rtol=1e-6)
        counts = head_argmax_counts(actual[:8], expected_heads)
        self.assertEqual(counts["class_flips"], 0)
        self.assertEqual(counts["dfl_flips"], 0)
        self.assertTrue(np.isfinite(actual[8]).all())
        self.assertGreater(float(np.abs(actual[9]).sum()), 0)


if __name__ == "__main__":
    result = unittest.main(exit=False)
    if not result.result.wasSuccessful():
        raise SystemExit(1)
