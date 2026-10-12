#!/usr/bin/env python3
"""Generate the `ocr` task probe fixtures from a pinned F32 GGUF.

The manifest's historical ocr_fixtures/*.npy release assets are gone (the
PP-OCR release now ships only the model GGUFs plus the image bundle), so
the det/rec/layout probes cannot run and the task has no numeric gate.
This script rebuilds the fixtures locally and deterministically:

1. Read the F32 GGUF (weights + embedded `ocr.graph.ops` topology).
2. Replay the audited Paddle graph in plain numpy NCHW semantics.
3. Feed a pinned, seed-stable synthetic input and save input + reference.

The numpy replay is an independent implementation of the same source graph
the C++ ggml builder executes: the probe's element-wise comparison against
these references is therefore a true parity gate (graph semantics, layout
mapping, activation formulas), not a self-check. Convert once per model
and publish the .npy files alongside the GGUFs.

Usage:
    python3 gen_ocr_fixtures.py --model <gguf> --out <dir> [--seed S]
                             [--stem rec_en]

The published det/rec fixtures are NOT produced here any more: on a uniform
random input the DB head is ~0 everywhere (max 0.0036) and the CTC head is
blank at all 40 steps, so those gates could not fail. They are real official
images through the Paddle Inference predictor of the GGUF source archive
(core/AICore/tools/export_ocr_fixture.py). This script still serves the
layout fixture and the --fetch-at bisection simulator.

Default output layout mirrors the manifest relative paths:
    <out>/rec_ch_input.npy  <out>/rec_ch_reference.npy
    <out>/rec_en_input.npy  <out>/rec_en_reference.npy   (--stem rec_en)
    <out>/det_ch_input.npy  <out>/det_ch_reference.npy
    <out>/layout_input.npy  <out>/layout_head0.npy .. layout_head7.npy

Publishing: the manifest input_assets point at the PP-OCR release asset
names below, so uploading the generated files under those exact names makes
the runner's fetch path work on any host (no manifest edit needed):

    gh release upload PP-OCR \\
        rec_ch_input.npy rec_ch_reference.npy \\
        rec_en_input.npy rec_en_reference.npy \\
        det_ch_input.npy det_ch_reference.npy \\
        layout_input.npy layout_head{0..7}.npy \\
        --repo Asher-1/cloudViewer_downloads
"""

import argparse
import hashlib
import json
import struct
import sys
from collections import Counter

import numpy as np

import os as _os
_DEBUG = _os.environ.get("OCR_SIM_DEBUG") == "1"

GGUF_ALIGNMENT = 32
GGML_TYPE_F32 = 0


# ---------------------------------------------------------------------------
# Minimal GGUF reader (matches the pinned ggml gguf.cpp wire format: string
# lengths are u64 for both keys and values, counts are u64).
# ---------------------------------------------------------------------------

def _read_scalar(f, code):
    return struct.unpack(code, f.read(struct.calcsize(code)))[0]


def _read_string(f):
    n = _read_scalar(f, "<Q")
    return f.read(n).decode("utf-8", "replace")


def _read_value(f, gtype):
    if gtype == 8:  # string
        return _read_string(f)
    if gtype == 9:  # array
        elem = _read_scalar(f, "<I")
        n = _read_scalar(f, "<Q")
        return [_read_value(f, elem) for _ in range(n)]
    codes = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f",
             7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
    return _read_scalar(f, codes[gtype])


class Gguf:
    def __init__(self, path):
        with open(path, "rb") as f:
            assert f.read(4) == b"GGUF"
            _read_scalar(f, "<I")  # version
            n_tensors = _read_scalar(f, "<Q")
            n_kv = _read_scalar(f, "<Q")
            self.kv = {}
            for _ in range(n_kv):
                key = _read_string(f)
                self.kv[key] = _read_value(f, _read_scalar(f, "<I"))
            self.tensors = {}
            data_start = None
            for _ in range(n_tensors):
                name = _read_string(f)
                n_dims = _read_scalar(f, "<I")
                ne = [_read_scalar(f, "<Q") for _ in range(n_dims)]
                ttype = _read_scalar(f, "<i")
                offset = _read_scalar(f, "<Q")
                self.tensors[name] = (ne, ttype, offset)
            pos = f.tell()
            data_start = (pos + GGUF_ALIGNMENT - 1) & ~(GGUF_ALIGNMENT - 1)
            self.data_start = data_start
        self._path = path
        # Tensor info names carry the task prefix ("ocr.conv2d_157.w_0");
        # expose them stripped so op args resolve directly.
        self._names = {name[len("ocr."):] if name.startswith("ocr.")
                       else name for name in self.tensors}

    def kv_tensor_names(self):
        return self._names

    def tensor(self, name):
        """Return the tensor as float32 with its original numpy shape.

        The converter stores numpy arrays (fastest axis last); GGUF ne is
        the reversed shape, so reversing ne restores the numpy shape. This
        generator targets F32 containers only.
        """
        ne, ttype, offset = self.tensors[name]
        assert ttype == GGML_TYPE_F32, (
            f"{name}: only F32 containers are supported, got type {ttype}")
        count = int(np.prod(ne)) if ne else 1
        with open(self._path, "rb") as f:
            f.seek(self.data_start + offset)
            raw = f.read(count * 4)
        assert len(raw) == count * 4, f"{name}: truncated tensor data"
        arr = np.frombuffer(raw, dtype="<f4").copy()
        shape = tuple(reversed(ne)) if ne else ()
        return arr.reshape(shape)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Pinned deterministic inputs (seed-stable PCG64 stream; NEP 19 guarantees
# the stream for a given bit generator + seed).
# ---------------------------------------------------------------------------

def pinned_input(shape, lo, hi, seed):
    rng = np.random.default_rng(seed)
    return (rng.random(shape, dtype=np.float64) * (hi - lo) + lo).astype(
        np.float32)


# ---------------------------------------------------------------------------
# Graph simulator (Paddle inference semantics, numpy NCHW)
# ---------------------------------------------------------------------------

class Sim:
    def __init__(self, gguf: Gguf, ops, ranks):
        self.g = gguf
        self.ops = ops
        self.ranks = ranks
        self.values = {}
        self.consts = {}

    def resolve(self, name):
        if name in self.values:
            return self.values[name]
        if name in self.g.kv_tensor_names():
            return self.g.tensor("ocr." + name)
        return None

    # -- elementwise broadcast (Paddle axis semantics with the channel-match
    # fallback the runtime documents for 1-D operands). Returns the operand
    # pair (A, B) with B made broadcastable against A. -----------------------
    @staticmethod
    def broadcast(x, y, axis):
        if y.ndim > x.ndim:
            x, y = y, x
        if y.ndim == 0:
            return x, y.reshape(1)
        if axis >= 0 and y.ndim == 1:
            nd = max(x.ndim, axis + 1)
            shape = [1] * nd
            shape[axis] = y.shape[0]
            if x.ndim < nd:
                x = x.reshape([1] * (nd - x.ndim) + list(x.shape))
            return x, y.reshape(shape)
        if y.ndim == 1 and y.shape[0] != 1 and not all(
                d == 1 or d == y.shape[0] for d in x.shape):
            # Place the vector on the first axis (W,H,C,N scan order, i.e.
            # numpy axes reversed) whose size matches the vector length.
            for d in (3, 2, 1, 0):
                if d < x.ndim and x.shape[d] == y.shape[0]:
                    shape = [1] * x.ndim
                    shape[d] = y.shape[0]
                    return x, y.reshape(shape)
        return x, y

    def op_inputs(self, op, param):
        out = []
        for io in op.get("inputs", []):
            if io.get("param") == param:
                out.extend(io.get("args", []))
        return out

    def first(self, op, param=None):
        if param is not None:
            # A named parameter with empty args (e.g. "Bias": []) must
            # resolve to None — never fall through to another input.
            names = self.op_inputs(op, param)
            return self.resolve(names[0]) if names else None
        for io in op.get("inputs", []):
            for a in io.get("args", []):
                t = self.resolve(a)
                if t is not None:
                    return t
        return None

    def attr(self, op, name, default=None):
        for a in op.get("attrs", []):
            if a.get("name") == name:
                return a.get("value")
        return default

    def _shape(self, name):
        v = self.resolve(name)
        return list(v.shape) if v is not None else None

    def run(self, feed_name, feed_value, want_fetch_count=1,
            stop_after=None):
        ops = self.ops
        if stop_after is not None:
            # Bisection mode: replay only the first stop_after+1 ops and
            # report the last activation's sum (pairs with the C++ probe's
            # AICORE_OCR_FETCH_AT truncation).
            ops = ops[:stop_after + 1]
        # FPN alignment pre-scan (nearest_interp_v2 with dynamic OutSize
        # resizes to the branch it is fused with).
        nearest_outs = {o["outputs"][0]["args"][0] for o in ops
                        if o["type"] == "nearest_interp_v2" and o.get("outputs")}
        nearest_target = {}
        for op in ops:
            if op["type"] not in ("elementwise_add", "concat"):
                continue
            args = [a for io in op.get("inputs", [])
                    for a in io.get("args", []) if a]
            near = [a for a in args if a in nearest_outs]
            other = [a for a in args if a not in nearest_outs]
            if near and other:
                for n in near:
                    nearest_target[n] = other[0]

        fetches = []
        for op_index, op in enumerate(ops):
            t = op["type"]
            # Output args bind by parameter: ops like batch_norm emit
            # MeanOut/SavedMean/... alongside Y, and only Y carries the
            # activation (its MeanOut reuses the running-mean var name).
            out_ios = op.get("outputs", [])
            out_params = {io.get("param") for io in out_ios}
            primary = None
            if len(out_params) > 1:
                primary = "Y" if "Y" in out_params else (
                    "Output" if "Output" in out_params else "Out")
            out_names = [a for io in out_ios
                         if (primary is None or io.get("param") == primary)
                         for a in io.get("args", [])]
            out = out_names[0] if out_names else ""
            # Failure locator for the caller (printed on replay errors).
            self.last_op = f"op[{op_index}] {t} out={out}"

            if t == "feed":
                self.values[out] = feed_value
                continue
            if t in ("fetch", "assign"):
                v = self.first(op)
                self.values[out] = v
                if t == "fetch":
                    fetches.append(v)
                continue
            if t == "dropout":
                self.values[out] = self.first(op)
                continue

            if t in ("conv2d", "depthwise_conv2d"):
                x = self.first(op, "Input")
                w = self.first(op, "Filter")
                b = self.first(op, "Bias")
                groups = int(self.attr(op, "groups", 1) or 1)
                strides = self.attr(op, "strides") or [1, 1]
                dilations = self.attr(op, "dilations") or [1, 1]
                paddings = self.attr(op, "paddings") or [0, 0]
                if len(paddings) == 2:
                    ph, pw = int(paddings[0]), int(paddings[1])
                else:
                    ph = (int(paddings[0]) + int(paddings[2])) // 2
                    pw = (int(paddings[1]) + int(paddings[3])) // 2
                sh, sw = int(strides[0]), int(strides[1])
                dh, dw = int(dilations[0]), int(dilations[1])
                res = self.conv(x, w, b, sh, sw, ph, pw, dh, dw, groups,
                                t == "depthwise_conv2d")
                resid = self.first(op, "ResidualData")
                if resid is not None:
                    res = res + resid
                self.values[out] = res
                continue

            if t == "conv2d_transpose":
                x = self.first(op, "Input")
                w = self.first(op, "Filter")
                b = self.first(op, "Bias")
                strides = self.attr(op, "strides") or [1, 1]
                res = self.conv_transpose(x, w, int(strides[0]),
                                          int(strides[1]))
                if b is not None:
                    _, b = self.broadcast(res, b, 1)
                    res = res + b
                self.values[out] = res
                continue

            if t == "batch_norm":
                x = self.first(op, "X")
                scale = self.first(op, "Scale")
                bias = self.first(op, "Bias")
                mean = self.first(op, "Mean")
                var = self.first(op, "Variance")
                eps = float(self.attr(op, "epsilon", 1e-5))
                inv = scale / np.sqrt(var + eps)
                self.values[out] = x * inv.reshape(1, -1, 1, 1) + \
                    (bias - mean * inv).reshape(1, -1, 1, 1)
                continue

            if t in ("elementwise_add", "elementwise_mul"):
                a = self.first(op, "X")
                b = self.first(op, "Y")
                if a is None or b is None:
                    args = [n for io in op.get("inputs", [])
                            for n in io.get("args", [])]
                    a = self.resolve(args[0])
                    for n in args[1:]:
                        t2 = self.resolve(n)
                        if t2 is not None and t2 is not a:
                            b = t2
                            break
                axis = int(self.attr(op, "axis", -1))
                a, b = self.broadcast(a, b, axis)
                self.values[out] = (a + b) if t == "elementwise_add" \
                    else (a * b)
                continue

            if t == "relu":
                self.values[out] = np.maximum(self.first(op), 0.0)
                continue
            if t == "sigmoid":
                x = self.first(op)
                self.values[out] = 1.0 / (1.0 + np.exp(-x))
                continue
            if t == "hard_swish":
                x = self.first(op)
                scale = float(self.attr(op, "scale", 6.0))
                offset = float(self.attr(op, "offset", 3.0))
                self.values[out] = x * np.clip(x + offset, 0.0, scale) / scale
                continue
            if t == "hard_sigmoid":
                x = self.first(op)
                slope = float(self.attr(op, "slope", 0.2))
                offset = float(self.attr(op, "offset", 0.5))
                self.values[out] = np.clip(x * slope + offset, 0.0, 1.0)
                continue
            if t == "swish":
                x = self.first(op)
                beta = float(self.attr(op, "beta", 1.0))
                self.values[out] = x / (1.0 + np.exp(-beta * x))
                continue
            if t == "softmax":
                x = self.first(op)
                axis = int(self.attr(op, "axis", -1))
                e = np.exp(x - x.max(axis=axis, keepdims=True))
                self.values[out] = e / e.sum(axis=axis, keepdims=True)
                continue
            if t == "layer_norm":
                # Inputs arrive alphabetically (Bias, Scale, X) — the
                # activation must be resolved by its "X" parameter.
                x = self.first(op, "X")
                eps = float(self.attr(op, "epsilon", 1e-5))
                bna = int(self.attr(op, "begin_norm_axis", -1))
                if bna < 0:
                    bna = x.ndim + bna
                axes = tuple(range(bna, x.ndim))
                mu = x.mean(axis=axes, keepdims=True)
                var = x.var(axis=axes, keepdims=True)
                y = (x - mu) / np.sqrt(var + eps)
                scale = self.first(op, "Scale")
                bias = self.first(op, "Bias")
                if scale is not None:
                    y = y * scale.reshape([1] * (x.ndim - 1) + [-1])
                if bias is not None:
                    y = y + bias.reshape([1] * (x.ndim - 1) + [-1])
                self.values[out] = y
                continue
            if t == "scale":
                x = self.first(op)
                s = float(self.attr(op, "scale", 1.0))
                b = float(self.attr(op, "bias", 0.0))
                self.values[out] = x * s + b
                continue

            if t == "matmul_v2":
                a = self.first(op, "X")
                b = self.first(op, "Y")
                # Numpy space: Y stays as stored (in, out) for trans_y=False
                # — the ggml builder transposes only because gguf ne reverses
                # the numpy shape, which does not apply here.
                if self.attr(op, "trans_x"):
                    a = a.transpose()
                if self.attr(op, "trans_y"):
                    b = b.transpose()
                self.values[out] = np.matmul(a, b).astype(np.float32)
                continue
            if t == "transpose2":
                x = self.first(op, "X")
                perm = self.attr(op, "axis") or self.attr(op, "permute")
                if not perm:
                    perm = [0, 2, 1] if x.ndim == 3 else [0, 2, 3, 1]
                # NOTE: no rank-5 folding here — the C++ builder folds a
                # size-1 batch axis only because ggml tensors cap at 4-D;
                # numpy carries the natural rank directly.
                self.values[out] = x.transpose(perm)
                continue
            if t == "reshape2":
                x = self.first(op, "X")
                shape = list(self.attr(op, "shape") or [])
                st = 0
                for io in op.get("inputs", []):
                    if io.get("param") != "ShapeTensor":
                        continue
                    for arg in io.get("args", []):
                        if arg in self.consts and self.consts[arg]:
                            shape[st] = self.consts[arg][0]
                        st += 1
                total = int(x.size)
                stated = 1
                infer_at = -1
                for i, s in enumerate(shape):
                    if s == -1:
                        infer_at = i
                    elif s == 0:
                        # Paddle: 0 copies the input's dimension.
                        shape[i] = x.shape[i] if i < x.ndim else 1
                        stated *= shape[i]
                    else:
                        stated *= s
                if infer_at >= 0:
                    shape[infer_at] = total // stated
                self.values[out] = x.reshape(shape)
                continue
            if t in ("squeeze2", "unsqueeze2"):
                x = self.first(op, "X")
                axes = self.attr(op, "axes") or []
                if t == "squeeze2":
                    self.values[out] = np.squeeze(x, axis=tuple(axes))
                else:
                    self.values[out] = np.expand_dims(x, axis=list(axes))
                continue
            if t == "flatten_contiguous_range":
                x = self.first(op, "X")
                first = int(self.attr(op, "start_axis", 0))
                last = int(self.attr(op, "stop_axis", first))
                if first < 0:
                    first += x.ndim
                if last < 0:
                    last += x.ndim
                shape = list(x.shape)
                merged = int(np.prod(shape[first:last + 1])) if last >= first \
                    else 1
                out_shape = shape[:first] + [merged] + shape[last + 1:]
                self.values[out] = x.reshape(out_shape)
                continue

            if t == "concat":
                axis = int(self.attr(op, "axis", 1))
                parts = []
                for io in op.get("inputs", []):
                    if io.get("param") not in (None, "X"):
                        continue
                    for arg in io.get("args", []):
                        v = self.resolve(arg)
                        if v is not None:
                            parts.append(v)
                self.values[out] = np.concatenate(parts, axis=axis)
                continue
            if t == "split":
                x = self.first(op, "X")
                axis = int(self.attr(op, "axis", 1))
                sections = self.attr(op, "sections") or []
                num = int(self.attr(op, "num", 0) or 0)
                if sections:
                    sizes = [int(s) for s in sections]
                else:
                    sizes = [x.shape[axis] // num] * num
                pieces = np.split(x, np.cumsum(sizes)[:-1], axis=axis)
                for name, piece in zip(out_names, pieces):
                    self.values[name] = piece
                continue

            if t == "pool2d":
                x = self.first(op, "X")
                global_pool = bool(self.attr(op, "global_pooling", False))
                adaptive = bool(self.attr(op, "adaptive", False))
                ptype = self.attr(op, "pooling_type", "max")
                if global_pool or adaptive:
                    axes = (2, 3)
                    self.values[out] = x.mean(axis=axes, keepdims=True) \
                        if ptype == "avg" else x.max(axis=axes, keepdims=True)
                    continue
                ksize = self.attr(op, "ksize") or [1, 1]
                strides = self.attr(op, "strides") or [1, 1]
                paddings = self.attr(op, "paddings") or [0, 0]
                kh, kw = int(ksize[0]), int(ksize[1])
                sh, sw = int(strides[0]), int(strides[1])
                ph, pw = int(paddings[0]), int(paddings[1])
                self.values[out] = self.pool(x, kh, kw, sh, sw, ph, pw,
                                             ptype == "avg")
                continue

            if t == "nearest_interp_v2":
                x = self.first(op, "X")
                out_h, out_w = x.shape[2] * 2, x.shape[3] * 2
                ref = nearest_target.get(out)
                if ref is not None:
                    r = self.resolve(ref)
                    if r is not None:
                        out_h, out_w = r.shape[2], r.shape[3]
                self.values[out] = self.nearest(x, out_h, out_w)
                continue

            if t == "shape":
                x = self.first(op)
                self.consts[out] = [int(d) for d in x.shape]
                self.values[out] = np.asarray(x.shape, dtype=np.float32)
                continue
            if t == "fill_constant":
                sv = self.attr(op, "str_value", None)
                v = int(sv) if sv else int(float(self.attr(op, "value", 0.0)))
                self.consts[out] = [v]
                continue
            if t == "slice":
                names = self.op_inputs(op, "Input")
                starts = [int(s) for s in (self.attr(op, "starts") or [])]
                ends = [int(s) for s in (self.attr(op, "ends") or [])]
                axes = [int(s) for s in (self.attr(op, "axes") or [])]
                decrease = [int(s) for s in
                            (self.attr(op, "decrease_axis") or [])]
                if names and names[0] in self.consts:
                    vals = list(self.consts[names[0]])
                    s = starts[0] if starts else 0
                    e = ends[0] if ends else len(vals)
                    if s < 0:
                        s += len(vals)
                    if e < 0:
                        e += len(vals)
                    e = min(e, len(vals))
                    kept = vals[s:e] if 0 <= s <= e else []
                    self.consts[out] = kept
                else:
                    x = self.first(op)
                    idx = []
                    for d in range(x.ndim):
                        s = 0
                        e = x.shape[d]
                        if d in axes:
                            i = axes.index(d)
                            s = starts[i] if i < len(starts) else 0
                            e = ends[i] if i < len(ends) else e
                            if s < 0:
                                s += x.shape[d]
                            if e < 0:
                                e += x.shape[d]
                            e = min(e, x.shape[d])
                        idx.append(slice(s, e))
                    v = x[tuple(idx)]
                    if decrease:
                        v = np.squeeze(v, axis=tuple(decrease))
                    self.values[out] = v
                continue

            raise ValueError(f"unsupported op: {t}")

        if len(fetches) < want_fetch_count:
            raise ValueError(
                f"expected {want_fetch_count} fetches, got {len(fetches)}")
        return fetches

    def run_fetch_sum(self, feed_name, feed_value, stop_after):
        """Bisection helper: replay the first stop_after+1 ops and return
        (shape, sum) of the truncated build's trailing activation."""
        self.run(feed_name, feed_value, 0, stop_after=stop_after)
        last_name = next(reversed(self.values))
        v = self.values[last_name]
        v = np.asarray(v, dtype=np.float32)
        return list(v.shape), float(v.sum()), last_name

    @staticmethod
    def conv(x, w, b, sh, sw, ph, pw, dh, dw, groups, depthwise):
        # x (N,C,H,W), w (Cout, Cin/g, kh, kw) — paddle weight layout.
        n, c, h, wd = x.shape
        xp = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
        cout, cin_g, kh, kw = w.shape
        eh = (h + 2 * ph - dh * (kh - 1) - 1) // sh + 1
        ew = (wd + 2 * pw - dw * (kw - 1) - 1) // sw + 1
        out = np.zeros((n, cout, eh, ew), dtype=np.float32)
        for g in range(groups):
            cg = slice(g * cin_g, (g + 1) * cin_g)
            og = slice(g * (cout // groups), (g + 1) * (cout // groups))
            wR = w[og].reshape(cout // groups, -1).astype(np.float32)
            taps = cin_g * kh * kw
            acc = np.zeros((n, taps, eh * ew), dtype=np.float32)
            t = 0
            for ci in range(cin_g):
                for u in range(kh):
                    for v in range(kw):
                        acc[:, t] = xp[:, g * cin_g + ci,
                                       u * dh:u * dh + eh * sh:sh,
                                       v * dw:v * dw + ew * sw:sw].reshape(
                                           n, -1)
                        t += 1
            res = np.einsum("ok,nkp->nop", wR, acc, optimize=True)
            out[:, og] = res.reshape(n, cout // groups, eh, ew)
        if b is not None:
            out = out + b.reshape(1, -1, 1, 1)
        return out.astype(np.float32)

    @staticmethod
    def conv_transpose(x, w, sh, sw):
        # w (Cin, Cout, kh, kw): transposed convolution, zero padding.
        # Each kernel tap scatters its input pixel to a strided output
        # position: out[n, co, ys*sh+u, xs*sw+v] += sum_cin x[n,ci,ys,xs] *
        # w[ci,co,u,v].
        n, c, h, wd = x.shape
        cin, cout, kh, kw = w.shape
        oh = (h - 1) * sh + kh
        ow = (wd - 1) * sw + kw
        out = np.zeros((n, cout, oh, ow), dtype=np.float32)
        wr = w.reshape(cin, cout, kh * kw).astype(np.float32)
        xr = x.reshape(n, c, h * wd).astype(np.float32)
        contrib = np.einsum("ncp,coq->nopq", xr, wr,
                            optimize=True).reshape(n, cout, kh * kw, h, wd)
        for q in range(kh * kw):
            u, v = divmod(q, kw)
            out[:, :, u:u + (h - 1) * sh + 1:sh,
                v:v + (wd - 1) * sw + 1:sw] += contrib[:, :, q]
        return out.astype(np.float32)

    @staticmethod
    def pool(x, kh, kw, sh, sw, ph, pw, avg):
        n, c, h, wd = x.shape
        if ph or pw:
            if avg:
                mode = "constant"
                xp = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)),
                            mode=mode)
            else:
                xp = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)),
                            mode="constant", constant_values=-np.inf)
        else:
            xp = x
        eh = (h + 2 * ph - kh) // sh + 1
        ew = (wd + 2 * pw - kw) // sw + 1
        out = np.zeros((n, c, eh, ew), dtype=np.float32)
        for u in range(kh):
            for v in range(kw):
                win = xp[:, :, u:u + eh * sh:sh, v:v + ew * sw:sw]
                out = out + win if avg else np.maximum(out, win)
        if avg:
            out /= kh * kw
        return out.astype(np.float32)

    @staticmethod
    def nearest(x, oh, ow):
        # Nearest-neighbor resize by pixel-area replication (the paddle
        # nearest index: floor(i * in / out)).
        n, c, h, w = x.shape
        yi = (np.arange(oh) * h // oh).clip(0, h - 1)
        xi = (np.arange(ow) * w // ow).clip(0, w - 1)
        return x[:, :, yi][:, :, :, xi].astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="F32 OCR GGUF")
    parser.add_argument("--out", default=None,
                        help="output directory (default: the AICore "
                             "shared data root ocr_fixtures folder)")
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--det-h", type=int, default=512)
    parser.add_argument("--det-w", type=int, default=960)
    parser.add_argument("--fetch-at", type=int, default=None,
                        help="bisection mode: replay only op[0..N] and "
                             "print the trailing activation's sum (pairs "
                             "with the probe's AICORE_OCR_FETCH_AT)")
    parser.add_argument("--stem", default=None,
                        help="fixture file-name stem (default: rec_ch / "
                             "det_ch / layout by role; pass rec_en for the "
                             "English rec container)")
    args = parser.parse_args()

    if args.fetch_at is not None:
        gguf = Gguf(args.model)
        graph = json.loads(gguf.kv["ocr.graph.ops"])
        ops = graph["ops"] if isinstance(graph, dict) else graph
        role = gguf.kv.get("ocr.role")
        if role == "rec":
            x = pinned_input((1, 3, 48, 320), -1.0, 1.0, args.seed)
        elif role == "layout":
            x = pinned_input((1, 3, 800, 608), -2.5, 2.5, args.seed)
        else:
            x = pinned_input((1, 3, args.det_h, args.det_w), -2.5, 2.5,
                             args.seed)
        sim = Sim(gguf, ops, {})
        shape, total, name = sim.run_fetch_sum("x", x, args.fetch_at)
        print(f"[fetch-sim] {name} shape={shape} sum={total:.6e}")
        return
    import os
    out = args.out
    if not out:
        root = os.path.expanduser("~/cloudViewer_data/extract")
        out = os.path.join(root, "ocr_fixtures")
    os.makedirs(out, exist_ok=True)

    gguf = Gguf(args.model)
    role = gguf.kv.get("ocr.role", "")
    ops_json = gguf.kv.get("ocr.graph.ops")
    assert ops_json, "GGUF is missing ocr.graph.ops"
    graph = json.loads(ops_json)
    ops = graph["ops"] if isinstance(graph, dict) else graph
    ranks = graph.get("ranks", {}) if isinstance(graph, dict) else {}
    hist = Counter(o["type"] for o in ops)
    print(f"{args.model}: role={role} ops={len(ops)} hist={dict(hist)}")

    sim = Sim(gguf, ops, ranks)
    if role == "rec":
        stem = args.stem or "rec_ch"
        x = pinned_input((1, 3, 48, 320), -1.0, 1.0, args.seed)
        fetches = sim.run("x", x, 1)
        ref = fetches[0]
        # The rec fetch arrives as (T, vocab) for N=1; the probe consumes
        # the numpy (1, T, vocab) reference shape.
        if ref.ndim == 2:
            ref = ref[np.newaxis, ...]
        np.save(os.path.join(out, f"{stem}_input.npy"), x)
        np.save(os.path.join(out, f"{stem}_reference.npy"), ref)
        print(f"rec: input {x.shape} ref {ref.shape} stem={stem}")
    elif role == "det":
        x = pinned_input((1, 3, args.det_h, args.det_w), -2.5, 2.5,
                         args.seed)
        fetches = sim.run("x", x, 1)
        ref = fetches[0]
        if ref.ndim == 2:
            ref = ref[np.newaxis, np.newaxis, ...]
        elif ref.ndim == 3:
            ref = ref[np.newaxis, ...]
        np.save(os.path.join(out, "det_ch_input.npy"), x)
        np.save(os.path.join(out, "det_ch_reference.npy"), ref)
        print(f"det: input {x.shape} ref {ref.shape}")
    elif role == "layout":
        x = pinned_input((1, 3, 800, 608), -2.5, 2.5, args.seed)
        np.save(os.path.join(out, "layout_input.npy"), x)
        fetches = sim.run("x", x, 8)
        for i, head in enumerate(fetches):
            # PaddleDetection saves each head as (1, N, C).
            if head.ndim == 2:
                head = head[np.newaxis, ...]
            np.save(os.path.join(out, f"layout_head{i}.npy"), head)
            print(f"layout head{i}: {head.shape}")
    else:
        raise SystemExit(f"unsupported role: {role}")

    for name in sorted(os.listdir(out)):
        if name.endswith(".npy"):
            p = os.path.join(out, name)
            print(f"  {name}: sha256={sha256_file(p)}")


if __name__ == "__main__":
    main()
