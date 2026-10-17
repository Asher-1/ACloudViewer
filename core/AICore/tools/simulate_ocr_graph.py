#!/usr/bin/env python3
"""Numpy reference simulator for the AICore `ocr` graph.

Replays the exported Paddle topology with plain numpy (paddle NCHW
semantics) so the C++ ggml builder can be compared op-by-op. Reads the
same graph JSON + Paddle weight stream the converters use. Conversion
host only.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from paddle_ocr_extract import load_ocr_tensors  # noqa: E402


def conv2d(x, w, b, strides, paddings, dilations, groups):
    # x NCHW, w (Cout, Cin/g, kh, kw)
    n, c, h, wd = x.shape
    cout, cin_g, kh, kw = w.shape
    sh, sw = strides
    dh, dw = dilations
    ph = paddings[0] if len(paddings) <= 2 else paddings[0]
    pw = paddings[1] if len(paddings) <= 2 else paddings[1]
    if len(paddings) > 2:
        ph = (paddings[0] + paddings[2]) // 2
        pw = (paddings[1] + paddings[3]) // 2
    xp = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
    eh = (h + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    ew = (wd + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    # Per-tap accumulation keeps memory at O(output) even for large C.
    out = np.zeros((n, cout, eh, ew), dtype=np.float32)
    for g in range(groups):
        cg = slice(g * cin_g, (g + 1) * cin_g)
        og = slice(g * (cout // groups), (g + 1) * (cout // groups))
        wR = w[og].reshape(cout // groups, -1)
        taps = cin_g * kh * kw
        acc = np.zeros((n, taps, eh * ew), dtype=np.float32)
        t = 0
        for ci in range(cin_g):
            for u in range(kh):
                for v in range(kw):
                    acc[:, t] = (xp[:, g * cin_g + ci,
                                    u * dh:u * dh + eh * sh:sh,
                                    v * dw:v * dw + ew * sw:sw]).reshape(
                        n, -1)
                    t += 1
        res = np.einsum("ok,nkp->nop", wR, acc, optimize=True)
        out[:, og] = res.reshape(n, cout // groups, eh, ew)
    if b is not None:
        out += b.reshape(1, -1, 1, 1)
    return out


def depthwise_conv2d(x, w, b, strides, paddings, dilations):
    # w (Cin, 1, kh, kw): per-channel convolution
    n, c, h, wd = x.shape
    kh, kw = w.shape[2], w.shape[3]
    sh, sw = strides
    dh, dw = dilations
    ph = paddings[0] if len(paddings) <= 2 else (paddings[0] + paddings[2]) // 2
    pw = paddings[1] if len(paddings) <= 2 else (paddings[1] + paddings[3]) // 2
    xp = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
    eh = (h + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    ew = (wd + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    out = np.empty((n, c, eh, ew), dtype=np.float32)
    for ci in range(c):
        k = w[ci, 0]
        acc = np.zeros((n, eh, ew), dtype=np.float32)
        for u in range(kh):
            for v in range(kw):
                acc += xp[:, ci, u * dh:u * dh + eh * sh:sh,
                          v * dw:v * dw + ew * sw:sw] * k[u, v]
        out[:, ci] = acc
    if b is not None:
        out += b.reshape(1, -1, 1, 1)
    return out


def pool2d(x, ksize, strides, paddings, adaptive, method):
    if adaptive:
        return x.mean(axis=(2, 3), keepdims=True)
    kh, kw = ksize[1], ksize[0]  # paddle ksize = [h, w]
    sh, sw = strides[0], strides[1]
    ph = paddings[0]
    pw = paddings[1]
    n, c, h, w = x.shape
    xp = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
    eh = (h + 2 * ph - kh) // sh + 1
    ew = (w + 2 * pw - kw) // sw + 1
    out = np.empty((n, c, eh, ew), dtype=np.float32)
    for u in range(kh):
        for v in range(kw):
            patch = xp[:, :, u:u + eh * sh:sh, v:v + ew * sw:sw]
            out += patch
    out /= kh * kw
    return out


def nearest_resize(x, oh, ow):
    n, c, h, w = x.shape
    yi = (np.arange(oh) * h // oh).clip(0, h - 1)
    xi = (np.arange(ow) * w // ow).clip(0, w - 1)
    return x[:, :, yi][:, :, :, xi]


def simulate(ops, tensors, x, log=None):
    values = {}
    params = {k[len("ocr."):]: v for k, v in tensors.items()}

    # FPN alignment: pin each nearest to its fused branch size.
    nearest_outs = {o["outputs"][0]["args"][0] for o in ops
                    if o["type"] == "nearest_interp_v2" and o.get("outputs")}
    nearest_target = {}
    for o in ops:
        if o["type"] not in ("elementwise_add", "concat"):
            continue
        xs = [a for io in o.get("inputs", [])
              for a in io.get("args", []) if a]
        nearest_vars = [v for v in xs if v in nearest_outs]
        others = [v for v in xs if v not in nearest_outs]
        if nearest_vars and others:
            for v in nearest_vars:
                nearest_target[v] = others[0]

    def resolve(name):
        if name in values:
            return values[name]
        return params.get(name)

    for i, op in enumerate(ops):
        t = op["type"]
        attrs = {a["name"]: a.get("value") for a in op.get("attrs", [])}
        ins = {io["param"]: io["args"] for io in op.get("inputs", [])}
        outs = [a for io in op.get("outputs", []) for a in io.get("args", [])]
        outName = outs[0] if outs else ""

        def arg(param):
            args = ins.get(param) or []
            if not args:
                return None
            v = resolve(args[0])
            if v is None:
                raise SystemExit(
                    f"simulator: unresolved input {param}={args[0]} at [{i}] {t}")
            return v

        if t == "feed":
            # The output variable name varies (x / image / ...); bind the
            # declared output instead of a fixed name.
            values[outName or "x"] = x
            continue
        if t in ("fetch", "assign"):
            values[outName] = resolve(ins["X"][0])
            continue
        if t == "conv2d":
            w = arg("Filter")
            if w.ndim == 2:
                raise SystemExit("folded kernel in simulator input")
            values[outName] = conv2d(arg("Input"), w, arg("Bias"),
                                     attrs.get("strides", [1, 1]),
                                     attrs.get("paddings", [0, 0]),
                                     attrs.get("dilations", [1, 1]),
                                     attrs.get("groups", 1))
        elif t == "depthwise_conv2d":
            values[outName] = depthwise_conv2d(
                arg("Input"), arg("Filter"), arg("Bias"),
                attrs.get("strides", [1, 1]), attrs.get("paddings", [0, 0]),
                attrs.get("dilations", [1, 1]))
        elif t == "conv2d_transpose":
            w = arg("Filter")  # (Cin, Cout, kh, kw)
            x_in = arg("Input")
            stride = attrs.get("strides", [1, 1])[0]
            cin, cout, kh, kw = w.shape
            n, c, h, wd = x_in.shape
            oh = (h - 1) * stride + kh
            ow = (wd - 1) * stride + kw
            out = np.zeros((n, cout, oh, ow), dtype=np.float32)
            for ci in range(cin):
                for cu in range(cout):
                    k2 = w[ci, cu]  # (kh, kw)
                    for u in range(kh):
                        for v in range(kw):
                            out[:, cu, u:oh:stride, v:ow:stride] += \
                                x_in[:, ci] * k2[u, v]
            b = arg("Bias")
            if b is not None:
                out += b.reshape(1, -1, 1, 1)
            values[outName] = out
        elif t == "batch_norm":
            xv = arg("X")
            scale = arg("Scale")
            bias = arg("Bias")
            mean = arg("Mean")
            var = arg("Variance")
            eps = attrs.get("epsilon", 1e-5)
            m = scale / np.sqrt(var + eps)
            values[outName] = (xv - mean.reshape(1, -1, 1, 1)) * m.reshape(
                1, -1, 1, 1) + bias.reshape(1, -1, 1, 1)
        elif t == "elementwise_add" or t == "elementwise_mul":
            a = arg("X")
            bv = arg("Y")
            if np.prod(a.shape) < np.prod(bv.shape):
                a, bv = bv, a
            # Paddle alignment: the smaller operand aligns starting at
            # `axis` (default -1 -> trailing), not numpy's leading rule.
            if a.ndim != bv.ndim:
                axis = attrs.get("axis", -1)
                if axis < 0:
                    axis = a.ndim - bv.ndim
                shape = [1] * a.ndim
                for i, d in enumerate(bv.shape):
                    shape[axis + i] = d
                bv = bv.reshape(shape)
            values[outName] = a + bv if t == "elementwise_add" else a * bv
        elif t == "relu":
            values[outName] = np.maximum(arg("X"), 0)
        elif t == "sigmoid":
            values[outName] = 1.0 / (1.0 + np.exp(-arg("X")))
        elif t == "swish":
            xv = arg("X")
            values[outName] = xv / (1.0 + np.exp(-xv))
        elif t == "hard_swish":
            xv = arg("X")
            scale = attrs.get("scale", 6.0)
            offset = attrs.get("offset", 3.0)
            values[outName] = xv * np.clip(np.maximum(xv + offset, 0), 0,
                                           scale) / scale
        elif t == "hard_sigmoid":
            xv = arg("X")
            slope = attrs.get("slope", 0.2)
            offset = attrs.get("offset", 0.5)
            values[outName] = np.clip(xv * slope + offset, 0, 1)
        elif t == "pool2d":
            values[outName] = pool2d(arg("X"), attrs.get("ksize", [1, 1]),
                                     attrs.get("strides", [1, 1]),
                                     attrs.get("paddings", [0, 0]),
                                     attrs.get("adaptive", False),
                                     attrs.get("pooling_type", "avg"))
        elif t == "nearest_interp_v2":
            xv = arg("X")
            target = nearest_target.get(outName)
            if target is not None:
                ref = resolve(target)
                oh, ow = ref.shape[2], ref.shape[3]
            else:
                oh, ow = xv.shape[2] * 2, xv.shape[3] * 2
            values[outName] = nearest_resize(xv, oh, ow)
        elif t == "concat":
            axis = attrs.get("axis", 1)
            parts = [resolve(a) for a in ins.get("X", []) if resolve(a) is not None]
            values[outName] = np.concatenate(parts, axis=axis)
        elif t == "split":
            # numpy axis over (N,C,H,W); sections fall back to equal split
            xv = arg("X")
            sections = attrs.get("sections")
            axis = attrs.get("axis", 1)
            if not sections:
                num = attrs.get("num", len(outs))
                sections = [xv.shape[axis] // num] * num
            off = 0
            for nm, sec in zip(outs, sections):
                values[nm] = np.take(xv, range(off, off + sec), axis=axis)
                off += sec
            continue  # outputs already bound individually
        elif t == "sigmoid" or t == "softmax":
            if t == "softmax":
                xv = arg("X")
                e = np.exp(xv - xv.max(axis=2, keepdims=True))
                values[outName] = e / e.sum(axis=2, keepdims=True)
        elif t == "reshape2":
            shape_attr = attrs.get("shape")
            xv = arg("X")
            if shape_attr:
                target = [d if d != -1 else -1 for d in shape_attr]
                known = [d for d in target if d != -1]
                if len(known) < len(target):
                    rest = xv.size // (int(np.prod(known)) if known else 1)
                    target = [d if d != -1 else rest for d in target]
                values[outName] = xv.reshape(target)
            else:
                values[outName] = xv.reshape(xv.shape)
        elif t == "transpose2":
            perm = attrs.get("permute", attrs.get("perm"))
            xv = arg("X")
            if not perm:
                # AnyLayout exports omit the attribute: the convention is
                # NCHW -> NHWC, generalized to ND as (0, 2, 1, 3, ...).
                perm = [0, 2] + list(range(3, xv.ndim)) + [1]
            # numpy gather semantics
            values[outName] = np.ascontiguousarray(xv.transpose(perm))
        elif t == "matmul_v2" or t == "matmul":
            a = arg("X")
            b = arg("Y")
            values[outName] = np.matmul(a, b)
        elif t == "layer_norm":
            xv = arg("X")
            eps = attrs.get("epsilon", 1e-5)
            beg = attrs.get("begin_norm_axis", 1)
            ax = tuple(range(beg, xv.ndim))
            mu = xv.mean(axis=ax, keepdims=True)
            var = xv.var(axis=ax, keepdims=True)
            values[outName] = (xv - mu) / np.sqrt(var + eps)
        elif t == "scale":
            xv = arg("X")
            values[outName] = xv * attrs.get("scale", 1.0) + attrs.get("bias", 0.0)
        elif t == "elementwise_sub" or t == "elementwise_div" or t == "elementwise_max" or t == "elementwise_min":
            a = arg("X")
            b = arg("Y")
            opf = {"elementwise_sub": np.subtract, "elementwise_div": np.divide,
                   "elementwise_max": np.maximum, "elementwise_min": np.minimum}[t]
            values[outName] = opf(a, b)
        else:
            raise SystemExit(f"simulator: unsupported op {t} at [{i}]")
        # batch_norm's front output is the Mean parameter var, not Y;
        # prefer the activation (.tmp_) var when mirroring all outputs
        # (same semantics as the C++ builder, which stores result in
        # every declared output).
        result_var = values.get(outName)
        for nm in outs:
            if ".tmp_" in nm and values.get(nm) is not None:
                result_var = values[nm]
                break
        for nm in outs:
            values[nm] = result_var
        if log is not None and outName:
            log.write(f"[{i:3d}] {t} {outName} "
                      f"sum={float(np.sum(result_var.astype(np.float64))):.6e}\n")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("graph_json", type=Path)
    parser.add_argument("--input-npy", type=Path, required=True)
    parser.add_argument("--log", type=Path, default=None)
    args = parser.parse_args()

    doc = json.loads(args.graph_json.read_text())
    ops = doc["ops"] if isinstance(doc, dict) else doc
    named, _feed, _fetch = load_ocr_tensors(args.model_dir)
    tensors = {"ocr." + name: arr for name, arr in named}
    x = np.load(args.input_npy).astype(np.float32)
    log = open(args.log, "w") if args.log else None
    values = simulate(ops, tensors, x, log)
    # final fetch
    fetch = values.get("save_infer_model/scale_0.tmp_0")
    if fetch is None:
        last = [v for v in values.values()]
        fetch = last[-1]
    print("simulated output shape:", fetch.shape,
          "sum:", float(np.sum(fetch.astype(np.float64))))
    np.save("/tmp/ocr_sim_reference.npy", fetch)
    if log:
        log.close()


if __name__ == "__main__":
    main()
