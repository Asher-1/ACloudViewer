#!/usr/bin/env python3
"""Export the Paddle inference op topology for the AICore `ocr` task.

Produces the authoritative operator sequence (types, tensor arguments and
selected attributes) the C++ graph builders must reproduce, plus reference
fixtures (fixed input -> paddle output) used by the parity probe. Runs on
the conversion host only.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from paddle_ocr_extract import (  # noqa: E402
    _parse_tensor_desc,
    parse_fields,
)


def var_shapes(pdmodel: Path):
    """Static serialized VarDesc shape (dims may contain -1) per tensor."""
    data = pdmodel.read_bytes()
    shapes = {}
    for field, wire, value in parse_fields(data):
        if field != 1 or wire != 2:
            continue  # ProgramDesc.blocks
        for bfield, bwire, bvalue in parse_fields(value):
            if bfield != 3 or bwire != 2:
                continue  # BlockDesc.vars map entries
            name = None
            dims = []
            for efield, ewire, evalue in parse_fields(bvalue):
                if efield == 1 and ewire == 2:
                    name = evalue.decode("utf-8")
                elif efield == 2 and ewire == 2:
                    for vfield, vwire, vvalue in parse_fields(evalue):
                        if vfield == 3 and vwire == 2:
                            for wfield, wwire, wvalue in parse_fields(vvalue):
                                if wfield == 1 and wwire == 2:
                                    _, dims = _parse_tensor_desc(wvalue)
                                    break
            if name and dims:
                shapes[name] = list(dims)
    return shapes


def op_sequence(pdmodel: Path):
    data = pdmodel.read_bytes()
    block = next(v for f, w, v in parse_fields(data) if f == 1 and w == 2)
    ops = []
    for op in (v for f, w, v in parse_fields(block) if f == 4 and w == 2):
        otype = None
        ins, outs = [], []
        attrs = []
        for f, w, v in parse_fields(op):
            if f == 3 and w == 2:
                otype = v.decode()
            elif f in (1, 2) and w == 2:
                names, param = [], None
                for vf, vw, vv in parse_fields(v):
                    if vf == 1:
                        param = vv.decode()
                    elif vf == 2:
                        names.append(vv.decode())
                (ins if f == 1 else outs).append(
                    {"param": param, "args": names})
            elif f == 4 and w == 2:
                # OpDesc.Attr: name=1, type=2, i=3, f=4, s=5, ints=6,
                # floats=7, strings=8, b=10.
                attr = {"name": None}
                def _ints(buf):
                    out, pos = [], 0
                    while pos < len(buf):
                        v = 0; shift = 0
                        while True:
                            b = buf[pos]; pos += 1
                            v |= (b & 0x7F) << shift
                            if not b & 0x80: break
                            shift += 7
                        out.append(v - (1 << 64) if v >= (1 << 63) else v)
                    return out
                for af, aw, av in parse_fields(v):
                    if af == 1 and aw == 2:
                        attr["name"] = av.decode("utf-8", "replace")
                    elif af == 3 and aw == 0:
                        attr.setdefault("value", av if av < (1 << 63) else av - (1 << 64))
                    elif af == 4 and aw == 5:
                        import struct as _s
                        attr.setdefault("value", _s.unpack("<f", av)[0])
                    elif af == 5 and aw == 2:
                        attr.setdefault("value", av.decode("utf-8", "replace"))
                    elif af == 6:
                        if aw == 2:
                            attr.setdefault("value", _ints(av))
                        else:
                            attr.setdefault("value", []).append(av)
                    elif af == 7 and aw == 2:
                        import struct as _s
                        attr.setdefault("value",
                                        [_s.unpack_from("<f", av, i)[0]
                                         for i in range(0, len(av), 4)])
                    elif af == 10 and aw == 0:
                        attr.setdefault("value", bool(av))
                attrs.append(attr)
        ops.append({"type": otype, "inputs": ins, "outputs": outs,
                    "attrs": attrs})
    return ops


def run_reference(model_dir: Path, x: np.ndarray):
    from paddle.inference import Config, create_predictor
    config = Config(str(model_dir / "inference.pdmodel"),
                    str(model_dir / "inference.pdiparams"))
    config.disable_gpu()
    config.switch_ir_optim(False)
    for name in ("disable_onednn", "disable_mkldnn"):
        try:
            getattr(config, name)()
            break
        except Exception:
            continue
    predictor = create_predictor(config)
    in_name = predictor.get_input_names()[0]
    handle = predictor.get_input_handle(in_name)
    handle.reshape(list(x.shape))
    handle.copy_from_cpu(x)
    predictor.run()
    out_name = predictor.get_output_names()[0]
    return predictor.get_output_handle(out_name).copy_to_cpu()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("out_json", type=Path)
    parser.add_argument("--fixture", type=Path, default=None,
                        help="write (input, output) .npz pair for parity")
    parser.add_argument("--input-shape", default="1,3,48,320")
    parser.add_argument("--seed", type=int, default=20260928)
    args = parser.parse_args()

    ops = op_sequence(args.model_dir / "inference.pdmodel")
    shapes = var_shapes(args.model_dir / "inference.pdmodel")
    ranks = {name: len(dims) for name, dims in shapes.items()}
    # Object envelope: ops plus per-tensor logical ranks and static
    # shapes. A bare array means a legacy export without ranks.
    Path(args.out_json).write_text(
        json.dumps({"ops": ops, "ranks": ranks, "shapes": shapes},
                   indent=1))
    print(f"wrote {args.out_json} ({len(ops)} ops, {len(ranks)} ranks)")

    if args.fixture:
        rng = np.random.default_rng(args.seed)
        shape = [int(v) for v in args.input_shape.split(",")]
        x = rng.standard_normal(shape).astype(np.float32)
        y = run_reference(args.model_dir, x)
        np.savez(str(args.fixture), input=x, reference=y)
        print(f"wrote {args.fixture} input{shape} -> {y.shape}")


if __name__ == "__main__":
    main()
