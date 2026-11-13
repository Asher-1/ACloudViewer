#!/usr/bin/env python3
"""Dependency-free Paddle inference format parser for the OCR converter.

The controlled conversion host may ship a paddle runtime whose Python
bindings are unusable, so this module reads the two inference files
directly:

* ``*.pdiparams`` — a stream of serialized LodTensors (uint32 version,
  uint64 LoD length + bytes, uint32 tensor version, uint32 descriptor
  length, ``VarType.TensorDesc`` protobuf, raw little-endian float data).
* ``*.pdmodel`` — a ``ProgramDesc`` protobuf. Only the subset needed for
  weight extraction is decoded: BlockDesc(idx/parent/vars), the
  map<string, VarDesc> entries, VarDesc(name/type/persistable) and the
  LOD_TENSOR VarType branch carrying dtype and dims.
* ``inference.json`` — a Paddle 3.x PIR program (``base_code.magic ==
  "pir"``). Persisted weights appear as ``"#":"p"`` ops whose attribute
  list ends with the parameter name and whose output ``t_dtensor`` carries
  the f32 element type and the shape.

Parameter order in ``*.pdiparams`` matches the program's parameter order,
so the loader zips the parsed name list with the parsed tensor stream and
verifies every shape. A shape mismatch aborts instead of guessing, keeping
the conversion auditable against the source graph.
"""

import json
import struct
from pathlib import Path

import numpy as np


def _read_varint(buf: bytes, pos: int):
    result = 0
    shift = 0
    while True:
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7


def _signed(value: int) -> int:
    return value - (1 << 64) if value >= (1 << 63) else value


def parse_fields(buf: bytes):
    """Yield (field_number, wire_type, payload) over a protobuf buffer."""
    pos = 0
    size = len(buf)
    while pos < size:
        key, pos = _read_varint(buf, pos)
        field, wire = key >> 3, key & 7
        if wire == 0:
            value, pos = _read_varint(buf, pos)
            yield field, wire, _signed(value)
        elif wire == 2:
            length, pos = _read_varint(buf, pos)
            yield field, wire, buf[pos:pos + length]
            pos += length
        elif wire == 5:
            yield field, wire, buf[pos:pos + 4]
            pos += 4
        elif wire == 1:
            yield field, wire, buf[pos:pos + 8]
            pos += 8
        else:
            raise ValueError(f"unsupported wire type {wire} at {pos}")


def _parse_tensor_desc(buf: bytes):
    """VarType.TensorDesc: field 1 = data_type varint, field 2 = dims."""
    dtype = None
    dims = []
    for field, wire, value in parse_fields(buf):
        if field == 1 and wire == 0:
            dtype = value
        elif field == 2:
            if wire == 2:  # packed repeated int64
                pos = 0
                while pos < len(value):
                    dim, pos = _read_varint(value, pos)
                    dims.append(_signed(dim))
            else:
                dims.append(_signed(value))
    return dtype, dims


def parse_pdmodel_vars(path: Path):
    """Return [(name, dims)] for every LOD_TENSOR var, in insertion order.

    Paddle 3.x inference programs do not serialize a persistable flag, so
    the caller separates parameters from activations by sequence-matching
    against the param stream instead of trusting a var attribute.
    """
    data = path.read_bytes()
    variables = []
    for field, wire, value in parse_fields(data):
        if field != 1 or wire != 2:
            continue  # ProgramDesc.blocks
        for bfield, bwire, bvalue in parse_fields(value):
            if bfield != 3 or bwire != 2:
                continue  # BlockDesc.vars map entries
            # Map entry layout: field 1 = variable name, field 2 = the
            # VarDesc message (type enum in field 1, tensor descriptor in
            # field 3), field 3 = persistable flag (1 for parameters).
            name = None
            persistable = None
            var_type = None
            dtype = None
            dims = []
            for efield, ewire, evalue in parse_fields(bvalue):
                if efield == 1 and ewire == 2:
                    name = evalue.decode("utf-8")
                elif efield == 2 and ewire == 2:
                    for vfield, vwire, vvalue in parse_fields(evalue):
                        if vfield == 1 and vwire == 0:
                            var_type = vvalue
                        elif vfield == 3 and vwire == 2:
                            # Field 3 wraps the LOD_TENSOR descriptor; its
                            # field 1 is the TensorDesc (08 05 = FP32,
                            # 10 .. = dims).
                            for wfield, wwire, wvalue in parse_fields(vvalue):
                                if wfield == 1 and wwire == 2:
                                    dtype, dims = _parse_tensor_desc(wvalue)
                                    break
                elif efield == 3 and ewire == 0:
                    persistable = bool(evalue)
            if not name or persistable is not True:
                continue
            if var_type not in (None, 7) or dtype not in (None, 5):
                continue
            variables.append((name, tuple(dims)))
    return variables


def parse_pir_json_vars(path: Path):
    """Return [(name, dims)] for every persisted f32 var in a PIR program.

    Mirrors parse_pdmodel_vars for the JSON container PaddleX 3.x ships
    instead of *.pdmodel. Only ``t_f32`` parameter ops are collected; the
    loader binds them to the param stream by shape (trying insertion and
    lexicographic order) exactly as for the protobuf path, so a wrong set
    aborts rather than guessing.
    """
    root = json.loads(path.read_text(encoding="utf-8"))
    if root.get("base_code", {}).get("magic") != "pir":
        raise ValueError(f"{path} is not a PIR program")
    variables = []
    for region in root["program"]["regions"]:
        for block in region["blocks"]:
            for op in block["ops"]:
                if op.get("#") != "p":
                    continue
                name = next((a for a in reversed(op.get("A", []))
                             if isinstance(a, str)), None)
                tensor = op.get("O", {}).get("TT")
                if not name or not tensor or \
                        tensor.get("#") != "0.t_dtensor":
                    continue
                desc = tensor["D"]
                if desc[0].get("#") != "0.t_f32":
                    continue
                variables.append((name, tuple(int(d) for d in desc[1])))
    return variables


def parse_pdparams(path: Path):
    """Return [ndarray] in file order (float32, reshaped)."""
    data = path.read_bytes()
    pos = 0
    size = len(data)
    tensors = []
    while pos < size:
        _version, = struct.unpack_from("<I", data, pos)
        pos += 4
        lod_len, = struct.unpack_from("<Q", data, pos)
        pos += 8 + lod_len
        _tensor_version, = struct.unpack_from("<I", data, pos)
        pos += 4
        desc_len, = struct.unpack_from("<I", data, pos)
        pos += 4
        dtype, dims = _parse_tensor_desc(data[pos:pos + desc_len])
        pos += desc_len
        if dtype not in (None, 5):
            raise ValueError(f"non-fp32 param tensor dtype={dtype}")
        count = int(np.prod(dims)) if dims else 0
        raw = data[pos:pos + count * 4]
        if len(raw) != count * 4:
            raise ValueError("truncated param data stream")
        pos += count * 4
        tensors.append(
            np.frombuffer(raw, dtype="<f4").reshape(tuple(dims)).copy())
    return tensors


def _match_stream(variables, streamed):
    """Greedily bind the param stream to an ordered variable list.

    Persistable fetch targets (`save_infer_model/scale_*`) are written to
    the program but not to the param stream; skipping them is the only
    tolerated divergence.
    """
    named = []
    cursor = 0
    for tensor in streamed:
        shape = tuple(tensor.shape)
        while cursor < len(variables):
            name, dims = variables[cursor]
            cursor += 1
            if dims == shape:
                named.append((name, tensor))
                break
            if name.startswith("save_infer_model/scale_") or ".tmp_" in name:
                continue
            raise ValueError(
                f"param stream shape {shape} does not match program "
                f"variable {name!r} {dims} at stream index {len(named)}")
        else:
            raise ValueError(f"param stream shape {shape} exhausted the "
                             f"program variable list at stream index "
                             f"{len(named)}")
    return named


def load_ocr_tensors(model_dir: Path):
    """Return ([(name, ndarray)], feed_names, fetch_names).

    Parameters are recovered by greedily matching the param stream's shape
    sequence inside the ordered program variable list. This mirrors how
    Paddle writes ``*.pdiparams``: the stream follows
    ``block.all_parameters()``, i.e. the parameter subsequence of the
    variable insertion order. Any mismatch aborts the conversion instead
    of guessing.
    """
    model_dir = Path(model_dir)
    pdmodel = model_dir / "inference.pdmodel"
    pdiparams = model_dir / "inference.pdiparams"
    pir_json = model_dir / "inference.json"
    if not pdmodel.exists():
        pdmodels = sorted(model_dir.glob("*.pdmodel"))
        if pdmodels:
            pdmodel = pdmodels[0]
        elif pir_json.exists():
            pdmodel = None  # PaddleX 3.x JSON program, no protobuf
        else:
            raise SystemExit(f"no *.pdmodel or inference.json under "
                             f"{model_dir}")
    if not pdiparams.exists():
        pdiparams = sorted(model_dir.glob("*.pdiparams"))[0]

    if pdmodel is not None:
        variables = parse_pdmodel_vars(pdmodel)
    else:
        variables = parse_pir_json_vars(pir_json)
    streamed = parse_pdparams(pdiparams)

    # Older exports stream parameters in creation order; Paddle 3.x writes
    # them in lexicographic name order (verified against
    # model.pdiparams.info). Try both orders and keep the one whose shape
    # sequence matches the stream without guessing.
    orders = [variables, sorted(variables, key=lambda kv: kv[0])]
    named = None
    last_error = None
    for order in orders:
        try:
            named = _match_stream(order, streamed)
            break
        except ValueError as exc:
            last_error = exc
    if named is None:
        raise ValueError(str(last_error))

    for name, _ in named:
        if ".tmp_" in name or name.startswith(("feed_", "fetch_")) or \
                name.startswith("save_infer_model/scale_"):
            raise ValueError(f"activation variable {name!r} matched the "
                             "parameter stream; alignment is invalid")
    fetch_names = [name for name, _ in named
                   if name.startswith("save_infer_model/scale_")]
    return named, [], fetch_names
