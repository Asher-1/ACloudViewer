#!/usr/bin/env python3
"""Same-source GLB parity check: bake the *upstream* sidecar mesh (t2mesh)
through ACloudViewer's libAICore and compare against the upstream CLI GLB.

Isolates the data-generation layer from the display layer:
  1. parse e2e_out/<name>.t2mesh  (T2MESH03: nv, nt, verts, normals, pbr, tris)
  2. aicore_trellis_bake_glb(verts, tris, pbr, 2048, keep-tiny) via libAICore
  3. compare the two GLBs: glTF material JSON + decoded PNG texture stats

Usage:
  python3 core/AICore/src/tasks/trellis/tools/trellis_glb_parity_check.py \
      --sidecar /path/cuda_q8.t2mesh --upstream /path/cuda_q8.glb \
      [--lib build_app/bin/libAICore.so] [--out /tmp/acv_parity.glb]
"""

import argparse
import ctypes
import io
import json
import os
import struct
import sys
import tempfile


def parse_t2mesh(path):
    data = open(path, "rb").read()
    magic = data[:8]
    if magic not in (b"T2MESH03", b"T2MESH01"):
        raise SystemExit(f"{path}: unexpected magic {magic!r}")
    nv, nt = struct.unpack_from("<II", data, 8)
    off = 16
    f32 = lambda n: struct.unpack_from(f"<{n}f", data, off)
    verts = f32(nv * 3)
    off += nv * 12
    normals = f32(nv * 3)
    off += nv * 12
    pbr = None
    if magic == b"T2MESH03":
        pbr = f32(nv * 6)
        off += nv * 24
    tris = struct.unpack_from(f"<{nt * 3}i", data, off)
    print(f"sidecar: magic={magic.decode()} nv={nv} nt={nt} pbr={'yes' if pbr else 'no'}")
    return verts, normals, tris, pbr, nv, nt


def bake_with_libaicore(lib_path, verts, tris, pbr, nv, nt, debug=False):
    lib = ctypes.CDLL(lib_path)
    _P = ctypes.c_void_p
    if debug:
        try:
            lib.aicore_set_log_level.argtypes = [ctypes.c_int]
            lib.aicore_set_log_level(0)  # DEBUG: unwrap path + atlas stats
        except AttributeError:
            print("(aicore_set_log_level not exported; logs stay at default)")
    F32A = ctypes.c_float * len(verts)
    I32A = ctypes.c_int * len(tris)
    FP = ctypes.POINTER(ctypes.c_float)
    lib.aicore_trellis_bake_glb.argtypes = [
        F32A, ctypes.c_int, I32A, ctypes.c_int, FP, ctypes.c_int,
        ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_char_p,
        ctypes.c_int]
    lib.aicore_trellis_bake_glb.restype = _P
    lib.aicore_trellis_free_buffer.argtypes = [_P]
    pbr_arg = (ctypes.c_float * len(pbr))(*pbr) if pbr else (ctypes.c_float * 1)()
    out_len = ctypes.c_int(0)
    err = ctypes.create_string_buffer(512)
    glb = lib.aicore_trellis_bake_glb(
        (F32A(*verts)), nv, (I32A(*tris)), nt, pbr_arg if pbr else None,
        2048, 0, ctypes.byref(out_len), err, 512)
    if not glb:
        raise SystemExit(f"bake failed: {err.value.decode() or 'no message'}")
    raw = ctypes.string_at(glb, out_len.value)
    lib.aicore_trellis_free_buffer(glb)
    print(f"ACV bake: {out_len.value} bytes")
    return raw


def glb_chunks(data):
    magic, ver, length = struct.unpack("<III", data[:12])
    assert magic == 0x46546C67, "not a GLB"
    off = 12
    clen, ctype = struct.unpack("<II", data[off:off + 8])
    js = json.loads(data[off + 8:off + 8 + clen])
    off += 8 + clen
    bin_chunk = b""
    if off < length:
        blen, btype = struct.unpack("<II", data[off:off + 8])
        bin_chunk = data[off + 8:off + 8 + blen]
    return js, bin_chunk


def png_stats(png_bytes):
    from PIL import Image
    img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    px = img.tobytes()
    n = len(px) // 4
    sums = [0, 0, 0, 0]
    for ch in range(4):
        sums[ch] = sum(px[ch::4]) / n
    return img.size, sums


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sidecar", required=True)
    ap.add_argument("--upstream", required=True)
    ap.add_argument("--lib", default=None)
    ap.add_argument("--out", default=os.path.join(tempfile.gettempdir(), "acv_parity.glb"))
    ap.add_argument("--debug", action="store_true",
                    help="enable aicore DEBUG logs (unwrap path trace)")
    args = ap.parse_args()

    verts, normals, tris, pbr, nv, nt = parse_t2mesh(args.sidecar)
    lib = args.lib
    if not lib:
        cand = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                            "..", "..", "..",
                            "build_app/bin/libAICore.so")
        lib = os.path.abspath(cand)
    acv = bake_with_libaicore(lib, verts, tris, pbr, nv, nt,
                              debug=args.debug)
    open(args.out, "wb").write(acv)

    up_data = open(args.upstream, "rb").read()
    js_up, bin_up = glb_chunks(up_data)
    js_acv, bin_acv = glb_chunks(acv)

    print("\n== glTF material JSON ==")
    mat_up = js_up.get("materials", [])
    mat_acv = js_acv.get("materials", [])
    same_mat = json.dumps(mat_up, sort_keys=True) == json.dumps(mat_acv, sort_keys=True)
    print("materials identical:", same_mat)
    if not same_mat:
        print(" upstream :", json.dumps(mat_up))
        print(" acv      :", json.dumps(mat_acv))

    print("\n== texture stats ==")
    from PIL import Image

    def images_of(js, bindata):
        out = []
        for img in js.get("images", []):
            bv = img.get("bufferView")
            view = js["bufferViews"][bv]
            off0 = view.get("byteOffset", 0)
            png = bindata[off0:off0 + view["byteLength"]]
            out.append(png)
        return out

    ups = images_of(js_up, bin_up)
    acs = images_of(js_acv, bin_acv)
    print(f"upstream images: {len(ups)}, acv images: {len(acs)}")
    for i, (u, a) in enumerate(zip(ups, acs)):
        (wu, hu), su = png_stats(u)
        (wa, ha), sa = png_stats(a)
        print(f" image[{i}]: upstream {wu}x{hu} rgba-mean="
              f"({su[0]:.1f},{su[1]:.1f},{su[2]:.1f},{su[3]:.1f})  |  "
              f"acv {wa}x{ha} rgba-mean=({sa[0]:.1f},{sa[1]:.1f},{sa[2]:.1f},{sa[3]:.1f})")
        identical = (u == a)
        print(f"           bytes identical: {identical}")
    print(f"\nACV GLB written to {args.out}")
    print("PASS criterion: identical material JSON + same-size textures with")
    print("matching per-channel means (xatlas UV non-determinism tolerates")
    print("small layout drift, so pixel-exact equality is not required).")


if __name__ == "__main__":
    sys.exit(main())
