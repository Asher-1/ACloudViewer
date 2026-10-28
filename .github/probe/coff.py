"""Probe-only object forensics (branch ci/probe-nvcc-sccache, never merged).

Answers one question: when two .cu.obj files built from the same command
differ, WHICH bytes differ — the COFF timestamp, a temp-name string, debug
info, or the host/device code itself?

  digest OUT.json DIR            per-section SHA-256 of every *.obj in DIR
  compare A.json B.json DIR_A DIR_B LABEL OUT.json
                                 section-level diff summary (+ string samples)
  dump CUOBJDUMP DIR [CMP.json [LIMIT]]
                                 normalized `cuobjdump -sass` / `-ptx` texts
                                 (with CMP.json: only the LIMIT smallest
                                 objects whose fatbin section differs)
  textcmp DIR_A DIR_B LABEL OUT.json
                                 SASS/PTX equality, line-multiset diff samples

Memory: one object at a time (largest .cu.obj is tens of MB); the text diff
is a Counter over lines, O(n), never difflib's O(n*m).
"""
import collections
import concurrent.futures
import hashlib
import itertools
import json
import os
import re
import struct
import subprocess
import sys


def sha(b):
    return hashlib.sha256(b).hexdigest()


B64 = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"


def parse_coff(data):
    """Return (timestamp, header-with-zeroed-timestamp, sections, symtab)."""
    if data[:4] == b"\x00\x00\xff\xff" and struct.unpack_from("<H", data, 4)[0] >= 2:
        ts_off, shdr, symsz = 8, 56, 20  # ANON_OBJECT_HEADER_BIGOBJ
        nsec, symptr, nsym = struct.unpack_from("<III", data, 44)
    else:
        ts_off, symsz = 4, 18
        (nsec,) = struct.unpack_from("<H", data, 2)
        symptr, nsym = struct.unpack_from("<II", data, 8)
        shdr = 20 + struct.unpack_from("<H", data, 16)[0]
    (ts,) = struct.unpack_from("<I", data, ts_off)
    str_off = symptr + nsym * symsz
    strtab = b""
    if symptr and str_off + 4 <= len(data):
        (strsz,) = struct.unpack_from("<I", data, str_off)
        if strsz < 4 or str_off + strsz > len(data):
            raise ValueError("invalid COFF string table length")
        strtab = data[str_off:str_off + strsz]

    def name(raw):
        label = raw.rstrip(b"\0")
        if label[:1] != b"/":
            return label.decode("latin1")
        # 长 section 名称有两种偏移编码：/十进制 与 //base64（微软对超出
        # 十进制字段宽度的偏移改用后者）。只解十进制会把 .nv_fatbin 读成
        # 字面 "//ABQDup"，令 is_fatbin 静默少算该对象；无法解码时必须失败，
        # 回退字面名称正是这个漏计的来源。
        if label[:2] == b"//":
            digits = label[2:]
            if not (1 <= len(digits) <= 6 and all(c in B64 for c in digits)):
                raise ValueError("invalid COFF base64 section name: %r" % label)
            off = 0
            for digit in digits:
                off = off * 64 + B64.index(digit)
            if off > 0xffffffff:
                raise ValueError("COFF base64 string offset overflow: %r" % label)
        elif label[1:].isdigit():
            off = int(label[1:])
        else:
            raise ValueError("invalid COFF long section name: %r" % label)
        end = strtab.find(b"\0", off)
        if not 4 <= off < len(strtab) or end < 0:
            raise ValueError("COFF string offset out of range: %r -> %d/%d"
                             % (label, off, len(strtab)))
        return strtab[off:end].decode("latin1")

    secs = []
    for i in range(nsec):
        h = shdr + 40 * i
        size, ptr, rel_ptr = struct.unpack_from("<III", data, h + 16)
        (nrel,) = struct.unpack_from("<H", data, h + 32)
        raw = data[ptr:ptr + size] if ptr else b""
        rel = data[rel_ptr:rel_ptr + 10 * nrel] if rel_ptr else b""
        secs.append((i, name(data[h:h + 8]), raw, rel))
    hdr = bytearray(data[:shdr + 40 * nsec])
    hdr[ts_off:ts_off + 4] = b"\0\0\0\0"
    symtab = data[symptr:str_off + len(strtab)] if symptr else b""
    return ts, bytes(hdr), secs, symtab


def digest(out, root):
    res = {}
    for f in sorted(os.listdir(root)):
        if not f.endswith(".obj"):
            continue
        data = open(os.path.join(root, f), "rb").read()
        ts, hdr, secs, symtab = parse_coff(data)
        res[f] = {
            "size": len(data), "full": sha(data), "ts": ts, "hdr": sha(hdr),
            "symtab": sha(symtab),
            "sections": [[i, n, sha(raw), sha(rel), len(raw)] for i, n, raw, rel in secs],
        }
    json.dump(res, open(out, "w"), indent=1)


def strings(b, n=6):
    return set(m.group().decode("latin1") for m in re.finditer(rb"[\x20-\x7e]{%d,}" % n, b))


def compare(a_json, b_json, dir_a, dir_b, label, out):
    A, B = json.load(open(a_json)), json.load(open(b_json))
    rows, kinds = {}, collections.Counter()
    for f in sorted(set(A) & set(B)):
        a, b = A[f], B[f]
        if a["full"] == b["full"]:
            kinds["identical"] += 1
            rows[f] = "identical"
            continue
        diff = []
        if a["ts"] != b["ts"]:
            diff.append("TimeDateStamp")
        if a["hdr"] != b["hdr"]:
            diff.append("header/section-table")
        if a["symtab"] != b["symtab"]:
            diff.append("symtab/strtab")
        sa = {(s[0], s[1]): s for s in a["sections"]}
        sb = {(s[0], s[1]): s for s in b["sections"]}
        if set(sa) != set(sb):
            diff.append("section-set")
        for k in sorted(set(sa) & set(sb)):
            if sa[k][2] != sb[k][2]:
                diff.append("data:" + k[1])
            if sa[k][3] != sb[k][3]:
                diff.append("reloc:" + k[1])
        uniq = sorted(set(diff))
        kinds[" + ".join(uniq)] += 1
        rows[f] = {"differs_in": uniq}
        pa, pb = os.path.join(dir_a, f), os.path.join(dir_b, f)
        if os.path.exists(pa) and os.path.exists(pb):  # A may live on another runner
            sa_, sb_ = strings(open(pa, "rb").read()), strings(open(pb, "rb").read())
            rows[f].update({"only_a": sorted(sa_ - sb_)[:12], "only_b": sorted(sb_ - sa_)[:12],
                            "n_only_a": len(sa_ - sb_), "n_only_b": len(sb_ - sa_)})
    missing = {"only_in_a": sorted(set(A) - set(B)), "only_in_b": sorted(set(B) - set(A))}
    # The embedded device code: identical fatbin bytes == identical SASS/PTX.
    fat = {f: [s for s in A[f]["sections"] if is_fatbin(s[1])] for f in rows}
    fat_diff = sorted(f for f in rows if rows[f] != "identical"
                      and any("data:" + s[1] in rows[f]["differs_in"] for s in fat[f]))
    n_fat = sum(1 for f in rows if fat[f])
    json.dump({"label": label, "kinds": kinds, "missing": missing, "fatbin_objects": n_fat,
               "fatbin_differs": fat_diff, "objects": rows}, open(out, "w"), indent=1)
    print(f"[{label}] {len(rows)} common objects (+{len(missing['only_in_a'])} only in A, "
          f"+{len(missing['only_in_b'])} only in B): " + "; ".join(f"{v}x {k}" for k, v in kinds.most_common())
          + f"; fatbin bytes differ in {len(fat_diff)}/{n_fat} objects carrying one")


def is_fatbin(name):
    return "fatbin" in name.lower()


def _dump_one(tool, path, mode):
    # Streamed to disk: a full-arch-matrix SASS listing can be hundreds of MB.
    raw, txt = path + "." + mode + ".raw", path + "." + mode + ".txt"
    with open(raw, "wb") as fh:
        p = subprocess.run([tool, "-" + mode, path], stdout=fh, stderr=subprocess.PIPE)
    n = 0
    with open(raw, encoding="utf-8", errors="replace") as src, open(txt, "w", encoding="utf-8") as dst:
        for ln in src:
            if not re.match(r"\s*identifier\s*=", ln):
                dst.write(ln)
                n += 1
    os.remove(raw)
    return path, mode, p.returncode, n, p.stderr.decode("utf-8", "replace").strip()[:200]


def dump(tool, root, cmp_json=None, limit="6"):
    objs = [os.path.join(root, f) for f in sorted(os.listdir(root)) if f.endswith(".obj")]
    if cmp_json:
        c = json.load(open(cmp_json))
        # No fatbin section recognized at all: sample the smallest objects.
        want = set(c["fatbin_differs"]) if c["fatbin_objects"] else set(map(os.path.basename, objs))
        objs = sorted((o for o in objs if os.path.basename(o) in want), key=os.path.getsize)[:int(limit)]
    print(f"[dump] {len(objs)} objects from {root}")
    with concurrent.futures.ThreadPoolExecutor(os.cpu_count() or 4) as ex:
        for path, mode, rc, n, err in ex.map(lambda t: _dump_one(tool, *t),
                                            [(o, m) for o in objs for m in ("sass", "ptx")]):
            if rc != 0 or n < 5:
                print(f"[dump] {os.path.basename(path)} -{mode}: rc={rc} lines={n} {err}")


def textcmp(dir_a, dir_b, label, out):
    res, kinds = {}, collections.Counter()
    for f in sorted(os.listdir(dir_a)):
        m = re.match(r"(.*\.obj)\.(sass|ptx)\.txt$", f)
        if not m or not os.path.exists(os.path.join(dir_b, f)):
            continue
        pa, pb = os.path.join(dir_a, f), os.path.join(dir_b, f)
        mode = m.group(2)
        if os.path.getsize(pa) == 0:
            kinds[mode + ":empty"] += 1
            res[f] = "empty"
            continue
        # Positional streaming compare, O(1) memory: the first differing
        # line pairs are what tells a renamed symbol from changed code.
        n = ndiff = 0
        samples = []
        with open(pa, encoding="utf-8") as fa, open(pb, encoding="utf-8") as fb:
            for la, lb in itertools.zip_longest(fa, fb, fillvalue="<EOF>"):
                n += 1
                if la != lb:
                    ndiff += 1
                    if len(samples) < 12:
                        samples.append([n, la.rstrip()[:240], lb.rstrip()[:240]])
        kinds[mode + (":identical" if ndiff == 0 else ":different")] += 1
        res[f] = {"identical": ndiff == 0, "lines": n, "differing_lines": ndiff, "samples": samples}
    json.dump({"label": label, "kinds": kinds, "files": res}, open(out, "w"), indent=1)
    print(f"[{label}] " + "; ".join(f"{v}x {k}" for k, v in sorted(kinds.items())))


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    {"digest": digest, "compare": compare, "dump": dump, "textcmp": textcmp}[cmd](*args)
