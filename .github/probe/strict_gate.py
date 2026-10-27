"""隔离 app/wheel 全量门禁；报告写完再非零退出，失败对象不被跳过。"""
import argparse
import collections
import json
from pathlib import Path, PureWindowsPath
import re
import struct
import sys
from strict_coff import Coff, Invalid, Names, lambda_bijection, namespace_bijection, require, sha
import strict_device


def inventory(root, expected):
    root = Path(root)
    files = {p.name: p for p in root.glob("*.obj")}
    require(files.keys() == set(expected) and files, f"对象集合不符: missing={sorted(set(expected) - files.keys())} extra={sorted(files.keys() - set(expected))}")
    return files


def fat_sections(coff):
    return {i: s for i, s in coff.sections.items() if s["name"] == ".nv_fatbin"}


def bound_fatbin(coff, section):
    # 替换容器的物理字节大小之前，必须证明宿主仅引用容器起始地址。
    for symbol in coff.symbols.values():
        if symbol["section"] == section:
            require(symbol["value"] == 0, "宿主符号指向 fatbin 内部偏移")
    for sec in coff.sections.values():
        for offset, target, kind in sec["rel"]:
            sym = coff.symbols[target]
            if sym["section"] == section:
                require(kind in (1, 3, 4), "未知 fatbin 宿主引用类型")
                width = 8 if kind == 1 else 4
                require(int.from_bytes(sec["raw"][offset:offset + width], "little") == 0, "宿主引用 fatbin 带非零 addend")


def source_owner(record):
    """所有权来自实际 -c 输入；仅支持本实验已确认的显式 CUDA 编译命令。"""
    pattern = r'(?:^|\s)-c\s+(?:"([^"\r\n]+\.cu)"|(\S+\.cu))(?=\s|$)'
    sources = []
    for field in ("cmd", "original"):
        matches = re.findall(pattern, record[field])
        require(len(matches) == 1, "实际编译源缺失/歧义")
        sources.append(matches[0][0] or matches[0][1])
    require(sources[0] == sources[1], "launcher 与裸编译源不符")
    owner = PureWindowsPath(sources[0]).name
    require(PureWindowsPath(record["obj"]).name == owner + ".obj", "实际源与对象所有权不符")
    return owner


def identity_tokens(coff):
    pattern = r"<lambda_[0-9a-f]{32}>|\?A0x[0-9a-f]{8}(?=@@)"
    tokens = set()
    for symbol in coff.symbols.values():
        tokens.update(re.findall(pattern, symbol["name"]))
    for sec in coff.sections.values():
        if sec["name"] == ".rdata":
            for text in re.findall(rb"[\x20-\x7e]{6,}\x00", sec["raw"]):
                tokens.update(re.findall(pattern, text.decode("latin1")))
    return tokens


def merge_identity_maps(forward, reverse, mapping):
    # 所有对象共同约束同一身份；稳定 token 也必须作为不可更换锚点。
    for original, target in mapping.items():
        require(forward.setdefault(original, target) == target, "跨对象生成身份映射冲突")
        require(reverse.setdefault(target, original) == original, "跨对象生成身份映射碰撞")


def check_composition(pairs):
    sa, ab, sb = (pairs[label]["identity_map"] for label in ("S1-vs-A", "A-vs-B", "S1-vs-B"))
    require(set(ab) == set(sb) and set(ab.values()) == set(sa), "三快照身份覆盖不符")
    require({token: sa[mid] for token, mid in ab.items()} == sb, "三快照身份映射组合不一致")


def compare_object(tool, pa, pb, work, owner=None):
    a, b = Coff(pa.read_bytes()), Coff(pb.read_bytes())
    namespace_map = namespace_bijection(a, b, owner)
    mapping = lambda_bijection(a, b, Names(owner), Names(owner, namespace_map=namespace_map))
    na, nb = Names(owner), Names(owner, mapping, namespace_map)
    tokens_a, tokens_b = identity_tokens(a), identity_tokens(b)
    changed = dict(mapping, **namespace_map)
    identities = {token: changed.get(token, token) for token in tokens_b}
    require(set(identities.values()) == tokens_a and len(set(identities.values())) == len(identities), "完整生成身份覆盖/一一映射不符")
    x, y = a.canonical(na), b.canonical(nb)
    ignored_a, ignored_b = x.pop("ignored"), y.pop("ignored")
    require([(r[0], r[1], r[2]) for r in ignored_a] == [(r[0], r[1], r[2]) for r in ignored_b],
            "被丢弃校验和 section 结构不同")
    fa, fb = fat_sections(a), fat_sections(b)
    require(len(fa) == len(fb) <= 1, "fatbin section 数量不同/多容器未知")
    device = None
    if fa:
        ia, sa = next(iter(fa.items()))
        ib, sb = next(iter(fb.items()))
        entries_a, entries_b = strict_device.fatbin_entries(sa["raw"]), strict_device.fatbin_entries(sb["raw"])
        if sa["raw"] == sb["raw"]:
            device = dict(equivalent=True, proof="full-fatbin-bytes", entries=entries_a)
        else:
            device = strict_device.verify(tool, sa["raw"], sb["raw"], work, na, nb)
        bound_fatbin(a, ia)
        bound_fatbin(b, ib)
        for model in (x, y):
            key = next(k for k in model["sections"] if json.loads(k)[0] == ".nv_fatbin")
            sec = model["sections"][key]
            # 全载荷已独立核查，不能用 COFF 容器压缩长度代替运行时结构。
            sec["data"] = "device-verified" if device["equivalent"] else "device-failed"
            sec["size"] = "device-container"
            if sec["aux"]:
                sec["aux"][0] = "device-container"
    good = x == y and (device is None or device["equivalent"])
    errors = {}
    if not good:
        errors = dict(host_a=x, host_b=y)
    return dict(equivalent=good, full_bytes_equal=a.data == b.data, raw_sha_a=sha(a.data), raw_sha_b=sha(b.data),
                owner=owner, identity_map=identities, namespace_map=namespace_map, lambda_map=mapping,
                host_a=sha(json.dumps(x, sort_keys=True).encode()), host_b=sha(json.dumps(y, sort_keys=True).encode()),
                device=device, names_a=na.changes, names_b=nb.changes, ignored_a=ignored_a, ignored_b=ignored_b), errors


def main():
    # Windows 重定向输出默认 cp1252；诊断必须可输出全部中文，不能中断全量遍历。
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="strict")
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", required=True)
    parser.add_argument("--snap", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--lane", required=True, choices=("app", "wheel"))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # run 40443832739 的完整 S1/A/B 实物：app 33/33，wheel 58/58；不从旧解析失败数推断覆盖。
    expected_count, expected_fat = (33, 33) if args.lane == "app" else (58, 58)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8-sig"))
    expected = [m["obj"].replace("\\", "__").replace("/", "__").replace(":", "__") for m in manifest]
    report = dict(schema="nvcc-equivalence/4", controls="fully-decoded-physical-lengths",
                  ptx_extraction="exact-or-windows-lf-expansion", lane=args.lane,
                  expected=expected_count, manifest_sha=sha(Path(args.manifest).read_bytes()), pairs={}, equivalent=False)
    try:
        require(len(expected) == len(set(expected)) == expected_count, "命令清单对象数量/映射不符")
        require(all(m["launched"] for m in manifest), "存在未经过 launcher 的对象")
        owners = {name: source_owner(record) for name, record in zip(expected, manifest)}
        files = {s: inventory(Path(args.snap) / s, expected) for s in ("S1", "A", "B")}
        for left, right in (("S1", "A"), ("A", "B"), ("S1", "B")):
            label = left + "-vs-" + right
            pair = report["pairs"][label] = dict(objects={}, errors={}, fatbin=0, identity_map={}, equivalent=False)
            reverse = {}
            for name in sorted(expected):
                print(label, name, flush=True)
                try:
                    work = out / label / name
                    row, errors = compare_object(args.tool, files[left][name], files[right][name], work, owners[name])
                    merge_identity_maps(pair["identity_map"], reverse, row["identity_map"])
                    pair["objects"][name] = row
                    pair["fatbin"] += row["device"] is not None
                    if errors:
                        work.mkdir(parents=True, exist_ok=True)
                        detail = work / "host-differences.json"
                        detail.write_text(json.dumps(errors, indent=1), encoding="utf-8")
                        pair["errors"][name] = dict(file=str(detail.relative_to(out)), sha=sha(detail.read_bytes()))
                        print("OBJECT FAIL", label, name, "完整宿主/设备结构不同；详情", pair["errors"][name]["file"], flush=True)
                    del row, errors
                except Exception as err:
                    pair["objects"][name] = dict(equivalent=False, error=type(err).__name__ + ": " + str(err))
                    print("OBJECT FAIL", label, name, pair["objects"][name]["error"], flush=True)
            pair["equivalent"] = len(pair["objects"]) == expected_count and pair["fatbin"] == expected_fat and not pair["errors"] and all(r["equivalent"] for r in pair["objects"].values())
            print("PAIR", label, "objects", len(pair["objects"]), "/", expected_count,
                  "fatbin", pair["fatbin"], "/", expected_fat,
                  "failed", sum(not row["equivalent"] for row in pair["objects"].values()),
                  "PASS" if pair["equivalent"] else "FAIL", flush=True)
            # 每对比较立即刷新完整报告；远程超时也保留已取得的证据。
            (out / "gate.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
        check_composition(report["pairs"])
        report["equivalent"] = all(p["equivalent"] for p in report["pairs"].values())
    except Exception as err:
        report["fatal"] = type(err).__name__ + ": " + str(err)
        print("GATE FATAL", report["fatal"], flush=True)
    (out / "gate.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("STRICT GATE", "PASS" if report["equivalent"] else "FAIL", flush=True)
    return 0 if report["equivalent"] else 1


if __name__ == "__main__":
    sys.exit(main())
