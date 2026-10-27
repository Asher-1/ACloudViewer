"""全量 fatbin 验证：全部 ELF 指令/常量/重定位和全部 PTX；无抽样。

以 cubin 指令字节代替耗时的 SASS 文本生成；指令字节相同是更强的证据。
内存 O(单个 cubin)；PTX 按行哈希。排序 O(n log n) 用于消除表的物理顺序。
"""
import collections
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess
from strict_coff import Names, blob, cstring, require, sha


def elf(data, names):
    require(data[:7] == b"\x7fELF\x02\x01\x01", "非 little-endian ELF64")
    header = struct.unpack("<16sHHIQQQIHHHHHH", blob(data, 0, 64))
    ident, type_, machine, version, entry, phoff, shoff, flags, ehsize, phsize, phnum, shsize, shnum, shstr = header
    require(machine == 190 and version == 126 and ehsize == 64 and shsize == 64 and shnum > 0,
            "未知 CUDA ELF header")
    require(shstr < shnum, "无效 ELF shstr index")
    sections = [struct.unpack("<IIQQQQIIQQ", blob(data, shoff + i * 64, 64)) for i in range(shnum)]
    stringdata = blob(data, sections[shstr][4], sections[shstr][5])
    secnames = [names(cstring(stringdata, s[0])) for s in sections]
    require(len(set(secnames)) == len(secnames), "ELF section 名称碰撞")
    symbols = {}
    symrows = {}
    for i, s in enumerate(sections):
        if s[1] not in (2, 11):
            continue
        require(s[9] == 24 and s[5] % 24 == 0 and s[6] < shnum and sections[s[6]][1] == 3,
                "无效 ELF symbol table")
        strings = blob(data, sections[s[6]][4], sections[s[6]][5])
        rows = []
        for n in range(s[5] // 24):
            name, info, other, section, value, size = struct.unpack("<IBBHQQ", blob(data, s[4] + n * 24, 24))
            require(section < shnum or section >= 0xff00, "ELF symbol section 越界")
            row = [names(cstring(strings, name)), info, other,
                   secnames[section] if section < shnum else section, value, size]
            rows.append(row)
        # 同名局部符号允许出现，但完整结构和出现次数必须完全相同。
        symbols[i] = rows
        # .nv.info 等属性包含 NVIDIA 私有 symbol index；未解析的索引禁止重排。
        symrows[i] = rows
    out = {}
    for i, s in enumerate(sections):
        name, type_, sflags, address, offset, size, link, info, align, entsize = s
        require(link < shnum, "ELF section link 越界")
        content = blob(data, offset, size) if type_ != 8 else b""
        if type_ in (2, 11):
            payload = symrows[i]
        elif type_ == 3:
            require(not content or content.endswith(b"\0"), "ELF 字符串表未终止")
            payload = [names(t.decode("latin1")) for t in content.split(b"\0")]
        elif type_ in (4, 9):
            width = 24 if type_ == 4 else 16
            require(entsize == width and size % width == 0 and link in symbols and info < shnum,
                    "无效 ELF relocation table")
            rows = []
            for n in range(size // width):
                values = struct.unpack("<QQq" if type_ == 4 else "<QQ", blob(data, offset + n * width, width))
                roff, rinfo = values[:2]
                target, kind = rinfo >> 32, rinfo & 0xffffffff
                require(target < len(symbols[link]), "ELF relocation symbol 越界")
                rows.append([roff, kind, symbols[link][target], *values[2:]])
            payload = sorted(rows, key=lambda x: json.dumps(x))
        else:
            # 包括 .text.*, .nv.constant*, .nv.info*, unwind 和未知属性，逐字节核查。
            payload = sha(content)
        out[secnames[i]] = [type_, sflags, address, secnames[link],
                           secnames[info] if type_ in (4, 9) else info, align, entsize,
                           size if type_ not in (2, 3, 4, 9, 11) else None, payload]
    # program header 是加载合同；不忽略实际地址/大小/权限，仅文件偏移可漂移。
    programs = []
    require(phnum == 0 or phsize == 56, "未知 ELF program header")
    for n in range(phnum):
        ptype, pflags, poff, vaddr, paddr, filesz, memsz, align = struct.unpack("<IIQQQQQQ", blob(data, phoff + n * phsize, phsize))
        covered = [secnames[i] for i, s in enumerate(sections) if s[1] != 8 and s[5] and poff <= s[4] and s[4] + s[5] <= poff + filesz]
        programs.append([ptype, pflags, vaddr, paddr, filesz, memsz, align, sorted(covered)])
    return dict(header=[ident.hex(), type_, machine, version, entry, flags], programs=programs, section_order=secnames, sections=out)


def fatbin_entries(raw):
    magic, version, header, size = struct.unpack("<IHHQ", blob(raw, 0, 16))
    require(magic == 0xba55ed50 and version == 1 and header == 16 and header + size == len(raw),
            "未知 fatbin 容器或未核查尾部")
    pos = header
    entries = []
    while pos < len(raw):
        kind, version, hsize, payload = struct.unpack("<HHIQ", blob(raw, pos, 16))
        require(kind in (1, 2) and hsize >= 32 and payload > 0, "未知 fatbin entry")
        entry = blob(raw, pos, hsize + payload)
        # 除压缩后的 payload 大小，全部控制字段保留供配对；identifier 单独检查。
        architecture = struct.unpack_from("<I", entry, 28)[0]
        entries.append((kind, architecture, version, hsize, sha(entry[16:hsize])))
        pos += hsize + payload
    require(pos == len(raw) and entries, "fatbin 覆盖失败")
    return entries


def run(tool, args, cwd):
    proc = subprocess.run([str(tool), *args], cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    require(proc.returncode == 0, "cuobjdump 失败: " + proc.stderr.decode("utf-8", "replace")[:1000])
    return proc.stdout.decode("utf-8", "strict")


def extract(tool, raw, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    source = directory / "input.fatbin"
    source.write_bytes(raw)
    expected = fatbin_entries(raw)
    manifest = []
    for mode, kind, suffix in (("elf", 2, ".cubin"), ("ptx", 1, ".ptx")):
        listing = run(tool, ["-all", "-l" + mode, str(source.resolve())], directory)
        listed = re.findall(r"^\s*" + mode.upper() + r" file\s+\d+:\s*(\S+)\s*$", listing, re.MULTILINE)
        want = sum(e[0] == kind for e in expected)
        require(len(listed) == want, f"fatbin/{mode} coverage: listed={len(listed)} expected={want}")
        if not want:
            continue
        run(tool, ["-all", "-x" + mode, "all", str(source.resolve())], directory)
        actual = {p.name for p in directory.glob("*" + suffix)}
        require(actual == set(listed) and len(listed) == len(set(listed)), "缺失/重复提取文件")
        for name in listed:
            require(Path(name).name == name, "不安全的提取文件名")
            arch = re.search(r"\.sm_(\d+)\.", name)
            require(arch is not None, "未知提取架构名: " + name)
            manifest.append(dict(kind=kind, arch=int(arch[1]), file=name))
    require(collections.Counter((m["kind"], m["arch"]) for m in manifest) == collections.Counter((e[0], e[1]) for e in expected),
            "fatbin 架构覆盖不完整")
    return manifest, expected


def ptx(path, names):
    digest = hashlib.sha256()
    lines = 0
    targets = []
    with Path(path).open(encoding="utf-8", errors="strict") as src:
        for line in src:
            lines += 1
            line = names(line)
            digest.update(line.encode("utf-8"))
            if re.match(r"\s*\.target\s+", line):
                targets.append(line.rstrip())
    require(lines > 5 and targets, "空/无 target 的 PTX")
    return dict(sha=digest.hexdigest(), lines=lines, targets=targets)


def verify(tool, raw_a, raw_b, work, na=None, nb=None):
    root = Path(work)
    root.mkdir(parents=True, exist_ok=False)
    ma, ea = extract(tool, raw_a, root / "a")
    mb, eb = extract(tool, raw_b, root / "b")
    require(ea == eb, "fatbin 控制字段/条目架构不同")
    require([(m["kind"], m["arch"]) for m in ma] == [(m["kind"], m["arch"]) for m in mb], "提取条目顺序不同")
    na, nb = na or Names(), nb or Names()
    rows = []
    errors = []
    for index, (a, b) in enumerate(zip(ma, mb)):
        pa, pb = root / "a" / a["file"], root / "b" / b["file"]
        x = elf(pa.read_bytes(), na) if a["kind"] == 2 else ptx(pa, na)
        y = elf(pb.read_bytes(), nb) if b["kind"] == 2 else ptx(pb, nb)
        good = x == y
        rows.append(dict(index=index, kind=a["kind"], arch=a["arch"], identical=good, canonical_a=sha(json.dumps(x, sort_keys=True).encode()),
                         canonical_b=sha(json.dumps(y, sort_keys=True).encode())))
        if not good:
            errors.append(dict(index=index, a=x, b=y))
    (root / "differences.json").write_text(json.dumps(errors, indent=1), encoding="utf-8")
    return dict(equivalent=not errors, entries=rows, names_a=na.changes, names_b=nb.changes)
