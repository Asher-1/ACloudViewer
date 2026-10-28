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


def elf(data, names, layout_evidence=None):
    require(data[:7] == b"\x7fELF\x02\x01\x01", "非 little-endian ELF64")
    header = struct.unpack("<16sHHIQQQIHHHHHH", blob(data, 0, 64))
    ident, elf_type, machine, version, entry, phoff, shoff, flags, ehsize, phsize, phnum, shsize, shnum, shstr = header
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
    # program header 是加载合同；权限、地址、对齐、覆盖集合、bss 量均不忽略。
    # 但 filesz/memsz 是文件坐标的派生量：nvcc 把编译进程 PID 嵌入匿名 namespace
    # 名称，PID 的十进制位数变化改变 .shstrtab/.strtab 的 size —— 上面已允许该
    # 漂移 —— 后续 section 偏移随之移位，并被 .text 的 128 字节对齐吸收成变长零
    # padding，于是段长变化而段内容一字节未变。只对能逐项证明为「最小零对齐
    # padding」的段归一长度；非零间隙、非最小间隙、物理重叠、被段部分截断的
    # section、首尾未绑定 section 边界一律失败，无 section 覆盖的段（PT_PHDR 及
    # 装载它的 PT_LOAD）仍按原 filesz 比较。O(phnum * shnum)，shnum 为几十量级。
    programs = []
    require(phnum == 0 or phsize == 56, "未知 ELF program header")
    for n in range(phnum):
        ptype, pflags, poff, vaddr, paddr, filesz, memsz, align = struct.unpack("<IIQQQQQQ", blob(data, phoff + n * phsize, phsize))
        stop = poff + filesz
        blob(data, poff, filesz)
        require(ptype != 1 or memsz >= filesz, "PT_LOAD memsz 小于 filesz")
        inside, covered = [], []
        for i, s in enumerate(sections):
            if s[1] == 8 or not s[5]:
                continue
            start, end = s[4], s[4] + s[5]
            if not filesz or end <= poff or start >= stop:
                continue
            require(poff <= start and end <= stop, "program 部分截断 section")
            inside.append((start, end, s[8], i))
            covered.append(secnames[i])
        length, memory = filesz, memsz
        # 只许可实测的 CUDA ET_EXEC=2、零地址 RX PT_LOAD 且无 BSS 的代码/常量段。
        # 未知 ELF 类型、RW/R 段、非零地址、无覆盖和含未知 section 均保留原大小，
        # 不把其它加载格式中 padding 的内存含义推断成相同。
        eligible = (elf_type == 2 and ptype == 1 and pflags == 5 and
                    vaddr == paddr == 0 and filesz == memsz and inside and
                    all(sections[i][1] == 1 and sections[i][3] == 0 and
                        secnames[i].startswith(('.text.', '.nv.constant'))
                        for _, _, _, i in inside))
        sequence = None
        if eligible:
            # O(s log s)：排序是逐项证明物理区间覆盖的必要步骤，不能靠总字节数。
            inside.sort()
            require(inside[0][0] == poff and inside[-1][1] == stop,
                    "program 首尾未绑定 section 边界")
            content, cursor, gaps = 0, poff, []
            for start, end, salign, i in inside:
                require(start >= cursor, "program 内 section 物理重叠")
                require(salign > 0 and not salign & (salign - 1) and start % salign == 0,
                        "program 内 section 对齐无效")
                if start > cursor:
                    require(not any(data[cursor:start]), "program 内间隙非零")
                    require(start == (cursor + salign - 1) // salign * salign,
                            "program 内间隙非最小 section 对齐")
                    gaps.append(dict(start=cursor, end=start, section=secnames[i], align=salign))
                content += end - start
                cursor = end
            require(cursor == stop and filesz == content + sum(g['end'] - g['start'] for g in gaps),
                    "program 内容长度核对失败")
            length = memory = content
            # 归一长度后仍保留物理 section 顺序，防止同总长的布局被误认为相同。
            sequence = [secnames[i] for _, _, _, i in inside]
            if layout_evidence is not None:
                layout_evidence.append(dict(program=n, elf_type=elf_type, flags=pflags,
                    vaddr=vaddr, paddr=paddr, offset=poff, filesz=filesz, memsz=memsz,
                    section_bytes=content, removed_padding=filesz-content,
                    section_sequence=sequence, gaps=gaps))
        programs.append([ptype, pflags, vaddr, paddr, length, memory, align, sorted(covered), sequence])
    return dict(header=[ident.hex(), elf_type, machine, version, entry, flags], programs=programs, section_order=secnames, sections=out)


def decompress_payload(data, expected):
    """有界 LZ4 block 解码，O(输入＋输出)；不信任声明长度，不预分配该长度。"""
    output = bytearray()
    pos = 0

    def extended(length):
        nonlocal pos
        if length == 15:
            while True:
                require(pos < len(data), "压缩长度扩展被截断")
                extra = data[pos]
                pos += 1
                length += extra
                if extra != 255:
                    break
        return length

    while pos < len(data):
        token = data[pos]
        pos += 1
        literal = extended(token >> 4)
        require(pos + literal <= len(data) and len(output) + literal <= expected,
                "压缩 literal 越界/超过声明输出")
        output.extend(data[pos:pos + literal])
        pos += literal
        if pos == len(data):
            break
        require(pos + 2 <= len(data), "压缩引用被截断")
        back = int.from_bytes(data[pos:pos + 2], "little")
        pos += 2
        count = extended(token & 15) + 4
        require(0 < back <= len(output) and len(output) + count <= expected,
                "压缩引用越界/超过声明输出")
        # 重叠引用重复已有周期，按块构造，避免逐字节 Python 循环。
        pattern = output[len(output) - back:len(output) - back + min(back, count)]
        output.extend((pattern * ((count + len(pattern) - 1) // len(pattern)))[:count])
    require(pos == len(data) and len(output) == expected and expected > 0,
            "压缩输入未完全消费/解压长度不符")
    return bytes(output)


def decoded_entries(raw):
    """逐条目验证物理编码并返回完整解码载荷；未知格式失败。"""
    magic, version, header, size = struct.unpack("<IHHQ", blob(raw, 0, 16))
    require(magic == 0xba55ed50 and version == 1 and header == 16 and header + size == len(raw),
            "未知 fatbin 容器或未核查尾部")
    pos = header
    while pos < len(raw):
        kind, version, hsize, payload = struct.unpack("<HHIQ", blob(raw, pos, 16))
        require(version == 0x101 and payload > 0 and
                ((kind == 2 and hsize == 64) or (kind == 1 and hsize in (80, 104))),
                "未知 fatbin entry")
        entry = blob(raw, pos, hsize + payload)
        compressed = struct.unpack_from("<I", entry, 16)[0]
        flags = struct.unpack_from("<Q", entry, 40)[0]
        declared = struct.unpack_from("<Q", entry, 56)[0]
        require(flags in (0x41, 0x2041), "未验证 fatbin flags")
        encoded = entry[hsize:]
        if flags & 0x2000:
            require(0 < compressed <= payload and payload == (compressed + 7) // 8 * 8,
                    "fatbin 压缩长度/对齐错误")
            require(not any(encoded[compressed:]), "fatbin 压缩 padding 非零")
            decoded = decompress_payload(encoded[:compressed], declared)
        else:
            require(compressed == 0 and declared == 0, "未压缩 fatbin 存在压缩长度")
            decoded = encoded
        yield dict(kind=kind, arch=struct.unpack_from("<I", entry, 28)[0],
                   version=version, hsize=hsize, header=entry[:hsize],
                   compressed_size=compressed, payload_size=payload,
                   declared_size=declared, decoded=decoded)
        pos += hsize + payload
    require(pos == len(raw) and pos > header, "fatbin 覆盖失败")


def fatbin_entries(raw):
    entries = []
    for item in decoded_entries(raw):
        # 完整消费压缩数据、padding 和精确解压长度后，物理长度不作为语义控制字段。
        # [56:64] 是已验证的解码长度，生成名称位数可令它变化；全部载荷仍须独立比较。
        # flags、保留字段和 [64:hsize] 附加控制内容全部保留；未知格式仍失败。
        entries.append((item["kind"], item["arch"], item["version"], item["hsize"],
                        sha(item["header"][20:56] + item["header"][64:])))
    require(entries, "fatbin 无载荷")
    return entries


def check_extracted(raw, manifest, directory):
    """绑定完整载荷；PTX 只允许单个终止 NUL 省略及确定的 Windows LF 扩展。"""
    files = {(m["kind"], m["arch"]): m for m in manifest}
    require(len(files) == len(manifest), "同架构载荷映射歧义")
    seen = set()
    for item in decoded_entries(raw):
        key = (item["kind"], item["arch"])
        require(key in files and key not in seen, "物理解码与提取条目覆盖不符")
        seen.add(key)
        extracted = (Path(directory) / files[key]["file"]).read_bytes()
        if item["kind"] == 1:
            decoded = item["decoded"]
            require(decoded.endswith(b"\0") and b"\0" not in decoded[:-1] and b"\0" not in extracted,
                    "PTX 单个终止 NUL 合同不符")
            exact = decoded[:-1] == extracted
            # Windows cuobjdump 的文本模式对每个 LF 加一个 CR，包括原有 CRLF。
            # 比较完整正向转换；不任意 strip，不允许部分转换或其它内容变化。
            windows_text = not exact and decoded[:-1].replace(b"\n", b"\r\n") == extracted
            require(exact or windows_text, "PTX 提取省略了非终止内容/物理解码不符")
            files[key]["windows_text"] = windows_text
        else:
            require(item["decoded"] == extracted, "cubin 提取字节与完整物理解码不符")
        files[key]["decoded_sha"] = sha(item["decoded"])
        files[key]["decoded_size"] = len(item["decoded"])
    require(seen == files.keys(), "存在未验证提取条目")


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
    check_extracted(raw, manifest, directory)
    return manifest, expected


def ptx(path, names, windows_text=False):
    digest = hashlib.sha256()
    lines = 0
    targets = []
    # 仅按 LF 分行，避免 universal-newline 把 Windows 扩展产生的 CRCRLF 拆成两行。
    with Path(path).open(encoding="utf-8", errors="strict", newline="\n") as src:
        for line in src:
            if windows_text:
                line = line.replace("\r\n", "\n")
            line = line.replace("\r\n", "\n")
            require("\r" not in line, "未验证的 PTX 单独 CR")
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
        layout_a, layout_b = [], []
        x = elf(pa.read_bytes(), na, layout_a) if a["kind"] == 2 else ptx(pa, na, a["windows_text"])
        y = elf(pb.read_bytes(), nb, layout_b) if b["kind"] == 2 else ptx(pb, nb, b["windows_text"])
        good = x == y
        rows.append(dict(index=index, kind=a["kind"], arch=a["arch"], identical=good, canonical_a=sha(json.dumps(x, sort_keys=True).encode()),
                         canonical_b=sha(json.dumps(y, sort_keys=True).encode()),
                         decoded_sha_a=a["decoded_sha"], decoded_sha_b=b["decoded_sha"],
                         decoded_size_a=a["decoded_size"], decoded_size_b=b["decoded_size"],
                         windows_text_a=a.get("windows_text"), windows_text_b=b.get("windows_text"),
                         load_layout_a=layout_a, load_layout_b=layout_b))
        if not good:
            errors.append(dict(index=index, a=x, b=y))
    (root / "differences.json").write_text(json.dumps(errors, indent=1), encoding="utf-8")
    return dict(equivalent=not errors, entries=rows, names_a=na.changes, names_b=nb.changes)
