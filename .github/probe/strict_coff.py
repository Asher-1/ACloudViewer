"""隔离实验的 COFF 全量结构核查；未知格式、映射碰撞、缺失项均失败。

内存：逐对象持有文件与 section 数据，O(对象大小)。排序 O(n log n)：
section/symbol 的物理排列不参与链接语义，必须按实际身份比较。
"""
import collections
import hashlib
import json
from pathlib import Path
import re
import struct
import sys


class Invalid(ValueError):
    pass


def require(ok, why):
    if not ok:
        raise Invalid(why)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def blob(data, offset, size):
    require(offset >= 0 and size >= 0 and offset + size <= len(data),
            f"越界数据 offset={offset} size={size} total={len(data)}")
    return data[offset:offset + size]


def cstring(data, offset):
    require(0 <= offset < len(data), "字符串偏移越界")
    end = data.find(b"\0", offset)
    require(end >= 0, "字符串未终止")
    return data[offset:end].decode("latin1")


class Names:
    """仅规范编译单元生成 id；声明长度限定范围，不吞掉后续函数长度。"""
    def __init__(self):
        self.originals = {}
        self.changes = {}

    def remember(self, original, canonical):
        if original != canonical:
            previous = self.originals.setdefault(canonical, original)
            require(previous == original, f"名称映射碰撞: {previous} / {original}")
            self.changes[original] = canonical
        return canonical

    def __call__(self, text):
        pattern = r"(\d+)(_INTERNAL_|_GLOBAL__N__)"
        pos = 0
        while match := re.search(pattern, text[pos:]):
            digits = match[1]
            token_start = pos + match.start(2)
            candidates = []
            # 相邻 namespace 的长度也可能属于连续数字；只能由完整 token 反证。
            for cut in range(len(digits)):
                length = int(digits[cut:])
                token = text[token_start:token_start + length]
                parsed = re.fullmatch(r"(_INTERNAL_|_GLOBAL__N__)([0-9a-f]{8})_(\d+)_(.*)", token)
                if parsed is None:
                    continue
                count = int(parsed[3])
                source, suffix = parsed[4][:count], parsed[4][count:]
                if source.endswith("_cu") and re.fullmatch(r"_[0-9a-f]{8}(?:_\d+)?", suffix):
                    candidates.append((cut, length, token, parsed[1] + "UNIT_" + parsed[3] + "_" + source + suffix[:9]))
            require(len(candidates) == 1, "生成名称长度不唯一/未知: " + text[token_start:token_start + 100])
            cut, length, token, canonical = candidates[0]
            start = pos + match.start() + cut
            self.remember(token, canonical)
            replacement = str(len(canonical)) + canonical
            text = text[:start] + replacement + text[token_start + length:]
            pos = start + len(replacement)
        # MSVC 的匿名 namespace 不带 Itanium 长度；保留源文件标识与源 hash。
        def msvc(match):
            canonical = "_GLOBAL__N__UNIT_" + match[1] + "_" + match[2] + "_" + match[3]
            return self.remember(match[0], canonical)
        text = re.sub(r"_GLOBAL__N__[0-9a-f]{8}_(\d+)_([A-Za-z0-9_]+?_cu)_([0-9a-f]{8})(?:_\d+)?(?=@)", msvc, text)
        return text

    def data(self, raw):
        # 只处理完整 ASCII C 字符串；数值、指令和其它二进制不参与名称替换。
        def replace(match):
            text = match[0][:-1].decode("latin1")
            if "_INTERNAL_" in text or "_GLOBAL__N__" in text:
                text = self(text)
            if re.fullmatch(r"[A-Za-z]:[\\/].*\.cu", text):
                text = text.replace("\\", "/")
            return text.encode("latin1") + b"\0"
        return re.sub(rb"[\x20-\x7e]{6,}\x00", replace, raw)


class Coff:
    def __init__(self, data):
        self.data = data
        self.big = data[:4] == b"\0\0\xff\xff"
        if self.big:
            require(len(data) >= 56, "截断 BigObj")
            version, self.machine = struct.unpack_from("<HH", data, 4)
            require(version == 2 and data[12:28].hex() == "c7a1bad1eebaa94baf20faf66aa4dcb8", "未知 BigObj")
            nsec, ptr, count = struct.unpack_from("<III", data, 44)
            size, base = 20, 56
            self.header = [self.machine, blob(data, 28, 16).hex()]
        else:
            require(len(data) >= 20, "截断 COFF")
            self.machine, nsec, timestamp, ptr, count, optional, flags = struct.unpack_from("<HHIIIHH", data)
            require(optional == 0, "非对象 COFF")
            size, base = 18, 20
            self.header = [self.machine, flags]
        require(self.machine == 0x8664 and ptr > 0 and count > 0 and nsec > 0, "非 AMD64 对象")
        blob(data, base, nsec * 40)
        blob(data, ptr, count * size + 4)
        table = ptr + count * size
        length = struct.unpack_from("<I", data, table)[0]
        require(length >= 4, "无效字符串表")
        strings = blob(data, table, length)
        require(table + length == len(data), "对象存在未知尾部数据")

        def name(raw, section=False):
            if section and raw.startswith(b"/"):
                require(raw.rstrip(b"\0")[1:].isdigit(), "无效 section 名称")
                return cstring(strings, int(raw.rstrip(b"\0")[1:]))
            if raw[:4] == b"\0" * 4:
                return cstring(strings, struct.unpack_from("<I", raw, 4)[0])
            return raw.rstrip(b"\0").decode("latin1")

        self.sections = {}
        for number in range(1, nsec + 1):
            h = base + (number - 1) * 40
            rawname, virtual, address, rawsize, rawptr, relptr, lineptr, nrel, nline, flags = struct.unpack_from("<8sIIIIIIHHI", data, h)
            require(lineptr == 0 and nline == 0, "未支持 COFF 行号记录")
            raw = blob(data, rawptr, rawsize) if rawptr else b""
            require(rawptr or not rawsize or flags & 0x80, "缺失 section 数据")
            skip = 0
            if flags & 0x1000000:
                require(nrel == 65535 and relptr > 0, "无效 relocation overflow")
                nrel, sentinel, kind = struct.unpack("<IIH", blob(data, relptr, 10))
                require(nrel > 65535 and sentinel == 0 and kind == 0, "无效 overflow 哨兵")
                skip = 1
            require(not nrel or relptr > 0, "缺失 relocation 表")
            rel = list(struct.iter_unpack("<IIH", blob(data, relptr, nrel * 10)))[skip:]
            self.sections[number] = dict(name=name(rawname, True), raw=raw, flags=flags & ~0x1000000,
                                         virtual=virtual, address=address, size=rawsize, rel=rel, aux=None)
        self.symbols = {}
        index = 0
        while index < count:
            h = ptr + index * size
            rawname, value = struct.unpack_from("<8sI", data, h)
            section = struct.unpack_from("<i" if self.big else "<h", data, h + 12)[0]
            type_, storage, naux = struct.unpack_from("<HBB", data, h + (16 if self.big else 14))
            require(section <= nsec and index + naux < count, "无效 symbol")
            sym = dict(name=name(rawname), value=value, section=section, type=type_, storage=storage,
                       aux=blob(data, h + size, naux * size), index=index)
            self.symbols[index] = sym
            if storage == 3 and section > 0 and naux and sym["name"] == self.sections[section]["name"]:
                require(naux == 1, "未知 section aux")
                a = sym["aux"]
                length_, nr, nl, checksum, low, selection, reserved, high = struct.unpack_from("<IHHIHBBH", a)
                parent = low | (high << 16)
                require(reserved == 0 and nl == 0 and not any(a[18:]), "未知 section aux 字段")
                self.sections[section]["aux"] = dict(length=length_, reloc=nr, checksum=checksum,
                                                     parent=parent, selection=selection, tail=b"")
            index += 1 + naux
        for sec in self.sections.values():
            for offset, target, kind in sec["rel"]:
                require(target in self.symbols and kind in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), "未知 relocation 类型或目标")
                width = 8 if kind == 1 else 2 if kind == 10 else 0 if kind == 0 else 4
                require(offset + width <= len(sec["raw"]), "relocation 超出 section")

    def canonical(self, names):
        symbols = self.symbols
        sections = self.sections
        normal = {i: names(s["name"]) for i, s in symbols.items() if not s["name"].startswith("??_C@")}
        for i, s in symbols.items():
            if s["name"].startswith("??_C@"):
                require(s["section"] > 0 and s["value"] == 0, "字符串 COMDAT 定义无效")
                sec = sections[s["section"]]
                require(not sec["rel"] and sec["name"] == ".rdata", "字符串 COMDAT 非纯数据")
                normal[i] = "literal:" + sha(names.data(sec["raw"]))
        defined = collections.defaultdict(list)
        for i, s in symbols.items():
            if s["section"] > 0 and not (s["storage"] == 3 and s["name"] == sections[s["section"]]["name"]):
                defined[s["section"]].append((normal[i], s["value"], s["type"], s["storage"]))
        keys = {}

        def key(number, active=()):
            if number <= 0:
                return str(number)
            if number in keys:
                return keys[number]
            require(number not in active, "COMDAT 关联环")
            sec = sections[number]
            aux = sec["aux"]
            if aux and aux["selection"] == 5:
                require(aux["parent"] in sections and aux["parent"] != number, "无效 COMDAT parent")
                identity = [sec["name"], "assoc", key(aux["parent"], active + (number,)), sorted(defined[number])]
            elif defined[number]:
                identity = [sec["name"], sorted(defined[number])]
            else:
                identity = [sec["name"]]
            keys[number] = json.dumps(identity, separators=(",", ":"))
            return keys[number]

        for number in sections:
            key(number)
        duplicates = [k for k, n in collections.Counter(keys.values()).items() if n != 1]
        require(not duplicates, "section 身份不唯一: " + str(duplicates[:4]))
        result = {}
        ignored = []
        for number, sec in sections.items():
            aux = sec["aux"]
            # LNK_REMOVE 的 section 不进入最终映像；仍核查不存在 runtime 引用。
            if sec["name"] == ".chks64":
                require(sec["flags"] == 0xa00 and not sec["rel"] and len(sec["raw"]) == len(sections) * 8,
                        "校验和 section 非已验证格式")
                require(all(s["storage"] == 3 and s["name"] == ".chks64" and s["value"] == 0 for s in symbols.values() if s["section"] == number),
                        "校验和 section 存在可见绑定")
                for other in sections.values():
                    if not other["flags"] & 0x800:
                        require(not any(symbols[t]["section"] == number for _, t, _ in other["rel"]), "runtime 引用被丢弃 section")
                ignored.append([sec["name"], sec["flags"], len(sec["raw"]), sha(sec["raw"])])
                continue
            raw = names.data(sec["raw"]) if sec["name"] == ".rdata" else sec["raw"]
            reloc = []
            for offset, target, kind in sec["rel"]:
                s = symbols[target]
                reloc.append([offset, kind, normal[target], key(s["section"]), s["value"], s["type"], s["storage"]])
            auxiliary = None if aux is None else [len(raw) if sec["raw"] else aux["length"], aux["reloc"], aux["selection"],
                                                  key(aux["parent"]) if aux["selection"] == 5 else aux["parent"], aux["tail"].hex()]
            result[key(number)] = dict(flags=sec["flags"], virtual=sec["virtual"], address=sec["address"],
                                       size=sec["size"] if not sec["raw"] else len(raw),
                                       data=sha(raw), reloc=sorted(reloc), aux=auxiliary)
        symrows = []
        for i, s in symbols.items():
            if s["section"] > 0 and sections[s["section"]]["name"] == ".chks64":
                continue
            if s["section"] > 0 and s["storage"] == 3 and sections[s["section"]]["aux"] and s["name"] == sections[s["section"]]["name"]:
                auxiliary = "section-definition"
            elif s["storage"] == 105:
                require(len(s["aux"]) == (20 if self.big else 18), "未知 weak external aux")
                target, search = struct.unpack_from("<II", s["aux"])
                require(target in symbols and search in (1, 2, 3, 4) and not any(s["aux"][8:]), "无效 weak external aux")
                bound = symbols[target]
                auxiliary = [normal[target], key(bound["section"]), bound["value"], bound["type"], bound["storage"], search]
            else:
                require(not s["aux"] or s["storage"] == 103, "未知非 section/weak/file auxiliary binding")
                auxiliary = s["aux"].hex()
            symrows.append([normal[i], s["value"], key(s["section"]), s["type"], s["storage"], auxiliary])
        return dict(header=self.header, sections=result, symbols=sorted(symrows, key=lambda s: json.dumps(s)), ignored=ignored)


def diagnose(root):
    for path in sorted(Path(root).glob("*.obj")):
        c = Coff(path.read_bytes())
        print(path.name, "bigobj", c.big, "sections", len(c.sections), "symbols", len(c.symbols))
        print("relocations", dict(collections.Counter(k for s in c.sections.values() for _, _, k in s["rel"])))
        print("aux", dict(collections.Counter((s["storage"], len(s["aux"])) for s in c.symbols.values() if s["aux"])))
        print("discard", dict(collections.Counter((s["name"], hex(s["flags"])) for s in c.sections.values() if s["flags"] & 0x800)))
        try:
            c.canonical(Names())
            print("canonical OK")
        except Invalid as err:
            print("canonical FAIL:", err)


if __name__ == "__main__":
    diagnose(sys.argv[1])
