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
    def __init__(self, unit_owner=None, lambda_map=None, namespace_map=None):
        self.lambda_map = lambda_map or {}
        self.namespace_map = namespace_map or {}
        require(len(set(self.lambda_map.values())) == len(self.lambda_map), "lambda 改名非一一映射")
        require(len(set(self.namespace_map.values())) == len(self.namespace_map), "namespace 改名非一一映射")
        require(not self.namespace_map or unit_owner, "namespace 改名缺少实际源所有权")
        require(all(re.fullmatch(r"\?A0x[0-9a-f]{8}", t) for pair in self.namespace_map.items() for t in pair), "未知 namespace 映射 token")
        self.originals = {}
        self.changes = {}
        self.unit_owner = unit_owner.replace(".", "_") if unit_owner else None
        self.unit_ids = {}

    def unit_hash(self, source, source_hash):
        if self.unit_owner is None:
            return source_hash
        require(source == self.unit_owner, "生成单元名称与实际编译源不符")
        previous = self.unit_ids.setdefault(source, source_hash)
        require(previous == source_hash, "同一对象存在多个生成单元 ID")
        # 第二段 ID 不是稳定源码哈希；只为已绑定到实际源文件的单元改名。
        return "ID"

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
                    candidates.append((cut, length, token, parsed[1] + "UNIT_" + parsed[3] + "_" + source,
                                       source, suffix[1:9]))
            require(len(candidates) == 1, "生成名称长度不唯一/未知: " + text[token_start:token_start + 100])
            cut, length, token, canonical, source, source_hash = candidates[0]
            canonical += "_" + self.unit_hash(source, source_hash)
            start = pos + match.start() + cut
            self.remember(token, canonical)
            replacement = str(len(canonical)) + canonical
            text = text[:start] + replacement + text[token_start + length:]
            pos = start + len(replacement)
        # MSVC 的匿名 namespace 不带 Itanium 长度；源文件所有权约束相同。
        def msvc(match):
            require(int(match[1]) == len(match[2]), "MSVC 生成单元源名称长度错误")
            canonical = "_GLOBAL__N__UNIT_" + match[1] + "_" + match[2] + "_" + self.unit_hash(match[2], match[3])
            return self.remember(match[0], canonical)
        text = re.sub(r"_GLOBAL__N__[0-9a-f]{8}_(\d+)_([A-Za-z0-9_]+?_cu)_([0-9a-f]{8})(?:_\d+)?(?=@)", msvc, text)
        def namespace(match):
            original = match[0]
            return self.remember(original, self.namespace_map.get(original, original))
        text = re.sub(r"\?A0x[0-9a-f]{8}(?=@@)", namespace, text)
        def renamed(match):
            original = match[0]
            return self.remember(original, self.lambda_map.get(original, original))
        return re.sub(r"<lambda_[0-9a-f]{32}>", renamed, text)

    def data(self, raw):
        # 只处理完整 ASCII C 字符串；数值、指令和其它二进制不参与名称替换。
        def replace(match):
            text = match[0][:-1].decode("latin1")
            if "_INTERNAL_" in text or "_GLOBAL__N__" in text or "<lambda_" in text or "?A0x" in text:
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

        def table_name(offset):
            require(4 <= offset < len(strings), "COFF 字符串偏移越界/指向表头")
            return cstring(strings, offset)

        def name(raw, section=False):
            if section and raw.startswith(b"/"):
                label = raw.rstrip(b"\0")
                if label.startswith(b"//"):
                    # COFF 的 // 六位 base64 表示大字符串偏移，不是被忽略的名称。
                    digits = label[2:]
                    alphabet = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
                    require(1 <= len(digits) <= 6 and all(c in alphabet for c in digits), "无效 COFF base64 section 名称")
                    offset = 0
                    for digit in digits:
                        offset = offset * 64 + alphabet.index(digit)
                    require(offset <= 0xffffffff, "COFF base64 偏移溢出")
                else:
                    require(label[1:].isdigit(), "无效 section 名称")
                    offset = int(label[1:])
                return table_name(offset)
            if raw[:4] == b"\0" * 4:
                return table_name(struct.unpack_from("<I", raw, 4)[0])
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


def namespace_bijection(a, b, owner):
    """只为实物确认的静态 nvcc helper namespace 构造唯一候选，不宣称等价。

    helper 定义名称没有 lambda，因而可先锚定 namespace，再细化 lambda；
    所有 incoming/outgoing 重定位仍由后续 lambda 指纹及完整 canonical 核查。
    本实验每个源最多一个这种 namespace；未知形态、多候选、外部绑定均失败。
    定义多重集排序 O(n log n)，不按十六进制值或物理顺序选择映射。
    """
    pattern = r"\?A0x[0-9a-f]{8}(?=@@)"
    def profile(coff):
        found = set()
        rows = []
        targets = set()
        for index, symbol in coff.symbols.items():
            tokens = re.findall(pattern, symbol["name"])
            if not tokens:
                continue
            require(owner and len(tokens) == 1, "namespace 缺少实际源所有权/名称歧义")
            require(symbol["section"] > 0 and symbol["storage"] == 3 and symbol["type"] == 0 and not symbol["aux"], "namespace 存在非静态 helper 绑定")
            require(re.match(r"\?fp_(?:copier|caller|deleter|noobject_caller)@\?\$__nv_hdl_helper@", symbol["name"]) and "<lambda_" not in symbol["name"], "未验证 namespace 定义形态")
            sec = coff.sections[symbol["section"]]
            require(sec["name"] == ".bss" and not sec["rel"], "未验证 namespace 存储 section")
            found.update(tokens)
            targets.add(index)
            rows.append([re.sub(pattern, "?A0xSELF", symbol["name"]), symbol["value"], symbol["type"], symbol["storage"],
                         sec["name"], sec["flags"], sec["virtual"], sec["address"], sec["size"], sha(sec["raw"]),
                         None if sec["aux"] is None else [sec["aux"][k] for k in ("length", "reloc", "selection", "parent")]])
        require(len(found) <= 1, "同一实际源存在多个 namespace，拒绝任意映射")
        for sec in coff.sections.values():
            for offset, target, kind in sec["rel"]:
                if target in targets:
                    require(kind == 4, "未验证 namespace 引用类型")
        return found, sorted(rows, key=lambda row: json.dumps(row))
    ga, pa = profile(a)
    gb, pb = profile(b)
    require(len(ga) == len(gb) and pa == pb, "namespace 完整定义/数据绑定指纹不符")
    if ga == gb:
        return {}
    # 两侧各唯一，且完整定义多重集相同；后续不得省略任何重定位或实际载荷。
    return {next(iter(gb)): next(iter(ga))}


def lambda_records(coff, names):
    """仅构造候选指纹；最终等价仍由完整 canonical 和设备门禁决定。"""
    pattern = r"<lambda_[0-9a-f]{32}>"
    symbols, sections = coff.symbols, coff.sections
    labels = {}
    defined = collections.defaultdict(list)
    for i, symbol in symbols.items():
        if symbol["name"].startswith("??_C@"):
            sec = sections[symbol["section"]]
            labels[i] = "literal:" + names.data(sec["raw"]).decode("latin1")
        else:
            labels[i] = names(symbol["name"])
        if symbol["section"] > 0 and not (symbol["storage"] == 3 and symbol["name"] == sections[symbol["section"]]["name"]):
            defined[symbol["section"]].append([labels[i], symbol["value"], symbol["type"], symbol["storage"]])
    identities = {}

    def identity(number, active=()):
        if number <= 0:
            return str(number)
        if number in identities:
            return identities[number]
        require(number not in active, "COMDAT 关联环")
        sec = sections[number]
        aux = sec["aux"]
        row = dict(name=sec["name"], definitions=defined[number])
        if aux and aux["selection"] == 5:
            require(aux["parent"] in sections and aux["parent"] != number, "无效 COMDAT parent")
            row["parent"] = identity(aux["parent"], active + (number,))
        identities[number] = row
        return row

    def binding(index):
        symbol = symbols[index]
        return [labels[index], symbol["value"], identity(symbol["section"]), symbol["type"], symbol["storage"]]

    records = []
    for number, sec in sections.items():
        raw = names.data(sec["raw"]) if sec["name"] == ".rdata" else sec["raw"]
        # 设备载荷和被丢弃校验和不是 GUID 映射锚点，最终门禁另行完整验证。
        separate = sec["name"] in (".nv_fatbin", ".chks64")
        content = "separate-gate" if separate else raw.decode("latin1") if sec["name"] == ".rdata" else sha(raw)
        aux = sec["aux"]
        metadata = [sec["flags"], sec["virtual"], sec["address"],
                    "separate-gate" if separate else len(raw) if raw else sec["size"], content,
                    None if aux is None else [aux["reloc"], aux["selection"],
                                             identity(aux["parent"]) if aux["selection"] == 5 else aux["parent"]]]
        records.append(["section", identity(number), metadata])
        for offset, target, kind in sec["rel"]:
            # incoming/outgoing 引用都出现在记录内；原始数据摘要保留 addend。
            records.append(["relocation", identity(number), metadata, offset, kind, binding(target)])
    for index, symbol in symbols.items():
        if symbol["storage"] == 105:
            target, search = struct.unpack_from("<II", symbol["aux"])
            require(target in symbols, "weak external 目标缺失")
            auxiliary = [binding(target), search]
        elif symbol["storage"] == 3 and symbol["section"] > 0 and symbol["name"] == sections[symbol["section"]]["name"]:
            auxiliary = "section-definition"
        else:
            auxiliary = symbol["aux"].hex()
        records.append(["symbol", binding(index), auxiliary])
    mentions = collections.defaultdict(list)
    for row in records:
        text = json.dumps(row, separators=(",", ":"))
        for guid in set(re.findall(pattern, text)):
            mentions[guid].append(text)
    return mentions


def lambda_bijection(a, b, na, nb):
    """结构细化只接受唯一映射，绝不穷举歧义图；改名本身不是等价证明。

    每轮扫描引用记录并排序其多重集 O(n log n)，最多 g 轮 O(g*n log n)。
    迭代只细分候选类；相同 GUID 为不可更换的锚点，稳定不分裂即拒绝。
    """
    pattern = r"<lambda_[0-9a-f]{32}>"
    ma, mb = lambda_records(a, na), lambda_records(b, nb)
    shared = ma.keys() & mb.keys()
    ga, gb = ma.keys() - shared, mb.keys() - shared
    require(len(ga) == len(gb), "lambda 生成身份数量不符")
    if not ga:
        return {}
    colors_a = dict.fromkeys(ga, "candidate")
    colors_b = dict.fromkeys(gb, "candidate")
    previous_groups = 1
    for iteration in range(len(ga) + 1):
        def profiles(mentions, colors):
            result = {}
            for guid in colors:
                def replace(match):
                    token = match[0]
                    return "SELF" if token == guid else colors.get(token, token)
                def unordered_definitions(value):
                    if isinstance(value, list):
                        return [unordered_definitions(v) for v in value]
                    if isinstance(value, dict):
                        transformed = {k: unordered_definitions(v) for k, v in value.items()}
                        if "definitions" in transformed:
                            # 改名后才排序定义多重集，不能让物理符号顺序制造唯一映射。
                            transformed["definitions"] = sorted(transformed["definitions"], key=lambda v: json.dumps(v, sort_keys=True))
                        return transformed
                    return value
                rows = sorted(json.dumps(unordered_definitions(json.loads(re.sub(pattern, replace, row))),
                                         sort_keys=True, separators=(",", ":")) for row in mentions[guid])
                result[guid] = sha(json.dumps([colors[guid], rows], separators=(",", ":")).encode())
            return result
        ca, cb = profiles(ma, colors_a), profiles(mb, colors_b)
        ba, bb = collections.defaultdict(list), collections.defaultdict(list)
        for guid, color in ca.items():
            ba[color].append(guid)
        for guid, color in cb.items():
            bb[color].append(guid)
        require({c: len(v) for c, v in ba.items()} == {c: len(v) for c, v in bb.items()},
                "lambda 指令/常量/定义/引用指纹不符")
        if all(len(v) == 1 for v in ba.values()):
            mapping = {bb[color][0]: values[0] for color, values in ba.items()}
            require(set(mapping) == gb and set(mapping.values()) == ga, "lambda 映射覆盖不完整")
            return mapping
        require(len(ba) > previous_groups, "lambda 结构映射歧义，拒绝任意排列")
        previous_groups = len(ba)
        colors_a, colors_b = ca, cb
    raise Invalid("lambda 结构细化未收敛")


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
