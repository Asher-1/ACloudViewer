"""严格校验器的破坏性负例，以及格式/布局正例；无第三方依赖。"""
import copy
import io
from contextlib import redirect_stderr, redirect_stdout
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch
from strict_coff import Coff, Invalid, Names, lambda_bijection, namespace_bijection
from strict_device import check_extracted, decoded_entries, decompress_payload, elf, extract, fatbin_entries, ptx, run, verify
from strict_gate import check_composition, compare_object, identity_tokens, inventory, merge_identity_maps, source_owner
import strict_gate


def object_bytes(big=False):
    width, base = (20, 56) if big else (18, 20)
    code = b"\x48\x83\xec\x28\0\0\0\0\xc3"
    constant = b"\x11\x22\x33\x44"
    payload = code + constant + struct.pack("<IIH", 4, 5, 4)
    pointer = base + 80 + len(payload)
    if big:
        header = struct.pack("<HHHHI", 0, 65535, 2, 0x8664, 1) + bytes.fromhex("c7a1bad1eebaa94baf20faf66aa4dcb8")
        header += struct.pack("<IIIIIII", 0, 0, 0, 0, 2, pointer, 7)
    else:
        header = struct.pack("<HHIIIHH", 0x8664, 2, 1, pointer, 7, 0, 0)
    sections = b""
    sections += struct.pack("<8sIIIIIIHHI", b".text", 0, 0, len(code), base + 80, base + 80 + len(code) + len(constant), 0, 1, 0, 0x60501020)
    sections += struct.pack("<8sIIIIIIHHI", b".rdata", 0, 0, len(constant), base + 80 + len(code), 0, 0, 0, 0, 0x40301040)
    def symbol(name, value, section, type_, storage, aux=0):
        return struct.pack("<8sIiHBB" if big else "<8sIhHBB", name, value, section, type_, storage, aux)
    def auxiliary(length, nr):
        return struct.pack("<IHHIHBBH", length, nr, 0, 0, 0, 2, 0, 0) + bytes(width - 18)
    symbols = symbol(b".text", 0, 1, 0, 3, 1) + auxiliary(len(code), 1)
    symbols += symbol(b".rdata", 0, 2, 0, 3, 1) + auxiliary(len(constant), 0)
    symbols += symbol(b"func", 0, 1, 32, 2) + symbol(b"data", 0, 2, 0, 2) + symbol(b"other", 0, 0, 0, 2)
    return header + sections + payload + symbols + struct.pack("<I", 4)


def cubin_bytes():
    labels = ["", ".text.func", ".nv.constant0.func", ".strtab", ".shstrtab", ".symtab", ".rel.text.func"]
    strings = b"\0func\0data\0other\0"
    shstrings = b"\0" + b"\0".join(n.encode() for n in labels[1:]) + b"\0"
    symbols = bytes(24) + struct.pack("<IBBHQQ", 1, 18, 0, 1, 0, 16) + struct.pack("<IBBHQQ", 6, 17, 0, 2, 0, 4) + struct.pack("<IBBHQQ", 11, 17, 0, 0, 0, 0)
    values = [b"", bytes(range(16)), b"\x11\x22\x33\x44", strings, shstrings, symbols, struct.pack("<QQ", 0, (2 << 32) | 43)]
    types = [0, 1, 1, 3, 3, 2, 9]
    data, sections = b"", []
    offset = 64
    for i, raw in enumerate(values):
        n = 0 if i == 0 else shstrings.index(labels[i].encode())
        sections.append(struct.pack("<IIQQQQIIQQ", n, types[i], 0, 0, offset, len(raw), 3 if i == 5 else 5 if i == 6 else 0,
                                    1 if i in (5, 6) else 0, 1, 24 if i == 5 else 16 if i == 6 else 0))
        data += raw
        offset += len(raw)
    ident = b"\x7fELF\x02\x01\x013\x07" + bytes(7)
    header = struct.pack("<16sHHIQQQIHHHHHH", ident, 2, 190, 126, 0, 0, offset, 90, 64, 56, 0, 64, len(labels), 4)
    return header + data + b"".join(sections)


class HostTests(unittest.TestCase):
    def model(self):
        return Coff(object_bytes())

    def test_plain_and_bigobj(self):
        self.assertEqual(Coff(object_bytes()).canonical(Names())["sections"], Coff(object_bytes(True)).canonical(Names())["sections"])

    def long_name_object(self, encoding, big=False):
        raw = bytearray(object_bytes(big))
        base = 56 if big else 20
        pointer = struct.unpack_from("<I", raw, 48 if big else 8)[0]
        raw[base:base + 8] = encoding.ljust(8, b"\0")
        raw[pointer:pointer + 8] = struct.pack("<II", 0, 4)
        text = b".long.section\0"
        raw[-4:] = struct.pack("<I", 4 + len(text)) + text
        return bytes(raw)

    def test_section_base64_and_decimal_names(self):
        for big in (False, True):
            decimal = Coff(self.long_name_object(b"/4", big)).canonical(Names())
            encoded = Coff(self.long_name_object(b"//AAAAAE", big)).canonical(Names())
            self.assertEqual(decimal, encoded)

    def test_invalid_section_offset_encoding(self):
        for encoding in (b"//??????", b"////////", b"//AAAAAA", b"//", b"/0", b"/9999999", b"//AA////"):
            for big in (False, True):
                with self.subTest(encoding=encoding, big=big), self.assertRaises(Invalid):
                    Coff(self.long_name_object(encoding, big))
        raw = bytearray(self.long_name_object(b"/4"))
        raw[-1] = ord("x")
        with self.assertRaises(Invalid):
            Coff(bytes(raw))

    def test_truncated_input(self):
        raw = object_bytes()
        for cut in range(len(raw)):
            with self.assertRaises((Invalid, struct.error)):
                Coff(raw[:cut])

    def changed(self, mutate):
        a, b = self.model(), self.model()
        mutate(b)
        self.assertNotEqual(a.canonical(Names()), b.canonical(Names()))

    def test_instruction_change(self):
        self.changed(lambda c: c.sections[1].update(raw=b"\x90" + c.sections[1]["raw"][1:]))

    def test_constant_change(self):
        self.changed(lambda c: c.sections[2].update(raw=b"\0" + c.sections[2]["raw"][1:]))

    def test_relocation_target_change(self):
        self.changed(lambda c: c.sections[1].update(rel=[(4, 6, 4)]))

    def test_relocation_type_change(self):
        self.changed(lambda c: c.sections[1].update(rel=[(4, 5, 3)]))

    def test_relocation_addend_change(self):
        self.changed(lambda c: c.sections[1].update(raw=c.sections[1]["raw"][:4] + b"\1\0\0\0" + c.sections[1]["raw"][8:]))

    def test_binding_change(self):
        self.changed(lambda c: c.symbols[5].update(storage=3))

    def test_comdat_change(self):
        self.changed(lambda c: c.sections[1]["aux"].update(selection=4))

    def test_association_cycle(self):
        c = self.model()
        c.sections[1]["aux"].update(selection=5, parent=2)
        c.sections[2]["aux"].update(selection=5, parent=1)
        with self.assertRaises(Invalid):
            c.canonical(Names())

    def test_unknown_auxiliary_is_rejected(self):
        c = self.model()
        c.symbols[4]["aux"] = bytes(18)
        with self.assertRaises(Invalid):
            c.canonical(Names())

    def test_symbol_order(self):
        a, b = self.model(), self.model()
        b.symbols[5], b.symbols[6] = b.symbols[6], b.symbols[5]
        b.sections[1]["rel"] = [(4, 6, 4)]
        self.assertEqual(a.canonical(Names()), b.canonical(Names()))

    def test_missing_object(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(Invalid):
                inventory(temp, ["missing.obj"])


class DeviceTests(unittest.TestCase):
    def test_compressed_payload_exact_consumption(self):
        self.assertEqual(decompress_payload(b"\x50hello", 5), b"hello")
        self.assertEqual(decompress_payload(b"\x14a\x01\x00\x50hello", 14), b"a" * 9 + b"hello")
        self.assertEqual(decompress_payload(b"\xf0\x05" + b"x" * 20, 20), b"x" * 20)
        self.assertEqual(decompress_payload(b"\x1fa\x01\x00\x02\x50hello", 27), b"a" * 22 + b"hello")
        for data, size in ((b"", 0), (b"\xf0", 15), (b"\x50he", 5),
                           (b"\x10a\x00\x00", 5), (b"\x10a\x02\x00", 5),
                           (b"\x10a\x01", 5), (b"\x1fa\x01\x00", 20),
                           (b"\x50hello", 4), (b"\x50hello", 6),
                           (b"\x50hello\x00", 5)):
            with self.subTest(data=data, size=size), self.assertRaises(Invalid):
                decompress_payload(data, size)

    def compressed_container(self):
        payload = b"\x50hello\0\0"
        header = bytearray(64)
        struct.pack_into("<HHIQ", header, 0, 2, 0x101, 64, len(payload))
        struct.pack_into("<I", header, 16, 6)
        struct.pack_into("<I", header, 28, 90)
        struct.pack_into("<Q", header, 40, 0x2041)
        struct.pack_into("<Q", header, 56, 5)
        return struct.pack("<IHHQ", 0xba55ed50, 1, 16, 72) + header + payload

    def test_fatbin_physical_lengths_and_padding(self):
        raw = self.compressed_container()
        self.assertEqual(list(decoded_entries(raw))[0]["decoded"], b"hello")
        for offset, fmt, value in ((32, "I", 0), (32, "I", 9), (32, "I", 5),
                                   (72, "Q", 4), (72, "Q", 6), (56, "Q", 0x4041),
                                   (24, "Q", 9), (20, "I", 72)):
            bad = bytearray(raw)
            struct.pack_into("<" + fmt, bad, offset, value)
            with self.subTest(offset=offset, value=value), self.assertRaises((Invalid, struct.error)):
                list(decoded_entries(bytes(bad)))
        bad = bytearray(raw)
        bad[-1] = 1
        with self.assertRaises(Invalid):
            list(decoded_entries(bytes(bad)))

    def test_uncompressed_payload_is_not_padding(self):
        raw = bytearray(self.compressed_container())
        struct.pack_into("<I", raw, 32, 0)
        struct.pack_into("<Q", raw, 56, 0x41)
        struct.pack_into("<Q", raw, 72, 0)
        self.assertEqual(list(decoded_entries(bytes(raw)))[0]["decoded"], b"\x50hello\0\0")
        struct.pack_into("<I", raw, 32, 6)
        with self.assertRaises(Invalid):
            list(decoded_entries(bytes(raw)))

    def test_extraction_cannot_skip_payload_bytes(self):
        raw = self.compressed_container()
        manifest = [dict(kind=2, arch=90, file="test.cubin")]
        with tempfile.TemporaryDirectory() as temp:
            file = Path(temp) / "test.cubin"
            file.write_bytes(b"hello")
            check_extracted(raw, manifest, temp)
            for bad in (b"hell", b"hello\0", b"jello"):
                file.write_bytes(bad)
                with self.assertRaises(Invalid):
                    check_extracted(raw, manifest, temp)
            with self.assertRaises(Invalid):
                check_extracted(raw, manifest + manifest, temp)
            header = bytearray(80)
            struct.pack_into("<HHIQ", header, 0, 1, 0x101, 80, 6)
            struct.pack_into("<I", header, 28, 90)
            struct.pack_into("<Q", header, 40, 0x41)
            ptx_raw = struct.pack("<IHHQ", 0xba55ed50, 1, 16, 86) + header + b"hello\0"
            manifest = [dict(kind=1, arch=90, file="test.cubin")]
            file.write_bytes(b"hello")
            check_extracted(ptx_raw, manifest, temp)
            for bad in (b"hell", b"hello\0", b"hello\0trailing"):
                file.write_bytes(bad)
                with self.assertRaises(Invalid):
                    check_extracted(ptx_raw, manifest, temp)

    def ptx_container(self, pid, instruction="ret;", hsize=80, newline="\n", terminator=b"\0"):
        name = NamesTests().mangled(pid)
        decoded = (f".version 8.5\n.target sm_90\n.address_size 64\n.visible .entry {name}()\n{{\n{instruction}\n}}\n").replace("\n", newline).encode() + terminator
        length = len(decoded)
        encoded = bytes([min(15, length) << 4])
        if length >= 15:
            count, tail = divmod(length - 15, 255)
            encoded += bytes([255]) * count + bytes([tail])
        encoded += decoded
        compressed = len(encoded)
        encoded += bytes((-compressed) % 8)
        header = bytearray(hsize)
        struct.pack_into("<HHIQ", header, 0, 1, 0x101, hsize, len(encoded))
        struct.pack_into("<I", header, 16, compressed)
        struct.pack_into("<I", header, 28, 90)
        struct.pack_into("<Q", header, 40, 0x2041)
        struct.pack_into("<Q", header, 56, length)
        return struct.pack("<IHHQ", 0xba55ed50, 1, 16, hsize + len(encoded)) + header + encoded

    def verify_ptx_fixture(self, a, b, windows=(False, False)):
        def extracted(tool, raw, directory):
            # 仅替换外部工具；物理解码、精确提取绑定和完整 PTX 比较均运行真实逻辑。
            directory = Path(directory)
            directory.mkdir()
            item, = decoded_entries(raw)
            text = item["decoded"][:-1]
            if windows[0 if directory.name == "a" else 1]:
                text = text.replace(b"\n", b"\r\n")
            (directory / "test.ptx").write_bytes(text)
            manifest = [dict(kind=1, arch=90, file="test.ptx")]
            check_extracted(raw, manifest, directory)
            return manifest, fatbin_entries(raw)
        with tempfile.TemporaryDirectory() as temp, patch("strict_device.extract", side_effect=extracted):
            return verify("fixture", a, b, Path(temp) / "compare", Names("FixedRadiusSearchOps.cu"), Names("FixedRadiusSearchOps.cu"))

    def test_exact_and_windows_text_are_bound_to_complete_ptx(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "test.ptx"
            for newline in ("\n", "\r\n"):
                raw = self.ptx_container("1904", newline=newline)
                item, = decoded_entries(raw)
                exact = item["decoded"][:-1]
                expected = None
                for windows in (False, True):
                    with self.subTest(newline=newline, windows=windows):
                        path.write_bytes(exact.replace(b"\n", b"\r\n") if windows else exact)
                        manifest = [dict(kind=1, arch=90, file=path.name)]
                        check_extracted(raw, manifest, temp)
                        self.assertIs(manifest[0]["windows_text"], windows)
                        self.assertEqual(manifest[0]["decoded_size"], len(item["decoded"]))
                        result = ptx(path, Names("FixedRadiusSearchOps.cu"), manifest[0]["windows_text"])
                        if expected is None:
                            expected = result
                        self.assertEqual(expected, result)
                        self.assertEqual(result["lines"], 7)
                expanded = exact.replace(b"\n", b"\r\n")
                for bad in (exact.replace(b"\n", b"\r\n", 1), expanded + b"\r",
                            expanded[:-1], expanded.replace(b"ret;", b"trap;"),
                            expanded + b"\0", expanded + b"\0trailing", expanded.replace(b"\r\n", b"\r\r\n", 1)):
                    path.write_bytes(bad)
                    with self.subTest(newline=newline, bad=bad), self.assertRaises(Invalid):
                        check_extracted(raw, [dict(kind=1, arch=90, file=path.name)], temp)

    def test_ptx_requires_one_terminal_nul_and_no_embedded_nul(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "test.ptx"
            for terminator, instruction in ((b"", "ret;"), (b"\0\0", "ret;"), (b"\0", "re\0t;")):
                raw = self.ptx_container("1904", instruction, terminator=terminator)
                item, = decoded_entries(raw)
                path.write_bytes(item["decoded"].rstrip(b"\0"))
                with self.subTest(terminator=terminator, instruction=instruction), self.assertRaises(Invalid):
                    check_extracted(raw, [dict(kind=1, arch=90, file=path.name)], temp)

    def test_windows_mode_is_per_entry_and_does_not_hide_instruction_changes(self):
        a, b = self.ptx_container("1904"), self.ptx_container("19040", newline="\r\n")
        for windows in ((False, False), (False, True), (True, False), (True, True)):
            with self.subTest(windows=windows):
                report = self.verify_ptx_fixture(a, b, windows)
                self.assertTrue(report["equivalent"])
                entry = report["entries"][0]
                self.assertIs(entry["windows_text_a"], windows[0])
                self.assertIs(entry["windows_text_b"], windows[1])
                changed = self.ptx_container("19040", "trap;", newline="\r\n")
                self.assertFalse(self.verify_ptx_fixture(a, changed, windows)["equivalent"])

    def test_standalone_cr_and_unbound_crcrlf_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "test.ptx"
            item, = decoded_entries(self.ptx_container("1904"))
            exact = item["decoded"][:-1]
            for raw in (exact.replace(b"ret;", b"re\rt;"), exact.replace(b"\n", b"\r\r\n")):
                path.write_bytes(raw)
                with self.assertRaises(Invalid):
                    ptx(path, Names("FixedRadiusSearchOps.cu"))
            path.write_bytes(exact.replace(b"ret;", b"re\rt;").replace(b"\n", b"\r\n"))
            with self.assertRaises(Invalid):
                ptx(path, Names("FixedRadiusSearchOps.cu"), True)

    def test_generated_name_length_is_not_a_control_field(self):
        a, b = self.ptx_container("1904"), self.ptx_container("19040")
        x, = decoded_entries(a)
        y, = decoded_entries(b)
        self.assertNotEqual(x["declared_size"], y["declared_size"])
        self.assertEqual(fatbin_entries(a), fatbin_entries(b))
        report = self.verify_ptx_fixture(a, b)
        self.assertTrue(report["equivalent"])
        self.assertNotEqual(report["entries"][0]["decoded_size_a"], report["entries"][0]["decoded_size_b"])

    def test_length_separation_does_not_hide_changed_instructions(self):
        a, b = self.ptx_container("1904"), self.ptx_container("19040", "trap;")
        self.assertEqual(fatbin_entries(a), fatbin_entries(b))
        self.assertFalse(self.verify_ptx_fixture(a, b)["equivalent"])

    def test_declared_length_must_still_match_complete_decode(self):
        raw = bytearray(self.ptx_container("1904"))
        declared = struct.unpack_from("<Q", raw, 72)[0]
        for value in (0, declared - 1, declared + 1):
            bad = bytearray(raw)
            struct.pack_into("<Q", bad, 72, value)
            with self.assertRaises(Invalid):
                fatbin_entries(bytes(bad))

    def test_reserved_and_extended_control_bytes_are_retained(self):
        for hsize in (80, 104):
            raw = self.ptx_container("1904", hsize=hsize)
            for offset in (20, 24, 32, 36, 48, 64, hsize - 1):
                bad = bytearray(raw)
                bad[16 + offset] ^= 1
                self.assertNotEqual(fatbin_entries(raw), fatbin_entries(bytes(bad)))
                with self.assertRaises(Invalid):
                    self.verify_ptx_fixture(raw, bytes(bad))

    def test_instruction_and_constant(self):
        raw = cubin_bytes()
        model = elf(raw, Names())
        for offset in (64, 80):
            bad = bytearray(raw)
            bad[offset] ^= 1
            self.assertNotEqual(model, elf(bytes(bad), Names()))

    def test_device_relocation_target_and_type(self):
        raw = cubin_bytes()
        model = elf(raw, Names())
        header = struct.unpack_from("<16sHHIQQQIHHHHHH", raw)
        section = struct.unpack_from("<IIQQQQIIQQ", raw, header[6] + 6 * 64)
        for info in ((3 << 32) | 43, (2 << 32) | 44):
            bad = bytearray(raw)
            struct.pack_into("<Q", bad, section[4] + 8, info)
            self.assertNotEqual(model, elf(bytes(bad), Names()))

    def test_private_metadata_indexes_forbid_symbol_reordering(self):
        raw = cubin_bytes()
        header = struct.unpack_from("<16sHHIQQQIHHHHHH", raw)
        symtab = struct.unpack_from("<IIQQQQIIQQ", raw, header[6] + 5 * 64)
        bad = bytearray(raw)
        offset = symtab[4]
        bad[offset + 48:offset + 72], bad[offset + 72:offset + 96] = raw[offset + 72:offset + 96], raw[offset + 48:offset + 72]
        self.assertNotEqual(elf(raw, Names()), elf(bytes(bad), Names()))

    def test_ptx_operand(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "kernel.ptx"
            text = ".version 8.5\n.target sm_90\n.address_size 64\n.visible .entry kernel() {\n mov.u32 %r0, 1;\n ret;\n}\n"
            path.write_text(text)
            first = ptx(path, Names())
            path.write_text(text.replace("%r0, 1", "%r0, 2"))
            self.assertNotEqual(first, ptx(path, Names()))

    def test_tool_failure(self):
        with self.assertRaises(Invalid):
            run(sys.executable, ["-c", "import sys;sys.exit(2)"], ".")

    def test_missing_extraction(self):
        header = struct.pack("<HHIQ", 2, 0x101, 64, 16) + bytes(12) + struct.pack("<I", 90) + bytes(8) + struct.pack("<Q", 0x41) + bytes(16)
        raw = struct.pack("<IHHQ", 0xba55ed50, 1, 16, 80) + header + bytes(16)
        self.assertEqual(len(fatbin_entries(raw)), 1)
        with tempfile.TemporaryDirectory() as temp, patch("strict_device.run", return_value=""):
            with self.assertRaises(Invalid):
                extract("unused", raw, Path(temp) / "out")

    def test_control_architecture_and_additional_header_bytes_are_preserved(self):
        raw = self.compressed_container()
        for offset in (16 + 20, 16 + 28, 16 + 32, 16 + 48):
            changed = bytearray(raw)
            changed[offset] ^= 1
            self.assertNotEqual(fatbin_entries(raw), fatbin_entries(bytes(changed)))
        header = bytearray(104)
        struct.pack_into("<HHIQ", header, 0, 1, 0x101, 104, 6)
        struct.pack_into("<Q", header, 40, 0x41)
        ptx_raw = struct.pack("<IHHQ", 0xba55ed50, 1, 16, 110) + header + b"hello\0"
        changed = bytearray(ptx_raw)
        changed[16 + 100] = 1
        self.assertNotEqual(fatbin_entries(ptx_raw), fatbin_entries(bytes(changed)))

    def test_fatbin_unknown_trailing(self):
        with self.assertRaises((Invalid, struct.error)):
            fatbin_entries(bytes(17))


class LambdaTests(unittest.TestCase):
    first = "<lambda_" + "1" * 32 + ">"
    second = "<lambda_" + "2" * 32 + ">"
    renamed_first = "<lambda_" + "a" * 32 + ">"
    renamed_second = "<lambda_" + "b" * 32 + ">"

    def pair(self):
        a, b = Coff(object_bytes()), Coff(object_bytes())
        a.symbols[4]["name"] = "func@" + self.first
        a.symbols[5]["name"] = "data@" + self.second
        b.symbols[4]["name"] = "func@" + self.renamed_first
        b.symbols[5]["name"] = "data@" + self.renamed_second
        return a, b

    def mapped(self, a, b):
        mapping = lambda_bijection(a, b, Names(), Names())
        return a.canonical(Names()), b.canonical(Names(lambda_map=mapping)), mapping

    def test_unique_full_definition_and_binding(self):
        x, y, mapping = self.mapped(*self.pair())
        self.assertEqual(mapping, {self.renamed_first: self.first, self.renamed_second: self.second})
        self.assertEqual(x, y)

    def test_changed_code_constant_binding_and_addend_fail(self):
        mutations = (
            lambda c: c.sections[1].update(raw=b"\x90" + c.sections[1]["raw"][1:]),
            lambda c: c.sections[2].update(raw=b"\0" + c.sections[2]["raw"][1:]),
            lambda c: c.sections[1].update(rel=[(4, 6, 4)]),
            lambda c: c.sections[1].update(rel=[(4, 5, 3)]),
            lambda c: c.sections[1].update(raw=c.sections[1]["raw"][:4] + b"\1\0\0\0" + c.sections[1]["raw"][8:]),
            lambda c: c.symbols[5].update(storage=3),
            lambda c: c.sections[1]["aux"].update(selection=4))
        for mutation in mutations:
            a, b = self.pair()
            mutation(b)
            with self.assertRaises(Invalid):
                self.mapped(a, b)

    def test_ambiguous_equal_definitions_rejected(self):
        a, b = self.pair()
        for c, one, two in ((a, self.first, self.second), (b, self.renamed_first, self.renamed_second)):
            c.symbols[4].update(name="func@" + one, section=1, value=0, storage=2, type=32)
            c.symbols[5].update(name="func@" + two, section=1, value=0, storage=2, type=32)
            c.sections[1]["rel"] = []
        with self.assertRaises(Invalid):
            self.mapped(a, b)

    def test_unchanged_guid_cannot_be_swapped(self):
        a, b = self.pair()
        b.symbols[4]["name"] = "func@" + self.second
        b.symbols[5]["name"] = "data@" + self.renamed_second
        with self.assertRaises(Invalid):
            self.mapped(a, b)

    def test_many_to_one_mapping_rejected(self):
        with self.assertRaises(Invalid):
            Names(lambda_map={self.renamed_first: self.first, self.renamed_second: self.first})

    def test_physical_symbol_order_does_not_choose_mapping(self):
        a, b = self.pair()
        b.symbols = dict(reversed(list(b.symbols.items())))
        # 两个定义在不同 section 内，物理迭代顺序不参与身份选择。
        x, y, _ = self.mapped(a, b)
        self.assertEqual(x, y)


class NamespaceTests(unittest.TestCase):
    def pair(self):
        a, b = LambdaTests().pair()
        for c, token in ((a, "?A0x11111111"), (b, "?A0xaaaaaaaa")):
            c.sections[2].update(name=".bss", aux=None, rel=[])
            c.symbols[2].update(name=".bss", aux=b"")
            c.symbols[5].update(name="?fp_copier@?$__nv_hdl_helper@Tag@" + token + "@@2P6APEAXPEAX@ZEA", storage=3, type=0)
        return a, b

    def mapped(self, a, b):
        namespace = namespace_bijection(a, b, "Example.cu")
        mapping = lambda_bijection(a, b, Names("Example.cu"), Names("Example.cu", namespace_map=namespace))
        return a.canonical(Names("Example.cu")), b.canonical(Names("Example.cu", mapping, namespace))

    def test_unique_namespace_then_lambda_complete_binding(self):
        a, b = self.pair()
        self.assertEqual(namespace_bijection(a, b, "Example.cu"), {"?A0xaaaaaaaa": "?A0x11111111"})
        self.assertEqual(*self.mapped(a, b))
        with self.assertRaises(Invalid):
            namespace_bijection(a, b, None)

    def test_namespace_definition_storage_content_and_presence_changes_fail(self):
        mutations = (
            lambda c: c.symbols[5].update(value=1),
            lambda c: c.symbols[5].update(storage=2),
            lambda c: c.symbols[5].update(section=0),
            lambda c: c.symbols[5].update(name="?other@?A0xaaaaaaaa@@"),
            lambda c: c.sections[2].update(raw=b"modified"),
            lambda c: c.sections[2].update(flags=0),
            lambda c: c.sections[1].update(rel=[(4, 5, 3)]))
        for mutate in mutations:
            a, b = self.pair()
            mutate(b)
            with self.assertRaises(Invalid):
                self.mapped(a, b)

    def test_namespace_does_not_hide_wrong_target_addend_or_instructions(self):
        mutations = (
            lambda c: c.sections[1].update(rel=[(4, 6, 4)]),
            lambda c: c.sections[1].update(raw=c.sections[1]["raw"][:4] + b"\1\0\0\0" + c.sections[1]["raw"][8:]),
            lambda c: c.sections[1].update(raw=b"\x90" + c.sections[1]["raw"][1:]))
        for mutate in mutations:
            a, b = self.pair()
            mutate(b)
            with self.assertRaises(Invalid):
                self.mapped(a, b)

    def test_multiple_namespaces_and_map_collision_rejected(self):
        a, b = self.pair()
        b.symbols[6].update(name="?fp_caller@?$__nv_hdl_helper@Tag@?A0xbbbbbbbb@@2P6AXPEAX@ZEA", section=2, storage=3, type=0)
        with self.assertRaises(Invalid):
            self.mapped(a, b)
        for mapping in ({"?A0xaaaaaaaa": "?A0x11111111", "?A0xbbbbbbbb": "?A0x11111111"}, {"?A0xBAD": "?A0x11111111"}):
            with self.assertRaises(Invalid):
                Names("Example.cu", namespace_map=mapping)

    def test_namespace_only_matches_complete_token(self):
        names = Names("Example.cu", namespace_map={"?A0xaaaaaaaa": "?A0x11111111"})
        self.assertEqual(names("x@?A0xaaaaaaaa@@"), "x@?A0x11111111@@")
        self.assertEqual(names("x@?A0xaaaaaaaa0@@"), "x@?A0xaaaaaaaa0@@")
        self.assertEqual(names("x@?A0xaaaaaaaa@other"), "x@?A0xaaaaaaaa@other")


class MappingGateTests(unittest.TestCase):
    def test_cross_object_anchor_conflict_and_many_to_one_fail(self):
        forward, reverse = {}, {}
        merge_identity_maps(forward, reverse, {"B1": "A1", "stable": "stable"})
        merge_identity_maps(forward, reverse, {"B1": "A1"})
        for mapping in ({"B1": "A2"}, {"B2": "A1"}, {"stable": "other"}):
            with self.assertRaises(Invalid):
                merge_identity_maps(dict(forward), dict(reverse), mapping)

    def test_three_snapshot_composition_and_coverage(self):
        pairs = {"S1-vs-A": {"identity_map": {"A1": "S1"}},
                 "A-vs-B": {"identity_map": {"B1": "A1"}},
                 "S1-vs-B": {"identity_map": {"B1": "S1"}}}
        check_composition(pairs)
        bad = copy.deepcopy(pairs)
        bad["S1-vs-B"]["identity_map"]["B1"] = "other"
        with self.assertRaises(Invalid):
            check_composition(bad)
        bad = copy.deepcopy(pairs)
        bad["A-vs-B"]["identity_map"] = {}
        with self.assertRaises(Invalid):
            check_composition(bad)

    def test_full_identity_inventory_includes_literal_only_tokens(self):
        coff = Coff(object_bytes())
        token = LambdaTests.first
        coff.sections[2]["raw"] = ("name@" + token + "\0").encode()
        self.assertIn(token, identity_tokens(coff))


class GateExecutionTests(unittest.TestCase):
    def execute(self, row=None, exception=None, inventory_error=False, lane="app", rows=None, encoding=None):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / "commands.json"
            count = 33 if lane == "app" else 58
            records = [dict(obj=f"{i}.cu.obj", cmd=f"nvcc -c D:\\repo\\{i}.cu -o {i}.cu.obj",
                            original=f"sccache nvcc -c D:\\repo\\{i}.cu -o {i}.cu.obj", launched=True) for i in range(count)]
            manifest.write_text(json.dumps(records))
            argv = ["strict_gate.py", "--tool", "unused", "--snap", temp, "--manifest", str(manifest), "--out", temp, "--lane", lane]
            buffer = io.BytesIO() if encoding else None
            output = io.TextIOWrapper(buffer, encoding=encoding, errors="strict") if encoding else io.StringIO()
            with patch.object(sys, "argv", argv), patch.object(strict_gate, "inventory") as inv, patch.object(strict_gate, "compare_object") as compare, redirect_stdout(output), redirect_stderr(output):
                if inventory_error:
                    inv.side_effect = inventory_error if isinstance(inventory_error, Exception) else Invalid("missing-object-fixture")
                else:
                    inv.return_value = {r["obj"]: root / r["obj"] for r in records}
                compare.side_effect = exception if rows is None else [(r, {}) for r in rows] * 3
                compare.return_value = (row, {})
                code = strict_gate.main()
            if encoding:
                self.assertEqual(output.encoding, "utf-8")
                self.assertEqual(output.errors, "strict")
                output.flush()
                log = buffer.getvalue().decode("utf-8", "strict")
                output.close()
            else:
                log = output.getvalue()
            return code, json.loads((root / "gate.json").read_text(encoding="utf-8")), log

    def test_cp1252_chinese_errors_do_not_interrupt_full_traversal(self):
        for lane, count in (("app", 33), ("wheel", 58)):
            with self.subTest(lane=lane):
                code, report, log = self.execute(exception=Invalid("物理解码不符"), lane=lane, encoding="cp1252")
                self.assertEqual(code, 1)
                self.assertFalse(report["equivalent"])
                self.assertNotIn("fatal", report)
                self.assertEqual(log.count("OBJECT FAIL"), count * 3)
                self.assertEqual(log.count("物理解码不符"), count * 3)
                self.assertEqual(len(report["pairs"]), 3)
                for pair in report["pairs"].values():
                    self.assertEqual(len(pair["objects"]), count)
                    self.assertTrue(all(row["error"] == "Invalid: 物理解码不符" for row in pair["objects"].values()))

    def test_cp1252_chinese_fatal_is_preserved_and_never_green(self):
        code, report, log = self.execute(inventory_error=Invalid("对象集合不符"), encoding="cp1252")
        self.assertEqual(code, 1)
        self.assertEqual(report["fatal"], "Invalid: 对象集合不符")
        self.assertIn("GATE FATAL Invalid: 对象集合不符", log)
        self.assertEqual(report["pairs"], {})

    def test_schema4_reports_exact_presentation_contract(self):
        row = dict(equivalent=True, device=dict(equivalent=True), identity_map={})
        code, report, _ = self.execute(row=row)
        self.assertEqual(code, 0)
        self.assertEqual(report["schema"], "nvcc-equivalence/4")
        self.assertEqual(report["controls"], "fully-decoded-physical-lengths")
        self.assertEqual(report["ptx_extraction"], "exact-or-windows-lf-expansion")

    def test_object_exception_is_reported_and_never_green(self):
        code, report, log = self.execute(exception=Invalid("binding-fixture"))
        self.assertEqual(code, 1)
        self.assertFalse(report["equivalent"])
        self.assertEqual(log.count("OBJECT FAIL"), 99)
        for pair in report["pairs"].values():
            self.assertEqual(len(pair["objects"]), 33)
            self.assertTrue(all("binding-fixture" in row["error"] for row in pair["objects"].values()))

    def test_fatbin_count_mismatch_is_visible_and_never_green(self):
        code, report, log = self.execute(row=dict(equivalent=True, device=None, identity_map={}))
        self.assertEqual(code, 1)
        self.assertFalse(report["equivalent"])
        self.assertEqual(log.count("fatbin 0 / 33 failed 0 FAIL"), 3)

    def test_wheel_requires_all_58_fatbins(self):
        # 只验证计数合同；mock 的绿色不作为真实对象等价证据。
        row = dict(equivalent=True, device=dict(equivalent=True), identity_map={})
        code, report, log = self.execute(row=row, lane="wheel")
        self.assertEqual(code, 0)
        self.assertTrue(report["equivalent"])
        self.assertEqual(log.count("fatbin 58 / 58 failed 0 PASS"), 3)
        for pair in report["pairs"].values():
            self.assertEqual(len(pair["objects"]), 58)

    def test_wheel_56_fatbins_cannot_pass(self):
        row = dict(equivalent=True, device=dict(equivalent=True), identity_map={})
        rows = [row] * 56 + [dict(row, device=None)] * 2
        code, report, log = self.execute(rows=rows, lane="wheel")
        self.assertEqual(code, 1)
        self.assertFalse(report["equivalent"])
        self.assertEqual(log.count("fatbin 56 / 58 failed 0 FAIL"), 3)

    def test_inventory_failure_is_reported_before_comparison(self):
        code, report, log = self.execute(inventory_error=True)
        self.assertEqual(code, 1)
        self.assertIn("missing-object-fixture", report["fatal"])
        self.assertIn("GATE FATAL", log)
        self.assertEqual(report["pairs"], {})


class NamesTests(unittest.TestCase):
    def mangled(self, pid):
        token = "_INTERNAL_4e739417_23_FixedRadiusSearchOps_cu_10ce75af_" + pid
        return "_ZN" + str(len(token)) + token + "4cuda3std3__45__cpo5beginE"

    def test_pid_without_swallowing_next_length(self):
        self.assertEqual(Names()(self.mangled("1904")), Names()(self.mangled("7352")))
        self.assertIn("4cuda", Names()(self.mangled("1904")))

    def test_collision(self):
        names = Names()
        names(self.mangled("1904"))
        with self.assertRaises(Invalid):
            names(self.mangled("7352"))

    def test_second_unit_id_requires_source_owner(self):
        a = self.mangled("1904")
        b = self.mangled("7352").replace("10ce75af", "8e8a4250")
        self.assertNotEqual(Names()(a), Names()(b))
        self.assertEqual(Names("FixedRadiusSearchOps.cu")(a), Names("FixedRadiusSearchOps.cu")(b))
        with self.assertRaises(Invalid):
            Names("WrongOwner.cu")(a)
        names = Names("FixedRadiusSearchOps.cu")
        names(a)
        with self.assertRaises(Invalid):
            names(b)

    def test_msvc_unit_declared_length(self):
        with self.assertRaises(Invalid):
            Names()("symbol@_GLOBAL__N__12345678_999_FixedRadiusSearchOps_cu_10ce75af@")

    def test_source_owner_from_actual_commands(self):
        for source in (r"D:\repo\FixedRadiusSearchOps.cu", '"D:\\repo with space\\FixedRadiusSearchOps.cu"'):
            record = dict(obj=r"target\FixedRadiusSearchOps.cu.obj", cmd="nvcc -x cu -c " + source + " -o target", original="sccache nvcc -x cu -c " + source + " -o target")
            self.assertEqual(source_owner(record), "FixedRadiusSearchOps.cu")
            for bad in (dict(record, obj="Wrong.cu.obj"), dict(record, cmd="nvcc -c a.cu -c b.cu"),
                        dict(record, cmd="nvcc -c Other.cu"), dict(record, cmd="nvcc -c response.rsp")):
                with self.assertRaises(Invalid):
                    source_owner(bad)

    def test_public_symbol_is_not_normalized(self):
        self.assertNotEqual(Names()("function123"), Names()("function456"))

    def test_unknown_token_fails(self):
        with self.assertRaises(Invalid):
            Names()("_ZN99_INTERNAL_12345678_invalid")


class RealSamples(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("STRICT_SAMPLE_ROOT") and os.environ.get("STRICT_CUDA_TOOL"), "真实样本由调用者显式提供")
    def test_nine_raw_objects(self):
        root = Path(os.environ["STRICT_SAMPLE_ROOT"])
        tool = os.environ["STRICT_CUDA_TOOL"]
        samples = {label: {path.name: path for path in (root / label).glob("*.obj")} for label in ("S1", "A", "B")}
        self.assertEqual(len(samples["S1"]), 3, "真实样本必须包含三个对象，不能空循环通过")
        self.assertEqual(samples["S1"].keys(), samples["A"].keys())
        self.assertEqual(samples["S1"].keys(), samples["B"].keys())
        with tempfile.TemporaryDirectory() as temp:
            for a, b in (("S1", "A"), ("A", "B"), ("S1", "B")):
                for path in samples[a].values():
                    report, errors = compare_object(tool, path, root / b / path.name, Path(temp) / (a + b) / path.name)
                    self.assertTrue(report["equivalent"], (path.name, errors))


if __name__ == "__main__":
    unittest.main()
