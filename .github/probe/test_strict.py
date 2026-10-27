"""严格校验器的破坏性负例，以及格式/布局正例；无第三方依赖。"""
import copy
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch
from strict_coff import Coff, Invalid, Names
from strict_device import elf, extract, fatbin_entries, ptx, run
from strict_gate import compare_object, inventory


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
        header = struct.pack("<HHIQ", 2, 0x101, 64, 16) + bytes(12) + struct.pack("<I", 90) + bytes(32)
        raw = struct.pack("<IHHQ", 0xba55ed50, 1, 16, 80) + header + bytes(16)
        self.assertEqual(len(fatbin_entries(raw)), 1)
        with tempfile.TemporaryDirectory() as temp, patch("strict_device.run", return_value=""):
            with self.assertRaises(Invalid):
                extract("unused", raw, Path(temp) / "out")

    def test_fatbin_unknown_trailing(self):
        with self.assertRaises((Invalid, struct.error)):
            fatbin_entries(bytes(17))


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
        with tempfile.TemporaryDirectory() as temp:
            for a, b in (("S1", "A"), ("A", "B"), ("S1", "B")):
                for path in (root / a).glob("*.obj"):
                    report, errors = compare_object(tool, path, root / b / path.name, Path(temp) / (a + b) / path.name)
                    self.assertTrue(report["equivalent"], (path.name, errors))


if __name__ == "__main__":
    unittest.main()
