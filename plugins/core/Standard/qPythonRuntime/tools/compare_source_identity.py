"""Classify byte changes with libclang tokens and ordered preprocessing directives.

This does not compile a translation unit against the application SDK. Include
reordering and preprocessing-sensitive changes require fresh build/runtime
verification; a matching lexical identity is never a runtime receipt.
"""
import argparse
import ctypes as C
import ctypes.util
import hashlib
import json
from pathlib import Path
import re


class CXString(C.Structure):
    _fields_ = [("data", C.c_void_p), ("flags", C.c_uint)]


class CXLocation(C.Structure):
    _fields_ = [("ptrs", C.c_void_p * 2), ("data", C.c_uint)]


class CXRange(C.Structure):
    _fields_ = [("ptrs", C.c_void_p * 2), ("begin", C.c_uint), ("end", C.c_uint)]


class CXToken(C.Structure):
    _fields_ = [("data", C.c_uint * 4), ("ptr", C.c_void_p)]


class CXUnsavedFile(C.Structure):
    _fields_ = [("filename", C.c_char_p), ("contents", C.c_char_p), ("length", C.c_ulong)]


class ClangTokenizer:
    def __init__(self, library):
        self.lib = C.CDLL(library)
        signatures = {
            "clang_createIndex": (C.c_void_p, [C.c_int, C.c_int]),
            "clang_disposeIndex": (None, [C.c_void_p]),
            "clang_parseTranslationUnit2": (C.c_int, [C.c_void_p, C.c_char_p, C.POINTER(C.c_char_p), C.c_int,
                                                      C.POINTER(CXUnsavedFile), C.c_uint, C.c_uint, C.POINTER(C.c_void_p)]),
            "clang_disposeTranslationUnit": (None, [C.c_void_p]),
            "clang_getFile": (C.c_void_p, [C.c_void_p, C.c_char_p]),
            "clang_getLocationForOffset": (CXLocation, [C.c_void_p, C.c_void_p, C.c_uint]),
            "clang_getRange": (CXRange, [CXLocation, CXLocation]),
            "clang_tokenize": (None, [C.c_void_p, CXRange, C.POINTER(C.POINTER(CXToken)), C.POINTER(C.c_uint)]),
            "clang_disposeTokens": (None, [C.c_void_p, C.POINTER(CXToken), C.c_uint]),
            "clang_getTokenKind": (C.c_uint, [CXToken]),
            "clang_getTokenSpelling": (CXString, [C.c_void_p, CXToken]),
            "clang_getCString": (C.c_char_p, [CXString]),
            "clang_disposeString": (None, [CXString]),
            "clang_getTokenExtent": (CXRange, [C.c_void_p, CXToken]),
            "clang_getRangeStart": (CXLocation, [CXRange]),
            "clang_getRangeEnd": (CXLocation, [CXRange]),
            "clang_getSpellingLocation": (None, [CXLocation, C.POINTER(C.c_void_p), C.POINTER(C.c_uint),
                                                  C.POINTER(C.c_uint), C.POINTER(C.c_uint)]),
            "clang_getClangVersion": (CXString, []),
        }
        for name, (result, arguments) in signatures.items():
            function = getattr(self.lib, name)
            function.restype, function.argtypes = result, arguments
        self.version = self.string(self.lib.clang_getClangVersion())

    def string(self, value):
        try:
            return self.lib.clang_getCString(value).decode("utf-8")
        finally:
            self.lib.clang_disposeString(value)

    def offset(self, location):
        offset = C.c_uint()
        self.lib.clang_getSpellingLocation(location, None, None, None, C.byref(offset))
        return offset.value

    def tokens(self, text):
        data = text.encode("utf-8")
        name = b"source-identity.cpp"
        index = self.lib.clang_createIndex(0, 0)
        unit, tokens, count = C.c_void_p(), C.POINTER(CXToken)(), C.c_uint()
        unsaved = CXUnsavedFile(name, data, len(data))
        args = (C.c_char_p * 4)(b"-std=c++17", b"-nostdinc", b"-nostdinc++", b"-ferror-limit=0")
        try:
            error = self.lib.clang_parseTranslationUnit2(index, name, args, 4, C.byref(unsaved),
                                                         1, 0x240, C.byref(unit))
            if error or not unit:
                raise RuntimeError(f"libclang could not create lexical translation unit: {error}")
            file = self.lib.clang_getFile(unit, name)
            if not file:
                raise RuntimeError("libclang did not retain the complete source buffer")
            begin = self.lib.clang_getLocationForOffset(unit, file, 0)
            end = self.lib.clang_getLocationForOffset(unit, file, len(data))
            self.lib.clang_tokenize(unit, self.lib.clang_getRange(begin, end), C.byref(tokens), C.byref(count))
            result = []
            for index_token in range(count.value):
                token = tokens[index_token]
                kind = self.lib.clang_getTokenKind(token)
                if kind == 4:
                    continue
                extent = self.lib.clang_getTokenExtent(unit, token)
                result.append({"kind": kind, "spelling": self.string(self.lib.clang_getTokenSpelling(unit, token)),
                               "start": self.offset(self.lib.clang_getRangeStart(extent)),
                               "end": self.offset(self.lib.clang_getRangeEnd(extent))})
            return result
        finally:
            if tokens:
                self.lib.clang_disposeTokens(unit, tokens, count)
            if unit:
                self.lib.clang_disposeTranslationUnit(unit)
            self.lib.clang_disposeIndex(index)


def source_identity(text, tokenizer):
    tokens = tokenizer.tokens(text)
    data = text.encode("utf-8")
    logical_lines = {}
    offset, logical = 0, 0
    for line in data.splitlines(keepends=True):
        for position in range(offset, offset + len(line)):
            logical_lines[position] = logical
        offset += len(line)
        if not line.endswith((b"\\\n", b"\\\r\n")):
            logical += 1
    groups = {}
    for token in tokens:
        groups.setdefault(logical_lines.get(token["start"], logical), []).append(token)
    directives, body = [], []
    sensitive = False
    for group in groups.values():
        spellings = [token["spelling"] for token in group]
        if spellings[0] in {"#", "%:"}:
            directive = {"tokens": spellings, "body_token_offset": len(body)}
            if len(group) > 3 and spellings[1] == "define" and spellings[3] == "(":
                gap = data[group[2]["end"]:group[3]["start"]]
                directive["function_like"] = re.sub(rb"\\\r?\n", b"", gap) == b""
            directives.append(directive)
            sensitive |= (len(spellings) > 1 and spellings[1] in {"include", "include_next"})
            sensitive |= any(value in {"#", "##", "%:", "%:%:"} for value in spellings[1:])
        else:
            body.extend((token["kind"], token["spelling"]) for token in group)
    sensitive |= any(token["spelling"] in {
        "__LINE__", "__FILE__", "__BASE_FILE__", "__COUNTER__", "__DATE__", "__TIME__", "__TIMESTAMP__", "_Pragma"}
        for token in tokens)
    return {"tokens": [(token["kind"], token["spelling"]) for token in tokens],
            "body": body, "directives": directives, "sensitive": sensitive}


def compare_sources(old, new, tokenizer):
    before, after = source_identity(old, tokenizer), source_identity(new, tokenizer)
    directives_equal, body_equal = before["directives"] == after["directives"], before["body"] == after["body"]
    if not body_equal:
        classification = "TOKEN_CHANGE"
    elif not directives_equal:
        classification = "DIRECTIVE_CHANGE_REQUIRES_REBUILD"
    elif old != new and (before["sensitive"] or after["sensitive"]):
        classification = "PREPROCESSING_SENSITIVE_REQUIRES_REBUILD"
    else:
        classification = "TOKEN_AND_DIRECTIVE_EQUIVALENT"
    digest = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    return {"classification": classification, "tokens_equal": before["tokens"] == after["tokens"],
            "body_tokens_equal": body_equal, "directives_equal": directives_equal,
            "old_token_sha256": digest(before["tokens"]), "new_token_sha256": digest(after["tokens"]),
            "old_directives": before["directives"], "new_directives": after["directives"],
            "runtime_verified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery", type=Path, required=True)
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--libclang", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Never overwrite an existing source identity receipt")
    recovery = json.loads(args.recovery.read_text())
    tokenizer = ClangTokenizer(args.libclang)
    rows = []
    for row in recovery["rows"]:
        if not row["recovered_file"]:
            raise ValueError(f"Missing original bytes for {row['file']}")
        path = args.plugin / row["file"]
        old, new = Path(row["recovered_file"]).read_bytes(), path.read_bytes()
        if (hashlib.sha256(old).hexdigest() != row["old_sha256"]
                or hashlib.sha256(new).hexdigest() != row["current_sha256"]):
            raise ValueError(f"Source identity changed during recovery: {row['file']}")
        result = compare_sources(old.decode("utf-8"), new.decode("utf-8"), tokenizer)
        rows.append(dict(row, **result))
    report = {"schema": "acloudviewer.pythonruntime.source_identity/1", "libclang_version": tokenizer.version,
              "recovery_receipt_sha256": hashlib.sha256(args.recovery.read_bytes()).hexdigest(),
              "scope": "Lexical tokens and ordered directives only; fresh SDK preprocessing/build/runtime required",
              "status": "IDENTITY_CLASSIFIED_NOT_RUNTIME_VERIFIED", "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "rows": len(rows),
                      "body_token_changes": sum(not row["body_tokens_equal"] for row in rows),
                      "directive_changes": sum(not row["directives_equal"] for row in rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
