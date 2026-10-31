"""Extract pinned upstream registrations without importing inference frameworks.

This is a scope and declaration checker, not a model correctness/release gate.
Outputs are append-only receipts; missing consumers never establish alignment.
"""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile

import yaml

ANYLABELING_REVISION = "cd26942a007b6ce7bd521de801a6cfd455d7fd9b"
LLAMA_REVISION = "707616feb5f379e99ff7442680a55b4a1704c7c9"
LLAMA_ARCHIVE_SHA256 = "a532d0601f65cee41446cfbdc6872b98e689aa727eb366c6b6a61a70e9965759"


class UniqueYamlLoader(yaml.SafeLoader):
    pass


def _yaml_mapping(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"duplicate YAML key: {key}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


UniqueYamlLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _yaml_mapping)


def _digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _literal_maps(text):
    maps = {}
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names = [node.target.id]
        else:
            continue
        if not isinstance(node.value, ast.Dict):
            continue
        try:
            keys = [ast.literal_eval(key) for key in node.value.keys]
            if len(keys) != len(set(keys)):
                raise ValueError(f"duplicate dictionary key in {names}")
            value = ast.literal_eval(node.value)
        except (TypeError, SyntaxError):
            continue
        except ValueError as error:
            if "duplicate" in str(error):
                raise
            continue
        for name in names:
            maps[name] = value
    return maps


def _model_loaders(text):
    function = next((node for node in ast.parse(text).body
                     if isinstance(node, ast.FunctionDef) and node.name == "_get_model_class"), None)
    if function is None:
        raise ValueError("model manager has no _get_model_class loader registry")
    rows = {}
    for node in ast.walk(function):
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
            continue
        test = node.test
        if not (isinstance(test.left, ast.Name) and test.left.id == "model_type"
                and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq)
                and isinstance(test.comparators[0], ast.Constant)):
            raise ValueError("unrecognized loader registry condition")
        family = test.comparators[0].value
        imports = [item for item in node.body if isinstance(item, ast.ImportFrom) and item.level == 1]
        if len(imports) != 1 or len(imports[0].names) != 1:
            raise ValueError(f"unrecognized loader import for {family}")
        if family in rows:
            raise ValueError(f"duplicate loader family: {family}")
        imported = imports[0].names[0]
        returns = [item.value for item in node.body if isinstance(item, ast.Return)]
        if (len(returns) != 1 or not isinstance(returns[0], ast.Name)
                or returns[0].id != (imported.asname or imported.name)):
            raise ValueError(f"loader return does not match its import: {family}")
        rows[family] = {"family": family, "module": imports[0].module,
                        "class": imported.name}
    if not rows:
        raise ValueError("empty loader registry")
    return rows


def extract_anylabeling(read):
    manager_path = "anylabeling/services/auto_labeling/model_manager.py"
    manager = read(manager_path)
    loaders = _model_loaders(manager)
    registry_path = "anylabeling/configs/models.yaml"
    registry = yaml.load(read(registry_path), Loader=UniqueYamlLoader)
    if not isinstance(registry, list) or not registry:
        raise ValueError("empty models.yaml registry")
    models, seen, seen_configs = [], set(), set()
    for registration in registry:
        name, resource = registration["model_name"], registration["config_file"]
        if not isinstance(name, str) or not name:
            raise ValueError("invalid registered model_name")
        if name in seen:
            raise ValueError(f"duplicate registered model: {name}")
        seen.add(name)
        if not isinstance(resource, str) or not re.fullmatch(r":/[A-Za-z0-9_-]+\.yaml", resource):
            raise ValueError(f"invalid config_file: {resource}")
        if resource in seen_configs:
            raise ValueError(f"duplicate registered config_file: {resource}")
        seen_configs.add(resource)
        config_path = "anylabeling/configs/auto_labeling/" + resource[2:]
        config_text = read(config_path)
        config = yaml.load(config_text, Loader=UniqueYamlLoader)
        if not isinstance(config, dict) or config.get("type") not in loaders:
            raise ValueError(f"registered model has no loader: {name}")
        loader = loaders[config["type"]]
        module_path = "anylabeling/services/auto_labeling/" + loader["module"].replace(".", "/") + ".py"
        module = read(module_path)
        assets = {key: value for key, value in config.items() if key.endswith("_path")}
        identity_keys = {"type", "name", "provider", "display_name", "license"}
        models.append({
            "id": "anylabeling/" + name, "family": config["type"],
            "config_name": config.get("name"), "config_file": config_path,
            "config_sha256": _digest(config_text), "loader": loader,
            "module_file": module_path, "module_sha256": _digest(module),
            "assets": assets,
            "defaults": {key: value for key, value in config.items()
                         if key not in identity_keys and key not in assets},
            "task_tokens": _literal_maps(module).get("TASK_MAPPING", {}),
            "license": config.get("license"), "license_status": "NEEDS_REVIEW",
            "status": "UNVERIFIED", "current_owner": None, "consumer": None,
            "contract": None, "numeric_gate": None, "performance_gate": None,
            "catalog": [], "publication": [], "ui": [], "automation": [], "receipts": [],
        })
    active = {row["family"] for row in models}
    return {"models": models,
            "custom_loaders": [row for family, row in loaders.items() if family not in active],
            "registry_file": registry_path, "registry_sha256": _digest(read(registry_path)),
            "manager_file": manager_path, "manager_sha256": _digest(manager)}


def extract_converters(text):
    maps = _literal_maps(text)
    rows = []
    for registry, role in [("TEXT_MODEL_MAP", "text"), ("MMPROJ_MODEL_MAP", "mmproj")]:
        if registry not in maps:
            raise ValueError(f"missing converter registry: {registry}")
        for architecture, module in maps[registry].items():
            rows.append({"role": role, "architecture": architecture, "module": module})
    return rows


def extract_converter_classes(text):
    rows = []
    for node in ast.parse(text).body:
        if not isinstance(node, ast.ClassDef):
            continue
        names, examples = [], []
        for decorator in node.decorator_list:
            if (not isinstance(decorator, ast.Call) or not isinstance(decorator.func, ast.Attribute)
                    or not isinstance(decorator.func.value, ast.Name)
                    or decorator.func.value.id != "ModelBase"
                    or decorator.func.attr not in {"register", "example"}):
                continue
            values = [ast.literal_eval(value) for value in decorator.args]
            if not all(isinstance(value, str) for value in values):
                raise ValueError("nonliteral converter registration")
            (names if decorator.func.attr == "register" else examples).extend(values)
        if names:
            rows.append({"class": node.name, "bases": [ast.unparse(base) for base in node.bases],
                         "architectures": names, "examples": examples})
    return rows


def extract_mtmd_sources(text):
    source = re.sub(r"#[^\n]*", "", text)
    library = re.search(r"\badd_library\s*\(\s*mtmd\s+([^)]*)\)", source, flags=re.IGNORECASE)
    if not library:
        raise ValueError("missing mtmd library source registration")
    return ["tools/mtmd/" + name for name in library[1].split()
            if name.startswith("models/") and name.endswith(".cpp")]


def _without_cpp_comments(text):
    pattern = r'R"(?P<delimiter>[^\s()\\]{0,16})\([\s\S]*?\)(?P=delimiter)"|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|//[^\n]*|/\*[\s\S]*?\*/'
    return re.sub(pattern, lambda match: " " if match.group().startswith(("//", "/*")) else match.group(), text)


def extract_architectures(text):
    source = _without_cpp_comments(text)
    registry = re.search(r"\bLLM_ARCH_NAMES\s*=\s*\{([\s\S]*?)\n\};", source)
    if not registry:
        raise ValueError("missing architecture registry")
    rows = [{"symbol": symbol, "architecture": name,
             "loader_kind": "quantize-only" if symbol == "LLM_ARCH_CLIP" else "model"}
            for symbol, name in re.findall(r'\{\s*(LLM_ARCH_\w+)\s*,\s*"([^"\n]+)"\s*\}', registry[1])]
    if not rows or len({row["architecture"] for row in rows}) != len(rows):
        raise ValueError("empty or duplicate architecture registry")
    return rows


def extract_cmake_subdirectories(text):
    stack, rows = [], []
    for command, argument in re.findall(
            r"\b(if|elseif|else|endif|add_subdirectory)\s*\(([^)]*)\)",
            re.sub(r"#[^\n]*", "", text), flags=re.IGNORECASE):
        command, argument = command.lower(), " ".join(argument.split())
        if command == "if":
            stack.append({"previous": [argument], "active": argument})
        elif command in {"else", "elseif"}:
            if not stack:
                raise ValueError("unbalanced CMake conditional")
            frame = stack[-1]
            exclusion = "NOT (" + " OR ".join("(" + item + ")" for item in frame["previous"]) + ")"
            frame["active"] = exclusion if command == "else" else exclusion + " AND (" + argument + ")"
            if command == "elseif":
                frame["previous"].append(argument)
        elif command == "endif":
            if not stack:
                raise ValueError("unbalanced CMake conditional")
            stack.pop()
        else:
            rows.append({"directory": argument.split()[0].strip('"'),
                         "conditions": [frame["active"] for frame in stack]})
    if stack:
        raise ValueError("unbalanced CMake conditional")
    return rows


def extract_routes(text):
    routes = {}
    for method, path, handler in re.findall(
            r'\bctx_http\.(get|post|put|patch|del)\s*\(\s*"([^"\n]+)"\s*,\s*ex_wrapper\(\s*([\w.]+)\s*\)',
            _without_cpp_comments(text)):
        method = "DELETE" if method == "del" else method.upper()
        row = routes.setdefault((method, path), {"method": method, "path": path, "handlers": []})
        if handler not in row["handlers"]:
            row["handlers"].append(handler)
    return list(routes.values())


def compare_coverage(expected, declared):
    errors = []
    counts = Counter(row["id"] for row in declared)
    errors.extend(f"duplicate declaration: {name}" for name, count in counts.items() if count > 1)
    actual = {row["id"]: row for row in declared}
    wanted = {row["id"]: row for row in expected}
    errors.extend(f"missing declaration: {name}" for name in sorted(wanted.keys() - actual.keys()))
    errors.extend(f"extra declaration: {name}" for name in sorted(actual.keys() - wanted.keys()))
    for name in wanted.keys() & actual.keys():
        source, candidate = wanted[name], actual[name]
        if candidate.get("family") != source["family"]:
            errors.append(f"family mismatch: {name}")
        if candidate.get("assets") != source["assets"]:
            errors.append(f"checkpoint or asset role mismatch: {name}")
        if json.dumps(candidate.get("defaults"), sort_keys=True) != json.dumps(source["defaults"], sort_keys=True):
            errors.append(f"default drift: {name}")
        if candidate.get("config_sha256") != source["config_sha256"]:
            errors.append(f"config identity mismatch: {name}")
        if not isinstance(candidate.get("consumer"), str) or not candidate["consumer"].strip():
            errors.append(f"missing inference consumer: {name}")
    return errors


def derive_manifest_coverage(expected, manifest, aliases=()):
    """Match pinned source declarations without claiming an executable consumer.

    input_ids declare dependencies, not actual inference. Preserve matching
    scenarios separately; consumer remains unverified until executable evidence
    is supplied. Aliases additionally require matching local bytes and size.
    """
    pins = {}
    for asset in manifest["input_assets"]:
        digest = asset.get("sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            continue
        previous = pins.setdefault(asset["url"], (asset["id"], digest))
        if previous[1] != digest:
            raise ValueError(f"conflicting source pins for {asset['url']}")
    by_digest = {digest: asset_id for asset_id, digest in pins.values()}
    for alias in aliases:
        if alias["url"] in pins:
            if pins[alias["url"]][1] != alias["sha256"]:
                raise ValueError(f"conflicting alias digest for {alias['url']}")
        elif alias["sha256"] in by_digest and alias.get("verified_file"):
            path = Path(alias["verified_file"])
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != alias["sha256"] or path.stat().st_size != alias.get("bytes"):
                raise ValueError(f"alias bytes or SHA-256 mismatch for {alias['url']}")
            pins[alias["url"]] = (by_digest[alias["sha256"]], alias["sha256"])
    declared = []
    for row in expected:
        roles, local = {}, []
        for role, value in sorted(row["assets"].items()):
            if isinstance(value, str) and value.startswith(("https://", "http://")):
                roles[role] = pins.get(value, (None,))[0]
            else:
                local.append(role)
        wanted = set(roles.values())
        complete, partial = [], []
        for scenario in manifest["scenarios"]:
            inputs = set(scenario.get("input_ids", []))
            if roles and None not in wanted and wanted <= inputs:
                complete.append(scenario["id"])
            elif inputs & (wanted - {None}):
                partial.append(scenario["id"])
        declared.append(dict(row, consumer=None,
                             declared_consumer="manifest:" + sorted(complete)[0] if complete else None,
                             binding={"roles": roles, "unverified_local_roles": local,
                                      "scenarios": sorted(complete), "partial_scenarios": sorted(partial)}))
    return declared


class GitSource:
    def __init__(self, root, revision):
        self.root, self.revision = root, revision
        self.files = subprocess.check_output(
            ["git", "-C", str(root), "ls-tree", "-r", "--name-only", revision], text=True).splitlines()
        self.digests = {}

    def read(self, path):
        if path not in self.files:
            raise ValueError(f"missing pinned source: {path}")
        data = subprocess.check_output(["git", "-C", str(self.root), "show", self.revision + ":" + path])
        self.digests[path] = hashlib.sha256(data).hexdigest()
        return data.decode("utf-8")


class ArchiveSource:
    def __init__(self, path):
        self.digests = {}
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != LLAMA_ARCHIVE_SHA256:
                raise ValueError("llama archive SHA-256 does not match the approved pin")
            stream.seek(0)
            with tarfile.open(fileobj=stream, mode="r:gz") as archive:
                prefix = "llama.cpp-" + LLAMA_REVISION + "/"
                self.contents = {}
                for member in archive.getmembers():
                    if not member.isfile() or not member.name.startswith(prefix):
                        continue
                    relative = member.name[len(prefix):]
                    if PurePosixPath(relative).is_absolute() or ".." in PurePosixPath(relative).parts:
                        raise ValueError("unsafe archive member")
                    if relative in self.contents:
                        raise ValueError(f"duplicate archive member: {relative}")
                    if relative.endswith((".py", ".cpp", ".h", ".cmake", "CMakeLists.txt", ".md")):
                        self.contents[relative] = archive.extractfile(member).read()
        self.files = list(self.contents)

    def read(self, path):
        data = self.contents[path]
        self.digests[path] = hashlib.sha256(data).hexdigest()
        return data.decode("utf-8")


def extract_llama(source):
    read = source.read
    architectures = extract_architectures(read("src/llama-arch.cpp"))
    converters = extract_converters(read("conversion/__init__.py"))
    converter_classes = []
    for module in dict.fromkeys(row["module"] for row in converters):
        path = "conversion/" + module + ".py"
        text = read(path)
        converter_classes.extend(dict(row, source_file=path, source_sha256=_digest(text))
                                 for row in extract_converter_classes(text))
    for row in converters:
        path = "conversion/" + row["module"] + ".py"
        row["source_file"] = path
        row["source_sha256"] = source.digests[path]
    tools = extract_cmake_subdirectories(read("tools/CMakeLists.txt"))
    for row in tools:
        path = "tools/" + row["directory"] + "/CMakeLists.txt"
        row["cmake_file"] = path
        row["cmake_sha256"] = _digest(read(path))
    mtmd_text = read("tools/mtmd/CMakeLists.txt")
    mtmd_sources = extract_mtmd_sources(mtmd_text)
    for path in mtmd_sources:
        read(path)
    return {"architectures": architectures, "converters": converters,
            "converter_classes": converter_classes, "tool_directories": tools,
            "routes": extract_routes(read("tools/server/server.cpp")),
            "mtmd_model_sources": mtmd_sources,
            "tests": [path for path in sorted(source.files) if path.startswith("tests/")
                      and path.endswith((".cpp", ".py", "CMakeLists.txt"))],
            "tts_source_sha256": _digest(read("tools/tts/tts.cpp")),
            "mtmd_cmake_sha256": _digest(mtmd_text)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anylabeling", type=Path, required=True)
    parser.add_argument("--llama-archive", type=Path, required=True)
    declarations_source = parser.add_mutually_exclusive_group()
    declarations_source.add_argument("--coverage", type=Path)
    declarations_source.add_argument("--manifest", type=Path,
                                     help="Match declared checkpoint dependencies; not inference evidence")
    parser.add_argument("--source-aliases", type=Path,
                        help="JSON with identical_source_aliases verified by downloaded bytes")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Never overwrite an existing inventory receipt")
    any_source = GitSource(args.anylabeling, ANYLABELING_REVISION)
    llama_source = ArchiveSource(args.llama_archive)
    report = {"schema": "acloudviewer.upstream_inventory/1",
              "anylabeling_revision": ANYLABELING_REVISION, "llama_revision": LLAMA_REVISION,
              "llama_archive_sha256": LLAMA_ARCHIVE_SHA256,
              "anylabeling": extract_anylabeling(any_source.read), "llama": extract_llama(llama_source),
              "complete_upstream_parity": False,
              "scope": "Pinned registrations and structural declarations only; not inference/release qualification"}
    report["source_sha256"] = {"anylabeling": any_source.digests, "llama": llama_source.digests}
    if args.manifest:
        manifest_bytes = args.manifest.read_bytes()
        aliases = (json.loads(args.source_aliases.read_text())["identical_source_aliases"]
                   if args.source_aliases else [])
        declarations = derive_manifest_coverage(report["anylabeling"]["models"], json.loads(manifest_bytes),
                                                aliases)
        report["coverage_source"] = {"manifest": args.manifest.name,
                                     "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                                     "source_aliases": aliases}
        report["coverage"] = [{"id": row["id"], "consumer": row["consumer"],
                               "declared_consumer": row["declared_consumer"], "binding": row["binding"]}
                              for row in declarations]
        report["declared_checkpoint_matches"] = sum(row["declared_consumer"] is not None
                                                    for row in declarations)
    else:
        declarations = json.loads(args.coverage.read_text())["models"] if args.coverage else []
    report["coverage_errors"] = compare_coverage(report["anylabeling"]["models"], declarations)
    report["status"] = "PARTIAL" if report["coverage_errors"] else "DECLARATIONS_MATCH_NOT_RELEASE"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=True)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "models": len(report["anylabeling"]["models"]),
                      "families": len({row["family"] for row in report["anylabeling"]["models"]}),
                      "architectures": len(report["llama"]["architectures"]),
                      "coverage_errors": len(report["coverage_errors"])}))
    return 1 if report["coverage_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
