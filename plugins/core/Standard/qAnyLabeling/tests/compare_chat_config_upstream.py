"""Execute the pinned source configuration and provider contract against Qt fixtures."""
import argparse
import ast
import hashlib
import json
import threading
from copy import deepcopy
from difflib import SequenceMatcher
from types import SimpleNamespace
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--upstream", type=Path, required=True)
parser.add_argument("--artifacts", type=Path, required=True)
args = parser.parse_args()
source = args.upstream / "anylabeling/views/labeling/chatbot/config.py"
provider = source.with_name("provider.py")
dropdown = source.parent.parent / "widgets/model_dropdown_widget.py"
constants = {}
for path in (source, provider):
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and len(
                node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                constants[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
fixtures = {
    name: args.artifacts / name
    for name in ("chat-config-defaults-native.json",
                 "chat-config-roundtrip-native.json",
                 "chat-capabilities-native.json",
                 "chat-model-search-native.json",
                 "chat-model-favorite-native.json",
                 "chat-model-refresh-native.json",
                 "chat-model-geometry-native.json")
}
defaults = json.loads(fixtures["chat-config-defaults-native.json"].read_text())
upstream_providers = constants["DEFAULT_PROVIDERS_DATA"]
local_provider = {
    "api_address": "aicore://llm",
    "api_key": "",
    "api_docs_url": None,
    "api_key_url": None,
    "model_docs_url": None,
}
assert "local" not in upstream_providers
assert set(defaults["providers"]) == set(upstream_providers) | {"local"}
for name, expected in upstream_providers.items():
    assert defaults["providers"][name] == expected, name
assert defaults["providers"]["local"] == local_provider
assert defaults["models"] == dict(
    settings=constants["DEFAULT_SETTINGS"],
    models_data={},
    supported_vision_models=constants["SUPPORTED_VISION_MODELS"])
namespace = dict(constants)
namespace["MODELS_CONFIG_LOCK"] = threading.Lock()
functions = {
    "save_model_selection", "model_supports_vision",
    "_metadata_supports_vision", "_explicit_vision_value",
    "_modalities_support_vision", "_modality_string_supports_vision",
    "_model_name_suggests_vision"
}
nodes = [
    node for node in ast.parse(provider.read_text()).body
    if isinstance(node, ast.FunctionDef) and node.name in functions
]
assert len(nodes) == len(functions)
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(provider), "exec"),
     namespace)
capabilities = json.loads(fixtures["chat-capabilities-native.json"].read_text())
for case in capabilities:
    expected = namespace["model_supports_vision"](
        case["id"], case["metadata"], constants["SUPPORTED_VISION_MODELS"])
    assert case["vision"] == expected, (case, expected)
roundtrip = json.loads(
    fixtures["chat-config-roundtrip-native.json"].read_text())
original = deepcopy(roundtrip["models"])
for entries in original["models_data"].values():
    for entry in entries.values():
        entry["selected"] = False
original["settings"]["provider"] = "custom"
original["settings"]["model_id"] = "my-private-model"
saved = []
namespace.update(get_models_config_path=lambda: "fixture",
                 load_json=lambda _: original,
                 save_json=lambda value, _: saved.append(deepcopy(value)))
namespace["save_model_selection"]("ollama", "vision-test")
assert saved[-1] == roundtrip["models"]
assert roundtrip["providers"]["custom"]["model_name"] == "my-private-model"
assert roundtrip["providers"]["ollama"]["vendor"] == "preserve-provider"
assert roundtrip["models"]["settings"]["temperature"] == 13
assert roundtrip["models"]["settings"]["max_length"] == 9876543
assert roundtrip["models"]["settings"]["system_prompt"] is None


# Execute the source popup methods with presentation-only stand-ins. Search
# decisions and config mutations remain the actual pinned upstream code.
class Widget:

    def __init__(self, *args):
        self.visible = True

    def setVisible(self, value):
        self.visible = value

    def setAlignment(self, value):
        pass

    def setStyleSheet(self, value):
        pass


class Layout:

    def count(self):
        return 0

    def addWidget(self, widget):
        pass


popup_class = next(
    node for node in ast.parse(dropdown.read_text()).body
    if isinstance(node, ast.ClassDef) and node.name == "ModelDropdown")
methods = [
    node for node in popup_class.body if isinstance(node, ast.FunctionDef) and
    node.name in {"filter_models", "save_models_data"}
]
assert len(methods) == 2
popup_namespace = dict(
    namespace,
    SequenceMatcher=SequenceMatcher,
    QLabel=Widget,
    QFrame=Widget,
    ProviderSection=Widget,
    Qt=SimpleNamespace(AlignmentFlag=SimpleNamespace(AlignCenter=0)),
    get_theme=lambda: {"text_secondary": "#888888"})
exec(compile(ast.Module(body=methods, type_ignores=[]), str(dropdown), "exec"),
     popup_namespace)
search_cases = json.loads(fixtures["chat-model-search-native.json"].read_text())
for case in search_cases:
    expected = SequenceMatcher(None, case["query"], case["name"]).ratio()
    assert abs(case["ratio"] - expected) < 1e-14, (case, expected)
    item = Widget()
    popup = SimpleNamespace(container_layout=Layout(),
                            model_items={case["name"]: item})
    popup_namespace["filter_models"](popup, case["query"])
    assert item.visible == case["match"], case
favorite = json.loads(fixtures["chat-model-favorite-native.json"].read_text())
favorite_saved = []
popup_namespace.update(
    load_json=lambda _: deepcopy(favorite["before"]),
    save_json=lambda value, _: favorite_saved.append(deepcopy(value)))
popup_namespace["save_models_data"](SimpleNamespace(), favorite["provider"],
                                    favorite["model"], False)
assert favorite_saved == [favorite["after"]]

# Execute both original functions with a controlled clock and transport.
# The source reset keeps timer=0; verify behavior after the first interval,
# rather than silently substituting a conventional rolling cooldown.
utils = source.with_name("utils.py")
tracker_class = next(node for node in ast.parse(utils.read_text()).body
                     if isinstance(node, ast.ClassDef) and
                     node.name == "EventTracker")
refresh_function = next(node for node in ast.parse(provider.read_text()).body
                        if isinstance(node, ast.FunctionDef) and
                        node.name == "get_models_data")
clock = [0.0]
refreshes = []
refresh_document = {}


class RefreshThread:
    def __init__(self, *, target, args):
        self.target, self.args = target, args

    def start(self):
        self.target(*self.args)


def refresh_models(name, *args):
    refreshes.append(name)
    return {"models_data": {name: {}}}


refresh_namespace = dict(
    constants,
    time=SimpleNamespace(time=lambda: clock[0]),
    get_models_config_path=lambda: "fixture",
    load_json=lambda _: deepcopy(refresh_document),
    _refresh_models_data=refresh_models,
    fetch_models_async=refresh_models,
    threading=SimpleNamespace(Thread=RefreshThread))
exec(compile(ast.Module(body=[tracker_class, refresh_function], type_ignores=[]),
             str(provider), "exec"), refresh_namespace)
refresh_namespace["api_call_tracker"] = refresh_namespace["EventTracker"]()
refresh_cases = json.loads(fixtures["chat-model-refresh-native.json"].read_text())
for case in refresh_cases:
    clock[0] = case["time"]
    refresh_document = {"models_data":
                        {case["provider"]: {"cached": {}}}
                        if case["cached"] else {}}
    before = len(refreshes)
    refresh_namespace["get_models_data"](case["provider"], "fixture", "fixture")
    assert (len(refreshes) > before) == case["fetch"], case

# Execute the actual positioning method using coordinates captured from a
# real plugin window. This checks placement only, not full-window visuals.
dialog = source.parent.parent / "widgets/chatbot_dialog.py"
dialog_class = next(node for node in ast.parse(dialog.read_text()).body
                    if isinstance(node, ast.ClassDef) and
                    node.name == "ChatbotDialog")
position_function = next(node for node in dialog_class.body
                         if isinstance(node, ast.FunctionDef) and
                         node.name == "show_model_dropdown")
geometry = json.loads(fixtures["chat-model-geometry-native.json"].read_text())
point = lambda x, y: SimpleNamespace(x=lambda: x, y=lambda: y)


class DropdownGeometry:
    def __init__(self):
        self.w, self.h = geometry["initial_width"], 500

    def width(self):
        return self.w

    def height(self):
        return self.h

    def resize(self, width, height):
        self.w, self.h = width, height

    def move(self, x, y):
        self.x, self.y = x, y

    def show(self):
        pass


position_namespace = dict(QPoint=point)
exec(compile(ast.Module(body=[position_function], type_ignores=[]),
             str(dialog), "exec"), position_namespace)
positioned = DropdownGeometry()
position_namespace["show_model_dropdown"](SimpleNamespace(
    refresh_model_list=lambda: None,
    model_button=SimpleNamespace(
        rect=lambda: SimpleNamespace(width=lambda: geometry["button_width"]),
        mapToGlobal=lambda _: point(geometry["button_x"], geometry["button_y"])),
    right_widget=SimpleNamespace(
        width=lambda: geometry["panel_width"],
        mapToGlobal=lambda _: point(0, geometry["panel_top"])),
    model_dropdown=positioned))
assert [positioned.x, positioned.y, positioned.w, positioned.h] == [
    geometry[key] for key in
    ("popup_x", "popup_y", "popup_width", "popup_height")], geometry
receipt = dict(
    verdict="PASS",
    local_provider_checks=sorted(local_provider),
    upstream_provider_checks=sorted(upstream_providers),
    comparisons=[
        "exact_upstream_default_providers", "exact_local_provider_extension",
        "exact_default_models",
        "source_save_model_selection", "unknown_fields_and_nullable_parameters",
        "source_popup_search", "source_popup_favorite_persistence",
        "source_model_refresh_tracker_and_cache_decisions",
        "source_model_popup_position_in_native_window"
    ],
    capability_cases=len(capabilities),
    search_cases=len(search_cases),
    refresh_cases=len(refresh_cases),
    upstream_tracker_sha256=hashlib.sha256(utils.read_bytes()).hexdigest(),
    upstream_dialog_sha256=hashlib.sha256(dialog.read_bytes()).hexdigest(),
    upstream_dropdown_sha256=hashlib.sha256(dropdown.read_bytes()).hexdigest(),
    upstream_config_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    upstream_provider_sha256=hashlib.sha256(provider.read_bytes()).hexdigest(),
    fixtures_sha256={
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in fixtures.items()
    })
(args.artifacts /
 "chat-config-upstream.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt))
