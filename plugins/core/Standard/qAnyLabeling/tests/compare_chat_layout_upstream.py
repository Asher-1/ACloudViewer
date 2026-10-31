"""Check native Chat panel constraints against the pinned source constructor.

This is structural evidence only. It does not certify pixel, font, Markdown,
formula or Qt6 rendering parity.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--upstream", type=Path, required=True)
parser.add_argument("--artifacts", type=Path, required=True)
args = parser.parse_args()
source = (args.upstream /
          "anylabeling/views/labeling/widgets/chatbot_dialog.py")
tree = ast.parse(source.read_text())
theme_file = args.upstream / "anylabeling/views/labeling/utils/theme.py"
themes = {}
for node in ast.parse(theme_file.read_text()).body:
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        if node.target.id in ("LIGHT", "DARK"):
            themes[node.target.id.lower()] = ast.literal_eval(node.value)
dialog = next(node for node in tree.body
              if isinstance(node, ast.ClassDef) and node.name == "ChatbotDialog")
constructor = next(node for node in dialog.body
                   if isinstance(node, ast.FunctionDef) and
                   node.name == "__init__")


def calls(owner, method):
    return [
        node for node in ast.walk(constructor)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and ast.unparse(node.func.value) == owner and node.func.attr == method
    ]


def literal_arguments(owner, method):
    matches = calls(owner, method)
    assert len(matches) == 1, (owner, method, len(matches))
    return [ast.literal_eval(value) for value in matches[0].args]


native = json.loads((args.artifacts / "chat-layout-native.json").read_text())
for sample in native["theme_samples"]:
    expected = themes[sample["mode"]]
    assert sample["provider_surface"] == expected["surface"].lower()
    assert sample["provider_text"] == expected["text"].lower()
    assert sample["conversation_background"] == expected["background"].lower()
assert len(native["columns"]) == 3
assert native["columns"][0]["minimum_width"] == literal_arguments(
    "self.left_widget", "setMinimumWidth")[0]
assert native["columns"][0]["maximum_width"] == literal_arguments(
    "self.left_widget", "setMaximumWidth")[0]
right = literal_arguments("self.right_widget", "setFixedWidth")[0]
assert native["columns"][2]["minimum_width"] == right
assert native["columns"][2]["maximum_width"] == right
assert [ast.unparse(call.args[0])
        for call in calls("self.main_splitter", "addWidget")] == [
            "self.left_widget", "self.middle_wrapper_widget",
            "self.right_widget"
        ]
assert native["main_handle_width"] == literal_arguments(
    "self.main_splitter", "setHandleWidth")[0]
assert native["right_handle_width"] == literal_arguments(
    "right_splitter", "setHandleWidth")[0]
assert native["root_margins"] == literal_arguments(
    "main_layout", "setContentsMargins")
assert native["root_spacing"] == literal_arguments(
    "main_layout", "setSpacing")[0]
assert native["preview_minimum_height"] == literal_arguments(
    "self.image_preview", "setMinimumHeight")[0]
assert native["temperature_range"] == [
    literal_arguments("self.temp_slider", "setMinimum")[0],
    literal_arguments("self.temp_slider", "setMaximum")[0]
]
assert native["settings_tabs"] == [
    ast.literal_eval(call.args[1].args[0])
    for call in calls("self.settings_tabs", "addTab")
]
message_source = args.upstream / "anylabeling/views/labeling/chatbot/chat.py"
message_class = next(node for node in ast.parse(message_source.read_text()).body
                     if isinstance(node, ast.ClassDef) and node.name == "ChatMessage")
message_methods = {node.name: node for node in message_class.body
                   if isinstance(node, ast.FunctionDef)}
message_native = json.loads(
    (args.artifacts / "chat-message-controls-native.json").read_text())
for sample in message_native["themes"]:
    expected = themes[sample["mode"]]
    assert sample["surface"] == expected["surface"].lower()
    assert sample["text"] == expected["text"].lower()
menu_method = message_methods["show_context_menu"]
for sample in message_native["menus"]:
    def menu_titles(statements):
        for statement in statements:
            if isinstance(statement, ast.If):
                if sample["role"] != "user":
                    yield from menu_titles(statement.body)
                continue
            for node in ast.walk(statement):
                if (isinstance(node, ast.Call) and
                        isinstance(node.func, ast.Attribute) and
                        node.func.attr == "addAction"):
                    yield ast.literal_eval(node.args[0].args[0])
    assert sample["items"] == list(menu_titles(menu_method.body))
copy_method = message_methods["copy_content_to_clipboard"]
timer = next(node for node in ast.walk(copy_method)
             if isinstance(node, ast.Call) and
             isinstance(node.func, ast.Attribute) and
             node.func.attr == "singleShot")
assert message_native["copy_feedback_ms"] == ast.literal_eval(timer.args[0])
receipt = dict(
    verdict="PASS",
    scope="Panel structure and constraints; full visual parity remains open",
    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    theme_source_sha256=hashlib.sha256(theme_file.read_bytes()).hexdigest(),
    message_source_sha256=hashlib.sha256(message_source.read_bytes()).hexdigest(),
    fixture_sha256=hashlib.sha256(
        (args.artifacts / "chat-layout-native.json").read_bytes()).hexdigest(),
    comparisons=["three_column_order", "provider_width_limits",
                 "right_fixed_width", "splitter_handles", "root_layout",
                 "preview_minimum_height", "settings_tabs",
                 "temperature_range", "workspace_light_dark_backgrounds"])
receipt["comparisons"] += ["user_assistant_context_menus", "copy_feedback_delay",
                           "message_light_dark_surface_and_text"]
(args.artifacts / "chat-layout-upstream.json").write_text(
    json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt))
