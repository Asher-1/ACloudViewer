"""Compare native message-history mutations against the pinned source methods.

Fixtures come from the actual plugin Qt test. Lightweight widget stand-ins only
supply the UI operations needed to execute the original Python implementations;
they do not replace the source's history slicing, indexing or mutations.
"""
import argparse
import ast
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--upstream", type=Path, required=True)
parser.add_argument("--artifacts", type=Path, required=True)
args = parser.parse_args()
source = args.upstream / "anylabeling/views/labeling/widgets/chatbot_dialog.py"
message_source = args.upstream / "anylabeling/views/labeling/chatbot/chat.py"
fixture = args.artifacts / "chat-message-controls-native.json"
data = json.loads(fixture.read_text())


class Layout:

    def __init__(self, widgets):
        self.widgets = list(widgets)

    def count(self):
        return len(self.widgets) + 1

    def itemAt(self, index):
        widget = self.widgets[index] if index < len(self.widgets) else None
        return SimpleNamespace(widget=lambda: widget,
                               spacerItem=lambda: widget is None)

    def removeWidget(self, widget):
        self.widgets.remove(widget)

    def addStretch(self):
        pass


class ChatMessage:

    def __init__(self, dialog, entry):
        self.dialog = dialog
        self.content = entry["content"]
        self.role = entry["role"]
        self.edit_area = SimpleNamespace(
            toPlainText=lambda: data["edited_content"])
        self.content_label = SimpleNamespace(setText=lambda text: None)

    def window(self):
        return self.dialog

    def setParent(self, _):
        pass

    def deleteLater(self):
        pass

    def exit_edit_mode(self):
        pass


class Dialog:

    def __init__(self, history):
        self.chat_history = deepcopy(history)
        self.chat_messages_layout = Layout(
            [ChatMessage(self, entry) for entry in self.chat_history])
        self.streaming = False
        self.current_api_address, self.current_api_key = "address", "key"
        self.stream_handler = SimpleNamespace(reset=lambda: None)
        self.message_input = SimpleNamespace(setPlainText=self.set_input)
        self.parent = lambda: None
        self.started = False

    def set_input(self, text):
        self.input = text

    def set_components_enabled(self, enabled):
        self.enabled = enabled

    def add_loading_message(self):
        pass

    def start_generation(self):
        self.started = True

    def stream_generation(self, *args):
        self.started = True


class Thread:

    def __init__(self, target, args):
        self.target, self.args = target, args

    def start(self):
        self.target(*self.args)


namespace = dict(ChatMessage=ChatMessage,
                 QApplication=SimpleNamespace(processEvents=lambda: None),
                 threading=SimpleNamespace(Thread=Thread),
                 convert_markdown_to_qt_html=lambda text: text)


def install(path, clsname, target, methods):
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body
               if isinstance(n, ast.ClassDef) and n.name == clsname)
    selected = [
        n for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name in methods
    ]
    assert len(selected) == len(methods)
    module = ast.Module(body=[
        ast.ImportFrom(module="__future__",
                       names=[ast.alias(name="annotations")],
                       level=0), *selected
    ],
                        type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"),
         namespace)
    for name in methods:
        setattr(target, name, namespace[name])


install(
    source, "ChatbotDialog", Dialog,
    {"clear_messages_after", "resubmit_edited_message", "regenerate_response"})
install(message_source, "ChatMessage", ChatMessage,
        {"save_edit", "delete_message"})

dialog = Dialog(data["initial"])
dialog.chat_messages_layout.widgets[1].save_edit()
assert dialog.chat_history == data["saved"]
dialog.chat_messages_layout.widgets[3].delete_message()
assert dialog.chat_history == data["deleted"]
dialog.regenerate_response(dialog.chat_messages_layout.widgets[1])
assert dialog.started and dialog.chat_history == data["regenerated"][:-1]
assert not dialog.enabled
# Resend must truncate the edited user and every subsequent message before
# delegating to the normal generation path. Wire encoding is covered by Qt.
dialog = Dialog(data["regenerated"])
dialog.resubmit_edited_message(dialog.chat_messages_layout.widgets[0],
                               "@image Edited request")
assert dialog.started and dialog.chat_history == data["resent"][:-2]
assert dialog.input == "@image Edited request"
receipt = dict(
    verdict="PASS",
    comparisons=[
        "save_single_message", "delete_single_message",
        "regenerate_history_prefix", "resend_history_prefix"
    ],
    upstream_dialog_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    upstream_messages_sha256=hashlib.sha256(
        message_source.read_bytes()).hexdigest(),
    native_fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest())
(args.artifacts / "chat-message-controls-upstream.json"
).write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt))
