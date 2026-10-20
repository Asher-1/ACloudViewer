"""Run the pinned source table parsers/serializer on native Qt fixtures."""
import argparse
import ast
import hashlib
import json
import re
from html import escape as html_escape, unescape as html_unescape
from html.parser import HTMLParser
from pathlib import Path
from types import SimpleNamespace
import __future__

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--upstream", type=Path, required=True)
parser.add_argument("--artifacts", type=Path, required=True)
args = parser.parse_args()
source = args.upstream / "anylabeling/views/labeling/ppocr/editors.py"
fixture = args.artifacts / "ppocr-table-native.json"
tree = ast.parse(source.read_text())
methods = {
    "_parse_token_table_content", "_parse_html_table_content",
    "_decode_cell_payload", "_load_content", "_rebuild_span_index",
    "_cell_anchor", "_encode_cell_payload", "_serialize_table_tokens",
    "_anchor_span", "_sync_column_widths", "_column_content_width_hint",
    "_sync_row_heights", "_default_row_height", "_frame_style", "_table_style",
    "_table_content_height", "_sync_table_height", "_available_table_height"
}
nodes = []
for node in tree.body:
    if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and
            (t.id.startswith("_TABLE_") or t.id.startswith("_HTML_TABLE_") or
             t.id.startswith("PPOCR_TABLE_"))
            for t in node.targets):
        nodes.append(node)
    elif isinstance(node, ast.FunctionDef) and node.name in {
            "_parse_span_int", "_is_dark_mode", "_editor_panel_background",
            "_editor_border_color", "_editor_selected_background",
            "_rich_editor_format_button_style"}:
        nodes.append(node)
    elif isinstance(node, ast.ClassDef) and node.name == "_PPOCRHTMLTableParser":
        nodes.append(node)
    elif isinstance(node, ast.ClassDef) and node.name == "PPOCRTableBlockEditor":
        selected = [m for m in node.body
                    if isinstance(m, ast.FunctionDef) and m.name in methods]
        assert len(selected) == len(methods)
        nodes.append(ast.ClassDef(name="SourceEditor", bases=[], keywords=[],
                                  body=selected, decorator_list=[]))
scope = dict(re=re, HTMLParser=HTMLParser,
             html_escape=html_escape, html_unescape=html_unescape)
module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
exec(compile(module, str(source), "exec",
             flags=__future__.annotations.compiler_flag), scope)
SourceEditor = scope["SourceEditor"]


class Table:
    """Presentation stand-in only: no parsing, formatting, or token decisions."""

    def __init__(self, rows, columns, states, spans):
        self.rows, self.columns = rows, columns
        self.states, self.spans = states, spans

    def rowCount(self):
        return self.rows

    def columnCount(self):
        return self.columns

    def item(self, r, c):
        text, bold, italic, strike = self.states.get(
            (r, c), ("", False, False, False))
        return SimpleNamespace(
            text=lambda: text,
            font=lambda: SimpleNamespace(bold=lambda: bold,
                                         italic=lambda: italic,
                                         strikeOut=lambda: strike))

    def span(self, r, c):
        for (y, x), (h, w) in self.spans.items():
            if y <= r < y + h and x <= c < x + w:
                return h, w
        return 1, 1

    def rowSpan(self, r, c):
        return self.span(r, c)[0]

    def columnSpan(self, r, c):
        return self.span(r, c)[1]


def apply(self, rows, columns, states, spans):
    self.table = Table(rows, columns, states, spans)


SourceEditor._apply_parsed_table = apply
data = json.loads(fixture.read_text())
for case in data["cases"]:
    reference = SourceEditor()
    reference._load_content(case["input"])
    native = case["output"]
    assert (native["rows"], native["columns"]) == (
        reference.table.rows, reference.table.columns), case
    for row in native["states"]:
        r, c = row["row"], row["column"]
        expected = reference.table.states.get(
            (r, c), ("", False, False, False))
        actual = tuple(row[k] for k in ("text", "bold", "italic", "strike"))
        assert actual == expected, (case["input"], r, c, actual, expected)
        assert (row["rows"], row["columns"]) == reference.table.span(r, c)
    expected = reference._serialize_table_tokens()
    assert native["tokens"] == expected, (case["input"], native["tokens"], expected)
    reference._load_content(expected)
    assert case["replay"]["tokens"] == reference._serialize_table_tokens()

for case in data.get("geometry_cases", []):
    reference = SourceEditor()
    reference._load_content(case["input"])
    table = reference.table
    reference._rebuild_span_index()
    widths, heights = {}, {}
    table.viewport = lambda: SimpleNamespace(width=lambda: case["viewport"])
    table.fontMetrics = lambda: SimpleNamespace(
        height=lambda: case["font_height"],
        horizontalAdvance=lambda text: case["advances"][text])
    table.setColumnWidth = lambda column, width: widths.update({column: width})
    table.verticalHeader = lambda: SimpleNamespace(setDefaultSectionSize=lambda height: None)
    table.rowHeight = lambda row: heights.get(row, -1)
    table.setRowHeight = lambda row, height: heights.update({row: height})
    reference._sync_column_widths()
    reference._sync_row_heights()
    assert case["widths"] == [widths[c] for c in range(table.columns)], case
    assert case["heights"] == [heights[r] for r in range(table.rows)], case

scope["QPoint"] = lambda x, y: (x, y)
for case in data.get("height_cases", []):
    reference = SourceEditor()
    heights = dict(minimum=-1, maximum=-1)
    viewport = SimpleNamespace(height=lambda: case["viewport_height"])
    reference._find_parent_scroll_area = lambda: SimpleNamespace(
        viewport=lambda: viewport)
    reference._horizontal_scrollbar_needed = lambda: case["horizontal_scrollbar"]
    reference.table = SimpleNamespace(
        rowCount=lambda: case["rows"],
        fontMetrics=lambda: SimpleNamespace(height=lambda: case["font_height"]),
        frameWidth=lambda: case["frame_width"],
        showGrid=lambda: case["show_grid"],
        mapTo=lambda parent, point: SimpleNamespace(y=lambda: case["table_top"]),
        horizontalScrollBar=lambda: SimpleNamespace(
            sizeHint=lambda: SimpleNamespace(height=lambda: case["scrollbar_height"])),
        minimumHeight=lambda: heights["minimum"],
        maximumHeight=lambda: heights["maximum"],
        setMinimumHeight=lambda height: heights.update(minimum=height),
        setMaximumHeight=lambda height: heights.update(maximum=height))
    reference._sync_table_height()
    assert case["minimum_height"] == heights["minimum"], case
    assert case["maximum_height"] == heights["maximum"], case

style_source = source.with_name("style.py")
style_nodes = [node for node in ast.parse(style_source.read_text()).body
               if isinstance(node, ast.FunctionDef) and node.name in {
                   "get_primary_button_style", "get_secondary_button_style"}]
exec(compile(ast.Module(body=style_nodes, type_ignores=[]), str(style_source),
             "exec"), scope)
theme_source = source.parent.parent / "utils/theme.py"
themes = {node.target.id: ast.literal_eval(node.value)
          for node in ast.parse(theme_source.read_text()).body
          if isinstance(node, ast.AnnAssign) and
          isinstance(node.target, ast.Name) and node.target.id in {"LIGHT", "DARK"}}
config_source = source.with_name("config.py")
for node in ast.parse(config_source.read_text()).body:
    if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "PPOCR_COLOR_TEXT"
            for t in node.targets):
        scope["PPOCR_COLOR_TEXT"] = ast.literal_eval(node.value)
for case in data.get("style_cases", []):
    scope["get_theme"] = lambda: themes[case["mode"].upper()]
    scope["get_mode"] = lambda: case["mode"]
    reference = SourceEditor()
    expected = dict(frame=reference._frame_style(), table=reference._table_style(),
                    format=scope["_rich_editor_format_button_style"](),
                    save=scope["get_primary_button_style"](),
                    cancel=scope["get_secondary_button_style"]())
    for key, value in expected.items():
        assert re.sub(r"\s+", "", case[key]) == re.sub(r"\s+", "", value), (
            case["mode"], key, case[key], value)

print(json.dumps(dict(verdict="PASS", cases=len(data["cases"]),
                      geometry_cases=len(data.get("geometry_cases", [])),
                      height_cases=len(data.get("height_cases", [])),
                      style_cases=len(data.get("style_cases", [])),
                      source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                      style_sha256=hashlib.sha256(style_source.read_bytes()).hexdigest(),
                      theme_sha256=hashlib.sha256(theme_source.read_bytes()).hexdigest(),
                      config_sha256=hashlib.sha256(config_source.read_bytes()).hexdigest(),
                      fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),
                      scope="Actual pinned parsers, serializer, geometry and light/dark QSS; native content/format/span and runtime font metrics; Qt stand-ins do not prove full-window pixels")))
