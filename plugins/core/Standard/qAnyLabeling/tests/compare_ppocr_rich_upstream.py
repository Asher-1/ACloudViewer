"""Execute the pinned source serialization/heading methods on native Qt output.

Qt presentation exports are supplied by the native probe. This checks source
decisions and formatting metrics; it does not claim Qt5/Qt6 pixel equivalence.
"""
import argparse
import ast
import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace
import __future__
from urllib.parse import unquote, urlparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--upstream', type=Path, required=True)
parser.add_argument('--artifacts', type=Path, required=True)
args = parser.parse_args()
source = args.upstream / 'anylabeling/views/labeling/ppocr/editors.py'
methods = {'_serialized_text', '_markdown_text', '_document_has_strikeout', '_html_fragment_text',
           '_max_content_height', '_document_text_height', '_update_editor_height',
           '_load_image_preview'}
paste_methods = {'_contains_image_urls', '_sanitized_mime_source',
                 '_should_start_new_block_for_paste', '_pasted_block_headings'}
style_methods = {'_is_dark_mode', '_editor_panel_background', '_editor_body_background',
                 '_editor_border_color', '_editor_selected_background',
                 '_rich_editor_container_style', '_rich_editor_format_button_style',
                 '_rich_editor_heading_button_style', '_rich_editor_style'}
nodes = []
for node in ast.parse(source.read_text()).body:
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and
            t.id in ('PPOCR_RICH_EDITOR_HEADING_LEVELS',
                     'PPOCR_RICH_EDITOR_BODY_MARGIN_BOTTOM',
                     'PPOCR_RICH_EDITOR_IMAGE_TEXT_SPACING_PX',
                     'PPOCR_EDITOR_IMAGE_MAX_WIDTH', 'PPOCR_EDITOR_IMAGE_MAX_HEIGHT',
                     'PPOCR_RICH_EDITOR_MIN_HEIGHT_PX', 'PPOCR_RICH_EDITOR_MAX_HEIGHT_RATIO',
                     'PPOCR_RICH_EDITOR_HEIGHT_PADDING_PX', 'PPOCR_RICH_EDITOR_BODY_OVERHEAD_PX')
            for t in node.targets):
        nodes.append(node)
    elif isinstance(node, ast.FunctionDef) and node.name == '_apply_rich_block_metrics':
        nodes.append(node)
    elif isinstance(node, ast.FunctionDef) and node.name in style_methods:
        nodes.append(node)
    elif isinstance(node, ast.ClassDef) and node.name == 'PPOCRRichTextBlockEditor':
        selected = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in methods]
        assert len(selected) == len(methods)
        nodes.append(ast.ClassDef(name='SourceEditor', bases=[], keywords=[], body=selected, decorator_list=[]))
    elif isinstance(node, ast.ClassDef) and node.name == '_PPOCRStyledRichTextEdit':
        selected = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in paste_methods]
        assert len(selected) == len(paste_methods)
        nodes.append(ast.ClassDef(name='SourcePaste', bases=[], keywords=[], body=selected, decorator_list=[]))
        image_method = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == '_image_rect']
        assert len(image_method) == 1
        nodes.append(ast.ClassDef(name='SourceImage', bases=[], keywords=[], body=image_method, decorator_list=[]))
    elif isinstance(node, ast.ClassDef) and node.name in {
            '_PPOCRRichHeadingDropdown', '_PPOCRRichHeadingMenuItem'}:
        name, method = ('SourceDropdown', 'set_current_level') if node.name.endswith('Dropdown') else ('SourceHeadingItem', 'set_selected')
        selected = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == method]
        assert len(selected) == 1
        nodes.append(ast.ClassDef(name=name, bases=[], keywords=[], body=selected, decorator_list=[]))

class Mime:
    def __init__(self, row=None):
        self.row = dict(row or {})
    def hasHtml(self): return isinstance(self.row.get('html'), str)
    def hasText(self): return isinstance(self.row.get('text'), str)
    def hasImage(self): return self.row.get('image', False)
    def hasUrls(self): return 'url' in self.row
    def html(self): return self.row['html']
    def text(self): return self.row['text']
    def setHtml(self, value): self.row['html'] = value
    def setText(self, value): self.row['text'] = value
    def urls(self):
        url = urlparse(self.row['url'])
        return [SimpleNamespace(toLocalFile=lambda: unquote(url.path) if url.scheme == 'file' else '')]
weights = SimpleNamespace(DemiBold=600)
scope = dict(re=re, Path=Path, QMimeData=Mime, QFont=SimpleNamespace(Weight=weights),
             QTextBlockFormat=SimpleNamespace(LineHeightTypes=SimpleNamespace(
                 SingleHeight=SimpleNamespace(value=0), ProportionalHeight=SimpleNamespace(value=1))))
exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
             str(source), 'exec', flags=__future__.annotations.compiler_flag), scope)
fixture = args.artifacts
data = json.loads(fixture.read_text())
for row in data['cases']:
    reference = scope['SourceEditor']()
    # Native QTextDocument supplies raw Qt exports. Execute the actual source
    # serializer for its Markdown/HTML choice and body/fragment trimming.
    reference.editor = SimpleNamespace(toMarkdown=lambda: row['markdown'], document=lambda: None)
    reference._document_has_strikeout = lambda: row['strike']
    scope['QTextCursor'] = lambda document: SimpleNamespace(
        select=lambda selection: None, selection=lambda: SimpleNamespace(toHtml=lambda: row['html']))
    scope['QTextCursor'].SelectionType = SimpleNamespace(Document=0)
    assert reference._serialized_text() == row['output'], row
for row in data['heading_cases']:
    values = {}
    format = SimpleNamespace(setTopMargin=lambda x: values.update(top=x),
        setBottomMargin=lambda x: values.update(bottom=x),
        setLineHeight=lambda h, t: values.update(line_height=h, line_type=t))
    scope['_apply_rich_block_metrics'](format, row['level'])
    assert all(row[k] == values[k] for k in ('bottom', 'line_height', 'line_type')), row
for row in data['paste_cases']:
    reference = scope['SourcePaste']()
    reference.textCursor = lambda: SimpleNamespace(positionInBlock=lambda: 4,
        block=lambda: SimpleNamespace(text=lambda: 'base'))
    sanitized = reference._sanitized_mime_source(Mime(row))
    assert (sanitized is not None) == row['allowed'], row
    if sanitized is None:
        continue
    assert reference._should_start_new_block_for_paste(sanitized) == row['new_block'], row
    if sanitized.hasHtml():
        levels = reference._pasted_block_headings(sanitized.html())
        for index, level in enumerate(levels):
            if level > 0:
                assert row['pasted_levels'][index] == level, row
for row in data['height_cases']:
    actual = {}
    reference = scope['SourceEditor']()
    reference._bind_scroll_viewport = lambda: None
    reference._tracked_scroll_viewport = SimpleNamespace(height=lambda: row['viewport_height'])
    reference._image_height = row['image_height']
    reference.editor = SimpleNamespace(
        document=lambda: SimpleNamespace(size=lambda: SimpleNamespace(height=lambda: row['document_height'])),
        height=lambda: -1, setFixedHeight=lambda h: actual.update(height=h))
    reference._update_editor_height()
    assert actual['height'] == row['actual'], row
for row in data['preview_cases']:
    # Execute the source loader with a native-size pixmap adapter. Only the
    # scaling geometry is supplied; the bounds and preview decision are source.
    actual = {}
    def scaled(width, height, aspect, transform):
        if row['width'] * height > row['height'] * width:
            actual.update(width=width, height=row['height'] * width // row['width'])
        else:
            actual.update(width=row['width'] * height // row['height'], height=height)
        return SimpleNamespace(height=lambda: actual['height'])
    scope['QPixmap'] = lambda path: SimpleNamespace(isNull=lambda: False, scaled=scaled)
    scope['Qt'] = SimpleNamespace(AspectRatioMode=SimpleNamespace(KeepAspectRatio=0),
        TransformationMode=SimpleNamespace(SmoothTransformation=0))
    reference = scope['SourceEditor']()
    reference._update_editor_height = lambda: None
    reference.image_label = SimpleNamespace(setPixmap=lambda pixmap: None, show=lambda: actual.update(shown=True))
    reference._load_image_preview(fixture)  # Existing transient file for the source existence check.
    assert actual['shown'] and actual['width'] == row['preview_width'], row
    assert actual['height'] == row['preview_height'] == reference._image_height, row
style_source = source.with_name('style.py')
style_nodes = [node for node in ast.parse(style_source.read_text()).body
               if isinstance(node, ast.FunctionDef) and node.name in {
                   'get_primary_button_style', 'get_secondary_button_style'}]
exec(compile(ast.Module(body=style_nodes, type_ignores=[]), str(style_source), 'exec'), scope)
theme_source = source.parent.parent / 'utils/theme.py'
themes = {node.target.id: ast.literal_eval(node.value)
          for node in ast.parse(theme_source.read_text()).body
          if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
          and node.target.id in {'LIGHT', 'DARK'}}
for node in ast.parse(source.with_name('config.py').read_text()).body:
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and
            t.id == 'PPOCR_COLOR_TEXT' for t in node.targets):
        scope['PPOCR_COLOR_TEXT'] = ast.literal_eval(node.value)
for row in data['style_cases']:
    scope['get_mode'] = lambda: row['mode']
    scope['get_theme'] = lambda: themes[row['mode'].upper()]
    expected = dict(frame=scope['_rich_editor_container_style'](),
                    text=scope['_rich_editor_style'](),
                    format=scope['_rich_editor_format_button_style'](),
                    heading=scope['_rich_editor_heading_button_style'](False),
                    heading_active=scope['_rich_editor_heading_button_style'](True),
                    save=scope['get_primary_button_style'](),
                    cancel=scope['get_secondary_button_style']())
    for key, value in expected.items():
        assert re.sub(r'\s+', '', row[key]) == re.sub(r'\s+', '', value), (row['mode'], key, row[key], value)
for row in data['image_selection_cases']:
    class Cursor:
        def __init__(self, document): self.position = 0
        def setPosition(self, position): self.position = position
        def charFormat(self):
            return SimpleNamespace(isImageFormat=lambda: self.position == row['position'] + 1,
                toImageFormat=lambda: SimpleNamespace(width=lambda: row['image_width']))
    def rect(values):
        return SimpleNamespace(x=lambda: values[0], y=lambda: values[1],
                               height=lambda: values[3])
    reference = scope['SourceImage']()
    reference.document = lambda: SimpleNamespace(characterCount=lambda: row['character_count'])
    reference.cursorRect = lambda cursor: rect(row['start_rect'] if cursor.position == row['position'] else row['end_rect'])
    scope['QTextCursor'] = Cursor
    scope['QRectF'] = lambda *args: list(args)
    assert reference._image_rect(row['position']) == row['image_rect'], row
    assert reference._image_rect(row['character_count'] - 1) is None, row
for row in data['dropdown_cases']:
    scope['get_theme'] = lambda: themes[row['mode'].upper()]
    dropdown = scope['SourceDropdown']()
    dropdown._items = []
    for native in row['rows']:
        item = scope['SourceHeadingItem']()
        item._level = native['level']
        item._label = SimpleNamespace(setStyleSheet=lambda value, native=native:
            native.update(source_style=value))
        dropdown._items.append(item)
    dropdown.set_current_level(1)
    for native in row['rows']:
        assert native['selected'] == (native['level'] == 1), native
        assert native['points'] == scope['PPOCR_RICH_EDITOR_HEADING_LEVELS'][native['level']]['point_size'], native
        assert re.sub(r'\s+', '', native['style']) == re.sub(r'\s+', '', native['source_style']), native
receipt = dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),
    serialization_cases=len(data['cases']), heading_cases=len(data['heading_cases']),
    paste_cases=len(data['paste_cases']),
    preview_cases=len(data['preview_cases']), height_cases=len(data['height_cases']),
    style_cases=len(data['style_cases']),
    dropdown_cases=len(data['dropdown_cases']),
    image_selection_cases=len(data['image_selection_cases']),
    style_sha256=hashlib.sha256(style_source.read_bytes()).hexdigest(),
    theme_sha256=hashlib.sha256(theme_source.read_bytes()).hexdigest(),
    scope='actual source decisions on native Qt exports; not full-window or cross-Qt rendering', passed=True)
fixture.with_suffix('.source-comparison.json').write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps(receipt))
