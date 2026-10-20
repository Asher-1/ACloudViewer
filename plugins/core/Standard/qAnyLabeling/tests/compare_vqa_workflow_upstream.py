"""Execute the pinned upstream export method, with only UI dialogs substituted.

This is a contract fixture, not a screenshot or full upstream GUI acceptance.
"""
import argparse, ast, hashlib, json, logging, os
from pathlib import Path
from types import SimpleNamespace
from PIL import Image

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--upstream", type=Path, required=True)
parser.add_argument("--artifacts", type=Path, required=True)
args = parser.parse_args()
artifacts = args.artifacts.resolve()
b = artifacts
source = args.upstream.resolve(
) / "anylabeling/views/labeling/widgets/vqa_dialog.py"
tree = ast.parse(source.read_text())
cls = next(n for n in tree.body
           if isinstance(n, ast.ClassDef) and n.name == 'VQADialog')
method = next(n for n in cls.body
              if isinstance(n, ast.FunctionDef) and n.name == 'export_labels')
config = json.loads((artifacts / 'export-config.json').read_text())
target = artifacts / 'upstream-selected.jsonl'


class Export:

    def __init__(self, *args):
        pass

    def exec(self):
        return 1

    def get_export_config(self):
        return config['mapping']


class Pixmap:

    def __init__(self, file):
        with Image.open(file) as im:
            self.size = im.size

    def isNull(self):
        return False

    def width(self):
        return self.size[0]

    def height(self):
        return self.size[1]


class Dialog:
    DialogCode = SimpleNamespace(Accepted=1)


messages = []
globals_ = {
    'os':
        os,
    'json':
        json,
    'QDialog':
        Dialog,
    'ExportLabelsDialog':
        Export,
    'QFileDialog':
        SimpleNamespace(getSaveFileName=lambda *a: (str(target), '')),
    'QMessageBox':
        SimpleNamespace(
            warning=lambda *a: messages.append(['warning', str(a[2])]),
            information=lambda *a: messages.append(['information',
                                                    str(a[2])])),
    'QPixmap':
        Pixmap,
    'get_label_file_path':
        lambda file, output: str(Path(output) / (Path(file).stem + '.json')),
    'logger':
        logging.getLogger('upstream-vqa')
}
exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'),
     globals_)
instance = SimpleNamespace(
    image_files=list(
        map(str, [artifacts / 'first.png', artifacts / 'second.png'])),
    custom_components=config['components'],
    tr=lambda x: x,
    save_current_image_data=lambda: None,
    parent=lambda: SimpleNamespace(output_dir=str(artifacts)))
globals_['export_labels'](instance)


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


actual = rows(artifacts / 'vqa-selected.jsonl')
expected = rows(target)
result = {
    'scope':
        'two-page selected/renamed/default-fields export; UI dialogs substituted only for the upstream reference',
    'source':
        str(source),
    'source_sha256':
        hashlib.sha256(source.read_bytes()).hexdigest(),
    'plugin_rows':
        actual,
    'upstream_rows':
        expected,
    'match':
        actual == expected,
    'messages':
        messages
}
(artifacts / 'comparison.json'
).write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(result, ensure_ascii=False))
assert result['match']

# Run the source page and clear operations with Qt storage primitives replaced.
# Native tests exercise real widgets and sidecar writes; these calls supply an
# independent source contract without requiring the upstream application's Qt6.
from types import MethodType

navigation_fixture = artifacts / 'vqa-navigation-native.json'
if navigation_fixture.exists():
    native_navigation = json.loads(navigation_fixture.read_text())
    methods = {
        'validate_page_input', 'on_page_input_finished',
        'restore_current_page_number', 'update_navigation_state',
        'clear_current', 'save_current_image_data', 'get_component_value'
    }
    namespace = dict(globals_)
    namespace['QMessageBox'] = SimpleNamespace(
        StandardButton=SimpleNamespace(Yes=1, No=2),
        question=lambda *args: 1)
    exec(compile(ast.Module(body=[
        node for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name in methods
    ], type_ignores=[]), str(source), 'exec'), namespace)

    class Text:
        def __init__(self, text=''):
            self.value = text
        def text(self):
            return self.value
        def setText(self, text):
            self.value = text
        def cursorPosition(self):
            return len(self.value)
        def setCursorPosition(self, position):
            pass
        def toPlainText(self):
            return self.value
        def clear(self):
            self.value = ''

    class Option:
        def __init__(self, name):
            self.name, self.checked = name, True
        def setChecked(self, checked):
            self.checked = checked
        def isChecked(self):
            return self.checked
        def toolTip(self):
            return self.name

    class Options:
        def __init__(self, name):
            button = Option(name)
            self.itemList = [SimpleNamespace(widget=lambda: button)]
        def layout(self):
            return self
        def count(self):
            return len(self.itemList)
        def itemAt(self, index):
            return self.itemList[index]

    class Combo:
        def __init__(self):
            self.index = 1
        def setCurrentIndex(self, index):
            self.index = index
        def currentText(self):
            return ['-- Select --', 'green'][self.index]

    parent = SimpleNamespace(filename='first', image_list=['first', 'second'],
                             other_data={'vqaData': {'Unknown': 'preserve'}},
                             set_dirty=lambda: None)
    previous, following = SimpleNamespace(), SimpleNamespace()
    previous.setEnabled = lambda value: setattr(previous, 'enabled', value)
    following.setEnabled = lambda value: setattr(following, 'enabled', value)
    reference = SimpleNamespace(parent=lambda: parent, tr=lambda text: text,
                                page_input=Text('99'), prev_button=previous,
                                next_button=following)
    for name in methods:
        setattr(reference, name, MethodType(namespace[name], reference))
    reference.update_navigation_state()
    assert native_navigation['first_page'] == [previous.enabled, following.enabled]
    parent.filename = 'second'
    reference.update_navigation_state()
    assert native_navigation['last_page'] == [previous.enabled, following.enabled]
    reference.validate_page_input('99')
    assert native_navigation['over_maximum'] == reference.page_input.text()
    reference.page_input.clear()
    reference.on_page_input_finished()
    assert native_navigation['empty_restored'] == reference.page_input.text()
    reference.custom_components = [
        dict(title='Caption', type='QLineEdit', widget=Text('manual caption')),
        dict(title='Radio', type='QRadioButton', widget=Options('yes')),
        dict(title='Choice', type='QComboBox', widget=Combo()),
        dict(title='Multi', type='QCheckBox', widget=Options('road'))
    ]
    reference.clear_current()
    assert native_navigation['cleared_vqa'] == parent.other_data['vqaData']
    constants = {}
    vqa_config_source = source.parent.parent / 'vqa/config.py'
    for node in ast.parse(vqa_config_source.read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                constants[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    assert native_navigation['page_size'] == [68, constants['DEFAULT_COMPONENT_HEIGHT']]
    assert native_navigation['nav_button_size'] == list(constants['ICON_SIZE_NORMAL'])
    panel_size = next(ast.literal_eval(node.value) for node in tree.body
                      if isinstance(node, ast.Assign) and
                      isinstance(node.targets[0], ast.Name) and
                      node.targets[0].id == 'PANEL_BUTTON_SIZE')
    assert native_navigation['panel_button_size'] == [panel_size, panel_size]
    (artifacts / 'vqa-navigation-upstream.json').write_text(json.dumps(dict(
        verdict='PASS', source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        scope='Original page validation, boundaries, clear and serialization methods; Qt storage primitives substituted; native real-widget and sidecar assertions also required'
    ), indent=2) + '\n')

# Execute original sidecar mutation methods on copies of the native fixture.
fixture = json.loads((artifacts / 'migration-fixture.json').read_text())
utils = source.parent.parent / 'vqa/utils.py'
names = {'apply_option_mapping', 'value_contains_deleted_options'}
functions = [
    node for node in ast.parse(utils.read_text()).body
    if isinstance(node, ast.FunctionDef) and node.name in names
]
mutations = {
    'update_title_in_json_file', 'update_modified_options_in_json',
    'reset_deleted_options_in_json', 'remove_component_from_json_file'
}
functions += [
    node for node in cls.body
    if isinstance(node, ast.FunctionDef) and node.name in mutations
]
exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'),
     globals_)
upstream = []
for i, document in enumerate(fixture['before']):
    path = artifacts / f'upstream-migration-{i}.json'
    path.write_text(json.dumps(document))
    globals_['update_title_in_json_file'](instance, str(path), 'Caption',
                                          'Renamed')
    globals_['update_modified_options_in_json'](instance, str(path), 'Radio', {
        'yes': 'affirmative',
        'no': 'negative'
    })
    globals_['reset_deleted_options_in_json'](instance, str(path), 'Radio',
                                              {'yes', 'no'}, 'affirmative')
    globals_['reset_deleted_options_in_json'](instance, str(path), 'Multi',
                                              {'road'}, [])
    globals_['remove_component_from_json_file'](instance, str(path), 'Renamed')
    upstream.append(json.loads(path.read_text()))
migration = {
    'match':
        upstream == fixture['after'],
    'source_sha256':
        result['source_sha256'],
    'utils_sha256':
        hashlib.sha256(utils.read_bytes()).hexdigest(),
    'scope':
        'title rename, indexed radio option rename, removed checkbox option reset, component deletion; all unrelated JSON fields retained',
    'upstream':
        upstream,
    'native':
        fixture['after']
}
(artifacts / 'migration-comparison.json'
).write_text(json.dumps(migration, ensure_ascii=False, indent=2) + '\n')
print(
    json.dumps(
        {
            k: v
            for k, v in migration.items()
            if k not in ('upstream', 'native')
        },
        ensure_ascii=False))
assert migration['match']

# Compare the actual source defaults without importing its Qt UI modules.
config_source = args.upstream.resolve(
) / 'anylabeling/views/labeling/vqa/config.py'
assignment = next(n for n in ast.parse(config_source.read_text()).body
                  if isinstance(n, ast.Assign) and any(
                      isinstance(t, ast.Name) and t.id == 'DEFAULT_TEMPLATES'
                      for t in n.targets))
defaults = ast.literal_eval(assignment.value)
defaults = [
    dict(name=name, content=content) for name, content in defaults.items()
]
actual = json.loads((artifacts / 'default-templates.json').read_text())
result = dict(match=actual == defaults,
              default_template_count=len(defaults),
              source_sha256=hashlib.sha256(
                  config_source.read_bytes()).hexdigest())
(artifacts /
 'template-comparison.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
assert result['match']

# Execute actual upstream reference replacement on the disk-sidecar fixture.
import re

utils_source = args.upstream.resolve(
) / 'anylabeling/views/labeling/vqa/utils.py'
worker = next(n for n in ast.parse(utils_source.read_text()).body
              if isinstance(n, ast.ClassDef) and n.name == 'AIWorkerThread')
methods = [
    n for n in worker.body if isinstance(n, ast.FunctionDef) and n.name in (
        'process_special_references', 'get_label_data')
]
reference = json.loads((artifacts / 'prompt-references.json').read_text())
label = artifacts / 'prompt-reference-sidecar.json'
label.write_text(reference['labels_raw'])
namespace = dict(os=os,
                 json=json,
                 re=re,
                 logger=logging.getLogger('upstream-vqa'),
                 get_label_file_path=lambda *_: str(label))
exec(
    compile(ast.Module(body=methods, type_ignores=[]), str(utils_source),
            'exec'), namespace)
worker = SimpleNamespace(
    current_text=reference['current_text'].strip(),
    components=[
        dict(title='Caption',
             type='QLineEdit',
             widget=SimpleNamespace(
                 toPlainText=lambda: reference['widget_text']))
    ],
    image_path=str(artifacts / 'first.png'),
    parent=SimpleNamespace(output_dir=str(artifacts)))
worker.get_label_data = lambda: namespace['get_label_data'](worker)
expected = namespace['process_special_references'](worker, reference['prompt'])
result = dict(
    match=expected == reference['actual'] and
    worker.has_image_reference == reference['has_image'],
    expected=expected,
    actual=reference['actual'],
    source_sha256=hashlib.sha256(utils_source.read_bytes()).hexdigest(),
    scope=
    'actual upstream special-reference method, ordered disk JSON, nested Python repr, image whitespace, overlapping VQA keys and unknown reference'
)
(artifacts / 'prompt-reference-comparison.json'
).write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(result, ensure_ascii=False))
assert result['match']

# Independently execute the original pure QSS functions against the actual
# native widgets after live workspace theme switches.
style_fixture = artifacts / 'vqa-styles-native.json'
if style_fixture.exists():
    base = args.upstream / 'anylabeling/views/labeling'
    theme_source = base / 'utils/theme.py'
    config_source = base / 'vqa/config.py'
    style_source = base / 'vqa/style.py'
    theme_values = {}
    for node in ast.parse(theme_source.read_text()).body:
        if (isinstance(node, ast.AnnAssign) and
                isinstance(node.target, ast.Name) and
                node.target.id in ('LIGHT', 'DARK')):
            theme_values[node.target.id.lower()] = ast.literal_eval(node.value)
    config_nodes = [node for node in ast.parse(config_source.read_text()).body
                    if isinstance(node, ast.Assign)]
    functions = [node for node in ast.parse(style_source.read_text()).body
                 if isinstance(node, ast.FunctionDef)]
    roles = {
        'VQAMainSplitter': ('main_splitter', ()),
        'VQAFilename': ('filename_label', ()),
        'VQAImageContainer': ('image_container', ()),
        'VQAImagePreview': ('image_label', ()),
        'VQAPageInput': ('page_input', ()),
        'VQAAddComponent': ('dialog_button', ('primary', 'medium')),
        'VQAExport': ('dialog_button', ('success', 'medium')),
        'VQAClearCurrent': ('dialog_button', ('secondary', 'medium')),
    }
    for name in ('VQARefresh', 'VQARefreshRight', 'VQAToggleSidebar',
                 'VQAToggleSidebarRight'):
        roles[name] = ('panel_button', ())
    for name in ('VQAPrevious', 'VQANext'):
        roles[name] = ('button', ())
    cases = []
    for sample in json.loads(style_fixture.read_text()):
        theme = theme_values[sample['mode']]
        env = dict(Dict=dict, get_theme=lambda: theme, THEME=theme)
        for node in config_nodes:
            try:
                exec(compile(ast.Module(body=[node], type_ignores=[]),
                             str(config_source), 'exec'), env)
            except NameError:
                # Config path and work-directory functions are not styles.
                pass
        exec(compile(ast.Module(body=functions, type_ignores=[]),
                     str(style_source), 'exec'), env)
        for widget, (role, arguments) in roles.items():
            expected = env['get_' + role + '_style'](*arguments)
            assert sample['styles'][widget] == expected, (sample['mode'], widget)
        assert sample['image_container_surface'] == theme['background_secondary'].lower()
        assert sample['filename_text'] == theme['text'].lower()
        assert sample['export_background'] == env['BUTTON_COLORS']['success']['background']
        cases.append(dict(mode=sample['mode'], widgets=len(roles)))
    result = dict(verdict='PASS', cases=cases,
                  source_sha256=hashlib.sha256(style_source.read_bytes()).hexdigest(),
                  fixture_sha256=hashlib.sha256(style_fixture.read_bytes()).hexdigest(),
                  scope='Original QSS equality, live theme switching and selected rendered colors; full pixel/font/DPI parity remains open')
    (artifacts / 'vqa-styles-upstream.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
