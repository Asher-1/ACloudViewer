"""Execute the pinned ShareGPT importer against the native import fixture."""
import argparse, ast, hashlib, json, logging, os, shutil, tempfile
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
) / "anylabeling/views/labeling/widgets/chatbot_dialog.py"
fixture = artifacts / "sharegpt-import"
cls = next(n for n in ast.parse(source.read_text()).body
           if isinstance(n, ast.ClassDef) and n.name == 'ChatbotDialog')
method = next(
    n for n in cls.body
    if isinstance(n, ast.FunctionDef) and n.name == 'import_export_dataset')
importer = next(
    n for n in method.body
    if isinstance(n, ast.FunctionDef) and n.name == 'import_dataset')
work = Path(tempfile.mkdtemp(prefix='sharegpt-import-upstream-', dir=b))
target = work / 'output'
target.mkdir()
shutil.copyfile(fixture / 'native/anchor.png', target / 'anchor.png')
messages = []
parent = SimpleNamespace(filename=str(target / 'anchor.png'),
                         image_list=[str(target / 'anchor.png')])
version_file = args.upstream.resolve() / 'anylabeling/app_info.py'
version = ast.literal_eval(
    next(n.value
         for n in ast.parse(version_file.read_text()).body
         if isinstance(n, ast.Assign) and any(
             isinstance(t, ast.Name) and t.id == '__version__'
             for t in n.targets)))
ns = dict(
    os=os,
    json=json,
    shutil=shutil,
    Image=Image,
    __version__=version,
    logger=logging.getLogger('upstream-chat'),
    option_dialog=SimpleNamespace(accept=lambda: None),
    QFileDialog=SimpleNamespace(getOpenFileName=lambda *args:
                                (str(fixture / 'input/dataset.json'), '')),
    QMessageBox=SimpleNamespace(
        **{
            name: (lambda *args: messages.append(str(args[2])))
            for name in ['warning', 'information', 'critical']
        }),
    self=SimpleNamespace(parent=lambda: parent,
                         tr=lambda x: x,
                         navigate_image=lambda **kwargs: None))
exec(compile(ast.Module(body=[importer], type_ignores=[]), str(source), 'exec'),
     ns)
ns['import_dataset']()


def payload(folder):
    result = {}
    for f in folder.glob('*.json'):
        d = json.loads(f.read_text())
        for msg in d['chat_history']:
            if msg['image'] is not None:
                msg['image'] = Path(msg['image']).name
        result[f.name] = d
    return result


actual = payload(fixture / 'native')
expected = payload(target)
actual_images = {
    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
    for p in (fixture / 'native').glob('*.png')
}
expected_images = {
    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
    for p in target.glob('*.png')
}
result = dict(
    match=actual == expected and actual_images == expected_images and
    len(actual) == 2,
    native_records=actual,
    upstream_records=expected,
    image_sha256=actual_images,
    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    source_version=version,
    source_version_sha256=hashlib.sha256(version_file.read_bytes()).hexdigest(),
    upstream_output=str(target),
    messages=messages,
    scope=
    'actual pinned nested import_dataset; UI selection/dialogs/navigation substituted; image roots normalized only'
)
(artifacts / 'sharegpt-import-comparison.json'
).write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(result, ensure_ascii=False))
assert result['match']
