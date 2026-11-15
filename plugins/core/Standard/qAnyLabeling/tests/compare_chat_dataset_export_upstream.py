"""Execute the pinned source export on the actual native dataset fixture."""
import argparse, ast, datetime, hashlib, json, logging, os, shutil, tempfile, zipfile
from pathlib import Path
from types import SimpleNamespace

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--upstream", type=Path, required=True)
parser.add_argument("--artifacts", type=Path, required=True)
args = parser.parse_args()
artifacts = args.artifacts.resolve()
b = artifacts
source = args.upstream.resolve(
) / "anylabeling/views/labeling/widgets/chatbot_dialog.py"
cls = next(n for n in ast.parse(source.read_text()).body
           if isinstance(n, ast.ClassDef) and n.name == 'ChatbotDialog')
method = next(
    n for n in cls.body
    if isinstance(n, ast.FunctionDef) and n.name == 'import_export_dataset')
export = next(n for n in method.body
              if isinstance(n, ast.FunctionDef) and n.name == 'export_dataset')
work = Path(tempfile.mkdtemp(prefix='sharegpt-upstream-', dir=b))
output = work / 'output'
output.mkdir()
messages = []


class FileDialog:
    Option = SimpleNamespace(ShowDirsOnly=1)

    @staticmethod
    def getExistingDirectory(*args):
        return str(output)


ns = dict(os=os,
          json=json,
          datetime=datetime,
          shutil=shutil,
          zipfile=zipfile,
          logger=logging.getLogger('upstream-chat'),
          option_dialog=SimpleNamespace(accept=lambda: None),
          QFileDialog=FileDialog,
          get_work_directory=lambda: str(work),
          QMessageBox=SimpleNamespace(
              **{
                  name: (lambda *args: messages.append(str(args[2])))
                  for name in ['warning', 'information', 'critical']
              }),
          self=SimpleNamespace(parent=lambda: SimpleNamespace(filename=str(
              artifacts / 'sharegpt-input/second.png')),
                               tr=lambda x: x))
exec(compile(ast.Module(body=[export], type_ignores=[]), str(source), 'exec'),
     ns)
ns['export_dataset']()
archives = list(output.glob('*.zip'))
assert len(archives) == 1, messages
with zipfile.ZipFile(artifacts /
                     'sharegpt-native.zip') as native, zipfile.ZipFile(
                         archives[0]) as upstream:
    assert native.testzip() is None and upstream.testzip() is None

    def payload(archive):
        rows = [n for n in archive.namelist() if n.endswith('.json')]
        assert len(rows) == 1
        return json.loads(archive.read(rows[0])), {
            n: hashlib.sha256(archive.read(n)).hexdigest()
            for n in archive.namelist()
            if n.startswith('images/')
        }

    actual, actual_images = payload(native)
    expected, expected_images = payload(upstream)
    result = dict(
        match=actual == expected and actual_images == expected_images,
        native_records=actual,
        upstream_records=expected,
        image_sha256=actual_images,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        native_archive_sha256=hashlib.sha256(
            (artifacts / 'sharegpt-native.zip').read_bytes()).hexdigest(),
        upstream_archive=str(archives[0]),
        scope=
        'actual pinned export method; only UI selection/message boxes substituted; multi-turn messages, image paths and original image bytes'
    )
(artifacts / 'sharegpt-comparison.json'
).write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(result, ensure_ascii=False))
assert result['match']

# Run the original batch sidecar writer on the same before/after documents.
finish = next(
    n for n in cls.body
    if isinstance(n, ast.FunctionDef) and n.name == 'finish_batch_processing')
batch = work / 'batch'
batch.mkdir()
for name in ('first.json', 'second.json'):
    shutil.copyfile(artifacts / 'sharegpt-input' / name, batch / name)
files = [str(batch / (stem + '.png')) for stem in ('first', 'second')]
owner = SimpleNamespace(
    batch_results={
        name: dict(content='Generated answer', error=None) for name in files
    },
    batch_prompt='<image>Describe this image',
    progress_dialog=SimpleNamespace(close=lambda: None),
    current_index=0,
    parent=lambda: SimpleNamespace(image_list=files, filename=files[0]),
    navigate_image=lambda **kwargs: None,
    tr=lambda x: x)
exec(compile(ast.Module(body=[finish], type_ignores=[]), str(source), 'exec'),
     ns)
ns['finish_batch_processing'](owner)


def normalized(document):
    for entry in document['chat_history']:
        if entry.get('image'):
            entry['image'] = Path(entry['image']).name
    return document


actual = [
    normalized(d)
    for d in json.loads((artifacts / 'chat-batch-native.json').read_text())
]
expected = [
    normalized(json.loads((batch / name).read_text()))
    for name in ('first.json', 'second.json')
]
result = dict(
    match=actual == expected,
    native=actual,
    upstream=expected,
    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    scope=
    'actual original finish_batch_processing; identical input documents; only local image roots normalized'
)
(artifacts / 'chat-batch-comparison.json'
).write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(
    json.dumps({
        k: v for k, v in result.items() if k not in ('native', 'upstream')
    }))
assert result['match']
