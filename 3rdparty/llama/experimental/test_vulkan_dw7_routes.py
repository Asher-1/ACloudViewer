#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import tempfile

p = argparse.ArgumentParser()
p.add_argument('--build', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
build = a.build.resolve()
base = a.output.resolve()
base.mkdir(parents=True, exist_ok=True)
out = Path(tempfile.mkdtemp(prefix='run-', dir=base))
source = build / 'llama/src/ext_llama/ggml/src/ggml-vulkan'
vkbuild = build / 'llama/src/ext_llama-build/ggml/src/ggml-vulkan'
flags_file = (vkbuild / 'CMakeFiles/ggml-vulkan.dir/flags.make').read_text()
compiler = re.search(r'^# compile CXX with (.+)$', flags_file, re.M).group(1)
flags = ['-I' + str(source)]
for key in ('CXX_DEFINES', 'CXX_INCLUDES', 'CXX_FLAGS'):
    flags += shlex.split(re.search(r'^' + key + r' = (.*)$', flags_file, re.M).group(1))
obj = out / 'routes.o'
subprocess.run([compiler, *flags, '-c', str(Path(__file__).with_name('vulkan_dw7_routes_test.cpp')),
                '-o', str(obj)], check=True)
link = shlex.split((vkbuild / 'CMakeFiles/ggml-vulkan.dir/link.txt').read_text())
link.remove('-shared')
link.insert(1, '-no-pie')
link[link.index('CMakeFiles/ggml-vulkan.dir/ggml-vulkan.cpp.o')] = str(obj)
executable = out / 'test_dw7_routes'
assert not executable.exists(), 'Test output must use a new directory to preserve existing receipts'
link[link.index('-o') + 1] = str(executable)
subprocess.run(link, cwd=vkbuild, check=True)
result = subprocess.run([str(executable)], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
print(result.stdout)
(out / 'routes.log').write_text(result.stdout)
(out / 'receipt.json').write_text(json.dumps({
    'source_sha256': hashlib.sha256((source / 'ggml-vulkan.cpp').read_bytes()).hexdigest(),
    'executable_sha256': hashlib.sha256(executable.read_bytes()).hexdigest(),
    'returncode': result.returncode, 'cases': 32,
    'scope': 'Real host pipeline selection; shader numerics and performance are qualified separately',
}, indent=2, ensure_ascii=False) + '\n')
raise SystemExit(result.returncode)
