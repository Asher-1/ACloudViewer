#!/usr/bin/env python3
"""隔离编译并执行真实 Vulkan checker，保留 source-owner 的对象和运行库。"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--build', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
build = a.build.resolve()
root = Path(__file__).resolve().parents[3]
out = a.output.resolve()
out.mkdir(parents=True, exist_ok=True)
source = build / 'llama/src/ext_llama/ggml/src/ggml-vulkan'
vkbuild = build / 'llama/src/ext_llama-build/ggml/src/ggml-vulkan'
cpp = source / 'ggml-vulkan-debug.cpp'
text = (vkbuild / 'CMakeFiles/ggml-vulkan.dir/flags.make').read_text()
compiler = re.search(r'^# compile CXX with (.+)$', text, re.M).group(1)
flags = ['-DGGML_VULKAN_CHECK_RESULTS', '-I' + str(source)]
for key in ('CXX_DEFINES', 'CXX_INCLUDES', 'CXX_FLAGS'):
    flags += shlex.split(re.search(r'^' + key + r' = (.*)$', text, re.M).group(1))
debug, main = out / 'debug.o', out / 'test.o'
subprocess.run([compiler, *flags, '-c', str(cpp), '-o', str(debug)], check=True)
subprocess.run([compiler, *flags, '-c', str(root / '3rdparty/llama/experimental/vulkan_dw_debug_clone_test.cpp'), '-o', str(main)], check=True)
link = shlex.split((vkbuild / 'CMakeFiles/ggml-vulkan.dir/link.txt').read_text())
link.remove('-shared')
link.insert(1, '-no-pie')
link[link.index('CMakeFiles/ggml-vulkan.dir/ggml-vulkan-debug.cpp.o')] = str(debug)
executable = out / 'test_dw_debug_clone'
assert not executable.is_symlink(), 'link output must not overwrite a frozen artifact'
link[link.index('-o') + 1] = str(executable)
link.append(str(main))
modules = build / 'install/bin'
cpu = modules / 'libggml-cpu-haswell.so'
link += [str(cpu), str(build / 'install/lib/libggml.so'), '-Wl,-rpath,' + str(modules)]
subprocess.run(link, cwd=vkbuild, check=True)
result = subprocess.run([str(executable)])
(out / 'receipt.json').write_text(json.dumps({
    'checker_sha256': hashlib.sha256(cpp.read_bytes()).hexdigest(),
    'cpu_module_sha256': hashlib.sha256(cpu.read_bytes()).hexdigest(),
    'executable_sha256': hashlib.sha256(executable.read_bytes()).hexdigest(),
    'returncode': result.returncode, 'checker_enabled': True,
    'input_buffer': 'host', 'input_type': 'f32', 'cases': 3,
}, indent=2) + '\n')
raise SystemExit(result.returncode)
