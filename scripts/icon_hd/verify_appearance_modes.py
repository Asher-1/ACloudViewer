"""Linux configured-build regression for the real appearance owner and dialog.

Unlike verify_qt_components.py this links CVAppCommon, tests Native/Custom and
settings migration, and clicks the real Display Options buttons. User settings
are isolated; this does not launch an ACloudViewer scene or infer CLI options.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, default=ROOT / 'build_app')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if not os.sys.platform.startswith('linux'):
        parser.error(
            'The actual-owner runner currently requires a Linux compile_commands build'
        )
    build = args.build.resolve()
    out = args.out.resolve() if args.out else Path(
        tempfile.mkdtemp(prefix='acv-appearance-'))
    out.mkdir(parents=True, exist_ok=True)
    cache = {}
    for line in (build / 'CMakeCache.txt').read_text().splitlines():
        if '=' in line and not line.startswith(('#', '//')):
            key, value = line.split('=', 1)
            cache[key.split(':')[0]] = value
    major = '6' if cache.get('USE_QT6') == 'ON' else '5'
    qt = Path(cache['Qt' + major + '_DIR']).parents[2]
    env = dict(os.environ)
    env['PKG_CONFIG_PATH'] = str(qt / 'lib/pkgconfig') + os.pathsep + env.get(
        'PKG_CONFIG_PATH', '')
    flags = shlex.split(
        subprocess.check_output([
            'pkg-config', '--cflags', '--libs', 'Qt' + major + 'Widgets',
            'Qt' + major + 'Test'
        ],
                                text=True,
                                env=env))
    entries = json.loads((build / 'compile_commands.json').read_text())
    entry = next(
        e for e in entries if e['file'].endswith('/ecvDisplayOptionsDlg.cpp'))
    parts = shlex.split(entry['command'])
    compile_flags = []
    for i, arg in enumerate(parts):
        if arg.startswith('-I') or (arg.startswith('-D') and not arg.startswith(
            ('-DCVAPPCOMMON_LIBRARY_BUILD', '-DCVAppCommon_EXPORTS'))):
            compile_flags.append(arg)
        elif arg == '-isystem':
            compile_flags.extend([arg, parts[i + 1]])
    subprocess.run([
        cache['CMAKE_CXX_COMPILER'], '-std=c++17', '-fPIC',
        '-Wno-deprecated-declarations', *compile_flags,
        str(ROOT / 'scripts/icon_hd/tests/appearance_modes.cpp'),
        '-L' + str(build / 'bin'), '-lCVAppCommon',
        '-Wl,-rpath,' + str(build / 'bin'), *flags, '-o',
        str(out / 'appearance_probe')
    ],
                   check=True)
    env.update(QT_QPA_PLATFORM='offscreen',
               LD_LIBRARY_PATH=str(qt / 'lib') + ':' +
               env.get('LD_LIBRARY_PATH', ''))
    results = []
    with (out / 'appearance-probe.log').open('w') as log:
        for scale in (1, 2):
            result = subprocess.run([str(out / 'appearance_probe'),
                                     str(out)],
                                    env={
                                        **env, 'QT_SCALE_FACTOR': str(scale)
                                    },
                                    stdout=log,
                                    stderr=subprocess.STDOUT)
            results.append({
                'phase': 'owner-dialog',
                'scale': scale,
                'exit_code': result.returncode
            })
        for phase in ('save-dark', 'restore-dark'):
            result = subprocess.run(
                [str(out / 'appearance_probe'),
                 str(out), phase],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT)
            results.append({'phase': phase, 'exit_code': result.returncode})
    (out / 'owner-verification.json').write_text(
        json.dumps(
            {
                'configured_build':
                    str(build),
                'qt_major':
                    major,
                'scope':
                    'Real CVAppCommon owner + Display Options; isolated settings; not full application scene automation',
                'results':
                    results
            },
            indent=2) + '\n')
    if any(row['exit_code'] for row in results):
        raise RuntimeError('Appearance regression failed; see ' +
                           str(out / 'appearance-probe.log'))
    print(
        'PASS actual owner/Display Options at DPR1/2 and separate-process restore:',
        out)


if __name__ == '__main__':
    main()
