"""Run the shared CI gate against real, isolated CTest registrations."""
from pathlib import Path
import os
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[5]
NAMES = ("test_qpythonruntime_binding_types", "test_qpythonruntime_plugin_loading",
         "test_qpythonruntime_binding_contracts")


def configure_gate(scenario, tmp_path):
    assert shutil.which("cmake") and shutil.which("ctest")
    source = tmp_path / "source with spaces"
    build = source / "build"
    source.mkdir()
    marker = tmp_path / "executed"
    marker.mkdir()
    cmake = "cmake_minimum_required(VERSION 3.24)\nproject(gate NONE)\nenable_testing()\n"
    cmake += "option(PLUGIN_PYTHON \"Plugin enabled\" ON)\n"
    cmake += "option(BUILD_PYTHON_MODULE \"Python modules enabled\" ON)\n"
    cmake += "option(BUILD_UNIT_TESTS \"Tests enabled\" ON)\n"
    for number, name in enumerate(NAMES):
        if scenario == "missing" and number == 1:
            continue
        script = source / (name + ".py")
        script.write_text(
            "from pathlib import Path\n"
            + f"Path({str(marker / name)!r}).touch()\n"
            + f"raise SystemExit({1 if scenario == 'failure' and number == 1 else 0})\n",
            encoding="utf-8",
        )
        cmake += f'add_test(NAME {name} COMMAND "{Path(sys.executable).as_posix()}" "{script.as_posix()}")\n'
    (source / "CMakeLists.txt").write_text(cmake, encoding="utf-8")
    options = ["-DPLUGIN_PYTHON=OFF"] if scenario == "plugin_off" else []
    if scenario == "tests_off":
        options += ["-DBUILD_UNIT_TESTS=OFF"]
    if scenario == "module_off":
        options += ["-DBUILD_PYTHON_MODULE=OFF"]
    subprocess.run(["cmake", "-S", str(source), "-B", str(build), *options], check=True,
                   capture_output=True, text=True)
    return source, build, marker


@pytest.mark.skipif(sys.platform == "win32", reason="The Bash gate is exercised on Linux and macOS")
@pytest.mark.parametrize("scenario", ["pass", "failure", "missing", "plugin_off", "tests_off", "module_off"])
@pytest.mark.parametrize("scope", ["all", "native"])
def test_ci_gate_executes_registered_plugin_tests(scenario, scope, tmp_path):
    _, build, marker = configure_gate(scenario, tmp_path)
    result = subprocess.run(
        ["bash", "-c", 'source "$1"; run_qpythonruntime_tests "$2" "$3"', "gate",
         str(ROOT / "util/ci_utils.sh"), str(build), scope],
        capture_output=True, text=True,
    )
    if scenario in ("failure", "missing"):
        assert result.returncode != 0, result.stdout + result.stderr
    else:
        assert result.returncode == 0, result.stdout + result.stderr
    executed = {path.name for path in marker.iterdir()}
    if scenario in ("pass", "failure"):
        expected = set(NAMES[:2] if scope == "native" else NAMES)
        assert executed == expected, result.stdout + result.stderr
    else:
        assert not executed, executed


def test_registered_wrapper_inherits_the_shared_binding_contract(tmp_path):
    source = tmp_path / "wrapper policy with spaces"
    source.mkdir()
    plugin = ROOT / "plugins/core/Standard/qPythonRuntime"
    (source / "entry.cpp").write_text("int entry() { return 0; }\n", encoding="utf-8")
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.24)\n"
        "project(wrapper_policy LANGUAGES CXX)\n"
        f'include("{(plugin / "cmake/Helpers.cmake").as_posix()}")\n'
        f'include("{(plugin / "wrapper/pycc/plugins/CMakeLists.txt").as_posix()}")\n'
        "add_library(sample MODULE entry.cpp)\n"
        "AddPluginWrapper(NAME sample WRAPPED wrapped)\n"
        "get_target_property(options sample COMPILE_OPTIONS)\n"
        "get_target_property(ipo sample INTERPROCEDURAL_OPTIMIZATION)\n"
        'file(WRITE "${CMAKE_BINARY_DIR}/contract.txt" "${options}\\n${ipo}\\n")\n',
        encoding="utf-8",
    )
    build = source / "build"
    subprocess.run(["cmake", "-S", str(source), "-B", str(build)], check=True,
                   capture_output=True, text=True)
    contract = (build / "contract.txt").read_text(encoding="utf-8")
    assert "BindingTypes.h" in contract, "Every registered wrapper must see the same converters"
    assert contract.splitlines()[-1] == "ON", contract


@pytest.mark.skipif(sys.platform != "win32", reason="The real app workflow gate requires native Windows PowerShell")
@pytest.mark.parametrize("scenario", ["pass", "failure", "missing", "plugin_off", "tests_off", "module_off"])
def test_windows_app_gate_runs_even_when_build_tests_are_ignored(scenario, tmp_path):
    source, _, marker = configure_gate(scenario, tmp_path)
    powershell = shutil.which("pwsh") or shutil.which("powershell")
    assert powershell, "Windows CI requires PowerShell"
    workflow = (ROOT / ".github/workflows/windows.yml").read_text(encoding="utf-8")
    steps = workflow.split("      - name: ")
    gate = next((step for step in steps if step.startswith("Run qPythonRuntime native tests\n")), "")
    # Execute the maintained workflow's script, not a test copy of its helper invocation.
    body = gate.partition("        run: |\n")[2]
    script = "\n".join(line[10:] for line in body.splitlines() if line.startswith("          "))
    script = script.replace("${{ matrix.CONFIG }}", "Release")
    result = subprocess.run(
        [powershell, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", script],
        cwd=ROOT, env=dict(os.environ, SRC_DIR=str(source), IGNORE_TEST="ON"),
        capture_output=True, text=True,
    )
    assert (result.returncode != 0) == (scenario in ("failure", "missing")), result.stdout + result.stderr
    executed = {path.name for path in marker.iterdir()}
    expected = set(NAMES[:2]) if scenario in ("pass", "failure") else set()
    assert executed == expected, result.stdout + result.stderr
