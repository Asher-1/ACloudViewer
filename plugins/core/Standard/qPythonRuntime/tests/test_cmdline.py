import os
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

abspath = os.path.dirname(os.path.abspath(__file__))


def assert_command_runs(binary, silent, command, *args):
    # Share the real binding test toolchain; macOS open must not return a false success.
    assert silent == "-SILENT"
    cli = shutil.which("cli-anything-acloudviewer")
    assert cli is not None, "cli-anything-acloudviewer is required"
    with tempfile.TemporaryDirectory() as config:
        env = dict(os.environ, ACV_BINARY=str(Path(binary).resolve()),
                   QT_QPA_PLATFORM="offscreen", PYTHONDONTWRITEBYTECODE="1",
                   XDG_CONFIG_HOME=config)
        completed = subprocess.run(
            [cli, "--json", "--mode", "headless", "plugin", "run",
             command.lstrip("-")] + ["--arg=" + str(arg) for arg in args],
            env=env, capture_output=True, text=True, timeout=180,
        )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads(completed.stdout)["exit_code"] == 0, completed.stdout


def test_cloud_sampling(acloudviewer_exe):
    assert_command_runs(
        acloudviewer_exe, "-SILENT", "-O", f"{abspath}/data/a_cloud.bin", "-PYTHON_SCRIPT",
        f"{abspath}/scripts/cloud_sampling_tools.py")


def test_argv(acloudviewer_exe):
    assert_command_runs(
        acloudviewer_exe, "-SILENT", "-PYTHON_SCRIPT", os.path.join(abspath, "scripts", "cmdline_args.py"), "--city",
        "dalaran")


def test_auto_segmentation(acloudviewer_exe):
    assert_command_runs(acloudviewer_exe, "-SILENT", "-O", f"{abspath}/data/a_cloud.bin", "-PYTHON_SCRIPT",
                        f"{abspath}/scripts/auto_segmentation_tools.py")


def test_normal_distribution(acloudviewer_exe):
    assert_command_runs(
        acloudviewer_exe, "-SILENT", "-O", f"{abspath}/data/a_cloud.bin", "-PYTHON_SCRIPT",
        os.path.join(abspath, "scripts", "normal_distribution.py"))


def test_normal_kdtree(acloudviewer_exe):
    assert_command_runs(
        acloudviewer_exe, "-SILENT", "-O", f"{abspath}/data/a_cloud.bin", "-PYTHON_SCRIPT",
        os.path.join(abspath, "scripts", "true_kdtree.py"))


def test_cloud_for_each(acloudviewer_exe):
    assert_command_runs(
        acloudviewer_exe, "-SILENT", "-O", f"{abspath}/data/a_cloud.bin", "-PYTHON_SCRIPT",
        os.path.join(abspath, "scripts", "cloud.py"))


def test_cloud_creation(acloudviewer_exe):
    assert_command_runs(
        acloudviewer_exe, "-SILENT", "-O", f"{abspath}/data/a_cloud.bin", "-PYTHON_SCRIPT",
        os.path.join(abspath, "scripts", "point_cloud.py"))


def test_scalarfield_tools(acloudviewer_exe):
    assert_command_runs(
        acloudviewer_exe, "-SILENT", "-O", f"{abspath}/data/a_cloud.bin", "-PYTHON_SCRIPT",
        os.path.join(abspath, "scripts", "scalarfield.py"))


def test_cchobjects_lifetimes(acloudviewer_exe):
    assert_command_runs(
        acloudviewer_exe, "-SILENT", "-PYTHON_SCRIPT", os.path.join(abspath, "scripts", "ccHObject_keep_alive.py"))


def test_pycc_tests(acloudviewer_exe):
    assert_command_runs(
        acloudviewer_exe, "-SILENT", "-PYTHON_SCRIPT", os.path.join(abspath, "pytest_runner.py"))


def test_can_call_m3c2_plugin(acloudviewer_exe):
    assert_command_runs(
        acloudviewer_exe, "-SILENT", "-O", f"{abspath}/data/a_cloud.bin", "-PYTHON_SCRIPT",
        os.path.join(abspath, "scripts", "m3c2_plugin.py"))
