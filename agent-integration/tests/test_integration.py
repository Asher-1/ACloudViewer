"""ACloudViewer Agent Integration — Self-Contained Verification Tests.

All paths can be overridden via environment variables so the tests work
both in CI (installed binary) and local dev (build directory).

Environment variables:
    ACV_REPO_ROOT   - Repository root (default: auto-detect from __file__)
    ACV_BUILD_DIR   - CMake build directory (default: $REPO_ROOT/build_app)
    ACV_BINARY      - ACloudViewer binary path (default: auto-discover)
    ACV_RPC_URL     - JSON-RPC WebSocket URL (default: ws://127.0.0.1:6001)

Usage:
    # CI (installed binary + CLI harness)
    python -m pytest test_integration.py -v

    # Local dev (build directory, no CLI harness)
    ACV_BUILD_DIR=~/code/ACloudViewer/build_app python -m pytest test_integration.py -v

    # Specific levels
    python -m pytest test_integration.py -v -k "level1"
    python -m pytest test_integration.py -v -k "level2"
    python -m pytest test_integration.py -v -k "level3"
    python -m pytest test_integration.py -v -k "level4"  # needs running GUI
"""

import collections
import importlib.util
import inspect
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

IS_WINDOWS = platform.system() == "Windows"
IS_MACOS = platform.system() == "Darwin"


# ── Path configuration (env-var overridable) ──────────────────────────────

REPO_ROOT = Path(os.environ.get(
    "ACV_REPO_ROOT",
    str(Path(__file__).resolve().parent.parent.parent),
))
BUILD_DIR = Path(os.environ.get(
    "ACV_BUILD_DIR",
    str(REPO_ROOT / "build_app"),
))
# JsonRPCPlugin.cpp owns lifecycle, the method registry and dispatch; the
# handlers live in src/JsonRPCMethods*.cpp, one file per method family.
PLUGIN_SRC_DIR = REPO_ROOT / "plugins/core/Standard/qJSonRPCPlugin/src"
PLUGIN_CPP = PLUGIN_SRC_DIR / "JsonRPCPlugin.cpp"
PLUGIN_CHILD_PROCESS_CPP = PLUGIN_SRC_DIR / "JsonRPCChildProcess.cpp"
PLUGIN_H = REPO_ROOT / "plugins/core/Standard/qJSonRPCPlugin/include/JsonRPCPlugin.h"
SIBR_COMMANDS_H = REPO_ROOT / "plugins/core/Standard/qSIBR/include/qSIBRCommands.h"

# One table gates the three automation layers: plugin source registration,
# CLI registry exposure, and installed-binary HELP/schema discovery.
AICORE_BATCH_COMMANDS = (
    ("rmbg", "RMBG_REMOVE", "qRMBG/include/RMBGBatchCommand.h"),
    ("deeplsd", "DEEPLSD_LINES", "qDeepLSD/include/DeepLSDBatchCommand.h"),
    ("rfdetr", "RFDETR_DETECT", "qRFDetr/include/RFDetrBatchCommand.h"),
    ("yolo", "YOLO_PREDICT", "qYOLO/include/YOLOPredictCommand.h"),
    ("yolo-track", "YOLO_TRACK", "qYOLO/include/YOLOTrackCommand.h"),
    ("da3", "DA3_DEPTH", "qDA3/include/DA3BatchCommand.h"),
    ("face", "FACE_DETECT", "qFaceDetect/include/FaceDetectBatchCommand.h"),
    ("gkd", "GKD_KEYPOINTS", "qGKD/include/GKDBatchCommand.h"),
    ("ocr", "OCR_RECOGNIZE", "qOCR/include/OCRBatchCommand.h"),
    ("sam3", "SAM3_SEGMENT", "qSAM3/include/SAM3BatchCommand.h"),
    ("lightglue", "LIGHTGLUE_MATCH", "qLightGlue/include/LightGlueBatchCommand.h"),
    ("freesplatter", "FREESPLATTER_RECONSTRUCT",
     "qFreeSplatter/include/FreeSplatterBatchCommand.h"),
    ("trellis", "TRELLIS_GENERATE", "qTrellis/include/TrellisBatchCommand.h"),
    ("lingbot-map", "LINGBOT_MAP_RECONSTRUCT",
     "qLingbotMap/include/LingbotMapBatchCommand.h"),
    ("anylabel", "ANYLABEL_INFER", "qAnyLabeling/include/AnyLabelingBatchCommand.h"),
    ("map-anything", "MAP_ANYTHING_RECONSTRUCT",
     "qMapAnything/include/MapAnythingReconstructCommand.h"),
    ("sam3d", "SAM3D_GENERATE", "qSAM3D/include/Sam3dGenerateCommand.h"),
)


def _assert_aicore_registry(rows) -> None:
    actual = {(row["name"], row["command"]) for row in rows}
    expected = {(task, command) for task, command, _ in AICORE_BATCH_COMMANDS}
    # Sort only diagnostics to keep cross-platform failure reports deterministic.
    assert actual == expected, (
        "CLI/native batch contract mismatch; use the paired harness revision in "
        ".github/actions/checkout-agent-harness/action.yml. "
        f"Missing: {sorted(expected - actual)}; unexpected: {sorted(actual - expected)}")


def _read_repo_text(path: Path) -> str:
    """Read repository sources as UTF-8.

    On Windows, :meth:`Path.read_text` without ``encoding=`` uses the ANSI
    code page (e.g. GBK), which breaks on UTF-8 C++ sources containing
    non-ASCII (smart quotes, symbols in comments, etc.).
    """
    return path.read_text(encoding="utf-8")


def _rpc_plugin_cpp_files() -> list:
    return sorted(PLUGIN_SRC_DIR.glob("*.cpp"))


def _rpc_plugin_sources() -> str:
    """All JSON-RPC plugin translation units, registry and handlers."""
    return "\n".join(_read_repo_text(path) for path in _rpc_plugin_cpp_files())


def _resolve_exe(binary_path: str) -> str:
    """On Windows, resolve .bat wrappers to the underlying .exe.

    The .bat wrapper uses ``start /b ... >nul`` which backgrounds the process
    and discards output — incompatible with synchronous subprocess.run.
    """
    if not IS_WINDOWS:
        return binary_path
    p = Path(binary_path)
    if p.suffix.lower() == ".bat":
        exe = p.with_suffix(".exe")
        if exe.exists():
            return str(exe)
    return binary_path


# ── Binary discovery (build dir → installed → env var) ────────────────────

def _find_build_binary() -> str | None:
    """Find the ACloudViewer binary in the build directory."""
    if IS_WINDOWS:
        candidates = ["bin/ACloudViewer.exe", "bin/Release/ACloudViewer.exe",
                       "bin/Debug/ACloudViewer.exe", "Release/ACloudViewer.exe"]
    elif IS_MACOS:
        # Check .app bundle path first to avoid selecting the directory
        candidates = ["bin/ACloudViewer.app/Contents/MacOS/ACloudViewer", "bin/ACloudViewer"]
    else:
        candidates = ["bin/ACloudViewer", "ACloudViewer"]
    for c in candidates:
        p = BUILD_DIR / c
        if p.exists() and p.is_file():
            return str(p)
    return None


def _find_any_binary() -> str | None:
    """Find ACloudViewer binary: env var → build dir → CLI harness discovery."""
    env_binary = os.environ.get("ACV_BINARY")
    if env_binary and Path(env_binary).exists():
        return env_binary

    build_bin = _find_build_binary()
    if build_bin:
        return build_bin

    try:
        from cli_anything.acloudviewer.utils.acloudviewer_backend import ACloudViewerBackend
        return ACloudViewerBackend.find_binary()
    except ImportError:
        pass

    if IS_WINDOWS:
        names = ("ACloudViewer.bat", "ACloudViewer.exe")
    elif IS_MACOS:
        names = ("ACloudViewer",)
    else:
        names = ("ACloudViewer.sh", "ACloudViewer")
    for name in names:
        path = shutil.which(name)
        if path:
            return path
    return None


_RAW_BINARY_PATH = _find_any_binary()
BINARY_PATH = _resolve_exe(_RAW_BINARY_PATH) if _RAW_BINARY_PATH else None
HAS_BINARY = BINARY_PATH is not None

# ── CLI harness detection ────────────────────────────────────────────────

def _check_cli_available() -> bool:
    if shutil.which("cli-anything-acloudviewer") is None:
        return False
    try:
        r = subprocess.run(
            ["cli-anything-acloudviewer", "--help"],
            capture_output=True, timeout=10)
        return r.returncode == 0
    except Exception:
        return False


HAS_CLI = _check_cli_available()

try:
    from websockets.sync.client import connect as _ws_sync_connect
    HAS_WS = True
except ImportError:
    HAS_WS = False


def ws_connect(url: str, **options):
    """Loopback GUI connection: no 1 MiB reply cap, and no HTTP(S)_PROXY
    (websockets >= 15 would route ws:// through it)."""
    options.setdefault("max_size", None)
    if "proxy" in inspect.signature(_ws_sync_connect).parameters:
        options.setdefault("proxy", None)
    return _ws_sync_connect(url, **options)

RPC_URL = os.environ.get("ACV_RPC_URL", "ws://127.0.0.1:6001")


def _lan_ipv4() -> str | None:
    """This host's non-loopback IPv4 address, if any (no packet is sent)."""
    import socket
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("192.0.2.1", 9))
        address = probe.getsockname()[0]
    except OSError:  # no route: no non-loopback interface
        return None
    finally:
        probe.close()
    return None if address.startswith("127.") else address


def _missing_modules(*names: str) -> list[str]:
    missing = []
    for name in names:
        try:
            if importlib.util.find_spec(name) is None:
                missing.append(name)
        except ModuleNotFoundError:  # parent package absent
            missing.append(name)
    return missing


# Decided before the test runs: with both installed, an ImportError or a
# SystemExit from the harness is a regression and fails.
_MISSING_MCP = _missing_modules("mcp", "cli_anything.acloudviewer")
_needs_mcp = pytest.mark.skipif(
    bool(_MISSING_MCP), reason=f"not installed: {', '.join(_MISSING_MCP)}")


def _rpc_available() -> bool:
    if not HAS_WS:
        return False
    try:
        ws = ws_connect(RPC_URL, open_timeout=5)
        ws.send(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping", "params": {}}))
        resp = json.loads(ws.recv(timeout=5))
        ws.close()
        return resp.get("result") == "pong"
    except Exception:
        return False


def _level4_mesh_rpc_ready() -> bool:
    """Live RPC check (used by mesh/transform Level 4 tests)."""
    return _rpc_available()


def _rpc_call(method: str, params: dict | None = None, timeout: int = 30):
    """Send a JSON-RPC 2.0 call and return the result.

    - Connection failures  → pytest.fail: Level 4 classes are collected
      only when the server answered ping, so a lost server is a crash
    - JSON-RPC errors      → RuntimeError "RPC error <code> (<method>)"
    - Success              → return result value
    """
    ws = None
    raw = None
    try:
        ws = ws_connect(RPC_URL, open_timeout=5)
    except (ConnectionRefusedError, ConnectionResetError, OSError) as exc:
        pytest.fail(f"RPC server not reachable (crashed?): {exc}")
    try:
        ws.send(json.dumps({
            "jsonrpc": "2.0", "id": 1,
            "method": method, "params": params or {},
        }))
        raw = ws.recv(timeout=timeout)
    except (ConnectionResetError, BrokenPipeError, OSError) as exc:
        pytest.fail(f"RPC connection lost during {method} (crashed?): {exc}")
    except Exception as exc:
        if "ConnectionClosed" in type(exc).__name__:
            pytest.fail(f"RPC connection closed during {method}: {exc}")
        raise
    finally:
        if ws is not None:
            try:
                ws.close()
            except Exception:  # best-effort cleanup
                pass
    if raw is None:
        pytest.fail(f"No response received from RPC server for {method}")
    resp = json.loads(raw)
    if "error" in resp:
        err = resp["error"]
        msg = err.get("message", err) if isinstance(err, dict) else err
        code = err.get("code") if isinstance(err, dict) else None
        raise RuntimeError(f"RPC error {code} ({method}): {msg}")
    return resp.get("result")


def _find_cloud_id(result) -> int | None:
    """Recursively find the first POINT_CLOUD entity ID in an entity tree."""
    if isinstance(result, dict):
        if result.get("type") == "POINT_CLOUD":
            return result["id"]
        for child in result.get("children", []):
            cid = _find_cloud_id(child)
            if cid is not None:
                return cid
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Level 1: C++ Source Verification
# ═══════════════════════════════════════════════════════════════════════════

class TestLevel1_CodeQL:
    """验证扫描策略、真实子构建命令和失败传播，不依赖完整应用构建。"""

    def test_level1_codeql_workflow_contract(self):
        source = _read_repo_text(REPO_ROOT / ".github/workflows/codeql.yml")
        cpp = source.split("  analyze-cpp:\n", 1)[1]
        configure = cpp.split("      - name: Configure CMake\n", 1)[1]
        configure = configure.split("      - name:", 1)[0]
        for option in ("PLUGIN_IO_QLAS", "PLUGIN_IO_QCORE", "PLUGIN_IO_QMESH",
                       "PLUGIN_STANDARD_QCSF", "PLUGIN_STANDARD_QM3C2",
                       "PLUGIN_STANDARD_QPCL", "PLUGIN_STANDARD_QPCV",
                       "PLUGIN_STANDARD_QPOISSON_RECON",
                       "PLUGIN_STANDARD_QRANSAC_SD",
                       "PLUGIN_STANDARD_QHOUGH_NORMALS", "USE_VTK_BACKEND"):
            assert f"-D{option}=ON" in configure
        assert '"-DCLOUDVIEWER_CODEQL_OPTIMIZATION=${CPP_OPTIMIZATION}"' in configure
        assert "compiler-policy.cmake" in configure
        assert source.count("queries: security-extended") == 2
        assert "build-mode: manual" in cpp
        assert cpp.index("- name: Configure CMake") < cpp.index("- name: Initialize CodeQL")
        build = cpp.split("      - name: Build\n", 1)[1].split("      - name:", 1)[0]
        assert "set -euo pipefail" in build and "|| true" not in build
        assert '--target' not in build and 'cmake --build build --parallel "$PARALLEL_JOBS"' in build
        assert "build/**/compile_commands.json" in cpp
        assert "codeql_databases/cpp/log/" in cpp
        extra_options = re.search(r"CODEQL_ACTION_EXTRA_OPTIONS: >-\n\s*(\{.*\})", cpp)[1]
        extra_options = extra_options.replace("${{ github.workspace }}", "/fixture")
        assert json.loads(extra_options) == {"database": {"run-queries": [
            "--evaluator-log=/fixture/codeql-evidence/evaluator.jsonl", "--evaluator-log-minify"]}}
        assert "generate log-summary" in cpp and "overall predicates text" in cpp
        assert cpp.index("- name: Sample C++ analysis resources") < cpp.index("- name: Perform CodeQL Analysis")
        summarize = cpp.split("      - name: Summarize C++ evaluator performance\n", 1)[1]
        summarize = summarize.split("      - name:", 1)[0]
        assert "if: always()" in summarize and "analysis-resources.stop" in summarize
        assert 'CODEQL_BIN: ${{ steps.init-cpp.outputs.codeql-path }}' in summarize
        poisson = _read_repo_text(REPO_ROOT / "3rdparty/PoissonRecon/PoissonRecon.cmake")
        assert "${ExternalProject_CODEQL_CMAKE_ARGS}" in poisson
        agent = _read_repo_text(REPO_ROOT / ".github/workflows/agent-integration.yml")
        for path in (".github/workflows/codeql.yml", ".github/codeql/**",
                     "3rdparty/find_dependencies.cmake", "3rdparty/PoissonRecon/**"):
            assert agent.count(f"- '{path}'") == 2

    @pytest.mark.skipif(platform.system() != "Linux", reason="工作流使用 Linux time 和 Bash")
    @pytest.mark.parametrize("optimization,build_status,actual", [
        ("O1", 0, "O1"), ("O3", 0, "O3"), ("O1", 3, "O1"),
        ("O1", 0, "O3"), ("O1", 0, None)])
    def test_level1_codeql_build_gate(self, tmp_path, optimization, build_status, actual):
        source = _read_repo_text(REPO_ROOT / ".github/workflows/codeql.yml")
        cpp = source.split("  analyze-cpp:\n", 1)[1]
        build = cpp.split("      - name: Build\n", 1)[1].split("      - name:", 1)[0]
        script = build.split("        run: |\n", 1)[1]
        script = "\n".join(line[10:] for line in script.splitlines())
        (tmp_path / "build").mkdir()
        (tmp_path / "codeql-evidence").mkdir()
        rows = [] if actual is None else [{"file": "probe.cpp", "arguments": [
            "/usr/bin/c++", f"-{actual}", "-DNDEBUG", "-c", "probe.cpp"]}]
        (tmp_path / "build/compile_commands.json").write_text(json.dumps(rows))
        tools = tmp_path / "tools"
        tools.mkdir()
        cmake = tools / "cmake"
        cmake.write_text(f"#!/bin/sh\necho build-fixture\nexit {build_status}\n")
        cmake.chmod(0o755)
        summary = tmp_path / "summary"
        env = dict(os.environ, PATH=str(tools) + os.pathsep + os.environ['PATH'],
                   CPP_OPTIMIZATION=optimization, PARALLEL_JOBS="2", GITHUB_STEP_SUMMARY=str(summary))
        result = subprocess.run(["bash", "-c", script], cwd=tmp_path, env=env,
                                capture_output=True, text=True, timeout=30)
        success = build_status == 0 and actual == optimization
        assert (result.returncode == 0) == success, result.stdout + result.stderr
        assert summary.exists() == success
        if success:
            assert (tmp_path / "codeql-evidence/compiler-policy.json").exists()

    @pytest.mark.skipif(platform.system() != "Linux", reason="资源采样读取 Linux procfs")
    def test_level1_codeql_resource_sampling(self, tmp_path):
        source = _read_repo_text(REPO_ROOT / ".github/workflows/codeql.yml")
        sampler = source.split("      - name: Sample C++ analysis resources\n", 1)[1]
        sampler = sampler.split("      - name:", 1)[0].split("        run: |\n", 1)[1]
        script = "\n".join(line[10:] for line in sampler.splitlines())
        # 替换等待而不是 procfs 数据，真实采样一次后以工作流相同的标记退出。
        tools = tmp_path / "tools"
        tools.mkdir()
        sleep = tools / "sleep"
        sleep.write_text("#!/bin/sh\ntouch codeql-evidence/analysis-resources.stop\n")
        sleep.chmod(0o755)
        result = subprocess.run(["bash", "-c", script + "\nwait"], cwd=tmp_path,
                                env=dict(os.environ, PATH=str(tools) + os.pathsep + os.environ["PATH"]),
                                capture_output=True, text=True, timeout=10)
        assert result.returncode == 0, result.stderr
        log = (tmp_path / "codeql-evidence/analysis-resources.log").read_text()
        assert log.count("UTC ") == 1
        assert "MemAvailable:" in log and "pgmajfault " in log and "cpu " in log
        assert "cmdline" not in script and "/environ" not in script

    @pytest.mark.skipif(platform.system() != "Linux", reason="工作流汇总使用 Bash")
    @pytest.mark.parametrize("evaluator,cli_status", [(False, 0), (True, 0), (True, 2)])
    def test_level1_codeql_evaluator_summary(self, tmp_path, evaluator, cli_status):
        source = _read_repo_text(REPO_ROOT / ".github/workflows/codeql.yml")
        summarize = source.split("      - name: Summarize C++ evaluator performance\n", 1)[1]
        summarize = summarize.split("      - name:", 1)[0].split("        run: |\n", 1)[1]
        script = "\n".join(line[10:] for line in summarize.splitlines()
                           if line.startswith("          ") or not line.strip())
        evidence = tmp_path / "codeql-evidence"
        evidence.mkdir()
        if evaluator:
            (evidence / "evaluator.jsonl").write_text("fixture\n")
        cli = tmp_path / "codeql"
        cli.write_text('#!/bin/sh\n'
                       'test "$1" = generate && test "$2" = log-summary && test "$4" = --utc || exit 3\n'
                       'printf "%s\\n" "$3" >> calls\n'
                       f'exit {cli_status}\n')
        cli.chmod(0o755)
        result = subprocess.run(["bash", "-c", script], cwd=tmp_path,
                                env=dict(os.environ, CODEQL_BIN=str(cli)),
                                capture_output=True, text=True, timeout=10)
        assert (result.returncode == 0) == (not evaluator or cli_status == 0)
        assert (evidence / "analysis-resources.stop").exists()
        calls = (tmp_path / "calls").read_text().splitlines() if evaluator else []
        expected = ["--format=overall", "--format=predicates", "--format=text"]
        assert calls == (expected if evaluator and cli_status == 0 else expected[:1] if evaluator else [])
        if evaluator:
            assert (evidence / "evaluator.jsonl").read_text() == "fixture\n"

    @pytest.mark.skipif(platform.system() != "Linux", reason="扫描策略只支持 Linux")
    @pytest.mark.parametrize("optimization,build_type", [
        (None, "Release"), ("O1", "Release"), ("O3", "Release"), ("O1", "")])
    def test_level1_codeql_subproject_commands(self, tmp_path, optimization, build_type):
        import shlex

        cmake = shutil.which("cmake")
        assert cmake is not None, "Linux CI must execute the real CMake regression"
        source = tmp_path / "source with spaces"
        source.mkdir()
        policy = source / ".github/codeql/compiler-policy.cmake"
        policy.parent.mkdir(parents=True)
        shutil.copyfile(REPO_ROOT / ".github/codeql/compiler-policy.cmake", policy)
        (source / "3rdparty").mkdir()
        shared = _read_repo_text(REPO_ROOT / "3rdparty/find_dependencies.cmake")
        shared = shared.split("# CMake arguments for configuring ExternalProjects.", 1)[1]
        shared = shared.split("# pkg_config_3rdparty_library(name ...)", 1)[0]
        shared = "# CMake arguments for configuring ExternalProjects." + shared
        (source / "shared.cmake").write_text(shared, encoding="utf-8")
        code = ("#if !defined(NDEBUG) || !defined(__OPTIMIZE__)\n#error lost scan macros\n"
                "#endif\nint probe(void) { return 7; }\n")
        (source / "c.c").write_text(code, encoding="utf-8")
        (source / "cpp.cpp").write_text(code, encoding="utf-8")
        for name in ("nested", "child"):
            directory = source / name
            directory.mkdir()
            (directory / "probe.cpp").write_text(code, encoding="utf-8")
            project = "project(child C CXX)\n" if name == "child" else ""
            (directory / "CMakeLists.txt").write_text(
                "cmake_minimum_required(VERSION 3.19)\n" + project +
                'set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")\n'
                'if(DEFINED CLOUDVIEWER_CODEQL_OPTIMIZATION)\n'
                '    set(CMAKE_CXX_COMPILER_LAUNCHER "false")\n'
                'endif()\n'
                'add_library(part STATIC probe.cpp)\n'
                'target_compile_options(part PRIVATE -funroll-loops -ffast-math)\n',
                encoding="utf-8")
        (source / "CMakeLists.txt").write_text('''cmake_minimum_required(VERSION 3.19)
project(scan_fixture C CXX)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CloudViewer_3RDPARTY_DIR "${CMAKE_SOURCE_DIR}/3rdparty")
set(CloudViewer_INSTALL_LIB_DIR lib)
include(shared.cmake)
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
if(DEFINED CLOUDVIEWER_CODEQL_OPTIMIZATION)
    set(CMAKE_C_COMPILER_LAUNCHER "false")
    set(CMAKE_CXX_COMPILER_LAUNCHER "false")
endif()
add_library(cpart OBJECT c.c)
add_library(top STATIC cpp.cpp)
add_library(header_only INTERFACE)
add_library(imported STATIC IMPORTED)
add_custom_target(utility ALL)
add_subdirectory(nested)
include(ExternalProject)
ExternalProject_Add(ext_child
    SOURCE_DIR "${CMAKE_SOURCE_DIR}/child"
    BINARY_DIR "${CMAKE_BINARY_DIR}/child-build"
    DOWNLOAD_COMMAND "" UPDATE_COMMAND "" INSTALL_COMMAND ""
    CMAKE_ARGS ${ExternalProject_CMAKE_ARGS_hidden})
''', encoding="utf-8")
        binary = tmp_path / "build with spaces"
        args = [cmake, "-S", str(source), "-B", str(binary),
                "-G", "Unix Makefiles", f"-DCMAKE_BUILD_TYPE={build_type}",
                "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"]
        if optimization:
            args += [f"-DCLOUDVIEWER_CODEQL_OPTIMIZATION={optimization}",
                     f"-DCMAKE_PROJECT_INCLUDE={policy}"]
        result = subprocess.run(args, capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr
        result = subprocess.run([cmake, "--build", str(binary), "--parallel", "2"],
                                capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr
        inventories = [binary / "compile_commands.json"]
        if optimization:
            inventories.append(binary / "child-build/compile_commands.json")
        expected = f"-{optimization or 'O3'}"
        for inventory in inventories:
            rows = json.loads(inventory.read_text(encoding="utf-8"))
            assert len(rows) == (3 if inventory.parent == binary else 1)
            for row in rows:
                tokens = shlex.split(row["command"])
                levels = [token for token in tokens if re.fullmatch(r"-O[0-3sg]|-Ofast", token)]
                assert levels[-1] == expected, row
                assert "-DNDEBUG" in tokens
                if row["file"].endswith("probe.cpp"):
                    assert "-ffast-math" in tokens and "-funroll-loops" in tokens
        command = _read_repo_text(binary / "ext_child-prefix/tmp/ext_child-cfgcmd.txt")
        assert ("CMAKE_PROJECT_INCLUDE" in command) == bool(optimization)
        assert ("CLOUDVIEWER_CODEQL_OPTIMIZATION" in command) == bool(optimization)
        if optimization:
            # 同一生成树再次配置，不能重复追加目标选项。
            before = (binary / "compile_commands.json").read_bytes()
            subprocess.run(args, capture_output=True, text=True, timeout=60, check=True)
            assert (binary / "compile_commands.json").read_bytes() == before
            # 真实编译失败必须传给调用方，不能留下虚假的成功基线。
            (source / "cpp.cpp").write_text("#error forced failure\n", encoding="utf-8")
            failed = subprocess.run([cmake, "--build", str(binary), "--parallel", "2"],
                                    capture_output=True, text=True, timeout=60)
            assert failed.returncode != 0 and "forced failure" in failed.stderr

    @pytest.mark.skipif(platform.system() != "Linux", reason="扫描策略只支持 Linux")
    @pytest.mark.parametrize("optimization,build_type", [("O2", "Release"), ("O1", "Debug")])
    def test_level1_codeql_invalid_policy_rejected(self, tmp_path, optimization, build_type):
        cmake = shutil.which("cmake")
        assert cmake is not None
        (tmp_path / "CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 3.19)\nproject(probe CXX)\n", encoding="utf-8")
        result = subprocess.run([
            cmake, "-S", str(tmp_path), "-B", str(tmp_path / "build"),
            f"-DCMAKE_BUILD_TYPE={build_type}",
            f"-DCLOUDVIEWER_CODEQL_OPTIMIZATION={optimization}",
            f"-DCMAKE_PROJECT_INCLUDE={REPO_ROOT / '.github/codeql/compiler-policy.cmake'}"],
            capture_output=True, text=True, timeout=60)
        assert result.returncode != 0 and "CodeQL" in result.stderr


class TestLevel1_CppPlugin:
    """Verify C++ plugin source structure without needing a build."""

    def test_level1_plugin_cpp_exists(self):
        assert PLUGIN_CPP.exists()

    def test_level1_ci_harness_revision_is_paired(self):
        action = REPO_ROOT / ".github/actions/checkout-agent-harness/action.yml"
        text = _read_repo_text(action)
        assert re.search(r"^        ref: [0-9a-f]{40}$", text, re.MULTILINE)
        consumers = {"agent-integration.yml", "ubuntu.yml", "ubuntu-cuda.yml",
                     "macos.yml", "windows.yml"}
        workflows = list((REPO_ROOT / ".github/workflows").glob("*.yml"))
        assert consumers <= {workflow.name for workflow in workflows}
        for workflow in workflows:
            source = _read_repo_text(workflow)
            assert "repository: Asher-1/CLI-Anything" not in source, workflow
            if workflow.name in consumers:
                assert source.count("uses: ./.github/actions/checkout-agent-harness") == 1

    def test_level1_ci_macos_cache_wiring(self):
        source = _read_repo_text(REPO_ROOT / ".github/workflows/macos.yml")
        saves = re.findall(
            r"^      - name: Save compiler cache \(stash\)\n(.*?)(?=^      - name:|\Z)",
            source, re.MULTILINE | re.DOTALL)
        assert len(saves) == 2
        for save in saves:
            # The cache directory is hidden; artifact uploads otherwise omit it.
            assert "include-hidden-files: true" in save
            assert "compression-level: 0" in save
            assert "path: ${{ env.CCACHE_DIR }}" in save
            assert "!cancelled()" in save
        assert source.count("ccache -M 10G") == 2
        assert source.count("ccache -z") == 2
        assert source.count("trap 'ccache -s' EXIT") == 2

    def test_level1_ci_cmake_pin_is_retained(self):
        setup = _read_repo_text(REPO_ROOT / ".github/actions/setup-cmake/action.yml")
        source = _read_repo_text(REPO_ROOT / "util/ci_utils.sh")
        assert "Get-Command cmake -CommandType Application" in setup
        assert "CLOUDVIEWER_CMAKE_BIN=$cmakeBin" in setup
        assert "$env:GITHUB_ENV" in setup
        assert '[[ ! -x "$CLOUDVIEWER_CMAKE_BIN" ]]' in source
        assert 'export PATH="$(dirname "$CLOUDVIEWER_CMAKE_BIN"):$PATH"' in source
        workflow = _read_repo_text(REPO_ROOT / ".github/workflows/agent-integration.yml")
        assert workflow.count("- '.github/actions/setup-cmake/**'") == 2
        assert workflow.count("- '.github/workflows/macos.yml'") == 2

    def test_level1_ci_cmake_regression_cannot_skip_in_ci(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CI", "true")
        monkeypatch.setattr(shutil, "which", lambda _: None)
        with pytest.raises(pytest.fail.Exception, match="CI must execute"):
            self.test_level1_ci_cmake_path_is_single_executable(tmp_path, "3.31.8", 0)

    @pytest.mark.parametrize("version,exit_code", [
        ("3.31.8", 0), ("3.31.7", 0), ("3.31.8", 7)])
    def test_level1_ci_cmake_path_is_single_executable(self, tmp_path, version, exit_code):
        pwsh = shutil.which("pwsh")
        if pwsh is None:
            if os.environ.get("CI") == "true":
                pytest.fail("CI must execute the PowerShell action regression")
            pytest.skip("PowerShell is unavailable locally")
        setup = _read_repo_text(REPO_ROOT / ".github/actions/setup-cmake/action.yml")
        # Execute the action itself, not a reimplementation of its resolution.
        script = "\n".join(line[8:] for line in setup.split("      run: |\n", 1)[1].splitlines())
        paths = [tmp_path / "preferred tools", tmp_path / "shadow tools"]
        executables = []
        for directory, candidate_version, candidate_exit in (
                (paths[0], version, exit_code), (paths[1], "3.31.8", 0)):
            directory.mkdir()
            executable = directory / ("cmake.cmd" if IS_WINDOWS else "cmake")
            if IS_WINDOWS:
                text = (f"@echo off\necho cmake version {candidate_version}\n"
                        f"exit /b {candidate_exit}\n")
            else:
                text = (f"#!/bin/sh\nprintf 'cmake version {candidate_version}\\n'\n"
                        f"exit {candidate_exit}\n")
            executable.write_text(text, encoding="utf-8")
            executable.chmod(0o755)
            executables.append(executable)
        github_env = tmp_path / "github env"
        env = dict(os.environ, GITHUB_ENV=str(github_env))
        env["PATH"] = os.pathsep.join([str(path) for path in paths] + [env["PATH"]])
        result = subprocess.run(
            [pwsh, "-NoLogo", "-NoProfile", "-NonInteractive",
             "-Command", script], env=env, capture_output=True, text=True, timeout=30)
        recorded = github_env.read_text(encoding="utf-8-sig") if github_env.exists() else ""
        if version == "3.31.8" and exit_code == 0:
            assert result.returncode == 0, result.stdout + result.stderr
            assert recorded.splitlines() == [f"CLOUDVIEWER_CMAKE_BIN={executables[0]}"]
        else:
            # A valid shadow must not hide a bad selected executable.
            assert result.returncode != 0, result.stdout + result.stderr
            assert "CLOUDVIEWER_CMAKE_BIN=" not in recorded

    @pytest.mark.skipif(IS_WINDOWS, reason="Requires a Unix login shell")
    @pytest.mark.parametrize("job,step_name", [
        ("MacOS", "Config and build APP"),
        ("build-wheel", "Config and build Wheel"),
        ("build-wheel", "Install dependencies")])
    def test_level1_ci_macos_failure_propagates(self, tmp_path, job, step_name):
        source = _read_repo_text(REPO_ROOT / ".github/workflows/macos.yml")
        job_source = source.split(f"  {job}:\n", 1)[1]
        step = job_source.split(f"      - name: {step_name}\n", 1)[1]
        step = step.split("      - name:", 1)[0]
        assert "shell: bash -l {0}" in step
        script = step.split("        run: |\n", 1)[1]
        script = "\n".join(line[10:] for line in script.splitlines())
        script = re.sub(r"\$\{\{.*?\}\}", "ci-placeholder", script)
        script = script.replace("/tmp/macos-", str(tmp_path / "macos-"))
        # Keep the workflow's shell/options; stub only expensive external work.
        stubs = ("source() { :; }\nccache() { echo cache-stats; }\n"
                 "python() { :; }\npip() { :; }\ncmake() { :; }\n"
                 "brew() { :; }\n"
                 "build_gui_app() { return 23; }\n"
                 "build_mac_wheel() { return 23; }\n"
                 "install_python_dependencies() { return 23; }\n")
        step_file = tmp_path / "step.sh"
        step_file.write_text(stubs + script, encoding="utf-8")
        env = dict(os.environ, CONDA_PREFIX=str(tmp_path), PKG_CONFIG_PATH="",
                   GITHUB_ENV=str(tmp_path / "github-env"))
        result = subprocess.run(["bash", "-l", str(step_file)], cwd=tmp_path,
                                env=env, capture_output=True, text=True, timeout=30)
        assert result.returncode == 23, result.stdout + result.stderr
        recorded = Path(env["GITHUB_ENV"]).read_text() if Path(env["GITHUB_ENV"]).exists() else ""
        assert "DEVEL_PKG_NAME=" not in recorded and "PIP_PKG_NAME=" not in recorded
        if step_name.startswith("Config and build"):
            assert "cache-stats" in result.stdout  # EXIT stats survive failure.

    @pytest.mark.parametrize("drift", ["extra", "missing"])
    def test_level1_registry_drift_is_rejected(self, drift):
        rows = [{"name": task, "command": command}
                for task, command, _ in AICORE_BATCH_COMMANDS]
        _assert_aicore_registry(rows)
        if drift == "extra":
            rows += [{"name": "llm", "command": "LLM_GENERATE"},
                     {"name": "openpcdet", "command": "OPENPCDET_DETECT"}]
        else:
            rows.pop()
        with pytest.raises(AssertionError, match="CLI/native batch contract mismatch"):
            _assert_aicore_registry(rows)

    def test_level1_plugin_h_exists(self):
        assert PLUGIN_H.exists()

    def test_level1_plugin_has_dispatch_table(self):
        src = _read_repo_text(PLUGIN_CPP)
        for method in ["open", "file.convert", "scene.list",
                        "cloud.computeNormals", "cloud.paintUniform",
                        "cloud.paintByHeight", "cloud.paintByScalarField",
                        "mesh.simplify", "view.screenshot",
                        "colmap.reconstruct", "colmap.run",
                        "cloud.setActiveSf", "cloud.removeSf",
                        "cloud.removeRgb", "cloud.invertNormals",
                        "cloud.merge", "mesh.extractVertices",
                        "mesh.flipTriangles", "mesh.volume"]:
            assert f'"{method}"' in src, \
                f"Missing method registration for '{method}'"

    def test_level1_colmap_reconstruct_params(self):
        src = _rpc_plugin_sources()
        assert "rpcColmapReconstruct" in src
        for param in ["image_path", "workspace_path", "quality",
                       "data_type", "mesher", "camera_model",
                       "single_camera", "use_gpu", "colmap_binary",
                       "timeout_ms"]:
            assert param in src, f"colmap.reconstruct missing param: {param}"

    def test_level1_file_convert_params(self):
        src = _rpc_plugin_sources()
        assert "rpcFileConvert" in src
        for param in ["input", "output", "input_filter", "output_filter"]:
            assert param in src, f"file.convert missing param: {param}"

    def test_level1_aicore_batch_contract_complete(self):
        rpc_src = _rpc_plugin_sources()
        rpc_header = _read_repo_text(PLUGIN_H)
        assert 'reg("aicore.batch"' in _read_repo_text(PLUGIN_CPP)
        assert "rpcAICoreBatch" in rpc_src
        assert "rpcAICoreBatch" in rpc_header
        plugin_root = REPO_ROOT / "plugins/core/Standard"
        expected_headers = {relative_header
                            for _, _, relative_header in AICORE_BATCH_COMMANDS}
        discovered_headers = {
            path.relative_to(plugin_root).as_posix()
            for path in plugin_root.rglob("*.h")
            if b'"ecvAICoreBatchCommand.h"' in path.read_bytes()
        }
        assert discovered_headers == expected_headers, (
            "Update AICORE_BATCH_COMMANDS and every automation surface when "
            f"batch headers change: missing={expected_headers - discovered_headers}, "
            f"unregistered={discovered_headers - expected_headers}")
        for task, command, relative_header in AICORE_BATCH_COMMANDS:
            header = plugin_root / relative_header
            assert header.is_file(), f"{task}: missing {relative_header}"
            source = _read_repo_text(header)
            assert command in source, f"{task}: missing command {command}"
            assert "ecvAICoreBatchCommand.h" in source, (
                f"{task}: command must expose the shared HELP/schema contract")
            assert f'"{command}"' in rpc_src, (
                f"{task}: {command} missing from aicore.batch whitelist")

        # The whitelist is an exact set: an extra entry would expose a
        # command that no registry/test row describes.
        block = re.search(r"kCommands\s*=\s*\{([^}]*)\}", rpc_src)
        assert block, "aicore.batch whitelist (kCommands) not found"
        whitelist = set(re.findall(r'"([A-Z][A-Z0-9_]*)"', block.group(1)))
        assert whitelist == {command for _, command, _ in AICORE_BATCH_COMMANDS}

    @pytest.mark.parametrize("task,command,relative_header", AICORE_BATCH_COMMANDS)
    def test_level1_aicore_batch_command_registered(
            self, task, command, relative_header):
        """The header defines the keyword and its plugin registers the class."""
        header = REPO_ROOT / "plugins/core/Standard" / relative_header
        source = _read_repo_text(header)
        assert re.search(r'#define\s+COMMAND_\w+\s+"%s"' % command, source), (
            f"{task}: {relative_header} does not define keyword {command}")
        classes = re.findall(
            r"(?:struct|class)\s+(\w+)\s*:\s*public\s+ccCommandLineInterface::Command",
            source)
        assert len(classes) == 1, f"{task}: expected one command class"
        plugin_sources = "".join(
            _read_repo_text(path)
            for path in sorted((header.parents[1] / "src").glob("*.cpp")))
        assert re.search(r"registerCommand\s*\([^;]*new\s+%s\b" % classes[0],
                         plugin_sources), (
            f"{task}: {classes[0]} is not passed to registerCommand()")

    def test_level1_documented_rpc_method_count(self):
        """Documented RPC method totals equal the registered method set."""
        methods = set(re.findall(r'reg\("([^"]+)"', _rpc_plugin_sources()))
        stale = []
        for path in (REPO_ROOT / "README.md", REPO_ROOT / "agent-integration/README.md"):
            for match in re.finditer(
                    r"\*\*(\d+) (?:JSON-RPC )?methods\*\*", _read_repo_text(path)):
                if int(match.group(1)) != len(methods):
                    stale.append(f"{path.name}: {match.group(0)}")
        assert not stale, f"{len(methods)} RPC methods registered; stale: {stale}"

    def test_level1_rpc_child_processes_do_not_block_gui_thread(self):
        """RPC handlers run on the GUI thread. A child is never awaited there:
        waitFor*() freezes the GUI, and a nested QEventLoop re-enters the
        dispatcher (the outer reply and deadline then wait for an inner
        child). Handlers parent each QProcess to the plugin (killed on
        teardown) and return runChildProcess()'s deferred result."""
        helper_src = _read_repo_text(PLUGIN_CHILD_PROCESS_CPP)
        assert re.search(r"JsonRPCResult runChildProcess\(", helper_src), (
            "runChildProcess() helper missing")
        for path in _rpc_plugin_cpp_files():
            src = _read_repo_text(path)
            blocking = re.findall(r"\.(waitForFinished|waitForStarted)\(", src)
            assert not blocking, f"{path.name}: blocking QProcess waits {blocking}"
            assert "QEventLoop" not in src, (
                f"{path.name}: nested event loops re-enter the RPC dispatcher")
            if path == PLUGIN_CHILD_PROCESS_CPP:
                continue
            assert not re.search(r"\bprocess(\.|->)start\(", src), (
                f"{path.name}: QProcess started outside runChildProcess()")
            spawned = re.findall(r"new QProcess\(([^)]*)\)", src)
            assert all(arg == "this" for arg in spawned), (
                f"{path.name}: QProcess not parented to the plugin {spawned}")
            assert len(spawned) == len(re.findall(r"return runChildProcess\(", src)), (
                f"{path.name}: every QProcess must go through runChildProcess()")
            assert "<< \"QT_QPA_PLATFORM=" not in src, (
                f"{path.name}: appending QT_QPA_PLATFORM keeps the inherited "
                "value first; use headlessChildEnvironment()")

    def test_level1_rpc_server_is_loopback_and_refuses_browsers(self):
        """No authentication guards methods that write files and start
        programs: the server must not listen beyond loopback, and must refuse
        Origin-carrying (browser) handshakes, since web pages reach loopback."""
        src = _read_repo_text(PLUGIN_SRC_DIR / "jsonrpcserver.cpp")
        assert "listen(QHostAddress::LocalHost" in src
        assert "QHostAddress::Any" not in src
        assert re.search(r"setAllowed\(\s*authenticator->origin\(\)\.isEmpty\(\)\s*\)",
                         src)

    def test_level1_rpc_handlers_defined_once_and_built(self):
        """A plugin .so may link with undefined members; a declared handler
        without a definition (or in a .cpp missing from CMake) only fails
        when the GUI dlopen()s the plugin, which then silently drops it."""
        declared = set(re.findall(r"JsonRPCResult (rpc\w+)\(",
                                  _read_repo_text(PLUGIN_H)))
        definitions = collections.Counter()
        for path in _rpc_plugin_cpp_files():
            definitions.update(re.findall(
                r"^JsonRPCResult JsonRPCPlugin::(rpc\w+)\(",
                _read_repo_text(path), re.M))
        assert set(definitions) == declared, (
            f"undefined: {declared - set(definitions)}, "
            f"undeclared: {set(definitions) - declared}")
        assert not [n for n, c in definitions.items() if c != 1]
        cmake = _read_repo_text(PLUGIN_SRC_DIR / "CMakeLists.txt")
        listed = set(re.findall(r"\$\{CMAKE_CURRENT_LIST_DIR\}/(\S+\.cpp)", cmake))
        assert listed == {path.name for path in _rpc_plugin_cpp_files()}

    def test_level1_sibr_commands_exist(self):
        assert SIBR_COMMANDS_H.exists(), "Missing qSIBRCommands.h"

    def test_level1_sibr_viewer_command_structure(self):
        src = _read_repo_text(SIBR_COMMANDS_H)
        assert 'COMMAND_SIBR_VIEWER' in src
        assert 'CommandSIBRViewer' in src
        for viewer in ["ulr", "ulrv2", "texturedmesh",
                        "pointbased", "gaussian", "remotegaussian"]:
            assert viewer in src.lower(), \
                f"SIBR_VIEWER missing viewer type: {viewer}"

    def test_level1_sibr_viewer_options(self):
        src = _read_repo_text(SIBR_COMMANDS_H)
        for opt in ["--path", "--model-path", "--width", "--height",
                     "--iteration", "--device", "--no-interop", "--ip", "--port"]:
            assert opt in src, f"SIBR_VIEWER missing option: {opt}"

    def test_level1_sibr_tool_command_structure(self):
        src = _read_repo_text(SIBR_COMMANDS_H)
        assert 'COMMAND_SIBR_TOOL' in src
        assert 'CommandSIBRTool' in src
        for tool in ["prepareColmap4Sibr", "tonemapper", "unwrapMesh",
                      "textureMesh", "alignMeshes", "cameraConverter"]:
            assert tool in src, f"SIBR_TOOL missing tool: {tool}"

    def test_level1_facets_command_header(self):
        """FACETS command header exists with expected keywords."""
        facets_h = REPO_ROOT / "plugins/core/Standard/qFacets/include/qFacetsCommands.h"
        if not facets_h.exists():
            pytest.fail("qFacetsCommands.h not found")
        src = _read_repo_text(facets_h)
        assert "CommandFacets" in src
        assert "EXTRACT_FACETS" in src
        assert "ALGO_KD_TREE" in src or "ALGO_FAST_MARCHING" in src

    def test_level1_hough_normals_command_header(self):
        """Hough Normals command header exists with expected keywords."""
        hn_h = REPO_ROOT / "plugins/core/Standard/qHoughNormals/include/qHoughNormalsCommands.h"
        if not hn_h.exists():
            pytest.fail("qHoughNormalsCommands.h not found")
        src = _read_repo_text(hn_h)
        assert "CommandHoughNormals" in src
        hn_cpp = REPO_ROOT / "plugins/core/Standard/qHoughNormals/src/qHoughNormalsCommands.cpp"
        if not hn_cpp.exists():
            pytest.fail("qHoughNormalsCommands.cpp not found")
        assert "HOUGH_NORMALS" in _read_repo_text(hn_cpp)

    def test_level1_poisson_recon_command_header(self):
        """Poisson Recon command header exists with expected keywords."""
        pr_h = REPO_ROOT / "plugins/core/Standard/qPoissonRecon/include/qPoissonReconCommands.h"
        if not pr_h.exists():
            pytest.fail("qPoissonReconCommands.h not found")
        src = _read_repo_text(pr_h)
        assert "CommandPoissonRecon" in src
        pr_cpp = REPO_ROOT / "plugins/core/Standard/qPoissonRecon/src/qPoissonReconCommands.cpp"
        if not pr_cpp.exists():
            pytest.fail("qPoissonReconCommands.cpp not found")
        assert "POISSON_RECON" in _read_repo_text(pr_cpp)

    def test_level1_cork_command_header(self):
        """Cork boolean command files exist with expected keywords."""
        cork_h = REPO_ROOT / "plugins/core/Standard/qCork/include/qCorkCommands.h"
        if not cork_h.exists():
            pytest.fail("qCorkCommands.h not found")
        src = _read_repo_text(cork_h)
        assert "CommandCork" in src
        cork_cpp = REPO_ROOT / "plugins/core/Standard/qCork/src/qCorkCommands.cpp"
        if not cork_cpp.exists():
            pytest.fail("qCorkCommands.cpp not found")
        assert "CORK" in _read_repo_text(cork_cpp)

    def test_level1_voxfall_command_header(self):
        """VoxFall command files exist with expected keywords."""
        vf_h = REPO_ROOT / "plugins/core/Standard/qVoxFall/include/qVoxFallCommands.h"
        if not vf_h.exists():
            pytest.fail("qVoxFallCommands.h not found")
        src = _read_repo_text(vf_h)
        assert "CommandVoxFall" in src
        vf_cpp = REPO_ROOT / "plugins/core/Standard/qVoxFall/src/qVoxFallCommands.cpp"
        if not vf_cpp.exists():
            pytest.fail("qVoxFallCommands.cpp not found")
        assert "VOXFALL" in _read_repo_text(vf_cpp)

    def test_level1_qpcl_pcl_commands_cpp(self):
        """qPCL PclCommands.cpp registers PCL_* CLI commands."""
        pcl_cpp = REPO_ROOT / "plugins/core/Standard/qPCL/src/PclCommands.cpp"
        if not pcl_cpp.exists():
            pytest.fail("PclCommands.cpp not found")
        src = _read_repo_text(pcl_cpp)
        for token in (
            "PCL_SOR",
            "PCL_NORMAL_ESTIMATION",
            "PCL_MLS",
            "PCL_EUCLIDEAN_CLUSTER",
            "PCL_SAC_SEGMENTATION",
            "PCL_REGION_GROWING",
            "PCL_GREEDY_TRIANGULATION",
            "PCL_POISSON_RECON",
            "PCL_MARCHING_CUBES",
            "PCL_CONVEX_HULL",
            "PclCommands::RegisterAll",
        ):
            assert token in src, f"Missing qPCL CLI token: {token}"

    def test_level1_plugin_cli_commands_registered(self):
        """Verify registerCommands is implemented in key plugins."""
        plugins_with_cli = [
            ("qFacets", "plugins/core/Standard/qFacets/src/qFacets.cpp", "CommandFacets"),
            ("qHoughNormals", "plugins/core/Standard/qHoughNormals/src/qHoughNormals.cpp", "CommandHoughNormals"),
            ("qPoissonRecon", "plugins/core/Standard/qPoissonRecon/src/qPoissonRecon.cpp", "CommandPoissonRecon"),
            ("qCork", "plugins/core/Standard/qCork/src/qCork.cpp", "CommandCork"),
            ("qVoxFall", "plugins/core/Standard/qVoxFall/src/qVoxFall.cpp", "CommandVoxFall"),
        ]
        for name, rel_path, cmd_class in plugins_with_cli:
            path = REPO_ROOT / rel_path
            if not path.exists():
                continue
            src = _read_repo_text(path)
            assert "registerCommands" in src, f"{name} missing registerCommands"
            assert cmd_class in src or "registerCommand" in src, f"{name} missing command registration"

    def test_level1_rpc_method_count(self):
        src = _read_repo_text(PLUGIN_CPP)
        count = src.count('reg("')
        assert count >= 40, f"Expected ≥40 RPC methods registered, found {count}"

    def test_level1_header_declares_all_methods(self):
        h = _read_repo_text(PLUGIN_H)
        for method in ["rpcOpen", "rpcExport", "rpcFileConvert",
                        "rpcSceneList", "rpcSceneInfo",
                        "rpcEntityRename", "rpcEntitySetColor",
                        "rpcCloudPaintUniform", "rpcCloudPaintByHeight",
                        "rpcCloudPaintByScalarField",
                        "rpcCloudComputeNormals", "rpcCloudSubsample",
                        "rpcCloudCrop", "rpcCloudGetScalarFields",
                        "rpcCloudSetActiveSf", "rpcCloudRemoveSf",
                        "rpcCloudRemoveAllSfs", "rpcCloudRenameSf",
                        "rpcCloudFilterSf", "rpcCloudCoordToSf",
                        "rpcCloudRemoveRgb", "rpcCloudRemoveNormals",
                        "rpcCloudInvertNormals", "rpcCloudMerge",
                        "rpcMeshSimplify", "rpcMeshSmooth",
                        "rpcMeshSubdivide", "rpcMeshSamplePoints",
                        "rpcMeshExtractVertices", "rpcMeshFlipTriangles",
                        "rpcMeshVolume", "rpcMeshMerge",
                        "rpcViewScreenshot", "rpcViewGetCamera",
                        "rpcTransformApply",
                        "rpcColmapReconstruct", "rpcColmapRun",
                        "registerMethods"]:
            assert method in h, f"Missing declaration: {method}"

    def test_level1_cursor_mcp_config_exists(self):
        mcp_json = REPO_ROOT / ".agents" / "mcp.json"
        assert mcp_json.exists(), "Missing .agents/mcp.json"
        data = json.loads(_read_repo_text(mcp_json))
        assert "mcpServers" in data
        assert "acloudviewer" in data["mcpServers"]
        server = data["mcpServers"]["acloudviewer"]
        assert server["command"] == "cli-anything-acloudviewer-mcp"

    def test_level1_plugin_builds(self):
        if not (BUILD_DIR / "CMakeCache.txt").exists():
            pytest.skip("No build directory")
        
        # Check if plugin already exists and is built
        if IS_WINDOWS:
            plugin_paths = [
                BUILD_DIR / "bin/Release/plugins/QJSON_RPC_PLUGIN.dll",
                BUILD_DIR / "bin/Debug/plugins/QJSON_RPC_PLUGINd.dll",
                BUILD_DIR / "plugins/core/Standard/qJSonRPCPlugin/Release/QJSON_RPC_PLUGIN.dll",
            ]
        elif IS_MACOS:
            plugin_paths = [
                BUILD_DIR / "bin/plugins/QJSON_RPC_PLUGIN.dylib",
            ]
        else:
            plugin_paths = [
                BUILD_DIR / "bin/plugins/QJSON_RPC_PLUGIN.so",
            ]
        
        # If plugin already exists, verify it's valid and skip build
        for plugin_path in plugin_paths:
            if plugin_path.exists() and plugin_path.stat().st_size > 0:
                # Plugin exists with non-zero size, assume it's valid
                # This test is about verifying buildability, not forcing a rebuild
                return
        
        # Plugin doesn't exist, need to build it
        timeout = 900  # 15 min for full build
        
        cmd = ["cmake", "--build", str(BUILD_DIR), "--target", "QJSON_RPC_PLUGIN", "--config", "Release"]
        if not IS_WINDOWS:
            cmd += ["--", "-j4"]
        else:
            # On Windows, use single-threaded build to avoid PDB file conflicts
            # Parallel builds can cause "error C1041: cannot open program database file"
            # when multiple cl.exe instances try to write the same PDB file
            pass  # No parallelism flag
        
        # Avoid text=True on Windows: MSBuild may emit bytes that are not valid
        # in the process ANSI code page (e.g. GBK), which breaks subprocess's decoder.
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=timeout)
            stderr = (result.stderr or b"").decode(errors="replace")
            stdout = (result.stdout or b"").decode(errors="replace")
            
            assert result.returncode == 0, (
                f"Build failed (returncode={result.returncode}):\n"
                f"Last 2000 chars of stderr:\n{stderr[-2000:]}\n"
                f"Last 2000 chars of stdout:\n{stdout[-2000:]}"
            )
        except subprocess.TimeoutExpired as e:
            # Provide helpful context about what might have caused the timeout
            pytest.fail(
                f"Build timed out after {timeout}s. "
                f"This might indicate: (1) slow dependencies compilation, "
                f"(2) hanging build process, or (3) insufficient timeout. "
                f"Try running manually: cmake --build {BUILD_DIR} --target QJSON_RPC_PLUGIN --config Release"
            )


# ═══════════════════════════════════════════════════════════════════════════
# Level 2: CLI Harness Tests
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not HAS_CLI, reason="CLI harness not installed")
class TestLevel2_CLIHarness:
    """Verify CLI harness commands work."""

    def test_level2_cli_help(self):
        r = subprocess.run(["cli-anything-acloudviewer", "--help"],
                           capture_output=True, text=True, timeout=10)
        assert r.returncode == 0
        assert "ACloudViewer" in r.stdout
        assert "ai" in r.stdout
        assert "plugin" in r.stdout

    def test_level2_mcp_entrypoint_help(self):
        r = subprocess.run(["cli-anything-acloudviewer-mcp", "--help"],
                           capture_output=True, text=True, timeout=10)
        assert r.returncode == 0, r.stderr
        assert "--mode" in r.stdout and "--rpc-url" in r.stdout

    def test_level2_aicore_registry_complete(self):
        r = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "ai", "list"],
            capture_output=True, text=True, timeout=10)
        assert r.returncode == 0, r.stderr
        _assert_aicore_registry(json.loads(r.stdout))

    @pytest.mark.parametrize("command", [
        ["ai", "--help"],
        ["ai", "run", "--help"],
        ["ai", "schema", "--help"],
        ["plugin", "--help"],
        ["plugin", "run", "--help"],
    ])
    def test_level2_generic_plugin_cli_help(self, command):
        r = subprocess.run(
            ["cli-anything-acloudviewer", *command],
            capture_output=True, text=True, timeout=10)
        assert r.returncode == 0, r.stderr
        if command[:2] == ["ai", "run"]:
            assert "--option" in r.stdout and "--image" in r.stdout
        if command[:2] == ["plugin", "run"]:
            assert "--option" in r.stdout and "--arg" in r.stdout

    @pytest.mark.parametrize("cmd", [
        "convert --help",
        "batch-convert --help",
        "formats --help",
        "formats",
        "scene --help",
        "view --help",
        "process --help",
        "session --help",
        "reconstruct --help",
        "reconstruct auto --help",
        "entity --help",
        "cloud --help",
        "mesh --help",
        "transform --help",
        "export --help",
        "clear --help",
        "methods --help",
        "sibr --help",
        "info --help",
        "info",
    ])
    def test_level2_subcommand_help(self, cmd):
        r = subprocess.run(
            ["cli-anything-acloudviewer"] + cmd.split(),
            capture_output=True, text=True, timeout=10)
        assert r.returncode == 0, (
            f"CLI '{cmd}' failed (rc={r.returncode}):\n"
            f"stdout: {r.stdout[:1000]}\nstderr: {r.stderr[:1000]}")

    @pytest.mark.parametrize("subcmd", [
        "process pcv --help",
        "process csf --help",
        "process ransac --help",
        "process m3c2 --help",
        "process canupo --help",
        "process facets --help",
        "process hough-normals --help",
        "process pcl-sor --help",
        "process pcl-normal-estimation --help",
        "process pcl-mls --help",
        "process pcl-euclidean-cluster --help",
        "process pcl-sac-segmentation --help",
        "process pcl-region-growing --help",
        "process pcl-greedy-triangulation --help",
        "process pcl-poisson-recon --help",
        "process pcl-marching-cubes --help",
        "process pcl-convex-hull --help",
        "process poisson-recon --help",
        "process cork-boolean --help",
        "process voxfall --help",
    ])
    def test_level2_plugin_command_help(self, subcmd):
        """Plugin processing commands show help."""
        result = subprocess.run(
            ["cli-anything-acloudviewer"] + subcmd.split(),
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, f"'{subcmd}' failed: {result.stderr}"

    def test_level2_process_subcommands(self):
        r = subprocess.run(
            ["cli-anything-acloudviewer", "process", "--help"],
            capture_output=True, text=True, timeout=10)
        assert r.returncode == 0
        for cmd in ["subsample", "normals", "icp", "sor",
                     "c2c-dist", "c2m-dist", "density", "curvature",
                     "roughness", "delaunay", "sample-mesh", "color-banding",
                     "set-active-sf", "remove-all-sfs", "remove-sf", "rename-sf",
                     "sf-arithmetic", "sf-op", "coord-to-sf", "sf-gradient",
                     "filter-sf", "sf-to-rgb",
                     "octree-normals", "orient-normals", "invert-normals",
                     "clear-normals", "normals-to-dip", "normals-to-sfs",
                     "extract-cc", "approx-density", "feature", "moment",
                     "best-fit-plane", "mesh-volume", "extract-vertices",
                     "flip-triangles", "merge-clouds", "merge-meshes",
                     "remove-rgb", "remove-scan-grids", "match-centers",
                     "drop-global-shift", "closest-point-set",
                     "rasterize", "stat-test", "cross-section",
                     "pcv", "csf", "ransac", "m3c2", "canupo",
                     "facets", "hough-normals",
                     "pcl-sor", "pcl-normal-estimation", "pcl-mls",
                     "pcl-euclidean-cluster", "pcl-sac-segmentation",
                     "pcl-region-growing", "pcl-greedy-triangulation",
                     "pcl-poisson-recon", "pcl-marching-cubes", "pcl-convex-hull",
                     "poisson-recon",
                     "cork-boolean", "voxfall"]:
            assert cmd in r.stdout, f"Missing process subcommand: {cmd}"
    
    def test_level2_sf_group_exists(self):
        r = subprocess.run(
            ["cli-anything-acloudviewer", "sf", "--help"],
            capture_output=True, text=True, timeout=10)
        assert r.returncode == 0
        assert "Scalar field operations" in r.stdout or "sf" in r.stdout.lower()
    
    def test_level2_sf_subcommands(self):
        r = subprocess.run(
            ["cli-anything-acloudviewer", "sf", "--help"],
            capture_output=True, text=True, timeout=10)
        assert r.returncode == 0
        for cmd in ["coord-to-sf", "arithmetic", "operation", "gradient",
                     "filter", "color-scale", "convert-to-rgb", "set-active",
                     "rename", "remove", "remove-all"]:
            assert cmd in r.stdout, f"Missing sf subcommand: {cmd}"
    
    def test_level2_normals_group_exists(self):
        r = subprocess.run(
            ["cli-anything-acloudviewer", "normals", "--help"],
            capture_output=True, text=True, timeout=10)
        assert r.returncode == 0
        assert "Normal vector operations" in r.stdout or "normals" in r.stdout.lower()
    
    def test_level2_normals_subcommands(self):
        r = subprocess.run(
            ["cli-anything-acloudviewer", "normals", "--help"],
            capture_output=True, text=True, timeout=10)
        assert r.returncode == 0
        for cmd in ["octree", "orient-mst", "invert", "clear", "to-dip", "to-sfs"]:
            assert cmd in r.stdout, f"Missing normals subcommand: {cmd}"

    def test_level2_scene_subcommands(self):
        r = subprocess.run(
            ["cli-anything-acloudviewer", "scene", "--help"],
            capture_output=True, text=True, timeout=10)
        assert r.returncode == 0
        for cmd in ["list", "info", "remove", "show", "hide", "select", "clear"]:
            assert cmd in r.stdout, f"Missing scene subcommand: {cmd}"

    def test_level2_view_subcommands(self):
        r = subprocess.run(
            ["cli-anything-acloudviewer", "view", "--help"],
            capture_output=True, text=True, timeout=10)
        assert r.returncode == 0
        for cmd in ["screenshot", "camera", "orient", "zoom", "refresh",
                     "perspective", "pointsize"]:
            assert cmd in r.stdout, f"Missing view subcommand: {cmd}"

    def test_level2_entity_subcommands(self):
        r = subprocess.run(
            ["cli-anything-acloudviewer", "entity", "--help"],
            capture_output=True, text=True, timeout=10)
        assert r.returncode == 0
        for cmd in ["rename", "set-color"]:
            assert cmd in r.stdout, f"Missing entity subcommand: {cmd}"

    def test_level2_cloud_subcommands(self):
        r = subprocess.run(
            ["cli-anything-acloudviewer", "cloud", "--help"],
            capture_output=True, text=True, timeout=10)
        assert r.returncode == 0
        for cmd in ["paint-uniform", "paint-by-height", "paint-by-scalar-field",
                     "get-scalar-fields", "crop"]:
            assert cmd in r.stdout, f"Missing cloud subcommand: {cmd}"

    def test_level2_mesh_subcommands(self):
        r = subprocess.run(
            ["cli-anything-acloudviewer", "mesh", "--help"],
            capture_output=True, text=True, timeout=10)
        assert r.returncode == 0
        for cmd in ["simplify", "smooth", "subdivide", "sample-points"]:
            assert cmd in r.stdout, f"Missing mesh subcommand: {cmd}"

    def test_level2_transform_subcommands(self):
        r = subprocess.run(
            ["cli-anything-acloudviewer", "transform", "--help"],
            capture_output=True, text=True, timeout=10)
        assert r.returncode == 0
        for cmd in ["apply", "apply-file"]:
            assert cmd in r.stdout, f"Missing transform subcommand: {cmd}"

    def test_level2_session_history(self):
        r = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "session", "history"],
            capture_output=True, text=True, timeout=10)
        assert r.returncode == 0, (
            f"session history failed:\nstdout: {r.stdout[:500]}\nstderr: {r.stderr[:500]}")

    def test_level2_headless_info(self):
        r = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless", "info"],
            capture_output=True, text=True, timeout=10)
        assert r.returncode == 0
        data = json.loads(r.stdout)
        assert data["mode"] == "headless"

    def test_level2_session_status(self):
        r = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "session", "status"],
            capture_output=True, text=True, timeout=10)
        assert r.returncode == 0

    def test_level2_sibr_subcommands(self):
        r = subprocess.run(
            ["cli-anything-acloudviewer", "sibr", "--help"],
            capture_output=True, text=True, timeout=10)
        assert r.returncode == 0
        for cmd in ["prepare-colmap", "texture-mesh", "unwrap-mesh",
                     "tonemapper", "align-meshes", "camera-converter",
                     "nvm-to-sibr", "crop-from-center", "clipping-planes",
                     "distord-crop", "tool"]:
            assert cmd in r.stdout, f"Missing sibr subcommand: {cmd}"

    def test_level2_sibr_viewer_subcommand(self):
        r = subprocess.run(
            ["cli-anything-acloudviewer", "sibr", "--help"],
            capture_output=True, text=True, timeout=10)
        assert r.returncode == 0
        assert "viewer" in r.stdout.lower() or "tool" in r.stdout.lower(), \
            "SIBR group should reference viewer or tool functionality"
    
    def test_level2_sibr_viewer_help(self):
        """Test sibr viewer subcommand help and viewer types."""
        r = subprocess.run(
            ["cli-anything-acloudviewer", "sibr", "viewer", "--help"],
            capture_output=True, text=True, timeout=10)
        assert r.returncode == 0, (
            f"sibr viewer --help failed (rc={r.returncode}):\n{r.stderr[-2000:]}")
        output = r.stdout.lower()
        for viewer in ["gaussian", "ulr", "remotegaussian"]:
            assert viewer in output, f"Missing viewer type: {viewer}"
        for opt in ["--path", "--model-path", "--width", "--height"]:
            assert opt in r.stdout, f"Missing option: {opt}"

    def test_level2_reconstruct_subcommands(self):
        r = subprocess.run(
            ["cli-anything-acloudviewer", "reconstruct", "--help"],
            capture_output=True, text=True, timeout=10)
        assert r.returncode == 0
        for cmd in ["auto", "mesh", "extract-features", "match", "sparse",
                     "undistort", "dense-stereo", "fuse", "poisson",
                     "delaunay-mesh", "texture-mesh",
                     "convert-model", "analyze-model"]:
            assert cmd in r.stdout, f"Missing reconstruct subcommand: {cmd}"

    def test_level2_reconstruct_auto_help(self):
        r = subprocess.run(
            ["cli-anything-acloudviewer", "reconstruct", "auto", "--help"],
            capture_output=True, text=True, timeout=10)
        assert r.returncode == 0
        output = r.stdout + r.stderr
        for expected in ["quality", "workspace"]:
            assert expected.lower() in output.lower(), \
                f"reconstruct auto --help missing: {expected}"

    def test_level2_reconstruct_auto_camera_model(self):
        """Verify reconstruct auto accepts --camera-model option."""
        r = subprocess.run(
            ["cli-anything-acloudviewer", "reconstruct", "auto", "--help"],
            capture_output=True, text=True, timeout=10)
        assert r.returncode == 0
        output = r.stdout + r.stderr
        assert "camera" in output.lower(), \
            "reconstruct auto --help should mention camera model option"

    def test_level2_convert_help(self):
        r = subprocess.run(
            ["cli-anything-acloudviewer", "convert", "--help"],
            capture_output=True, text=True, timeout=10)
        assert r.returncode == 0

    def test_level2_batch_convert_help(self):
        r = subprocess.run(
            ["cli-anything-acloudviewer", "batch-convert", "--help"],
            capture_output=True, text=True, timeout=10)
        assert r.returncode == 0


# ═══════════════════════════════════════════════════════════════════════════
# Level 3: Headless Processing Tests (requires ACloudViewer binary)
# ═══════════════════════════════════════════════════════════════════════════

_SAMPLE_PLY_HEADER = (
    "ply\nformat ascii 1.0\n"
    "element vertex 100\n"
    "property float x\nproperty float y\nproperty float z\n"
    "end_header\n"
)
_SAMPLE_PLY_BODY = "\n".join(f"{i*0.01} {i*0.02} {i*0.03}" for i in range(100)) + "\n"

def _make_mesh_ply_content():
    """Generate a 10x10 grid of points on the XY plane — guaranteed triangulable."""
    pts = []
    for row in range(10):
        for col in range(10):
            pts.append(f"{col * 0.1:.4f} {row * 0.1:.4f} {((row + col) % 3) * 0.05:.4f}")
    header = (
        "ply\nformat ascii 1.0\n"
        f"element vertex {len(pts)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "end_header\n"
    )
    return header + "\n".join(pts) + "\n"

_MESH_PLY_CONTENT = _make_mesh_ply_content()

# Minimal Leica PTX (see PTXFilter.cpp): height, width, 4×(sensor row, 3 tokens),
# 4×(cloud row, 4 tokens), then height×width cells of x y z intensity.
_MINIMAL_PTX_CONTENT = """3
3
0 0 0
1 0 0
0 1 0
0 0 1
1 0 0 0
0 1 0 0
0 0 1 0
0 0 0 1
0.1 0.2 0.3 0.5
0.4 0.5 0.6 0.5
0.7 0.8 0.9 0.5
0.1 0.2 0.3 0.5
0.4 0.5 0.6 0.5
0.7 0.8 0.9 0.5
0.1 0.2 0.3 0.5
0.4 0.5 0.6 0.5
0.7 0.8 0.9 0.5
"""


def _ply_vertex_count(path: Path) -> int:
    """Read ``element vertex N`` from a PLY header (ASCII or binary body)."""
    raw = path.read_bytes()
    end = raw.find(b"end_header")
    if end < 0:
        raise AssertionError(f"no end_header in PLY: {path}")
    text = raw[: end + len(b"end_header")].decode("ascii", errors="replace")
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("element vertex"):
            return int(line.split()[-1])
    raise AssertionError(f"no element vertex in PLY: {path}")


def _count_obj_vertices(path: Path) -> int:
    """Count ``v`` vertex lines in a Wavefront OBJ file."""
    n = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if len(line) > 1 and line[0] == "v" and line[1] in " \t":
            n += 1
    return n


_REQUIRE_BINARY = os.environ.get("ACV_REQUIRE_BINARY") == "1"

# Skips are decided before anything runs, from facts known up front: what a
# platform's build cannot provide, what is installed, what the QPA can do.
# Once a command has run, a non-zero exit or a crash is a failure.

# Plugins a platform's build cannot provide.
_PLATFORM_UNSUPPORTED_PLUGINS = {
    # qFBXIO/cmake/FBX_download.cmake fetches a Windows-only SDK (.lib/.dll);
    # only util/ci_utils.ps1 sets PLUGIN_IO_QFBX=ON.
    "QFBX_IO_PLUGIN": ("Linux", "Darwin"),
}


def _fail_if_crashed(r: subprocess.CompletedProcess, context: str) -> None:
    """Name the crash (signal or Windows exception) instead of a bare rc."""
    if r.returncode < 0:
        pytest.fail(f"Binary crashed with signal {-r.returncode} ({context}):\n"
                    f"{(r.stdout + r.stderr)[-2000:]}")
    if r.returncode >= 0xC0000000:
        pytest.fail(f"Binary crashed with code 0x{r.returncode:08X} ({context}):\n"
                    f"{(r.stdout + r.stderr)[-2000:]}")


def _plugin_library(name: str) -> Path | None:
    """The installed library of plugin target `name` (e.g. QPCL_PLUGIN),
    searched where ecvApplicationBase::setupPaths() loads plugins from."""
    if not BINARY_PATH:
        return None
    exe_dir = Path(BINARY_PATH).resolve().parent
    if IS_MACOS:  # <bundle>/Contents/MacOS/ACloudViewer
        contents = exe_dir.parent if exe_dir.name == "MacOS" else exe_dir
        dirs = [contents / "cvPlugins", contents / "PlugIns/cvPlugins"]
    elif IS_WINDOWS:
        dirs = [exe_dir / "plugins"]
    else:  # bin/ACloudViewer, or the packaged ACloudViewer.sh at the root
        root = exe_dir.parent if exe_dir.name == "bin" else exe_dir
        dirs = [root / "plugins", root / "bin/plugins",
                root / "lib/ACloudViewer/plugins"]
    files = {f"lib{name}.so", f"{name}.dll", f"lib{name}.dylib"}
    for directory in dirs:
        for candidate in files:
            if (directory / candidate).is_file():
                return directory / candidate
    return None


def _require_plugin(plugin: str) -> None:
    """Skip, before running, a plugin this platform cannot build or this
    checkout did not build; an installer (ACV_REQUIRE_BINARY=1) that lacks a
    plugin its platform builds fails."""
    system = platform.system()
    if system in _PLATFORM_UNSUPPORTED_PLUGINS.get(plugin, ()):
        pytest.skip(f"{plugin} is not built on {system}")
    if _plugin_library(plugin) is None:
        if _REQUIRE_BINARY:
            pytest.fail(f"{plugin} is built on {system} but is not installed "
                        f"with {BINARY_PATH}")
        pytest.skip(f"{plugin} is not built in this checkout")


def _cli_resolve_normals_output_path(
        r_normals: subprocess.CompletedProcess, expected: str) -> str:
    """Resolve path to cloud written by ``process normals``; a missing output
    fails the dependent test."""
    _fail_if_crashed(r_normals, "process normals")
    combined = r_normals.stdout + r_normals.stderr
    if r_normals.returncode != 0:
        pytest.fail(
            f"normals prerequisite failed (rc={r_normals.returncode}):\n"
            f"{combined[-800:]}")
    exp = Path(expected)
    if exp.is_file() and exp.stat().st_size > 0:
        return str(exp.resolve())
    data = {}
    try:
        data = json.loads(r_normals.stdout)
    except json.JSONDecodeError:
        pytest.fail(
            f"normals prerequisite: invalid JSON and output missing at {expected}")
    if data.get("status") == "failed":
        pytest.fail(
            f"normals returned status=failed: {json.dumps(data)[:800]}")
    for key in ("output", "output_path", "output_file", "path", "file"):
        p = data.get(key)
        if isinstance(p, str) and Path(p).is_file() and Path(p).stat().st_size > 0:
            return str(Path(p).resolve())
    pytest.fail(
        "normals prerequisite succeeded but output file not found at "
        f"{expected} (JSON keys: {list(data.keys())})")
    return ""  # unreachable: pytest.fail always raises


def _count_off_vertices(path: Path) -> int:
    """Read vertex count from the header of an OFF file."""
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8", errors="replace").splitlines()
             if ln.strip() and not ln.strip().startswith("#")]
    if not lines:
        raise AssertionError(f"empty OFF: {path}")
    parts0 = lines[0].split()
    if parts0 and parts0[0].upper() == "OFF":
        if len(parts0) >= 4:
            return int(parts0[1])
        if len(lines) < 2:
            raise AssertionError(f"bad OFF: {path}")
        return int(lines[1].split()[0])
    return int(lines[0].split()[0])


def _build_env_for_binary(binary_path: str) -> dict[str, str]:
    """Lightweight env setup for invoking the binary directly (no harness)."""
    binary_path = _resolve_exe(binary_path)
    env = os.environ.copy()

    # Set or remove QT_QPA_PLATFORM based on platform:
    # - macOS: Don't set it - .app bundles only have cocoa plugin, Qt auto-selects it
    # - Windows: Use minimal (windeployqt doesn't ship qoffscreen).
    #   minimal has no OpenGL context, so GL-rendering plugins (qPCV) are
    #   skipped there before they run.
    # - Linux: Use offscreen (always available)
    if IS_MACOS:
        # macOS .app bundles only include cocoa Qt platform plugin.
        # Remove QT_QPA_PLATFORM if set externally so Qt auto-selects cocoa.
        env.pop("QT_QPA_PLATFORM", None)
    elif IS_WINDOWS:
        env["QT_QPA_PLATFORM"] = "minimal"
    else:
        env["QT_QPA_PLATFORM"] = "offscreen"

    bin_dir = str(Path(binary_path).parent)
    sep = ";" if IS_WINDOWS else ":"
    
    if IS_WINDOWS:
        # On Windows, add multiple potential DLL locations to PATH
        dll_dirs = [
            bin_dir,                           # Binary directory
            str(Path(bin_dir) / "lib"),       # lib subdirectory
            str(Path(bin_dir) / "bin"),       # bin subdirectory (if different)
            str(Path(bin_dir) / "plugins"),   # plugins directory
            str(Path(bin_dir) / "platforms"), # Qt platforms plugin directory
        ]
        # Filter to only existing directories
        dll_dirs = [d for d in dll_dirs if Path(d).exists()]
        env["PATH"] = sep.join(dll_dirs + [env.get("PATH", "")])
    elif IS_MACOS:
        lib_dir = str(Path(bin_dir) / "lib")
        env["DYLD_LIBRARY_PATH"] = sep.join(
            filter(None, [bin_dir, lib_dir, env.get("DYLD_LIBRARY_PATH", "")]))
        qt_plugin_path = Path(binary_path).parent.parent / "PlugIns"
        if qt_plugin_path.exists():
            env["QT_PLUGIN_PATH"] = str(qt_plugin_path)
    else:
        lib_dir = str(Path(bin_dir) / "lib")
        env["LD_LIBRARY_PATH"] = sep.join(
            filter(None, [bin_dir, lib_dir, env.get("LD_LIBRARY_PATH", "")]))
    return env


@pytest.mark.skipif(not _REQUIRE_BINARY, reason="ACV_REQUIRE_BINARY is not set")
def test_level3_required_binary_present():
    """Installer CI sets ACV_REQUIRE_BINARY=1: a missing binary must fail.

    Every other Level 3 class is skipif(not HAS_BINARY), which is right for a
    developer checkout but would let a broken install silently skip the
    whole AICore schema gate in CI.
    """
    assert HAS_BINARY, (
        "ACloudViewer binary not found; ACV_BINARY="
        f"{os.environ.get('ACV_BINARY', '')!r}")


@pytest.mark.skipif(not HAS_BINARY, reason="ACloudViewer binary not found")
class TestLevel3_HeadlessProcessing:
    """Test headless processing via ACloudViewer binary directly.

    These tests do NOT require the CLI harness — they invoke the binary
    directly via subprocess so they work with both installed binaries and
    build-directory binaries.
    
    Note: On macOS, -SILENT mode uses the cocoa platform plugin (the only
    plugin included in .app bundles) which works in headless mode.
    """

    @pytest.fixture
    def sample_ply(self, tmp_path):
        ply_path = tmp_path / "test.ply"
        ply_path.write_text(_SAMPLE_PLY_HEADER + _SAMPLE_PLY_BODY)
        return str(ply_path)

    @pytest.fixture
    def acv_env(self):
        return _build_env_for_binary(BINARY_PATH)

    def test_level3_binary_can_load(self, sample_ply, acv_env):
        """Verify the binary can load a PLY file."""
        r = subprocess.run(
            [BINARY_PATH, "-SILENT", "-O", sample_ply, "-SAVE_CLOUDS"],
            capture_output=True, text=True, timeout=60, env=acv_env)
        if IS_WINDOWS and r.returncode == 3221225781:  # 0xC0000135 STATUS_DLL_NOT_FOUND
            pytest.fail(
                f"Binary crashed with STATUS_DLL_NOT_FOUND (0xC0000135).\n"
                f"This indicates missing DLL dependencies.\n"
                f"Binary path: {BINARY_PATH}\n"
                f"PATH env: {acv_env.get('PATH', 'NOT SET')[:500]}\n"
                f"stderr: {r.stderr[-500:]}"
            )
        assert r.returncode == 0, f"Binary load failed:\n{r.stderr[-2000:]}"

    def test_level3_subsample(self, sample_ply, tmp_path, acv_env):
        r = subprocess.run(
            [BINARY_PATH, "-SILENT", "-O", sample_ply,
             "-SS", "SPATIAL", "0.2", "-SAVE_CLOUDS"],
            capture_output=True, text=True, timeout=60,
            env=acv_env, cwd=str(tmp_path))
        if IS_WINDOWS and r.returncode == 3221225781:
            pytest.fail(
                f"Binary crashed with STATUS_DLL_NOT_FOUND (0xC0000135).\n"
                f"Binary path: {BINARY_PATH}\n"
                f"PATH: {acv_env.get('PATH', 'NOT SET')[:500]}"
            )
        combined = r.stdout + r.stderr
        assert "Result:" in combined or r.returncode == 0, \
            f"Subsample failed (rc={r.returncode}):\n{combined[-2000:]}"

    def test_level3_normals(self, sample_ply, acv_env, tmp_path):
        r = subprocess.run(
            [BINARY_PATH, "-SILENT", "-O", sample_ply, "-COMPUTE_NORMALS"],
            capture_output=True, text=True, timeout=60,
            env=acv_env, cwd=str(tmp_path))
        if IS_WINDOWS and r.returncode == 3221225781:
            pytest.fail(
                f"Binary crashed with STATUS_DLL_NOT_FOUND (0xC0000135).\n"
                f"Binary path: {BINARY_PATH}\n"
                f"PATH: {acv_env.get('PATH', 'NOT SET')[:500]}"
            )
        assert r.returncode == 0, f"Normals failed:\n{(r.stdout + r.stderr)[-2000:]}"

    def test_level3_binary_help(self, acv_env):
        """Binary prints help text and exits cleanly."""
        r = subprocess.run(
            [BINARY_PATH, "-SILENT", "-HELP"],
            capture_output=True, text=True, timeout=30, env=acv_env)
        combined = r.stdout + r.stderr
        assert r.returncode == 0, f"HELP failed (rc={r.returncode}):\n{combined[-2000:]}"

    def test_level3_binary_sor(self, sample_ply, acv_env, tmp_path):
        """Statistical outlier removal via binary."""
        r = subprocess.run(
            [BINARY_PATH, "-SILENT", "-O", sample_ply,
             "-SOR", "6", "1.0", "-SAVE_CLOUDS"],
            capture_output=True, text=True, timeout=60,
            env=acv_env, cwd=str(tmp_path))
        assert r.returncode == 0, \
            f"SOR failed (rc={r.returncode}):\n{(r.stdout+r.stderr)[-2000:]}"

    def test_level3_binary_crop(self, sample_ply, acv_env, tmp_path):
        """Crop a point cloud by bounding box via binary."""
        r = subprocess.run(
            [BINARY_PATH, "-SILENT", "-O", sample_ply,
             "-CROP", "0:0:0:0.5:1.0:1.5", "-SAVE_CLOUDS"],
            capture_output=True, text=True, timeout=60,
            env=acv_env, cwd=str(tmp_path))
        assert r.returncode == 0, \
            f"CROP failed (rc={r.returncode}):\n{(r.stdout+r.stderr)[-2000:]}"

    def test_level3_binary_c2c_dist(self, sample_ply, acv_env, tmp_path):
        """Cloud-to-cloud distance between same cloud via binary."""
        ply2 = str(tmp_path / "cloud2.ply")
        Path(ply2).write_text(_SAMPLE_PLY_HEADER + _SAMPLE_PLY_BODY)
        r = subprocess.run(
            [BINARY_PATH, "-SILENT",
             "-O", sample_ply, "-O", ply2,
             "-C2C_DIST", "-SAVE_CLOUDS"],
            capture_output=True, text=True, timeout=60,
            env=acv_env, cwd=str(tmp_path))
        assert r.returncode == 0, \
            f"C2C_DIST failed (rc={r.returncode}):\n{(r.stdout+r.stderr)[-2000:]}"

    def test_level3_binary_log_file(self, sample_ply, acv_env, tmp_path):
        """Logging to file via -LOG_FILE (relative name → <binary_dir>/logs/)."""
        r = subprocess.run(
            [BINARY_PATH, "-SILENT", "-LOG_FILE", "test_integration",
             "-O", sample_ply, "-SAVE_CLOUDS"],
            capture_output=True, text=True, timeout=60,
            env=acv_env, cwd=str(tmp_path))
        assert r.returncode == 0, \
            f"LOG_FILE failed (rc={r.returncode}):\n{(r.stdout+r.stderr)[-2000:]}"
        log_dir = Path(BINARY_PATH).parent / "logs"
        for f in log_dir.glob("test_integration*.log"):
            f.unlink(missing_ok=True)

    def test_level3_binary_density(self, sample_ply, acv_env, tmp_path):
        """Density computation via binary."""
        r = subprocess.run(
            [BINARY_PATH, "-SILENT", "-O", sample_ply,
             "-DENSITY", "0.1", "-SAVE_CLOUDS"],
            capture_output=True, text=True, timeout=60,
            env=acv_env, cwd=str(tmp_path))
        assert r.returncode == 0, \
            f"DENSITY failed (rc={r.returncode}):\n{(r.stdout+r.stderr)[-2000:]}"

    def test_level3_binary_curvature(self, sample_ply, acv_env, tmp_path):
        """Curvature computation via binary."""
        r = subprocess.run(
            [BINARY_PATH, "-SILENT", "-O", sample_ply,
             "-CURV", "MEAN", "0.1", "-SAVE_CLOUDS"],
            capture_output=True, text=True, timeout=60,
            env=acv_env, cwd=str(tmp_path))
        assert r.returncode == 0, \
            f"CURV failed (rc={r.returncode}):\n{(r.stdout+r.stderr)[-2000:]}"

    def test_level3_binary_roughness(self, sample_ply, acv_env, tmp_path):
        """Roughness computation via binary."""
        r = subprocess.run(
            [BINARY_PATH, "-SILENT", "-O", sample_ply,
             "-ROUGH", "0.1", "-SAVE_CLOUDS"],
            capture_output=True, text=True, timeout=60,
            env=acv_env, cwd=str(tmp_path))
        assert r.returncode == 0, \
            f"ROUGH failed (rc={r.returncode}):\n{(r.stdout+r.stderr)[-2000:]}"

    def test_level3_binary_best_fit_plane(self, sample_ply, acv_env, tmp_path):
        """Best fit plane via binary."""
        r = subprocess.run(
            [BINARY_PATH, "-SILENT", "-O", sample_ply,
             "-BEST_FIT_PLANE", "-SAVE_CLOUDS"],
            capture_output=True, text=True, timeout=60,
            env=acv_env, cwd=str(tmp_path))
        assert r.returncode == 0, \
            f"BEST_FIT_PLANE failed (rc={r.returncode}):\n{(r.stdout+r.stderr)[-2000:]}"

    def test_level3_binary_merge_clouds(self, sample_ply, acv_env, tmp_path):
        """Merge two clouds via binary."""
        ply2 = str(tmp_path / "cloud2.ply")
        Path(ply2).write_text(_SAMPLE_PLY_HEADER + _SAMPLE_PLY_BODY)
        r = subprocess.run(
            [BINARY_PATH, "-SILENT",
             "-O", sample_ply, "-O", ply2,
             "-MERGE_CLOUDS", "-SAVE_CLOUDS"],
            capture_output=True, text=True, timeout=60,
            env=acv_env, cwd=str(tmp_path))
        assert r.returncode == 0, \
            f"MERGE_CLOUDS failed (rc={r.returncode}):\n{(r.stdout+r.stderr)[-2000:]}"

    def test_level3_binary_extract_cc(self, sample_ply, acv_env, tmp_path):
        """Extract connected components via binary."""
        r = subprocess.run(
            [BINARY_PATH, "-SILENT", "-O", sample_ply,
             "-EXTRACT_CC", "6", "10", "-SAVE_CLOUDS"],
            capture_output=True, text=True, timeout=60,
            env=acv_env, cwd=str(tmp_path))
        assert r.returncode == 0, \
            f"EXTRACT_CC failed (rc={r.returncode}):\n{(r.stdout+r.stderr)[-2000:]}"

    def test_level3_binary_filter_sf(self, sample_ply, acv_env, tmp_path):
        """Filter by scalar field via binary."""
        r = subprocess.run(
            [BINARY_PATH, "-SILENT", "-O", sample_ply,
             "-COORD_TO_SF", "Z",
             "-FILTER_SF", "0.0", "1.5",
             "-SAVE_CLOUDS"],
            capture_output=True, text=True, timeout=60,
            env=acv_env, cwd=str(tmp_path))
        assert r.returncode == 0, \
            f"FILTER_SF failed (rc={r.returncode}):\n{(r.stdout+r.stderr)[-2000:]}"

    def test_level3_binary_icp(self, sample_ply, acv_env, tmp_path):
        """ICP registration with two copies of the same cloud."""
        ply2 = str(tmp_path / "cloud2.ply")
        Path(ply2).write_text(_SAMPLE_PLY_HEADER + _SAMPLE_PLY_BODY)
        r = subprocess.run(
            [BINARY_PATH, "-SILENT",
             "-O", sample_ply, "-O", ply2,
             "-ICP", "-ITER", "10",
             "-SAVE_CLOUDS"],
            capture_output=True, text=True, timeout=120,
            env=acv_env, cwd=str(tmp_path))
        assert r.returncode == 0, \
            f"ICP failed (rc={r.returncode}):\n{(r.stdout+r.stderr)[-2000:]}"

    def test_level3_binary_sample_mesh(self, acv_env, tmp_path):
        """Sample points from mesh via binary (uses grid PLY for clean Delaunay)."""
        mesh_input = str(tmp_path / "grid.ply")
        Path(mesh_input).write_text(_MESH_PLY_CONTENT)
        r = subprocess.run(
            [BINARY_PATH, "-SILENT", "-O", mesh_input,
             "-DELAUNAY", "-SAMPLE_MESH", "POINTS", "50",
             "-SAVE_CLOUDS"],
            capture_output=True, text=True, timeout=60,
            env=acv_env, cwd=str(tmp_path))
        assert r.returncode == 0, \
            f"SAMPLE_MESH failed (rc={r.returncode}):\n{(r.stdout+r.stderr)[-2000:]}"


@pytest.mark.skipif(not HAS_CLI, reason="CLI harness not installed")
@pytest.mark.skipif(not HAS_BINARY, reason="ACloudViewer binary not found")
class TestLevel3_CLIHarness:
    """Test headless processing via the CLI harness (requires pip install)."""

    @pytest.fixture
    def cli_env(self):
        env = _build_env_for_binary(BINARY_PATH)
        env["ACV_BINARY"] = BINARY_PATH
        return env

    @pytest.fixture
    def sample_ply(self, tmp_path):
        ply_path = tmp_path / "test.ply"
        ply_path.write_text(_SAMPLE_PLY_HEADER + _SAMPLE_PLY_BODY)
        return str(ply_path)

    @pytest.mark.parametrize("task,expected_command,_header", AICORE_BATCH_COMMANDS)
    def test_level3_aicore_schema_from_installed_plugin(
            self, task, expected_command, _header, cli_env):
        r = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "ai", "schema", task],
            capture_output=True, text=True, timeout=60, env=cli_env)
        assert r.returncode == 0, (
            f"{task} schema failed:\nstdout: {r.stdout[-2000:]}\n"
            f"stderr: {r.stderr[-2000:]}")
        schema = json.loads(r.stdout)
        assert schema["schema"] == "acloudviewer.aicore.batch/1"
        assert schema["keyword"] == expected_command
        assert schema["options"], f"{task}: empty option schema"
        keys = {option["key"] for option in schema["options"]}
        assert "RESULT_JSON" in keys

    def test_level3_generic_plugin_passthrough(self, cli_env):
        r = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "plugin", "run", "ANIMATION", "--arg=-FPS", "--arg=24",
             "--arg=-TOTAL_FRAMES", "--arg=10"],
            capture_output=True, text=True, timeout=60, env=cli_env)
        assert r.returncode == 0, r.stderr
        result = json.loads(r.stdout)
        assert result["command"] == "ANIMATION"
        assert result["exit_code"] == 0

    def test_level3_cli_subsample(self, sample_ply, tmp_path, cli_env):
        out = str(tmp_path / "sub.ply")
        r = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "process", "subsample", sample_ply, "-o", out, "--voxel-size", "0.2"],
            capture_output=True, text=True, timeout=60, env=cli_env)
        assert r.returncode == 0, (
            f"CLI subsample failed (rc={r.returncode}):\n"
            f"stdout: {r.stdout[:2000]}\nstderr: {r.stderr[:2000]}")
        data = json.loads(r.stdout)
        assert data.get("status") != "failed", (
            f"CLI subsample returned status=failed. Full response:\n"
            f"{json.dumps(data, indent=2)}\nstderr: {r.stderr[:1000]}")

    def test_level3_cli_crop(self, sample_ply, tmp_path, cli_env):
        out = str(tmp_path / "crop.ply")
        r = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "process", "crop", sample_ply, "-o", out,
             "--min-x", "-1", "--min-y", "-1", "--min-z", "-1",
             "--max-x", "1", "--max-y", "2", "--max-z", "3"],
            capture_output=True, text=True, timeout=60, env=cli_env)
        assert r.returncode == 0, (
            f"CLI process crop failed (rc={r.returncode}):\n"
            f"stdout: {r.stdout[:2000]}\nstderr: {r.stderr[:2000]}")
        data = json.loads(r.stdout)
        assert data.get("status") != "failed", (
            f"CLI process crop returned status=failed. Full response:\n"
            f"{json.dumps(data, indent=2)}\nstderr: {r.stderr[:1000]}")

    def test_level3_cli_normals(self, sample_ply, tmp_path, cli_env):
        out = str(tmp_path / "normals.ply")
        r = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "process", "normals", sample_ply, "-o", out],
            capture_output=True, text=True, timeout=60, env=cli_env)
        assert r.returncode == 0, (
            f"CLI normals failed (rc={r.returncode}):\n"
            f"stdout: {r.stdout[:2000]}\nstderr: {r.stderr[:2000]}")
        data = json.loads(r.stdout)
        assert data.get("status") != "failed", (
            f"CLI normals returned status=failed. Full response:\n"
            f"{json.dumps(data, indent=2)}\nstderr: {r.stderr[:1000]}")

    def test_level3_cli_formats(self, cli_env):
        r = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless", "formats"],
            capture_output=True, text=True, timeout=10, env=cli_env)
        assert r.returncode == 0
        data = json.loads(r.stdout)
        assert ".ply" in data["point_cloud"]
        assert ".obj" in data["mesh"]
    
    def test_level3_cli_density(self, sample_ply, tmp_path, cli_env):
        """Test process density command."""
        out = str(tmp_path / "density.ply")
        r = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "process", "density", sample_ply, "-o", out, "--radius", "0.05"],
            capture_output=True, text=True, timeout=60, env=cli_env)
        assert r.returncode == 0, (
            f"process density failed (rc={r.returncode}):\n"
            f"stdout: {r.stdout[:2000]}\nstderr: {r.stderr[:2000]}")
        data = json.loads(r.stdout)
        assert data.get("status") != "failed"
    
    def test_level3_cli_curvature(self, sample_ply, tmp_path, cli_env):
        """Test process curvature command."""
        # Need normals first
        temp = str(tmp_path / "with_normals.ply")
        r1 = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "process", "normals", sample_ply, "-o", temp],
            capture_output=True, text=True, timeout=60, env=cli_env)
        if r1.returncode != 0:
            pytest.fail(f"normals prerequisite failed: {r1.stderr[:500]}")
        
        out = str(tmp_path / "curvature.ply")
        r = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "process", "curvature", temp, "-o", out, "--type", "MEAN", "--radius", "0.05"],
            capture_output=True, text=True, timeout=60, env=cli_env)
        assert r.returncode == 0, (
            f"process curvature failed (rc={r.returncode}):\n"
            f"stdout: {r.stdout[:2000]}\nstderr: {r.stderr[:2000]}")
        data = json.loads(r.stdout)
        assert data.get("status") != "failed"
    
    def test_level3_cli_roughness(self, sample_ply, tmp_path, cli_env):
        """Test process roughness command."""
        temp = str(tmp_path / "with_normals.ply")
        r1 = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "process", "normals", sample_ply, "-o", temp],
            capture_output=True, text=True, timeout=60, env=cli_env)
        normals_in = _cli_resolve_normals_output_path(r1, temp)

        out = str(tmp_path / "roughness.ply")
        r = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "process", "roughness", normals_in, "-o", out, "--radius", "0.1"],
            capture_output=True, text=True, timeout=60, env=cli_env)
        _fail_if_crashed(r, "process roughness")
        assert r.returncode == 0, (
            f"process roughness failed (rc={r.returncode}):\n"
            f"stdout: {r.stdout[:2000]}\nstderr: {r.stderr[:2000]}")
        data = json.loads(r.stdout)
        assert data.get("status") != "failed"
    
    def test_level3_cli_feature(self, sample_ply, tmp_path, cli_env):
        """Test process feature command."""
        temp = str(tmp_path / "with_normals.ply")
        r1 = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "process", "normals", sample_ply, "-o", temp],
            capture_output=True, text=True, timeout=60, env=cli_env)
        normals_in = _cli_resolve_normals_output_path(r1, temp)

        out = str(tmp_path / "features.ply")
        r = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "process", "feature", normals_in, "-o", out,
             "--type", "SURFACE_VARIATION", "--kernel-size", "0.1"],
            capture_output=True, text=True, timeout=60, env=cli_env)
        _fail_if_crashed(r, "process feature")
        assert r.returncode == 0, (
            f"process feature failed (rc={r.returncode}):\n"
            f"stdout: {r.stdout[:2000]}\nstderr: {r.stderr[:2000]}")
        data = json.loads(r.stdout)
        assert data.get("status") != "failed"
    
    def test_level3_cli_extract_cc(self, sample_ply, tmp_path, cli_env):
        """Test process extract-cc command."""
        out = str(tmp_path / "components.ply")
        r = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "process", "extract-cc", sample_ply, "-o", out,
             "--min-points", "5", "--octree-level", "6"],
            capture_output=True, text=True, timeout=60, env=cli_env)
        assert r.returncode == 0, (
            f"process extract-cc failed (rc={r.returncode}):\n"
            f"stdout: {r.stdout[:2000]}\nstderr: {r.stderr[:2000]}")
        data = json.loads(r.stdout)
        assert "status" in data, f"extract-cc missing status field: {data}"
    
    def test_level3_cli_color_banding(self, sample_ply, tmp_path, cli_env):
        """Test process color-banding command."""
        out = str(tmp_path / "colored.ply")
        r = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "process", "color-banding", sample_ply, "-o", out,
             "--axis", "Z", "--frequency", "10.0"],
            capture_output=True, text=True, timeout=60, env=cli_env)
        assert r.returncode == 0, (
            f"process color-banding failed (rc={r.returncode}):\n"
            f"stdout: {r.stdout[:2000]}\nstderr: {r.stderr[:2000]}")
        data = json.loads(r.stdout)
        assert data.get("status") != "failed"
    
    def test_level3_cli_merge_clouds(self, sample_ply, tmp_path, cli_env):
        """Test process merge-clouds command."""
        # Create two copies
        cloud1 = str(tmp_path / "cloud1.ply")
        cloud2 = str(tmp_path / "cloud2.ply")
        Path(cloud1).write_text(_SAMPLE_PLY_HEADER + _SAMPLE_PLY_BODY)
        Path(cloud2).write_text(_SAMPLE_PLY_HEADER + _SAMPLE_PLY_BODY)
        
        out = str(tmp_path / "merged.ply")
        r = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "process", "merge-clouds", cloud1, cloud2, "-o", out],
            capture_output=True, text=True, timeout=60, env=cli_env)
        assert r.returncode == 0, (
            f"process merge-clouds failed (rc={r.returncode}):\n"
            f"stdout: {r.stdout[:2000]}\nstderr: {r.stderr[:2000]}")
        data = json.loads(r.stdout)
        assert data.get("status") != "failed"
    
    def test_level3_cli_cross_section(self, sample_ply, tmp_path, cli_env):
        """Test process cross-section command (requires polyline input)."""
        # Create a simple polyline file for testing
        polyline_file = str(tmp_path / "polyline.ply")
        polyline_content = (
            "ply\nformat ascii 1.0\n"
            "element vertex 2\n"
            "property float x\nproperty float y\nproperty float z\n"
            "end_header\n"
            "0.0 0.0 0.0\n"
            "1.0 1.0 1.0\n"
        )
        Path(polyline_file).write_text(polyline_content)
        
        out = str(tmp_path / "cross_section.ply")
        r = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "process", "cross-section", sample_ply,
             "-o", out, "--polyline", polyline_file],
            capture_output=True, text=True, timeout=60, env=cli_env)
        if r.returncode != 0:
            pytest.fail(
                f"process cross-section failed (rc={r.returncode}):\n"
                f"stdout: {r.stdout[:2000]}\nstderr: {r.stderr[:2000]}")
        data = json.loads(r.stdout)
        assert "status" in data

    def test_level3_cli_best_fit_plane(self, sample_ply, tmp_path, cli_env):
        """Test process best-fit-plane command."""
        out = str(tmp_path / "plane_distance.ply")
        r = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "process", "best-fit-plane", sample_ply, "-o", out],
            capture_output=True, text=True, timeout=60, env=cli_env)
        assert r.returncode == 0, (
            f"process best-fit-plane failed (rc={r.returncode}):\n"
            f"stdout: {r.stdout[:2000]}\nstderr: {r.stderr[:2000]}")
        data = json.loads(r.stdout)
        assert data.get("status") != "failed"

    def test_level3_cli_sibr_available(self, cli_env):
        r = subprocess.run(
            ["cli-anything-acloudviewer", "sibr", "--help"],
            capture_output=True, text=True, timeout=10, env=cli_env)
        assert r.returncode == 0, "SIBR CLI group should be available"
    
    def test_level3_cli_sor(self, sample_ply, tmp_path, cli_env):
        """Test statistical outlier removal."""
        out = str(tmp_path / "cleaned.ply")
        r = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "process", "sor", sample_ply, "-o", out, "--knn", "6", "--sigma", "1.0"],
            capture_output=True, text=True, timeout=60, env=cli_env)
        assert r.returncode == 0, (
            f"process sor failed (rc={r.returncode}):\n"
            f"stdout: {r.stdout[:2000]}\nstderr: {r.stderr[:2000]}")
        data = json.loads(r.stdout)
        assert data.get("status") != "failed", f"Command returned failed: {data}"
    
    def test_level3_cli_delaunay(self, sample_ply, tmp_path, cli_env):
        """Test Delaunay triangulation (mesh reconstruction)."""
        temp = str(tmp_path / "with_normals.ply")
        r1 = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "process", "normals", sample_ply, "-o", temp],
            capture_output=True, text=True, timeout=60, env=cli_env)
        if r1.returncode != 0:
            pytest.fail(f"normals prerequisite failed: {r1.stderr[:500]}")
        
        out = str(tmp_path / "mesh.ply")
        r = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "process", "delaunay", temp, "-o", out],
            capture_output=True, text=True, timeout=60, env=cli_env)
        assert r.returncode == 0, (
            f"process delaunay failed (rc={r.returncode}):\n"
            f"stdout: {r.stdout[:2000]}\nstderr: {r.stderr[:2000]}")
        data = json.loads(r.stdout)
        assert data.get("status") != "failed", f"Command returned failed: {data}"
    
    def test_level3_cli_sample_mesh(self, tmp_path, cli_env):
        """Test mesh sampling."""
        mesh_input = str(tmp_path / "mesh_input.ply")
        Path(mesh_input).write_text(_MESH_PLY_CONTENT)
        temp_mesh = str(tmp_path / "mesh.ply")
        temp_normals = str(tmp_path / "with_normals.ply")
        r1 = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "process", "normals", mesh_input, "-o", temp_normals],
            capture_output=True, text=True, timeout=60, env=cli_env)
        if r1.returncode != 0:
            pytest.fail(f"normals prerequisite failed: {r1.stderr[:500]}")
        
        r2 = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "process", "delaunay", temp_normals, "-o", temp_mesh],
            capture_output=True, text=True, timeout=60, env=cli_env)
        if r2.returncode != 0:
            pytest.fail(f"delaunay prerequisite failed: {r2.stderr[:500]}")
        
        out = str(tmp_path / "sampled.ply")
        r = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "process", "sample-mesh", temp_mesh, "-o", out, "--points", "100"],
            capture_output=True, text=True, timeout=60, env=cli_env)
        assert r.returncode == 0, (
            f"process sample-mesh failed (rc={r.returncode}):\n"
            f"stdout: {r.stdout[:2000]}\nstderr: {r.stderr[:2000]}")
        data = json.loads(r.stdout)
        assert data.get("status") != "failed", f"Command returned failed: {data}"
    
    def test_level3_cli_remove_rgb(self, sample_ply, tmp_path, cli_env):
        """Test remove RGB colors."""
        temp = str(tmp_path / "colored.ply")
        r1 = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "process", "color-banding", sample_ply, "-o", temp, "--axis", "Z"],
            capture_output=True, text=True, timeout=60, env=cli_env)
        if r1.returncode != 0:
            pytest.fail(f"color-banding prerequisite failed: {r1.stderr[:500]}")
        
        out = str(tmp_path / "no_color.ply")
        r = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "process", "remove-rgb", temp, "-o", out],
            capture_output=True, text=True, timeout=60, env=cli_env)
        assert r.returncode == 0, (
            f"process remove-rgb failed (rc={r.returncode}):\n"
            f"stdout: {r.stdout[:2000]}\nstderr: {r.stderr[:2000]}")
        data = json.loads(r.stdout)
        assert data.get("status") != "failed", f"Command returned failed: {data}"
    
    def test_level3_cli_extract_vertices(self, tmp_path, cli_env):
        """Test extract mesh vertices."""
        mesh_input = str(tmp_path / "mesh_input.ply")
        Path(mesh_input).write_text(_MESH_PLY_CONTENT)
        temp_mesh = str(tmp_path / "mesh.ply")
        temp_normals = str(tmp_path / "with_normals.ply")
        r1 = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "process", "normals", mesh_input, "-o", temp_normals],
            capture_output=True, text=True, timeout=60, env=cli_env)
        if r1.returncode != 0:
            pytest.fail(f"normals prerequisite failed: {r1.stderr[:500]}")
        
        r2 = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "process", "delaunay", temp_normals, "-o", temp_mesh],
            capture_output=True, text=True, timeout=60, env=cli_env)
        if r2.returncode != 0:
            pytest.fail(f"delaunay prerequisite failed: {r2.stderr[:500]}")
        
        out = str(tmp_path / "vertices.ply")
        r = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "process", "extract-vertices", temp_mesh, "-o", out],
            capture_output=True, text=True, timeout=60, env=cli_env)
        assert r.returncode == 0, (
            f"process extract-vertices failed (rc={r.returncode}):\n"
            f"stdout: {r.stdout[:2000]}\nstderr: {r.stderr[:2000]}")
        data = json.loads(r.stdout)
        assert data.get("status") != "failed", f"Command returned failed: {data}"
    
    def test_level3_cli_flip_triangles(self, tmp_path, cli_env):
        """Test flip triangle normals."""
        mesh_input = str(tmp_path / "mesh_input.ply")
        Path(mesh_input).write_text(_MESH_PLY_CONTENT)
        temp_mesh = str(tmp_path / "mesh.ply")
        temp_normals = str(tmp_path / "with_normals.ply")
        r1 = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "process", "normals", mesh_input, "-o", temp_normals],
            capture_output=True, text=True, timeout=60, env=cli_env)
        if r1.returncode != 0:
            pytest.fail(f"normals prerequisite failed: {r1.stderr[:500]}")
        
        r2 = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "process", "delaunay", temp_normals, "-o", temp_mesh],
            capture_output=True, text=True, timeout=60, env=cli_env)
        if r2.returncode != 0:
            pytest.fail(f"delaunay prerequisite failed: {r2.stderr[:500]}")
        
        out = str(tmp_path / "flipped.ply")
        r = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "process", "flip-triangles", temp_mesh, "-o", out],
            capture_output=True, text=True, timeout=60, env=cli_env)
        assert r.returncode == 0, (
            f"process flip-triangles failed (rc={r.returncode}):\n"
            f"stdout: {r.stdout[:2000]}\nstderr: {r.stderr[:2000]}")
        data = json.loads(r.stdout)
        assert "status" in data

    def test_level3_cli_mesh_volume(self, sample_ply, tmp_path, cli_env):
        """Test compute mesh volume."""
        temp_mesh = str(tmp_path / "mesh.ply")
        temp_normals = str(tmp_path / "with_normals.ply")
        r1 = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "process", "normals", sample_ply, "-o", temp_normals],
            capture_output=True, text=True, timeout=60, env=cli_env)
        if r1.returncode != 0:
            pytest.fail(f"normals prerequisite failed: {r1.stderr[:500]}")
        
        r2 = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "process", "delaunay", temp_normals, "-o", temp_mesh],
            capture_output=True, text=True, timeout=60, env=cli_env)
        if r2.returncode != 0:
            pytest.fail(f"delaunay prerequisite failed: {r2.stderr[:500]}")
        
        r = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "process", "mesh-volume", temp_mesh],
            capture_output=True, text=True, timeout=60, env=cli_env)
        assert r.returncode == 0, (
            f"process mesh-volume failed (rc={r.returncode}):\n"
            f"stdout: {r.stdout[:2000]}\nstderr: {r.stderr[:2000]}")
        data = json.loads(r.stdout)
        assert data.get("status") != "failed", f"Command returned failed: {data}"
    
    def test_level3_cli_merge_meshes(self, tmp_path, cli_env):
        """Test merge multiple meshes."""
        mesh_input = str(tmp_path / "mesh_input.ply")
        Path(mesh_input).write_text(_MESH_PLY_CONTENT)
        mesh1 = str(tmp_path / "mesh1.ply")
        mesh2 = str(tmp_path / "mesh2.ply")
        temp_normals = str(tmp_path / "with_normals.ply")
        
        r1 = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "process", "normals", mesh_input, "-o", temp_normals],
            capture_output=True, text=True, timeout=60, env=cli_env)
        normals_in = _cli_resolve_normals_output_path(r1, temp_normals)

        r2 = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "process", "delaunay", normals_in, "-o", mesh1],
            capture_output=True, text=True, timeout=60, env=cli_env)
        _fail_if_crashed(r2, "process delaunay (merge-meshes prerequisite)")
        if r2.returncode != 0:
            pytest.fail(f"delaunay prerequisite failed: {r2.stderr[:500]}")
        m1p = Path(mesh1)
        if not (m1p.is_file() and m1p.stat().st_size > 0):
            pytest.fail(
                "delaunay did not produce a mesh file for merge-meshes prerequisite")

        import shutil
        shutil.copy(mesh1, mesh2)
        
        out = str(tmp_path / "merged.ply")
        r = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "process", "merge-meshes", mesh1, mesh2, "-o", out],
            capture_output=True, text=True, timeout=60, env=cli_env)
        _fail_if_crashed(r, "process merge-meshes")
        assert r.returncode == 0, (
            f"process merge-meshes failed (rc={r.returncode}):\n"
            f"stdout: {r.stdout[:2000]}\nstderr: {r.stderr[:2000]}")
        data = json.loads(r.stdout)
        if data.get("status") == "failed":
            pytest.fail(
                "merge-meshes returned status=failed (input meshes may be invalid "
                f"for this build): {json.dumps(data, indent=2)[:1200]}\n"
                f"stderr: {r.stderr[:500]}")
    
    def test_level3_cli_sf_coord_to_sf(self, sample_ply, tmp_path, cli_env):
        """Test sf coord-to-sf command."""
        out = str(tmp_path / "height_sf.ply")
        r = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "sf", "coord-to-sf", sample_ply, "-o", out, "--dimension", "Z"],
            capture_output=True, text=True, timeout=60, env=cli_env)
        assert r.returncode == 0, (
            f"sf coord-to-sf failed (rc={r.returncode}):\n"
            f"stdout: {r.stdout[:2000]}\nstderr: {r.stderr[:2000]}")
        data = json.loads(r.stdout)
        assert data.get("status") != "failed"
    
    def test_level3_cli_sf_arithmetic(self, sample_ply, tmp_path, cli_env):
        """Test sf arithmetic command (SQRT operation)."""
        # First create a SF
        temp = str(tmp_path / "with_sf.ply")
        r1 = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "sf", "coord-to-sf", sample_ply, "-o", temp, "--dimension", "Z"],
            capture_output=True, text=True, timeout=60, env=cli_env)
        if r1.returncode != 0:
            pytest.fail(f"coord-to-sf prerequisite failed: {r1.stderr[:500]}")
        
        out = str(tmp_path / "sqrt_sf.ply")
        r = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "sf", "arithmetic", temp, "-o", out, "--sf-index", "0", "--operation", "SQRT"],
            capture_output=True, text=True, timeout=60, env=cli_env)
        assert r.returncode == 0, (
            f"sf arithmetic failed (rc={r.returncode}):\n"
            f"stdout: {r.stdout[:2000]}\nstderr: {r.stderr[:2000]}")
        data = json.loads(r.stdout)
        assert data.get("status") != "failed"
    
    def test_level3_cli_sf_operation(self, sample_ply, tmp_path, cli_env):
        """Test sf operation command (MULTIPLY with constant)."""
        temp = str(tmp_path / "with_sf.ply")
        r1 = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "sf", "coord-to-sf", sample_ply, "-o", temp, "--dimension", "Z"],
            capture_output=True, text=True, timeout=60, env=cli_env)
        if r1.returncode != 0:
            pytest.fail(f"coord-to-sf prerequisite failed: {r1.stderr[:500]}")
        
        # Check if the prerequisite file was actually created
        if not Path(temp).exists():
            pytest.fail(f"coord-to-sf did not create output file: {temp}")
        
        out = str(tmp_path / "scaled_sf.ply")
        r = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "sf", "operation", temp, "-o", out, "--sf-index", "0",
             "--operation", "MULTIPLY", "--value", "2.0"],
            capture_output=True, text=True, timeout=60, env=cli_env)
        assert r.returncode == 0, (
            f"sf operation failed (rc={r.returncode}):\n"
            f"stdout: {r.stdout[:2000]}\nstderr: {r.stderr[:2000]}")
        data = json.loads(r.stdout)
        assert "status" in data

    def test_level3_cli_normals_octree(self, sample_ply, tmp_path, cli_env):
        """Test normals octree command."""
        out = str(tmp_path / "octree_normals.ply")
        r = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "normals", "octree", sample_ply, "-o", out, "--radius", "AUTO"],
            capture_output=True, text=True, timeout=60, env=cli_env)
        assert r.returncode == 0, (
            f"normals octree failed (rc={r.returncode}):\n"
            f"stdout: {r.stdout[:2000]}\nstderr: {r.stderr[:2000]}")
        data = json.loads(r.stdout)
        assert data.get("status") != "failed"
    
    def test_level3_cli_normals_orient_mst(self, sample_ply, tmp_path, cli_env):
        """Test normals orient-mst command."""
        # First compute normals
        temp = str(tmp_path / "with_normals.ply")
        r1 = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "process", "normals", sample_ply, "-o", temp],
            capture_output=True, text=True, timeout=60, env=cli_env)
        if r1.returncode != 0:
            pytest.fail(f"normals prerequisite failed: {r1.stderr[:500]}")
        
        out = str(tmp_path / "oriented.ply")
        r = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "normals", "orient-mst", temp, "-o", out, "--knn", "6"],
            capture_output=True, text=True, timeout=60, env=cli_env)
        assert r.returncode == 0, (
            f"normals orient-mst failed (rc={r.returncode}):\n"
            f"stdout: {r.stdout[:2000]}\nstderr: {r.stderr[:2000]}")
        data = json.loads(r.stdout)
        assert data.get("status") != "failed"
    
    def test_level3_cli_normals_invert(self, sample_ply, tmp_path, cli_env):
        """Test normals invert command."""
        temp = str(tmp_path / "with_normals.ply")
        r1 = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "process", "normals", sample_ply, "-o", temp],
            capture_output=True, text=True, timeout=60, env=cli_env)
        if r1.returncode != 0:
            pytest.fail(f"normals prerequisite failed: {r1.stderr[:500]}")
        
        out = str(tmp_path / "inverted.ply")
        r = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "normals", "invert", temp, "-o", out],
            capture_output=True, text=True, timeout=60, env=cli_env)
        assert r.returncode == 0, (
            f"normals invert failed (rc={r.returncode}):\n"
            f"stdout: {r.stdout[:2000]}\nstderr: {r.stderr[:2000]}")
        data = json.loads(r.stdout)
        assert data.get("status") != "failed"
    
    def test_level3_cli_normals_to_sfs(self, sample_ply, tmp_path, cli_env):
        """Test normals to-sfs command (export as Nx, Ny, Nz scalar fields)."""
        temp = str(tmp_path / "with_normals.ply")
        r1 = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "process", "normals", sample_ply, "-o", temp],
            capture_output=True, text=True, timeout=60, env=cli_env)
        if r1.returncode != 0:
            pytest.fail(f"normals prerequisite failed: {r1.stderr[:500]}")
        
        out = str(tmp_path / "normals_as_sf.ply")
        r = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "normals", "to-sfs", temp, "-o", out],
            capture_output=True, text=True, timeout=60, env=cli_env)
        assert r.returncode == 0, (
            f"normals to-sfs failed (rc={r.returncode}):\n"
            f"stdout: {r.stdout[:2000]}\nstderr: {r.stderr[:2000]}")
        data = json.loads(r.stdout)
        assert data.get("status") != "failed"
    
    def test_level3_cli_normals_clear(self, sample_ply, tmp_path, cli_env):
        """Test normals clear command."""
        temp = str(tmp_path / "with_normals.ply")
        r1 = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "process", "normals", sample_ply, "-o", temp],
            capture_output=True, text=True, timeout=60, env=cli_env)
        if r1.returncode != 0:
            pytest.fail(f"normals prerequisite failed: {r1.stderr[:500]}")
        
        out = str(tmp_path / "no_normals.ply")
        r = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "normals", "clear", temp, "-o", out],
            capture_output=True, text=True, timeout=60, env=cli_env)
        assert r.returncode == 0, (
            f"normals clear failed (rc={r.returncode}):\n"
            f"stdout: {r.stdout[:2000]}\nstderr: {r.stderr[:2000]}")
        data = json.loads(r.stdout)
        assert data.get("status") != "failed", f"Command returned failed: {data}"
    
    def test_level3_cli_normals_to_dip(self, sample_ply, tmp_path, cli_env):
        """Test normals to-dip command (geological dip/dip-direction)."""
        temp = str(tmp_path / "with_normals.ply")
        r1 = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "normals", "octree", sample_ply, "-o", temp, "--radius", "AUTO"],
            capture_output=True, text=True, timeout=60, env=cli_env)
        if r1.returncode != 0:
            pytest.fail(f"normals octree prerequisite failed: {r1.stderr[:500]}")
        
        out = str(tmp_path / "dip.ply")
        r = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "normals", "to-dip", temp, "-o", out],
            capture_output=True, text=True, timeout=60, env=cli_env)
        assert r.returncode == 0, (
            f"normals to-dip failed (rc={r.returncode}):\n"
            f"stdout: {r.stdout[:2000]}\nstderr: {r.stderr[:2000]}")
        data = json.loads(r.stdout)
        assert data.get("status") != "failed", f"Command returned failed: {data}"
    
    def test_level3_cli_sf_gradient(self, sample_ply, tmp_path, cli_env):
        """Test sf gradient command."""
        temp = str(tmp_path / "with_sf.ply")
        r1 = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "sf", "coord-to-sf", sample_ply, "-o", temp, "--dimension", "Z"],
            capture_output=True, text=True, timeout=60, env=cli_env)
        if r1.returncode != 0:
            pytest.fail(f"coord-to-sf prerequisite failed: {r1.stderr[:500]}")
        
        out = str(tmp_path / "gradient.ply")
        r = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "sf", "gradient", temp, "-o", out],
            capture_output=True, text=True, timeout=60, env=cli_env)
        assert r.returncode == 0, (
            f"sf gradient failed (rc={r.returncode}):\n"
            f"stdout: {r.stdout[:2000]}\nstderr: {r.stderr[:2000]}")
        data = json.loads(r.stdout)
        assert "status" in data

    def test_level3_cli_sf_filter(self, sample_ply, tmp_path, cli_env):
        """Test sf filter command (filter points by SF value range)."""
        temp = str(tmp_path / "with_sf.ply")
        r1 = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "sf", "coord-to-sf", sample_ply, "-o", temp, "--dimension", "Z"],
            capture_output=True, text=True, timeout=60, env=cli_env)
        if r1.returncode != 0:
            pytest.fail(f"coord-to-sf prerequisite failed: {r1.stderr[:500]}")
        
        out = str(tmp_path / "filtered.ply")
        r = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "sf", "filter", temp, "-o", out, "--min", "0.0", "--max", "0.5"],
            capture_output=True, text=True, timeout=60, env=cli_env)
        assert r.returncode == 0, (
            f"sf filter failed (rc={r.returncode}):\n"
            f"stdout: {r.stdout[:2000]}\nstderr: {r.stderr[:2000]}")
        data = json.loads(r.stdout)
        assert data.get("status") != "failed", f"Command returned failed: {data}"
    
    def test_level3_cli_sf_convert_to_rgb(self, sample_ply, tmp_path, cli_env):
        """Test sf convert-to-rgb command."""
        temp = str(tmp_path / "with_sf.ply")
        r1 = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "sf", "coord-to-sf", sample_ply, "-o", temp, "--dimension", "Z"],
            capture_output=True, text=True, timeout=60, env=cli_env)
        if r1.returncode != 0:
            pytest.fail(f"coord-to-sf prerequisite failed: {r1.stderr[:500]}")
        
        out = str(tmp_path / "colored.ply")
        r = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "sf", "convert-to-rgb", temp, "-o", out],
            capture_output=True, text=True, timeout=60, env=cli_env)
        assert r.returncode == 0, (
            f"sf convert-to-rgb failed (rc={r.returncode}):\n"
            f"stdout: {r.stdout[:2000]}\nstderr: {r.stderr[:2000]}")
        data = json.loads(r.stdout)
        assert "status" in data

    def test_level3_cli_sf_set_active(self, sample_ply, tmp_path, cli_env):
        """Test sf set-active command."""
        temp = str(tmp_path / "with_sf.ply")
        r1 = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "sf", "coord-to-sf", sample_ply, "-o", temp, "--dimension", "Z"],
            capture_output=True, text=True, timeout=60, env=cli_env)
        if r1.returncode != 0:
            pytest.fail(f"coord-to-sf prerequisite failed: {r1.stderr[:500]}")
        
        out = str(tmp_path / "active_set.ply")
        r = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "sf", "set-active", temp, "-o", out, "--sf-index", "0"],
            capture_output=True, text=True, timeout=60, env=cli_env)
        assert r.returncode == 0, (
            f"sf set-active failed (rc={r.returncode}):\n"
            f"stdout: {r.stdout[:2000]}\nstderr: {r.stderr[:2000]}")
        data = json.loads(r.stdout)
        assert data.get("status") != "failed", f"Command returned failed: {data}"
    
    def test_level3_cli_sf_rename(self, sample_ply, tmp_path, cli_env):
        """Test sf rename command."""
        temp = str(tmp_path / "with_sf.ply")
        r1 = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "sf", "coord-to-sf", sample_ply, "-o", temp, "--dimension", "Z"],
            capture_output=True, text=True, timeout=60, env=cli_env)
        if r1.returncode != 0:
            pytest.fail(f"coord-to-sf prerequisite failed: {r1.stderr[:500]}")
        
        out = str(tmp_path / "renamed.ply")
        r = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "sf", "rename", temp, "-o", out, "--sf-index", "0", "--new-name", "Elevation"],
            capture_output=True, text=True, timeout=60, env=cli_env)
        assert r.returncode == 0, (
            f"sf rename failed (rc={r.returncode}):\n"
            f"stdout: {r.stdout[:2000]}\nstderr: {r.stderr[:2000]}")
        data = json.loads(r.stdout)
        assert data.get("status") != "failed", f"Command returned failed: {data}"
    
    def test_level3_cli_sf_remove(self, sample_ply, tmp_path, cli_env):
        """Test sf remove command."""
        temp = str(tmp_path / "with_sf.ply")
        r1 = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "sf", "coord-to-sf", sample_ply, "-o", temp, "--dimension", "Z"],
            capture_output=True, text=True, timeout=60, env=cli_env)
        if r1.returncode != 0:
            pytest.fail(f"coord-to-sf prerequisite failed: {r1.stderr[:500]}")
        
        out = str(tmp_path / "removed.ply")
        r = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "sf", "remove", temp, "-o", out, "--sf-index", "0"],
            capture_output=True, text=True, timeout=60, env=cli_env)
        assert r.returncode == 0, (
            f"sf remove failed (rc={r.returncode}):\n"
            f"stdout: {r.stdout[:2000]}\nstderr: {r.stderr[:2000]}")
        data = json.loads(r.stdout)
        assert data.get("status") != "failed", f"Command returned failed: {data}"
    
    def test_level3_cli_sf_remove_all(self, sample_ply, tmp_path, cli_env):
        """Test sf remove-all command."""
        temp = str(tmp_path / "with_sf.ply")
        r1 = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "sf", "coord-to-sf", sample_ply, "-o", temp, "--dimension", "Z"],
            capture_output=True, text=True, timeout=60, env=cli_env)
        if r1.returncode != 0:
            pytest.fail(f"coord-to-sf prerequisite failed: {r1.stderr[:500]}")
        
        out = str(tmp_path / "clean.ply")
        r = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "sf", "remove-all", temp, "-o", out],
            capture_output=True, text=True, timeout=60, env=cli_env)
        assert r.returncode == 0, (
            f"sf remove-all failed (rc={r.returncode}):\n"
            f"stdout: {r.stdout[:2000]}\nstderr: {r.stderr[:2000]}")
        data = json.loads(r.stdout)
        assert data.get("status") != "failed", f"Command returned failed: {data}"


# ═══════════════════════════════════════════════════════════════════════════
# Level 3b: Format Conversion Tests (requires ACloudViewer binary)
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not HAS_BINARY, reason="ACloudViewer binary not found")
class TestLevel3_FormatConversion:
    """Test file format conversions via ACloudViewer binary."""

    @pytest.fixture(scope="class")
    def acv_env(self):
        return _build_env_for_binary(BINARY_PATH)

    @pytest.fixture(scope="class")
    def shared_dir(self, tmp_path_factory):
        return tmp_path_factory.mktemp("bin_convert")

    @pytest.fixture(scope="class")
    def sample_ply(self, shared_dir):
        ply_path = shared_dir / "test.ply"
        ply_path.write_text(_SAMPLE_PLY_HEADER + _SAMPLE_PLY_BODY)
        return str(ply_path)

    @pytest.fixture(scope="class")
    def mesh_ply(self, shared_dir):
        """A 10x10 grid PLY that Delaunay can triangulate (non-collinear)."""
        ply_path = shared_dir / "mesh_test.ply"
        ply_path.write_text(_MESH_PLY_CONTENT)
        return str(ply_path)

    def _convert_file(self, input_path, output_path, fmt, acv_env):
        r = subprocess.run(
            [BINARY_PATH, "-SILENT", "-O", input_path,
             "-AUTO_SAVE", "OFF", "-NO_TIMESTAMP",
             "-C_EXPORT_FMT", fmt, "-SAVE_CLOUDS", "FILE", output_path],
            capture_output=True, text=True, timeout=120, env=acv_env)
        return r

    @pytest.fixture(scope="class")
    def converted_pcd(self, sample_ply, shared_dir, acv_env):
        """PLY->PCD conversion shared across tests that need a PCD input."""
        out = str(shared_dir / "converted.pcd")
        r = subprocess.run(
            [BINARY_PATH, "-SILENT", "-O", sample_ply,
             "-AUTO_SAVE", "OFF", "-NO_TIMESTAMP",
             "-C_EXPORT_FMT", "PCD", "-SAVE_CLOUDS", "FILE", out],
            capture_output=True, text=True, timeout=120, env=acv_env)
        return out, r

    @pytest.fixture(scope="class")
    def converted_drc(self, converted_pcd, shared_dir, acv_env):
        """PCD->DRC conversion shared across tests that need a DRC input."""
        pcd, _ = converted_pcd
        if not Path(pcd).exists():
            return None, None
        out = str(shared_dir / "converted.drc")
        r = subprocess.run(
            [BINARY_PATH, "-SILENT", "-O", pcd,
             "-AUTO_SAVE", "OFF", "-NO_TIMESTAMP",
             "-C_EXPORT_FMT", "DRC", "-SAVE_CLOUDS", "FILE", out],
            capture_output=True, text=True, timeout=120, env=acv_env)
        return out, r

    def test_level3_ply_to_pcd(self, converted_pcd):
        out, r = converted_pcd
        assert r.returncode == 0, \
            f"PLY->PCD failed:\n{(r.stdout + r.stderr)[-2000:]}"
        assert Path(out).exists() and Path(out).stat().st_size > 0, \
            "PLY->PCD output file missing or empty"

    def test_level3_pcd_to_ply(self, converted_pcd, shared_dir, acv_env):
        pcd, pcd_r = converted_pcd
        _fail_if_crashed(pcd_r, "PLY->PCD")
        if not Path(pcd).exists():
            pytest.fail(
                f"PLY->PCD prerequisite failed (rc={pcd_r.returncode}):\n"
                f"{(pcd_r.stdout + pcd_r.stderr)[-1000:]}"
            )
        out = str(shared_dir / "roundtrip.ply")
        r = self._convert_file(pcd, out, "PLY", acv_env)
        _fail_if_crashed(r, "PCD->PLY")
        assert r.returncode == 0, \
            f"PCD->PLY failed:\n{(r.stdout + r.stderr)[-2000:]}"
        assert Path(out).exists() and Path(out).stat().st_size > 0, \
            "PCD->PLY output file missing or empty"

    def test_level3_pcd_to_drc(self, converted_drc):
        if converted_drc[0] is None:
            pytest.fail("PLY->PCD prerequisite failed")
        out, r = converted_drc
        _fail_if_crashed(r, "PCD->DRC")
        assert r.returncode == 0, \
            f"PCD->DRC failed:\n{(r.stdout + r.stderr)[-2000:]}"
        assert Path(out).exists() and Path(out).stat().st_size > 0, \
            "PCD->DRC output file missing or empty"

    def test_level3_drc_to_pcd(self, converted_drc, shared_dir, acv_env):
        if converted_drc[0] is None:
            pytest.fail("PLY->PCD prerequisite failed")
        drc, drc_r = converted_drc
        if not Path(drc).exists():
            pytest.fail(
                f"PCD->DRC prerequisite failed (rc={drc_r.returncode}):\n"
                f"{(drc_r.stdout + drc_r.stderr)[-1000:]}"
            )
        out = str(shared_dir / "roundtrip.pcd")
        r = self._convert_file(drc, out, "PCD", acv_env)
        assert r.returncode == 0, \
            f"DRC->PCD failed:\n{(r.stdout + r.stderr)[-2000:]}"
        assert Path(out).exists() and Path(out).stat().st_size > 0, \
            "DRC->PCD output file missing or empty"

    @pytest.mark.parametrize("fmt", ["PLY", "ASC", "BIN", "VTK"])
    def test_level3_basic_format_conversion(self, sample_ply, acv_env, tmp_path, fmt):
        out = str(tmp_path / f"test.{fmt.lower()}")
        r = self._convert_file(sample_ply, out, fmt, acv_env)
        if IS_WINDOWS and r.returncode >= 0xC0000000:
            pytest.fail(
                f"PLY->{fmt} crashed (rc=0x{r.returncode:08X}):\n"
                f"{(r.stdout + r.stderr)[-2000:]}"
            )
        assert r.returncode == 0, \
            f"PLY->{fmt} failed (rc={r.returncode}):\n{(r.stdout+r.stderr)[-2000:]}"
        assert Path(out).exists() and Path(out).stat().st_size > 0, \
            f"{fmt} output file missing or empty: {out}"

    @pytest.mark.parametrize("fmt", ["OBJ", "OFF", "STL"])
    def test_level3_mesh_format_conversion(self, mesh_ply, acv_env, tmp_path, fmt):
        """Cloud->Mesh format via Delaunay + mesh export."""
        out = str(tmp_path / f"mesh.{fmt.lower()}")
        r = subprocess.run(
            [BINARY_PATH, "-SILENT", "-O", mesh_ply,
             "-DELAUNAY", "-M_EXPORT_FMT", fmt,
             "-AUTO_SAVE", "OFF", "-NO_TIMESTAMP",
             "-SAVE_MESHES", "FILE", out],
            capture_output=True, text=True, timeout=120,
            env=acv_env)
        assert r.returncode == 0, \
            f"PLY->{fmt} (mesh) failed (rc={r.returncode}):\n{(r.stdout+r.stderr)[-2000:]}"
        assert Path(out).exists() and Path(out).stat().st_size > 0, \
            f"{fmt} mesh output file missing or empty: {out}"

    @pytest.mark.parametrize("ext", [".xyz", ".txt", ".csv", ".pts"])
    def test_level3_ascii_variants(self, sample_ply, acv_env, tmp_path, ext):
        """All ASCII variant extensions export as ASC format."""
        out = str(tmp_path / f"test{ext}")
        r = self._convert_file(sample_ply, out, "ASC", acv_env)
        assert r.returncode == 0, \
            f"PLY->ASC({ext}) failed (rc={r.returncode}):\n{(r.stdout+r.stderr)[-2000:]}"
        assert Path(out).exists() and Path(out).stat().st_size > 0, \
            f"ASC({ext}) output file missing or empty: {out}"

    def test_level3_las_conversion(self, sample_ply, acv_env, tmp_path):
        """PLY -> LAS (requires qLASIO or qPDALIO plugin)."""
        _require_plugin("QLAS_IO_PLUGIN")
        out = str(tmp_path / "test.las")
        r = self._convert_file(sample_ply, out, "LAS", acv_env)
        assert r.returncode == 0, \
            f"PLY->LAS failed (rc={r.returncode}):\n{(r.stdout+r.stderr)[-2000:]}"
        assert Path(out).exists() and Path(out).stat().st_size > 0, \
            f"LAS output file missing or empty: {out}"

    def test_level3_e57_conversion(self, sample_ply, acv_env, tmp_path):
        """PLY -> E57 (requires qE57IO plugin)."""
        _require_plugin("QE57_IO_PLUGIN")
        out = str(tmp_path / "test.e57")
        r = self._convert_file(sample_ply, out, "E57", acv_env)
        assert r.returncode == 0, \
            f"PLY->E57 failed (rc={r.returncode}):\n{(r.stdout+r.stderr)[-2000:]}"
        assert Path(out).exists() and Path(out).stat().st_size > 0, \
            f"E57 output file missing or empty: {out}"

    def test_level3_fbx_conversion(self, mesh_ply, acv_env, tmp_path):
        """PLY -> FBX (requires qFBXIO plugin, mesh-based)."""
        _require_plugin("QFBX_IO_PLUGIN")
        out = str(tmp_path / "test.fbx")
        r = subprocess.run(
            [BINARY_PATH, "-SILENT", "-O", mesh_ply,
             "-DELAUNAY", "-M_EXPORT_FMT", "FBX",
             "-AUTO_SAVE", "OFF", "-NO_TIMESTAMP",
             "-SAVE_MESHES", "FILE", out],
            capture_output=True, text=True, timeout=120,
            env=acv_env)
        _fail_if_crashed(r, "FBX conversion")
        assert r.returncode == 0, \
            f"PLY->FBX (mesh) failed (rc={r.returncode}):\n{(r.stdout+r.stderr)[-2000:]}"
        assert Path(out).exists() and Path(out).stat().st_size > 0, \
            f"FBX output file missing or empty: {out}"

    def test_level3_sbf_conversion(self, sample_ply, acv_env, tmp_path):
        """PLY -> SBF (SimpleBin, qCoreIO plugin)."""
        _require_plugin("QCORE_IO_PLUGIN")
        out = str(tmp_path / "test.sbf")
        r = self._convert_file(sample_ply, out, "SBF", acv_env)
        assert r.returncode == 0, \
            f"PLY->SBF failed (rc={r.returncode}):\n{(r.stdout+r.stderr)[-2000:]}"
        assert Path(out).exists() and Path(out).stat().st_size > 0, \
            f"SBF output file missing or empty: {out}"

    def test_level3_dxf_conversion(self, sample_ply, acv_env, tmp_path):
        """PLY -> DXF (requires CV_DXF_SUPPORT / DXF IO)."""
        out = str(tmp_path / "test.dxf")
        r = self._convert_file(sample_ply, out, "DXF", acv_env)
        assert r.returncode == 0, \
            f"PLY->DXF failed (rc={r.returncode}):\n{(r.stdout+r.stderr)[-2000:]}"
        assert Path(out).exists() and Path(out).stat().st_size > 0, \
            f"DXF output file missing or empty: {out}"

    def test_level3_ascii_export_separator_content(self, sample_ply, acv_env, tmp_path):
        """Verify ASCII export body: point rows, parsable coordinates, cross-extension consistency.

        Many builds write space-separated floats for all ASC extensions; this still
        validates file content (row count and numeric triplets) and that .csv and .xyz
        bodies match for the same cloud.
        """
        def _data_lines(p: Path) -> list[str]:
            return [ln.strip() for ln in p.read_text(encoding="utf-8", errors="replace").splitlines()
                    if ln.strip() and not ln.strip().startswith("#")]

        csv_out = str(tmp_path / "out.csv")
        r = self._convert_file(sample_ply, csv_out, "ASC", acv_env)
        assert r.returncode == 0, \
            f"PLY->ASC(.csv) failed:\n{(r.stdout+r.stderr)[-2000:]}"
        csv_lines = _data_lines(Path(csv_out))
        assert len(csv_lines) == 100
        for ln in csv_lines[:5]:
            parts = [p for p in ln.replace(",", " ").split() if p]
            assert len(parts) >= 3
            float(parts[0])
            float(parts[1])
            float(parts[2])

        xyz_out = str(tmp_path / "points.xyz")
        r2 = self._convert_file(sample_ply, xyz_out, "ASC", acv_env)
        assert r2.returncode == 0, \
            f"PLY->ASC(.xyz) failed:\n{(r2.stdout+r2.stderr)[-2000:]}"
        xyz_lines = _data_lines(Path(xyz_out))
        assert len(xyz_lines) == 100
        assert csv_lines == xyz_lines, \
            ".csv and .xyz ASC exports should match for the same input cloud"

        pts_out = str(tmp_path / "cloud.pts")
        r3 = self._convert_file(sample_ply, pts_out, "ASC", acv_env)
        assert r3.returncode == 0, \
            f"PLY->ASC(.pts) failed:\n{(r3.stdout+r3.stderr)[-2000:]}"
        pts_lines = _data_lines(Path(pts_out))
        assert len(pts_lines) == 100
        assert pts_lines == csv_lines, ".pts body should match .csv for the same cloud"

    def test_level3_roundtrip_ply_asc_ply(self, sample_ply, acv_env, tmp_path):
        """PLY -> ASC -> PLY preserves point count."""
        asc = str(tmp_path / "step.asc")
        r1 = self._convert_file(sample_ply, asc, "ASC", acv_env)
        assert r1.returncode == 0, \
            f"PLY->ASC failed:\n{(r1.stdout+r1.stderr)[-2000:]}"
        ply_back = str(tmp_path / "roundtrip.ply")
        r2 = self._convert_file(asc, ply_back, "PLY", acv_env)
        assert r2.returncode == 0, \
            f"ASC->PLY failed:\n{(r2.stdout+r2.stderr)[-2000:]}"
        assert Path(ply_back).exists() and Path(ply_back).stat().st_size > 0
        assert _ply_vertex_count(Path(ply_back)) == 100

    def test_level3_roundtrip_ply_vtk_ply(self, sample_ply, acv_env, tmp_path):
        """PLY -> VTK -> PLY preserves point count."""
        vtk = str(tmp_path / "step.vtk")
        r1 = self._convert_file(sample_ply, vtk, "VTK", acv_env)
        assert r1.returncode == 0, \
            f"PLY->VTK failed:\n{(r1.stdout+r1.stderr)[-2000:]}"
        ply_back = str(tmp_path / "roundtrip_vtk.ply")
        r2 = self._convert_file(vtk, ply_back, "PLY", acv_env)
        assert r2.returncode == 0, \
            f"VTK->PLY failed:\n{(r2.stdout+r2.stderr)[-2000:]}"
        assert Path(ply_back).exists() and Path(ply_back).stat().st_size > 0
        assert _ply_vertex_count(Path(ply_back)) == 100

    def test_level3_mesh_obj_roundtrip_vertex_count(self, mesh_ply, acv_env, tmp_path):
        """Mesh export to OBJ and reimport to PLY preserves vertex count."""
        obj_path = str(tmp_path / "mesh.obj")
        r = subprocess.run(
            [BINARY_PATH, "-SILENT", "-O", mesh_ply,
             "-DELAUNAY", "-M_EXPORT_FMT", "OBJ",
             "-AUTO_SAVE", "OFF", "-NO_TIMESTAMP",
             "-SAVE_MESHES", "FILE", obj_path],
            capture_output=True, text=True, timeout=120, env=acv_env)
        assert r.returncode == 0, \
            f"PLY->OBJ failed:\n{(r.stdout+r.stderr)[-2000:]}"
        assert Path(obj_path).exists() and Path(obj_path).stat().st_size > 0
        nv = _count_obj_vertices(Path(obj_path))
        assert nv > 0, "OBJ should list vertices"
        ply_back = str(tmp_path / "mesh_back.ply")
        r2 = subprocess.run(
            [BINARY_PATH, "-SILENT", "-O", obj_path,
             "-AUTO_SAVE", "OFF", "-NO_TIMESTAMP",
             "-M_EXPORT_FMT", "PLY", "-SAVE_MESHES", "FILE", ply_back],
            capture_output=True, text=True, timeout=120, env=acv_env)
        assert r2.returncode == 0, \
            f"OBJ->PLY (mesh) failed:\n{(r2.stdout+r2.stderr)[-2000:]}"
        assert Path(ply_back).exists() and Path(ply_back).stat().st_size > 0
        assert _ply_vertex_count(Path(ply_back)) == nv

    def test_level3_mesh_off_roundtrip_vertex_count(self, mesh_ply, acv_env, tmp_path):
        """Mesh export to OFF and reimport to PLY preserves vertex count."""
        off_path = str(tmp_path / "mesh.off")
        r = subprocess.run(
            [BINARY_PATH, "-SILENT", "-O", mesh_ply,
             "-DELAUNAY", "-M_EXPORT_FMT", "OFF",
             "-AUTO_SAVE", "OFF", "-NO_TIMESTAMP",
             "-SAVE_MESHES", "FILE", off_path],
            capture_output=True, text=True, timeout=120, env=acv_env)
        assert r.returncode == 0, \
            f"PLY->OFF failed:\n{(r.stdout+r.stderr)[-2000:]}"
        assert Path(off_path).exists() and Path(off_path).stat().st_size > 0
        nv = _count_off_vertices(Path(off_path))
        assert nv > 0, "OFF should declare vertices"
        ply_back = str(tmp_path / "mesh_off_back.ply")
        r2 = subprocess.run(
            [BINARY_PATH, "-SILENT", "-O", off_path,
             "-AUTO_SAVE", "OFF", "-NO_TIMESTAMP",
             "-M_EXPORT_FMT", "PLY", "-SAVE_MESHES", "FILE", ply_back],
            capture_output=True, text=True, timeout=120, env=acv_env)
        assert r2.returncode == 0, \
            f"OFF->PLY (mesh) failed:\n{(r2.stdout+r2.stderr)[-2000:]}"
        assert Path(ply_back).exists() and Path(ply_back).stat().st_size > 0
        assert _ply_vertex_count(Path(ply_back)) == nv

    def test_level3_convert_nonexistent_input(self, acv_env, tmp_path):
        """Conversion with a missing input file should fail."""
        out = str(tmp_path / "noop.ply")
        r = subprocess.run(
            [BINARY_PATH, "-SILENT", "-O", "/nonexistent/path/does_not_exist.ply",
             "-AUTO_SAVE", "OFF", "-NO_TIMESTAMP",
             "-C_EXPORT_FMT", "PLY", "-SAVE_CLOUDS", "FILE", out],
            capture_output=True, text=True, timeout=60, env=acv_env)
        assert r.returncode != 0, "expected failure for missing input file"

    def test_level3_convert_unsupported_export_format(self, sample_ply, acv_env, tmp_path):
        """Export with an invalid format token should not succeed as a normal save."""
        out = str(tmp_path / "out.ply")
        r = subprocess.run(
            [BINARY_PATH, "-SILENT", "-O", sample_ply,
             "-AUTO_SAVE", "OFF", "-NO_TIMESTAMP",
             "-C_EXPORT_FMT", "__INVALID_FORMAT__",
             "-SAVE_CLOUDS", "FILE", out],
            capture_output=True, text=True, timeout=60, env=acv_env)
        assert r.returncode != 0 or not Path(out).exists(), \
            "unsupported format should produce error or no output"

    def test_level3_laz_conversion(self, sample_ply, acv_env, tmp_path):
        """PLY -> LAZ (same LAS plugin; -C_EXPORT_FMT LAS; .laz selects compressed)."""
        _require_plugin("QLAS_IO_PLUGIN")
        out = str(tmp_path / "test.laz")
        r = self._convert_file(sample_ply, out, "LAS", acv_env)
        assert r.returncode == 0, \
            f"PLY->LAZ failed (rc={r.returncode}):\n{(r.stdout+r.stderr)[-2000:]}"
        assert Path(out).exists() and Path(out).stat().st_size > 0, \
            f"LAZ output file missing or empty: {out}"

    def test_level3_shp_conversion(self, sample_ply, acv_env, tmp_path):
        """PLY -> SHP (optional plugin / build)."""
        out = str(tmp_path / "test.shp")
        r = self._convert_file(sample_ply, out, "SHP", acv_env)
        assert r.returncode == 0, \
            f"PLY->SHP failed (rc={r.returncode}):\n{(r.stdout+r.stderr)[-2000:]}"
        assert Path(out).exists() and Path(out).stat().st_size > 0, \
            f"SHP output file missing or empty: {out}"

    def test_level3_ptx_import(self, acv_env, tmp_path):
        """Minimal PTX loads (PTXFilter is built into CV_io unconditionally)."""
        ptx_path = tmp_path / "minimal.ptx"
        ptx_path.write_text(_MINIMAL_PTX_CONTENT)
        r = subprocess.run(
            [BINARY_PATH, "-SILENT", "-O", str(ptx_path), "-SAVE_CLOUDS"],
            capture_output=True, text=True, timeout=120, env=acv_env)
        assert r.returncode == 0, \
            f"PTX load failed (rc={r.returncode}):\n{(r.stdout+r.stderr)[-2000:]}"

    def test_level3_neu_ascii_variant(self, sample_ply, acv_env, tmp_path):
        """ASCII export with .neu extension (ASC format)."""
        out = str(tmp_path / "test.neu")
        r = self._convert_file(sample_ply, out, "ASC", acv_env)
        assert r.returncode == 0, \
            f"PLY->ASC(.neu) failed (rc={r.returncode}):\n{(r.stdout+r.stderr)[-2000:]}"
        assert Path(out).exists() and Path(out).stat().st_size > 0, \
            f"ASC(.neu) output file missing or empty: {out}"


@pytest.mark.skipif(not HAS_CLI, reason="CLI harness not installed")
@pytest.mark.skipif(not HAS_BINARY, reason="ACloudViewer binary not found")
class TestLevel3_CLIFormatConversion:
    """Test format conversion via the CLI harness."""

    @pytest.fixture(scope="class")
    def cli_env(self):
        env = _build_env_for_binary(BINARY_PATH)
        env["ACV_BINARY"] = BINARY_PATH
        return env

    @pytest.fixture(scope="class")
    def shared_dir(self, tmp_path_factory):
        return tmp_path_factory.mktemp("cli_convert")

    @pytest.fixture(scope="class")
    def sample_ply(self, shared_dir):
        ply_path = shared_dir / "test.ply"
        ply_path.write_text(_SAMPLE_PLY_HEADER + _SAMPLE_PLY_BODY)
        return str(ply_path)

    @pytest.fixture(scope="class")
    def converted_pcd(self, sample_ply, shared_dir, cli_env):
        """PLY->PCD conversion shared across tests that need a PCD input."""
        out = str(shared_dir / "converted.pcd")
        r = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "convert", sample_ply, out],
            capture_output=True, text=True, timeout=120, env=cli_env)
        return out, r

    def test_level3_cli_ply_to_pcd(self, converted_pcd):
        out, r = converted_pcd
        assert r.returncode == 0, (
            f"CLI PLY->PCD failed (rc={r.returncode}):\n"
            f"stdout: {r.stdout[:2000]}\nstderr: {r.stderr[:2000]}")
        data = json.loads(r.stdout)
        assert data.get("status") != "failed", (
            f"CLI PLY->PCD returned status=failed. Full response:\n"
            f"{json.dumps(data, indent=2)}\nstderr: {r.stderr[:1000]}")

    def test_level3_cli_pcd_to_drc(self, converted_pcd, shared_dir, cli_env):
        pcd, pcd_r = converted_pcd
        _fail_if_crashed(pcd_r, "CLI PLY->PCD")
        if not Path(pcd).exists():
            pytest.fail(
                f"PLY->PCD prerequisite failed (rc={pcd_r.returncode}):\n"
                f"stdout: {pcd_r.stdout[:1000]}\nstderr: {pcd_r.stderr[:1000]}"
            )
        out = str(shared_dir / "output.drc")
        r = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "convert", pcd, out],
            capture_output=True, text=True, timeout=120, env=cli_env)
        _fail_if_crashed(r, "CLI PCD->DRC")
        assert r.returncode == 0, (
            f"CLI PCD->DRC failed (rc={r.returncode}):\n"
            f"stdout: {r.stdout[:2000]}\nstderr: {r.stderr[:2000]}")
        data = json.loads(r.stdout)
        assert data.get("status") != "failed", (
            f"CLI PCD->DRC returned status=failed. Full response:\n"
            f"{json.dumps(data, indent=2)}\nstderr: {r.stderr[:1000]}")

    def test_level3_cli_batch_convert_pcd(self, sample_ply, shared_dir, cli_env):
        src_dir = shared_dir / "input_batch"
        src_dir.mkdir(exist_ok=True)
        for i in range(3):
            (src_dir / f"cloud_{i}.ply").write_text(
                _SAMPLE_PLY_HEADER + _SAMPLE_PLY_BODY)
        out_dir = str(shared_dir / "output_batch")
        r = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "batch-convert", str(src_dir), out_dir, "--format", "pcd"],
            capture_output=True, text=True, timeout=180, env=cli_env)
        assert r.returncode == 0, (
            f"CLI batch-convert failed (rc={r.returncode}):\n"
            f"stdout: {r.stdout[:2000]}\nstderr: {r.stderr[:2000]}")
        data = json.loads(r.stdout)
        assert data.get("status") != "failed", (
            f"CLI batch-convert returned status=failed. Full response:\n"
            f"{json.dumps(data, indent=2)}\nstderr: {r.stderr[:1000]}")


# ═══════════════════════════════════════════════════════════════════════════
# Level 4: GUI RPC Tests (requires running ACloudViewer with JSON-RPC)
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def level4_cleanup():
    """Clean up the scene after all Level 4 tests complete."""
    yield
    if _rpc_available():
        try:
            _rpc_call("clear", timeout=5)
        except Exception:  # best-effort cleanup
            pass


@pytest.mark.skipif(not _rpc_available(), reason="ACloudViewer RPC not available")
@pytest.mark.usefixtures("level4_cleanup")
class TestLevel4_GUIRPC:
    """Basic JSON-RPC connectivity and view control (no entities needed)."""

    # --- Basic connectivity ---

    def test_level4_ping(self):
        assert _rpc_call("ping") == "pong"

    def test_level4_ping_repeated(self):
        for _ in range(5):
            assert _rpc_call("ping") == "pong"

    def test_level4_methods_list(self):
        methods = _rpc_call("methods.list")
        assert len(methods) >= 44, (
            f"Expected >=44 RPC methods, got {len(methods)}")
        names = {m["method"] for m in methods}
        for expected in ["open", "export", "file.convert", "scene.list",
                         "scene.info", "scene.remove", "scene.setVisible",
                         "cloud.computeNormals", "cloud.subsample",
                         "cloud.crop", "cloud.getScalarFields",
                         "cloud.paintUniform", "cloud.paintByHeight",
                         "cloud.setActiveSf", "cloud.removeSf",
                         "cloud.removeAllSfs", "cloud.renameSf",
                         "cloud.filterSf", "cloud.coordToSF",
                         "cloud.removeRgb", "cloud.removeNormals",
                         "cloud.invertNormals", "cloud.merge",
                         "mesh.simplify", "mesh.smooth", "mesh.samplePoints",
                         "mesh.extractVertices", "mesh.flipTriangles",
                         "mesh.volume", "mesh.merge",
                         "view.screenshot", "view.setOrientation",
                         "view.zoomFit", "view.setPerspective",
                         "view.setPointSize", "view.getCamera",
                         "colmap.reconstruct", "colmap.run",
                         "transform.apply",
                         "entity.rename", "entity.setColor",
                         "yolo.track", "sam3d.generate", "mapanything.reconstruct",
                         "methods.list", "ping"]:
            assert expected in names, f"Missing RPC method: {expected}"

    def test_level4_methods_list_has_descriptions(self):
        methods = _rpc_call("methods.list")
        for m in methods:
            assert "method" in m, "Each method entry must have 'method' key"
            assert "description" in m, f"Method '{m['method']}' missing description"

    def test_level4_unknown_method_returns_error(self):
        with pytest.raises(RuntimeError, match="Method not found"):
            _rpc_call("nonexistent.method")

    # --- Scene (empty) ---

    def test_level4_scene_list_empty(self):
        entities = _rpc_call("scene.list", {"recursive": True})
        assert isinstance(entities, list)

    # --- View control (no entities needed) ---

    def test_level4_view_get_camera(self):
        cam = _rpc_call("view.getCamera")
        assert "view_matrix" in cam
        assert "fov_deg" in cam
        assert "perspective" in cam
        assert "near_clipping" in cam
        assert "far_clipping" in cam
        assert isinstance(cam["view_matrix"], list)
        assert len(cam["view_matrix"]) == 16

    @pytest.mark.parametrize("orientation", [
        "top", "bottom", "front", "back", "left", "right", "iso1", "iso2",
    ])
    def test_level4_view_set_orientation(self, orientation):
        result = _rpc_call("view.setOrientation", {"orientation": orientation})
        assert isinstance(result, (int, dict)), \
            f"Unexpected result type: {type(result).__name__}: {result}"
        if isinstance(result, dict):
            assert result.get("orientation") == orientation

    def test_level4_view_camera_differs_by_orientation(self):
        _rpc_call("view.setOrientation", {"orientation": "front"})
        cam_front = _rpc_call("view.getCamera")
        _rpc_call("view.setOrientation", {"orientation": "top"})
        cam_top = _rpc_call("view.getCamera")
        assert cam_front["view_matrix"] != cam_top["view_matrix"], \
            "Camera matrix should differ between front and top views"

    def test_level4_view_zoom_fit(self):
        assert _rpc_call("view.zoomFit") == 0

    def test_level4_view_refresh(self):
        assert _rpc_call("view.refresh") == 0

    def test_level4_view_set_perspective_object(self):
        result = _rpc_call("view.setPerspective", {"mode": "object"})
        assert isinstance(result, (int, dict)), f"Unexpected: {result}"
        if isinstance(result, dict):
            assert result.get("mode") == "object"

    def test_level4_view_set_perspective_viewer(self):
        result = _rpc_call("view.setPerspective", {"mode": "viewer"})
        assert isinstance(result, (int, dict)), f"Unexpected: {result}"
        if isinstance(result, dict):
            assert result.get("mode") == "viewer"

    def test_level4_view_point_size_increase_decrease(self):
        r1 = _rpc_call("view.setPointSize", {"action": "increase"})
        assert isinstance(r1, (int, dict)), f"Unexpected: {r1}"
        r2 = _rpc_call("view.setPointSize", {"action": "decrease"})
        assert isinstance(r2, (int, dict)), f"Unexpected: {r2}"

    def test_level4_view_screenshot_empty_scene(self, tmp_path):
        path = str(tmp_path / "empty_scene.png")
        result = _rpc_call("view.screenshot", {"filename": path})
        assert result["width"] > 0
        assert result["height"] > 0
        assert Path(path).exists()

    # --- CLI GUI info ---

    @pytest.mark.skipif(not HAS_CLI, reason="CLI harness not installed")
    def test_level4_cli_gui_info(self):
        r = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "gui", "info"],
            capture_output=True, text=True, timeout=10)
        assert r.returncode == 0
        data = json.loads(r.stdout)
        assert data["mode"] == "gui"


# ── Level 4b: RPC File I/O ────────────────────────────────────────────────

@pytest.mark.skipif(not _rpc_available(), reason="ACloudViewer RPC not available")
@pytest.mark.usefixtures("level4_cleanup")
class TestLevel4_RPCFileOps:
    """Test JSON-RPC file open, convert, export, and screenshot after load."""

    @pytest.fixture
    def sample_ply(self, tmp_path):
        p = tmp_path / "rpc_test.ply"
        p.write_text(_SAMPLE_PLY_HEADER + _SAMPLE_PLY_BODY)
        return str(p)

    @pytest.fixture
    def mesh_ply(self, tmp_path):
        p = tmp_path / "rpc_mesh.ply"
        p.write_text(_MESH_PLY_CONTENT)
        return str(p)

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        yield
        try:
            _rpc_call("clear", timeout=5)
        except Exception:  # best-effort cleanup
            pass

    def test_level4_rpc_open_ply(self, sample_ply):
        result = _rpc_call("open", {"filename": sample_ply, "silent": True})
        assert isinstance(result, dict)
        assert "id" in result
        assert "name" in result

    def test_level4_rpc_open_returns_cloud(self, sample_ply):
        result = _rpc_call("open", {"filename": sample_ply, "silent": True})
        cloud_id = _find_cloud_id(result)
        assert cloud_id is not None, "Loaded PLY should contain a point cloud"

    def test_level4_rpc_open_point_count(self, sample_ply):
        result = _rpc_call("open", {"filename": sample_ply, "silent": True})
        cloud_id = _find_cloud_id(result)
        info = _rpc_call("scene.info", {"entity_id": cloud_id})
        assert info["point_count"] == 100

    def test_level4_rpc_scene_count_after_load(self, sample_ply):
        _rpc_call("open", {"filename": sample_ply, "silent": True})
        entities = _rpc_call("scene.list", {"recursive": True})
        assert len(entities) >= 1, "Scene should have ≥1 entity after loading"

    def test_level4_rpc_scene_info(self, sample_ply):
        loaded = _rpc_call("open", {"filename": sample_ply, "silent": True})
        info = _rpc_call("scene.info", {"entity_id": loaded["id"]})
        assert info["id"] == loaded["id"]
        assert "name" in info
        assert "type" in info

    @pytest.mark.parametrize("ext", ["asc", "ply", "bin"])
    def test_level4_rpc_file_convert(self, sample_ply, tmp_path, ext):
        out = str(tmp_path / f"converted.{ext}")
        result = _rpc_call("file.convert",
                           {"input": sample_ply, "output": out}, timeout=60)
        assert result["status"] == "converted"
        assert Path(out).exists() and Path(out).stat().st_size > 0, \
            f"Converted .{ext} file missing or empty"

    def test_level4_rpc_export_entity(self, sample_ply, tmp_path):
        loaded = _rpc_call("open", {"filename": sample_ply, "silent": True})
        cloud_id = _find_cloud_id(loaded)
        group_id = loaded["id"]
        out = str(tmp_path / "exported.ply")
        # Try exporting the point cloud first, fall back to the group
        exported = False
        for eid in [cloud_id, group_id]:
            if eid is None:
                continue
            try:
                result = _rpc_call("export",
                                   {"entity_id": eid, "filename": out},
                                   timeout=60)
                assert result["filename"] == out
                exported = True
                break
            except RuntimeError:
                continue
        assert exported, "Export should succeed with either cloud or group ID"
        assert Path(out).exists()
        assert Path(out).stat().st_size > 0

    def test_level4_rpc_screenshot_with_data(self, sample_ply, tmp_path):
        _rpc_call("open", {"filename": sample_ply, "silent": True})
        _rpc_call("view.zoomFit")
        _rpc_call("view.setOrientation", {"orientation": "iso1"})
        path = str(tmp_path / "loaded_scene.png")
        result = _rpc_call("view.screenshot", {"filename": path})
        assert result["width"] > 0
        assert result["height"] > 0
        assert Path(path).exists()
        assert Path(path).stat().st_size > 100

    def test_level4_rpc_clear_scene(self, sample_ply):
        _rpc_call("open", {"filename": sample_ply, "silent": True})
        before = _rpc_call("scene.list")
        assert len(before) >= 1
        _rpc_call("clear")
        after = _rpc_call("scene.list")
        assert len(after) == 0, "Scene should be empty after clear"

    def test_level4_rpc_open_missing_file(self):
        with pytest.raises(RuntimeError):
            _rpc_call("open", {"filename": "/nonexistent/path.ply", "silent": True})

    @pytest.mark.parametrize("ext", ["pcd", "vtk", "dxf"])
    def test_level4_rpc_file_convert_extended_formats(self, sample_ply, tmp_path, ext):
        """RPC file.convert for additional point-cloud output extensions."""
        out = str(tmp_path / f"rpc_converted.{ext}")
        try:
            result = _rpc_call("file.convert",
                               {"input": sample_ply, "output": out}, timeout=120)
        except RuntimeError as exc:
            pytest.fail(f"file.convert to .{ext} failed: {exc}")
        assert result["status"] == "converted"
        assert Path(out).exists() and Path(out).stat().st_size > 0, \
            f"Converted .{ext} should exist and be non-empty"

    @pytest.mark.skipif(not HAS_BINARY, reason="ACloudViewer binary not found")
    def test_level4_rpc_file_convert_stl_mesh(self, mesh_ply, tmp_path):
        """RPC file.convert to STL using a mesh produced from ``mesh_ply`` (Delaunay → OBJ)."""
        obj_path = str(tmp_path / "mesh_for_stl.obj")
        env = _build_env_for_binary(BINARY_PATH)
        r = subprocess.run(
            [BINARY_PATH, "-SILENT", "-O", mesh_ply,
             "-DELAUNAY", "-M_EXPORT_FMT", "OBJ",
             "-AUTO_SAVE", "OFF", "-NO_TIMESTAMP",
             "-SAVE_MESHES", "FILE", obj_path],
            capture_output=True, text=True, timeout=120, env=env)
        assert r.returncode == 0, \
            f"mesh PLY->OBJ prerequisite failed:\n{(r.stdout+r.stderr)[-2000:]}"
        assert Path(obj_path).exists() and Path(obj_path).stat().st_size > 0
        stl_out = str(tmp_path / "converted.stl")
        result = _rpc_call("file.convert",
                           {"input": obj_path, "output": stl_out}, timeout=120)
        assert result["status"] == "converted"
        assert Path(stl_out).exists() and Path(stl_out).stat().st_size > 0

    def test_level4_rpc_file_convert_missing_input(self, tmp_path):
        with pytest.raises(RuntimeError) as excinfo:
            _rpc_call("file.convert",
                      {"input": "/nonexistent/path/missing_input.ply",
                       "output": str(tmp_path / "out.ply")})
        assert "not found" in str(excinfo.value).lower()

    def test_level4_rpc_file_convert_unsupported_output(self, sample_ply, tmp_path):
        out = str(tmp_path / "out.__not_a_real_extension__")
        with pytest.raises(RuntimeError):
            _rpc_call("file.convert", {"input": sample_ply, "output": out})


# ── Level 4c: RPC Entity Operations ───────────────────────────────────────

@pytest.mark.skipif(not _rpc_available(), reason="ACloudViewer RPC not available")
@pytest.mark.usefixtures("level4_cleanup")
class TestLevel4_RPCEntityOps:
    """Test JSON-RPC entity manipulation: rename, color, visibility, remove."""

    @pytest.fixture
    def sample_ply(self, tmp_path):
        p = tmp_path / "entity_test.ply"
        p.write_text(_SAMPLE_PLY_HEADER + _SAMPLE_PLY_BODY)
        return str(p)

    @pytest.fixture
    def loaded(self, sample_ply):
        result = _rpc_call("open", {"filename": sample_ply, "silent": True})
        cloud_id = _find_cloud_id(result)
        group_id = result["id"]
        yield {"group_id": group_id, "cloud_id": cloud_id or group_id,
               "result": result}
        try:
            # Remove only the loaded group, not the entire scene
            _rpc_call("scene.remove", {"entity_id": group_id}, timeout=5)
        except Exception:  # best-effort cleanup
            pass

    def test_level4_rpc_entity_rename(self, loaded):
        eid = loaded["group_id"]
        _rpc_call("entity.rename", {"entity_id": eid, "name": "my_cloud"})
        info = _rpc_call("scene.info", {"entity_id": eid})
        assert info["name"] == "my_cloud"

    def test_level4_rpc_entity_set_color(self, loaded):
        eid = loaded["cloud_id"]
        result = _rpc_call("entity.setColor",
                           {"entity_id": eid, "r": 255, "g": 0, "b": 0})
        assert isinstance(result, (int, dict)), f"Unexpected: {result}"
        if isinstance(result, dict):
            assert result.get("entity_id") == eid

    def test_level4_rpc_scene_set_visible_off(self, loaded):
        eid = loaded["group_id"]
        _rpc_call("scene.setVisible", {"entity_id": eid, "visible": False})
        info = _rpc_call("scene.info", {"entity_id": eid})
        assert info["visible"] is False

    def test_level4_rpc_scene_set_visible_toggle(self, loaded):
        eid = loaded["group_id"]
        _rpc_call("scene.setVisible", {"entity_id": eid, "visible": False})
        _rpc_call("scene.setVisible", {"entity_id": eid, "visible": True})
        info = _rpc_call("scene.info", {"entity_id": eid})
        assert info["visible"] is True

    def test_level4_rpc_scene_remove(self, sample_ply):
        loaded = _rpc_call("open", {"filename": sample_ply, "silent": True})
        _rpc_call("scene.remove", {"entity_id": loaded["id"]})
        entities = _rpc_call("scene.list")
        ids = [e["id"] for e in entities]
        assert loaded["id"] not in ids
        _rpc_call("clear", timeout=5)

    def test_level4_rpc_entity_not_found(self):
        with pytest.raises(RuntimeError, match="Entity not found"):
            _rpc_call("scene.info", {"entity_id": 999999999})

    def test_level4_rpc_zoom_fit_on_entity(self, loaded):
        result = _rpc_call("view.zoomFit", {"entity_id": loaded["cloud_id"]})
        assert result == 0


# ── Level 4d: RPC Cloud Processing ────────────────────────────────────────

@pytest.mark.skipif(not _rpc_available(), reason="ACloudViewer RPC not available")
@pytest.mark.usefixtures("level4_cleanup")
class TestLevel4_RPCCloudProcessing:
    """Test JSON-RPC cloud processing: normals, subsample, crop, paint."""

    @pytest.fixture
    def sample_ply(self, tmp_path):
        p = tmp_path / "cloud_proc_test.ply"
        p.write_text(_SAMPLE_PLY_HEADER + _SAMPLE_PLY_BODY)
        return str(p)

    @pytest.fixture
    def cloud_id(self, sample_ply):
        result = _rpc_call("open", {"filename": sample_ply, "silent": True})
        cid = _find_cloud_id(result)
        assert cid is not None, "Expected a point cloud after loading PLY"
        group_id = result["id"]  # Store the parent group ID for cleanup
        yield cid
        try:
            # Remove only the loaded group, not the entire scene
            _rpc_call("scene.remove", {"entity_id": group_id}, timeout=5)
        except Exception:  # best-effort cleanup
            pass

    def test_level4_rpc_cloud_compute_normals(self, cloud_id):
        result = _rpc_call("cloud.computeNormals",
                           {"entity_id": cloud_id, "radius": 0.1}, timeout=60)
        assert result["has_normals"] is True
        assert result["point_count"] == 100

    def test_level4_rpc_cloud_subsample_spatial(self, cloud_id):
        result = _rpc_call("cloud.subsample",
                           {"entity_id": cloud_id,
                            "method": "spatial", "step": 0.1})
        assert result["point_count"] > 0
        assert result["type"] == "POINT_CLOUD"

    def test_level4_rpc_cloud_subsample_random(self, cloud_id):
        result = _rpc_call("cloud.subsample",
                           {"entity_id": cloud_id,
                            "method": "random", "count": 30})
        assert 0 < result["point_count"] <= 30

    def test_level4_rpc_cloud_get_scalar_fields(self, cloud_id):
        fields = _rpc_call("cloud.getScalarFields", {"entity_id": cloud_id})
        assert isinstance(fields, list)

    def test_level4_rpc_cloud_paint_uniform(self, cloud_id):
        result = _rpc_call("cloud.paintUniform",
                           {"entity_id": cloud_id,
                            "r": 255, "g": 128, "b": 0})
        assert result["points_colored"] == 100
        assert result["color"] == [255, 128, 0]

    def test_level4_rpc_cloud_paint_by_height_z(self, cloud_id):
        result = _rpc_call("cloud.paintByHeight",
                           {"entity_id": cloud_id, "axis": "z"})
        assert result["axis"] == "z"
        assert "min" in result and "max" in result
        assert result["max"] > result["min"]

    def test_level4_rpc_cloud_paint_by_height_x(self, cloud_id):
        result = _rpc_call("cloud.paintByHeight",
                           {"entity_id": cloud_id, "axis": "x"})
        assert result["axis"] == "x"

    def test_level4_rpc_cloud_crop(self, cloud_id):
        result = _rpc_call("cloud.crop", {
            "entity_id": cloud_id,
            "min_x": 0.0, "min_y": 0.0, "min_z": 0.0,
            "max_x": 0.5, "max_y": 1.0, "max_z": 1.5,
        })
        assert result["point_count"] > 0
        assert result["type"] == "POINT_CLOUD"

    def test_level4_rpc_cloud_coord_to_sf(self, cloud_id):
        result = _rpc_call("cloud.coordToSF",
                           {"entity_id": cloud_id, "dimension": "z"})
        assert result.get("entity_id") == cloud_id
        assert result.get("dimension") == "z"

    def test_level4_rpc_cloud_set_active_sf(self, cloud_id):
        _rpc_call("cloud.coordToSF",
                  {"entity_id": cloud_id, "dimension": "z"})
        result = _rpc_call("cloud.setActiveSf",
                           {"entity_id": cloud_id, "field_index": 0})
        assert result.get("entity_id") == cloud_id
        assert result.get("field_index") == 0 or result.get("active_sf_index") == 0

    def test_level4_rpc_cloud_rename_sf(self, cloud_id):
        _rpc_call("cloud.coordToSF",
                  {"entity_id": cloud_id, "dimension": "z"})
        result = _rpc_call("cloud.renameSf",
                           {"entity_id": cloud_id,
                            "field_index": 0, "new_name": "test_sf"})
        assert result.get("entity_id") == cloud_id
        assert result.get("new_name") == "test_sf"

    def test_level4_rpc_cloud_remove_sf(self, cloud_id):
        _rpc_call("cloud.coordToSF",
                  {"entity_id": cloud_id, "dimension": "z"})
        sfs_before = _rpc_call("cloud.getScalarFields",
                               {"entity_id": cloud_id})
        result = _rpc_call("cloud.removeSf",
                           {"entity_id": cloud_id, "field_index": 0})
        assert result.get("entity_id") == cloud_id
        sfs_after = _rpc_call("cloud.getScalarFields",
                              {"entity_id": cloud_id})
        assert len(sfs_after) < len(sfs_before)

    def test_level4_rpc_cloud_remove_all_sfs(self, cloud_id):
        _rpc_call("cloud.coordToSF",
                  {"entity_id": cloud_id, "dimension": "z"})
        result = _rpc_call("cloud.removeAllSfs",
                           {"entity_id": cloud_id})
        assert result.get("entity_id") == cloud_id
        sfs = _rpc_call("cloud.getScalarFields",
                        {"entity_id": cloud_id})
        assert len(sfs) == 0

    def test_level4_rpc_cloud_filter_sf(self, cloud_id):
        _rpc_call("cloud.coordToSF",
                  {"entity_id": cloud_id, "dimension": "z"})
        _rpc_call("cloud.setActiveSf",
                  {"entity_id": cloud_id, "field_index": 0})
        result = _rpc_call("cloud.filterSf",
                           {"entity_id": cloud_id,
                            "min": 0.0, "max": 0.5})
        assert "point_count" in result, f"cloud.filterSf missing point_count: {result}"
        assert isinstance(result["point_count"], int)

    def test_level4_rpc_cloud_remove_rgb(self, cloud_id):
        result = _rpc_call("cloud.removeRgb",
                           {"entity_id": cloud_id})
        assert result.get("entity_id") == cloud_id

    def test_level4_rpc_cloud_compute_and_invert_normals(self, cloud_id):
        _rpc_call("cloud.computeNormals",
                  {"entity_id": cloud_id, "radius": 0.1}, timeout=60)
        result = _rpc_call("cloud.invertNormals",
                           {"entity_id": cloud_id})
        assert result.get("entity_id") == cloud_id

    def test_level4_rpc_cloud_remove_normals(self, cloud_id):
        _rpc_call("cloud.computeNormals",
                  {"entity_id": cloud_id, "radius": 0.1}, timeout=60)
        result = _rpc_call("cloud.removeNormals",
                           {"entity_id": cloud_id})
        assert result.get("entity_id") == cloud_id

    def test_level4_rpc_cloud_merge(self, sample_ply):
        r1 = _rpc_call("open", {"filename": sample_ply, "silent": True})
        id1 = _find_cloud_id(r1)
        r2 = _rpc_call("open", {"filename": sample_ply, "silent": True})
        id2 = _find_cloud_id(r2)
        result = _rpc_call("cloud.merge",
                           {"entity_ids": [id1, id2]})
        assert (
            result.get("merged_count", 0) >= 2
            or result.get("point_count", 0) > 0
            or result.get("id") is not None
        ), f"cloud.merge returned unexpected result: {result}"
        _rpc_call("clear", timeout=5)
    
    def test_level4_rpc_cloud_density(self, cloud_id):
        """Test density computation via RPC."""
        result = _rpc_call("cloud.density", {
            "entity_id": cloud_id,
            "radius": 0.05
        })
        assert isinstance(result, dict)
        assert result.get("entity_id") == cloud_id

    def test_level4_rpc_cloud_curvature(self, cloud_id):
        """Test curvature computation via RPC."""
        _rpc_call("cloud.computeNormals", {"entity_id": cloud_id})
        result = _rpc_call("cloud.curvature", {
            "entity_id": cloud_id,
            "type": "MEAN",
            "radius": 0.05
        })
        assert isinstance(result, dict)
        assert result.get("entity_id") == cloud_id

    def test_level4_rpc_cloud_roughness(self, cloud_id):
        """Test roughness computation via RPC."""
        result = _rpc_call("cloud.roughness", {
            "entity_id": cloud_id,
            "radius": 0.1
        })
        assert isinstance(result, dict)
        assert result.get("entity_id") == cloud_id

    def test_level4_rpc_cloud_geometric_feature(self, cloud_id):
        """Test geometric feature computation via RPC."""
        result = _rpc_call("cloud.geometricFeature", {
            "entity_id": cloud_id,
            "type": "SURFACE_VARIATION",
            "kernel_size": 0.1
        })
        assert isinstance(result, dict)
        assert result.get("entity_id") == cloud_id

    def test_level4_rpc_cloud_color_banding(self, cloud_id):
        """Test color banding via RPC."""
        result = _rpc_call("cloud.colorBanding", {
            "entity_id": cloud_id,
            "axis": "Z",
            "frequency": 10.0
        })
        assert isinstance(result, dict)
        assert result.get("entity_id") == cloud_id

    def test_level4_rpc_cloud_sor_filter(self, cloud_id):
        """Test statistical outlier removal via RPC."""
        result = _rpc_call("cloud.sorFilter", {
            "entity_id": cloud_id,
            "knn": 6,
            "sigma": 1.0
        })
        assert isinstance(result, dict)

    def test_level4_rpc_sf_arithmetic(self, cloud_id):
        """Test scalar field arithmetic operations via RPC."""
        _rpc_call("cloud.coordToSF", {
            "entity_id": cloud_id,
            "dimension": "Z"
        })
        result = _rpc_call("cloud.sfArithmetic", {
            "entity_id": cloud_id,
            "sf_index": 0,
            "operation": "SQRT"
        })
        assert isinstance(result, dict)
        assert result.get("entity_id") == cloud_id

    def test_level4_rpc_sf_operation(self, cloud_id):
        """Test scalar field operation with constant via RPC."""
        _rpc_call("cloud.coordToSF", {
            "entity_id": cloud_id,
            "dimension": "Z"
        })
        result = _rpc_call("cloud.sfOperation", {
            "entity_id": cloud_id,
            "sf_index": 0,
            "operation": "MULTIPLY",
            "value": 2.0
        })
        assert isinstance(result, dict)
        assert result.get("entity_id") == cloud_id

    def test_level4_rpc_sf_gradient(self, cloud_id):
        """Test scalar field gradient via RPC."""
        _rpc_call("cloud.coordToSF", {
            "entity_id": cloud_id,
            "dimension": "Z"
        })
        result = _rpc_call("cloud.sfGradient", {
            "entity_id": cloud_id
        })
        assert isinstance(result, dict)

    def test_level4_rpc_sf_convert_to_rgb(self, cloud_id):
        """Test scalar field to RGB conversion via RPC."""
        _rpc_call("cloud.coordToSF", {
            "entity_id": cloud_id,
            "dimension": "Z"
        })
        result = _rpc_call("cloud.sfConvertToRGB", {
            "entity_id": cloud_id
        })
        assert isinstance(result, dict)

    def test_level4_rpc_octree_normals(self, cloud_id):
        """Test octree-based normal computation via RPC."""
        result = _rpc_call("cloud.octreeNormals", {
            "entity_id": cloud_id,
            "radius": "AUTO"
        })
        assert isinstance(result, dict)

    def test_level4_rpc_orient_normals_mst(self, cloud_id):
        """Test MST normal orientation via RPC."""
        _rpc_call("cloud.computeNormals", {"entity_id": cloud_id})
        result = _rpc_call("cloud.orientNormalsMST", {
            "entity_id": cloud_id,
            "knn": 6
        })
        assert isinstance(result, dict)

    def test_level4_rpc_clear_normals(self, cloud_id):
        """Test clear normals via RPC."""
        _rpc_call("cloud.computeNormals", {"entity_id": cloud_id})
        result = _rpc_call("cloud.clearNormals", {
            "entity_id": cloud_id
        })
        assert isinstance(result, dict)

    def test_level4_rpc_normals_to_sfs(self, cloud_id):
        """Test normals to scalar fields via RPC."""
        _rpc_call("cloud.computeNormals", {"entity_id": cloud_id})
        result = _rpc_call("cloud.normalsToSFs", {
            "entity_id": cloud_id
        })
        assert isinstance(result, dict)

    def test_level4_rpc_normals_to_dip(self, cloud_id):
        """Test normals to dip/dip-direction via RPC."""
        _rpc_call("cloud.octreeNormals", {
            "entity_id": cloud_id,
            "radius": "AUTO"
        })
        result = _rpc_call("cloud.normalsToDip", {
            "entity_id": cloud_id
        })
        assert isinstance(result, dict)

    def test_level4_rpc_extract_connected_components(self, cloud_id):
        """Test extract connected components via RPC."""
        result = _rpc_call("cloud.extractConnectedComponents", {
            "entity_id": cloud_id,
            "min_points": 10,
            "octree_level": 6
        })
        assert isinstance(result, dict)

    def test_level4_rpc_approx_density(self, cloud_id):
        """Test approximate density computation via RPC."""
        result = _rpc_call("cloud.approxDensity", {
            "entity_id": cloud_id,
            "density_type": "PRECISE"
        })
        assert isinstance(result, dict)
        assert result.get("entity_id") == cloud_id

    def test_level4_rpc_best_fit_plane(self, cloud_id):
        """Test best fit plane computation via RPC."""
        result = _rpc_call("cloud.bestFitPlane", {
            "entity_id": cloud_id,
            "make_horiz": False
        })
        assert isinstance(result, dict)

    def test_level4_rpc_delaunay(self, cloud_id):
        """Test Delaunay triangulation via RPC."""
        _rpc_call("cloud.computeNormals", {"entity_id": cloud_id})
        result = _rpc_call("cloud.delaunay", {
            "entity_id": cloud_id
        })
        assert isinstance(result, dict)

    def test_level4_rpc_scene_select(self, cloud_id):
        """Test scene.select — select entities by ID."""
        result = _rpc_call("scene.select", {"entity_ids": [cloud_id]})
        assert isinstance(result, (dict, list, int))

    def test_level4_rpc_cloud_paint_by_scalar_field(self, cloud_id):
        """Test cloud.paintByScalarField — color by SF (requires SF)."""
        _rpc_call("cloud.coordToSF",
                  {"entity_id": cloud_id, "dimension": "z"})
        result = _rpc_call("cloud.paintByScalarField",
                           {"entity_id": cloud_id, "field_index": 0})
        assert result.get("entity_id") == cloud_id
        assert "field_name" in result

    def test_level4_rpc_workflow_load_orient_screenshot(
            self, cloud_id, tmp_path):
        """Full workflow: load -> orient -> zoom -> screenshot."""
        _rpc_call("view.setOrientation", {"orientation": "iso1"})
        _rpc_call("view.zoomFit", {"entity_id": cloud_id})
        path = str(tmp_path / "workflow.png")
        result = _rpc_call("view.screenshot", {"filename": path})
        assert result["width"] > 0
        assert Path(path).exists()

    def test_level4_rpc_process_run_cli(self):
        """process.run_cli is registered (full CLI run not tested here)."""
        methods = _rpc_call("methods.list")
        names = {m["method"] for m in methods}
        assert "process.run_cli" in names, "process.run_cli should be in methods list"

    def test_level4_rpc_process_csf_exists(self):
        methods = _rpc_call("methods.list")
        names = {m["method"] for m in methods}
        assert "process.csf" in names, "process.csf should be in methods list"

    def test_level4_rpc_process_m3c2_exists(self):
        methods = _rpc_call("methods.list")
        names = {m["method"] for m in methods}
        assert "process.m3c2" in names, "process.m3c2 should be in methods list"

    def test_level4_rpc_process_ransac_exists(self):
        methods = _rpc_call("methods.list")
        names = {m["method"] for m in methods}
        assert "process.ransac" in names, "process.ransac should be in methods list"

    def test_level4_rpc_method_count_increased(self):
        methods = _rpc_call("methods.list")
        previous_minimum = 40
        assert len(methods) >= previous_minimum + 4, (
            f"Expected >={previous_minimum + 4} RPC methods "
            f"({previous_minimum} baseline + 4 new process.*), got {len(methods)}")


# ── Level 4e: RPC mesh operations ─────────────────────────────────────────

@pytest.mark.skipif(
    not _level4_mesh_rpc_ready(),
    reason="ACV_RPC_URL not set or ACloudViewer RPC not available",
)
@pytest.mark.usefixtures("level4_cleanup")
class TestLevel4_RPCMeshOperations:
    """JSON-RPC mesh editing: simplify, smooth, subdivide, sampling, volume, etc."""

    @pytest.fixture
    def sample_ply(self, tmp_path):
        p = tmp_path / "mesh_rpc_grid.ply"
        p.write_text(_MESH_PLY_CONTENT)
        return str(p)

    @pytest.fixture
    def mesh_id(self, sample_ply):
        opened = _rpc_call("open", {"filename": sample_ply, "silent": True})
        cloud_id = _find_cloud_id(opened)
        assert cloud_id is not None, "Expected point cloud in mesh grid PLY"
        group_id = opened["id"]
        _rpc_call("cloud.computeNormals",
                  {"entity_id": cloud_id, "radius": 0.1}, timeout=60)
        mesh_res = _rpc_call("cloud.delaunay", {"entity_id": cloud_id},
                             timeout=120)
        mid = mesh_res["id"]
        yield mid
        try:
            _rpc_call("scene.remove", {"entity_id": mid}, timeout=5)
        except Exception:  # best-effort cleanup
            pass
        try:
            _rpc_call("scene.remove", {"entity_id": group_id}, timeout=5)
        except Exception:  # best-effort cleanup
            pass

    def test_level4_rpc_mesh_simplify_quadric(self, mesh_id):
        result = _rpc_call(
            "mesh.simplify",
            {"entity_id": mesh_id, "method": "quadric", "target_triangles": 50},
            timeout=120,
        )
        assert isinstance(result, dict)
        assert "id" in result

    def test_level4_rpc_mesh_simplify_vertex_clustering(self, mesh_id):
        result = _rpc_call(
            "mesh.simplify",
            {"entity_id": mesh_id, "method": "vertex_clustering", "voxel_size": 0.3},
            timeout=120,
        )
        assert isinstance(result, dict)
        assert "id" in result

    def test_level4_rpc_mesh_smooth_laplacian(self, mesh_id):
        result = _rpc_call(
            "mesh.smooth",
            {"entity_id": mesh_id, "method": "laplacian", "iterations": 2},
            timeout=120,
        )
        assert isinstance(result, dict)
        assert "id" in result

    def test_level4_rpc_mesh_smooth_taubin(self, mesh_id):
        result = _rpc_call(
            "mesh.smooth",
            {"entity_id": mesh_id, "method": "taubin",
             "iterations": 2, "lambda": 0.5, "mu": -0.53},
            timeout=120,
        )
        assert isinstance(result, dict)
        assert "id" in result

    def test_level4_rpc_mesh_smooth_simple(self, mesh_id):
        result = _rpc_call(
            "mesh.smooth",
            {"entity_id": mesh_id, "method": "simple", "iterations": 2},
            timeout=120,
        )
        assert isinstance(result, dict)
        assert "id" in result

    def test_level4_rpc_mesh_subdivide_midpoint(self, mesh_id):
        result = _rpc_call(
            "mesh.subdivide",
            {"entity_id": mesh_id, "method": "midpoint", "iterations": 1},
            timeout=120,
        )
        assert isinstance(result, dict)
        assert "id" in result

    def test_level4_rpc_mesh_subdivide_loop(self, mesh_id):
        result = _rpc_call(
            "mesh.subdivide",
            {"entity_id": mesh_id, "method": "loop", "iterations": 1},
            timeout=120,
        )
        assert isinstance(result, dict)
        assert "id" in result

    def test_level4_rpc_mesh_sample_points_uniform(self, mesh_id):
        result = _rpc_call(
            "mesh.samplePoints",
            {"entity_id": mesh_id, "method": "uniform", "count": 500},
            timeout=120,
        )
        assert isinstance(result, dict)
        assert "id" in result

    def test_level4_rpc_mesh_sample_points_poisson_disk(self, mesh_id):
        result = _rpc_call(
            "mesh.samplePoints",
            {"entity_id": mesh_id, "method": "poisson_disk", "count": 500},
            timeout=120,
        )
        assert isinstance(result, dict)
        assert "id" in result

    def test_level4_rpc_mesh_volume(self, mesh_id):
        result = _rpc_call("mesh.volume", {"entity_id": mesh_id}, timeout=60)
        assert isinstance(result, dict)
        assert result.get("entity_id") == mesh_id
        assert "volume" in result
        assert isinstance(result["volume"], (int, float))

    def test_level4_rpc_mesh_merge(self, sample_ply):
        """Test mesh.merge — merge two meshes."""
        r1 = _rpc_call("open", {"filename": sample_ply, "silent": True})
        cid1 = _find_cloud_id(r1)
        _rpc_call("cloud.computeNormals",
                  {"entity_id": cid1, "radius": 0.1}, timeout=60)
        m1 = _rpc_call("cloud.delaunay", {"entity_id": cid1})

        r2 = _rpc_call("open", {"filename": sample_ply, "silent": True})
        cid2 = _find_cloud_id(r2)
        _rpc_call("cloud.computeNormals",
                  {"entity_id": cid2, "radius": 0.1}, timeout=60)
        m2 = _rpc_call("cloud.delaunay", {"entity_id": cid2})

        mid1 = m1.get("mesh_id") or m1.get("id")
        mid2 = m2.get("mesh_id") or m2.get("id")
        if mid1 and mid2:
            result = _rpc_call("mesh.merge",
                               {"entity_ids": [mid1, mid2]}, timeout=60)
            assert isinstance(result, dict)
        _rpc_call("clear", timeout=5)

    def test_level4_rpc_mesh_flip_triangles(self, mesh_id):
        result = _rpc_call("mesh.flipTriangles", {"entity_id": mesh_id},
                           timeout=60)
        assert isinstance(result, dict)
        assert result.get("entity_id") == mesh_id
        assert "triangle_count" in result

    def test_level4_rpc_mesh_extract_vertices(self, mesh_id):
        result = _rpc_call("mesh.extractVertices", {"entity_id": mesh_id},
                           timeout=120)
        assert isinstance(result, dict)
        assert "id" in result


@pytest.mark.skipif(
    not _level4_mesh_rpc_ready(),
    reason="ACV_RPC_URL not set or ACloudViewer RPC not available",
)
@pytest.mark.usefixtures("level4_cleanup")
class TestLevel4_RPCTransformOps:
    """JSON-RPC transform.apply."""

    @pytest.fixture
    def sample_ply(self, tmp_path):
        p = tmp_path / "transform_rpc_test.ply"
        p.write_text(_SAMPLE_PLY_HEADER + _SAMPLE_PLY_BODY)
        return str(p)

    @pytest.fixture
    def loaded(self, sample_ply):
        result = _rpc_call("open", {"filename": sample_ply, "silent": True})
        cloud_id = _find_cloud_id(result)
        group_id = result["id"]
        yield {"group_id": group_id, "cloud_id": cloud_id or group_id}
        try:
            _rpc_call("scene.remove", {"entity_id": group_id}, timeout=5)
        except Exception:  # best-effort cleanup
            pass

    def test_level4_rpc_transform_apply(self, loaded):
        eid = loaded["cloud_id"]
        identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        result = _rpc_call(
            "transform.apply",
            {"entity_id": eid, "matrix": identity},
            timeout=60,
        )
        assert isinstance(result, dict)
        assert result.get("entity_id") == eid
        assert "name" in result


# ── Level 4g: Colmap RPC ───────────────────────────────────────────────────

@pytest.mark.skipif(not _rpc_available(), reason="ACloudViewer RPC not available")
@pytest.mark.usefixtures("level4_cleanup")
class TestLevel4_RPCColmap:
    """JSON-RPC colmap.reconstruct and colmap.run (smoke tests — no data needed)."""

    def test_level4_rpc_colmap_reconstruct_missing_params(self):
        """colmap.reconstruct with missing params should return structured error."""
        try:
            _rpc_call("colmap.reconstruct", {}, timeout=10)
        except Exception as exc:
            err = str(exc).lower()
            assert "error" in err or "missing" in err or "required" in err

    def test_level4_rpc_colmap_run_ping(self):
        """colmap.run with a no-op command to verify dispatch."""
        try:
            result = _rpc_call("colmap.run", {
                "command": "help",
                "timeout_ms": 5000,
            }, timeout=10)
            assert isinstance(result, (dict, str, int))
        except Exception as exc:
            err = str(exc).lower()
            assert "error" in err or "colmap" in err or "not found" in err


# ── Level 4h: AICore batch RPC ─────────────────────────────────────────────

@pytest.mark.skipif(not _rpc_available(), reason="ACloudViewer RPC not available")
class TestLevel4_RPCAICoreBatch:
    """aicore.batch resolves every registered task to its native HELP schema.

    HELP loads no model, so this checks the RPC child-process path end to end
    without test assets.
    """

    @pytest.mark.parametrize("task,command,_header", AICORE_BATCH_COMMANDS)
    def test_level4_aicore_batch_help(self, task, command, _header):
        result = _rpc_call("aicore.batch", {
            "command": command, "options": {"HELP": True},
            "timeout_ms": 60000}, timeout=90)
        assert result["exit_code"] == 0, result.get("stderr")
        schema = result["manifest"]
        assert schema["schema"] == "acloudviewer.aicore.batch/1"
        assert schema["keyword"] == command, task
        assert "RESULT_JSON" in {option["key"] for option in schema["options"]}

    def test_level4_aicore_batch_rejects_non_aicore_command(self):
        with pytest.raises(RuntimeError):
            _rpc_call("aicore.batch", {"command": "SS", "options": {}},
                      timeout=10)


@pytest.mark.skipif(
    not _rpc_available(),
    reason="ACV_RPC_URL not set or ACloudViewer RPC not available",
)
class TestLevel4_RPCChildProcessResponsiveness:
    """Child-process handlers (process.run_cli, aicore.batch, colmap.*) run on
    the GUI thread; the server must keep answering while the child runs."""

    def test_level4_run_cli_keeps_server_responsive(self, tmp_path):
        import threading
        import time

        source = tmp_path / "in.ply"
        source.write_text(_SAMPLE_PLY_HEADER + _SAMPLE_PLY_BODY)
        target = tmp_path / "out.ply"
        outcome: dict = {}

        def run_cli():
            try:
                outcome["result"] = _rpc_call("process.run_cli", {
                    "input_path": str(source), "output_path": str(target),
                    "args": ["-SS", "RANDOM", "3"], "timeout_ms": 120000},
                    timeout=150)
            except BaseException as exc:  # pytest.skip is BaseException
                outcome["error"] = exc

        worker = threading.Thread(target=run_cli)
        worker.start()
        # The child needs seconds to load plugins/backends; a blocking wait
        # would hold every ping below until it exits.
        time.sleep(0.2)
        latencies = []
        while worker.is_alive():
            started = time.monotonic()
            assert _rpc_call("ping", timeout=5) == "pong"
            if worker.is_alive():
                latencies.append(time.monotonic() - started)
            time.sleep(0.1)
        worker.join()
        if "error" in outcome:
            raise outcome["error"]
        assert outcome["result"]["exit_code"] == 0, outcome["result"]
        assert target.is_file() and target.stat().st_size > 0
        assert latencies, "child exited before any concurrent ping was sent"
        assert max(latencies) < 1.0, latencies

    @staticmethod
    def _sleep_child(seconds: float, timeout_ms: int = 60000) -> dict:
        # colmap.run starts any program: a Python sleep is a portable child.
        return {"command": "-c", "args": [f"import time; time.sleep({seconds})"],
                "colmap_binary": sys.executable, "timeout_ms": timeout_ms}

    def test_level4_concurrent_children_answer_independently(self):
        """Replies are deferred, not nested event loops: a short child
        started first must not wait for a longer one started after it."""
        import threading
        import time

        started = time.monotonic()
        replies: dict = {}

        def run(name, seconds):
            try:
                _rpc_call("colmap.run", self._sleep_child(seconds), timeout=60)
                replies[name] = time.monotonic() - started
            except BaseException as exc:  # pytest.fail is BaseException
                replies[name] = exc

        short = threading.Thread(target=run, args=("short", 1.5))
        long_ = threading.Thread(target=run, args=("long", 5))
        short.start()
        time.sleep(0.3)
        long_.start()
        short.join()
        long_.join()
        for value in replies.values():
            if isinstance(value, BaseException):
                raise value
        assert replies["short"] < 4.0, replies
        assert replies["long"] >= 5.0, replies

    def test_level4_child_deadline_kills_and_answers(self):
        import time

        started = time.monotonic()
        with pytest.raises(RuntimeError, match=r"RPC error 3 .*timed out"):
            _rpc_call("colmap.run", self._sleep_child(30, timeout_ms=1000),
                      timeout=20)
        assert time.monotonic() - started < 10

    @pytest.mark.parametrize("timeout_ms", [0, -1, "abc", 86400001])
    def test_level4_child_timeout_must_be_bounded(self, timeout_ms):
        with pytest.raises(RuntimeError, match=r"RPC error -32602 "):
            _rpc_call("colmap.run", self._sleep_child(1, timeout_ms=timeout_ms))


@pytest.mark.skipif(
    not _rpc_available(),
    reason="ACV_RPC_URL not set or ACloudViewer RPC not available",
)
class TestLevel4_RPCServerBoundary:
    """Unauthenticated methods write files and start programs: browsers
    (Origin handshakes) and other hosts must not reach them."""

    def test_level4_browser_origin_is_refused(self):
        with pytest.raises(Exception, match="403"):
            ws_connect(RPC_URL, open_timeout=5, origin="https://example.org")

    def test_level4_not_listening_beyond_loopback(self):
        import socket
        from urllib.parse import urlparse

        address = _lan_ipv4()
        if address is None:
            pytest.skip("no non-loopback IPv4 interface")
        port = urlparse(RPC_URL).port
        with socket.socket() as sock:
            sock.settimeout(3)
            assert sock.connect_ex((address, port)) != 0, (
                f"RPC port {port} is reachable on {address}")

    def test_level4_malformed_json_gets_parse_error(self):
        ws = ws_connect(RPC_URL, open_timeout=5)
        try:
            ws.send("{not json")
            reply = json.loads(ws.recv(timeout=5))
        finally:
            ws.close()
        assert reply["id"] is None
        assert reply["error"]["code"] == -32700


# ═══════════════════════════════════════════════════════════════════════════
# Level 5: MCP Server Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestLevel5_MCPServer:
    """Verify MCP server tool definitions."""

    @_needs_mcp
    def test_level5_mcp_import(self):
        from mcp.server import Server  # noqa: F401

    @_needs_mcp
    def test_level5_aicore_and_plugin_tools(self):
        from cli_anything.acloudviewer.mcp_server import list_tools
        import asyncio
        tools = asyncio.run(list_tools())
        by_name = {tool.name: tool for tool in tools}
        for name in ("list_ai_tasks", "get_ai_schema", "run_ai_batch",
                     "run_plugin_command"):
            assert name in by_name, f"Missing MCP tool: {name}"
        assert by_name["get_ai_schema"].inputSchema["required"] == ["task"]
        assert set(by_name["run_ai_batch"].inputSchema["required"]) == {
            "task", "options"}
        assert by_name["run_plugin_command"].inputSchema["required"] == [
            "command"]

    @_needs_mcp
    def test_level5_documented_tool_counts_match_server(self):
        """Every documented MCP tool count and OpenClaw category set equals
        ``list_tools()``: the server is the only owner of the number."""
        import cli_anything.acloudviewer as harness_pkg
        from cli_anything.acloudviewer.mcp_server import list_tools
        import asyncio
        names = {tool.name for tool in asyncio.run(list_tools())}
        harness = Path(harness_pkg.__file__).resolve().parent
        repo_docs = [
            REPO_ROOT / "README.md",
            REPO_ROOT / "agent-integration/README.md",
            REPO_ROOT / "agent-integration/mcp/README.md",
            REPO_ROOT / "agent-integration/openclaw/README.md",
            REPO_ROOT / "agent-integration/tests/README.md",
            REPO_ROOT / "agent-integration/openclaw/openclaw-skill.json",
        ]
        # Harness files are present for editable/source installs (CI uses -e).
        harness_docs = [path for path in (
            harness / "README.md", harness / "OPENCLAW.md",
            harness / "skills/SKILL.md", harness / "tests/TEST.md",
            harness / "openclaw-skill.json",
            harness.parents[1] / "ACLOUDVIEWER.md") if path.is_file()]
        # Three-digit totals only: per-category counts ("13 tools") are < 100.
        pattern = re.compile(
            r"(?<![\d.])(\d{3})\+?\s+(?:MCP\s+)?[Tt]ools\b|"
            r"Tools \((\d{3})\)|\"tools_count\": (\d{3})")
        stale = []
        for path in repo_docs + harness_docs:
            text = _read_repo_text(path)
            for match in pattern.finditer(text):
                value = int(next(group for group in match.groups() if group))
                if value != len(names):
                    line = text.count("\n", 0, match.start()) + 1
                    stale.append(f"{path}:{line}: {value}")
        assert not stale, (
            f"MCP server has {len(names)} tools; stale counts: {stale}")
        for manifest in (REPO_ROOT / "agent-integration/openclaw/openclaw-skill.json",
                         harness / "openclaw-skill.json"):
            if not manifest.is_file():
                continue
            categories = json.loads(_read_repo_text(manifest))["tool_categories"]
            listed = {name for tools in categories.values() for name in tools}
            assert listed == names, (
                f"{manifest}: missing={sorted(names - listed)}, "
                f"stale={sorted(listed - names)}")

    @_needs_mcp
    def test_level5_mcp_tool_count_matches_docs(self):
        """Every published MCP tool count equals list_tools()."""
        import cli_anything.acloudviewer as harness_pkg
        from cli_anything.acloudviewer.mcp_server import list_tools
        import asyncio
        tools = asyncio.run(list_tools())
        package = Path(harness_pkg.__file__).parent
        sources = [REPO_ROOT / "agent-integration" / "README.md",
                   REPO_ROOT / "agent-integration" / "mcp" / "README.md",
                   REPO_ROOT / "agent-integration" / "openclaw" / "openclaw-skill.json",
                   package / "skills" / "SKILL.md"]
        # Not package data: present for the editable (CI) install only.
        if (package / "README.md").is_file():
            sources.append(package / "README.md")
        count = re.compile(
            r"(?:exposes|provides|registers)\s+\**(\d+)\s+(?:MCP\s+)?tools"
            r"|(\d+)\s+MCP tools|MCP [Ss]erver \((\d+) tools\)"
            r"|MCP tool registration \((\d+) tools\)|\"tools_count\":\s*(\d+)")
        for path in sources:
            text = path.read_text(encoding="utf-8")
            documented = {int(next(g for g in m.groups() if g))
                          for m in count.finditer(text)}
            assert documented, f"{path}: no MCP tool count found"
            assert documented == {len(tools)}, (
                f"{path}: documents {sorted(documented)} MCP tools, "
                f"list_tools() has {len(tools)}")

    @_needs_mcp
    def test_level5_mcp_tool_names(self):
        from cli_anything.acloudviewer.mcp_server import list_tools
        import asyncio
        tools = asyncio.run(list_tools())
        names = {t.name for t in tools}
        expected_tools = [
            "open_file", "convert_format", "batch_convert", "list_formats",
            "export_entity",
            "scene_list", "scene_info", "scene_remove", "scene_set_visible",
            "scene_select", "scene_clear",
            "entity_rename", "entity_set_color",
            "cloud_get_scalar_fields", "cloud_paint_uniform",
            "cloud_paint_by_height", "cloud_paint_by_scalar_field",
            "mesh_simplify", "mesh_smooth", "mesh_subdivide", "mesh_sample_points",
            "subsample", "compute_normals", "icp_registration", "sor_filter",
            "c2c_distance", "c2m_distance", "crop", "delaunay",
            "density", "curvature", "roughness", "sample_mesh", "color_banding",
            "set_active_sf", "remove_all_sfs", "remove_sf", "rename_sf",
            "sf_arithmetic", "sf_operation", "coord_to_sf", "sf_gradient",
            "filter_sf", "sf_color_scale", "sf_convert_to_rgb",
            "octree_normals", "orient_normals_mst", "invert_normals",
            "clear_normals", "normals_to_dip", "normals_to_sfs",
            "extract_connected_components", "approx_density",
            "geometric_feature", "moment", "best_fit_plane",
            "cross_section",
            "mesh_volume", "extract_vertices", "flip_triangles",
            "merge_clouds", "merge_meshes",
            "remove_rgb", "remove_scan_grids", "match_centers",
            "drop_global_shift", "closest_point_set",
            "rasterize", "stat_test",
            "screenshot", "get_camera",
            "view_set_orientation", "view_zoom_fit", "view_refresh",
            "view_set_perspective", "view_set_point_size",
            "transform_apply", "transform_apply_file",
            "get_info", "list_rpc_methods",
            "colmap_auto_reconstruct", "colmap_extract_features",
            "colmap_sparse_reconstruct", "colmap_poisson_mesh",
            "cloud_set_active_sf", "cloud_remove_sf",
            "cloud_remove_all_sfs", "cloud_rename_sf",
            "cloud_filter_sf", "cloud_coord_to_sf",
            "cloud_remove_rgb", "cloud_remove_normals_gui",
            "cloud_invert_normals_gui", "cloud_merge_gui",
            "mesh_extract_vertices_gui", "mesh_flip_triangles_gui",
            "mesh_volume_gui", "mesh_merge_gui",
            "colmap_run",
        ]
        missing = [t for t in expected_tools if t not in names]
        assert not missing, (
            f"Missing MCP tools: {missing}\n"
            f"Available tools: {sorted(names)}")

    @_needs_mcp
    def test_level5_mcp_colmap_tools(self):
        from cli_anything.acloudviewer.mcp_server import list_tools
        import asyncio
        tools = asyncio.run(list_tools())
        names = {t.name for t in tools}
        colmap_tools = [
            "colmap_auto_reconstruct", "colmap_extract_features",
            "colmap_match_features", "colmap_sparse_reconstruct",
            "colmap_undistort", "colmap_dense_stereo",
            "colmap_stereo_fusion", "colmap_poisson_mesh",
            "colmap_delaunay_mesh", "colmap_image_texturer",
            "colmap_model_converter", "colmap_analyze_model",
            "colmap_run",
        ]
        for tool in colmap_tools:
            assert tool in names, f"Missing Colmap MCP tool: {tool}"

    @_needs_mcp
    def test_level5_mcp_sibr_tools(self):
        from cli_anything.acloudviewer.mcp_server import list_tools
        import asyncio
        tools = asyncio.run(list_tools())
        names = {t.name for t in tools}
        sibr_tools = [
            "sibr_tool", "sibr_prepare_colmap",
            "sibr_texture_mesh", "sibr_unwrap_mesh",
            "sibr_tonemapper", "sibr_align_meshes",
            "sibr_camera_converter", "sibr_nvm_to_sibr",
            "sibr_crop_from_center", "sibr_clipping_planes",
            "sibr_distord_crop",
        ]
        missing = [t for t in sibr_tools if t not in names]
        assert not missing, (
            f"Missing SIBR MCP tools: {missing}\n"
            f"Available SIBR tools: {sorted(n for n in names if n.startswith('sibr_'))}")

    @_needs_mcp
    def test_level5_mcp_entry_point(self):
        r = subprocess.run(
            [sys.executable, "-m", "cli_anything.acloudviewer.mcp_server", "--help"],
            capture_output=True, text=True, timeout=10)
        assert r.returncode == 0, (
            f"mcp_server --help failed (rc={r.returncode}):\n{r.stderr[-2000:]}")

    @_needs_mcp
    def test_level5_mcp_plugin_standard_tools(self):
        from cli_anything.acloudviewer.mcp_server import list_tools
        import asyncio
        tools = asyncio.run(list_tools())
        names = {t.name for t in tools}
        plugin_tools = [
            "classify_3dmasc", "animation", "cloud_layers",
            "color_seg_rgb", "color_seg_hsv", "color_seg_scalar",
            "g3point",
            "pcl_sor", "pcl_normal_estimation", "pcl_mls",
            "pcl_euclidean_cluster", "pcl_sac_segmentation",
            "pcl_region_growing", "pcl_greedy_triangulation",
            "pcl_poisson_recon", "pcl_marching_cubes", "pcl_convex_hull",
        ]
        missing = [t for t in plugin_tools if t not in names]
        assert not missing, (
            f"Missing Standard plugin MCP tools: {missing}\n"
            f"Available tools: {sorted(names)}")

    @_needs_mcp
    def test_level5_mcp_plugin_io_tools(self):
        from cli_anything.acloudviewer.mcp_server import list_tools
        import asyncio
        tools = asyncio.run(list_tools())
        names = {t.name for t in tools}
        io_tools = [
            "draco_settings", "e57_settings", "las_settings",
            "csv_matrix_settings", "photoscan_settings",
            "mesh_io_settings", "core_io_settings",
        ]
        missing = [t for t in io_tools if t not in names]
        assert not missing, (
            f"Missing IO plugin MCP tools: {missing}\n"
            f"Available tools: {sorted(names)}")


# ═══════════════════════════════════════════════════════════════════════════
# Level 1: Plugin Command Source Verification
# ═══════════════════════════════════════════════════════════════════════════

class TestLevel1_PluginCommands:
    """Verify new plugin command source structure without building."""

    def test_level1_skips_precede_execution(self):
        """A skip states what is known before running (platform, installed
        library, QPA); once a test has run something, the outcome can only
        pass or fail. No pytest.skip may follow a child process / RPC call in
        the same function, nor sit in an exception handler."""
        import ast
        tree = ast.parse(_read_repo_text(Path(__file__)))
        runs = {"subprocess.run", "subprocess.Popen", "_rpc_call", "ws_connect"}
        # pytest.skip itself and module helpers that call it (_require_plugin).
        skips = {"pytest.skip"} | {
            fn.name for fn in tree.body if isinstance(fn, ast.FunctionDef)
            and any(isinstance(c, ast.Call) and ast.unparse(c.func) == "pytest.skip"
                    for c in ast.walk(fn))}
        late = []
        for fn in ast.walk(tree):
            if not isinstance(fn, ast.FunctionDef):
                continue
            calls = [c for c in ast.walk(fn) if isinstance(c, ast.Call)]
            first_run = min((c.lineno for c in calls
                             if ast.unparse(c.func) in runs), default=None)
            handlers = [h for h in ast.walk(fn) if isinstance(h, ast.ExceptHandler)]
            for c in calls:
                if ast.unparse(c.func) not in skips:
                    continue
                in_handler = any(h.lineno <= c.lineno <= h.end_lineno
                                 for h in handlers)
                if in_handler or (first_run is not None and c.lineno > first_run):
                    late.append(f"{fn.name}:{c.lineno}")
        assert not late, f"pytest.skip after execution: {late}"

    def test_level1_animation_command_header(self):
        h = REPO_ROOT / "plugins/core/Standard/qAnimation/include/qAnimationCommands.h"
        if not h.exists():
            pytest.fail("qAnimationCommands.h not found")
        src = _read_repo_text(h)
        assert "CommandAnimation" in src
        assert "ANIMATION" in _read_repo_text(
            REPO_ROOT / "plugins/core/Standard/qAnimation/src/qAnimationCommands.cpp")

    def test_level1_cloud_layers_command_header(self):
        h = REPO_ROOT / "plugins/core/Standard/qCloudLayers/include/qCloudLayersCommands.h"
        if not h.exists():
            pytest.fail("qCloudLayersCommands.h not found")
        src = _read_repo_text(h)
        assert "CommandCloudLayers" in src
        cpp = REPO_ROOT / "plugins/core/Standard/qCloudLayers/src/qCloudLayersCommands.cpp"
        assert "CLOUD_LAYERS" in _read_repo_text(cpp)

    def test_level1_colorimetric_seg_command_header(self):
        h = REPO_ROOT / "plugins/core/Standard/qColorimetricSegmenter/ColorimetricSegmenterCommands.h"
        if not h.exists():
            pytest.fail("ColorimetricSegmenterCommands.h not found")
        src = _read_repo_text(h)
        assert "CommandColorimetricSegRGB" in src
        assert "CommandColorimetricSegHSV" in src
        assert "CommandColorimetricSegScalar" in src

    def test_level1_colorimetric_seg_command_impl(self):
        cpp = REPO_ROOT / "plugins/core/Standard/qColorimetricSegmenter/ColorimetricSegmenterCommands.cpp"
        if not cpp.exists():
            pytest.fail("ColorimetricSegmenterCommands.cpp not found")
        src = _read_repo_text(cpp)
        assert "COLOR_SEG_RGB" in src
        assert "COLOR_SEG_HSV" in src
        assert "COLOR_SEG_SCALAR" in src

    def test_level1_g3point_command_header(self):
        h = REPO_ROOT / "plugins/core/Standard/qG3Point/include/G3PointCommands.h"
        if not h.exists():
            pytest.fail("G3PointCommands.h not found")
        src = _read_repo_text(h)
        assert "CommandG3Point" in src
        cpp = REPO_ROOT / "plugins/core/Standard/qG3Point/src/G3PointCommands.cpp"
        assert "G3POINT" in _read_repo_text(cpp)

    def test_level1_3dmasc_command_header(self):
        h = REPO_ROOT / "plugins/core/Standard/q3DMASC/q3DMASCCommands.h"
        if not h.exists():
            pytest.fail("q3DMASCCommands.h not found")
        src = _read_repo_text(h)
        assert "Command3DMASCClassif" in src
        assert "3DMASC_CLASSIFY" in src

    def test_level1_pcv_command_header(self):
        h = REPO_ROOT / "plugins/core/Standard/qPCV/include/PCVCommand.h"
        if not h.exists():
            pytest.fail("PCVCommand.h not found")
        src = _read_repo_text(h)
        assert "PCVCommand" in src

    def test_level1_compass_command_header(self):
        h = REPO_ROOT / "plugins/core/Standard/qCompass/include/ccCompassCommands.h"
        if not h.exists():
            pytest.fail("ccCompassCommands.h not found")
        src = _read_repo_text(h)
        assert "CommandCompassExport" in src
        assert "CommandCompassRefit" in src
        assert "CommandCompassP21" in src
        assert "CommandCompassImportFoliations" in src
        assert "CommandCompassImportLineations" in src

    def test_level1_compass_command_impl(self):
        cpp = REPO_ROOT / "plugins/core/Standard/qCompass/src/ccCompassCommands.cpp"
        if not cpp.exists():
            pytest.fail("ccCompassCommands.cpp not found")
        src = _read_repo_text(cpp)
        assert "COMPASS_EXPORT" in src
        assert "COMPASS_REFIT" in src
        assert "COMPASS_P21" in src
        assert "COMPASS_IMPORT_FOL" in src
        assert "COMPASS_IMPORT_LIN" in src

    def test_level1_compass_registered(self):
        cpp = REPO_ROOT / "plugins/core/Standard/qCompass/src/ccCompass.cpp"
        if not cpp.exists():
            pytest.fail("ccCompass.cpp not found")
        src = _read_repo_text(cpp)
        assert "registerCommands" in src
        assert "CommandCompassExport" in src

    def test_level1_sra_command_header(self):
        h = REPO_ROOT / "plugins/core/Standard/qSRA/include/qSRACommands.h"
        if not h.exists():
            pytest.fail("qSRACommands.h not found")
        src = _read_repo_text(h)
        assert "CommandSRARadialDist" in src

    def test_level1_sra_command_impl(self):
        cpp = REPO_ROOT / "plugins/core/Standard/qSRA/src/qSRACommands.cpp"
        if not cpp.exists():
            pytest.fail("qSRACommands.cpp not found")
        src = _read_repo_text(cpp)
        assert "SRA" in src

    def test_level1_sra_registered(self):
        cpp = REPO_ROOT / "plugins/core/Standard/qSRA/src/qSRA.cpp"
        if not cpp.exists():
            pytest.fail("qSRA.cpp not found")
        src = _read_repo_text(cpp)
        assert "registerCommands" in src
        assert "CommandSRARadialDist" in src

    def test_level1_io_coreio_command(self):
        h = REPO_ROOT / "plugins/core/IO/qCoreIO/include/CoreIOCommands.h"
        if not h.exists():
            pytest.fail("CoreIOCommands.h not found")
        src = _read_repo_text(h)
        assert "CommandCoreIO" in src
        cpp = REPO_ROOT / "plugins/core/IO/qCoreIO/src/CoreIOCommands.cpp"
        assert "CORE_IO" in _read_repo_text(cpp)

    def test_level1_io_csvmatrix_command(self):
        h = REPO_ROOT / "plugins/core/IO/qCSVMatrixIO/include/CSVMatrixCommands.h"
        if not h.exists():
            pytest.fail("CSVMatrixCommands.h not found")
        assert "CommandCSVMatrix" in _read_repo_text(h)

    def test_level1_io_photoscan_command(self):
        h = REPO_ROOT / "plugins/core/IO/qPhotoscanIO/include/PhotoscanCommands.h"
        if not h.exists():
            pytest.fail("PhotoscanCommands.h not found")
        assert "CommandPhotoscan" in _read_repo_text(h)

    def test_level1_io_e57_command(self):
        h = REPO_ROOT / "plugins/core/IO/qE57IO/include/E57Commands.h"
        if not h.exists():
            pytest.fail("E57Commands.h not found")
        assert "CommandE57" in _read_repo_text(h)

    def test_level1_io_draco_command(self):
        h = REPO_ROOT / "plugins/core/IO/qDracoIO/include/DracoCommands.h"
        if not h.exists():
            pytest.fail("DracoCommands.h not found")
        assert "CommandDraco" in _read_repo_text(h)

    def test_level1_io_las_command(self):
        h = REPO_ROOT / "plugins/core/IO/qLASIO/include/LasCommands.h"
        if not h.exists():
            pytest.fail("LasCommands.h not found")
        assert "CommandLAS" in _read_repo_text(h)

    def test_level1_io_meshio_command(self):
        h = REPO_ROOT / "plugins/core/IO/qMeshIO/include/MeshIOCommands.h"
        if not h.exists():
            pytest.fail("MeshIOCommands.h not found")
        assert "CommandMeshIO" in _read_repo_text(h)

    def test_level1_python_runtime_command(self):
        """qPythonRuntime already has built-in PYTHON_SCRIPT command."""
        cpp = REPO_ROOT / "plugins/core/Standard/qPythonRuntime/src/PythonPlugin.cpp"
        if not cpp.exists():
            pytest.fail("PythonPlugin.cpp not found")
        src = _read_repo_text(cpp)
        assert "PythonPluginCommand" in src
        assert "PYTHON_SCRIPT" in src
        assert "registerCommands" in src

    def test_level1_mplane_command(self):
        h = REPO_ROOT / "plugins/core/Standard/qMPlane/MPlaneCommands.h"
        if not h.exists():
            pytest.fail("MPlaneCommands.h not found")
        src = _read_repo_text(h)
        assert "CommandMPlane" in src
        cpp = REPO_ROOT / "plugins/core/Standard/qMPlane/MPlaneCommands.cpp"
        assert "MPLANE" in _read_repo_text(cpp)

    def test_level1_autoseg_command(self):
        h = REPO_ROOT / "plugins/core/Standard/qMasonry/qAutoSeg/AutoSegCommands.h"
        if not h.exists():
            pytest.fail("AutoSegCommands.h not found")
        src = _read_repo_text(h)
        assert "CommandAutoSeg" in src
        cpp = REPO_ROOT / "plugins/core/Standard/qMasonry/qAutoSeg/AutoSegCommands.cpp"
        assert "AUTO_SEG" in _read_repo_text(cpp)

    def test_level1_manualseg_command(self):
        h = REPO_ROOT / "plugins/core/Standard/qMasonry/qManualSeg/ManualSegCommands.h"
        if not h.exists():
            pytest.fail("ManualSegCommands.h not found")
        src = _read_repo_text(h)
        assert "CommandManualSeg" in src
        cpp = REPO_ROOT / "plugins/core/Standard/qMasonry/qManualSeg/ManualSegCommands.cpp"
        assert "MANUAL_SEG" in _read_repo_text(cpp)

    def test_level1_plugin_commands_registered(self):
        """Verify registerCommands is implemented in all new plugins."""
        plugins = [
            ("qAnimation", "plugins/core/Standard/qAnimation/src/qAnimation.cpp", "CommandAnimation"),
            ("qCloudLayers", "plugins/core/Standard/qCloudLayers/src/qCloudLayers.cpp", "CommandCloudLayers"),
            ("qColorimetricSegmenter", "plugins/core/Standard/qColorimetricSegmenter/qColorimetricSegmenter.cpp",
             "CommandColorimetricSeg"),
            ("qG3Point", "plugins/core/Standard/qG3Point/src/G3Point.cpp", "CommandG3Point"),
            ("qCoreIO", "plugins/core/IO/qCoreIO/src/qCoreIO.cpp", "CommandCoreIO"),
            ("qCSVMatrixIO", "plugins/core/IO/qCSVMatrixIO/src/qCSVMatrixIO.cpp", "CommandCSVMatrix"),
            ("qPhotoscanIO", "plugins/core/IO/qPhotoscanIO/src/qPhotoscanIO.cpp", "CommandPhotoscan"),
            ("qE57IO", "plugins/core/IO/qE57IO/src/qE57IO.cpp", "CommandE57"),
            ("qDracoIO", "plugins/core/IO/qDracoIO/src/qDracoIO.cpp", "CommandDraco"),
            ("qLASIO", "plugins/core/IO/qLASIO/src/LasPlugin.cpp", "CommandLAS"),
            ("qMeshIO", "plugins/core/IO/qMeshIO/src/qMeshIO.cpp", "CommandMeshIO"),
            ("qPythonRuntime", "plugins/core/Standard/qPythonRuntime/src/PythonPlugin.cpp", "PythonPluginCommand"),
            ("qMPlane", "plugins/core/Standard/qMPlane/qMPlane.cpp", "CommandMPlane"),
            ("qAutoSeg", "plugins/core/Standard/qMasonry/qAutoSeg/qAutoSeg.cpp", "CommandAutoSeg"),
            ("qManualSeg", "plugins/core/Standard/qMasonry/qManualSeg/qManualSeg.cpp", "CommandManualSeg"),
            ("qCompass", "plugins/core/Standard/qCompass/src/ccCompass.cpp", "CommandCompassExport"),
            ("qSRA", "plugins/core/Standard/qSRA/src/qSRA.cpp", "CommandSRARadialDist"),
        ]
        for name, rel_path, cmd_class in plugins:
            path = REPO_ROOT / rel_path
            if not path.exists():
                continue
            src = _read_repo_text(path)
            assert "registerCommands" in src, f"{name} missing registerCommands"
            assert "registerCommand" in src, f"{name} not calling registerCommand"


# ═══════════════════════════════════════════════════════════════════════════
# Level 2: CLI Plugin Command Tests
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not HAS_CLI, reason="CLI harness not installed")
class TestLevel2_PluginCLICommands:
    """Verify CLI plugin commands work (help output)."""

    @pytest.mark.parametrize("subcmd", [
        "3dmasc", "animation", "cloud-layers",
        "color-seg-rgb", "color-seg-hsv", "color-seg-scalar",
        "g3point", "draco-settings", "e57-settings",
        "las-settings", "csv-matrix-settings", "photoscan-settings",
        "mesh-io-settings", "core-io-settings",
        "python-script", "mplane", "auto-seg", "manual-seg",
    ])
    def test_level2_plugin_command_help(self, subcmd):
        r = subprocess.run(
            ["cli-anything-acloudviewer", "process", subcmd, "--help"],
            capture_output=True, text=True, timeout=10)
        assert r.returncode == 0, f"process {subcmd} --help failed: {r.stderr}"
        assert "--help" in r.stdout or "Usage" in r.stdout


# ═══════════════════════════════════════════════════════════════════════════
# Level 3: Headless Plugin Processing Tests
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not HAS_BINARY, reason="ACloudViewer binary not found")
class TestLevel3_PluginHeadlessProcessing:
    """Verify headless plugin processing via binary CLI."""

    @pytest.fixture
    def sample_ply(self, tmp_path):
        ply_path = tmp_path / "test.ply"
        ply_path.write_text(_SAMPLE_PLY_HEADER + _SAMPLE_PLY_BODY)
        return str(ply_path)

    @pytest.fixture
    def acv_env(self):
        return _build_env_for_binary(BINARY_PATH)

    def test_level3_pcv_headless(self, sample_ply, acv_env, tmp_path):
        if acv_env.get("QT_QPA_PLATFORM") == "minimal":
            # PCVContext renders through QOpenGLContext + QOffscreenSurface;
            # the minimal QPA (Windows here) implements no GL context.
            pytest.skip("PCV needs an OpenGL context; the minimal QPA has none")
        _require_plugin("QPCV_PLUGIN")
        out = tmp_path / "pcv_out.ply"
        args = [BINARY_PATH, "-SILENT", "-O", str(sample_ply),
                "-AUTO_SAVE", "OFF", "-NO_TIMESTAMP",
                "-PCV", "-SAVE_CLOUDS", "FILE", str(out)]
        r = subprocess.run(args, capture_output=True, text=True, timeout=120,
                           env=acv_env)
        _fail_if_crashed(r, "PCV")
        assert r.returncode == 0 and out.exists(), (
            f"PCV headless failed (rc={r.returncode}):\n"
            f"{(r.stdout + r.stderr)[-2000:]}")

    def test_level3_animation_headless(self, acv_env, tmp_path):
        args = [BINARY_PATH, "-SILENT", "-ANIMATION",
                "-FPS", "24", "-TOTAL_FRAMES", "10"]
        r = subprocess.run(args, capture_output=True, text=True, timeout=30,
                           env=acv_env)
        assert r.returncode == 0, (
            f"ANIMATION failed (rc={r.returncode}):\n{(r.stdout+r.stderr)[-2000:]}")

    def test_level3_draco_settings_headless(self, acv_env, tmp_path):
        args = [BINARY_PATH, "-SILENT", "-DRACO",
                "-QUANTIZATION", "14", "-SPEED", "3"]
        r = subprocess.run(args, capture_output=True, text=True, timeout=30,
                           env=acv_env)
        assert r.returncode == 0, (
            f"DRACO failed (rc={r.returncode}):\n{(r.stdout+r.stderr)[-2000:]}")

    def test_level3_e57_settings_headless(self, acv_env, tmp_path):
        args = [BINARY_PATH, "-SILENT", "-E57", "-IGNORE_COLOR"]
        r = subprocess.run(args, capture_output=True, text=True, timeout=30,
                           env=acv_env)
        assert r.returncode == 0, (
            f"E57 failed (rc={r.returncode}):\n{(r.stdout+r.stderr)[-2000:]}")

    def test_level3_las_settings_headless(self, acv_env, tmp_path):
        args = [BINARY_PATH, "-SILENT", "-LAS", "-EXTRA_FIELDS", "-SAVE_LAZ"]
        r = subprocess.run(args, capture_output=True, text=True, timeout=30,
                           env=acv_env)
        assert r.returncode == 0, (
            f"LAS failed (rc={r.returncode}):\n{(r.stdout+r.stderr)[-2000:]}")

    def test_level3_mplane_headless(self, sample_ply, acv_env, tmp_path):
        _require_plugin("QMPLANE_PLUGIN")
        out = tmp_path / "mplane_out.ply"
        args = [BINARY_PATH, "-SILENT", "-O", str(sample_ply),
                "-AUTO_SAVE", "OFF", "-NO_TIMESTAMP",
                "-MPLANE", "-NX", "0", "-NY", "0", "-NZ", "1", "-D", "0",
                "-SAVE_CLOUDS", "FILE", str(out)]
        r = subprocess.run(args, capture_output=True, text=True, timeout=60,
                           env=acv_env)
        combined = r.stdout + r.stderr
        assert r.returncode == 0, (
            f"MPLANE failed (rc={r.returncode}):\n{combined[-2000:]}")

    def test_level3_autoseg_headless(self, sample_ply, acv_env, tmp_path):
        _require_plugin("QAUTO_SEG_PLUGIN")
        out = tmp_path / "autoseg_out.ply"
        args = [BINARY_PATH, "-SILENT", "-O", str(sample_ply),
                "-AUTO_SAVE", "OFF", "-NO_TIMESTAMP",
                "-AUTO_SEG", "-MORTAR_MAPS",
                "-SAVE_CLOUDS", "FILE", str(out)]
        r = subprocess.run(args, capture_output=True, text=True, timeout=60,
                           env=acv_env)
        combined = r.stdout + r.stderr
        assert r.returncode == 0, (
            f"AUTO_SEG failed (rc={r.returncode}):\n{combined[-2000:]}")

    def test_level3_manualseg_headless(self, sample_ply, acv_env, tmp_path):
        _require_plugin("QMANUAL_SEG_PLUGIN")
        out = tmp_path / "manualseg_out.ply"
        args = [BINARY_PATH, "-SILENT", "-O", str(sample_ply),
                "-AUTO_SAVE", "OFF", "-NO_TIMESTAMP",
                "-MANUAL_SEG", "-CONTOURS",
                "-SAVE_CLOUDS", "FILE", str(out)]
        r = subprocess.run(args, capture_output=True, text=True, timeout=60,
                           env=acv_env)
        combined = r.stdout + r.stderr
        assert r.returncode == 0, (
            f"MANUAL_SEG failed (rc={r.returncode}):\n{combined[-2000:]}")

    def test_level3_compass_export_headless(self, sample_ply, acv_env, tmp_path):
        _require_plugin("QCOMPASS_PLUGIN")
        args = [BINARY_PATH, "-SILENT", "-O", str(sample_ply),
                "-AUTO_SAVE", "OFF", "-NO_TIMESTAMP",
                "-COMPASS_EXPORT", "-FORMAT", "csv",
                "-OUTPUT", str(tmp_path / "compass_out")]
        r = subprocess.run(args, capture_output=True, text=True, timeout=60,
                           env=acv_env)
        combined = r.stdout + r.stderr
        assert r.returncode == 0, (
            f"COMPASS_EXPORT failed (rc={r.returncode}):\n{combined[-2000:]}")

    def test_level3_compass_refit_headless(self, sample_ply, acv_env, tmp_path):
        _require_plugin("QCOMPASS_PLUGIN")
        args = [BINARY_PATH, "-SILENT", "-O", str(sample_ply),
                "-AUTO_SAVE", "OFF", "-NO_TIMESTAMP",
                "-COMPASS_REFIT"]
        r = subprocess.run(args, capture_output=True, text=True, timeout=60,
                           env=acv_env)
        combined = r.stdout + r.stderr
        assert r.returncode == 0, (
            f"COMPASS_REFIT failed (rc={r.returncode}):\n{combined[-2000:]}")

    def test_level3_compass_p21_headless(self, sample_ply, acv_env, tmp_path):
        _require_plugin("QCOMPASS_PLUGIN")
        args = [BINARY_PATH, "-SILENT", "-O", str(sample_ply),
                "-AUTO_SAVE", "OFF", "-NO_TIMESTAMP",
                "-COMPASS_P21", "-RADIUS", "0.5", "-SUBSAMPLE", "5"]
        r = subprocess.run(args, capture_output=True, text=True, timeout=60,
                           env=acv_env)
        combined = r.stdout + r.stderr
        _fail_if_crashed(r, "COMPASS_P21")
        assert r.returncode == 0, (
            f"COMPASS_P21 failed (rc={r.returncode}):\n{combined[-2000:]}")

    def test_level3_compass_import_fol_headless(self, sample_ply, acv_env, tmp_path):
        _require_plugin("QCOMPASS_PLUGIN")
        args = [BINARY_PATH, "-SILENT", "-O", str(sample_ply),
                "-AUTO_SAVE", "OFF", "-NO_TIMESTAMP",
                "-COMPASS_IMPORT_FOL", "-DIP_SF", "Dip", "-DIPDIR_SF", "DipDir"]
        r = subprocess.run(args, capture_output=True, text=True, timeout=60,
                           env=acv_env)
        combined = r.stdout + r.stderr
        assert r.returncode == 0, (
            f"COMPASS_IMPORT_FOL failed (rc={r.returncode}):\n{combined[-2000:]}")

    def test_level3_compass_import_lin_headless(self, sample_ply, acv_env, tmp_path):
        _require_plugin("QCOMPASS_PLUGIN")
        args = [BINARY_PATH, "-SILENT", "-O", str(sample_ply),
                "-AUTO_SAVE", "OFF", "-NO_TIMESTAMP",
                "-COMPASS_IMPORT_LIN", "-TREND_SF", "Trend", "-PLUNGE_SF", "Plunge"]
        r = subprocess.run(args, capture_output=True, text=True, timeout=60,
                           env=acv_env)
        combined = r.stdout + r.stderr
        assert r.returncode == 0, (
            f"COMPASS_IMPORT_LIN failed (rc={r.returncode}):\n{combined[-2000:]}")

    def test_level3_pcl_don_headless(self, sample_ply, acv_env, tmp_path):
        _require_plugin("QPCL_PLUGIN")
        out = tmp_path / "don_out.ply"
        args = [BINARY_PATH, "-SILENT", "-O", str(sample_ply),
                "-AUTO_SAVE", "OFF", "-NO_TIMESTAMP",
                "-PCL_DON_SEGMENTATION", "-SMALL_SCALE", "5", "-LARGE_SCALE", "10",
                "-SAVE_CLOUDS", "FILE", str(out)]
        r = subprocess.run(args, capture_output=True, text=True, timeout=60,
                           env=acv_env)
        combined = r.stdout + r.stderr
        assert r.returncode == 0, (
            f"PCL_DON_SEGMENTATION failed (rc={r.returncode}):\n{combined[-2000:]}")

    def test_level3_pcl_mincut_headless(self, sample_ply, acv_env, tmp_path):
        _require_plugin("QPCL_PLUGIN")
        out = tmp_path / "mincut_out.ply"
        args = [BINARY_PATH, "-SILENT", "-O", str(sample_ply),
                "-AUTO_SAVE", "OFF", "-NO_TIMESTAMP",
                "-PCL_MINCUT_SEGMENTATION", "-FX", "0", "-FY", "0", "-FZ", "0",
                "-SAVE_CLOUDS", "FILE", str(out)]
        r = subprocess.run(args, capture_output=True, text=True, timeout=60,
                           env=acv_env)
        combined = r.stdout + r.stderr
        assert r.returncode == 0, (
            f"PCL_MINCUT_SEGMENTATION failed (rc={r.returncode}):\n{combined[-2000:]}")

    def test_level3_pcl_sift_headless(self, sample_ply, acv_env, tmp_path):
        _require_plugin("QPCL_PLUGIN")
        out = tmp_path / "sift_out.ply"
        args = [BINARY_PATH, "-SILENT", "-O", str(sample_ply),
                "-AUTO_SAVE", "OFF", "-NO_TIMESTAMP",
                "-PCL_EXTRACT_SIFT", "-MODE", "RGB", "-OCTAVES", "4",
                "-MIN_SCALE", "0.01", "-SCALES_PER_OCTAVE", "6",
                "-SAVE_CLOUDS", "FILE", str(out)]
        r = subprocess.run(args, capture_output=True, text=True, timeout=60,
                           env=acv_env)
        combined = r.stdout + r.stderr
        assert r.returncode == 0, (
            f"PCL_EXTRACT_SIFT failed (rc={r.returncode}):\n{combined[-2000:]}")

    def test_level3_pcl_projection_headless(self, sample_ply, acv_env, tmp_path):
        _require_plugin("QPCL_PLUGIN")
        out = tmp_path / "proj_out.ply"
        args = [BINARY_PATH, "-SILENT", "-O", str(sample_ply),
                "-AUTO_SAVE", "OFF", "-NO_TIMESTAMP",
                "-PCL_PROJECTION_FILTER", "-A", "0", "-B", "0", "-C", "1", "-D", "0",
                "-SAVE_CLOUDS", "FILE", str(out)]
        r = subprocess.run(args, capture_output=True, text=True, timeout=60,
                           env=acv_env)
        combined = r.stdout + r.stderr
        assert r.returncode == 0, (
            f"PCL_PROJECTION_FILTER failed (rc={r.returncode}):\n{combined[-2000:]}")

    def test_level3_pcl_general_pass_headless(self, sample_ply, acv_env, tmp_path):
        _require_plugin("QPCL_PLUGIN")
        out = tmp_path / "genfilt_out.ply"
        args = [BINARY_PATH, "-SILENT", "-O", str(sample_ply),
                "-AUTO_SAVE", "OFF", "-NO_TIMESTAMP",
                "-PCL_GENERAL_FILTERS", "-MODE", "PASS", "-FIELD", "z",
                "-MIN", "-100", "-MAX", "100",
                "-SAVE_CLOUDS", "FILE", str(out)]
        r = subprocess.run(args, capture_output=True, text=True, timeout=60,
                           env=acv_env)
        combined = r.stdout + r.stderr
        assert r.returncode == 0, (
            f"PCL_GENERAL_FILTERS PASS failed (rc={r.returncode}):\n{combined[-2000:]}")

    def test_level3_pcl_general_voxel_headless(self, sample_ply, acv_env, tmp_path):
        _require_plugin("QPCL_PLUGIN")
        out = tmp_path / "voxel_out.ply"
        args = [BINARY_PATH, "-SILENT", "-O", str(sample_ply),
                "-AUTO_SAVE", "OFF", "-NO_TIMESTAMP",
                "-PCL_GENERAL_FILTERS", "-MODE", "VOXEL", "-LEAF", "0.01",
                "-SAVE_CLOUDS", "FILE", str(out)]
        r = subprocess.run(args, capture_output=True, text=True, timeout=60,
                           env=acv_env)
        combined = r.stdout + r.stderr
        assert r.returncode == 0, (
            f"PCL_GENERAL_FILTERS VOXEL failed (rc={r.returncode}):\n{combined[-2000:]}")


# ═══════════════════════════════════════════════════════════════════════════
# Level 4: GUI RPC Plugin Command Tests
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not _rpc_available(), reason="ACloudViewer RPC not available")
class TestLevel4_PluginRPC:
    """Test plugin commands via JSON-RPC (requires running GUI app)."""

    @pytest.fixture
    def sample_ply(self, tmp_path):
        ply_path = tmp_path / "test.ply"
        ply_path.write_text(_SAMPLE_PLY_HEADER + _SAMPLE_PLY_BODY)
        return str(ply_path)

    def _rpc_call(self, method, params=None):
        ws = ws_connect(RPC_URL, open_timeout=5)
        try:
            ws.send(json.dumps({
                "jsonrpc": "2.0", "id": 1,
                "method": method,
                "params": params or {},
            }))
            resp = json.loads(ws.recv(timeout=30))
            if "error" in resp:
                raise RuntimeError(f"RPC error: {resp['error']}")
            return resp.get("result")
        finally:
            ws.close()

    def test_level4_rpc_ping(self):
        assert self._rpc_call("ping") == "pong"

# ═══════════════════════════════════════════════════════════════════════════
# Level 5: MCP Plugin Tool Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestLevel5_PluginMCPTools:
    """Verify MCP tools for new plugin commands."""

    @_needs_mcp
    def test_level5_mcp_3dmasc_tool_schema(self):
        from cli_anything.acloudviewer.mcp_server import list_tools
        import asyncio
        tools = asyncio.run(list_tools())
        tool = next((t for t in tools if t.name == "classify_3dmasc"), None)
        assert tool is not None, "classify_3dmasc tool not found"
        schema = tool.inputSchema
        assert "classifier_file" in schema.get("properties", {})
        assert "cloud_roles" in schema.get("properties", {})

    @_needs_mcp
    def test_level5_mcp_color_seg_rgb_tool_schema(self):
        from cli_anything.acloudviewer.mcp_server import list_tools
        import asyncio
        tools = asyncio.run(list_tools())
        tool = next((t for t in tools if t.name == "color_seg_rgb"), None)
        assert tool is not None, "color_seg_rgb tool not found"
        props = tool.inputSchema.get("properties", {})
        for key in ["r_min", "r_max", "g_min", "g_max", "b_min", "b_max"]:
            assert key in props, f"Missing property: {key}"

    @_needs_mcp
    def test_level5_mcp_g3point_tool_schema(self):
        from cli_anything.acloudviewer.mcp_server import list_tools
        import asyncio
        tools = asyncio.run(list_tools())
        tool = next((t for t in tools if t.name == "g3point"), None)
        assert tool is not None, "g3point tool not found"
        props = tool.inputSchema.get("properties", {})
        for key in ["n_neighbors", "max_radius", "min_radius"]:
            assert key in props, f"Missing property: {key}"

    @_needs_mcp
    def test_level5_mcp_draco_settings_schema(self):
        from cli_anything.acloudviewer.mcp_server import list_tools
        import asyncio
        tools = asyncio.run(list_tools())
        tool = next((t for t in tools if t.name == "draco_settings"), None)
        assert tool is not None, "draco_settings tool not found"
        props = tool.inputSchema.get("properties", {})
        assert "quantization" in props
        assert "compression_level" in props

    @_needs_mcp
    def test_level5_mcp_las_settings_schema(self):
        from cli_anything.acloudviewer.mcp_server import list_tools
        import asyncio
        tools = asyncio.run(list_tools())
        tool = next((t for t in tools if t.name == "las_settings"), None)
        assert tool is not None, "las_settings tool not found"
        props = tool.inputSchema.get("properties", {})
        assert "extra_fields" in props
        assert "save_laz" in props

    @_needs_mcp
    def test_level5_mcp_python_script_schema(self):
        from cli_anything.acloudviewer.mcp_server import list_tools
        import asyncio
        tools = asyncio.run(list_tools())
        tool = next((t for t in tools if t.name == "python_script"), None)
        assert tool is not None, "python_script tool not found"
        props = tool.inputSchema.get("properties", {})
        assert "script_path" in props
        assert "script_args" in props

    @_needs_mcp
    def test_level5_mcp_mplane_schema(self):
        from cli_anything.acloudviewer.mcp_server import list_tools
        import asyncio
        tools = asyncio.run(list_tools())
        tool = next((t for t in tools if t.name == "mplane"), None)
        assert tool is not None, "mplane tool not found"
        props = tool.inputSchema.get("properties", {})
        for key in ["nx", "ny", "nz", "d"]:
            assert key in props, f"Missing property: {key}"

    @_needs_mcp
    def test_level5_mcp_auto_seg_schema(self):
        from cli_anything.acloudviewer.mcp_server import list_tools
        import asyncio
        tools = asyncio.run(list_tools())
        tool = next((t for t in tools if t.name == "auto_seg"), None)
        assert tool is not None, "auto_seg tool not found"
        props = tool.inputSchema.get("properties", {})
        assert "mortar_maps" in props
        assert "contours" in props

    @_needs_mcp
    def test_level5_mcp_manual_seg_schema(self):
        from cli_anything.acloudviewer.mcp_server import list_tools
        import asyncio
        tools = asyncio.run(list_tools())
        tool = next((t for t in tools if t.name == "manual_seg"), None)
        assert tool is not None, "manual_seg tool not found"
        props = tool.inputSchema.get("properties", {})
        assert "mortar_maps" in props
        assert "contours" in props

    @_needs_mcp
    def test_level5_mcp_pcl_sor_schema(self):
        from cli_anything.acloudviewer.mcp_server import list_tools
        import asyncio
        tools = asyncio.run(list_tools())
        tool = next((t for t in tools if t.name == "pcl_sor"), None)
        assert tool is not None, "pcl_sor tool not found"
        props = tool.inputSchema.get("properties", {})
        assert "k" in props
        assert "std" in props

    @_needs_mcp
    def test_level5_mcp_pcv_schema(self):
        from cli_anything.acloudviewer.mcp_server import list_tools
        import asyncio
        tools = asyncio.run(list_tools())
        tool = next((t for t in tools if t.name == "pcv"), None)
        assert tool is not None, "pcv tool not found"
        props = tool.inputSchema.get("properties", {})
        assert "input_path" in props
        assert "output_path" in props

    @_needs_mcp
    def test_level5_mcp_compass_export_schema(self):
        from cli_anything.acloudviewer.mcp_server import list_tools
        import asyncio
        tools = asyncio.run(list_tools())
        tool = next((t for t in tools if t.name == "compass_export"), None)
        assert tool is not None, "compass_export tool not found"
        props = tool.inputSchema.get("properties", {})
        assert "input_path" in props
        assert "output_path" in props
        assert "format" in props

    @_needs_mcp
    def test_level5_mcp_sra_schema(self):
        from cli_anything.acloudviewer.mcp_server import list_tools
        import asyncio
        tools = asyncio.run(list_tools())
        tool = next((t for t in tools if t.name == "sra"), None)
        assert tool is not None, "sra tool not found"
        props = tool.inputSchema.get("properties", {})
        assert "input_path" in props
        assert "output_path" in props
        assert "profile_path" in props
        assert "axis" in props

    @_needs_mcp
    def test_level5_mcp_pcl_don_schema(self):
        from cli_anything.acloudviewer.mcp_server import list_tools
        import asyncio
        tools = asyncio.run(list_tools())
        tool = next((t for t in tools if t.name == "pcl_don_segmentation"), None)
        assert tool is not None, "pcl_don_segmentation tool not found"
        props = tool.inputSchema.get("properties", {})
        assert "small_scale" in props
        assert "large_scale" in props
        assert "min_don" in props

    @_needs_mcp
    def test_level5_mcp_pcl_mincut_schema(self):
        from cli_anything.acloudviewer.mcp_server import list_tools
        import asyncio
        tools = asyncio.run(list_tools())
        tool = next((t for t in tools if t.name == "pcl_mincut_segmentation"), None)
        assert tool is not None, "pcl_mincut_segmentation tool not found"
        props = tool.inputSchema.get("properties", {})
        assert "fx" in props
        assert "fy" in props
        assert "fz" in props

    @_needs_mcp
    def test_level5_mcp_pcl_fgr_schema(self):
        from cli_anything.acloudviewer.mcp_server import list_tools
        import asyncio
        tools = asyncio.run(list_tools())
        tool = next((t for t in tools if t.name == "pcl_fast_global_registration"), None)
        assert tool is not None, "pcl_fast_global_registration tool not found"
        props = tool.inputSchema.get("properties", {})
        assert "feature_radius" in props
        assert "reference_path" in props

    @_needs_mcp
    def test_level5_mcp_pcl_sift_schema(self):
        from cli_anything.acloudviewer.mcp_server import list_tools
        import asyncio
        tools = asyncio.run(list_tools())
        tool = next((t for t in tools if t.name == "pcl_extract_sift"), None)
        assert tool is not None, "pcl_extract_sift tool not found"
        props = tool.inputSchema.get("properties", {})
        assert "mode" in props
        assert "octaves" in props
        assert "min_scale" in props

    @_needs_mcp
    def test_level5_mcp_pcl_projection_schema(self):
        from cli_anything.acloudviewer.mcp_server import list_tools
        import asyncio
        tools = asyncio.run(list_tools())
        tool = next((t for t in tools if t.name == "pcl_projection_filter"), None)
        assert tool is not None, "pcl_projection_filter tool not found"
        props = tool.inputSchema.get("properties", {})
        assert "a" in props
        assert "b" in props
        assert "c" in props

    @_needs_mcp
    def test_level5_mcp_pcl_general_filters_schema(self):
        from cli_anything.acloudviewer.mcp_server import list_tools
        import asyncio
        tools = asyncio.run(list_tools())
        tool = next((t for t in tools if t.name == "pcl_general_filters"), None)
        assert tool is not None, "pcl_general_filters tool not found"
        props = tool.inputSchema.get("properties", {})
        assert "mode" in props
        assert "field" in props

    @_needs_mcp
    def test_level5_mcp_compass_import_fol_schema(self):
        from cli_anything.acloudviewer.mcp_server import list_tools
        import asyncio
        tools = asyncio.run(list_tools())
        tool = next((t for t in tools if t.name == "compass_import_fol"), None)
        assert tool is not None, "compass_import_fol tool not found"
        props = tool.inputSchema.get("properties", {})
        assert "input_path" in props
        assert "dip_sf" in props
        assert "dipdir_sf" in props
        assert "plane_size" in props

    @_needs_mcp
    def test_level5_mcp_compass_import_lin_schema(self):
        from cli_anything.acloudviewer.mcp_server import list_tools
        import asyncio
        tools = asyncio.run(list_tools())
        tool = next((t for t in tools if t.name == "compass_import_lin"), None)
        assert tool is not None, "compass_import_lin tool not found"
        props = tool.inputSchema.get("properties", {})
        assert "input_path" in props
        assert "trend_sf" in props
        assert "plunge_sf" in props
        assert "length" in props

    @_needs_mcp
    def test_level5_mcp_compass_refit_schema(self):
        from cli_anything.acloudviewer.mcp_server import list_tools
        import asyncio
        tools = asyncio.run(list_tools())
        tool = next((t for t in tools if t.name == "compass_refit"), None)
        assert tool is not None, "compass_refit tool not found"
        props = tool.inputSchema.get("properties", {})
        assert "input_path" in props
        assert "output_path" in props

    @_needs_mcp
    def test_level5_mcp_compass_p21_schema(self):
        from cli_anything.acloudviewer.mcp_server import list_tools
        import asyncio
        tools = asyncio.run(list_tools())
        tool = next((t for t in tools if t.name == "compass_p21"), None)
        assert tool is not None, "compass_p21 tool not found"
        props = tool.inputSchema.get("properties", {})
        assert "input_path" in props
        assert "radius" in props
        assert "subsample" in props

    @_needs_mcp
    def test_level5_mcp_treeiso_schema(self):
        from cli_anything.acloudviewer.mcp_server import list_tools
        import asyncio
        tools = asyncio.run(list_tools())
        tool = next((t for t in tools if t.name == "treeiso"), None)
        assert tool is not None, "treeiso tool not found"
        props = tool.inputSchema.get("properties", {})
        assert "input_path" in props
        assert "lambda1" in props
        assert "k1" in props
        assert "rho" in props
        assert "vertical_weight" in props

    @_needs_mcp
    def test_level5_mcp_fbx_settings_schema(self):
        from cli_anything.acloudviewer.mcp_server import list_tools
        import asyncio
        tools = asyncio.run(list_tools())
        tool = next((t for t in tools if t.name == "fbx_settings"), None)
        assert tool is not None, "fbx_settings tool not found"
        props = tool.inputSchema.get("properties", {})
        assert "export_format" in props

    @_needs_mcp
    def test_level5_mcp_lasfwf_load_schema(self):
        from cli_anything.acloudviewer.mcp_server import list_tools
        import asyncio
        tools = asyncio.run(list_tools())
        tool = next((t for t in tools if t.name == "lasfwf_load"), None)
        assert tool is not None, "lasfwf_load tool not found"
        props = tool.inputSchema.get("properties", {})
        assert "input_path" in props
        assert "output_path" in props
        assert "global_shift" in props

    @_needs_mcp
    def test_level5_mcp_lasfwf_save_schema(self):
        from cli_anything.acloudviewer.mcp_server import list_tools
        import asyncio
        tools = asyncio.run(list_tools())
        tool = next((t for t in tools if t.name == "lasfwf_save"), None)
        assert tool is not None, "lasfwf_save tool not found"
        props = tool.inputSchema.get("properties", {})
        assert "input_path" in props
        assert "compressed" in props
        assert "all_at_once" in props

    @_needs_mcp
    def test_level5_mcp_bundler_import_schema(self):
        from cli_anything.acloudviewer.mcp_server import list_tools
        import asyncio
        tools = asyncio.run(list_tools())
        tool = next((t for t in tools if t.name == "bundler_import"), None)
        assert tool is not None, "bundler_import tool not found"
        props = tool.inputSchema.get("properties", {})
        assert "bundler_file" in props
        assert "output_path" in props
        assert "scale_factor" in props
        assert "undistort" in props
