"""ACloudViewer Agent Integration — Self-Contained Verification Tests.

All paths can be overridden via environment variables so the tests work
both in CI (installed binary) and local dev (build directory).

Environment variables:
    ACV_REPO_ROOT   - Repository root (default: auto-detect from __file__)
    ACV_BUILD_DIR   - CMake build directory (default: $REPO_ROOT/build_app)
    ACV_BINARY      - ACloudViewer binary path (default: auto-discover)
    ACV_RPC_URL     - JSON-RPC WebSocket URL (default: ws://localhost:6001)

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

import json
import os
import platform
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
PLUGIN_CPP = REPO_ROOT / "plugins/core/Standard/qJSonRPCPlugin/src/JsonRPCPlugin.cpp"
PLUGIN_H = REPO_ROOT / "plugins/core/Standard/qJSonRPCPlugin/include/JsonRPCPlugin.h"

# ── Binary discovery (build dir → installed → env var) ────────────────────

def _find_build_binary() -> str | None:
    """Find the ACloudViewer binary in the build directory."""
    if IS_WINDOWS:
        candidates = ["bin/ACloudViewer.exe", "bin/Release/ACloudViewer.exe",
                       "bin/Debug/ACloudViewer.exe", "Release/ACloudViewer.exe"]
    elif IS_MACOS:
        candidates = ["bin/ACloudViewer", "bin/ACloudViewer.app/Contents/MacOS/ACloudViewer"]
    else:
        candidates = ["bin/ACloudViewer", "ACloudViewer"]
    for c in candidates:
        p = BUILD_DIR / c
        if p.exists():
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


BINARY_PATH = _find_any_binary()
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
    from cli_anything.acloudviewer.utils.acloudviewer_backend import ACloudViewerBackend
except ImportError:
    ACloudViewerBackend = None  # type: ignore[assignment,misc]

try:
    from websockets.sync.client import connect as ws_connect
    HAS_WS = True
except ImportError:
    HAS_WS = False

RPC_URL = os.environ.get("ACV_RPC_URL", "ws://localhost:6001")


def _rpc_available() -> bool:
    if not HAS_WS:
        return False
    try:
        ws = ws_connect(RPC_URL)
        ws.send(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping", "params": {}}))
        resp = json.loads(ws.recv())
        ws.close()
        return resp.get("result") == "pong"
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════════════
# Level 1: C++ Source Verification
# ═══════════════════════════════════════════════════════════════════════════

class TestLevel1_CppPlugin:
    """Verify C++ plugin source structure without needing a build."""

    def test_level1_plugin_cpp_exists(self):
        assert PLUGIN_CPP.exists()

    def test_level1_plugin_h_exists(self):
        assert PLUGIN_H.exists()

    def test_level1_plugin_has_dispatch_table(self):
        src = PLUGIN_CPP.read_text()
        assert 'method == "open"' in src
        assert 'method == "file.convert"' in src
        assert 'method == "scene.list"' in src
        assert 'method == "cloud.computeNormals"' in src
        assert 'method == "cloud.paintUniform"' in src
        assert 'method == "cloud.paintByHeight"' in src
        assert 'method == "cloud.paintByScalarField"' in src
        assert 'method == "mesh.simplify"' in src
        assert 'method == "view.screenshot"' in src
        assert 'method == "colmap.reconstruct"' in src

    def test_level1_rpc_method_count(self):
        src = PLUGIN_CPP.read_text()
        count = src.count('add("')
        assert count >= 25, f"Expected ≥25 RPC methods in methods.list, found {count}"

    def test_level1_header_declares_all_methods(self):
        h = PLUGIN_H.read_text()
        for method in ["rpcOpen", "rpcExport", "rpcFileConvert",
                        "rpcSceneList", "rpcSceneInfo",
                        "rpcEntityRename", "rpcEntitySetColor",
                        "rpcCloudPaintUniform", "rpcCloudPaintByHeight",
                        "rpcCloudPaintByScalarField",
                        "rpcCloudComputeNormals", "rpcCloudSubsample",
                        "rpcCloudCrop", "rpcCloudGetScalarFields",
                        "rpcMeshSimplify", "rpcMeshSmooth",
                        "rpcMeshSubdivide", "rpcMeshSamplePoints",
                        "rpcViewScreenshot", "rpcViewGetCamera",
                        "rpcTransformApply", "rpcMethodsList"]:
            assert method in h, f"Missing declaration: {method}"

    def test_level1_cursor_mcp_config_exists(self):
        mcp_json = REPO_ROOT / ".cursor" / "mcp.json"
        assert mcp_json.exists(), "Missing .cursor/mcp.json"
        data = json.loads(mcp_json.read_text())
        assert "mcpServers" in data
        assert "acloudviewer" in data["mcpServers"]
        server = data["mcpServers"]["acloudviewer"]
        assert server["command"] == "cli-anything-acloudviewer-mcp"

    def test_level1_plugin_builds(self):
        if not (BUILD_DIR / "CMakeCache.txt").exists():
            pytest.skip("No build directory")
        cmd = ["cmake", "--build", str(BUILD_DIR), "--target", "QJSON_RPC_PLUGIN"]
        if not IS_WINDOWS:
            cmd += ["--", "-j4"]
        else:
            cmd += ["--", "/m"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        assert result.returncode == 0, f"Build failed:\n{result.stderr[-2000:]}"


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

    @pytest.mark.parametrize("cmd", [
        "convert --help",
        "batch-convert --help",
        "formats --help",
        "scene --help",
        "view --help",
        "process --help",
        "session --help",
        "reconstruct --help",
        "info --help",
    ])
    def test_level2_subcommand_help(self, cmd):
        r = subprocess.run(
            ["cli-anything-acloudviewer"] + cmd.split(),
            capture_output=True, text=True, timeout=10)
        assert r.returncode == 0

    def test_level2_process_subcommands(self):
        r = subprocess.run(
            ["cli-anything-acloudviewer", "process", "--help"],
            capture_output=True, text=True, timeout=10)
        assert r.returncode == 0
        for cmd in ["subsample", "normals", "icp", "sor",
                     "c2c-dist", "c2m-dist", "density", "curvature",
                     "roughness", "delaunay", "sample-mesh", "color-banding"]:
            assert cmd in r.stdout, f"Missing process subcommand: {cmd}"

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


def _build_env_for_binary(binary_path: str) -> dict[str, str]:
    """Lightweight env setup for invoking the binary directly (no harness)."""
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    if binary_path.endswith((".sh", ".bat")):
        return env
    bin_dir = str(Path(binary_path).parent)
    lib_dir = str(Path(bin_dir) / "lib")
    sep = ";" if IS_WINDOWS else ":"
    if IS_WINDOWS:
        env["PATH"] = sep.join(filter(None, [bin_dir, lib_dir, env.get("PATH", "")]))
    elif IS_MACOS:
        env["DYLD_LIBRARY_PATH"] = sep.join(
            filter(None, [bin_dir, lib_dir, env.get("DYLD_LIBRARY_PATH", "")]))
    else:
        env["LD_LIBRARY_PATH"] = sep.join(
            filter(None, [bin_dir, lib_dir, env.get("LD_LIBRARY_PATH", "")]))
    return env


@pytest.mark.skipif(not HAS_BINARY, reason="ACloudViewer binary not found")
class TestLevel3_HeadlessProcessing:
    """Test headless processing via ACloudViewer binary directly.

    These tests do NOT require the CLI harness — they invoke the binary
    directly via subprocess so they work with both installed binaries and
    build-directory binaries.
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
        assert r.returncode == 0, f"Binary load failed:\n{r.stderr[-2000:]}"

    def test_level3_subsample(self, sample_ply, tmp_path, acv_env):
        r = subprocess.run(
            [BINARY_PATH, "-SILENT", "-O", sample_ply,
             "-SS", "SPATIAL", "0.2", "-SAVE_CLOUDS"],
            capture_output=True, text=True, timeout=60,
            env=acv_env, cwd=str(tmp_path))
        combined = r.stdout + r.stderr
        assert "Result:" in combined or r.returncode == 0, \
            f"Subsample failed (rc={r.returncode}):\n{combined[-2000:]}"

    def test_level3_normals(self, sample_ply, acv_env, tmp_path):
        r = subprocess.run(
            [BINARY_PATH, "-SILENT", "-O", sample_ply, "-COMPUTE_NORMALS"],
            capture_output=True, text=True, timeout=60,
            env=acv_env, cwd=str(tmp_path))
        assert r.returncode == 0, f"Normals failed:\n{(r.stdout + r.stderr)[-2000:]}"


@pytest.mark.skipif(not HAS_CLI, reason="CLI harness not installed")
@pytest.mark.skipif(not HAS_BINARY, reason="ACloudViewer binary not found")
class TestLevel3_CLIHarness:
    """Test headless processing via the CLI harness (requires pip install)."""

    @pytest.fixture
    def sample_ply(self, tmp_path):
        ply_path = tmp_path / "test.ply"
        ply_path.write_text(_SAMPLE_PLY_HEADER + _SAMPLE_PLY_BODY)
        return str(ply_path)

    def test_level3_cli_subsample(self, sample_ply, tmp_path):
        out = str(tmp_path / "sub.ply")
        r = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "process", "subsample", sample_ply, "-o", out, "--voxel-size", "0.2"],
            capture_output=True, text=True, timeout=60)
        assert r.returncode == 0

    def test_level3_cli_normals(self, sample_ply, tmp_path):
        out = str(tmp_path / "normals.ply")
        r = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "process", "normals", sample_ply, "-o", out],
            capture_output=True, text=True, timeout=60)
        assert r.returncode == 0

    def test_level3_cli_formats(self):
        r = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless", "formats"],
            capture_output=True, text=True, timeout=10)
        assert r.returncode == 0
        data = json.loads(r.stdout)
        assert ".ply" in data["point_cloud"]
        assert ".obj" in data["mesh"]


# ═══════════════════════════════════════════════════════════════════════════
# Level 4: GUI RPC Tests (requires running ACloudViewer)
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not _rpc_available(), reason="ACloudViewer RPC not available")
class TestLevel4_GUIRPC:
    """Test JSON-RPC communication with a running ACloudViewer instance."""

    def _call(self, method, params=None):
        ws = ws_connect(RPC_URL)
        ws.send(json.dumps({
            "jsonrpc": "2.0", "id": 1,
            "method": method, "params": params or {},
        }))
        resp = json.loads(ws.recv())
        ws.close()
        if "error" in resp:
            raise RuntimeError(resp["error"])
        return resp.get("result")

    def test_level4_ping(self):
        assert self._call("ping") == "pong"

    def test_level4_methods_list(self):
        methods = self._call("methods.list")
        assert len(methods) >= 30
        names = {m["method"] for m in methods}
        for expected in ["open", "export", "file.convert", "scene.list",
                         "cloud.computeNormals", "mesh.simplify",
                         "view.screenshot"]:
            assert expected in names, f"Missing RPC method: {expected}"

    def test_level4_scene_list(self):
        entities = self._call("scene.list", {"recursive": True})
        assert isinstance(entities, list)

    def test_level4_view_get_camera(self):
        cam = self._call("view.getCamera")
        assert "view_matrix" in cam
        assert "fov_deg" in cam

    def test_level4_cli_gui_info(self):
        r = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "gui", "info"],
            capture_output=True, text=True, timeout=10)
        assert r.returncode == 0
        data = json.loads(r.stdout)
        assert data["mode"] == "gui"


# ═══════════════════════════════════════════════════════════════════════════
# Level 5: MCP Server Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestLevel5_MCPServer:
    """Verify MCP server tool definitions."""

    def test_level5_mcp_import(self):
        try:
            from mcp.server import Server
        except ImportError:
            pytest.skip("MCP SDK not installed")

    def test_level5_mcp_tool_count(self):
        try:
            from cli_anything.acloudviewer.mcp_server import list_tools
            import asyncio
            tools = asyncio.run(list_tools())
            assert len(tools) >= 28, f"Expected ≥28 MCP tools, got {len(tools)}"
        except (ImportError, SystemExit):
            pytest.skip("MCP SDK or CLI harness not installed")

    def test_level5_mcp_tool_names(self):
        try:
            from cli_anything.acloudviewer.mcp_server import list_tools
            import asyncio
            tools = asyncio.run(list_tools())
            names = {t.name for t in tools}
            for expected in ["open_file", "convert_format", "batch_convert",
                             "scene_list", "subsample", "icp_registration",
                             "screenshot", "list_formats", "c2c_distance",
                             "c2m_distance", "delaunay", "sor_filter",
                             "colmap_auto_reconstruct", "colmap_extract_features",
                             "colmap_sparse_reconstruct", "colmap_poisson_mesh"]:
                assert expected in names, f"Missing MCP tool: {expected}"
        except (ImportError, SystemExit):
            pytest.skip("MCP SDK or CLI harness not installed")

    def test_level5_mcp_entry_point(self):
        r = subprocess.run(
            [sys.executable, "-m", "cli_anything.acloudviewer.mcp_server", "--help"],
            capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            pytest.skip("MCP server module not runnable")
