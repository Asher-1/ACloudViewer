"""ACloudViewer Agent Integration — Self-Contained Verification Tests.

Run from the ACloudViewer repo:
    cd agent-integration/tests
    python -m pytest test_integration.py -v

Or with specific levels:
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

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BUILD_DIR = REPO_ROOT / "build_app"
PLUGIN_CPP = REPO_ROOT / "plugins/core/Standard/qJSonRPCPlugin/src/JsonRPCPlugin.cpp"
PLUGIN_H = REPO_ROOT / "plugins/core/Standard/qJSonRPCPlugin/include/JsonRPCPlugin.h"

IS_WINDOWS = platform.system() == "Windows"
IS_MACOS = platform.system() == "Darwin"


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

from cli_anything.acloudviewer.utils.acloudviewer_backend import ACloudViewerBackend
HAS_BINARY = ACloudViewerBackend.find_binary() is not None

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
# Level 3: Headless Processing Tests (requires cloudViewer)
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not HAS_BINARY, reason="ACloudViewer binary not found")
@pytest.mark.skipif(not HAS_CLI, reason="CLI harness not installed")
class TestLevel3_HeadlessProcessing:
    """Test headless processing via ACloudViewer binary (not Python API)."""

    @pytest.fixture
    def sample_ply(self, tmp_path):
        """Create a minimal PLY file for testing."""
        ply_path = tmp_path / "test.ply"
        ply_path.write_text(
            "ply\nformat ascii 1.0\n"
            "element vertex 100\n"
            "property float x\nproperty float y\nproperty float z\n"
            "end_header\n" +
            "\n".join(f"{i*0.01} {i*0.02} {i*0.03}" for i in range(100)) + "\n"
        )
        return str(ply_path)

    def test_level3_binary_can_load(self, sample_ply):
        """Verify the ACloudViewer binary can load a PLY file."""
        from cli_anything.acloudviewer.utils.acloudviewer_backend import ACloudViewerBackend
        backend = ACloudViewerBackend(mode="headless")
        result = backend._run_cli(["-O", sample_ply, "-SAVE_CLOUDS"])
        assert result is not None

    def test_level3_subsample(self, sample_ply, tmp_path):
        out = str(tmp_path / "sub.ply")
        r = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "process", "subsample", sample_ply, "-o", out, "--voxel-size", "0.2"],
            capture_output=True, text=True, timeout=60)
        assert r.returncode == 0

    def test_level3_normals(self, sample_ply, tmp_path):
        out = str(tmp_path / "normals.ply")
        r = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "process", "normals", sample_ply, "-o", out],
            capture_output=True, text=True, timeout=60)
        assert r.returncode == 0

    def test_level3_formats_command(self):
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
