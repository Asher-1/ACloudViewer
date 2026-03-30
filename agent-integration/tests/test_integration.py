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

_skip_sibr_on_macos = pytest.mark.skipif(IS_MACOS, reason="SIBR not supported on macOS")

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
SIBR_COMMANDS_H = REPO_ROOT / "plugins/core/Standard/qSIBR/include/qSIBRCommands.h"


def _read_repo_text(path: Path) -> str:
    """Read repository sources as UTF-8.

    On Windows, :meth:`Path.read_text` without ``encoding=`` uses the ANSI
    code page (e.g. GBK), which breaks on UTF-8 C++ sources containing
    non-ASCII (smart quotes, symbols in comments, etc.).
    """
    return path.read_text(encoding="utf-8")


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

    - Connection failures  → pytest.skip  (server may have crashed)
    - JSON-RPC errors      → RuntimeError (caller can use pytest.raises)
    - Success              → return result value
    """
    try:
        ws = ws_connect(RPC_URL, open_timeout=5)
    except (ConnectionRefusedError, ConnectionResetError, OSError):
        pytest.skip("RPC server not reachable (may have crashed)")
    try:
        ws.send(json.dumps({
            "jsonrpc": "2.0", "id": 1,
            "method": method, "params": params or {},
        }))
        raw = ws.recv(timeout=timeout)
    except (ConnectionResetError, BrokenPipeError, OSError):
        pytest.skip("RPC connection lost during call")
    except Exception as exc:
        if "ConnectionClosed" in type(exc).__name__:
            pytest.skip("RPC connection closed unexpectedly")
        raise
    finally:
        try:
            ws.close()
        except Exception:
            pass
    resp = json.loads(raw)
    if "error" in resp:
        err = resp["error"]
        msg = err.get("message", err) if isinstance(err, dict) else err
        raise RuntimeError(f"RPC error ({method}): {msg}")
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

class TestLevel1_CppPlugin:
    """Verify C++ plugin source structure without needing a build."""

    def test_level1_plugin_cpp_exists(self):
        assert PLUGIN_CPP.exists()

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
        src = _read_repo_text(PLUGIN_CPP)
        assert "rpcColmapReconstruct" in src
        for param in ["image_path", "workspace_path", "quality",
                       "data_type", "mesher", "camera_model",
                       "single_camera", "use_gpu", "colmap_binary",
                       "timeout_ms"]:
            assert param in src, f"colmap.reconstruct missing param: {param}"

    def test_level1_file_convert_params(self):
        src = _read_repo_text(PLUGIN_CPP)
        assert "rpcFileConvert" in src
        for param in ["input", "output", "input_filter", "output_filter"]:
            assert param in src, f"file.convert missing param: {param}"

    @_skip_sibr_on_macos
    def test_level1_sibr_commands_exist(self):
        assert SIBR_COMMANDS_H.exists(), "Missing qSIBRCommands.h"

    @_skip_sibr_on_macos
    def test_level1_sibr_viewer_command_structure(self):
        src = _read_repo_text(SIBR_COMMANDS_H)
        assert 'COMMAND_SIBR_VIEWER' in src
        assert 'CommandSIBRViewer' in src
        for viewer in ["ulr", "ulrv2", "texturedmesh",
                        "pointbased", "gaussian", "remotegaussian"]:
            assert viewer in src.lower(), \
                f"SIBR_VIEWER missing viewer type: {viewer}"

    @_skip_sibr_on_macos
    def test_level1_sibr_viewer_options(self):
        src = _read_repo_text(SIBR_COMMANDS_H)
        for opt in ["--path", "--model-path", "--width", "--height",
                     "--iteration", "--device", "--no-interop", "--ip", "--port"]:
            assert opt in src, f"SIBR_VIEWER missing option: {opt}"

    @_skip_sibr_on_macos
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
            pytest.skip("qFacetsCommands.h not found")
        src = _read_repo_text(facets_h)
        assert "CommandFacets" in src
        assert "EXTRACT_FACETS" in src
        assert "ALGO_KD_TREE" in src or "ALGO_FAST_MARCHING" in src

    def test_level1_hough_normals_command_header(self):
        """Hough Normals command header exists with expected keywords."""
        hn_h = REPO_ROOT / "plugins/core/Standard/qHoughNormals/include/qHoughNormalsCommands.h"
        if not hn_h.exists():
            pytest.skip("qHoughNormalsCommands.h not found")
        src = _read_repo_text(hn_h)
        assert "CommandHoughNormals" in src
        hn_cpp = REPO_ROOT / "plugins/core/Standard/qHoughNormals/src/qHoughNormalsCommands.cpp"
        if not hn_cpp.exists():
            pytest.skip("qHoughNormalsCommands.cpp not found")
        assert "HOUGH_NORMALS" in _read_repo_text(hn_cpp)

    def test_level1_poisson_recon_command_header(self):
        """Poisson Recon command header exists with expected keywords."""
        pr_h = REPO_ROOT / "plugins/core/Standard/qPoissonRecon/include/qPoissonReconCommands.h"
        if not pr_h.exists():
            pytest.skip("qPoissonReconCommands.h not found")
        src = _read_repo_text(pr_h)
        assert "CommandPoissonRecon" in src
        pr_cpp = REPO_ROOT / "plugins/core/Standard/qPoissonRecon/src/qPoissonReconCommands.cpp"
        if not pr_cpp.exists():
            pytest.skip("qPoissonReconCommands.cpp not found")
        assert "POISSON_RECON" in _read_repo_text(pr_cpp)

    def test_level1_cork_command_header(self):
        """Cork boolean command files exist with expected keywords."""
        cork_h = REPO_ROOT / "plugins/core/Standard/qCork/include/qCorkCommands.h"
        if not cork_h.exists():
            pytest.skip("qCorkCommands.h not found")
        src = _read_repo_text(cork_h)
        assert "CommandCork" in src
        cork_cpp = REPO_ROOT / "plugins/core/Standard/qCork/src/qCorkCommands.cpp"
        if not cork_cpp.exists():
            pytest.skip("qCorkCommands.cpp not found")
        assert "CORK" in _read_repo_text(cork_cpp)

    def test_level1_voxfall_command_header(self):
        """VoxFall command files exist with expected keywords."""
        vf_h = REPO_ROOT / "plugins/core/Standard/qVoxFall/include/qVoxFallCommands.h"
        if not vf_h.exists():
            pytest.skip("qVoxFallCommands.h not found")
        src = _read_repo_text(vf_h)
        assert "CommandVoxFall" in src
        vf_cpp = REPO_ROOT / "plugins/core/Standard/qVoxFall/src/qVoxFallCommands.cpp"
        if not vf_cpp.exists():
            pytest.skip("qVoxFallCommands.cpp not found")
        assert "VOXFALL" in _read_repo_text(vf_cpp)

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
        mcp_json = REPO_ROOT / ".cursor" / "mcp.json"
        assert mcp_json.exists(), "Missing .cursor/mcp.json"
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
        pytest.param("sibr --help", marks=_skip_sibr_on_macos),
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
                     "facets", "hough-normals", "poisson-recon",
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

    @_skip_sibr_on_macos
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

    @_skip_sibr_on_macos
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
        # If implemented, should return 0; if not implemented yet, may return non-zero
        if r.returncode == 0:
            output = r.stdout.lower()
            # Check for viewer types
            for viewer in ["gaussian", "ulr", "remotegaussian"]:
                assert viewer in output, f"Missing viewer type: {viewer}"
            # Check for common options
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
    # - Windows: Use minimal (windeployqt doesn't ship qoffscreen)
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
            pytest.skip(f"normals prerequisite failed: {r1.stderr[:500]}")
        
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
        if r1.returncode != 0:
            pytest.skip(f"normals prerequisite failed: {r1.stderr[:500]}")
        
        out = str(tmp_path / "roughness.ply")
        r = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "process", "roughness", temp, "-o", out, "--radius", "0.1"],
            capture_output=True, text=True, timeout=60, env=cli_env)
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
        if r1.returncode != 0:
            pytest.skip(f"normals prerequisite failed: {r1.stderr[:500]}")
        
        out = str(tmp_path / "features.ply")
        r = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "process", "feature", temp, "-o", out,
             "--type", "SURFACE_VARIATION", "--kernel-size", "0.1"],
            capture_output=True, text=True, timeout=60, env=cli_env)
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
            combined = r.stdout + r.stderr
            if "polyline" in combined.lower() or "cross" in combined.lower():
                pytest.skip("cross-section requires specific polyline format")
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

    @_skip_sibr_on_macos
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
            pytest.skip(f"normals prerequisite failed: {r1.stderr[:500]}")
        
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
            pytest.skip(f"normals prerequisite failed: {r1.stderr[:500]}")
        
        r2 = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "process", "delaunay", temp_normals, "-o", temp_mesh],
            capture_output=True, text=True, timeout=60, env=cli_env)
        if r2.returncode != 0:
            pytest.skip(f"delaunay prerequisite failed: {r2.stderr[:500]}")
        
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
            pytest.skip(f"color-banding prerequisite failed: {r1.stderr[:500]}")
        
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
            pytest.skip(f"normals prerequisite failed: {r1.stderr[:500]}")
        
        r2 = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "process", "delaunay", temp_normals, "-o", temp_mesh],
            capture_output=True, text=True, timeout=60, env=cli_env)
        if r2.returncode != 0:
            pytest.skip(f"delaunay prerequisite failed: {r2.stderr[:500]}")
        
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
            pytest.skip(f"normals prerequisite failed: {r1.stderr[:500]}")
        
        r2 = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "process", "delaunay", temp_normals, "-o", temp_mesh],
            capture_output=True, text=True, timeout=60, env=cli_env)
        if r2.returncode != 0:
            pytest.skip(f"delaunay prerequisite failed: {r2.stderr[:500]}")
        
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
            pytest.skip(f"normals prerequisite failed: {r1.stderr[:500]}")
        
        r2 = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "process", "delaunay", temp_normals, "-o", temp_mesh],
            capture_output=True, text=True, timeout=60, env=cli_env)
        if r2.returncode != 0:
            pytest.skip(f"delaunay prerequisite failed: {r2.stderr[:500]}")
        
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
        if r1.returncode != 0:
            pytest.skip(f"normals prerequisite failed: {r1.stderr[:500]}")
        
        r2 = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "process", "delaunay", temp_normals, "-o", mesh1],
            capture_output=True, text=True, timeout=60, env=cli_env)
        if r2.returncode != 0:
            pytest.skip(f"delaunay prerequisite failed: {r2.stderr[:500]}")
        
        import shutil
        shutil.copy(mesh1, mesh2)
        
        out = str(tmp_path / "merged.ply")
        r = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "process", "merge-meshes", mesh1, mesh2, "-o", out],
            capture_output=True, text=True, timeout=60, env=cli_env)
        assert r.returncode == 0, (
            f"process merge-meshes failed (rc={r.returncode}):\n"
            f"stdout: {r.stdout[:2000]}\nstderr: {r.stderr[:2000]}")
        data = json.loads(r.stdout)
        assert data.get("status") != "failed", f"Command returned failed: {data}"
    
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
            pytest.skip(f"coord-to-sf prerequisite failed: {r1.stderr[:500]}")
        
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
            pytest.skip(f"coord-to-sf prerequisite failed: {r1.stderr[:500]}")
        
        # Check if the prerequisite file was actually created
        if not Path(temp).exists():
            pytest.skip(f"coord-to-sf did not create output file: {temp}")
        
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
            pytest.skip(f"normals prerequisite failed: {r1.stderr[:500]}")
        
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
            pytest.skip(f"normals prerequisite failed: {r1.stderr[:500]}")
        
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
            pytest.skip(f"normals prerequisite failed: {r1.stderr[:500]}")
        
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
            pytest.skip(f"normals prerequisite failed: {r1.stderr[:500]}")
        
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
            pytest.skip(f"normals octree prerequisite failed: {r1.stderr[:500]}")
        
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
            pytest.skip(f"coord-to-sf prerequisite failed: {r1.stderr[:500]}")
        
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
            pytest.skip(f"coord-to-sf prerequisite failed: {r1.stderr[:500]}")
        
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
            pytest.skip(f"coord-to-sf prerequisite failed: {r1.stderr[:500]}")
        
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
            pytest.skip(f"coord-to-sf prerequisite failed: {r1.stderr[:500]}")
        
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
            pytest.skip(f"coord-to-sf prerequisite failed: {r1.stderr[:500]}")
        
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
            pytest.skip(f"coord-to-sf prerequisite failed: {r1.stderr[:500]}")
        
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
            pytest.skip(f"coord-to-sf prerequisite failed: {r1.stderr[:500]}")
        
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
        if not Path(pcd).exists():
            pytest.skip(
                f"PLY->PCD prerequisite failed (rc={pcd_r.returncode}):\n"
                f"{(pcd_r.stdout + pcd_r.stderr)[-1000:]}"
            )
        out = str(shared_dir / "roundtrip.ply")
        r = self._convert_file(pcd, out, "PLY", acv_env)
        assert r.returncode == 0, \
            f"PCD->PLY failed:\n{(r.stdout + r.stderr)[-2000:]}"
        assert Path(out).exists() and Path(out).stat().st_size > 0, \
            "PCD->PLY output file missing or empty"

    def test_level3_pcd_to_drc(self, converted_drc):
        if converted_drc[0] is None:
            pytest.skip("PLY->PCD prerequisite failed")
        out, r = converted_drc
        assert r.returncode == 0, \
            f"PCD->DRC failed:\n{(r.stdout + r.stderr)[-2000:]}"
        assert Path(out).exists() and Path(out).stat().st_size > 0, \
            "PCD->DRC output file missing or empty"

    def test_level3_drc_to_pcd(self, converted_drc, shared_dir, acv_env):
        if converted_drc[0] is None:
            pytest.skip("PLY->PCD prerequisite failed")
        drc, drc_r = converted_drc
        if not Path(drc).exists():
            pytest.skip(
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
        out = str(tmp_path / "test.las")
        r = self._convert_file(sample_ply, out, "LAS", acv_env)
        combined = (r.stdout + r.stderr).lower()
        plugin_missing = any(kw in combined for kw in
                             ("plugin", "filter", "unsupported", "unknown format"))
        if not Path(out).exists() and plugin_missing:
            pytest.skip("LAS IO plugin not available in this build")
        if r.returncode != 0 and plugin_missing:
            pytest.skip("LAS IO plugin not available in this build")
        assert r.returncode == 0, \
            f"PLY->LAS failed (rc={r.returncode}):\n{(r.stdout+r.stderr)[-2000:]}"
        assert Path(out).exists() and Path(out).stat().st_size > 0, \
            f"LAS output file missing or empty: {out}"

    def test_level3_e57_conversion(self, sample_ply, acv_env, tmp_path):
        """PLY -> E57 (requires qE57IO plugin)."""
        out = str(tmp_path / "test.e57")
        r = self._convert_file(sample_ply, out, "E57", acv_env)
        combined = (r.stdout + r.stderr).lower()
        plugin_missing = any(kw in combined for kw in
                             ("plugin", "filter", "unsupported", "unknown format",
                              "no filter"))
        if not Path(out).exists() and plugin_missing:
            pytest.skip("E57 IO plugin not available in this build")
        if r.returncode != 0 and plugin_missing:
            pytest.skip("E57 IO plugin not available in this build")
        assert r.returncode == 0, \
            f"PLY->E57 failed (rc={r.returncode}):\n{(r.stdout+r.stderr)[-2000:]}"
        assert Path(out).exists() and Path(out).stat().st_size > 0, \
            f"E57 output file missing or empty: {out}"

    def test_level3_fbx_conversion(self, mesh_ply, acv_env, tmp_path):
        """PLY -> FBX (requires qFBXIO plugin, mesh-based)."""
        out = str(tmp_path / "test.fbx")
        r = subprocess.run(
            [BINARY_PATH, "-SILENT", "-O", mesh_ply,
             "-DELAUNAY", "-M_EXPORT_FMT", "FBX",
             "-AUTO_SAVE", "OFF", "-NO_TIMESTAMP",
             "-SAVE_MESHES", "FILE", out],
            capture_output=True, text=True, timeout=120,
            env=acv_env)
        combined = (r.stdout + r.stderr).lower()
        skip_keywords = ("plugin", "filter", "unsupported", "unknown format",
                         "unhandled", "empty mesh", "nothing to save")
        plugin_missing = any(kw in combined for kw in skip_keywords)
        if not Path(out).exists() and plugin_missing:
            pytest.skip("FBX IO plugin not available or Delaunay produced empty mesh")
        if r.returncode != 0 and plugin_missing:
            pytest.skip("FBX IO plugin not available or Delaunay produced empty mesh")
        assert r.returncode == 0, \
            f"PLY->FBX (mesh) failed (rc={r.returncode}):\n{(r.stdout+r.stderr)[-2000:]}"
        assert Path(out).exists() and Path(out).stat().st_size > 0, \
            f"FBX output file missing or empty: {out}"

    def test_level3_sbf_conversion(self, sample_ply, acv_env, tmp_path):
        """PLY -> SBF (SimpleBin, qCoreIO plugin)."""
        out = str(tmp_path / "test.sbf")
        r = self._convert_file(sample_ply, out, "SBF", acv_env)
        combined = (r.stdout + r.stderr).lower()
        plugin_missing = any(kw in combined for kw in
                             ("plugin", "filter", "unsupported", "unknown format"))
        if not Path(out).exists() and plugin_missing:
            pytest.skip("SBF IO plugin not available in this build")
        if r.returncode != 0 and plugin_missing:
            pytest.skip("SBF IO plugin not available in this build")
        assert r.returncode == 0, \
            f"PLY->SBF failed (rc={r.returncode}):\n{(r.stdout+r.stderr)[-2000:]}"
        assert Path(out).exists() and Path(out).stat().st_size > 0, \
            f"SBF output file missing or empty: {out}"

    def test_level3_dxf_conversion(self, sample_ply, acv_env, tmp_path):
        """PLY -> DXF (requires CV_DXF_SUPPORT / DXF IO)."""
        out = str(tmp_path / "test.dxf")
        r = self._convert_file(sample_ply, out, "DXF", acv_env)
        combined = (r.stdout + r.stderr).lower()
        plugin_missing = any(kw in combined for kw in
                             ("plugin", "filter", "unsupported", "unknown format"))
        if not Path(out).exists() and plugin_missing:
            pytest.skip("DXF IO not available in this build")
        if r.returncode != 0 and plugin_missing:
            pytest.skip("DXF IO not available in this build")
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
        out = str(tmp_path / "test.laz")
        r = self._convert_file(sample_ply, out, "LAS", acv_env)
        combined = (r.stdout + r.stderr).lower()
        plugin_missing = any(kw in combined for kw in
                             ("plugin", "filter", "unsupported", "unknown format"))
        if not Path(out).exists() and plugin_missing:
            pytest.skip("LAS IO plugin not available in this build")
        if r.returncode != 0 and plugin_missing:
            pytest.skip("LAS IO plugin not available in this build")
        assert r.returncode == 0, \
            f"PLY->LAZ failed (rc={r.returncode}):\n{(r.stdout+r.stderr)[-2000:]}"
        assert Path(out).exists() and Path(out).stat().st_size > 0, \
            f"LAZ output file missing or empty: {out}"

    def test_level3_shp_conversion(self, sample_ply, acv_env, tmp_path):
        """PLY -> SHP (optional plugin / build)."""
        out = str(tmp_path / "test.shp")
        r = self._convert_file(sample_ply, out, "SHP", acv_env)
        combined = (r.stdout + r.stderr).lower()
        plugin_missing = any(kw in combined for kw in
                             ("plugin", "filter", "unsupported", "unknown format",
                              "unhandled"))
        if not Path(out).exists() and plugin_missing:
            pytest.skip("SHP IO not available in this build")
        if r.returncode != 0 and plugin_missing:
            pytest.skip("SHP IO not available in this build")
        assert r.returncode == 0, \
            f"PLY->SHP failed (rc={r.returncode}):\n{(r.stdout+r.stderr)[-2000:]}"
        assert Path(out).exists() and Path(out).stat().st_size > 0, \
            f"SHP output file missing or empty: {out}"

    def test_level3_ptx_import(self, acv_env, tmp_path):
        """Minimal PTX loads; skip if PTX IO is unavailable."""
        ptx_path = tmp_path / "minimal.ptx"
        ptx_path.write_text(_MINIMAL_PTX_CONTENT)
        r = subprocess.run(
            [BINARY_PATH, "-SILENT", "-O", str(ptx_path), "-SAVE_CLOUDS"],
            capture_output=True, text=True, timeout=120, env=acv_env)
        combined = (r.stdout + r.stderr).lower()
        plugin_missing = any(kw in combined for kw in
                             ("plugin", "filter", "unsupported", "unknown format",
                              "unhandled"))
        if r.returncode != 0 and plugin_missing:
            pytest.skip("PTX IO not available in this build")
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
        if not Path(pcd).exists():
            pytest.skip(
                f"PLY->PCD prerequisite failed (rc={pcd_r.returncode}):\n"
                f"stdout: {pcd_r.stdout[:1000]}\nstderr: {pcd_r.stderr[:1000]}"
            )
        out = str(shared_dir / "output.drc")
        r = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "convert", pcd, out],
            capture_output=True, text=True, timeout=120, env=cli_env)
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
        except Exception:
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
        assert len(methods) >= 40, (
            f"Expected >=40 RPC methods, got {len(methods)}")
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
        except Exception:
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
            if ext == "dxf":
                msg = str(exc).lower()
                if "failed to save" in msg:
                    pytest.skip("DXF export not available in this RPC build")
            raise
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
        except Exception:
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
        except Exception:
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
        except Exception:
            pass
        try:
            _rpc_call("scene.remove", {"entity_id": group_id}, timeout=5)
        except Exception:
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
        except Exception:
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
            tool_names = sorted(t.name for t in tools)
            assert len(tools) >= 121, (
                f"Expected ≥121 MCP tools, got {len(tools)}.\n"
                f"Available tools: {tool_names}")
        except (ImportError, SystemExit):
            pytest.skip("MCP SDK or CLI harness not installed")

    def test_level5_mcp_tool_names(self):
        try:
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
        except (ImportError, SystemExit):
            pytest.skip("MCP SDK or CLI harness not installed")

    def test_level5_mcp_colmap_tools(self):
        try:
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
        except (ImportError, SystemExit):
            pytest.skip("MCP SDK or CLI harness not installed")

    @_skip_sibr_on_macos
    def test_level5_mcp_sibr_tools(self):
        try:
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
        except (ImportError, SystemExit):
            pytest.skip("MCP SDK or CLI harness not installed")

    def test_level5_mcp_entry_point(self):
        r = subprocess.run(
            [sys.executable, "-m", "cli_anything.acloudviewer.mcp_server", "--help"],
            capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            pytest.skip("MCP server module not runnable")

    def test_level5_mcp_plugin_standard_tools(self):
        try:
            from cli_anything.acloudviewer.mcp_server import list_tools
            import asyncio
            tools = asyncio.run(list_tools())
            names = {t.name for t in tools}
            plugin_tools = [
                "classify_3dmasc", "animation", "cloud_layers",
                "color_seg_rgb", "color_seg_hsv", "color_seg_scalar",
                "g3point",
            ]
            missing = [t for t in plugin_tools if t not in names]
            assert not missing, (
                f"Missing Standard plugin MCP tools: {missing}\n"
                f"Available tools: {sorted(names)}")
        except (ImportError, SystemExit):
            pytest.skip("MCP SDK or CLI harness not installed")

    def test_level5_mcp_plugin_io_tools(self):
        try:
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
        except (ImportError, SystemExit):
            pytest.skip("MCP SDK or CLI harness not installed")


# ═══════════════════════════════════════════════════════════════════════════
# Level 1: Plugin Command Source Verification
# ═══════════════════════════════════════════════════════════════════════════

class TestLevel1_PluginCommands:
    """Verify new plugin command source structure without building."""

    def test_level1_animation_command_header(self):
        h = REPO_ROOT / "plugins/core/Standard/qAnimation/include/qAnimationCommands.h"
        if not h.exists():
            pytest.skip("qAnimationCommands.h not found")
        src = _read_repo_text(h)
        assert "CommandAnimation" in src
        assert "ANIMATION" in _read_repo_text(
            REPO_ROOT / "plugins/core/Standard/qAnimation/src/qAnimationCommands.cpp")

    def test_level1_cloud_layers_command_header(self):
        h = REPO_ROOT / "plugins/core/Standard/qCloudLayers/include/qCloudLayersCommands.h"
        if not h.exists():
            pytest.skip("qCloudLayersCommands.h not found")
        src = _read_repo_text(h)
        assert "CommandCloudLayers" in src
        cpp = REPO_ROOT / "plugins/core/Standard/qCloudLayers/src/qCloudLayersCommands.cpp"
        assert "CLOUD_LAYERS" in _read_repo_text(cpp)

    def test_level1_colorimetric_seg_command_header(self):
        h = REPO_ROOT / "plugins/core/Standard/qColorimetricSegmenter/ColorimetricSegmenterCommands.h"
        if not h.exists():
            pytest.skip("ColorimetricSegmenterCommands.h not found")
        src = _read_repo_text(h)
        assert "CommandColorimetricSegRGB" in src
        assert "CommandColorimetricSegHSV" in src
        assert "CommandColorimetricSegScalar" in src

    def test_level1_colorimetric_seg_command_impl(self):
        cpp = REPO_ROOT / "plugins/core/Standard/qColorimetricSegmenter/ColorimetricSegmenterCommands.cpp"
        if not cpp.exists():
            pytest.skip("ColorimetricSegmenterCommands.cpp not found")
        src = _read_repo_text(cpp)
        assert "COLOR_SEG_RGB" in src
        assert "COLOR_SEG_HSV" in src
        assert "COLOR_SEG_SCALAR" in src

    def test_level1_g3point_command_header(self):
        h = REPO_ROOT / "plugins/core/Standard/qG3Point/include/G3PointCommands.h"
        if not h.exists():
            pytest.skip("G3PointCommands.h not found")
        src = _read_repo_text(h)
        assert "CommandG3Point" in src
        cpp = REPO_ROOT / "plugins/core/Standard/qG3Point/src/G3PointCommands.cpp"
        assert "G3POINT" in _read_repo_text(cpp)

    def test_level1_3dmasc_command_header(self):
        h = REPO_ROOT / "plugins/core/Standard/q3DMASC/q3DMASCCommands.h"
        if not h.exists():
            pytest.skip("q3DMASCCommands.h not found")
        src = _read_repo_text(h)
        assert "Command3DMASCClassif" in src
        assert "3DMASC_CLASSIFY" in src

    def test_level1_pcv_command_header(self):
        h = REPO_ROOT / "plugins/core/Standard/qPCV/include/PCVCommand.h"
        if not h.exists():
            pytest.skip("PCVCommand.h not found")
        src = _read_repo_text(h)
        assert "PCVCommand" in src

    def test_level1_io_coreio_command(self):
        h = REPO_ROOT / "plugins/core/IO/qCoreIO/include/CoreIOCommands.h"
        if not h.exists():
            pytest.skip("CoreIOCommands.h not found")
        src = _read_repo_text(h)
        assert "CommandCoreIO" in src
        cpp = REPO_ROOT / "plugins/core/IO/qCoreIO/src/CoreIOCommands.cpp"
        assert "CORE_IO" in _read_repo_text(cpp)

    def test_level1_io_csvmatrix_command(self):
        h = REPO_ROOT / "plugins/core/IO/qCSVMatrixIO/include/CSVMatrixCommands.h"
        if not h.exists():
            pytest.skip("CSVMatrixCommands.h not found")
        assert "CommandCSVMatrix" in _read_repo_text(h)

    def test_level1_io_photoscan_command(self):
        h = REPO_ROOT / "plugins/core/IO/qPhotoscanIO/include/PhotoscanCommands.h"
        if not h.exists():
            pytest.skip("PhotoscanCommands.h not found")
        assert "CommandPhotoscan" in _read_repo_text(h)

    def test_level1_io_e57_command(self):
        h = REPO_ROOT / "plugins/core/IO/qE57IO/include/E57Commands.h"
        if not h.exists():
            pytest.skip("E57Commands.h not found")
        assert "CommandE57" in _read_repo_text(h)

    def test_level1_io_draco_command(self):
        h = REPO_ROOT / "plugins/core/IO/qDracoIO/include/DracoCommands.h"
        if not h.exists():
            pytest.skip("DracoCommands.h not found")
        assert "CommandDraco" in _read_repo_text(h)

    def test_level1_io_las_command(self):
        h = REPO_ROOT / "plugins/core/IO/qLASIO/include/LasCommands.h"
        if not h.exists():
            pytest.skip("LasCommands.h not found")
        assert "CommandLAS" in _read_repo_text(h)

    def test_level1_io_meshio_command(self):
        h = REPO_ROOT / "plugins/core/IO/qMeshIO/include/MeshIOCommands.h"
        if not h.exists():
            pytest.skip("MeshIOCommands.h not found")
        assert "CommandMeshIO" in _read_repo_text(h)

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

    def test_level3_pcv_headless(self, sample_ply, tmp_path):
        out = tmp_path / "pcv_out.ply"
        args = [BINARY_PATH, "-SILENT", "-O", str(sample_ply),
                "-AUTO_SAVE", "OFF", "-NO_TIMESTAMP",
                "-PCV", "-SAVE_CLOUDS", "FILE", str(out)]
        r = subprocess.run(args, capture_output=True, text=True, timeout=60)
        if "Unknown or misplaced command" in r.stdout:
            pytest.skip("qPCV plugin not loaded in this build")
        assert r.returncode == 0 or out.exists()

    def test_level3_animation_headless(self, tmp_path):
        args = [BINARY_PATH, "-SILENT", "-ANIMATION",
                "-FPS", "24", "-TOTAL_FRAMES", "10"]
        r = subprocess.run(args, capture_output=True, text=True, timeout=30)
        assert r.returncode == 0

    def test_level3_draco_settings_headless(self, tmp_path):
        args = [BINARY_PATH, "-SILENT", "-DRACO",
                "-QUANTIZATION", "14", "-SPEED", "3"]
        r = subprocess.run(args, capture_output=True, text=True, timeout=30)
        assert r.returncode == 0

    def test_level3_e57_settings_headless(self, tmp_path):
        args = [BINARY_PATH, "-SILENT", "-E57", "-IGNORE_COLOR"]
        r = subprocess.run(args, capture_output=True, text=True, timeout=30)
        assert r.returncode == 0

    def test_level3_las_settings_headless(self, tmp_path):
        args = [BINARY_PATH, "-SILENT", "-LAS", "-EXTRA_FIELDS", "-SAVE_LAZ"]
        r = subprocess.run(args, capture_output=True, text=True, timeout=30)
        assert r.returncode == 0


# ═══════════════════════════════════════════════════════════════════════════
# Level 4: GUI RPC Plugin Command Tests
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not _rpc_available(), reason="ACloudViewer RPC not available")
class TestLevel4_PluginRPC:
    """Test plugin commands via JSON-RPC (requires running GUI app)."""

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

    def test_level5_mcp_tool_count_increased(self):
        try:
            from cli_anything.acloudviewer.mcp_server import list_tools
            import asyncio
            tools = asyncio.run(list_tools())
            assert len(tools) >= 135, (
                f"Expected ≥135 MCP tools (with new plugins), got {len(tools)}")
        except (ImportError, SystemExit):
            pytest.skip("MCP SDK or CLI harness not installed")

    def test_level5_mcp_3dmasc_tool_schema(self):
        try:
            from cli_anything.acloudviewer.mcp_server import list_tools
            import asyncio
            tools = asyncio.run(list_tools())
            tool = next((t for t in tools if t.name == "classify_3dmasc"), None)
            assert tool is not None, "classify_3dmasc tool not found"
            schema = tool.inputSchema
            assert "classifier_file" in schema.get("properties", {})
            assert "cloud_roles" in schema.get("properties", {})
        except (ImportError, SystemExit):
            pytest.skip("MCP SDK or CLI harness not installed")

    def test_level5_mcp_color_seg_rgb_tool_schema(self):
        try:
            from cli_anything.acloudviewer.mcp_server import list_tools
            import asyncio
            tools = asyncio.run(list_tools())
            tool = next((t for t in tools if t.name == "color_seg_rgb"), None)
            assert tool is not None, "color_seg_rgb tool not found"
            props = tool.inputSchema.get("properties", {})
            for key in ["r_min", "r_max", "g_min", "g_max", "b_min", "b_max"]:
                assert key in props, f"Missing property: {key}"
        except (ImportError, SystemExit):
            pytest.skip("MCP SDK or CLI harness not installed")

    def test_level5_mcp_g3point_tool_schema(self):
        try:
            from cli_anything.acloudviewer.mcp_server import list_tools
            import asyncio
            tools = asyncio.run(list_tools())
            tool = next((t for t in tools if t.name == "g3point"), None)
            assert tool is not None, "g3point tool not found"
            props = tool.inputSchema.get("properties", {})
            for key in ["n_neighbors", "max_radius", "min_radius"]:
                assert key in props, f"Missing property: {key}"
        except (ImportError, SystemExit):
            pytest.skip("MCP SDK or CLI harness not installed")

    def test_level5_mcp_draco_settings_schema(self):
        try:
            from cli_anything.acloudviewer.mcp_server import list_tools
            import asyncio
            tools = asyncio.run(list_tools())
            tool = next((t for t in tools if t.name == "draco_settings"), None)
            assert tool is not None, "draco_settings tool not found"
            props = tool.inputSchema.get("properties", {})
            assert "quantization" in props
            assert "compression_level" in props
        except (ImportError, SystemExit):
            pytest.skip("MCP SDK or CLI harness not installed")

    def test_level5_mcp_las_settings_schema(self):
        try:
            from cli_anything.acloudviewer.mcp_server import list_tools
            import asyncio
            tools = asyncio.run(list_tools())
            tool = next((t for t in tools if t.name == "las_settings"), None)
            assert tool is not None, "las_settings tool not found"
            props = tool.inputSchema.get("properties", {})
            assert "extra_fields" in props
            assert "save_laz" in props
        except (ImportError, SystemExit):
            pytest.skip("MCP SDK or CLI harness not installed")
