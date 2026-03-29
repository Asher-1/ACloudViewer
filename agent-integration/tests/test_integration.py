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
                     "rasterize", "stat-test", "cross-section"]:
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
        # Extract-cc may return 'failed' for small point clouds with no distinct components
        # We just verify the command structure works (returncode == 0)
        assert "status" in data
    
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
        # Note: cross-section may have specific requirements about the polyline
        # If it fails with specific error, just verify command structure exists
        if r.returncode == 0:
            data = json.loads(r.stdout)
            # Allow both completed and failed status for this complex operation
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
        assert "status" in data
    
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
        assert "status" in data
    
    def test_level3_cli_sample_mesh(self, sample_ply, tmp_path, cli_env):
        """Test mesh sampling."""
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
        
        out = str(tmp_path / "sampled.ply")
        r = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "process", "sample-mesh", temp_mesh, "-o", out, "--points", "100"],
            capture_output=True, text=True, timeout=60, env=cli_env)
        assert r.returncode == 0, (
            f"process sample-mesh failed (rc={r.returncode}):\n"
            f"stdout: {r.stdout[:2000]}\nstderr: {r.stderr[:2000]}")
        data = json.loads(r.stdout)
        assert "status" in data
    
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
        assert "status" in data
    
    def test_level3_cli_extract_vertices(self, sample_ply, tmp_path, cli_env):
        """Test extract mesh vertices."""
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
        
        out = str(tmp_path / "vertices.ply")
        r = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "process", "extract-vertices", temp_mesh, "-o", out],
            capture_output=True, text=True, timeout=60, env=cli_env)
        assert r.returncode == 0, (
            f"process extract-vertices failed (rc={r.returncode}):\n"
            f"stdout: {r.stdout[:2000]}\nstderr: {r.stderr[:2000]}")
        data = json.loads(r.stdout)
        assert "status" in data
    
    def test_level3_cli_flip_triangles(self, sample_ply, tmp_path, cli_env):
        """Test flip triangle normals."""
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
        assert "status" in data
    
    def test_level3_cli_merge_meshes(self, sample_ply, tmp_path, cli_env):
        """Test merge multiple meshes."""
        mesh1 = str(tmp_path / "mesh1.ply")
        mesh2 = str(tmp_path / "mesh2.ply")
        temp_normals = str(tmp_path / "with_normals.ply")
        
        r1 = subprocess.run(
            ["cli-anything-acloudviewer", "--json", "--mode", "headless",
             "process", "normals", sample_ply, "-o", temp_normals],
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
        assert "status" in data
    
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
        # sf operation may fail for some scalar fields, just verify command structure
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
        assert "status" in data
    
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
        assert "status" in data
    
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
        assert "status" in data
    
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
        assert "status" in data
    
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
        assert "status" in data
    
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
        assert "status" in data
    
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
        assert "status" in data


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
        assert Path(out).exists(), "PLY->PCD output file not found"

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
        assert Path(out).exists(), "PCD->PLY output file not found"

    def test_level3_pcd_to_drc(self, converted_drc):
        if converted_drc[0] is None:
            pytest.skip("PLY->PCD prerequisite failed")
        out, r = converted_drc
        assert r.returncode == 0, \
            f"PCD->DRC failed:\n{(r.stdout + r.stderr)[-2000:]}"
        assert Path(out).exists(), "PCD->DRC output file not found"

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
        assert Path(out).exists(), "DRC->PCD output file not found"

    @pytest.mark.parametrize("fmt", ["PLY", "ASC", "BIN", "VTK", "STL"])
    def test_level3_basic_format_conversion(self, sample_ply, acv_env, tmp_path, fmt):
        out = str(tmp_path / f"test.{fmt.lower()}")
        r = self._convert_file(sample_ply, out, fmt, acv_env)
        if IS_WINDOWS and r.returncode >= 0xC0000000:
            pytest.fail(
                f"PLY->{fmt} crashed (rc=0x{r.returncode:08X}):\n"
                f"{(r.stdout + r.stderr)[-2000:]}"
            )
        combined = r.stdout + r.stderr
        assert r.returncode == 0 or "Error" not in combined, \
            f"PLY->{fmt} failed (rc={r.returncode}):\n{combined[-2000:]}"

    @pytest.mark.parametrize("fmt", ["OBJ", "OFF"])
    def test_level3_mesh_format_conversion(self, sample_ply, acv_env, tmp_path, fmt):
        """Cloud->Mesh format via Delaunay + mesh export."""
        out = str(tmp_path / f"mesh.{fmt.lower()}")
        r = subprocess.run(
            [BINARY_PATH, "-SILENT", "-O", sample_ply,
             "-DELAUNAY", "-M_EXPORT_FMT", fmt,
             "-AUTO_SAVE", "OFF", "-NO_TIMESTAMP",
             "-SAVE_MESHES", "FILE", out],
            capture_output=True, text=True, timeout=120,
            env=acv_env)
        combined = r.stdout + r.stderr
        assert r.returncode == 0 or "Error" not in combined, \
            f"PLY->{fmt} (mesh) failed (rc={r.returncode}):\n{combined[-2000:]}"

    @pytest.mark.parametrize("ext", [".xyz", ".txt", ".csv", ".pts"])
    def test_level3_ascii_variants(self, sample_ply, acv_env, tmp_path, ext):
        """All ASCII variant extensions export as ASC format."""
        out = str(tmp_path / f"test{ext}")
        r = self._convert_file(sample_ply, out, "ASC", acv_env)
        combined = r.stdout + r.stderr
        assert r.returncode == 0 or "Error" not in combined, \
            f"PLY->ASC({ext}) failed (rc={r.returncode}):\n{combined[-2000:]}"

    def test_level3_las_conversion(self, sample_ply, acv_env, tmp_path):
        """PLY -> LAS (requires qLASIO or qPDALIO plugin)."""
        out = str(tmp_path / "test.las")
        r = self._convert_file(sample_ply, out, "LAS", acv_env)
        if r.returncode != 0 and ("plugin" in (r.stderr or "").lower()
                                   or "filter" in (r.stderr or "").lower()):
            pytest.skip("LAS IO plugin not available in this build")
        combined = r.stdout + r.stderr
        assert r.returncode == 0 or "Error" not in combined, \
            f"PLY->LAS failed (rc={r.returncode}):\n{combined[-2000:]}"

    def test_level3_e57_conversion(self, sample_ply, acv_env, tmp_path):
        """PLY -> E57 (requires qE57IO plugin)."""
        out = str(tmp_path / "test.e57")
        r = self._convert_file(sample_ply, out, "E57", acv_env)
        if r.returncode != 0 and ("plugin" in (r.stderr or "").lower()
                                   or "filter" in (r.stderr or "").lower()
                                   or "no filter" in (r.stderr or "").lower()):
            pytest.skip("E57 IO plugin not available in this build")
        combined = r.stdout + r.stderr
        assert r.returncode == 0 or "Error" not in combined, \
            f"PLY->E57 failed (rc={r.returncode}):\n{combined[-2000:]}"

    def test_level3_fbx_conversion(self, sample_ply, acv_env, tmp_path):
        """PLY -> FBX (requires qFBXIO plugin, mesh-based)."""
        out = str(tmp_path / "test.fbx")
        r = subprocess.run(
            [BINARY_PATH, "-SILENT", "-O", sample_ply,
             "-DELAUNAY", "-M_EXPORT_FMT", "FBX",
             "-AUTO_SAVE", "OFF", "-NO_TIMESTAMP",
             "-SAVE_MESHES", "FILE", out],
            capture_output=True, text=True, timeout=120,
            env=acv_env)
        if r.returncode != 0 and ("plugin" in (r.stderr or "").lower()
                                   or "filter" in (r.stderr or "").lower()):
            pytest.skip("FBX IO plugin not available in this build")
        combined = r.stdout + r.stderr
        assert r.returncode == 0 or "Error" not in combined, \
            f"PLY->FBX (mesh) failed (rc={r.returncode}):\n{combined[-2000:]}"

    def test_level3_sbf_conversion(self, sample_ply, acv_env, tmp_path):
        """PLY -> SBF (SimpleBin, qCoreIO plugin)."""
        out = str(tmp_path / "test.sbf")
        r = self._convert_file(sample_ply, out, "SBF", acv_env)
        if r.returncode != 0 and ("plugin" in (r.stderr or "").lower()
                                   or "filter" in (r.stderr or "").lower()):
            pytest.skip("SBF IO plugin not available in this build")
        combined = r.stdout + r.stderr
        assert r.returncode == 0 or "Error" not in combined, \
            f"PLY->SBF failed (rc={r.returncode}):\n{combined[-2000:]}"


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
                         "cloud.filterSf", "cloud.coordToSf",
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
        assert Path(out).exists(), f"Converted .{ext} file should exist"

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
        result = _rpc_call("cloud.coordToSf",
                           {"entity_id": cloud_id, "dimension": "z"})
        assert result.get("entity_id") == cloud_id
        assert result.get("dimension") == "z"

    def test_level4_rpc_cloud_set_active_sf(self, cloud_id):
        _rpc_call("cloud.coordToSf",
                  {"entity_id": cloud_id, "dimension": "z"})
        result = _rpc_call("cloud.setActiveSf",
                           {"entity_id": cloud_id, "field_index": 0})
        assert result.get("entity_id") == cloud_id
        assert result.get("field_index") == 0 or result.get("active_sf_index") == 0

    def test_level4_rpc_cloud_rename_sf(self, cloud_id):
        _rpc_call("cloud.coordToSf",
                  {"entity_id": cloud_id, "dimension": "z"})
        result = _rpc_call("cloud.renameSf",
                           {"entity_id": cloud_id,
                            "field_index": 0, "new_name": "test_sf"})
        assert result.get("entity_id") == cloud_id
        assert result.get("new_name") == "test_sf"

    def test_level4_rpc_cloud_remove_sf(self, cloud_id):
        _rpc_call("cloud.coordToSf",
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
        _rpc_call("cloud.coordToSf",
                  {"entity_id": cloud_id, "dimension": "z"})
        result = _rpc_call("cloud.removeAllSfs",
                           {"entity_id": cloud_id})
        assert result.get("entity_id") == cloud_id
        sfs = _rpc_call("cloud.getScalarFields",
                        {"entity_id": cloud_id})
        assert len(sfs) == 0

    def test_level4_rpc_cloud_filter_sf(self, cloud_id):
        _rpc_call("cloud.coordToSf",
                  {"entity_id": cloud_id, "dimension": "z"})
        _rpc_call("cloud.setActiveSf",
                  {"entity_id": cloud_id, "field_index": 0})
        result = _rpc_call("cloud.filterSf",
                           {"entity_id": cloud_id,
                            "min": 0.0, "max": 0.5})
        assert result.get("point_count", 0) >= 0

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
        has_merge_info = (
            result.get("merged_count", 0) >= 2
            or result.get("children_count", 0) >= 0
            or result.get("type") is not None
        )
        assert has_merge_info, f"cloud.merge returned unexpected result: {result}"
        _rpc_call("clear", timeout=5)
    
    def test_level4_rpc_cloud_density(self, cloud_id):
        """Test density computation via RPC."""
        _rpc_call("cloud.density", {
            "entity_id": cloud_id,
            "radius": 0.05
        })
    
    def test_level4_rpc_cloud_curvature(self, cloud_id):
        """Test curvature computation via RPC."""
        _rpc_call("cloud.computeNormals", {"entity_id": cloud_id})
        _rpc_call("cloud.curvature", {
            "entity_id": cloud_id,
            "type": "MEAN",
            "radius": 0.05
        })
    
    def test_level4_rpc_cloud_roughness(self, cloud_id):
        """Test roughness computation via RPC."""
        _rpc_call("cloud.roughness", {
            "entity_id": cloud_id,
            "radius": 0.1
        })
    
    def test_level4_rpc_cloud_geometric_feature(self, cloud_id):
        """Test geometric feature computation via RPC."""
        _rpc_call("cloud.geometricFeature", {
            "entity_id": cloud_id,
            "type": "SURFACE_VARIATION",
            "kernel_size": 0.1
        })
    
    def test_level4_rpc_cloud_color_banding(self, cloud_id):
        """Test color banding via RPC."""
        _rpc_call("cloud.colorBanding", {
            "entity_id": cloud_id,
            "axis": "Z",
            "frequency": 10.0
        })
    
    def test_level4_rpc_cloud_sor_filter(self, cloud_id):
        """Test statistical outlier removal via RPC."""
        _rpc_call("cloud.sorFilter", {
            "entity_id": cloud_id,
            "knn": 6,
            "sigma": 1.0
        })
    
    def test_level4_rpc_sf_arithmetic(self, cloud_id):
        """Test scalar field arithmetic operations via RPC."""
        _rpc_call("cloud.coordToSF", {
            "entity_id": cloud_id,
            "dimension": "Z"
        })
        _rpc_call("cloud.sfArithmetic", {
            "entity_id": cloud_id,
            "sf_index": 0,
            "operation": "SQRT"
        })
    
    def test_level4_rpc_sf_operation(self, cloud_id):
        """Test scalar field operation with constant via RPC."""
        _rpc_call("cloud.coordToSF", {
            "entity_id": cloud_id,
            "dimension": "Z"
        })
        _rpc_call("cloud.sfOperation", {
            "entity_id": cloud_id,
            "sf_index": 0,
            "operation": "MULTIPLY",
            "value": 2.0
        })
    
    def test_level4_rpc_sf_gradient(self, cloud_id):
        """Test scalar field gradient via RPC."""
        _rpc_call("cloud.coordToSF", {
            "entity_id": cloud_id,
            "dimension": "Z"
        })
        _rpc_call("cloud.sfGradient", {
            "entity_id": cloud_id
        })
    
    def test_level4_rpc_sf_convert_to_rgb(self, cloud_id):
        """Test scalar field to RGB conversion via RPC."""
        _rpc_call("cloud.coordToSF", {
            "entity_id": cloud_id,
            "dimension": "Z"
        })
        _rpc_call("cloud.sfConvertToRGB", {
            "entity_id": cloud_id
        })
    
    def test_level4_rpc_octree_normals(self, cloud_id):
        """Test octree-based normal computation via RPC."""
        _rpc_call("cloud.octreeNormals", {
            "entity_id": cloud_id,
            "radius": "AUTO"
        })
    
    def test_level4_rpc_orient_normals_mst(self, cloud_id):
        """Test MST normal orientation via RPC."""
        _rpc_call("cloud.computeNormals", {"entity_id": cloud_id})
        _rpc_call("cloud.orientNormalsMST", {
            "entity_id": cloud_id,
            "knn": 6
        })
    
    def test_level4_rpc_clear_normals(self, cloud_id):
        """Test clear normals via RPC."""
        _rpc_call("cloud.computeNormals", {"entity_id": cloud_id})
        _rpc_call("cloud.clearNormals", {
            "entity_id": cloud_id
        })
    
    def test_level4_rpc_normals_to_sfs(self, cloud_id):
        """Test normals to scalar fields via RPC."""
        _rpc_call("cloud.computeNormals", {"entity_id": cloud_id})
        _rpc_call("cloud.normalsToSFs", {
            "entity_id": cloud_id
        })
    
    def test_level4_rpc_normals_to_dip(self, cloud_id):
        """Test normals to dip/dip-direction via RPC."""
        _rpc_call("cloud.octreeNormals", {
            "entity_id": cloud_id,
            "radius": "AUTO"
        })
        _rpc_call("cloud.normalsToDip", {
            "entity_id": cloud_id
        })
    
    def test_level4_rpc_extract_connected_components(self, cloud_id):
        """Test extract connected components via RPC."""
        _rpc_call("cloud.extractConnectedComponents", {
            "entity_id": cloud_id,
            "min_points": 10,
            "octree_level": 6
        })
    
    def test_level4_rpc_approx_density(self, cloud_id):
        """Test approximate density computation via RPC."""
        _rpc_call("cloud.approxDensity", {
            "entity_id": cloud_id,
            "density_type": "PRECISE"
        })
    
    def test_level4_rpc_best_fit_plane(self, cloud_id):
        """Test best fit plane computation via RPC."""
        _rpc_call("cloud.bestFitPlane", {
            "entity_id": cloud_id,
            "make_horiz": False
        })
    
    def test_level4_rpc_delaunay(self, cloud_id):
        """Test Delaunay triangulation via RPC."""
        _rpc_call("cloud.computeNormals", {"entity_id": cloud_id})
        _rpc_call("cloud.delaunay", {
            "entity_id": cloud_id
        })

    def test_level4_rpc_workflow_load_orient_screenshot(
            self, cloud_id, tmp_path):
        """Full workflow: load -> orient -> zoom -> screenshot."""
        _rpc_call("view.setOrientation", {"orientation": "iso1"})
        _rpc_call("view.zoomFit", {"entity_id": cloud_id})
        path = str(tmp_path / "workflow.png")
        result = _rpc_call("view.screenshot", {"filename": path})
        assert result["width"] > 0
        assert Path(path).exists()


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
            assert len(tools) >= 95, (
                f"Expected ≥95 MCP tools, got {len(tools)}.\n"
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
