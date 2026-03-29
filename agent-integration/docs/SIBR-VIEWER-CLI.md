# SIBR Viewer CLI Implementation Guide

## Overview

This document describes the implementation requirements for the `cli-anything-acloudviewer sibr viewer` command to launch SIBR viewers through the unified CLI interface.

## Background

Previously, SIBR viewers were launched directly via the ACloudViewer binary:

```bash
# Old approach (deprecated)
ACloudViewer -SIBR_VIEWER gaussian --model-path ./output/ --path ./dataset/
```

Now they should be launched through the unified CLI:

```bash
# New approach (recommended)
cli-anything-acloudviewer sibr viewer gaussian --model-path ./output/ --path ./dataset/
```

## Implementation Requirements

### 1. CLI Command Structure

Add a new `viewer` subcommand under the `sibr` command group in `acloudviewer_cli.py`:

```python
@sibr.command()
@click.argument('viewer_type', type=click.Choice([
    'gaussian', 'ulr', 'ulrv2', 'texturedmesh', 
    'pointbased', 'remoteGaussian'
]))
@click.option('--path', type=click.Path(exists=True), 
              help='Dataset directory')
@click.option('--model-path', type=click.Path(exists=True), 
              help='Trained model directory (for gaussian viewer)')
@click.option('--width', type=int, default=1920, 
              help='Window width')
@click.option('--height', type=int, default=1080, 
              help='Window height')
@click.option('--iteration', type=int, 
              help='Specific iteration to load (gaussian viewer)')
@click.option('--device', type=int, default=0, 
              help='CUDA device ID')
@click.option('--no-interop', is_flag=True, 
              help='Disable CUDA-OpenGL interop')
@click.option('--ip', default='127.0.0.1', 
              help='IP address for remote connection')
@click.option('--port', type=int, default=6009, 
              help='Port for remote connection')
@click.pass_context
def viewer(ctx, viewer_type, path, model_path, width, height, 
           iteration, device, no_interop, ip, port):
    """Launch a SIBR viewer for novel view synthesis visualization."""
    # Implementation details below
```

### 2. Backend Implementation

The viewer command should invoke the ACloudViewer binary with the `-SIBR_VIEWER` flag:

```python
def launch_sibr_viewer(viewer_type: str, **kwargs):
    """
    Launch a SIBR viewer by invoking ACloudViewer binary.
    
    Args:
        viewer_type: One of 'gaussian', 'ulr', 'ulrv2', 'texturedmesh',
                     'pointbased', 'remoteGaussian'
        **kwargs: Viewer-specific options
    
    Returns:
        subprocess.Popen: Running viewer process
    """
    from .utils.acloudviewer_backend import find_binary
    
    binary = find_binary()
    if not binary:
        raise RuntimeError("ACloudViewer binary not found")
    
    # Build command line arguments
    cmd = [binary, '-SIBR_VIEWER', viewer_type]
    
    if kwargs.get('path'):
        cmd.extend(['--path', kwargs['path']])
    
    if kwargs.get('model_path'):
        cmd.extend(['--model-path', kwargs['model_path']])
    
    if kwargs.get('width'):
        cmd.extend(['--width', str(kwargs['width'])])
    
    if kwargs.get('height'):
        cmd.extend(['--height', str(kwargs['height'])])
    
    if kwargs.get('iteration'):
        cmd.extend(['--iteration', str(kwargs['iteration'])])
    
    if kwargs.get('device') is not None:
        cmd.extend(['--device', str(kwargs['device'])])
    
    if kwargs.get('no_interop'):
        cmd.append('--no-interop')
    
    if viewer_type == 'remoteGaussian':
        cmd.extend(['--ip', kwargs.get('ip', '127.0.0.1')])
        cmd.extend(['--port', str(kwargs.get('port', 6009))])
    
    # Launch viewer process
    import subprocess
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    return process
```

### 3. Viewer Types and Required Parameters

| Viewer Type | Required Options | Optional Options |
|-------------|------------------|------------------|
| `gaussian` | `--model-path`, `--path` | `--iteration`, `--device`, `--width`, `--height`, `--no-interop` |
| `ulr` | `--path` | `--width`, `--height` |
| `ulrv2` | `--path` | `--width`, `--height` |
| `texturedmesh` | `--path`, `--mesh` | `--width`, `--height` |
| `pointbased` | `--path` | `--width`, `--height` |
| `remoteGaussian` | `--ip`, `--port` | `--width`, `--height` |

### 4. MCP Tool Integration

Add a corresponding MCP tool in `mcp_server.py`:

```python
Tool(
    name="sibr_viewer",
    description="""Launch a SIBR viewer for novel view synthesis visualization.
    
    Supported viewer types:
    - gaussian: Gaussian Splatting viewer (requires --model-path and --path)
    - ulr: Unstructured Lumigraph Rendering
    - ulrv2: ULR version 2
    - texturedmesh: Textured mesh viewer
    - pointbased: Point-based rendering
    - remoteGaussian: Remote Gaussian viewer (connects to training process)
    """,
    inputSchema={
        "type": "object",
        "properties": {
            "viewer_type": {
                "type": "string",
                "enum": ["gaussian", "ulr", "ulrv2", "texturedmesh", 
                         "pointbased", "remoteGaussian"],
                "description": "Type of SIBR viewer to launch"
            },
            "path": {
                "type": "string",
                "description": "Dataset directory path"
            },
            "model_path": {
                "type": "string",
                "description": "Trained model directory (for gaussian viewer)"
            },
            "width": {
                "type": "integer",
                "default": 1920,
                "description": "Window width"
            },
            "height": {
                "type": "integer",
                "default": 1080,
                "description": "Window height"
            },
            "iteration": {
                "type": "integer",
                "description": "Specific iteration to load (gaussian viewer)"
            },
            "device": {
                "type": "integer",
                "default": 0,
                "description": "CUDA device ID"
            },
            "no_interop": {
                "type": "boolean",
                "default": False,
                "description": "Disable CUDA-OpenGL interop"
            },
            "ip": {
                "type": "string",
                "default": "127.0.0.1",
                "description": "IP address for remote connection"
            },
            "port": {
                "type": "integer",
                "default": 6009,
                "description": "Port for remote connection"
            }
        },
        "required": ["viewer_type"]
    }
)
```

### 5. Testing Requirements

Add tests in `test_integration.py`:

```python
class TestLevel2CLI:
    def test_level2_sibr_viewer_help(self):
        """Test sibr viewer subcommand help."""
        r = subprocess.run(
            ["cli-anything-acloudviewer", "sibr", "viewer", "--help"],
            capture_output=True, text=True, timeout=10)
        assert r.returncode == 0
        output = r.stdout.lower()
        assert "gaussian" in output
        assert "ulr" in output
        assert "remotegaussian" in output
    
    def test_level2_sibr_viewer_types(self):
        """Test all viewer types are recognized."""
        for viewer in ["gaussian", "ulr", "ulrv2", "texturedmesh", 
                       "pointbased", "remoteGaussian"]:
            r = subprocess.run(
                ["cli-anything-acloudviewer", "sibr", "viewer", 
                 viewer, "--help"],
                capture_output=True, text=True, timeout=10)
            # Should show help without error even without required args
            assert "--path" in r.stdout or "--ip" in r.stdout
```

### 6. Error Handling

The implementation should handle:

1. **Missing binary**: Clear error message if ACloudViewer is not installed
2. **Missing SIBR plugin**: Detect if SIBR plugin is not built
3. **Invalid viewer type**: Validate viewer type before launching
4. **Missing required parameters**: Check required options for each viewer type
5. **Process management**: Handle viewer process lifecycle properly

Example:

```python
def validate_viewer_params(viewer_type: str, **kwargs):
    """Validate required parameters for viewer type."""
    if viewer_type == 'gaussian':
        if not kwargs.get('model_path'):
            raise ValueError("--model-path is required for gaussian viewer")
        if not kwargs.get('path'):
            raise ValueError("--path is required for gaussian viewer")
    
    elif viewer_type == 'remoteGaussian':
        if not kwargs.get('ip'):
            raise ValueError("--ip is required for remoteGaussian viewer")
        if not kwargs.get('port'):
            raise ValueError("--port is required for remoteGaussian viewer")
    
    elif viewer_type in ['ulr', 'ulrv2', 'texturedmesh', 'pointbased']:
        if not kwargs.get('path'):
            raise ValueError(f"--path is required for {viewer_type} viewer")
    
    if viewer_type == 'texturedmesh' and not kwargs.get('mesh'):
        raise ValueError("--mesh is required for texturedmesh viewer")
```

## Implementation Checklist

- [ ] Add `viewer` subcommand to `sibr` command group
- [ ] Implement `launch_sibr_viewer()` function in backend
- [ ] Add parameter validation for each viewer type
- [ ] Add `sibr_viewer` MCP tool
- [ ] Update `list_tools()` in MCP server
- [ ] Add CLI tests for viewer command
- [ ] Add MCP tests for viewer tool
- [ ] Update documentation examples
- [ ] Test on Windows, Linux, and macOS

## Related Files

- **CLI-Anything repo**:
  - `acloudviewer/agent-harness/cli_anything/acloudviewer/acloudviewer_cli.py`
  - `acloudviewer/agent-harness/cli_anything/acloudviewer/mcp_server.py`
  - `acloudviewer/agent-harness/cli_anything/acloudviewer/utils/acloudviewer_backend.py`

- **ACloudViewer repo**:
  - `agent-integration/tests/test_integration.py`
  - `agent-integration/README.md`
  - `plugins/core/Standard/qSIBRCommands/qSIBRCommands.h` (C++ reference)

## References

- SIBR commands implementation: `qSIBRCommands.h`
- Test expectations: `test_integration.py::test_level1_sibr_viewer_*`
- Command-line parsing reference: `CommandLine.cpp`
