# ACloudViewer Agent Skill Integration

> **Reference copy only.** Canonical harness code, MCP server, and skill manifests live in
> [CLI-Anything](https://github.com/Asher-1/CLI-Anything) under `acloudviewer/agent-harness/`,
> following the same layout as `blender/agent-harness/` and other CLI-Anything projects.

Use ACloudViewer from AI agents via MCP with **178 tools** for 3D point cloud and mesh processing.

## Canonical Sources (CLI-Anything)

| Artifact | Canonical path |
|----------|----------------|
| CLI / MCP package | `CLI-Anything/acloudviewer/agent-harness/` |
| Agent SKILL.md (package) | `acloudviewer/agent-harness/cli_anything/acloudviewer/skills/SKILL.md` |
| Agent SKILL.md (hub mirror) | `CLI-Anything/skills/cli-anything-acloudviewer/SKILL.md` |
| OpenClaw manifest | `acloudviewer/agent-harness/cli_anything/acloudviewer/openclaw-skill.json` |
| Registry entry | `CLI-Anything/registry.json` → `acloudviewer` |
| CLI-Hub | https://asher-1.github.io/CLI-Anything/ |
| Agent catalog | https://asher-1.github.io/CLI-Anything/SKILL.txt |

Install from the fork (same pattern as Blender and other CLIs):

```bash
pip install git+https://github.com/Asher-1/CLI-Anything.git#subdirectory=acloudviewer/agent-harness
# or
pip install cli-anything-acloudviewer
```

Local development:

```bash
cd /path/to/CLI-Anything/acloudviewer/agent-harness
pip install -e ".[mcp,dev]"
```

## Agent Frameworks

### OpenClaw / ClawHub

1. Search for **acloudviewer** in ClawHub, or self-host:

```bash
pip install cli-anything-acloudviewer
```

2. Add to `openclaw.json`:

```json
{
  "plugins": {
    "acloudviewer": {
      "command": "cli-anything-acloudviewer-mcp",
      "args": ["--mode", "auto"],
      "type": "mcp",
      "description": "3D point cloud and mesh processing with ACloudViewer (178 tools)"
    }
  }
}
```

3. Restart OpenClaw.

### Cursor / Claude Code (MCP)

See [../mcp/README.md](../mcp/README.md). Example `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "acloudviewer": {
      "command": "cli-anything-acloudviewer-mcp",
      "args": ["--mode", "auto"]
    }
  }
}
```

### Codex / Cursor Skills (SKILL.md)

Point your agent at the hub mirror (same content as the package skill):

- https://github.com/Asher-1/CLI-Anything/blob/main/skills/cli-anything-acloudviewer/SKILL.md
- Live catalog: https://asher-1.github.io/CLI-Anything/SKILL.txt

Or copy `skills/cli-anything-acloudviewer/SKILL.md` into your project's skills directory.

## Operating Modes

| Mode | Mechanism | When used |
|------|-----------|-----------|
| **GUI** | JSON-RPC WebSocket (`ws://localhost:6001`) | ACloudViewer running with qJSonRPC plugin |
| **Headless** | `ACloudViewer -SILENT -O …` subprocess | No GUI, or batch/CI |
| **Auto** | Try GUI RPC, fall back to headless | Default for MCP and `--mode auto` |

Set `ACV_BINARY` to your build or install path. For integration tests, prefer
`build_app/bin/ACloudViewer` over older system installs.

## Overview

The MCP skill covers:

- **File I/O**: 30+ formats (PLY, PCD, OBJ, STL, LAS, E57, DRC, …)
- **Point cloud processing**: subsample, normals, SOR, density, curvature, roughness
- **Scalar fields & normals**: full SF pipeline and MST orientation
- **Distance & registration**: C2C, C2M, ICP
- **Mesh**: Delaunay, simplify, smooth, volume, sampling
- **Plugins & PCL**: PCV, Compass, CSF, Poisson, Cork, PCL filters, etc.
- **Colmap**: full SfM/MVS pipeline (13 tools)
- **SIBR**: dataset prep, texturing, viewers (12 tools)
- **Scene / view / transform** (GUI RPC)

Full tool list: [../mcp/README.md](../mcp/README.md) and `mcp_server.py` `list_tools()`.

## Testing

Cross-layer tests live in this repo; harness unit tests live in CLI-Anything:

```bash
# ACloudViewer integration (Levels 1–5: C++ plugin + CLI + RPC + MCP)
cd agent-integration/tests
export ACV_BINARY=/path/to/ACloudViewer
export ACV_RPC_URL=ws://127.0.0.1:6001
python -m pytest test_integration.py -v

# CLI-Anything harness unit tests (592 tests, same repo layout as blender/)
cd ../CLI-Anything/acloudviewer/agent-harness
pytest cli_anything/acloudviewer/tests/ -q
```

## Maintenance

When updating skill metadata:

1. Edit **canonical** `openclaw-skill.json` in CLI-Anything first
2. Update `tools_count` and `tool_categories` if MCP tools change
3. Sync this reference copy (`agent-integration/openclaw/openclaw-skill.json`)
4. Keep `version` aligned with `acloudviewer/agent-harness/setup.py`

## Further Reading

- [ACloudViewer agent-integration README](../README.md)
- [CLI-Anything ACloudViewer harness](https://github.com/Asher-1/CLI-Anything/tree/main/acloudviewer/agent-harness)
- [OPENCLAW.md in harness](https://github.com/Asher-1/CLI-Anything/blob/main/acloudviewer/agent-harness/cli_anything/acloudviewer/OPENCLAW.md)
- [MCP tool reference](../mcp/README.md)
- [JSON-RPC API](../docs/JSON-RPC-API.md)
