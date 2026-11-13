---
name: acloudviewer-cli-agent
description: One-click deployment and operation of the cli-anything-acloudviewer CLI and MCP toolchain for ACloudViewer. Use when an agent or user needs to install the harness, install/verify the ACloudViewer binary, register MCP clients (Claude Code, Codex, Cursor, Claude Desktop), run headless batch processing, format conversion, 3D reconstruction, AICore batch inference, choose GUI vs headless mode, or verify outputs from automation.
---

# ACloudViewer CLI: One-Click Deploy & Operate

Single entry point for deploying and driving `cli-anything-acloudviewer`
from a shell, an AI agent, or an MCP client. Every command below is verified
against the harness source (`acloudviewer_cli.py`, v3.1.0) and
`agent-integration/README.md`. Repo rule: all agent/automation interaction
MUST go through this toolchain; never guess ACloudViewer binary flags.

## Sources of Truth

| Topic | Canonical reference |
|---|---|
| Unified interface reference | `agent-integration/README.md` |
| Command cheat sheet | `agent-integration/docs/CLI-QUICK-REFERENCE.md` |
| Command groups & aliases | `agent-integration/docs/COMMAND-MAPPING.md` |
| JSON-RPC methods (89) | `agent-integration/docs/JSON-RPC-API.md` |
| Known pitfalls | `agent-integration/docs/TROUBLESHOOTING.md` |
| Harness package (canonical) | `CLI-Anything/acloudviewer/agent-harness/` |
| MCP tool reference (187 tools) | `agent-integration/mcp/README.md` |

## Phase A — One-Click Deployment

Run `check` first; it tells you exactly which step you still need.

```bash
# 0. Diagnose: what is installed, what is missing, suggested fix
cli-anything-acloudviewer check

# 1. Install the CLI/MCP harness (pick one)
pip install cli-anything-acloudviewer            # PyPI
pip install "git+https://github.com/Asher-1/CLI-Anything.git#subdirectory=acloudviewer/agent-harness"

# 2. Install the ACloudViewer binary (auto-detects platform + NVIDIA GPU)
cli-anything-acloudviewer install auto           # installs whatever is missing
cli-anything-acloudviewer install app            # CUDA build when GPU present
cli-anything-acloudviewer install app --cpu-only # smaller CPU-only build
cli-anything-acloudviewer install app --channel beta
cli-anything-acloudviewer install app --from-file /path/to/ACloudViewer-*.run   # .run/.dmg/.exe
cli-anything-acloudviewer install wheel [--cpu-only]   # optional cloudViewer Python wheel

# One-line script alternative (Linux / macOS): harness install + check + MCP
# client registration; ACV_SKIP_CLIENTS=1 skips registration
curl -LsSf https://raw.githubusercontent.com/Asher-1/CLI-Anything/main/acloudviewer/agent-harness/install.sh | sh

# Register the bundled MCP server with detected AI clients (writes .bak backups)
cli-anything-acloudviewer install clients --all          # every detected client
cli-anything-acloudviewer install clients --dry-run      # preview, no writes
cli-anything-acloudviewer install clients --client codex --client cursor --mode headless
cli-anything-acloudviewer install clients --all --remove # symmetric undo (unregister)

# Health check: installation + RPC reachability + AICore registry
cli-anything-acloudviewer --json doctor [--skip-rpc]

# Source build one command (mirrors docs/guides/compiling_doc: Linux apt+pyenv
# script, macOS/Windows conda); always --dry-run first - build takes hours
cli-anything-acloudviewer install source --dry-run
cli-anything-acloudviewer install source --jobs 8 -y

# Headless AICore model prefetch (native DOWNLOAD=1; --list shows the matrix)
cli-anything-acloudviewer --json install models --list
cli-anything-acloudviewer install models --task ocr --image ./page.png

# 3. Verify end to end (headless, no GUI required)
cli-anything-acloudviewer --json info            # binary path, version, device
cli-anything-acloudviewer formats                # supported I/O formats
```

The installer prints the resolved path as `export ACV_BINARY=<path>`; set it
whenever the binary is not on `PATH`.

### Platform Notes (installer handles platform + arch detection)

| Platform | Artifact | Headless install / caveats |
|---|---|---|
| Linux (.run) | Qt IFW installer | `QT_QPA_PLATFORM=minimal installer --root <dir> --accept-licenses --accept-messages --confirm-command install`; headless `-SILENT` runs may need `sudo apt-get install libxcb-icccm4 libxcb-shape0` |
| Windows (.exe) | Qt IFW installer | Point `ACV_BINARY` at the **.exe**, never a `.bat` wrapper (wrapper job control hangs pytest/CI) |
| macOS (.dmg) | Disk image | `hdiutil attach <dmg> -nobrowse -noverify && cp -R /Volumes/ACloudViewer*/*.app ~/`; do **not** set `QT_QPA_PLATFORM` (cocoa-only plugin) |

### MCP Client Registration (Claude / OpenAI / IDEs)

The MCP server ships in the same package and runs over stdio. Pick the mode
explicitly: `--mode auto` (default), `--mode gui` (running app via WebSocket),
`--mode headless` (binary only, safest for unattended clients).

Automated registration (writes each client's own config with a `.bak`
backup; `--all` covers Claude Code, Claude Desktop, Codex, and Cursor when
detected; non-TTY callers must pass `--all` or `--client`):

```bash
cli-anything-acloudviewer install clients --all
```

Manual registration (equivalent result, useful when a client is not
auto-detected):

```bash
# Claude Code — one command
claude mcp add acloudviewer -- cli-anything-acloudviewer-mcp --mode auto

# OpenAI Codex CLI — one command
codex mcp add acloudviewer -- cli-anything-acloudviewer-mcp --mode auto
```

Cursor / VS Code style JSON config (a ready `.agents/mcp.json` ships in this
repo with three profiles: `acloudviewer`, `acloudviewer-headless`,
`acloudviewer-gui`):

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

Claude Desktop (`claude_desktop_config.json` → Settings → Developer → Edit
Config): same JSON as above with the full command path if launchers cannot
resolve PATH. For headless-only environments use `"args": ["--mode", "headless"]`.
Only one MCP client instance should own the GUI WebSocket at a time.

## Phase B — Operating the CLI

### Mode Selection (decide once per script)

| Situation | Mode | Why |
|---|---|---|
| Batch, CI, conversion, AI inference, reconstruction | `--mode headless` | Calls the binary directly; no port probing; immune to the Windows `auto` hang |
| entity_id-based ops (`scene`, `view`, `mesh`, `cloud` GUI groups), screenshots | GUI | Requires ACloudViewer running with qJSonRPCPlugin enabled (WebSocket 6001) |
| Interactive exploration on Linux/macOS | default `auto` | Tries GUI then falls back headless; avoid in scripts and on Windows |

```bash
# GUI mode contract: start ACloudViewer, enable the JSON-RPC plugin, then
cli-anything-acloudviewer open /path/to/scene.ply
cli-anything-acloudviewer --json scene list        # get entity ids here
cli-anything-acloudviewer view screenshot out.png --width 1920 --height 1080
cli-anything-acloudviewer mesh simplify <entity_id> --method quadric --target-triangles 50000
```

### Machine-Readable Contract (agents MUST follow this)

```bash
cli-anything-acloudviewer --json <command> ...     # {"status": "...", "result": {...}}
echo $?                                            # 0 = CLI succeeded
```

1. Always pass `--json` when an agent consumes the output.
2. Exit code 0 does **not** guarantee data was produced: some conversions fail
   silently (e.g. STL export from non-mesh data). Automation MUST assert the
   output file exists and is non-empty, not just `returncode == 0`.
3. Use absolute paths for all inputs/outputs.

### Environment Variables

```bash
export ACV_BINARY=/path/to/ACloudViewer      # override binary discovery
export ACV_CLI_MODE=headless                 # default mode for every call
export ACV_RPC_URL=ws://127.0.0.1:6001       # GUI WebSocket endpoint
export ACV_LOG_LEVEL=DEBUG                   # CLI verbosity
export COLMAP_PATH=/path/to/colmap           # reconstruction prerequisite
```

### Headless Batch Cookbook

```bash
# Directory conversion with extension filter
cli-anything-acloudviewer --mode headless batch-convert ./scans/ ./out/ -f .ply --filter-ext .las .laz

# Multi-stage per-file pipeline (subsample -> normals -> roughness)
for f in ./in/*.ply; do
  b=$(basename "$f" .ply)
  cli-anything-acloudviewer --mode headless process subsample "$f" -o "./down/${b}.ply" --voxel-size 0.05 \
    && cli-anything-acloudviewer --mode headless process normals "./down/${b}.ply" -o "./norm/${b}.ply" \
    && cli-anything-acloudviewer --mode headless process roughness "./norm/${b}.ply" -o "./rough/${b}.ply"
done

# 55+ processing ops: subsample, normals, crop, icp, sor, c2c-dist, c2m-dist,
# density, curvature, roughness, extract-cc, rasterize, delaunay, sample-mesh,
# mesh-volume, merge-clouds, poisson-recon, csf, ransac, m3c2, canupo, pcv, ...
cli-anything-acloudviewer --mode headless process poisson-recon in.ply -o mesh.ply --depth 8
```

Scalar-field and normal families are alias groups of `process`:
`sf coord-to-sf/arithmetic/operation/gradient/filter/...`,
`normals octree/orient-mst/invert/to-dip/...` (see COMMAND-MAPPING.md).

### 3D Reconstruction (headless)

```bash
# End-to-end
cli-anything-acloudviewer reconstruct auto ./images/ -w ./ws/ --quality high --use-gpu
cli-anything-acloudviewer reconstruct auto ./images/ -w ./ws/ --quality low --no-dense

# Step-by-step
cli-anything-acloudviewer reconstruct extract-features ./imgs/ -d db.db
cli-anything-acloudviewer reconstruct match db.db --method exhaustive
cli-anything-acloudviewer reconstruct sparse db.db --image-path ./imgs/ -o ./sparse/ \
  --random-seed 17 --mapper-option Mapper.min_num_matches=32
cli-anything-acloudviewer reconstruct dense-stereo ./dense/
cli-anything-acloudviewer reconstruct fuse ./dense/ -o fused.ply
cli-anything-acloudviewer reconstruct poisson fused.ply -o mesh.ply
cli-anything-acloudviewer reconstruct texture-mesh ./dense/ -o textured/ --mesh mesh.ply

# SIBR dataset prep / 3DGS training (standalone binary, no GUI)
cli-anything-acloudviewer sibr prepare-colmap ./ws/
spirula train ./dataset/ --output ./runs/x && spirula mesh ./runs/x --output mesh.ply
```

### AICore Batch Inference (headless, 19 tasks)

The installed binary is the source of truth for task names and options —
always `schema` before `run`. Shortcut flags (`--image`, `--out-dir`,
`--model`, `--device`, `--threads`) are valid only when the schema lists the
same uppercase key; everything else goes through repeatable
`--option KEY=VALUE` (JSON values keep their type; repeat a key for lists).

```bash
cli-anything-acloudviewer --json --mode headless ai list
cli-anything-acloudviewer --json --mode headless ai schema ocr
cli-anything-acloudviewer --json --mode headless ai run ocr \
  --image ./scan-1.png --image ./scan-2.png --out-dir ./ocr-out \
  --option LANG=en --option SAVE_VIS=true
cli-anything-acloudviewer --json --mode headless ai run llm \
  --model qwen2.5-0.5b-instruct-q8_0.gguf --out-dir ./llm-out \
  --option 'PROMPT=...' --option MAX_TOKENS=32

# Ordinary/third-party plugin passthrough (legacy argv order preserved)
cli-anything-acloudviewer --mode headless plugin run ANIMATION \
  --arg=-FPS --arg=24 --arg=-TOTAL_FRAMES --arg=10
```

Notes: DOWNLOAD support is judged by a zero-side-effect parser probe
against the installed binary — `prepare()`-based commands (querysplat,
anylabel, yolo, ...) accept `DOWNLOAD 1` and prefetch headlessly via
`install models --task <name>` (fetch through the shared
ecvModelDownloader), while the legacy handleSchemaHelp bridge commands
(map-anything, yolo-track, sam3d) reject the token (prefetch through the
GUI dialog or place GGUFs under `~/cloudViewer_data/extract/`). Model
assets are SHA-256 verified against the catalog.

## Phase C — Verification & Troubleshooting

```bash
# Layered test suite (Levels 1-5; auto-skips missing deps)
cd agent-integration/tests && python -m pytest test_integration.py -v

# Harness unit tests (CLI-Anything repo)
pytest cli_anything/acloudviewer/tests/ -q
```

| Symptom | Cause | Fix |
|---|---|---|
| `auto` mode hangs (esp. Windows) | port 6001 listening but unresponsive | Always `--mode headless` for batch/scripts |
| Command "succeeds" (exit 0) but no data | silent conversion failure | Assert output exists and size > 0 |
| pytest/batch hangs on Windows | `ACV_BINARY` points at a `.bat` wrapper | Point it at the real `.exe` |
| Headless Qt fails on Linux | missing XCB libs | `sudo apt-get install libxcb-icccm4 libxcb-shape0` |
| macOS app won't start in tests | `QT_QPA_PLATFORM` forced | Unset it for the `.app` bundle (cocoa only) |
| `uvx`-style ENOENT from GUI-launched clients | GUI apps don't inherit shell PATH | Use the absolute command path in MCP config |
| AI task schema mismatch | harness option cache | Re-run `ai schema <task>`; binary is authoritative |

## First-Principles Rules for Agents

1. Mode = data location: data on disk → headless; data in a running scene → GUI.
2. Never construct `ACloudViewer -SILENT ...` flag strings by hand; always go
   through this CLI (it owns the exact argv per operation).
3. `--json` + exit code + output-existence assertion is the complete success
   contract; nothing else is evidence.
4. Ask the binary (`info`, `formats`, `ai schema`) before hard-coding
   task/option names; the catalog, not the docs, is current.

## Version

3.2.0 (aligns with `acloudviewer/agent-harness/setup.py`)
