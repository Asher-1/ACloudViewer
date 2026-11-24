# Deployment and operation recipes

These recipes preserve the CLI deployment/operation coverage of skill 3.9.5. That skill cited `acloudviewer_cli.py` v3.1.0 and `agent-integration/README.md`; validate against the installed CLI and current repository rather than assuming those historical versions or counts are still current. Application automation MUST use this toolchain; never guess binary flags.

## Sources of truth

| Topic | Repository reference |
|---|---|
| Unified interface | `agent-integration/README.md` |
| Command cheat sheet | `agent-integration/docs/CLI-QUICK-REFERENCE.md` |
| Groups and aliases | `agent-integration/docs/COMMAND-MAPPING.md` |
| JSON-RPC methods (historically 89) | `agent-integration/docs/JSON-RPC-API.md` |
| Troubleshooting | `agent-integration/docs/TROUBLESHOOTING.md` |
| Harness package | `CLI-Anything/acloudviewer/agent-harness/` (separate checkout; locate it) |
| MCP tools (historically 187) | `agent-integration/mcp/README.md` |

## Install routing

Ask whether the user wants to **use** the app, **automate** it, or **build** it.

- Consumer: npm bootstrap; no prerequisites beyond Node/npm (Node >=18).
- Agent/automation: CLI harness plus health checks, MCP registration and headless operations.
- Developer needing app and clean source build: CLI harness plus `install source`; it wraps system dependencies via the platform installer/conda, clone, CMake configure and build. Follow current platform guides under `docs/guides/compiling_doc`.

### Consumer: npm bootstrap

```bash
npm install -g @asher-1/acloudviewer
acloudviewer install
acloudviewer install --silent
acloudviewer uninstall --yes
acloudviewer list
```

`install` downloads and launches the official matching Windows/macOS/Linux installer with CPU/CUDA detection. Silent install lands in `~/.local/share/ACloudViewer` without sudo; uninstall uses QtIFW maintenancetool purge. `list` inspects release assets. `GITHUB_TOKEN=<pat>` raises GitHub API rate limits on shared CI; do not expose its value. Downloads use parallel Range segments and resume. If rolling `main-devel` release assets change during download, `asset is no longer downloadable` fails fast: re-run. npm does not build from source.

### Developer: source build

Always preview first: a full build can take hours and dependency installation needs authorization.

```bash
cli-anything-acloudviewer install source --dry-run
cli-anything-acloudviewer install source --jobs 8 -y
```

Historically this followed Linux apt+pyenv and macOS/Windows conda recipes. Current platform guides override stale script names. Do not invoke this path to fix an existing install when the user forbids installation or replacement.

## One-click harness deployment

Run `check` first (if command not installed, install the harness using one of the supported package routes).

```bash
cli-anything-acloudviewer check

# Choose ONE harness source:
pip install cli-anything-acloudviewer
pip install "git+https://github.com/Asher-1/CLI-Anything.git#subdirectory=acloudviewer/agent-harness"

# App installation: choose the needed variant, not every line.
cli-anything-acloudviewer install auto
cli-anything-acloudviewer install app
cli-anything-acloudviewer install app --cuda
cli-anything-acloudviewer install app --channel beta
cli-anything-acloudviewer install app --from-file /path/to/ACloudViewer-RELEASE.run
cli-anything-acloudviewer install wheel
cli-anything-acloudviewer install wheel --cuda

# Alternative Linux/macOS harness installer: inspect/authorize remote script first.
# ACV_SKIP_CLIENTS=1 skips client registration.
curl -LsSf https://raw.githubusercontent.com/Asher-1/CLI-Anything/main/acloudviewer/agent-harness/install.sh | sh

cli-anything-acloudviewer install clients --dry-run
cli-anything-acloudviewer install clients --all
cli-anything-acloudviewer install clients --client codex --client cursor --mode headless
cli-anything-acloudviewer install clients --all --remove

cli-anything-acloudviewer --json doctor
cli-anything-acloudviewer --json doctor --skip-rpc
cli-anything-acloudviewer --json install models --list
cli-anything-acloudviewer install models --task ocr --image /absolute/path/page.png
cli-anything-acloudviewer --json --mode headless info
cli-anything-acloudviewer formats
```

CPU is default for app/auto/wheel; `--cuda` checks the driver before download. `--from-file` accepts platform `.run`, `.dmg`, `.exe` artifacts (substitute actual path). Doctor checks installation, RPC reachability and AICore registry; `--skip-rpc` deliberately omits reachability and is not GUI evidence. Model prefetch uses native DOWNLOAD=1; `--list` shows the matrix. Installer prints `export ACV_BINARY=<path>`; set it if the executable is not discoverable, but verify actual identity before launch claims.

### Platform notes

| Platform | Artifact / guidance |
|---|---|
| Linux | Qt IFW `.run`; installer invocation pattern below; headless application `-SILENT` runs historically needed XCB packages `libxcb-icccm4 libxcb-shape0` |
| Windows | Qt IFW `.exe`; `ACV_BINARY` must be the actual `.exe`, not a `.bat` wrapper (job control can hang pytest/CI) |
| macOS | `.dmg`; mount and copy app; do NOT force `QT_QPA_PLATFORM`, because app uses cocoa only |

```bash
# Installer invocation, NOT guessed application operation flags:
QT_QPA_PLATFORM=minimal installer --root <dir> --accept-licenses --accept-messages --confirm-command install
# Authorized Linux dependency remedy only; do not run for a no-system-changes task:
sudo apt-get install libxcb-icccm4 libxcb-shape0
# macOS (replace <dmg> with actual artifact):
hdiutil attach <dmg> -nobrowse -noverify && cp -R /Volumes/ACloudViewer*/*.app ~/
```

## MCP client registration

Bundled MCP server runs over stdio. Choose mode: `auto` tries running GUI via WebSocket, `gui` requires it, `headless` uses binary only and is safest unattended. Only one MCP client instance should own the GUI WebSocket at a time.

Automatic registration writes each detected client's config and `.bak` backups. `--all` covers Claude Code, Claude Desktop, Codex and Cursor when detected; non-TTY callers must pass `--all` or `--client`. `--remove` symmetrically unregisters. Preview with `--dry-run`.

```bash
cli-anything-acloudviewer install clients --all
# Manual alternatives:
claude mcp add acloudviewer -- cli-anything-acloudviewer-mcp --mode auto
codex mcp add acloudviewer -- cli-anything-acloudviewer-mcp --mode auto
```

Cursor/VS Code-style JSON (repository `.agents/mcp.json` historically ships `acloudviewer`, `acloudviewer-headless`, `acloudviewer-gui` profiles):

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

Claude Desktop: Settings → Developer → Edit Config (`claude_desktop_config.json`), same JSON. Use absolute command path when GUI launchers do not inherit shell PATH; headless-only args are `["--mode", "headless"]`.

## Operating modes and machine-readable contract

| Situation | Mode |
|---|---|
| Batch, CI, conversion, inference, reconstruction; data on disk | `--mode headless`: direct binary, no port probing, avoids Windows auto hang |
| Scene/entity-ID operations in scene/view/mesh/cloud GUI groups; screenshots | `--mode gui`: running app, enabled qJSonRPCPlugin, normally WebSocket 6001 |
| Interactive Linux/macOS exploration | default auto: GUI then headless fallback; avoid in scripts and Windows |

GUI example (start/enable the app/plugin using supported documented controls; do not restart an existing scene to obtain RPC):

```bash
cli-anything-acloudviewer --mode gui open /absolute/path/scene.ply
cli-anything-acloudviewer --json --mode gui scene list
cli-anything-acloudviewer --mode gui view screenshot /absolute/path/out.png --width 1920 --height 1080
cli-anything-acloudviewer --mode gui mesh simplify <entity_id> --method quadric --target-triangles 50000
```

Get entity IDs from the current scene list, never invent them. For agents always add `--json` when consuming command output. Its historical envelope is `{"status":"...","result":{...}}`; inspect actual schema and errors.

```bash
cli-anything-acloudviewer --json <command> ...
echo $?
```

Exit 0 alone does not establish data production: STL export from non-mesh data can fail silently. Assert every expected file exists and is nonempty, and validate contents where relevant. Use absolute inputs/outputs in actual automation (relative paths below are compact examples, not a script contract).

```bash
export ACV_BINARY=/path/to/ACloudViewer
export ACV_CLI_MODE=headless
export ACV_RPC_URL=ws://127.0.0.1:6001
export ACV_LOG_LEVEL=DEBUG
export COLMAP_PATH=/path/to/colmap
```

These respectively override binary discovery, default mode, GUI endpoint, logging and reconstruction prerequisite. `ACV_BINARY` is not proof that an existing RPC process runs that executable.

## Headless batch cookbook

```bash
cli-anything-acloudviewer --mode headless batch-convert ./scans/ ./out/ -f .ply --filter-ext .las .laz

for f in ./in/*.ply; do
  b=$(basename "$f" .ply)
  cli-anything-acloudviewer --mode headless process subsample "$f" -o "./down/${b}.ply" --voxel-size 0.05 \
    && cli-anything-acloudviewer --mode headless process normals "./down/${b}.ply" -o "./norm/${b}.ply" \
    && cli-anything-acloudviewer --mode headless process roughness "./norm/${b}.ply" -o "./rough/${b}.ply"
done

cli-anything-acloudviewer --mode headless process poisson-recon in.ply -o mesh.ply --depth 8
```

For machine use add `--json`, resolve paths absolutely, create/check required output directories and assert outputs after each stage. Historically 55+ `process` operations include subsample, normals, crop, icp, sor, c2c-dist, c2m-dist, density, curvature, roughness, extract-cc, rasterize, delaunay, sample-mesh, mesh-volume, merge-clouds, poisson-recon, csf, ransac, m3c2, canupo, pcv. Discover current help. Scalar-field and normal families alias `process`: `sf coord-to-sf/arithmetic/operation/gradient/filter/...`, `normals octree/orient-mst/invert/to-dip/...`; see COMMAND-MAPPING.

## 3D reconstruction

Use explicit headless mode (or `ACV_CLI_MODE=headless`) for these disk-data operations; add JSON for machine consumption.

```bash
cli-anything-acloudviewer --mode headless reconstruct auto ./images/ -w ./ws/ --quality high --use-gpu
cli-anything-acloudviewer --mode headless reconstruct auto ./images/ -w ./ws/ --quality low --no-dense

cli-anything-acloudviewer --mode headless reconstruct extract-features ./imgs/ -d db.db
cli-anything-acloudviewer --mode headless reconstruct match db.db --method exhaustive
cli-anything-acloudviewer --mode headless reconstruct sparse db.db --image-path ./imgs/ -o ./sparse/ \
  --random-seed 17 --mapper-option Mapper.min_num_matches=32
cli-anything-acloudviewer --mode headless reconstruct dense-stereo ./dense/
cli-anything-acloudviewer --mode headless reconstruct fuse ./dense/ -o fused.ply
cli-anything-acloudviewer --mode headless reconstruct poisson fused.ply -o mesh.ply
cli-anything-acloudviewer --mode headless reconstruct texture-mesh ./dense/ -o textured/ --mesh mesh.ply

cli-anything-acloudviewer --mode headless sibr prepare-colmap ./ws/
# Separate standalone 3DGS training tool, not ACloudViewer binary flags:
spirula train ./dataset/ --output ./runs/x && spirula mesh ./runs/x --output mesh.ply
```

## AICore batch inference

Historically 19 tasks; installed binary is authoritative. Always `schema` before `run`. Shortcut flags `--image`, `--out-dir`, `--model`, `--device`, `--threads` are valid only if schema lists corresponding uppercase keys. Everything else uses repeatable `--option KEY=VALUE`; JSON values retain type; repeat a key for lists.

```bash
cli-anything-acloudviewer --json --mode headless ai list
cli-anything-acloudviewer --json --mode headless ai schema ocr
cli-anything-acloudviewer --json --mode headless ai run ocr \
  --image ./scan-1.png --image ./scan-2.png --out-dir ./ocr-out \
  --option LANG=en --option SAVE_VIS=true
cli-anything-acloudviewer --json --mode headless ai run llm \
  --model qwen2.5-0.5b-instruct-q8_0.gguf --out-dir ./llm-out \
  --option 'PROMPT=...' --option MAX_TOKENS=32

# Ordinary/third-party plugin passthrough, preserves legacy argv ordering:
cli-anything-acloudviewer --mode headless plugin run ANIMATION \
  --arg=-FPS --arg=24 --arg=-TOTAL_FRAMES --arg=10
```

DOWNLOAD support is established by a zero-side-effect parser probe against the installed binary, not a blanket assumption. `prepare()` commands (querysplat, anylabel, yolo, etc.) historically accept DOWNLOAD 1 and prefetch with `install models --task <name>` through shared ecvModelDownloader. Legacy handleSchemaHelp bridge commands (map-anything, yolo-track, sam3d) historically reject it: prefetch via GUI dialog or place GGUFs under `~/cloudViewer_data/extract/`. Assets are SHA-256 checked against the catalog. Query current schema/capability; do not convert these historical observations into an invented flag.

## Integration tests and troubleshooting

```bash
# ACloudViewer checkout, layered Levels 1–5; auto-skips missing deps:
cd agent-integration/tests && python -m pytest test_integration.py -v
# Harness checkout, not necessarily same working directory:
pytest cli_anything/acloudviewer/tests/ -q
```

Report skipped levels/dependencies; they do not establish end-to-end acceptance.

| Symptom | Cause / action |
|---|---|
| auto hangs, especially Windows | Port 6001 listens but is unresponsive; force headless for disk-data batch/scripts, not for live scene validation |
| Exit 0 without output | Silent conversion failure; assert file exists and size >0 |
| Windows pytest/batch hangs | `.bat` wrapper in ACV_BINARY; use actual `.exe` |
| Linux headless Qt failure | Check missing XCB libs; only install `libxcb-icccm4 libxcb-shape0` if authorized |
| macOS test launch fails | Unset forced QT_QPA_PLATFORM for cocoa app |
| GUI-launched client ENOENT (e.g. uvx) | Missing shell PATH; use absolute executable in MCP config |
| AI schema mismatch | Harness option cache; rerun `ai schema <task>`; binary wins |

First principles: mode follows data location; CLI owns exact application argv; JSON + exit + verified output is the automation contract; query `info`, `formats`, `ai schema` before hard-coding names. Source/native GUI regressions have additional acceptance layers in the companion reference.
