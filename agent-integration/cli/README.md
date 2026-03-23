# ACloudViewer CLI Harness

Stateful CLI for controlling ACloudViewer from the command line or AI agents.
Built with the [CLI-Anything](https://github.com/HKUDS/CLI-Anything) methodology.

## Installation

```bash
pip install cli-anything-acloudviewer
```

## Modes

| Mode | Description | Requirements |
|------|-------------|-------------|
| `auto` | Try GUI, fall back to headless | Either |
| `gui` | Control running ACloudViewer via WebSocket | ACloudViewer + JSON-RPC plugin |
| `headless` | Invoke ACloudViewer binary in silent CLI mode | ACloudViewer binary on PATH |

## Command Reference

### REPL Mode (default)

```bash
cli-anything-acloudviewer                          # enter interactive REPL
cli-anything-acloudviewer --mode gui               # force GUI backend
cli-anything-acloudviewer --mode headless           # force headless
```

### File I/O

```bash
cli-anything-acloudviewer open /path/to/scene.ply
cli-anything-acloudviewer open scene.ply --silent
cli-anything-acloudviewer export 42 output.obj
```

### Scene Tree

```bash
cli-anything-acloudviewer scene list               # list all entities
cli-anything-acloudviewer scene list --flat         # non-recursive
cli-anything-acloudviewer scene info 42             # entity details
cli-anything-acloudviewer scene remove 42           # remove entity
cli-anything-acloudviewer scene show 42             # make visible
cli-anything-acloudviewer scene hide 42             # make hidden
cli-anything-acloudviewer scene select 42 43 44     # select entities
cli-anything-acloudviewer clear                     # clear all
```

### View Control (GUI mode)

```bash
cli-anything-acloudviewer view orient top           # set view orientation
cli-anything-acloudviewer view orient iso1           # isometric view
cli-anything-acloudviewer view zoom                  # zoom to fit all
cli-anything-acloudviewer view zoom --entity 42      # zoom to entity
cli-anything-acloudviewer view refresh               # force redraw
cli-anything-acloudviewer view perspective object    # object-centered
cli-anything-acloudviewer view perspective viewer    # viewer-centered
cli-anything-acloudviewer view pointsize +           # increase point size
cli-anything-acloudviewer view pointsize -           # decrease point size
```

### Processing (Headless)

```bash
cli-anything-acloudviewer process subsample input.ply -o out.ply --voxel-size 0.05
cli-anything-acloudviewer process subsample input.ply --method random --voxel-size 0.5
cli-anything-acloudviewer process normals input.ply -o out.ply --radius 0.1
cli-anything-acloudviewer process icp source.ply target.ply --threshold 0.02
```

### Session Management

```bash
cli-anything-acloudviewer session status            # show session info
cli-anything-acloudviewer session undo              # undo last operation
cli-anything-acloudviewer session redo              # redo
cli-anything-acloudviewer session save project.json # save session
cli-anything-acloudviewer session history           # show undo history
```

### Utility

```bash
cli-anything-acloudviewer info                      # backend and version info
cli-anything-acloudviewer methods                   # list RPC methods (GUI)
```

### JSON Output

Add `--json` to any command for machine-readable output:

```bash
cli-anything-acloudviewer --json scene list
cli-anything-acloudviewer --json open scene.ply
cli-anything-acloudviewer --json process icp src.ply tgt.ply
```
