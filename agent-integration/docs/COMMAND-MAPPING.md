# CLI Command Mapping

This document clarifies the relationship between different command groups and their implementations.

## Command Group Overview

The CLI has three main processing command groups:

1. **`process`** - Core processing commands (headless)
2. **`sf`** - Scalar field convenience wrappers
3. **`normals`** - Normal vector convenience wrappers

## Scalar Field Commands

The `sf` group provides convenient shortcuts for scalar field operations:

| `sf` Command | Equivalent `process` Command | Notes |
|--------------|------------------------------|-------|
| `sf coord-to-sf` | `process coord-to-sf` | Direct mapping |
| `sf arithmetic` | `process sf-arithmetic` | Unary operations (SQRT, ABS, etc.) |
| `sf operation` | `process sf-op` | Binary operations with constant |
| `sf gradient` | `process sf-gradient` | Computes gradient magnitude |
| `sf filter` | `process filter-sf` | Filters by SF value range |
| `sf color-scale` | `process sf-color-scale` | Applies color scale from file |
| `sf convert-to-rgb` | `process sf-to-rgb` | SF → RGB colors |
| `sf set-active` | `process set-active-sf` | Sets active SF |
| `sf rename` | `process rename-sf` | Renames SF |
| `sf remove` | `process remove-sf` | Removes one SF |
| `sf remove-all` | `process remove-all-sfs` | Removes all SFs |

**Key Differences**:
- `sf` commands support `--sf` as an alias for `--sf-index`
- `sf gradient/filter/color-scale/convert-to-rgb` can optionally specify SF, defaulting to active SF
- `sf rename` uses `--old` and `--new` instead of `--sf-index` and `--new-name`

## Normal Vector Commands

The `normals` group provides shortcuts for normal-related operations:

| `normals` Command | Equivalent `process` Command | Notes |
|-------------------|------------------------------|-------|
| `normals octree` | `process octree-normals` | Advanced octree-based computation |
| `normals orient-mst` | `process orient-normals` | MST-based orientation |
| `normals invert` | `process invert-normals` | Flip all normals 180° |
| `normals clear` | `process clear-normals` | Remove all normals |
| `normals to-dip` | `process normals-to-dip` | Geological dip/dip-direction |
| `normals to-sfs` | `process normals-to-sfs` | Export as Nx, Ny, Nz SFs |

**Standard Normal Computation**:
- `process normals` - Basic k-NN method (no `normals` group equivalent)

## GUI-Only Operations

Some operations **require GUI mode** (running ACloudViewer with JSON-RPC) and operate on `entity_id` instead of file paths:

### Cloud Operations (GUI)
- `cloud paint-uniform <entity_id> R G B` - Paint solid color
- `cloud paint-by-height <entity_id> --axis z` - Height gradient
- `cloud paint-by-scalar-field <entity_id> --field "Name"` - Color by SF
- `cloud crop <entity_id> --min-x ... --max-z ...` - Interactive crop
- `cloud get-scalar-fields <entity_id>` - List SFs on loaded entity

### Mesh Operations (GUI)
- `mesh simplify <entity_id>` - Reduce triangle count
- `mesh smooth <entity_id>` - Laplacian smoothing
- `mesh subdivide <entity_id>` - Increase resolution
- `mesh sample-points <entity_id>` - Sample to point cloud

### Workaround for Headless Mode

If you need GUI-only features in headless workflows:

```python
# Use cloudViewer Python API instead
import cloudViewer as cv

# Load, process, save
pcd = cv.io.read_point_cloud("input.ply")
pcd.paint_uniform_color([1, 0, 0])  # Red
cv.io.write_point_cloud("output.ply", pcd)
```

## Complete Command Reference

### Mode Overview

```
headless mode:
  ├── convert, batch-convert
  ├── process (30+ commands)
  ├── sf (11 commands)
  ├── normals (6 commands)
  ├── reconstruct (12+ Colmap commands)
  ├── sibr (10+ commands)
  └── transform apply-file

gui mode (requires running ACloudViewer + JSON-RPC):
  ├── open, export, clear
  ├── scene (list, info, remove, show, hide, select, clear)
  ├── entity (rename, set-color)
  ├── view (screenshot, camera, orient, zoom, refresh, perspective, pointsize)
  ├── cloud (5 commands - paint, crop, get-scalar-fields)
  ├── mesh (4 commands - simplify, smooth, subdivide, sample-points)
  └── transform apply

both modes:
  ├── info, check, formats
  ├── install (auto, app, wheel)
  ├── session (status, undo, redo, save, history)
  ├── methods (GUI only, but command exists)
  └── repl (interactive shell)
```

## Recommendations

1. **For automation/scripting**: Use `process` commands directly (explicit and stable)
2. **For interactive use**: Use `sf` and `normals` groups (shorter and more intuitive)
3. **For GUI control**: Use `scene`, `view`, `cloud`, `mesh` groups
4. **For pipelines**: Combine headless process commands with file I/O

## Common Gotchas

### SF Index vs Name
Most SF commands accept either:
- Numeric index: `--sf-index 0` (first SF)
- String name: `--sf-index "Density"` (by name)

### Active SF Requirement
Some commands operate on the **active scalar field**:
- `sf gradient`, `sf filter`, `sf convert-to-rgb` (if `--sf-index` not specified)
- `process sf-gradient`, `process filter-sf`, `process sf-to-rgb`

Use `sf set-active` first if needed.

### Temporary Files in Workflows
When chaining SF operations that require setting active SF:

```bash
# Method 1: Manual temp files
cli-anything-acloudviewer sf set-active input.ply -o temp.ply --sf-index 1
cli-anything-acloudviewer sf gradient temp.ply -o gradient.ply --euclidean

# Method 2: SF group auto-handles (if --sf-index specified)
cli-anything-acloudviewer sf gradient input.ply -o gradient.ply --sf-index 1 --euclidean
```

The `sf` group commands can automatically set active SF if you provide `--sf-index`.

## See Also

- [CLI Quick Reference](CLI-QUICK-REFERENCE.md) - Comprehensive command listing
- [README.md](../README.md) - Full documentation with examples
- [TESTING.md](TESTING.md) - Test suite documentation
