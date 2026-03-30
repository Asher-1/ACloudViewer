# CLI Command Mapping

This document maps **`cli-anything-acloudviewer` CLI** commands to **MCP tools** (from `cli-anything-acloudviewer-mcp`) and **JSON-RPC methods** (GUI WebSocket plugin). It also clarifies internal CLI aliases (`sf`, `normals`).

In **`cli-anything-acloudviewer --help`**, each top-level group is tagged **`[GUI]`** or **`[Headless]`** to show whether it targets JSON-RPC (running app) or the headless binary.

Legend for the **RPC method** column: **(H)** headless binary / process API only. **(G)** GUI JSON-RPC (`ws://…:6001`). **(B)** either path depending on MCP `--mode`.

## CLI ↔ MCP ↔ JSON-RPC (major commands)

| CLI command / group | MCP tool(s) | JSON-RPC method(s) |
|---------------------|-------------|---------------------|
| `info`, `formats`, `version` | `get_info`, `list_formats` | — **(H)** |
| `convert` | `convert_format` | `file.convert` **(G)**; headless MCP uses binary **(H)** |
| `batch-convert` | `batch_convert` | — **(H)** |
| `process` (subsample, normals, crop, SOR, density, …) | `subsample`, `compute_normals`, `crop`, `sor_filter`, `density`, … | `cloud.subsample`, `cloud.computeNormals`, … **(G)** when routed via RPC; `process crop` is **(H)** |
| `process delaunay` | `delaunay` | — **(H)** |
| `process pcv` | `pcv` | — **(H)** |
| `process csf` | `csf` | — **(H)** |
| `process ransac` | `ransac` | — **(H)** |
| `process m3c2` | `m3c2` | — **(H)** |
| `process canupo` | `canupo` | — **(H)** |
| `process facets` | `facets` | — **(H)** |
| `process hough-normals` | `hough_normals` | — **(H)** |
| `process poisson-recon` | `poisson_recon` | — **(H)** |
| `process cork-boolean` | `cork_boolean` | — **(H)** |
| `process voxfall` | `voxfall` | — **(H)** |
| `sf` * | `coord_to_sf`, `set_active_sf`, `sf_gradient`, … | `cloud.coordToSf`, `cloud.setActiveSf`, … **(G)** |
| `normals` * | `octree_normals`, `orient_normals_mst`, … | — **(H)**; GUI cloud ops overlap **(G)** |
| `cloud` (paint, crop, …) | `cloud_paint_uniform`, `crop`, … | `cloud.paintUniform`, `cloud.crop`, … **(G)** |
| `mesh` simplify / smooth / subdivide / sample-points | `mesh_simplify`, `mesh_smooth`, `mesh_subdivide`, `mesh_sample_points` | `mesh.simplify`, `mesh.smooth`, `mesh.subdivide`, `mesh.samplePoints` **(G)** |
| `scene` | `scene_list`, `scene_info`, `scene_remove`, … | `scene.list`, `scene.info`, `scene.remove`, … **(G)** |
| `view` | `screenshot`, `get_camera`, `view_set_orientation`, … | `view.screenshot`, `view.getCamera`, `view.setOrientation`, … **(G)** |
| `entity` | `entity_rename`, `entity_set_color` | `entity.rename`, `entity.setColor` **(G)** |
| `open`, `export` | `open_file`, `export_entity` | `open`, `export` **(G)** |
| `transform apply` / `transform apply-file` | `transform_apply`, `transform_apply_file` | `transform.apply` **(B)** |
| `reconstruct` (Colmap) | `colmap_*`, `colmap_run` | `colmap.reconstruct`, `colmap.run` **(H)** subprocess / **(G)** RPC |
| `sibr` | `sibr_*` | — **(H)** (external SIBR tools) |
| `methods` | `list_rpc_methods` | `methods.list` **(G)** |
| `session` | `get_session_info` (and CLI session commands) | — **(B)** |

\* The `sf` and `normals` CLI groups are **aliases** (same implementations as the matching `process …` subcommands); see [Scalar Field Commands](#scalar-field-commands) and [Normal Vector Commands](#normal-vector-commands).

When the **MCP server** runs in **`--mode headless`**, tools invoke the **ACloudViewer binary** (`-SILENT` …) instead of RPC—there is no JSON-RPC method in that path. When **`--mode gui`**, GUI tools forward to the matching **`category.action`** RPC methods above.

## Plugin Processing Commands

These commands wrap ACloudViewer's native C++ plugin CLI interfaces. They require the ACloudViewer binary and run in headless mode only.

| CLI Command | Native Flag | Plugin | Description |
|-------------|-------------|--------|-------------|
| `process pcv` | `-PCV` | qPCV | Ambient occlusion / sky visibility |
| `process csf` | `-CSF` | qCSF | Cloth Simulation ground filtering |
| `process ransac` | `-RANSAC` | qRANSAC_SD | Shape detection (planes, spheres, etc.) |
| `process m3c2` | `-M3C2` | qM3C2 | Multiscale cloud comparison |
| `process canupo` | `-CANUPO_CLASSIF` | qCanupo | Point cloud classification |
| `process facets` | `-FACETS` | qFacets | Planar facet extraction |
| `process hough-normals` | `-HOUGH_NORMALS` | qHoughNormals | Hough-based normal estimation |
| `process poisson-recon` | `-POISSON_RECON` | qPoissonRecon | Poisson surface reconstruction |
| `process cork-boolean` | `-CORK` | qCork | Mesh boolean ops (union/intersect/diff/sym_diff) |
| `process voxfall` | `-VOXFALL` | qVoxFall | Voxel-based rockfall/change detection |

## Command Group Overview

The CLI has three main processing command groups (all **`[Headless]`** in `--help`):

1. **`process`** — Core processing commands
2. **`sf`** — Scalar field alias group (mirrors `process` SF subcommands)
3. **`normals`** — Normal-vector alias group (mirrors `process` advanced normal subcommands)

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
- `cloud crop <entity_id> --min-x … --min-y … --min-z … --max-x … --max-y … --max-z …` — bbox crop in GUI (six bounds; **not** the legacy wrong axis order `Xmin:Xmax:Ymin:Ymax` style string)
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
  ├── process (38+ commands, incl. crop and 8 plugin processors: pcv, csf, ransac, m3c2, canupo, facets, hough-normals, poisson-recon)
  ├── sf (11 commands)
  ├── normals (6 commands)
  ├── reconstruct (12+ Colmap commands)
  ├── sibr (10+ commands)
  └── transform apply-file

gui mode (requires running ACloudViewer + JSON-RPC):
  ├── open, export
  ├── scene (list, info, remove, show, hide, select, clear)   # use scene clear; top-level `clear` is deprecated
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

- [CLI Quick Reference](CLI-QUICK-REFERENCE.md) — Full command listing
- [JSON-RPC API](JSON-RPC-API.md) — Method names and parameters
- [README.md](../README.md) — Overview and examples
- [TESTING.md](TESTING.md) — Test suite documentation
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) — Platform and CLI issues
