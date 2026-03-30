# CLI Quick Reference

Fast lookup for all `cli-anything-acloudviewer` commands.

Command groups in `cli-anything-acloudviewer --help` are labeled **`[GUI]`** or **`[Headless]`** so you can see which backend each group expects.

## Global Options

```bash
cli-anything-acloudviewer [OPTIONS] COMMAND [ARGS...]

Options:
  --json              # Output JSON format
  --mode MODE         # auto|headless|gui (default: auto)
  -v, --verbose       # Increase verbosity
  -vv                 # Debug level logging
  --help              # Show help message
```

## Core Commands

### Info & Formats

```bash
cli-anything-acloudviewer info              # Show binary info, version, device
cli-anything-acloudviewer formats           # List all supported formats
cli-anything-acloudviewer version           # Show CLI version
```

### File Conversion

```bash
# Single file
cli-anything-acloudviewer convert input.ply output.pcd

# Batch directory
cli-anything-acloudviewer batch-convert ./scans/ ./out/ --format .ply

# Filter by extension
cli-anything-acloudviewer batch-convert ./scans/ ./out/ -f .ply --filter-ext .las .laz
```

---

## Processing Commands

All processing commands follow this pattern:
```bash
cli-anything-acloudviewer process <operation> input.ply [input2.ply] -o output.ply [OPTIONS]
```

Headless **`process`** exposes 52+ subcommands: core geometry and analysis operations, plus plugin-backed commands (`pcv`, `csf`, `ransac`, `m3c2`, `canupo`, `facets`, `hough-normals`, `poisson-recon`, `cork-boolean`, `voxfall`, `3dmasc`, etc.). See [Plugin Processing Commands](#plugin-processing-commands) below.

### Basic Operations

| Command | Description | Key Options |
|---------|-------------|-------------|
| `subsample` | Downsample point cloud | `--method` SPATIAL\|RANDOM\|OCTREE, `--voxel-size` (float) |
| `normals` | Compute normals (k-NN) | `--radius` (float, 0=auto) |
| `crop` | Axis-aligned bbox crop (headless) | `--min-x` … `--max-z` (six values; see below) |
| `sor` | Statistical outlier removal | `--knn` (int), `--sigma` (float) |

**Crop**: Use **`process crop`** in headless mode with **`--min-x/--min-y/--min-z` and `--max-x/--max-y/--max-z`** (six separate bounds). In GUI mode, **`cloud crop <entity_id>`** takes the same min/max semantics (not the old erroneous `min_x:max_x:min_y:max_y` ordering).

### Distance Computation

| Command | Description | Key Options |
|---------|-------------|-------------|
| `c2c-dist` | Cloud-to-cloud distance | `--max-distance` (float) |
| `c2m-dist` | Cloud-to-mesh distance | `--max-distance` (float) |
| `closest-point-set` | Find closest points | Second cloud as reference |

### Geometric Features

| Command | Description | Key Options |
|---------|-------------|-------------|
| `density` | Local point density | `--radius` (float) |
| `approx-density` | Approximate density | `--type` (optional) |
| `curvature` | Surface curvature | `--type` MEAN\|GAUSS, `--radius` (float) |
| `roughness` | Surface roughness | `--radius` (float) |

**See also**: `process feature` in Advanced Processing section for multi-scale geometric features.

### Advanced Processing

| Command | Description | Key Options |
|---------|-------------|-------------|
| `extract-cc` | Extract connected components | `--min-points` (int), `--octree-level` (int) |
| `feature` | Compute geometric features | `--type` SURFACE_VARIATION\|PLANARITY\|..., `--kernel-size` (float) |
| `approx-density` | Approximate density | `--type` (optional) |
| `moment` | Compute 1st order moment | `--kernel-size` (float) |
| `best-fit-plane` | Fit plane to points | `--make-horiz`, `--keep-loaded` |
| `rasterize` | Convert to 2.5D grid | `--grid-step` (float), `--proj` MIN\|MAX\|AVG |
| `stat-test` | Statistical outlier test | `--distribution` GAUSS\|WEIBULL, `--p-value` (float) |
| `cross-section` | Extract cross-section | `--polyline` (file, optional) |

### Mesh Operations (Headless)

| Command | Description | Key Options |
|---------|-------------|-------------|
| `delaunay` | Delaunay triangulation | `--max-edge-length` (float, 0=auto) |
| `sample-mesh` | Sample points from mesh | `--points` (int, point count) |
| `mesh-volume` | Compute mesh volume | (optional `--output-file` for result) |
| `extract-vertices` | Mesh → point cloud vertices | `-o` |
| `flip-triangles` | Reverse triangle winding | `-o` |
| `merge-meshes` | Combine multiple meshes | 2+ inputs, `-o` |

**Note**: mesh-simplify, mesh-smooth, mesh-subdivide require GUI mode (see `mesh` group below).

### Color Operations

| Command | Description | Key Options |
|---------|-------------|-------------|
| `color-banding` | Rainbow bands by axis | `--axis` X\|Y\|Z, `--frequency` (float) |
| `remove-rgb` | Remove all colors | `-o` |

**GUI mode only**: `cloud paint-uniform`, `cloud paint-by-height`, `cloud paint-by-scalar-field` (require entity_id)

### Registration

| Command | Description | Key Options |
|---------|-------------|-------------|
| `icp` | ICP registration | source, target inputs |
| `match-centers` | Translate to align centers | source, target inputs |

### Utility

| Command | Description | Key Options |
|---------|-------------|-------------|
| `merge-clouds` | Merge multiple clouds | 2+ inputs, `-o` |
| `match-centers` | Align bounding box centers | 2+ inputs, `-o` |
| `closest-point-set` | Find closest points | 2+ inputs, `-o` |
| `drop-global-shift` | Remove coordinate offset | `-o` |
| `remove-scan-grids` | Clean scan artifacts | `-o` |
| `remove-rgb` | Remove colors | `-o` |

### Plugin Processing Commands

These invoke ACloudViewer C++ plugins via the headless binary (`-SILENT` and native flags such as `-PCV`, `-FACETS`, `-HOUGH_NORMALS`, `-POISSON_RECON`).

```bash
# PCV (ambient occlusion)
cli-anything-acloudviewer process pcv input.ply -o output.ply --n-rays 256 --resolution 1024

# CSF ground filtering
cli-anything-acloudviewer process csf input.ply -o output.ply --scenes RELIEF --cloth-resolution 2.0
cli-anything-acloudviewer process csf input.ply -o output.ply --scenes SLOPE --proc-slope --export-ground

# RANSAC shape detection
cli-anything-acloudviewer process ransac input.ply -o output.ply --epsilon 0.005 --support-points 500
cli-anything-acloudviewer process ransac input.ply -o output.ply --primitives PLANE --primitives SPHERE --primitives CYLINDER

# M3C2 cloud comparison
cli-anything-acloudviewer process m3c2 cloud1.ply cloud2.ply -o dist.ply --params-file m3c2_params.txt
cli-anything-acloudviewer process m3c2 cloud1.ply cloud2.ply -o dist.ply --params-file m3c2_params.txt --core-points core.ply

# CANUPO classification
cli-anything-acloudviewer process canupo input.ply -o classified.ply --classifier model.prm
cli-anything-acloudviewer process canupo input.ply -o classified.ply --classifier model.prm --use-confidence 0.5

# Facet extraction
cli-anything-acloudviewer process facets input.ply -o output.ply --algo KD_TREE --error-max 0.2 --classify

# Hough-based normals
cli-anything-acloudviewer process hough-normals input.ply -o output.ply --k 100 --t 1000

# Poisson reconstruction
cli-anything-acloudviewer process poisson-recon input.ply -o output.ply --depth 8 --boundary NEUMANN

# Cork mesh boolean (union, intersect, diff, sym_diff)
cli-anything-acloudviewer process cork-boolean mesh1.ply mesh2.ply -o result.ply --operation UNION
cli-anything-acloudviewer process cork-boolean mesh1.ply mesh2.ply -o diff.ply --operation DIFF --swap

# VoxFall rockfall / change detection
cli-anything-acloudviewer process voxfall ref_mesh.ply comp_mesh.ply -o changes.ply --voxel-size 0.05 --azimuth 45 --loss-gain

# Compass — Export measurements
cli-anything-acloudviewer process compass-export project.bin -o compass_data --format csv
cli-anything-acloudviewer process compass-export project.bin -o compass_data.xml --format xml

# Compass — Import foliations / lineations from scalar fields
cli-anything-acloudviewer process compass-import-fol input.ply --dip-sf Dip --dipdir-sf DipDir --plane-size 2.0
cli-anything-acloudviewer process compass-import-lin input.ply --trend-sf Trend --plunge-sf Plunge --length 2.0

# Compass — Refit planes & P21 intensity
cli-anything-acloudviewer process compass-refit project.bin
cli-anything-acloudviewer process compass-p21 input.ply --radius 10.0 --subsample 25 -o p21_output.ply

# SRA — Surface of revolution radial distance
cli-anything-acloudviewer process sra input.ply -o output.ply --profile profile.txt --axis Z
```

---

## Scalar Field Commands

Pattern: `cli-anything-acloudviewer sf <operation> input.ply [OPTIONS]`

### Creation & Conversion

| Command | Description | Key Options |
|---------|-------------|-------------|
| `coord-to-sf` | Coordinate → SF | `-o`, `--dimension` X\|Y\|Z |
| `convert-to-rgb` | Active SF → RGB | `-o` (uses active SF) |

### Arithmetic

| Command | Description | Key Options |
|---------|-------------|-------------|
| `arithmetic` | Unary operations | `-o`, `--sf-index`, `--operation` SQRT\|ABS\|INV\|EXP\|LOG\|LOG10 |
| `operation` | Binary op with constant | `-o`, `--sf-index`, `--operation` ADD\|SUB\|MULTIPLY\|DIVIDE, `--value` (float) |

### Analysis

| Command | Description | Key Options |
|---------|-------------|-------------|
| `gradient` | Compute SF gradient | `-o`, `--euclidean` (flag, uses active SF) |
| `filter` | Filter by SF range | `-o`, `--min`, `--max` (uses active SF) |

### Visualization

| Command | Description | Key Options |
|---------|-------------|-------------|
| `color-scale` | Apply color scale file | `-o`, `--scale-file` (XML path, uses active SF) |

### Management

| Command | Description | Key Options |
|---------|-------------|-------------|
| `set-active` | Set active SF | `-o`, `--sf-index` (int or name) |
| `rename` | Rename SF | `-o`, `--old` (index/name), `--new` (name) |
| `remove` | Remove SF | `-o`, `--sf-index` (int or name) |
| `remove-all` | Remove all SFs | `-o` |

---

## Normal Commands

Pattern: `cli-anything-acloudviewer normals <operation> input.ply [OPTIONS]`

| Command | Description | Key Options |
|---------|-------------|-------------|
| `octree` | Compute with octree | `-o`, `--radius` AUTO\|float, `--model` LS\|TRI\|QUADRIC |
| `orient-mst` | Orient consistently (MST) | `-o`, `--knn` (int, default 6) |
| `invert` | Flip directions | `-o` |
| `clear` | Remove all normals | `-o` |
| `to-dip` | Convert to dip/direction | `-o` (geology, creates Dip & Dip direction SFs) |
| `to-sfs` | Export as Nx, Ny, Nz SFs | `-o` (3 scalar fields for X/Y/Z components) |

**Note**: `process normals` also available for standard k-NN method.

---

## Reconstruction Commands

### Automatic (One Command)

```bash
cli-anything-acloudviewer reconstruct auto images/ -w workspace/ [OPTIONS]

Options:
  --quality low|medium|high|extreme
  --camera-model SIMPLE_RADIAL|PINHOLE|OPENCV|OPENCV_FISHEYE|...
  --no-dense                    # Skip dense reconstruction
  --use-gpu                     # Enable GPU acceleration
```

### Step-by-Step (Advanced)

| Command | Description | Key Options |
|---------|-------------|-------------|
| `extract-features` | SIFT feature extraction | `--database`, `--image-path` |
| `match` | Match features | `--method` exhaustive\|sequential\|spatial |
| `sparse` | Sparse SfM | `--database`, `--image-path`, `-o` |
| `undistort` | Undistort images | `--image-path`, `-i`, `-o` |
| `dense-stereo` | Compute depth maps | workspace_path |
| `fuse` | Fuse depth maps | workspace_path, `-o` |
| `poisson` | Poisson surface reconstruction | input.ply, `-o` |
| `delaunay-mesh` | Delaunay meshing | input.ply, `-o` |
| `texture-mesh` | Apply textures | workspace, `--mesh`, `-o` |
| `analyze-model` | Model statistics | model_path |
| `convert-model` | Export Colmap model | model_path, `-o`, `--output-type` PLY\|TXT\|BIN |

---

## SIBR Commands

```bash
# Prepare Colmap dataset for SIBR
cli-anything-acloudviewer sibr prepare-colmap workspace/ -o output/

# Prepare bundle file
cli-anything-acloudviewer sibr prepare-bundle scene.out -o output/

# Launch SIBR viewer (if implemented)
cli-anything-acloudviewer sibr viewer gaussian --model-path ./output/ --path ./dataset/
```

---

## Session Commands

```bash
cli-anything-acloudviewer session status              # Show current session state
cli-anything-acloudviewer session save output.ccx     # Save session (GUI mode)
```

---

## Scene Commands (GUI Mode)

```bash
cli-anything-acloudviewer scene list                  # List all entities
cli-anything-acloudviewer scene info <entity-id>      # Get entity details
cli-anything-acloudviewer scene add file.ply          # Load file to scene
cli-anything-acloudviewer scene remove <entity-id>    # Remove entity
cli-anything-acloudviewer scene show <entity-id>      # Show entity
cli-anything-acloudviewer scene hide <entity-id>      # Hide entity
cli-anything-acloudviewer scene clear                 # Remove all entities
```

---

## Entity Commands (GUI Mode)

```bash
cli-anything-acloudviewer entity rename <id> --name "New Name"
cli-anything-acloudviewer entity setColor <id> --r 255 --g 0 --b 0
cli-anything-acloudviewer entity setVisible <id> --visible true
```

---

## View Commands (GUI Mode)

```bash
cli-anything-acloudviewer view screenshot output.png
cli-anything-acloudviewer view screenshot output.png --width 1920 --height 1080
cli-anything-acloudviewer view orient front|back|top|bottom|left|right
cli-anything-acloudviewer view zoom <factor>
cli-anything-acloudviewer view reset
```

---

## Examples by Use Case

### Batch Processing Pipeline

```bash
# 1. Batch convert LAS to PLY
cli-anything-acloudviewer batch-convert ./scans/ ./ply/ -f .ply --filter-ext .las

# 2. Process each file: subsample → normals → roughness
for f in ./ply/*.ply; do
  base=$(basename "$f" .ply)
  cli-anything-acloudviewer process subsample "$f" -o "./down/${base}.ply" --voxel-size 0.05
  cli-anything-acloudviewer process normals "./down/${base}.ply" -o "./norm/${base}.ply"
  cli-anything-acloudviewer process roughness "./norm/${base}.ply" -o "./rough/${base}.ply"
done
```

### Registration Workflow

```bash
# 1. Subsample both clouds
cli-anything-acloudviewer process subsample source.ply -o source_down.ply --voxel-size 0.05
cli-anything-acloudviewer process subsample target.ply -o target_down.ply --voxel-size 0.05

# 2. Compute normals
cli-anything-acloudviewer process normals source_down.ply -o source_norm.ply --radius 0.1
cli-anything-acloudviewer process normals target_down.ply -o target_norm.ply --radius 0.1

# 3. Coarse alignment
cli-anything-acloudviewer process match-centers source_norm.ply target_norm.ply -o coarse.ply

# 4. ICP refinement
cli-anything-acloudviewer process icp coarse.ply target_norm.ply -o aligned.ply --iterations 100

# 5. Validate
cli-anything-acloudviewer process c2c-dist aligned.ply target_norm.ply -o validation.ply
```

### Mesh Reconstruction & Processing

```bash
# 1. From point cloud to mesh (headless)
cli-anything-acloudviewer process normals cloud.ply -o cloud_normals.ply --radius 0.05
cli-anything-acloudviewer process delaunay cloud_normals.ply -o mesh.ply --max-edge-length 0.0

# 2. Sample points from mesh (headless)
cli-anything-acloudviewer process sample-mesh mesh.ply -o sampled.ply --points 100000

# 3. Extract vertices (headless)
cli-anything-acloudviewer process extract-vertices mesh.ply -o vertices.ply

# 4. Compute volume (headless)
cli-anything-acloudviewer process mesh-volume mesh.ply

# 5. Convert format (headless)
cli-anything-acloudviewer convert mesh.ply mesh.obj

# Advanced mesh operations (GUI mode - requires entity_id):
# First: cli-anything-acloudviewer open mesh.ply (get entity_id from scene list)
# Then:
cli-anything-acloudviewer mesh simplify <entity_id> --method quadric --target-triangles 50000
cli-anything-acloudviewer mesh smooth <entity_id> --method laplacian --iterations 5
cli-anything-acloudviewer mesh subdivide <entity_id> --method loop --iterations 1
```

### Scalar Field Analysis Pipeline

```bash
# 1. Create height scalar field
cli-anything-acloudviewer sf coord-to-sf input.ply -o height.ply --dimension Z

# 2. Compute density
cli-anything-acloudviewer process density height.ply -o density.ply --radius 0.05

# 3. Scale density by 2.0
cli-anything-acloudviewer sf operation density.ply -o scaled.ply \
  --sf-index 0 --operation MULTIPLY --value 2.0

# 4. Compute sqrt of density
cli-anything-acloudviewer sf arithmetic density.ply -o sqrt_dens.ply \
  --sf-index 0 --operation SQRT

# 5. Set active SF and compute gradient
cli-anything-acloudviewer sf set-active density.ply -o density_active.ply --sf-index 0
cli-anything-acloudviewer sf gradient density_active.ply -o gradient.ply --euclidean

# 6. Convert gradient SF to RGB colors
cli-anything-acloudviewer sf convert-to-rgb gradient.ply -o colored.ply

# 7. Filter by value (uses active SF)
cli-anything-acloudviewer sf filter density_active.ply -o filtered.ply --min 0.5 --max 2.0
```

### 3D Reconstruction (Automatic)

```bash
# High quality with GPU
cli-anything-acloudviewer reconstruct auto ./images/ -w ./workspace/ \
  --quality high --use-gpu

# Medium quality with specific camera
cli-anything-acloudviewer reconstruct auto ./images/ -w ./workspace/ \
  --quality medium --camera-model OPENCV

# Sparse only (fast)
cli-anything-acloudviewer reconstruct auto ./images/ -w ./workspace/ \
  --quality low --no-dense
```

### 3D Reconstruction (Step-by-Step)

```bash
# 1. Extract features
cli-anything-acloudviewer reconstruct extract-features ./images/ -d db.db

# 2. Match features
cli-anything-acloudviewer reconstruct match db.db --method exhaustive

# 3. Sparse reconstruction
cli-anything-acloudviewer reconstruct sparse -d db.db --image-path ./images/ -o ./sparse/

# 4. Undistort images
cli-anything-acloudviewer reconstruct undistort --image-path ./images/ -i ./sparse/0 -o ./dense/

# 5. Dense stereo
cli-anything-acloudviewer reconstruct dense-stereo ./dense/

# 6. Fuse depth maps
cli-anything-acloudviewer reconstruct fuse ./dense/ -o ./dense/fused.ply

# 7. Poisson meshing
cli-anything-acloudviewer reconstruct poisson ./dense/fused.ply -o mesh.ply

# 8. Texture mapping
cli-anything-acloudviewer reconstruct texture-mesh ./dense/ -o ./textured/ --mesh mesh.ply
```

---

## Common Option Patterns

### Input/Output

```bash
-o, --output PATH          # Output file or directory
-i, --input PATH           # Input file (some commands)
-w, --workspace PATH       # Workspace directory (reconstruction)
-d, --database PATH        # Database file (Colmap)
```

### Processing Parameters

```bash
--voxel-size FLOAT         # Voxel size for subsampling
--radius FLOAT             # Neighborhood radius
--knn INTEGER              # K nearest neighbors
--sigma FLOAT              # Standard deviation multiplier threshold
--min-points INTEGER       # Minimum points per component
--target-count INTEGER     # Target triangle count
--iterations INTEGER       # Number of iterations
```

### Filters

```bash
--filter-ext EXT [EXT...]  # Filter by extensions (batch-convert)
--max-distance FLOAT       # Max distance threshold
--min FLOAT --max FLOAT    # Value range filter
```

### Axes & Dimensions

```bash
--axis X|Y|Z               # Coordinate axis
--dimension X|Y|Z          # Dimension selector
--type MEAN|GAUSS          # Computation type
```

### Colors

```bash
--r INTEGER --g INTEGER --b INTEGER    # RGB values (0-255)
--scale-type RAINBOW|GRAY|BWR|HSV     # Color scale type
```

---

## Output Modes

### JSON Mode (for scripts)

```bash
cli-anything-acloudviewer --json info
cli-anything-acloudviewer --json --mode headless process subsample input.ply -o out.ply
```

JSON output structure:
```json
{
  "status": "success",
  "result": { ... },
  "errors": []
}
```

### Text Mode (for humans)

```bash
cli-anything-acloudviewer info
cli-anything-acloudviewer formats
```

Human-readable formatted output.

---

## Modes

### Auto Mode (default)

- Tries GUI connection first (port 6001)
- Falls back to headless if GUI unavailable
- Best for interactive development

```bash
cli-anything-acloudviewer process subsample input.ply -o out.ply
```

### Headless Mode (recommended for CI/scripting)

- No GUI connection attempt
- Runs binary with `-SILENT` flag
- Faster startup, no port scanning

```bash
cli-anything-acloudviewer --mode headless process subsample input.ply -o out.ply
```

### GUI Mode (interactive)

- Requires running ACloudViewer with JSON-RPC enabled
- Real-time scene updates
- Supports view/scene/entity commands

```bash
# Start ACloudViewer, enable JSON-RPC plugin, then:
cli-anything-acloudviewer --mode gui scene list
cli-anything-acloudviewer --mode gui view screenshot capture.png
```

---

## Environment Variables

```bash
# Override binary path
export ACV_BINARY=/path/to/ACloudViewer

# Set default mode
export ACV_CLI_MODE=headless

# WebSocket URL (GUI mode)
export ACV_RPC_URL=ws://localhost:6001

# Log level
export ACV_LOG_LEVEL=DEBUG
```

---

## Tips & Tricks

### Performance

1. **Use headless mode for batch processing:**
   ```bash
   cli-anything-acloudviewer --mode headless batch-convert ...
   ```

2. **Subsample large clouds first:**
   ```bash
   cli-anything-acloudviewer process subsample huge.ply -o small.ply --voxel-size 0.1
   cli-anything-acloudviewer process normals small.ply -o result.ply
   ```

3. **Use CUDA builds for GPU acceleration** (10-100× faster)

### Debugging

1. **Enable verbose output:**
   ```bash
   cli-anything-acloudviewer -vv process subsample ...
   ```

2. **Check binary info:**
   ```bash
   cli-anything-acloudviewer info
   ```

3. **Validate formats:**
   ```bash
   cli-anything-acloudviewer formats | grep .las
   ```

### Scripting

1. **Use JSON output:**
   ```bash
   result=$(cli-anything-acloudviewer --json info)
   version=$(echo "$result" | jq -r '.version')
   ```

2. **Check exit codes:**
   ```bash
   if cli-anything-acloudviewer --mode headless convert in.ply out.pcd; then
     echo "Success"
   else
     echo "Failed with code $?"
   fi
   ```

3. **Parse JSON in Python:**
   ```python
   import subprocess, json
   result = subprocess.run(
       ["cli-anything-acloudviewer", "--json", "info"],
       capture_output=True, text=True
   )
   info = json.loads(result.stdout)
   print(info["device_api"])
   ```

---

## See Also

- **Full documentation**: [agent-integration/README.md](../README.md)
- **Troubleshooting**: [docs/TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- **JSON-RPC API**: [docs/JSON-RPC-API.md](JSON-RPC-API.md)
- **Testing guide**: [docs/TESTING.md](TESTING.md)
- **Examples**: `agent-integration/examples/`

---

## Quick Cheat Sheet

```bash
# Get help
cli-anything-acloudviewer --help
cli-anything-acloudviewer process --help
cli-anything-acloudviewer sf --help

# Common operations
cli-anything-acloudviewer convert input.ply output.pcd
cli-anything-acloudviewer process subsample input.ply -o out.ply --voxel-size 0.05
cli-anything-acloudviewer process crop input.ply -o cropped.ply --min-x -1 --min-y -1 --min-z -1 --max-x 1 --max-y 1 --max-z 1
cli-anything-acloudviewer process normals input.ply -o out.ply --radius 0.1
cli-anything-acloudviewer process icp source.ply target.ply -o aligned.ply --iterations 100
cli-anything-acloudviewer sf coord-to-sf input.ply -o height.ply --dimension Z
cli-anything-acloudviewer normals orient-mst input.ply -o oriented.ply --knn 6
cli-anything-acloudviewer process density input.ply -o density.ply --radius 0.05

# Batch operations
cli-anything-acloudviewer batch-convert ./scans/ ./out/ -f .ply

# 3D reconstruction
cli-anything-acloudviewer reconstruct auto ./images/ -w ./workspace/ --quality high

# Info
cli-anything-acloudviewer info
cli-anything-acloudviewer formats
```
