# Example: Fixing Docstrings in geometry/pointcloud.cpp

This document shows a step-by-step example of fixing docstring issues in a real file.

## Current Issues (Example)

Let's say we have warnings like this:

```
WARNING: Inline emphasis start-string without end-string.
WARNING: Unexpected indentation.
ERROR: Undefined substitution referenced: "X".
```

## Step 1: Identify the Problem Code

Looking at `libs/Python/pybind/geometry/pointcloud.cpp`, we might find:

```cpp
// ❌ PROBLEM: Unescaped asterisk
.def("scale", &PointCloud::Scale,
     "Scale points by a factor *k",
     "k"_a)

// ❌ PROBLEM: Missing indentation in Args
.def("transform", &PointCloud::Transform,
     "Transform the point cloud.\n"
     "Args:\n"
     "matrix: 4x4 transformation matrix\n"  // Missing indentation!
     "Returns:\n"
     "Transformed point cloud.",
     "matrix"_a)
```

## Step 2: Fix Using Templates

### Fix 1: Simple One-liner

```cpp
// ✅ FIXED: Escaped asterisk or removed emphasis
.def("scale", &PointCloud::Scale,
     "Scale points by a factor k",
     "k"_a)
```

### Fix 2: Proper Args/Returns Formatting

```cpp
// ✅ FIXED: Proper indentation
.def("transform", &PointCloud::Transform,
     "Transform the point cloud.\n\n"
     "Args:\n"
     "    matrix (np.ndarray): 4x4 transformation matrix.\n\n"
     "Returns:\n"
     "    PointCloud: Transformed point cloud.",
     "matrix"_a)
```

## Step 3: Use Docstring Injection System

ACloudViewer already has a docstring injection system (like Open3D). For complex functions:

```cpp
// In the .def() - keep it simple
m.def("compute_point_cloud_distance", &PointCloud::ComputeDistance,
      "Compute distance between two point clouds", 
      "source"_a, "target"_a)

// Then inject detailed documentation
docstring::FunctionDocInject(
    m, "compute_point_cloud_distance",
    {{"source", "The source point cloud."},
     {"target", "The target point cloud."}});
```

## Step 4: Common Patterns

### Pattern 1: Boolean Parameters

```cpp
// ✅ GOOD
.def("remove_duplicates", &PointCloud::RemoveDuplicates,
     "Remove duplicate points.\n\n"
     "Args:\n"
     "    remove_nan (bool): If True, also remove NaN points. Default: False.\n\n"
     "Returns:\n"
     "    PointCloud: Point cloud with duplicates removed.",
     "remove_nan"_a = false)
```

### Pattern 2: Optional Parameters with Defaults

```cpp
// ✅ GOOD
.def("voxel_down_sample", &PointCloud::VoxelDownSample,
     "Downsample point cloud using voxel grid.\n\n"
     "Args:\n"
     "    voxel_size (float): Size of voxel for downsampling.\n"
     "    min_bound (np.ndarray, optional): Minimum bound. Default: None.\n"
     "    max_bound (np.ndarray, optional): Maximum bound. Default: None.\n\n"
     "Returns:\n"
     "    PointCloud: Downsampled point cloud.",
     "voxel_size"_a, "min_bound"_a = py::none(), "max_bound"_a = py::none())
```

### Pattern 3: With Example Code

```cpp
// ✅ GOOD
.def("estimate_normals", &PointCloud::EstimateNormals,
     "Estimate normals for all points.\n\n"
     "Args:\n"
     "    search_param (KDTreeSearchParam): KDTree search parameters.\n\n"
     "Returns:\n"
     "    bool: True if successful, False otherwise.\n\n"
     "Example::\n\n"
     "    pcd = cloudViewer.geometry.PointCloud()\n"
     "    pcd.estimate_normals(\n"
     "        search_param=cloudViewer.geometry.KDTreeSearchParamHybrid(\n"
     "            radius=0.1, max_nn=30))",
     "search_param"_a)
```

## Step 5: Properties

```cpp
// ✅ GOOD
.def_property("points", &PointCloud::GetPoints, &PointCloud::SetPoints,
              "Point coordinates.\n\n"
              "Type:\n"
              "    np.ndarray: Nx3 float64 array of points.")
```

## Step 6: Verify the Fix

After making changes:

```bash
# 1. Rebuild documentation
cd build_app
make sphinx-html 2>&1 | grep pointcloud | grep WARNING

# 2. Check before/after count
# Before: 15 warnings in pointcloud
# After:   3 warnings in pointcloud

# 3. View the generated docs
firefox docs/_out/html/python_api/cloudViewer.geometry.PointCloud.html
```

## Complete Example: Before & After

### Before (Has Warnings)

```cpp
.def("compute_convex_hull", &PointCloud::ComputeConvexHull,
     "Compute convex hull. Returns: tuple (mesh, indices)")
```

**Problems:**
- No proper Args section
- Malformed Returns section (should use proper RST format)
- No type information

### After (No Warnings)

```cpp
.def("compute_convex_hull", &PointCloud::ComputeConvexHull,
     "Compute the convex hull of the point cloud.\n\n"
     "Returns:\n"
     "    tuple: A tuple containing:\n"
     "        - mesh (TriangleMesh): The convex hull mesh.\n"
     "        - indices (list): Indices of points on the hull.")
```

**Or using injection system:**

```cpp
// In binding
m.def("compute_convex_hull", &PointCloud::ComputeConvexHull,
      "Compute the convex hull of the point cloud")

// Inject detailed docs
docstring::FunctionDocInject(
    m, "compute_convex_hull",
    {}, // No args
    "Returns a tuple (mesh, indices) where mesh is the convex hull "
    "TriangleMesh and indices is the list of point indices on the hull.");
```

## Quick Reference: Common Fixes

| Problem | Fix |
|---------|-----|
| Unescaped `*` | Replace with `\\*` or remove emphasis |
| Missing Args indentation | Add 4 spaces before argument names |
| Missing Returns indentation | Add 4 spaces before return description |
| Example without `::` | Change `Example:` to `Example::` |
| `'True'` in text | Change to `` `True` `` |
| Broken line continuation | Add `\n\n` between sections |
| Missing type info | Add `(type)` after arg names |

## Testing Your Fixes

```bash
# Quick test for single module
cd build_app
make sphinx-html 2>&1 | grep "geometry" | grep WARNING | wc -l

# Full test
./docs/scripts/count_warnings.sh

# Visual inspection
firefox docs/_out/html/python_api/cloudViewer.geometry.html
```

## Tips

1. **Start Small**: Fix one function, verify, then continue
2. **Use Examples**: Copy patterns from Open3D or fixed functions
3. **Check Generated Docs**: Always view the HTML to ensure it looks right
4. **Follow Templates**: Use the templates in DOCSTRING_STYLE_GUIDE.md
5. **Be Consistent**: Use the same style throughout the module

## Next Steps

After fixing one file:

1. Count warnings: `./docs/scripts/count_warnings.sh`
2. Compare before/after numbers
3. Commit: `git commit -m "docs: fix pointcloud.cpp docstrings"`
4. Move to next file

---

**See Also:**
- `docs/DOCSTRING_STYLE_GUIDE.md` - Complete style guide
- `docs/scripts/README.md` - Tool usage
- `docs/DOCUMENTATION_WARNINGS.md` - Problem explanation
