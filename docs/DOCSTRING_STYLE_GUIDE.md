# ACloudViewer Docstring Style Guide

## Overview

This guide defines the standard format for docstrings in ACloudViewer's C++ pybind11 bindings to ensure compatibility with Sphinx and generate high-quality Python API documentation.

## Motivation

**Problem**: Our current C++ pybind11 docstrings generate ~200+ Sphinx warnings due to improper RST formatting.

**Goal**: Follow Open3D's approach - write clean, well-formatted docstrings that Sphinx can parse without warnings.

## Basic Principles

### 1. Keep It Simple

Use plain, concise descriptions without complex RST directives in inline docstrings:

```cpp
// ✅ GOOD - Simple and clear
.def("has_points", &PointCloud::HasPoints,
     "Returns True if the point cloud contains points.")

// ❌ BAD - Overly complex with broken RST
.def("has_points", &PointCloud::HasPoints,
     "Returns: bool\n"  // This breaks RST parsing
     "  True if has points")
```

### 2. Use Proper RST Format for Multi-Line Docs

When documenting complex functions, use proper RST format with correct indentation:

```cpp
// ✅ GOOD - Proper RST format
.def("voxel_down_sample", &PointCloud::VoxelDownSample,
     "Function to downsample input pointcloud into output pointcloud with "
     "a voxel.\n\n"
     "Normals and colors are averaged if they exist.\n\n"
     "Args:\n"
     "    voxel_size (float): Size of voxel for downsampling.\n\n"
     "Returns:\n"
     "    PointCloud: Downsampled point cloud.",
     "voxel_size"_a)

// ❌ BAD - Improper indentation and format
.def("voxel_down_sample", &PointCloud::VoxelDownSample,
     "Function to downsample\n"
     "Args:\nvoxel_size: float\n"  // Missing proper indentation
     "Returns:\nPointCloud",
     "voxel_size"_a)
```

### 3. Escape Special RST Characters

RST treats certain characters as markup. Escape them when needed:

```cpp
// ✅ GOOD - Escaped asterisks
.def("compute", &Class::Compute,
     "Compute value using formula: result \\* 2")

// ❌ BAD - Unescaped asterisks create inline emphasis
.def("compute", &Class::Compute,
     "Compute value using *result*")  // Sphinx sees this as *result*
```

### 4. Use Consistent Argument Documentation

```cpp
// ✅ GOOD - Consistent format
.def("crop", &PointCloud::Crop,
     "Function to crop input pointcloud into output pointcloud.\n\n"
     "Args:\n"
     "    bounding_box (AxisAlignedBoundingBox): The bounding box.\n"
     "    invert (bool): If True, invert the cropping. Default: False.\n\n"
     "Returns:\n"
     "    PointCloud: Cropped point cloud.",
     "bounding_box"_a, "invert"_a = false)
```

## Open3D Docstring Strategy

Open3D uses a sophisticated two-layer system:

### Layer 1: Inline Docstrings (Minimal)

```cpp
// Simple one-liner for basic functions
.def("has_points", &PointCloud::HasPoints,
     "Returns True if the point cloud contains points.")
```

### Layer 2: Docstring Injection System

For complex functions, Open3D uses a separate docstring injection system:

```cpp
// In binding code - minimal docstring
m.def("read_point_cloud", &io::ReadPointCloud,
      "Function to read PointCloud from file", "filename"_a, ...)

// In separate docstring file - detailed documentation
docstring::FunctionDocInject(
    m, "read_point_cloud",
    {{"filename", "Path to file."},
     {"format", "File format (auto-detected if not specified)."},
     {"remove_nan_points", "If true, remove NaN points."}});
```

**Benefits:**
- Separates code from documentation
- Easier to maintain and update
- Cleaner C++ code

## Common RST Formatting Issues

### Issue 1: Unexpected Indentation

```cpp
// ❌ BAD
"Args:\n"
"filename: str\n"  // Missing indentation
"    The file path"

// ✅ GOOD
"Args:\n"
"    filename (str): The file path\n"  // Proper 4-space indentation
```

### Issue 2: Inline Emphasis Without End String

```cpp
// ❌ BAD
"Set to *True to enable"  // Unclosed emphasis

// ✅ GOOD
"Set to ``True`` to enable"  // Use double backticks for literals
```

### Issue 3: Literal Block Expected

```cpp
// ❌ BAD
"Example:\nload_file('test.pcd')"  // No double colon before code

// ✅ GOOD
"Example::\n\n    load_file('test.pcd')"  // Double colon and indentation
```

### Issue 4: Undefined Substitutions

```cpp
// ❌ BAD
"Returns |X| and |Y|"  // Undefined substitutions

// ✅ GOOD
"Returns X and Y"  // Use plain text
```

## Template Examples

### Template 1: Simple Function

```cpp
.def("method_name", &Class::MethodName,
     "Brief description of what this method does.")
```

### Template 2: Function with Arguments

```cpp
.def("method_name", &Class::MethodName,
     "Brief description.\n\n"
     "Args:\n"
     "    arg1 (type): Description of arg1.\n"
     "    arg2 (type): Description of arg2. Default: value.\n\n"
     "Returns:\n"
     "    ReturnType: Description of return value.",
     "arg1"_a, "arg2"_a = default_value)
```

### Template 3: Function with Example

```cpp
.def("method_name", &Class::MethodName,
     "Brief description.\n\n"
     "This function does something interesting with the data.\n\n"
     "Args:\n"
     "    input (Type): Input data.\n\n"
     "Returns:\n"
     "    Type: Output data.\n\n"
     "Example::\n\n"
     "    result = obj.method_name(input_data)\n"
     "    print(result)",
     "input"_a)
```

### Template 4: Property

```cpp
.def_property("property_name", &Class::GetProperty, &Class::SetProperty,
              "Description of the property.\n\n"
              "Type:\n"
              "    PropertyType")
```

### Template 5: Static Method

```cpp
.def_static("static_method", &Class::StaticMethod,
            "Static method description.\n\n"
            "Args:\n"
            "    arg (type): Argument description.\n\n"
            "Returns:\n"
            "    ReturnType: Return value description.",
            "arg"_a)
```

## Migration Strategy

### Phase 1: Audit (1-2 weeks)

1. **Generate Warning Report**
   ```bash
   cd build_app
   make sphinx-html 2>&1 | grep -E "^(WARNING|ERROR)" > /tmp/doc_warnings.txt
   ```

2. **Categorize Warnings**
   - RST format errors by type
   - Duplicate object descriptions
   - Missing/broken references

3. **Prioritize Modules**
   - Start with core modules: `geometry`, `io`, `utility`
   - Then pipelines: `registration`, `segmentation`
   - Finally specialized: `ml`, `reconstruction`

### Phase 2: Fix (2-3 months)

**Week-by-week plan:**

- **Weeks 1-2**: `cloudViewer.utility` (~50 warnings)
- **Weeks 3-4**: `cloudViewer.io` (~40 warnings)
- **Weeks 5-6**: `cloudViewer.geometry` (~80 warnings)
- **Weeks 7-8**: `cloudViewer.pipelines` (~30 warnings)
- **Weeks 9-10**: `cloudViewer.visualization` (~20 warnings)
- **Weeks 11-12**: Remaining modules

### Phase 3: Prevention (Ongoing)

1. **Pre-commit Hook**
   - Check docstring format before commit
   - Validate RST syntax

2. **CI/CD Check**
   - Build docs and count warnings
   - Fail if warnings increase

3. **Code Review Checklist**
   - [ ] Docstrings follow style guide
   - [ ] No new Sphinx warnings
   - [ ] Examples tested

## Tools and Scripts

### 1. Docstring Checker Script

```bash
#!/bin/bash
# Check for common docstring issues in C++ files

find python/pybind -name "*.cpp" -exec grep -H "\.def(" {} \; | \
  grep -v "    \"" | \
  echo "Found functions without proper docstring indentation"
```

### 2. Warning Counter

```bash
#!/bin/bash
# Count Sphinx warnings by category

cd build_app
make sphinx-html 2>&1 | \
  grep -E "^WARNING:" | \
  cut -d: -f3 | \
  sort | uniq -c | sort -rn
```

### 3. Module-specific Report

```bash
#!/bin/bash
# Generate warning report for specific module

MODULE="geometry"
cd build_app
make sphinx-html 2>&1 | \
  grep "python_api/cloudViewer.${MODULE}" | \
  grep -E "^(WARNING|ERROR)"
```

## References

- [Sphinx RST Primer](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)
- [pybind11 Documentation](https://pybind11.readthedocs.io/en/stable/advanced/misc.html#documentation)
- [Open3D Docstring System](https://github.com/isl-org/Open3D/blob/main/cpp/pybind/docstring.h)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)

## Getting Help

- **Documentation Issues**: `docs/DOCUMENTATION_WARNINGS.md`
- **Build Guide**: `docs/automation/BUILD_DOCUMENTATION.md`
- **GitHub Issues**: Tag with `documentation`

---

**Last Updated**: January 2026  
**Maintainer**: ACloudViewer Documentation Team
