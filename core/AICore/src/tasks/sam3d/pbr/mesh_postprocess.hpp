// Native geometry-only stages from the official SAM 3D Objects postprocess.
#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "asset_io.hpp"

namespace sam3d {

struct MeshSimplifyOptions {
    // This is vtkQuadricDecimation's target reduction: 0.95 removes 95% of
    // the input triangles, exactly as PyVista PolyData.decimate(0.95) does.
    float target_reduction = 0.95f;
};

// Applies the exact VTK filter and default parameters selected by the checked
// in PyVista 0.48.4 call. UVs and vertex attributes are intentionally cleared:
// the official texture path parameterizes only after simplification.
bool simplify_mesh_official_vtk(NativeMesh& mesh, const MeshSimplifyOptions& options,
                                std::string& error);

// Statistics from the visibility/mincut portion of the official
// postprocessing path. Boundary remeshing is deliberately not included: the
// official implementation delegates that last operation to GPL MeshFix, which
// must not silently enter this distributable runtime.
struct MeshVisibilityFilterStats {
    size_t invisible_face_count = 0;
    size_t outer_face_count = 0;
    size_t mincut_face_count = 0;
    size_t removed_face_count = 0;
};

struct MeshVisibilityFilterOptions {
    // Same default as postprocessing_utils._fill_holes(). This is the
    // official cut-loop proxy, not an arbitrary geometric cleanup setting.
    float max_hole_size = 0.04f;
};

// Replays the official _fill_holes visibility-to-mincut decision once the
// per-face visibility frequencies have been produced. `visibility` is one
// value per current mesh face, normalized by the view count. The function
// implements the official component quantile threshold, weighted undirected
// dual graph, source/target mincut, and cut validation. It intentionally does
// not call MeshFix; native boundary remeshing remains a separate parity item.
bool filter_invisible_faces_official(NativeMesh& mesh, const std::vector<float>& visibility,
                                     const MeshVisibilityFilterOptions& options,
                                     MeshVisibilityFilterStats* stats, std::string& error,
                                     std::vector<uint32_t>* mincut_candidates = nullptr);

// Calls the same C++ TMesh implementation used by the official
// pymeshfix.PyTMesh binding: fill_small_boundaries(nbe, refine). This exact
// path is compiled only when SAM3D_GGML_MESHFIX_GPL=ON because its upstream is
// GPL-3.0-or-commercial. The normal build returns a clear configuration error
// instead of silently substituting a different hole triangulator.
bool repair_mesh_boundaries_official_meshfix(NativeMesh& mesh, int max_boundary_edges,
                                             bool refine, std::string& error);

}  // namespace sam3d
