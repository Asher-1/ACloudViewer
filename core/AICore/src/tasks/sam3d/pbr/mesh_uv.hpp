// xatlas UV parameterization for the native post-processing path.
#pragma once

#include <string>

#include "asset_io.hpp"

namespace sam3d {

// Matches xatlas.parametrize(vertices, faces): duplicate seam vertices using
// xref, replace the triangle index buffer, and write normalized UVs.
bool parameterize_mesh_xatlas(NativeMesh& mesh, std::string& error);

}  // namespace sam3d
