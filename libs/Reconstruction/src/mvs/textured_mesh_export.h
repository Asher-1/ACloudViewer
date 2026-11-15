// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <filesystem>

#include "mvs/texture_mapping.h"
#include "util/ply.h"

namespace colmap {
namespace mvs {

// Export the mesh-texture-mapping result as a standard OBJ + MTL pair through
// the repository's canonical IO chain instead of an ad-hoc writer: the mesh is
// assembled as a ccMesh with per-triangle texture-coordinate and material
// indexes plus a ccMaterial carrying the atlas texture, and handed to
// ObjFilter::saveToFile, so mtllib/usemtl/vt/f-vt syntax and the MTL texture
// emission are owned by the same reader/writer pair used everywhere else
// (GUI load/save, qMeshIO, cli convert).
//
// The atlas PNG is written next to the OBJ (as <stem>_map_Kd.png) and the MTL
// references it by its relative filename, as produced by
// ccMaterialSet::saveAsMTL.
//
// UV convention: MeshTextureMappingResult::face_uvs carries GL texture
// coordinates (v = 0 at the image bottom, see texture_mapping.cc
// ComputeFaceUVs), which IS the OBJ standard, so the vt pool is written
// verbatim. ACloudViewer's render chain stays consistent without any
// conversion: ObjFilter stores vt as-is and the VTK texture upload flips
// the image (vtkQImageToImageSource), so t = 0 samples the image bottom
// row. Do NOT add a 1 - v adaptation here: face_uvs are GL-convention,
// unlike MvsTexturing::SaveOBJModel whose input atlas.texcoords are
// image-top normalized and therefore DO need the flip before ObjFilter.
//
// Returns false and logs on failure.
bool ExportTexturedMeshAsObj(const MeshTextureMappingResult& result,
                             const PlyMesh& mesh,
                             const std::filesystem::path& output_path);

}  // namespace mvs
}  // namespace colmap
