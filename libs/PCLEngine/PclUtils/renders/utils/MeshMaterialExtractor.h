// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "renders/TextureRenderManager.h"

class vtkLODActor;
class vtkPolyData;
class ccGenericMesh;

namespace PclUtils {
namespace renders {

/**
 * @brief Utility class for extracting and applying materials from mesh
 *
 * Handles material extraction from ccGenericMesh and application to VTK actors.
 */
class MeshMaterialExtractor {
public:
    /**
     * @brief Extract and apply material from ccMesh to actor
     * @param actor VTK actor to apply material to
     * @param mesh ccMesh object containing materials
     * @param polydata Polygon data (for texture coordinates)
     * @param render_manager Texture render manager for applying materials
     * @param renderer VTK renderer (optional)
     * @return true on success
     */
    static bool ApplyMaterialFromMesh(vtkLODActor* actor,
                                      const ccGenericMesh* mesh,
                                      vtkPolyData* polydata,
                                      TextureRenderManager* render_manager,
                                      class vtkRenderer* renderer = nullptr);
};

}  // namespace renders
}  // namespace PclUtils
