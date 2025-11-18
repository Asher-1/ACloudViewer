// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <pcl/TextureMesh.h>

#include <Eigen/Dense>
#include <vector>

// Forward declarations
class vtkLODActor;
class vtkPolyData;
class vtkRenderer;
class vtkRenderWindow;
class ccGenericMesh;
class ccMaterialSet;

namespace PclUtils {
namespace renders {

class TextureRenderManager;

/**
 * @brief Utility class for applying textures to mesh actors
 *
 * Handles texture application from mesh to VTK actors.
 * Supports both PCLTextureMesh (legacy) and ccGenericMesh (preferred) sources.
 */
class MeshTextureApplier {
public:
    /**
     * @brief Apply textures from ccGenericMesh (preferred method)
     * @param actor VTK actor to apply textures to
     * @param mesh ccGenericMesh containing materials and texture coordinates
     * @param polydata Polygon data (for texture coordinates)
     * @param render_manager Texture render manager for applying materials
     * @param renderer VTK renderer (optional)
     * @return true on success
     */
    static bool ApplyTexturesFromCCMesh(
            vtkLODActor* actor,
            ccGenericMesh* mesh,  // Non-const because getTriangleVertIndexes is
                                  // non-const
            vtkPolyData* polydata,
            TextureRenderManager* render_manager,
            vtkRenderer* renderer);

    /**
     * @brief Apply textures from ccMaterialSet with texture coordinates
     * @param actor VTK actor to apply textures to
     * @param materials Material set containing texture information
     * @param tex_coordinates Texture coordinates per material
     * @param polydata Polygon data (for texture coordinates)
     * @param render_manager Texture render manager for applying materials
     * @param renderer VTK renderer (optional)
     * @return true on success
     */
    static bool ApplyTexturesFromMaterialSet(
            vtkLODActor* actor,
            const ccMaterialSet* materials,
            const std::vector<std::vector<Eigen::Vector2f>>& tex_coordinates,
            vtkPolyData* polydata,
            TextureRenderManager* render_manager,
            vtkRenderer* renderer);

    /**
     * @brief Apply PBR textures from PCLTextureMesh (deprecated)
     * @deprecated Use ApplyTexturesFromCCMesh or ApplyTexturesFromMaterialSet
     * instead
     * @note This method converts pcl::TexMaterial to ccMaterialSet internally
     */
    static bool ApplyPBRTextures(vtkLODActor* actor,
                                 const pcl::TextureMesh& mesh,
                                 vtkPolyData* polydata,
                                 TextureRenderManager* render_manager,
                                 vtkRenderer* renderer);

    /**
     * @brief Apply textures from PCLTextureMesh to actor (traditional path)
     * @deprecated Use TextureRenderManager instead
     */
    static bool ApplyTraditionalTextures(vtkLODActor* actor,
                                         const pcl::TextureMesh& mesh,
                                         vtkPolyData* polydata,
                                         vtkRenderWindow* render_window);
};

}  // namespace renders
}  // namespace PclUtils
