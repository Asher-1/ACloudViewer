// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/** @file MultiTextureRenderer.h
 *  @brief Traditional multi-texture rendering using VTK multi-texture API
 */

#include <ecvMaterial.h>

#include "TextureRendererBase.h"

// Forward declarations
class vtkTexture;
class vtkActor;
class vtkTextureUnitManager;
class vtkRenderWindow;

namespace Visualization {
namespace renders {

/**
 * @brief Multi-texture renderer
 *
 * Traditional multi-texture rendering using VTK's multi-texture API.
 * Supports multiple textures with blending modes (REPLACE, ADD).
 * Used when:
 * - Multiple map_Kd textures are present (VTK PBR limitation)
 * - Traditional texture rendering is preferred
 */
class MultiTextureRenderer : public TextureRendererBase {
public:
    MultiTextureRenderer();
    ~MultiTextureRenderer() override = default;

    /// @param material_count Number of materials
    /// @param has_pbr_textures Whether PBR textures present
    /// @param has_multiple_map_kd Whether multiple map_Kd textures
    /// @return true if this renderer can handle
    bool CanHandle(size_t material_count,
                   bool has_pbr_textures,
                   bool has_multiple_map_kd) const override;

    /// @return MULTI_TEXTURE
    RenderingMode GetMode() const override;

    /// @param actor VTK actor to render
    /// @param materials Material set
    /// @param polydata Polygon data
    /// @param renderer VTK renderer
    /// @return true on success
    bool Apply(vtkLODActor* actor,
               const class ccMaterialSet* materials,
               vtkPolyData* polydata,
               vtkRenderer* renderer) override;

    /// @param actor VTK actor to update
    /// @param materials Material set
    /// @param polydata Polygon data
    /// @param renderer VTK renderer
    /// @return true on success
    bool Update(vtkActor* actor,
                const class ccMaterialSet* materials,
                vtkPolyData* polydata,
                vtkRenderer* renderer) override;

    /// @return Renderer name string
    std::string GetName() const override;

    /**
     * @brief Apply material properties to actor
     * @param material Material to apply
     * @param actor VTK actor
     * @param intensity_scale Intensity multiplier
     * @return true on success
     * @note This is a public method that can be used standalone
     */
    bool ApplyMaterial(ccMaterial::CShared material,
                       vtkActor* actor,
                       float intensity_scale = 1.0f) const;

private:
    /**
     * @brief Load texture from material
     */
    int LoadTexture(ccMaterial::CShared material, ::vtkTexture* vtk_tex) const;

    /**
     * @brief Get texture unit manager
     */
    ::vtkTextureUnitManager* GetTextureUnitManager(
            ::vtkRenderWindow* render_window) const;
};

}  // namespace renders
}  // namespace Visualization
