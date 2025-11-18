// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <ecvMaterial.h>

#include "renders/base/TextureRendererBase.h"

// Forward declarations
class vtkTexture;
class vtkActor;
class vtkTextureUnitManager;
class vtkRenderWindow;

namespace PclUtils {
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

    bool CanHandle(size_t material_count,
                   bool has_pbr_textures,
                   bool has_multiple_map_kd) const override;

    RenderingMode GetMode() const override;

    bool Apply(vtkLODActor* actor,
               const class ccMaterialSet* materials,
               vtkPolyData* polydata,
               vtkRenderer* renderer) override;

    bool Update(vtkActor* actor,
                const class ccMaterialSet* materials,
                vtkPolyData* polydata,
                vtkRenderer* renderer) override;

    std::string GetName() const override;

    /**
     * @brief Apply material properties to actor
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
}  // namespace PclUtils
