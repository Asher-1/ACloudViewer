// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <memory>

#include "renders/base/TextureRendererBase.h"
#include "renders/pbr/VtkMultiTextureRenderer.h"

namespace PclUtils {
namespace renders {

/**
 * @brief Unified PBR (Physically Based Rendering) renderer
 *
 * Uses VTK 9+ PBR rendering pipeline for realistic material rendering.
 * Supports both:
 * - PBR textures (baseColor, normal, metallic, roughness, AO)
 * - Material-only rendering (no textures, only material properties)
 *
 * VtkMultiTextureRenderer::ApplyPBRMaterial automatically detects the rendering
 * mode (PBR/TEXTURED/MATERIAL_ONLY) based on material properties and applies
 * accordingly.
 */
class PBRRenderer : public TextureRendererBase {
public:
    PBRRenderer();
    ~PBRRenderer() override = default;

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

private:
    std::unique_ptr<VtkUtils::VtkMultiTextureRenderer> vtk_renderer_;
};

}  // namespace renders
}  // namespace PclUtils
