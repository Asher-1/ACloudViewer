// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <memory>
#include <vector>

#include "renders/base/TextureRendererBase.h"

class vtkActor;
class vtkLODActor;
class vtkPolyData;
class vtkRenderer;
class ccMaterialSet;

namespace PclUtils {
namespace renders {

/**
 * @brief Texture rendering manager
 *
 * Centralized manager for texture rendering that:
 * - Automatically selects the appropriate renderer based on material properties
 * - Provides unified interface for texture rendering operations
 * - Manages multiple renderer instances (PBR, multi-texture, material-only)
 */
class TextureRenderManager {
public:
    TextureRenderManager();
    ~TextureRenderManager();

    /**
     * @brief Apply rendering to actor
     * @param actor VTK actor to render
     * @param materials Material set
     * @param polydata Polygon data (for texture coordinates)
     * @param renderer VTK renderer (for lighting setup)
     * @return true on success
     */
    bool Apply(vtkLODActor* actor,
               const ccMaterialSet* materials,
               vtkPolyData* polydata,
               vtkRenderer* renderer);

    /**
     * @brief Update existing actor with new materials
     * @param actor VTK actor to update
     * @param materials Material set
     * @param polydata Polygon data
     * @param renderer VTK renderer
     * @return true on success
     */
    bool Update(vtkActor* actor,
                const ccMaterialSet* materials,
                vtkPolyData* polydata,
                vtkRenderer* renderer);

    /**
     * @brief Detect and select appropriate renderer for materials
     * @param materials Material set
     * @return Selected renderer, or nullptr if none can handle
     */
    TextureRendererBase* SelectRenderer(const ccMaterialSet* materials) const;

    /**
     * @brief Get renderer by mode
     */
    TextureRendererBase* GetRenderer(RenderingMode mode) const;

private:
    /**
     * @brief Initialize all available renderers
     */
    void InitializeRenderers();

    /**
     * @brief Analyze material properties
     */
    struct MaterialAnalysis {
        bool has_pbr_textures = false;
        bool has_multiple_map_kd = false;
        bool has_any_texture = false;  // Any texture type (not just PBR)
        size_t material_count = 0;
    };
    MaterialAnalysis AnalyzeMaterials(const ccMaterialSet* materials) const;

    // Helper method to check if materials have any texture files
    bool HasAnyTextureFiles(const ccMaterialSet* materials) const;

    std::vector<std::unique_ptr<TextureRendererBase>> renderers_;
};

}  // namespace renders
}  // namespace PclUtils
