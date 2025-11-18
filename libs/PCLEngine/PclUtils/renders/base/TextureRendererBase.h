// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <vtkSmartPointer.h>

#include <memory>
#include <string>
#include <vector>

class vtkActor;
class vtkLODActor;
class vtkPolyData;
class vtkRenderer;
class vtkRenderWindow;

// Forward declarations
class ccMaterialSet;
class ccMaterial;

namespace PclUtils {
namespace renders {

// Forward declarations
class MaterialConverter;

/**
 * @brief Rendering mode enumeration
 */
enum class RenderingMode {
    PBR,            // Physically Based Rendering (VTK 9+)
    MULTI_TEXTURE,  // Traditional multi-texture rendering
    MATERIAL_ONLY   // Material properties only, no textures
};

/**
 * @brief Base class for all texture renderers
 *
 * Provides common interface and shared functionality for different rendering
 * modes. Subclasses implement specific rendering logic (PBR, multi-texture,
 * etc.)
 */
class TextureRendererBase {
public:
    TextureRendererBase() = default;
    virtual ~TextureRendererBase() = default;

    /**
     * @brief Check if this renderer can handle the given material
     * @param material_count Number of materials
     * @param has_pbr_textures Whether material has PBR textures
     * @param has_multiple_map_kd Whether material has multiple map_Kd textures
     * @return true if this renderer can handle the material
     */
    virtual bool CanHandle(size_t material_count,
                           bool has_pbr_textures,
                           bool has_multiple_map_kd) const = 0;

    /**
     * @brief Get the rendering mode this renderer implements
     */
    virtual RenderingMode GetMode() const = 0;

    /**
     * @brief Apply rendering to actor
     * @param actor VTK actor to render
     * @param materials Material set
     * @param polydata Polygon data (for texture coordinates)
     * @param renderer VTK renderer (for lighting setup)
     * @return true on success
     */
    virtual bool Apply(vtkLODActor* actor,
                       const class ccMaterialSet* materials,
                       vtkPolyData* polydata,
                       vtkRenderer* renderer) = 0;

    /**
     * @brief Update existing actor with new materials
     * @param actor VTK actor to update
     * @param materials Material set
     * @param polydata Polygon data
     * @param renderer VTK renderer
     * @return true on success
     */
    virtual bool Update(vtkActor* actor,
                        const class ccMaterialSet* materials,
                        vtkPolyData* polydata,
                        vtkRenderer* renderer) = 0;

    /**
     * @brief Get renderer name for logging
     */
    virtual std::string GetName() const = 0;

protected:
    /**
     * @brief Helper: Clear all textures from actor
     */
    void ClearTextures(vtkActor* actor);

    /**
     * @brief Helper: Validate actor
     */
    bool ValidateActor(vtkActor* actor) const;

    /**
     * @brief Helper: Validate materials
     */
    bool ValidateMaterials(const class ccMaterialSet* materials) const;
};

}  // namespace renders
}  // namespace PclUtils
