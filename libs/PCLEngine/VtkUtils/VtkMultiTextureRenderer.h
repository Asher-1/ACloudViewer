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
#include <unordered_map>
#include <vector>

class vtkActor;
class vtkPolyData;
class vtkTexture;
class vtkImageData;
class vtkPolyDataMapper;
class vtkRenderer;
class vtkProperty;

namespace VtkUtils {

/**
 * @brief Multi-texture renderer - Efficient and robust multi-texture mesh
 * rendering support
 *
 * Features:
 * - Support PBR material rendering with multiple textures
 * - Automatic texture caching and management
 * - Support VTK 9.0+ PBR rendering pipeline
 * - Thread-safe texture loading
 */
class VtkMultiTextureRenderer {
public:
    /**
     * @brief Generic PBR material structure (supports multi-texture)
     */
    struct PBRMaterial {
        std::string name;  // Material name

        // PBR texture paths
        std::string baseColorTexture;  // Albedo/Diffuse
        std::string normalTexture;     // Normal map
        std::string metallicTexture;   // Metallic
        std::string roughnessTexture;  // Roughness
        std::string aoTexture;         // Ambient Occlusion
        std::string emissiveTexture;   // Emissive

        // Advanced PBR texture paths (VTK 9.2+)
        std::string sheenTexture;               // Sheen
        std::string clearcoatTexture;           // Clearcoat
        std::string clearcoatRoughnessTexture;  // Clearcoat roughness
        std::string anisotropyTexture;          // Anisotropy

        // Material parameters (use these values if no texture)
        float baseColor[3] = {1.0f, 1.0f, 1.0f};
        float metallic = 0.0f;
        float roughness = 0.5f;
        float ao = 1.0f;
        float emissive[3] = {0.0f, 0.0f, 0.0f};
        float opacity = 1.0f;

        // Advanced PBR parameters (VTK 9.2+)
        float anisotropy = 0.0f;          // Anisotropic reflection [0,1]
        float anisotropyRotation = 0.0f;  // Anisotropy rotation angle [0,1]
        float clearcoat = 0.0f;           // Clearcoat layer strength [0,1]
        float clearcoatRoughness = 0.0f;  // Clearcoat roughness [0,1]
        float sheen = 0.0f;      // Sheen for fabric-like materials [0,1]
        float sheenTint = 0.0f;  // Sheen color tint [0,1]

        // Phong lighting parameters (for non-PBR rendering)
        float ambient = 0.3f;
        float diffuse = 0.7f;
        float specular = 0.2f;
        float shininess = 30.0f;

        // Check if has PBR textures
        bool hasPBRTextures() const {
            return !baseColorTexture.empty() || !normalTexture.empty() ||
                   !metallicTexture.empty() || !roughnessTexture.empty() ||
                   !aoTexture.empty();
        }

        // Check if has any texture
        bool hasAnyTexture() const {
            return hasPBRTextures() || !emissiveTexture.empty();
        }
    };

public:
    VtkMultiTextureRenderer();
    ~VtkMultiTextureRenderer();

    /**
     * @brief Apply PBR material to actor (Filament style)
     * @param actor VTK actor object
     * @param material PBR material structure
     * @param polydata Polygon data (for texture coordinates)
     * @param renderer VTK renderer (for IBL configuration)
     * @return Returns true on success
     */
    bool ApplyPBRMaterial(vtkSmartPointer<vtkActor> actor,
                          const PBRMaterial& material,
                          vtkSmartPointer<vtkPolyData> polydata,
                          vtkRenderer* renderer = nullptr);

private:
    /**
     * @brief Rendering mode enumeration
     */
    enum class RenderingMode {
        PBR,       // Full PBR rendering (with PBR textures)
        TEXTURED,  // Single texture Phong rendering (only baseColor texture)
        MATERIAL_ONLY  // Pure material Phong rendering (no texture)
    };

    /**
     * @brief Texture quality settings
     */
    enum class TextureQuality {
        LOW,      // Low quality, fast loading
        MEDIUM,   // Medium quality
        HIGH,     // High quality
        ORIGINAL  // Original quality, no compression
    };

    /**
     * @brief Texture filter mode
     */
    enum class FilterMode {
        NEAREST,  // Nearest neighbor
        LINEAR,   // Linear interpolation
        MIPMAP    // Mipmap (best quality)
    };

    /**
     * @brief Rendering configuration options
     */
    struct RenderOptions {
        TextureQuality quality = TextureQuality::HIGH;
        FilterMode filter_mode = FilterMode::MIPMAP;
        bool enable_mipmaps = true;
        bool enable_texture_cache = true;
        bool enable_compression = false;  // Texture compression (save VRAM)
        int max_texture_size = 4096;      // Maximum texture size
        bool interpolate_scalars = true;  // Scalar interpolation
        bool use_phong_shading = true;    // Use Phong shading
    };

    /**
     * @brief Texture material info (traditional single texture)
     */
    struct MaterialInfo {
        std::string name;                        // Material name
        std::string texture_file;                // Texture file path
        float ambient[3] = {1.0f, 1.0f, 1.0f};   // Ambient light
        float diffuse[3] = {1.0f, 1.0f, 1.0f};   // Diffuse
        float specular[3] = {0.0f, 0.0f, 0.0f};  // Specular
        float shininess = 0.0f;                  // Shininess
        float opacity = 1.0f;                    // Opacity
        int illum_model = 2;                     // Illumination model
    };

    /**
     * @brief Detect rendering mode of material
     * @param material PBR material structure
     * @return Rendering mode enumeration
     */
    RenderingMode DetectRenderingMode(const PBRMaterial& material) const;

    /**
     * @brief Apply PBR rendering mode
     * @param actor VTK actor object
     * @param material PBR material structure
     * @param polydata Polygon data
     * @param renderer VTK renderer
     * @param metallic Metallic (clamped)
     * @param roughness Roughness (clamped)
     * @param ao Ambient occlusion (clamped)
     * @param opacity Opacity (clamped)
     * @return Returns true on success
     */
    bool ApplyPBRRendering(vtkSmartPointer<vtkActor> actor,
                           const PBRMaterial& material,
                           vtkSmartPointer<vtkPolyData> polydata,
                           vtkRenderer* renderer,
                           float metallic,
                           float roughness,
                           float ao,
                           float opacity);

    /**
     * @brief Apply single texture Phong rendering mode
     * @param actor VTK actor object
     * @param material PBR material structure
     * @param opacity Opacity (clamped)
     * @return Returns true on success
     */
    bool ApplyTexturedRendering(vtkSmartPointer<vtkActor> actor,
                                const PBRMaterial& material,
                                float opacity);

    /**
     * @brief Apply pure material Phong rendering mode
     * @param actor VTK actor object
     * @param material PBR material structure
     * @param opacity Opacity (clamped)
     * @return Returns true on success
     */
    bool ApplyMaterialOnlyRendering(vtkSmartPointer<vtkActor> actor,
                                    const PBRMaterial& material,
                                    float opacity);

    /**
     * @brief Set common Phong material properties
     * @param property VTK property object
     * @param material PBR material structure
     * @param opacity Opacity (clamped)
     */
    void SetPhongProperties(vtkProperty* property,
                            const PBRMaterial& material,
                            float opacity);

    /**
     * @brief Load texture image
     * @param texture_path Texture file path
     * @param options Rendering options
     * @return vtkTexture pointer, returns nullptr on failure
     */
    vtkSmartPointer<vtkTexture> LoadTexture(const std::string& texture_path,
                                            const RenderOptions& options);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    RenderOptions default_options_;

    // Texture cache: path -> texture
    std::unordered_map<std::string, vtkSmartPointer<vtkTexture>> texture_cache_;

    // Texture memory usage statistics
    std::unordered_map<std::string, size_t> texture_memory_usage_;
};

}  // namespace VtkUtils
