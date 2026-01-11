// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "renders/textured/MultiTextureRenderer.h"

#include <CVLog.h>
#include <CVTools.h>
#include <FileSystem.h>
#include <ecvMaterial.h>
#include <ecvMaterialSet.h>

#include <cmath>
#include <map>
#include <set>
#include <sstream>

#include "renders/base/TextureRendererBase.h"
#include "renders/utils/MaterialConverter.h"

// VTK
#include <vtkActor.h>
#include <vtkBMPReader.h>
#include <vtkDataArray.h>
#include <vtkJPEGReader.h>
#include <vtkLODActor.h>
#include <vtkOpenGLRenderWindow.h>
#include <vtkPNGReader.h>
#include <vtkPNMReader.h>
#include <vtkPointData.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkQImageToImageSource.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkTIFFReader.h>
#include <vtkTexture.h>
#include <vtkTextureUnitManager.h>

#include <sstream>

namespace PclUtils {
namespace renders {

MultiTextureRenderer::MultiTextureRenderer() = default;

bool MultiTextureRenderer::CanHandle(size_t material_count,
                                     bool has_pbr_textures,
                                     bool has_multiple_map_kd) const {
    // Multi-texture renderer handles:
    // 1. Multiple map_Kd textures (VTK PBR limitation) - highest priority
    // 2. Traditional multi-texture scenarios (multiple materials)
    // 3. Single material with traditional textures (not PBR)
    // Note: If has_pbr_textures is true but has_multiple_map_kd is false,
    // PBRRenderer should handle it instead
    return has_multiple_map_kd || (material_count > 1 && !has_pbr_textures);
}

RenderingMode MultiTextureRenderer::GetMode() const {
    return RenderingMode::MULTI_TEXTURE;
}

bool MultiTextureRenderer::Apply(vtkLODActor* actor,
                                 const ccMaterialSet* materials,
                                 vtkPolyData* polydata,
                                 vtkRenderer* renderer) {
    if (!ValidateActor(actor) || !ValidateMaterials(materials)) {
        return false;
    }

    CVLog::PrintDebug(
            "[MultiTextureRenderer::Apply] Applying multi-texture rendering "
            "for %zu materials",
            materials->size());

    // Get render window
    vtkRenderWindow* render_window = nullptr;
    if (renderer) {
        render_window = renderer->GetRenderWindow();
    }
    if (!render_window) {
        CVLog::Error("[MultiTextureRenderer::Apply] Cannot get render window");
        return false;
    }

    // Get texture unit manager
    ::vtkTextureUnitManager* tex_manager = GetTextureUnitManager(render_window);
    if (!tex_manager) {
        CVLog::Error(
                "[MultiTextureRenderer::Apply] Cannot get texture unit "
                "manager");
        return false;
    }

    // Check available texture units
    int texture_units = tex_manager->GetNumberOfTextureUnits();
    if (static_cast<size_t>(texture_units) < materials->size()) {
        CVLog::Warning(
                "[MultiTextureRenderer::Apply] GPU texture units %d < mesh "
                "textures %zu!",
                texture_units, materials->size());
    }

    // Collect all map_Kd textures from all materials
    // A material may have multiple map_Kd textures, so we need to collect them
    // all
    struct TextureInfo {
        ccMaterial::CShared material;
        QString texturePath;
        size_t materialIndex;
        size_t textureIndexInMaterial;
    };
    std::vector<TextureInfo> allMapKdTextures;

    using TexType = ccMaterial::TextureMapType;
    for (size_t mat_idx = 0; mat_idx < materials->size(); ++mat_idx) {
        auto material = materials->at(mat_idx);
        if (!material) continue;

        // Get all DIFFUSE (map_Kd) textures for this material
        std::vector<QString> diffuseTextures =
                material->getTextureFilenames(TexType::DIFFUSE);

        for (size_t tex_idx = 0; tex_idx < diffuseTextures.size(); ++tex_idx) {
            if (!diffuseTextures[tex_idx].isEmpty()) {
                TextureInfo info;
                info.material = material;
                info.texturePath = diffuseTextures[tex_idx];
                info.materialIndex = mat_idx;
                info.textureIndexInMaterial = tex_idx;
                allMapKdTextures.push_back(info);
            }
        }
    }

    // Count map_Kd textures for intensity scaling
    // IMPORTANT: All textures from all materials will be blended together,
    // so we need a unified intensity scaling strategy to maintain balanced
    // brightness and visual quality
    int map_kd_count = static_cast<int>(allMapKdTextures.size());

    // Calculate per-material texture counts (for logging and per-material
    // scaling)
    std::map<size_t, int> materialTextureCounts;
    for (const auto& texInfo : allMapKdTextures) {
        materialTextureCounts[texInfo.materialIndex]++;
    }

    // Global intensity scale: apply to ALL textures to maintain balanced
    // brightness Use a more conservative scaling approach to prevent
    // over-darkening Formula: 1.0 / sqrt(N) for better visual balance with
    // MODULATE blending This provides better results than logarithmic scaling
    // for MODULATE mode
    float global_intensity_scale = 1.0f;
    if (map_kd_count > 1) {
        // Use square root scaling for MODULATE blending
        // For 2 textures: ~0.71, for 4 textures: ~0.5, for 8 textures: ~0.35
        // This prevents over-darkening while maintaining balanced brightness
        global_intensity_scale = 1.0f / map_kd_count;
    }

    if (map_kd_count > 1) {
        CVLog::Print(
                "[MultiTextureRenderer::Apply] Detected %d map_Kd textures "
                "across %zu materials. Applying global intensity scale %.3f "
                "to maintain balanced brightness. All textures will be "
                "rendered "
                "and blended together.",
                map_kd_count, materials->size(), global_intensity_scale);
    } else if (map_kd_count == 0) {
        CVLog::Warning(
                "[MultiTextureRenderer::Apply] No map_Kd textures found in "
                "materials");
        return false;
    }

    // Clear existing textures
    ClearTextures(actor);

    // Apply textures
    size_t last_tex_id = std::min(allMapKdTextures.size(),
                                  static_cast<size_t>(texture_units));
    vtkPolyDataMapper* mapper =
            vtkPolyDataMapper::SafeDownCast(actor->GetMapper());

    // Track which materials have been applied to avoid duplicate material
    // property application
    std::set<size_t> appliedMaterials;

    for (size_t tex_id = 0; tex_id < last_tex_id; ++tex_id) {
        const TextureInfo& texInfo = allMapKdTextures[tex_id];

        // Get texture unit
#if (VTK_MAJOR_VERSION == 8 && VTK_MINOR_VERSION >= 2) || VTK_MAJOR_VERSION > 8
        std::string tex_name =
                CVTools::FromQString(texInfo.material->getName());
        std::stringstream tu_ss;
        tu_ss << tex_name << "_tex" << tex_id;
        std::string tu_str = tu_ss.str();
        const char* tu = tu_str.c_str();
#else
        int tu = vtkProperty::VTK_TEXTURE_UNIT_0 + tex_id;
#endif

        // Load texture from the specific path
        vtkSmartPointer<vtkTexture> texture =
                vtkSmartPointer<vtkTexture>::New();
        QImage qimage = ccMaterial::GetTexture(texInfo.texturePath);
        if (qimage.isNull()) {
            CVLog::Warning(
                    "[MultiTextureRenderer::Apply] Failed to load texture %s "
                    "for material %s, skipping!",
                    CVTools::FromQString(texInfo.texturePath).c_str(),
                    CVTools::FromQString(texInfo.material->getName()).c_str());
            continue;
        }

        // Check if texture has alpha channel (transparency support)
        bool hasAlpha = qimage.hasAlphaChannel();
        if (hasAlpha) {
            // Convert to RGBA format to preserve alpha channel
            if (qimage.format() != QImage::Format_RGBA8888 &&
                qimage.format() != QImage::Format_ARGB32 &&
                qimage.format() != QImage::Format_ARGB32_Premultiplied) {
                qimage = qimage.convertToFormat(QImage::Format_RGBA8888);
            }
            CVLog::PrintDebug(
                    "[MultiTextureRenderer::Apply] Texture %zu has alpha "
                    "channel, "
                    "enabling transparency support",
                    tex_id);
        }

        vtkSmartPointer<vtkQImageToImageSource> qimageToImageSource =
                vtkSmartPointer<vtkQImageToImageSource>::New();
        qimageToImageSource->SetQImage(&qimage);
        qimageToImageSource->Update();
        texture->SetInputConnection(qimageToImageSource->GetOutputPort());

        // Enable alpha blending for textures with transparency
        if (hasAlpha) {
            texture->SetBlendingMode(
                    vtkTexture::VTK_TEXTURE_BLENDING_MODE_INTERPOLATE);
            CVLog::Print(
                    "[MultiTextureRenderer::Apply] Texture %zu: Using "
                    "INTERPOLATE "
                    "blending mode for alpha transparency",
                    tex_id);
        }

        // Set blending mode for unified multi-texture rendering
        // Strategy: All textures from all materials will be blended together
        // 1. First texture: REPLACE (base layer - establishes the foundation)
        // 2. All subsequent textures: MODULATE (multiply blending - prevents
        // over-brightness) MODULATE is better than ADD because:
        // - ADD causes exponential brightness increase with multiple textures
        // - MODULATE maintains color balance and prevents over-saturation
        // - Works well with intensity scaling to achieve balanced results
        // NOTE: If texture has alpha channel, blending mode is already set to
        // INTERPOLATE above
        if (!hasAlpha) {
            if (tex_id == 0) {
                texture->SetBlendingMode(
                        vtkTexture::VTK_TEXTURE_BLENDING_MODE_REPLACE);
                CVLog::PrintDebug(
                        "[MultiTextureRenderer::Apply] Texture %zu (material "
                        "%zu): "
                        "REPLACE mode (base layer)",
                        tex_id, texInfo.materialIndex);
            } else {
                texture->SetBlendingMode(
                        vtkTexture::VTK_TEXTURE_BLENDING_MODE_ADD);
                CVLog::PrintDebug(
                        "[MultiTextureRenderer::Apply] Texture %zu (material "
                        "%zu): "
                        "MODULATE mode (blending layer)",
                        tex_id, texInfo.materialIndex);
            }
        }

        // Map texture coordinates to multi-texture attribute
        // MapDataArrayToMultiTextureAttribute purpose:
        // 1. Maps texture coordinate arrays in polydata (e.g., "TCoords0",
        // "TCoords1") to the specified texture unit (tu)
        // 2. Parameter description:
        //    - tu: Texture unit identifier, can be a string or integer
        //    - coords_name: Name of the texture coordinate array (e.g.,
        //    "TCoords0")
        //    - FIELD_ASSOCIATION_POINTS:
        //    Specifies that data is associated with points, not cells
        // 3. This tells VTK which texture coordinate array to use for which
        //    texture unit during rendering
        // 4. For multi-texture rendering, each texture needs its own texture
        //    coordinate array
        // CRITICAL: Use global texture index (tex_id) instead of material index
        // because ApplyTexturesFromMaterialSet creates separate TCoords arrays
        // for each texture (TCoords0, TCoords1, ...) regardless of material
        if (mapper && polydata) {
            std::stringstream ss;
            // Use global texture index (tex_id) to find corresponding TCoords
            // array This matches the naming convention in
            // ApplyTexturesFromMaterialSet
            if (last_tex_id == 1) {
                ss << "TCoords";
            } else {
                ss << "TCoords" << tex_id;
            }
            std::string coords_name = ss.str();

            // Check if the texture coordinates array exists
            vtkDataArray* tcoords_array =
                    polydata->GetPointData()->GetArray(coords_name.c_str());
            if (tcoords_array) {
                mapper->MapDataArrayToMultiTextureAttribute(
                        tu, coords_name.c_str(),
                        vtkDataObject::FIELD_ASSOCIATION_POINTS);
                CVLog::PrintDebug(
                        "[MultiTextureRenderer::Apply] Mapped texture %zu to "
                        "texture coordinates array '%s'",
                        tex_id, coords_name.c_str());
            } else {
                CVLog::Warning(
                        "[MultiTextureRenderer::Apply] Texture coordinates "
                        "array "
                        "'%s' not found for texture %zu (material %zu, texture "
                        "index %zu), texture may not display correctly",
                        coords_name.c_str(), tex_id, texInfo.materialIndex,
                        texInfo.textureIndexInMaterial);
            }
        }

        // Set texture
        actor->GetProperty()->SetTexture(tu, texture);

        // Apply material properties only once per material (not per texture)
        // Use unified global intensity scaling to maintain balanced brightness
        // across all materials and textures
        if (appliedMaterials.find(texInfo.materialIndex) ==
            appliedMaterials.end()) {
            // Apply global intensity scale to all materials uniformly
            // This ensures balanced brightness when textures from different
            // materials are blended together
            ApplyMaterial(texInfo.material, actor, global_intensity_scale);
            appliedMaterials.insert(texInfo.materialIndex);
        }
    }

    // Set model transparency based on material opacity
    // This allows the entire textured model to be transparent, regardless of
    // whether textures themselves have alpha channels
    // The opacity can be set externally via actor->GetProperty()->SetOpacity()
    // and will be respected here
    float modelOpacity = 1.0f;
    bool hasMaterialOpacity = false;

    // Check material opacity values
    for (size_t i = 0; i < materials->size(); ++i) {
        auto mat = materials->at(i);
        if (mat) {
            const ecvColor::Rgbaf& diffuse = mat->getDiffuseFront();
            const ecvColor::Rgbaf& ambient = mat->getAmbient();
            float opacity = std::max(diffuse.a, ambient.a);
            if (opacity < 1.0f) {
                hasMaterialOpacity = true;
                modelOpacity = std::min(modelOpacity, opacity);
            }
        }
    }

    // Get current opacity from actor property (may have been set externally)
    float currentOpacity = actor->GetProperty()->GetOpacity();

    // If material has opacity, use it; otherwise preserve external setting
    // If external setting is less than 1.0, it means transparency was requested
    if (hasMaterialOpacity) {
        // Material defines opacity, use it
        actor->GetProperty()->SetOpacity(modelOpacity);
        CVLog::Print(
                "[MultiTextureRenderer::Apply] Set model opacity from "
                "material: "
                "%.3f",
                modelOpacity);
    } else if (currentOpacity < 1.0f) {
        // External opacity setting exists, preserve it
        actor->GetProperty()->SetOpacity(currentOpacity);
        CVLog::Print(
                "[MultiTextureRenderer::Apply] Preserving external opacity "
                "setting: %.3f",
                currentOpacity);
    } else {
        // Default: fully opaque
        actor->GetProperty()->SetOpacity(1.0f);
    }

    // Enable transparency rendering support if opacity < 1.0
    float finalOpacity = actor->GetProperty()->GetOpacity();
    if (finalOpacity < 1.0f) {
        // Configure renderer for transparency rendering
        if (renderer) {
            vtkRenderWindow* renderWindow = renderer->GetRenderWindow();
            if (renderWindow) {
                // Enable alpha bit planes for transparency rendering
                renderWindow->SetAlphaBitPlanes(1);

                // Enable depth sorting for proper transparency rendering
                // VTK will automatically sort transparent objects by depth
                renderer->SetUseDepthPeeling(1);
                renderer->SetMaximumNumberOfPeels(4);  // Reasonable default
                renderer->SetOcclusionRatio(0.0);  // Full transparency support

                CVLog::Print(
                        "[MultiTextureRenderer::Apply] Enabled transparency "
                        "rendering support: opacity=%.3f, depth peeling "
                        "enabled",
                        finalOpacity);
            }
        }
    }

    actor->Modified();
    return true;
}

bool MultiTextureRenderer::Update(vtkActor* actor,
                                  const ccMaterialSet* materials,
                                  vtkPolyData* polydata,
                                  vtkRenderer* renderer) {
    if (!ValidateActor(actor) || !ValidateMaterials(materials)) {
        return false;
    }

    vtkLODActor* lod_actor = vtkLODActor::SafeDownCast(actor);
    if (!lod_actor) {
        CVLog::Warning(
                "[MultiTextureRenderer::Update] Actor is not a vtkLODActor");
        return false;
    }

    return Apply(lod_actor, materials, polydata, renderer);
}

std::string MultiTextureRenderer::GetName() const {
    return "MultiTextureRenderer";
}

int MultiTextureRenderer::LoadTexture(ccMaterial::CShared material,
                                      ::vtkTexture* vtk_tex) const {
    if (!material || !vtk_tex) {
        return -1;
    }

    // For traditional multi-texture rendering, prioritize DIFFUSE texture
    // If not available, try other texture types
    using TexType = ccMaterial::TextureMapType;
    QString tex_file = material->getTextureFilename(TexType::DIFFUSE);
    if (tex_file.isEmpty()) {
        // Fallback to legacy texture filename (for backward compatibility)
        tex_file = material->getTextureFilename();
    }
    QImage qimage = ccMaterial::GetTexture(tex_file);
    if (!qimage.isNull()) {
        vtkSmartPointer<vtkQImageToImageSource> qimageToImageSource =
                vtkSmartPointer<vtkQImageToImageSource>::New();
        qimageToImageSource->SetQImage(&qimage);
        qimageToImageSource->Update();
        vtk_tex->SetInputConnection(qimageToImageSource->GetOutputPort());
        return 1;
    }

    if (tex_file.isEmpty()) {
        return -1;
    }

    std::string full_path = CVTools::FromQString(tex_file);
    if (!cloudViewer::utility::filesystem::FileExists(full_path)) {
        std::string parent_dir =
                cloudViewer::utility::filesystem::GetFileParentDirectory(
                        full_path);
        std::string upper_filename = cloudViewer::utility::ToUpper(full_path);
        std::string real_name;

        try {
            if (!cloudViewer::utility::filesystem::DirectoryExists(
                        parent_dir)) {
                CVLog::Warning(
                        "[MultiTextureRenderer::LoadTexture] Parent directory "
                        "'%s' doesn't exist!",
                        parent_dir.c_str());
                return -1;
            }

            if (!cloudViewer::utility::filesystem::IsDirectory(parent_dir)) {
                CVLog::Warning(
                        "[MultiTextureRenderer::LoadTexture] Parent '%s' is "
                        "not "
                        "a directory !",
                        parent_dir.c_str());
                return -1;
            }

            std::vector<std::string> paths_vector;
            cloudViewer::utility::filesystem::ListFilesInDirectory(
                    parent_dir, paths_vector);

            for (const auto& path : paths_vector) {
                if (cloudViewer::utility::filesystem::IsFile(path)) {
                    if (cloudViewer::utility::ToUpper(path) == upper_filename) {
                        real_name = path;
                        break;
                    }
                }
            }

            if (real_name.empty()) {
                CVLog::Warning(
                        "[MultiTextureRenderer::LoadTexture] Can not find "
                        "texture file %s!",
                        full_path.c_str());
                return -1;
            }
        } catch (const std::exception& ex) {
            CVLog::Warning(
                    "[MultiTextureRenderer::LoadTexture] Error %s when "
                    "looking for file %s!",
                    ex.what(), full_path.c_str());
            return -1;
        }

        full_path = real_name;
    }

    std::string extension =
            cloudViewer::utility::filesystem::GetFileExtensionInLowerCase(
                    full_path);

    if ((extension == "jpg") || (extension == "jpeg")) {
        vtkSmartPointer<vtkJPEGReader> jpeg_reader =
                vtkSmartPointer<vtkJPEGReader>::New();
        jpeg_reader->SetFileName(full_path.c_str());
        jpeg_reader->Update();
        vtk_tex->SetInputConnection(jpeg_reader->GetOutputPort());
    } else if (extension == "bmp") {
        vtkSmartPointer<vtkBMPReader> bmp_reader =
                vtkSmartPointer<vtkBMPReader>::New();
        bmp_reader->SetFileName(full_path.c_str());
        bmp_reader->Update();
        vtk_tex->SetInputConnection(bmp_reader->GetOutputPort());
    } else if (extension == "pnm") {
        vtkSmartPointer<vtkPNMReader> pnm_reader =
                vtkSmartPointer<vtkPNMReader>::New();
        pnm_reader->SetFileName(full_path.c_str());
        pnm_reader->Update();
        vtk_tex->SetInputConnection(pnm_reader->GetOutputPort());
    } else if (extension == "png") {
        vtkSmartPointer<vtkPNGReader> png_reader =
                vtkSmartPointer<vtkPNGReader>::New();
        png_reader->SetFileName(full_path.c_str());
        png_reader->Update();
        vtk_tex->SetInputConnection(png_reader->GetOutputPort());
    } else if ((extension == "tiff") || (extension == "tif")) {
        vtkSmartPointer<vtkTIFFReader> tiff_reader =
                vtkSmartPointer<vtkTIFFReader>::New();
        tiff_reader->SetFileName(full_path.c_str());
        tiff_reader->Update();
        vtk_tex->SetInputConnection(tiff_reader->GetOutputPort());
    } else {
        CVLog::Warning(
                "[MultiTextureRenderer::LoadTexture] Unhandled image %s "
                "(extension: '%s') for material %s!",
                full_path.c_str(), extension.c_str(),
                CVTools::FromQString(material->getName()).c_str());
        return -1;
    }

    return 1;
}

bool MultiTextureRenderer::ApplyMaterial(ccMaterial::CShared material,
                                         vtkActor* actor,
                                         float intensity_scale) const {
    if (!actor || !material) return false;

    const ecvColor::Rgbaf& ambientColor = material->getAmbient();
    const ecvColor::Rgbaf& diffuseColor = material->getDiffuseFront();
    const ecvColor::Rgbaf& specularColor = material->getSpecular();
    actor->GetProperty()->SetDiffuseColor(diffuseColor.r, diffuseColor.g,
                                          diffuseColor.b);
    actor->GetProperty()->SetSpecularColor(specularColor.r, specularColor.g,
                                           specularColor.b);
    actor->GetProperty()->SetAmbientColor(ambientColor.r, ambientColor.g,
                                          ambientColor.b);
    // Opacity: prefer from diffuse alpha, then ambient alpha
    // Note: This sets the base opacity from material, but external opacity
    // settings (via actor->GetProperty()->SetOpacity()) will override this
    float materialOpacity = std::max(0.0f, std::min(1.0f, diffuseColor.a));
    if (materialOpacity <= 0.0f) {
        materialOpacity = std::max(0.0f, std::min(1.0f, ambientColor.a));
    }

    // Only set opacity if it's different from current, or if current is 1.0
    // This preserves external opacity settings
    float currentOpacity = actor->GetProperty()->GetOpacity();
    if (currentOpacity >= 1.0f || materialOpacity < currentOpacity) {
        actor->GetProperty()->SetOpacity(materialOpacity);
    }
    actor->GetProperty()->SetInterpolationToPhong();

    // Apply illumination mode with proper intensity scaling
    // MTL illumination modes:
    // 0 = Color on and Ambient off (no lighting, use diffuse color)
    // 1 = Color on and Ambient on (diffuse lighting only, no specular)
    // 2 = Highlight on (Phong shading with specular)
    switch (material->getIllum()) {
        case 1: {
            // Diffuse only: disable specular, enable diffuse and ambient
            actor->GetProperty()->SetDiffuse(1.0f * intensity_scale);
            actor->GetProperty()->SetSpecular(0.0f);
            actor->GetProperty()->SetAmbient(1.0f * intensity_scale);
            break;
        }
        case 2: {
            // Phong with specular: enable all components
            actor->GetProperty()->SetDiffuse(1.0f * intensity_scale);
            actor->GetProperty()->SetSpecular(1.0f * intensity_scale);
            actor->GetProperty()->SetAmbient(1.0f * intensity_scale);
            // Use material's shininess for specular power
            float shininess = std::max(
                    0.0f, std::min(128.0f, material->getShininessFront()));
            actor->GetProperty()->SetSpecularPower(shininess > 0.0f ? shininess
                                                                    : 4.0f);
            break;
        }
        default:
        case 0: {
            // No lighting: disable lighting, use diffuse color directly
            actor->GetProperty()->SetLighting(false);
            actor->GetProperty()->SetDiffuse(0.0f);
            actor->GetProperty()->SetSpecular(0.0f);
            actor->GetProperty()->SetAmbient(1.0f * intensity_scale);
            actor->GetProperty()->SetColor(diffuseColor.r, diffuseColor.g,
                                           diffuseColor.b);
            break;
        }
    }

    return true;
}

::vtkTextureUnitManager* MultiTextureRenderer::GetTextureUnitManager(
        ::vtkRenderWindow* render_window) const {
    if (!render_window) {
        return nullptr;
    }

    vtkOpenGLRenderWindow* gl_window =
            vtkOpenGLRenderWindow::SafeDownCast(render_window);
    if (!gl_window) {
        return nullptr;
    }

    return gl_window->GetTextureUnitManager();
}

}  // namespace renders
}  // namespace PclUtils
