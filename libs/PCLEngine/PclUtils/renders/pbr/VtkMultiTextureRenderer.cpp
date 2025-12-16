// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "renders/pbr/VtkMultiTextureRenderer.h"

#include <algorithm>
#include <fstream>
#include <mutex>
#include <sstream>

// VTK
#include <vtkActor.h>
#include <vtkCellArray.h>
#include <vtkCellData.h>
#include <vtkFloatArray.h>
#include <vtkImageData.h>
#include <vtkImageResize.h>
#include <vtkLODActor.h>
#include <vtkLight.h>
#include <vtkLightCollection.h>
#include <vtkOpenGLRenderWindow.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkRenderer.h>
#include <vtkTexture.h>
#include <vtkTextureUnitManager.h>

// Qt
#include <QFileInfo>
#include <QImage>
#include <QImageReader>
#include <QString>

// CloudViewer
#include <CVLog.h>
#include <FileSystem.h>
#include <ecvMaterial.h>

namespace VtkUtils {

// Pimpl implementation
struct VtkMultiTextureRenderer::Impl {
    std::mutex cache_mutex;  // Thread-safe cache access

    // Helper function: Create vtkImageData from QImage
    vtkSmartPointer<vtkImageData> QImageToVtkImage(const QImage& qimage) {
        if (qimage.isNull()) {
            return nullptr;
        }

        // Convert to RGBA format
        QImage rgba_image = qimage.convertToFormat(QImage::Format_RGBA8888);

        vtkSmartPointer<vtkImageData> image_data =
                vtkSmartPointer<vtkImageData>::New();
        image_data->SetDimensions(rgba_image.width(), rgba_image.height(), 1);
        image_data->AllocateScalars(VTK_UNSIGNED_CHAR, 4);

        // Copy pixel data
        unsigned char* vtk_ptr =
                static_cast<unsigned char*>(image_data->GetScalarPointer());
        const uchar* qt_ptr = rgba_image.constBits();

        int width = rgba_image.width();
        int height = rgba_image.height();

        // VTK's Y-axis is bottom-up, Qt is top-down, need to flip
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int vtk_idx = ((height - 1 - y) * width + x) * 4;
                int qt_idx = (y * width + x) * 4;

                vtk_ptr[vtk_idx + 0] = qt_ptr[qt_idx + 0];  // R
                vtk_ptr[vtk_idx + 1] = qt_ptr[qt_idx + 1];  // G
                vtk_ptr[vtk_idx + 2] = qt_ptr[qt_idx + 2];  // B
                vtk_ptr[vtk_idx + 3] = qt_ptr[qt_idx + 3];  // A
            }
        }

        return image_data;
    }
};

VtkMultiTextureRenderer::VtkMultiTextureRenderer()
    : impl_(std::make_unique<Impl>()) {
    // Set default options
    default_options_.quality = TextureQuality::HIGH;
    default_options_.filter_mode = FilterMode::MIPMAP;
    default_options_.enable_mipmaps = true;
    default_options_.enable_texture_cache = true;
}

VtkMultiTextureRenderer::~VtkMultiTextureRenderer() {
    std::lock_guard<std::mutex> lock(impl_->cache_mutex);
    texture_cache_.clear();
    texture_memory_usage_.clear();
}

// Removed unused CreateMultiTextureActor - not used in codebase

// Removed unused UpdateTextures - not used in codebase

vtkSmartPointer<vtkTexture> VtkMultiTextureRenderer::LoadTexture(
        const std::string& texture_path, const RenderOptions& options) {
    // Check cache first
    {
        std::lock_guard<std::mutex> lock(impl_->cache_mutex);
        auto it = texture_cache_.find(texture_path);
        if (it != texture_cache_.end()) {
            return it->second;
        }
    }

    // Load image using Qt (via ccMaterial for consistency)
    QImage qimage = ccMaterial::GetTexture(texture_path.c_str());
    if (qimage.isNull()) {
        CVLog::Warning(
                "[VtkMultiTextureRenderer::LoadTexture] Failed to load "
                "texture: %s",
                texture_path.c_str());
        return nullptr;
    }

    // Convert to vtkImageData
    vtkSmartPointer<vtkImageData> image_data = impl_->QImageToVtkImage(qimage);
    if (!image_data) {
        CVLog::Warning(
                "[VtkMultiTextureRenderer::LoadTexture] Failed to convert "
                "QImage to vtkImageData");
        return nullptr;
    }

    // Create texture
    vtkSmartPointer<vtkTexture> texture = vtkSmartPointer<vtkTexture>::New();
    texture->SetInputData(image_data);
    texture->SetInterpolate(1);  // Linear interpolation
    texture->MipmapOn();         // Enable mipmaps
    texture->SetRepeat(1);       // Allow texture repeat
    texture->SetEdgeClamp(0);

    // Cache texture
    {
        std::lock_guard<std::mutex> lock(impl_->cache_mutex);
        texture_cache_[texture_path] = texture;

        // Record memory usage
        size_t memory_size =
                image_data->GetActualMemorySize() * 1024;  // KB to bytes
        texture_memory_usage_[texture_path] = memory_size;
    }

    CVLog::PrintDebug(
            "[VtkMultiTextureRenderer::LoadTexture] Loaded and cached texture: "
            "%s (%dx%d)",
            texture_path.c_str(), qimage.width(), qimage.height());

    return texture;
}

// ============================================================================
// PBR Material Rendering Implementation (Filament-style)
// ============================================================================
VtkMultiTextureRenderer::RenderingMode
VtkMultiTextureRenderer::DetectRenderingMode(
        const PBRMaterial& material) const {
    // If there are multiple map_Kd textures, VTK PBR doesn't support it,
    // so we must use traditional multi-texture rendering
    if (material.hasMultipleMapKd) {
        CVLog::Print(
                "[VtkMultiTextureRenderer::DetectRenderingMode] Multiple "
                "map_Kd detected, using traditional multi-texture rendering "
                "instead of PBR");
        return RenderingMode::TEXTURED;
    }

    if (material.hasPBRTextures()) {
        return RenderingMode::PBR;
    } else if (material.hasAnyTexture()) {
        return RenderingMode::TEXTURED;
    } else {
        return RenderingMode::MATERIAL_ONLY;
    }
}

void VtkMultiTextureRenderer::SetProperties(vtkProperty* property,
                                            const PBRMaterial& material,
                                            float opacity) {
    property->SetInterpolationToPBR();
    property->SetColor(material.baseColor[0], material.baseColor[1],
                       material.baseColor[2]);
    // Set RGB colors for ambient, diffuse, specular
    property->SetAmbientColor(material.ambientColor[0],
                              material.ambientColor[1],
                              material.ambientColor[2]);
    property->SetDiffuseColor(material.diffuseColor[0],
                              material.diffuseColor[1],
                              material.diffuseColor[2]);
    property->SetSpecularColor(material.specularColor[0],
                               material.specularColor[1],
                               material.specularColor[2]);
    // Set intensity coefficients to 1.0 to preserve RGB color brightness
    property->SetAmbient(1.0f);
    property->SetDiffuse(1.0f);
    property->SetSpecular(1.0f);
    property->SetSpecularPower(material.shininess);
    property->SetOpacity(opacity);

    property->BackfaceCullingOff();
    property->FrontfaceCullingOff();

    CVLog::PrintDebug(
            "[VtkMultiTextureRenderer::SetProperties] Setting properties: "
            "specularPower=%.2f, opacity=%.2f",
            material.shininess, opacity);
}

// ============================================================================
// Public method implementation - Unified material application interface
// ============================================================================

bool VtkMultiTextureRenderer::ApplyPBRMaterial(
        vtkSmartPointer<vtkActor> actor,
        const PBRMaterial& material,
        vtkSmartPointer<vtkPolyData> polydata,
        vtkRenderer* renderer) {
    // ========================================================================
    // Input validation
    // ========================================================================
    if (!actor) {
        CVLog::Error(
                "[VtkMultiTextureRenderer::ApplyPBRMaterial] Actor is null");
        return false;
    }

    vtkProperty* property = actor->GetProperty();
    if (!property) {
        CVLog::Error(
                "[VtkMultiTextureRenderer::ApplyPBRMaterial] Property is null");
        return false;
    }

    // Validate and clamp material parameter ranges
    float opacity = std::max(0.0f, std::min(1.0f, material.opacity));

    // Enable transparency rendering support ONLY if opacity < 1.0
    // This avoids unnecessary performance overhead for opaque materials
    if (opacity < 1.0f) {
        actor->ForceTranslucentOn();
        actor->ForceOpaqueOff();

        if (renderer) {
            vtkRenderWindow* renderWindow = renderer->GetRenderWindow();
            if (renderWindow) {
                // Enable alpha bit planes for transparency rendering
                renderWindow->SetAlphaBitPlanes(1);
            }

            // Enable depth peeling for proper transparency rendering
            renderer->UseDepthPeelingOn();
            renderer->SetMaximumNumberOfPeels(4);  // Reasonable default
            renderer->SetOcclusionRatio(0.0);      // Full transparency support

            CVLog::Print(
                    "[VtkMultiTextureRenderer::ApplyPBRMaterial] Enabled "
                    "transparency rendering support: opacity=%.3f, depth "
                    "peeling enabled",
                    opacity);
        }
    } else {
        // Fully opaque material - ensure transparent rendering is disabled
        actor->ForceTranslucentOff();
        actor->ForceOpaqueOn();

        CVLog::PrintDebug(
                "[VtkMultiTextureRenderer::ApplyPBRMaterial] Material is fully "
                "opaque (opacity=%.3f), transparent rendering disabled",
                opacity);
    }
    float metallic = std::max(0.0f, std::min(1.0f, material.metallic));
    float roughness = std::max(0.0f, std::min(1.0f, material.roughness));
    float ao = std::max(
            0.0f,
            std::min(2.0f,
                     material.ao));  // AO can exceed 1.0 for enhanced effect

    if (material.opacity != opacity) {
        CVLog::Warning(
                "[VtkMultiTextureRenderer::ApplyPBRMaterial] Invalid opacity "
                "%.2f, clamped to %.2f",
                material.opacity, opacity);
    }

    if (material.metallic != metallic) {
        CVLog::Warning(
                "[VtkMultiTextureRenderer::ApplyPBRMaterial] Invalid metallic "
                "%.2f, clamped to %.2f",
                material.metallic, metallic);
    }

    if (material.roughness != roughness) {
        CVLog::Warning(
                "[VtkMultiTextureRenderer::ApplyPBRMaterial] Invalid roughness "
                "%.2f, clamped to %.2f",
                material.roughness, roughness);
    }

    if (material.ao != ao) {
        CVLog::Warning(
                "[VtkMultiTextureRenderer::ApplyPBRMaterial] Invalid AO %.2f, "
                "clamped to %.2f",
                material.ao, ao);
    }

    // ========================================================================
    // Detect rendering mode and dispatch to corresponding handler
    // ========================================================================
    RenderingMode mode = DetectRenderingMode(material);

    CVLog::Print(
            "[VtkMultiTextureRenderer::ApplyPBRMaterial] Detected rendering "
            "mode: %s",
            mode == RenderingMode::PBR        ? "PBR"
            : mode == RenderingMode::TEXTURED ? "TEXTURED"
                                              : "MATERIAL_ONLY");

    bool result = false;
    switch (mode) {
        case RenderingMode::PBR:
            result = ApplyPBRRendering(actor, material, polydata, renderer,
                                       metallic, roughness, ao, opacity);
            break;
        case RenderingMode::TEXTURED:
            result = ApplyTexturedRendering(actor, material, opacity);
            break;
        case RenderingMode::MATERIAL_ONLY:
            result = ApplyMaterialOnlyRendering(actor, material, opacity);
            break;
    }

    if (result) {
        CVLog::PrintDebug(
                "[VtkMultiTextureRenderer::ApplyPBRMaterial] Material "
                "application complete");
    } else {
        CVLog::Error(
                "[VtkMultiTextureRenderer::ApplyPBRMaterial] Material "
                "application failed");
    }

    return result;
}

bool VtkMultiTextureRenderer::ApplyPBRRendering(
        vtkSmartPointer<vtkActor> actor,
        const PBRMaterial& material,
        vtkSmartPointer<vtkPolyData> polydata,
        vtkRenderer* renderer,
        float metallic,
        float roughness,
        float ao,
        float opacity) {
    // ========================================================================
    // Mode 1: PBR rendering (with PBR textures)
    // ========================================================================
#if VTK_MAJOR_VERSION >= 9
    CVLog::PrintDebug(
            "[VtkMultiTextureRenderer::ApplyPBRRendering] Using VTK 9+ PBR "
            "rendering");

    vtkProperty* property = actor->GetProperty();
    if (!property) {
        CVLog::Error(
                "[VtkMultiTextureRenderer::ApplyPBRRendering] Property is "
                "null");
        return false;
    }

    // Clear all traditional texture settings to avoid conflicts
    // with PBR API
    property->RemoveAllTextures();

    // Ensure polydata and texture coordinates exist
    if (!polydata) {
        CVLog::Error(
                "[VtkMultiTextureRenderer::ApplyPBRRendering] PolyData is null "
                "for PBR rendering");
        return false;
    }

    // Check and set texture coordinates
    vtkDataArray* tcoords = polydata->GetPointData()->GetTCoords();
    if (!tcoords) {
        // Try to find TCoords0
        tcoords = polydata->GetPointData()->GetArray("TCoords0");
        if (tcoords) {
            polydata->GetPointData()->SetTCoords(tcoords);
            CVLog::PrintDebug(
                    "[VtkMultiTextureRenderer::ApplyPBRRendering] Set TCoords0 "
                    "as active texture coordinates");
        } else {
            CVLog::Warning(
                    "[VtkMultiTextureRenderer::ApplyPBRRendering] No texture "
                    "coordinates found, PBR textures may not display "
                    "correctly");
        }
    }

    if (tcoords) {
        CVLog::PrintDebug(
                "[VtkMultiTextureRenderer::ApplyPBRRendering] Texture "
                "coordinates: %lld points",
                tcoords->GetNumberOfTuples());
    }

    // 1. Enable PBR interpolation mode
    property->SetInterpolationToPBR();

    // 2. Set base material properties (using clamped values)
    property->SetColor(material.baseColor[0], material.baseColor[1],
                       material.baseColor[2]);
    property->SetOpacity(opacity);

    // Check if we will use ORM texture
    // If ORM texture is present, scalar values (metallic, roughness, ao) will
    // be ignored/overridden by texture values. Setting them may cause
    // multiplication/overlay effects leading to darker rendering.
    bool has_orm_texture = (!material.aoTexture.empty() ||
                            !material.roughnessTexture.empty() ||
                            !material.metallicTexture.empty());

    // Only set scalar values if NO ORM texture will be used
    // When ORM texture is present, VTK uses texture values directly
    if (!has_orm_texture) {
        property->SetMetallic(metallic);
        property->SetRoughness(roughness);
        property->SetOcclusionStrength(ao);
        CVLog::PrintDebug(
                "[VtkMultiTextureRenderer::ApplyPBRRendering] Using scalar "
                "values: metallic=%.2f, roughness=%.2f, ao=%.2f",
                metallic, roughness, ao);
    } else {
        CVLog::PrintDebug(
                "[VtkMultiTextureRenderer::ApplyPBRRendering] ORM texture will "
                "be used, skipping scalar metallic/roughness/ao settings to "
                "avoid conflicts");
    }

#if VTK_VERSION_NUMBER >= VTK_VERSION_CHECK(9, 2, 0)
    // Advanced PBR parameters (VTK 9.2+)
    property->SetAnisotropy(std::max(
            0.0, std::min(1.0, static_cast<double>(material.anisotropy))));
    property->SetAnisotropyRotation(std::max(
            0.0,
            std::min(1.0, static_cast<double>(material.anisotropyRotation))));
    property->SetCoatStrength(std::max(
            0.0, std::min(1.0, static_cast<double>(material.clearcoat))));
    property->SetCoatRoughness(std::max(
            0.0,
            std::min(1.0, static_cast<double>(material.clearcoatRoughness))));
    // Note: VTK uses "Coat" terminology for clearcoat

    CVLog::PrintDebug(
            "[VtkMultiTextureRenderer::ApplyPBRRendering] Advanced PBR: "
            "anisotropy=%.2f, clearcoat=%.2f, clearcoatRoughness=%.2f",
            material.anisotropy, material.clearcoat,
            material.clearcoatRoughness);
#endif

    CVLog::PrintDebug(
            "[VtkMultiTextureRenderer::ApplyPBRRendering] Base properties: "
            "color=(%.2f,%.2f,%.2f), metallic=%.2f, roughness=%.2f, ao=%.2f",
            material.baseColor[0], material.baseColor[1], material.baseColor[2],
            metallic, roughness, ao);

    // 3. Load BaseColor texture (Albedo)
    if (!material.baseColorTexture.empty()) {
        // Check if file exists
        if (!cloudViewer::utility::filesystem::FileExists(
                    material.baseColorTexture)) {
            CVLog::Warning(
                    "[VtkMultiTextureRenderer::ApplyPBRMaterial] BaseColor "
                    "texture file not found: %s",
                    material.baseColorTexture.c_str());
        } else {
            QImage albedo_img =
                    ccMaterial::GetTexture(material.baseColorTexture.c_str());
            if (!albedo_img.isNull()) {
                // Validate image dimensions
                if (albedo_img.width() <= 0 || albedo_img.height() <= 0) {
                    CVLog::Error(
                            "[VtkMultiTextureRenderer::ApplyPBRMaterial] "
                            "Invalid BaseColor texture dimensions: %dx%d",
                            albedo_img.width(), albedo_img.height());
                } else {
                    // Convert to RGB888 format (sRGB)
                    if (albedo_img.format() != QImage::Format_RGB888) {
                        albedo_img = albedo_img.convertToFormat(
                                QImage::Format_RGB888);
                    }

                    vtkSmartPointer<vtkImageData> image_data =
                            impl_->QImageToVtkImage(albedo_img);
                    if (image_data) {
                        vtkSmartPointer<vtkTexture> tex =
                                vtkSmartPointer<vtkTexture>::New();
                        tex->SetInputData(image_data);
                        tex->SetInterpolate(1);
                        tex->MipmapOn();
                        tex->UseSRGBColorSpaceOn();  // Filament uses sRGB color
                                                     // space

                        property->SetBaseColorTexture(tex);
                        CVLog::PrintDebug(
                                "[VtkMultiTextureRenderer::ApplyPBRMaterial] "
                                "Applied BaseColor texture: %dx%d",
                                albedo_img.width(), albedo_img.height());
                    } else {
                        CVLog::Warning(
                                "[VtkMultiTextureRenderer::ApplyPBRMaterial] "
                                "Failed to convert BaseColor image to VTK "
                                "format");
                    }
                }
            } else {
                CVLog::Warning(
                        "[VtkMultiTextureRenderer::ApplyPBRMaterial] Failed to "
                        "load BaseColor texture: %s",
                        material.baseColorTexture.c_str());
            }
        }
    } else {
        CVLog::Print(
                "[VtkMultiTextureRenderer::ApplyPBRMaterial] No BaseColor "
                "texture specified, using base color: (%.2f,%.2f,%.2f)",
                material.baseColor[0], material.baseColor[1],
                material.baseColor[2]);
    }

    // 4. Create and apply ORM texture (Occlusion-Roughness-Metallic)
    if (!material.aoTexture.empty() || !material.roughnessTexture.empty() ||
        !material.metallicTexture.empty()) {
        CVLog::PrintDebug(
                "[VtkMultiTextureRenderer::ApplyPBRMaterial] Creating ORM "
                "texture (O=%d, R=%d, M=%d)",
                !material.aoTexture.empty(), !material.roughnessTexture.empty(),
                !material.metallicTexture.empty());

        // Load each channel
        QImage ao_img, roughness_img, metallic_img;
        int width = 1024, height = 1024;

        if (!material.aoTexture.empty()) {
            ao_img = ccMaterial::GetTexture(material.aoTexture.c_str());
            if (!ao_img.isNull()) {
                width = ao_img.width();
                height = ao_img.height();
            }
        }
        if (!material.roughnessTexture.empty()) {
            roughness_img =
                    ccMaterial::GetTexture(material.roughnessTexture.c_str());
            if (!roughness_img.isNull() && roughness_img.width() > width) {
                width = roughness_img.width();
                height = roughness_img.height();
            }
        }
        if (!material.metallicTexture.empty()) {
            metallic_img =
                    ccMaterial::GetTexture(material.metallicTexture.c_str());
            if (!metallic_img.isNull() && metallic_img.width() > width) {
                width = metallic_img.width();
                height = metallic_img.height();
            }
        }

        // Validate size reasonableness
        if (width <= 0 || height <= 0 || width > 8192 || height > 8192) {
            CVLog::Warning(
                    "[VtkMultiTextureRenderer::ApplyPBRMaterial] Invalid ORM "
                    "texture dimensions: %dx%d, using default 1024x1024",
                    width, height);
            width = 1024;
            height = 1024;
        }

        // Create ORM texture (R=Occlusion, G=Roughness, B=Metallic)
        QImage orm(width, height, QImage::Format_RGB888);
        if (orm.isNull()) {
            CVLog::Error(
                    "[VtkMultiTextureRenderer::ApplyPBRMaterial] Failed to "
                    "create ORM texture");
        } else {
            // Preprocess each channel to grayscale
            QImage o_gray =
                    (!material.aoTexture.empty() && !ao_img.isNull())
                            ? ao_img.scaled(width, height,
                                            Qt::IgnoreAspectRatio,
                                            Qt::SmoothTransformation)
                                      .convertToFormat(
                                              QImage::Format_Grayscale8)
                            : QImage(width, height, QImage::Format_Grayscale8);
            if (material.aoTexture.empty() || ao_img.isNull())
                o_gray.fill(Qt::white);

            QImage r_gray =
                    (!material.roughnessTexture.empty() &&
                     !roughness_img.isNull())
                            ? roughness_img
                                      .scaled(width, height,
                                              Qt::IgnoreAspectRatio,
                                              Qt::SmoothTransformation)
                                      .convertToFormat(
                                              QImage::Format_Grayscale8)
                            : QImage(width, height, QImage::Format_Grayscale8);
            if (material.roughnessTexture.empty() || roughness_img.isNull())
                r_gray.fill(qRgb(128, 128, 128));

            QImage m_gray =
                    (!material.metallicTexture.empty() &&
                     !metallic_img.isNull())
                            ? metallic_img
                                      .scaled(width, height,
                                              Qt::IgnoreAspectRatio,
                                              Qt::SmoothTransformation)
                                      .convertToFormat(
                                              QImage::Format_Grayscale8)
                            : QImage(width, height, QImage::Format_Grayscale8);
            if (material.metallicTexture.empty() || metallic_img.isNull())
                m_gray.fill(Qt::black);

            // Merge into ORM texture
            for (int y = 0; y < height; ++y) {
                uchar* o_line = o_gray.scanLine(y);
                uchar* r_line = r_gray.scanLine(y);
                uchar* m_line = m_gray.scanLine(y);
                uchar* orm_line = orm.scanLine(y);

                for (int x = 0; x < width; ++x) {
                    orm_line[x * 3 + 0] = o_line[x];  // R = Occlusion
                    orm_line[x * 3 + 1] = r_line[x];  // G = Roughness
                    orm_line[x * 3 + 2] = m_line[x];  // B = Metallic
                }
            }

            // Apply ORM texture
            vtkSmartPointer<vtkImageData> orm_data =
                    impl_->QImageToVtkImage(orm);
            if (orm_data) {
                vtkSmartPointer<vtkTexture> tex =
                        vtkSmartPointer<vtkTexture>::New();
                tex->SetInputData(orm_data);
                tex->SetInterpolate(1);
                tex->MipmapOn();

                property->SetORMTexture(tex);
                CVLog::PrintDebug(
                        "[VtkMultiTextureRenderer::ApplyPBRMaterial] Applied "
                        "ORM texture (%dx%d)",
                        width, height);
            }
        }  // end of orm.isNull() check
    }

    // 5. Configure renderer lighting (enhanced illumination)
    // Only set up lighting once per renderer to avoid repeated operations
    if (renderer) {
        vtkLightCollection* lights = renderer->GetLights();
        int num_lights = lights ? lights->GetNumberOfItems() : 0;

        // Enable IBL
        renderer->UseImageBasedLightingOn();
        renderer->AutomaticLightCreationOn();

        // Significantly enhance ambient light to avoid being too dark
        renderer->SetAmbient(1.0, 1.0, 1.0);  // Maximum ambient light

        // Only configure lighting if not already set up
        // Check if we have the expected number of lights (1 headlight + auto
        // lights)
        if (num_lights == 0) {
            // Add additional directional light source
            vtkSmartPointer<vtkLight> light = vtkSmartPointer<vtkLight>::New();
            light->SetLightTypeToHeadlight();  // Light source that follows
                                               // camera
            light->SetIntensity(1.5);          // Enhanced brightness
            light->SetColor(1.0, 1.0, 1.0);
            renderer->AddLight(light);

            CVLog::PrintDebug(
                    "[VtkMultiTextureRenderer::ApplyPBRMaterial] Lighting "
                    "configured (first time): ambient=1.0, headlight "
                    "intensity=1.5");
        } else {
            // Enhance existing lights for brighter textured model rendering
            CVLog::PrintDebug(
                    "[VtkMultiTextureRenderer::ApplyPBRMaterial] Found %d "
                    "existing lights, enhancing for textured model",
                    num_lights);
        }
    } else {
        CVLog::Warning(
                "[VtkMultiTextureRenderer::ApplyPBRMaterial] No renderer "
                "provided, lighting may be insufficient");
    }

    // 6. Set rendering quality
    property->BackfaceCullingOff();   // Allow seeing backface
    property->FrontfaceCullingOff();  // Allow seeing frontface

    property->SetLighting(true);
    property->SetShading(true);
    property->SetAmbient(1.0f);
    property->SetDiffuse(1.0f);
    property->SetSpecular(1.0f);
#else
    CVLog::Warning(
            "[VtkMultiTextureRenderer::ApplyPBRMaterial] VTK < 9.0, falling "
            "back to Phong");
    property->SetInterpolationToPhong();
    property->SetColor(material.baseColor[0], material.baseColor[1],
                       material.baseColor[2]);
    // Set RGB colors for ambient, diffuse, specular
    property->SetAmbientColor(material.ambientColor[0],
                              material.ambientColor[1],
                              material.ambientColor[2]);
    property->SetDiffuseColor(material.diffuseColor[0],
                              material.diffuseColor[1],
                              material.diffuseColor[2]);
    property->SetSpecularColor(material.specularColor[0],
                               material.specularColor[1],
                               material.specularColor[2]);
    // Set intensity coefficients to 1.0 to preserve RGB color brightness
    // Since we're using RGB colors, setting intensity to 1.0 preserves the
    // original color brightness
    property->SetAmbient(1.0f);
    property->SetDiffuse(1.0f);
    property->SetSpecular(1.0f);
    property->SetSpecularPower(material.shininess);
    property->SetOpacity(opacity);  // Use clamped value

    // Only apply BaseColor texture
    if (!material.baseColorTexture.empty()) {
        QImage img = ccMaterial::GetTexture(material.baseColorTexture.c_str());
        if (!img.isNull()) {
            vtkSmartPointer<vtkImageData> image_data =
                    impl_->QImageToVtkImage(img);
            if (image_data) {
                vtkSmartPointer<vtkTexture> tex =
                        vtkSmartPointer<vtkTexture>::New();
                tex->SetInputData(image_data);
                tex->SetInterpolate(1);
                tex->MipmapOn();
                actor->SetTexture(tex);
            }
        }
    }
#endif

    CVLog::PrintDebug(
            "[VtkMultiTextureRenderer::ApplyPBRRendering] PBR rendering setup "
            "complete");
    return true;
}

bool VtkMultiTextureRenderer::ApplyTexturedRendering(
        vtkSmartPointer<vtkActor> actor,
        const PBRMaterial& material,
        float opacity) {
    // ========================================================================
    // Mode 2: Single texture rendering (only baseColor texture)
    // ========================================================================
    CVLog::PrintDebug(
            "[VtkMultiTextureRenderer::ApplyTexturedRendering] Using single "
            "texture mode");

    vtkProperty* property = actor->GetProperty();
    if (!property) {
        CVLog::Error(
                "[VtkMultiTextureRenderer::ApplyTexturedRendering] Property is "
                "null");
        return false;
    }

    // Set material properties
    SetProperties(property, material, opacity);

    // Load texture
    std::string tex_path = material.baseColorTexture;
    if (tex_path.empty()) tex_path = material.emissiveTexture;

    if (!tex_path.empty()) {
        // Check if file exists
        if (!cloudViewer::utility::filesystem::FileExists(tex_path)) {
            CVLog::Warning(
                    "[VtkMultiTextureRenderer::ApplyTexturedRendering] Texture "
                    "file not found: %s",
                    tex_path.c_str());
            return false;
        }

        QImage img = ccMaterial::GetTexture(tex_path.c_str());
        if (!img.isNull()) {
            vtkSmartPointer<vtkImageData> image_data =
                    impl_->QImageToVtkImage(img);
            if (image_data) {
                vtkSmartPointer<vtkTexture> tex =
                        vtkSmartPointer<vtkTexture>::New();
                tex->SetInputData(image_data);
                tex->SetInterpolate(1);
                tex->MipmapOn();
                actor->SetTexture(tex);
                CVLog::PrintDebug(
                        "[VtkMultiTextureRenderer::ApplyTexturedRendering] "
                        "Applied single texture: %dx%d",
                        img.width(), img.height());
            } else {
                CVLog::Warning(
                        "[VtkMultiTextureRenderer::ApplyTexturedRendering] "
                        "Failed to convert QImage to vtkImageData");
                return false;
            }
        } else {
            CVLog::Warning(
                    "[VtkMultiTextureRenderer::ApplyTexturedRendering] Failed "
                    "to load texture: %s",
                    tex_path.c_str());
            return false;
        }
    } else {
        CVLog::Warning(
                "[VtkMultiTextureRenderer::ApplyTexturedRendering] No texture "
                "path provided");
        return false;
    }

    CVLog::PrintDebug(
            "[VtkMultiTextureRenderer::ApplyTexturedRendering] Textured "
            "rendering setup complete");
    return true;
}

bool VtkMultiTextureRenderer::ApplyMaterialOnlyRendering(
        vtkSmartPointer<vtkActor> actor,
        const PBRMaterial& material,
        float opacity) {
    // ========================================================================
    // Mode 3: Pure material rendering (no texture) - PBR mode
    // ========================================================================
    CVLog::PrintDebug(
            "[VtkMultiTextureRenderer::ApplyMaterialOnlyRendering] Using "
            "material-only PBR mode");

    vtkProperty* property = actor->GetProperty();
    if (!property) {
        CVLog::Error(
                "[VtkMultiTextureRenderer::ApplyMaterialOnlyRendering] "
                "Property is null");
        return false;
    }

    // Enable PBR interpolation mode (same as ApplyPBRRendering)
    property->SetInterpolationToPBR();

    // Set base material properties
    property->SetColor(material.baseColor[0], material.baseColor[1],
                       material.baseColor[2]);
    property->SetOpacity(opacity);

    // Set PBR scalar parameters (metallic, roughness, ao)
    // These are essential for PBR rendering to show proper glossy/reflective
    // surfaces
    float metallic = std::max(0.0f, std::min(1.0f, material.metallic));
    float roughness = std::max(0.0f, std::min(1.0f, material.roughness));
    float ao = std::max(0.0f, std::min(2.0f, material.ao));

    property->SetMetallic(metallic);
    property->SetRoughness(roughness);
    property->SetOcclusionStrength(ao);

#if VTK_VERSION_NUMBER >= VTK_VERSION_CHECK(9, 2, 0)
    // Advanced PBR parameters (VTK 9.2+)
    property->SetAnisotropy(std::max(
            0.0, std::min(1.0, static_cast<double>(material.anisotropy))));
    property->SetAnisotropyRotation(std::max(
            0.0,
            std::min(1.0, static_cast<double>(material.anisotropyRotation))));
    property->SetCoatStrength(std::max(
            0.0, std::min(1.0, static_cast<double>(material.clearcoat))));
    property->SetCoatRoughness(std::max(
            0.0,
            std::min(1.0, static_cast<double>(material.clearcoatRoughness))));
    // Note: VTK uses "Coat" terminology for clearcoat

    CVLog::PrintDebug(
            "[VtkMultiTextureRenderer::ApplyMaterialOnlyRendering] Advanced "
            "PBR: "
            "anisotropy=%.2f, clearcoat=%.2f, clearcoatRoughness=%.2f",
            material.anisotropy, material.clearcoat,
            material.clearcoatRoughness);
#endif

    // Set lighting properties for proper PBR rendering
    property->SetLighting(true);
    property->SetShading(true);
    property->BackfaceCullingOff();
    property->FrontfaceCullingOff();

    CVLog::PrintDebug(
            "[VtkMultiTextureRenderer::ApplyMaterialOnlyRendering] Material "
            "properties: "
            "color=(%.2f,%.2f,%.2f), metallic=%.2f, roughness=%.2f, ao=%.2f, "
            "opacity=%.2f",
            material.baseColor[0], material.baseColor[1], material.baseColor[2],
            metallic, roughness, ao, opacity);
    return true;
}

}  // namespace VtkUtils
