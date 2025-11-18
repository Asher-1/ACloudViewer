// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "renders/TextureRenderManager.h"

#include <CVLog.h>
#include <ecvMaterialSet.h>

#include "renders/base/TextureRendererBase.h"
#include "renders/pbr/PBRRenderer.h"
#include "renders/textured/MultiTextureRenderer.h"
#include "renders/utils/MaterialConverter.h"

// VTK
#include <vtkActor.h>
#include <vtkLODActor.h>
#include <vtkPolyData.h>
#include <vtkRenderer.h>

namespace PclUtils {
namespace renders {

TextureRenderManager::TextureRenderManager() { InitializeRenderers(); }

TextureRenderManager::~TextureRenderManager() = default;

void TextureRenderManager::InitializeRenderers() {
    // Register all available renderers in priority order
    // Higher priority renderers are checked first
    // Note: PBRRenderer now handles both PBR textures and material-only cases
    // (VtkMultiTextureRenderer::ApplyPBRMaterial automatically detects the
    // mode)
    renderers_.push_back(std::make_unique<PBRRenderer>());
    renderers_.push_back(std::make_unique<MultiTextureRenderer>());

    CVLog::PrintDebug("[TextureRenderManager] Initialized %zu renderers",
                      renderers_.size());
}

TextureRenderManager::MaterialAnalysis TextureRenderManager::AnalyzeMaterials(
        const ccMaterialSet* materials) const {
    MaterialAnalysis analysis;

    if (!materials || materials->empty()) {
        return analysis;
    }

    analysis.material_count = materials->size();

    // Check first material for PBR encoding
    auto firstMaterial = materials->at(0);
    if (firstMaterial) {
        analysis.has_pbr_textures =
                MaterialConverter::HasPBREncoding(firstMaterial.data());
    }

    // Check for multiple map_Kd textures
    analysis.has_multiple_map_kd =
            MaterialConverter::HasMultipleMapKd(materials);

    // Check if any material has PBR textures (not just encoding)
    if (!analysis.has_pbr_textures) {
        auto pbr_material = MaterialConverter::FromMaterialSet(materials);
        analysis.has_pbr_textures = pbr_material.hasPBRTextures();
    }

    // Also check if any material has any texture files at all
    // This helps determine if materials have textures or are material-only
    // PBRRenderer now handles both cases (with and without textures)
    analysis.has_any_texture = HasAnyTextureFiles(materials);

    CVLog::PrintDebug(
            "[TextureRenderManager::AnalyzeMaterials] Analysis complete: "
            "count=%zu, has_pbr=%d, has_multiple_map_kd=%d, has_any_texture=%d",
            analysis.material_count, analysis.has_pbr_textures,
            analysis.has_multiple_map_kd, analysis.has_any_texture);

    return analysis;
}

TextureRendererBase* TextureRenderManager::SelectRenderer(
        const ccMaterialSet* materials) const {
    if (!materials || materials->empty()) {
        CVLog::Warning(
                "[TextureRenderManager::SelectRenderer] No materials "
                "provided");
        return nullptr;
    }

    MaterialAnalysis analysis = AnalyzeMaterials(materials);

    CVLog::PrintDebug(
            "[TextureRenderManager::SelectRenderer] Analysis: count=%zu, "
            "has_pbr=%d, has_multiple_map_kd=%d",
            analysis.material_count, analysis.has_pbr_textures,
            analysis.has_multiple_map_kd);

    // Try each renderer in order until one can handle the materials
    for (auto& renderer : renderers_) {
        if (renderer->CanHandle(analysis.material_count,
                                analysis.has_pbr_textures,
                                analysis.has_multiple_map_kd)) {
            CVLog::PrintDebug(
                    "[TextureRenderManager::SelectRenderer] Selected: %s",
                    renderer->GetName().c_str());
            return renderer.get();
        }
    }

    CVLog::Warning(
            "[TextureRenderManager::SelectRenderer] No renderer can handle "
            "materials");
    return nullptr;
}

TextureRendererBase* TextureRenderManager::GetRenderer(
        RenderingMode mode) const {
    for (auto& renderer : renderers_) {
        if (renderer->GetMode() == mode) {
            return renderer.get();
        }
    }
    return nullptr;
}

bool TextureRenderManager::HasAnyTextureFiles(
        const ccMaterialSet* materials) const {
    if (!materials || materials->empty()) {
        return false;
    }

    for (size_t i = 0; i < materials->size(); ++i) {
        auto mat = materials->at(i);
        if (!mat) continue;

        // Check if material has any texture file (not just texture map type)
        using TexType = ccMaterial::TextureMapType;
        for (int type = 0; type < 15; ++type) {  // All texture types
            if (mat->hasTextureMap(static_cast<TexType>(type))) {
                // Check if this texture type has actual files
                std::vector<QString> textureFiles =
                        mat->getTextureFilenames(static_cast<TexType>(type));
                for (const auto& file : textureFiles) {
                    if (!file.isEmpty()) {
                        return true;  // Found at least one texture file
                    }
                }
            }
        }
    }

    return false;  // No texture files found
}

bool TextureRenderManager::Apply(vtkLODActor* actor,
                                 const ccMaterialSet* materials,
                                 vtkPolyData* polydata,
                                 vtkRenderer* renderer) {
    if (!actor) {
        CVLog::Error("[TextureRenderManager::Apply] Actor is null");
        return false;
    }

    TextureRendererBase* selected_renderer = SelectRenderer(materials);
    if (!selected_renderer) {
        CVLog::Error(
                "[TextureRenderManager::Apply] Cannot select renderer for "
                "materials");
        return false;
    }

    CVLog::PrintDebug(
            "[TextureRenderManager::Apply] Applying rendering using %s",
            selected_renderer->GetName().c_str());

    return selected_renderer->Apply(actor, materials, polydata, renderer);
}

bool TextureRenderManager::Update(vtkActor* actor,
                                  const ccMaterialSet* materials,
                                  vtkPolyData* polydata,
                                  vtkRenderer* renderer) {
    if (!actor) {
        CVLog::Error("[TextureRenderManager::Update] Actor is null");
        return false;
    }

    TextureRendererBase* selected_renderer = SelectRenderer(materials);
    if (!selected_renderer) {
        CVLog::Error(
                "[TextureRenderManager::Update] Cannot select renderer for "
                "materials");
        return false;
    }

    CVLog::PrintDebug(
            "[TextureRenderManager::Update] Updating rendering using %s",
            selected_renderer->GetName().c_str());

    return selected_renderer->Update(actor, materials, polydata, renderer);
}

}  // namespace renders
}  // namespace PclUtils
