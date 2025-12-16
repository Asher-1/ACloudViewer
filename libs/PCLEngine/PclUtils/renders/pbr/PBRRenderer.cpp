// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "renders/pbr/PBRRenderer.h"

#include <CVLog.h>
#include <ecvMaterialSet.h>

#include "renders/base/TextureRendererBase.h"
#include "renders/pbr/VtkMultiTextureRenderer.h"
#include "renders/utils/MaterialConverter.h"

// VTK
#include <vtkActor.h>
#include <vtkLODActor.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkRenderer.h>

namespace PclUtils {
namespace renders {

PBRRenderer::PBRRenderer()
    : vtk_renderer_(std::make_unique<VtkUtils::VtkMultiTextureRenderer>()) {}

bool PBRRenderer::CanHandle(size_t material_count,
                            bool has_pbr_textures,
                            bool has_multiple_map_kd) const {
    // PBR renderer can handle:
    // 1. Only ONE material (VTK PBR limitation: each actor can only have one
    // PBR material)
    // 2. Does NOT have multiple map_Kd (VTK PBR limitation)
    // 3. Can handle both cases:
    //    - Has PBR textures (will use PBR rendering mode)
    //    - No textures, only material properties (will use MATERIAL_ONLY
    //    rendering mode)
    // VtkMultiTextureRenderer::ApplyPBRMaterial will automatically detect and
    // use the appropriate mode
    return material_count == 1 && !has_multiple_map_kd;
}

RenderingMode PBRRenderer::GetMode() const { return RenderingMode::PBR; }

bool PBRRenderer::Apply(vtkLODActor* actor,
                        const ccMaterialSet* materials,
                        vtkPolyData* polydata,
                        vtkRenderer* renderer) {
    if (!ValidateActor(actor) || !ValidateMaterials(materials)) {
        return false;
    }

    CVLog::PrintDebug(
            "[PBRRenderer::Apply] Applying unified PBR rendering for %zu "
            "materials "
            "(VTK PBR only supports single material per actor, using first "
            "material). VtkMultiTextureRenderer will automatically detect "
            "rendering mode "
            "(PBR/TEXTURED/MATERIAL_ONLY) based on material properties.",
            materials->size());

    // Convert material
    auto pbr_material = MaterialConverter::FromMaterialSet(materials);

    CVLog::PrintDebug(
            "[PBRRenderer::Apply] Converted material '%s': "
            "baseColorTexture=%s, normalTexture=%s, metallicTexture=%s, "
            "roughnessTexture=%s, aoTexture=%s, hasPBRTextures=%d, "
            "hasAnyTexture=%d",
            pbr_material.name.c_str(), pbr_material.baseColorTexture.c_str(),
            pbr_material.normalTexture.c_str(),
            pbr_material.metallicTexture.c_str(),
            pbr_material.roughnessTexture.c_str(),
            pbr_material.aoTexture.c_str(), pbr_material.hasPBRTextures(),
            pbr_material.hasAnyTexture());

    // Apply using VtkMultiTextureRenderer
    vtkSmartPointer<vtkActor> actor_ptr(actor);
    vtkSmartPointer<vtkPolyData> poly_ptr(polydata);
    bool result = vtk_renderer_->ApplyPBRMaterial(actor_ptr, pbr_material,
                                                  poly_ptr, renderer);

    if (result) {
        CVLog::PrintDebug(
                "[PBRRenderer::Apply] PBR material applied successfully");
    } else {
        CVLog::Error("[PBRRenderer::Apply] Failed to apply PBR material");
    }

    return result;
}

bool PBRRenderer::Update(vtkActor* actor,
                         const ccMaterialSet* materials,
                         vtkPolyData* polydata,
                         vtkRenderer* renderer) {
    if (!ValidateActor(actor) || !ValidateMaterials(materials)) {
        return false;
    }

    vtkLODActor* lod_actor = vtkLODActor::SafeDownCast(actor);
    if (!lod_actor) {
        CVLog::Warning(
                "[PBRRenderer::Update] Actor is not a vtkLODActor, cannot "
                "apply PBR");
        return false;
    }

    return Apply(lod_actor, materials, polydata, renderer);
}

std::string PBRRenderer::GetName() const { return "PBRRenderer"; }

}  // namespace renders
}  // namespace PclUtils
