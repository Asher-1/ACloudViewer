// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "renders/utils/MeshMaterialExtractor.h"

#include <CVLog.h>
#include <ecvGenericMesh.h>
#include <ecvMaterialSet.h>

#include "renders/TextureRenderManager.h"
#include "renders/utils/MaterialConverter.h"

// VTK
#include <vtkLODActor.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkRenderer.h>

namespace PclUtils {
namespace renders {

bool MeshMaterialExtractor::ApplyMaterialFromMesh(
        vtkLODActor* actor,
        const ccGenericMesh* mesh,
        vtkPolyData* polydata,
        TextureRenderManager* render_manager,
        vtkRenderer* renderer) {
    if (!actor) {
        CVLog::Error(
                "[MeshMaterialExtractor::ApplyMaterialFromMesh] Actor is null");
        return false;
    }

    if (!render_manager) {
        CVLog::Error(
                "[MeshMaterialExtractor::ApplyMaterialFromMesh] Render manager "
                "is null");
        return false;
    }

    if (!mesh) {
        CVLog::Warning(
                "[MeshMaterialExtractor::ApplyMaterialFromMesh] Mesh is null, "
                "using default material");
        // Use default material via render manager
        ccMaterialSet default_materials("DefaultMaterial");
        ccMaterial::Shared default_mat = ccMaterial::Shared(new ccMaterial);
        default_mat->setName("default");
        default_mat->setDiffuse(ecvColor::Rgbaf(0.8f, 0.8f, 0.8f, 1.0f));
        default_materials.addMaterial(default_mat);

        return render_manager->Apply(actor, &default_materials, polydata,
                                     renderer);
    }

    CVLog::Print(
            "[MeshMaterialExtractor::ApplyMaterialFromMesh] Extracting "
            "material from mesh");

    // Try to get materials
    const ccMaterialSet* materials =
            const_cast<ccGenericMesh*>(mesh)->getMaterialSet();
    if (!materials || materials->size() == 0) {
        CVLog::Warning(
                "[MeshMaterialExtractor::ApplyMaterialFromMesh] No materials "
                "found in mesh, using default");
        // Use default material
        ccMaterialSet default_materials("DefaultMaterial");
        ccMaterial::Shared default_mat = ccMaterial::Shared(new ccMaterial);
        default_mat->setName("mesh_default");

        // Try to get color from mesh
        if (mesh->hasColors()) {
            // Use first vertex color as base color
            // Note: Simplified handling, may need more complex logic in
            // practice
            default_mat->setDiffuse(ecvColor::Rgbaf(0.8f, 0.8f, 0.8f, 1.0f));
        } else {
            default_mat->setDiffuse(ecvColor::Rgbaf(0.8f, 0.8f, 0.8f, 1.0f));
        }
        default_materials.addMaterial(default_mat);

        return render_manager->Apply(actor, &default_materials, polydata,
                                     renderer);
    }

    // Apply materials using render manager
    return render_manager->Apply(actor, materials, polydata, renderer);
}

}  // namespace renders
}  // namespace PclUtils
