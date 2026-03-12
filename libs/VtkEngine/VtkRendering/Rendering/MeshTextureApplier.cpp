// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "MeshTextureApplier.h"

#include <CVLog.h>
#include <CVTools.h>
#include <Converters/Cc2Vtk.h>
#include <ecvGenericMesh.h>
#include <ecvHObjectCaster.h>
#include <ecvMaterial.h>
#include <ecvMaterialSet.h>
#include <ecvPointCloud.h>

#include <Eigen/Dense>

#include "TextureRenderManager.h"

// VTK
#include <vtkDataObject.h>
#include <vtkFloatArray.h>
#include <vtkLODActor.h>
#include <vtkMatrix4x4.h>
#include <vtkOpenGLRenderWindow.h>
#include <vtkPointData.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkSmartPointer.h>
#include <vtkTexture.h>
#include <vtkTextureUnitManager.h>

namespace Visualization {
namespace renders {

bool MeshTextureApplier::ApplyTexturesFromCCMesh(
        vtkLODActor* actor,
        ccGenericMesh*
                mesh,  // Non-const because getTriangleVertIndexes is non-const
        vtkPolyData* polydata,
        TextureRenderManager* render_manager,
        vtkRenderer* renderer) {
    if (!actor || !mesh || !render_manager) {
        return false;
    }

    const ccMaterialSet* materials = mesh->getMaterialSet();
    if (!materials || materials->empty()) {
        CVLog::Warning(
                "[MeshTextureApplier::ApplyTexturesFromCCMesh] No materials "
                "found in mesh");
        return false;
    }

    ccPointCloud* cloud =
            ccHObjectCaster::ToPointCloud(mesh->getAssociatedCloud());
    if (!cloud) {
        CVLog::Error(
                "[MeshTextureApplier::ApplyTexturesFromCCMesh] Failed to get "
                "point cloud from mesh!");
        return false;
    }

    vtkSmartPointer<vtkPolyData> new_polydata;
    vtkSmartPointer<vtkMatrix4x4> transformation;
    std::vector<std::vector<Eigen::Vector2f>> tex_coordinates;

    if (!Converters::Cc2Vtk::TextureMeshToPolyData(
                cloud, mesh, new_polydata, transformation, tex_coordinates)) {
        CVLog::Error(
                "[MeshTextureApplier::ApplyTexturesFromCCMesh] Failed to "
                "convert mesh to VTK with textures!");
        return false;
    }

    // Replace polydata with the new one (which has correct texture coordinates)
    if (polydata && new_polydata) {
        polydata->ShallowCopy(new_polydata);
    } else if (new_polydata) {
        polydata = new_polydata.Get();
    }

    // transformation is automatically managed by smart pointer, no manual
    // cleanup needed

    CVLog::Print(
            "[MeshTextureApplier::ApplyTexturesFromCCMesh] Successfully "
            "converted mesh using getVtkPolyDataWithTextures with %zu texture "
            "coordinate groups",
            tex_coordinates.size());

    // Apply textures using material set
    return ApplyTexturesFromMaterialSet(actor, materials, tex_coordinates,
                                        polydata, render_manager, renderer);
}

bool MeshTextureApplier::ApplyTexturesFromMaterialSet(
        vtkLODActor* actor,
        const ccMaterialSet* materials,
        const std::vector<std::vector<Eigen::Vector2f>>& tex_coordinates,
        vtkPolyData* polydata,
        TextureRenderManager* render_manager,
        vtkRenderer* renderer) {
    if (!actor || !materials || !render_manager) {
        return false;
    }

    // Setup texture coordinate arrays in polydata
    // CRITICAL: This must match the logic in addTextureMesh
    // (VtkVis.cpp:1577-1598)
    //
    // In getPclTextureMesh, texture coordinates are grouped by material:
    // - tex_coordinates[0] contains coordinates for material 0's triangles
    // - tex_coordinates[1] contains coordinates for material 1's triangles
    // - etc.
    //
    // Each material's texture coordinates are already in the correct order
    // (matching the triangle order from getPclCloud2).
    //
    // For multi-texture rendering, we need:
    // - TCoords0: coordinates for material 0 (only material 0's triangles have
    // valid coords)
    // - TCoords1: coordinates for material 1 (only material 1's triangles have
    // valid coords)
    // - etc.
    //
    // But VTK requires arrays with the same size as points, so we need to fill
    // invalid regions with (-1, -1) or (0, 0) for triangles that don't belong
    // to that material.
    //
    // However, since getPclCloud2 creates expanded points (each triangle vertex
    // is separate), and tex_coordinates[mat_idx] already contains coordinates
    // in the correct order, we can directly use them. But we need to create a
    // full-size array and fill other regions appropriately.

    if (polydata) {
        vtkIdType numPoints = polydata->GetNumberOfPoints();

        // Create texture coordinate arrays for each material
        // Each array corresponds to one material's texture coordinates
        // This matches the logic in addTextureMesh where tex_id corresponds to
        // material index
        size_t texture_index = 0;

        for (size_t mat_idx = 0;
             mat_idx < materials->size() && mat_idx < tex_coordinates.size();
             ++mat_idx) {
            if (tex_coordinates[mat_idx].empty()) {
                continue;
            }

            auto material = materials->at(mat_idx);
            if (!material) continue;

            // Get all DIFFUSE (map_Kd) textures for this material
            using TexType = ccMaterial::TextureMapType;
            std::vector<QString> diffuseTextures =
                    material->getTextureFilenames(TexType::DIFFUSE);

            // Create a texture coordinate array for EACH map_Kd texture in this
            // material All textures from the same material share the same
            // coordinates
            for (size_t tex_in_mat = 0; tex_in_mat < diffuseTextures.size();
                 ++tex_in_mat) {
                if (diffuseTextures[tex_in_mat].isEmpty()) {
                    continue;
                }

                // Create texture coordinate array for this texture
                // Use the material's texture coordinates directly
                vtkSmartPointer<vtkFloatArray> coordinates =
                        vtkSmartPointer<vtkFloatArray>::New();
                coordinates->SetNumberOfComponents(2);
                std::stringstream ss;
                ss << "TCoords" << texture_index;
                std::string coords_name = ss.str();
                coordinates->SetName(coords_name.c_str());

                // Fill coordinates for this material
                // tex_coordinates[mat_idx] already contains coordinates in the
                // correct order matching the point order from getPclCloud2
                for (const auto& tc : tex_coordinates[mat_idx]) {
                    coordinates->InsertNextTuple2(tc[0], tc[1]);
                }

                // If the coordinate array size doesn't match numPoints, we need
                // to pad This shouldn't happen if getPclCloud2 and
                // getPclTextureMesh are consistent, but we handle it for safety
                vtkIdType currentSize = coordinates->GetNumberOfTuples();
                if (currentSize < numPoints) {
                    CVLog::Warning(
                            "[MeshTextureApplier::ApplyTexturesFromMaterialSet]"
                            " "
                            "Material %zu texture coordinate count (%ld) < "
                            "point count "
                            "(%ld), padding with default coordinates",
                            mat_idx, currentSize, numPoints);
                    for (vtkIdType i = currentSize; i < numPoints; ++i) {
                        coordinates->InsertNextTuple2(0.0f, 0.0f);
                    }
                } else if (currentSize > numPoints) {
                    CVLog::Warning(
                            "[MeshTextureApplier::ApplyTexturesFromMaterialSet]"
                            " "
                            "Material %zu texture coordinate count (%ld) > "
                            "point count "
                            "(%ld), truncating",
                            mat_idx, currentSize, numPoints);
                    // Note: VTK arrays can't be easily truncated, so we'll just
                    // use what we have This indicates a data inconsistency that
                    // should be fixed upstream
                }

                polydata->GetPointData()->AddArray(coordinates);

                // Set first texture coordinates as active TCoords (for PBR)
                if (texture_index == 0) {
                    polydata->GetPointData()->SetTCoords(coordinates);
                }

                texture_index++;
            }
        }

        CVLog::PrintDebug(
                "[MeshTextureApplier::ApplyTexturesFromMaterialSet] Created "
                "%zu "
                "texture coordinate arrays for %zu points",
                texture_index, numPoints);
    }

    // Use TextureRenderManager to apply materials
    return render_manager->Apply(actor, materials, polydata, renderer);
}

}  // namespace renders
}  // namespace Visualization
