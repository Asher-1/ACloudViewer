// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "renders/utils/MeshTextureApplier.h"

#include <CVLog.h>
#include <CVTools.h>
#include <ecvGenericMesh.h>
#include <ecvHObjectCaster.h>  // For ccHObjectCaster
#include <ecvMaterial.h>
#include <ecvMaterialSet.h>
#include <ecvPointCloud.h>  // For POINT_VISIBLE

#include "cc2sm.h"  // For cc2smReader::getVtkPolyDataWithTextures
#include "renders/TextureRenderManager.h"
#include "sm2cc.h"

// PCL
#include <pcl/TextureMesh.h>

#include <Eigen/Dense>

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

namespace PclUtils {
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

    // Use cc2smReader::getVtkPolyDataWithTextures to ensure consistency
    // This reuses the proven logic from getPclTextureMesh and addTextureMesh
    // which ensures texture coordinates match point order perfectly
    ccPointCloud* cloud =
            ccHObjectCaster::ToPointCloud(mesh->getAssociatedCloud());
    if (!cloud) {
        CVLog::Error(
                "[MeshTextureApplier::ApplyTexturesFromCCMesh] Failed to get "
                "point cloud from mesh!");
        return false;
    }

    cc2smReader reader(cloud, true);
    vtkSmartPointer<vtkPolyData> new_polydata;
    vtkSmartPointer<vtkMatrix4x4> transformation;
    std::vector<std::vector<Eigen::Vector2f>> tex_coordinates;

    if (!reader.getVtkPolyDataWithTextures(mesh, new_polydata, transformation,
                                           tex_coordinates)) {
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
    // (PCLVis.cpp:1577-1598)
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

bool MeshTextureApplier::ApplyPBRTextures(vtkLODActor* actor,
                                          const pcl::TextureMesh& mesh,
                                          vtkPolyData* polydata,
                                          TextureRenderManager* render_manager,
                                          vtkRenderer* renderer) {
    if (!actor || !render_manager) {
        return false;
    }

    CVLog::Print(
            "[MeshTextureApplier::ApplyPBRTextures] DEPRECATED: This method "
            "uses pcl::TexMaterial encoding. Consider using "
            "ApplyTexturesFromCCMesh instead.");

    // Convert pcl::TexMaterial to ccMaterialSet for TextureRenderManager
    ccMaterialSet* temp_material_set = new ccMaterialSet("TempMaterials");
    for (const auto& pcl_mat : mesh.tex_materials) {
        ccMaterial::Shared temp_material = ccMaterial::Shared(new ccMaterial);
        pcl2cc::FromPCLMaterial(pcl_mat, temp_material);
        temp_material_set->addMaterial(temp_material);
    }

    // Convert texture coordinates
    // pcl::TextureMesh uses Eigen::aligned_allocator, we need to convert
    std::vector<std::vector<Eigen::Vector2f>> tex_coordinates;
    tex_coordinates.reserve(mesh.tex_coordinates.size());
    for (const auto& coords : mesh.tex_coordinates) {
        std::vector<Eigen::Vector2f> coord_vec;
        coord_vec.reserve(coords.size());
        for (const auto& coord : coords) {
            coord_vec.push_back(Eigen::Vector2f(coord[0], coord[1]));
        }
        tex_coordinates.push_back(coord_vec);
    }

    // Use new method to apply textures
    bool success = ApplyTexturesFromMaterialSet(actor, temp_material_set,
                                                tex_coordinates, polydata,
                                                render_manager, renderer);

    delete temp_material_set;
    return success;
}

bool MeshTextureApplier::ApplyTraditionalTextures(
        vtkLODActor* actor,
        const pcl::TextureMesh& mesh,
        vtkPolyData* polydata,
        vtkRenderWindow* render_window) {
    if (!actor || !polydata || !render_window) {
        return false;
    }

    // This function is kept for backward compatibility but should use
    // TextureRenderManager instead
    CVLog::Warning(
            "[MeshTextureApplier::ApplyTraditionalTextures] This function is "
            "deprecated. Use TextureRenderManager instead.");

    // Get texture unit manager
    vtkOpenGLRenderWindow* gl_window = vtkOpenGLRenderWindow::SafeDownCast(
            static_cast<vtkObjectBase*>(render_window));
    if (!gl_window) {
        return false;
    }
    vtkTextureUnitManager* tex_manager = gl_window->GetTextureUnitManager();
    if (!tex_manager) {
        return false;
    }

    // Check available texture units
    int texture_units = tex_manager->GetNumberOfTextureUnits();
    if (static_cast<size_t>(texture_units) < mesh.tex_materials.size()) {
        CVLog::Warning(
                "[MeshTextureApplier::ApplyTraditionalTextures] GPU texture "
                "units %d < mesh textures %zu!",
                texture_units, mesh.tex_materials.size());
    }

    vtkPolyDataMapper* mapper =
            vtkPolyDataMapper::SafeDownCast(actor->GetMapper());
    if (!mapper) {
        return false;
    }

    // Apply textures (simplified version - full implementation should use
    // MultiTextureRenderer)
    size_t last_tex_id = std::min(mesh.tex_materials.size(),
                                  static_cast<size_t>(texture_units));

    for (size_t tex_id = 0; tex_id < last_tex_id; ++tex_id) {
#if (VTK_MAJOR_VERSION == 8 && VTK_MINOR_VERSION >= 2) || VTK_MAJOR_VERSION > 8
        const char* tu = mesh.tex_materials[tex_id].tex_name.c_str();
#else
        int tu = vtkProperty::VTK_TEXTURE_UNIT_0 + tex_id;
#endif

        // Add texture coordinates array
        vtkSmartPointer<vtkFloatArray> coordinates =
                vtkSmartPointer<vtkFloatArray>::New();
        coordinates->SetNumberOfComponents(2);
        std::stringstream ss;
        ss << "TCoords" << tex_id;
        std::string coords_name = ss.str();
        coordinates->SetName(coords_name.c_str());

        // Fill coordinates
        for (size_t t = 0; t < mesh.tex_coordinates.size(); ++t) {
            if (t == tex_id) {
                for (const auto& tc : mesh.tex_coordinates[t]) {
                    coordinates->InsertNextTuple2(tc[0], tc[1]);
                }
            } else {
                for (size_t tc = 0; tc < mesh.tex_coordinates[t].size(); ++tc) {
                    coordinates->InsertNextTuple2(-1.0, -1.0);
                }
            }
        }

        mapper->MapDataArrayToMultiTextureAttribute(
                tu, coords_name.c_str(),
                vtkDataObject::FIELD_ASSOCIATION_POINTS);
        polydata->GetPointData()->AddArray(coordinates);
    }

    return true;
}

}  // namespace renders
}  // namespace PclUtils
