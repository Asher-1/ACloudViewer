// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "mvs/textured_mesh_export.h"

#include <unordered_map>

// CV_CORE_LIB
#include <CVLog.h>

// CV_DB_LIB
#include <ecvAdvancedTypes.h>
#include <ecvMaterial.h>
#include <ecvMaterialSet.h>
#include <ecvMesh.h>
#include <ecvPointCloud.h>

// CV_IO_LIB
#include <ObjFilter.h>

#include "util/logging.h"

namespace colmap {
namespace mvs {

bool ExportTexturedMeshAsObj(const MeshTextureMappingResult& result,
                             const PlyMesh& mesh,
                             const std::filesystem::path& output_path) {
    THROW_CHECK(!mesh.vertices.empty());
    THROW_CHECK_EQ(result.face_uvs.size() / 6, mesh.faces.size());
    THROW_CHECK(result.texture_atlas.Width() > 0 &&
                result.texture_atlas.Height() > 0);

    // Write the atlas next to the OBJ; ccMaterialSet::saveAsMTL re-saves the
    // material texture under its (absolute) filename's basename inside the
    // MTL directory and references it relatively.
    const std::string texture_filename =
            output_path.stem().string() + "_map_Kd.png";
    const std::filesystem::path texture_path =
            output_path.parent_path() / texture_filename;
    THROW_CHECK(result.texture_atlas.Write(texture_path));

    // Vertices.
    ccPointCloud* cloud = new ccPointCloud("vertices");
    if (!cloud->reserve(static_cast<unsigned>(mesh.vertices.size()))) {
        delete cloud;
        CVLog::Error("[ObjExport] Not enough memory for vertices");
        return false;
    }
    for (const PlyMeshVertex& v : mesh.vertices) {
        cloud->addPoint(CCVector3(static_cast<PointCoordinateType>(v.x),
                                  static_cast<PointCoordinateType>(v.y),
                                  static_cast<PointCoordinateType>(v.z)));
    }

    ccMesh* export_mesh = new ccMesh(cloud);
    export_mesh->setName("textured-mesh");
    export_mesh->addChild(cloud);
    cloud->setEnabled(false);

    if (!export_mesh->reserve(static_cast<unsigned>(mesh.faces.size())) ||
        !export_mesh->reservePerTriangleMtlIndexes() ||
        !export_mesh->reservePerTriangleTexCoordIndexes()) {
        delete export_mesh;
        CVLog::Error("[ObjExport] Not enough memory for triangles");
        return false;
    }

    // Per-corner texture coordinates (the OBJ vt pool). Quantized dedup keeps
    // the pool compact across shared corners without splitting vertices.
    TextureCoordsContainer* texCoords = new TextureCoordsContainer();
    texCoords->link();
    if (!texCoords->reserveSafe(mesh.faces.size() * 3)) {
        texCoords->release();
        delete export_mesh;
        CVLog::Error("[ObjExport] Not enough memory for texture coordinates");
        return false;
    }
    std::unordered_map<uint64_t, int> uv_lut;
    uv_lut.reserve(mesh.faces.size() * 3);
    constexpr float kQuant = 1e6f;

    int* tc_indexes = new int[mesh.faces.size() * 3];
    for (size_t fi = 0; fi < mesh.faces.size(); ++fi) {
        const PlyMeshFace& face = mesh.faces[fi];
        export_mesh->addTriangle(face.vertex_idx1, face.vertex_idx2,
                                 face.vertex_idx3);
        export_mesh->addTriangleMtlIndex(0);

        for (size_t k = 0; k < 3; ++k) {
            // face_uvs already carries standard OBJ texture coordinates
            // (GL convention, v = 0 at the image bottom - see
            // texture_mapping.cc ComputeFaceUVs) and must be written
            // verbatim: ObjFilter stores vt as-is, and ACloudViewer's VTK
            // upload chain flips the image on GPU upload
            // (vtkQImageToImageSource), so t = 0 samples the image BOTTOM
            // row and the standard vt pair renders right side up. An extra
            // 1 - v here inverts the mapping and pushes ~half of the faces
            // onto the atlas black background. (MvsTexturing::SaveOBJModel
            // does flip because ITS input atlas.texcoords are image-top
            // normalized; face_uvs are NOT - the conventions differ.)
            const float u = result.face_uvs[fi * 6 + k * 2 + 0];
            const float v = result.face_uvs[fi * 6 + k * 2 + 1];
            const uint64_t key =
                    (static_cast<uint64_t>(
                             static_cast<int64_t>(std::lrintf(u * kQuant)))
                     << 32) |
                    static_cast<uint32_t>(
                            static_cast<int64_t>(std::lrintf(v * kQuant)));
            auto it = uv_lut.find(key);
            if (it == uv_lut.end()) {
                const int idx = static_cast<int>(texCoords->currentSize());
                texCoords->addElement(TexCoords2D(u, v));
                it = uv_lut.emplace(key, idx).first;
            }
            tc_indexes[fi * 3 + k] = it->second;
        }
        export_mesh->addTriangleTexCoordIndexes(
                tc_indexes[fi * 3 + 0], tc_indexes[fi * 3 + 1],
                tc_indexes[fi * 3 + 2]);
    }
    delete[] tc_indexes;

    export_mesh->setTexCoordinatesTable(texCoords);

    // Vertex normals: without them the OBJ lights to near-black under the
    // lit shading path (the renderer logs "Mesh has no normal" and the
    // diffuse term N.L has no valid normal). Computed from the triangle
    // topology; the winding order of the Poisson output orients them
    // outward. ObjFilter emits one vn per vertex and face lines f v/vt/vn.
    if (!export_mesh->computePerVertexNormals()) {
        CVLog::Warning("[ObjExport] Vertex normal computation failed; "
                       "the OBJ will have no vn records");
    }

    // Single material carrying the atlas; saveAsMTL emits the map_Kd entry.
    ccMaterialSet* materials = new ccMaterialSet("materials");
    ccMaterial::Shared mat(new ccMaterial("material0000"));
    mat->setAmbient(ecvColor::Rgbaf(1.0f, 1.0f, 1.0f, 1.0f));
    mat->setDiffuse(ecvColor::Rgbaf(1.0f, 1.0f, 1.0f, 1.0f));
    mat->setSpecular(ecvColor::Rgbaf(0.0f, 0.0f, 0.0f, 1.0f));
    mat->setShininess(1.0f);
    mat->setTexture(QImage(QString::fromStdString(texture_path.string())),
                    QString::fromStdString(texture_path.string()),
                    /*mirrorImage=*/false);
    if (mat->getTexture().isNull()) {
        CVLog::Warning(
                "[ObjExport] Atlas texture could not be loaded as QImage; the "
                "MTL will reference the file but the material has no image");
    }
    materials->addMaterial(mat);
    export_mesh->setMaterialSet(materials);

    // Canonical writer: same OBJ/MTL reader/writer pair as everywhere else.
    ObjFilter obj_filter;
    FileIOFilter::SaveParameters params;
    params.alwaysDisplaySaveDialog = false;
    params.parentWidget = nullptr;

    const CC_FILE_ERROR err =
            obj_filter.saveToFile(export_mesh,
                                  QString::fromStdString(output_path.string()),
                                  params);
    delete export_mesh;

    if (err != CC_FERR_NO_ERROR) {
        CVLog::Error(QString("[ObjExport] ObjFilter failed with error %1")
                             .arg(static_cast<int>(err)));
        return false;
    }

    LOG(INFO) << "Exported OBJ model to " << output_path.string()
              << " via ObjFilter";
    return true;
}

}  // namespace mvs
}  // namespace colmap
