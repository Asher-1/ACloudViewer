// ----------------------------------------------------------------------------
// -                        CloudViewer: www.erow.cn                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2020 www.erow.cn
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include <fstream>
#include <numeric>
#include <vector>

#include <Console.h>
#include <FileSystem.h>

#include <ecvPointCloud.h>

#include "assimp/Importer.hpp"
#include "assimp/pbrmaterial.h"
#include "assimp/postprocess.h"
#include "assimp/scene.h"
#include <ImageIO.h>
#include <FileFormatIO.h>
#include <TriangleMeshIO.h>
#include "visualization/rendering/Material.h"
#include "visualization/rendering/Model.h"

#define AI_MATKEY_CLEARCOAT_THICKNESS "$mat.clearcoatthickness", 0, 0
#define AI_MATKEY_CLEARCOAT_ROUGHNESS "$mat.clearcoatroughness", 0, 0
#define AI_MATKEY_SHEEN "$mat.sheen", 0, 0
#define AI_MATKEY_ANISOTROPY "$mat.anisotropy", 0, 0

namespace cloudViewer {
namespace io {
    using namespace CVLib;

const unsigned int kPostProcessFlags =
        aiProcess_GenSmoothNormals | aiProcess_JoinIdenticalVertices |
        aiProcess_ImproveCacheLocality | aiProcess_RemoveRedundantMaterials |
        aiProcess_Triangulate | aiProcess_GenUVCoords | aiProcess_SortByPType |
        aiProcess_FindDegenerates | aiProcess_OptimizeMeshes |
        aiProcess_PreTransformVertices;

struct TextureImages {
    std::shared_ptr<geometry::Image> albedo;
    std::shared_ptr<geometry::Image> normal;
    std::shared_ptr<geometry::Image> ao;
    std::shared_ptr<geometry::Image> roughness;
    std::shared_ptr<geometry::Image> metallic;
    std::shared_ptr<geometry::Image> reflectance;
    std::shared_ptr<geometry::Image> clearcoat;
    std::shared_ptr<geometry::Image> clearcoat_roughness;
    std::shared_ptr<geometry::Image> anisotropy;
    std::shared_ptr<geometry::Image> gltf_rough_metal;
};

void LoadTextures(const std::string& filename,
                  aiMaterial* mat,
                  TextureImages& maps) {
    // Retrieve textures
    std::string base_path =
            utility::filesystem::GetFileParentDirectory(filename);

    auto texture_loader = [&base_path, &mat](
                                  aiTextureType type,
                                  std::shared_ptr<geometry::Image>& img) {
        if (mat->GetTextureCount(type) > 0) {
            aiString path;
            mat->GetTexture(type, 0, &path);
            std::string strpath(path.C_Str());
            // normalize path separators
            auto p_win = strpath.find("\\");
            while (p_win != std::string::npos) {
                strpath[p_win] = '/';
                p_win = strpath.find("\\", p_win + 1);
            }
            // if absolute path convert to relative to base path
            if (strpath.length() > 1 &&
                (strpath[0] == '/' || strpath[1] == ':')) {
                strpath = utility::filesystem::GetFileNameWithoutDirectory(
                        strpath);
            }
            auto image = io::CreateImageFromFile(base_path + strpath);
            if (image->HasData()) {
                img = image;
            }
        }
    };

    texture_loader(aiTextureType_DIFFUSE, maps.albedo);
    texture_loader(aiTextureType_NORMALS, maps.normal);
    // Assimp may place ambient occlusion texture in AMBIENT_OCCLUSION if
    // format has AO support. Prefer that texture if it is preset. Otherwise,
    // try AMBIENT where OBJ and FBX typically put AO textures.
    if (mat->GetTextureCount(aiTextureType_AMBIENT_OCCLUSION) > 0) {
        texture_loader(aiTextureType_AMBIENT_OCCLUSION, maps.ao);
    } else {
        texture_loader(aiTextureType_AMBIENT, maps.ao);
    }
    texture_loader(aiTextureType_METALNESS, maps.metallic);
    if (mat->GetTextureCount(aiTextureType_DIFFUSE_ROUGHNESS) > 0) {
        texture_loader(aiTextureType_DIFFUSE_ROUGHNESS, maps.roughness);
    } else if (mat->GetTextureCount(aiTextureType_SHININESS) > 0) {
        // NOTE: In some FBX files assimp puts the roughness texture in
        // shininess slot
        texture_loader(aiTextureType_SHININESS, maps.roughness);
    }
    // NOTE: Assimp doesn't have a texture type for GLTF's combined
    // roughness/metallic texture so it puts it in the 'unknown' texture slot
    texture_loader(aiTextureType_UNKNOWN, maps.gltf_rough_metal);
    // NOTE: the following may be non-standard. We are using REFLECTION texture
    // type to store OBJ map_Ps 'sheen' PBR map
    texture_loader(aiTextureType_REFLECTION, maps.reflectance);

    // NOTE: ASSIMP doesn't appear to provide texture params for the following:
    // clearcoat
    // clearcoat_roughness
    // anisotropy
}

bool ReadTriangleMeshUsingASSIMP(const std::string& filename,
                                 ccMesh& mesh, bool print_progress) {
    Assimp::Importer importer;
    const auto* scene = importer.ReadFile(filename.c_str(), kPostProcessFlags);
    if (!scene) {
        utility::LogWarning("Unable to load file {} with ASSIMP", filename);
        return false;
    }

    mesh.clear();
    if (!mesh.getAssociatedCloud())
    {
        if (scene->mNumMeshes > 0) {
            const auto* assimp_mesh = scene->mMeshes[0];
            mesh.createInternalCloud();
            if(!mesh.reserveAssociatedCloud(assimp_mesh->mNumVertices)) {
                return false;
            }
        } else {
            utility::LogWarning("Must call createInternalCloud first!");
            return false;
        }
    }

    size_t current_vidx = 0;
    // Merge individual meshes in aiScene into a single TriangleMesh
    for (size_t midx = 0; midx < scene->mNumMeshes; ++midx) {
        const auto* assimp_mesh = scene->mMeshes[midx];
        // Only process triangle meshes
        if (assimp_mesh->mPrimitiveTypes != aiPrimitiveType_TRIANGLE) {
            utility::LogInfo(
                    "Skipping non-triangle primitive geometry of type: "
                    "{}",
                    assimp_mesh->mPrimitiveTypes);
            continue;
        }

        // copy vertex data
        for (size_t vidx = 0; vidx < assimp_mesh->mNumVertices; ++vidx) {
            auto& vertex = assimp_mesh->mVertices[vidx];
            mesh.addVertice({ vertex.x, vertex.y, vertex.z });
        }

        // copy face indices data
        for (size_t fidx = 0; fidx < assimp_mesh->mNumFaces; ++fidx) {
            auto& face = assimp_mesh->mFaces[fidx];
            Eigen::Vector3i facet(
                    face.mIndices[0] + static_cast<int>(current_vidx),
                    face.mIndices[1] + static_cast<int>(current_vidx),
                    face.mIndices[2] + static_cast<int>(current_vidx));
            mesh.addTriangle(facet);
        }

        if (assimp_mesh->mNormals) {
            for (size_t nidx = 0; nidx < assimp_mesh->mNumVertices; ++nidx) {
                auto& normal = assimp_mesh->mNormals[nidx];
                mesh.addVertexNormal({ normal.x, normal.y, normal.z });
            }
        }

        // NOTE: only support a single UV channel
        if (assimp_mesh->HasTextureCoords(0)) {
            for (size_t fidx = 0; fidx < assimp_mesh->mNumFaces; ++fidx) {
                auto& face = assimp_mesh->mFaces[fidx];
                auto& uv1 = assimp_mesh->mTextureCoords[0][face.mIndices[0]];
                auto& uv2 = assimp_mesh->mTextureCoords[0][face.mIndices[1]];
                auto& uv3 = assimp_mesh->mTextureCoords[0][face.mIndices[2]];
                mesh.triangle_uvs_.push_back(Eigen::Vector2d(uv1.x, uv1.y));
                mesh.triangle_uvs_.push_back(Eigen::Vector2d(uv2.x, uv2.y));
                mesh.triangle_uvs_.push_back(Eigen::Vector2d(uv3.x, uv3.y));
            }
        }

        // NOTE: only support a single per-vertex color attribute
        if (assimp_mesh->HasVertexColors(0)) {
            for (size_t cidx = 0; cidx < assimp_mesh->mNumVertices; ++cidx) {
                auto& c = assimp_mesh->mColors[0][cidx];
                mesh.addVertexColor({c.r, c.g, c.b});
            }
        }

        // Adjust face indices to index into combined mesh vertex array
        current_vidx += assimp_mesh->mNumVertices;
    }

    if (scene->mNumMaterials > 1) {
        utility::LogWarning(
                "{} has {} materials but only a single material per object is "
                "currently supported",
                filename, scene->mNumMaterials);
    }

    // Load material data
    auto* mat = scene->mMaterials[0];

    // create material structure to match this name
    auto& mesh_material = mesh.materials_[std::string(mat->GetName().C_Str())];

    using MaterialParameter =
            ccMesh::Material::MaterialParameter;

    // Retrieve base material properties
    aiColor3D color(1.f, 1.f, 1.f);

    mat->Get(AI_MATKEY_COLOR_DIFFUSE, color);
    mesh_material.baseColor =
            MaterialParameter::CreateRGB(color.r, color.g, color.b);
    mat->Get(AI_MATKEY_GLTF_PBRMETALLICROUGHNESS_METALLIC_FACTOR,
             mesh_material.baseMetallic);
    mat->Get(AI_MATKEY_GLTF_PBRMETALLICROUGHNESS_ROUGHNESS_FACTOR,
             mesh_material.baseRoughness);
    // NOTE: We prefer sheen to reflectivity so the following code works since
    // if sheen is not present it won't modify baseReflectance
    mat->Get(AI_MATKEY_REFLECTIVITY, mesh_material.baseReflectance);
    mat->Get(AI_MATKEY_SHEEN, mesh_material.baseReflectance);

    mat->Get(AI_MATKEY_CLEARCOAT_THICKNESS, mesh_material.baseClearCoat);
    mat->Get(AI_MATKEY_CLEARCOAT_ROUGHNESS,
             mesh_material.baseClearCoatRoughness);
    mat->Get(AI_MATKEY_ANISOTROPY, mesh_material.baseAnisotropy);

    // Retrieve textures
    TextureImages maps;
    LoadTextures(filename, mat, maps);
    mesh_material.albedo = maps.albedo;
    mesh_material.normalMap = maps.normal;
    mesh_material.ambientOcclusion = maps.ao;
    mesh_material.metallic = maps.metallic;
    mesh_material.roughness = maps.roughness;
    mesh_material.reflectance = maps.reflectance;

    return true;
}

bool ReadModelUsingAssimp(const std::string& filename,
                          visualization::rendering::TriangleMeshModel& model,
                          bool print_progress) {
    Assimp::Importer importer;
    const auto* scene = importer.ReadFile(filename.c_str(), kPostProcessFlags);
    if (!scene) {
        utility::LogWarning("Unable to load file {} with ASSIMP", filename);
        return false;
    }

    // Process each Assimp mesh into a ccMesh
    for (size_t midx = 0; midx < scene->mNumMeshes; ++midx) {
        const auto* assimp_mesh = scene->mMeshes[midx];
        // Only process triangle meshes
        if (assimp_mesh->mPrimitiveTypes != aiPrimitiveType_TRIANGLE) {
            utility::LogInfo(
                    "Skipping non-triangle primitive geometry of type: "
                    "{}",
                    assimp_mesh->mPrimitiveTypes);
            continue;
        }

        ccPointCloud* baseVertices = new ccPointCloud("vertices");
        assert(baseVertices);
        auto mesh = std::make_shared<ccMesh>(baseVertices);

        if(!baseVertices->reserveThePointsTable(assimp_mesh->mNumVertices)) {
            utility::LogError("[reserveThePointsTable] Not have enough memory! ");
            return false;
        }
        if (assimp_mesh->mNormals) {
            if (!baseVertices->reserveTheNormsTable()) {
                utility::LogError("[reserveTheNormsTable] Not have enough memory! ");
                return false;
            }
        }
        if (assimp_mesh->HasVertexColors(0)) {
            if (!baseVertices->reserveTheRGBTable()) {
                utility::LogError(
                        "[reserveTheRGBTable] Not have enough memory! ");
                return false;
            }
        }

        // copy vertex data
        for (size_t vidx = 0; vidx < assimp_mesh->mNumVertices; ++vidx) {
            auto& vertex = assimp_mesh->mVertices[vidx];
            baseVertices->addEigenPoint( { vertex.x, vertex.y, vertex.z } );
        }

        // copy face indices data
        for (size_t fidx = 0; fidx < assimp_mesh->mNumFaces; ++fidx) {
            auto& face = assimp_mesh->mFaces[fidx];
            Eigen::Vector3i facet(face.mIndices[0], face.mIndices[1],
                                  face.mIndices[2]);
            mesh->addTriangle(facet);
        }

        if (assimp_mesh->mNormals) {
            for (size_t nidx = 0; nidx < assimp_mesh->mNumVertices; ++nidx) {
                auto& normal = assimp_mesh->mNormals[nidx];
                baseVertices->addEigenNorm({normal.x, normal.y, normal.z});
            }
        }

        // NOTE: only use the first UV channel
        if (assimp_mesh->HasTextureCoords(0)) {
            for (size_t fidx = 0; fidx < assimp_mesh->mNumFaces; ++fidx) {
                auto& face = assimp_mesh->mFaces[fidx];
                auto& uv1 = assimp_mesh->mTextureCoords[0][face.mIndices[0]];
                auto& uv2 = assimp_mesh->mTextureCoords[0][face.mIndices[1]];
                auto& uv3 = assimp_mesh->mTextureCoords[0][face.mIndices[2]];
                mesh->triangle_uvs_.push_back(Eigen::Vector2d(uv1.x, uv1.y));
                mesh->triangle_uvs_.push_back(Eigen::Vector2d(uv2.x, uv2.y));
                mesh->triangle_uvs_.push_back(Eigen::Vector2d(uv3.x, uv3.y));
            }
        }

        // NOTE: only use the first color attribute
        if (assimp_mesh->HasVertexColors(0)) {
            for (size_t cidx = 0; cidx < assimp_mesh->mNumVertices; ++cidx) {
                auto& c = assimp_mesh->mColors[0][cidx];
                baseVertices->addEigenColor({c.r, c.g, c.b});
            }
        }

        //do some cleaning
        {
            baseVertices->shrinkToFit();
            mesh->shrinkToFit();
            NormsIndexesTableType* normals = mesh->getTriNormsTable();
            if (normals)
            {
                normals->shrink_to_fit();
            }
        }

        baseVertices->setEnabled(false);
        // DGM: no need to lock it as it is only used by one mesh!
        baseVertices->setLocked(false);
        mesh->addChild(baseVertices);

        // Add the mesh to the model
        model.meshes_.push_back({mesh, std::string(assimp_mesh->mName.C_Str()),
                                 assimp_mesh->mMaterialIndex});
    }

    // Load materials
    for (size_t i = 0; i < scene->mNumMaterials; ++i) {
        auto* mat = scene->mMaterials[i];

        visualization::rendering::Material cv3d_mat;

        cv3d_mat.name = mat->GetName().C_Str();

        // Retrieve base material properties
        aiColor3D color(1.f, 1.f, 1.f);

        mat->Get(AI_MATKEY_COLOR_DIFFUSE, color);
        cv3d_mat.base_color = Eigen::Vector4f(color.r, color.g, color.b, 1.f);
        mat->Get(AI_MATKEY_GLTF_PBRMETALLICROUGHNESS_METALLIC_FACTOR,
                 cv3d_mat.base_metallic);
        mat->Get(AI_MATKEY_GLTF_PBRMETALLICROUGHNESS_ROUGHNESS_FACTOR,
                 cv3d_mat.base_roughness);
        mat->Get(AI_MATKEY_REFLECTIVITY, cv3d_mat.base_reflectance);
        mat->Get(AI_MATKEY_SHEEN, cv3d_mat.base_reflectance);

        mat->Get(AI_MATKEY_CLEARCOAT_THICKNESS, cv3d_mat.base_clearcoat);
        mat->Get(AI_MATKEY_CLEARCOAT_ROUGHNESS,
                 cv3d_mat.base_clearcoat_roughness);
        mat->Get(AI_MATKEY_ANISOTROPY, cv3d_mat.base_anisotropy);
        aiString alpha_mode;
        mat->Get(AI_MATKEY_GLTF_ALPHAMODE, alpha_mode);
        std::string alpha_mode_str(alpha_mode.C_Str());
        if (alpha_mode_str == "BLEND" || alpha_mode_str == "MASK") {
            cv3d_mat.has_alpha = true;
        }

        // Retrieve textures
        TextureImages maps;
        LoadTextures(filename, mat, maps);
        cv3d_mat.albedo_img = maps.albedo;
        cv3d_mat.normal_img = maps.normal;
        cv3d_mat.ao_img = maps.ao;
        cv3d_mat.metallic_img = maps.metallic;
        cv3d_mat.roughness_img = maps.roughness;
        cv3d_mat.reflectance_img = maps.reflectance;
        cv3d_mat.ao_rough_metal_img = maps.gltf_rough_metal;

        if (cv3d_mat.has_alpha) {
            cv3d_mat.shader = "defaultLitTransparency";
        } else {
            cv3d_mat.shader = "defaultLit";
        }

        model.materials_.push_back(cv3d_mat);
    }

    return true;
}

}  // namespace io
}  // namespace cloudViewer
