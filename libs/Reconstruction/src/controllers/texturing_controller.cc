// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "controllers/texturing_controller.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <locale>

#include "mvs/model.h"
#include "mvs/texture_mapping.h"
#include "util/logging.h"
#include "util/misc.h"
#include "util/ply.h"

// CV_IO_LIB
#include <AutoIO.h>
#include <ecvHObjectCaster.h>
#include <ecvMesh.h>
#include <ecvPointCloud.h>

namespace colmap {
namespace {

PlyMesh ConvertToPlyMesh(ccMesh* mesh) {
    PlyMesh result;
    ccPointCloud* cloud =
            ccHObjectCaster::ToPointCloud(mesh->getAssociatedCloud());
    CHECK_NOTNULL(cloud);

    result.vertices.reserve(cloud->size());
    for (unsigned i = 0; i < cloud->size(); ++i) {
        const CCVector3* point = cloud->getPoint(i);
        if (cloud->isShifted()) {
            const CCVector3d global = cloud->toGlobal3d(*point);
            result.vertices.emplace_back(static_cast<float>(global.x),
                                         static_cast<float>(global.y),
                                         static_cast<float>(global.z));
        } else {
            result.vertices.emplace_back(point->x, point->y, point->z);
        }
    }

    result.faces.reserve(mesh->size());
    for (unsigned i = 0; i < mesh->size(); ++i) {
        const cloudViewer::VerticesIndexes* triangle =
                mesh->getTriangleVertIndexes(i);
        result.faces.emplace_back(triangle->i1, triangle->i2, triangle->i3);
    }
    return result;
}

std::vector<Eigen::Vector3f> ComputeVertexNormals(const PlyMesh& mesh) {
    std::vector<Eigen::Vector3f> normals(mesh.vertices.size(),
                                         Eigen::Vector3f::Zero());
    for (const PlyMeshFace& face : mesh.faces) {
        const PlyMeshVertex& av = mesh.vertices[face.vertex_idx1];
        const PlyMeshVertex& bv = mesh.vertices[face.vertex_idx2];
        const PlyMeshVertex& cv = mesh.vertices[face.vertex_idx3];
        const Eigen::Vector3f a(av.x, av.y, av.z);
        const Eigen::Vector3f b(bv.x, bv.y, bv.z);
        const Eigen::Vector3f c(cv.x, cv.y, cv.z);
        const Eigen::Vector3f normal = (b - a).cross(c - a);
        normals[face.vertex_idx1] += normal;
        normals[face.vertex_idx2] += normal;
        normals[face.vertex_idx3] += normal;
    }
    for (Eigen::Vector3f& normal : normals) {
        if (normal.squaredNorm() > 1e-20f) {
            normal.normalize();
        } else {
            normal = Eigen::Vector3f::UnitZ();
        }
    }
    return normals;
}

bool WriteObjMaterial(const std::string& obj_path,
                      const std::string& texture_filename,
                      const PlyMesh& mesh,
                      const std::vector<float>& face_uvs) {
    CHECK_EQ(face_uvs.size(), mesh.faces.size() * 6);

    std::string prefix;
    std::string extension;
    SplitFileExtension(obj_path, &prefix, &extension);
    const std::string mtl_path = prefix + ".mtl";
    const std::string mtl_filename = GetPathBaseName(mtl_path);

    std::ofstream mtl(mtl_path);
    if (!mtl.is_open()) {
        return false;
    }
    mtl << "newmtl material0000\n"
        << "Ka 1.000000 1.000000 1.000000\n"
        << "Kd 1.000000 1.000000 1.000000\n"
        << "Ks 0.000000 0.000000 0.000000\n"
        << "d 1.000000\n"
        << "illum 1\n"
        << "map_Kd " << texture_filename << "\n";
    mtl.close();

    std::ofstream obj(obj_path);
    if (!obj.is_open()) {
        return false;
    }
    obj.imbue(std::locale::classic());
    obj << std::setprecision(9);
    obj << "mtllib " << mtl_filename << "\n";
    for (const PlyMeshVertex& vertex : mesh.vertices) {
        obj << "v " << vertex.x << " " << vertex.y << " " << vertex.z
            << "\n";
    }
    for (size_t i = 0; i < mesh.faces.size(); ++i) {
        for (size_t corner = 0; corner < 3; ++corner) {
            obj << "vt " << face_uvs[i * 6 + corner * 2] << " "
                << face_uvs[i * 6 + corner * 2 + 1] << "\n";
        }
    }
    const std::vector<Eigen::Vector3f> normals = ComputeVertexNormals(mesh);
    for (const Eigen::Vector3f& normal : normals) {
        obj << "vn " << normal.x() << " " << normal.y() << " "
            << normal.z() << "\n";
    }
    obj << "usemtl material0000\n";
    for (size_t i = 0; i < mesh.faces.size(); ++i) {
        const PlyMeshFace& face = mesh.faces[i];
        const size_t texture_index = i * 3 + 1;
        obj << "f " << face.vertex_idx1 + 1 << "/" << texture_index << "/"
            << face.vertex_idx1 + 1 << " " << face.vertex_idx2 + 1 << "/"
            << texture_index + 1 << "/" << face.vertex_idx2 + 1 << " "
            << face.vertex_idx3 + 1 << "/" << texture_index + 2 << "/"
            << face.vertex_idx3 + 1 << "\n";
    }
    return obj.good();
}

}  // namespace

TexturingReconstruction::TexturingReconstruction(
        const TexturingOptions& options,
        const std::string& output_path)
    : options_(options), output_path_(output_path) {}

void TexturingReconstruction::Run() {
    success_.store(false);
    PrintHeading1("Mesh Texturing");

    if (options_.verbose) {
        options_.Print();
    }
    if (!options_.Check()) {
        LOG(ERROR) << "Invalid mesh texturing options";
        return;
    }

    ccMesh mesh;
    if (options_.meshed_file_path.empty() ||
        !ExistsFile(options_.meshed_file_path) || !mesh.CreateInternalCloud()) {
        LOG(ERROR) << "Invalid input mesh: " << options_.meshed_file_path;
        return;
    }
    cloudViewer::io::ReadTriangleMeshOptions mesh_options;
    mesh_options.print_progress = false;
    if (!cloudViewer::io::AutoReadMesh(options_.meshed_file_path, mesh,
                                       mesh_options)) {
        LOG(ERROR) << "Failed to load mesh: " << options_.meshed_file_path;
        return;
    }
    const PlyMesh ply_mesh = ConvertToPlyMesh(&mesh);
    LOG(INFO) << "Loaded mesh: " << ply_mesh.vertices.size() << " vertices, "
              << ply_mesh.faces.size() << " faces";

    mvs::Model model;
    model.ReadFromCOLMAP(output_path_);
    LOG(INFO) << "Loading " << model.images.size()
              << " undistorted workspace images";
    for (mvs::Image& image : model.images) {
        if (IsStopped()) {
            return;
        }
        Bitmap bitmap;
        if (!bitmap.Read(image.GetPath(), /*as_rgb=*/true)) {
            LOG(ERROR) << "Failed to read image: " << image.GetPath();
            return;
        }
        if (bitmap.Width() != static_cast<int>(image.GetWidth()) ||
            bitmap.Height() != static_cast<int>(image.GetHeight())) {
            bitmap.Rescale(static_cast<int>(image.GetWidth()),
                           static_cast<int>(image.GetHeight()));
        }
        image.SetBitmap(bitmap);
    }

    mvs::MeshTextureMappingOptions mapping_options;
    mapping_options.min_cos_normal_angle = options_.min_cos_normal_angle;
    mapping_options.min_visible_vertices = options_.min_visible_vertices;
    mapping_options.view_selection_smoothing_iterations =
            options_.view_selection_smoothing_iterations;
    mapping_options.atlas_patch_padding = options_.atlas_patch_padding;
    mapping_options.inpaint_radius = options_.inpaint_radius;
    mapping_options.apply_color_correction =
            options_.apply_color_correction;
    mapping_options.color_correction_regularization =
            options_.color_correction_regularization;
    mapping_options.num_threads = options_.num_threads;
    mapping_options.texture_scale_factor = options_.texture_scale_factor;

    const mvs::MeshTextureMappingResult result =
            mvs::MeshTextureMapping(ply_mesh, model.images, mapping_options);
    if (result.texture_atlas.Data() == nullptr || result.face_uvs.empty()) {
        LOG(ERROR) << "COLMAP mesh texture mapping produced no atlas";
        return;
    }

    std::string output_prefix;
    std::string output_extension;
    SplitFileExtension(options_.textured_file_path.string(), &output_prefix,
                       &output_extension);
    const std::string output_dir = GetParentDir(output_prefix);
    if (!output_dir.empty()) {
        CreateDirIfNotExists(output_dir);
    }
    const std::string base_name = GetPathBaseName(output_prefix);
    const std::string texture_filename =
            base_name + "_material0000_map_Kd.png";
    const std::string texture_path = JoinPaths(output_dir, texture_filename);
    if (!result.texture_atlas.Write(texture_path)) {
        LOG(ERROR) << "Failed to write texture atlas: " << texture_path;
        return;
    }
    if (!WriteObjMaterial(options_.textured_file_path, texture_filename,
                          ply_mesh, result.face_uvs)) {
        LOG(ERROR) << "Failed to write OBJ/MTL: "
                   << options_.textured_file_path;
        return;
    }

    PlyTexturedMesh textured_ply;
    textured_ply.mesh = ply_mesh;
    textured_ply.face_uvs = result.face_uvs;
    textured_ply.texture_file = texture_filename;
    const std::string ply_path = output_prefix + ".ply";
    WriteBinaryPlyMesh(ply_path, textured_ply);

    const size_t assigned_faces = std::count_if(
            result.face_view_ids.begin(), result.face_view_ids.end(),
            [](int view_id) { return view_id >= 0; });
    LOG(INFO) << "Textured " << assigned_faces << " / "
              << ply_mesh.faces.size() << " faces; atlas "
              << result.atlas_width << " x " << result.atlas_height;
    LOG(INFO) << "Wrote textured mesh: " << options_.textured_file_path
              << " and " << ply_path;
    success_.store(true);
    GetTimer().PrintMinutes();
}

bool TexturingOptions::Check() const {
    CHECK_GT(min_cos_normal_angle, 0.0f);
    CHECK_LE(min_cos_normal_angle, 1.0f);
    CHECK_GE(min_visible_vertices, 1);
    CHECK_LE(min_visible_vertices, 3);
    CHECK_GE(view_selection_smoothing_iterations, 0);
    CHECK_GE(atlas_patch_padding, 0);
    CHECK_GE(inpaint_radius, 0);
    CHECK_GT(color_correction_regularization, 0.0);
    CHECK_GT(texture_scale_factor, 0.0);
    return true;
}

void TexturingOptions::Print() const {
#define PrintOption(option) \
    std::cout << "  " << #option ": " << option << std::endl
    PrintHeading2("TexturingOptions");
    PrintOption(verbose);
    PrintOption(meshed_file_path);
    PrintOption(textured_file_path);
    PrintOption(min_cos_normal_angle);
    PrintOption(min_visible_vertices);
    PrintOption(view_selection_smoothing_iterations);
    PrintOption(atlas_patch_padding);
    PrintOption(inpaint_radius);
    PrintOption(apply_color_correction);
    PrintOption(color_correction_regularization);
    PrintOption(num_threads);
    PrintOption(texture_scale_factor);
    PrintOption(mesh_source);
#undef PrintOption
}

}  // namespace colmap
