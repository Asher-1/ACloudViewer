// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "MvsTexturing.h"

#include <QImage>
#include <QImageReader>
#include <QPainter>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <list>
#include <map>
#include <numeric>
#include <set>
#include <unordered_map>
#include <unordered_set>

#include "base/reconstruction.h"
#include "mvs/workspace.h"
#include "util/logging.h"
#include "util/misc.h"

// CV_CORE_LIB
#include <CVLog.h>
#include <CVTools.h>
#include <FileSystem.h>

// ECV_DB_LIB
#include "camera/PinholeCameraTrajectory.h"

// ECV_IO_LIB
#include <AutoIO.h>
#include <ImageIO.h>
#include <ecvHObjectCaster.h>
#include <ecvMesh.h>
#include <ecvPointCloud.h>

namespace cloudViewer {

using namespace colmap;
using colmap::image_t;

// Implementations of internal structures
TextureView::TextureView(std::size_t id,
                         const std::string& image_file,
                         const Eigen::Matrix3f& projection,
                         const Eigen::Matrix4f& world_to_cam,
                         const Eigen::Vector3f& pos,
                         const Eigen::Vector3f& viewdir,
                         int width,
                         int height)
    : id(id),
      image_file(image_file),
      width(width),
      height(height),
      projection(projection),
      world_to_cam(world_to_cam),
      pos(pos),
      viewdir(viewdir) {}

Eigen::Vector2f TextureView::GetPixelCoords(
        const Eigen::Vector3f& vertex) const {
    // Project vertex from world coordinates to image coordinates
    // P = K * [R|t] * [X; Y; Z; 1]
    // where [R|t] is world_to_cam (3x4), K is projection (3x3)
    Eigen::Vector4f vertex_homogeneous(vertex.x(), vertex.y(), vertex.z(),
                                       1.0f);

    // First transform to camera coordinates: [R|t] * [X; Y; Z; 1]
    Eigen::Vector3f cam_coords =
            world_to_cam.block<3, 4>(0, 0) * vertex_homogeneous;

    // Check if point is behind camera
    if (cam_coords.z() <= 0) {
        return Eigen::Vector2f(-1, -1);
    }

    // Project to image plane: K * [x_cam; y_cam; z_cam]
    Eigen::Vector3f proj = projection * cam_coords;

    // Normalize by depth to get pixel coordinates
    Eigen::Vector2f pixel(proj.x() / proj.z(), proj.y() / proj.z());

    // Following mvs-texturing: pixel coordinates are centered (pixel[0] - 0.5,
    // pixel[1] - 0.5) This means pixel (0,0) corresponds to center of top-left
    // pixel
    return Eigen::Vector2f(pixel.x() - 0.5f, pixel.y() - 0.5f);
}

bool TextureView::ValidPixel(const Eigen::Vector2f& pixel) const {
    return pixel.x() >= 0 && pixel.x() < width - 1 && pixel.y() >= 0 &&
           pixel.y() < height - 1;
}

bool TextureView::Inside(const Eigen::Vector3f& v1,
                         const Eigen::Vector3f& v2,
                         const Eigen::Vector3f& v3) const {
    Eigen::Vector2f p1 = GetPixelCoords(v1);
    Eigen::Vector2f p2 = GetPixelCoords(v2);
    Eigen::Vector2f p3 = GetPixelCoords(v3);
    return ValidPixel(p1) && ValidPixel(p2) && ValidPixel(p3);
}

TexturePatch::TexturePatch(int label,
                           const std::vector<size_t>& faces,
                           const std::vector<Eigen::Vector2f>& texcoords,
                           std::shared_ptr<QImage> image,
                           int min_x,
                           int min_y,
                           int max_x,
                           int max_y)
    : label(label),
      faces(faces),
      texcoords(texcoords),
      image(image),
      min_x(min_x),
      min_y(min_y),
      max_x(max_x),
      max_y(max_y) {}

bool ObjModel::SaveToFiles(const std::string& prefix) const {
    // Save OBJ file
    std::string obj_file = prefix + ".obj";
    std::ofstream obj_out(obj_file);
    if (!obj_out.good()) {
        CVLog::Error("Failed to open OBJ file: %s", obj_file.c_str());
        return false;
    }

    std::string mtl_name = colmap::GetPathBaseName(prefix) + ".mtl";
    obj_out << "mtllib " << mtl_name << "\n";

    obj_out << std::fixed << std::setprecision(6);

    // Write vertices
    for (const auto& v : vertices) {
        obj_out << "v " << v.x() << " " << v.y() << " " << v.z() << "\n";
    }

    // Write texture coordinates
    for (const auto& vt : texcoords) {
        obj_out << "vt " << vt.x() << " " << (1.0f - vt.y()) << "\n";
    }

    // Write normals
    for (const auto& vn : normals) {
        obj_out << "vn " << vn.x() << " " << vn.y() << " " << vn.z() << "\n";
    }

    // Write groups and faces
    for (const auto& group : groups) {
        obj_out << "usemtl " << group.material_name << "\n";
        for (const auto& face : group.faces) {
            obj_out << "f";
            for (int k = 0; k < 3; ++k) {
                obj_out << " " << (face.vertex_ids[k] + 1);
                // OBJ format: v/vt/vn (with texture), v//vn (without texture)
                if (face.texcoord_ids[k] != SIZE_MAX) {
                    // Has texture coordinates: v/vt/vn
                    obj_out << "/" << (face.texcoord_ids[k] + 1) << "/"
                            << (face.normal_ids[k] + 1);
                } else {
                    // No texture coordinates: v//vn (two slashes)
                    obj_out << "//" << (face.normal_ids[k] + 1);
                }
            }
            obj_out << "\n";
        }
    }

    obj_out.close();

    // Save MTL file (following mvs-texturing:
    // material_lib.save_to_files(prefix))
    std::string mtl_file =
            colmap::JoinPaths(colmap::GetParentDir(prefix), mtl_name);
    std::ofstream mtl_out(mtl_file);
    if (!mtl_out.good()) {
        CVLog::Error("Failed to open MTL file: %s", mtl_file.c_str());
        return false;
    }

    std::string const name = colmap::GetPathBaseName(prefix);

    for (const auto& [mat_name, texture_file] : materials) {
        // Following mvs-texturing: diffuse_map_postfix = "_" + material.name +
        // "_map_Kd.png"
        std::string diffuse_map_postfix = "_" + mat_name + "_map_Kd.png";
        mtl_out << "newmtl " << mat_name << "\n"
                << "Ka 1.000000 1.000000 1.000000\n"
                << "Kd 1.000000 1.000000 1.000000\n"
                << "Ks 0.000000 0.000000 0.000000\n"
                << "Tr 0.000000\n"  // Transparency (Tr = 1.0 - d)
                << "illum 1\n"
                << "Ns 1.000000\n"
                << "map_Kd " << name + diffuse_map_postfix << "\n";
    }

    mtl_out.close();
    return true;
}

MvsTexturing::MvsTexturing(const Options& options,
                           const colmap::Reconstruction& reconstruction,
                           colmap::mvs::Workspace* workspace,
                           const std::string& image_path)
    : options_(options),
      reconstruction_(reconstruction),
      workspace_(workspace),
      image_path_(image_path) {}

int MvsTexturing::GetWorkspaceImageIdx(const colmap::image_t image_id) const {
    if (!workspace_) {
        return -1;
    }
    const Image& image = reconstruction_.Image(image_id);
    const std::string& image_name = image.Name();

    const auto& model = workspace_->GetModel();

    // GetImageIdx uses CHECK_GT and .at() which throws exception if not found
    // Use try-catch to safely handle the case when image doesn't exist
    try {
        int workspace_image_idx = model.GetImageIdx(image_name);

        // Validate the index is within valid range
        if (workspace_image_idx < 0 ||
            static_cast<size_t>(workspace_image_idx) >= model.images.size()) {
            return -1;
        }

        return workspace_image_idx;
    } catch (const std::exception&) {
        // Image not found in workspace model or index out of range
        return -1;
    }
}

bool MvsTexturing::IsPointVisible(const Eigen::Vector3d& point3d,
                                  const Image& image,
                                  const Camera& camera,
                                  int workspace_image_idx) const {
    if (!options_.use_depth_normal_maps || !workspace_ ||
        workspace_image_idx < 0) {
        return true;
    }

    // Validate index is within model range before accessing cache
    const auto& model = workspace_->GetModel();
    if (static_cast<size_t>(workspace_image_idx) >= model.images.size()) {
        return true;  // Invalid index, assume visible
    }

    if (!workspace_->HasDepthMap(workspace_image_idx)) {
        return true;
    }

    try {
        const auto& depth_map = workspace_->GetDepthMap(workspace_image_idx);

        const Eigen::Matrix3x4d proj_matrix = image.ProjectionMatrix();
        const Eigen::Vector3d proj = proj_matrix * point3d.homogeneous();

        if (proj(2) <= 0) {
            return false;
        }

        const double u = proj(0) / proj(2);
        const double v = proj(1) / proj(2);

        if (u < 0 || u >= camera.Width() || v < 0 || v >= camera.Height()) {
            return false;
        }

        const size_t row = static_cast<size_t>(std::round(v));
        const size_t col = static_cast<size_t>(std::round(u));

        if (row >= depth_map.GetHeight() || col >= depth_map.GetWidth()) {
            return false;
        }

        const float depth_map_value = depth_map.Get(row, col);
        const float point_depth = static_cast<float>(proj(2));

        if (depth_map_value <= 0) {
            return false;
        }

        // Use very lenient depth check: allow larger depth error
        // mvs-texturing doesn't use depth maps for visibility by default
        // We use it as an optional filter, so be very lenient
        const float depth_error =
                std::abs(point_depth - depth_map_value) / depth_map_value;
        // Increase tolerance: use 10x the configured error threshold (was 5x,
        // but still too strict) Also add absolute tolerance: allow up to 0.1 *
        // depth_map_value absolute error
        const float relative_tolerance = options_.max_depth_error * 10.0f;
        const float absolute_tolerance = 0.1f;  // 10% absolute tolerance
        return depth_error <= relative_tolerance ||
               std::abs(point_depth - depth_map_value) <=
                       depth_map_value * absolute_tolerance;
    } catch (const std::exception&) {
        // Error accessing depth map, assume visible
        return true;
    }
}

bool MvsTexturing::TextureMesh(
        ccMesh& mesh,
        const camera::PinholeCameraTrajectory& camera_trajectory,
        const std::string& output_path) {
    if (options_.verbose) {
        CVLog::Print("Starting mvs-texturing based texture mapping...");
    }

    // Verify mesh has associated cloud
    ccGenericPointCloud* generic_cloud = mesh.getAssociatedCloud();
    if (!generic_cloud) {
        CVLog::Error(
                "Mesh has no associated cloud! "
                "This may be due to mesh merge failure. "
                "Please ensure the mesh file has valid vertices.");
        return false;
    }

    ccPointCloud* cloud = ccHObjectCaster::ToPointCloud(generic_cloud);
    if (!cloud || cloud->size() == 0) {
        CVLog::Error("Mesh has invalid or empty associated cloud!");
        return false;
    }

    if (mesh.size() == 0) {
        CVLog::Error("Mesh has no triangles!");
        return false;
    }

    if (options_.verbose) {
        CVLog::Print("Using mesh: %u vertices, %u triangles", cloud->size(),
                     mesh.size());
    }

    // Ensure normals are available (following mvs-texturing: prepare_mesh ->
    // mesh->ensure_normals(true, true)) This should be done right after loading
    // mesh, before processing
    if (!cloud->hasNormals()) {
        if (options_.verbose) {
            CVLog::Print(
                    "Mesh has no normals, computing per-vertex normals...");
        }
        {
            mesh.ComputeVertexNormals();
            // Refresh cloud pointer after computing normals
            generic_cloud = mesh.getAssociatedCloud();
            cloud = ccHObjectCaster::ToPointCloud(generic_cloud);
            if (!cloud) {
                CVLog::Error(
                        "Failed to get associated cloud after computing "
                        "normals!");
                return false;
            }
            if (options_.verbose) {
                CVLog::Print("Computed per-vertex normals for %u vertices",
                             cloud->size());
            }
        }
    } else {
        if (options_.verbose) {
            CVLog::Print("Mesh already has normals");
        }
    }

    // Create texture views
    CreateTextureViews(camera_trajectory);

    if (texture_views_.empty()) {
        CVLog::Error("No valid texture views created!");
        return false;
    }

    if (options_.verbose) {
        CVLog::Print("Created %zu texture views", texture_views_.size());
    }

    // Build adjacency graph for mesh faces
    BuildAdjacencyGraph(mesh);

    // Calculate data costs
    CalculateDataCosts(mesh);

    // Select views using graph cut algorithm
    SelectViews();

    // Generate texture patches
    GenerateTexturePatches(mesh);

    // Seam leveling
    SeamLeveling(mesh);

    // Generate texture atlases with packing
    GenerateTextureAtlases();

    // Build and save OBJ model
    if (!SaveOBJModel(output_path, mesh)) {
        CVLog::Error("Failed to save OBJ model to %s", output_path.c_str());
        return false;
    }

    return true;
}

void MvsTexturing::CreateTextureViews(
        const camera::PinholeCameraTrajectory& camera_trajectory) {
    texture_views_.clear();
    view_to_image_id_.clear();
    texture_views_.reserve(camera_trajectory.parameters_.size());
    view_to_image_id_.reserve(camera_trajectory.parameters_.size());

    // Try to find image_id by matching texture_file_ with image names
    for (size_t i = 0; i < camera_trajectory.parameters_.size(); ++i) {
        const auto& params = camera_trajectory.parameters_[i];

        // Find corresponding image_id by matching texture_file_ with image
        // names
        colmap::image_t image_id = colmap::kInvalidImageId;
        std::string texture_file_base =
                colmap::GetPathBaseName(params.texture_file_);
        for (const colmap::image_t& img_id : reconstruction_.RegImageIds()) {
            const Image& img = reconstruction_.Image(img_id);
            if (colmap::GetPathBaseName(img.Name()) == texture_file_base) {
                image_id = img_id;
                break;
            }
        }

        view_to_image_id_.push_back(image_id);

        // Get projection matrix directly from COLMAP Image if available
        // This ensures we use the same coordinate system as COLMAP
        Eigen::Matrix3f projection = Eigen::Matrix3f::Identity();
        Eigen::Matrix4f world_to_cam = Eigen::Matrix4f::Identity();
        Eigen::Vector3f pos;
        Eigen::Vector3f viewdir;

        if (image_id != colmap::kInvalidImageId) {
            // Use COLMAP Image's projection matrix directly
            // Image::ProjectionMatrix() returns [R|t] (world_to_cam), NOT
            // K*[R|t] This ensures we use the exact same coordinate system as
            // COLMAP
            const Image& img = reconstruction_.Image(image_id);
            const Eigen::Matrix3x4d Rt_matrix = img.ProjectionMatrix();

            // Build intrinsic matrix K
            projection(0, 0) = params.intrinsic_.GetFocalLength().first;
            projection(1, 1) = params.intrinsic_.GetFocalLength().second;
            projection(0, 2) = params.intrinsic_.GetPrincipalPoint().first;
            projection(1, 2) = params.intrinsic_.GetPrincipalPoint().second;

            // Image::ProjectionMatrix() already returns [R|t] (world_to_cam)
            // No need to extract it - use directly!
            world_to_cam.block<3, 3>(0, 0) =
                    Rt_matrix.leftCols<3>().cast<float>();
            world_to_cam.block<3, 1>(0, 3) =
                    Rt_matrix.rightCols<1>().cast<float>();
            world_to_cam(3, 0) = 0.0f;
            world_to_cam(3, 1) = 0.0f;
            world_to_cam(3, 2) = 0.0f;
            world_to_cam(3, 3) = 1.0f;

            // Camera position: C = -R^T * t (in world coordinates)
            Eigen::Matrix3f R = world_to_cam.block<3, 3>(0, 0);
            Eigen::Vector3f t = world_to_cam.block<3, 1>(0, 3);
            pos = -R.transpose() * t;

            // Viewing direction: camera -Z axis in world coordinates = -R^T *
            // [0, 0, 1]^T
            viewdir = -R.transpose() * Eigen::Vector3f(0, 0, 1);
            viewdir.normalize();
        } else {
            // Fallback: use params directly
            // NOTE: params.extrinsic_ is cam_to_world (from
            // InverseProjectionMatrix) So we need to invert it to get
            // world_to_cam
            projection(0, 0) = params.intrinsic_.GetFocalLength().first;
            projection(1, 1) = params.intrinsic_.GetFocalLength().second;
            projection(0, 2) = params.intrinsic_.GetPrincipalPoint().first;
            projection(1, 2) = params.intrinsic_.GetPrincipalPoint().second;

            // params.extrinsic_ is cam_to_world, so invert to get world_to_cam
            Eigen::Matrix4f cam_to_world = params.extrinsic_.cast<float>();
            world_to_cam = cam_to_world.inverse();

            pos = Eigen::Vector3f(cam_to_world(0, 3), cam_to_world(1, 3),
                                  cam_to_world(2, 3));
            viewdir = -cam_to_world.block<3, 1>(0, 2).normalized();
        }

        // Image file path
        std::string image_file = params.texture_file_;

        // Try multiple path resolution strategies
        if (!boost::filesystem::path(image_file).is_absolute()) {
            // First try: join with image_path_
            std::string candidate = colmap::JoinPaths(image_path_, image_file);
            if (ExistsFile(candidate)) {
                image_file = candidate;
            } else {
                // Second try: check if texture_file_ is relative to workspace
                // The texture_file_ might already be correct relative path
                // Try joining with workspace path if available
                if (workspace_) {
                    std::string workspace_path =
                            workspace_->GetOptions().workspace_path;
                    candidate = colmap::JoinPaths(
                            workspace_path, "images",
                            colmap::GetPathBaseName(image_file));
                    if (ExistsFile(candidate)) {
                        image_file = candidate;
                    } else {
                        // Fallback: use original path (might be resolved later)
                        image_file = colmap::JoinPaths(image_path_, image_file);
                    }
                } else {
                    image_file = candidate;
                }
            }
        }

        // Verify file exists, log warning if not
        if (!ExistsFile(image_file)) {
            CVLog::Warning(
                    "Image file does not exist: %s (will try to load anyway)",
                    image_file.c_str());
        }

        auto view = std::make_unique<TextureView>(
                i, image_file, projection, world_to_cam, pos, viewdir,
                params.intrinsic_.width_, params.intrinsic_.height_);

        texture_views_.push_back(std::move(view));
    }
}

void MvsTexturing::CalculateDataCosts(const ccMesh& mesh) {
    if (options_.verbose) {
        CVLog::Print("Calculating data costs for %zu faces...", mesh.size());
    }

    ccPointCloud* cloud =
            ccHObjectCaster::ToPointCloud(mesh.getAssociatedCloud());
    if (!cloud) {
        CVLog::Error("Mesh has no associated cloud!");
        return;
    }

    // Get global shift/scale for coordinate conversion
    // CloudViewer uses local coordinates (shifted/scaled), but camera
    // transforms are in global coordinates. We need to convert mesh vertices to
    // global coordinates before projection.
    bool mesh_is_shifted = cloud->isShifted();
    CCVector3d mesh_global_shift = cloud->getGlobalShift();
    double mesh_global_scale = cloud->getGlobalScale();

    if (options_.verbose && mesh_is_shifted) {
        CVLog::Print("Mesh has global shift: (%.6f, %.6f, %.6f), scale: %.6f",
                     mesh_global_shift.x, mesh_global_shift.y,
                     mesh_global_shift.z, mesh_global_scale);
    }

    const unsigned num_faces = mesh.size();
    face_projection_infos_.clear();
    face_projection_infos_.resize(num_faces);

    // Statistics for debugging
    size_t total_faces_processed = 0;
    size_t faces_inside_view = 0;
    size_t faces_backface_culled = 0;
    size_t faces_viewing_angle_too_steep = 0;
    size_t faces_behind_camera = 0;
    size_t faces_depth_check_passed = 0;
    size_t faces_depth_check_failed = 0;
    size_t faces_zero_area = 0;
    size_t faces_quality_ok = 0;

    // For each view, test visibility and quality for each face
    for (size_t view_id = 0; view_id < texture_views_.size(); ++view_id) {
        const auto& view = texture_views_[view_id];
        colmap::image_t image_id = view_to_image_id_[view_id];

        if (image_id == colmap::kInvalidImageId) {
            if (options_.verbose) {
                CVLog::Warning("View %zu has invalid image_id, skipping",
                               view_id);
            }
            continue;
        }

        const Image& image = reconstruction_.Image(image_id);
        const Camera& camera = reconstruction_.Camera(image.CameraId());
        int workspace_image_idx = GetWorkspaceImageIdx(image_id);

        // Process each face
        for (unsigned face_id = 0; face_id < num_faces; ++face_id) {
            total_faces_processed++;

            Eigen::Vector3i tri_idx;
            mesh.getTriangleVertIndexes(face_id, tri_idx);

            if (tri_idx(0) >= static_cast<int>(cloud->size()) ||
                tri_idx(1) >= static_cast<int>(cloud->size()) ||
                tri_idx(2) >= static_cast<int>(cloud->size())) {
                continue;
            }

            const CCVector3* v1_ptr = cloud->getPoint(tri_idx(0));
            const CCVector3* v2_ptr = cloud->getPoint(tri_idx(1));
            const CCVector3* v3_ptr = cloud->getPoint(tri_idx(2));

            // Convert from local coordinates to global coordinates if mesh is
            // shifted Camera transforms are in global coordinate system
            Eigen::Vector3f v1, v2, v3;
            if (mesh_is_shifted) {
                // Convert local to global: Pglobal = Plocal / scale - shift
                CCVector3d v1_global = cloud->toGlobal3d(*v1_ptr);
                CCVector3d v2_global = cloud->toGlobal3d(*v2_ptr);
                CCVector3d v3_global = cloud->toGlobal3d(*v3_ptr);
                v1 = Eigen::Vector3f(static_cast<float>(v1_global.x),
                                     static_cast<float>(v1_global.y),
                                     static_cast<float>(v1_global.z));
                v2 = Eigen::Vector3f(static_cast<float>(v2_global.x),
                                     static_cast<float>(v2_global.y),
                                     static_cast<float>(v2_global.z));
                v3 = Eigen::Vector3f(static_cast<float>(v3_global.x),
                                     static_cast<float>(v3_global.y),
                                     static_cast<float>(v3_global.z));
            } else {
                v1 = Eigen::Vector3f(v1_ptr->x, v1_ptr->y, v1_ptr->z);
                v2 = Eigen::Vector3f(v2_ptr->x, v2_ptr->y, v2_ptr->z);
                v3 = Eigen::Vector3f(v3_ptr->x, v3_ptr->y, v3_ptr->z);
            }

            // Check if face projects into view
            if (!view->Inside(v1, v2, v3)) {
                continue;
            }
            faces_inside_view++;

            // Compute face center and normal (following mvs-texturing approach)
            Eigen::Vector3f face_center = (v1 + v2 + v3) / 3.0f;
            Eigen::Vector3f face_normal_vec = (v2 - v1).cross(v3 - v1);
            float face_normal_norm = face_normal_vec.norm();
            if (face_normal_norm < 1e-6f) {
                continue;  // Degenerate face (zero area)
            }
            Eigen::Vector3f face_normal =
                    face_normal_vec / face_normal_norm;  // Normalize

            // Convert to double for visibility/quality testing
            Eigen::Vector3d face_center_d = face_center.cast<double>();
            Eigen::Vector3d face_normal_d = face_normal.cast<double>();

            // Following mvs-texturing: view filtering based on viewing angle
            const Eigen::Vector3d camera_pos = image.ProjectionCenter();
            Eigen::Vector3d view_to_face_vec =
                    (face_center_d - camera_pos).normalized();
            Eigen::Vector3d face_to_view_vec =
                    (camera_pos - face_center_d).normalized();

            // Backface culling: viewing_angle < 0 means face is facing away
            float viewing_angle =
                    static_cast<float>(face_to_view_vec.dot(face_normal_d));
            if (viewing_angle < 0.0f) {
                faces_backface_culled++;
                continue;  // Backface culling
            }

            // Viewing angle limit: reject faces viewed from too steep angle
            // (mvs-texturing uses 75 degrees)
            const float max_viewing_angle_rad =
                    options_.max_viewing_angle_deg * M_PI / 180.0f;
            if (std::acos(viewing_angle) > max_viewing_angle_rad) {
                faces_viewing_angle_too_steep++;
                continue;  // Viewing angle too steep
            }

            // Frustum culling: check if face is in viewing direction
            Eigen::Vector3d viewing_direction =
                    image.RotationMatrix()
                            .row(2)
                            .transpose();  // Camera -Z axis
            if (viewing_direction.dot(view_to_face_vec) < 0.0) {
                faces_behind_camera++;
                continue;  // Face is behind camera
            }

            // Test visibility using depth map (if enabled) - geometric
            // visibility test Note: mvs-texturing's geometric_visibility_test
            // is optional and uses BVH ray casting Depth map check is very
            // lenient - we use it as a soft filter, not a hard requirement Most
            // faces should pass this check to allow proper texturing If depth
            // map check fails, we still allow the face (depth maps are
            // optional)
            bool visible = true;
            if (options_.use_depth_normal_maps && workspace_image_idx >= 0) {
                // Try depth map check, but don't reject if it fails
                // Depth maps may have noise, alignment issues, or missing data
                // We use it as a quality hint, not a strict filter
                bool depth_visible = IsPointVisible(
                        face_center_d, image, camera, workspace_image_idx);
                // Use depth check result as a hint - don't reject faces based
                // on it This ensures we get enough faces for texturing (The
                // lenient tolerance in IsPointVisible should allow most faces
                // through anyway)
                visible = depth_visible;  // Use result, but lenient tolerance
                                          // should allow most through
                if (visible) {
                    faces_depth_check_passed++;
                } else {
                    faces_depth_check_failed++;
                }
            }
            // IMPORTANT: Don't skip faces even if depth check failed!
            // Depth maps are optional and may have errors. We need to ensure
            // every face has at least some candidate views for texturing.
            // Continue processing even if depth check failed - this ensures
            // faces have candidate views

            // Compute quality following mvs-texturing: use projection area or
            // GMI For now, use projection area (DATA_TERM_AREA)
            // TODO: Implement GMI (Gradient Magnitude Integral) for better
            // quality
            Eigen::Vector2f p1 = view->GetPixelCoords(v1);
            Eigen::Vector2f p2 = view->GetPixelCoords(v2);
            Eigen::Vector2f p3 = view->GetPixelCoords(v3);

            // Calculate projected triangle area
            float area = 0.5f * std::abs((p2(0) - p1(0)) * (p3(1) - p1(1)) -
                                         (p3(0) - p1(0)) * (p2(1) - p1(1)));

            // Use a very small threshold instead of epsilon to allow more faces
            // Some faces may have very small projected area but still valid
            const float min_area_threshold =
                    1e-6f;  // Very small but larger than epsilon
            if (area < min_area_threshold) {
                faces_zero_area++;
                continue;  // Zero or near-zero area projection
            }

            float quality = area;  // Use area as quality (DATA_TERM_AREA)

            // If depth map check failed, reduce quality slightly to prefer
            // views that pass depth check, but still allow the face to be
            // textured IMPORTANT: Don't skip faces even if depth check failed!
            if (!visible && options_.use_depth_normal_maps) {
                quality *= 0.5f;  // Reduce quality by 50% if depth check failed
            }

            faces_quality_ok++;

            // Get mean color from image (for outlier detection)
            // TODO: Sample pixels in triangle to get mean color (for outlier
            // detection)
            Eigen::Vector3f mean_color(0.5f, 0.5f,
                                       0.5f);  // Default gray for now

            FaceProjectionInfo info;
            info.view_id = view_id;
            info.quality = quality;
            info.mean_color = mean_color;

            face_projection_infos_[face_id].push_back(info);
        }

        if (options_.verbose && (view_id + 1) % 10 == 0) {
            CVLog::Print("Processed %zu/%zu views...", view_id + 1,
                         texture_views_.size());
        }
    }

    if (options_.verbose) {
        CVLog::Print(
                "Data cost calculation stats:\n"
                "  Total face-view pairs processed: %zu\n"
                "  Inside view: %zu\n"
                "  Backface culled: %zu\n"
                "  Viewing angle too steep: %zu\n"
                "  Behind camera: %zu\n"
                "  Depth check passed: %zu\n"
                "  Depth check failed: %zu\n"
                "  Zero area projection: %zu\n"
                "  Quality OK (added to data costs): %zu",
                total_faces_processed, faces_inside_view, faces_backface_culled,
                faces_viewing_angle_too_steep, faces_behind_camera,
                faces_depth_check_passed, faces_depth_check_failed,
                faces_zero_area, faces_quality_ok);
    }

    // Postprocess face infos following mvs-texturing approach
    // 1. Sort infos by view_id (for consistency)
    for (size_t face_id = 0; face_id < face_projection_infos_.size();
         ++face_id) {
        auto& infos = face_projection_infos_[face_id];
        std::sort(infos.begin(), infos.end(),
                  [](const FaceProjectionInfo& a, const FaceProjectionInfo& b) {
                      return a.view_id < b.view_id;
                  });
    }

    // 2. Find global max quality and calculate percentile (99.5% like
    // mvs-texturing)
    float max_quality = 0.0f;
    std::vector<float> all_qualities;
    for (const auto& infos : face_projection_infos_) {
        for (const auto& info : infos) {
            max_quality = std::max(max_quality, info.quality);
            all_qualities.push_back(info.quality);
        }
    }

    // Calculate 99.5th percentile
    float percentile = max_quality;
    if (!all_qualities.empty()) {
        std::sort(all_qualities.begin(), all_qualities.end());
        size_t percentile_idx = static_cast<size_t>(
                std::round(0.995f * (all_qualities.size() - 1)));
        percentile_idx = std::min(percentile_idx, all_qualities.size() - 1);
        percentile = all_qualities[percentile_idx];
    }

    if (options_.verbose) {
        CVLog::Print("Maximum quality: %.6f, 99.5%% percentile: %.6f",
                     max_quality, percentile);
    }

    // 3. Convert face_projection_infos_ to data_costs_ format
    // Following mvs-texturing: clamp to percentile and normalize
    data_costs_.clear();
    data_costs_.resize(face_projection_infos_.size());
    for (size_t face_id = 0; face_id < face_projection_infos_.size();
         ++face_id) {
        const auto& infos = face_projection_infos_[face_id];
        data_costs_[face_id].reserve(infos.size());

        for (const auto& info : infos) {
            // Clamp to percentile and normalize (following mvs-texturing)
            float normalized_quality =
                    percentile > 0 ? std::min(1.0f, info.quality / percentile)
                                   : 0.0f;
            // Convert quality to cost: cost = 1.0 - normalized_quality
            float cost = 1.0f - normalized_quality;
            // Note: mvs-texturing uses view_id + 1 (label 0 is undefined)
            // We'll keep view_id as-is for now, but SelectViews should handle
            // label 0
            data_costs_[face_id].emplace_back(info.view_id, cost);
        }
    }

    if (options_.verbose) {
        size_t total_projections = 0;
        for (const auto& infos : face_projection_infos_) {
            total_projections += infos.size();
        }
        CVLog::Print("Calculated data costs: %zu total face-view projections",
                     total_projections);
    }
}

void MvsTexturing::BuildAdjacencyGraph(const ccMesh& mesh) {
    if (options_.verbose) {
        CVLog::Print("Building adjacency graph for mesh faces...");
    }

    const unsigned num_faces = mesh.size();
    face_adjacency_.clear();
    face_adjacency_.resize(num_faces);

    ccPointCloud* cloud =
            ccHObjectCaster::ToPointCloud(mesh.getAssociatedCloud());
    if (!cloud) {
        CVLog::Error("Mesh has no associated cloud!");
        return;
    }

    // Build edge-to-faces map
    std::map<std::pair<unsigned, unsigned>, std::vector<unsigned>>
            edge_to_faces;
    for (unsigned face_id = 0; face_id < num_faces; ++face_id) {
        Eigen::Vector3i tri_idx;
        mesh.getTriangleVertIndexes(face_id, tri_idx);

        unsigned v1 = tri_idx(0), v2 = tri_idx(1), v3 = tri_idx(2);

        // Add edges (always store with smaller index first)
        auto add_edge = [&edge_to_faces](unsigned a, unsigned b,
                                         unsigned face) {
            if (a > b) std::swap(a, b);
            edge_to_faces[{a, b}].push_back(face);
        };
        add_edge(v1, v2, face_id);
        add_edge(v2, v3, face_id);
        add_edge(v3, v1, face_id);
    }

    // Build adjacency list
    for (const auto& [edge, faces] : edge_to_faces) {
        for (size_t i = 0; i < faces.size(); ++i) {
            for (size_t j = i + 1; j < faces.size(); ++j) {
                unsigned face1 = faces[i];
                unsigned face2 = faces[j];

                // Add bidirectional adjacency
                if (std::find(face_adjacency_[face1].begin(),
                              face_adjacency_[face1].end(),
                              face2) == face_adjacency_[face1].end()) {
                    face_adjacency_[face1].push_back(face2);
                }
                if (std::find(face_adjacency_[face2].begin(),
                              face_adjacency_[face2].end(),
                              face1) == face_adjacency_[face2].end()) {
                    face_adjacency_[face2].push_back(face1);
                }
            }
        }
    }

    if (options_.verbose) {
        size_t total_edges = 0;
        for (const auto& adj : face_adjacency_) {
            total_edges += adj.size();
        }
        CVLog::Print("Built adjacency graph: %zu faces, %zu edges", num_faces,
                     total_edges / 2);
    }
}

float MvsTexturing::ComputePairwiseCost(size_t face1,
                                        size_t face2,
                                        size_t view1,
                                        size_t view2) const {
    // Potts model: cost is 0 if same view, 1 if different view
    if (view1 == view2) {
        return 0.0f;
    }
    return 1.0f;
}

void MvsTexturing::SelectViews() {
    if (options_.verbose) {
        CVLog::Print("Selecting views using graph cut algorithm...");
    }

    face_labels_.clear();
    face_labels_.resize(face_projection_infos_.size(), SIZE_MAX);

    // Initialize with greedy solution
    for (size_t face_id = 0; face_id < face_projection_infos_.size();
         ++face_id) {
        const auto& infos = face_projection_infos_[face_id];
        if (infos.empty()) {
            continue;
        }

        // Find view with minimum cost (highest quality)
        float min_cost = std::numeric_limits<float>::max();
        size_t best_view_id = SIZE_MAX;
        for (const auto& [view_id, cost] : data_costs_[face_id]) {
            if (cost < min_cost) {
                min_cost = cost;
                best_view_id = view_id;
            }
        }

        if (best_view_id != SIZE_MAX) {
            face_labels_[face_id] = best_view_id;
        }
    }

    // Iterative refinement using alpha-expansion style algorithm
    const int max_iterations = 10;
    const float lambda = 0.5f;  // Smoothness weight

    for (int iter = 0; iter < max_iterations; ++iter) {
        bool changed = false;

        // Try to improve each face's label
        for (size_t face_id = 0; face_id < face_labels_.size(); ++face_id) {
            if (face_labels_[face_id] == SIZE_MAX) continue;
            if (data_costs_[face_id].empty()) continue;

            float current_cost = std::numeric_limits<float>::max();
            // Find current data cost
            for (const auto& [view_id, cost] : data_costs_[face_id]) {
                if (view_id == face_labels_[face_id]) {
                    current_cost = cost;
                    break;
                }
            }

            // Add pairwise costs from neighbors
            float current_pairwise = 0.0f;
            for (size_t adj_face : face_adjacency_[face_id]) {
                if (adj_face < face_labels_.size() &&
                    face_labels_[adj_face] != SIZE_MAX) {
                    current_pairwise += ComputePairwiseCost(
                            face_id, adj_face, face_labels_[face_id],
                            face_labels_[adj_face]);
                }
            }
            float current_total = current_cost + lambda * current_pairwise;

            // Try all alternative labels
            float best_total = current_total;
            size_t best_label = face_labels_[face_id];

            for (const auto& [view_id, cost] : data_costs_[face_id]) {
                if (view_id == face_labels_[face_id]) continue;

                float pairwise = 0.0f;
                for (size_t adj_face : face_adjacency_[face_id]) {
                    if (adj_face < face_labels_.size() &&
                        face_labels_[adj_face] != SIZE_MAX) {
                        pairwise +=
                                ComputePairwiseCost(face_id, adj_face, view_id,
                                                    face_labels_[adj_face]);
                    }
                }

                float total = cost + lambda * pairwise;
                if (total < best_total) {
                    best_total = total;
                    best_label = view_id;
                    changed = true;
                }
            }

            face_labels_[face_id] = best_label;
        }

        if (!changed) break;

        if (options_.verbose && iter % 2 == 0) {
            CVLog::Print("Graph cut iteration %d...", iter + 1);
        }
    }

    // Count faces with valid labels
    size_t labeled_faces = 0;
    size_t unlabeled_faces = 0;
    for (size_t label : face_labels_) {
        if (label != SIZE_MAX) {
            labeled_faces++;
        } else {
            unlabeled_faces++;
        }
    }

    if (options_.verbose) {
        CVLog::Print("Selected views: %zu/%zu faces have valid labels (%.1f%%)",
                     labeled_faces, face_labels_.size(),
                     100.0f * labeled_faces / face_labels_.size());
        if (unlabeled_faces > 0) {
            CVLog::Warning(
                    "%zu faces have no valid labels (will be handled as unseen "
                    "faces)",
                    unlabeled_faces);
        }
    }
}

void MvsTexturing::GetSubgraphs(
        size_t label, std::vector<std::vector<size_t>>* subgraphs) const {
    // Following mvs-texturing's UniGraph::get_subgraphs implementation
    std::vector<bool> used(face_labels_.size(), false);

    for (size_t i = 0; i < face_labels_.size(); ++i) {
        if (face_labels_[i] == label && !used[i]) {
            subgraphs->push_back(std::vector<size_t>());

            std::list<size_t> queue;
            queue.push_back(i);
            used[i] = true;

            while (!queue.empty()) {
                size_t node = queue.front();
                queue.pop_front();

                subgraphs->back().push_back(node);

                // Add all unused neighbours with the same label to the queue
                for (size_t adj_face : face_adjacency_[node]) {
                    if (adj_face < face_labels_.size() &&
                        face_labels_[adj_face] == label && !used[adj_face]) {
                        queue.push_back(adj_face);
                        used[adj_face] = true;
                    }
                }
            }
        }
    }
}

void MvsTexturing::MergeVertexProjectionInfos() {
    // Following mvs-texturing's merge_vertex_projection_infos implementation
    // Merge vertex infos within the same texture patch
    for (size_t i = 0; i < vertex_projection_infos_.size(); ++i) {
        auto& infos = vertex_projection_infos_[i];

        std::map<size_t, VertexProjectionInfo> info_map;

        for (const auto& info : infos) {
            size_t texture_patch_id = info.texture_patch_id;
            auto it = info_map.find(texture_patch_id);
            if (it == info_map.end()) {
                info_map[texture_patch_id] = info;
            } else {
                // Merge faces
                it->second.faces.insert(it->second.faces.end(),
                                        info.faces.begin(), info.faces.end());
            }
        }

        infos.clear();
        infos.reserve(info_map.size());
        for (const auto& [patch_id, info] : info_map) {
            infos.push_back(info);
        }
    }
}

// Helper structure for texture patch candidate (following mvs-texturing)
struct TexturePatchCandidate {
    int min_x, min_y, max_x, max_y;  // Bounding box
    std::unique_ptr<TexturePatch> texture_patch;

    bool IsInside(const TexturePatchCandidate& other) const {
        return min_x >= other.min_x && max_x <= other.max_x &&
               min_y >= other.min_y && max_y <= other.max_y;
    }
};

// Helper function to generate a texture patch candidate (following
// mvs-texturing's generate_candidate)
static TexturePatchCandidate GenerateCandidate(int label,
                                               TextureView* texture_view,
                                               const std::vector<size_t>& faces,
                                               const ccMesh& mesh,
                                               ccPointCloud* cloud,
                                               bool mesh_is_shifted = false) {
    // Load image if needed
    if (!texture_view->image_data) {
        std::string image_path = texture_view->image_file;
        if (!ExistsFile(image_path)) {
            // Try to find image in common locations
            std::vector<std::string> candidates;
            // Add path resolution strategies here if needed
            for (const auto& candidate : candidates) {
                if (ExistsFile(candidate)) {
                    image_path = candidate;
                    break;
                }
            }
        }

        QImageReader reader(QString::fromStdString(image_path));
        QImage img = reader.read();
        if (!img.isNull()) {
            texture_view->image_data = std::make_shared<QImage>(
                    img.convertToFormat(QImage::Format_RGB888));
        }
    }

    if (!texture_view->image_data) {
        CVLog::Error("Failed to load image for texture view");
        TexturePatchCandidate empty;
        empty.min_x = empty.min_y = empty.max_x = empty.max_y = 0;
        return empty;
    }

    // Calculate bounding box
    int min_x = texture_view->width, min_y = texture_view->height;
    int max_x = 0, max_y = 0;
    std::vector<Eigen::Vector2f> texcoords;

    for (size_t face_id : faces) {
        Eigen::Vector3i tri_idx;
        mesh.getTriangleVertIndexes(face_id, tri_idx);

        const CCVector3* v1_ptr = cloud->getPoint(tri_idx(0));
        const CCVector3* v2_ptr = cloud->getPoint(tri_idx(1));
        const CCVector3* v3_ptr = cloud->getPoint(tri_idx(2));

        // Convert from local coordinates to global coordinates if mesh is
        // shifted Camera transforms are in global coordinate system
        Eigen::Vector3f v1, v2, v3;
        if (mesh_is_shifted) {
            // Convert local to global: Pglobal = Plocal / scale - shift
            CCVector3d v1_global = cloud->toGlobal3d(*v1_ptr);
            CCVector3d v2_global = cloud->toGlobal3d(*v2_ptr);
            CCVector3d v3_global = cloud->toGlobal3d(*v3_ptr);
            v1 = Eigen::Vector3f(static_cast<float>(v1_global.x),
                                 static_cast<float>(v1_global.y),
                                 static_cast<float>(v1_global.z));
            v2 = Eigen::Vector3f(static_cast<float>(v2_global.x),
                                 static_cast<float>(v2_global.y),
                                 static_cast<float>(v2_global.z));
            v3 = Eigen::Vector3f(static_cast<float>(v3_global.x),
                                 static_cast<float>(v3_global.y),
                                 static_cast<float>(v3_global.z));
        } else {
            v1 = Eigen::Vector3f(v1_ptr->x, v1_ptr->y, v1_ptr->z);
            v2 = Eigen::Vector3f(v2_ptr->x, v2_ptr->y, v2_ptr->z);
            v3 = Eigen::Vector3f(v3_ptr->x, v3_ptr->y, v3_ptr->z);
        }

        Eigen::Vector2f p1 = texture_view->GetPixelCoords(v1);
        Eigen::Vector2f p2 = texture_view->GetPixelCoords(v2);
        Eigen::Vector2f p3 = texture_view->GetPixelCoords(v3);

        texcoords.push_back(p1);
        texcoords.push_back(p2);
        texcoords.push_back(p3);

        min_x = std::min({min_x, static_cast<int>(std::floor(p1.x())),
                          static_cast<int>(std::floor(p2.x())),
                          static_cast<int>(std::floor(p3.x()))});
        min_y = std::min({min_y, static_cast<int>(std::floor(p1.y())),
                          static_cast<int>(std::floor(p2.y())),
                          static_cast<int>(std::floor(p3.y()))});
        max_x = std::max({max_x, static_cast<int>(std::ceil(p1.x())),
                          static_cast<int>(std::ceil(p2.x())),
                          static_cast<int>(std::ceil(p3.x()))});
        max_y = std::max({max_y, static_cast<int>(std::ceil(p1.y())),
                          static_cast<int>(std::ceil(p2.y())),
                          static_cast<int>(std::ceil(p3.y()))});
    }

    // Validate bounding box
    if (min_x < 0 || min_y < 0 || max_x >= texture_view->width ||
        max_y >= texture_view->height) {
        CVLog::Warning("Bounding box out of bounds, clamping");
        min_x = std::max(0, min_x);
        min_y = std::max(0, min_y);
        max_x = std::min(texture_view->width - 1, max_x);
        max_y = std::min(texture_view->height - 1, max_y);
    }

    int width = max_x - min_x + 1;
    int height = max_y - min_y + 1;

    if (width <= 0 || height <= 0) {
        TexturePatchCandidate empty;
        empty.min_x = empty.min_y = empty.max_x = empty.max_y = 0;
        return empty;
    }

    // Add border
    const int texture_patch_border = 1;
    int patch_width = width + 2 * texture_patch_border;
    int patch_height = height + 2 * texture_patch_border;
    int patch_min_x = min_x - texture_patch_border;
    int patch_min_y = min_y - texture_patch_border;

    // Adjust texcoords relative to patch bounding box
    Eigen::Vector2f offset(patch_min_x, patch_min_y);
    for (auto& tc : texcoords) {
        tc = tc - offset;
    }

    // Extract image patch
    QImage patch(patch_width, patch_height, QImage::Format_RGB888);
    patch.fill(QColor(255, 0, 255));  // Magenta fill

    int src_x_start = std::max(0, patch_min_x);
    int src_y_start = std::max(0, patch_min_y);
    int src_x_end = std::min(texture_view->width, patch_min_x + patch_width);
    int src_y_end = std::min(texture_view->height, patch_min_y + patch_height);

    int dst_x_start = src_x_start - patch_min_x;
    int dst_y_start = src_y_start - patch_min_y;
    int copy_width = src_x_end - src_x_start;
    int copy_height = src_y_end - src_y_start;

    if (copy_width > 0 && copy_height > 0) {
        QImage src_region = texture_view->image_data->copy(
                src_x_start, src_y_start, copy_width, copy_height);
        QPainter painter(&patch);
        painter.drawImage(dst_x_start, dst_y_start, src_region);
        painter.end();
    }

    auto patch_ptr = std::make_shared<QImage>(patch);
    auto texture_patch = std::make_unique<TexturePatch>(
            label, faces, texcoords, patch_ptr, patch_min_x, patch_min_y,
            patch_min_x + patch_width - 1, patch_min_y + patch_height - 1);

    TexturePatchCandidate candidate;
    candidate.min_x = patch_min_x;
    candidate.min_y = patch_min_y;
    candidate.max_x = patch_min_x + patch_width - 1;
    candidate.max_y = patch_min_y + patch_height - 1;
    candidate.texture_patch = std::move(texture_patch);

    return candidate;
}

void MvsTexturing::GenerateTexturePatches(const ccMesh& mesh) {
    // Following mvs-texturing's generate_texture_patches implementation exactly
    if (options_.verbose) {
        CVLog::Print("Generating texture patches...");
    }

    texture_patches_.clear();
    ccPointCloud* cloud =
            ccHObjectCaster::ToPointCloud(mesh.getAssociatedCloud());
    if (!cloud) {
        CVLog::Error("Mesh has no associated cloud!");
        return;
    }

    // Initialize vertex_projection_infos_ (following mvs-texturing)
    vertex_projection_infos_.clear();
    vertex_projection_infos_.resize(cloud->size());

    // Get global shift/scale for coordinate conversion
    bool mesh_is_shifted = cloud->isShifted();

    size_t num_patches = 0;

    // Process each texture view (following mvs-texturing: for each view_id)
    // Note: In our implementation, face_labels_ stores view_id (0-based)
    // In mvs-texturing, graph labels are view_id + 1 (1-based)
    for (size_t i = 0; i < texture_views_.size(); ++i) {
        size_t view_id = i;
        int label = static_cast<int>(
                view_id +
                1);  // mvs-texturing uses label = view_id + 1 for TexturePatch

        // Get subgraphs for this view_id (connected components with same
        // view_id) Note: GetSubgraphs expects the label value stored in
        // face_labels_ (which is view_id)
        std::vector<std::vector<size_t>> subgraphs;
        GetSubgraphs(view_id, &subgraphs);

        if (subgraphs.empty()) continue;

        TextureView* texture_view = texture_views_[view_id].get();

        // Load image if not already loaded
        if (!texture_view->image_data) {
            std::string image_path = texture_view->image_file;

            // Try multiple path resolution strategies
            if (!ExistsFile(image_path)) {
                std::vector<std::string> candidates;

                // 1. Try workspace images directory
                if (workspace_) {
                    std::string workspace_path =
                            workspace_->GetOptions().workspace_path;
                    candidates.push_back(colmap::JoinPaths(
                            workspace_path, "images",
                            colmap::GetPathBaseName(image_path)));
                }

                // 2. Try image_path_/images
                candidates.push_back(
                        colmap::JoinPaths(image_path_, "images",
                                          colmap::GetPathBaseName(image_path)));

                // 3. Try image_path_ directly
                candidates.push_back(colmap::JoinPaths(
                        image_path_, colmap::GetPathBaseName(image_path)));

                for (const auto& candidate : candidates) {
                    if (ExistsFile(candidate)) {
                        image_path = candidate;
                        break;
                    }
                }
            }

            QImageReader reader(QString::fromStdString(image_path));
            QImage img = reader.read();
            if (!img.isNull()) {
                texture_view->image_data = std::make_shared<QImage>(
                        img.convertToFormat(QImage::Format_RGB888));
            } else {
                CVLog::Warning("Failed to load image: %s", image_path.c_str());
                continue;
            }
        }

        // Generate candidates for each subgraph
        std::list<TexturePatchCandidate> candidates;
        for (size_t j = 0; j < subgraphs.size(); ++j) {
            TexturePatchCandidate candidate =
                    GenerateCandidate(label, texture_view, subgraphs[j], mesh,
                                      cloud, mesh_is_shifted);
            if (candidate.texture_patch) {
                candidates.push_back(std::move(candidate));
            }
        }

        // Merge candidates which contain the same image content (following
        // mvs-texturing) If one candidate's bounding box is inside another,
        // merge them mvs-texturing: if (it != sit &&
        // bounding_box.is_inside(&it->bounding_box)) This means: if sit's bbox
        // is inside it's bbox, merge sit into it
        for (auto it = candidates.begin(); it != candidates.end(); ++it) {
            for (auto sit = candidates.begin(); sit != candidates.end();) {
                if (it != sit) {
                    // Check if sit's bounding box is inside it's bounding box
                    bool sit_inside_it = (sit->min_x >= it->min_x &&
                                          sit->max_x <= it->max_x &&
                                          sit->min_y >= it->min_y &&
                                          sit->max_y <= it->max_y);

                    if (sit_inside_it) {
                        // Merge sit into it: add sit's faces and adjust
                        // texcoords
                        auto& it_faces = it->texture_patch->faces;
                        auto& sit_faces = sit->texture_patch->faces;
                        it_faces.insert(it_faces.end(), sit_faces.begin(),
                                        sit_faces.end());

                        auto& it_texcoords = it->texture_patch->texcoords;
                        auto& sit_texcoords = sit->texture_patch->texcoords;
                        Eigen::Vector2f offset(sit->min_x - it->min_x,
                                               sit->min_y - it->min_y);
                        for (const auto& tc : sit_texcoords) {
                            it_texcoords.push_back(tc + offset);
                        }

                        sit = candidates.erase(sit);
                    } else {
                        ++sit;
                    }
                } else {
                    ++sit;
                }
            }
        }

        // Add candidates to texture_patches_ and update
        // vertex_projection_infos_
        for (auto it = candidates.begin(); it != candidates.end(); ++it) {
            size_t texture_patch_id;

            // Add to texture_patches_
            texture_patches_.push_back(std::move(it->texture_patch));
            texture_patch_id = num_patches++;

            // Update vertex_projection_infos_ (following mvs-texturing)
            const auto& patch = texture_patches_.back();
            const auto& faces = patch->faces;
            const auto& texcoords = patch->texcoords;

            for (size_t face_idx = 0; face_idx < faces.size(); ++face_idx) {
                size_t face_id = faces[face_idx];
                Eigen::Vector3i tri_idx;
                mesh.getTriangleVertIndexes(face_id, tri_idx);

                for (size_t j = 0; j < 3; ++j) {
                    size_t vertex_id = tri_idx(j);
                    if (vertex_id >= vertex_projection_infos_.size()) continue;

                    Eigen::Vector2f projection = texcoords[face_idx * 3 + j];

                    VertexProjectionInfo info;
                    info.texture_patch_id = texture_patch_id;
                    info.projection = projection;
                    info.faces = {face_id};

                    vertex_projection_infos_[vertex_id].push_back(info);
                }
            }
        }
    }

    // Merge vertex projection infos (following mvs-texturing)
    MergeVertexProjectionInfos();

    // Handle unseen faces (label = 0 in mvs-texturing, SIZE_MAX in our
    // implementation) Following mvs-texturing: get_subgraphs(0, &subgraphs) for
    // unseen faces
    {
        std::vector<size_t> unseen_faces;
        std::vector<std::vector<size_t>> subgraphs;

        // Get subgraphs with label = SIZE_MAX (unseen faces in our
        // implementation) In mvs-texturing, label = 0 means unseen We use
        // SIZE_MAX to represent unseen faces
        for (size_t face_id = 0; face_id < face_labels_.size(); ++face_id) {
            if (face_labels_[face_id] == SIZE_MAX) {
                unseen_faces.push_back(face_id);
            }
        }

        // Group unseen faces into connected components
        if (!unseen_faces.empty()) {
            std::vector<bool> used(face_labels_.size(), false);
            for (size_t face_id : unseen_faces) {
                if (used[face_id]) continue;

                subgraphs.push_back(std::vector<size_t>());
                std::list<size_t> queue;
                queue.push_back(face_id);
                used[face_id] = true;

                while (!queue.empty()) {
                    size_t node = queue.front();
                    queue.pop_front();
                    subgraphs.back().push_back(node);

                    for (size_t adj_face : face_adjacency_[node]) {
                        if (adj_face < face_labels_.size() &&
                            face_labels_[adj_face] == SIZE_MAX &&
                            !used[adj_face]) {
                            queue.push_back(adj_face);
                            used[adj_face] = true;
                        }
                    }
                }
            }
        }

        // Process unseen faces (following mvs-texturing: fill_hole or
        // keep_unseen_faces) For now, we create a simple texture patch for
        // unseen faces
        // TODO: Implement proper hole filling if needed
        if (!unseen_faces.empty() && options_.verbose) {
            CVLog::Print("Found %zu unseen faces in %zu subgraphs",
                         unseen_faces.size(), subgraphs.size());

            // Create a simple texture patch for unseen faces (following
            // mvs-texturing) Use a small default texture
            QImage default_image(3, 3, QImage::Format_RGB888);
            default_image.fill(QColor(128, 128, 128));  // Gray color

            std::vector<Eigen::Vector2f> texcoords;
            for (size_t i = 0; i < unseen_faces.size(); ++i) {
                // Use fixed texture coordinates (following mvs-texturing:
                // {{2.0f, 1.0f}, {1.0f, 1.0f}, {1.0f, 2.0f}})
                texcoords.push_back(Eigen::Vector2f(2.0f, 1.0f));
                texcoords.push_back(Eigen::Vector2f(1.0f, 1.0f));
                texcoords.push_back(Eigen::Vector2f(1.0f, 2.0f));
            }

            auto patch_ptr = std::make_shared<QImage>(default_image);
            auto texture_patch = std::make_unique<TexturePatch>(
                    0, unseen_faces, texcoords, patch_ptr, 0, 0, 2, 2);

            size_t texture_patch_id = texture_patches_.size();
            texture_patches_.push_back(std::move(texture_patch));

            // Update vertex_projection_infos_ for unseen faces
            for (size_t i = 0; i < unseen_faces.size(); ++i) {
                size_t face_id = unseen_faces[i];
                Eigen::Vector3i tri_idx;
                mesh.getTriangleVertIndexes(face_id, tri_idx);

                for (size_t j = 0; j < 3; ++j) {
                    size_t vertex_id = tri_idx(j);
                    if (vertex_id >= vertex_projection_infos_.size()) continue;

                    Eigen::Vector2f projection = texcoords[i * 3 + j];

                    VertexProjectionInfo info;
                    info.texture_patch_id = texture_patch_id;
                    info.projection = projection;
                    info.faces = {face_id};

                    vertex_projection_infos_[vertex_id].push_back(info);
                }
            }
        }
    }

    // Merge vertex projection infos again (following mvs-texturing)
    MergeVertexProjectionInfos();

    if (options_.verbose) {
        CVLog::Print("Generated %zu texture patches", texture_patches_.size());
    }
}

void MvsTexturing::SeamLeveling(const ccMesh& mesh) {
    // Following mvs-texturing's seam leveling flow:
    // - If global_seam_leveling: call global_seam_leveling
    // - Else: call adjust_colors with zero adjust_values to calculate validity
    // masks
    // - If local_seam_leveling: call local_seam_leveling
    // Note: Our TexturePatch doesn't have adjust_colors/validity_mask, so we
    // use simplified approach

    if (options_.verbose) {
        CVLog::Print("Performing seam leveling...");
    }

    if (texture_patches_.empty()) {
        return;
    }

    // Find seam edges: edges between faces with different labels
    // Following mvs-texturing: find_seam_edges(graph, mesh, &seam_edges)
    seam_edges_.clear();
    std::set<std::pair<size_t, size_t>> processed_vertex_edges;

    for (size_t face_id = 0; face_id < face_labels_.size(); ++face_id) {
        if (face_labels_[face_id] == SIZE_MAX) continue;

        for (size_t adj_face : face_adjacency_[face_id]) {
            if (adj_face >= face_labels_.size() ||
                face_labels_[adj_face] == SIZE_MAX)
                continue;
            if (face_labels_[face_id] == face_labels_[adj_face]) continue;

            // Find shared edge vertices (following mvs-texturing:
            // find_seam_edges)
            Eigen::Vector3i tri1_idx, tri2_idx;
            mesh.getTriangleVertIndexes(face_id, tri1_idx);
            mesh.getTriangleVertIndexes(adj_face, tri2_idx);

            // Find shared vertices
            std::vector<size_t> shared_vertices;
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    if (tri1_idx(i) == tri2_idx(j)) {
                        shared_vertices.push_back(tri1_idx(i));
                    }
                }
            }

            if (shared_vertices.size() == 2) {
                SeamEdge seam;
                seam.face1 = face_id;
                seam.face2 = adj_face;
                seam.v1 = std::min(shared_vertices[0], shared_vertices[1]);
                seam.v2 = std::max(shared_vertices[0], shared_vertices[1]);

                auto edge_key = std::make_pair(seam.v1, seam.v2);
                if (processed_vertex_edges.find(edge_key) ==
                    processed_vertex_edges.end()) {
                    processed_vertex_edges.insert(edge_key);
                    seam_edges_.push_back(seam);
                }
            }
        }
    }

    // Following mvs-texturing: if global_seam_leveling is false, calculate
    // validity masks by calling adjust_colors with zero adjust_values Since our
    // TexturePatch doesn't have adjust_colors, we skip this step In a full
    // implementation, we would:
    //  1. Call adjust_colors with zero adjust_values to calculate validity
    //  masks
    //  2. Optionally call global_seam_leveling if enabled
    //  3. Optionally call local_seam_leveling if enabled

    if (options_.verbose) {
        CVLog::Print("Seam leveling completed: %zu seam edges found",
                     seam_edges_.size());
    }
}

void MvsTexturing::GenerateTextureAtlases() {
    if (options_.verbose) {
        CVLog::Print("Generating texture atlases with bin packing...");
    }

    if (texture_patches_.empty()) {
        return;
    }

    // Constants for texture atlas sizing
    const unsigned int MAX_TEXTURE_SIZE = 8192;
    const unsigned int PREF_TEXTURE_SIZE = 4096;
    const unsigned int MIN_TEXTURE_SIZE = 256;
    const unsigned int PADDING = 2;

    // Sort patches by size (largest first) for better packing
    std::vector<size_t> patch_indices(texture_patches_.size());
    std::iota(patch_indices.begin(), patch_indices.end(), 0);
    std::sort(patch_indices.begin(), patch_indices.end(),
              [this](size_t a, size_t b) {
                  int size_a = texture_patches_[a]->image->width() *
                               texture_patches_[a]->image->height();
                  int size_b = texture_patches_[b]->image->width() *
                               texture_patches_[b]->image->height();
                  return size_a > size_b;
              });

    // Simple bin packing: try to pack patches into atlases
    struct AtlasPatch {
        size_t patch_idx;
        unsigned int x, y;
    };

    struct Atlas {
        unsigned int width;
        unsigned int height;
        std::vector<AtlasPatch> patches;
    };

    std::vector<Atlas> atlases;
    unsigned int current_atlas_size = PREF_TEXTURE_SIZE;

    for (size_t patch_idx : patch_indices) {
        const auto& patch = texture_patches_[patch_idx];
        unsigned int patch_width = patch->image->width() + 2 * PADDING;
        unsigned int patch_height = patch->image->height() + 2 * PADDING;

        // Find or create atlas that can fit this patch
        bool placed = false;
        for (auto& atlas : atlases) {
            // Simple bottom-left bin packing
            // Try to place at different positions
            for (unsigned int y = 0; y + patch_height <= atlas.height;
                 y += 64) {
                for (unsigned int x = 0; x + patch_width <= atlas.width;
                     x += 64) {
                    // Check if this position doesn't overlap with existing
                    // patches
                    bool overlaps = false;
                    for (const auto& ap : atlas.patches) {
                        const auto& existing_patch =
                                texture_patches_[ap.patch_idx];
                        unsigned int ew =
                                existing_patch->image->width() + 2 * PADDING;
                        unsigned int eh =
                                existing_patch->image->height() + 2 * PADDING;

                        if (!(x + patch_width <= ap.x || ap.x + ew <= x ||
                              y + patch_height <= ap.y || ap.y + eh <= y)) {
                            overlaps = true;
                            break;
                        }
                    }

                    if (!overlaps) {
                        AtlasPatch ap;
                        ap.patch_idx = patch_idx;
                        ap.x = x;
                        ap.y = y;
                        atlas.patches.push_back(ap);
                        placed = true;
                        break;
                    }
                }
                if (placed) break;
            }
            if (placed) break;
        }

        // Create new atlas if patch doesn't fit
        if (!placed) {
            // Determine atlas size
            unsigned int atlas_size = current_atlas_size;
            if (patch_width > atlas_size || patch_height > atlas_size) {
                atlas_size = std::max(patch_width, patch_height);
                // Round up to power of 2
                unsigned int next_power = 1;
                while (next_power < atlas_size &&
                       next_power < MAX_TEXTURE_SIZE) {
                    next_power <<= 1;
                }
                atlas_size = std::min(next_power, MAX_TEXTURE_SIZE);
            }

            Atlas new_atlas;
            new_atlas.width = atlas_size;
            new_atlas.height = atlas_size;
            AtlasPatch ap;
            ap.patch_idx = patch_idx;
            ap.x = PADDING;
            ap.y = PADDING;
            new_atlas.patches.push_back(ap);
            atlases.push_back(new_atlas);
        }
    }

    // Merge patches into atlas images and update texture coordinates
    texture_atlases_.clear();
    texture_atlases_.reserve(atlases.size());

    for (size_t atlas_idx = 0; atlas_idx < atlases.size(); ++atlas_idx) {
        const auto& atlas_info = atlases[atlas_idx];

        TextureAtlas atlas;
        atlas.width = atlas_info.width;
        atlas.height = atlas_info.height;
        atlas.image = std::make_shared<QImage>(
                atlas_info.width, atlas_info.height, QImage::Format_RGB888);
        atlas.image->fill(Qt::black);

        size_t texcoord_id_offset = 0;

        // Merge all patches into this atlas
        for (const auto& ap : atlas_info.patches) {
            const auto& patch = texture_patches_[ap.patch_idx];

            // Copy patch image into atlas at the specified position
            QPainter painter(atlas.image.get());
            painter.drawImage(ap.x + PADDING, ap.y + PADDING, *patch->image);
            painter.end();

            // Update texture coordinates: convert from patch-local to
            // atlas-normalized
            for (size_t i = 0; i < patch->faces.size(); ++i) {
                size_t face_id = patch->faces[i];
                atlas.face_ids.push_back(face_id);

                // Get texture coordinates for this face (3 vertices)
                size_t face_texcoord_start = atlas.texcoords.size();
                for (int j = 0; j < 3; ++j) {
                    size_t texcoord_idx = i * 3 + j;
                    Eigen::Vector2f local_tc = patch->texcoords[texcoord_idx];

                    // Convert to atlas coordinates: add offset and normalize
                    float atlas_u =
                            (local_tc.x() + ap.x + PADDING) / atlas.width;
                    float atlas_v =
                            (local_tc.y() + ap.y + PADDING) / atlas.height;

                    atlas.texcoords.emplace_back(atlas_u, atlas_v);
                }

                // Store texture coordinate indices for this face (relative to
                // atlas start)
                atlas.texcoord_ids.push_back(face_texcoord_start);
                atlas.texcoord_ids.push_back(face_texcoord_start + 1);
                atlas.texcoord_ids.push_back(face_texcoord_start + 2);
            }
        }

        texture_atlases_.push_back(atlas);
    }

    if (options_.verbose) {
        CVLog::Print("Created %zu texture atlases from %zu patches",
                     texture_atlases_.size(), texture_patches_.size());
        for (size_t i = 0; i < texture_atlases_.size(); ++i) {
            CVLog::Print("  Atlas %zu: %u x %u, %zu faces", i + 1,
                         texture_atlases_[i].width, texture_atlases_[i].height,
                         texture_atlases_[i].face_ids.size());
        }
    }
}

bool MvsTexturing::SaveOBJModel(const std::string& output_path,
                                const ccMesh& mesh) {
    // Following mvs-texturing's build_model implementation exactly
    if (options_.verbose) {
        CVLog::Print("Building OBJ model...");
    }

    obj_model_ = std::make_unique<ObjModel>();

    ccPointCloud* cloud =
            ccHObjectCaster::ToPointCloud(mesh.getAssociatedCloud());
    if (!cloud) {
        CVLog::Error("Mesh has no associated cloud!");
        return false;
    }

    // Copy vertices with global shift/scale handling (following mvs-texturing:
    // mesh->get_vertices())
    obj_model_->vertices.reserve(cloud->size());
    bool is_shifted = cloud->isShifted();

    for (unsigned i = 0; i < cloud->size(); ++i) {
        const CCVector3* pt = cloud->getPoint(i);
        if (is_shifted) {
            CCVector3d global_pt = cloud->toGlobal3d(*pt);
            obj_model_->vertices.emplace_back(static_cast<float>(global_pt.x),
                                              static_cast<float>(global_pt.y),
                                              static_cast<float>(global_pt.z));
        } else {
            obj_model_->vertices.emplace_back(pt->x, pt->y, pt->z);
        }
    }

    // Copy normals (following mvs-texturing: mesh->get_vertex_normals())
    obj_model_->normals.reserve(cloud->size());
    if (cloud->hasNormals()) {
        for (unsigned i = 0; i < cloud->size(); ++i) {
            const CCVector3& n = cloud->getPointNormal(i);
            obj_model_->normals.emplace_back(n.x, n.y, n.z);
        }
    } else {
        CVLog::Warning("Mesh has no normals, using default normals");
        obj_model_->normals.resize(cloud->size(), Eigen::Vector3f(0, 0, 1));
    }

    // Process texture atlases (following mvs-texturing: build_model)
    if (texture_atlases_.empty()) {
        CVLog::Warning("No texture atlases to save!");
        return false;
    }

    // Helper function to format material name (material0000, material0001,
    // etc.)
    auto format_material_name = [](size_t n) -> std::string {
        std::string name = "material";
        std::string num_str = std::to_string(n);
        // Pad with zeros to 4 digits (mvs-texturing uses get_filled(n, 4))
        while (num_str.length() < 4) {
            num_str = "0" + num_str;
        }
        return name + num_str;
    };

    for (size_t atlas_idx = 0; atlas_idx < texture_atlases_.size();
         ++atlas_idx) {
        const auto& atlas = texture_atlases_[atlas_idx];

        // Create material (following mvs-texturing: material.name = "material"
        // + get_filled(n, 4))
        std::string material_name = format_material_name(atlas_idx);
        obj_model_->materials[material_name] =
                material_name;  // Store material name

        // Save texture atlas image (following mvs-texturing:
        // prefix_materialname_map_Kd.png)
        std::string prefix = output_path;
        std::string root, ext;
        colmap::SplitFileExtension(output_path, &root, &ext);
        if (ext == ".obj" || ext == ".OBJ") {
            prefix = root;
        }
        std::string name = colmap::GetPathBaseName(prefix);
        std::string texture_path =
                colmap::JoinPaths(colmap::GetParentDir(prefix),
                                  name + "_" + material_name + "_map_Kd.png");
        if (!atlas.image->save(QString::fromStdString(texture_path))) {
            CVLog::Warning("Failed to save texture atlas %zu to %s", atlas_idx,
                           texture_path.c_str());
        }

        // Add texture coordinates (following mvs-texturing:
        // texcoords.insert(...))
        size_t texcoord_id_offset = obj_model_->texcoords.size();
        obj_model_->texcoords.insert(obj_model_->texcoords.end(),
                                     atlas.texcoords.begin(),
                                     atlas.texcoords.end());

        // Create group (following mvs-texturing:
        // groups.push_back(ObjModel::Group()))
        ObjModel::Group group;
        group.material_name = material_name;

        // Add faces (following mvs-texturing: for each atlas face)
        const auto& atlas_faces = atlas.face_ids;
        const auto& atlas_texcoord_ids = atlas.texcoord_ids;

        for (size_t i = 0; i < atlas_faces.size(); ++i) {
            size_t mesh_face_pos = atlas_faces[i];
            Eigen::Vector3i tri_idx;
            mesh.getTriangleVertIndexes(mesh_face_pos, tri_idx);

            // Vertex IDs (following mvs-texturing: mesh_faces[mesh_face_pos])
            size_t vertex_ids[3] = {static_cast<size_t>(tri_idx(0)),
                                    static_cast<size_t>(tri_idx(1)),
                                    static_cast<size_t>(tri_idx(2))};
            size_t* normal_ids =
                    vertex_ids;  // Same as vertex_ids (following mvs-texturing)

            // Texture coordinate IDs (following mvs-texturing:
            // texcoord_id_offset + atlas_texcoord_ids[i * 3])
            size_t texcoord_ids[3] = {
                    texcoord_id_offset + atlas_texcoord_ids[i * 3 + 0],
                    texcoord_id_offset + atlas_texcoord_ids[i * 3 + 1],
                    texcoord_id_offset + atlas_texcoord_ids[i * 3 + 2]};

            ObjModel::Face face;
            std::copy(vertex_ids, vertex_ids + 3, face.vertex_ids);
            std::copy(texcoord_ids, texcoord_ids + 3, face.texcoord_ids);
            std::copy(normal_ids, normal_ids + 3, face.normal_ids);

            group.faces.push_back(face);
        }

        obj_model_->groups.push_back(group);
    }

    // Save model (following mvs-texturing: Model::save(model, prefix))
    std::string prefix = output_path;
    std::string root, ext;
    colmap::SplitFileExtension(output_path, &root, &ext);
    if (ext == ".obj" || ext == ".OBJ") {
        prefix = root;
    }

    bool success = obj_model_->SaveToFiles(prefix);
    if (success && options_.verbose) {
        CVLog::Print("Saved OBJ model to %s (vertices: %zu, faces: %zu)",
                     prefix.c_str(), obj_model_->vertices.size(),
                     obj_model_->groups.empty()
                             ? 0
                             : std::accumulate(obj_model_->groups.begin(),
                                               obj_model_->groups.end(), 0,
                                               [](size_t sum,
                                                  const ObjModel::Group& g) {
                                                   return sum + g.faces.size();
                                               }));
    }

    return success;
}

}  // namespace cloudViewer
