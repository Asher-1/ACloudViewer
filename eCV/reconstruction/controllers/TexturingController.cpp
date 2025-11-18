// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "TexturingController.h"

#include <cmath>
#include <fstream>

#include "base/reconstruction.h"
#include "mvs/workspace.h"
#include "util/logging.h"

// CV_CORE_LIB
#include <FileSystem.h>

// ECV_DB_LIB
#include "camera/PinholeCameraTrajectory.h"

// ECV_IO_LIB
#include <AutoIO.h>
#include <ecvHObjectCaster.h>
#include <ecvMesh.h>
#include <ecvPointCloud.h>

// Local
#include "MvsTexturing.h"

namespace cloudViewer {

using namespace colmap;

TexturingReconstruction::TexturingReconstruction(
        const TexturingOptions& options,
        const Reconstruction& reconstruction,
        const std::string& image_path,
        const std::string& output_path,
        const std::vector<image_t>& image_ids)
    : options_(options),
      image_path_(image_path),
      output_path_(output_path),
      image_ids_(image_ids),
      reconstruction_(reconstruction) {
    camera_trajectory_ = std::make_shared<camera::PinholeCameraTrajectory>();
}

void TexturingReconstruction::Run() {
    PrintHeading1("Mesh Texturing");

    CreateDirIfNotExists(JoinPaths(output_path_, "images"));
    CreateDirIfNotExists(JoinPaths(output_path_, "sparse"));
    CreateDirIfNotExists(JoinPaths(output_path_, "stereo"));
    CreateDirIfNotExists(JoinPaths(output_path_, "stereo/depth_maps"));

    // Create workspace for accessing depth maps and normal maps
    if (options_.use_depth_normal_maps) {
        colmap::mvs::Workspace::Options workspace_options;
        workspace_options.workspace_path = output_path_;
        workspace_options.workspace_format = "COLMAP";
        workspace_options.input_type = options_.depth_map_type;
        workspace_options.stereo_folder = "stereo";
        workspace_options.max_image_size = -1;
        workspace_options.image_as_rgb = true;
        workspace_options.cache_size = 32.0;
        workspace_options.num_threads = -1;

        workspace_ = std::make_unique<colmap::mvs::CachedWorkspace>(
                workspace_options);
    }
    std::string parent_path = utility::filesystem::GetFileParentDirectory(
            options_.textured_file_path);
    CreateDirIfNotExists(parent_path);
    CreateDirIfNotExists(JoinPaths(parent_path, "images"));

    ThreadPool thread_pool;
    std::vector<std::future<bool>> futures;
    futures.reserve(reconstruction_.NumRegImages());
    camera_trajectory_->parameters_.clear();
    if (image_ids_.empty()) {
        camera_trajectory_->parameters_.resize(reconstruction_.NumRegImages());
        for (size_t i = 0; i < reconstruction_.NumRegImages(); ++i) {
            const image_t image_id = reconstruction_.RegImageIds().at(i);
            futures.push_back(thread_pool.AddTask(
                    &TexturingReconstruction::Texturing, this, image_id, i));
        }
    } else {
        std::size_t index = 0;
        camera_trajectory_->parameters_.resize(image_ids_.size());
        for (const image_t image_id : image_ids_) {
            futures.push_back(
                    thread_pool.AddTask(&TexturingReconstruction::Texturing,
                                        this, image_id, index));
            index += 1;
        }
    }

    // Only use the image names for the successfully textured mesh
    // when writing the MVS config files
    image_names_.clear();
    for (size_t i = 0; i < futures.size(); ++i) {
        if (IsStopped()) {
            break;
        }

        if (options_.verbose) {
            std::cout << StringPrintf("texture image [%d/%d]", i + 1,
                                      futures.size())
                      << std::endl;
            CVLog::Print("texture image [%d/%d]", i + 1, futures.size());
        }

        if (futures[i].get()) {
            if (image_ids_.empty()) {
                const image_t image_id = reconstruction_.RegImageIds().at(i);
                image_names_.push_back(reconstruction_.Image(image_id).Name());
            } else {
                image_names_.push_back(
                        reconstruction_.Image(image_ids_[i]).Name());
            }
        }
    }

    // check camera trajectory validation
    for (int i = 0; i < camera_trajectory_->parameters_.size(); ++i) {
        auto& cameraParams = camera_trajectory_->parameters_[i];
        if (!cameraParams.intrinsic_.IsValid()) {
            CVLog::Error(
                    "Invalid camera intrinsic parameters found and ignore "
                    "texturing!");
            return;
        }
    }

    // Load mesh once and reuse for both filtering and texturing
    ccMesh mesh;
    bool mesh_loaded = false;
    if (!options_.meshed_file_path.empty() &&
        ExistsFile(options_.meshed_file_path)) {
        if (!mesh.CreateInternalCloud()) {
            CVLog::Error("creating internal cloud failed!");
            return;
        }
        cloudViewer::io::ReadTriangleMeshOptions mesh_options;
        mesh_options.print_progress = false;
        if (cloudViewer::io::AutoReadMesh(options_.meshed_file_path, mesh,
                                          mesh_options)) {
            mesh_loaded = true;
            if (!mesh.hasNormals()) {
                mesh.computeNormals(true);
            }
            if (options_.verbose) {
                CVLog::Print("Loaded mesh: %zu vertices, %zu triangles",
                             mesh.getAssociatedCloud()
                                     ? mesh.getAssociatedCloud()->size()
                                     : 0,
                             mesh.size());
            }
        } else {
            if (options_.verbose) {
                CVLog::Warning("Failed to load mesh from %s",
                               options_.meshed_file_path.c_str());
            }
        }
    } else {
        CVLog::Error("Mesh file path is empty or mesh file does not exist: %s",
                     options_.meshed_file_path.c_str());
        return;
    }

    // Filter camera trajectory based on depth/normal maps if enabled
    std::shared_ptr<camera::PinholeCameraTrajectory> filtered_trajectory =
            camera_trajectory_;
    if (options_.use_depth_normal_maps && workspace_) {
        filtered_trajectory = FilterCameraTrajectory(&mesh);
        if (options_.verbose) {
            CVLog::Print("Filtered camera trajectory: %zu -> %zu cameras",
                         camera_trajectory_->parameters_.size(),
                         filtered_trajectory->parameters_.size());
        }

        // Additional safety check: ensure filtered trajectory is valid
        if (filtered_trajectory->parameters_.empty()) {
            CVLog::Error(
                    "No valid cameras after filtering! Aborting texturing.");
            return;
        }

        // Validate all cameras in filtered trajectory
        for (size_t i = 0; i < filtered_trajectory->parameters_.size(); ++i) {
            const auto& params = filtered_trajectory->parameters_[i];
            if (!params.intrinsic_.IsValid()) {
                CVLog::Error(
                        "Camera %zu in filtered trajectory has invalid "
                        "intrinsics!",
                        i);
                return;
            }
            // Check extrinsic matrix
            for (int row = 0; row < 4; ++row) {
                for (int col = 0; col < 4; ++col) {
                    if (!std::isfinite(params.extrinsic_(row, col))) {
                        CVLog::Error(
                                "Camera %zu in filtered trajectory has invalid "
                                "extrinsics!",
                                i);
                        return;
                    }
                }
            }
        }
    }

    // Use mvs-texturing approach
    MvsTexturing::Options mvs_options;
    mvs_options.verbose = options_.verbose;
    mvs_options.max_depth_error = options_.max_depth_error;
    mvs_options.max_viewing_angle_deg = 75.0f;  // mvs-texturing uses 75 degrees
    mvs_options.use_depth_normal_maps = options_.use_depth_normal_maps;
    mvs_options.use_gradient_magnitude =
            false;  // Use area for now, can enable GMI later

    MvsTexturing texturing(mvs_options, reconstruction_, workspace_.get(),
                           image_path_);

    // Pass loaded mesh (must be loaded at this point)
    if (!mesh_loaded) {
        CVLog::Error("Mesh must be loaded before texturing!");
        return;
    }
    if (texturing.TextureMesh(mesh, *filtered_trajectory,
                              options_.textured_file_path)) {
        CVLog::Print("Save textured mesh to %s successfully!",
                     options_.textured_file_path.c_str());
    } else {
        CVLog::Warning("Texturing reconstruction failed!");
    }
    GetTimer().PrintMinutes();
}

bool TexturingReconstruction::Texturing(const image_t image_id,
                                        std::size_t index) {
    const Image& image = reconstruction_.Image(image_id);
    const Camera& camera = reconstruction_.Camera(image.CameraId());

    const std::string input_image_path = JoinPaths(image_path_, image.Name());
    const std::string texture_file = JoinPaths("images", image.Name());

    std::string target_file_path =
            JoinPaths(utility::filesystem::GetFileParentDirectory(
                              options_.textured_file_path),
                      texture_file);
    if (!ExistsFile(target_file_path) &&
        ExistsFile(JoinPaths(output_path_, texture_file))) {
        FileCopy(JoinPaths(output_path_, texture_file), target_file_path);
    }

    // Check if workspace is available and has depth/normal maps
    int workspace_image_idx = GetWorkspaceImageIdx(image_id);

    CVLog::Print(
            "[Texturing] Processing image_id=%u, image_name='%s', "
            "workspace_image_idx=%d",
            image_id, image.Name().c_str(), workspace_image_idx);

    // Store workspace_image_idx for later use in visibility checking
    // This will be used when filtering cameras during texture mapping

    auto& cameraParams = camera_trajectory_->parameters_[index];
    cameraParams.texture_file_ = texture_file;
    const Eigen::Matrix3x4d proj_matrix = image.InverseProjectionMatrix();
    Eigen::Matrix3d rotation = proj_matrix.leftCols<3>();
    Eigen::Vector3d translation = proj_matrix.rightCols<1>();
    ccGLMatrixd extrinsic = ccGLMatrixd::FromEigenMatrix3(rotation);
    extrinsic.setTranslation(translation.data());
    cameraParams.extrinsic_ = ccGLMatrixd::ToEigenMatrix4(extrinsic);
    std::string model_name = camera.ModelName();
    // https://github.com/colmap/colmap/blob/dev/src/base/camera_models.h
    if (model_name == "SIMPLE_PINHOLE" || model_name == "SIMPLE_RADIAL" ||
        model_name == "SIMPLE_RADIAL_FISHEYE" || model_name == "RADIAL" ||
        model_name == "RADIAL_FISHEYE") {
        // Simple pinhole: f, cx, cy
        cameraParams.intrinsic_.SetIntrinsics(
                static_cast<int>(camera.Width()),
                static_cast<int>(camera.Height()), camera.FocalLength(),
                camera.FocalLength(), camera.PrincipalPointX(),
                camera.PrincipalPointY());
    } else if (model_name == "PINHOLE" || model_name == "OPENCV" ||
               model_name == "OPENCV_FISHEYE" || model_name == "FULL_OPENCV" ||
               model_name == "FOV" || model_name == "THIN_PRISM_FISHEYE") {
        // Pinhole: fx, fy, cx, cy
        cameraParams.intrinsic_.SetIntrinsics(
                static_cast<int>(camera.Width()),
                static_cast<int>(camera.Height()), camera.FocalLengthX(),
                camera.FocalLengthY(), camera.PrincipalPointX(),
                camera.PrincipalPointY());
        //      float fx = params[0];
        //      float fy = params[1];
        //      float dim_aspect = static_cast<float>(width) / height;
        //      float pixel_aspect = fy / fx;
        //      float img_aspect = dim_aspect * pixel_aspect;
        //      if (img_aspect < 1.0f) {
        //          camera_info.flen = fy / height;
        //      } else {
        //          camera_info.flen = fx / width;
        //      }
        //      camera_info.ppoint[0] = params[2] / width;
        //      camera_info.ppoint[1] = params[3] / height;
    } else {
        std::string msg =
                "Unsupported camera model with texturing "
                "detected! If possible, re-run the SfM reconstruction with the "
                "SIMPLE_PINHOLE or the PINHOLE camera model (recommended). "
                "Otherwise, use the undistortion step in Colmap to obtain "
                "undistorted images and corresponding camera models without "
                "radial "
                "distortion.";
        CVLog::Error(msg.c_str());
        return false;
    }
    return true;
}

bool TexturingOptions::Check() const {
    CHECK_OPTION_GT(save_precision, 0);
    CHECK_GT(max_depth_error, 0.0f);
    CHECK_GE(min_normal_consistency, -1.0f);
    CHECK_LE(min_normal_consistency, 1.0f);
    return true;
}

int TexturingReconstruction::GetWorkspaceImageIdx(
        const image_t image_id) const {
    if (!workspace_) {
        CVLog::Warning(
                "[GetWorkspaceImageIdx] workspace_ is null for image_id=%u",
                image_id);
        return -1;
    }

    const Image& image = reconstruction_.Image(image_id);
    const std::string& image_name = image.Name();

    const auto& model = workspace_->GetModel();

    CVLog::PrintDebug(
            "[GetWorkspaceImageIdx] Looking up image_name='%s' (image_id=%u) "
            "in workspace model (model has %zu images)",
            image_name.c_str(), image_id, model.images.size());

    // GetImageIdx uses CHECK_GT and .at() which throws exception if not found
    // Use try-catch to safely handle the case when image doesn't exist
    try {
        int workspace_image_idx = model.GetImageIdx(image_name);

        CVLog::PrintDebug(
                "[GetWorkspaceImageIdx] Found workspace_image_idx=%d for "
                "image_name='%s'",
                workspace_image_idx, image_name.c_str());

        // Validate the index is within valid range
        if (workspace_image_idx < 0) {
            CVLog::Warning(
                    "[GetWorkspaceImageIdx] Invalid negative "
                    "workspace_image_idx=%d for image_name='%s'",
                    workspace_image_idx, image_name.c_str());
            return -1;
        }

        if (static_cast<size_t>(workspace_image_idx) >= model.images.size()) {
            CVLog::Warning(
                    "[GetWorkspaceImageIdx] workspace_image_idx=%d >= "
                    "model.images.size()=%zu for image_name='%s'",
                    workspace_image_idx, model.images.size(),
                    image_name.c_str());
            return -1;
        }

        CVLog::PrintDebug(
                "[GetWorkspaceImageIdx] Successfully mapped image_name='%s' -> "
                "workspace_image_idx=%d",
                image_name.c_str(), workspace_image_idx);
        return workspace_image_idx;
    } catch (const std::exception& e) {
        // Image not found in workspace model or index out of range
        CVLog::Warning(
                "[GetWorkspaceImageIdx] Exception when looking up "
                "image_name='%s': %s",
                image_name.c_str(), e.what());
        return -1;
    }
}

bool TexturingReconstruction::IsPointVisible(const Eigen::Vector3d& point3d,
                                             const Image& image,
                                             const Camera& camera,
                                             int workspace_image_idx) const {
    if (!options_.use_depth_normal_maps || !workspace_ ||
        workspace_image_idx < 0) {
        // Without depth map, assume visible if projection is valid
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

        // Project 3D point to image coordinates
        // First transform to camera coordinates: [R|t] * [X; Y; Z; 1]
        const Eigen::Matrix3x4d world_to_cam =
                image.ProjectionMatrix();  // [R|t]
        Eigen::Vector4d point_homogeneous(point3d.x(), point3d.y(), point3d.z(),
                                          1.0);
        Eigen::Vector3d cam_coords = world_to_cam * point_homogeneous;

        // Check if point is behind camera
        if (cam_coords.z() <= 0) {
            return false;
        }

        // Build intrinsic matrix K
        Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
        K(0, 0) = camera.MeanFocalLength();
        K(1, 1) = camera.MeanFocalLength();
        K(0, 2) = camera.PrincipalPointX();
        K(1, 2) = camera.PrincipalPointY();

        // Project to pixel coordinates: K * [x_cam; y_cam; z_cam]
        Eigen::Vector3d proj = K * cam_coords;
        const double u = proj.x() / proj.z();
        const double v = proj.y() / proj.z();

        // Check if projection is within image bounds
        if (u < 0 || u >= camera.Width() || v < 0 || v >= camera.Height()) {
            return false;
        }

        const size_t row = static_cast<size_t>(std::round(v));
        const size_t col = static_cast<size_t>(std::round(u));

        // Check if depth map dimensions match
        if (row >= depth_map.GetHeight() || col >= depth_map.GetWidth()) {
            return false;
        }

        const float depth_map_value = depth_map.Get(row, col);
        const float point_depth = static_cast<float>(proj(2));

        // Check if depth is valid (positive)
        if (depth_map_value <= 0) {
            return false;
        }

        // Check depth consistency: point depth should be close to depth map
        // value
        const float depth_error =
                std::abs(point_depth - depth_map_value) / depth_map_value;
        return depth_error <= options_.max_depth_error;
    } catch (const std::exception&) {
        // Error accessing depth map, assume visible
        return true;
    }
}

float TexturingReconstruction::ComputeViewQuality(
        const Eigen::Vector3d& point3d,
        const Eigen::Vector3d& face_normal,
        const Image& image,
        const Camera& camera,
        int workspace_image_idx) const {
    if (!options_.use_depth_normal_maps || !workspace_ ||
        workspace_image_idx < 0) {
        // Without normal map, use viewing angle as quality measure
        const Eigen::Vector3d camera_pos = image.ProjectionCenter();
        const Eigen::Vector3d view_dir = (point3d - camera_pos).normalized();
        const double cos_angle = face_normal.normalized().dot(view_dir);
        return static_cast<float>(std::max(0.0, cos_angle));
    }

    // Validate index is within model range before accessing cache
    const auto& model = workspace_->GetModel();
    if (static_cast<size_t>(workspace_image_idx) >= model.images.size()) {
        // Invalid index, use fallback
        const Eigen::Vector3d camera_pos = image.ProjectionCenter();
        const Eigen::Vector3d view_dir = (point3d - camera_pos).normalized();
        const double cos_angle = face_normal.normalized().dot(view_dir);
        return static_cast<float>(std::max(0.0, cos_angle));
    }

    if (!workspace_->HasNormalMap(workspace_image_idx)) {
        // Fallback to viewing angle
        const Eigen::Vector3d camera_pos = image.ProjectionCenter();
        const Eigen::Vector3d view_dir = (point3d - camera_pos).normalized();
        const double cos_angle = face_normal.normalized().dot(view_dir);
        return static_cast<float>(std::max(0.0, cos_angle));
    }

    try {
        const auto& normal_map = workspace_->GetNormalMap(workspace_image_idx);

        // Project 3D point to image coordinates
        // First transform to camera coordinates: [R|t] * [X; Y; Z; 1]
        const Eigen::Matrix3x4d world_to_cam =
                image.ProjectionMatrix();  // [R|t]
        Eigen::Vector4d point_homogeneous(point3d.x(), point3d.y(), point3d.z(),
                                          1.0);
        Eigen::Vector3d cam_coords = world_to_cam * point_homogeneous;

        if (cam_coords.z() <= 0) {
            return 0.0f;
        }

        // Build intrinsic matrix K
        Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
        K(0, 0) = camera.MeanFocalLength();
        K(1, 1) = camera.MeanFocalLength();
        K(0, 2) = camera.PrincipalPointX();
        K(1, 2) = camera.PrincipalPointY();

        // Project to pixel coordinates: K * [x_cam; y_cam; z_cam]
        Eigen::Vector3d proj = K * cam_coords;
        const double u = proj.x() / proj.z();
        const double v = proj.y() / proj.z();

        if (u < 0 || u >= camera.Width() || v < 0 || v >= camera.Height()) {
            return 0.0f;
        }

        const size_t row = static_cast<size_t>(std::round(v));
        const size_t col = static_cast<size_t>(std::round(u));

        if (row >= normal_map.GetHeight() || col >= normal_map.GetWidth()) {
            return 0.0f;
        }

        // Get normal from normal map (in camera coordinate system)
        Eigen::Vector3d normal_map_normal(normal_map.Get(row, col, 0),
                                          normal_map.Get(row, col, 1),
                                          normal_map.Get(row, col, 2));

        // Transform face normal to camera coordinate system
        // The rotation matrix transforms from world to camera coordinates
        const Eigen::Matrix3d R = image.RotationMatrix();
        const Eigen::Vector3d face_normal_cam = R * face_normal.normalized();

        // Compute consistency: cosine of angle between normals
        const double cos_angle = face_normal_cam.normalized().dot(
                normal_map_normal.normalized());

        return static_cast<float>(cos_angle);
    } catch (const std::exception&) {
        // Error accessing normal map, use fallback
        const Eigen::Vector3d camera_pos = image.ProjectionCenter();
        const Eigen::Vector3d view_dir = (point3d - camera_pos).normalized();
        const double cos_angle = face_normal.normalized().dot(view_dir);
        return static_cast<float>(std::max(0.0, cos_angle));
    }
}

std::shared_ptr<camera::PinholeCameraTrajectory>
TexturingReconstruction::FilterCameraTrajectory(ccMesh* mesh) const {
    auto filtered_trajectory =
            std::make_shared<camera::PinholeCameraTrajectory>();

    // Sample mesh vertices for visibility testing
    std::vector<Eigen::Vector3d> mesh_vertices;
    if (!mesh || !mesh->getAssociatedCloud()) {
        CVLog::Error("Mesh is null or has no associated cloud!");
        return filtered_trajectory;
    }

    ccGenericPointCloud* generic_cloud = mesh->getAssociatedCloud();
    ccPointCloud* cloud = ccHObjectCaster::ToPointCloud(generic_cloud);
    int sample_step = 1;  // Initialize for later use
    if (cloud && cloud->size() > 0) {
        // Sample vertices (every Nth vertex to speed up testing)
        // IMPORTANT: Convert from local to global coordinates for visibility
        // testing Camera transforms are in global coordinate system
        const unsigned point_count = cloud->size();
        sample_step = std::max(1, static_cast<int>(point_count / 1000));
        mesh_vertices.reserve(point_count / sample_step);

        bool mesh_is_shifted = cloud->isShifted();
        for (unsigned i = 0; i < point_count; i += sample_step) {
            const CCVector3* pt = cloud->getPoint(i);
            if (mesh_is_shifted) {
                // Convert local to global coordinates
                CCVector3d pt_global = cloud->toGlobal3d(*pt);
                mesh_vertices.emplace_back(pt_global.x, pt_global.y,
                                           pt_global.z);
            } else {
                mesh_vertices.emplace_back(pt->x, pt->y, pt->z);
            }
        }

        if (options_.verbose) {
            CVLog::Print(
                    "Sampled %zu vertices from mesh for visibility testing",
                    mesh_vertices.size());
        }
    } else {
        if (options_.verbose) {
            CVLog::Warning("Mesh has no valid vertices for visibility testing");
        }
    }

    for (size_t i = 0; i < camera_trajectory_->parameters_.size(); ++i) {
        const auto& cameraParams = camera_trajectory_->parameters_[i];

        // Find corresponding image_id
        image_t image_id = kInvalidImageId;
        if (image_ids_.empty()) {
            if (i < reconstruction_.RegImageIds().size()) {
                image_id = reconstruction_.RegImageIds().at(i);
            }
        } else {
            if (i < image_ids_.size()) {
                image_id = image_ids_[i];
            }
        }

        if (image_id == kInvalidImageId) {
            continue;
        }

        const Image& image = reconstruction_.Image(image_id);
        const Camera& camera = reconstruction_.Camera(image.CameraId());
        int workspace_image_idx = GetWorkspaceImageIdx(image_id);

        // Check if camera has valid depth/normal maps
        // GetWorkspaceImageIdx now safely returns -1 if image not found or
        // invalid
        bool has_valid_maps = false;
        if (workspace_image_idx >= 0 && workspace_) {
            has_valid_maps = workspace_->HasDepthMap(workspace_image_idx) &&
                             workspace_->HasNormalMap(workspace_image_idx);
        }

        // If using depth/normal maps, perform visibility testing
        bool should_include = true;
        if (options_.use_depth_normal_maps) {
            if (!has_valid_maps) {
                should_include = false;
                if (options_.verbose) {
                    CVLog::Warning(
                            "Excluding camera %s from texturing (no "
                            "depth/normal maps)",
                            image.Name().c_str());
                }
            } else if (!mesh_vertices.empty()) {
                // Test visibility on sampled mesh vertices
                size_t visible_count = 0;
                size_t quality_count = 0;

                // Get vertex normals if available
                bool has_normals = cloud->hasNormals();

                for (size_t idx = 0; idx < mesh_vertices.size(); ++idx) {
                    const auto& vertex = mesh_vertices[idx];
                    // Test visibility using depth map
                    if (IsPointVisible(vertex, image, camera,
                                       workspace_image_idx)) {
                        visible_count++;

                        // Get actual vertex normal if available
                        Eigen::Vector3d face_normal(0, 0, 1);  // Default normal
                        if (has_normals) {
                            // Map back to original vertex index
                            size_t vertex_idx = idx * sample_step;
                            if (vertex_idx < cloud->size()) {
                                const CCVector3& n =
                                        cloud->getPointNormal(vertex_idx);
                                face_normal = Eigen::Vector3d(n.x, n.y, n.z);
                                // Convert to global coordinate system if needed
                                if (cloud->isShifted()) {
                                    // Normals don't need shift/scale, but may
                                    // need rotation For now, use as-is
                                    // (assuming normals are already in global)
                                }
                            }
                        }

                        float quality =
                                ComputeViewQuality(vertex, face_normal, image,
                                                   camera, workspace_image_idx);

                        // Check if quality meets threshold
                        // Use absolute value since normals might be oriented
                        // differently
                        if (std::abs(quality) >=
                            options_.min_normal_consistency) {
                            quality_count++;
                        }
                    }
                }

                // Include camera if it can see enough vertices with good
                // quality
                const double visibility_ratio =
                        static_cast<double>(visible_count) /
                        mesh_vertices.size();
                const double quality_ratio =
                        visible_count > 0 ? static_cast<double>(quality_count) /
                                                    visible_count
                                          : 0.0;

                const double min_visibility_ratio =
                        0.1;  // At least 10% of vertices visible
                // Lower quality ratio threshold: at least 5% of visible
                // vertices have good quality This is more lenient since normal
                // consistency can vary significantly
                const double min_quality_ratio = 0.05;

                should_include = visibility_ratio >= min_visibility_ratio &&
                                 quality_ratio >= min_quality_ratio;

                if (options_.verbose) {
                    if (should_include) {
                        CVLog::Print(
                                "Camera %s: visibility=%.2f%%, quality=%.2f%% "
                                "(included)",
                                image.Name().c_str(), visibility_ratio * 100.0,
                                quality_ratio * 100.0);
                    } else {
                        CVLog::Warning(
                                "Camera %s: visibility=%.2f%%, quality=%.2f%% "
                                "(excluded)",
                                image.Name().c_str(), visibility_ratio * 100.0,
                                quality_ratio * 100.0);
                    }
                }
            }
        }

        if (should_include) {
            // Validate camera parameters before adding
            if (!cameraParams.intrinsic_.IsValid()) {
                if (options_.verbose) {
                    CVLog::Warning(
                            "Excluding camera %s from texturing (invalid "
                            "intrinsic parameters)",
                            image.Name().c_str());
                }
                continue;
            }

            // Check if extrinsic matrix is valid (no NaN or Inf)
            bool extrinsic_valid = true;
            for (int row = 0; row < 4; ++row) {
                for (int col = 0; col < 4; ++col) {
                    const double val = cameraParams.extrinsic_(row, col);
                    if (!std::isfinite(val)) {
                        extrinsic_valid = false;
                        break;
                    }
                }
                if (!extrinsic_valid) break;
            }

            if (!extrinsic_valid) {
                if (options_.verbose) {
                    CVLog::Warning(
                            "Excluding camera %s from texturing (invalid "
                            "extrinsic matrix)",
                            image.Name().c_str());
                }
                continue;
            }

            filtered_trajectory->parameters_.push_back(cameraParams);
        }
    }

    // Final validation: ensure we have at least one valid camera
    if (filtered_trajectory->parameters_.empty()) {
        CVLog::Warning(
                "No valid cameras found after filtering! Using original "
                "trajectory.");
        return camera_trajectory_;
    }

    return filtered_trajectory;
}

}  // namespace cloudViewer
