// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "controllers/da3_depth_controller.h"

#include "base/camera.h"
#include "base/database.h"
#include "base/image.h"
#include "base/point3d.h"
#include "base/reconstruction.h"
#include "util/logging.h"
#include "util/misc.h"
#include "util/threading.h"

#include "util/reconstruction_log.h"

#ifdef AICore_ENABLED
#include "aicore/backend_capi.h"
#include "aicore/depth_capi.h"
#endif

#ifdef COLMAP_DOWNLOAD_ENABLED
#include "util/download.h"
#endif

#include "util/bitmap.h"

#include <Eigen/Core>

#include "mvs/consistency_graph.h"
#include "mvs/image.h"

#include <filesystem>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace colmap {

const std::map<std::pair<DA3ModelType, DA3QuantType>, std::string> kModelFilenames = {
    {{DA3ModelType::BASE, DA3QuantType::F16}, "depth-anything-base-f16.gguf"},
    {{DA3ModelType::BASE, DA3QuantType::Q8_0}, "depth-anything-base-q8_0.gguf"},
    {{DA3ModelType::BASE, DA3QuantType::Q4_K}, "depth-anything-base-q4_k.gguf"},
    {{DA3ModelType::LARGE, DA3QuantType::Q8_0}, "depth-anything-large-q8_0.gguf"},
    {{DA3ModelType::LARGE, DA3QuantType::Q4_K}, "depth-anything-large-q4_k.gguf"},
    {{DA3ModelType::GIANT, DA3QuantType::Q8_0}, "depth-anything-giant-q8_0.gguf"},
    {{DA3ModelType::GIANT, DA3QuantType::Q4_K}, "depth-anything-giant-q4_k.gguf"},
    {{DA3ModelType::NESTED_ANYVIEW, DA3QuantType::Q8_0}, "depth-anything-nested-anyview-q8_0.gguf"},
    {{DA3ModelType::NESTED_ANYVIEW, DA3QuantType::Q4_K}, "depth-anything-nested-anyview-q4_k.gguf"},
    {{DA3ModelType::NESTED_METRIC, DA3QuantType::F32}, "depth-anything-nested-metric.gguf"},
};

const std::string kDA3DownloadBaseURL =
    "https://github.com/Asher-1/cloudViewer_downloads/releases/download/DA3/";

namespace {

// Normals in COLMAP camera frame (x-right, y-down, z-forward) using calibrated
// intrinsics and 3-point cross products on back-projected surface points.
std::vector<float> ComputeNormalsFromDepthWithIntrinsics(const float* depth,
                                                         int w,
                                                         int h,
                                                         float fx,
                                                         float fy,
                                                         float cx,
                                                         float cy);

// Estimate per-pixel surface normals from a depth map via central finite
// differences in pixel space (legacy fallback).
std::vector<float> ComputeNormalsFromDepth(const float* depth, int w, int h) {
    return ComputeNormalsFromDepthWithIntrinsics(depth, w, h, 1.0f, 1.0f, 0.0f,
                                                 0.0f);
}

// Normals in COLMAP camera frame (x-right, y-down, z-forward) using calibrated
// intrinsics and 3-point cross products on back-projected surface points.
std::vector<float> ComputeNormalsFromDepthWithIntrinsics(const float* depth,
                                                         int w,
                                                         int h,
                                                         float fx,
                                                         float fy,
                                                         float cx,
                                                         float cy) {
    const size_t num_pixels = static_cast<size_t>(w) * h;
    std::vector<float> normals(num_pixels * 3, 0.0f);

    auto idx = [w](int row, int col) {
        return static_cast<size_t>(row) * w + col;
    };

    auto unproject = [&](int col, int row, float d) -> Eigen::Vector3f {
        if (d <= 0.0f || !std::isfinite(d)) {
            return Eigen::Vector3f::Zero();
        }
        return Eigen::Vector3f((static_cast<float>(col) - cx) * d / fx,
                               (static_cast<float>(row) - cy) * d / fy, d);
    };

    for (int row = 0; row < h; ++row) {
        for (int col = 0; col < w; ++col) {
            const size_t p = idx(row, col);
            const float d = depth[p];
            if (d <= 0.0f || !std::isfinite(d)) {
                normals[p * 3 + 0] = 0.0f;
                normals[p * 3 + 1] = 0.0f;
                normals[p * 3 + 2] = 1.0f;
                continue;
            }

            const Eigen::Vector3f center = unproject(col, row, d);
            Eigen::Vector3f right = Eigen::Vector3f::Zero();
            Eigen::Vector3f down = Eigen::Vector3f::Zero();
            if (col + 1 < w) {
                const float d_right = depth[idx(row, col + 1)];
                right = unproject(col + 1, row, d_right);
            }
            if (row + 1 < h) {
                const float d_down = depth[idx(row + 1, col)];
                down = unproject(col, row + 1, d_down);
            }

            Eigen::Vector3f normal = Eigen::Vector3f::Zero();
            if (right.norm() > 0.0f && down.norm() > 0.0f) {
                normal = (right - center).cross(down - center);
            } else if (col > 0 && row + 1 < h) {
                const float d_left = depth[idx(row, col - 1)];
                const float d_down = depth[idx(row + 1, col)];
                normal = (center - unproject(col - 1, row, d_left))
                             .cross(unproject(col, row + 1, d_down) - center);
            } else if (col + 1 < w && row > 0) {
                const float d_right = depth[idx(row, col + 1)];
                right = unproject(col + 1, row, d_right);
                const float d_up = depth[idx(row - 1, col)];
                normal = (right - center).cross(center - unproject(col, row - 1, d_up));
            }

            const float norm = normal.norm();
            if (norm > 1e-10f) {
                normal /= norm;
                if (normal.z() < 0.0f) {
                    normal = -normal;
                }
                normals[p * 3 + 0] = normal.x();
                normals[p * 3 + 1] = normal.y();
                normals[p * 3 + 2] = normal.z();
            } else {
                normals[p * 3 + 0] = 0.0f;
                normals[p * 3 + 1] = 0.0f;
                normals[p * 3 + 2] = 1.0f;
            }
        }
    }
    return normals;
}

bool LookupUndistortedCameraIntrinsics(const Reconstruction& reconstruction,
                                       const std::string& image_name,
                                       int out_w,
                                       int out_h,
                                       float* fx,
                                       float* fy,
                                       float* cx,
                                       float* cy) {
    for (const auto image_id : reconstruction.RegImageIds()) {
        const Image& image = reconstruction.Image(image_id);
        if (image.Name() != image_name) {
            continue;
        }
        const Camera& camera = reconstruction.Camera(image.CameraId());
        if (camera.Width() <= 0 || camera.Height() <= 0) {
            return false;
        }
        const double scale_x =
            static_cast<double>(out_w) / static_cast<double>(camera.Width());
        const double scale_y =
            static_cast<double>(out_h) / static_cast<double>(camera.Height());
        *fx = static_cast<float>(camera.FocalLengthX() * scale_x);
        *fy = static_cast<float>(camera.FocalLengthY() * scale_y);
        *cx = static_cast<float>(camera.PrincipalPointX() * scale_x);
        *cy = static_cast<float>(camera.PrincipalPointY() * scale_y);
        return true;
    }
    return false;
}

bool IsImageFile(const std::string& path) {
    return HasFileExtension(path, ".jpg") || HasFileExtension(path, ".jpeg") ||
           HasFileExtension(path, ".png") || HasFileExtension(path, ".bmp") ||
           HasFileExtension(path, ".tiff") || HasFileExtension(path, ".tif") ||
           HasFileExtension(path, ".webp") || HasFileExtension(path, ".gif") ||
           HasFileExtension(path, ".ppm") || HasFileExtension(path, ".pgm") ||
           HasFileExtension(path, ".pbm");
}

}  // namespace

std::vector<DA3ImageEntry> CollectDA3ImageEntries(
    const std::string& image_root) {
    const std::string root =
        EnsureTrailingSlash(StringReplace(image_root, "\\", "/"));
    std::vector<std::string> files = GetRecursiveFileList(root);
    std::sort(files.begin(), files.end());

    std::vector<DA3ImageEntry> entries;
    entries.reserve(files.size());
    for (const auto& abs_path : files) {
        if (!IsImageFile(abs_path)) {
            continue;
        }
        std::string rel = abs_path;
        rel = StringReplace(rel, "\\", "/");
        if (rel.compare(0, root.size(), root) == 0) {
            rel = rel.substr(root.size());
        }
        entries.push_back({abs_path, rel});
    }
    return entries;
}

bool DA3ConfigsMatchForStereoReuse(const DA3Config& sparse,
                                   const DA3Config& stereo) {
    return sparse.model_type == stereo.model_type &&
           sparse.quant_type == stereo.quant_type &&
           sparse.model_path == stereo.model_path &&
           sparse.metric_model_path == stereo.metric_model_path &&
           DA3ModelSupportsStereo(stereo.model_type);
}

bool DA3OutputsAreStale(const std::string& image_root,
                        const std::string& output_marker_path,
                        bool force_recompute) {
    if (force_recompute) {
        return true;
    }
    if (!ExistsFile(output_marker_path) && !ExistsDir(output_marker_path)) {
        return true;
    }

    std::error_code ec;
    std::filesystem::file_time_type marker_time;
    if (ExistsFile(output_marker_path)) {
        marker_time = std::filesystem::last_write_time(output_marker_path, ec);
    } else {
        marker_time = std::filesystem::last_write_time(output_marker_path, ec);
    }
    if (ec) {
        return true;
    }

    for (const auto& entry : CollectDA3ImageEntries(image_root)) {
        const auto src_time =
            std::filesystem::last_write_time(entry.abs_path, ec);
        if (!ec && src_time > marker_time) {
            return true;
        }
    }
    return false;
}

bool DA3StereoDepthMapsReady(const std::string& dense_path) {
    return ColmapGeometricDepthMapsReady(dense_path);
}

namespace {

std::vector<std::string> FusionConfigImageNames(const std::string& dense_path) {
    const auto fusion_cfg = JoinPaths(dense_path, "stereo", "fusion.cfg");
    if (!ExistsFile(fusion_cfg)) {
        return {};
    }
    std::vector<std::string> names;
    for (const auto& line : ReadTextFileLines(fusion_cfg)) {
        std::string trimmed = line;
        StringTrim(&trimmed);
        if (!trimmed.empty() && trimmed[0] != '#') {
            names.push_back(trimmed);
        }
    }
    return names;
}

bool StereoDepthMapsReadyWithSuffix(const std::string& dense_path,
                                    const char* suffix) {
    const auto image_names = FusionConfigImageNames(dense_path);
    if (image_names.empty()) {
        return false;
    }

    const auto depth_dir = JoinPaths(dense_path, "stereo", "depth_maps");
    const auto normal_dir = JoinPaths(dense_path, "stereo", "normal_maps");
    if (!ExistsDir(depth_dir) || !ExistsDir(normal_dir)) {
        return false;
    }

    constexpr std::uintmax_t kMinMapBytes = 128;
    for (const auto& name : image_names) {
        const std::string map_name = name + suffix;
        const auto depth_path = JoinPaths(depth_dir, map_name);
        const auto normal_path = JoinPaths(normal_dir, map_name);
        if (!ExistsFile(depth_path) || !ExistsFile(normal_path)) {
            return false;
        }
        std::error_code ec;
        const auto depth_size = std::filesystem::file_size(depth_path, ec);
        if (ec || depth_size < kMinMapBytes) {
            return false;
        }
        const auto normal_size = std::filesystem::file_size(normal_path, ec);
        if (ec || normal_size < kMinMapBytes) {
            return false;
        }
    }
    return true;
}

std::filesystem::file_time_type NewestFileTimeWithSuffix(
        const std::string& dir_path, const char* suffix) {
    std::filesystem::file_time_type newest{};
    bool found = false;
    std::error_code ec;
    if (!ExistsDir(dir_path)) {
        return newest;
    }
    for (const auto& entry :
         std::filesystem::directory_iterator(dir_path, ec)) {
        if (ec || !entry.is_regular_file()) {
            continue;
        }
        const std::string name = entry.path().filename().string();
        if (name.find(suffix) == std::string::npos) {
            continue;
        }
        const auto t = entry.last_write_time(ec);
        if (ec) {
            continue;
        }
        if (!found || t > newest) {
            newest = t;
            found = true;
        }
    }
    return newest;
}

std::filesystem::file_time_type OldestFileTimeWithSuffix(
        const std::string& dir_path, const char* suffix) {
    std::filesystem::file_time_type oldest{};
    bool found = false;
    std::error_code ec;
    if (!ExistsDir(dir_path)) {
        return oldest;
    }
    for (const auto& entry :
         std::filesystem::directory_iterator(dir_path, ec)) {
        if (ec || !entry.is_regular_file()) {
            continue;
        }
        const std::string name = entry.path().filename().string();
        if (name.find(suffix) == std::string::npos) {
            continue;
        }
        const auto t = entry.last_write_time(ec);
        if (ec) {
            continue;
        }
        if (!found || t < oldest) {
            oldest = t;
            found = true;
        }
    }
    return oldest;
}

void RemoveStereoMapsWithSuffix(const std::string& dense_path,
                                const char* suffix) {
    const auto stereo_path = JoinPaths(dense_path, "stereo");
    for (const char* sub : {"depth_maps", "normal_maps", "consistency_graphs"}) {
        const auto dir_path = JoinPaths(stereo_path, sub);
        if (!ExistsDir(dir_path)) {
            continue;
        }
        std::error_code ec;
        for (const auto& entry :
             std::filesystem::directory_iterator(dir_path, ec)) {
            if (ec || !entry.is_regular_file()) {
                continue;
            }
            const std::string name = entry.path().filename().string();
            if (name.find(suffix) != std::string::npos) {
                std::filesystem::remove(entry.path(), ec);
            }
        }
    }
}

}  // namespace

bool DA3DepthPriorReady(const std::string& dense_path) {
    return StereoDepthMapsReadyWithSuffix(dense_path, ".photometric.bin");
}

bool ColmapGeometricDepthMapsReady(const std::string& dense_path) {
    return StereoDepthMapsReadyWithSuffix(dense_path, ".geometric.bin");
}

bool DA3PatchMatchRefineStale(const std::string& dense_path) {
    if (!DA3DepthPriorReady(dense_path)) {
        return true;
    }
    if (!ColmapGeometricDepthMapsReady(dense_path)) {
        return true;
    }
    const auto depth_dir = JoinPaths(dense_path, "stereo", "depth_maps");
    const auto newest_prior =
        NewestFileTimeWithSuffix(depth_dir, ".photometric.bin");
    const auto oldest_geometric =
        OldestFileTimeWithSuffix(depth_dir, ".geometric.bin");
    return newest_prior > oldest_geometric;
}

void RemoveColmapGeometricStereoMaps(const std::string& dense_path) {
    RemoveStereoMapsWithSuffix(dense_path, ".geometric.bin");
}

void WriteDA3PlaceholderDatabase(const std::string& database_path,
                                 const Reconstruction& reconstruction) {
    Database database(database_path);
    std::unordered_map<camera_t, camera_t> camera_id_map;
    for (const auto& [camera_id, camera] : reconstruction.Cameras()) {
        camera_id_map[camera_id] = database.WriteCamera(camera);
    }
    for (const auto image_id : reconstruction.RegImageIds()) {
        Image image = reconstruction.Image(image_id);
        image.SetCameraId(camera_id_map.at(image.CameraId()));
        database.WriteImage(image);
    }
}

bool SyncWorkspaceSparseFromDense(const std::string& workspace_path,
                                  const std::string& dense_path,
                                  int reconstruction_index) {
    std::string src_sparse = JoinPaths(dense_path, "sparse");
    if (!ExistsFile(JoinPaths(src_sparse, "images.bin")) &&
        !ExistsFile(JoinPaths(src_sparse, "images.txt"))) {
        src_sparse = JoinPaths(dense_path, "sparse", "0");
    }
    if (!ExistsFile(JoinPaths(src_sparse, "images.bin")) &&
        !ExistsFile(JoinPaths(src_sparse, "images.txt"))) {
        LOG(ERROR) << "DA3: no dense sparse model to sync from " << dense_path;
        return false;
    }

    const std::string dst_sparse =
        JoinPaths(workspace_path, "sparse", std::to_string(reconstruction_index));
    CreateDirIfNotExists(JoinPaths(workspace_path, "sparse"));

    std::error_code ec;
    std::filesystem::remove_all(dst_sparse, ec);
    std::filesystem::create_directories(dst_sparse, ec);

    static const char* kSparseFiles[] = {
        "cameras.bin", "images.bin", "points3D.bin",
        "cameras.txt", "images.txt", "points3D.txt",
        "project.ini",
    };
    size_t copied = 0;
    for (const char* name : kSparseFiles) {
        const std::string src = JoinPaths(src_sparse, name);
        if (!ExistsFile(src)) {
            continue;
        }
        std::filesystem::copy(
            src, JoinPaths(dst_sparse, name),
            std::filesystem::copy_options::overwrite_existing, ec);
        if (!ec) {
            ++copied;
        }
    }
    if (copied == 0) {
        LOG(ERROR) << "DA3: dense sparse model at " << src_sparse
                   << " contains no COLMAP files";
        return false;
    }

    {
        std::ofstream marker(JoinPaths(dst_sparse, ".da3_undistorted_sync"),
                             std::ios::trunc);
        marker << "synced_from=" << dense_path << '\n';
    }

    RECON_LOG_DEBUG("DA3: synced workspace sparse/%d from %s (%zu files)\n", reconstruction_index, src_sparse.c_str(), copied);
    return true;
}

size_t CountDA3Images(const std::string& image_root) {
    return CollectDA3ImageEntries(image_root).size();
}

int ComputeDA3ImgResizeTarget(const std::vector<std::string>& image_paths,
                              int max_image_size) {
    constexpr int kPatchSize = 14;
    constexpr int kDefaultTarget = 504;

    int long_edge = 0;
    for (const auto& path : image_paths) {
        Bitmap bitmap;
        if (!bitmap.Read(path, false)) {
            continue;
        }
        long_edge = std::max(
            long_edge, static_cast<int>(
                           std::max(bitmap.Width(), bitmap.Height())));
    }
    if (long_edge <= 0) {
        return kDefaultTarget;
    }
    if (max_image_size > 0) {
        long_edge = std::min(long_edge, max_image_size);
    }
    long_edge = std::max(kDefaultTarget, (long_edge / kPatchSize) * kPatchSize);
    return long_edge;
}

bool WriteExifPlaceholderSparseModel(const std::string& image_root,
                                     const std::string& sparse_output_path,
                                     double default_focal_length_factor) {
    const auto entries = CollectDA3ImageEntries(image_root);
    if (entries.empty()) {
        LOG(ERROR) << "EXIF bootstrap: no images under " << image_root;
        return false;
    }

    Reconstruction reconstruction;
    using CameraKey = std::tuple<size_t, size_t, int64_t>;
    std::map<CameraKey, camera_t> camera_key_to_id;
    camera_t next_camera_id = 1;
    image_t next_image_id = 1;
    size_t registered = 0;

    for (const auto& entry : entries) {
        Bitmap bitmap;
        if (!bitmap.Read(entry.abs_path, false)) {
            LOG(WARNING) << "EXIF bootstrap: failed to read " << entry.abs_path;
            continue;
        }

        const size_t width = bitmap.Width();
        const size_t height = bitmap.Height();
        double focal_length = 0.0;
        if (!bitmap.ExifFocalLength(&focal_length)) {
            focal_length = default_focal_length_factor *
                           static_cast<double>(std::max(width, height));
        }

        const CameraKey key = {width, height,
                               static_cast<int64_t>(std::llround(
                                   focal_length * 1000.0))};
        camera_t camera_id = kInvalidCameraId;
        const auto cached = camera_key_to_id.find(key);
        if (cached != camera_key_to_id.end()) {
            camera_id = cached->second;
        } else {
            Camera camera;
            camera.SetCameraId(next_camera_id++);
            camera.InitializeWithName("SIMPLE_PINHOLE", focal_length, width,
                                      height);
            if (!camera.VerifyParams()) {
                LOG(WARNING) << "EXIF bootstrap: invalid camera for "
                             << entry.colmap_name;
                continue;
            }
            reconstruction.AddCamera(camera);
            camera_id = camera.CameraId();
            camera_key_to_id.emplace(key, camera_id);
        }

        Image image;
        image.SetImageId(next_image_id++);
        image.SetName(entry.colmap_name);
        image.SetCameraId(camera_id);
        image.Qvec(0) = 1.0;
        image.Qvec(1) = 0.0;
        image.Qvec(2) = 0.0;
        image.Qvec(3) = 0.0;
        image.SetTvec(Eigen::Vector3d::Zero());
        reconstruction.AddImage(image);
        reconstruction.RegisterImage(image.ImageId());
        ++registered;
    }

    if (registered == 0) {
        LOG(ERROR) << "EXIF bootstrap: no images registered";
        return false;
    }

    if (!ExistsDir(sparse_output_path)) {
        boost::filesystem::create_directories(sparse_output_path);
    }
    reconstruction.Write(sparse_output_path);

    {
        std::ofstream marker(JoinPaths(sparse_output_path, ".da3_exif_bootstrap"),
                             std::ios::trunc);
        marker << "exif_placeholder=1\n";
    }

    RECON_LOG_DEBUG("DA3 EXIF bootstrap: wrote placeholder sparse model (%d images, %zu cameras) to %s\n",
                    registered, camera_key_to_id.size(), sparse_output_path.c_str());
    return true;
}

namespace {

std::vector<std::string> CollectImagePaths(const std::string& image_dir) {
    std::vector<std::string> paths;
    for (const auto& entry : CollectDA3ImageEntries(image_dir)) {
        paths.push_back(entry.abs_path);
    }
    return paths;
}

struct DepthPoseMultiResult {
    int h = 0;
    int w = 0;
    int n = 0;
    std::vector<float> depth;
    std::vector<float> ext;
    std::vector<float> intr;
};

using StereoExportProgressCallback =
    std::function<void(int current, int total, const std::string& status)>;

bool WriteStereoMapsFromMultiview(
    const std::string& output_path,
    const std::vector<std::string>& undist_image_paths,
    const std::vector<std::string>& fusion_colmap_names,
    const DepthPoseMultiResult& multi,
    const StereoExportProgressCallback& progress_cb,
    const std::function<bool()>& is_stopped,
    bool use_colmap_poses_only,
    bool export_photometric_prior,
    int export_max_image_size = -1,
    bool fast_depth_export = false);

#ifdef AICore_ENABLED
void LogDA3InferenceDevice(aicore_depth_ctx* ctx, const char* requested_device) {
    if (!ctx) {
        return;
    }
    const char* requested =
        (requested_device && requested_device[0]) ? requested_device : "auto";
    const char* active = aicore_depth_device_name(ctx);
    if (!active || !active[0]) {
        active = "cpu";
    }
    if (std::strcmp(requested, "auto") == 0) {
        RECON_LOG_INFO(
            "DA3: Using device: auto (%s), active backend: %s\n",
            aicore_auto_device_order(), active);
    } else {
        RECON_LOG_INFO("DA3: Using device: %s, active backend: %s\n",
                       requested, active);
    }
}

aicore_depth_ctx* LoadDA3Context(const DA3Config& config, int n_threads) {
    const char* requested_device = std::getenv("DA_DEVICE");
    if (!requested_device || requested_device[0] == '\0') {
        requested_device = "auto";
    }
    aicore_depth_ctx* ctx = nullptr;
    if (config.model_type != DA3ModelType::NESTED_METRIC &&
        config.model_type != DA3ModelType::NESTED_ANYVIEW) {
        const std::string model_path = DA3DepthController::ResolveModelPath(config);
        if (model_path.empty()) {
            return nullptr;
        }
        ctx = aicore_depth_load_device(model_path.c_str(), n_threads,
                                       requested_device);
    } else {
        std::string anyview_path;
        std::string metric_path;

        if (config.model_type == DA3ModelType::NESTED_ANYVIEW) {
            anyview_path = DA3DepthController::ResolveModelPath(config);
            metric_path =
                config.metric_model_path.empty()
                    ? DA3DepthController::ResolveModelPath(
                          DA3Config{DA3ModelType::NESTED_METRIC, DA3QuantType::F32})
                    : config.metric_model_path;
        } else {
            // NESTED_METRIC UI option selects the metric-branch GGUF; anyview is required.
            metric_path = DA3DepthController::ResolveModelPath(config);
            DA3Config anyview_config;
            anyview_config.model_type = DA3ModelType::NESTED_ANYVIEW;
            anyview_config.quant_type = DA3QuantType::Q8_0;
            if (DA3ModelExists(DA3ModelType::NESTED_ANYVIEW, config.quant_type)) {
                anyview_config.quant_type = config.quant_type;
            }
            anyview_path = DA3DepthController::ResolveModelPath(anyview_config);
        }

        if (anyview_path.empty() || metric_path.empty()) {
            return nullptr;
        }
        ctx = aicore_depth_load_nested_device(anyview_path.c_str(),
                                              metric_path.c_str(), n_threads,
                                              requested_device);
    }
    if (ctx) {
        LogDA3InferenceDevice(ctx, requested_device);
    }
    return ctx;
}

float BilinearSampleDepth(const float* depth, int w, int h, float x, float y) {
    x = std::max(0.0f, std::min(static_cast<float>(w - 1), x));
    y = std::max(0.0f, std::min(static_cast<float>(h - 1), y));
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const int x1 = std::min(x0 + 1, w - 1);
    const int y1 = std::min(y0 + 1, h - 1);
    const float dx = x - static_cast<float>(x0);
    const float dy = y - static_cast<float>(y0);

    const auto at = [depth, w](int row, int col) {
        return depth[static_cast<size_t>(row) * w + col];
    };

    const float v00 = at(y0, x0);
    const float v01 = at(y0, x1);
    const float v10 = at(y1, x0);
    const float v11 = at(y1, x1);
    const float v0 = v00 * (1.0f - dx) + v01 * dx;
    const float v1 = v10 * (1.0f - dx) + v11 * dx;
    return v0 * (1.0f - dy) + v1 * dy;
}

std::vector<float> UpsampleDepthBilinear(const float* src, int src_w, int src_h,
                                         int dst_w, int dst_h) {
    const size_t dst_size = static_cast<size_t>(dst_w) * static_cast<size_t>(dst_h);
    std::vector<float> dst(dst_size);
    if (src_w <= 0 || src_h <= 0 || dst_w <= 0 || dst_h <= 0) {
        return dst;
    }
    if (src_w == dst_w && src_h == dst_h) {
        std::copy(src, src + static_cast<size_t>(src_w) * src_h, dst.begin());
        return dst;
    }

    const float scale_x = static_cast<float>(src_w) / static_cast<float>(dst_w);
    const float scale_y = static_cast<float>(src_h) / static_cast<float>(dst_h);
    for (int row = 0; row < dst_h; ++row) {
        for (int col = 0; col < dst_w; ++col) {
            const float src_x = (static_cast<float>(col) + 0.5f) * scale_x - 0.5f;
            const float src_y = (static_cast<float>(row) + 0.5f) * scale_y - 0.5f;
            dst[static_cast<size_t>(row) * dst_w + col] =
                BilinearSampleDepth(src, src_w, src_h, src_x, src_y);
        }
    }
    return dst;
}

std::vector<float> UpsampleDepthGuided(const float* src, int src_w, int src_h,
                                       const Bitmap* guide_rgb, int dst_w,
                                       int dst_h) {
    if (src_w == dst_w && src_h == dst_h) {
        const size_t n = static_cast<size_t>(src_w) * static_cast<size_t>(src_h);
        return std::vector<float>(src, src + n);
    }
    if (!guide_rgb || guide_rgb->Width() != static_cast<size_t>(dst_w) ||
        guide_rgb->Height() != static_cast<size_t>(dst_h)) {
        return UpsampleDepthBilinear(src, src_w, src_h, dst_w, dst_h);
    }

    std::vector<float> dst =
        UpsampleDepthBilinear(src, src_w, src_h, dst_w, dst_h);
    if (src_w <= 0 || src_h <= 0 || dst_w <= 0 || dst_h <= 0) {
        return dst;
    }

    const float scale_x = static_cast<float>(src_w) / static_cast<float>(dst_w);
    const float scale_y = static_cast<float>(src_h) / static_cast<float>(dst_h);
    constexpr int kRadius = 2;
    constexpr float kSigmaSpatial = 2.0f;
    constexpr float kSigmaColor = 0.12f;
    const float inv_sigma_s2 = 1.0f / (2.0f * kSigmaSpatial * kSigmaSpatial);
    const float inv_sigma_c2 = 1.0f / (2.0f * kSigmaColor * kSigmaColor);

    for (int row = 0; row < dst_h; ++row) {
        for (int col = 0; col < dst_w; ++col) {
            BitmapColor<uint8_t> center_color;
            if (!guide_rgb->GetPixel(col, row, &center_color)) {
                continue;
            }
            const float center_r = center_color.r / 255.0f;
            const float center_g = center_color.g / 255.0f;
            const float center_b = center_color.b / 255.0f;

            const float src_x =
                (static_cast<float>(col) + 0.5f) * scale_x - 0.5f;
            const float src_y =
                (static_cast<float>(row) + 0.5f) * scale_y - 0.5f;
            const int center_sx = static_cast<int>(std::lround(src_x));
            const int center_sy = static_cast<int>(std::lround(src_y));

            float sum_w = 0.0f;
            float sum_d = 0.0f;
            for (int dy = -kRadius; dy <= kRadius; ++dy) {
                for (int dx = -kRadius; dx <= kRadius; ++dx) {
                    const int sx = center_sx + dx;
                    const int sy = center_sy + dy;
                    if (sx < 0 || sy < 0 || sx >= src_w || sy >= src_h) {
                        continue;
                    }
                    const float depth =
                        src[static_cast<size_t>(sy) * src_w +
                            static_cast<size_t>(sx)];
                    if (depth <= 0.0f || !std::isfinite(depth)) {
                        continue;
                    }

                    const int gx = std::max(
                        0, std::min(dst_w - 1,
                                    static_cast<int>(std::lround(
                                        (static_cast<float>(sx) + 0.5f) /
                                            scale_x -
                                        0.5f))));
                    const int gy = std::max(
                        0, std::min(dst_h - 1,
                                    static_cast<int>(std::lround(
                                        (static_cast<float>(sy) + 0.5f) /
                                            scale_y -
                                        0.5f))));
                    BitmapColor<uint8_t> sample_color;
                    if (!guide_rgb->GetPixel(gx, gy, &sample_color)) {
                        continue;
                    }
                    const float dr =
                        sample_color.r / 255.0f - center_r;
                    const float dg =
                        sample_color.g / 255.0f - center_g;
                    const float db =
                        sample_color.b / 255.0f - center_b;
                    const float spatial_w =
                        std::exp(-static_cast<float>(dx * dx + dy * dy) *
                                 inv_sigma_s2);
                    const float color_w =
                        std::exp(-(dr * dr + dg * dg + db * db) * inv_sigma_c2);
                    const float weight = spatial_w * color_w;
                    sum_w += weight;
                    sum_d += weight * depth;
                }
            }
            if (sum_w > 0.0f) {
                dst[static_cast<size_t>(row) * dst_w +
                    static_cast<size_t>(col)] = sum_d / sum_w;
            }
        }
    }
    return dst;
}

void FilterMetricDepthInPlace(std::vector<float>& depth,
                              float min_depth = 0.01f,
                              float max_depth = 100.0f) {
    for (float& d : depth) {
        if (d <= min_depth || d > max_depth || !std::isfinite(d)) {
            d = 0.0f;
        }
    }
}

void CapExportDimensions(int* width, int* height, int export_max_image_size) {
    if (export_max_image_size <= 0 || width == nullptr || height == nullptr) {
        return;
    }
    const int max_dim = std::max(*width, *height);
    if (max_dim <= export_max_image_size) {
        return;
    }
    const double scale =
        static_cast<double>(export_max_image_size) / static_cast<double>(max_dim);
    *width = std::max(1, static_cast<int>(std::lround(*width * scale)));
    *height = std::max(1, static_cast<int>(std::lround(*height * scale)));
}

void WriteStereoFusionConfig(const std::string& stereo_path,
                             const std::vector<std::string>& image_names) {
    std::ofstream file(JoinPaths(stereo_path, "fusion.cfg"), std::ios::trunc);
    if (!file) {
        return;
    }
    for (const auto& name : image_names) {
        file << name << '\n';
    }
}

bool RunDepthPoseMulti(aicore_depth_ctx* ctx, const std::vector<std::string>& image_paths,
                       DepthPoseMultiResult& result, int max_image_size) {
    if (!ctx || image_paths.empty()) {
        return false;
    }

    const int N = static_cast<int>(image_paths.size());
    const int requested =
        ComputeDA3ImgResizeTarget(image_paths, max_image_size);
    int target = aicore_depth_cap_img_resize_target(ctx, requested);
    if (target < requested) {
        DA3NoteVramCap(requested, target);
        RECON_LOG_DEBUG("DA3: VRAM-safe img_resize_target %zu -> %zu (single-view peak, not view count)\n", requested, target);
    }

    constexpr int kMinTarget = 504;
    constexpr int kPatchStep = 28;

    while (target >= kMinTarget) {
        aicore_depth_set_img_resize_target(ctx, target);
        RECON_LOG_DEBUG("DA3: sequential per-view inference on single backend, img_resize_target=%zu (%zu images)\n", target, N);

        result.ext.assign(static_cast<size_t>(N) * 12, 0.f);
        result.intr.assign(static_cast<size_t>(N) * 9, 0.f);
        result.n = N;
        result.h = 0;
        result.w = 0;
        result.depth.clear();

        bool oom = false;
        for (int i = 0; i < N; ++i) {
            RECON_LOG_DEBUG("DA3: infer view %d/%d\n", i + 1, N);

            int h = 0;
            int w = 0;
            float* depth_ptr = nullptr;
            float ext[12] = {};
            float intr[9] = {};
            int is_metric = 0;
            if (aicore_depth_depth_dense(
                    ctx, image_paths[static_cast<size_t>(i)].c_str(), &h, &w,
                    &depth_ptr, nullptr, nullptr, ext, intr, &is_metric) != 0) {
                if (depth_ptr) {
                    aicore_depth_free_floats(depth_ptr);
                }
                const std::string err = aicore_depth_last_error(ctx);
                LOG(ERROR) << "DA3: per-view inference failed for "
                           << image_paths[static_cast<size_t>(i)] << ": "
                           << err;
                oom = err.find("galloc") != std::string::npos ||
                      err.find("out of memory") != std::string::npos ||
                      err.find("OOM") != std::string::npos ||
                      err.find("CUDA out of memory") != std::string::npos;
                aicore_depth_release_gpu_working_memory(ctx);
                break;
            }
            if (h <= 0 || w <= 0 || !depth_ptr) {
                aicore_depth_free_floats(depth_ptr);
                LOG(ERROR) << "DA3: empty depth for view " << i;
                aicore_depth_release_gpu_working_memory(ctx);
                oom = true;
                break;
            }

            if (i == 0) {
                result.h = h;
                result.w = w;
            } else if (h != result.h || w != result.w) {
                aicore_depth_free_floats(depth_ptr);
                LOG(ERROR) << "DA3: view " << i << " size mismatch (" << w << "x"
                           << h << " vs " << result.w << "x" << result.h << ")";
                return false;
            }

            const size_t per_view =
                static_cast<size_t>(h) * static_cast<size_t>(w);
            result.depth.insert(result.depth.end(), depth_ptr,
                                depth_ptr + per_view);
            std::memcpy(result.ext.data() + static_cast<size_t>(i) * 12, ext,
                        12 * sizeof(float));
            std::memcpy(result.intr.data() + static_cast<size_t>(i) * 9, intr,
                        9 * sizeof(float));
            aicore_depth_free_floats(depth_ptr);
            aicore_depth_release_gpu_working_memory(ctx);
        }

        if (static_cast<int>(result.depth.size()) ==
            N * result.h * result.w) {
            return true;
        }

        const int next = std::max(kMinTarget, target - kPatchStep);
        if (!oom || next >= target) {
            return false;
        }
        RECON_LOG_WARN("WARNING: DA3 GPU OOM at img_resize_target=%d; retrying all views at %d\n",
                       target, next);
        target = next;
    }

    return false;
}

bool WriteDenseSparseFromMultiview(
        aicore_depth_ctx* ctx,
        const std::string& dense_path,
        const std::vector<std::string>& image_paths,
        const std::vector<std::string>& image_names,
        const DepthPoseMultiResult& multi) {
    if (!ctx || image_paths.empty() ||
        static_cast<int>(image_paths.size()) != multi.n ||
        static_cast<int>(image_names.size()) != multi.n) {
        return false;
    }

    const std::string sparse_dir = JoinPaths(dense_path, "sparse");
    CreateDirIfNotExists(sparse_dir);

    std::vector<const char*> cpaths(static_cast<size_t>(multi.n));
    std::vector<const char*> cnames(static_cast<size_t>(multi.n));
    for (int i = 0; i < multi.n; ++i) {
        cpaths[static_cast<size_t>(i)] = image_paths[static_cast<size_t>(i)].c_str();
        cnames[static_cast<size_t>(i)] = image_names[static_cast<size_t>(i)].c_str();
    }

    if (aicore_depth_write_colmap_from_multiview(
            ctx, cpaths.data(), cnames.data(), multi.n, multi.depth.data(),
            multi.ext.data(), multi.intr.data(), multi.h, multi.w,
            sparse_dir.c_str(), 1) != 0) {
        LOG(ERROR) << "DA3: failed to sync dense sparse model: "
                   << aicore_depth_last_error(ctx);
        RECON_LOG_ERROR("ERROR: DA3 dense sparse sync failed: %s\n", aicore_depth_last_error(ctx));
        return false;
    }

    RECON_LOG_DEBUG("DA3 depth export: synced dense sparse model (cameras/poses) to %s\n", sparse_dir.c_str());
    return true;
}

void WriteColmapDepthMap(const std::string& path, int w, int h,
                         const float* depth) {
    std::ofstream ofs(path);
    if (!ofs) {
        return;
    }
    ofs << w << "&" << h << "&" << 1 << "&";
    ofs.close();
    std::ofstream bin(path, std::ios::binary | std::ios::app);
    if (bin) {
        bin.write(reinterpret_cast<const char*>(depth),
                  static_cast<std::streamsize>(sizeof(float) * w * h));
    }
}

void WriteColmapNormalMap(const std::string& path, int w, int h,
                          const std::vector<float>& normals) {
    std::ofstream ofs(path);
    if (!ofs) {
        return;
    }
    ofs << w << "&" << h << "&" << 3 << "&";
    ofs.close();
    std::ofstream bin(path, std::ios::binary | std::ios::app);
    if (bin) {
        bin.write(reinterpret_cast<const char*>(normals.data()),
                  static_cast<std::streamsize>(sizeof(float) * normals.size()));
    }
}

struct DA3ViewCamera {
    Eigen::Matrix<float, 3, 4, Eigen::RowMajor> P;
    Eigen::Matrix<float, 3, 4, Eigen::RowMajor> inv_P;
};

DA3ViewCamera MakeViewCamera(const float* ext12, const float* intr9) {
    DA3ViewCamera cam;
    const float R[9] = {ext12[0], ext12[1], ext12[2], ext12[4], ext12[5],
                        ext12[6], ext12[8], ext12[9], ext12[10]};
    const float T[3] = {ext12[3], ext12[7], ext12[11]};
    mvs::ComposeProjectionMatrix(intr9, R, T, cam.P.data());
    mvs::ComposeInverseProjectionMatrix(intr9, R, T, cam.inv_P.data());
    return cam;
}

float SampleDepthGrid(const float* depth, int w, int h, float col, float row) {
    col = std::max(0.0f, std::min(static_cast<float>(w - 1), col));
    row = std::max(0.0f, std::min(static_cast<float>(h - 1), row));
    const int col_i = static_cast<int>(std::lround(col));
    const int row_i = static_cast<int>(std::lround(row));
    return depth[static_cast<size_t>(row_i) * w + col_i];
}

bool MakeViewCameraFromColmapImage(const Reconstruction& reconstruction,
                                   const std::string& image_name,
                                   int out_w,
                                   int out_h,
                                   DA3ViewCamera* camera) {
    if (camera == nullptr || out_w <= 0 || out_h <= 0) {
        return false;
    }
    for (const auto image_id : reconstruction.RegImageIds()) {
        const Image& image = reconstruction.Image(image_id);
        if (image.Name() != image_name) {
            continue;
        }
        const Camera& cam = reconstruction.Camera(image.CameraId());
        if (cam.Width() <= 0 || cam.Height() <= 0) {
            return false;
        }
        const double scale_x =
            static_cast<double>(out_w) / static_cast<double>(cam.Width());
        const double scale_y =
            static_cast<double>(out_h) / static_cast<double>(cam.Height());
        const float K[9] = {
            static_cast<float>(cam.FocalLengthX() * scale_x),
            0.0f,
            static_cast<float>(cam.PrincipalPointX() * scale_x),
            0.0f,
            static_cast<float>(cam.FocalLengthY() * scale_y),
            static_cast<float>(cam.PrincipalPointY() * scale_y),
            0.0f,
            0.0f,
            1.0f};
        const Eigen::Matrix<float, 3, 3, Eigen::RowMajor> R =
            image.RotationMatrix().cast<float>();
        const Eigen::Vector3f T = image.Tvec().cast<float>();
        mvs::ComposeProjectionMatrix(K, R.data(), T.data(), camera->P.data());
        mvs::ComposeInverseProjectionMatrix(K, R.data(), T.data(),
                                            camera->inv_P.data());
        return true;
    }
    return false;
}

// EXIF bootstrap writes identity rotation + zero translation for every image.
// These are not valid multi-view poses and must not drive depth cleaning/fusion.
bool IsExifPlaceholderReconstruction(const Reconstruction& reconstruction) {
    if (reconstruction.NumRegImages() == 0) {
        return true;
    }
    for (const auto image_id : reconstruction.RegImageIds()) {
        const Image& image = reconstruction.Image(image_id);
        const Eigen::Vector4d& q = image.Qvec();
        const Eigen::Vector3d& t = image.Tvec();
        const bool identity_q =
            std::abs(q(0) - 1.0) < 1e-6 && std::abs(q(1)) < 1e-6 &&
            std::abs(q(2)) < 1e-6 && std::abs(q(3)) < 1e-6;
        if (!identity_q || t.norm() > 1e-6) {
            return false;
        }
    }
    return true;
}

struct ExportedDepthView {
    int w = 0;
    int h = 0;
    std::vector<float> depth;
    DA3ViewCamera camera;
    bool has_camera = false;
    float fx = 1.0f;
    float fy = 1.0f;
    float cx = 0.0f;
    float cy = 0.0f;
};

// COLMAPUndistorter writes undistorted cameras to dense/<i>/sparse/ (no /0).
// DA3 sparse export may write dense/<i>/sparse/0/. Prefer the undistorted model.
std::string FindUndistortedSparseModelPath(const std::string& dense_path) {
    const auto undist_sparse = JoinPaths(dense_path, "sparse");
    if (ExistsFile(JoinPaths(undist_sparse, "images.bin")) ||
        ExistsFile(JoinPaths(undist_sparse, "images.txt"))) {
        return undist_sparse;
    }
    const auto sparse0 = JoinPaths(dense_path, "sparse", "0");
    if (ExistsFile(JoinPaths(sparse0, "images.bin")) ||
        ExistsFile(JoinPaths(sparse0, "images.txt"))) {
        return sparse0;
    }
    return {};
}

size_t CountValidDepthPixels(const std::vector<ExportedDepthView>& views) {
    size_t count = 0;
    for (const auto& view : views) {
        for (float depth : view.depth) {
            if (depth > 0.0f && std::isfinite(depth)) {
                ++count;
            }
        }
    }
    return count;
}

std::vector<int> BuildConsistencyGraphData(
        size_t ref_idx,
        const std::vector<ExportedDepthView>& views,
        float max_depth_error,
        float max_reproj_error,
        int min_partner_views,
        int sample_stride) {
    std::vector<int> data;
    if (ref_idx >= views.size() || views.size() < 2) {
        return data;
    }
    const auto& ref_view = views[ref_idx];
    if (!ref_view.has_camera || ref_view.depth.empty() || ref_view.w <= 0 ||
        ref_view.h <= 0) {
        return data;
    }

    const float max_squared_reproj = max_reproj_error * max_reproj_error;
    sample_stride = std::max(1, sample_stride);

    for (int row = 0; row < ref_view.h; row += sample_stride) {
        for (int col = 0; col < ref_view.w; col += sample_stride) {
            const size_t p =
                static_cast<size_t>(row) * ref_view.w + static_cast<size_t>(col);
            const float depth = ref_view.depth[p];
            if (depth <= 0.0f || !std::isfinite(depth)) {
                continue;
            }

            const Eigen::Vector3f xyz =
                ref_view.camera.inv_P *
                Eigen::Vector4f(col * depth, row * depth, depth, 1.0f);

            std::vector<int> partners;
            partners.reserve(views.size());
            for (size_t tgt_idx = 0; tgt_idx < views.size(); ++tgt_idx) {
                if (tgt_idx == ref_idx) {
                    continue;
                }
                const auto& tgt_view = views[tgt_idx];
                if (!tgt_view.has_camera) {
                    continue;
                }
                const Eigen::Vector3f proj =
                    tgt_view.camera.P *
                    Eigen::Vector4f(xyz.x(), xyz.y(), xyz.z(), 1.0f);
                if (std::abs(proj.z()) <
                    std::numeric_limits<float>::epsilon()) {
                    continue;
                }
                const float col_proj = proj.x() / proj.z();
                const float row_proj = proj.y() / proj.z();
                if (col_proj < 0.0f || row_proj < 0.0f ||
                    col_proj >= static_cast<float>(tgt_view.w) ||
                    row_proj >= static_cast<float>(tgt_view.h)) {
                    continue;
                }
                const float tgt_depth_val = BilinearSampleDepth(
                    tgt_view.depth.data(), tgt_view.w, tgt_view.h, col_proj,
                    row_proj);
                if (tgt_depth_val <= 0.0f || !std::isfinite(tgt_depth_val)) {
                    continue;
                }
                const float depth_error =
                    std::abs((proj.z() - tgt_depth_val) / tgt_depth_val);
                if (depth_error > max_depth_error) {
                    continue;
                }
                const float col_diff = col_proj - std::lround(col_proj);
                const float row_diff = row_proj - std::lround(row_proj);
                if (col_diff * col_diff + row_diff * row_diff >
                    max_squared_reproj) {
                    continue;
                }
                partners.push_back(static_cast<int>(tgt_idx));
            }

            if (static_cast<int>(partners.size()) < min_partner_views) {
                continue;
            }

            data.push_back(col);
            data.push_back(row);
            data.push_back(static_cast<int>(partners.size()));
            for (const int partner : partners) {
                data.push_back(partner);
            }
        }
    }
    return data;
}

// PatchMatch / StereoFusion need vis.dat + patch-match.cfg; consistency graphs
// are only used by legacy photometric paths and are expensive to compute.
void WritePatchMatchStereoConfigFromSparse(
        const std::string& stereo_path,
        const Reconstruction& reconstruction,
        const std::vector<std::string>& fusion_colmap_names,
        size_t max_neighbors) {
    const int num_images = static_cast<int>(fusion_colmap_names.size());
    if (num_images <= 0) {
        return;
    }

    std::unordered_map<std::string, int> name_to_fusion_idx;
    name_to_fusion_idx.reserve(fusion_colmap_names.size());
    for (int i = 0; i < num_images; ++i) {
        name_to_fusion_idx.emplace(fusion_colmap_names[static_cast<size_t>(i)], i);
    }

    std::vector<std::map<int, int>> cooccurrence(static_cast<size_t>(num_images));
    for (const point3D_t point3D_id : reconstruction.Point3DIds()) {
        const Point3D& point3D = reconstruction.Point3D(point3D_id);
        std::vector<int> visible_fusion_idxs;
        visible_fusion_idxs.reserve(point3D.Track().Length());
        for (const auto& track_el : point3D.Track().Elements()) {
            const Image& image = reconstruction.Image(track_el.image_id);
            const auto it = name_to_fusion_idx.find(image.Name());
            if (it != name_to_fusion_idx.end()) {
                visible_fusion_idxs.push_back(it->second);
            }
        }
        for (size_t i = 0; i < visible_fusion_idxs.size(); ++i) {
            for (size_t j = i + 1; j < visible_fusion_idxs.size(); ++j) {
                ++cooccurrence[static_cast<size_t>(visible_fusion_idxs[i])]
                              [visible_fusion_idxs[j]];
                ++cooccurrence[static_cast<size_t>(visible_fusion_idxs[j])]
                              [visible_fusion_idxs[i]];
            }
        }
    }

    std::ofstream vis_file(JoinPaths(stereo_path, "vis.dat"), std::ios::trunc);
    std::ofstream patch_match_file(JoinPaths(stereo_path, "patch-match.cfg"),
                                   std::ios::trunc);
    if (!vis_file) {
        return;
    }
    vis_file << "VISDATA\n" << num_images << '\n';
    for (int ref_idx = 0; ref_idx < num_images; ++ref_idx) {
        std::vector<std::pair<int, int>> ranked;
        ranked.reserve(cooccurrence[static_cast<size_t>(ref_idx)].size());
        for (const auto& item : cooccurrence[static_cast<size_t>(ref_idx)]) {
            if (item.first != ref_idx) {
                ranked.emplace_back(item.first, item.second);
            }
        }
        std::sort(ranked.begin(), ranked.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        const size_t keep = std::min(ranked.size(), max_neighbors);

        vis_file << ref_idx << ' ' << keep << '\n';
        for (size_t i = 0; i < keep; ++i) {
            vis_file << ranked[i].first << '\n';
        }

        if (patch_match_file) {
            patch_match_file << fusion_colmap_names[static_cast<size_t>(ref_idx)]
                             << '\n';
            for (size_t i = 0; i < keep; ++i) {
                patch_match_file
                    << fusion_colmap_names[static_cast<size_t>(ranked[i].first)]
                    << ", ";
            }
            patch_match_file << '\n';
        }
    }

    RECON_LOG_DEBUG("DA3 depth export: wrote sparse co-visibility vis.dat and "                  "patch-match.cfg (skipped consistency graphs) under %s\n", stereo_path.c_str());
}

void WriteDA3ConsistencyGraphsAndVisDat(
        const std::string& stereo_path,
        const std::vector<ExportedDepthView>& views,
        const std::vector<std::string>& fusion_colmap_names,
        float max_depth_error,
        float max_reproj_error,
        int min_partner_views,
        size_t max_neighbors) {
    const std::string consistency_path =
        JoinPaths(stereo_path, "consistency_graphs");
    CreateDirIfNotExists(consistency_path);

    std::vector<std::map<int, int>> cooccurrence(views.size());
    std::vector<std::vector<std::pair<int, int>>> ranked_neighbors(views.size());

    for (size_t ref_idx = 0; ref_idx < views.size(); ++ref_idx) {
        const auto data = BuildConsistencyGraphData(
            ref_idx, views, max_depth_error, max_reproj_error, min_partner_views,
            /*sample_stride=*/1);
        if (data.empty()) {
            continue;
        }
        const auto& ref_view = views[ref_idx];
        mvs::ConsistencyGraph graph(static_cast<size_t>(ref_view.w),
                                    static_cast<size_t>(ref_view.h), data);
        graph.Write(JoinPaths(consistency_path,
                              fusion_colmap_names[ref_idx] + ".geometric.bin"));

        for (size_t i = 0; i < data.size();) {
            const int num = data[i + 2];
            for (int j = 0; j < num; ++j) {
                const int partner = data[i + 3 + j];
                if (partner >= 0 &&
                    static_cast<size_t>(partner) < views.size()) {
                    cooccurrence[ref_idx][partner]++;
                }
            }
            i += static_cast<size_t>(3 + num);
        }
    }

    const int num_images = static_cast<int>(views.size());
    std::ofstream vis_file(JoinPaths(stereo_path, "vis.dat"), std::ios::trunc);
    std::ofstream patch_match_file(JoinPaths(stereo_path, "patch-match.cfg"),
                                   std::ios::trunc);
    if (!vis_file) {
        return;
    }
    vis_file << "VISDATA\n" << num_images << '\n';
    for (int ref_idx = 0; ref_idx < num_images; ++ref_idx) {
        std::vector<std::pair<int, int>> ranked;
        ranked.reserve(cooccurrence[static_cast<size_t>(ref_idx)].size());
        for (const auto& item : cooccurrence[static_cast<size_t>(ref_idx)]) {
            if (item.first != ref_idx) {
                ranked.emplace_back(item.first, item.second);
            }
        }
        std::sort(ranked.begin(), ranked.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        const size_t keep = std::min(ranked.size(), max_neighbors);
        ranked_neighbors[static_cast<size_t>(ref_idx)] =
            std::vector<std::pair<int, int>>(ranked.begin(),
                                             ranked.begin() + keep);

        vis_file << ref_idx << ' ' << keep << '\n';
        for (size_t i = 0; i < keep; ++i) {
            vis_file << ranked[i].first << '\n';
        }

        if (patch_match_file) {
            patch_match_file << fusion_colmap_names[static_cast<size_t>(ref_idx)]
                             << '\n';
            for (size_t i = 0; i < keep; ++i) {
                patch_match_file
                    << fusion_colmap_names[static_cast<size_t>(ranked[i].first)]
                    << ", ";
            }
            patch_match_file << '\n';
        }
    }

    RECON_LOG_DEBUG("DA3 depth export: wrote consistency graphs, vis.dat, "                  "patch-match.cfg under %s\n", stereo_path.c_str());
}

void FilterExportedDepthMapsColmapInPlace(
        std::vector<ExportedDepthView>* views,
        float max_depth_error,
        float max_reproj_error,
        int min_views,
        int sample_stride) {
    if (views == nullptr || views->size() < 2) {
        return;
    }
    for (const auto& view : *views) {
        if (!view.has_camera || view.depth.empty() || view.w <= 0 ||
            view.h <= 0) {
            RECON_LOG_DEBUG("DA3 depth export: skip COLMAP-camera cleaning "                          "(missing undistorted camera for one or more views)\n");
            return;
        }
    }

    const float max_squared_reproj = max_reproj_error * max_reproj_error;
    size_t invalidated = 0;
    size_t checked = 0;

    for (size_t ref_idx = 0; ref_idx < views->size(); ++ref_idx) {
        auto& ref_view = views->at(ref_idx);
        for (int row = 0; row < ref_view.h; row += sample_stride) {
            for (int col = 0; col < ref_view.w; col += sample_stride) {
                const size_t p =
                    static_cast<size_t>(row) * ref_view.w +
                    static_cast<size_t>(col);
                const float depth = ref_view.depth[p];
                if (depth <= 0.0f || !std::isfinite(depth)) {
                    continue;
                }
                ++checked;

                const Eigen::Vector3f xyz =
                    ref_view.camera.inv_P *
                    Eigen::Vector4f(col * depth, row * depth, depth, 1.0f);

                int consistent_views = 1;
                for (size_t tgt_idx = 0; tgt_idx < views->size(); ++tgt_idx) {
                    if (tgt_idx == ref_idx) {
                        continue;
                    }
                    const auto& tgt_view = views->at(tgt_idx);
                    const Eigen::Vector3f proj =
                        tgt_view.camera.P *
                        Eigen::Vector4f(xyz.x(), xyz.y(), xyz.z(), 1.0f);
                    if (std::abs(proj.z()) <
                        std::numeric_limits<float>::epsilon()) {
                        continue;
                    }
                    const float col_proj = proj.x() / proj.z();
                    const float row_proj = proj.y() / proj.z();
                    if (col_proj < 0.0f || row_proj < 0.0f ||
                        col_proj >= static_cast<float>(tgt_view.w) ||
                        row_proj >= static_cast<float>(tgt_view.h)) {
                        continue;
                    }
                    const float tgt_depth_val = BilinearSampleDepth(
                        tgt_view.depth.data(), tgt_view.w, tgt_view.h, col_proj,
                        row_proj);
                    if (tgt_depth_val <= 0.0f || !std::isfinite(tgt_depth_val)) {
                        continue;
                    }
                    const float depth_error =
                        std::abs((proj.z() - tgt_depth_val) / tgt_depth_val);
                    if (depth_error > max_depth_error) {
                        continue;
                    }
                    const float col_diff = col_proj - std::lround(col_proj);
                    const float row_diff = row_proj - std::lround(row_proj);
                    if (col_diff * col_diff + row_diff * row_diff >
                        max_squared_reproj) {
                        continue;
                    }
                    ++consistent_views;
                }

                if (consistent_views < min_views) {
                    for (int dr = 0; dr < sample_stride && row + dr < ref_view.h;
                         ++dr) {
                        for (int dc = 0; dc < sample_stride &&
                                        col + dc < ref_view.w;
                             ++dc) {
                            const size_t ip =
                                static_cast<size_t>(row + dr) * ref_view.w +
                                static_cast<size_t>(col + dc);
                            if (ref_view.depth[ip] > 0.0f) {
                                ref_view.depth[ip] = 0.0f;
                                ++invalidated;
                            }
                        }
                    }
                }
            }
        }
    }

    if (checked > 0) {
        const size_t total_sampled =
            static_cast<size_t>(sample_stride) *
            static_cast<size_t>(sample_stride) * checked;
        RECON_LOG_DEBUG(
                "DA3 depth export: COLMAP-camera cleaning invalidated %zu / %zu "
                "sampled pixels (stride=%d, min_views=%d, depth_err=%f, "
                "reproj=%fpx)\n",
                invalidated, total_sampled, sample_stride, min_views,
                max_depth_error, max_reproj_error);
    }
}

void ApplyDepthScaleInPlace(std::vector<ExportedDepthView>* views, float scale) {
    if (views == nullptr || !std::isfinite(scale) || scale <= 0.0f ||
        std::abs(scale - 1.0f) < 1e-4f) {
        return;
    }
    for (auto& view : *views) {
        for (float& depth : view.depth) {
            if (depth > 0.0f && std::isfinite(depth)) {
                depth *= scale;
            }
        }
    }
}

// DA3 nested metric depth is in meters; COLMAP sparse uses an arbitrary scale.
// Estimate a global scale from triangulated 3D points so z-depth matches COLMAP.
struct ColmapDepthScaleResult {
    float scale = 1.0f;
    bool aligned = false;
};

ColmapDepthScaleResult EstimateColmapDepthScaleFromSparse(
        const Reconstruction& reconstruction,
        const std::vector<ExportedDepthView>& export_views,
        const std::vector<std::string>& fusion_colmap_names) {
    ColmapDepthScaleResult result;
    std::unordered_map<std::string, size_t> name_to_idx;
    name_to_idx.reserve(fusion_colmap_names.size());
    for (size_t i = 0; i < fusion_colmap_names.size(); ++i) {
        name_to_idx.emplace(fusion_colmap_names[i], i);
    }

    std::vector<double> ratios;
    constexpr size_t kMaxScaleSamples = 2048;
    ratios.reserve(kMaxScaleSamples);

    bool enough_samples = false;
    for (const point3D_t point3D_id : reconstruction.Point3DIds()) {
        if (enough_samples) {
            break;
        }
        const Point3D& point3D = reconstruction.Point3D(point3D_id);
        const Eigen::Vector3d& X = point3D.XYZ();

        for (const auto& track_el : point3D.Track().Elements()) {
            const Image& image = reconstruction.Image(track_el.image_id);
            const auto it = name_to_idx.find(image.Name());
            if (it == name_to_idx.end()) {
                continue;
            }
            const ExportedDepthView& view = export_views[it->second];
            if (!view.has_camera || view.depth.empty() || view.w <= 0 ||
                view.h <= 0) {
                continue;
            }

            const Camera& cam = reconstruction.Camera(image.CameraId());
            if (cam.Width() <= 0 || cam.Height() <= 0) {
                continue;
            }

            const Point2D& point2D = image.Point2D(track_el.point2D_idx);
            const double scale_x =
                static_cast<double>(view.w) / static_cast<double>(cam.Width());
            const double scale_y =
                static_cast<double>(view.h) / static_cast<double>(cam.Height());
            const float col = static_cast<float>(point2D.X() * scale_x);
            const float row = static_cast<float>(point2D.Y() * scale_y);

            const Eigen::Vector3d X_cam =
                image.RotationMatrix() * X + image.Tvec();
            const double colmap_z = X_cam.z();
            if (colmap_z <= 1e-6) {
                continue;
            }

            const float da3_z = BilinearSampleDepth(view.depth.data(), view.w,
                                                    view.h, col, row);
            if (da3_z <= 0.05f || !std::isfinite(da3_z)) {
                continue;
            }

            const double ratio = colmap_z / static_cast<double>(da3_z);
            if (ratio > 0.01 && ratio < 1000.0 && std::isfinite(ratio)) {
                ratios.push_back(ratio);
                if (ratios.size() >= kMaxScaleSamples) {
                    enough_samples = true;
                    break;
                }
            }
        }
    }

    constexpr size_t kMinSamples = 20;
    if (ratios.size() < kMinSamples) {
        RECON_LOG_DEBUG(
                "DA3 depth export: sparse scale alignment skipped (%zu samples, "
                "need %zu)\n",
                ratios.size(), kMinSamples);
        return result;
    }

    const size_t mid = ratios.size() / 2;
    std::nth_element(ratios.begin(), ratios.begin() + mid, ratios.end());
    result.scale = static_cast<float>(ratios[mid]);
    result.aligned = std::isfinite(result.scale) && result.scale > 0.0f;
    if (result.aligned) {
        RECON_LOG_DEBUG(
                "DA3 depth export: aligned metric depth to COLMAP scale x%f (%zu "
                "sparse samples)\n",
                result.scale, ratios.size());
    }
    return result;
}

void FilterMultiviewDepthMapsInPlace(DepthPoseMultiResult& multi,
                                     float max_depth_error = 0.08f,
                                     float max_reproj_error = 4.0f,
                                     int min_views = 2,
                                     int sample_stride = 4) {
    if (multi.n <= 1 || multi.w <= 0 || multi.h <= 0 || multi.ext.empty() ||
        multi.intr.empty()) {
        return;
    }

    std::vector<DA3ViewCamera> cameras;
    cameras.reserve(static_cast<size_t>(multi.n));
    for (int i = 0; i < multi.n; ++i) {
        cameras.push_back(MakeViewCamera(
            multi.ext.data() + static_cast<size_t>(i) * 12,
            multi.intr.data() + static_cast<size_t>(i) * 9));
    }

    const float max_squared_reproj = max_reproj_error * max_reproj_error;
    const size_t per_view =
        static_cast<size_t>(multi.w) * static_cast<size_t>(multi.h);
    size_t invalidated = 0;
    size_t checked = 0;

    for (int ref_idx = 0; ref_idx < multi.n; ++ref_idx) {
        float* ref_depth =
            multi.depth.data() + static_cast<size_t>(ref_idx) * per_view;
        const auto& ref_cam = cameras[ref_idx];
        for (int row = 0; row < multi.h; row += sample_stride) {
            for (int col = 0; col < multi.w; col += sample_stride) {
                const size_t p =
                    static_cast<size_t>(row) * multi.w + static_cast<size_t>(col);
                const float depth = ref_depth[p];
                if (depth <= 0.0f || !std::isfinite(depth)) {
                    continue;
                }
                ++checked;

                const Eigen::Vector3f xyz =
                    ref_cam.inv_P *
                    Eigen::Vector4f(col * depth, row * depth, depth, 1.0f);

                int consistent_views = 1;
                for (int tgt_idx = 0; tgt_idx < multi.n; ++tgt_idx) {
                    if (tgt_idx == ref_idx) {
                        continue;
                    }
                    const auto& tgt_cam = cameras[tgt_idx];
                    const float* tgt_depth =
                        multi.depth.data() +
                        static_cast<size_t>(tgt_idx) * per_view;
                    const Eigen::Vector3f proj =
                        tgt_cam.P *
                        Eigen::Vector4f(xyz.x(), xyz.y(), xyz.z(), 1.0f);
                    if (std::abs(proj.z()) <
                        std::numeric_limits<float>::epsilon()) {
                        continue;
                    }
                    const float col_proj = proj.x() / proj.z();
                    const float row_proj = proj.y() / proj.z();
                    if (col_proj < 0.0f || row_proj < 0.0f ||
                        col_proj >= static_cast<float>(multi.w) ||
                        row_proj >= static_cast<float>(multi.h)) {
                        continue;
                    }
                    const float tgt_depth_val = SampleDepthGrid(
                        tgt_depth, multi.w, multi.h, col_proj, row_proj);
                    if (tgt_depth_val <= 0.0f ||
                        !std::isfinite(tgt_depth_val)) {
                        continue;
                    }
                    const float depth_error =
                        std::abs((proj.z() - tgt_depth_val) / tgt_depth_val);
                    if (depth_error > max_depth_error) {
                        continue;
                    }
                    const float col_diff = col_proj - std::lround(col_proj);
                    const float row_diff = row_proj - std::lround(row_proj);
                    if (col_diff * col_diff + row_diff * row_diff >
                        max_squared_reproj) {
                        continue;
                    }
                    ++consistent_views;
                }

                if (consistent_views < min_views) {
                    ref_depth[p] = 0.0f;
                    ++invalidated;
                }
            }
        }
    }

    if (checked > 0) {
        RECON_LOG_DEBUG(
                "DA3 depth export: invalidated %zu / %zu inconsistent low-res "
                "samples (stride=%d, min_views=%d)\n",
                invalidated, checked, sample_stride, min_views);
    }
}

bool WriteStereoMapsFromMultiview(
    const std::string& output_path,
    const std::vector<std::string>& undist_image_paths,
    const std::vector<std::string>& fusion_colmap_names,
    const DepthPoseMultiResult& multi,
    const StereoExportProgressCallback& progress_cb,
    const std::function<bool()>& is_stopped,
    bool use_colmap_poses_only,
    bool export_photometric_prior,
    int export_max_image_size,
    bool fast_depth_export) {
    if (static_cast<int>(undist_image_paths.size()) != multi.n ||
        static_cast<int>(fusion_colmap_names.size()) != multi.n) {
        LOG(ERROR) << "DA3: stereo export view count mismatch";
        return false;
    }

    const std::string stereo_path = JoinPaths(output_path, "stereo");
    const std::string depth_maps_path = JoinPaths(stereo_path, "depth_maps");
    const std::string normal_maps_path = JoinPaths(stereo_path, "normal_maps");
    CreateDirIfNotExists(stereo_path);
    CreateDirIfNotExists(depth_maps_path);
    CreateDirIfNotExists(normal_maps_path);

    const size_t per_view =
        static_cast<size_t>(multi.h) * static_cast<size_t>(multi.w);

    const float kFusionDepthErr = 0.04f;
    const float kFusionReprojErr = 3.0f;

    DepthPoseMultiResult filtered_multi = multi;

    // COLMAP+DA3 uses COLMAP poses for fusion; per-view DA3 poses are not
    // globally consistent and must not drive low-res cleaning.
    if (!use_colmap_poses_only) {
        FilterMultiviewDepthMapsInPlace(filtered_multi,
                                        /*max_depth_error=*/0.06f,
                                        /*max_reproj_error=*/3.5f,
                                        /*min_views=*/2,
                                        /*sample_stride=*/4);
    } else {
        RECON_LOG_DEBUG("DA3 depth export: skip low-res DA3-camera cleaning "                      "(COLMAP poses authoritative)\n");
    }

    std::unique_ptr<Reconstruction> undist_reconstruction;
    const std::string undist_sparse_path =
        FindUndistortedSparseModelPath(output_path);
    if (!undist_sparse_path.empty()) {
        undist_reconstruction = std::make_unique<Reconstruction>();
        undist_reconstruction->Read(undist_sparse_path);
        RECON_LOG_DEBUG("DA3 depth export: using undistorted sparse model at %s\n", undist_sparse_path.c_str());
    }

    std::vector<ExportedDepthView> export_views(static_cast<size_t>(multi.n));
    std::vector<std::unique_ptr<Bitmap>> undist_bitmaps(static_cast<size_t>(multi.n));
    std::vector<int> target_widths(static_cast<size_t>(multi.n), multi.w);
    std::vector<int> target_heights(static_cast<size_t>(multi.n), multi.h);

    // Phase 1: keep inference-resolution depth for fast sparse scale alignment.
    for (int i = 0; i < multi.n; ++i) {
        if (is_stopped && is_stopped()) {
            return false;
        }
        if (progress_cb) {
            progress_cb(i + 1, multi.n,
                        "DA3 depth export (prepare): " + fusion_colmap_names[i]);
        }

        const float* view_depth =
            filtered_multi.depth.data() + static_cast<size_t>(i) * per_view;

        auto undist_bmp = std::make_unique<Bitmap>();
        int out_w = multi.w;
        int out_h = multi.h;
        if (undist_bmp->Read(undist_image_paths[i])) {
            undist_bitmaps[static_cast<size_t>(i)] = std::move(undist_bmp);
            out_w = static_cast<int>(undist_bitmaps[static_cast<size_t>(i)]->Width());
            out_h = static_cast<int>(undist_bitmaps[static_cast<size_t>(i)]->Height());
        } else {
            undist_bitmaps[static_cast<size_t>(i)].reset();
        }
        CapExportDimensions(&out_w, &out_h, export_max_image_size);
        target_widths[static_cast<size_t>(i)] = out_w;
        target_heights[static_cast<size_t>(i)] = out_h;

        export_views[static_cast<size_t>(i)].w = multi.w;
        export_views[static_cast<size_t>(i)].h = multi.h;
        export_views[static_cast<size_t>(i)].depth.assign(
            view_depth, view_depth + per_view);
        FilterMetricDepthInPlace(export_views[static_cast<size_t>(i)].depth);

        const std::string& colmap_name = fusion_colmap_names[i];
        auto& export_view = export_views[static_cast<size_t>(i)];
        const bool colmap_poses_usable =
            undist_reconstruction &&
            !IsExifPlaceholderReconstruction(*undist_reconstruction);
        if (colmap_poses_usable &&
            MakeViewCameraFromColmapImage(*undist_reconstruction, colmap_name,
                                          multi.w, multi.h, &export_view.camera)) {
            export_view.has_camera = true;
            LookupUndistortedCameraIntrinsics(*undist_reconstruction, colmap_name,
                                              multi.w, multi.h, &export_view.fx,
                                              &export_view.fy, &export_view.cx,
                                              &export_view.cy);
        } else if (multi.ext.size() >= static_cast<size_t>((i + 1) * 12) &&
                   multi.intr.size() >= static_cast<size_t>((i + 1) * 9)) {
            const float* ext =
                multi.ext.data() + static_cast<size_t>(i) * 12;
            const float* intr =
                multi.intr.data() + static_cast<size_t>(i) * 9;
            export_view.camera = MakeViewCamera(ext, intr);
            export_view.has_camera = true;
            export_view.fx = intr[0];
            export_view.fy = intr[4];
            export_view.cx = intr[2];
            export_view.cy = intr[5];
        }
    }

    if (undist_reconstruction &&
        !IsExifPlaceholderReconstruction(*undist_reconstruction)) {
        const ColmapDepthScaleResult scale_result =
            EstimateColmapDepthScaleFromSparse(*undist_reconstruction,
                                               export_views,
                                               fusion_colmap_names);
        if (scale_result.aligned) {
            ApplyDepthScaleInPlace(&export_views, scale_result.scale);
        }

        if (export_photometric_prior && fast_depth_export) {
            RECON_LOG_DEBUG("DA3 depth export: skip COLMAP-camera cleaning "                          "(direct fusion path; voxel consensus filters priors)\n");
        } else if (export_photometric_prior && !fast_depth_export) {
            RECON_LOG_DEBUG("DA3 depth export: skip COLMAP-camera cleaning "                          "(PatchMatch geometric refine will optimize depth)\n");
        } else {
        if (export_photometric_prior) {
            RECON_LOG_DEBUG("DA3 depth export: COLMAP-camera cleaning on priors\n");
        }
        const size_t pixels_before = CountValidDepthPixels(export_views);
        std::vector<std::vector<float>> depth_backup;
        depth_backup.reserve(export_views.size());
        for (const auto& view : export_views) {
            depth_backup.push_back(view.depth);
        }

        const bool run_colmap_cleaning = scale_result.aligned;
        if (run_colmap_cleaning) {
            FilterExportedDepthMapsColmapInPlace(
                &export_views,
                kFusionDepthErr,
                kFusionReprojErr,
                /*min_views=*/2,
                /*sample_stride=*/4);
        } else {
            RECON_LOG_DEBUG("DA3 depth export: skip COLMAP-camera cleaning "                          "(depth scale not aligned to sparse model)\n");
        }

        const size_t pixels_after = CountValidDepthPixels(export_views);
        const size_t min_keep_pixels = std::max<size_t>(
            pixels_before / 10, static_cast<size_t>(multi.w) * multi.h / 20);
        if (run_colmap_cleaning && pixels_before > 0 &&
            pixels_after < min_keep_pixels) {
            for (size_t i = 0; i < export_views.size(); ++i) {
                export_views[i].depth = std::move(depth_backup[i]);
            }
            RECON_LOG_WARN(
                    "WARNING: DA3 COLMAP-camera cleaning removed %zu%% of valid "
                    "depth pixels; reverted to unfiltered depth\n",
                    100 - pixels_after * 100 / pixels_before);
        } else if (run_colmap_cleaning && pixels_before > 0) {
            RECON_LOG_DEBUG(
                    "DA3 depth export: COLMAP-camera cleaning kept %zu / %zu "
                    "valid depth pixels.\n",
                    pixels_after, pixels_before);
        }
        }
    } else if (undist_reconstruction &&
               IsExifPlaceholderReconstruction(*undist_reconstruction)) {
        RECON_LOG_DEBUG("DA3 depth export: undistorted sparse has EXIF placeholder "                      "poses; using DA3 inference cameras for cleaning\n");
        const size_t pixels_before = CountValidDepthPixels(export_views);
        FilterExportedDepthMapsColmapInPlace(
            &export_views,
            kFusionDepthErr,
            kFusionReprojErr,
            /*min_views=*/2,
            /*sample_stride=*/4);
        const size_t pixels_after = CountValidDepthPixels(export_views);
        if (pixels_before > 0) {
            RECON_LOG_DEBUG(
                    "DA3 depth export: DA3-camera cleaning kept %zu / %zu valid "
                    "depth pixels.\n",
                    pixels_after, pixels_before);
        }
    } else {
        RECON_LOG_DEBUG("DA3 depth export: undistorted sparse model not found under %s; skipping COLMAP-camera depth cleaning\n", JoinPaths(output_path, "sparse").c_str());
    }

    if (fast_depth_export) {
        RECON_LOG_DEBUG("DA3 depth export: fast bilinear upsample (skip guided filter)\n");
    }
    if (export_max_image_size > 0) {
        RECON_LOG_DEBUG(
                "DA3 depth export: capped export resolution to max side %d px\n",
                export_max_image_size);
    }

    ThreadPool upsample_pool(
        std::min(multi.n, GetEffectiveNumThreads(-1)));
    for (int i = 0; i < multi.n; ++i) {
        upsample_pool.AddTask([&, i]() {
            if (is_stopped && is_stopped()) {
                return;
            }
            if (progress_cb) {
                progress_cb(i + 1, multi.n,
                            "DA3 depth export (upsample): " +
                                fusion_colmap_names[i]);
            }

            auto& export_view = export_views[static_cast<size_t>(i)];
            const int out_w = target_widths[static_cast<size_t>(i)];
            const int out_h = target_heights[static_cast<size_t>(i)];
            if (multi.w == out_w && multi.h == out_h) {
                export_view.w = out_w;
                export_view.h = out_h;
                return;
            }

            const float* src_depth = export_view.depth.data();
            Bitmap* guide = undist_bitmaps[static_cast<size_t>(i)].get();
            if (fast_depth_export || guide == nullptr) {
                export_view.depth =
                    UpsampleDepthBilinear(src_depth, multi.w, multi.h, out_w, out_h);
            } else {
                export_view.depth = UpsampleDepthGuided(
                    src_depth, multi.w, multi.h, guide, out_w, out_h);
            }
            FilterMetricDepthInPlace(export_view.depth);
            export_view.w = out_w;
            export_view.h = out_h;

            const std::string& colmap_name = fusion_colmap_names[i];
            const bool colmap_poses_usable =
                undist_reconstruction &&
                !IsExifPlaceholderReconstruction(*undist_reconstruction);
            if (colmap_poses_usable && export_view.has_camera) {
                LookupUndistortedCameraIntrinsics(
                    *undist_reconstruction, colmap_name, out_w, out_h,
                    &export_view.fx, &export_view.fy, &export_view.cx,
                    &export_view.cy);
            } else if (multi.ext.size() >= static_cast<size_t>((i + 1) * 12) &&
                       multi.intr.size() >= static_cast<size_t>((i + 1) * 9)) {
                const float scale_x =
                    static_cast<float>(out_w) / static_cast<float>(multi.w);
                const float scale_y =
                    static_cast<float>(out_h) / static_cast<float>(multi.h);
                const float* intr =
                    multi.intr.data() + static_cast<size_t>(i) * 9;
                export_view.fx = intr[0] * scale_x;
                export_view.fy = intr[4] * scale_y;
                export_view.cx = intr[2] * scale_x;
                export_view.cy = intr[5] * scale_y;
            }
        });
    }
    upsample_pool.Wait();
    if (is_stopped && is_stopped()) {
        return false;
    }

    for (int i = 0; i < multi.n; ++i) {
        if (is_stopped && is_stopped()) {
            return false;
        }
        if (progress_cb) {
            progress_cb(i + 1, multi.n,
                        "DA3 depth export (write): " + fusion_colmap_names[i]);
        }

        const auto& export_view = export_views[static_cast<size_t>(i)];
        const int out_w = export_view.w;
        const int out_h = export_view.h;
        const std::string& colmap_name = fusion_colmap_names[i];
        const std::string map_suffix =
            export_photometric_prior ? ".photometric.bin" : ".geometric.bin";

        WriteColmapDepthMap(
            JoinPaths(depth_maps_path, colmap_name + map_suffix), out_w,
            out_h, export_view.depth.data());

        const std::vector<float> normals = ComputeNormalsFromDepthWithIntrinsics(
            export_view.depth.data(), out_w, out_h, export_view.fx,
            export_view.fy, export_view.cx, export_view.cy);
        WriteColmapNormalMap(
            JoinPaths(normal_maps_path, colmap_name + map_suffix), out_w,
            out_h, normals);
    }

    constexpr size_t kMaxPatchMatchNeighbors = 20;
    if (export_photometric_prior && undist_reconstruction &&
        !IsExifPlaceholderReconstruction(*undist_reconstruction)) {
        WritePatchMatchStereoConfigFromSparse(
            stereo_path, *undist_reconstruction, fusion_colmap_names,
            kMaxPatchMatchNeighbors);
    } else {
        WriteDA3ConsistencyGraphsAndVisDat(
            stereo_path, export_views, fusion_colmap_names, kFusionDepthErr,
            kFusionReprojErr,
            /*min_partner_views=*/1, kMaxPatchMatchNeighbors);
    }

    WriteStereoFusionConfig(stereo_path, fusion_colmap_names);
    RECON_LOG_DEBUG(
            "DA3: Depth maps and normal maps written to: %s (%d views%s)\n",
            stereo_path.c_str(), multi.n,
            export_photometric_prior ? ", photometric priors" : "");
    return true;
}

#endif  // AICore_ENABLED

}  // namespace

std::string DA3ModelFilename(DA3ModelType model, DA3QuantType quant) {
    auto it = kModelFilenames.find({model, quant});
    if (it != kModelFilenames.end()) return it->second;

    std::string name = "depth-anything-";
    switch (model) {
        case DA3ModelType::BASE: name += "base"; break;
        case DA3ModelType::LARGE: name += "large"; break;
        case DA3ModelType::GIANT: name += "giant"; break;
        case DA3ModelType::NESTED_METRIC: name += "nested-metric"; break;
        case DA3ModelType::NESTED_ANYVIEW: name += "nested-anyview"; break;
    }
    name += "-";
    switch (quant) {
        case DA3QuantType::F32: name += "f32"; break;
        case DA3QuantType::F16: name += "f16"; break;
        case DA3QuantType::Q8_0: name += "q8_0"; break;
        case DA3QuantType::Q4_K: name += "q4_k"; break;
    }
    name += ".gguf";
    return name;
}

std::string DA3ModelDownloadURL(DA3ModelType model, DA3QuantType quant) {
    return kDA3DownloadBaseURL + DA3ModelFilename(model, quant);
}

std::string DA3ModelDownloadURI(DA3ModelType model, DA3QuantType quant) {
    return DA3ModelDownloadURL(model, quant);
}

bool DA3ModelExists(DA3ModelType model, DA3QuantType quant) {
    return kModelFilenames.count({model, quant}) > 0;
}

bool DA3ModelIsNested(DA3ModelType model) {
    return model == DA3ModelType::NESTED_METRIC ||
           model == DA3ModelType::NESTED_ANYVIEW;
}

bool DA3ModelSupportsStereo(DA3ModelType model) {
    return DA3ModelIsNested(model);
}

std::vector<DA3QuantType> DA3SupportedQuantTypes(DA3ModelType model) {
    std::vector<DA3QuantType> result;
    for (auto qt : {DA3QuantType::Q8_0, DA3QuantType::Q4_K,
                    DA3QuantType::F16, DA3QuantType::F32}) {
        if (kModelFilenames.count({model, qt}) > 0) {
            result.push_back(qt);
        }
    }
    return result;
}

std::string DA3ModelCacheDir() {
#ifdef AICore_ENABLED
    char* dir = aicore_depth_model_cache_dir();
    if (!dir) return ".cache/da3_models";
    std::string result(dir);
    aicore_depth_free_string(dir);
    return result;
#else
    return ".cache/da3_models";
#endif
}

void CollectDA3ModelCacheNeeds(const std::string& cache_dir,
                               DA3ModelType model_type,
                               DA3QuantType quant_type,
                               std::string& model_path,
                               std::string& metric_model_path,
                               std::vector<DA3ModelCacheNeed>& needed) {
    const std::filesystem::path cache_path(cache_dir);

    if (model_path.empty()) {
        const std::string main_filename = DA3ModelFilename(model_type, quant_type);
        const auto main_cached = cache_path / main_filename;
        if (!std::filesystem::exists(main_cached)) {
            needed.push_back({main_filename, DA3ModelDownloadURL(model_type, quant_type),
                              &model_path});
        } else {
            model_path = main_cached.string();
        }
    }

    if (!DA3ModelIsNested(model_type)) {
        return;
    }

    if (metric_model_path.empty()) {
        const std::string metric_filename =
            DA3ModelFilename(DA3ModelType::NESTED_METRIC, DA3QuantType::F32);
        const auto metric_cached = cache_path / metric_filename;
        if (!std::filesystem::exists(metric_cached)) {
            needed.push_back({metric_filename,
                              DA3ModelDownloadURL(DA3ModelType::NESTED_METRIC,
                                                  DA3QuantType::F32),
                              &metric_model_path});
        } else {
            metric_model_path = metric_cached.string();
        }
    }

    if (model_type != DA3ModelType::NESTED_METRIC) {
        return;
    }

    DA3QuantType anyview_quant = DA3QuantType::Q8_0;
    if (DA3ModelExists(DA3ModelType::NESTED_ANYVIEW, quant_type)) {
        anyview_quant = quant_type;
    }
    const std::string anyview_filename =
        DA3ModelFilename(DA3ModelType::NESTED_ANYVIEW, anyview_quant);
    const auto anyview_cached = cache_path / anyview_filename;
    if (!std::filesystem::exists(anyview_cached)) {
        needed.push_back({anyview_filename,
                          DA3ModelDownloadURL(DA3ModelType::NESTED_ANYVIEW,
                                              anyview_quant),
                          nullptr});
    }
}

std::string DA3DepthController::ResolveModelPath(const DA3Config& config) {
    if (!config.model_path.empty() && ExistsFile(config.model_path)) {
        return config.model_path;
    }

    const std::string filename = DA3ModelFilename(config.model_type, config.quant_type);

#ifdef COLMAP_DOWNLOAD_ENABLED
    const std::string url = DA3ModelDownloadURL(config.model_type, config.quant_type);

    const std::filesystem::path cache_dir(DA3ModelCacheDir());
    std::filesystem::create_directories(cache_dir);
    const auto cached_path = cache_dir / filename;

    if (std::filesystem::exists(cached_path)) {
        RECON_LOG_DEBUG("DA3: Using cached model: %s\n", cached_path.string().c_str());
        return cached_path.string();
    }

    RECON_LOG_DEBUG("DA3: Downloading model from: %s\n", url.c_str());
    const auto blob = DownloadFile(url);
    if (!blob.has_value()) {
        LOG(ERROR) << "DA3: Failed to download model from: " << url;
        return "";
    }

    std::ofstream ofs(cached_path, std::ios::binary);
    if (!ofs) {
        LOG(ERROR) << "DA3: Failed to write cached model to: " << cached_path;
        return "";
    }
    ofs.write(blob->data(), static_cast<std::streamsize>(blob->size()));
    ofs.close();
    RECON_LOG_DEBUG("DA3: Model cached at: %s\n", cached_path.string().c_str());
    return cached_path.string();
#else
    LOG(ERROR) << "DA3: Download support not enabled and no local model path provided. "
               << "Please provide the model path directly or enable DOWNLOAD_ENABLED.";
    return "";
#endif
}

DA3DepthController::DA3DepthController(const DA3Config& config,
                                       const std::string& image_path,
                                       const std::string& output_path)
    : config_(config), image_path_(image_path), output_path_(output_path) {}

void DA3DepthController::Run() {
    success_ = true;
    if (config_.sparse_mode == SparseModelMode::DA3_DEPTH_POSE) {
        if (!GenerateSparseModel()) {
            success_ = false;
        }
    }
    if (config_.stereo_mode == StereoPipelineMode::DA3_DEPTH_INFERENCE) {
        if (!DA3ModelSupportsStereo(config_.model_type)) {
            LOG(ERROR) << "DA3: stereo depth inference requires a nested model "
                       << "(Nested AnyView or Nested Metric)";
            RECON_LOG_ERROR("ERROR: DA3 stereo requires nested model. Select Nested AnyView/Metric or use COLMAP PatchMatch.\n");
            success_ = false;
            return;
        }
        if (!GenerateDepthMaps(multiview_cache_in_)) {
            success_ = false;
            RECON_LOG_ERROR("ERROR: DA3 depth inference failed. If you see CUDA OOM, "                          "try: lower max image size, Q4_K quant, or "                          "DA_DEVICE=cpu ./ACloudViewer\n");
        }
    }
}

bool DA3DepthController::GenerateSparseModel() {
#ifndef AICore_ENABLED
    LOG(ERROR) << "DA3: DA3 core library not enabled. Cannot run depth estimation.";
    RECON_LOG_ERROR("ERROR: AICore not enabled (AICore_ENABLED not set at "                  "compile time). Rebuild with -DAICore_ENABLED=ON.\n");
    return false;
#else
    std::vector<DA3ImageEntry> entries = CollectDA3ImageEntries(image_path_);
    if (entries.empty()) {
        LOG(ERROR) << "DA3: No images found in: " << image_path_;
        RECON_LOG_ERROR("ERROR: DA3 no images found in: %s\n", image_path_.c_str());
        return false;
    }

    const int n_threads = config_.num_threads > 0 ? config_.num_threads : 4;
    const int N = static_cast<int>(entries.size());

    RECON_LOG_DEBUG("DA3 GenerateSparseModel: sequential depth+pose (threads=%d, images=%zu)\n",
                    n_threads, N);

    aicore_depth_ctx* ctx = LoadDA3Context(config_, n_threads);
    if (!ctx) {
        LOG(ERROR) << "DA3: Failed to load model";
        RECON_LOG_ERROR("ERROR: DA3 failed to load model\n");
        return false;
    }

    if (progress_cb_) {
        progress_cb_(0, N, "DA3 multiview depth+pose+export");
    }

    std::vector<const char*> cpaths(static_cast<size_t>(N));
    std::vector<const char*> cnames(static_cast<size_t>(N));
    std::vector<std::string> name_storage;
    if (!colmap_image_names_.empty()) {
        if (static_cast<int>(colmap_image_names_.size()) != N) {
            LOG(WARNING) << "DA3: colmap_image_names size mismatch; using "
                            "derived relative names";
            colmap_image_names_.clear();
        }
    }
    if (colmap_image_names_.empty()) {
        name_storage.reserve(static_cast<size_t>(N));
        for (const auto& entry : entries) {
            name_storage.push_back(entry.colmap_name);
        }
    } else {
        name_storage = colmap_image_names_;
    }

    for (int i = 0; i < N; ++i) {
        cpaths[static_cast<size_t>(i)] = entries[static_cast<size_t>(i)].abs_path.c_str();
        cnames[static_cast<size_t>(i)] = name_storage[static_cast<size_t>(i)].c_str();
    }

    std::vector<std::string> abs_paths;
    abs_paths.reserve(static_cast<size_t>(N));
    for (const auto& entry : entries) {
        abs_paths.push_back(entry.abs_path);
    }

    DepthPoseMultiResult multi;
    const bool cache_stereo =
        multiview_cache_out_ != nullptr &&
        DA3ModelSupportsStereo(config_.model_type);

    const std::string sparse_path = JoinPaths(output_path_, "sparse", "0");
    if (cache_stereo) {
        if (!RunDepthPoseMulti(ctx, abs_paths, multi, config_.max_image_size)) {
            LOG(ERROR) << "DA3: aicore_depth_depth_pose_multi failed: "
                       << aicore_depth_last_error(ctx);
            RECON_LOG_ERROR("ERROR: DA3 multiview depth failed: %s\n", aicore_depth_last_error(ctx));
            aicore_depth_free(ctx);
            return false;
        }
        if (aicore_depth_write_colmap_from_multiview(
                ctx, cpaths.data(), cnames.data(), N, multi.depth.data(),
                multi.ext.data(), multi.intr.data(), multi.h, multi.w,
                sparse_path.c_str(), 1) != 0) {
            LOG(ERROR) << "DA3: aicore_depth_write_colmap_from_multiview failed: "
                       << aicore_depth_last_error(ctx);
            RECON_LOG_ERROR("ERROR: DA3 sparse COLMAP export failed: %s\n", aicore_depth_last_error(ctx));
            aicore_depth_free(ctx);
            return false;
        }
    } else if (aicore_depth_export_colmap_multi_named(
                   ctx, cpaths.data(), cnames.data(), N, sparse_path.c_str(),
                   1) != 0) {
        LOG(ERROR) << "DA3: aicore_depth_export_colmap_multi failed: "
                   << aicore_depth_last_error(ctx);
        RECON_LOG_ERROR("ERROR: DA3 sparse COLMAP export failed: %s\n", aicore_depth_last_error(ctx));
        aicore_depth_free(ctx);
        return false;
    }
    aicore_depth_free(ctx);

    if (IsStopped()) {
        return false;
    }

    if (cache_stereo && multiview_cache_out_) {
        multiview_cache_out_->config = config_;
        multiview_cache_out_->h = multi.h;
        multiview_cache_out_->w = multi.w;
        multiview_cache_out_->n = multi.n;
        multiview_cache_out_->colmap_names = name_storage;
        multiview_cache_out_->depth = std::move(multi.depth);
        multiview_cache_out_->ext = std::move(multi.ext);
        multiview_cache_out_->intr = std::move(multi.intr);
        multiview_cache_out_->valid = true;
        multiview_cache_out_->undistorted_images = false;
        RECON_LOG_DEBUG("DA3: cached multiview result for stereo reuse (%d views)\n",
                        multiview_cache_out_->n);
    }

    RECON_LOG_DEBUG(
            "DA3: COLMAP sparse model written to: %s (%d images, with "
            "back-projected points3D)\n",
            sparse_path.c_str(), N);
    return true;
#endif  // AICore_ENABLED
}

bool DA3DepthController::GenerateDepthMaps(
    const DA3MultiviewCache* multiview_cache) {
#ifndef AICore_ENABLED
    LOG(ERROR) << "DA3: DA3 core library not enabled. Cannot run depth estimation.";
    RECON_LOG_ERROR("ERROR: AICore not enabled (AICore_ENABLED not set at "                  "compile time). Rebuild with -DAICore_ENABLED=ON.\n");
    return false;
#else
    if (!DA3ModelSupportsStereo(config_.model_type)) {
        LOG(ERROR) << "DA3: stereo depth inference requires a nested model";
        RECON_LOG_ERROR("ERROR: DA3 stereo requires nested model "                      "(Nested AnyView / Nested Metric).\n");
        return false;
    }

    const auto image_paths = explicit_image_paths_.empty()
                                   ? CollectImagePaths(image_path_)
                                   : explicit_image_paths_;
    if (image_paths.empty()) {
        LOG(ERROR) << "DA3: No images found in: " << image_path_;
        RECON_LOG_ERROR("ERROR: DA3 no images found in: %s\n", image_path_.c_str());
        return false;
    }

    std::vector<std::string> fusion_names;
    if (!colmap_image_names_.empty()) {
        fusion_names = colmap_image_names_;
    } else {
        for (const auto& entry : CollectDA3ImageEntries(image_path_)) {
            fusion_names.push_back(entry.colmap_name);
        }
    }
    if (static_cast<int>(image_paths.size()) != static_cast<int>(fusion_names.size())) {
        fusion_names.clear();
        for (const auto& path : image_paths) {
            fusion_names.push_back(GetPathBaseName(path));
        }
    }

    const int n_threads = config_.num_threads > 0 ? config_.num_threads : 4;
    const int N = static_cast<int>(image_paths.size());

    DepthPoseMultiResult multi;
    const bool on_undistorted = !explicit_image_paths_.empty();
    const DA3MultiviewCache* cache =
        multiview_cache != nullptr ? multiview_cache : multiview_cache_in_;
    const bool reuse_cache =
        cache != nullptr && cache->valid &&
        DA3ConfigsMatchForStereoReuse(cache->config, config_) &&
        cache->n == N &&
        cache->undistorted_images == on_undistorted;

    if (cache != nullptr && cache->valid && on_undistorted &&
        !cache->undistorted_images) {
        RECON_LOG_DEBUG("DA3 GenerateDepthMaps: not reusing sparse multiview cache "                      "(need undistorted-image inference).\n");
    }

    if (reuse_cache) {
        RECON_LOG_DEBUG(
                "DA3 GenerateDepthMaps: reusing cached multiview inference (%zu "
                "views%s)\n",
                static_cast<size_t>(N),
                on_undistorted ? ", undistorted" : "");
        multi.h = cache->h;
        multi.w = cache->w;
        multi.n = cache->n;
        multi.depth = cache->depth;
        multi.ext = cache->ext;
        multi.intr = cache->intr;
        if (progress_cb_) {
            progress_cb_(0, N, "DA3 stereo export (cached depth)");
        }
    } else {
        RECON_LOG_DEBUG(
                "DA3 GenerateDepthMaps: sequential per-view depth (threads=%d, "
                "images=%zu%s)\n",
                n_threads, static_cast<size_t>(N),
                on_undistorted ? ", undistorted images" : "");

        aicore_depth_ctx* ctx = LoadDA3Context(config_, n_threads);
        if (!ctx) {
            LOG(ERROR) << "DA3: Failed to load nested model";
            RECON_LOG_ERROR("ERROR: DA3 failed to load nested model\n");
            return false;
        }

        if (progress_cb_) {
            progress_cb_(0, N, "DA3 multiview metric depth");
        }

        if (!RunDepthPoseMulti(ctx, image_paths, multi, config_.max_image_size)) {
            LOG(ERROR) << "DA3: aicore_depth_depth_pose_multi failed: "
                       << aicore_depth_last_error(ctx);
            RECON_LOG_ERROR("ERROR: DA3 multiview depth failed: %s\n", aicore_depth_last_error(ctx));
            aicore_depth_free(ctx);
            return false;
        }

        if (!use_colmap_poses_only_ && !explicit_image_paths_.empty() &&
            !WriteDenseSparseFromMultiview(ctx, output_path_, image_paths,
                                           fusion_names, multi)) {
            aicore_depth_free(ctx);
            return false;
        }

        aicore_depth_free(ctx);

        if (multiview_cache_out_) {
            multiview_cache_out_->config = config_;
            multiview_cache_out_->h = multi.h;
            multiview_cache_out_->w = multi.w;
            multiview_cache_out_->n = multi.n;
            multiview_cache_out_->colmap_names = fusion_names;
            multiview_cache_out_->depth = multi.depth;
            multiview_cache_out_->ext = multi.ext;
            multiview_cache_out_->intr = multi.intr;
            multiview_cache_out_->valid = true;
            multiview_cache_out_->undistorted_images = on_undistorted;
            RECON_LOG_DEBUG("DA3: cached multiview result (%d views%s)\n",
                            multiview_cache_out_->n,
                            on_undistorted ? ", undistorted" : "");
        }
    }

    if (IsStopped()) {
        return false;
    }

    DepthPoseMultiResult export_multi = multi;
    return WriteStereoMapsFromMultiview(
        output_path_, image_paths, fusion_names, export_multi, progress_cb_,
        [this]() { return IsStopped(); }, use_colmap_poses_only_,
        export_photometric_prior_, export_max_image_size_, fast_depth_export_);
#endif  // AICore_ENABLED
}

namespace {

DA3VramCapWarning g_da3_vram_warning;

}  // namespace

void DA3ClearVramCapWarning() { g_da3_vram_warning = {}; }

const DA3VramCapWarning& DA3PeekVramCapWarning() { return g_da3_vram_warning; }

void DA3NoteVramCap(int requested, int capped) {
    if (requested <= 0 || capped >= requested) {
        return;
    }
    const double ratio = static_cast<double>(capped) / static_cast<double>(requested);
    constexpr double kWarnRatio = 0.70;
    constexpr int kWarnAbsoluteCap = 800;
    if (ratio >= kWarnRatio && capped > kWarnAbsoluteCap) {
        return;
    }
    g_da3_vram_warning.active = true;
    g_da3_vram_warning.requested = requested;
    g_da3_vram_warning.capped = capped;
    RECON_LOG_WARN(
            "WARNING: DA3 GPU memory pressure — preprocess resolution capped %d "
            "-> %d px (close other GPU apps for higher depth quality).\n",
            requested, capped);
}

std::string DA3VramCapWarningMessage() {
    if (!g_da3_vram_warning.active) {
        return {};
    }
    return StringPrintf(
        "GPU memory pressure reduced DA3 inference resolution from %d to %d px. "
        "Close other GPU applications and re-run for higher depth quality.",
        g_da3_vram_warning.requested, g_da3_vram_warning.capped);
}

}  // namespace colmap
