// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "controllers/da3_depth_controller.h"

#include "util/logging.h"
#include "util/misc.h"

#ifdef DA3_ENABLED
#include "da_capi.h"
#include "path_util.hpp"
#endif

#ifdef COLMAP_DOWNLOAD_ENABLED
#include "util/download.h"
#endif

#include "util/bitmap.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
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

// Estimate per-pixel surface normals from a depth map via central finite
// differences.  Matches plugins/core/Standard/qDA3/scripts/da3_to_colmap_stereo.py
// and COLMAP camera convention (x-right, y-down, z-forward).
std::vector<float> ComputeNormalsFromDepth(const float* depth, int w, int h) {
    const size_t num_pixels = static_cast<size_t>(w) * h;
    std::vector<float> normals(num_pixels * 3, 0.0f);

    auto idx = [w](int row, int col) {
        return static_cast<size_t>(row) * w + col;
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

            float dz_dx = 0.0f;
            float dz_dy = 0.0f;
            if (col > 0 && col < w - 1) {
                dz_dx = 0.5f * (depth[idx(row, col + 1)] - depth[idx(row, col - 1)]);
            }
            if (row > 0 && row < h - 1) {
                dz_dy = 0.5f * (depth[idx(row + 1, col)] - depth[idx(row - 1, col)]);
            }

            const float nx = -dz_dx;
            const float ny = -dz_dy;
            const float nz = 1.0f;
            const float norm =
                std::sqrt(nx * nx + ny * ny + nz * nz);
            if (norm > 1e-10f) {
                normals[p * 3 + 0] = nx / norm;
                normals[p * 3 + 1] = ny / norm;
                normals[p * 3 + 2] = nz / norm;
            } else {
                normals[p * 3 + 0] = 0.0f;
                normals[p * 3 + 1] = 0.0f;
                normals[p * 3 + 2] = 1.0f;
            }
        }
    }
    return normals;
}

bool IsImageFile(const std::string& path) {
    return HasFileExtension(path, ".jpg") || HasFileExtension(path, ".jpeg") ||
           HasFileExtension(path, ".png") || HasFileExtension(path, ".bmp") ||
           HasFileExtension(path, ".tiff") || HasFileExtension(path, ".tif") ||
           HasFileExtension(path, ".webp") || HasFileExtension(path, ".gif") ||
           HasFileExtension(path, ".ppm") || HasFileExtension(path, ".pgm") ||
           HasFileExtension(path, ".pbm");
}

std::vector<std::string> CollectImagePaths(const std::string& image_dir) {
    const auto file_list = GetFileList(image_dir);
    std::vector<std::string> paths;
    for (const auto& p : file_list) {
        if (IsImageFile(p)) paths.push_back(p);
    }
    std::sort(paths.begin(), paths.end());
    return paths;
}

#ifdef DA3_ENABLED
struct DepthPoseMultiResult {
    int h = 0;
    int w = 0;
    int n = 0;
    std::vector<float> depth;
    std::vector<float> ext;
    std::vector<float> intr;
};

da_ctx* LoadDA3Context(const DA3Config& config, int n_threads) {
    if (config.model_type != DA3ModelType::NESTED_METRIC &&
        config.model_type != DA3ModelType::NESTED_ANYVIEW) {
        const std::string model_path = DA3DepthController::ResolveModelPath(config);
        if (model_path.empty()) {
            return nullptr;
        }
        return da_capi_load(model_path.c_str(), n_threads);
    }

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
    return da_capi_load_nested(anyview_path.c_str(), metric_path.c_str(), n_threads);
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

void FilterMetricDepthInPlace(std::vector<float>& depth,
                              float min_depth = 0.01f,
                              float max_depth = 100.0f) {
    for (float& d : depth) {
        if (d <= min_depth || d > max_depth || !std::isfinite(d)) {
            d = 0.0f;
        }
    }
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

bool RunDepthPoseMulti(da_ctx* ctx, const std::vector<std::string>& image_paths,
                       DepthPoseMultiResult& result) {
    if (!ctx || image_paths.empty()) {
        return false;
    }

    const int N = static_cast<int>(image_paths.size());
    std::vector<const char*> cpaths(N);
    for (int i = 0; i < N; ++i) {
        cpaths[i] = image_paths[i].c_str();
    }

    result.ext.resize(static_cast<size_t>(N) * 12);
    result.intr.resize(static_cast<size_t>(N) * 9);

    float* depth = da_capi_depth_pose_multi(
        ctx, cpaths.data(), N, &result.h, &result.w, &result.n,
        result.ext.data(), result.intr.data());
    if (!depth || result.n <= 0 || result.h <= 0 || result.w <= 0) {
        if (depth) {
            da_capi_free_floats(depth);
        }
        return false;
    }

    const size_t per_view =
        static_cast<size_t>(result.h) * static_cast<size_t>(result.w);
    result.depth.assign(depth, depth + static_cast<size_t>(result.n) * per_view);
    da_capi_free_floats(depth);
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

#endif  // DA3_ENABLED

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
#ifdef DA3_ENABLED
    return da::default_model_cache_dir();
#else
    return ".cache/da3_models";
#endif
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
        LOG(INFO) << "DA3: Using cached model: " << cached_path;
        return cached_path.string();
    }

    LOG(INFO) << "DA3: Downloading model from: " << url;
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
    LOG(INFO) << "DA3: Model cached at: " << cached_path;
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
    if (config_.sparse_mode == SparseModelMode::DA3_DEPTH_POSE) {
        GenerateSparseModel();
    }
    if (config_.stereo_mode == StereoPipelineMode::DA3_DEPTH_INFERENCE) {
        if (!DA3ModelSupportsStereo(config_.model_type)) {
            LOG(ERROR) << "DA3: stereo depth inference requires a nested model "
                       << "(Nested AnyView or Nested Metric)";
            std::cout << "ERROR: DA3 stereo requires nested model. "
                      << "Select Nested AnyView/Metric or use COLMAP PatchMatch."
                      << std::endl;
            return;
        }
        GenerateDepthMaps();
    }
}

bool DA3DepthController::GenerateSparseModel() {
#ifndef DA3_ENABLED
    LOG(ERROR) << "DA3: DA3 core library not enabled. Cannot run depth estimation.";
    std::cout << "ERROR: DA3 core library not enabled (DA3_ENABLED not set at "
                 "compile time). Rebuild with -DDA3_ENABLED=ON." << std::endl;
    return false;
#else
    const auto image_paths = CollectImagePaths(image_path_);
    if (image_paths.empty()) {
        LOG(ERROR) << "DA3: No images found in: " << image_path_;
        std::cout << "ERROR: DA3 no images found in: " << image_path_ << std::endl;
        return false;
    }

    const int n_threads = config_.num_threads > 0 ? config_.num_threads : 4;
    const int N = static_cast<int>(image_paths.size());

    std::cout << "DA3 GenerateSparseModel: multiview depth+pose (threads="
              << n_threads << ", images=" << N << ")" << std::endl;

    da_ctx* ctx = LoadDA3Context(config_, n_threads);
    if (!ctx) {
        LOG(ERROR) << "DA3: Failed to load model";
        std::cout << "ERROR: DA3 failed to load model" << std::endl;
        return false;
    }

    if (progress_cb_) {
        progress_cb_(0, N, "DA3 multiview depth+pose+export");
    }

    std::vector<const char*> cpaths(static_cast<size_t>(N));
    for (int i = 0; i < N; ++i) {
        cpaths[static_cast<size_t>(i)] = image_paths[i].c_str();
    }

    const std::string sparse_path = JoinPaths(output_path_, "sparse", "0");
    if (da_capi_export_colmap_multi(ctx, cpaths.data(), N, sparse_path.c_str(), 1) != 0) {
        LOG(ERROR) << "DA3: da_capi_export_colmap_multi failed: "
                   << da_capi_last_error(ctx);
        std::cout << "ERROR: DA3 sparse COLMAP export failed: "
                  << da_capi_last_error(ctx) << std::endl;
        da_capi_free(ctx);
        return false;
    }
    da_capi_free(ctx);

    if (IsStopped()) {
        return false;
    }

    LOG(INFO) << "DA3: COLMAP sparse model written to: " << sparse_path
              << " (" << N << " images, with back-projected points3D)";
    return true;
#endif  // DA3_ENABLED
}

bool DA3DepthController::GenerateDepthMaps() {
#ifndef DA3_ENABLED
    LOG(ERROR) << "DA3: DA3 core library not enabled. Cannot run depth estimation.";
    std::cout << "ERROR: DA3 core library not enabled (DA3_ENABLED not set at "
                 "compile time). Rebuild with -DDA3_ENABLED=ON." << std::endl;
    return false;
#else
    if (!DA3ModelSupportsStereo(config_.model_type)) {
        LOG(ERROR) << "DA3: stereo depth inference requires a nested model";
        std::cout << "ERROR: DA3 stereo requires nested model "
                     "(Nested AnyView / Nested Metric)."
                  << std::endl;
        return false;
    }

    const auto image_paths = CollectImagePaths(image_path_);
    if (image_paths.empty()) {
        LOG(ERROR) << "DA3: No images found in: " << image_path_;
        std::cout << "ERROR: DA3 no images found in: " << image_path_ << std::endl;
        return false;
    }

    const int n_threads = config_.num_threads > 0 ? config_.num_threads : 4;
    const int N = static_cast<int>(image_paths.size());

    std::cout << "DA3 GenerateDepthMaps: nested multiview (threads=" << n_threads
              << ", images=" << N << ")" << std::endl;

    da_ctx* ctx = LoadDA3Context(config_, n_threads);
    if (!ctx) {
        LOG(ERROR) << "DA3: Failed to load nested model";
        std::cout << "ERROR: DA3 failed to load nested model" << std::endl;
        return false;
    }

    if (progress_cb_) {
        progress_cb_(0, N, "DA3 multiview metric depth");
    }

    DepthPoseMultiResult multi;
    if (!RunDepthPoseMulti(ctx, image_paths, multi)) {
        LOG(ERROR) << "DA3: da_capi_depth_pose_multi failed: "
                   << da_capi_last_error(ctx);
        std::cout << "ERROR: DA3 multiview depth failed: "
                  << da_capi_last_error(ctx) << std::endl;
        da_capi_free(ctx);
        return false;
    }
    da_capi_free(ctx);

    if (IsStopped()) {
        return false;
    }

    const std::string stereo_path = JoinPaths(output_path_, "stereo");
    const std::string depth_maps_path = JoinPaths(stereo_path, "depth_maps");
    const std::string normal_maps_path = JoinPaths(stereo_path, "normal_maps");
    CreateDirIfNotExists(stereo_path);
    CreateDirIfNotExists(depth_maps_path);
    CreateDirIfNotExists(normal_maps_path);

    const size_t per_view =
        static_cast<size_t>(multi.h) * static_cast<size_t>(multi.w);

    std::vector<std::string> fusion_image_names;
    fusion_image_names.reserve(static_cast<size_t>(multi.n));

    for (int i = 0; i < multi.n; ++i) {
        if (IsStopped()) {
            return false;
        }

        if (progress_cb_) {
            progress_cb_(i + 1, multi.n,
                         "DA3 depth export: " + GetPathBaseName(image_paths[i]));
        }

        const float* view_depth = multi.depth.data() + static_cast<size_t>(i) * per_view;
        const std::string base = GetPathBaseName(image_paths[i]);
        fusion_image_names.push_back(base);

        Bitmap undist_bmp;
        int out_w = multi.w;
        int out_h = multi.h;
        if (undist_bmp.Read(image_paths[i])) {
            out_w = undist_bmp.Width();
            out_h = undist_bmp.Height();
        }

        std::vector<float> export_depth =
            UpsampleDepthBilinear(view_depth, multi.w, multi.h, out_w, out_h);
        FilterMetricDepthInPlace(export_depth);

        WriteColmapDepthMap(JoinPaths(depth_maps_path, base + ".geometric.bin"),
                            out_w, out_h, export_depth.data());

        const std::vector<float> normals =
            ComputeNormalsFromDepth(export_depth.data(), out_w, out_h);
        WriteColmapNormalMap(JoinPaths(normal_maps_path, base + ".geometric.bin"),
                             out_w, out_h, normals);
    }

    WriteStereoFusionConfig(stereo_path, fusion_image_names);

    LOG(INFO) << "DA3: Depth maps and normal maps written to: " << stereo_path
              << " (" << multi.n << " views, upsampled to undistorted resolution)";
    return true;
#endif  // DA3_ENABLED
}

}  // namespace colmap
