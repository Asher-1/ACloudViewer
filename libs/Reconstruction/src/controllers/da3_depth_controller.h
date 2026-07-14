// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "util/threading.h"

namespace colmap {

// DA3 model type: base/large/giant/nested
enum class DA3ModelType {
    BASE,
    LARGE,
    GIANT,
    NESTED_METRIC,
    NESTED_ANYVIEW
};

// DA3 quantization type
enum class DA3QuantType {
    F32,
    F16,
    Q8_0,
    Q4_K
};

// Returns the GGUF filename for a given model/quant combination.
std::string DA3ModelFilename(DA3ModelType model, DA3QuantType quant);

// Returns the download URL for the GGUF model.
std::string DA3ModelDownloadURL(DA3ModelType model, DA3QuantType quant);

// Returns the download URI (url;name;sha256) for DownloadAndCacheFile.
// If SHA256 is not yet known, returns the URL-only format for direct download.
std::string DA3ModelDownloadURI(DA3ModelType model, DA3QuantType quant);

// Returns the list of quantization types supported for a given model type.
std::vector<DA3QuantType> DA3SupportedQuantTypes(DA3ModelType model);

// Returns true if the given model+quant combination is a known GGUF model.
bool DA3ModelExists(DA3ModelType model, DA3QuantType quant);

// True for nested anyview/metric models (da_capi_load_nested).
bool DA3ModelIsNested(DA3ModelType model);

// DA3 stereo / dense depth maps require nested metric alignment.
bool DA3ModelSupportsStereo(DA3ModelType model);

// Returns the cache directory path consistent with the qDA3 plugin.
std::string DA3ModelCacheDir();

// Sparse model generation mode
enum class SparseModelMode {
    COLMAP_NATIVE,
    DA3_DEPTH_POSE
};

// Dense/stereo pipeline mode
enum class StereoPipelineMode {
    COLMAP_PATCH_MATCH,
    DA3_DEPTH_INFERENCE
};

// Configuration for DA3 integration
struct DA3Config {
    DA3ModelType model_type = DA3ModelType::BASE;
    DA3QuantType quant_type = DA3QuantType::Q8_0;

    // Path to GGUF model file (auto-downloaded if empty)
    std::string model_path;

    // For nested metric depth: path to the metric model
    std::string metric_model_path;

    int num_threads = -1;

    SparseModelMode sparse_mode = SparseModelMode::COLMAP_NATIVE;
    StereoPipelineMode stereo_mode = StereoPipelineMode::COLMAP_PATCH_MATCH;
};

// Controller for running DA3 depth estimation on a set of images and producing
// a COLMAP-compatible sparse model or depth maps for dense reconstruction.
class DA3DepthController : public Thread {
public:
    using ProgressCallback = std::function<void(int current, int total, const std::string& status)>;

    DA3DepthController(const DA3Config& config,
                       const std::string& image_path,
                       const std::string& output_path);

    void SetProgressCallback(ProgressCallback cb) { progress_cb_ = std::move(cb); }

    // Generate COLMAP sparse model from DA3 depth+pose estimation.
    // Output: cameras.bin, images.bin, points3D.bin in output_path/sparse/0/
    bool GenerateSparseModel();

    // Generate per-image depth maps for the dense pipeline.
    // Output: .geometric.bin depth maps in output_path/stereo/depth_maps/
    bool GenerateDepthMaps();

    // Resolve (download if needed) the model path. Returns empty on failure.
    static std::string ResolveModelPath(const DA3Config& config);

protected:
    void Run() override;

private:
    DA3Config config_;
    std::string image_path_;
    std::string output_path_;
    ProgressCallback progress_cb_;
};

}  // namespace colmap
