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

class Reconstruction;

// DA3 model type: base/large/giant/nested
enum class DA3ModelType { BASE, LARGE, GIANT, NESTED_METRIC, NESTED_ANYVIEW };

// DA3 quantization type
enum class DA3QuantType { F32, F16, Q8_0, Q4_K };

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

// True for nested anyview/metric models (aicore_depth_load_nested).
bool DA3ModelIsNested(DA3ModelType model);

// DA3 stereo / dense depth maps require nested metric alignment.
bool DA3ModelSupportsStereo(DA3ModelType model);

// Returns the cache directory path consistent with the qDA3 plugin.
std::string DA3ModelCacheDir();

struct DA3ModelCacheNeed {
    std::string filename;
    std::string url;
    std::string* dest_path = nullptr;
};

// Resolve cached GGUF paths when present; append missing files to `needed`.
void CollectDA3ModelCacheNeeds(const std::string& cache_dir,
                               DA3ModelType model_type,
                               DA3QuantType quant_type,
                               std::string& model_path,
                               std::string& metric_model_path,
                               std::vector<DA3ModelCacheNeed>& needed);

// Sparse model generation mode
enum class SparseModelMode { COLMAP_NATIVE, DA3_DEPTH_POSE };

// Dense/stereo pipeline mode
enum class StereoPipelineMode { COLMAP_PATCH_MATCH, DA3_DEPTH_INFERENCE };

// Configuration for DA3 integration
struct DA3Config {
    DA3ModelType model_type = DA3ModelType::BASE;
    DA3QuantType quant_type = DA3QuantType::Q8_0;

    // Path to GGUF model file (auto-downloaded if empty)
    std::string model_path;

    // For nested metric depth: path to the metric model
    std::string metric_model_path;

    int num_threads = -1;

    // COLMAP PatchMatch max_image_size; when >0, DA3 preprocess longest side
    // is aligned to min(image long edge, max_image_size).
    int max_image_size = -1;

    SparseModelMode sparse_mode = SparseModelMode::COLMAP_NATIVE;
    StereoPipelineMode stereo_mode = StereoPipelineMode::COLMAP_PATCH_MATCH;
};

// Absolute path + COLMAP Image.Name() relative to the image root.
struct DA3ImageEntry {
    std::string abs_path;
    std::string colmap_name;
};

// Cached multiview inference (sparse step) for reuse in the stereo step.
struct DA3MultiviewCache {
    DA3Config config;
    int h = 0;
    int w = 0;
    int n = 0;
    std::vector<std::string> colmap_names;
    std::vector<float> depth;
    std::vector<float> ext;
    std::vector<float> intr;
    bool valid = false;
    // True when depth/poses were computed on dense/.../images (undistorted).
    bool undistorted_images = false;
};

bool DA3ConfigsMatchForStereoReuse(const DA3Config& sparse,
                                   const DA3Config& stereo);

// Copy dense/<i>/sparse into workspace/sparse/<i> (unified undistorted poses).
bool SyncWorkspaceSparseFromDense(const std::string& workspace_path,
                                  const std::string& dense_path,
                                  int reconstruction_index = 0);

// Minimal sparse/0 from EXIF intrinsics + identity poses (undistort bootstrap).
bool WriteExifPlaceholderSparseModel(const std::string& image_root,
                                     const std::string& sparse_output_path,
                                     double default_focal_length_factor = 1.2);

// Longest-side preprocess target aligned with undistorted image size.
int ComputeDA3ImgResizeTarget(const std::vector<std::string>& image_paths,
                              int max_image_size);

// Count images under image_root (recursive), same set as
// CollectDA3ImageEntries.
size_t CountDA3Images(const std::string& image_root);

// At or above this count, automatic reconstruction uses COLMAP SfM for globally
// consistent camera poses and DA3 sequential per-view depth only. Per-view DA3
// poses are not mutually consistent; fusion requires COLMAP (or joint MV)
// poses.
constexpr int kDA3ColmapSparseAutoMinViews = 3;

// Unified undistorted bootstrap (EXIF placeholder) is only valid for small
// sets.
constexpr int kDA3UnifiedUndistortedMaxViews = 8;

// Collect images under image_root (recursive), sorted by colmap_name.
std::vector<DA3ImageEntry> CollectDA3ImageEntries(
        const std::string& image_root);

// Write a minimal database.db (cameras + images, no features) for DA3-only
// runs.
void WriteDA3PlaceholderDatabase(const std::string& database_path,
                                 const Reconstruction& reconstruction);

// True when any source image is newer than the output marker file/dir.
bool DA3OutputsAreStale(const std::string& image_root,
                        const std::string& output_marker_path,
                        bool force_recompute);

// True when dense/<i>/stereo/depth_maps contains at least one non-empty map.
bool DA3StereoDepthMapsReady(const std::string& dense_path);

// True when DA3 exported PatchMatch photometric priors (.photometric.bin).
bool DA3DepthPriorReady(const std::string& dense_path);

// True when PatchMatch geometric refinement outputs exist (.geometric.bin).
bool ColmapGeometricDepthMapsReady(const std::string& dense_path);

// True when photometric priors are newer than geometric outputs (or geometric
// missing).
bool DA3PatchMatchRefineStale(const std::string& dense_path);

// Remove PatchMatch geometric depth/normal maps (before re-refine from DA3
// priors).
void RemoveColmapGeometricStereoMaps(const std::string& dense_path);

// Last DA3 VRAM-based preprocess cap (set during sequential depth inference).
struct DA3VramCapWarning {
    bool active = false;
    int requested = 0;
    int capped = 0;
};

void DA3ClearVramCapWarning();
const DA3VramCapWarning& DA3PeekVramCapWarning();
std::string DA3VramCapWarningMessage();
void DA3NoteVramCap(int requested, int capped);

// Controller for running DA3 depth estimation on a set of images and producing
// a COLMAP-compatible sparse model or depth maps for dense reconstruction.
class DA3DepthController : public Thread {
public:
    using ProgressCallback = std::function<void(
            int current, int total, const std::string& status)>;

    DA3DepthController(const DA3Config& config,
                       const std::string& image_path,
                       const std::string& output_path);

    void SetProgressCallback(ProgressCallback cb) {
        progress_cb_ = std::move(cb);
    }

    void SetMultiviewCacheOut(DA3MultiviewCache* cache_out) {
        multiview_cache_out_ = cache_out;
    }

    // When set, GenerateDepthMaps reads this cache instead of re-inferring.
    void SetMultiviewCacheIn(const DA3MultiviewCache* cache_in) {
        multiview_cache_in_ = cache_in;
    }

    // Optional COLMAP image names (parallel to sorted image files). When empty,
    // names are derived from CollectDA3ImageEntries(image_path_).
    void SetColmapImageNames(std::vector<std::string> names) {
        colmap_image_names_ = std::move(names);
    }

    // When set, use these absolute image paths (and matching
    // colmap_image_names_) instead of scanning image_path_.
    void SetExplicitImagePaths(std::vector<std::string> abs_paths) {
        explicit_image_paths_ = std::move(abs_paths);
    }

    // When true, keep COLMAP undistorted sparse poses for export/fusion; do not
    // overwrite dense/sparse with per-view DA3 poses.
    void SetUseColmapPosesOnly(bool v) { use_colmap_poses_only_ = v; }

    // When true, write .photometric.bin priors for COLMAP PatchMatch geometric
    // refinement instead of final .geometric.bin maps.
    void SetExportPhotometricPrior(bool v) { export_photometric_prior_ = v; }

    // Cap exported depth/normal resolution (-1 = match undistorted image size).
    void SetExportMaxImageSize(int v) { export_max_image_size_ = v; }

    // Bilinear upsample only (skip guided filter); much faster for direct
    // fusion.
    void SetFastDepthExport(bool v) { fast_depth_export_ = v; }

    // Generate COLMAP sparse model from DA3 depth+pose estimation.
    // Output: cameras.bin, images.bin, points3D.bin in output_path/sparse/0/
    // (undistorted workspace cameras live in output_path/sparse/ from COLMAP
    // undistort)
    bool GenerateSparseModel();

    // Generate per-image depth maps for the dense pipeline.
    // Output: .geometric.bin or .photometric.bin in
    // output_path/stereo/depth_maps/ If multiview_cache is valid and matches
    // config, skips model inference.
    bool GenerateDepthMaps(const DA3MultiviewCache* multiview_cache = nullptr);

    // Resolve (download if needed) the model path. Returns empty on failure.
    static std::string ResolveModelPath(const DA3Config& config);

    bool Success() const { return success_; }

protected:
    void Run() override;

private:
    bool success_ = true;
    DA3Config config_;
    std::string image_path_;
    std::string output_path_;
    ProgressCallback progress_cb_;
    DA3MultiviewCache* multiview_cache_out_ = nullptr;
    const DA3MultiviewCache* multiview_cache_in_ = nullptr;
    std::vector<std::string> colmap_image_names_;
    std::vector<std::string> explicit_image_paths_;
    bool use_colmap_poses_only_ = false;
    bool export_photometric_prior_ = false;
    int export_max_image_size_ = -1;
    bool fast_depth_export_ = false;
};

}  // namespace colmap
