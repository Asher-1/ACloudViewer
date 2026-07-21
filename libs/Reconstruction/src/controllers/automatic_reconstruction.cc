// Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#include "controllers/automatic_reconstruction.h"

#include <cstdlib>
#include <cstring>
#include <filesystem>

#include "base/undistortion.h"
#include "base/database.h"
#include "controllers/da3_depth_controller.h"
#include "controllers/incremental_mapper.h"
#include "controllers/texturing_controller.h"
#include "feature/extraction.h"
#include "feature/matching.h"
#include "mvs/da3_fusion.h"
#include "mvs/fusion.h"
#include "mvs/meshing.h"
#include "mvs/patch_match.h"
#include "util/download.h"
#include "util/misc.h"
#include "util/option_manager.h"
#include "util/ply.h"
#include "util/ply_point_filter.h"
#include "util/reconstruction_log.h"

namespace colmap {

namespace {

constexpr size_t kDA3ColmapFusionFallbackMinPoints = 5000;

bool EnvFlagEnabled(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

bool DA3SkipGeometricRefineRequested() {
  if (EnvFlagEnabled("DA3_FULL_PATCHMATCH")) {
    return false;
  }
  return EnvFlagEnabled("DA3_SKIP_GEOMETRIC_REFINE");
}

struct DirectFusionQualityGate {
  size_t min_points = 0;
  bool should_fallback = false;
  // True when multi-view voxel consensus rejects too many depth samples.
  bool poor_consensus = false;
  double consensus_acceptance_rate = -1.0;
};

DirectFusionQualityGate EvaluateDirectFusionQuality(
    const mvs::DA3FusionResult& fusion, int num_reg_images) {
  DirectFusionQualityGate gate;
  const int num_images = std::max(num_reg_images, 0);
  const size_t num_points = fusion.points.size();

  constexpr size_t kMinPointsPerImage = 500;
  // Enough per-image output to trust voxel fusion without PatchMatch refine.
  constexpr size_t kHealthyPointsPerImage = 25000;
  // Multi-view consistency: accepted / (accepted + skipped inconsistent).
  constexpr double kMinConsensusAcceptanceRate = 0.03;
  // Minimum accepted-sample rate vs valid depth pixels (catches 2440-point runs).
  constexpr double kMinAcceptanceRate = 1e-5;
  // Lower bound: expect at least this fraction of valid depth pixels as points.
  constexpr double kMinPointsPerValidPixel = 2e-5;
  // Upper cap on required minimum (avoid huge scenes demanding tens of millions).
  constexpr double kMaxPointsPerValidPixel = 0.02;

  const size_t per_image_floor =
      static_cast<size_t>(num_images) * kMinPointsPerImage;
  const size_t absolute_floor =
      std::max(kDA3ColmapFusionFallbackMinPoints, per_image_floor);
  gate.min_points = absolute_floor;

  const size_t consensus_total =
      fusion.num_accepted_samples + fusion.num_skipped_samples;
  if (consensus_total > 0) {
    gate.consensus_acceptance_rate =
        static_cast<double>(fusion.num_accepted_samples) /
        static_cast<double>(consensus_total);
    if (gate.consensus_acceptance_rate < kMinConsensusAcceptanceRate) {
      gate.poor_consensus = true;
      gate.should_fallback = true;
      gate.min_points = std::max(
          gate.min_points,
          static_cast<size_t>(num_images) * kHealthyPointsPerImage);
      return gate;
    }
  }

  if (const char* env_min = std::getenv("DA3_DIRECT_FUSION_MIN_POINTS");
      env_min != nullptr && env_min[0] != '\0') {
    char* end = nullptr;
    const unsigned long parsed = std::strtoul(env_min, &end, 10);
    if (end != env_min && parsed > 0) {
      gate.min_points = static_cast<size_t>(parsed);
      gate.should_fallback = num_points < gate.min_points;
      return gate;
    }
  }

  if (num_images > 0 &&
      num_points >= static_cast<size_t>(num_images) * kHealthyPointsPerImage) {
    gate.min_points =
        static_cast<size_t>(num_images) * kHealthyPointsPerImage;
    gate.should_fallback = false;
    return gate;
  }

  if (fusion.num_valid_depth_pixels > 0) {
    const size_t density_min = static_cast<size_t>(
        static_cast<double>(fusion.num_valid_depth_pixels) *
        kMinPointsPerValidPixel);
    const size_t density_cap = std::max(
        absolute_floor,
        static_cast<size_t>(static_cast<double>(fusion.num_valid_depth_pixels) *
                            kMaxPointsPerValidPixel));
    gate.min_points = std::min(
        density_cap,
        std::max(gate.min_points, density_min));
  }

  if (num_points < absolute_floor) {
    gate.should_fallback = true;
    return gate;
  }

  if (num_points >= gate.min_points) {
    gate.should_fallback = false;
    return gate;
  }

  // Below density-based minimum: fallback only when acceptance is also poor.
  if (fusion.num_valid_depth_pixels > 0 && fusion.num_accepted_samples > 0) {
    const double acceptance_rate =
        static_cast<double>(fusion.num_accepted_samples) /
        static_cast<double>(fusion.num_valid_depth_pixels);
    gate.should_fallback = acceptance_rate < kMinAcceptanceRate;
    return gate;
  }

  gate.should_fallback = num_points < gate.min_points;
  return gate;
}

mvs::StereoFusionOptions ApplyDA3ColmapStereoFusionProfile(
    const mvs::StereoFusionOptions& base,
    int num_reg_images,
    AutomaticReconstructionController::Quality quality) {
  mvs::StereoFusionOptions opts = base;
  opts.min_num_pixels = std::min(3, opts.min_num_pixels);
  opts.max_reproj_error = std::max(opts.max_reproj_error, 3.0);
  opts.max_normal_error = std::max(opts.max_normal_error, 25.0);

  switch (quality) {
    case AutomaticReconstructionController::Quality::LOW:
      opts.min_num_pixels = std::min(2, opts.min_num_pixels);
      opts.max_depth_error = std::max(opts.max_depth_error, 0.05);
      opts.check_num_images =
          std::max(opts.check_num_images, num_reg_images);
      break;
    case AutomaticReconstructionController::Quality::MEDIUM:
      opts.min_num_pixels = std::min(3, opts.min_num_pixels);
      opts.max_depth_error = std::max(opts.max_depth_error, 0.04);
      break;
    case AutomaticReconstructionController::Quality::HIGH:
    case AutomaticReconstructionController::Quality::EXTREME:
      opts.min_num_pixels = std::max(2, std::min(3, opts.min_num_pixels));
      opts.max_depth_error = std::max(opts.max_depth_error, 0.035);
      opts.max_traversal_depth = std::max(opts.max_traversal_depth, 100);
      break;
    default:
      opts.max_depth_error = std::max(opts.max_depth_error, 0.04);
      break;
  }

  if (num_reg_images > 0 && num_reg_images <= 12) {
    opts.check_num_images = std::max(opts.check_num_images, num_reg_images);
  }
  return opts;
}

colmap::mvs::DA3FusionOptions MakeDA3FusionOptions(
    const AutomaticReconstructionController::Options& options,
    int max_image_size,
    const colmap::mvs::StereoFusionOptions& stereo_fusion) {
  colmap::mvs::DA3FusionOptions fusion_options;
  fusion_options.max_image_size = max_image_size;
  fusion_options.min_num_pixels = 1;
  fusion_options.min_num_views = 2;
  fusion_options.use_voxel_consensus = true;
  fusion_options.use_dense_fusion = false;
  fusion_options.restrict_to_overlapping_views = true;
  fusion_options.max_num_pixels = stereo_fusion.max_num_pixels;
  fusion_options.max_traversal_depth = stereo_fusion.max_traversal_depth;
  fusion_options.check_num_images = stereo_fusion.check_num_images;
  fusion_options.num_threads = options.num_threads;

  // DA3-specific fusion thresholds (do not affect native COLMAP StereoFusion).
  // COLMAP StereoFusion emits ~one point per fused pixel group at full resolution
  // (often 10^5–10^6 points). DA3 voxel consensus merges multi-view samples into
  // one point per cell; cell size dominates output count (not the UI "Fused point
  // filter", which is skipped for DA3 via skip_voxel_downsample).
  fusion_options.max_reproj_error =
      std::max(stereo_fusion.max_reproj_error, 3.0);
  fusion_options.max_depth_error =
      std::max(stereo_fusion.max_depth_error, 0.05);
  fusion_options.max_point_dist = 0.028;
  fusion_options.max_normal_error =
      std::max(stereo_fusion.max_normal_error, 22.0);

  switch (options.quality) {
    case AutomaticReconstructionController::Quality::LOW:
      fusion_options.pixel_stride = 1;
      fusion_options.fusion_voxel_size = 0.008;
      fusion_options.max_depth_error = 0.035;
      fusion_options.max_point_dist = 0.020;
      break;
    case AutomaticReconstructionController::Quality::MEDIUM:
      fusion_options.pixel_stride = 1;
      fusion_options.fusion_voxel_size = 0.008;
      break;
    case AutomaticReconstructionController::Quality::HIGH:
    case AutomaticReconstructionController::Quality::EXTREME:
      fusion_options.pixel_stride = 1;
      // ~6 mm cell: with ~7e5 accepted samples targets ~3e5–4e5 points
      // (COLMAP fused.ply on mini6 is ~3.5e5), while point_dist still blocks ghosts.
      fusion_options.fusion_voxel_size = 0.006;
      break;
    default:
      fusion_options.pixel_stride = 1;
      fusion_options.fusion_voxel_size = 0.008;
      break;
  }
  return fusion_options;
}

colmap::mvs::DA3FusionOptions MakeDA3DirectPriorFusionOptions(
    const AutomaticReconstructionController::Options& options,
    int max_image_size,
    const colmap::mvs::StereoFusionOptions& stereo_fusion) {
  colmap::mvs::DA3FusionOptions fusion_options =
      MakeDA3FusionOptions(options, max_image_size, stereo_fusion);
  // Raw DA3 priors are per-view monocular depth. Keep thresholds stricter than
  // the old 0.15m / 0.08 settings that passed ~98% inconsistent samples yet
  // still emitted ghosted multi-shell clouds.
  fusion_options.min_num_views = 2;
  fusion_options.min_num_pixels = 1;
  fusion_options.max_depth_error = std::max(fusion_options.max_depth_error, 0.06);
  fusion_options.max_reproj_error = std::max(fusion_options.max_reproj_error, 4.0);
  fusion_options.max_point_dist = 0.07;
  fusion_options.max_normal_error = std::max(fusion_options.max_normal_error, 25.0);
  fusion_options.fusion_voxel_size =
      std::max(fusion_options.fusion_voxel_size, 0.008);
  return fusion_options;
}

colmap::mvs::DA3FusionOptions MakeDA3DirectPriorFusionRetryOptions(
    const colmap::mvs::DA3FusionOptions& base) {
  colmap::mvs::DA3FusionOptions fusion_options = base;
  fusion_options.min_num_views = 2;
  fusion_options.max_depth_error = std::max(fusion_options.max_depth_error, 0.08);
  fusion_options.max_reproj_error = std::max(fusion_options.max_reproj_error, 4.5);
  fusion_options.max_point_dist = std::max(fusion_options.max_point_dist, 0.10);
  fusion_options.fusion_voxel_size =
      std::max(fusion_options.fusion_voxel_size, 0.010);
  fusion_options.check_num_images = 0;
  return fusion_options;
}

std::vector<PlyPoint> SparsePointsToPly(const Reconstruction& reconstruction) {
  std::vector<PlyPoint> points;
  points.reserve(reconstruction.NumPoints3D());
  for (const auto& [_, point3D] : reconstruction.Points3D()) {
    PlyPoint point;
    point.x = static_cast<float>(point3D.XYZ(0));
    point.y = static_cast<float>(point3D.XYZ(1));
    point.z = static_cast<float>(point3D.XYZ(2));
    point.r = point3D.Color(0);
    point.g = point3D.Color(1);
    point.b = point3D.Color(2);
    points.push_back(point);
  }
  return points;
}

void RemovePathIfExists(const std::string& path) {
  std::error_code ec;
  if (ExistsFile(path)) {
    std::filesystem::remove(path, ec);
  } else if (ExistsDir(path)) {
    std::filesystem::remove_all(path, ec);
  }
}

void LoadWorkspaceSparseIfEmpty(ReconstructionManager* reconstruction_manager,
                                const std::string& workspace_path) {
  if (reconstruction_manager == nullptr || reconstruction_manager->Size() > 0) {
    return;
  }
  const auto sparse_path = JoinPaths(workspace_path, "sparse");
  if (!ExistsDir(sparse_path)) {
    return;
  }
  auto dir_list = GetDirList(sparse_path);
  std::sort(dir_list.begin(), dir_list.end());
  for (const auto& dir : dir_list) {
    reconstruction_manager->Read(dir);
  }
  if (reconstruction_manager->Size() > 0) {
    RECON_LOG_DEBUG("Loaded %zu existing sparse reconstruction(s) from workspace.\n", reconstruction_manager->Size());
  }
}

std::vector<std::string> RegisteredImageNames(const Reconstruction& reconstruction) {
  std::vector<std::string> names;
  names.reserve(reconstruction.NumRegImages());
  for (const auto image_id : reconstruction.RegImageIds()) {
    names.push_back(reconstruction.Image(image_id).Name());
  }
  return names;
}

std::vector<std::string> UndistortedImagePaths(
    const std::string& dense_path, const Reconstruction& reconstruction) {
  std::vector<std::string> paths;
  paths.reserve(reconstruction.NumRegImages());
  for (const auto image_id : reconstruction.RegImageIds()) {
    paths.push_back(
        JoinPaths(dense_path, "images", reconstruction.Image(image_id).Name()));
  }
  return paths;
}

mvs::PatchMatchOptions MakeDA3PatchMatchRefineOptions(
    const mvs::PatchMatchOptions& base) {
  mvs::PatchMatchOptions opts = base;
  opts.geom_consistency = true;

  const char* full_patchmatch = std::getenv("DA3_FULL_PATCHMATCH");
  if (full_patchmatch != nullptr && full_patchmatch[0] != '\0' &&
      std::strcmp(full_patchmatch, "0") != 0) {
    opts.skip_photometric_pass = false;
    opts.filter = true;
    opts.photometric_force_recompute = true;
    opts.photometric_use_existing_as_init = true;
    return opts;
  }

  // Fast path: DA3 metric depth replaces COLMAP photometric PatchMatch.
  opts.skip_photometric_pass = true;
  opts.filter = false;
  opts.photometric_force_recompute = false;
  opts.photometric_use_existing_as_init = false;
  opts.num_iterations = std::min(3, opts.num_iterations);
  return opts;
}

}  // namespace

AutomaticReconstructionController::AutomaticReconstructionController(
    const Options& options, ReconstructionManager* reconstruction_manager)
    : options_(options),
      reconstruction_manager_(reconstruction_manager),
      active_thread_(nullptr) {
  CHECK(ExistsDir(options_.workspace_path));
  CHECK(ExistsDir(options_.image_path));
  CHECK_NOTNULL(reconstruction_manager_);

  option_manager_.AddAllOptions();

  *option_manager_.image_path = options_.image_path;
  *option_manager_.database_path =
      JoinPaths(options_.workspace_path, "database.db");

  if (options_.data_type == DataType::VIDEO) {
    option_manager_.ModifyForVideoData();
  } else if (options_.data_type == DataType::INDIVIDUAL) {
    option_manager_.ModifyForIndividualData();
  } else if (options_.data_type == DataType::INTERNET) {
    option_manager_.ModifyForInternetData();
  } else {
    LOG(FATAL) << "Data type not supported";
  }

  CHECK(ExistsCameraModelWithName(options_.camera_model));

  if (options_.quality == Quality::LOW) {
    option_manager_.ModifyForLowQuality();
  } else if (options_.quality == Quality::MEDIUM) {
    option_manager_.ModifyForMediumQuality();
  } else if (options_.quality == Quality::HIGH) {
    option_manager_.ModifyForHighQuality();
  } else if (options_.quality == Quality::EXTREME) {
    option_manager_.ModifyForExtremeQuality();
  }

  option_manager_.sift_extraction->num_threads = options_.num_threads;
  option_manager_.sift_matching->num_threads = options_.num_threads;
  option_manager_.mapper->num_threads = options_.num_threads;
  option_manager_.poisson_meshing->num_threads = options_.num_threads;

  ImageReaderOptions reader_options = *option_manager_.image_reader;
  reader_options.database_path = *option_manager_.database_path;
  reader_options.image_path = *option_manager_.image_path;
  if (!options_.mask_path.empty()) {
    reader_options.mask_path = options_.mask_path;
    option_manager_.image_reader->mask_path = options_.mask_path;
  }
  reader_options.single_camera = options_.single_camera;
  reader_options.camera_model = options_.camera_model;

  option_manager_.sift_extraction->use_gpu = options_.use_gpu;
  option_manager_.sift_matching->use_gpu = options_.use_gpu;
  option_manager_.mapper->ba_use_gpu = options_.use_gpu;
  option_manager_.bundle_adjustment->use_gpu = options_.use_gpu;

  option_manager_.sift_extraction->gpu_index = options_.gpu_index;
  option_manager_.sift_matching->gpu_index = options_.gpu_index;
  option_manager_.patch_match_stereo->gpu_index = options_.gpu_index;
  option_manager_.mapper->ba_gpu_index = options_.gpu_index;
  option_manager_.bundle_adjustment->gpu_index = options_.gpu_index;

  feature_extractor_.reset(new SiftFeatureExtractor(
      reader_options, *option_manager_.sift_extraction));

  exhaustive_matcher_.reset(new ExhaustiveFeatureMatcher(
      *option_manager_.exhaustive_matching, *option_manager_.sift_matching,
      *option_manager_.database_path));

  // Resolve vocab_tree_path: use default if empty, and download/cache if URI
  std::string resolved_vocab_tree_path = options_.vocab_tree_path;
  if (resolved_vocab_tree_path.empty()) {
    resolved_vocab_tree_path = retrieval::kDefaultVocabTreeUri;
  }
  
  // Automatically download and cache if URI format is provided
  if (!resolved_vocab_tree_path.empty()) {
#ifdef COLMAP_DOWNLOAD_ENABLED
    // Download and cache the file if it's a URI, otherwise use as local path
    resolved_vocab_tree_path = MaybeDownloadAndCacheFile(resolved_vocab_tree_path).string();
    option_manager_.sequential_matching->loop_detection = true;
    option_manager_.sequential_matching->vocab_tree_path = resolved_vocab_tree_path;
#else
    // If download is disabled, check if it's a URI and warn, otherwise use as local path
    if (resolved_vocab_tree_path.find("http://") == 0 ||
        resolved_vocab_tree_path.find("https://") == 0 ||
        resolved_vocab_tree_path.find(';') != std::string::npos) {
      LOG(WARNING) << "vocab_tree_path appears to be a URI but download support is disabled. "
                   << "Please provide a local path or enable DOWNLOAD_ENABLED.";
      resolved_vocab_tree_path = "";  // Clear to avoid using invalid URI
    } else {
      // Use as local path
      option_manager_.sequential_matching->loop_detection = true;
      option_manager_.sequential_matching->vocab_tree_path = resolved_vocab_tree_path;
    }
#endif
  }

  sequential_matcher_.reset(new SequentialFeatureMatcher(
      *option_manager_.sequential_matching, *option_manager_.sift_matching,
      *option_manager_.database_path));

  if (!resolved_vocab_tree_path.empty()) {
    option_manager_.vocab_tree_matching->vocab_tree_path = resolved_vocab_tree_path;
    vocab_tree_matcher_.reset(new VocabTreeFeatureMatcher(
        *option_manager_.vocab_tree_matching, *option_manager_.sift_matching,
        *option_manager_.database_path));
  }
}

void AutomaticReconstructionController::Stop() {
  if (active_thread_ != nullptr) {
    active_thread_->Stop();
  }
  Thread::Stop();
}

void AutomaticReconstructionController::Run() {
  if (IsStopped()) {
    return;
  }

  use_da3_stereo_maps_ =
      options_.dense &&
      options_.stereo_mode == StereoPipelineMode::DA3_DEPTH_INFERENCE &&
      DA3ModelSupportsStereo(options_.da3_stereo_model_type);
  da3_reuse_sparse_stereo_ =
      options_.sparse &&
      options_.sparse_mode == SparseModelMode::DA3_DEPTH_POSE &&
      use_da3_stereo_maps_ &&
      DA3ConfigsMatchForStereoReuse(
          DA3Config{options_.da3_sparse_model_type,
                    options_.da3_sparse_quant_type,
                    options_.da3_sparse_model_path,
                    options_.da3_sparse_metric_model_path},
          DA3Config{options_.da3_stereo_model_type,
                    options_.da3_stereo_quant_type,
                    options_.da3_stereo_model_path,
                    options_.da3_stereo_metric_model_path});

  const size_t num_da3_images = CountDA3Images(options_.image_path);
  const bool da3_colmap_hybrid_dense =
      options_.sparse_mode == SparseModelMode::DA3_DEPTH_POSE &&
      use_da3_stereo_maps_ &&
      num_da3_images >= static_cast<size_t>(kDA3ColmapSparseAutoMinViews);

  da3_auto_colmap_sparse_ = options_.sparse && da3_colmap_hybrid_dense;
  da3_patchmatch_refine_ = da3_colmap_hybrid_dense;
  da3_skip_geometric_refine_ =
      da3_patchmatch_refine_ &&
      (options_.da3_skip_geometric_refine || DA3SkipGeometricRefineRequested());

  da3_unified_undistorted_ =
      da3_reuse_sparse_stereo_ && use_da3_stereo_maps_ &&
      !da3_auto_colmap_sparse_ &&
      num_da3_images <= static_cast<size_t>(kDA3UnifiedUndistortedMaxViews);
  da3_multiview_cache_.valid = false;

  LoadWorkspaceSparseIfEmpty(reconstruction_manager_, options_.workspace_path);

  if (da3_auto_colmap_sparse_) {
    RECON_LOG_DEBUG("DA3: %zu images — using COLMAP sparse reconstruction for global "                  "camera poses; DA3 sequential per-view depth for dense.\n", num_da3_images);
    if (da3_patchmatch_refine_) {
      if (da3_skip_geometric_refine_) {
        RECON_LOG_DEBUG("DA3: dense stage uses metric depth priors + DA3 voxel "                      "fusion (skip geometric refine; auto-fallback to "                      "PatchMatch geometric if fusion is sparse).\n");
      } else {
        RECON_LOG_DEBUG("DA3: dense stage uses metric depth priors (replace "                      "PatchMatch photometric) + fast geometric refine + "                      "StereoFusion.\n");
      }
    }
  } else if (da3_unified_undistorted_) {
    RECON_LOG_DEBUG("DA3: unified undistorted mode (sequential per-view depth on "                  "undistorted images; workspace sparse synced from dense)\n");
  } else if (da3_reuse_sparse_stereo_) {
    RECON_LOG_DEBUG("DA3: sparse and stereo share sequential depth inference cache\n");
  }

  // DA3 sparse model path: skip feature extraction/matching, use DA3 depth+pose
  if (options_.sparse_mode == SparseModelMode::DA3_DEPTH_POSE &&
      !da3_auto_colmap_sparse_) {
    if (options_.sparse) {
      RunDA3SparseMapper();
    }
  } else {
    const bool workspace_sparse_ready =
        reconstruction_manager_->Size() > 0 &&
        ExistsDir(JoinPaths(options_.workspace_path, "sparse"));
    const bool dense_only_resume =
        !options_.sparse && workspace_sparse_ready;
    const bool skip_feature_pipeline =
        dense_only_resume ||
        (workspace_sparse_ready && options_.sparse &&
         !options_.da3_force_recompute);
    if (!skip_feature_pipeline) {
      RunFeatureExtraction();

      if (IsStopped()) {
        return;
      }

      RunFeatureMatching();

      if (IsStopped()) {
        return;
      }
    } else {
      RECON_LOG_DEBUG("Skipping feature extraction/matching (workspace sparse "                    "model already available).\n");
    }

    if (options_.sparse) {
      RunSparseMapper();
    }
  }

  if (IsStopped()) {
    return;
  }

  if (options_.dense) {
    if (options_.stereo_mode == StereoPipelineMode::DA3_DEPTH_INFERENCE &&
        options_.sparse_mode != SparseModelMode::DA3_DEPTH_POSE &&
        !da3_auto_colmap_sparse_) {
      RECON_LOG_WARN("WARNING: DA3 depth inference is configured with a non-DA3 "                    "sparse model. Camera poses may not match metric depth; "                    "prefer Sparse mode = DA3 (depth+pose).\n");
    }
    if (options_.stereo_mode == StereoPipelineMode::DA3_DEPTH_INFERENCE &&
        !use_da3_stereo_maps_) {
      RECON_LOG_WARN("WARNING: DA3 depth inference requires a nested model "                    "(Nested AnyView / Nested Metric). "                    "Falling back to COLMAP PatchMatch stereo.\n");
    }
    if (use_da3_stereo_maps_) {
      RunDA3DepthMaps();
      if (IsStopped()) return;
    }
    RunDenseMapper();
  }

  GetTimer().PrintMinutes();
}

void AutomaticReconstructionController::RunFeatureExtraction() {
  CHECK(feature_extractor_);
  active_thread_ = feature_extractor_.get();
  feature_extractor_->Start();
  feature_extractor_->Wait();
  feature_extractor_.reset();
  active_thread_ = nullptr;
}

void AutomaticReconstructionController::RunFeatureMatching() {
  Thread* matcher = nullptr;
  if (options_.data_type == DataType::VIDEO) {
    matcher = sequential_matcher_.get();
  } else if (options_.data_type == DataType::INDIVIDUAL ||
             options_.data_type == DataType::INTERNET) {
    Database database(*option_manager_.database_path);
    const size_t num_images = database.NumImages();
    // Use vocab tree matcher if it was created (vocab_tree_path was resolved) and num_images >= 200
    if (vocab_tree_matcher_ && num_images >= 200) {
      matcher = vocab_tree_matcher_.get();
    } else {
      matcher = exhaustive_matcher_.get();
    }
  }

  CHECK(matcher);
  active_thread_ = matcher;
  matcher->Start();
  matcher->Wait();
  exhaustive_matcher_.reset();
  sequential_matcher_.reset();
  vocab_tree_matcher_.reset();
  active_thread_ = nullptr;
}

void AutomaticReconstructionController::RunSparseMapper() {
  const auto sparse_path = JoinPaths(options_.workspace_path, "sparse");
  if (ExistsDir(sparse_path)) {
    auto dir_list = GetDirList(sparse_path);
    std::sort(dir_list.begin(), dir_list.end());
    if (dir_list.size() > 0) {
      RECON_LOG_WARN("WARNING: Skipping sparse reconstruction because it is "                    "already computed\n");
      for (const auto& dir : dir_list) {
        reconstruction_manager_->Read(dir);
      }
      return;
    }
  }

  IncrementalMapperController mapper(
      option_manager_.mapper.get(), *option_manager_.image_path,
      *option_manager_.database_path, reconstruction_manager_);
  active_thread_ = &mapper;
  mapper.Start();
  mapper.Wait();
  active_thread_ = nullptr;

  CreateDirIfNotExists(sparse_path);
  reconstruction_manager_->Write(sparse_path, &option_manager_);
}

void AutomaticReconstructionController::RunDA3SparseMapper() {
  const auto sparse_path = JoinPaths(options_.workspace_path, "sparse");
  const auto sparse_0 = JoinPaths(sparse_path, "0");
  const std::string sparse_marker = JoinPaths(sparse_0, "images.bin");
  const std::string undistorted_sync_marker =
      JoinPaths(sparse_0, ".da3_undistorted_sync");

  if (options_.da3_force_recompute) {
    RemovePathIfExists(sparse_0);
  }

  if (ExistsDir(sparse_path)) {
    auto dir_list = GetDirList(sparse_path);
    std::sort(dir_list.begin(), dir_list.end());
    if (!dir_list.empty()) {
      const bool synced_undistorted = ExistsFile(undistorted_sync_marker);
      const std::string freshness_root =
          da3_unified_undistorted_ && synced_undistorted
              ? JoinPaths(options_.workspace_path, "dense", "0", "images")
              : options_.image_path;
      if (!DA3OutputsAreStale(freshness_root, sparse_marker,
                              options_.da3_force_recompute)) {
        RECON_LOG_WARN(
                "WARNING: Skipping DA3 sparse reconstruction because it is "
                "already computed%s\n",
                synced_undistorted ? " (undistorted sync)" : "");
        for (const auto& dir : dir_list) {
          reconstruction_manager_->Read(dir);
        }
        if (reconstruction_manager_->Size() > 0 &&
            !ExistsFile(*option_manager_.database_path)) {
          WriteDA3PlaceholderDatabase(*option_manager_.database_path,
                                      reconstruction_manager_->Get(0));
        }
        return;
      }
    }
  }

  if (da3_unified_undistorted_) {
    RECON_LOG_DEBUG(
            "========================================\n"
            "Running DA3 EXIF placeholder sparse (skip bootstrap inference)\n"
            "========================================\n");

    CreateDirIfNotExists(sparse_path);
    CreateDirIfNotExists(sparse_0);
    const double focal_factor =
        option_manager_.image_reader->default_focal_length_factor;
    if (WriteExifPlaceholderSparseModel(options_.image_path, sparse_0,
                                        focal_factor)) {
      reconstruction_manager_->Read(sparse_0);
      if (reconstruction_manager_->Size() > 0) {
        WriteDA3PlaceholderDatabase(*option_manager_.database_path,
                                    reconstruction_manager_->Get(0));
      }
      return;
    }

    RECON_LOG_WARN(
            "WARNING: EXIF placeholder sparse failed; falling back to "
            "DA3 bootstrap inference on original images.\n");
    RECON_LOG_DEBUG(
            "========================================\n"
            "Running DA3 bootstrap sparse (original images, for undistort)\n"
            "========================================\n");
  } else {
    RECON_LOG_DEBUG(
            "========================================\n"
            "Running DA3 depth+pose sparse model generation\n"
            "========================================\n");
  }

  DA3Config da3_config;
  da3_config.model_type = options_.da3_sparse_model_type;
  da3_config.quant_type = options_.da3_sparse_quant_type;
  da3_config.model_path = options_.da3_sparse_model_path;
  da3_config.metric_model_path = options_.da3_sparse_metric_model_path;
  da3_config.num_threads = options_.num_threads;
  da3_config.sparse_mode = SparseModelMode::DA3_DEPTH_POSE;

  if (da3_config.model_path.empty()) {
    da3_config.model_path = DA3DepthController::ResolveModelPath(da3_config);
    if (da3_config.model_path.empty()) {
      RECON_LOG_ERROR("ERROR: DA3 model could not be resolved. Skipping DA3 sparse model generation.\n");
      return;
    }
  }

  RECON_LOG_DEBUG("DA3 sparse: model_path=%s  image_path=%s\n", da3_config.model_path.c_str(), options_.image_path.c_str());

  DA3DepthController da3_controller(
      da3_config, options_.image_path, options_.workspace_path);
  if (da3_reuse_sparse_stereo_ && !da3_unified_undistorted_) {
    da3_controller.SetMultiviewCacheOut(&da3_multiview_cache_);
  }
  active_thread_ = &da3_controller;
  da3_controller.Start();
  da3_controller.Wait();
  active_thread_ = nullptr;

  // Read back the generated sparse model
  if (ExistsDir(sparse_0)) {
    reconstruction_manager_->Read(sparse_0);
    RECON_LOG_DEBUG("DA3 sparse: loaded %zu reconstruction(s) from %s\n", reconstruction_manager_->Size(), sparse_0.c_str());

    if (reconstruction_manager_->Size() > 0) {
      WriteDA3PlaceholderDatabase(*option_manager_.database_path,
                                  reconstruction_manager_->Get(0));
    }
  } else {
    RECON_LOG_ERROR("ERROR: DA3 sparse model generation produced no output at %s.  Check stderr / glog for details.\n", sparse_0.c_str());
  }
}

void AutomaticReconstructionController::RunDA3DepthMaps() {
  if (!DA3ModelSupportsStereo(options_.da3_stereo_model_type)) {
    RECON_LOG_ERROR("ERROR: DA3 depth map generation requires a nested model.\n");
    return;
  }

  DA3ClearVramCapWarning();

  RECON_LOG_DEBUG("========================================\n");
  if (da3_patchmatch_refine_) {
    RECON_LOG_DEBUG("Running DA3 metric depth priors (PatchMatch refine follows)\n");
  } else {
    RECON_LOG_DEBUG("Running DA3 depth map generation (replacing PatchMatch stereo)\n");
  }
  RECON_LOG_DEBUG("========================================\n");

  DA3Config da3_config;
  da3_config.model_type = options_.da3_stereo_model_type;
  da3_config.quant_type = options_.da3_stereo_quant_type;
  da3_config.model_path = options_.da3_stereo_model_path;
  da3_config.metric_model_path = options_.da3_stereo_metric_model_path;
  da3_config.num_threads = options_.num_threads;
  da3_config.max_image_size = option_manager_.patch_match_stereo->max_image_size;
  da3_config.stereo_mode = StereoPipelineMode::DA3_DEPTH_INFERENCE;

  // Resolve model path once before the loop to avoid repeated
  // filesystem probes / downloads for each reconstruction index.
  if (da3_config.model_path.empty()) {
    da3_config.model_path = DA3DepthController::ResolveModelPath(da3_config);
    if (da3_config.model_path.empty()) {
      RECON_LOG_ERROR("ERROR: DA3 model could not be resolved. Skipping DA3 depth map generation.\n");
      return;
    }
  }

  if (reconstruction_manager_->Size() == 0) {
    RECON_LOG_WARN("WARNING: DA3 depth map generation skipped — no sparse "                  "reconstructions available.  Run sparse reconstruction first.\n");
    return;
  }

  RECON_LOG_DEBUG("DA3 depth maps: model_path=%s  reconstructions=%zu\n", da3_config.model_path.c_str(), reconstruction_manager_->Size());

  CreateDirIfNotExists(JoinPaths(options_.workspace_path, "dense"));

  for (size_t i = 0; i < reconstruction_manager_->Size(); ++i) {
    if (IsStopped()) return;

    const std::string dense_path =
        JoinPaths(options_.workspace_path, "dense", std::to_string(i));
    const std::string stereo_marker =
        JoinPaths(dense_path, "stereo", "fusion.cfg");

    if (options_.da3_force_recompute) {
      RemovePathIfExists(JoinPaths(dense_path, "stereo"));
      RemovePathIfExists(JoinPaths(dense_path, "fused.ply"));
      RemovePathIfExists(JoinPaths(dense_path, "fused.ply.vis"));
    }

    CreateDirIfNotExists(dense_path);

    const bool had_undist_images =
        ExistsDir(JoinPaths(dense_path, "images"));

    // Undistort images first
    if (!had_undist_images) {
      UndistortCameraOptions undistortion_options;
      undistortion_options.max_image_size =
          option_manager_.patch_match_stereo->max_image_size;
      COLMAPUndistorter undistorter(undistortion_options,
                                    &reconstruction_manager_->Get(i),
                                    *option_manager_.image_path, dense_path);
      active_thread_ = &undistorter;
      undistorter.Start();
      undistorter.Wait();
      active_thread_ = nullptr;
      // COLMAPUndistorter writes empty stereo/ skeleton; depth must be rebuilt.
      RemovePathIfExists(JoinPaths(dense_path, "stereo", "fusion.cfg"));
      RemovePathIfExists(JoinPaths(dense_path, "stereo", "patch-match.cfg"));
    }

    if (IsStopped()) return;

    const std::string undist_images = JoinPaths(dense_path, "images");
    if (ExistsDir(undist_images)) {
      const bool stereo_ready =
          da3_patchmatch_refine_
              ? DA3DepthPriorReady(dense_path)
              : DA3StereoDepthMapsReady(dense_path);
      if (stereo_ready &&
          ExistsFile(stereo_marker) &&
          !DA3OutputsAreStale(undist_images, stereo_marker,
                              options_.da3_force_recompute)) {
        RECON_LOG_WARN("WARNING: Skipping DA3 depth map generation for dense/%zu (stereo outputs already up to date)\n", i);
        continue;
      }
      if (!stereo_ready) {
        RECON_LOG_DEBUG("DA3: stereo depth maps missing or empty; regenerating.\n");
      }

      const auto& reconstruction = reconstruction_manager_->Get(i);
      const auto colmap_names = RegisteredImageNames(reconstruction);
      const auto undist_paths = UndistortedImagePaths(dense_path, reconstruction);

      DA3DepthController da3_controller(da3_config, undist_images, dense_path);
      da3_controller.SetColmapImageNames(colmap_names);
      da3_controller.SetExplicitImagePaths(undist_paths);
      const bool use_colmap_poses =
          options_.sparse_mode == SparseModelMode::COLMAP_NATIVE ||
          da3_auto_colmap_sparse_;
      da3_controller.SetUseColmapPosesOnly(use_colmap_poses);
      da3_controller.SetExportPhotometricPrior(da3_patchmatch_refine_);
      if (da3_skip_geometric_refine_) {
        da3_controller.SetFastDepthExport(true);
        da3_controller.SetExportMaxImageSize(
            option_manager_.stereo_fusion->max_image_size);
      }
      if (da3_unified_undistorted_) {
        da3_controller.SetMultiviewCacheIn(&da3_multiview_cache_);
        da3_controller.SetMultiviewCacheOut(&da3_multiview_cache_);
      }
      active_thread_ = &da3_controller;
      da3_controller.Start();
      da3_controller.Wait();
      active_thread_ = nullptr;

      if (!da3_controller.Success()) {
        RECON_LOG_ERROR("ERROR: DA3 depth map generation failed for dense/%zu. Dense fusion will be skipped.\n", i);
        continue;
      }

      if (da3_patchmatch_refine_) {
        RemoveColmapGeometricStereoMaps(dense_path);
      }

      if (da3_unified_undistorted_ &&
          SyncWorkspaceSparseFromDense(options_.workspace_path, dense_path,
                                       static_cast<int>(i))) {
        const auto workspace_sparse_0 =
            JoinPaths(options_.workspace_path, "sparse", "0");
        reconstruction_manager_->Clear();
        if (ExistsDir(workspace_sparse_0)) {
          reconstruction_manager_->Read(workspace_sparse_0);
          RECON_LOG_DEBUG("DA3 unified: reloaded workspace sparse from dense (%s images, undistorted coordinates)\n", reconstruction_manager_->Get(0).NumRegImages());
          if (!ExistsFile(*option_manager_.database_path)) {
            WriteDA3PlaceholderDatabase(*option_manager_.database_path,
                                        reconstruction_manager_->Get(0));
          }
        }
      }
    }
  }
}

void AutomaticReconstructionController::RunDenseMapper() {
  CreateDirIfNotExists(JoinPaths(options_.workspace_path, "dense"));

  for (size_t i = 0; i < reconstruction_manager_->Size(); ++i) {
    if (IsStopped()) {
      return;
    }

    const std::string dense_path =
        JoinPaths(options_.workspace_path, "dense", std::to_string(i));
    const std::string fused_path = JoinPaths(dense_path, "fused.ply");
    const std::string stereo_marker =
        JoinPaths(dense_path, "stereo", "fusion.cfg");

    std::string meshing_path;
    if (options_.mesher == Mesher::POISSON) {
      meshing_path = JoinPaths(dense_path, "meshed-poisson.ply");
    } else if (options_.mesher == Mesher::DELAUNAY) {
      meshing_path = JoinPaths(dense_path, "meshed-delaunay.ply");
    }

    const std::string undist_images = JoinPaths(dense_path, "images");
    const bool fusion_freshness_root_is_undist =
        use_da3_stereo_maps_ && ExistsDir(undist_images);
    const std::string fusion_freshness_root =
        fusion_freshness_root_is_undist ? undist_images : options_.image_path;

    if (options_.da3_force_recompute) {
      RemovePathIfExists(fused_path);
      RemovePathIfExists(fused_path + ".vis");
      RemovePathIfExists(meshing_path);
      if (da3_patchmatch_refine_) {
        RemoveColmapGeometricStereoMaps(dense_path);
      }
    }

    const bool dense_outputs_stale =
        da3_patchmatch_refine_
            ? (da3_skip_geometric_refine_
                   ? (!DA3DepthPriorReady(dense_path) ||
                      DA3OutputsAreStale(fusion_freshness_root, stereo_marker,
                                           options_.da3_force_recompute))
                   : (DA3PatchMatchRefineStale(dense_path) ||
                      !ColmapGeometricDepthMapsReady(dense_path) ||
                      DA3OutputsAreStale(fusion_freshness_root, stereo_marker,
                                         options_.da3_force_recompute)))
            : (use_da3_stereo_maps_ &&
               (!DA3StereoDepthMapsReady(dense_path) ||
                DA3OutputsAreStale(fusion_freshness_root, stereo_marker,
                                   options_.da3_force_recompute)));

    const std::string freshness_marker =
        use_da3_stereo_maps_ && ExistsFile(stereo_marker) ? stereo_marker
                                                          : fused_path;
    if (ExistsFile(fused_path) &&
        (!options_.meshing || ExistsFile(meshing_path)) &&
        !dense_outputs_stale &&
        !DA3OutputsAreStale(options_.image_path, freshness_marker,
                            options_.da3_force_recompute)) {
      continue;
    }

    // Image undistortion.

    if (!ExistsDir(dense_path)) {
      CreateDirIfNotExists(dense_path);

      UndistortCameraOptions undistortion_options;
      undistortion_options.max_image_size =
          option_manager_.patch_match_stereo->max_image_size;
      COLMAPUndistorter undistorter(undistortion_options,
                                    &reconstruction_manager_->Get(i),
                                    *option_manager_.image_path, dense_path);
      active_thread_ = &undistorter;
      undistorter.Start();
      undistorter.Wait();
      active_thread_ = nullptr;
    }

    if (IsStopped()) {
      return;
    }

    // Patch match stereo: native COLMAP, or DA3 priors + geometric refine.

    bool da3_used_fast_patchmatch = false;
    const bool run_patchmatch =
        (!use_da3_stereo_maps_ || da3_patchmatch_refine_) &&
        !da3_skip_geometric_refine_;
    if (run_patchmatch) {
#ifdef CUDA_ENABLED
      if (da3_patchmatch_refine_) {
        if (!DA3DepthPriorReady(dense_path)) {
          RECON_LOG_ERROR("ERROR: DA3 photometric depth priors missing for dense/%zu. Run DA3 depth stage first.\n", i);
          continue;
        }
        if (DA3PatchMatchRefineStale(dense_path) ||
            options_.da3_force_recompute) {
          RemoveColmapGeometricStereoMaps(dense_path);
          auto patch_match_options =
              MakeDA3PatchMatchRefineOptions(*option_manager_.patch_match_stereo);
          da3_used_fast_patchmatch = patch_match_options.skip_photometric_pass;
          if (patch_match_options.skip_photometric_pass) {
            RECON_LOG_DEBUG("DA3: fast PatchMatch geometric refine from metric "                          "depth priors (photometric pass skipped; set "                          "DA3_FULL_PATCHMATCH=1 for full NCC re-optimization).\n");
          } else {
            RECON_LOG_DEBUG("DA3: full PatchMatch refine from metric depth priors "                          "(photometric NCC re-optimization enabled).\n");
          }
          mvs::PatchMatchController patch_match_controller(
              patch_match_options, dense_path, "COLMAP", "");
          active_thread_ = &patch_match_controller;
          patch_match_controller.Start();
          patch_match_controller.Wait();
          active_thread_ = nullptr;
        } else {
          RECON_LOG_WARN("WARNING: Skipping PatchMatch refine for dense/%zu (geometric depth maps already up to date)\n", i);
        }
      } else {
        mvs::PatchMatchController patch_match_controller(
            *option_manager_.patch_match_stereo, dense_path, "COLMAP", "");
        active_thread_ = &patch_match_controller;
        patch_match_controller.Start();
        patch_match_controller.Wait();
        active_thread_ = nullptr;
      }
#else   // CUDA_ENABLED
      RECON_LOG_WARN(
              "WARNING: Skipping patch match stereo for dense/%zu because CUDA "
              "is not available.\n",
              i);
      continue;
#endif  // CUDA_ENABLED
    } else if (use_da3_stereo_maps_ && da3_skip_geometric_refine_) {
      RECON_LOG_DEBUG("DA3: skipping PatchMatch geometric refine; fusing DA3 "                    "priors directly.\n");
    } else if (use_da3_stereo_maps_) {
      RECON_LOG_DEBUG("Skipping PatchMatch stereo: using DA3 depth maps directly.\n");
    }

    if (IsStopped()) {
      return;
    }

    // Stereo fusion.

    const bool fused_exists = ExistsFile(fused_path);
    const bool fused_is_empty =
        fused_exists && ReadPly(fused_path).empty();
    const bool fusion_stale =
        da3_patchmatch_refine_
            ? (da3_skip_geometric_refine_
                   ? (!DA3DepthPriorReady(dense_path) ||
                      DA3OutputsAreStale(fusion_freshness_root, stereo_marker,
                                           options_.da3_force_recompute))
                   : (DA3PatchMatchRefineStale(dense_path) ||
                      !ColmapGeometricDepthMapsReady(dense_path) ||
                      DA3OutputsAreStale(fusion_freshness_root, stereo_marker,
                                         options_.da3_force_recompute)))
            : (use_da3_stereo_maps_ &&
               (!DA3StereoDepthMapsReady(dense_path) ||
                DA3OutputsAreStale(fusion_freshness_root, stereo_marker,
                                   options_.da3_force_recompute)));
    if (!fused_exists || fused_is_empty || fusion_stale ||
        options_.da3_force_recompute) {
      if (da3_patchmatch_refine_) {
        if (!ExistsFile(stereo_marker) || !DA3DepthPriorReady(dense_path)) {
          RECON_LOG_ERROR("ERROR: DA3+PatchMatch stereo outputs missing for dense/%zu. Skipping fusion.\n", i);
          continue;
        }
        if (!da3_skip_geometric_refine_ &&
            !ColmapGeometricDepthMapsReady(dense_path)) {
          RECON_LOG_ERROR("ERROR: DA3+PatchMatch stereo outputs missing for dense/%zu. Skipping fusion.\n", i);
          continue;
        }
      } else if (use_da3_stereo_maps_ &&
                 (!ExistsFile(stereo_marker) ||
                  !DA3StereoDepthMapsReady(dense_path))) {
        RECON_LOG_ERROR("ERROR: DA3 stereo outputs missing for dense/%zu (fusion.cfg or depth maps). Skipping fusion — "                      "depth inference may have failed (e.g. GPU out of memory).\n", i);
        continue;
      }

      FusedPointCloudWithVisibility fused_cloud;
      bool used_da3_custom_fusion = false;
      if (use_da3_stereo_maps_ && !da3_patchmatch_refine_) {
        const auto da3_fusion_options = MakeDA3FusionOptions(
            options_, option_manager_.patch_match_stereo->max_image_size,
            *option_manager_.stereo_fusion);
        RECON_LOG_DEBUG("Using DA3 voxel fusion on geometric depth maps "                      "(consensus_cell=%fm, stride=%d, min_views=%d, depth_err=%f).\n", da3_fusion_options.fusion_voxel_size, da3_fusion_options.pixel_stride, da3_fusion_options.min_num_views, da3_fusion_options.max_depth_error);
        const auto da3_fusion_result =
            mvs::FuseDA3DepthMaps(dense_path, da3_fusion_options);
        fused_cloud.points = da3_fusion_result.points;
        fused_cloud.visibility = da3_fusion_result.visibility;
        used_da3_custom_fusion = true;

        if (fused_cloud.points.size() < kDA3ColmapFusionFallbackMinPoints) {
          const int num_reg_images =
              reconstruction_manager_->Get(i).NumRegImages();
          auto fusion_options = ApplyDA3ColmapStereoFusionProfile(
              *option_manager_.stereo_fusion, num_reg_images, options_.quality);
          fusion_options.num_threads = options_.num_threads;
          RECON_LOG_WARN(
                  "WARNING: DA3 voxel fusion produced %zu points (< %zu); "
                  "falling back to COLMAP StereoFusion (profile: min_pixels=%d, "
                  "depth_err=%f).\n",
                  fused_cloud.points.size(), kDA3ColmapFusionFallbackMinPoints,
                  fusion_options.min_num_pixels, fusion_options.max_depth_error);
          mvs::StereoFusion colmap_fuser(fusion_options, dense_path, "COLMAP",
                                         "", "geometric");
          active_thread_ = &colmap_fuser;
          colmap_fuser.Start();
          colmap_fuser.Wait();
          active_thread_ = nullptr;
          fused_cloud.points = colmap_fuser.GetFusedPoints();
          fused_cloud.visibility = colmap_fuser.GetFusedPointsVisibility();
          used_da3_custom_fusion = false;
        }
      } else {
        const int num_reg_images =
            reconstruction_manager_->Get(i).NumRegImages();
        auto fusion_options = ApplyDA3ColmapStereoFusionProfile(
            *option_manager_.stereo_fusion, num_reg_images, options_.quality);
        fusion_options.num_threads = options_.num_threads;
        const bool fuse_geometric_maps =
            da3_patchmatch_refine_ && !da3_skip_geometric_refine_ &&
            ColmapGeometricDepthMapsReady(dense_path);
        const std::string fusion_input_type =
            fuse_geometric_maps ||
                    (!da3_skip_geometric_refine_ &&
                     (options_.quality == Quality::HIGH ||
                      options_.quality == Quality::EXTREME))
                ? "geometric"
                : "photometric";

        colmap::mvs::DA3FusionOptions da3_fusion_options;
        mvs::DA3FusionResult direct_fusion_result;
        if (da3_skip_geometric_refine_) {
          da3_fusion_options = MakeDA3DirectPriorFusionOptions(
              options_, option_manager_.stereo_fusion->max_image_size,
              *option_manager_.stereo_fusion);
          RECON_LOG_DEBUG("Using DA3 voxel fusion on photometric depth priors "                        "(consensus_cell=%fm, depth_err=%f, point_dist=%f, min_views=%d).\n", da3_fusion_options.fusion_voxel_size, da3_fusion_options.max_depth_error, da3_fusion_options.max_point_dist, da3_fusion_options.min_num_views);
          direct_fusion_result = mvs::FuseDA3DepthMaps(
              dense_path, da3_fusion_options, "photometric");
          fused_cloud.points = direct_fusion_result.points;
          fused_cloud.visibility = direct_fusion_result.visibility;
        } else {
          if (da3_patchmatch_refine_) {
            RECON_LOG_DEBUG("Using COLMAP StereoFusion on PatchMatch-refined "                          "geometric depth maps.\n");
          }
          mvs::StereoFusion fuser(fusion_options, dense_path, "COLMAP", "",
                                  fusion_input_type);
          active_thread_ = &fuser;
          fuser.Start();
          fuser.Wait();
          active_thread_ = nullptr;
          fused_cloud.points = fuser.GetFusedPoints();
          fused_cloud.visibility = fuser.GetFusedPointsVisibility();
        }

        auto direct_fusion_gate = EvaluateDirectFusionQuality(
            direct_fusion_result, num_reg_images);
        if (da3_skip_geometric_refine_ && direct_fusion_gate.should_fallback) {
          const auto run_patchmatch_geometric_fallback = [&]() {
            RECON_LOG_WARN("WARNING: falling back to PatchMatch geometric refine "                          "(close SIBR/BEV viewers to free GPU memory).\n");
            RemoveColmapGeometricStereoMaps(dense_path);
            auto patch_match_options = MakeDA3PatchMatchRefineOptions(
                *option_manager_.patch_match_stereo);
            da3_used_fast_patchmatch = patch_match_options.skip_photometric_pass;
            mvs::PatchMatchController patch_match_controller(
                patch_match_options, dense_path, "COLMAP", "");
            active_thread_ = &patch_match_controller;
            patch_match_controller.Start();
            patch_match_controller.Wait();
            active_thread_ = nullptr;

            mvs::StereoFusion retry_fuser(fusion_options, dense_path, "COLMAP",
                                          "", "geometric");
            active_thread_ = &retry_fuser;
            retry_fuser.Start();
            retry_fuser.Wait();
            active_thread_ = nullptr;
            fused_cloud.points = retry_fuser.GetFusedPoints();
            fused_cloud.visibility = retry_fuser.GetFusedPointsVisibility();
          };

          if (direct_fusion_gate.consensus_acceptance_rate >= 0.0) {
            RECON_LOG_WARN(
                    "WARNING: direct DA3 prior fusion produced %zu points "
                    "(adaptive min %zu, valid_depth_pixels=%zu, "
                    "accepted_samples=%zu, skipped_samples=%zu, "
                    "consensus_acceptance=%.3f%%).\n",
                    fused_cloud.points.size(), direct_fusion_gate.min_points,
                    direct_fusion_result.num_valid_depth_pixels,
                    direct_fusion_result.num_accepted_samples,
                    direct_fusion_result.num_skipped_samples,
                    direct_fusion_gate.consensus_acceptance_rate * 100.0);
          } else {
            RECON_LOG_WARN(
                    "WARNING: direct DA3 prior fusion produced %zu points "
                    "(adaptive min %zu, valid_depth_pixels=%zu, "
                    "accepted_samples=%zu, skipped_samples=%zu).\n",
                    fused_cloud.points.size(), direct_fusion_gate.min_points,
                    direct_fusion_result.num_valid_depth_pixels,
                    direct_fusion_result.num_accepted_samples,
                    direct_fusion_result.num_skipped_samples);
          }

          if (direct_fusion_gate.poor_consensus) {
            RECON_LOG_WARN("WARNING: multi-view depth consensus too low; skipping "                          "relaxed fusion retry.\n");
            run_patchmatch_geometric_fallback();
          } else {
            RECON_LOG_DEBUG("Retrying with relaxed fusion thresholds.\n");
            const auto relaxed_fusion_options =
                MakeDA3DirectPriorFusionRetryOptions(da3_fusion_options);
            const auto relaxed_fusion_result = mvs::FuseDA3DepthMaps(
                dense_path, relaxed_fusion_options, "photometric");
            const auto relaxed_gate = EvaluateDirectFusionQuality(
                relaxed_fusion_result, num_reg_images);
            if (!relaxed_gate.should_fallback &&
                relaxed_fusion_result.points.size() >
                    direct_fusion_result.points.size()) {
              RECON_LOG_DEBUG("DA3 relaxed fusion produced %zu points; skipping PatchMatch geometric refine.\n", relaxed_fusion_result.points.size());
              fused_cloud.points = relaxed_fusion_result.points;
              fused_cloud.visibility = relaxed_fusion_result.visibility;
              direct_fusion_gate = relaxed_gate;
            } else {
              RECON_LOG_WARN("WARNING: relaxed DA3 fusion produced %zu points "                            "(adaptive min %zu).\n", relaxed_fusion_result.points.size(), relaxed_gate.min_points);
              run_patchmatch_geometric_fallback();
            }
          }
        } else if (da3_skip_geometric_refine_ &&
                   !direct_fusion_gate.should_fallback) {
          RECON_LOG_DEBUG("DA3 direct fusion accepted %zu points (adaptive min %zu); skipping PatchMatch geometric refine.\n", fused_cloud.points.size(), direct_fusion_gate.min_points);
        } else if (da3_patchmatch_refine_ && da3_used_fast_patchmatch &&
                   fused_cloud.points.size() < kDA3ColmapFusionFallbackMinPoints) {
          RECON_LOG_WARN(
                  "WARNING: fast DA3 PatchMatch fusion produced %zu points (< "
                  "%zu); falling back to full photometric+geometric PatchMatch.\n",
                  fused_cloud.points.size(), kDA3ColmapFusionFallbackMinPoints);
          RemoveColmapGeometricStereoMaps(dense_path);
          auto full_patch_match_options = *option_manager_.patch_match_stereo;
          full_patch_match_options.geom_consistency = true;
          full_patch_match_options.skip_photometric_pass = false;
          full_patch_match_options.filter = true;
          full_patch_match_options.photometric_force_recompute = true;
          full_patch_match_options.photometric_use_existing_as_init = true;
          mvs::PatchMatchController full_patch_match_controller(
              full_patch_match_options, dense_path, "COLMAP", "");
          active_thread_ = &full_patch_match_controller;
          full_patch_match_controller.Start();
          full_patch_match_controller.Wait();
          active_thread_ = nullptr;

          mvs::StereoFusion retry_fuser(fusion_options, dense_path, "COLMAP",
                                        "", fusion_input_type);
          active_thread_ = &retry_fuser;
          retry_fuser.Start();
          retry_fuser.Wait();
          active_thread_ = nullptr;
          fused_cloud.points = retry_fuser.GetFusedPoints();
          fused_cloud.visibility = retry_fuser.GetFusedPointsVisibility();
        }
      }

      if (fused_cloud.points.empty() && use_da3_stereo_maps_) {
        RECON_LOG_ERROR("ERROR: DA3 dense fusion failed; not writing sparse "                      "points3D into fused.ply (would cause fan-like ghosting). "                      "Enable Force recompute or check dense/0/stereo depth maps.\n");
      }
      if (options_.fused_point_filter.enabled && !fused_cloud.points.empty()) {
        const auto unfiltered_cloud = fused_cloud;
        const size_t before = fused_cloud.points.size();
        auto filter_options = options_.fused_point_filter;
        filter_options.num_threads = options_.num_threads;
        if (use_da3_stereo_maps_ && used_da3_custom_fusion) {
          filter_options.skip_voxel_downsample = true;
        }
        fused_cloud =
            FilterFusedPlyPointsWithVisibility(fused_cloud, filter_options);
        if (fused_cloud.points.empty()) {
          RECON_LOG_WARN("WARNING: Fused point filter removed all %zu points; writing unfiltered fusion output.\n", before);
          fused_cloud = unfiltered_cloud;
        }
      }
      if (fused_cloud.visibility.size() != fused_cloud.points.size()) {
        fused_cloud.visibility.assign(fused_cloud.points.size(),
                                      std::vector<int>{});
      }
      RECON_LOG_DEBUG("Writing output: %s\n", fused_path.c_str());
      WriteBinaryPlyPoints(fused_path, fused_cloud.points);
      if (!fused_cloud.points.empty()) {
        mvs::WritePointsVisibility(fused_path + ".vis", fused_cloud.visibility);
      }

      // Hook for derived classes
      OnFusedPointsGenerated(i, fused_cloud.points);
    }

    if (IsStopped()) {
      return;
    }

    if (IsStopped()) {
      return;
    }

    // Surface meshing.

    const bool has_fused_points = ExistsFile(fused_path) &&
                                  std::filesystem::file_size(fused_path) > 0;

    if (!options_.meshing) {
      RECON_LOG_DEBUG("Skipping surface meshing (disabled in options).\n");
    } else if (!ExistsFile(meshing_path)) {
      if (!has_fused_points) {
        RECON_LOG_WARN("WARNING: Skipping surface meshing because stereo fusion "                      "produced no points.\n");
      } else if (IsStopped()) {
        RECON_LOG_DEBUG("Skipping surface meshing (cancelled).\n");
      } else if (options_.mesher == Mesher::POISSON) {
        RECON_LOG_DEBUG("Starting Poisson meshing...\n");
        mvs::PoissonMeshing(*option_manager_.poisson_meshing, fused_path,
                            meshing_path);
      } else if (options_.mesher == Mesher::DELAUNAY) {
#ifdef CGAL_ENABLED
        RECON_LOG_DEBUG("Starting Delaunay meshing (this step cannot be "                      "interrupted until it finishes)...\n");
        mvs::DenseDelaunayMeshing(*option_manager_.delaunay_meshing, dense_path,
                                  meshing_path);
#else  // CGAL_ENABLED
        RECON_LOG_WARN("WARNING: Skipping Delaunay meshing because CGAL is "                      "not available.\n");
        return;

#endif  // CGAL_ENABLED
      }
      
      // Hook for derived classes
      if (ExistsFile(meshing_path)) {
        OnMeshGenerated(i, meshing_path);
      }
    } else {
      // Mesh already exists, notify derived classes
      OnMeshGenerated(i, meshing_path);
    }

    if (IsStopped()) {
      return;
    }

    // Surface texturing.
    if (options_.texturing && options_.meshing) {
      const std::string textured_path =
              JoinPaths(dense_path, "textured-mesh.obj");
      if (!ExistsFile(textured_path) && ExistsFile(meshing_path)) {
        option_manager_.texturing->meshed_file_path = meshing_path;
        option_manager_.texturing->textured_file_path = textured_path;

        // Set mesh_source based on which mesher was used
        if (options_.mesher == Mesher::POISSON) {
          option_manager_.texturing->mesh_source = "poisson";
        } else if (options_.mesher == Mesher::DELAUNAY) {
          option_manager_.texturing->mesh_source = "delaunay";
        }

        TexturingReconstruction texturing(
                *option_manager_.texturing,
                reconstruction_manager_->Get(i),
                *option_manager_.image_path, dense_path);
        active_thread_ = &texturing;
        texturing.Start();
        texturing.Wait();
        active_thread_ = nullptr;

        if (ExistsFile(textured_path)) {
          RECON_LOG_DEBUG("Writing textured mesh: %s\n", textured_path.c_str());
          // Hook for derived classes
          OnTexturedMeshGenerated(i, textured_path);
        }
      } else if (ExistsFile(textured_path)) {
        // Textured mesh already exists, notify derived classes
        OnTexturedMeshGenerated(i, textured_path);
      }
    }
  }
}

}  // namespace colmap
