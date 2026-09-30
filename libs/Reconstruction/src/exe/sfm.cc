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

#include "exe/sfm.h"

#include "controllers/global_pipeline.h"
#include "controllers/rotation_averaging.h"
#include "estimators/view_graph_calibration.h"

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include "scene/reconstruction.h"
#include "scene/projection.h"
#include "sfm/observation_manager.h"
#include "controllers/automatic_reconstruction.h"
#include "controllers/da3_depth_controller.h"
#include "controllers/da3_pipeline_defaults.h"
#include "controllers/bundle_adjustment.h"
#include "controllers/hierarchical_pipeline.h"
#include "exe/gui.h"
#include "util/misc.h"
#include "util/opengl_utils.h"
#include "controllers/option_manager.h"

namespace colmap {

namespace {

DA3ModelType ParseDA3ModelType(const std::string& model) {
  if (model == "base") {
    return DA3ModelType::BASE;
  } else if (model == "large") {
    return DA3ModelType::LARGE;
  } else if (model == "giant") {
    return DA3ModelType::GIANT;
  } else if (model == "nested" || model == "nested_metric") {
    return DA3ModelType::NESTED_METRIC;
  } else if (model == "nested_anyview") {
    return DA3ModelType::NESTED_ANYVIEW;
  }
  LOG(FATAL) << "Invalid DA3 model provided. "
             << "Use 'base', 'large', 'giant', 'nested_metric', or "
                "'nested_anyview'.";
  return DA3ModelType::BASE;
}

DA3QuantType ParseDA3QuantType(const std::string& quant) {
  if (quant == "f32") {
    return DA3QuantType::F32;
  } else if (quant == "f16") {
    return DA3QuantType::F16;
  } else if (quant == "q8_0") {
    return DA3QuantType::Q8_0;
  } else if (quant == "q4_k") {
    return DA3QuantType::Q4_K;
  }
  LOG(FATAL) << "Invalid da3 quant provided. Use 'f32', 'f16', 'q8_0', or "
                "'q4_k'.";
  return DA3QuantType::Q8_0;
}

}  // namespace

int RunAutomaticReconstructor(int argc, char** argv) {
  AutomaticReconstructionController::Options reconstruction_options;
  std::string data_type = "individual";
  std::string quality = "high";
  std::string mesher = "poisson";
#if defined(AICore_ENABLED) && !defined(CUDA_ENABLED)
  std::string sparse_mode = "da3";
  std::string stereo_mode = "da3";
  std::string da3_model = "nested_anyview";
#else
  std::string sparse_mode = "colmap";
  std::string stereo_mode = "colmap";
  std::string da3_model = "base";
#endif
  std::string da3_quant = "q8_0";
  std::string da3_sparse_model;
  std::string da3_sparse_quant;
  std::string da3_stereo_model = "nested_anyview";
  std::string da3_stereo_quant;

  OptionManager options;
  options.AddRequiredOption("workspace_path",
                            &reconstruction_options.workspace_path);
  options.AddRequiredOption("image_path", &reconstruction_options.image_path);
  options.AddDefaultOption("mask_path", &reconstruction_options.mask_path);
  options.AddDefaultOption("vocab_tree_path",
                           &reconstruction_options.vocab_tree_path);
  options.AddDefaultOption("data_type", &data_type,
                           "{individual, video, internet}");
  options.AddDefaultOption("quality", &quality, "{low, medium, high, extreme}");
  options.AddDefaultOption("camera_model",
                           &reconstruction_options.camera_model);
  options.AddDefaultOption("single_camera",
                           &reconstruction_options.single_camera);
  options.AddDefaultOption("sparse", &reconstruction_options.sparse);
  options.AddDefaultOption("dense", &reconstruction_options.dense);
  options.AddDefaultOption("meshing", &reconstruction_options.meshing,
                           "Generate surface mesh from fused point cloud");
  options.AddDefaultOption("texturing", &reconstruction_options.texturing,
                           "Texture the reconstructed mesh");
  options.AddDefaultOption("mesher", &mesher,
                           "{poisson, delaunay, advancing_front}");
  options.AddDefaultOption("num_threads", &reconstruction_options.num_threads);
  options.AddDefaultOption("use_gpu", &reconstruction_options.use_gpu);
  options.AddDefaultOption("gpu_index", &reconstruction_options.gpu_index);
  options.AddDefaultOption("sparse_mode", &sparse_mode,
                           "{colmap, da3}");
  options.AddDefaultOption("stereo_mode", &stereo_mode,
                           "{colmap, da3}");
  options.AddDefaultOption("da3_model", &da3_model,
                           "{base, large, giant, nested_metric, nested_anyview} "
                           "(alias for da3_sparse_model)");
  options.AddDefaultOption("da3_quant", &da3_quant,
                           "{f32, f16, q8_0, q4_k} (alias for da3_sparse_quant)");
  options.AddDefaultOption("da3_sparse_model", &da3_sparse_model,
                           "{base, large, giant, nested_metric, nested_anyview}");
  options.AddDefaultOption("da3_sparse_quant", &da3_sparse_quant,
                           "{f32, f16, q8_0, q4_k}");
  options.AddDefaultOption("da3_stereo_model", &da3_stereo_model,
                           "{nested_metric, nested_anyview}");
  options.AddDefaultOption("da3_stereo_quant", &da3_stereo_quant,
                           "{f32, f16, q8_0, q4_k}");
  options.AddDefaultOption("da3_sparse_model_path",
                           &reconstruction_options.da3_sparse_model_path);
  options.AddDefaultOption("da3_sparse_metric_model_path",
                           &reconstruction_options.da3_sparse_metric_model_path,
                           "Metric model path for nested DA3 sparse step");
  options.AddDefaultOption("da3_stereo_model_path",
                           &reconstruction_options.da3_stereo_model_path);
  options.AddDefaultOption("da3_stereo_metric_model_path",
                           &reconstruction_options.da3_stereo_metric_model_path,
                           "Metric model path for nested DA3 stereo step");
  options.AddDefaultOption("da3_force_recompute",
                           &reconstruction_options.da3_force_recompute,
                           "Recompute DA3 sparse/stereo/fusion even if cached");
  options.AddDefaultOption("da3_skip_geometric_refine",
                           &reconstruction_options.da3_skip_geometric_refine,
                           "Fuse DA3 priors directly; skip PatchMatch geometric "
                           "(auto-fallback if sparse)");
  options.AddDefaultOption("fused_point_filter",
                           &reconstruction_options.fused_point_filter.enabled,
                           "Apply voxel + SOR cleanup before Poisson meshing");
  options.AddDefaultOption("fused_voxel_size",
                           &reconstruction_options.fused_point_filter.voxel_size,
                           "Voxel edge length for fused point downsampling (m)");
  options.AddDefaultOption("fused_sor_enabled",
                           &reconstruction_options.fused_point_filter.sor_enabled,
                           "Apply statistical outlier removal after voxel filter");
  options.AddDefaultOption("fused_sor_nb_neighbors",
                           &reconstruction_options.fused_point_filter.sor_nb_neighbors);
  options.AddDefaultOption("fused_sor_std_ratio",
                           &reconstruction_options.fused_point_filter.sor_std_ratio);
  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  StringToLower(&data_type);
  if (data_type == "individual") {
    reconstruction_options.data_type =
        AutomaticReconstructionController::DataType::INDIVIDUAL;
  } else if (data_type == "video") {
    reconstruction_options.data_type =
        AutomaticReconstructionController::DataType::VIDEO;
  } else if (data_type == "internet") {
    reconstruction_options.data_type =
        AutomaticReconstructionController::DataType::INTERNET;
  } else {
    LOG(FATAL) << "Invalid data type provided";
  }

  StringToLower(&quality);
  if (quality == "low") {
    reconstruction_options.quality =
        AutomaticReconstructionController::Quality::LOW;
  } else if (quality == "medium") {
    reconstruction_options.quality =
        AutomaticReconstructionController::Quality::MEDIUM;
  } else if (quality == "high") {
    reconstruction_options.quality =
        AutomaticReconstructionController::Quality::HIGH;
  } else if (quality == "extreme") {
    reconstruction_options.quality =
        AutomaticReconstructionController::Quality::EXTREME;
  } else {
    LOG(FATAL) << "Invalid quality provided";
  }

  StringToLower(&mesher);
  if (mesher == "poisson") {
    reconstruction_options.mesher =
        AutomaticReconstructionController::Mesher::POISSON;
  } else if (mesher == "delaunay") {
    reconstruction_options.mesher =
        AutomaticReconstructionController::Mesher::DELAUNAY;
  } else if (mesher == "advancing_front") {
    reconstruction_options.mesher =
        AutomaticReconstructionController::Mesher::ADVANCING_FRONT;
  } else {
    LOG(FATAL) << "Invalid mesher provided";
  }

  // DA3 sparse model mode
  StringToLower(&sparse_mode);
  if (sparse_mode == "colmap") {
    reconstruction_options.sparse_mode = SparseModelMode::COLMAP_NATIVE;
  } else if (sparse_mode == "da3") {
    reconstruction_options.sparse_mode = SparseModelMode::DA3_DEPTH_POSE;
  } else {
    LOG(FATAL) << "Invalid sparse_mode provided. Use 'colmap' or 'da3'.";
  }

  // DA3 stereo pipeline mode
  StringToLower(&stereo_mode);
  if (stereo_mode == "colmap") {
    reconstruction_options.stereo_mode = StereoPipelineMode::COLMAP_PATCH_MATCH;
  } else if (stereo_mode == "da3") {
    reconstruction_options.stereo_mode = StereoPipelineMode::DA3_DEPTH_INFERENCE;
  } else {
    LOG(FATAL) << "Invalid stereo_mode provided. Use 'colmap' or 'da3'.";
  }

  // DA3 sparse/stereo model types (da3_model/da3_quant are legacy aliases)
  if (da3_sparse_model.empty()) {
    StringToLower(&da3_model);
    da3_sparse_model = da3_model;
  } else {
    StringToLower(&da3_sparse_model);
  }
  if (da3_sparse_quant.empty()) {
    StringToLower(&da3_quant);
    da3_sparse_quant = da3_quant;
  } else {
    StringToLower(&da3_sparse_quant);
  }
  StringToLower(&da3_stereo_model);
  if (!da3_stereo_quant.empty()) {
    StringToLower(&da3_stereo_quant);
  } else if (!da3_sparse_quant.empty()) {
    da3_stereo_quant = da3_sparse_quant;
  } else {
    da3_stereo_quant = "q8_0";
  }

  reconstruction_options.da3_sparse_model_type =
      ParseDA3ModelType(da3_sparse_model);
  reconstruction_options.da3_sparse_quant_type =
      ParseDA3QuantType(da3_sparse_quant);
  reconstruction_options.da3_stereo_model_type =
      ParseDA3ModelType(da3_stereo_model);
  reconstruction_options.da3_stereo_quant_type =
      ParseDA3QuantType(da3_stereo_quant);

  if (reconstruction_options.stereo_mode ==
          StereoPipelineMode::DA3_DEPTH_INFERENCE &&
      !DA3ModelSupportsStereo(reconstruction_options.da3_stereo_model_type)) {
    LOG(FATAL) << "DA3 stereo (--stereo_mode da3) requires a nested model. "
               << "Use --da3_stereo_model nested_anyview or nested_metric.";
  }

  if (reconstruction_options.stereo_mode ==
          StereoPipelineMode::DA3_DEPTH_INFERENCE &&
      reconstruction_options.sparse_mode != SparseModelMode::DA3_DEPTH_POSE) {
    LOG(WARNING) << "DA3 stereo with non-DA3 sparse model: use --sparse_mode da3 "
                    "for consistent poses.";
  }

  ReconstructionManager reconstruction_manager;

  if (reconstruction_options.use_gpu && kUseOpenGL) {
    QApplication app(argc, argv);
    AutomaticReconstructionController controller(reconstruction_options,
                                                 &reconstruction_manager);
    RunThreadWithOpenGLContext(&controller);
  } else {
    AutomaticReconstructionController controller(reconstruction_options,
                                                 &reconstruction_manager);
    controller.Start();
    controller.Wait();
  }

  return EXIT_SUCCESS;
}

int RunBundleAdjuster(int argc, char** argv) {
  std::string input_path;
  std::string output_path;

  OptionManager options;
  options.AddRequiredOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddBundleAdjustmentOptions();
  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  if (!ExistsDir(input_path)) {
    std::cerr << "ERROR: `input_path` is not a directory" << std::endl;
    return EXIT_FAILURE;
  }

  if (!ExistsDir(output_path)) {
    std::cerr << "ERROR: `output_path` is not a directory" << std::endl;
    return EXIT_FAILURE;
  }

  Reconstruction reconstruction;
  reconstruction.Read(input_path);

  BundleAdjustmentController ba_controller(options, &reconstruction);
  ba_controller.Start();
  ba_controller.Wait();

  reconstruction.Write(output_path);

  return EXIT_SUCCESS;
}

int RunColorExtractor(int argc, char** argv) {
  std::string input_path;
  std::string output_path;

  OptionManager options;
  options.AddImageOptions();
  options.AddDefaultOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);
  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  Reconstruction reconstruction;
  reconstruction.Read(input_path);
  reconstruction.ExtractColorsForAllImages(*options.image_path);
  reconstruction.Write(output_path);

  return EXIT_SUCCESS;
}

namespace {

void UpdateDatabasePosePriorsCovariance(
    const std::filesystem::path& database_path,
    const Eigen::Matrix3d& covariance) {
  auto database = Database::Open(database_path);
  DatabaseTransaction database_transaction(database.get());

  LOG(INFO)
      << "Setting up database pose priors with the same covariance matrix: \n"
      << covariance << '\n';

  for (auto& pose_prior : database->ReadAllPosePriors()) {
    pose_prior.position_covariance = covariance;
    database->UpdatePosePrior(pose_prior);
  }
}

}  // namespace

// Upstream parity (d3ccaf35 exe/sfm.cc RunIncrementalMapperImpl), driving
// the fork's IncrementalMapperController with the same contracts.
bool RunIncrementalMapperImpl(
    const std::filesystem::path& database_path,
    const std::filesystem::path& image_path,
    const std::filesystem::path& output_path,
    const std::shared_ptr<IncrementalMapperOptions>& mapper_options,
    std::shared_ptr<ReconstructionManager>& reconstruction_manager,
    std::function<void()> initial_image_pair_callback,
    std::function<void()> next_image_callback,
    std::function<bool()> check_if_stopped) {
  const bool exists_input_reconstruction =
      reconstruction_manager->Size() > 0;

  IncrementalMapperController mapper(mapper_options.get(), image_path,
                                     database_path,
                                     reconstruction_manager.get());
  mapper.SetCheckIfStoppedFunc(std::move(check_if_stopped));

  // In case a new reconstruction is started, write results of individual
  // sub-models as they finish instead of writing all results afterwards.
  size_t prev_num_reconstructions = 0;
  const auto write_new_reconstructions = [&]() {
    while (prev_num_reconstructions < reconstruction_manager->Size()) {
      const auto reconstruction_path =
          output_path / std::to_string(prev_num_reconstructions);
      CreateDirIfNotExists(reconstruction_path);
      reconstruction_manager->Get(prev_num_reconstructions)
          .Write(reconstruction_path);
      ++prev_num_reconstructions;
    }
  };
  if (!exists_input_reconstruction) {
    mapper.AddCallback(IncrementalMapperController::LAST_IMAGE_REG_CALLBACK,
                       write_new_reconstructions);
  }

  if (initial_image_pair_callback) {
    mapper.AddCallback(
        IncrementalMapperController::INITIAL_IMAGE_PAIR_REG_CALLBACK,
        std::move(initial_image_pair_callback));
  }

  if (next_image_callback) {
    mapper.AddCallback(IncrementalMapperController::NEXT_IMAGE_REG_CALLBACK,
                       std::move(next_image_callback));
  }

  mapper.Start();
  mapper.Wait();

  if (!exists_input_reconstruction) {
    // The final callback is intentionally not invoked for an interrupted
    // sub-model, so flush any reconstruction retained by the interruption
    // path before returning.
    write_new_reconstructions();
  }

  if (reconstruction_manager->Size() == 0) {
    LOG(ERROR) << "Failed to create any sparse model";
    return false;
  }

  // In case the reconstruction is continued from an existing reconstruction,
  // do not create sub-folders but directly write the results.
  if (exists_input_reconstruction) {
    reconstruction_manager->Get(0).Write(output_path);
  }

  return true;
}

int RunMapper(int argc, char** argv) {
  std::string input_path;
  std::string output_path;
  std::string image_list_path;

  OptionManager options;
  options.AddDatabaseOptions();
  options.AddImageOptions();
  options.AddDefaultOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddDefaultOption("image_list_path", &image_list_path);
  options.AddMapperOptions();
  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  if (!ExistsDir(output_path)) {
    std::cerr << "ERROR: `output_path` is not a directory." << std::endl;
    return EXIT_FAILURE;
  }

  if (!image_list_path.empty()) {
    const auto image_names = ReadTextFileLines(image_list_path);
    options.mapper->image_names =
        std::unordered_set<std::string>(image_names.begin(), image_names.end());
  }

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  if (input_path != "") {
    if (!ExistsDir(input_path)) {
      std::cerr << "ERROR: `input_path` is not a directory." << std::endl;
      return EXIT_FAILURE;
    }
    reconstruction_manager->Read(input_path);
  }

  if (!RunIncrementalMapperImpl(*options.database_path, *options.image_path,
                                output_path, options.mapper,
                                reconstruction_manager)) {
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

int RunPosePriorMapper(int argc, char** argv) {
  std::filesystem::path input_path;
  std::filesystem::path output_path;

  bool overwrite_priors_covariance = false;
  double prior_position_std_x = 1.;
  double prior_position_std_y = 1.;
  double prior_position_std_z = 1.;

  OptionManager options;
  options.AddDatabaseOptions();
  options.AddImageOptions();
  options.AddDefaultOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddMapperOptions();

  // Upstream parity: the prior switch lives on the pipeline options
  // (it also enables the ENU conversion of the pose priors).
  options.mapper->use_prior_position = true;

  options.AddDefaultOption(
      "overwrite_priors_covariance",
      &overwrite_priors_covariance,
      "Priors covariance read from database. If true, overwrite the priors "
      "covariance using the following prior_position_std_... options");
  options.AddDefaultOption("prior_position_std_x", &prior_position_std_x);
  options.AddDefaultOption("prior_position_std_y", &prior_position_std_y);
  options.AddDefaultOption("prior_position_std_z", &prior_position_std_z);
  options.AddDefaultOption("use_robust_loss_on_prior_position",
                           &options.mapper->mapper.use_robust_loss_on_prior_position);
  options.AddDefaultOption("prior_position_loss_scale",
                           &options.mapper->mapper.prior_position_loss_scale);
  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  if (!ExistsDir(output_path)) {
    std::cerr << "ERROR: `output_path` is not a directory." << std::endl;
    return EXIT_FAILURE;
  }

  if (overwrite_priors_covariance) {
    const Eigen::Matrix3d covariance =
        Eigen::Vector3d(
            prior_position_std_x, prior_position_std_y, prior_position_std_z)
            .cwiseAbs2()
            .asDiagonal();
    UpdateDatabasePosePriorsCovariance(*options.database_path, covariance);
  }

  ReconstructionManager reconstruction_manager;
  if (!input_path.empty()) {
    if (!ExistsDir(input_path)) {
      std::cerr << "ERROR: `input_path` is not a directory." << std::endl;
      return EXIT_FAILURE;
    }
    reconstruction_manager.Read(input_path);
  }

  IncrementalMapperController mapper(options.mapper.get(), *options.image_path,
                                     *options.database_path,
                                     &reconstruction_manager);
  mapper.Start();
  mapper.Wait();

  if (reconstruction_manager.Size() == 0) {
    std::cerr << "ERROR: failed to create sparse model" << std::endl;
    return EXIT_FAILURE;
  }

  // Upstream parity (d3ccaf35 RunPosePriorMapper): write every reconstruction
  // to output_path/<i> regardless of whether an input reconstruction was
  // continued; the old fork condition skipped the write entirely when no
  // input_path was given, silently producing an empty output.
  for (size_t i = 0; i < reconstruction_manager.Size(); ++i) {
    const auto reconstruction_path = output_path / std::to_string(i);
    CreateDirIfNotExists(reconstruction_path);
    reconstruction_manager.Get(i).Write(reconstruction_path);
    options.Write(reconstruction_path / "project.ini");
  }

  return EXIT_SUCCESS;
}

// Upstream parity (d3ccaf35 exe/sfm.cc RunHierarchicalMapperImpl), driving
// the fork's HierarchicalMapperController.
bool RunHierarchicalMapperImpl(
    const HierarchicalMapperController::Options& controller_options,
    const SceneClustering::Options& clustering_options,
    const IncrementalMapperOptions& mapper_options,
    const std::filesystem::path& output_path,
    ReconstructionManager* reconstruction_manager) {
  HierarchicalMapperController hierarchical_mapper(
      controller_options, clustering_options, mapper_options,
      reconstruction_manager);
  hierarchical_mapper.Start();
  hierarchical_mapper.Wait();
  return reconstruction_manager->Size() > 0;
}

int RunHierarchicalMapper(int argc, char** argv) {
  HierarchicalMapperController::Options hierarchical_options;
  SceneClustering::Options clustering_options;
  std::string output_path;

  OptionManager options;
  options.AddRequiredOption("database_path",
                            &hierarchical_options.database_path);
  options.AddRequiredOption("image_path", &hierarchical_options.image_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddDefaultOption("num_workers", &hierarchical_options.num_workers);
  options.AddDefaultOption("image_overlap", &clustering_options.image_overlap);
  options.AddDefaultOption("leaf_max_num_images",
                           &clustering_options.leaf_max_num_images);
  options.AddMapperOptions();
  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  if (!ExistsDir(output_path)) {
    std::cerr << "ERROR: `output_path` is not a directory." << std::endl;
    return EXIT_FAILURE;
  }

  ReconstructionManager reconstruction_manager;

  if (!RunHierarchicalMapperImpl(hierarchical_options, clustering_options,
                                 *options.mapper, output_path,
                                 &reconstruction_manager)) {
    std::cerr << "ERROR: failed to create sparse model" << std::endl;
    return EXIT_FAILURE;
  }

  reconstruction_manager.Write(output_path, &options);

  return EXIT_SUCCESS;
}

int RunPointFiltering(int argc, char** argv) {
  std::string input_path;
  std::string output_path;

  int min_track_len = 2;
  double max_reproj_error = 4.0;
  double min_tri_angle = 1.5;

  OptionManager options;
  options.AddRequiredOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddDefaultOption("min_track_len", &min_track_len);
  options.AddDefaultOption("max_reproj_error", &max_reproj_error);
  options.AddDefaultOption("min_tri_angle", &min_tri_angle);
  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  Reconstruction reconstruction;
  reconstruction.Read(input_path);

  // Upstream parity (d3ccaf35 exe/sfm.cc point_filtering): filtering runs
  // through the observation manager so the visibility ledger stays in sync;
  // short-track pruning uses the manager's dedicated filter instead of a
  // manual DeletePoint3D loop that bypasses the bookkeeping.
  ObservationManager obs_manager(reconstruction);
  size_t num_filtered =
      obs_manager.FilterAllPoints3D(max_reproj_error, min_tri_angle);
  num_filtered += obs_manager.FilterPoints3DWithShortTracks(min_track_len);

  std::cout << "Filtered observations: " << num_filtered << std::endl;

  reconstruction.Write(output_path);

  return EXIT_SUCCESS;
}

// Upstream parity (d3ccaf35 exe/sfm.cc RunPointTriangulatorImpl): the
// fork drives IncrementalMapper directly over the loaded model; the stop
// probe switches the write target to "<output>.partial" instead of
// overwriting the caller's reconstruction output.
void RunPointTriangulatorImpl(
    const std::shared_ptr<Reconstruction>& reconstruction,
    const std::filesystem::path& database_path,
    const std::filesystem::path& image_path,
    const std::filesystem::path& output_path,
    const IncrementalMapperOptions& options,
    const bool clear_points,
    const bool refine_intrinsics,
    std::function<bool()> check_if_stopped) {
  THROW_CHECK_GE(reconstruction->NumRegImages(), 2)
      << "Need at least two images for triangulation";

  if (clear_points) {
    reconstruction->DeleteAllPoints2DAndPoints3D();
    reconstruction->TranscribeImageIdsToDatabase(
        *Database::Open(database_path));
  }

  // Upstream parity: triangulation operates on every registered image, not
  // only the configured image-list subset.
  IncrementalMapperOptions mapper_options = options;
  mapper_options.image_names.clear();

  PrintHeading1("Loading database");

  DatabaseCache database_cache;
  {
    auto database = Database::Open(database_path);

    DatabaseCache::Options cache_options;
    cache_options.min_num_matches =
        static_cast<size_t>(mapper_options.min_num_matches);
    cache_options.ignore_watermarks = mapper_options.ignore_watermarks;
    // Upstream parity (load_all_images=true): triangulation operates on the
    // registered reconstruction regardless of pair connectivity.
    cache_options.load_all_images = true;
    database_cache.Load(*database, cache_options);
  }

  IncrementalMapper mapper(&database_cache);
  mapper.BeginReconstruction(reconstruction.get());

  bool stopped = false;

  // Triangulation
  const auto tri_options = mapper_options.Triangulation();
  const auto& reg_image_ids = reconstruction->RegImageIds();

  for (size_t i = 0; i < reg_image_ids.size(); ++i) {
    if (check_if_stopped && check_if_stopped()) {
      stopped = true;
      break;
    }
    const image_t image_id = reg_image_ids[i];
    const auto& image = reconstruction->Image(image_id);
    mapper.TriangulateImage(tri_options, image_id);
  }

  // Retriangulation and bundle adjustment
  if (!stopped) {
    CompleteAndMergeTracks(mapper_options, &mapper);

    auto ba_options = mapper_options.GlobalBundleAdjustment();
    ba_options.refine_focal_length = refine_intrinsics;
    ba_options.refine_principal_point = false;
    ba_options.refine_extra_params = refine_intrinsics;
    ba_options.refine_extrinsics = false;

    BundleAdjustmentConfig ba_config;
    for (const image_t image_id : reconstruction->RegImageIds()) {
      ba_config.AddImage(image_id);
    }

    for (int i = 0; i < mapper_options.ba_global_max_refinements; ++i) {
      if (check_if_stopped && check_if_stopped()) {
        stopped = true;
        break;
      }
      reconstruction->FilterObservationsWithNegativeDepth();

      const size_t num_observations = reconstruction->ComputeNumObservations();

      auto bundle_adjuster =
          CreateDefaultBundleAdjuster(ba_options, ba_config);
      CHECK(bundle_adjuster->Solve(reconstruction.get()));

      size_t num_changed_observations = 0;
      num_changed_observations +=
          CompleteAndMergeTracks(mapper_options, &mapper);
      num_changed_observations += FilterPoints(mapper_options, &mapper);
      const double changed =
          static_cast<double>(num_changed_observations) / num_observations;
      if (changed < mapper_options.ba_global_max_refinement_change) {
        break;
      }
    }
  }

  if (mapper_options.extract_colors && !stopped) {
    reconstruction->ExtractColorsForAllImages(image_path);
  }

  const bool kDiscardReconstruction = false;
  mapper.EndReconstruction(kDiscardReconstruction);

  std::filesystem::path write_path = output_path;
  if (stopped) {
    std::filesystem::path normalized_output_path =
        output_path.lexically_normal();
    if (normalized_output_path.filename().empty()) {
      normalized_output_path = normalized_output_path.parent_path();
    }
    write_path = normalized_output_path.parent_path() /
                 (normalized_output_path.filename().string() + ".partial");
    CreateDirIfNotExists(write_path);
    LOG(WARNING) << "Writing partial triangulation output to " << write_path;
  }
  reconstruction->Write(write_path);
}

int RunPointTriangulator(int argc, char** argv) {
  std::string input_path;
  std::string output_path;
  // Upstream parity (d3ccaf35 exe/sfm.cc): point triangulation clears the
  // existing points by default and recomputes them from the database.
  bool clear_points = true;
  bool refine_intrinsics = false;

  OptionManager options;
  options.AddDatabaseOptions();
  options.AddImageOptions();
  options.AddRequiredOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddDefaultOption(
      "clear_points", &clear_points,
      "Whether to clear all existing points and observations");
  options.AddDefaultOption("refine_intrinsics", &refine_intrinsics,
                           "Whether to refine the intrinsics of the cameras");
  options.AddMapperOptions();
  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  if (!ExistsDir(input_path)) {
    std::cerr << "ERROR: `input_path` is not a directory" << std::endl;
    return EXIT_FAILURE;
  }

  if (!ExistsDir(output_path)) {
    std::cerr << "ERROR: `output_path` is not a directory" << std::endl;
    return EXIT_FAILURE;
  }

  auto reconstruction = std::make_shared<Reconstruction>();
  reconstruction->Read(input_path);

  RunPointTriangulatorImpl(reconstruction,
                           *options.database_path,
                           *options.image_path,
                           output_path,
                           *options.mapper,
                           clear_points,
                           refine_intrinsics);
  return EXIT_SUCCESS;
}


namespace {

// Read the configuration of the camera rigs from a JSON file. The input images
// of a camera rig must be named consistently to assign them to the appropriate
// camera rig and the respective snapshots.
//
// An example configuration of a single camera rig:
// [
//   {
//     "ref_camera_id": 1,
//     "cameras":
//     [
//       {
//           "camera_id": 1,
//           "image_prefix": "left1_image"
//           "rel_tvec": [0, 0, 0],
//           "rel_qvec": [1, 0, 0, 0]
//       },
//       {
//           "camera_id": 2,
//           "image_prefix": "left2_image"
//           "rel_tvec": [0, 0, 0],
//           "rel_qvec": [0, 1, 0, 0]
//       },
//       {
//           "camera_id": 3,
//           "image_prefix": "right1_image"
//           "rel_tvec": [0, 0, 0],
//           "rel_qvec": [0, 0, 1, 0]
//       },
//       {
//           "camera_id": 4,
//           "image_prefix": "right2_image"
//           "rel_tvec": [0, 0, 0],
//           "rel_qvec": [0, 0, 0, 1]
//       }
//     ]
//   }
// ]
//
// The "camera_id" and "image_prefix" fields are required, whereas the
// "rel_tvec" and "rel_qvec" fields optionally specify the relative
// extrinsics of the camera rig in the form of a translation vector and a
// rotation quaternion. The relative extrinsics rel_qvec and rel_tvec transform
// coordinates from rig to camera coordinate space. If the relative extrinsics
// are not provided then they are automatically inferred from the
// reconstruction.
//
// This file specifies the configuration for a single camera rig and that you
// could potentially define multiple camera rigs. The rig is composed of 4
// cameras: all images of the first camera must have "left1_image" as a name
// prefix, e.g., "left1_image_frame000.png" or "left1_image/frame000.png".
// Images with the same suffix ("_frame000.png" and "/frame000.png") are
// assigned to the same snapshot, i.e., they are assumed to be captured at the
// same time. Only snapshots with the reference image registered will be added
// to the bundle adjustment problem. The remaining images will be added with
// independent poses to the bundle adjustment problem. The above configuration
// could have the following input image file structure:
//
//    /path/to/images/...
//        left1_image/...
//            frame000.png
//            frame001.png
//            frame002.png
//            ...
//        left2_image/...
//            frame000.png
//            frame001.png
//            frame002.png
//            ...
//        right1_image/...
//            frame000.png
//            frame001.png
//            frame002.png
//            ...
//        right2_image/...
//            frame000.png
//            frame001.png
//            frame002.png
//            ...
//
}  // namespace

// Upstream parity (d3ccaf35 exe/sfm.cc RunGlobalMapperImpl). The fork's
// Database is constructed directly (no Database::Open factory).
bool RunGlobalMapperImpl(
    const std::filesystem::path& database_path,
    const std::filesystem::path& image_path,
    const std::filesystem::path& output_path,
    const std::shared_ptr<GlobalPipelineOptions>& mapper_options,
    std::shared_ptr<ReconstructionManager>& reconstruction_manager) {
  GlobalPipelineOptions pipeline_options = *mapper_options;
  pipeline_options.image_path = image_path;

  GlobalPipeline global_mapper(std::move(pipeline_options),
                               Database::Open(database_path),
                               reconstruction_manager);
  global_mapper.Run();

  if (reconstruction_manager->Size() == 0) {
    LOG(ERROR) << "Failed to create sparse model";
    return false;
  }

  reconstruction_manager->Write(output_path);
  return true;
}

int RunGlobalMapper(int argc, char** argv) {
  std::filesystem::path output_path;

  OptionManager options;
  options.AddDatabaseOptions();
  options.AddImageOptions();
  options.AddRequiredOption("output_path", &output_path);
  options.AddGlobalMapperOptions();
  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  if (!ExistsDir(output_path)) {
    LOG(ERROR) << "`output_path` is not a directory.";
    return EXIT_FAILURE;
  }

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  if (!RunGlobalMapperImpl(*options.database_path,
                           *options.image_path,
                           output_path,
                           options.global_mapper,
                           reconstruction_manager)) {
    return EXIT_FAILURE;
  }

  // Upstream parity (d3ccaf35 exe/sfm.cc): the option file is written to a
  // file inside the output directory, not to the directory itself.
  options.Write(output_path / "project.ini");
  return EXIT_SUCCESS;
}


int RunRotationAverager(int argc, char** argv) {
  std::filesystem::path output_path;
  std::filesystem::path image_list_path;

  RotationAveragingPipelineOptions controller_options;

  OptionManager options;
  options.AddDatabaseOptions();
  options.AddRequiredOption("output_path", &output_path);
  options.AddDefaultOption("image_list_path", &image_list_path);
  options.AddDefaultOption("min_num_matches",
                           &controller_options.min_num_matches);
  options.AddDefaultOption("ignore_watermarks",
                           &controller_options.ignore_watermarks);
  options.AddDefaultOption("num_threads", &controller_options.num_threads);
  options.AddDefaultOption("random_seed", &controller_options.random_seed);
  options.AddDefaultOption("use_gravity",
                           &controller_options.rotation_estimation.use_gravity);
  options.AddDefaultOption(
      "use_stratified", &controller_options.rotation_estimation.use_stratified);
  options.AddDefaultOption("refine_gravity",
                           &controller_options.refine_gravity);
  options.AddGravityRefinerOptions();
  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  controller_options.gravity_refiner = *options.gravity_refiner;

  if (!image_list_path.empty()) {
    controller_options.image_names = ReadTextFileLines(image_list_path);
  }

  if (!ExistsDir(output_path)) {
    LOG(ERROR) << "`output_path` is not a directory";
    return EXIT_FAILURE;
  }

  // Fork parity: the fork constructs the Database directly (no Open factory).
  auto database = Database::Open(*options.database_path);
  auto reconstruction = std::make_shared<Reconstruction>();

  RotationAveragingPipeline controller(
      controller_options, std::move(database), reconstruction);
  controller.Run();

  if (reconstruction->NumRegFrames() == 0) {
    LOG(ERROR) << "No frames registered";
    return EXIT_FAILURE;
  }

  LOG(INFO) << "Writing reconstruction to " << output_path;
  reconstruction->Write(output_path);

  return EXIT_SUCCESS;
}

int RunViewGraphCalibrator(int argc, char** argv) {
  ViewGraphCalibrationOptions calibration_options;

  OptionManager options;
  options.AddDatabaseOptions();
  options.AddDefaultOption(
      "cross_validate_prior_focal_lengths",
      &calibration_options.cross_validate_prior_focal_lengths,
      "Cross-validate prior focal lengths");
  options.AddDefaultOption(
      "min_calibrated_pair_ratio",
      &calibration_options.min_calibrated_pair_ratio,
      "Minimum ratio of calibrated pairs for cross-validation");
  options.AddDefaultOption("reestimate_relative_pose",
                           &calibration_options.reestimate_relative_pose,
                           "Re-estimate relative poses after calibration");
  options.AddDefaultOption("min_focal_length_ratio",
                           &calibration_options.min_focal_length_ratio,
                           "Minimum ratio of estimated to prior focal length");
  options.AddDefaultOption("max_focal_length_ratio",
                           &calibration_options.max_focal_length_ratio,
                           "Maximum ratio of estimated to prior focal length");
  options.AddDefaultOption("max_calibration_error",
                           &calibration_options.max_calibration_error,
                           "Maximum calibration error for an image pair");
  options.AddDefaultOption("relpose_max_error",
                           &calibration_options.relpose_max_error,
                           "Maximum error for relative pose re-estimation");
  options.AddDefaultOption("relpose_min_num_inliers",
                           &calibration_options.relpose_min_num_inliers,
                           "Minimum inliers for relative pose re-estimation");
  options.AddDefaultOption(
      "relpose_min_inlier_ratio",
      &calibration_options.relpose_min_inlier_ratio,
      "Minimum inlier ratio for relative pose re-estimation");
  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  // Fork parity: the fork constructs the Database directly (no Open factory).
  auto database = Database::Open(*options.database_path);

  if (!CalibrateViewGraph(calibration_options, database.get())) {
    LOG(ERROR) << "View graph calibration failed";
    return EXIT_FAILURE;
  }

  LOG(INFO) << "View graph calibration completed successfully";
  return EXIT_SUCCESS;
}

}  // namespace colmap
