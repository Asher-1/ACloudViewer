// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "controllers/incremental_pipeline.h"

#include "sfm/observation_manager.h"

#include "util/hash_containers.h"

#include "estimators/alignment.h"
#include "estimators/bundle_adjustment_caspar.h"
#include "estimators/bundle_adjustment_ceres.h"
#include "scene/projection.h"

#include "util/misc.h"

namespace colmap {
namespace {

// Upstream parity (a2bd15c2 ReconstructSubModel/InitializeReconstruction):
// registration happens per frame, so every image of the registered frame is
// triangulated, not only the image that triggered the registration.
size_t TriangulateFrame(const IncrementalTriangulator::Options& tri_options,
                        const Image& image,
                        IncrementalMapper* mapper) {
    size_t num_tris = 0;
    for (const image_t frame_image_id : image.FramePtr()->ImageIds()) {
        num_tris += mapper->TriangulateImage(tri_options, frame_image_id);
    }
    std::cout << "  => Added observations: " << num_tris << std::endl;
    return num_tris;
}

// Upstream parity (d3ccaf35 incremental_pipeline.cc): anchor the final
// reconstruction to the metric prior positions when enabled, falling back to
// the original rig extrinsic scales; without this the BA gauge lets the
// model drift away from the GT scale.
void AlignReconstructionToPriorsOrRigScale(
        const IncrementalPipelineOptions& options,
        const DatabaseCache& database_cache,
        Reconstruction* reconstruction) {
    if (options.use_prior_position) {
        PosePriorBundleAdjustmentOptions prior_options;
        prior_options.alignment_ransac_options.random_seed =
                options.random_seed;

        Sim3d metric_from_reconstruction;
        if (AlignReconstructionToPosePriors(
                *reconstruction,
                database_cache.PosePriors(),
                prior_options.alignment_ransac_options,
                prior_options.prior_position_fallback_stddev,
                &metric_from_reconstruction)) {
            reconstruction->Transform(metric_from_reconstruction);
            return;
        }
        LOG(WARNING) << "Final alignment w.r.t. prior positions failed; "
                        "restoring the original rig scale instead";
    }

    AlignReconstructionToOrigRigScales(database_cache.Rigs(), reconstruction);
}

void IterativeGlobalRefinement(const IncrementalPipelineOptions& options,
                               const IncrementalMapper::Options& mapper_options,
                               IncrementalMapper* mapper,
                               std::function<bool()> check_if_stopped) {
    PrintHeading1("Retriangulation and Global bundle adjustment");
    // Upstream parity (a2bd15c2 incremental_pipeline.cc): the refinement
    // rounds run inside the mapper, which normalizes the scene after every
    // global bundle adjustment (numerical stability) and applies the same
    // complete/merge/retriangulate + changed-observation convergence logic.
    // The stop probe reaches the BA solver, and an interrupted refinement
    // skips the frame filter.
    BundleAdjustmentOptions ba_options = options.GlobalBundleAdjustment();
    ba_options.check_if_stopped = std::move(check_if_stopped);
    mapper->IterativeGlobalRefinement(options.ba_global_max_refinements,
                                      options.ba_global_max_refinement_change,
                                      mapper_options, ba_options,
                                      options.Triangulation());
    if (!ba_options.check_if_stopped || !ba_options.check_if_stopped()) {
        const size_t num_filtered_frames = mapper->FilterFrames(mapper_options);
        std::cout << "  => Filtered images: " << num_filtered_frames
                  << std::endl;
    }
}

void ExtractColors(const std::string& image_path,
                   const image_t image_id,
                   Reconstruction* reconstruction) {
    if (!reconstruction->ExtractColorsForImage(image_id, image_path)) {
        std::cout << StringPrintf(
                             "WARNING: Could not read image %s at path %s.",
                             reconstruction->Image(image_id).Name().c_str(),
                             image_path.c_str())
                  << std::endl;
    }
}

// Upstream parity (a2bd15c2): colors are extracted for every image of the
// registered frame.
void ExtractFrameColors(const std::string& image_path,
                        const Image& image,
                        Reconstruction* reconstruction) {
    for (const image_t frame_image_id : image.FramePtr()->ImageIds()) {
        ExtractColors(image_path, frame_image_id, reconstruction);
    }
}

// Upstream parity (d3ccaf35 controllers/incremental_pipeline.cc): detect a
// rig whose camera sensor_from_rig pose is unset; such a scene cannot be
// reconstructed with rig constraints and must be discarded gracefully.
bool HasUnknownSensorFromRig(const Reconstruction& reconstruction) {
    FlatHashSet<const Rig*> parameterized_rigs;
    for (const auto& [_, image] : reconstruction.Images()) {
        parameterized_rigs.insert(image.FramePtr()->RigPtr());
    }
    for (const Rig* rig : parameterized_rigs) {
        for (const auto& [sensor_id, sensor_from_rig] :
             rig->NonRefSensors()) {
            if (sensor_id.type == SensorType::CAMERA &&
                !sensor_from_rig.has_value()) {
                return true;
            }
        }
    }
    return false;
}

void WriteSnapshot(const Reconstruction& reconstruction,
                   const std::filesystem::path& snapshot_path) {
    PrintHeading1("Creating snapshot");
    // Get the current timestamp in milliseconds.
    const size_t timestamp =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now()
                            .time_since_epoch())
                    .count();
    // Write reconstruction to unique path with current timestamp.
    const std::string path =
            JoinPaths(snapshot_path, StringPrintf("%010d", timestamp));
    CreateDirIfNotExists(path);
    std::cout << "  => Writing to " << path << std::endl;
    reconstruction.Write(path);
}

}  // namespace

size_t FilterPoints(const IncrementalPipelineOptions& options,
                    IncrementalMapper* mapper) {
    const size_t num_filtered_observations =
            mapper->FilterPoints(options.Mapper());
    std::cout << "  => Filtered observations: " << num_filtered_observations
              << std::endl;
    return num_filtered_observations;
}

size_t FilterFrames(const IncrementalPipelineOptions& options,
                    IncrementalMapper* mapper) {
    const size_t num_filtered_images = mapper->FilterFrames(options.Mapper());
    std::cout << "  => Filtered images: " << num_filtered_images << std::endl;
    return num_filtered_images;
}

size_t CompleteAndMergeTracks(const IncrementalPipelineOptions& options,
                              IncrementalMapper* mapper) {
    const size_t num_completed_observations =
            mapper->CompleteTracks(options.Triangulation());
    std::cout << "  => Completed observations: " << num_completed_observations
              << std::endl;
    const size_t num_merged_observations =
            mapper->MergeTracks(options.Triangulation());
    std::cout << "  => Merged observations: " << num_merged_observations
              << std::endl;
    return num_completed_observations + num_merged_observations;
}

IncrementalMapper::Options IncrementalPipelineOptions::Mapper() const {
    IncrementalMapper::Options options = mapper;
    options.abs_pose_refine_focal_length = ba_refine_focal_length;
    options.abs_pose_refine_extra_params = ba_refine_extra_params;
    options.min_focal_length_ratio = min_focal_length_ratio;
    options.max_focal_length_ratio = max_focal_length_ratio;
    options.max_extra_param = max_extra_param;
    options.num_threads = num_threads;
    options.fix_existing_frames = fix_existing_frames;
    options.constant_rigs = constant_rigs;
    options.constant_cameras = constant_cameras;
    options.random_seed = random_seed;
    options.use_prior_position = use_prior_position;
    options.use_robust_loss_on_prior_position =
            use_robust_loss_on_prior_position;
    options.prior_position_loss_scale = prior_position_loss_scale;
    return options;
}

IncrementalTriangulator::Options IncrementalPipelineOptions::Triangulation()
        const {
    IncrementalTriangulator::Options options = triangulation;
    options.min_focal_length_ratio = min_focal_length_ratio;
    options.max_focal_length_ratio = max_focal_length_ratio;
    options.max_extra_param = max_extra_param;
    options.random_seed = random_seed;
    return options;
}

namespace {

// Upstream parity (d3ccaf35): the Ceres iteration defaults behind the
// -1 sentinel.
constexpr int kDefaultCeresLocalMaxNumIterations = 25;
constexpr int kDefaultCeresGlobalMaxNumIterations = 50;

}  // namespace

BundleAdjustmentOptions IncrementalPipelineOptions::LocalBundleAdjustment()
        const {
    BundleAdjustmentOptions options;
    options.print_summary = false;
    options.backend = ba_local_backend;
    options.refine_focal_length = ba_refine_focal_length;
    options.refine_principal_point = ba_refine_principal_point;
    options.refine_extra_params = ba_refine_extra_params;
    options.refine_sensor_from_rig = ba_refine_sensor_from_rig;
    if (options.ceres) {
        options.ceres->solver_options.function_tolerance =
                ba_local_function_tolerance;
        options.ceres->solver_options.gradient_tolerance = 10.0;
        options.ceres->solver_options.parameter_tolerance = 0.0;
        options.ceres->solver_options.max_num_iterations =
                (ba_local_max_num_iterations >= 0)
                        ? ba_local_max_num_iterations
                        : kDefaultCeresLocalMaxNumIterations;
        options.ceres->solver_options.max_linear_solver_iterations = 100;
        options.ceres->solver_options.logging_type =
                ceres::LoggingType::SILENT;
        options.ceres->solver_options.num_threads = num_threads;
#if CERES_VERSION_MAJOR < 2
        options.ceres->solver_options.num_linear_solver_threads = num_threads;
#endif  // CERES_VERSION_MAJOR
        options.ceres->min_num_residuals_for_cpu_multi_threading =
                ba_min_num_residuals_for_cpu_multi_threading;
        options.ceres->loss_function_scale = 1.0;
        options.ceres->loss_function_type = CeresLossFunctionType::SOFT_L1;
        options.ceres->use_gpu = ba_use_gpu;
        options.ceres->gpu_index = ba_gpu_index;
    }
    if (options.caspar) {
        // Only forward the iteration bound when the user explicitly set it
        // (i.e. it is not the -1 sentinel), leaving Caspar's own tuned
        // solver_iter_max default untouched otherwise, as Caspar has
        // different dampening and convergence criteria than Ceres.
        if (ba_local_max_num_iterations >= 0) {
            options.caspar->solver_iter_max = ba_local_max_num_iterations;
        }
        options.caspar->gpu_index = ba_gpu_index;
    }
    return options;
}


int IncrementalPipelineOptions::EffBaLocalMaxNumIterations() const {
    if (ba_local_max_num_iterations >= 0) {
        return ba_local_max_num_iterations;
    }
    if (ba_local_backend == BundleAdjustmentBackend::CASPAR) {
        return CasparBundleAdjustmentOptions().solver_iter_max;
    }
    return kDefaultCeresLocalMaxNumIterations;
}

int IncrementalPipelineOptions::EffBaGlobalMaxNumIterations() const {
    if (ba_global_max_num_iterations >= 0) {
        return ba_global_max_num_iterations;
    }
    if (ba_global_backend == BundleAdjustmentBackend::CASPAR) {
        return CasparBundleAdjustmentOptions().solver_iter_max;
    }
    return kDefaultCeresGlobalMaxNumIterations;
}

BundleAdjustmentOptions IncrementalPipelineOptions::GlobalBundleAdjustment()
        const {
    BundleAdjustmentOptions options;
    options.print_summary = false;
    options.backend = ba_global_backend;
    options.refine_focal_length = ba_refine_focal_length;
    options.refine_principal_point = ba_refine_principal_point;
    options.refine_extra_params = ba_refine_extra_params;
    options.refine_sensor_from_rig = ba_refine_sensor_from_rig;
    if (options.ceres) {
        options.ceres->solver_options.function_tolerance =
                ba_global_function_tolerance;
        options.ceres->solver_options.gradient_tolerance = 1.0;
        options.ceres->solver_options.parameter_tolerance = 0.0;
        options.ceres->solver_options.max_num_iterations =
                (ba_global_max_num_iterations >= 0)
                        ? ba_global_max_num_iterations
                        : kDefaultCeresGlobalMaxNumIterations;
        options.ceres->solver_options.max_linear_solver_iterations = 100;
        options.ceres->solver_options.logging_type =
                ceres::LoggingType::SILENT;
        if (VLOG_IS_ON(2)) {
            options.ceres->solver_options.minimizer_progress_to_stdout = true;
            options.ceres->solver_options.logging_type =
                    ceres::LoggingType::PER_MINIMIZER_ITERATION;
        }
        options.ceres->solver_options.num_threads = num_threads;
#if CERES_VERSION_MAJOR < 2
        options.ceres->solver_options.num_linear_solver_threads = num_threads;
#endif  // CERES_VERSION_MAJOR
        options.ceres->min_num_residuals_for_cpu_multi_threading =
                ba_min_num_residuals_for_cpu_multi_threading;
        options.ceres->loss_function_type = CeresLossFunctionType::TRIVIAL;
        options.ceres->use_gpu = ba_use_gpu;
        options.ceres->gpu_index = ba_gpu_index;
    }
    if (options.caspar) {
        // See LocalBundleAdjustment(): only forward the iteration bound to
        // the Caspar backend when the user explicitly set it, so Caspar's
        // own tuned solver_iter_max default is preserved otherwise.
        if (ba_global_max_num_iterations >= 0) {
            options.caspar->solver_iter_max = ba_global_max_num_iterations;
        }
        options.caspar->gpu_index = ba_gpu_index;
    }
    return options;
}

bool IncrementalPipelineOptions::Check() const {
    CHECK_OPTION_GT(min_num_matches, 0);
    CHECK_OPTION_GT(max_num_models, 0);
    CHECK_OPTION_GT(max_model_overlap, 0);
    CHECK_OPTION_GE(min_model_size, 0);
    CHECK_OPTION_GT(init_num_trials, 0);
    CHECK_OPTION_GT(min_focal_length_ratio, 0);
    CHECK_OPTION_GT(max_focal_length_ratio, 0);
    CHECK_OPTION_GE(max_extra_param, 0);
    // Upstream parity: -1 is the "backend default" sentinel and legal.
    CHECK_OPTION_GE(ba_local_max_num_iterations, -1);
    CHECK_OPTION_GT(ba_global_frames_ratio, 1.0);
    CHECK_OPTION_GT(ba_global_points_ratio, 1.0);
    CHECK_OPTION_GT(ba_global_frames_freq, 0);
    CHECK_OPTION_GT(ba_global_points_freq, 0);
    // Upstream parity: -1 is the "backend default" sentinel and legal, while
    // zero global iterations is rejected (a2bd15c2).
    CHECK_OPTION_GE(ba_global_max_num_iterations, -1);
    CHECK_OPTION_NE(ba_global_max_num_iterations, 0);
    CHECK_OPTION_GT(ba_local_max_refinements, 0);
    CHECK_OPTION_GE(ba_local_max_refinement_change, 0);
    CHECK_OPTION_GE(ba_global_max_refinements, 0);
    CHECK_OPTION_GE(ba_global_max_refinement_change, 0);
    CHECK_OPTION_GE(snapshot_frames_freq, 0);
    CHECK_OPTION_GT(prior_position_loss_scale, 0.);
    CHECK_OPTION_GE(num_threads, -1);
    CHECK_OPTION_GE(random_seed, -1);
    // structure_less_registration_only selects the structure-less path by
    // itself (Reconstruct uses {true}); upstream does not tie it to the
    // fallback toggle.
#ifndef CASPAR_ENABLED
    CHECK_OPTION(ba_local_backend != BundleAdjustmentBackend::CASPAR);
    CHECK_OPTION(ba_global_backend != BundleAdjustmentBackend::CASPAR);
#endif
    CHECK_OPTION(Mapper().Check());
    CHECK_OPTION(Triangulation().Check());
    return true;
}

IncrementalPipeline::IncrementalPipeline(
        const IncrementalPipelineOptions* options,
        const std::filesystem::path& image_path,
        const std::filesystem::path& database_path,
        ReconstructionManager* reconstruction_manager)
    : options_(options),
      image_path_(image_path.string()),
      database_path_(database_path.string()),
      reconstruction_manager_(reconstruction_manager) {
    CHECK(options_->Check());
    RegisterCallback(INITIAL_IMAGE_PAIR_REG_CALLBACK);
    RegisterCallback(NEXT_IMAGE_REG_CALLBACK);
    RegisterCallback(LAST_IMAGE_REG_CALLBACK);
}

IncrementalPipeline::IncrementalPipeline(
        const IncrementalPipelineOptions* options,
        const std::filesystem::path& image_path,
        std::shared_ptr<const DatabaseCache> database_cache,
        ReconstructionManager* reconstruction_manager)
    : options_(options),
      image_path_(image_path.string()),
      reconstruction_manager_(reconstruction_manager),
      source_database_cache_(std::move(database_cache)) {
    CHECK(options_->Check());
    CHECK_NOTNULL(source_database_cache_.get());
    RegisterCallback(INITIAL_IMAGE_PAIR_REG_CALLBACK);
    RegisterCallback(NEXT_IMAGE_REG_CALLBACK);
    RegisterCallback(LAST_IMAGE_REG_CALLBACK);
}

void IncrementalPipeline::SetCheckIfStoppedFunc(
        std::function<bool()> func) {
    check_if_stopped_func_ = std::move(func);
}

bool IncrementalPipeline::ShouldStop() {
    return IsStopped() ||
           (check_if_stopped_func_ && check_if_stopped_func_());
}

bool IncrementalPipeline::CheckReachedMaxRuntime() const {
    if (options_->max_runtime_seconds > 0 &&
        total_run_timer_.ElapsedSeconds() > options_->max_runtime_seconds) {
        LOG(INFO) << "Reached maximum runtime of "
                  << options_->max_runtime_seconds << " seconds.";
        return true;
    }
    return false;
}

// Upstream parity (a2bd15c2 IncrementalPipeline::Run): one mapper spans the
// initial attempt and both relaxation rounds, so the registration statistics
// and the total registered-image count carry over; relaxation continues while
// images remain unregistered, even if a model already exists.
void IncrementalPipeline::Run() {
    total_run_timer_.Start();

    if (!LoadDatabase()) {
        return;
    }

    if (options_->use_prior_position && database_cache_->NumPosePriors() == 0) {
        LOG(WARNING) << "No pose priors";
        return;
    }

    // Is there a sub-reconstruction before we start the reconstruction? I.e.
    // the user has imported an existing reconstruction.
    const bool continue_reconstruction = reconstruction_manager_->Size() > 0;
    CHECK_LE(reconstruction_manager_->Size(), 1)
            << "Can only continue from a single reconstruction, "
               "but multiple are given.";

    const size_t num_images = database_cache_->NumImages();

    IncrementalMapper::Options mapper_options = options_->Mapper();
    IncrementalMapper mapper(database_cache_.get());
    if (Reconstruct(mapper, mapper_options, continue_reconstruction) ==
        Status::STOP) {
        GetTimer().PrintMinutes();
        return;
    }

    auto ShouldStopRelaxation = [this, &mapper, &num_images]() {
        return mapper.NumTotalRegImages() == num_images || ShouldStop() ||
               CheckReachedMaxRuntime();
    };

    const size_t kNumInitRelaxations = 2;
    for (size_t i = 0; i < kNumInitRelaxations; ++i) {
        if (ShouldStopRelaxation()) {
            break;
        }

        std::cout << "  => Relaxing the initialization constraints."
                  << std::endl;
        mapper_options.init_min_num_inliers /= 2;
        mapper.ResetInitializationStats();
        if (Reconstruct(mapper, mapper_options,
                        /*continue_reconstruction=*/false) == Status::STOP) {
            break;
        }

        if (ShouldStopRelaxation()) {
            break;
        }

        std::cout << "  => Relaxing the initialization constraints."
                  << std::endl;
        mapper_options.init_min_tri_angle /= 2;
        mapper.ResetInitializationStats();
        if (Reconstruct(mapper, mapper_options,
                        /*continue_reconstruction=*/false) == Status::STOP) {
            break;
        }
    }

    std::cout << std::endl;
    GetTimer().PrintMinutes();
}

bool IncrementalPipeline::LoadDatabase() {
    PrintHeading1("Loading database");

    DatabaseCache::Options cache_options;
    cache_options.min_num_matches =
            static_cast<size_t>(options_->min_num_matches);
    cache_options.ignore_watermarks = options_->ignore_watermarks;
    cache_options.image_names = {options_->image_names.begin(),
                                 options_->image_names.end()};
    // Make sure images of the given reconstruction are also included when
    // manually specifying images for the reconstruction procedure.
    if (reconstruction_manager_->Size() == 1 &&
        !options_->image_names.empty()) {
        const Reconstruction& reconstruction = reconstruction_manager_->Get(0);
        for (const image_t image_id : reconstruction.RegImageIds()) {
            const auto& image = reconstruction.Image(image_id);
            cache_options.image_names.insert(image.Name());
        }
    }
    cache_options.load_all_images = options_->load_all_images;
    cache_options.convert_pose_priors_to_enu = options_->use_prior_position;

    Timer timer;
    timer.Start();
    if (source_database_cache_) {
        database_cache_ = DatabaseCache::CreateFromCache(
                *source_database_cache_, cache_options);
    } else {
        auto database = Database::Open(database_path_);
        database_cache_ = DatabaseCache::Create(*database, cache_options);
    }
    std::cout << std::endl;
    timer.PrintMinutes();

    std::cout << std::endl;

    // Upstream parity (a2bd15c2 CustomizeIncrementalPipelineOptions): if the
    // total number of images is small then do not enforce the minimum model
    // size so that we can reconstruct small image collections, i.e., if the
    // model is at least half of the total number of images, we always keep
    // it.
    min_model_size_ = static_cast<int>(
            std::min<size_t>(0.5 * database_cache_->NumImages(),
                             static_cast<size_t>(options_->min_model_size)));

    if (database_cache_->NumImages() == 0) {
        std::cout << "WARNING: No images with matches found in the database."
                  << std::endl
                  << std::endl;
        return false;
    }

    return true;
}

IncrementalPipeline::Status IncrementalPipeline::InitializeReconstruction(
        IncrementalMapper& mapper,
        const IncrementalMapper::Options& mapper_options,
        Reconstruction& reconstruction) {
    image_t image_id1 = static_cast<image_t>(options_->init_image_id1);
    image_t image_id2 = static_cast<image_t>(options_->init_image_id2);

    // Try to find good initial pair.
    Rigid3d cam2_from_cam1;
    if (options_->init_image_id1 == -1 || options_->init_image_id2 == -1) {
        PrintHeading1("Finding good initial image pair");
        const bool find_init_success = mapper.FindInitialImagePair(
                mapper_options, image_id1, image_id2, cam2_from_cam1);
        if (ShouldStop() || CheckReachedMaxRuntime()) {
            return Status::INTERRUPTED;
        }
        if (!find_init_success) {
            std::cout << "  => No good initial image pair found." << std::endl;
            return Status::NO_INITIAL_PAIR;
        }
    } else {
        if (!reconstruction.ExistsImage(image_id1) ||
            !reconstruction.ExistsImage(image_id2)) {
            std::cout << StringPrintf(
                                 "  => Initial image pair #%d and #%d does "
                                 "not exist.",
                                 image_id1, image_id2)
                      << std::endl;
            return Status::NO_INITIAL_PAIR;
        }
        const bool provided_init_success =
                mapper.EstimateInitialTwoViewGeometry(
                        mapper_options, image_id1, image_id2, cam2_from_cam1);
        if (!provided_init_success) {
            std::cout << "  => Provided pair is unsuitable for initialization."
                      << std::endl;
            return Status::BAD_INITIAL_PAIR;
        }
    }

    PrintHeading1(StringPrintf("Registering initial image pair #%d and #%d",
                               image_id1, image_id2));
    if (!mapper.RegisterInitialImagePair(mapper_options, image_id1, image_id2,
                                         cam2_from_cam1)) {
        return Status::BAD_INITIAL_PAIR;
    }

    IncrementalTriangulator::Options tri_options = options_->Triangulation();
    tri_options.min_angle = mapper_options.init_min_tri_angle;
    for (const image_t image_id : {image_id1, image_id2}) {
        TriangulateFrame(tri_options, reconstruction.Image(image_id), &mapper);
    }

    if (reconstruction.NumPoints3D() == 0) {
        return Status::BAD_INITIAL_PAIR;
    }

    PrintHeading1("Global bundle adjustment");
    BundleAdjustmentOptions ba_options = options_->GlobalBundleAdjustment();
    ba_options.check_if_stopped = [this]() { return ShouldStop(); };
    mapper.AdjustGlobalBundle(mapper_options, ba_options);
    if (ShouldStop() || CheckReachedMaxRuntime()) {
        return Status::INTERRUPTED;
    }
    reconstruction.Normalize();
    mapper.FilterPoints(mapper_options);
    mapper.FilterFrames(mapper_options);

    // Initial image pair failed to register.
    if (reconstruction.NumRegFrames() == 0 ||
        reconstruction.NumPoints3D() == 0) {
        return Status::BAD_INITIAL_PAIR;
    }

    // Number of triangulated points not enough for registering future images.
    if (static_cast<int>(reconstruction.NumPoints3D()) <
        mapper_options.abs_pose_min_num_inliers) {
        return Status::BAD_INITIAL_PAIR;
    }

    if (options_->extract_colors) {
        for (const image_t image_id : {image_id1, image_id2}) {
            ExtractFrameColors(image_path_, reconstruction.Image(image_id),
                               &reconstruction);
        }
    }
    return Status::SUCCESS;
}

bool IncrementalPipeline::CheckRunGlobalRefinement(
        const Reconstruction& reconstruction,
        const size_t ba_prev_num_reg_frames,
        const size_t ba_prev_num_points) const {
    return reconstruction.NumRegFrames() >=
                   options_->ba_global_frames_ratio * ba_prev_num_reg_frames ||
           reconstruction.NumRegFrames() >=
                   options_->ba_global_frames_freq + ba_prev_num_reg_frames ||
           reconstruction.NumPoints3D() >=
                   options_->ba_global_points_ratio * ba_prev_num_points ||
           reconstruction.NumPoints3D() >=
                   options_->ba_global_points_freq + ba_prev_num_points;
}

IncrementalPipeline::Status IncrementalPipeline::ReconstructSubModel(
        IncrementalMapper& mapper,
        const IncrementalMapper::Options& mapper_options,
        Reconstruction& reconstruction) {
    mapper.BeginReconstruction(&reconstruction);

    if (HasUnknownSensorFromRig(reconstruction)) {
        return Status::UNKNOWN_SENSOR_FROM_RIG;
    }

    ////////////////////////////////////////////////////////////////////////////
    // Register initial pair
    ////////////////////////////////////////////////////////////////////////////

    if (reconstruction.NumRegFrames() == 0) {
        const Status init_status =
                InitializeReconstruction(mapper, mapper_options, reconstruction);
        if (init_status != Status::SUCCESS) {
            return init_status;
        }
    }
    Callback(INITIAL_IMAGE_PAIR_REG_CALLBACK);

    ////////////////////////////////////////////////////////////////////////////
    // Incremental mapping
    ////////////////////////////////////////////////////////////////////////////

    size_t snapshot_prev_num_reg_frames = reconstruction.NumRegFrames();
    size_t ba_prev_num_reg_frames = reconstruction.NumRegFrames();
    size_t ba_prev_num_points = reconstruction.NumPoints3D();

    std::vector<bool> structure_less_flags;
    if (options_->structure_less_registration_only) {
        structure_less_flags = {true};
    } else if (options_->structure_less_registration_fallback) {
        structure_less_flags = {false, true};
    } else {
        structure_less_flags = {false};
    }

    const auto check_if_stopped = [this]() { return ShouldStop(); };

    bool reg_next_success = true;
    bool prev_reg_next_success = true;
    do {
        BlockIfPaused();
        if (ShouldStop() || CheckReachedMaxRuntime()) {
            break;
        }

        prev_reg_next_success = reg_next_success;
        reg_next_success = false;
        image_t next_image_id = kInvalidImageId;

        // Try to register next image. Always prefer structure-based
        // registration first, and if that fails, try (less reliable)
        // structure-less registration.
        for (const bool structure_less : structure_less_flags) {
            const std::vector<image_t> next_images = mapper.FindNextImages(
                    mapper_options, /*structure_less=*/structure_less);

            for (size_t reg_trial = 0; reg_trial < next_images.size();
                 ++reg_trial) {
                next_image_id = next_images[reg_trial];

                PrintHeading1(StringPrintf(
                        "Registering image #%d (num_reg_frames=%d)",
                        next_image_id, reconstruction.NumRegFrames()));
                std::cout << StringPrintf(
                                     "  => Image sees %d / %d points",
                                     mapper.ObservationManager()
                                             .NumVisiblePoints3D(next_image_id),
                                     mapper.ObservationManager()
                                             .NumObservations(next_image_id))
                          << std::endl;

                if (structure_less) {
                    std::cout << "  => Registering image with structure-less "
                                 "fallback"
                              << std::endl;
                    std::cout << StringPrintf(
                                         "  => Image sees %d / %d "
                                         "correspondences",
                                         mapper.ObservationManager()
                                                 .NumVisibleCorrespondences(
                                                         next_image_id),
                                         mapper.ObservationManager()
                                                 .NumCorrespondences(
                                                         next_image_id))
                              << std::endl;
                    reg_next_success = mapper.RegisterNextStructureLessImage(
                            mapper_options, next_image_id);
                } else {
                    reg_next_success = mapper.RegisterNextImage(mapper_options,
                                                                next_image_id);
                }

                if (reg_next_success) {
                    break;
                } else {
                    std::cout << "  => Could not register, trying another "
                                 "image."
                              << std::endl;

                    // If initial model fails to continue for some time,
                    // abort and try different initial pair.
                    const size_t kMinNumInitialRegTrials = 30;
                    if (reg_trial >= kMinNumInitialRegTrials &&
                        reconstruction.NumRegImages() <
                                static_cast<size_t>(min_model_size_)) {
                        break;
                    }
                }
            }

            if (reg_next_success) {
                break;
            }
        }

        if (reg_next_success) {
            const Image& image = reconstruction.Image(next_image_id);
            TriangulateFrame(options_->Triangulation(), image, &mapper);
            BundleAdjustmentOptions ba_options =
                    options_->LocalBundleAdjustment();
            ba_options.check_if_stopped = check_if_stopped;
            mapper.IterativeLocalRefinement(
                    options_->ba_local_max_refinements,
                    options_->ba_local_max_refinement_change, mapper_options,
                    ba_options, options_->Triangulation(), next_image_id);

            if (ShouldStop() || CheckReachedMaxRuntime()) {
                break;
            }

            if (CheckRunGlobalRefinement(reconstruction,
                                         ba_prev_num_reg_frames,
                                         ba_prev_num_points)) {
                IterativeGlobalRefinement(*options_, mapper_options, &mapper,
                                          check_if_stopped);
                if (ShouldStop() || CheckReachedMaxRuntime()) {
                    break;
                }
                ba_prev_num_points = reconstruction.NumPoints3D();
                ba_prev_num_reg_frames = reconstruction.NumRegFrames();
            }

            if (options_->extract_colors) {
                ExtractFrameColors(image_path_, image, &reconstruction);
            }

            if (options_->snapshot_frames_freq > 0 &&
                reconstruction.NumRegFrames() >=
                        options_->snapshot_frames_freq +
                                snapshot_prev_num_reg_frames) {
                snapshot_prev_num_reg_frames = reconstruction.NumRegFrames();
                WriteSnapshot(reconstruction, options_->snapshot_path);
            }

            Callback(NEXT_IMAGE_REG_CALLBACK);
        }

        const size_t max_model_overlap =
                static_cast<size_t>(options_->max_model_overlap);
        if (mapper.NumSharedRegImages() >= max_model_overlap) {
            break;
        }

        // If no image could be registered, try a single final global iterative
        // bundle adjustment and try again to register one image. If this fails
        // once, then exit the incremental mapping.
        if (!reg_next_success && prev_reg_next_success) {
            IterativeGlobalRefinement(*options_, mapper_options, &mapper,
                                      check_if_stopped);
        }
    } while (reg_next_success || prev_reg_next_success);

    if (ShouldStop() || CheckReachedMaxRuntime()) {
        return Status::INTERRUPTED;
    }

    // Only run final global BA, if last incremental BA was not global.
    if (reconstruction.NumRegFrames() > 0 &&
        reconstruction.NumRegFrames() != ba_prev_num_reg_frames &&
        reconstruction.NumPoints3D() != ba_prev_num_points) {
        IterativeGlobalRefinement(*options_, mapper_options, &mapper,
                                  check_if_stopped);
    }
    return Status::SUCCESS;
}

IncrementalPipeline::Status IncrementalPipeline::Reconstruct(
        IncrementalMapper& mapper,
        const IncrementalMapper::Options& mapper_options,
        bool continue_reconstruction) {
    for (int num_trials = 0; num_trials < options_->init_num_trials;
         ++num_trials) {
        BlockIfPaused();
        if (ShouldStop() || CheckReachedMaxRuntime()) {
            return Status::STOP;
        }

        const size_t reconstruction_idx =
                (!continue_reconstruction || num_trials > 0)
                        ? reconstruction_manager_->Add()
                        : 0;
        Reconstruction& reconstruction =
                reconstruction_manager_->Get(reconstruction_idx);

        const Status status =
                ReconstructSubModel(mapper, mapper_options, reconstruction);
        switch (status) {
            case Status::INTERRUPTED: {
                if (reconstruction.NumRegFrames() == 0) {
                    mapper.EndReconstruction(/*discard=*/true);
                    reconstruction_manager_->Delete(reconstruction_idx);
                    return Status::STOP;
                }
                reconstruction.UpdatePoint3DErrors();
                std::cout << "  => Keeping reconstruction due to interrupt"
                          << std::endl;
                mapper.EndReconstruction(/*discard=*/false);
                AlignReconstructionToPriorsOrRigScale(
                        *options_, *database_cache_, &reconstruction);
                return Status::STOP;
            }

            case Status::UNKNOWN_SENSOR_FROM_RIG: {
                LOG(ERROR) << "Discarding reconstruction due to unknown "
                              "sensor_from_rig poses. Either explicitly define "
                              "the poses by configuring the rigs or first run "
                              "reconstruction without configured rigs and then "
                              "derive the poses from the initial "
                              "reconstruction for a subsequent reconstruction "
                              "with rig constraints. See documentation for "
                              "detailed instructions.";
                mapper.EndReconstruction(/*discard=*/true);
                reconstruction_manager_->Delete(reconstruction_idx);
                // If the reconstruction was discarded due to an unknown sensor
                // from rig, we can stop the outer trial loop, because all
                // trials will fail.
                return Status::STOP;
            }

            case Status::BAD_INITIAL_PAIR: {
                std::cout << "  => Discarding reconstruction due to bad "
                             "initial pair"
                          << std::endl;
                mapper.EndReconstruction(/*discard=*/true);
                reconstruction_manager_->Delete(reconstruction_idx);
                // If an initial pair was found but it was bad, we discard and
                // attempt to initialize from any of the remaining pairs in the
                // next trials.
                break;
            }

            case Status::NO_INITIAL_PAIR: {
                std::cout << "  => Discarding reconstruction due to no "
                             "initial pair"
                          << std::endl;
                mapper.EndReconstruction(/*discard=*/true);
                reconstruction_manager_->Delete(reconstruction_idx);
                // If no pair could be found, we can exit the trial loop,
                // because the next trials in this loop will not find anything
                // unless the initialization thresholds are relaxed. However,
                // by relaxing the constraints in the outer loop we can
                // succeed.
                return Status::CONTINUE;
            }

            case Status::SUCCESS: {
                // Remember the total number of registered images before
                // potentially discarding it below due to small size, so we
                // can exit out of the main loop, if all images were
                // registered.
                const size_t num_reg_images = reconstruction.NumRegImages();
                const size_t total_num_reg_images = mapper.NumTotalRegImages();

                // Always keep the first reconstruction, independent of size.
                if ((options_->multiple_models &&
                     reconstruction_manager_->Size() > 1 &&
                     num_reg_images < static_cast<size_t>(min_model_size_)) ||
                    num_reg_images == 0) {
                    std::cout << "  => Discarding reconstruction due to "
                                 "insufficient size"
                              << std::endl;
                    mapper.EndReconstruction(/*discard=*/true);
                    reconstruction_manager_->Delete(reconstruction_idx);
                } else {
                    reconstruction.UpdatePoint3DErrors();
                    std::cout << "  => Keeping successful reconstruction"
                              << std::endl;
                    mapper.EndReconstruction(/*discard=*/false);
                    AlignReconstructionToPriorsOrRigScale(
                            *options_, *database_cache_, &reconstruction);
                }

                Callback(LAST_IMAGE_REG_CALLBACK);

                // Check if we should or can reconstruct another sub-model.
                if (!options_->multiple_models ||
                    reconstruction_manager_->Size() >=
                            static_cast<size_t>(options_->max_num_models) ||
                    total_num_reg_images >= database_cache_->NumImages() - 1) {
                    return Status::STOP;
                }

                // In case the reconstruction was successful and there are
                // remaining images, we try to reconstruct another sub-model in
                // the next trial.
                break;
            }

            default:
                LOG(FATAL) << "Unknown reconstruction status.";
        }
    }

    return Status::CONTINUE;
}

}  // namespace colmap
