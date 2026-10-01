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
#include "scene/projection.h"

#include "util/misc.h"

namespace colmap {
namespace {

size_t TriangulateImage(const IncrementalMapperOptions& options,
                        const Image& image,
                        IncrementalMapper* mapper) {
    std::cout << "  => Continued observations: " << image.NumPoints3D()
              << std::endl;
    const size_t num_tris =
            mapper->TriangulateImage(options.Triangulation(), image.ImageId());
    std::cout << "  => Added observations: " << num_tris << std::endl;
    return num_tris;
}

// Upstream parity (d3ccaf35 incremental_pipeline.cc): anchor the final
// reconstruction to the metric prior positions when enabled, falling back to
// the original rig extrinsic scales; without this the BA gauge lets the
// model drift away from the GT scale.
void AlignReconstructionToPriorsOrRigScale(
        const IncrementalMapperOptions& options,
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

void AdjustGlobalBundle(const IncrementalMapperOptions& options,
                        IncrementalMapper* mapper) {
    BundleAdjustmentOptions custom_ba_options =
            options.GlobalBundleAdjustment();

    const size_t num_reg_images = mapper->GetReconstruction().NumRegImages();

    // Use stricter convergence criteria for first registered images.
    const size_t kMinNumRegImagesForFastBA = 10;
    if (num_reg_images < kMinNumRegImagesForFastBA) {
        custom_ba_options.solver_options.function_tolerance /= 10;
        custom_ba_options.solver_options.gradient_tolerance /= 10;
        custom_ba_options.solver_options.parameter_tolerance /= 10;
        custom_ba_options.solver_options.max_num_iterations *= 2;
        custom_ba_options.solver_options.max_linear_solver_iterations = 200;
    }

    PrintHeading1("Global bundle adjustment");
    mapper->AdjustGlobalBundle(options.Mapper(), custom_ba_options);
}

void IterativeLocalRefinement(const IncrementalMapperOptions& options,
                              const image_t image_id,
                              IncrementalMapper* mapper) {
    // Upstream parity (d3ccaf35): the refinement loop lives in the mapper
    // (stop-aware, zero-adjustment guard, VLOG diagnostics, robust-loss
    // demotion after the first iteration).
    mapper->IterativeLocalRefinement(
            options.ba_local_max_refinements,
            options.ba_local_max_refinement_change, options.Mapper(),
            options.LocalBundleAdjustment(), options.Triangulation(),
            image_id);
}

void IterativeGlobalRefinement(const IncrementalMapperOptions& options,
                               IncrementalMapper* mapper) {
    PrintHeading1("Retriangulation and Global bundle adjustment");
    // Upstream parity (d3ccaf35 incremental_pipeline.cc): the refinement
    // rounds run inside the mapper, which normalizes the scene after every
    // global bundle adjustment (numerical stability) and applies the same
    // complete/merge/retriangulate + changed-observation convergence logic.
    // The old fork loop had no per-round Normalize, so the exported gauge
    // drifted relative to upstream (x1.37 on mini6).
    mapper->IterativeGlobalRefinement(
            options.ba_global_max_refinements,
            options.ba_global_max_refinement_change,
            options.Mapper(), options.GlobalBundleAdjustment(),
            options.Triangulation());
    FilterFrames(options, mapper);
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

size_t FilterPoints(const IncrementalMapperOptions& options,
                    IncrementalMapper* mapper) {
    const size_t num_filtered_observations =
            mapper->FilterPoints(options.Mapper());
    std::cout << "  => Filtered observations: " << num_filtered_observations
              << std::endl;
    return num_filtered_observations;
}

size_t FilterFrames(const IncrementalMapperOptions& options,
                    IncrementalMapper* mapper) {
    const size_t num_filtered_images = mapper->FilterFrames(options.Mapper());
    std::cout << "  => Filtered images: " << num_filtered_images << std::endl;
    return num_filtered_images;
}

size_t CompleteAndMergeTracks(const IncrementalMapperOptions& options,
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

IncrementalMapper::Options IncrementalMapperOptions::Mapper() const {
    IncrementalMapper::Options options = mapper;
    options.abs_pose_refine_focal_length = ba_refine_focal_length;
    options.abs_pose_refine_extra_params = ba_refine_extra_params;
    options.min_focal_length_ratio = min_focal_length_ratio;
    options.max_focal_length_ratio = max_focal_length_ratio;
    options.max_extra_param = max_extra_param;
    options.num_threads = num_threads;
    options.local_ba_num_images = ba_local_num_images;
    options.fix_existing_images = fix_existing_images;
    options.constant_rigs = constant_rigs;
    options.constant_cameras = constant_cameras;
    options.random_seed = random_seed;
    options.ba_global_ignore_redundant_points3D =
            ba_global_ignore_redundant_points3D;
    options.ba_global_ignore_redundant_points3D_min_coverage_gain =
            ba_global_ignore_redundant_points3D_min_coverage_gain;
    options.use_prior_position = use_prior_position;
    options.use_robust_loss_on_prior_position =
            use_robust_loss_on_prior_position;
    options.prior_position_loss_scale = prior_position_loss_scale;
    return options;
}

IncrementalTriangulator::Options IncrementalMapperOptions::Triangulation()
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

BundleAdjustmentOptions IncrementalMapperOptions::LocalBundleAdjustment()
        const {
    BundleAdjustmentOptions options;
    options.solver_options.function_tolerance = ba_local_function_tolerance;
    options.solver_options.gradient_tolerance = 10.0;
    options.solver_options.parameter_tolerance = 0.0;
    options.solver_options.max_num_iterations =
            EffBaLocalMaxNumIterations();
    options.solver_options.max_linear_solver_iterations = 100;
    options.solver_options.minimizer_progress_to_stdout = false;
    options.solver_options.num_threads = num_threads;
#if CERES_VERSION_MAJOR < 2
    options.solver_options.num_linear_solver_threads = num_threads;
#endif  // CERES_VERSION_MAJOR
    options.print_summary = true;
    options.refine_focal_length = ba_refine_focal_length;
    options.refine_principal_point = ba_refine_principal_point;
    options.refine_extra_params = ba_refine_extra_params;
    options.refine_sensor_from_rig = ba_refine_sensor_from_rig;
    options.min_num_residuals_for_cpu_multi_threading =
            ba_min_num_residuals_for_cpu_multi_threading;
    options.loss_function_scale = 1.0;
    options.loss_function_type =
            BundleAdjustmentOptions::LossFunctionType::SOFT_L1;
    options.use_gpu = ba_use_gpu;
    options.gpu_index = ba_gpu_index;
    options.backend = ba_local_backend;
    return options;
}


int IncrementalMapperOptions::EffBaLocalMaxNumIterations() const {
    if (ba_local_max_num_iterations >= 0) {
        return ba_local_max_num_iterations;
    }
    if (ba_local_backend == BundleAdjustmentBackend::CASPAR) {
        return CasparBundleAdjustmentOptions().solver_iter_max;
    }
    return kDefaultCeresLocalMaxNumIterations;
}

int IncrementalMapperOptions::EffBaGlobalMaxNumIterations() const {
    if (ba_global_max_num_iterations >= 0) {
        return ba_global_max_num_iterations;
    }
    if (ba_global_backend == BundleAdjustmentBackend::CASPAR) {
        return CasparBundleAdjustmentOptions().solver_iter_max;
    }
    return kDefaultCeresGlobalMaxNumIterations;
}

BundleAdjustmentOptions IncrementalMapperOptions::GlobalBundleAdjustment()
        const {
    BundleAdjustmentOptions options;
    options.solver_options.function_tolerance = ba_global_function_tolerance;
    options.solver_options.gradient_tolerance = 1.0;
    options.solver_options.parameter_tolerance = 0.0;
    options.solver_options.max_num_iterations =
            EffBaGlobalMaxNumIterations();
    options.solver_options.max_linear_solver_iterations = 100;
    options.solver_options.minimizer_progress_to_stdout = true;
    options.solver_options.num_threads = num_threads;
#if CERES_VERSION_MAJOR < 2
    options.solver_options.num_linear_solver_threads = num_threads;
#endif  // CERES_VERSION_MAJOR
    options.print_summary = true;
    options.refine_focal_length = ba_refine_focal_length;
    options.refine_principal_point = ba_refine_principal_point;
    options.refine_extra_params = ba_refine_extra_params;
    options.refine_sensor_from_rig = ba_refine_sensor_from_rig;
    options.min_num_residuals_for_cpu_multi_threading =
            ba_min_num_residuals_for_cpu_multi_threading;
    options.loss_function_type =
            BundleAdjustmentOptions::LossFunctionType::TRIVIAL;
    options.use_gpu = ba_use_gpu;
    options.gpu_index = ba_gpu_index;
    options.backend = ba_global_backend;
    return options;
}

bool IncrementalMapperOptions::Check() const {
    CHECK_OPTION_GT(min_num_matches, 0);
    CHECK_OPTION_GT(max_num_models, 0);
    CHECK_OPTION_GT(max_model_overlap, 0);
    CHECK_OPTION_GE(min_model_size, 0);
    CHECK_OPTION_GT(init_num_trials, 0);
    CHECK_OPTION_GT(min_focal_length_ratio, 0);
    CHECK_OPTION_GT(max_focal_length_ratio, 0);
    CHECK_OPTION_GE(max_extra_param, 0);
    CHECK_OPTION_GE(ba_local_num_images, 2);
    // Upstream parity: -1 is the "backend default" sentinel and legal.
    CHECK_OPTION_GE(ba_local_max_num_iterations, -1);
    CHECK_OPTION_GT(ba_global_images_ratio, 1.0);
    CHECK_OPTION_GT(ba_global_points_ratio, 1.0);
    CHECK_OPTION_GT(ba_global_images_freq, 0);
    CHECK_OPTION_GT(ba_global_points_freq, 0);
        // Upstream parity: -1 is the "backend default" sentinel and legal.
    CHECK_OPTION_GE(ba_global_max_num_iterations, -1);
    CHECK_OPTION_GT(ba_local_max_refinements, 0);
    CHECK_OPTION_GE(ba_local_max_refinement_change, 0);
    CHECK_OPTION_GT(ba_global_max_refinements, 0);
    CHECK_OPTION_GE(ba_global_max_refinement_change, 0);
    CHECK_OPTION_GE(snapshot_images_freq, 0);
    CHECK_OPTION(Mapper().Check());
    CHECK_OPTION(Triangulation().Check());
    return true;
}

IncrementalMapperController::IncrementalMapperController(
        const IncrementalMapperOptions* options,
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

void IncrementalMapperController::SetCheckIfStoppedFunc(
        std::function<bool()> func) {
    check_if_stopped_func_ = std::move(func);
}

bool IncrementalMapperController::ShouldStop() {
    return IsStopped() ||
           (check_if_stopped_func_ && check_if_stopped_func_());
}

void IncrementalMapperController::Run() {
    if (!LoadDatabase()) {
        return;
    }

    IncrementalMapper::Options init_mapper_options = options_->Mapper();
    Reconstruct(init_mapper_options);

    const size_t kNumInitRelaxations = 2;
    for (size_t i = 0; i < kNumInitRelaxations; ++i) {
        if (reconstruction_manager_->Size() > 0 || ShouldStop()) {
            break;
        }

        std::cout << "  => Relaxing the initialization constraints."
                  << std::endl;
        init_mapper_options.init_min_num_inliers /= 2;
        Reconstruct(init_mapper_options);

        if (reconstruction_manager_->Size() > 0 || ShouldStop()) {
            break;
        }

        std::cout << "  => Relaxing the initialization constraints."
                  << std::endl;
        init_mapper_options.init_min_tri_angle /= 2;
        Reconstruct(init_mapper_options);
    }

    std::cout << std::endl;
    GetTimer().PrintMinutes();
}

bool IncrementalMapperController::LoadDatabase() {
    PrintHeading1("Loading database");

    // Make sure images of the given reconstruction are also included when
    // manually specifying images for the reconstrunstruction procedure.
    std::unordered_set<std::string> image_names = options_->image_names;
    if (reconstruction_manager_->Size() == 1 &&
        !options_->image_names.empty()) {
        const Reconstruction& reconstruction = reconstruction_manager_->Get(0);
        for (const image_t image_id : reconstruction.RegImageIds()) {
            const auto& image = reconstruction.Image(image_id);
            image_names.insert(image.Name());
        }
    }

    auto database = Database::Open(database_path_);
    Timer timer;
    timer.Start();
    DatabaseCache::Options cache_options;
    cache_options.min_num_matches =
            static_cast<size_t>(options_->min_num_matches);
    cache_options.ignore_watermarks = options_->ignore_watermarks;
    cache_options.image_names = {image_names.begin(), image_names.end()};
    cache_options.convert_pose_priors_to_enu = options_->use_prior_position;
    database_cache_.Load(*database, cache_options);
    std::cout << std::endl;
    timer.PrintMinutes();

    std::cout << std::endl;

    if (database_cache_.NumImages() == 0) {
        std::cout << "WARNING: No images with matches found in the database."
                  << std::endl
                  << std::endl;
        return false;
    }

    return true;
}

void IncrementalMapperController::Reconstruct(
        const IncrementalMapper::Options& init_mapper_options) {
    const bool kDiscardReconstruction = true;

    //////////////////////////////////////////////////////////////////////////////
    // Main loop
    //////////////////////////////////////////////////////////////////////////////

    IncrementalMapper mapper(&database_cache_);

    // Is there a sub-model before we start the reconstruction? I.e. the user
    // has imported an existing reconstruction.
    const bool initial_reconstruction_given =
            reconstruction_manager_->Size() > 0;
    CHECK_LE(reconstruction_manager_->Size(), 1)
            << "Can only resume from a "
               "single reconstruction, but "
               "multiple are given.";

    for (int num_trials = 0; num_trials < options_->init_num_trials;
         ++num_trials) {
        BlockIfPaused();
        if (ShouldStop()) {
            break;
        }

        size_t reconstruction_idx;
        if (!initial_reconstruction_given || num_trials > 0) {
            reconstruction_idx = reconstruction_manager_->Add();
        } else {
            reconstruction_idx = 0;
        }

        Reconstruction& reconstruction =
                reconstruction_manager_->Get(reconstruction_idx);

        mapper.BeginReconstruction(&reconstruction);
        if (HasUnknownSensorFromRig(reconstruction)) {
            LOG(ERROR)
                    << "Discarding reconstruction due to unknown "
                       "sensor_from_rig poses. Either explicitly define the "
                       "poses by configuring the rigs or first run "
                       "reconstruction without configured rigs and then "
                       "derive the poses from the initial reconstruction for "
                       "a subsequent reconstruction with rig constraints. See "
                       "documentation for detailed instructions.";
            mapper.EndReconstruction(/*discard=*/true);
            reconstruction_manager_->Delete(reconstruction_idx);
            break;
        }

        ////////////////////////////////////////////////////////////////////////////
        // Register initial pair
        ////////////////////////////////////////////////////////////////////////////

        if (reconstruction.NumRegImages() == 0) {
            image_t image_id1 = static_cast<image_t>(options_->init_image_id1);
            image_t image_id2 = static_cast<image_t>(options_->init_image_id2);
            Rigid3d cam2_from_cam1;

            // Try to find good initial pair.
            if (options_->init_image_id1 == -1 ||
                options_->init_image_id2 == -1) {
                PrintHeading1("Finding good initial image pair");
                const bool find_init_success = mapper.FindInitialImagePair(
                        init_mapper_options, image_id1, image_id2,
                        cam2_from_cam1);
                if (!find_init_success) {
                    std::cout << "  => No good initial image pair found."
                              << std::endl;
                    mapper.EndReconstruction(kDiscardReconstruction);
                    reconstruction_manager_->Delete(reconstruction_idx);
                    break;
                }
            } else {
                if (!reconstruction.ExistsImage(image_id1) ||
                    !reconstruction.ExistsImage(image_id2)) {
                    std::cout << StringPrintf(
                                         "  => Initial image pair #%d and #%d "
                                         "do not exist.",
                                         image_id1, image_id2)
                              << std::endl;
                    mapper.EndReconstruction(kDiscardReconstruction);
                    reconstruction_manager_->Delete(reconstruction_idx);
                    return;
                }
                // Upstream parity (d3ccaf35): a user-provided initial pair
                // still needs its two-view geometry estimated and checked
                // against the initialization thresholds.
                const bool provided_init_success =
                        mapper.EstimateInitialTwoViewGeometry(
                                init_mapper_options, image_id1, image_id2,
                                cam2_from_cam1);
                if (!provided_init_success) {
                    std::cout << "  => Provided pair is unsuitable for "
                                 "initialization."
                              << std::endl;
                    mapper.EndReconstruction(kDiscardReconstruction);
                    reconstruction_manager_->Delete(reconstruction_idx);
                    return;
                }
            }

            PrintHeading1(
                    StringPrintf("Initializing with image pair #%d and #%d",
                                 image_id1, image_id2));
            const bool reg_init_success = mapper.RegisterInitialImagePair(
                    init_mapper_options, image_id1, image_id2,
                    cam2_from_cam1);
            if (!reg_init_success) {
                std::cout
                        << "  => Initialization failed - possible solutions:"
                        << std::endl
                        << "     - try to relax the initialization constraints"
                        << std::endl
                        << "     - manually select an initial image pair"
                        << std::endl;
                mapper.EndReconstruction(kDiscardReconstruction);
                reconstruction_manager_->Delete(reconstruction_idx);
                break;
            }

            // Upstream parity (d3ccaf35 incremental_pipeline.cc): triangulate
            // both initial images through the triangulator (RANSAC-based,
            // reprojection-filtered point creation) instead of unfiltered
            // inline triangulation inside the mapper.
            IncrementalTriangulator::Options init_tri_options =
                    options_->Triangulation();
            init_tri_options.min_angle =
                    init_mapper_options.init_min_tri_angle;
            for (const image_t init_image_id : {image_id1, image_id2}) {
                mapper.TriangulateImage(init_tri_options, init_image_id);
            }

            AdjustGlobalBundle(*options_, &mapper);
            reconstruction.Normalize();
            FilterPoints(*options_, &mapper);
            FilterFrames(*options_, &mapper);

            // Initial image pair failed to register.
            if (reconstruction.NumRegImages() == 0 ||
                reconstruction.NumPoints3D() == 0) {
                mapper.EndReconstruction(kDiscardReconstruction);
                reconstruction_manager_->Delete(reconstruction_idx);
                // If both initial images are manually specified, there is no
                // need for further initialization trials.
                if (options_->init_image_id1 != -1 &&
                    options_->init_image_id2 != -1) {
                    break;
                } else {
                    continue;
                }
            }

            if (options_->extract_colors) {
                ExtractColors(image_path_, image_id1, &reconstruction);
            }
        }

        Callback(INITIAL_IMAGE_PAIR_REG_CALLBACK);

        ////////////////////////////////////////////////////////////////////////////
        // Incremental mapping
        ////////////////////////////////////////////////////////////////////////////

        size_t snapshot_prev_num_reg_images = reconstruction.NumRegImages();
        size_t ba_prev_num_reg_images = reconstruction.NumRegImages();
        size_t ba_prev_num_points = reconstruction.NumPoints3D();

        bool reg_next_success = true;
        bool prev_reg_next_success = true;
        while (reg_next_success) {
            BlockIfPaused();
            if (ShouldStop()) {
                break;
            }

            reg_next_success = false;

            const std::vector<image_t> next_images =
                    mapper.FindNextImages(options_->Mapper());

            if (next_images.empty()) {
                break;
            }

            for (size_t reg_trial = 0; reg_trial < next_images.size();
                 ++reg_trial) {
                const image_t next_image_id = next_images[reg_trial];
                const Image& next_image = reconstruction.Image(next_image_id);

                PrintHeading1(StringPrintf("Registering image #%d (%d)",
                                           next_image_id,
                                           reconstruction.NumRegImages() + 1));

                std::cout << StringPrintf(
                                   "  => Image sees %d / %d points",
                                   mapper.ObservationManager()
                                           .NumVisiblePoints3D(next_image_id),
                                   mapper.ObservationManager().NumObservations(
                                           next_image_id))
                          << std::endl;

                reg_next_success = mapper.RegisterNextImage(options_->Mapper(),
                                                            next_image_id);

                // Upstream parity (d3ccaf35): structure-less fallback
                // registration when structure-based registration fails.
                if (!reg_next_success) {
                    reg_next_success = mapper.RegisterNextStructureLessImage(
                            options_->Mapper(), next_image_id);
                    if (reg_next_success) {
                        TriangulateImage(*options_, next_image, &mapper);
                        IterativeLocalRefinement(*options_, next_image_id,
                                                 &mapper);
                    }
                }

                if (reg_next_success) {
                    TriangulateImage(*options_, next_image, &mapper);
                    IterativeLocalRefinement(*options_, next_image_id, &mapper);

                    if (reconstruction.NumRegImages() >=
                                options_->ba_global_images_ratio *
                                        ba_prev_num_reg_images ||
                        reconstruction.NumRegImages() >=
                                options_->ba_global_images_freq +
                                        ba_prev_num_reg_images ||
                        reconstruction.NumPoints3D() >=
                                options_->ba_global_points_ratio *
                                        ba_prev_num_points ||
                        reconstruction.NumPoints3D() >=
                                options_->ba_global_points_freq +
                                        ba_prev_num_points) {
                        IterativeGlobalRefinement(*options_, &mapper);
                        ba_prev_num_points = reconstruction.NumPoints3D();
                        ba_prev_num_reg_images = reconstruction.NumRegImages();
                    }

                    if (options_->extract_colors) {
                        ExtractColors(image_path_, next_image_id,
                                      &reconstruction);
                    }

                    if (options_->snapshot_images_freq > 0 &&
                        reconstruction.NumRegImages() >=
                                options_->snapshot_images_freq +
                                        snapshot_prev_num_reg_images) {
                        snapshot_prev_num_reg_images =
                                reconstruction.NumRegImages();
                        WriteSnapshot(reconstruction, options_->snapshot_path);
                    }

                    Callback(NEXT_IMAGE_REG_CALLBACK);

                    break;
                } else {
                    std::cout
                            << "  => Could not register, trying another image."
                            << std::endl;

                    // If initial pair fails to continue for some time,
                    // abort and try different initial pair.
                    const size_t kMinNumInitialRegTrials = 30;
                    if (reg_trial >= kMinNumInitialRegTrials &&
                        reconstruction.NumRegImages() <
                                static_cast<size_t>(options_->min_model_size)) {
                        break;
                    }
                }
            }

            const size_t max_model_overlap =
                    static_cast<size_t>(options_->max_model_overlap);
            if (mapper.NumSharedRegImages() >= max_model_overlap) {
                break;
            }

            // If no image could be registered, try a single final global
            // iterative bundle adjustment and try again to register one image.
            // If this fails once, then exit the incremental mapping.
            if (!reg_next_success && prev_reg_next_success) {
                reg_next_success = true;
                prev_reg_next_success = false;
                IterativeGlobalRefinement(*options_, &mapper);
            } else {
                prev_reg_next_success = reg_next_success;
            }
        }

        if (ShouldStop()) {
            const bool kDiscardReconstruction = false;
            mapper.EndReconstruction(kDiscardReconstruction);
            AlignReconstructionToPriorsOrRigScale(
                    *options_, database_cache_, &reconstruction);
            break;
        }

        // Only run final global BA, if last incremental BA was not global.
        if (reconstruction.NumRegImages() >= 2 &&
            reconstruction.NumRegImages() != ba_prev_num_reg_images &&
            reconstruction.NumPoints3D() != ba_prev_num_points) {
            IterativeGlobalRefinement(*options_, &mapper);
        }

        // If the total number of images is small then do not enforce the
        // minimum model size so that we can reconstruct small image
        // collections.
        const size_t min_model_size =
                std::min(database_cache_.NumImages(),
                         static_cast<size_t>(options_->min_model_size));
        if ((options_->multiple_models &&
             reconstruction.NumRegImages() < min_model_size) ||
            reconstruction.NumRegImages() == 0) {
            mapper.EndReconstruction(kDiscardReconstruction);
            reconstruction_manager_->Delete(reconstruction_idx);
        } else {
            const bool kDiscardReconstruction = false;
            mapper.EndReconstruction(kDiscardReconstruction);
            AlignReconstructionToPriorsOrRigScale(
                    *options_, database_cache_, &reconstruction);
        }

        Callback(LAST_IMAGE_REG_CALLBACK);

        const size_t max_num_models =
                static_cast<size_t>(options_->max_num_models);
        if (initial_reconstruction_given || !options_->multiple_models ||
            reconstruction_manager_->Size() >= max_num_models ||
            mapper.NumTotalRegImages() >= database_cache_.NumImages() - 1) {
            break;
        }
    }
}

}  // namespace colmap
