// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "sfm/global_mapper.h"

#include "base/projection.h"
#include "base/triangulation.h"
#include "estimators/rotation_averaging.h"
#include "math/union_find.h"
#include "optim/bundle_adjustment_caspar.h"
#include "sfm/incremental_mapper.h"
#include "util/logging.h"
#include "util/misc.h"
#include "util/timer.h"

#include <algorithm>

namespace colmap {
namespace {

bool RunBundleAdjustment(const BundleAdjustmentOptions& options,
                         Reconstruction& reconstruction) {
  if (reconstruction.NumImages() == 0) {
    LOG(ERROR) << "Cannot run bundle adjustment: no registered images";
    return false;
  }
  if (reconstruction.NumPoints3D() == 0) {
    LOG(ERROR) << "Cannot run bundle adjustment: no 3D points to optimize";
    return false;
  }

  BundleAdjustmentConfig ba_config;
  for (const auto& [image_id, image] : reconstruction.Images()) {
    if (image.HasPose()) {
      ba_config.AddImage(image_id);
    }
  }

  // Fork note: the bundle adjuster still optimizes the legacy per-image
  // qvec_/tvec_ buffers (frame-aware BA lands with W3-2b step 4). Sync
  // those buffers from the frame-aware poses before solving and write the
  // optimized poses back to the frames afterwards, so both pose tracks
  // stay consistent. The qvec convention is [w, x, y, z].
  std::vector<image_t> posed_image_ids;
  posed_image_ids.reserve(reconstruction.NumImages());
  for (const auto& [image_id, image] : reconstruction.Images()) {
    if (image.HasPose()) {
      posed_image_ids.push_back(image_id);
    }
  }
  for (const image_t image_id : posed_image_ids) {
    Image& image = reconstruction.Image(image_id);
    const Rigid3d cam_from_world = image.CamFromWorld();
    const Eigen::Quaterniond q = cam_from_world.rotation();
    image.SetQvec(Eigen::Vector4d(q.w(), q.x(), q.y(), q.z()));
    image.SetTvec(cam_from_world.translation());
  }

  BundleAdjuster ba(options, ba_config);
  const bool success = ba.Solve(&reconstruction);

  for (const image_t image_id : posed_image_ids) {
    const Image& image = reconstruction.Image(image_id);
    const Eigen::Vector4d& qvec = image.Qvec();
    const Rigid3d cam_from_world(
        Eigen::Quaterniond(qvec(0), qvec(1), qvec(2), qvec(3)),
        image.Tvec());
    reconstruction.Frame(image.FrameId())
        .SetCamFromWorld(image.CameraId(), cam_from_world);
  }

  return success;
}

// Fork-legacy stand-in for the upstream ObservationManager filters: the
// per-image pose is read through ProjectionMatrix(), which is frame-aware
// (CamFromWorld when the image is wired into a reconstruction).
Eigen::Vector3d ProjectionCenterOf(const Reconstruction& reconstruction,
                                   image_t image_id) {
  return reconstruction.Image(image_id).ProjectionCenter();
}

void DeleteObservations(Reconstruction& reconstruction,
                        const std::vector<TrackElement>& outliers) {
  for (const auto& track_el : outliers) {
    if (!reconstruction.ExistsImage(track_el.image_id)) {
      continue;
    }
    const point3D_t point3D_id =
        reconstruction.Image(track_el.image_id)
            .Point2D(track_el.point2D_idx)
            .Point3DId();
    if (point3D_id != kInvalidPoint3DId &&
        reconstruction.ExistsPoint3D(point3D_id)) {
      reconstruction.DeleteObservation(track_el.image_id,
                                       track_el.point2D_idx);
    }
  }
}

// Fork parity of the upstream ObservationManager filtering loop
// (sfm/observation_manager.cc FilterPoints3DWithLargeReprojectionError):
// tracks shorter than 2 observations are removed entirely, outliers are
// collected first and deleted afterwards (mutating while iterating a track
// would dangle it), and points that would keep fewer than 2 observations
// are removed entirely. The remaining points carry the mean inlier error.
size_t FilterTracksByAngularError(Reconstruction& reconstruction,
                                  const std::unordered_set<point3D_t>& ids,
                                  double max_angular_error_rad) {
  size_t num_filtered_observations = 0;
  for (const point3D_t point3D_id : ids) {
    if (!reconstruction.ExistsPoint3D(point3D_id)) {
      continue;
    }
    Point3D& point3D = reconstruction.Point3D(point3D_id);
    if (point3D.Track().Length() < 2) {
      num_filtered_observations += point3D.Track().Length();
      reconstruction.DeletePoint3D(point3D_id);
      continue;
    }
    double error_sum = 0.0;
    std::vector<TrackElement> track_els_to_delete;
    for (const auto& track_el : point3D.Track().Elements()) {
      const Image& image = reconstruction.Image(track_el.image_id);
      const Camera& camera = image.CameraPtr() != nullptr
                                 ? *image.CameraPtr()
                                 : reconstruction.Camera(image.CameraId());
      const double error = CalculateAngularError(
          image.Point2D(track_el.point2D_idx).XY(), point3D.XYZ(),
          image.ProjectionMatrix(), camera);
      if (error > max_angular_error_rad) {
        track_els_to_delete.push_back(track_el);
      } else {
        error_sum += error;
      }
    }
    if (track_els_to_delete.size() >= point3D.Track().Length() - 1) {
      num_filtered_observations += point3D.Track().Length();
      reconstruction.DeletePoint3D(point3D_id);
    } else {
      num_filtered_observations += track_els_to_delete.size();
      DeleteObservations(reconstruction, track_els_to_delete);
      point3D.SetError(error_sum / point3D.Track().Length());
    }
  }
  return num_filtered_observations;
}

size_t FilterTracksByNormalizedError(Reconstruction& reconstruction,
                                     const std::unordered_set<point3D_t>& ids,
                                     double max_normalized_error) {
  size_t num_filtered_observations = 0;
  for (const point3D_t point3D_id : ids) {
    if (!reconstruction.ExistsPoint3D(point3D_id)) {
      continue;
    }
    Point3D& point3D = reconstruction.Point3D(point3D_id);
    if (point3D.Track().Length() < 2) {
      num_filtered_observations += point3D.Track().Length();
      reconstruction.DeletePoint3D(point3D_id);
      continue;
    }
    double error_sum = 0.0;
    std::vector<TrackElement> track_els_to_delete;
    for (const auto& track_el : point3D.Track().Elements()) {
      const Image& image = reconstruction.Image(track_el.image_id);
      const Camera& camera = image.CameraPtr() != nullptr
                                 ? *image.CameraPtr()
                                 : reconstruction.Camera(image.CameraId());
      const Point2D& point2D = image.Point2D(track_el.point2D_idx);
      // Upstream NORMALIZED error: distance on the z=1 normalized plane
      // between the projected point and the unprojected observation. The
      // fork has no Camera::CamFromImg, so unproject through the unit ray
      // (CamRayFromImg) and divide by z for perspective models; spherical
      // models compare unit bearings instead (chord distance).
      const Eigen::Vector3d point3D_in_cam =
          image.CamFromWorld() * point3D.XYZ();
      constexpr double kMinDepth = 1e-12;
      double observation_error;
      const std::optional<Eigen::Vector3d> cam_ray =
          camera.CamRayFromImg(point2D.XY());
      if (CameraModelIsPerspective(camera.ModelId())) {
        observation_error =
            (point3D_in_cam.z() >= kMinDepth && cam_ray.has_value())
                ? (point3D_in_cam.hnormalized().head<2>() -
                   cam_ray->hnormalized().head<2>())
                      .norm()
                : std::numeric_limits<double>::infinity();
      } else {
        observation_error =
            cam_ray.has_value()
                ? (point3D_in_cam.normalized() - *cam_ray).norm()
                : std::numeric_limits<double>::infinity();
      }
      if (observation_error > max_normalized_error) {
        track_els_to_delete.push_back(track_el);
      } else {
        error_sum += observation_error;
      }
    }
    if (track_els_to_delete.size() >= point3D.Track().Length() - 1) {
      num_filtered_observations += point3D.Track().Length();
      reconstruction.DeletePoint3D(point3D_id);
    } else {
      num_filtered_observations += track_els_to_delete.size();
      DeleteObservations(reconstruction, track_els_to_delete);
      point3D.SetError(error_sum / point3D.Track().Length());
    }
  }
  return num_filtered_observations;
}

size_t FilterTracksBySmallTriangulationAngle(
    Reconstruction& reconstruction,
    const std::unordered_set<point3D_t>& ids,
    double min_tri_angle_rad) {
  std::vector<point3D_t> points_to_delete;
  for (const point3D_t point3D_id : ids) {
    if (!reconstruction.ExistsPoint3D(point3D_id)) {
      continue;
    }
    const Point3D& point3D = reconstruction.Point3D(point3D_id);
    const auto& track = point3D.Track().Elements();
    if (track.size() < 2) {
      continue;
    }
    double min_angle = std::numeric_limits<double>::max();
    for (size_t i = 0; i < track.size(); ++i) {
      for (size_t j = i + 1; j < track.size(); ++j) {
        const Eigen::Vector3d center1 = ProjectionCenterOf(
            reconstruction, track[i].image_id);
        const Eigen::Vector3d center2 = ProjectionCenterOf(
            reconstruction, track[j].image_id);
        min_angle = std::min(
            min_angle,
            CalculateTriangulationAngle(center1, center2, point3D.XYZ()));
      }
    }
    if (min_angle < min_tri_angle_rad) {
      points_to_delete.push_back(point3D_id);
    }
  }
  for (const point3D_t point3D_id : points_to_delete) {
    if (reconstruction.ExistsPoint3D(point3D_id)) {
      reconstruction.DeletePoint3D(point3D_id);
    }
  }
  return points_to_delete.size();
}

}  // namespace

RotationEstimatorOptions GlobalMapperOptions::RotationAveraging() const {
  RotationEstimatorOptions opts = rotation_averaging;
  opts.refine_sensor_from_rig = refine_sensor_from_rig;
  if (random_seed >= 0) {
    opts.random_seed = random_seed;
  }
  return opts;
}

GlobalPositionerOptions GlobalMapperOptions::GlobalPositioning() const {
  GlobalPositionerOptions opts = global_positioning;
  opts.refine_sensor_from_rig = refine_sensor_from_rig;
  opts.solver_options.num_threads = num_threads;
  if (random_seed >= 0) {
    opts.random_seed = random_seed;
    opts.use_parameter_block_ordering = false;
  }
  return opts;
}

BundleAdjustmentOptions GlobalMapperOptions::BundleAdjustment() const {
  BundleAdjustmentOptions opts = bundle_adjustment;
  if (opts.backend == BundleAdjustmentBackend::CERES) {
    opts.solver_options.num_threads = num_threads;
    opts.gpu_index = ba_gpu_index;
  } else {
    opts.caspar_gpu_index =
        ba_gpu_index.empty() ? -1 : std::stoi(ba_gpu_index);
  }
  return opts;
}

IncrementalTriangulator::Options GlobalMapperOptions::Retriangulation() const {
  // Fork note: the fork's IncrementalTriangulator::Options has no random_seed
  // field yet; the retriangulation stage inherits the pipeline PRNG state.
  return retriangulation;
}

GlobalMapper::GlobalMapper(
    std::shared_ptr<const DatabaseCache> database_cache)
    : database_cache_(std::move(THROW_CHECK_NOTNULL(database_cache))) {}

void GlobalMapper::BeginReconstruction(
    const std::shared_ptr<class Reconstruction>& reconstruction) {
  THROW_CHECK_NOTNULL(reconstruction);
  reconstruction_ = reconstruction;
  reconstruction_->Load(*database_cache_);
  pose_graph_ = std::make_shared<class PoseGraph>();
  pose_graph_->Load(*database_cache_->CorrespondenceGraph());
}

std::shared_ptr<Reconstruction> GlobalMapper::Reconstruction() const {
  return reconstruction_;
}

bool GlobalMapper::RotationAveraging(const RotationEstimatorOptions& options) {
  THROW_CHECK_NOTNULL(reconstruction_);
  THROW_CHECK_NOTNULL(pose_graph_);

  if (pose_graph_->Empty()) {
    LOG(ERROR) << "Cannot continue with empty pose graph";
    return false;
  }

  // Read pose priors from the database cache.
  const std::vector<PosePrior>& pose_priors = database_cache_->PosePriors();

  // First pass: solve rotation averaging on all frames, then filter outlier
  // pairs by rotation error and de-register frames outside the largest
  // connected component.
  RotationEstimatorOptions custom_options = options;
  custom_options.filter_unregistered = false;
  if (!RunRotationAveraging(
          custom_options, *pose_graph_, *reconstruction_, pose_priors)) {
    return false;
  }

  // Second pass: re-solve on registered frames only to refine rotations
  // after outlier removal.
  custom_options.filter_unregistered = true;
  if (!RunRotationAveraging(
          custom_options, *pose_graph_, *reconstruction_, pose_priors)) {
    return false;
  }

  VLOG(1) << reconstruction_->NumRegImages() << " / "
          << reconstruction_->NumImages()
          << " images are within the connected component.";

  return true;
}

void GlobalMapper::EstablishTracks(const GlobalMapperOptions& options) {
  using Observation = std::pair<image_t, point2D_t>;
  THROW_CHECK_EQ(reconstruction_->NumPoints3D(), 0);

  // Build keypoints map from registered images.
  NodeHashMap<image_t, std::vector<Eigen::Vector2d>> image_id_to_keypoints;
  for (const image_t image_id : reconstruction_->RegImageIds()) {
    const auto& image = reconstruction_->Image(image_id);
    std::vector<Eigen::Vector2d> points;
    points.reserve(image.NumPoints2D());
    for (const auto& point2D : image.Points2D()) {
      points.push_back(point2D.XY());
    }
    image_id_to_keypoints.emplace(image_id, std::move(points));
  }

  auto corr_graph = database_cache_->CorrespondenceGraph();

  // Union all matching observations.
  UnionFind<Observation, PairHash> uf;
  FeatureMatches matches;
  for (const auto& [pair_id, edge] : pose_graph_->ValidEdges()) {
    const auto [image_id1, image_id2] =
        Database::PairIdToImagePair(pair_id);
    THROW_CHECK(image_id_to_keypoints.count(image_id1))
        << "Missing keypoints for image " << image_id1;
    THROW_CHECK(image_id_to_keypoints.count(image_id2))
        << "Missing keypoints for image " << image_id2;
    corr_graph->ExtractMatchesBetweenImages(image_id1, image_id2, matches);
    for (const auto& match : matches) {
      const Observation obs1(image_id1, match.point2D_idx1);
      const Observation obs2(image_id2, match.point2D_idx2);
      if (obs2 < obs1) {
        uf.Union(obs1, obs2);
      } else {
        uf.Union(obs2, obs1);
      }
    }
  }

  // Group observations by their root.
  uf.Compress();
  NodeHashMap<Observation, std::vector<Observation>, PairHash> track_map;
  for (const auto& [obs, root] : uf.Parents()) {
    track_map[root].push_back(obs);
  }
  LOG(INFO) << "Established " << track_map.size() << " tracks from "
            << uf.Parents().size() << " observations";

  // Validate tracks, check consistency, and collect valid ones with lengths.
  NodeHashMap<point3D_t, Point3D> candidate_points3D;
  std::vector<std::pair<size_t, point3D_t>> track_lengths;
  size_t discarded_counter = 0;
  point3D_t next_point3D_id = 0;

  for (const auto& [track_id, observations] : track_map) {
    NodeHashMap<image_t, std::vector<Eigen::Vector2d>> image_id_set;
    Point3D point3D;
    bool is_consistent = true;

    for (const auto& [image_id, feature_id] : observations) {
      const Eigen::Vector2d& xy =
          image_id_to_keypoints.at(image_id).at(feature_id);

      auto it = image_id_set.find(image_id);
      if (it != image_id_set.end()) {
        for (const auto& existing_xy : it->second) {
          const double sq_threshold =
              options.track_intra_image_consistency_threshold *
              options.track_intra_image_consistency_threshold;
          if ((existing_xy - xy).squaredNorm() > sq_threshold) {
            is_consistent = false;
            break;
          }
        }
        if (!is_consistent) {
          ++discarded_counter;
          break;
        }
        it->second.push_back(xy);
      } else {
        image_id_set[image_id].push_back(xy);
      }
      point3D.Track().AddElement(image_id, feature_id);
    }

    if (!is_consistent) continue;

    const size_t num_images = image_id_set.size();
    if (num_images <
        static_cast<size_t>(options.track_min_num_views_per_track)) {
      continue;
    }

    const point3D_t point3D_id = next_point3D_id++;
    track_lengths.emplace_back(point3D.Track().Length(), point3D_id);
    candidate_points3D.emplace(point3D_id, std::move(point3D));
  }

  LOG(INFO) << "Kept " << candidate_points3D.size() << " tracks, discarded "
            << discarded_counter << " due to inconsistency";

  // Sort tracks by length (descending) and select for problem.
  std::sort(track_lengths.begin(), track_lengths.end(), std::greater<>());

  NodeHashMap<image_t, size_t> tracks_per_image;
  size_t images_left = image_id_to_keypoints.size();
  const size_t max_num_tracks =
      static_cast<size_t>(options.keep_max_num_tracks);
  for (const auto& [track_length, point3D_id] : track_lengths) {
    // Stop once the global track budget is exhausted. As tracks are sorted by
    // decreasing length, this keeps the longest tracks and bounds memory
    // usage.
    if (reconstruction_->NumPoints3D() >= max_num_tracks) break;

    auto& point3D = candidate_points3D.at(point3D_id);

    // Check if any image in this track still needs more observations.
    const bool should_add = std::any_of(
        point3D.Track().Elements().begin(),
        point3D.Track().Elements().end(),
        [&](const auto& obs) {
          return tracks_per_image[obs.image_id] <=
                 static_cast<size_t>(options.track_required_tracks_per_view);
        });
    if (!should_add) continue;

    // Update image counts.
    for (const auto& obs : point3D.Track().Elements()) {
      auto& count = tracks_per_image[obs.image_id];
      if (count ==
          static_cast<size_t>(options.track_required_tracks_per_view)) {
        --images_left;
      }
      ++count;
    }

    // Add track after updating counts so we can move. The fork's
    // AddPoint3D assigns a fresh point3D_id; the XYZ coordinates stay zero
    // here and are estimated by the subsequent global positioning stage.
    reconstruction_->AddPoint3D(Eigen::Vector3d::Zero(), point3D.Track());

    if (images_left == 0) break;
  }

  LOG(INFO) << "Before filtering: " << candidate_points3D.size()
            << ", after filtering: " << reconstruction_->NumPoints3D();
}

bool GlobalMapper::GlobalPositioning(
    const GlobalPositionerOptions& options,
    double max_angular_reproj_error_deg,
    double max_normalized_reproj_error,
    double min_tri_angle_deg) {
  if (!RunGlobalPositioning(options, *pose_graph_, *reconstruction_)) {
    return false;
  }

  std::unordered_set<point3D_t> point3D_ids = reconstruction_->Point3DIds();

  // Filter tracks based on the estimation. The fork reads the per-image pose
  // through ProjectionMatrix() (frame-aware) and the angular error through
  // CalculateAngularError; upstream routes the same checks through the
  // ObservationManager with ReprojectionErrorType::ANGULAR.
  const double max_angular_error_rad = DegToRad(max_angular_reproj_error_deg);
  FilterTracksByAngularError(*reconstruction_,
                             point3D_ids,
                             2.0 * max_angular_error_rad);
  FilterTracksByAngularError(*reconstruction_,
                             point3D_ids,
                             max_angular_error_rad);

  // Filter tracks based on triangulation angle and reprojection error.
  FilterTracksBySmallTriangulationAngle(*reconstruction_,
                                        point3D_ids,
                                        DegToRad(min_tri_angle_deg));
  FilterTracksByNormalizedError(*reconstruction_,
                                point3D_ids,
                                10 * max_normalized_reproj_error);

  // Normalize the structure for numerical stability.
  reconstruction_->Normalize();

  return true;
}

bool GlobalMapper::IterativeBundleAdjustment(
    const BundleAdjustmentOptions& options,
    double max_normalized_reproj_error,
    double min_tri_angle_deg,
    int num_iterations,
    bool skip_fixed_rotation_stage,
    bool skip_joint_optimization_stage,
    const std::function<bool()>& on_progress) {
  // Fork note: the fixed-rotation stage requires the per-frame
  // constant_rig_from_world_rotation bundle-adjustment option that lands
  // with W3-2b step 4; until then the stage is silently skipped and the
  // joint optimization runs instead.
  for (int ite = 0; ite < num_iterations; ite++) {
    // Joint optimization stage: default BA
    if (!skip_joint_optimization_stage) {
      if (!RunBundleAdjustment(options, *reconstruction_)) {
        return false;
      }
    }
    LOG(INFO) << "Global bundle adjustment iteration " << ite + 1 << " / "
              << num_iterations << " finished";

    // Normalize the structure for numerical stability.
    reconstruction_->Normalize();

    // Report progress for this refinement iteration and stop early if
    // requested. The filter passes above leave point3D.error in normalized
    // units, so recompute it in pixels first to keep intermediate
    // visualizations consistent with the final reconstruction.
    if (on_progress) {
      reconstruction_->UpdatePoint3DErrors();
      if (on_progress()) {
        break;
      }
    }

    // Filter tracks based on the estimation. In each round, the criteria for
    // outliers is tightened. If only few tracks are changed, no need to start
    // bundle adjustment right away. Instead, use a more strict criteria to
    // filter.
    LOG(INFO) << "Filtering tracks by reprojection ...";

    bool status = true;
    size_t filtered_num = 0;
    while (status && ite < num_iterations) {
      double scaling = std::max(3 - ite, 1);
      filtered_num += FilterTracksByNormalizedError(
          *reconstruction_,
          reconstruction_->Point3DIds(),
          scaling * max_normalized_reproj_error);

      if (filtered_num > 1e-3 * reconstruction_->NumPoints3D()) {
        status = false;
      } else {
        ite++;
      }
    }
    if (status) {
      LOG(INFO) << "fewer than 0.1% tracks are filtered, stop the iteration.";
      break;
    }
  }

  // Filter tracks based on the estimation.
  FilterTracksByNormalizedError(*reconstruction_,
                                reconstruction_->Point3DIds(),
                                max_normalized_reproj_error);

  return true;
}

bool GlobalMapper::IterativeRetriangulateAndRefine(
    const IncrementalTriangulator::Options& options,
    const BundleAdjustmentOptions& ba_options,
    double max_normalized_reproj_error,
    double min_tri_angle_deg) {
  // Delete all existing 3D points and re-establish 2D-3D correspondences.
  reconstruction_->DeleteAllPoints2DAndPoints3D();

  // Initialize mapper.
  IncrementalMapper mapper(database_cache_.get());
  mapper.BeginReconstruction(reconstruction_.get());

  // Triangulate all registered images.
  for (const image_t image_id : reconstruction_->RegImageIds()) {
    mapper.TriangulateImage(options, image_id);
  }

  // Fork note: the upstream iterative global refinement loop
  // (IncrementalMapper::IterativeGlobalRefinement, an ObservationManager
  // consumer) is not ported until W3-2b step 3; the equivalent refinement is
  // a bounded filter-and-bundle-adjust loop over the fork's Reconstruction
  // filters with a pixel-based reprojection threshold.
  BundleAdjustmentOptions custom_ba_options = ba_options;
  custom_ba_options.print_summary = false;
  custom_ba_options.solver_options.max_num_iterations = 50;
  custom_ba_options.solver_options.max_linear_solver_iterations = 100;

  for (int ite = 0; ite < 5; ++ite) {
    FilterTracksByNormalizedError(*reconstruction_,
                                  reconstruction_->Point3DIds(),
                                  max_normalized_reproj_error);
    if (!RunBundleAdjustment(custom_ba_options, *reconstruction_)) {
      return false;
    }
    reconstruction_->Normalize();
  }

  mapper.EndReconstruction(/*discard=*/false);

  // Final filtering and bundle adjustment.
  FilterTracksByNormalizedError(*reconstruction_,
                                reconstruction_->Point3DIds(),
                                max_normalized_reproj_error);

  if (!RunBundleAdjustment(ba_options, *reconstruction_)) {
    return false;
  }

  // Normalize the structure for numerical stability.
  reconstruction_->Normalize();

  FilterTracksByNormalizedError(*reconstruction_,
                                reconstruction_->Point3DIds(),
                                max_normalized_reproj_error);
  FilterTracksBySmallTriangulationAngle(*reconstruction_,
                                        reconstruction_->Point3DIds(),
                                        DegToRad(min_tri_angle_deg));

  return true;
}

bool GlobalMapper::Solve(const GlobalMapperOptions& options,
                         const std::function<bool()>& on_progress) {
  THROW_CHECK_NOTNULL(reconstruction_);
  THROW_CHECK_NOTNULL(pose_graph_);

  if (pose_graph_->Empty()) {
    LOG(ERROR) << "Cannot continue with empty pose graph";
    return false;
  }

  // Reports the current reconstruction and returns whether a stop was
  // requested. Point errors are recomputed in pixels before reporting because
  // the preceding filter passes leave point3D.error in normalized units,
  // which would otherwise make intermediate visualizations inconsistent with
  // the final reconstruction.
  const auto report_and_check_stop = [&]() {
    if (!on_progress) {
      return false;
    }
    reconstruction_->UpdatePoint3DErrors();
    return on_progress();
  };

  // Run rotation averaging
  if (!options.skip_rotation_averaging) {
    PrintHeading1("Running rotation averaging");
    Timer run_timer;
    run_timer.Start();
    if (!RotationAveraging(options.RotationAveraging())) {
      return false;
    }
    LOG(INFO) << "Rotation averaging done in " << run_timer.ElapsedSeconds()
              << " seconds";
  }

  // Track establishment and selection
  if (!options.skip_track_establishment) {
    PrintHeading1("Running track establishment");
    Timer run_timer;
    run_timer.Start();
    EstablishTracks(options);
    LOG(INFO) << "Track establishment done in " << run_timer.ElapsedSeconds()
              << " seconds";
  }

  // Global positioning
  if (!options.skip_global_positioning) {
    PrintHeading1("Running global positioning");
    Timer run_timer;
    run_timer.Start();
    if (!GlobalPositioning(options.GlobalPositioning(),
                           options.max_angular_reproj_error_deg,
                           options.max_normalized_reproj_error,
                           options.min_tri_angle_deg)) {
      return false;
    }
    LOG(INFO) << "Global positioning done in " << run_timer.ElapsedSeconds()
              << " seconds";

    // Report the first 3D view after global positioning and stop early if
    // requested.
    if (report_and_check_stop()) {
      return true;
    }
  }

  // Bundle adjustment
  if (!options.skip_bundle_adjustment) {
    PrintHeading1("Running iterative bundle adjustment");
    Timer run_timer;
    run_timer.Start();
    if (!IterativeBundleAdjustment(options.BundleAdjustment(),
                                   options.max_normalized_reproj_error,
                                   options.min_tri_angle_deg,
                                   options.ba_num_iterations,
                                   options.ba_skip_fixed_rotation_stage,
                                   options.ba_skip_joint_optimization_stage,
                                   on_progress)) {
      return false;
    }
    LOG(INFO) << "Iterative bundle adjustment done in "
              << run_timer.ElapsedSeconds() << " seconds";
  }

  // Retriangulation
  if (!options.skip_retriangulation) {
    PrintHeading1("Running iterative retriangulation and refinement");
    Timer run_timer;
    run_timer.Start();
    if (!IterativeRetriangulateAndRefine(options.Retriangulation(),
                                         options.BundleAdjustment(),
                                         options.max_normalized_reproj_error,
                                         options.min_tri_angle_deg)) {
      return false;
    }
    LOG(INFO) << "Iterative retriangulation and refinement done in "
              << run_timer.ElapsedSeconds() << " seconds";

    // Report the result after retriangulation and stop early if requested.
    if (report_and_check_stop()) {
      return true;
    }
  }

  // Filter passes here use NORMALIZED/ANGULAR error, so point3D.error is
  // left in non-pixel units. Recompute in pixels for consistent reporting
  // in model_analyzer.
  reconstruction_->UpdatePoint3DErrors();

  return true;
}

}  // namespace colmap
