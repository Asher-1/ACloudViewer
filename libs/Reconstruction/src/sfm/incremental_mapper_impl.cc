// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
#include "util/hash_containers.h"
#include "sfm/incremental_mapper_impl.h"

#include "estimators/generalized_pose.h"
#include "estimators/two_view_geometry.h"
#include "geometry/rigid3.h"
#include "geometry/triangulation.h"
#include "math/math.h"
#include "scene/correspondence_graph.h"
#include "scene/rig.h"
#include "util/logging.h"
#include "util/misc.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <optional>
#include <utility>

namespace colmap {

namespace {

// Upstream parity (d3ccaf35 sfm/incremental_mapper_impl.cc): compact
// meta-data for ranking initial-pair seed images.
struct InitImageInfo {
    image_t image_id;
    bool prior_focal_length;
    size_t num_correspondences;
};

inline bool CompareImageInfo(const InitImageInfo& a, const InitImageInfo& b) {
    if (a.prior_focal_length != b.prior_focal_length) {
        return a.prior_focal_length > b.prior_focal_length;
    }
    return a.num_correspondences > b.num_correspondences;
}

std::vector<image_t> ExtractSortedImageIds(
        std::vector<InitImageInfo>& image_infos) {
    std::sort(image_infos.begin(), image_infos.end(), CompareImageInfo);
    std::vector<image_t> image_ids(image_infos.size());
    for (size_t i = 0; i < image_infos.size(); ++i) {
        image_ids[i] = image_infos[i].image_id;
    }
    return image_ids;
}

float RankNextImageMaxVisiblePointsNum(
        const image_t image_id, const ObservationManager& obs_manager) {
    return static_cast<float>(obs_manager.NumVisiblePoints3D(image_id));
}

float RankNextImageMaxVisiblePointsRatio(
        const image_t image_id, const ObservationManager& obs_manager) {
    return static_cast<float>(obs_manager.NumVisiblePoints3D(image_id)) /
           static_cast<float>(obs_manager.NumObservations(image_id));
}

float RankNextImageMinUncertainty(const image_t image_id,
                                  const ObservationManager& obs_manager) {
    return static_cast<float>(obs_manager.Point3DVisibilityScore(image_id));
}

// Upstream parity (d3ccaf35 sfm/incremental_mapper_impl.cc): generalized
// relative pose initialization for non-trivial rigs. Only reached after the
// plain two-view geometry between the seed images has been verified stable.
bool EstimateInitialGeneralizedTwoViewGeometry(
        const IncrementalMapper::Options& options,
        const DatabaseCache& database_cache,
        const Image& orig_image1,
        const Image& orig_image2,
        const Frame& frame1,
        const Frame& frame2,
        const Rig& rig1,
        const Rig& rig2,
        Rigid3d& orig_cam2_from_orig_cam1) {
    std::vector<Eigen::Vector2d> points2D1;
    std::vector<Eigen::Vector2d> points2D2;
    std::vector<size_t> camera_idxs1;
    std::vector<size_t> camera_idxs2;
    std::vector<Rigid3d> cams_from_rig;
    std::vector<Camera> cameras;

    NodeHashMap<camera_t, size_t> camera_id_to_idx;
    auto maybe_add_camera = [&cameras, &cams_from_rig, &camera_id_to_idx](
                                    const Rig& rig, const Camera& camera) {
        const auto [it, inserted] =
                camera_id_to_idx.emplace(camera.CameraId(), cameras.size());
        if (inserted) {
            cameras.push_back(camera);
            if (rig.IsRefSensor(
                        sensor_t(SensorType::CAMERA, camera.CameraId()))) {
                cams_from_rig.push_back(Rigid3d());
            } else {
                cams_from_rig.push_back(rig.SensorFromRig(sensor_t(
                        SensorType::CAMERA, camera.CameraId())));
            }
        }
        return it->second;
    };

    FeatureMatches matches;
    for (const image_t image_id1 : frame1.ImageIds()) {
        const Image& image1 = database_cache.Image(image_id1);
        const Camera& camera1 = database_cache.Camera(image1.CameraId());
        const size_t camera_idx1 = maybe_add_camera(rig1, camera1);

        for (const image_t image_id2 : frame2.ImageIds()) {
            const Image& image2 = database_cache.Image(image_id2);
            const Camera& camera2 = database_cache.Camera(image2.CameraId());
            const size_t camera_idx2 = maybe_add_camera(rig2, camera2);

            database_cache.CorrespondenceGraph()->ExtractMatchesBetweenImages(
                    image_id1, image_id2, matches);
            for (const auto& match : matches) {
                points2D1.push_back(
                        image1.Point2D(match.point2D_idx1).XY());
                points2D2.push_back(
                        image2.Point2D(match.point2D_idx2).XY());
                camera_idxs1.push_back(camera_idx1);
                camera_idxs2.push_back(camera_idx2);
            }
        }
    }

    RANSACOptions ransac_options;
    ransac_options.min_num_trials = 30;
    ransac_options.random_seed = options.random_seed;
    ransac_options.max_error = options.init_max_error;

    std::optional<Rigid3d> maybe_rig2_from_rig1;
    std::optional<Rigid3d> maybe_pano2_from_pano1;
    size_t num_inliers;
    std::vector<char> inlier_mask;
    if (!EstimateGeneralizedRelativePose(ransac_options,
                                         points2D1,
                                         points2D2,
                                         camera_idxs1,
                                         camera_idxs2,
                                         cams_from_rig,
                                         cameras,
                                         &maybe_rig2_from_rig1,
                                         &maybe_pano2_from_pano1,
                                         &num_inliers,
                                         &inlier_mask)) {
        return false;
    }

    VLOG(3) << "Initial general frame pair with " << num_inliers
            << " inlier matches";

    // Note that we already checked for stable geometry (i.e., non-forward
    // motion, sufficient triangulation angle) between the original image pair.
    if (static_cast<int>(num_inliers) < options.init_min_num_inliers) {
        return false;
    }

    const Rigid3d rig2_from_rig1 = maybe_rig2_from_rig1.has_value()
                                           ? maybe_rig2_from_rig1.value()
                                           : maybe_pano2_from_pano1.value();

    // Recompose the relative transformation between the original images.
    const sensor_t orig_camera_id1(SensorType::CAMERA,
                                   orig_image1.CameraId());
    Rigid3d orig_cam1_from_rig1;
    if (!rig1.IsRefSensor(orig_camera_id1)) {
        orig_cam1_from_rig1 = rig1.SensorFromRig(orig_camera_id1);
    }

    const sensor_t orig_camera_id2(SensorType::CAMERA,
                                   orig_image2.CameraId());
    Rigid3d orig_cam2_from_rig2;
    if (!rig2.IsRefSensor(orig_camera_id2)) {
        orig_cam2_from_rig2 = rig2.SensorFromRig(orig_camera_id2);
    }

    orig_cam2_from_orig_cam1 =
            orig_cam2_from_rig2 * rig2_from_rig1 * Inverse(orig_cam1_from_rig1);

    return true;
}

void SortAndAppendNextImages(std::vector<std::pair<image_t, float>> image_ranks,
                             std::vector<image_t>* sorted_images_ids) {
    std::sort(image_ranks.begin(), image_ranks.end(),
              [](const std::pair<image_t, float>& image1,
                 const std::pair<image_t, float>& image2) {
                  return image1.second > image2.second;
              });

    sorted_images_ids->reserve(sorted_images_ids->size() + image_ranks.size());
    for (const auto& image : image_ranks) {
        sorted_images_ids->push_back(image.first);
    }

    image_ranks.clear();
}

}  // namespace

std::vector<image_t> IncrementalMapperImpl::FindFirstInitialImage(
        const IncrementalMapper::Options& options,
        const CorrespondenceGraph& correspondence_graph,
        const Reconstruction& reconstruction,
        const NodeHashMap<image_t, size_t>& init_num_reg_trials,
        const NodeHashMap<image_t, size_t>& num_registrations) {
    // Struct to hold meta-data for ranking images.
    struct ImageInfo {
        image_t image_id;
        bool prior_focal_length;
        image_t num_correspondences;
    };

    const size_t init_max_reg_trials =
            static_cast<size_t>(options.init_max_reg_trials);

    // Collect information of all not yet registered images with
    // correspondences.
    std::vector<InitImageInfo> image_infos;
    image_infos.reserve(reconstruction.NumImages());
    for (const auto& image : reconstruction.Images()) {
        // Only images with correspondences can be registered. Upstream
        // d3ccaf35 derives this from the correspondence graph; the fork's
        // legacy per-image counter is not maintained on a fresh Load.
        const size_t num_correspondences =
                correspondence_graph.NumCorrespondencesForImage(image.first);
        if (num_correspondences == 0) {
            continue;
        }

        // Only use images for initialization a maximum number of times.
        if (init_num_reg_trials.count(image.first) &&
            init_num_reg_trials.at(image.first) >= init_max_reg_trials) {
            continue;
        }

        // Only use images for initialization that are not registered in any
        // of the other reconstructions.
        if (num_registrations.count(image.first) > 0 &&
            num_registrations.at(image.first) > 0) {
            continue;
        }

        const class Camera& camera =
                reconstruction.Camera(image.second.CameraId());
        image_infos.push_back({image.first,
                               camera.HasPriorFocalLength(),
                               num_correspondences});
    }

    return ExtractSortedImageIds(image_infos);
}

std::vector<image_t> IncrementalMapperImpl::FindSecondInitialImage(
        const IncrementalMapper::Options& options,
        const image_t image_id1,
        const CorrespondenceGraph& correspondence_graph,
        const Reconstruction& reconstruction,
        const NodeHashMap<image_t, size_t>& num_registrations) {
    // Collect images that are connected to the first seed image and have
    // not been registered before in other reconstructions.
    const class Image& image1 = reconstruction.Image(image_id1);
    NodeHashMap<image_t, point2D_t> num_correspondences;
    for (point2D_t point2D_idx = 0; point2D_idx < image1.NumPoints2D();
         ++point2D_idx) {
        const auto corr_range =
                correspondence_graph.FindCorrespondences(image_id1,
                                                         point2D_idx);
        for (const CorrespondenceGraph::Correspondence* corr = corr_range.beg;
             corr < corr_range.end; ++corr) {
            if (num_registrations.count(corr->image_id) == 0 ||
                num_registrations.at(corr->image_id) == 0) {
                num_correspondences[corr->image_id] += 1;
            }
        }
    }

    const size_t init_min_num_inliers =
            static_cast<size_t>(options.init_min_num_inliers);

    // Compose image information in a compact form for sorting.
    std::vector<InitImageInfo> image_infos;
    image_infos.reserve(num_correspondences.size());
    for (const auto elem : num_correspondences) {
        if (elem.second >= init_min_num_inliers) {
            const class Image& image = reconstruction.Image(elem.first);
            const class Camera& camera =
                    reconstruction.Camera(image.CameraId());
            image_infos.push_back({elem.first,
                                   camera.HasPriorFocalLength(),
                                   static_cast<size_t>(elem.second)});
        }
    }

    return ExtractSortedImageIds(image_infos);
}

std::vector<image_t> IncrementalMapperImpl::FindNextImages(
        const IncrementalMapper::Options& options,
        const ObservationManager& obs_manager,
        const NodeHashSet<frame_t>& filtered_frames,
        NodeHashMap<image_t, size_t>& num_reg_trials,
        bool structure_less) {
    CHECK(options.Check());
    const Reconstruction& reconstruction = obs_manager.Reconstruction();

    std::function<float(image_t, const ObservationManager&)> rank_image_func;
    if (structure_less) {
        rank_image_func = [](image_t image_id,
                             const ObservationManager& obs_manager) {
            return static_cast<float>(
                    obs_manager.NumVisibleCorrespondences(image_id));
        };
    } else {
        switch (options.image_selection_method) {
            case IncrementalMapper::Options::ImageSelectionMethod::
                    MAX_VISIBLE_POINTS_NUM:
                rank_image_func = RankNextImageMaxVisiblePointsNum;
                break;
            case IncrementalMapper::Options::ImageSelectionMethod::
                    MAX_VISIBLE_POINTS_RATIO:
                rank_image_func = RankNextImageMaxVisiblePointsRatio;
                break;
            case IncrementalMapper::Options::ImageSelectionMethod::
                    MIN_UNCERTAINTY:
                rank_image_func = RankNextImageMinUncertainty;
                break;
        }
    }

    std::vector<std::pair<image_t, float>> image_ranks;
    std::vector<std::pair<image_t, float>> other_image_ranks;

    // Append images that have not failed to register before.
    for (const auto& image : reconstruction.Images()) {
        // Skip images that are already registered.
        if (image.second.HasPose()) {
            continue;
        }

        // Only consider images with a sufficient number of visible points.
        if (obs_manager.NumVisiblePoints3D(image.first) <
            static_cast<size_t>(options.abs_pose_min_num_inliers)) {
            continue;
        }

        // Only try registration for a certain maximum number of times.
        // Upstream parity (d3ccaf35): the lookup inserts a zero entry for
        // images never attempted before, matching the member-side mutable
        // reference contract.
        const size_t image_num_reg_trials = num_reg_trials[image.first];
        if (image_num_reg_trials >=
            static_cast<size_t>(options.max_reg_trials)) {
            continue;
        }

        // If the image's frame has been filtered, or the image failed to
        // register before, place it in the second bucket and prefer images
        // that have not been tried before.
        const float rank = rank_image_func(image.first, obs_manager);
        const frame_t frame_id = image.second.FrameId();
        if (filtered_frames.count(frame_id) == 0 && image_num_reg_trials == 0) {
            image_ranks.emplace_back(image.first, rank);
        } else {
            other_image_ranks.emplace_back(image.first, rank);
        }
    }

    std::vector<image_t> ranked_images_ids;
    SortAndAppendNextImages(std::move(image_ranks), &ranked_images_ids);
    SortAndAppendNextImages(std::move(other_image_ranks), &ranked_images_ids);

    return ranked_images_ids;
}

std::vector<image_t> IncrementalMapperImpl::FindLocalBundle(
        const IncrementalMapper::Options& options,
        const image_t image_id,
        const Reconstruction& reconstruction) {
    CHECK(options.Check());

    const Image& image = reconstruction.Image(image_id);
    // Upstream parity (d3ccaf35): THROW_CHECK on the frame pose.
    THROW_CHECK(image.HasPose());

    // Extract all images that have at least one 3D point with the query image
    // in common, and simultaneously count the number of common 3D points.

    NodeHashMap<image_t, size_t> shared_observations;

    NodeHashSet<point3D_t> point3D_ids;
    point3D_ids.reserve(image.NumPoints3D());

    for (const Point2D& point2D : image.Points2D()) {
        if (point2D.HasPoint3D()) {
            point3D_ids.insert(point2D.Point3DId());
            const Point3D& point3D =
                    reconstruction.Point3D(point2D.Point3DId());
            for (const TrackElement& track_el : point3D.Track().Elements()) {
                if (track_el.image_id != image_id) {
                    shared_observations[track_el.image_id] += 1;
                }
            }
        }
    }

    // Sort overlapping images according to number of shared observations.

    std::vector<std::pair<image_t, size_t>> overlapping_images(
            shared_observations.begin(), shared_observations.end());
    std::sort(overlapping_images.begin(), overlapping_images.end(),
              [](const std::pair<image_t, size_t>& image1,
                 const std::pair<image_t, size_t>& image2) {
                  return image1.second > image2.second;
              });

    // The local bundle is composed of the given image and its most connected
    // neighbor images, hence the subtraction of 1.

    const size_t num_images =
            static_cast<size_t>(options.ba_local_num_images - 1);
    const size_t num_eff_images =
            std::min(num_images, overlapping_images.size());

    // Extract most connected images and ensure sufficient triangulation angle.

    std::vector<image_t> local_bundle_image_ids;
    local_bundle_image_ids.reserve(num_eff_images);

    // If the number of overlapping images equals the number of desired images
    // in the local bundle, then simply copy over the image identifiers.
    if (overlapping_images.size() == num_eff_images) {
        for (const auto& overlapping_image : overlapping_images) {
            local_bundle_image_ids.push_back(overlapping_image.first);
        }
        return local_bundle_image_ids;
    }

    // In the following iteration, we start with the most overlapping images and
    // check whether it has sufficient triangulation angle. If none of the
    // overlapping images has sufficient triangulation angle, we relax the
    // triangulation angle threshold and start from the most overlapping image
    // again. In the end, if we still haven't found enough images, we simply use
    // the most overlapping images.

    const double min_tri_angle_rad = DegToRad(options.ba_local_min_tri_angle);

    // The selection thresholds (minimum triangulation angle, minimum number of
    // shared observations), which are successively relaxed.
    const std::array<std::pair<double, double>, 8> selection_thresholds = {{
            std::make_pair(min_tri_angle_rad / 1.0, 0.6 * image.NumPoints3D()),
            std::make_pair(min_tri_angle_rad / 1.5, 0.6 * image.NumPoints3D()),
            std::make_pair(min_tri_angle_rad / 2.0, 0.5 * image.NumPoints3D()),
            std::make_pair(min_tri_angle_rad / 2.5, 0.4 * image.NumPoints3D()),
            std::make_pair(min_tri_angle_rad / 3.0, 0.3 * image.NumPoints3D()),
            std::make_pair(min_tri_angle_rad / 4.0, 0.2 * image.NumPoints3D()),
            std::make_pair(min_tri_angle_rad / 5.0, 0.1 * image.NumPoints3D()),
            std::make_pair(min_tri_angle_rad / 6.0, 0.1 * image.NumPoints3D()),
    }};

    const Eigen::Vector3d proj_center = image.ProjectionCenter();
    std::vector<Eigen::Vector3d> shared_points3D;
    shared_points3D.reserve(image.NumPoints3D());
    std::vector<double> tri_angles(overlapping_images.size(), -1.0);
    std::vector<char> used_overlapping_images(overlapping_images.size(), false);

    for (const auto& selection_threshold : selection_thresholds) {
        for (size_t overlapping_image_idx = 0;
             overlapping_image_idx < overlapping_images.size();
             ++overlapping_image_idx) {
            // Check if the image has sufficient overlap. Since the images are
            // ordered based on the overlap, we can just skip the remaining
            // ones.
            if (overlapping_images[overlapping_image_idx].second <
                selection_threshold.second) {
                break;
            }

            // Check if the image is already in the local bundle.
            if (used_overlapping_images[overlapping_image_idx]) {
                continue;
            }

            const auto& overlapping_image = reconstruction.Image(
                    overlapping_images[overlapping_image_idx].first);
            const Eigen::Vector3d overlapping_proj_center =
                    overlapping_image.ProjectionCenter();

            // In the first iteration, compute the triangulation angle. In later
            // iterations, reuse the previously computed value.
            double& tri_angle = tri_angles[overlapping_image_idx];
            if (tri_angle < 0.0) {
                // Collect the commonly observed 3D points.
                shared_points3D.clear();
                for (const Point2D& point2D : image.Points2D()) {
                    if (point2D.HasPoint3D() &&
                        point3D_ids.count(point2D.Point3DId())) {
                        shared_points3D.push_back(
                                reconstruction.Point3D(point2D.Point3DId())
                                        .XYZ());
                    }
                }

                // Calculate the triangulation angle at a certain percentile.
                const double kTriangulationAnglePercentile = 75;
                tri_angle =
                        Percentile(CalculateTriangulationAngles(
                                           proj_center, overlapping_proj_center,
                                           shared_points3D),
                                   kTriangulationAnglePercentile);
            }

            // Check that the image has sufficient triangulation angle.
            if (tri_angle >= selection_threshold.first) {
                local_bundle_image_ids.push_back(overlapping_image.ImageId());
                used_overlapping_images[overlapping_image_idx] = true;
                // Check if we already collected enough images.
                if (local_bundle_image_ids.size() >= num_eff_images) {
                    break;
                }
            }
        }

        // Check if we already collected enough images.
        if (local_bundle_image_ids.size() >= num_eff_images) {
            break;
        }
    }

    // In case there are not enough images with sufficient triangulation angle,
    // simply fill up the rest with the most overlapping images.

    if (local_bundle_image_ids.size() < num_eff_images) {
        for (size_t overlapping_image_idx = 0;
             overlapping_image_idx < overlapping_images.size();
             ++overlapping_image_idx) {
            // Collect image if it is not yet in the local bundle.
            if (!used_overlapping_images[overlapping_image_idx]) {
                local_bundle_image_ids.push_back(
                        overlapping_images[overlapping_image_idx].first);
                used_overlapping_images[overlapping_image_idx] = true;

                // Check if we already collected enough images.
                if (local_bundle_image_ids.size() >= num_eff_images) {
                    break;
                }
            }
        }
    }

    return local_bundle_image_ids;
}

std::optional<IncrementalMapperImpl::InitInfo>
IncrementalMapperImpl::FindInitialImagePair(
        const IncrementalMapper::Options& options,
        const DatabaseCache& database_cache,
        const Reconstruction& reconstruction,
        const NodeHashMap<image_t, size_t>& init_num_reg_trials,
        const NodeHashMap<image_t, size_t>& num_registrations,
        NodeHashSet<image_pair_t>& init_image_pairs,
        image_t image_id1,
        image_t image_id2) {
    CHECK(options.Check());

    const CorrespondenceGraph& correspondence_graph =
            *database_cache.CorrespondenceGraph();

    std::vector<image_t> image_ids1;
    if (image_id1 != kInvalidImageId && image_id2 == kInvalidImageId) {
        // Only image_id1 provided.
        if (!database_cache.ExistsImage(image_id1)) {
            return std::nullopt;
        }
        image_ids1.push_back(image_id1);
    } else if (image_id1 == kInvalidImageId && image_id2 != kInvalidImageId) {
        // Only image_id2 provided.
        if (!database_cache.ExistsImage(image_id2)) {
            return std::nullopt;
        }
        image_ids1.push_back(image_id2);
    } else {
        // No initial seed image provided.
        image_ids1 =
                IncrementalMapperImpl::FindFirstInitialImage(
                        options,
                        correspondence_graph,
                        reconstruction,
                        init_num_reg_trials,
                        num_registrations);
    }

    // Try to find good initial pair. The fork keeps the scan serial (A/A
    // reproducibility; see the header comment on the fork decision).
    for (const image_t seed_id1 : image_ids1) {
        const std::vector<image_t> image_ids2 =
                IncrementalMapperImpl::FindSecondInitialImage(
                        options,
                        seed_id1,
                        correspondence_graph,
                        reconstruction,
                        num_registrations);

        for (const image_t seed_id2 : image_ids2) {
            const image_pair_t pair_id =
                    ImagePairToPairId(seed_id1, seed_id2);

            // Try every pair only once.
            if (!init_image_pairs.emplace(pair_id).second) {
                continue;
            }

            std::optional<InitInfo> pair_init_info =
                    IncrementalMapperImpl::EstimateInitialTwoViewGeometry(
                            options, database_cache, seed_id1, seed_id2);
            if (pair_init_info.has_value()) {
                return pair_init_info;
            }
        }
    }

    // No suitable pair found in entire dataset.
    return std::nullopt;
}

std::optional<IncrementalMapperImpl::InitInfo>
IncrementalMapperImpl::EstimateInitialTwoViewGeometry(
        const IncrementalMapper::Options& options,
        const DatabaseCache& database_cache,
        const image_t image_id1,
        const image_t image_id2) {
    const Image& image1 = database_cache.Image(image_id1);
    const Camera& camera1 = database_cache.Camera(image1.CameraId());

    const Image& image2 = database_cache.Image(image_id2);
    const Camera& camera2 = database_cache.Camera(image2.CameraId());

    const CorrespondenceGraph& correspondence_graph =
            *database_cache.CorrespondenceGraph();
    FeatureMatches matches;
    correspondence_graph.ExtractMatchesBetweenImages(image_id1, image_id2,
                                                     matches);

    std::vector<Eigen::Vector2d> points1;
    points1.reserve(image1.NumPoints2D());
    for (const auto& point : image1.Points2D()) {
        points1.push_back(point.XY());
    }

    std::vector<Eigen::Vector2d> points2;
    points2.reserve(image2.NumPoints2D());
    for (const auto& point : image2.Points2D()) {
        points2.push_back(point.XY());
    }

    TwoViewGeometryOptions two_view_geometry_options;
    two_view_geometry_options.ransac_options.min_num_trials = 30;
    two_view_geometry_options.ransac_options.max_error = options.init_max_error;
    // Upstream parity (d3ccaf35): the initial TVG RANSAC consumes the mapper
    // seed; negative values fall back to the ambient PRNG stream (default
    // behaviour unchanged).
    two_view_geometry_options.ransac_options.random_seed = options.random_seed;
    // The upstream estimator folds the relative-pose decomposition into the
    // estimation, so cam2_from_cam1 carries what the legacy member call
    // EstimateRelativePose produced.
    TwoViewGeometry two_view_geometry = EstimateTwoViewGeometry(
            camera1, points1, camera2, points2, matches,
            two_view_geometry_options);

    // Upstream parity (d3ccaf35 sfm/incremental_mapper_impl.cc): the plain
    // estimation only classifies the configuration; the relative pose and
    // the triangulation angle are recovered by a dedicated decomposition
    // pass that must run before the initialization checks.
    if (!EstimateTwoViewGeometryPose(camera1, points1, camera2, points2,
                                     &two_view_geometry)) {
        return std::nullopt;
    }

    if (!two_view_geometry.cam2_from_cam1) {
        return std::nullopt;
    }

    VLOG(3) << "Initial image pair with config "
            << two_view_geometry.config << ", "
            << two_view_geometry.inlier_matches.size() << " inlier matches, "
            << two_view_geometry.cam2_from_cam1->translation().z()
            << " z translation, "
            << RadToDeg(two_view_geometry.tri_angle)
            << " deg triangulation angle";
    if (static_cast<int>(two_view_geometry.inlier_matches.size()) <
                options.init_min_num_inliers ||
        std::abs(two_view_geometry.cam2_from_cam1->translation().z()) >=
                options.init_max_forward_motion ||
        two_view_geometry.tri_angle <= DegToRad(options.init_min_tri_angle)) {
        return std::nullopt;
    }

    InitInfo info;
    info.image_id1 = image_id1;
    info.image_id2 = image_id2;

    // If one or both of the frames are non-trivial, initialize using the
    // generalized relative pose solver. Note that we intentionally do this
    // after ensuring that the given image pair has stable two-view geometry
    // (upstream d3ccaf35; the generalized solver does not recover
    // intrinsics).
    const Frame& frame1 = database_cache.Frame(image1.FrameId());
    const Frame& frame2 = database_cache.Frame(image2.FrameId());
    const Rig& rig1 = database_cache.Rig(frame1.RigId());
    const Rig& rig2 = database_cache.Rig(frame2.RigId());
    if (rig1.NumSensors() > 1 || rig2.NumSensors() > 1) {
        if (!EstimateInitialGeneralizedTwoViewGeometry(
                    options, database_cache, image1, image2, frame1, frame2,
                    rig1, rig2, info.cam2_from_cam1)) {
            return std::nullopt;
        }
        return info;
    }

    info.cam2_from_cam1 = *two_view_geometry.cam2_from_cam1;
    // Surface the intrinsics estimated by the two-view solver (if any) so the
    // caller can seed the cameras before registering the initial pair.
    info.camera1 = two_view_geometry.camera1;
    info.camera2 = two_view_geometry.camera2;
    return info;
}

}  // namespace colmap
