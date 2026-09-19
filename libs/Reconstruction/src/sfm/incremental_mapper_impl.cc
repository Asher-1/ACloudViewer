// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
#include "sfm/incremental_mapper_impl.h"

#include "geometry/triangulation.h"
#include "scene/correspondence_graph.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <unordered_map>
#include <unordered_set>

namespace colmap {

namespace {

float RankNextImageMaxVisiblePointsNum(const Image& image) {
    return static_cast<float>(image.NumVisiblePoints3D());
}

float RankNextImageMaxVisiblePointsRatio(const Image& image) {
    return static_cast<float>(image.NumVisiblePoints3D()) /
           static_cast<float>(image.NumObservations());
}

float RankNextImageMinUncertainty(const Image& image) {
    return static_cast<float>(image.Point3DVisibilityScore());
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
        const std::unordered_map<image_t, size_t>& init_num_reg_trials,
        const std::unordered_map<image_t, size_t>& num_registrations) {
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
    std::vector<ImageInfo> image_infos;
    image_infos.reserve(reconstruction.NumImages());
    for (const auto& image : reconstruction.Images()) {
        // Only images with correspondences can be registered. Upstream
        // d3ccaf35 derives this from the correspondence graph; the fork's
        // legacy per-image counter is not maintained on a fresh Load.
        if (correspondence_graph.NumCorrespondencesForImage(image.first) ==
            0) {
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
        ImageInfo image_info;
        image_info.image_id = image.first;
        image_info.prior_focal_length = camera.HasPriorFocalLength();
        image_info.num_correspondences =
                correspondence_graph.NumCorrespondencesForImage(image.first);
        image_infos.push_back(image_info);
    }

    // Sort images such that images with a prior focal length and more
    // correspondences are preferred, i.e. they appear in the front of the list.
    std::sort(image_infos.begin(), image_infos.end(),
              [](const ImageInfo& image_info1, const ImageInfo& image_info2) {
                  if (image_info1.prior_focal_length &&
                      !image_info2.prior_focal_length) {
                      return true;
                  } else if (!image_info1.prior_focal_length &&
                             image_info2.prior_focal_length) {
                      return false;
                  } else {
                      return image_info1.num_correspondences >
                             image_info2.num_correspondences;
                  }
              });

    // Extract image identifiers in sorted order.
    std::vector<image_t> image_ids;
    image_ids.reserve(image_infos.size());
    for (const ImageInfo& image_info : image_infos) {
        image_ids.push_back(image_info.image_id);
    }

    return image_ids;
}

std::vector<image_t> IncrementalMapperImpl::FindSecondInitialImage(
        const IncrementalMapper::Options& options,
        const image_t image_id1,
        const CorrespondenceGraph& correspondence_graph,
        const Reconstruction& reconstruction,
        const std::unordered_map<image_t, size_t>& num_registrations) {
    // Collect images that are connected to the first seed image and have
    // not been registered before in other reconstructions.
    const class Image& image1 = reconstruction.Image(image_id1);
    std::unordered_map<image_t, point2D_t> num_correspondences;
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

    // Struct to hold meta-data for ranking images.
    struct ImageInfo {
        image_t image_id;
        bool prior_focal_length;
        point2D_t num_correspondences;
    };

    const size_t init_min_num_inliers =
            static_cast<size_t>(options.init_min_num_inliers);

    // Compose image information in a compact form for sorting.
    std::vector<ImageInfo> image_infos;
    image_infos.reserve(reconstruction.NumImages());
    for (const auto elem : num_correspondences) {
        if (elem.second >= init_min_num_inliers) {
            const class Image& image = reconstruction.Image(elem.first);
            const class Camera& camera =
                    reconstruction.Camera(image.CameraId());
            ImageInfo image_info;
            image_info.image_id = elem.first;
            image_info.prior_focal_length = camera.HasPriorFocalLength();
            image_info.num_correspondences = elem.second;
            image_infos.push_back(image_info);
        }
    }

    // Sort images such that images with a prior focal length and more
    // correspondences are preferred, i.e. they appear in the front of the list.
    std::sort(image_infos.begin(), image_infos.end(),
              [](const ImageInfo& image_info1, const ImageInfo& image_info2) {
                  if (image_info1.prior_focal_length &&
                      !image_info2.prior_focal_length) {
                      return true;
                  } else if (!image_info1.prior_focal_length &&
                             image_info2.prior_focal_length) {
                      return false;
                  } else {
                      return image_info1.num_correspondences >
                             image_info2.num_correspondences;
                  }
              });

    // Extract image identifiers in sorted order.
    std::vector<image_t> image_ids;
    image_ids.reserve(image_infos.size());
    for (const ImageInfo& image_info : image_infos) {
        image_ids.push_back(image_info.image_id);
    }

    return image_ids;
}

std::vector<image_t> IncrementalMapperImpl::FindNextImages(
        const IncrementalMapper::Options& options,
        const Reconstruction& reconstruction,
        const std::unordered_map<image_t, size_t>& num_reg_trials,
        const std::unordered_set<image_t>& filtered_images) {
    CHECK(options.Check());

    std::function<float(const Image&)> rank_image_func;
    switch (options.image_selection_method) {
        case IncrementalMapper::Options::ImageSelectionMethod::MAX_VISIBLE_POINTS_NUM:
            rank_image_func = RankNextImageMaxVisiblePointsNum;
            break;
        case IncrementalMapper::Options::ImageSelectionMethod::MAX_VISIBLE_POINTS_RATIO:
            rank_image_func = RankNextImageMaxVisiblePointsRatio;
            break;
        case IncrementalMapper::Options::ImageSelectionMethod::MIN_UNCERTAINTY:
            rank_image_func = RankNextImageMinUncertainty;
            break;
    }

    std::vector<std::pair<image_t, float>> image_ranks;
    std::vector<std::pair<image_t, float>> other_image_ranks;

    // Append images that have not failed to register before.
    for (const auto& image : reconstruction.Images()) {
        // Skip images that are already registered.
        if (image.second.IsRegistered()) {
            continue;
        }

        // Only consider images with a sufficient number of visible points.
        if (image.second.NumVisiblePoints3D() <
            static_cast<size_t>(options.abs_pose_min_num_inliers)) {
            continue;
        }

        // Only try registration for a certain maximum number of times.
        // Upstream parity (d3ccaf35): images never attempted before count as
        // zero trials, so the lookup must not require the key to exist.
        size_t trials = 0;
        if (const auto trials_it = num_reg_trials.find(image.first);
            trials_it != num_reg_trials.end()) {
            trials = trials_it->second;
        }
        if (trials >= static_cast<size_t>(options.max_reg_trials)) {
            continue;
        }

        // If image has been filtered or failed to register, place it in the
        // second bucket and prefer images that have not been tried before.
        const float rank = rank_image_func(image.second);
        if (filtered_images.count(image.first) == 0 && trials == 0) {
            image_ranks.emplace_back(image.first, rank);
        } else {
            other_image_ranks.emplace_back(image.first, rank);
        }
    }

    std::vector<image_t> ranked_images_ids;
    SortAndAppendNextImages(image_ranks, &ranked_images_ids);
    SortAndAppendNextImages(other_image_ranks, &ranked_images_ids);

    return ranked_images_ids;
}

std::vector<image_t> IncrementalMapperImpl::FindLocalBundle(
        const IncrementalMapper::Options& options,
        const image_t image_id,
        const Reconstruction& reconstruction) {
    CHECK(options.Check());

    const Image& image = reconstruction.Image(image_id);
    CHECK(image.IsRegistered());

    // Extract all images that have at least one 3D point with the query image
    // in common, and simultaneously count the number of common 3D points.

    std::unordered_map<image_t, size_t> shared_observations;

    std::unordered_set<point3D_t> point3D_ids;
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
            static_cast<size_t>(options.local_ba_num_images - 1);
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

    const double min_tri_angle_rad = DegToRad(options.local_ba_min_tri_angle);

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

}  // namespace colmap
