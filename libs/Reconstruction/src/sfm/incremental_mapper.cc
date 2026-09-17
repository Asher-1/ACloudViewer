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

#include <ceres/ceres.h>

#include "estimators/triangulation.h"
#include "estimators/two_view_geometry.h"
#include "sfm/observation_manager.h"
#include "sfm/incremental_mapper.h"
#include "sfm/incremental_mapper_impl.h"

#include "geometry/pose_prior.h"
#include "scene/reconstruction_pruning.h"

#include <array>
#include <fstream>

#include "scene/projection.h"
#include "geometry/triangulation.h"
#include "estimators/pose.h"
#include "sensor/bitmap.h"
#include "util/misc.h"

namespace colmap {
namespace {

}  // namespace

bool IncrementalMapper::Options::Check() const {
    CHECK_OPTION_GT(init_min_num_inliers, 0);
    CHECK_OPTION_GT(init_max_error, 0.0);
    CHECK_OPTION_GE(init_max_forward_motion, 0.0);
    CHECK_OPTION_LE(init_max_forward_motion, 1.0);
    CHECK_OPTION_GE(init_min_tri_angle, 0.0);
    CHECK_OPTION_GE(init_max_reg_trials, 1);
    CHECK_OPTION_GT(abs_pose_max_error, 0.0);
    CHECK_OPTION_GT(abs_pose_min_num_inliers, 0);
    CHECK_OPTION_GE(abs_pose_min_inlier_ratio, 0.0);
    CHECK_OPTION_LE(abs_pose_min_inlier_ratio, 1.0);
    CHECK_OPTION_GE(local_ba_num_images, 2);
    CHECK_OPTION_GE(local_ba_min_tri_angle, 0.0);
    CHECK_OPTION_GE(min_focal_length_ratio, 0.0);
    CHECK_OPTION_GE(max_focal_length_ratio, min_focal_length_ratio);
    CHECK_OPTION_GE(max_extra_param, 0.0);
    CHECK_OPTION_GE(filter_max_reproj_error, 0.0);
    CHECK_OPTION_GE(filter_min_tri_angle, 0.0);
    CHECK_OPTION_GE(max_reg_trials, 1);
    CHECK_OPTION_GE(ba_global_ignore_redundant_points3D_min_coverage_gain,
                    0.0);
    return true;
}

IncrementalMapper::IncrementalMapper(const DatabaseCache* database_cache)
    : database_cache_(database_cache),
      reconstruction_(nullptr),
      triangulator_(nullptr),
      num_total_reg_images_(0),
      num_shared_reg_images_(0),
      prev_init_image_pair_id_(kInvalidImagePairId) {}

void IncrementalMapper::BeginReconstruction(Reconstruction* reconstruction) {
    CHECK(reconstruction_ == nullptr);
    reconstruction_ = reconstruction;
    reconstruction_->Load(*database_cache_);
    reconstruction_->SetUp(database_cache_->CorrespondenceGraph().get());
    triangulator_.reset(new IncrementalTriangulator(
            database_cache_->CorrespondenceGraph().get(), reconstruction));
    obs_manager_ = std::make_shared<class ObservationManager>(
            *reconstruction, database_cache_->CorrespondenceGraph());

    num_shared_reg_images_ = 0;
    num_reg_images_per_camera_.clear();
    num_reg_frames_per_rig_.clear();
    // Upstream parity (d3ccaf35): account for the already-registered frames
    // of a continued reconstruction at the frame level.
    for (const frame_t frame_id : reconstruction_->RegFrameIds()) {
        RegisterFrameEvent(frame_id);
    }

    existing_image_ids_ =
            std::unordered_set<image_t>(reconstruction->RegImageIds().begin(),
                                        reconstruction->RegImageIds().end());

    prev_init_image_pair_id_ = kInvalidImagePairId;
    prev_init_two_view_geometry_ = TwoViewGeometry();

    filtered_images_.clear();
    num_reg_trials_.clear();
}

void IncrementalMapper::EndReconstruction(const bool discard) {
    CHECK_NOTNULL(reconstruction_);

    if (discard) {
        // Upstream parity (d3ccaf35): de-register at the frame level.
        const auto reg_frame_ids = reconstruction_->RegFrameIds();
        for (const frame_t frame_id : reg_frame_ids) {
            if (obs_manager_) {
                obs_manager_->DeRegisterFrame(frame_id);
            }
            DeRegisterFrameEvent(frame_id);
        }
    }

    reconstruction_->TearDown();
    reconstruction_ = nullptr;
    triangulator_.reset();
    obs_manager_.reset();
}

bool IncrementalMapper::FindInitialImagePair(const Options& options,
                                             image_t* image_id1,
                                             image_t* image_id2) {
    CHECK(options.Check());

    std::vector<image_t> image_ids1;
    if (*image_id1 != kInvalidImageId && *image_id2 == kInvalidImageId) {
        // Only *image_id1 provided.
        if (!database_cache_->ExistsImage(*image_id1)) {
            return false;
        }
        image_ids1.push_back(*image_id1);
    } else if (*image_id1 == kInvalidImageId && *image_id2 != kInvalidImageId) {
        // Only *image_id2 provided.
        if (!database_cache_->ExistsImage(*image_id2)) {
            return false;
        }
        image_ids1.push_back(*image_id2);
    } else {
        // No initial seed image provided.
        image_ids1 = FindFirstInitialImage(options);
    }

    // Try to find good initial pair.
    for (size_t i1 = 0; i1 < image_ids1.size(); ++i1) {
        *image_id1 = image_ids1[i1];

        const std::vector<image_t> image_ids2 =
                FindSecondInitialImage(options, *image_id1);

        for (size_t i2 = 0; i2 < image_ids2.size(); ++i2) {
            *image_id2 = image_ids2[i2];

            const image_pair_t pair_id =
                    Database::ImagePairToPairId(*image_id1, *image_id2);

            // Try every pair only once.
            if (init_image_pairs_.count(pair_id) > 0) {
                continue;
            }

            init_image_pairs_.insert(pair_id);

            if (EstimateInitialTwoViewGeometry(options, *image_id1,
                                               *image_id2)) {
                return true;
            }
        }
    }

    // No suitable pair found in entire dataset.
    *image_id1 = kInvalidImageId;
    *image_id2 = kInvalidImageId;

    return false;
}

std::vector<image_t> IncrementalMapper::FindNextImages(
        const Options& options, const bool structure_less) {
    CHECK_NOTNULL(reconstruction_);
    CHECK(options.Check());
    // Upstream parity (d3ccaf35): structure-less registration tracks its
    // own trial counter so that failed structure-less attempts do not
    // consume the structure-based registration budget.
    return IncrementalMapperImpl::FindNextImages(
            options, *reconstruction_,
            structure_less ? num_structure_less_reg_trials_ : num_reg_trials_,
            filtered_images_);
}

bool IncrementalMapper::RegisterInitialImagePair(const Options& options,
                                                 const image_t image_id1,
                                                 const image_t image_id2) {
    CHECK_NOTNULL(reconstruction_);
    CHECK_EQ(reconstruction_->NumRegImages(), 0);

    CHECK(options.Check());

    init_num_reg_trials_[image_id1] += 1;
    init_num_reg_trials_[image_id2] += 1;
    num_reg_trials_[image_id1] += 1;
    num_reg_trials_[image_id2] += 1;

    const image_pair_t pair_id =
            Database::ImagePairToPairId(image_id1, image_id2);
    init_image_pairs_.insert(pair_id);

    Image& image1 = reconstruction_->Image(image_id1);

    Image& image2 = reconstruction_->Image(image_id2);

    //////////////////////////////////////////////////////////////////////////////
    // Estimate two-view geometry
    //////////////////////////////////////////////////////////////////////////////

    if (!EstimateInitialTwoViewGeometry(options, image_id1, image_id2)) {
        return false;
    }

    // Upstream parity (d3ccaf35): pose writes go through the frame-aware
    // model (Frame::SetCamFromWorld is rig-aware). The fork's mapper runtime
    // keeps frameless cache images (DatabaseCache::AddImage does not wire
    // frames yet), so the legacy qvec/tvec storage is the live path until
    // the cache wiring lands (W3-2b stage 2); the frame branch mirrors the
    // established dual-path pattern of Image::ProjectionMatrix.
    if (image1.HasFramePtr()) {
        image1.SetCamFromWorld(Rigid3d());
    } else {
        image1.Qvec() = ComposeIdentityQuaternion();
        image1.Tvec() = Eigen::Vector3d(0, 0, 0);
    }
    if (prev_init_two_view_geometry_.cam2_from_cam1) {
        const Eigen::Quaterniond& q =
                prev_init_two_view_geometry_.cam2_from_cam1->rotation();
        if (image2.HasFramePtr()) {
            image2.SetCamFromWorld(*prev_init_two_view_geometry_.cam2_from_cam1);
        } else {
            image2.Qvec() = Eigen::Vector4d(q.w(), q.x(), q.y(), q.z());
            image2.Tvec() =
                    prev_init_two_view_geometry_.cam2_from_cam1->translation();
        }
    }

    // Upstream parity (d3ccaf35): registration and bookkeeping happen at
    // the frame level (every image of the frame joins the reconstruction).
    obs_manager_->RegisterFrame(image1.FrameId());
    RegisterFrameEvent(image1.FrameId());
    if (image2.FrameId() != image1.FrameId()) {
        obs_manager_->RegisterFrame(image2.FrameId());
        RegisterFrameEvent(image2.FrameId());
    }

    // Upstream parity (d3ccaf35): the initial two-view points are NOT created
    // here. The pipeline triangulates both registered images through
    // TriangulateImage (the IncrementalTriangulator::Create path), whose
    // RANSAC-based estimation applies the reprojection-error filtering that
    // this function's legacy inline loop lacked (it kept every
    // angle/depth-valid match, inflating the initial reconstruction with
    // unfiltered observations).
    return true;
}

bool IncrementalMapper::RegisterNextImage(const Options& options,
                                          const image_t image_id) {
    CHECK_NOTNULL(reconstruction_);
    CHECK_GE(reconstruction_->NumRegImages(), 2);

    CHECK(options.Check());

    Image& image = reconstruction_->Image(image_id);
    Camera& camera = reconstruction_->Camera(image.CameraId());

    CHECK(!image.IsRegistered()) << "Image cannot be registered multiple times";

    // Upstream parity (d3ccaf35): multi-sensor frames with good focal
    // lengths for every camera register through the generalized absolute
    // pose estimator over all images of the frame.
    if (image.HasFramePtr() && image.FramePtr()->HasRigPtr() &&
        image.FramePtr()->RigPtr()->NumSensors() > 1) {
        bool all_cameras_have_good_focal_length = true;
        for (const image_t frame_image_id : image.FramePtr()->ImageIds()) {
            const Image& frame_image = reconstruction_->Image(frame_image_id);
            const Camera& frame_camera =
                    reconstruction_->Camera(frame_image.CameraId());
            if ((!frame_camera.HasPriorFocalLength() &&
                 num_reg_images_per_camera_[frame_camera.CameraId()] == 0) ||
                frame_camera.HasBogusParams(options.min_focal_length_ratio,
                                            options.max_focal_length_ratio,
                                            options.max_extra_param)) {
                all_cameras_have_good_focal_length = false;
                break;
            }
        }
        if (all_cameras_have_good_focal_length) {
            VLOG(2) << "Registering image using generalized pose estimation";
            return RegisterNextGeneralFrame(options, *image.FramePtr());
        }
    }

    num_reg_trials_[image_id] += 1;

    // Check if enough 2D-3D correspondences.
    if (image.NumVisiblePoints3D() <
        static_cast<size_t>(options.abs_pose_min_num_inliers)) {
        return false;
    }

    //////////////////////////////////////////////////////////////////////////////
    // Search for 2D-3D correspondences
    //////////////////////////////////////////////////////////////////////////////

    const int kCorrTransitivity = 1;

    std::vector<std::pair<point2D_t, point3D_t>> tri_corrs;
    std::vector<Eigen::Vector2d> tri_points2D;
    std::vector<Eigen::Vector3d> tri_points3D;

    for (point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D();
         ++point2D_idx) {
        const Point2D& point2D = image.Point2D(point2D_idx);
        const CorrespondenceGraph& correspondence_graph =
                *database_cache_->CorrespondenceGraph();
        std::vector<CorrespondenceGraph::Correspondence> corrs;
        correspondence_graph.ExtractTransitiveCorrespondences(
                image_id, point2D_idx, kCorrTransitivity, &corrs);

        std::unordered_set<point3D_t> point3D_ids;

        for (const auto corr : corrs) {
            if (!reconstruction_->ExistsImage(corr.image_id)) {
                std::cout << "[IncrementalMapper::RegisterNextImage] Image "
                          << corr.image_id << " does not exist" << std::endl;
                continue;
            }

            const Image& corr_image = reconstruction_->Image(corr.image_id);
            if (!corr_image.IsRegistered()) {
                continue;
            }

            const Point2D& corr_point2D = corr_image.Point2D(corr.point2D_idx);
            if (!corr_point2D.HasPoint3D()) {
                continue;
            }

            // Avoid duplicate correspondences.
            if (point3D_ids.count(corr_point2D.Point3DId()) > 0) {
                continue;
            }

            const Camera& corr_camera =
                    reconstruction_->Camera(corr_image.CameraId());

            // Avoid correspondences to images with bogus camera parameters.
            if (corr_camera.HasBogusParams(options.min_focal_length_ratio,
                                           options.max_focal_length_ratio,
                                           options.max_extra_param)) {
                continue;
            }

            const Point3D& point3D =
                    reconstruction_->Point3D(corr_point2D.Point3DId());

            tri_corrs.emplace_back(point2D_idx, corr_point2D.Point3DId());
            point3D_ids.insert(corr_point2D.Point3DId());
            tri_points2D.push_back(point2D.XY());
            tri_points3D.push_back(point3D.XYZ());
        }
    }

    // The size of `next_image.num_tri_obs` and `tri_corrs_point2D_idxs.size()`
    // can only differ, when there are images with bogus camera parameters, and
    // hence we skip some of the 2D-3D correspondences.
    if (tri_points2D.size() <
        static_cast<size_t>(options.abs_pose_min_num_inliers)) {
        return false;
    }

    //////////////////////////////////////////////////////////////////////////////
    // 2D-3D estimation
    //////////////////////////////////////////////////////////////////////////////

    // Only refine / estimate focal length, if no focal length was specified
    // (manually or through EXIF) and if it was not already estimated previously
    // from another image (when multiple images share the same camera
    // parameters)

    AbsolutePoseEstimationOptions abs_pose_options;
    abs_pose_options.num_threads = options.num_threads;
    abs_pose_options.num_focal_length_samples = 30;
    abs_pose_options.min_focal_length_ratio = options.min_focal_length_ratio;
    abs_pose_options.max_focal_length_ratio = options.max_focal_length_ratio;
    abs_pose_options.ransac_options.max_error = options.abs_pose_max_error;
    abs_pose_options.ransac_options.min_inlier_ratio =
            options.abs_pose_min_inlier_ratio;
    // Use high confidence to avoid preemptive termination of P3P RANSAC
    // - too early termination may lead to bad registration.
    abs_pose_options.ransac_options.min_num_trials = 100;
    abs_pose_options.ransac_options.max_num_trials = 10000;
    abs_pose_options.ransac_options.confidence = 0.99999;

    AbsolutePoseRefinementOptions abs_pose_refinement_options;
    if (num_reg_images_per_camera_[image.CameraId()] > 0) {
        // Camera already refined from another image with the same camera.
        if (camera.HasBogusParams(options.min_focal_length_ratio,
                                  options.max_focal_length_ratio,
                                  options.max_extra_param)) {
            // Previously refined camera has bogus parameters,
            // so reset parameters and try to re-estimage.
            camera.SetParams(
                    database_cache_->Camera(image.CameraId()).Params());
            abs_pose_options.estimate_focal_length =
                    !camera.HasPriorFocalLength();
            abs_pose_refinement_options.refine_focal_length = true;
            abs_pose_refinement_options.refine_extra_params = true;
        } else {
            abs_pose_options.estimate_focal_length = false;
            abs_pose_refinement_options.refine_focal_length = false;
            abs_pose_refinement_options.refine_extra_params = false;
        }
    } else {
        // Camera not refined before. Note that the camera parameters might have
        // been changed before but the image was filtered, so we explicitly
        // reset the camera parameters and try to re-estimate them.
        camera.SetParams(database_cache_->Camera(image.CameraId()).Params());
        abs_pose_options.estimate_focal_length = !camera.HasPriorFocalLength();
        abs_pose_refinement_options.refine_focal_length = true;
        abs_pose_refinement_options.refine_extra_params = true;
    }

    if (!options.abs_pose_refine_focal_length) {
        abs_pose_options.estimate_focal_length = false;
        abs_pose_refinement_options.refine_focal_length = false;
    }

    if (!options.abs_pose_refine_extra_params) {
        abs_pose_refinement_options.refine_extra_params = false;
    }

    size_t num_inliers;
    std::vector<char> inlier_mask;

    // Upstream parity (d3ccaf35): estimate and refine into locals and commit
    // the pose once through the frame-aware model; the image-local storage
    // stays in sync as the shadow (and remains the live path for frameless
    // cache images).
    Eigen::Vector4d cam_qvec = image.Qvec();
    Eigen::Vector3d cam_tvec = image.Tvec();

    if (!EstimateAbsolutePose(abs_pose_options, tri_points2D, tri_points3D,
                              &cam_qvec, &cam_tvec, &camera,
                              &num_inliers, &inlier_mask)) {
        return false;
    }

    if (num_inliers < static_cast<size_t>(options.abs_pose_min_num_inliers)) {
        return false;
    }

    //////////////////////////////////////////////////////////////////////////////
    // Pose refinement
    //////////////////////////////////////////////////////////////////////////////

    // Upstream parity (d3ccaf35 estimators/pose.cc): the pose is refined as
    // a single Rigid3d parameter block under a product (EigenQuaternion +
    // Euclidean) manifold.
    Rigid3d cam_from_world(
            Eigen::Quaterniond(cam_qvec(0), cam_qvec(1), cam_qvec(2),
                               cam_qvec(3)),
            cam_tvec);

    if (!RefineAbsolutePose(abs_pose_refinement_options, inlier_mask,
                            tri_points2D, tri_points3D, &cam_from_world,
                            &camera)) {
        return false;
    }

    if (image.HasFramePtr()) {
        image.SetCamFromWorld(cam_from_world);
    } else {
        const Eigen::Quaterniond quat = cam_from_world.rotation();
        image.Qvec() =
                Eigen::Vector4d(quat.w(), quat.x(), quat.y(), quat.z());
        image.Tvec() = cam_from_world.translation();
    }

    //////////////////////////////////////////////////////////////////////////////
    // Continue tracks
    //////////////////////////////////////////////////////////////////////////////

    // Upstream parity (d3ccaf35): frame-level registration.
    obs_manager_->RegisterFrame(image.FrameId());
    RegisterFrameEvent(image.FrameId());

    for (size_t i = 0; i < inlier_mask.size(); ++i) {
        if (inlier_mask[i]) {
            const point2D_t point2D_idx = tri_corrs[i].first;
            const Point2D& point2D = image.Point2D(point2D_idx);
            if (!point2D.HasPoint3D()) {
                const point3D_t point3D_id = tri_corrs[i].second;
                const TrackElement track_el(image_id, point2D_idx);
                obs_manager_->AddObservation(point3D_id, track_el);
                triangulator_->AddModifiedPoint3D(point3D_id);
            }
        }
    }

    return true;
}

size_t IncrementalMapper::TriangulateImage(
        const IncrementalTriangulator::Options& tri_options,
        const image_t image_id) {
    CHECK_NOTNULL(reconstruction_);
    return triangulator_->TriangulateImage(tri_options, image_id);
}

size_t IncrementalMapper::Retriangulate(
        const IncrementalTriangulator::Options& tri_options) {
    CHECK_NOTNULL(reconstruction_);
    return triangulator_->Retriangulate(tri_options);
}

size_t IncrementalMapper::CompleteTracks(
        const IncrementalTriangulator::Options& tri_options) {
    CHECK_NOTNULL(reconstruction_);
    return triangulator_->CompleteAllTracks(tri_options);
}

size_t IncrementalMapper::MergeTracks(
        const IncrementalTriangulator::Options& tri_options) {
    CHECK_NOTNULL(reconstruction_);
    return triangulator_->MergeAllTracks(tri_options);
}

IncrementalMapper::LocalBundleAdjustmentReport
IncrementalMapper::AdjustLocalBundle(
        const Options& options,
        const BundleAdjustmentOptions& ba_options,
        const IncrementalTriangulator::Options& tri_options,
        const image_t image_id,
        const std::unordered_set<point3D_t>& point3D_ids) {
    CHECK_NOTNULL(reconstruction_);
    CHECK(options.Check());

    LocalBundleAdjustmentReport report;

    // Find images that have most 3D points with given image in common.
    const std::vector<image_t> local_bundle =
            FindLocalBundle(options, image_id);

    // Do the bundle adjustment only if there is any connected images.
    if (local_bundle.size() > 0) {
        BundleAdjustmentConfig ba_config;
        ba_config.AddImage(image_id);
        for (const image_t local_image_id : local_bundle) {
            ba_config.AddImage(local_image_id);
        }
        // Fix the existing frames, if option specified (upstream parity,
        // d3ccaf35: frame-level rig-from-world constants; the fork option
        // name fix_existing_images is kept for CLI compatibility, and the
        // image-level constant remains the fallback for frameless images).
        if (options.fix_existing_images) {
            for (const image_t local_image_id : local_bundle) {
                if (existing_image_ids_.count(local_image_id)) {
                    const Image& local_image =
                            reconstruction_->Image(local_image_id);
                    if (local_image.HasFrameId()) {
                        ba_config.SetConstantRigFromWorldPose(
                                local_image.FrameId());
                    } else {
                        ba_config.SetConstantPose(local_image_id);
                    }
                }
            }
        }

        // Determine which cameras to fix, when not all the registered images
        // are within the current local bundle.
        std::unordered_map<camera_t, size_t> num_images_per_camera;
        for (const image_t image_id : ba_config.Images()) {
            const Image& image = reconstruction_->Image(image_id);
            num_images_per_camera[image.CameraId()] += 1;
        }

        for (const auto& camera_id_and_num_images_pair :
             num_images_per_camera) {
            const size_t num_reg_images_for_camera =
                    num_reg_images_per_camera_.at(
                            camera_id_and_num_images_pair.first);
            if (camera_id_and_num_images_pair.second <
                num_reg_images_for_camera) {
                ba_config.SetConstantCamera(
                        camera_id_and_num_images_pair.first);
            }
        }

        // Fix the gauge by fixing three 3D points, as in the upstream
        // local bundle adjustment (d3ccaf35); this avoids
        // scale/rotation/translation drift without over-constraining
        // specific image poses.
        ba_config.FixGauge(BundleAdjustmentGauge::THREE_POINTS);

        // Make sure, we refine all new and short-track 3D points, no matter if
        // they are fully contained in the local image set or not. Do not
        // include long track 3D points as they are usually already very stable
        // and adding to them to bundle adjustment and track merging/completion
        // would slow down the local bundle adjustment significantly.
        std::unordered_set<point3D_t> variable_point3D_ids;
        for (const point3D_t point3D_id : point3D_ids) {
            const Point3D& point3D = reconstruction_->Point3D(point3D_id);
            const size_t kMaxTrackLength = 15;
            if (!point3D.HasError() ||
                point3D.Track().Length() <= kMaxTrackLength) {
                ba_config.AddVariablePoint(point3D_id);
                variable_point3D_ids.insert(point3D_id);
            }
        }

        // Adjust the local bundle.
        auto bundle_adjuster = CreateDefaultBundleAdjuster(ba_options, ba_config);
        bundle_adjuster->Solve(reconstruction_);

        report.num_adjusted_observations =
                bundle_adjuster->Summary().num_residuals / 2;

        // Merge refined tracks with other existing points.
        report.num_merged_observations =
                triangulator_->MergeTracks(tri_options, variable_point3D_ids);
        // Complete tracks that may have failed to triangulate before refinement
        // of camera pose and calibration in bundle-adjustment. This may avoid
        // that some points are filtered and it helps for subsequent image
        // registrations.
        report.num_completed_observations = triangulator_->CompleteTracks(
                tri_options, variable_point3D_ids);
        report.num_completed_observations +=
                triangulator_->CompleteImage(tri_options, image_id);
    }

    // Filter both the modified images and all changed 3D points to make sure
    // there are no outlier points in the model. This results in duplicate work
    // as many of the provided 3D points may also be contained in the adjusted
    // images, but the filtering is not a bottleneck at this point.
    std::unordered_set<image_t> filter_image_ids;
    filter_image_ids.insert(image_id);
    filter_image_ids.insert(local_bundle.begin(), local_bundle.end());
    report.num_filtered_observations = reconstruction_->FilterPoints3DInImages(
            options.filter_max_reproj_error, options.filter_min_tri_angle,
            filter_image_ids);
    report.num_filtered_observations += reconstruction_->FilterPoints3D(
            options.filter_max_reproj_error, options.filter_min_tri_angle,
            point3D_ids);

    return report;
}

size_t NumRegisteredPosePriors(const std::vector<PosePrior>& pose_priors,
                               const BundleAdjustmentConfig& ba_config) {
    size_t num_registered_pose_priors = 0;
    for (const PosePrior& pose_prior : pose_priors) {
        if (pose_prior.HasPosition() &&
            pose_prior.corr_data_id.sensor_id.type == SensorType::CAMERA &&
            ba_config.HasImage(pose_prior.corr_data_id.id)) {
            ++num_registered_pose_priors;
        }
    }
    return num_registered_pose_priors;
}

bool IncrementalMapper::AdjustGlobalBundle(
        const Options& options, const BundleAdjustmentOptions& ba_options) {
    CHECK_NOTNULL(reconstruction_);

    const std::vector<image_t>& reg_image_ids = reconstruction_->RegImageIds();

    CHECK_GE(reg_image_ids.size(), 2) << "At least two images must be "
                                         "registered for global "
                                         "bundle-adjustment";

    BundleAdjustmentOptions custom_ba_options = ba_options;
    // Use stricter convergence criteria for first registered images
    // (upstream parity, d3ccaf35 AdjustGlobalBundle).
    constexpr size_t kMinNumRegFramesForFastBA = 10;
    const bool is_small_reconstruction =
        reconstruction_->NumRegFrames() < kMinNumRegFramesForFastBA;
    if (is_small_reconstruction) {
        custom_ba_options.solver_options.function_tolerance /= 10;
        custom_ba_options.solver_options.gradient_tolerance /= 10;
        custom_ba_options.solver_options.parameter_tolerance /= 10;
        custom_ba_options.solver_options.max_num_iterations *= 2;
        custom_ba_options.solver_options.max_linear_solver_iterations = 200;
    }

    // Avoid degeneracies in bundle adjustment.
    reconstruction_->FilterObservationsWithNegativeDepth();

    // Configure bundle adjustment.
    BundleAdjustmentConfig ba_config;
    for (const image_t image_id : reg_image_ids) {
        ba_config.AddImage(image_id);
    }

    // Fix the existing frames, if option specified (upstream parity,
    // d3ccaf35 AdjustGlobalBundle: frame-level rig-from-world constants;
    // fork option name kept, image-level fallback for frameless images).
    if (options.fix_existing_images) {
        for (const image_t image_id : reg_image_ids) {
            if (existing_image_ids_.count(image_id)) {
                const Image& image = reconstruction_->Image(image_id);
                if (image.HasFrameId()) {
                    ba_config.SetConstantRigFromWorldPose(image.FrameId());
                } else {
                    ba_config.SetConstantPose(image_id);
                }
            }
        }
    }

    // Ignore redundant 3D points in the first solve, if option specified
    // (upstream parity; only past the "small reconstruction" threshold).
    if (!is_small_reconstruction &&
        options.ba_global_ignore_redundant_points3D) {
        const std::vector<point3D_t> redundant_point3D_ids =
                FindRedundantPoints3D(
                        options
                                .ba_global_ignore_redundant_points3D_min_coverage_gain,
                        *reconstruction_);
        VLOG(1) << "=> Ignoring " << redundant_point3D_ids.size() << " / "
                << reconstruction_->NumPoints3D() << " redundant 3D points";
        for (const point3D_t point3D_id : redundant_point3D_ids) {
            ba_config.IgnorePoint(point3D_id);
        }
    }

    // Fix 7-DOFs of the bundle adjustment problem.
    ba_config.SetConstantPose(reg_image_ids[0]);
    if (!options.fix_existing_images ||
        !existing_image_ids_.count(reg_image_ids[1])) {
        ba_config.SetConstantTvec(reg_image_ids[1], {0});
    }

    // Run bundle adjustment. With position priors, the gauge is fixed by
    // the priors instead of the two-camera heuristic (upstream parity).
    const bool use_prior_position =
            options.use_prior_position &&
            NumRegisteredPosePriors(database_cache_->PosePriors(),
                                    ba_config) >= 3;
    if (use_prior_position) {
        PosePriorBundleAdjustmentOptions prior_options;
        if (options.use_robust_loss_on_prior_position) {
            prior_options.prior_position_loss_function_type =
                    BundleAdjustmentOptions::LossFunctionType::CAUCHY;
        }
        prior_options.prior_position_loss_scale =
                options.prior_position_loss_scale;
        auto bundle_adjuster = CreateDefaultBundleAdjuster(custom_ba_options, ba_config);
        bundle_adjuster->SetPosePriors(prior_options,
                                      database_cache_->PosePriors());
        if (!bundle_adjuster->Solve(reconstruction_)) {
            return false;
        }
    } else {
        auto bundle_adjuster = CreateDefaultBundleAdjuster(custom_ba_options, ba_config);
        if (!bundle_adjuster->Solve(reconstruction_)) {
            return false;
        }
    }

    // Normalize scene for numerical stability and
    // to avoid large scale changes in viewer.
    reconstruction_->Normalize();

    return true;
}


size_t IncrementalMapper::FilterImages(const Options& options) {
    CHECK_NOTNULL(reconstruction_);
    CHECK(options.Check());

    // Do not filter images in the early stage of the reconstruction, since the
    // calibration is often still refining a lot. Hence, the camera parameters
    // are not stable in the beginning.
    const size_t kMinNumImages = 20;
    if (reconstruction_->NumRegImages() < kMinNumImages) {
        return {};
    }

    const std::vector<image_t> image_ids = reconstruction_->FilterImages(
            options.min_focal_length_ratio, options.max_focal_length_ratio,
            options.max_extra_param);

    for (const image_t image_id : image_ids) {
        DeRegisterImageEvent(image_id);
        filtered_images_.insert(image_id);
    }

    return image_ids.size();
}

size_t IncrementalMapper::FilterPoints(const Options& options) {
    CHECK_NOTNULL(reconstruction_);
    CHECK(options.Check());
    return reconstruction_->FilterAllPoints3D(options.filter_max_reproj_error,
                                              options.filter_min_tri_angle);
}

size_t IncrementalMapper::CompleteAndMergeTracks(
        const IncrementalTriangulator::Options& tri_options) {
    const size_t num_completed_observations = CompleteTracks(tri_options);
    VLOG(1) << "=> Completed observations: " << num_completed_observations;
    const size_t num_merged_observations = MergeTracks(tri_options);
    VLOG(1) << "=> Merged observations: " << num_merged_observations;
    return num_completed_observations + num_merged_observations;
}

void IncrementalMapper::IterativeGlobalRefinement(
        const int max_num_refinements,
        const double max_refinement_change,
        const Options& options,
        const BundleAdjustmentOptions& ba_options,
        const IncrementalTriangulator::Options& tri_options,
        const bool normalize_reconstruction) {
    // Upstream parity (d3ccaf35 IncrementalMapper::IterativeGlobalRefinement).
    if (ba_options.check_if_stopped && ba_options.check_if_stopped()) {
        return;
    }
    CompleteAndMergeTracks(tri_options);
    const size_t num_retriangulated_observations = Retriangulate(tri_options);
    VLOG(1) << "=> Retriangulated observations: "
            << num_retriangulated_observations;
    for (int i = 0; i < max_num_refinements; ++i) {
        if (ba_options.check_if_stopped && ba_options.check_if_stopped()) {
            break;
        }
        const size_t num_observations = reconstruction_->ComputeNumObservations();
        AdjustGlobalBundle(options, ba_options);
        if (ba_options.check_if_stopped && ba_options.check_if_stopped()) {
            break;
        }
        if (normalize_reconstruction && !options.use_prior_position) {
            // Normalize scene for numerical stability and
            // to avoid large scale changes in the viewer.
            reconstruction_->Normalize();
        }
        size_t num_changed_observations = CompleteAndMergeTracks(tri_options);
        num_changed_observations += FilterPoints(options);
        const double changed =
                num_observations == 0
                        ? 0
                        : static_cast<double>(num_changed_observations) /
                              num_observations;
        VLOG(1) << StringPrintf("=> Changed observations: %.6f", changed);
        if (changed < max_refinement_change) {
            break;
        }
    }
    ClearModifiedPoints3D();
}

const Reconstruction& IncrementalMapper::GetReconstruction() const {
    CHECK_NOTNULL(reconstruction_);
    return *reconstruction_;
}

size_t IncrementalMapper::NumTotalRegImages() const {
    return num_total_reg_images_;
}

size_t IncrementalMapper::NumSharedRegImages() const {
    return num_shared_reg_images_;
}

const std::unordered_set<point3D_t>& IncrementalMapper::GetModifiedPoints3D() {
    return triangulator_->GetModifiedPoints3D();
}

void IncrementalMapper::ClearModifiedPoints3D() {
    triangulator_->ClearModifiedPoints3D();
}

std::vector<image_t> IncrementalMapper::FindFirstInitialImage(
        const Options& options) const {
    return IncrementalMapperImpl::FindFirstInitialImage(
            options, *database_cache_->CorrespondenceGraph(),
            *reconstruction_, init_num_reg_trials_, num_registrations_);
}

std::vector<image_t> IncrementalMapper::FindSecondInitialImage(
        const Options& options, const image_t image_id1) const {
    return IncrementalMapperImpl::FindSecondInitialImage(
            options, image_id1, *database_cache_->CorrespondenceGraph(),
            *reconstruction_, num_registrations_);
}

std::vector<image_t> IncrementalMapper::FindLocalBundle(
        const Options& options, const image_t image_id) const {
    CHECK(options.Check());
    return IncrementalMapperImpl::FindLocalBundle(options, image_id,
                                                  *reconstruction_);
}

void IncrementalMapper::RegisterImageEvent(const image_t image_id) {
    const Image& image = reconstruction_->Image(image_id);
    size_t& num_reg_images_for_camera =
            num_reg_images_per_camera_[image.CameraId()];
    num_reg_images_for_camera += 1;

    size_t& num_regs_for_image = num_registrations_[image_id];
    num_regs_for_image += 1;
    if (num_regs_for_image == 1) {
        num_total_reg_images_ += 1;
    } else if (num_regs_for_image > 1) {
        num_shared_reg_images_ += 1;
    }
}

void IncrementalMapper::DeRegisterImageEvent(const image_t image_id) {
    const Image& image = reconstruction_->Image(image_id);
    size_t& num_reg_images_for_camera =
            num_reg_images_per_camera_.at(image.CameraId());
    CHECK_GT(num_reg_images_for_camera, 0);
    num_reg_images_for_camera -= 1;

    size_t& num_regs_for_image = num_registrations_[image_id];
    num_regs_for_image -= 1;
    if (num_regs_for_image == 0) {
        num_total_reg_images_ -= 1;
    } else if (num_regs_for_image > 0) {
        num_shared_reg_images_ -= 1;
    }
}

// Upstream parity (d3ccaf35 sfm/incremental_mapper.cc): frame-level
// registration bookkeeping on top of the per-image counters.
void IncrementalMapper::RegisterFrameEvent(const frame_t frame_id) {
    const Frame& frame = reconstruction_->Frame(frame_id);

    size_t& num_reg_frames_for_rig = num_reg_frames_per_rig_[frame.RigId()];
    num_reg_frames_for_rig += 1;

    for (const image_t image_id : frame.ImageIds()) {
        RegisterImageEvent(image_id);
    }
}

void IncrementalMapper::DeRegisterFrameEvent(const frame_t frame_id) {
    const Frame& frame = reconstruction_->Frame(frame_id);

    size_t& num_reg_frames_for_rig =
            num_reg_frames_per_rig_.at(frame.RigId());
    CHECK_GT(num_reg_frames_for_rig, 0);
    num_reg_frames_for_rig -= 1;

    for (const image_t image_id : frame.ImageIds()) {
        DeRegisterImageEvent(image_id);
    }
}

const std::unordered_map<rig_t, size_t>&
IncrementalMapper::NumRegFramesPerRig() const {
    return num_reg_frames_per_rig_;
}

// Upstream parity (d3ccaf35 sfm/incremental_mapper.cc
// RegisterNextGeneralFrame): register a full multi-sensor frame by pooling
// the 2D-3D correspondences of every image in the frame and solving a
// generalized absolute pose.
bool IncrementalMapper::RegisterNextGeneralFrame(const Options& options,
                                                 Frame& frame) {
    THROW_CHECK_GT(frame.RigPtr()->NumSensors(), 1);

    struct Corr {
        point2D_t point2D_idx;
        image_t image_id;
        point3D_t point3D_id;
    };

    std::vector<Corr> tri_corrs;
    std::vector<Eigen::Vector2d> tri_points2D;
    std::vector<Eigen::Vector3d> tri_points3D;
    std::vector<size_t> tri_camera_idxs;

    std::vector<Rigid3d> cams_from_rig;
    cams_from_rig.reserve(frame.RigPtr()->NumSensors());
    std::vector<Camera> cameras;
    cameras.reserve(frame.RigPtr()->NumSensors());

    const auto correspondence_graph = database_cache_->CorrespondenceGraph();

    for (const image_t image_id : frame.ImageIds()) {
        const Image& image = reconstruction_->Image(image_id);
        const Camera& camera = *image.CameraPtr();

        const size_t camera_idx = cameras.size();
        if (frame.RigPtr()->IsRefSensor(camera.SensorId())) {
            cams_from_rig.push_back(Rigid3d());
        } else {
            cams_from_rig.push_back(
                    frame.RigPtr()->SensorFromRig(camera.SensorId()));
        }
        cameras.push_back(camera);

        num_reg_trials_[image_id] += 1;

        FlatHashSet<point3D_t> corr_point3D_ids;
        for (point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D();
             ++point2D_idx) {
            const Point2D& point2D = image.Point2D(point2D_idx);

            corr_point3D_ids.clear();
            const auto corr_range = correspondence_graph->FindCorrespondences(
                    image_id, point2D_idx);
            for (const auto* corr = corr_range.beg; corr < corr_range.end;
                 ++corr) {
                const Image& corr_image =
                        reconstruction_->Image(corr->image_id);
                if (!corr_image.HasPose()) {
                    continue;
                }

                const Point2D& corr_point2D =
                        corr_image.Point2D(corr->point2D_idx);
                if (!corr_point2D.HasPoint3D()) {
                    continue;
                }

                // Avoid duplicate correspondences.
                if (corr_point3D_ids.count(corr_point2D.Point3DId()) > 0) {
                    continue;
                }

                const Camera& corr_camera =
                        *corr_image.CameraPtr();

                // Avoid correspondences to images with bogus camera
                // parameters.
                if (corr_camera.HasBogusParams(
                            options.min_focal_length_ratio,
                            options.max_focal_length_ratio,
                            options.max_extra_param)) {
                    continue;
                }

                const Point3D& point3D =
                        reconstruction_->Point3D(corr_point2D.Point3DId());

                tri_corrs.push_back(
                        Corr{point2D_idx, image_id, corr_point2D.Point3DId()});
                corr_point3D_ids.insert(corr_point2D.Point3DId());
                tri_points2D.push_back(point2D.XY());
                tri_points3D.push_back(point3D.XYZ());
                tri_camera_idxs.push_back(camera_idx);
            }
        }
    }

    // The size of the correspondences can only differ when there are images
    // with bogus camera parameters, in which case some 2D-3D
    // correspondences are skipped above.
    if (tri_points2D.size() <
        static_cast<size_t>(options.abs_pose_min_num_inliers)) {
        VLOG(2) << "Insufficient number of 2D-3D correspondences for "
                   "registration ("
                << tri_points2D.size() << " < "
                << options.abs_pose_min_num_inliers << ")";
        return false;
    }

    // 2D-3D estimation. Only refine focal length if no focal length was
    // specified (manually or through EXIF) and if it was not already
    // estimated previously from another image sharing the camera.
    RANSACOptions abs_pose_options;
    abs_pose_options.max_error = options.abs_pose_max_error;
    abs_pose_options.min_inlier_ratio = options.abs_pose_min_inlier_ratio;

    AbsolutePoseRefinementOptions abs_pose_refinement_options;
    abs_pose_refinement_options.refine_focal_length = false;
    abs_pose_refinement_options.refine_extra_params = false;

    size_t num_inliers;
    std::vector<char> inlier_mask;
    Rigid3d rig_from_world;
    if (!EstimateGeneralizedAbsolutePose(abs_pose_options,
                                         tri_points2D,
                                         tri_points3D,
                                         tri_camera_idxs,
                                         cams_from_rig,
                                         cameras,
                                         &rig_from_world,
                                         &num_inliers,
                                         &inlier_mask)) {
        VLOG(2) << "Absolute pose estimation failed";
        return false;
    }

    if (num_inliers < static_cast<size_t>(options.abs_pose_min_num_inliers)) {
        VLOG(2) << "Absolute pose estimation failed due to insufficient "
                   "inliers ("
                << num_inliers << " < "
                << options.abs_pose_min_num_inliers << ")";
        return false;
    }

    // Pose refinement.
    if (!RefineGeneralizedAbsolutePose(abs_pose_refinement_options,
                                       inlier_mask,
                                       tri_points2D,
                                       tri_points3D,
                                       tri_camera_idxs,
                                       cams_from_rig,
                                       &rig_from_world,
                                       &cameras)) {
        VLOG(2) << "Absolute pose refinement failed";
        return false;
    }

    // Continue tracks.
    VLOG(2) << "Continuing tracks for " << num_inliers
            << " inlier 2D-3D correspondences";

    frame.SetRigFromWorld(rig_from_world);

    obs_manager_->RegisterFrame(frame.FrameId());
    RegisterFrameEvent(frame.FrameId());

    for (size_t i = 0; i < inlier_mask.size(); ++i) {
        if (inlier_mask[i]) {
            const Corr& corr = tri_corrs[i];
            const Image& image = reconstruction_->Image(corr.image_id);
            const Point2D& point2D = image.Point2D(corr.point2D_idx);
            if (!point2D.HasPoint3D()) {
                const TrackElement track_el(corr.image_id, corr.point2D_idx);
                obs_manager_->AddObservation(corr.point3D_id, track_el);
                triangulator_->AddModifiedPoint3D(corr.point3D_id);
            }
        }
    }

    return true;
}

// Upstream parity (d3ccaf35 sfm/incremental_mapper.cc
// RegisterNextStructureLessImage): structure-less resectioning from 2D-2D
// correspondences to already registered images, with robust triangulation
// of new points for the inlier correspondences.
bool IncrementalMapper::RegisterNextStructureLessImage(
        const Options& options, const image_t image_id) {
    THROW_CHECK_NOTNULL(reconstruction_);
    THROW_CHECK_NOTNULL(obs_manager_);
    if (reconstruction_->NumRegImages() < 2) {
        VLOG(2) << "Structure-less registration requires at least 2 "
                   "registered images; only "
                << reconstruction_->NumRegImages() << " available";
        return false;
    }

    THROW_CHECK(options.Check());

    num_structure_less_reg_trials_[image_id] += 1;

    Image& image = reconstruction_->Image(image_id);
    Camera& camera = reconstruction_->Camera(image.CameraId());

    // Each 2D-2D correspondence contributes 1 geometric constraint, whereas
    // each 2D-3D correspondence contributes 2, so require 2x the number of
    // inliers.
    const size_t min_num_inliers = 2 * options.abs_pose_min_num_inliers;

    // Check if enough 2D-2D correspondences.
    if (obs_manager_->NumVisibleCorrespondences(image_id) <
        min_num_inliers) {
        return false;
    }

    const auto correspondence_graph = database_cache_->CorrespondenceGraph();

    std::vector<point2D_t> point2D_idxs;
    std::vector<CorrespondenceGraph::Correspondence> corrs;
    std::vector<Eigen::Vector2d> points2D;
    std::vector<Eigen::Vector2d> world_points2D;
    std::vector<size_t> world_camera_idxs;
    std::vector<Rigid3d> world_cams_from_world;
    std::vector<Camera> world_cameras;
    FlatHashMap<image_t, size_t> world_image_id_to_camera_idx;

    const point2D_t num_points2D = image.NumPoints2D();
    for (point2D_t point2D_idx = 0; point2D_idx < num_points2D;
         ++point2D_idx) {
        const Point2D& point2D = image.Point2D(point2D_idx);

        const auto corr_range = correspondence_graph->FindCorrespondences(
                image_id, point2D_idx);
        for (const auto* corr = corr_range.beg; corr < corr_range.end;
             ++corr) {
            const Image& world_image = reconstruction_->Image(corr->image_id);
            if (!world_image.HasPose()) {
                continue;
            }

            const Camera& world_camera = *world_image.CameraPtr();

            // Avoid correspondences to images with bogus camera parameters.
            if (world_camera.HasBogusParams(options.min_focal_length_ratio,
                                            options.max_focal_length_ratio,
                                            options.max_extra_param)) {
                continue;
            }

            world_points2D.push_back(
                    world_image.Point2D(corr->point2D_idx).XY());
            points2D.push_back(point2D.XY());

            const auto res = world_image_id_to_camera_idx.emplace(
                    corr->image_id, world_cameras.size());
            if (res.second) {
                world_cams_from_world.push_back(
                        world_image.CamFromWorld());
                world_cameras.push_back(world_camera);
            }

            world_camera_idxs.push_back(res.first->second);

            point2D_idxs.push_back(point2D_idx);
            corrs.push_back(*corr);
        }
    }

    // Check if we pass the minimum number of inliers.
    if (world_points2D.size() < min_num_inliers) {
        VLOG(2) << "Image observes insufficient number of points for "
                   "registration ("
                << world_points2D.size() << " < " << min_num_inliers << ")";
        return false;
    }

    // Structure-less resectioning. It uses epipolar Sampson error, so the
    // acceptance threshold is stricter than for structure-based
    // resectioning. The minimal solver is expensive, hence multi-threading.
    StructureLessAbsolutePoseEstimationOptions abs_pose_options;
    abs_pose_options.ransac_options.max_error =
            0.5 * options.abs_pose_max_error;
    abs_pose_options.ransac_options.min_inlier_ratio =
            options.abs_pose_min_inlier_ratio;

    BundleAdjustmentOptions abs_pose_refinement_options;
    abs_pose_refinement_options.loss_function_type =
            BundleAdjustmentOptions::LossFunctionType::CAUCHY;
    abs_pose_refinement_options.solver_options.logging_type =
            ceres::LoggingType::SILENT;
    abs_pose_refinement_options.print_summary = false;
    if (num_reg_images_per_camera_[image.CameraId()] > 0) {
        // Camera already refined from another image with the same camera.
        if (camera.HasBogusParams(options.min_focal_length_ratio,
                                  options.max_focal_length_ratio,
                                  options.max_extra_param)) {
            // Previously refined camera has bogus parameters, so reset
            // parameters and try to re-estimate them.
            camera.SetParams(
                    database_cache_->Camera(image.CameraId()).Params());
            abs_pose_refinement_options.refine_focal_length = true;
            abs_pose_refinement_options.refine_extra_params = true;
        } else {
            abs_pose_refinement_options.refine_focal_length = false;
            abs_pose_refinement_options.refine_extra_params = false;
        }
    } else {
        // Camera not refined before. Note that the camera parameters might
        // have been changed before but the image was filtered, so we
        // explicitly reset the camera parameters and try to re-estimate
        // them.
        camera.SetParams(database_cache_->Camera(image.CameraId()).Params());
        abs_pose_refinement_options.refine_focal_length = true;
        abs_pose_refinement_options.refine_extra_params = true;
    }

    if (!options.abs_pose_refine_focal_length) {
        abs_pose_refinement_options.refine_focal_length = false;
    }

    if (!options.abs_pose_refine_extra_params) {
        abs_pose_refinement_options.refine_extra_params = false;
    }

    size_t num_inliers;
    std::vector<char> inlier_mask;
    Rigid3d cam_from_world;
    if (!EstimateStructureLessAbsolutePose(abs_pose_options,
                                           points2D,
                                           world_points2D,
                                           world_camera_idxs,
                                           world_cams_from_world,
                                           world_cameras,
                                           camera,
                                           &cam_from_world,
                                           &num_inliers,
                                           &inlier_mask)) {
        VLOG(2) << "Absolute pose estimation failed";
        return false;
    }

    if (num_inliers < min_num_inliers) {
        VLOG(2) << "Absolute pose estimation failed due to insufficient "
                   "inliers ("
                << num_inliers << " < " << min_num_inliers << ")";
        return false;
    }

    // Continue or triangulate tracks.
    VLOG(2) << "Continuing or triangulating tracks for " << num_inliers
            << " inlier 2D-2D correspondences";

    image.SetCamFromWorld(cam_from_world);

    obs_manager_->RegisterFrame(image.FrameId());
    RegisterFrameEvent(image.FrameId());

    THROW_CHECK_EQ(point2D_idxs.size(), corrs.size());
    THROW_CHECK_EQ(point2D_idxs.size(), inlier_mask.size());
    std::vector<std::vector<CorrespondenceGraph::Correspondence>>
            inlier_corrs(num_points2D);
    for (size_t i = 0; i < inlier_mask.size(); ++i) {
        if (inlier_mask[i]) {
            inlier_corrs[point2D_idxs[i]].push_back(corrs[i]);
        }
    }

    BundleAdjustmentConfig abs_pose_refinement_config;
    abs_pose_refinement_config.AddImage(image_id);

    for (point2D_t point2D_idx = 0; point2D_idx < num_points2D;
         ++point2D_idx) {
        if (inlier_corrs[point2D_idx].empty()) {
            continue;
        }

        // Check if any of the corresponding inlier points is already
        // triangulated. Simply add the current 2D point to the first track
        // we find.
        bool continued_track = false;
        for (const auto& corr : inlier_corrs[point2D_idx]) {
            const Image& corr_image = reconstruction_->Image(corr.image_id);
            const Point2D& corr_point2D =
                    corr_image.Point2D(corr.point2D_idx);
            if (corr_point2D.HasPoint3D()) {
                obs_manager_->AddObservation(
                        corr_point2D.Point3DId(),
                        TrackElement(image_id, point2D_idx));
                triangulator_->AddModifiedPoint3D(
                        corr_point2D.Point3DId());
                continued_track = true;
                break;
            }
        }

        if (continued_track) {
            continue;
        }

        // Otherwise, robustly triangulate a new point. Upstream parity
        // (d3ccaf35): the estimation entry point takes pixel observations and
        // camera poses directly.
        std::vector<Eigen::Vector2d> tri_points;
        std::vector<Rigid3d> tri_cams_from_world;
        std::vector<const Camera*> tri_cameras;
        for (const auto& corr : inlier_corrs[point2D_idx]) {
            const Image& corr_image = reconstruction_->Image(corr.image_id);
            const Camera& corr_camera =
                    reconstruction_->Camera(corr_image.CameraId());
            tri_points.emplace_back(
                    corr_image.Point2D(corr.point2D_idx).XY());
            tri_cams_from_world.push_back(corr_image.CamFromWorld());
            tri_cameras.push_back(&corr_camera);
        }

        tri_points.emplace_back(image.Point2D(point2D_idx).XY());
        tri_cams_from_world.push_back(image.CamFromWorld());
        tri_cameras.push_back(&camera);

        Eigen::Vector3d tri_xyz;
        EstimateTriangulationOptions tri_options;
        tri_options.min_tri_angle = DegToRad(options.filter_min_tri_angle);
        tri_options.ransac_options.max_error = options.abs_pose_max_error;
        std::vector<char> tri_inlier_mask;
        if (!EstimateTriangulation(tri_options,
                                   tri_points,
                                   tri_cams_from_world,
                                   tri_cameras,
                                   &tri_inlier_mask,
                                   &tri_xyz) ||
            !tri_inlier_mask.back()) {
            // Skip this 2D point if we failed to triangulate and if it is
            // itself not in the inlier set.
            continue;
        }

        Track track;
        track.AddElement(image_id, point2D_idx);
        for (size_t i = 0; i < tri_inlier_mask.size() - 1; ++i) {
            if (tri_inlier_mask[i]) {
                const auto& inlier_corr = inlier_corrs[point2D_idx][i];
                track.AddElement(inlier_corr.image_id,
                                 inlier_corr.point2D_idx);
            }
        }

        const point3D_t point3D_id =
                obs_manager_->AddPoint3D(tri_xyz, track);
        triangulator_->AddModifiedPoint3D(point3D_id);
        abs_pose_refinement_config.AddVariablePoint(point3D_id);
    }

    return true;
}

bool IncrementalMapper::EstimateInitialTwoViewGeometry(
        const Options& options,
        const image_t image_id1,
        const image_t image_id2) {
    const image_pair_t image_pair_id =
            Database::ImagePairToPairId(image_id1, image_id2);

    if (prev_init_image_pair_id_ == image_pair_id) {
        return true;
    }

    const Image& image1 = database_cache_->Image(image_id1);
    const Camera& camera1 = database_cache_->Camera(image1.CameraId());

    const Image& image2 = database_cache_->Image(image_id2);
    const Camera& camera2 = database_cache_->Camera(image2.CameraId());

    const CorrespondenceGraph& correspondence_graph =
            *database_cache_->CorrespondenceGraph();
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

    TwoViewGeometry two_view_geometry;
    TwoViewGeometryOptions two_view_geometry_options;
    two_view_geometry_options.ransac_options.min_num_trials = 30;
    two_view_geometry_options.ransac_options.max_error = options.init_max_error;
    // The upstream estimator folds the relative-pose decomposition into the
    // estimation, so cam2_from_cam1 carries what the legacy member call
    // EstimateRelativePose produced.
    two_view_geometry = EstimateTwoViewGeometry(camera1, points1, camera2,
                                                points2, matches,
                                                two_view_geometry_options);

    // Upstream parity (d3ccaf35 sfm/incremental_mapper_impl.cc): the plain
    // estimation only classifies the configuration; the relative pose and
    // the triangulation angle are recovered by a dedicated decomposition
    // pass that must run before the initialization checks.
    if (!EstimateTwoViewGeometryPose(camera1, points1, camera2, points2,
                                     &two_view_geometry)) {
        return false;
    }

    if (!two_view_geometry.cam2_from_cam1) {
        return false;
    }

    VLOG(3) << "Initial image pair with config "
            << two_view_geometry.config << ", "
            << two_view_geometry.inlier_matches.size() << " inlier matches, "
            << two_view_geometry.cam2_from_cam1->translation().z()
            << " z translation, "
            << RadToDeg(two_view_geometry.tri_angle)
            << " deg triangulation angle";
    if (static_cast<int>(two_view_geometry.inlier_matches.size()) >=
                options.init_min_num_inliers &&
        std::abs(two_view_geometry.cam2_from_cam1->translation().z()) <
                options.init_max_forward_motion &&
        two_view_geometry.tri_angle > DegToRad(options.init_min_tri_angle)) {
        prev_init_image_pair_id_ = image_pair_id;
        prev_init_two_view_geometry_ = two_view_geometry;
        return true;
    }

    return false;
}

}  // namespace colmap
