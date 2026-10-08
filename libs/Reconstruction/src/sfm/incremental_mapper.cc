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

#include "util/hash_containers.h"
#include <ceres/ceres.h>

#include "estimators/triangulation.h"
#include "estimators/two_view_geometry.h"
#include "sfm/observation_manager.h"
#include "sfm/incremental_mapper.h"
#include "sfm/incremental_mapper_impl.h"

#include "geometry/pose_prior.h"
#include "scene/camera.h"
#include "scene/reconstruction_pruning.h"

#include <array>
#include <fstream>
#include <optional>

#include "scene/projection.h"
#include "geometry/triangulation.h"
#include "estimators/pose.h"
#include "sensor/bitmap.h"
#include "util/misc.h"

namespace colmap {
namespace {

}  // namespace

// Upstream parity (d3ccaf35 sfm/incremental_mapper.cc): seed the intrinsics
// estimated by two-view solvers (e.g. shared focal) into the reconstruction
// cameras of the initial pair before registration.
void SeedEstimatedInitialCameras(Reconstruction& reconstruction,
                                 const image_t image_id1,
                                 const image_t image_id2,
                                 const std::optional<Camera>& camera1,
                                 const std::optional<Camera>& camera2) {
    if (camera1.has_value()) {
        *reconstruction.Image(image_id1).CameraPtr() = *camera1;
    }
    if (camera2.has_value()) {
        *reconstruction.Image(image_id2).CameraPtr() = *camera2;
    }
}

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
    CHECK_OPTION_GE(random_seed, -1);
    return true;
}

IncrementalMapper::IncrementalMapper(const DatabaseCache* database_cache)
    : database_cache_(database_cache),
      reconstruction_(nullptr),
      triangulator_(nullptr) {}

void IncrementalMapper::BeginReconstruction(Reconstruction* reconstruction) {
    CHECK(reconstruction_ == nullptr);
    reconstruction_ = reconstruction;
    // Snapshot the frames that already belong to a continued reconstruction
    // before the cache load (batch C observation-bookkeeping fix). Batch T:
    // RegFrameIds() now returns a reference to the fork's registration-order
    // vector (upstream parity); copy into the hash set explicitly.
    const NodeHashSet<frame_t> pre_existing_frames(
            reconstruction->RegFrameIds().begin(),
            reconstruction->RegFrameIds().end());
    reconstruction_->Load(*database_cache_);
    // Fork fix (surfaced by IncrementalMapperTest.EndReconstructionDiscard /
    // exe/sfm_test WritesInterruptedReconstruction): the fork database
    // round-trip persists frame poses (a fork extension), so Load() brings
    // cache frames in as "registered" through AddFrame's pose rule while the
    // observation bookkeeping is still empty - registration and
    // visible-correspondence accounting would be permanently out of sync
    // (visible-correspondence underflow on EndReconstruction(discard)), and
    // FindNextImages' HasPose() proxy would skip every frame, leaving no
    // candidates for the initial pair. Upstream does not hit this because its
    // database frames carry no poses. Undo the pose-based registration for
    // the frames the load introduced, so incremental SfM re-establishes
    // registration and accounting together; frames of a continued
    // reconstruction (registered before the load) are kept.
    // Batch T: de-registration mutates reg_frame_ids_, so iterate over a
    // copy (the old NodeHashSet return was already a detached snapshot).
    const std::vector<frame_t> post_load_reg_frames =
            reconstruction_->RegFrameIds();
    for (const frame_t frame_id : post_load_reg_frames) {
        if (pre_existing_frames.count(frame_id) == 0) {
            reconstruction_->DeRegisterFrame(frame_id);
        }
    }
    // Own the graph: the observation manager holds the graph shared_ptr so it
    // outlives this controller's cache in continue-scenarios (batch C
    // graph-lifetime fix). Upstream parity: the reconstruction carries no
    // graph state - the observation manager is the sole visibility ledger.
    obs_manager_ = std::make_shared<class ObservationManager>(
            *reconstruction, database_cache_->CorrespondenceGraph());
    // Upstream parity (d3ccaf35): the triangulator shares the mapper's
    // observation manager so point operations keep the visibility statistics
    // in sync with the reconstruction.
    triangulator_.reset(new IncrementalTriangulator(
            database_cache_->CorrespondenceGraph().get(), reconstruction,
            obs_manager_));

    reg_stats_.num_shared_reg_images = 0;
    reg_stats_.num_reg_images_per_camera.clear();
    reg_stats_.num_reg_frames_per_rig.clear();
    // Upstream parity (d3ccaf35): account for the already-registered frames
    // of a continued reconstruction at the frame level.
    for (const frame_t frame_id : reconstruction_->RegFrameIds()) {
        RegisterFrameEvent(frame_id);
    }

    existing_frame_ids_ =
            NodeHashSet<frame_t>(reconstruction->RegFrameIds().begin(),
                                 reconstruction->RegFrameIds().end());

    filtered_frames_.clear();
    reg_stats_.num_reg_trials.clear();
    reg_stats_.num_structure_less_reg_trials.clear();
}

void IncrementalMapper::EndReconstruction(const bool discard) {
    // Upstream parity: idempotent against a second end/reinit cycle (a
    // discard already ended the reconstruction; upstream's fixture guards
    // via the nullable Reconstruction() accessor, which the fork's raw
    // pointer surface expresses as an early return here).
    if (reconstruction_ == nullptr) {
        return;
    }

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
                                             image_t& image_id1,
                                             image_t& image_id2,
                                             Rigid3d& cam2_from_cam1) {
    const std::optional<IncrementalMapperImpl::InitInfo> init_info =
            IncrementalMapperImpl::FindInitialImagePair(
                    options,
                    *database_cache_,
                    *reconstruction_,
                    reg_stats_.init_num_reg_trials,
                    reg_stats_.num_registrations,
                    reg_stats_.init_image_pairs,
                    image_id1,
                    image_id2);
    if (!init_info.has_value()) {
        // No suitable pair found in entire dataset.
        image_id1 = kInvalidImageId;
        image_id2 = kInvalidImageId;
        return false;
    }
    image_id1 = init_info->image_id1;
    image_id2 = init_info->image_id2;
    cam2_from_cam1 = init_info->cam2_from_cam1;
    // Upstream parity (d3ccaf35): surface the intrinsics estimated by the
    // two-view solver (if any) into the reconstruction cameras before
    // registration.
    SeedEstimatedInitialCameras(*reconstruction_, image_id1, image_id2,
                                init_info->camera1, init_info->camera2);
    return true;
}

std::vector<image_t> IncrementalMapper::FindNextImages(
        const Options& options, const bool structure_less) {
    CHECK_NOTNULL(reconstruction_);
    CHECK(options.Check());
    // Upstream parity (d3ccaf35): structure-less registration tracks its
    // own trial counter so that failed structure-less attempts do not
    // consume the structure-based registration budget.
    return IncrementalMapperImpl::FindNextImages(
            options, *obs_manager_, filtered_frames_,
            structure_less ? reg_stats_.num_structure_less_reg_trials
                           : reg_stats_.num_reg_trials,
            /*structure_less=*/structure_less);
}

bool IncrementalMapper::RegisterInitialImagePair(
        const Options& options,
        const image_t image_id1,
        const image_t image_id2,
        const Rigid3d& cam2_from_cam1) {
    CHECK_NOTNULL(reconstruction_);
    CHECK_EQ(reconstruction_->NumRegImages(), 0);

    CHECK(options.Check());

    reg_stats_.init_num_reg_trials[image_id1] += 1;
    reg_stats_.init_num_reg_trials[image_id2] += 1;
    reg_stats_.num_reg_trials[image_id1] += 1;
    reg_stats_.num_reg_trials[image_id2] += 1;

    const image_pair_t pair_id = ImagePairToPairId(image_id1, image_id2);
    reg_stats_.init_image_pairs.insert(pair_id);

    Image& image1 = reconstruction_->Image(image_id1);

    Image& image2 = reconstruction_->Image(image_id2);

    //////////////////////////////////////////////////////////////////////////////
    // Apply two-view geometry
    //////////////////////////////////////////////////////////////////////////////

    // Upstream parity (d3ccaf35): the relative pose comes from the caller
    // (FindInitialImagePair returned it); this function no longer re-estimates
    // the two-view geometry. Pose writes go through the frame-aware model
    // (Reconstruction images are frame-wired by Reconstruction::Load, so the
    // former legacy qvec/tvec fallback branch was dead code and is retired).
    image1.SetCamFromWorld(Rigid3d());
    image2.SetCamFromWorld(cam2_from_cam1);

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
                 reg_stats_.num_reg_images_per_camera[frame_camera.CameraId()] == 0) ||
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

    reg_stats_.num_reg_trials[image_id] += 1;

    // Check if enough 2D-3D correspondences (upstream parity: the count is
    // read from the ObservationManager, the single ledger maintained by the
    // OM-era triangulation path; the fork's legacy Image-level counter was
    // stale and gated registrations off).
    if (obs_manager_->NumVisiblePoints3D(image_id) <
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

        NodeHashSet<point3D_t> point3D_ids;

        for (const auto corr : corrs) {
            if (!reconstruction_->ExistsImage(corr.image_id)) {
                std::cout << "[IncrementalMapper::RegisterNextImage] Image "
                          << corr.image_id << " does not exist" << std::endl;
                continue;
            }

            const Image& corr_image = reconstruction_->Image(corr.image_id);
            // Upstream parity (d3ccaf35): gate on the frame pose.
            if (!corr_image.HasPose()) {
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
    // Upstream parity (d3ccaf35): the registration RANSAC consumes the mapper
    // seed; negative values fall back to the ambient PRNG stream.
    abs_pose_options.ransac_options.random_seed = options.random_seed;
    // Use high confidence to avoid preemptive termination of P3P RANSAC
    // - too early termination may lead to bad registration.
    abs_pose_options.ransac_options.min_num_trials = 100;
    abs_pose_options.ransac_options.max_num_trials = 10000;
    abs_pose_options.ransac_options.confidence = 0.99999;

    AbsolutePoseRefinementOptions abs_pose_refinement_options;
    if (options.constant_cameras.count(image.CameraId()) > 0) {
        // Upstream parity (d3ccaf35): cameras marked constant never get
        // their intrinsics estimated or refined during registration.
        abs_pose_options.estimate_focal_length = false;
        abs_pose_refinement_options.refine_focal_length = false;
        abs_pose_refinement_options.refine_extra_params = false;
    } else if (reg_stats_.num_reg_images_per_camera[image.CameraId()] > 0) {
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

    // Upstream parity (d3ccaf35 sfm/incremental_mapper.cc): non-perspective
    // cameras (spherical / equirectangular) must take the camera-ray P3P
    // branch of EstimateAbsolutePose - the P4PF branch reads the principal
    // point indices, which are empty for these models and previously aborted
    // the process.
    if (!camera.IsPerspective()) {
        abs_pose_options.estimate_focal_length = false;
        abs_pose_refinement_options.refine_focal_length = false;
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
        const NodeHashSet<point3D_t>& point3D_ids) {
    CHECK_NOTNULL(reconstruction_);
    CHECK(options.Check());

    LocalBundleAdjustmentReport report;

    // Find images that have most 3D points with given image in common.
    const std::vector<image_t> local_bundle =
            IncrementalMapperImpl::FindLocalBundle(options, image_id,
                                                   *reconstruction_);

    // Do the bundle adjustment only if there is any connected images.
    if (local_bundle.size() > 0) {
        BundleAdjustmentConfig ba_config;
        ba_config.FixGauge(BundleAdjustmentGauge::THREE_POINTS);

        // Insert the images of all local frames (upstream d3ccaf35: the
        // whole frame joins the local bundle so that multi-sensor rigs are
        // adjusted jointly; for single-sensor rigs this equals the old
        // per-image insertion).
        NodeHashSet<frame_t> frame_ids;
        ba_config.AddImage(image_id);
        for (const image_t local_image_id : local_bundle) {
            const Image& local_image = reconstruction_->Image(local_image_id);
            frame_ids.insert(local_image.FrameId());
            ba_config.AddImage(local_image_id);
        }

        // Fix the existing frames, if option specified (the fork keeps the
        // fix_existing_images option name for CLI compatibility).
        if (options.fix_existing_images) {
            for (const frame_t frame_id : frame_ids) {
                if (existing_frame_ids_.count(frame_id)) {
                    ba_config.SetConstantRigFromWorldPose(frame_id);
                }
            }
        }

        // Fix rig poses, if not all frames within the local bundle
        // (upstream d3ccaf35).
        NodeHashMap<rig_t, size_t> num_frames_per_rig;
        num_frames_per_rig.reserve(frame_ids.size());
        for (const frame_t frame_id : frame_ids) {
            const Frame& frame = reconstruction_->Frame(frame_id);
            num_frames_per_rig[frame.RigId()] += 1;
        }
        for (const auto& rig_id_and_num_frames : num_frames_per_rig) {
            if (options.constant_rigs.count(rig_id_and_num_frames.first) ||
                rig_id_and_num_frames.second <
                        reg_stats_.num_reg_frames_per_rig.at(
                                rig_id_and_num_frames.first)) {
                const Rig& rig =
                        reconstruction_->Rig(rig_id_and_num_frames.first);
                for (const auto& sensor_id_and_pose : rig.NonRefSensors()) {
                    ba_config.SetConstantSensorFromRigPose(
                            sensor_id_and_pose.first);
                }
            }
        }

        // Determine which cameras to fix, when not all the registered images
        // are within the current local bundle.
        NodeHashMap<camera_t, size_t> num_images_per_camera;
        num_images_per_camera.reserve(ba_config.NumImages());
        for (const image_t config_image_id : ba_config.Images()) {
            const Image& config_image =
                    reconstruction_->Image(config_image_id);
            num_images_per_camera[config_image.CameraId()] += 1;
        }

        for (const auto& camera_id_and_num_images : num_images_per_camera) {
            if (options.constant_cameras.count(
                        camera_id_and_num_images.first) ||
                camera_id_and_num_images.second <
                        reg_stats_.num_reg_images_per_camera.at(
                                camera_id_and_num_images.first)) {
                ba_config.SetConstantCamera(
                        camera_id_and_num_images.first);
            }
        }

        // Make sure, we refine all new and short-track 3D points, no matter if
        // they are fully contained in the local image set or not. Do not
        // include long track 3D points as they are usually already very stable
        // and adding to them to bundle adjustment and track merging/completion
        // would slow down the local bundle adjustment significantly.
        NodeHashSet<point3D_t> variable_point3D_ids;
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
    NodeHashSet<image_t> filter_image_ids;
    filter_image_ids.insert(image_id);
    filter_image_ids.insert(local_bundle.begin(), local_bundle.end());
    // Upstream parity (d3ccaf35 sfm/incremental_mapper.cc): the post-BA
    // filtering must run through the ObservationManager so the triangulated
    // correspondence bookkeeping is decremented before the observations are
    // cleared. The fork's Reconstruction-level filters maintain a legacy
    // private copy of the pair stats and leave the manager's counters
    // inflated, which later trips the num_tri_corrs guard
    // (surfaced by the non-trivial-rig pipeline cases, batch C).
    report.num_filtered_observations =
            obs_manager_->FilterPoints3DInImages(
                    options.filter_max_reproj_error,
                    options.filter_min_tri_angle, filter_image_ids);
    report.num_filtered_observations += obs_manager_->FilterPoints3D(
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
    // Upstream parity: filtering goes through the ObservationManager (see
    // the AdjustLocalBundle note above).
    obs_manager_->FilterObservationsWithNegativeDepth();

    // Configure bundle adjustment.
    BundleAdjustmentConfig ba_config;
    for (const image_t image_id : reg_image_ids) {
        ba_config.AddImage(image_id);
    }

    // Fix the existing frames, if option specified (upstream d3ccaf35:
    // iterate registered frames directly).
    if (options.fix_existing_images) {
        for (const frame_t frame_id : reconstruction_->RegFrameIds()) {
            if (existing_frame_ids_.count(frame_id)) {
                ba_config.SetConstantRigFromWorldPose(frame_id);
            }
        }
    }

    // Fix rig poses for the configured constant rigs (upstream d3ccaf35).
    for (const rig_t rig_id : options.constant_rigs) {
        const Rig& rig = reconstruction_->Rig(rig_id);
        for (const auto& sensor_id_and_pose : rig.NonRefSensors()) {
            ba_config.SetConstantSensorFromRigPose(sensor_id_and_pose.first);
        }
    }

    // Fix camera intrinsics for the configured constant cameras
    // (upstream d3ccaf35; fork keeps the SetConstantCamera name).
    for (const camera_t camera_id : options.constant_cameras) {
        ba_config.SetConstantCamera(camera_id);
    }

    // Ignore redundant 3D points in the first solve, if option specified
    // (upstream parity; only past the "small reconstruction" threshold).
    std::vector<point3D_t> redundant_point3D_ids;
    if (!is_small_reconstruction &&
        options.ba_global_ignore_redundant_points3D) {
        redundant_point3D_ids = FindRedundantPoints3D(
                        options
                                .ba_global_ignore_redundant_points3D_min_coverage_gain,
                        *reconstruction_);
        VLOG(1) << "=> Ignoring " << redundant_point3D_ids.size() << " / "
                << reconstruction_->NumPoints3D() << " redundant 3D points";
        for (const point3D_t point3D_id : redundant_point3D_ids) {
            ba_config.IgnorePoint(point3D_id);
        }
    }

    // Run bundle adjustment (upstream parity, d3ccaf35): with position
    // priors the gauge is fixed by the priors themselves, so no explicit
    // gauge fixation is applied; without priors the gauge is fixed with the
    // two-cameras heuristic (more stable and fewer steps than fixing three
    // points). The old fork form fixed image0's full pose plus one tvec
    // component unconditionally, which fought the priors in
    // use_prior_position runs and degraded their convergence.
    const bool use_prior_position =
            options.use_prior_position &&
            NumRegisteredPosePriors(database_cache_->PosePriors(),
                                    ba_config) >= 3;
    std::unique_ptr<BundleAdjuster> bundle_adjuster;
    if (!use_prior_position) {
        ba_config.FixGauge(BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);
        bundle_adjuster =
                CreateDefaultBundleAdjuster(custom_ba_options, ba_config);
    } else {
        PosePriorBundleAdjustmentOptions prior_options;
        if (options.use_robust_loss_on_prior_position) {
            prior_options.prior_position_loss_function_type =
                    BundleAdjustmentOptions::LossFunctionType::CAUCHY;
        }
        prior_options.prior_position_loss_scale =
                options.prior_position_loss_scale;
        prior_options.alignment_ransac_options.random_seed =
                options.random_seed;
        bundle_adjuster =
                CreateDefaultBundleAdjuster(custom_ba_options, ba_config);
        bundle_adjuster->SetPosePriors(prior_options,
                                       database_cache_->PosePriors());
    }
    if (!bundle_adjuster->Solve(reconstruction_)) {
        return false;
    }

    // Optimize the redundant 3D points with all other parameters fixed
    // (upstream parity, d3ccaf35): the first solve ignored them, so a
    // second dedicated solve pulls them back in without disturbing the
    // rest of the reconstruction.
    if (!is_small_reconstruction &&
        options.ba_global_ignore_redundant_points3D) {
        if (custom_ba_options.check_if_stopped &&
            custom_ba_options.check_if_stopped()) {
            return true;
        }

        ba_config = BundleAdjustmentConfig();
        for (const point3D_t point3D_id : redundant_point3D_ids) {
            ba_config.AddVariablePoint(point3D_id);
        }
        for (const frame_t frame_id : reconstruction_->RegFrameIds()) {
            ba_config.SetConstantRigFromWorldPose(frame_id);
        }
        for (const auto& camera_id_and_camera : reconstruction_->Cameras()) {
            ba_config.SetConstantCamera(camera_id_and_camera.first);
        }
        for (const auto& rig_id_and_rig : reconstruction_->Rigs()) {
            for (const auto& sensor_id_and_pose :
                    rig_id_and_rig.second.NonRefSensors()) {
                ba_config.SetConstantSensorFromRigPose(
                        sensor_id_and_pose.first);
            }
        }

        bundle_adjuster =
                CreateDefaultBundleAdjuster(custom_ba_options, ba_config);
        if (!bundle_adjuster->Solve(reconstruction_)) {
            return false;
        }
    }

    return true;
}


size_t IncrementalMapper::FilterFrames(const Options& options) {
    CHECK_NOTNULL(reconstruction_);
    CHECK_NOTNULL(obs_manager_);
    CHECK(options.Check());

    // Do not filter frames in the early stage of the reconstruction, since the
    // calibration is often still refining a lot. Hence, the camera parameters
    // are not stable in the beginning. (Upstream parity, d3ccaf35: filtering
    // and bookkeeping happen at frame granularity.)
    const size_t kMinNumFrames = 20;
    if (reconstruction_->NumRegFrames() < kMinNumFrames) {
        return {};
    }

    const std::vector<frame_t> filter_frame_ids =
            obs_manager_->FindFramesToFilter(
                    /*min_focal_length_ratio=*/options.min_focal_length_ratio,
                    /*max_focal_length_ratio=*/options.max_focal_length_ratio,
                    /*max_extra_param=*/options.max_extra_param,
                    /*min_num_observations=*/1);

    size_t num_filtered = 0;
    for (const frame_t frame_id : filter_frame_ids) {
        // The fork keeps the fix_existing_images option name for CLI
        // compatibility; the gate is frame-level (upstream
        // fix_existing_frames semantics).
        if (!options.fix_existing_images ||
            existing_frame_ids_.count(frame_id) == 0) {
            obs_manager_->DeRegisterFrame(frame_id);
            DeRegisterFrameEvent(frame_id);
            filtered_frames_.insert(frame_id);
            ++num_filtered;
        }
    }

    VLOG(1) << "=> Filtered frames: " << num_filtered;
    return num_filtered;
}

size_t IncrementalMapper::FilterPoints(const Options& options) {
    CHECK_NOTNULL(reconstruction_);
    CHECK(options.Check());
    // Upstream parity (d3ccaf35): filtering goes through the
    // ObservationManager so the triangulation bookkeeping stays in sync.
    return obs_manager_->FilterAllPoints3D(options.filter_max_reproj_error,
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
    return reg_stats_.num_total_reg_images;
}

size_t IncrementalMapper::NumSharedRegImages() const {
    return reg_stats_.num_shared_reg_images;
}

const NodeHashSet<point3D_t>& IncrementalMapper::GetModifiedPoints3D() {
    return triangulator_->GetModifiedPoints3D();
}

void IncrementalMapper::ClearModifiedPoints3D() {
    triangulator_->ClearModifiedPoints3D();
}

void IncrementalMapper::IterativeLocalRefinement(
        const int max_num_refinements,
        const double max_refinement_change,
        const Options& options,
        const BundleAdjustmentOptions& ba_options,
        const IncrementalTriangulator::Options& tri_options,
        const image_t image_id) {
    BundleAdjustmentOptions custom_ba_options = ba_options;
    for (int i = 0; i < max_num_refinements; ++i) {
        if (custom_ba_options.check_if_stopped &&
            custom_ba_options.check_if_stopped()) {
            break;
        }
        const auto report = AdjustLocalBundle(options,
                                              custom_ba_options,
                                              tri_options,
                                              image_id,
                                              GetModifiedPoints3D());
        VLOG(1) << "=> Merged observations: " << report.num_merged_observations;
        VLOG(1) << "=> Completed observations: "
                << report.num_completed_observations;
        VLOG(1) << "=> Filtered observations: "
                << report.num_filtered_observations;
        const double changed =
                report.num_adjusted_observations == 0
                        ? 0
                        : (report.num_merged_observations +
                           report.num_completed_observations +
                           report.num_filtered_observations) /
                                  static_cast<double>(
                                          report.num_adjusted_observations);
        VLOG(1) << StringPrintf("=> Changed observations: %.6f", changed);
        if (changed < max_refinement_change) {
            break;
        }
        // Only use robust cost function for first iteration.
        custom_ba_options.loss_function_type =
                BundleAdjustmentOptions::LossFunctionType::TRIVIAL;
    }
    ClearModifiedPoints3D();
}

bool IncrementalMapper::EstimateInitialTwoViewGeometry(
        const Options& options,
        const image_t image_id1,
        const image_t image_id2,
        Rigid3d& cam2_from_cam1) {
    const std::optional<IncrementalMapperImpl::InitInfo> init_info =
            IncrementalMapperImpl::EstimateInitialTwoViewGeometry(
                    options, *database_cache_, image_id1, image_id2);
    if (!init_info.has_value()) {
        return false;
    }
    cam2_from_cam1 = init_info->cam2_from_cam1;
    SeedEstimatedInitialCameras(*reconstruction_, image_id1, image_id2,
                                init_info->camera1, init_info->camera2);
    return true;
}

// Upstream parity (d3ccaf35 sfm/incremental_mapper.cc): frame-level
// registration bookkeeping. Bumps the per-rig frame count and, for every
// image in the frame, the per-camera and per-image counters (there are no
// standalone image-level events).
void IncrementalMapper::RegisterFrameEvent(const frame_t frame_id) {
    const Frame& frame = reconstruction_->Frame(frame_id);

    size_t& num_reg_frames_for_rig = reg_stats_.num_reg_frames_per_rig[frame.RigId()];
    num_reg_frames_for_rig += 1;

    for (const image_t image_id : frame.ImageIds()) {
        const Image& image = reconstruction_->Image(image_id);
        size_t& num_reg_images_for_camera =
                reg_stats_.num_reg_images_per_camera[image.CameraId()];
        num_reg_images_for_camera += 1;

        size_t& num_regs_for_image = reg_stats_.num_registrations[image_id];
        num_regs_for_image += 1;
        if (num_regs_for_image == 1) {
            reg_stats_.num_total_reg_images += 1;
        } else if (num_regs_for_image > 1) {
            reg_stats_.num_shared_reg_images += 1;
        }
    }
}

void IncrementalMapper::DeRegisterFrameEvent(const frame_t frame_id) {
    const Frame& frame = reconstruction_->Frame(frame_id);

    size_t& num_reg_frames_for_rig =
            reg_stats_.num_reg_frames_per_rig.at(frame.RigId());
    CHECK_GT(num_reg_frames_for_rig, 0);
    num_reg_frames_for_rig -= 1;

    for (const image_t image_id : frame.ImageIds()) {
        const Image& image = reconstruction_->Image(image_id);
        size_t& num_reg_images_for_camera =
                reg_stats_.num_reg_images_per_camera.at(image.CameraId());
        CHECK_GT(num_reg_images_for_camera, 0);
        num_reg_images_for_camera -= 1;

        size_t& num_regs_for_image = reg_stats_.num_registrations[image_id];
        num_regs_for_image -= 1;
        if (num_regs_for_image == 0) {
            reg_stats_.num_total_reg_images -= 1;
        } else if (num_regs_for_image > 0) {
            reg_stats_.num_shared_reg_images -= 1;
        }
    }
}

const NodeHashMap<rig_t, size_t>&
IncrementalMapper::NumRegFramesPerRig() const {
    return reg_stats_.num_reg_frames_per_rig;
}

const NodeHashMap<camera_t, size_t>&
IncrementalMapper::NumRegImagesPerCamera() const {
    return reg_stats_.num_reg_images_per_camera;
}

class ObservationManager& IncrementalMapper::ObservationManager() const {
    CHECK_NOTNULL(obs_manager_);
    return *obs_manager_;
}

IncrementalTriangulator& IncrementalMapper::Triangulator() const {
    CHECK_NOTNULL(triangulator_);
    return *triangulator_;
}

const NodeHashSet<frame_t>& IncrementalMapper::FilteredFrames() const {
    return filtered_frames_;
}

const NodeHashSet<frame_t>& IncrementalMapper::ExistingFrameIds() const {
    return existing_frame_ids_;
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

        reg_stats_.num_reg_trials[image_id] += 1;

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

    reg_stats_.num_structure_less_reg_trials[image_id] += 1;

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
    if (reg_stats_.num_reg_images_per_camera[image.CameraId()] > 0) {
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


void IncrementalMapper::ResetInitializationStats() {
  reg_stats_.init_num_reg_trials.clear();
  reg_stats_.init_image_pairs.clear();
}

}  // namespace colmap
