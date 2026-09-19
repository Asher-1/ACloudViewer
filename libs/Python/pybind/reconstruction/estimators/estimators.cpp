// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "pybind/reconstruction/estimators/estimators.h"

#include <sstream>
#include <unordered_map>

#include "estimators/alignment.h"
#include "estimators/bundle_adjustment.h"
#include "estimators/bundle_adjustment_ceres.h"
#include "estimators/generalized_pose.h"
#include "estimators/global_positioning.h"
#include "estimators/gravity_refinement.h"
#include "estimators/pose.h"
#include "estimators/rotation_averaging.h"
#include "estimators/solvers/affine_transform.h"
#include "estimators/solvers/essential_matrix.h"
#include "estimators/solvers/fundamental_matrix.h"
#include "estimators/solvers/homography_matrix.h"
#include "estimators/solvers/similarity_transform.h"
#include "estimators/triangulation.h"
#include "estimators/two_view_geometry.h"
#include "geometry/pose.h"
#include "geometry/pose_prior.h"
#include "geometry/rigid3.h"
#include "geometry/triangulation.h"
#include "optim/loransac.h"
#include "optim/ransac.h"
#include "optim/support_measurement.h"
#include "pybind/docstring.h"
#include "pybind/reconstruction/estimators/covariance_bindings.h"
#include "scene/camera.h"
#include "scene/pose_graph.h"
#include "scene/two_view_geometry.h"
#include "util/logging.h"

namespace cloudViewer {
namespace reconstruction {
namespace estimators {

// The COLMAP fork engine types live in namespace colmap.
using colmap::AbsolutePoseEstimationOptions;
using colmap::AbsolutePoseRefinementOptions;
using colmap::Camera;
using colmap::EstimateTriangulationOptions;
using colmap::FeatureMatch;
using colmap::FeatureMatches;
using colmap::ImageAlignmentError;
using colmap::MEstimatorSupportMeasurer;
using colmap::RANSACOptions;
using colmap::Reconstruction;
using colmap::Rigid3d;
using colmap::Sim3d;
using colmap::TwoViewGeometry;
using colmap::TwoViewGeometryOptions;

// ndarray (N, d) helpers. Open3D-style bindings keep the Python side on
// plain numpy arrays; conversion cost is O(N) with no copies on the way out.
std::vector<Eigen::Vector2d> Points2DFromNdarray(
        const py::array_t<double, py::array::c_style | py::array::forcecast>&
                arr) {
    THROW_CHECK_EQ(arr.ndim(), 2);
    THROW_CHECK_EQ(arr.shape(1), 2);
    std::vector<Eigen::Vector2d> out(arr.shape(0));
    auto buf = arr.unchecked<2>();
    for (py::ssize_t i = 0; i < buf.shape(0); ++i) {
        out[i] = Eigen::Vector2d(buf(i, 0), buf(i, 1));
    }
    return out;
}

std::vector<Eigen::Vector3d> Points3DFromNdarray(
        const py::array_t<double, py::array::c_style | py::array::forcecast>&
                arr) {
    THROW_CHECK_EQ(arr.ndim(), 2);
    THROW_CHECK_EQ(arr.shape(1), 3);
    std::vector<Eigen::Vector3d> out(arr.shape(0));
    auto buf = arr.unchecked<2>();
    for (py::ssize_t i = 0; i < buf.shape(0); ++i) {
        out[i] = Eigen::Vector3d(buf(i, 0), buf(i, 1), buf(i, 2));
    }
    return out;
}

py::list InlierMaskToList(const std::vector<char>& mask) {
    py::list out;
    for (const char c : mask) {
        out.append(c != 0);
    }
    return out;
}

// (N, 2) uint32 ndarray -> FeatureMatches (upstream MatchesFromMatrix
// equivalent). Matches may be None to fall back to the identity pairing.
FeatureMatches MatchesFromOptionalNdarray(const py::object& matches) {
    FeatureMatches out;
    if (matches.is_none()) {
        return out;
    }
    auto arr = matches.cast<
            py::array_t<uint32_t, py::array::c_style | py::array::forcecast>>();
    THROW_CHECK_EQ(arr.ndim(), 2);
    THROW_CHECK_EQ(arr.shape(1), 2);
    auto buf = arr.unchecked<2>();
    out.reserve(buf.shape(0));
    for (py::ssize_t i = 0; i < buf.shape(0); ++i) {
        out.emplace_back(buf(i, 0), buf(i, 1));
    }
    return out;
}

py::array_t<uint32_t> MatchesToNdarray(const FeatureMatches& matches) {
    py::array_t<uint32_t> arr(
            {static_cast<py::ssize_t>(matches.size()), py::ssize_t(2)});
    auto buf = arr.mutable_unchecked<2>();
    for (size_t i = 0; i < matches.size(); ++i) {
        buf(i, 0) = matches[i].point2D_idx1;
        buf(i, 1) = matches[i].point2D_idx2;
    }
    return arr;
}

static const std::unordered_map<std::string, std::string>
        map_shared_argument_docstrings = {
                {"points2D", "Array of shape (N, 2) with 2D points."},
                {"points3D", "Array of shape (N, 3) with 3D points."},
                {"inlier_mask", "Length-N boolean inlier mask."},
                {"camera",
                 "Camera with intrinsics (modified in place when "
                 "refining the focal length)."},
};

void pybind_estimators(py::module& m) {
    py::module m_estimators = m.def_submodule("estimators");

    py::class_<RANSACOptions> ransac(
            m_estimators, "RANSACOptions",
            "Robust estimator options (shared by the estimator entry "
            "points).");
    ransac.def(py::init<>())
            .def_readwrite("max_error", &RANSACOptions::max_error)
            .def_readwrite("min_inlier_ratio", &RANSACOptions::min_inlier_ratio)
            .def_readwrite("confidence", &RANSACOptions::confidence)
            .def_readwrite("dyn_num_trials_multiplier",
                           &RANSACOptions::dyn_num_trials_multiplier)
            .def_readwrite("min_num_trials", &RANSACOptions::min_num_trials)
            .def_readwrite("max_num_trials", &RANSACOptions::max_num_trials)
            .def_readwrite("random_seed", &RANSACOptions::random_seed);

    py::class_<AbsolutePoseEstimationOptions> abs_est(
            m_estimators, "AbsolutePoseEstimationOptions",
            "Options for absolute pose (PnP) estimation.");
    abs_est.def(py::init<>())
            .def_readwrite(
                    "estimate_focal_length",
                    &AbsolutePoseEstimationOptions::estimate_focal_length)
            .def_readwrite(
                    "num_focal_length_samples",
                    &AbsolutePoseEstimationOptions::num_focal_length_samples)
            .def_readwrite(
                    "min_focal_length_ratio",
                    &AbsolutePoseEstimationOptions::min_focal_length_ratio)
            .def_readwrite(
                    "max_focal_length_ratio",
                    &AbsolutePoseEstimationOptions::max_focal_length_ratio)
            .def_readwrite("num_threads",
                           &AbsolutePoseEstimationOptions::num_threads)
            .def_readwrite("ransac_options",
                           &AbsolutePoseEstimationOptions::ransac_options);

    py::class_<AbsolutePoseRefinementOptions> abs_ref(
            m_estimators, "AbsolutePoseRefinementOptions",
            "Options for absolute pose refinement.");
    abs_ref.def(py::init<>())
            .def_readwrite("gradient_tolerance",
                           &AbsolutePoseRefinementOptions::gradient_tolerance)
            .def_readwrite("max_num_iterations",
                           &AbsolutePoseRefinementOptions::max_num_iterations)
            .def_readwrite("loss_function_scale",
                           &AbsolutePoseRefinementOptions::loss_function_scale)
            .def_readwrite("refine_focal_length",
                           &AbsolutePoseRefinementOptions::refine_focal_length)
            .def_readwrite("print_summary",
                           &AbsolutePoseRefinementOptions::print_summary)
            .def_readwrite("refine_extra_params",
                           &AbsolutePoseRefinementOptions::refine_extra_params);

    py::class_<EstimateTriangulationOptions> tri_opt(
            m_estimators, "EstimateTriangulationOptions",
            "Options for multi-view triangulation.");
    tri_opt.def(py::init<>())
            .def_readwrite("min_tri_angle",
                           &EstimateTriangulationOptions::min_tri_angle)
            .def_readwrite("ransac_options",
                           &EstimateTriangulationOptions::ransac_options);

    m_estimators.def(
            "triangulate_point",
            [](const Rigid3d& cam_from_world1, const Rigid3d& cam_from_world2,
               const Eigen::Vector2d& point1, const Eigen::Vector2d& point2) {
                Eigen::Vector3d point3D;
                // Two-view DLT on normalized camera points; the ray variant
                // is equivalent for perspective cameras.
                if (!colmap::TriangulatePoint(cam_from_world1.ToMatrix(),
                                              cam_from_world2.ToMatrix(),
                                              point1, point2, &point3D)) {
                    throw std::runtime_error(
                            "Triangulation failed (degenerate rays)");
                }
                return point3D;
            },
            "cam_from_world1"_a, "cam_from_world2"_a, "point1"_a, "point2"_a,
            "Triangulate a 3D point from two normalized 2D observations.");

    m_estimators.def(
            "estimate_triangulation",
            [](const EstimateTriangulationOptions& options,
               const py::array_t<double,
                                 py::array::c_style | py::array::forcecast>&
                       points2D,
               const std::vector<Rigid3d>& cams_from_world,
               const std::vector<Camera>& cameras) {
                auto points = Points2DFromNdarray(points2D);
                THROW_CHECK_EQ(points.size(), cams_from_world.size());
                THROW_CHECK_EQ(points.size(), cameras.size());
                std::vector<const Camera*> camera_ptrs;
                camera_ptrs.reserve(cameras.size());
                for (const auto& camera : cameras) {
                    camera_ptrs.push_back(&camera);
                }
                std::vector<char> inlier_mask;
                Eigen::Vector3d xyz;
                if (!colmap::EstimateTriangulation(options, points,
                                                   cams_from_world, camera_ptrs,
                                                   &inlier_mask, &xyz)) {
                    return py::make_tuple(py::none(), py::none());
                }
                return py::make_tuple(xyz, InlierMaskToList(inlier_mask));
            },
            "options"_a, "points2D"_a, "cams_from_world"_a, "cameras"_a,
            "Multi-view triangulation with RANSAC; returns (xyz, inlier_mask) "
            "or (None, None) on failure.");

    m_estimators.def(
            "estimate_absolute_pose",
            [](const AbsolutePoseEstimationOptions& options,
               const py::array_t<double,
                                 py::array::c_style | py::array::forcecast>&
                       points2D,
               const py::array_t<double,
                                 py::array::c_style | py::array::forcecast>&
                       points3D,
               Camera& camera) {
                auto points2 = Points2DFromNdarray(points2D);
                auto points3 = Points3DFromNdarray(points3D);
                THROW_CHECK_EQ(points2.size(), points3.size());
                Eigen::Vector4d qvec;
                Eigen::Vector3d tvec;
                size_t num_inliers = 0;
                std::vector<char> inlier_mask;
                if (!colmap::EstimateAbsolutePose(options, points2, points3,
                                                  &qvec, &tvec, &camera,
                                                  &num_inliers, &inlier_mask)) {
                    return py::make_tuple(py::none(), py::none(), py::none(),
                                          py::none());
                }
                return py::make_tuple(qvec, tvec, num_inliers,
                                      InlierMaskToList(inlier_mask));
            },
            "options"_a, "points2D"_a, "points3D"_a, "camera"_a,
            "Estimate the absolute pose (PnP) from 2D-3D correspondences; "
            "returns (qvec, tvec, num_inliers, inlier_mask) or all None.");

    m_estimators.def(
            "refine_absolute_pose",
            [](const AbsolutePoseRefinementOptions& options,
               const py::array_t<bool,
                                 py::array::c_style | py::array::forcecast>&
                       inlier_mask,
               const py::array_t<double,
                                 py::array::c_style | py::array::forcecast>&
                       points2D,
               const py::array_t<double,
                                 py::array::c_style | py::array::forcecast>&
                       points3D,
               const Rigid3d& cam_from_world, Camera& camera,
               bool return_covariance) -> py::object {
                auto points2 = Points2DFromNdarray(points2D);
                auto points3 = Points3DFromNdarray(points3D);
                auto mask_buf = inlier_mask.unchecked<1>();
                THROW_CHECK_EQ(points2.size(), points3.size());
                THROW_CHECK_EQ(points2.size(),
                               static_cast<size_t>(mask_buf.shape(0)));
                std::vector<char> mask(points2.size());
                for (py::ssize_t i = 0; i < mask_buf.shape(0); ++i) {
                    mask[i] = mask_buf(i) ? 1 : 0;
                }
                Rigid3d refined = cam_from_world;
                Eigen::Matrix6d covariance;
                if (!colmap::RefineAbsolutePose(
                            options, mask, points2, points3, &refined, &camera,
                            return_covariance ? &covariance : nullptr)) {
                    return py::none();
                }
                if (return_covariance) {
                    // 6x6 tangent-space covariance ([rotation, translation]
                    // order) over the refined pose.
                    return py::make_tuple(refined, covariance);
                }
                return py::cast(refined);
            },
            "options"_a, "inlier_mask"_a, "points2D"_a, "points3D"_a,
            "cam_from_world"_a, "camera"_a, "return_covariance"_a = false,
            "Refine an absolute pose (optionally the camera focal length) "
            "from 2D-3D correspondences; returns the refined Rigid3d, or a "
            "(Rigid3d, 6x6 covariance) tuple with return_covariance=True, or "
            "None on failure.");

    py::class_<TwoViewGeometryOptions> two_view_options(
            m_estimators, "TwoViewGeometryOptions",
            "Options for two-view geometry estimation.");
    two_view_options.def(py::init<>())
            .def_readwrite("min_num_inliers",
                           &TwoViewGeometryOptions::min_num_inliers)
            .def_readwrite("min_inlier_ratio",
                           &TwoViewGeometryOptions::min_inlier_ratio)
            .def_readwrite("min_E_F_inlier_ratio",
                           &TwoViewGeometryOptions::min_E_F_inlier_ratio)
            .def_readwrite("max_H_inlier_ratio",
                           &TwoViewGeometryOptions::max_H_inlier_ratio)
            .def_readwrite("watermark_min_inlier_ratio",
                           &TwoViewGeometryOptions::watermark_min_inlier_ratio)
            .def_readwrite("watermark_border_size",
                           &TwoViewGeometryOptions::watermark_border_size)
            .def_readwrite("detect_watermark",
                           &TwoViewGeometryOptions::detect_watermark)
            .def_readwrite("multiple_ignore_watermark",
                           &TwoViewGeometryOptions::multiple_ignore_watermark)
            .def_readwrite(
                    "watermark_detection_max_error",
                    &TwoViewGeometryOptions::watermark_detection_max_error)
            .def_readwrite("filter_stationary_matches",
                           &TwoViewGeometryOptions::filter_stationary_matches)
            .def_readwrite(
                    "stationary_matches_max_error",
                    &TwoViewGeometryOptions::stationary_matches_max_error)
            .def_readwrite("force_H_use", &TwoViewGeometryOptions::force_H_use)
            .def_readwrite("use_degensac",
                           &TwoViewGeometryOptions::use_degensac)
            .def_readwrite("use_sampson_refinement",
                           &TwoViewGeometryOptions::use_sampson_refinement)
            .def_readwrite("compute_relative_pose",
                           &TwoViewGeometryOptions::compute_relative_pose)
            .def_readwrite("multiple_models",
                           &TwoViewGeometryOptions::multiple_models)
            .def_readwrite("ransac", &TwoViewGeometryOptions::ransac_options);

    m_estimators.def(
            "estimate_essential_matrix",
            [](const std::vector<Eigen::Vector2d>& points2D1,
               const std::vector<Eigen::Vector2d>& points2D2,
               const Camera& camera1, const Camera& camera2,
               const RANSACOptions& options) -> py::object {
                THROW_CHECK_EQ(points2D1.size(), points2D2.size());
                const size_t num_points2D = points2D1.size();

                // Unproject to rays + per-ray Jacobians (pixel-unit tangent
                // Sampson score). Unprojectable points are zeroed -> rejected.
                std::vector<colmap::CamRayWithJac> cam_rays1(num_points2D);
                std::vector<colmap::CamRayWithJac> cam_rays2(num_points2D);
                for (size_t i = 0; i < num_points2D; ++i) {
                    cam_rays1[i] =
                            camera1.CamRayFromImgWithJac(points2D1[i])
                                    .value_or(colmap::CamRayWithJac::Zero());
                    cam_rays2[i] =
                            camera2.CamRayFromImgWithJac(points2D2[i])
                                    .value_or(colmap::CamRayWithJac::Zero());
                }

                colmap::LORANSAC<colmap::EssentialMatrixTangentSampsonEstimator,
                                 colmap::EssentialMatrixTangentSampsonEstimator,
                                 colmap::MEstimatorSupportMeasurer>
                        ransac(options);
                const auto report = ransac.Estimate(cam_rays1, cam_rays2);
                if (!report.success) {
                    return py::none();
                }

                // Pose from the essential matrix over the inlier set.
                std::vector<Eigen::Vector3d> inlier_rays1;
                std::vector<Eigen::Vector3d> inlier_rays2;
                inlier_rays1.reserve(report.support.num_inliers);
                inlier_rays2.reserve(report.support.num_inliers);
                for (size_t i = 0; i < num_points2D; ++i) {
                    if (report.inlier_mask[i]) {
                        inlier_rays1.push_back(cam_rays1[i].ray);
                        inlier_rays2.push_back(cam_rays2[i].ray);
                    }
                }
                Rigid3d cam2_from_cam1;
                std::vector<int> valid_indices;
                colmap::PoseFromEssentialMatrix(report.model, inlier_rays1,
                                                inlier_rays2, &cam2_from_cam1,
                                                &valid_indices);
                py::dict out;
                out["E"] = report.model;
                out["cam2_from_cam1"] = cam2_from_cam1;
                out["num_inliers"] = report.support.num_inliers;
                out["inlier_mask"] = InlierMaskToList(report.inlier_mask);
                return out;
            },
            "points2D1"_a, "points2D2"_a, "camera1"_a, "camera2"_a,
            "estimation_options"_a = RANSACOptions(),
            "Robustly estimate the essential matrix with LO-RANSAC and "
            "decompose it using the cheirality check; returns a dict with "
            "E/cam2_from_cam1/num_inliers/inlier_mask or None.");

    m_estimators.def(
            "estimate_fundamental_matrix",
            [](const std::vector<Eigen::Vector2d>& points2D1,
               const std::vector<Eigen::Vector2d>& points2D2,
               const RANSACOptions& options,
               bool use_sampson_refinement) -> py::object {
                THROW_CHECK_EQ(points2D1.size(), points2D2.size());
                py::dict out;
                if (use_sampson_refinement) {
                    colmap::LORANSAC<
                            colmap::FundamentalMatrixSevenPointEstimator,
                            colmap::FundamentalMatrixSampsonEstimator,
                            colmap::MEstimatorSupportMeasurer>
                            ransac(options);
                    const auto report = ransac.Estimate(points2D1, points2D2);
                    if (!report.success) {
                        return py::none();
                    }
                    out["F"] = report.model;
                    out["num_inliers"] = report.support.num_inliers;
                    out["inlier_mask"] = InlierMaskToList(report.inlier_mask);
                    return out;
                }
                colmap::LORANSAC<colmap::FundamentalMatrixSevenPointEstimator,
                                 colmap::FundamentalMatrixEightPointEstimator,
                                 colmap::MEstimatorSupportMeasurer>
                        ransac(options);
                const auto report = ransac.Estimate(points2D1, points2D2);
                if (!report.success) {
                    return py::none();
                }
                out["F"] = report.model;
                out["num_inliers"] = report.support.num_inliers;
                out["inlier_mask"] = InlierMaskToList(report.inlier_mask);
                return out;
            },
            "points2D1"_a, "points2D2"_a,
            "estimation_options"_a = RANSACOptions(),
            "use_sampson_refinement"_a = true,
            "Robustly estimate the fundamental matrix with LO-RANSAC; "
            "returns a dict with F/num_inliers/inlier_mask or None.");

    m_estimators.def(
            "estimate_homography_matrix",
            [](const std::vector<Eigen::Vector2d>& points2D1,
               const std::vector<Eigen::Vector2d>& points2D2,
               const RANSACOptions& options) -> py::object {
                THROW_CHECK_EQ(points2D1.size(), points2D2.size());
                colmap::LORANSAC<colmap::HomographyMatrixEstimator,
                                 colmap::HomographyMatrixEstimator,
                                 colmap::MEstimatorSupportMeasurer>
                        ransac(options);
                const auto report = ransac.Estimate(points2D1, points2D2);
                if (!report.success) {
                    return py::none();
                }
                py::dict out;
                out["H"] = report.model;
                out["num_inliers"] = report.support.num_inliers;
                out["inlier_mask"] = InlierMaskToList(report.inlier_mask);
                return out;
            },
            "points2D1"_a, "points2D2"_a,
            "estimation_options"_a = RANSACOptions(),
            "Robustly estimate the homography matrix with LO-RANSAC; "
            "returns a dict with H/num_inliers/inlier_mask or None.");

    m_estimators.def(
            "estimate_two_view_geometry_pose",
            [](const Camera& camera1,
               const std::vector<Eigen::Vector2d>& points1,
               const Camera& camera2,
               const std::vector<Eigen::Vector2d>& points2,
               TwoViewGeometry& geometry) -> py::object {
                // The fork splits the relative-pose decomposition out of the
                // plain estimation (upstream d3ccaf35 parity); expose the
                // decomposition step on an already-estimated geometry.
                if (!colmap::EstimateTwoViewGeometryPose(
                            camera1, points1, camera2, points2, &geometry)) {
                    return py::none();
                }
                return py::cast(geometry);
            },
            "camera1"_a, "points1"_a, "camera2"_a, "points2"_a, "geometry"_a,
            "Decompose the relative pose and triangulation angle on an "
            "estimated two-view geometry; returns the updated geometry or "
            "None.");

    m_estimators.def(
            "estimate_two_view_geometry",
            [](const Camera& camera1,
               const std::vector<Eigen::Vector2d>& points1,
               const Camera& camera2,
               const std::vector<Eigen::Vector2d>& points2,
               const py::object& matches,
               const TwoViewGeometryOptions& options) {
                FeatureMatches feature_matches =
                        MatchesFromOptionalNdarray(matches);
                if (feature_matches.empty()) {
                    THROW_CHECK_EQ(points1.size(), points2.size());
                    feature_matches.reserve(points1.size());
                    for (size_t i = 0; i < points1.size(); ++i) {
                        feature_matches.emplace_back(i, i);
                    }
                }
                return colmap::EstimateTwoViewGeometry(
                        camera1, points1, camera2, points2,
                        std::move(feature_matches), options);
            },
            "camera1"_a, "points1"_a, "camera2"_a, "points2"_a,
            "matches"_a = py::none(), "options"_a = TwoViewGeometryOptions(),
            "Estimate the two-view geometry from an image pair (calibrated or "
            "uncalibrated depending on prior focal lengths); matches is an "
            "(N, 2) uint32 array or None for identity pairing.");

    m_estimators.def(
            "estimate_calibrated_two_view_geometry",
            [](const Camera& camera1,
               const std::vector<Eigen::Vector2d>& points1,
               const Camera& camera2,
               const std::vector<Eigen::Vector2d>& points2,
               const py::object& matches,
               const TwoViewGeometryOptions& options) {
                FeatureMatches feature_matches =
                        MatchesFromOptionalNdarray(matches);
                if (feature_matches.empty()) {
                    THROW_CHECK_EQ(points1.size(), points2.size());
                    feature_matches.reserve(points1.size());
                    for (size_t i = 0; i < points1.size(); ++i) {
                        feature_matches.emplace_back(i, i);
                    }
                }
                return colmap::EstimateCalibratedTwoViewGeometry(
                        camera1, points1, camera2, points2, feature_matches,
                        options);
            },
            "camera1"_a, "points1"_a, "camera2"_a, "points2"_a,
            "matches"_a = py::none(), "options"_a = TwoViewGeometryOptions(),
            "Estimate the two-view geometry from a calibrated image pair.");

    py::class_<ImageAlignmentError> alignment_error(
            m_estimators, "ImageAlignmentError",
            "Per-image alignment error statistics.");
    alignment_error.def(py::init<>())
            .def_readwrite("image_name", &ImageAlignmentError::image_name)
            .def_readwrite("rotation_error_deg",
                           &ImageAlignmentError::rotation_error_deg)
            .def_readwrite("proj_center_error",
                           &ImageAlignmentError::proj_center_error);

    m_estimators.def(
            "align_reconstructions_via_reprojections",
            [](const Reconstruction& src_reconstruction,
               const Reconstruction& tgt_reconstruction,
               double min_inlier_observations,
               double max_reproj_error) -> py::object {
                Sim3d tgt_from_src;
                if (!colmap::AlignReconstructionsViaReprojections(
                            src_reconstruction, tgt_reconstruction,
                            min_inlier_observations, max_reproj_error,
                            &tgt_from_src)) {
                    return py::none();
                }
                return py::cast(tgt_from_src);
            },
            "src_reconstruction"_a, "tgt_reconstruction"_a,
            "min_inlier_observations"_a = 0.3, "max_reproj_error"_a = 8.0,
            "Align two reconstructions via shared 2D-3D observations; "
            "returns the tgt_from_src Sim3d or None.");

    m_estimators.def(
            "align_reconstructions_via_proj_centers",
            [](const Reconstruction& src_reconstruction,
               const Reconstruction& tgt_reconstruction,
               double max_proj_center_error) -> py::object {
                Sim3d tgt_from_src;
                if (!colmap::AlignReconstructionsViaProjCenters(
                            src_reconstruction, tgt_reconstruction,
                            max_proj_center_error, &tgt_from_src)) {
                    return py::none();
                }
                return py::cast(tgt_from_src);
            },
            "src_reconstruction"_a, "tgt_reconstruction"_a,
            "max_proj_center_error"_a,
            "Align two reconstructions via camera projection centers; "
            "returns the tgt_from_src Sim3d or None.");

    m_estimators.def(
            "align_reconstructions_via_points",
            [](const Reconstruction& src_reconstruction,
               const Reconstruction& tgt_reconstruction,
               size_t min_common_observations, double max_error,
               double min_inlier_ratio) -> py::object {
                Sim3d tgt_from_src;
                if (!colmap::AlignReconstructionsViaPoints(
                            src_reconstruction, tgt_reconstruction,
                            min_common_observations, max_error,
                            min_inlier_ratio, &tgt_from_src)) {
                    return py::none();
                }
                return py::cast(tgt_from_src);
            },
            "src_reconstruction"_a, "tgt_reconstruction"_a,
            "min_common_observations"_a = 3, "max_error"_a = 0.005,
            "min_inlier_ratio"_a = 0.9,
            "Align two reconstructions via shared 3D points; returns the "
            "tgt_from_src Sim3d or None.");

    m_estimators.def(
            "align_reconstruction_to_locations",
            [](const Reconstruction& src,
               const std::vector<std::string>& tgt_image_names,
               const std::vector<Eigen::Vector3d>& tgt_locations,
               int min_common_images,
               const RANSACOptions& ransac_options) -> py::object {
                Sim3d locations_from_src;
                if (!colmap::AlignReconstructionToLocations(
                            src, tgt_image_names, tgt_locations,
                            min_common_images, ransac_options,
                            &locations_from_src)) {
                    return py::none();
                }
                return py::cast(locations_from_src);
            },
            "src"_a, "tgt_image_names"_a, "tgt_locations"_a,
            "min_common_images"_a, "ransac_options"_a = RANSACOptions(),
            "Geo-register a reconstruction to known image locations via "
            "robust similarity estimation; returns the locations_from_src "
            "Sim3d or None.");

    m_estimators.def(
            "estimate_rigid3d",
            [](const std::vector<Eigen::Vector3d>& src,
               const std::vector<Eigen::Vector3d>& tgt) -> py::object {
                Rigid3d tgt_from_src;
                if (!colmap::EstimateRigid3d(src, tgt, tgt_from_src)) {
                    return py::none();
                }
                return py::cast(tgt_from_src);
            },
            "src"_a, "tgt"_a, "Estimate the 3D rigid transform tgt_from_src.");

    m_estimators.def(
            "estimate_rigid3d_robust",
            [](const std::vector<Eigen::Vector3d>& src,
               const std::vector<Eigen::Vector3d>& tgt,
               const RANSACOptions& options) -> py::object {
                Rigid3d tgt_from_src;
                const auto report = colmap::EstimateRigid3dRobust(
                        src, tgt, options, tgt_from_src);
                if (!report.success) {
                    return py::none();
                }
                py::dict out;
                out["tgt_from_src"] = Rigid3d::FromMatrix(report.model);
                out["num_inliers"] = report.support.num_inliers;
                out["inlier_mask"] = InlierMaskToList(report.inlier_mask);
                return out;
            },
            "src"_a, "tgt"_a, "estimation_options"_a = RANSACOptions(),
            "Robustly estimate the 3D rigid transform with LO-RANSAC.");

    m_estimators.def(
            "estimate_sim3d",
            [](const std::vector<Eigen::Vector3d>& src,
               const std::vector<Eigen::Vector3d>& tgt) -> py::object {
                Sim3d tgt_from_src;
                if (!colmap::EstimateSim3d(src, tgt, tgt_from_src)) {
                    return py::none();
                }
                return py::cast(tgt_from_src);
            },
            "src"_a, "tgt"_a,
            "Estimate the 3D similarity transform tgt_from_src.");

    m_estimators.def(
            "estimate_sim3d_robust",
            [](const std::vector<Eigen::Vector3d>& src,
               const std::vector<Eigen::Vector3d>& tgt,
               const RANSACOptions& options) -> py::object {
                Sim3d tgt_from_src;
                const auto report = colmap::EstimateSim3dRobust(
                        src, tgt, options, tgt_from_src);
                if (!report.success) {
                    return py::none();
                }
                py::dict out;
                out["tgt_from_src"] = Sim3d::FromMatrix(report.model);
                out["num_inliers"] = report.support.num_inliers;
                out["inlier_mask"] = InlierMaskToList(report.inlier_mask);
                return out;
            },
            "src"_a, "tgt"_a, "estimation_options"_a = RANSACOptions(),
            "Robustly estimate the 3D similarity transform with LO-RANSAC.");

    // ---- Upstream pycolmap parity (src/pycolmap/estimators/bundle_
    // adjustment.cc): the bundle adjustment class surface. The fork keeps the
    // bool Solve(Reconstruction*) lifecycle; the ceres solver_options /
    // check_if_stopped internals stay engine-side and are not exposed.
    py::enum_<colmap::BundleAdjustmentTerminationType> ba_term(
            m_estimators, "BundleAdjustmentTerminationType",
            "Bundle adjustment termination status.");
    ba_term.value("CONVERGENCE",
                  colmap::BundleAdjustmentTerminationType::CONVERGENCE)
            .value("NO_CONVERGENCE",
                   colmap::BundleAdjustmentTerminationType::NO_CONVERGENCE)
            .value("FAILURE", colmap::BundleAdjustmentTerminationType::FAILURE)
            .value("USER_SUCCESS",
                   colmap::BundleAdjustmentTerminationType::USER_SUCCESS)
            .value("USER_FAILURE",
                   colmap::BundleAdjustmentTerminationType::USER_FAILURE);

    py::enum_<colmap::BundleAdjustmentGauge> ba_gauge(
            m_estimators, "BundleAdjustmentGauge",
            "The gauge fixing strategy (the problem has a global 7-DoF null "
            "space without fixing).");
    ba_gauge.value("UNSPECIFIED", colmap::BundleAdjustmentGauge::UNSPECIFIED)
            .value("TWO_CAMS_FROM_WORLD",
                   colmap::BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD)
            .value("THREE_POINTS", colmap::BundleAdjustmentGauge::THREE_POINTS);

    py::enum_<colmap::BundleAdjustmentBackend> ba_backend(
            m_estimators, "BundleAdjustmentBackend",
            "The bundle adjustment solver backend.");
    ba_backend.value("CERES", colmap::BundleAdjustmentBackend::CERES)
            .value("CASPAR", colmap::BundleAdjustmentBackend::CASPAR);

    py::class_<colmap::BundleAdjustmentSummary> ba_summary(
            m_estimators, "BundleAdjustmentSummary",
            "Summary of bundle adjustment results.");
    ba_summary.def(py::init<>())
            .def_readwrite("termination_type",
                           &colmap::BundleAdjustmentSummary::termination_type)
            .def_readwrite("num_residuals",
                           &colmap::BundleAdjustmentSummary::num_residuals)
            .def("is_solution_usable",
                 &colmap::BundleAdjustmentSummary::IsSolutionUsable)
            .def("brief_report", &colmap::BundleAdjustmentSummary::BriefReport);

    py::class_<colmap::BundleAdjustmentConfig> ba_config(
            m_estimators, "BundleAdjustmentConfig",
            "Configuration container to set up bundle adjustment problems.");
    ba_config.def(py::init<>())
            .def("add_image", &colmap::BundleAdjustmentConfig::AddImage,
                 "image_id"_a)
            .def("has_image", &colmap::BundleAdjustmentConfig::HasImage,
                 "image_id"_a)
            .def("remove_image", &colmap::BundleAdjustmentConfig::RemoveImage,
                 "image_id"_a)
            .def("num_images", &colmap::BundleAdjustmentConfig::NumImages)
            .def("num_points", &colmap::BundleAdjustmentConfig::NumPoints)
            .def("num_constant_cameras",
                 &colmap::BundleAdjustmentConfig::NumConstantCameras)
            .def("num_constant_poses",
                 &colmap::BundleAdjustmentConfig::NumConstantPoses)
            .def("num_constant_tvecs",
                 &colmap::BundleAdjustmentConfig::NumConstantTvecs)
            .def("num_variable_points",
                 &colmap::BundleAdjustmentConfig::NumVariablePoints)
            .def("num_constant_points",
                 &colmap::BundleAdjustmentConfig::NumConstantPoints)
            .def("num_residuals", &colmap::BundleAdjustmentConfig::NumResiduals,
                 "reconstruction"_a)
            .def("set_constant_camera",
                 &colmap::BundleAdjustmentConfig::SetConstantCamera,
                 "camera_id"_a,
                 "Freeze the camera intrinsics (upstream "
                 "set_constant_cam_intrinsics semantics).")
            .def("set_variable_camera",
                 &colmap::BundleAdjustmentConfig::SetVariableCamera,
                 "camera_id"_a)
            .def("is_constant_camera",
                 &colmap::BundleAdjustmentConfig::IsConstantCamera,
                 "camera_id"_a)
            .def("set_constant_pose",
                 &colmap::BundleAdjustmentConfig::SetConstantPose, "image_id"_a)
            .def("set_variable_pose",
                 &colmap::BundleAdjustmentConfig::SetVariablePose, "image_id"_a)
            .def("has_constant_pose",
                 &colmap::BundleAdjustmentConfig::HasConstantPose, "image_id"_a)
            .def("set_constant_rig_from_world_pose",
                 &colmap::BundleAdjustmentConfig::SetConstantRigFromWorldPose,
                 "frame_id"_a)
            .def("set_variable_rig_from_world_pose",
                 &colmap::BundleAdjustmentConfig::SetVariableRigFromWorldPose,
                 "frame_id"_a)
            .def("has_constant_rig_from_world_pose",
                 &colmap::BundleAdjustmentConfig::HasConstantRigFromWorldPose,
                 "frame_id"_a)
            .def("set_constant_sensor_from_rig_pose",
                 &colmap::BundleAdjustmentConfig::SetConstantSensorFromRigPose,
                 "sensor_id"_a)
            .def("set_variable_sensor_from_rig_pose",
                 &colmap::BundleAdjustmentConfig::SetVariableSensorFromRigPose,
                 "sensor_id"_a)
            .def("has_constant_sensor_from_rig_pose",
                 &colmap::BundleAdjustmentConfig::HasConstantSensorFromRigPose,
                 "sensor_id"_a)
            .def("fix_gauge", &colmap::BundleAdjustmentConfig::FixGauge,
                 "gauge"_a)
            .def_property_readonly("fixed_gauge",
                                   &colmap::BundleAdjustmentConfig::FixedGauge)
            .def(
                    "set_constant_tvec",
                    [](colmap::BundleAdjustmentConfig& self,
                       colmap::image_t image_id, const std::vector<int>& idxs) {
                        self.SetConstantTvec(image_id, idxs);
                    },
                    "image_id"_a, "idxs"_a)
            .def("remove_constant_tvec",
                 &colmap::BundleAdjustmentConfig::RemoveConstantTvec,
                 "image_id"_a)
            .def("add_variable_point",
                 &colmap::BundleAdjustmentConfig::AddVariablePoint,
                 "point3D_id"_a)
            .def("add_constant_point",
                 &colmap::BundleAdjustmentConfig::AddConstantPoint,
                 "point3D_id"_a)
            .def("has_point", &colmap::BundleAdjustmentConfig::HasPoint,
                 "point3D_id"_a)
            .def("has_variable_point",
                 &colmap::BundleAdjustmentConfig::HasVariablePoint,
                 "point3D_id"_a)
            .def("has_constant_point",
                 &colmap::BundleAdjustmentConfig::HasConstantPoint,
                 "point3D_id"_a)
            .def("remove_variable_point",
                 &colmap::BundleAdjustmentConfig::RemoveVariablePoint,
                 "point3D_id"_a)
            .def("remove_constant_point",
                 &colmap::BundleAdjustmentConfig::RemoveConstantPoint,
                 "point3D_id"_a)
            .def("ignore_point", &colmap::BundleAdjustmentConfig::IgnorePoint,
                 "point3D_id"_a)
            .def("is_ignored_point",
                 &colmap::BundleAdjustmentConfig::IsIgnoredPoint,
                 "point3D_id"_a);

    m_estimators.def(
            "adjust_bundle",
            [](const colmap::BundleAdjustmentOptions& options,
               const colmap::BundleAdjustmentConfig& config,
               Reconstruction& reconstruction) -> py::object {
                auto adjuster =
                        colmap::CreateDefaultBundleAdjuster(options, config);
                const bool success = adjuster->Solve(&reconstruction);
                if (!success) {
                    return py::none();
                }
                // Materialize the fork's engine-side summary into the
                // backend-agnostic BundleAdjustmentSummary value type.
                auto summary = colmap::CeresBundleAdjustmentSummary::Create(
                        adjuster->Summary());
                py::dict out;
                out["termination_type"] = summary->termination_type;
                out["num_residuals"] = summary->num_residuals;
                out["is_solution_usable"] = summary->IsSolutionUsable();
                out["brief_report"] = summary->BriefReport();
                return out;
            },
            "options"_a, "config"_a, "reconstruction"_a,
            "Run bundle adjustment on the reconstruction with the default "
            "backend dispatch (CASPAR-first, Ceres fallback); returns a "
            "summary dict or None on failure.");

    m_estimators.def(
            "estimate_affine2d",
            [](const std::vector<Eigen::Vector2d>& src,
               const std::vector<Eigen::Vector2d>& tgt) -> py::object {
                Eigen::Matrix2x3d tgt_from_src;
                if (!colmap::EstimateAffine2d(src, tgt, tgt_from_src)) {
                    return py::none();
                }
                return py::cast(tgt_from_src);
            },
            "src"_a, "tgt"_a, "Estimate the 2D affine transform tgt_from_src.");

    m_estimators.def(
            "estimate_affine2d_robust",
            [](const std::vector<Eigen::Vector2d>& src,
               const std::vector<Eigen::Vector2d>& tgt,
               const RANSACOptions& options) -> py::object {
                Eigen::Matrix2x3d tgt_from_src;
                const auto report = colmap::EstimateAffine2dRobust(
                        src, tgt, options, tgt_from_src);
                if (!report.success) {
                    return py::none();
                }
                py::dict out;
                out["tgt_from_src"] = tgt_from_src;
                out["num_inliers"] = report.support.num_inliers;
                out["inlier_mask"] = InlierMaskToList(report.inlier_mask);
                return out;
            },
            "src"_a, "tgt"_a, "estimation_options"_a = RANSACOptions(),
            "Robustly estimate the 2D affine transform with LO-RANSAC.");

    // Upstream pycolmap parity (src/pycolmap/estimators/covariance.cc):
    // the bundle-adjustment covariance surface (separate TU).
    pybind_covariance(m_estimators);

    // Upstream pycolmap parity (src/pycolmap/estimators/motion_averaging.cc):
    // the GLomap motion-averaging entry points.
    py::enum_<colmap::RotationEstimatorOptions::WeightType> rot_weight(
            m_estimators, "RotationWeightType",
            "Robust weight for rotation averaging IRLS.");
    rot_weight
            .value("GEMAN_MCCLURE",
                   colmap::RotationEstimatorOptions::WeightType::GEMAN_MCCLURE)
            .value("HALF_NORM",
                   colmap::RotationEstimatorOptions::WeightType::HALF_NORM);

    py::class_<colmap::RotationEstimatorOptions> ra_options(
            m_estimators, "RotationEstimatorOptions",
            "Options for (GLomap-style) rotation averaging.");
    ra_options.def(py::init<>())
            .def_readwrite("random_seed",
                           &colmap::RotationEstimatorOptions::random_seed)
            .def_readwrite(
                    "max_num_l1_iterations",
                    &colmap::RotationEstimatorOptions::max_num_l1_iterations)
            .def_readwrite("l1_step_convergence_threshold",
                           &colmap::RotationEstimatorOptions::
                                   l1_step_convergence_threshold)
            .def_readwrite(
                    "max_num_irls_iterations",
                    &colmap::RotationEstimatorOptions::max_num_irls_iterations)
            .def_readwrite("irls_step_convergence_threshold",
                           &colmap::RotationEstimatorOptions::
                                   irls_step_convergence_threshold)
            .def_readwrite("irls_loss_parameter_sigma",
                           &colmap::RotationEstimatorOptions::
                                   irls_loss_parameter_sigma)
            .def_readwrite(
                    "ridge_regularization",
                    &colmap::RotationEstimatorOptions::ridge_regularization)
            .def_readwrite(
                    "skip_initialization",
                    &colmap::RotationEstimatorOptions::skip_initialization)
            .def_readwrite("use_gravity",
                           &colmap::RotationEstimatorOptions::use_gravity)
            .def_readwrite("use_stratified",
                           &colmap::RotationEstimatorOptions::use_stratified)
            .def_readwrite(
                    "filter_unregistered",
                    &colmap::RotationEstimatorOptions::filter_unregistered)
            .def_readwrite(
                    "max_rotation_error_deg",
                    &colmap::RotationEstimatorOptions::max_rotation_error_deg)
            .def_readwrite(
                    "refine_sensor_from_rig",
                    &colmap::RotationEstimatorOptions::refine_sensor_from_rig)
            .def_readwrite("weight_type",
                           &colmap::RotationEstimatorOptions::weight_type);

    m_estimators.def(
            "estimate_rotations",
            [](const colmap::RotationEstimatorOptions& options,
               const colmap::PoseGraph& pose_graph,
               std::vector<colmap::PosePrior> pose_priors,
               const std::vector<colmap::image_t>& active_image_ids,
               Reconstruction& reconstruction) {
                // Solves rotation averaging and registers frames with
                // computed poses (upstream parity).
                colmap::RotationEstimator estimator(options);
                return estimator.EstimateRotations(
                        pose_graph, pose_priors,
                        colmap::FlatHashSet<colmap::image_t>(
                                active_image_ids.begin(),
                                active_image_ids.end()),
                        reconstruction);
            },
            "options"_a, "pose_graph"_a, "pose_priors"_a, "active_image_ids"_a,
            "reconstruction"_a,
            "Solve rotation averaging and register the frames with computed "
            "poses; returns True on success.");

    py::class_<colmap::GlobalPositionerOptions> gp_options(
            m_estimators, "GlobalPositionerOptions",
            "Options for (GLomap-style) global positioning.");
    gp_options.def(py::init<>())
            .def_readwrite(
                    "generate_random_positions",
                    &colmap::GlobalPositionerOptions::generate_random_positions)
            .def_readwrite(
                    "generate_random_points",
                    &colmap::GlobalPositionerOptions::generate_random_points)
            .def_readwrite("generate_scales",
                           &colmap::GlobalPositionerOptions::generate_scales)
            .def_readwrite("optimize_positions",
                           &colmap::GlobalPositionerOptions::optimize_positions)
            .def_readwrite("optimize_points",
                           &colmap::GlobalPositionerOptions::optimize_points)
            .def_readwrite("optimize_scales",
                           &colmap::GlobalPositionerOptions::optimize_scales)
            .def_readwrite(
                    "refine_sensor_from_rig",
                    &colmap::GlobalPositionerOptions::refine_sensor_from_rig)
            .def_readwrite("use_gpu", &colmap::GlobalPositionerOptions::use_gpu)
            .def_readwrite("gpu_index",
                           &colmap::GlobalPositionerOptions::gpu_index)
            .def_readwrite(
                    "min_num_images_gpu_solver",
                    &colmap::GlobalPositionerOptions::min_num_images_gpu_solver)
            .def_readwrite(
                    "min_num_view_per_track",
                    &colmap::GlobalPositionerOptions::min_num_view_per_track)
            .def_readwrite("random_seed",
                           &colmap::GlobalPositionerOptions::random_seed)
            .def_readwrite(
                    "loss_function_scale",
                    &colmap::GlobalPositionerOptions::loss_function_scale)
            .def_readwrite("use_parameter_block_ordering",
                           &colmap::GlobalPositionerOptions::
                                   use_parameter_block_ordering);

    m_estimators.def(
            "run_global_positioning",
            [](const colmap::GlobalPositionerOptions& options,
               const colmap::PoseGraph& pose_graph,
               Reconstruction& reconstruction) {
                return colmap::RunGlobalPositioning(options, pose_graph,
                                                    reconstruction);
            },
            "options"_a, "pose_graph"_a, "reconstruction"_a,
            "Solve global positioning using point-to-camera constraints; "
            "returns True on success.");

    py::class_<colmap::GravityRefinerOptions> grav_options(
            m_estimators, "GravityRefinerOptions",
            "Options for gravity refinement of pose priors.");
    grav_options.def(py::init<>())
            .def_readwrite("max_outlier_ratio",
                           &colmap::GravityRefinerOptions::max_outlier_ratio)
            .def_readwrite("max_gravity_error",
                           &colmap::GravityRefinerOptions::max_gravity_error)
            .def_readwrite("min_num_neighbors",
                           &colmap::GravityRefinerOptions::min_num_neighbors);

    m_estimators.def(
            "run_gravity_refinement",
            [](const colmap::GravityRefinerOptions& options,
               const colmap::PoseGraph& pose_graph,
               const Reconstruction& reconstruction, py::list pose_priors) {
                // The engine refines in place; pybind's list caster copies,
                // so round-trip the elements explicitly.
                std::vector<colmap::PosePrior> priors;
                priors.reserve(py::len(pose_priors));
                for (auto item : pose_priors) {
                    priors.push_back(item.cast<colmap::PosePrior>());
                }
                colmap::RunGravityRefinement(options, pose_graph,
                                             reconstruction, priors);
                py::list out;
                for (auto& prior : priors) {
                    out.append(prior);
                }
                return out;
            },
            "options"_a, "pose_graph"_a, "reconstruction"_a, "pose_priors"_a,
            "Refine the gravity of pose priors using relative rotations from "
            "the pose graph; returns the refined pose prior list.");

    // Upstream pycolmap parity (src/pycolmap/estimators/generalized_pose.cc):
    // the generalized (multi-camera rig) pose estimation surface. The fork
    // engine exposes the non-scaled Rigid3d/Matrix6d API; the upstream scaled
    // variants (Estimate/RefineScaledGeneralizedAbsolutePose, Sim3d + 7x7
    // covariance) have no fork engine counterpart yet and stay unbound.

    m_estimators.def(
            "estimate_generalized_absolute_pose",
            [](const py::array_t<double,
                                 py::array::c_style | py::array::forcecast>&
                       points2D,
               const py::array_t<double,
                                 py::array::c_style | py::array::forcecast>&
                       points3D,
               const std::vector<size_t>& camera_idxs,
               const std::vector<Rigid3d>& cams_from_rig,
               const std::vector<Camera>& cameras,
               const RANSACOptions& options) -> py::object {
                auto points2 = Points2DFromNdarray(points2D);
                auto points3 = Points3DFromNdarray(points3D);
                THROW_CHECK_EQ(points2.size(), points3.size());
                THROW_CHECK_EQ(points2.size(), camera_idxs.size());
                THROW_CHECK_EQ(cams_from_rig.size(), cameras.size());
                Rigid3d rig_from_world;
                size_t num_inliers = 0;
                std::vector<char> inlier_mask;
                if (!colmap::EstimateGeneralizedAbsolutePose(
                            options, points2, points3, camera_idxs,
                            cams_from_rig, cameras, &rig_from_world,
                            &num_inliers, &inlier_mask)) {
                    return py::none();
                }
                py::dict out;
                out["rig_from_world"] = rig_from_world;
                out["num_inliers"] = num_inliers;
                out["inlier_mask"] = InlierMaskToList(inlier_mask);
                return out;
            },
            "points2D"_a, "points3D"_a, "camera_idxs"_a, "cams_from_rig"_a,
            "cameras"_a, "estimation_options"_a = RANSACOptions(),
            "Robustly estimate the generalized absolute pose (rig from world) "
            "from 2D-3D correspondences across a rig with LO-RANSAC; returns a "
            "dict with rig_from_world/num_inliers/inlier_mask or None.");

    m_estimators.def(
            "refine_generalized_absolute_pose",
            [](const Rigid3d& rig_from_world,
               const py::array_t<double,
                                 py::array::c_style | py::array::forcecast>&
                       points2D,
               const py::array_t<double,
                                 py::array::c_style | py::array::forcecast>&
                       points3D,
               const py::array_t<bool,
                                 py::array::c_style | py::array::forcecast>&
                       inlier_mask,
               const std::vector<size_t>& camera_idxs,
               const std::vector<Rigid3d>& cams_from_rig,
               const std::vector<Camera>& cameras,
               const AbsolutePoseRefinementOptions& options,
               bool return_covariance) -> py::object {
                auto points2 = Points2DFromNdarray(points2D);
                auto points3 = Points3DFromNdarray(points3D);
                auto mask_buf = inlier_mask.unchecked<1>();
                THROW_CHECK_EQ(points2.size(), points3.size());
                THROW_CHECK_EQ(points2.size(),
                               static_cast<size_t>(mask_buf.shape(0)));
                THROW_CHECK_EQ(points2.size(), camera_idxs.size());
                THROW_CHECK_EQ(cams_from_rig.size(), cameras.size());
                std::vector<char> mask(points2.size());
                for (py::ssize_t i = 0; i < mask_buf.shape(0); ++i) {
                    mask[i] = mask_buf(i) ? 1 : 0;
                }
                Rigid3d refined = rig_from_world;
                // The STL conversion of the input is a temporary, so the
                // refined cameras are returned explicitly (upstream parity).
                std::vector<Camera> refined_cameras = cameras;
                Eigen::Matrix6d covariance;
                if (!colmap::RefineGeneralizedAbsolutePose(
                            options, mask, points2, points3, camera_idxs,
                            cams_from_rig, &refined, &refined_cameras,
                            return_covariance ? &covariance : nullptr)) {
                    return py::none();
                }
                py::dict out;
                out["rig_from_world"] = refined;
                out["cameras"] = refined_cameras;
                if (return_covariance) {
                    // 6x6 tangent-space covariance ([rotation, translation]
                    // order) over the refined pose, edge-marginalized over
                    // the refined camera parameters.
                    out["covariance"] = covariance;
                }
                return out;
            },
            "rig_from_world"_a, "points2D"_a, "points3D"_a, "inlier_mask"_a,
            "camera_idxs"_a, "cams_from_rig"_a, "cameras"_a,
            "refinement_options"_a = AbsolutePoseRefinementOptions(),
            "return_covariance"_a = false,
            "Refine the generalized absolute pose (optionally focal lengths) "
            "with non-linear refinement; returns a dict with rig_from_world "
            "and cameras (covariance is None until the fork engine "
            "assembles it; the scaled variant provides the 7x7 one), or "
            "None.");

    m_estimators.def(
            "estimate_and_refine_generalized_absolute_pose",
            [](const py::array_t<double,
                                 py::array::c_style | py::array::forcecast>&
                       points2D,
               const py::array_t<double,
                                 py::array::c_style | py::array::forcecast>&
                       points3D,
               const std::vector<size_t>& camera_idxs,
               const std::vector<Rigid3d>& cams_from_rig,
               const std::vector<Camera>& cameras,
               const RANSACOptions& estimation_options,
               const AbsolutePoseRefinementOptions& refinement_options,
               bool return_covariance) -> py::object {
                auto points2 = Points2DFromNdarray(points2D);
                auto points3 = Points3DFromNdarray(points3D);
                THROW_CHECK_EQ(points2.size(), points3.size());
                THROW_CHECK_EQ(points2.size(), camera_idxs.size());
                THROW_CHECK_EQ(cams_from_rig.size(), cameras.size());
                Rigid3d rig_from_world;
                size_t num_inliers = 0;
                std::vector<char> inlier_mask;
                if (!colmap::EstimateGeneralizedAbsolutePose(
                            estimation_options, points2, points3, camera_idxs,
                            cams_from_rig, cameras, &rig_from_world,
                            &num_inliers, &inlier_mask)) {
                    return py::none();
                }
                std::vector<Camera> refined_cameras = cameras;
                Eigen::Matrix6d covariance;
                if (!colmap::RefineGeneralizedAbsolutePose(
                            refinement_options, inlier_mask, points2, points3,
                            camera_idxs, cams_from_rig, &rig_from_world,
                            &refined_cameras,
                            return_covariance ? &covariance : nullptr)) {
                    return py::none();
                }
                py::dict out;
                out["rig_from_world"] = rig_from_world;
                out["num_inliers"] = num_inliers;
                out["inlier_mask"] = InlierMaskToList(inlier_mask);
                out["cameras"] = refined_cameras;
                if (return_covariance) {
                    out["covariance"] = covariance;
                }
                return out;
            },
            "points2D"_a, "points3D"_a, "camera_idxs"_a, "cams_from_rig"_a,
            "cameras"_a, "estimation_options"_a = RANSACOptions(),
            "refinement_options"_a = AbsolutePoseRefinementOptions(),
            "return_covariance"_a = false,
            "Robustly estimate the generalized absolute pose with LO-RANSAC "
            "followed by non-linear refinement; returns a dict with the "
            "refined rig_from_world, inliers, and cameras (and optionally the "
            "6x6 covariance), or None.");

    m_estimators.def(
            "estimate_scaled_generalized_absolute_pose",
            [](const py::array_t<double,
                                 py::array::c_style | py::array::forcecast>&
                       points2D,
               const py::array_t<double,
                                 py::array::c_style | py::array::forcecast>&
                       points3D,
               const std::vector<size_t>& camera_idxs,
               const std::vector<Rigid3d>& cams_from_rig,
               const std::vector<Camera>& cameras,
               const RANSACOptions& options) -> py::object {
                auto points2 = Points2DFromNdarray(points2D);
                auto points3 = Points3DFromNdarray(points3D);
                THROW_CHECK_EQ(points2.size(), points3.size());
                THROW_CHECK_EQ(points2.size(), camera_idxs.size());
                THROW_CHECK_EQ(cams_from_rig.size(), cameras.size());
                Sim3d rig_from_world;
                size_t num_inliers = 0;
                std::vector<char> inlier_mask;
                if (!colmap::EstimateScaledGeneralizedAbsolutePose(
                            options, points2, points3, camera_idxs,
                            cams_from_rig, cameras, &rig_from_world,
                            &num_inliers, &inlier_mask)) {
                    return py::none();
                }
                py::dict out;
                out["rig_from_world"] = rig_from_world;
                out["num_inliers"] = num_inliers;
                out["inlier_mask"] = InlierMaskToList(inlier_mask);
                return out;
            },
            "points2D"_a, "points3D"_a, "camera_idxs"_a, "cams_from_rig"_a,
            "cameras"_a, "estimation_options"_a = RANSACOptions(),
            "Robustly estimate the generalized absolute pose and the rig "
            "geometry scale (GP4PS) with LO-RANSAC; returns a dict with "
            "rig_from_world (Sim3d)/num_inliers/inlier_mask or None.");

    m_estimators.def(
            "refine_scaled_generalized_absolute_pose",
            [](const Sim3d& rig_from_world,
               const py::array_t<double,
                                 py::array::c_style | py::array::forcecast>&
                       points2D,
               const py::array_t<double,
                                 py::array::c_style | py::array::forcecast>&
                       points3D,
               const py::array_t<bool,
                                 py::array::c_style | py::array::forcecast>&
                       inlier_mask,
               const std::vector<size_t>& camera_idxs,
               const std::vector<Rigid3d>& cams_from_rig,
               const std::vector<Camera>& cameras,
               const AbsolutePoseRefinementOptions& options,
               bool return_covariance) -> py::object {
                auto points2 = Points2DFromNdarray(points2D);
                auto points3 = Points3DFromNdarray(points3D);
                auto mask_buf = inlier_mask.unchecked<1>();
                THROW_CHECK_EQ(points2.size(), points3.size());
                THROW_CHECK_EQ(points2.size(),
                               static_cast<size_t>(mask_buf.shape(0)));
                THROW_CHECK_EQ(points2.size(), camera_idxs.size());
                THROW_CHECK_EQ(cams_from_rig.size(), cameras.size());
                std::vector<char> mask(points2.size());
                for (py::ssize_t i = 0; i < mask_buf.shape(0); ++i) {
                    mask[i] = mask_buf(i) ? 1 : 0;
                }
                Sim3d refined = rig_from_world;
                std::vector<Camera> refined_cameras = cameras;
                Eigen::Matrix7d covariance;
                if (!colmap::RefineScaledGeneralizedAbsolutePose(
                            options, mask, points2, points3, camera_idxs,
                            cams_from_rig, &refined, &refined_cameras,
                            return_covariance ? &covariance : nullptr)) {
                    return py::none();
                }
                py::dict out;
                out["rig_from_world"] = refined;
                out["cameras"] = refined_cameras;
                if (return_covariance) {
                    out["covariance"] = covariance;
                }
                return out;
            },
            "rig_from_world"_a, "points2D"_a, "points3D"_a, "inlier_mask"_a,
            "camera_idxs"_a, "cams_from_rig"_a, "cameras"_a,
            "refinement_options"_a = AbsolutePoseRefinementOptions(),
            "return_covariance"_a = false,
            "Refine the generalized absolute pose and the rig scale "
            "(optionally focal lengths); returns a dict with rig_from_world "
            "(Sim3d) and cameras (and optionally the 7x7 covariance over "
            "rotation/translation/scale), or None.");

    m_estimators.def(
            "estimate_and_refine_scaled_generalized_absolute_pose",
            [](const py::array_t<double,
                                 py::array::c_style | py::array::forcecast>&
                       points2D,
               const py::array_t<double,
                                 py::array::c_style | py::array::forcecast>&
                       points3D,
               const std::vector<size_t>& camera_idxs,
               const std::vector<Rigid3d>& cams_from_rig,
               const std::vector<Camera>& cameras,
               const RANSACOptions& estimation_options,
               const AbsolutePoseRefinementOptions& refinement_options,
               bool return_covariance) -> py::object {
                auto points2 = Points2DFromNdarray(points2D);
                auto points3 = Points3DFromNdarray(points3D);
                THROW_CHECK_EQ(points2.size(), points3.size());
                THROW_CHECK_EQ(points2.size(), camera_idxs.size());
                THROW_CHECK_EQ(cams_from_rig.size(), cameras.size());
                Sim3d rig_from_world;
                size_t num_inliers = 0;
                std::vector<char> inlier_mask;
                if (!colmap::EstimateScaledGeneralizedAbsolutePose(
                            estimation_options, points2, points3, camera_idxs,
                            cams_from_rig, cameras, &rig_from_world,
                            &num_inliers, &inlier_mask)) {
                    return py::none();
                }
                std::vector<Camera> refined_cameras = cameras;
                Eigen::Matrix7d covariance;
                if (!colmap::RefineScaledGeneralizedAbsolutePose(
                            refinement_options, inlier_mask, points2, points3,
                            camera_idxs, cams_from_rig, &rig_from_world,
                            &refined_cameras,
                            return_covariance ? &covariance : nullptr)) {
                    return py::none();
                }
                py::dict out;
                out["rig_from_world"] = rig_from_world;
                out["num_inliers"] = num_inliers;
                out["inlier_mask"] = InlierMaskToList(inlier_mask);
                out["cameras"] = refined_cameras;
                if (return_covariance) {
                    out["covariance"] = covariance;
                }
                return out;
            },
            "points2D"_a, "points3D"_a, "camera_idxs"_a, "cams_from_rig"_a,
            "cameras"_a, "estimation_options"_a = RANSACOptions(),
            "refinement_options"_a = AbsolutePoseRefinementOptions(),
            "return_covariance"_a = false,
            "Robustly estimate the generalized absolute pose and the rig "
            "scale with LO-RANSAC followed by non-linear refinement; returns "
            "a dict with the refined rig_from_world (Sim3d), inliers, and "
            "cameras (and optionally the 7x7 covariance), or None.");

    m_estimators.def(
            "estimate_generalized_relative_pose",
            [](const py::array_t<double,
                                 py::array::c_style | py::array::forcecast>&
                       points2D1,
               const py::array_t<double,
                                 py::array::c_style | py::array::forcecast>&
                       points2D2,
               const std::vector<size_t>& camera_idxs1,
               const std::vector<size_t>& camera_idxs2,
               const std::vector<Rigid3d>& cams_from_rig,
               const std::vector<Camera>& cameras,
               const RANSACOptions& options) -> py::object {
                auto points2_1 = Points2DFromNdarray(points2D1);
                auto points2_2 = Points2DFromNdarray(points2D2);
                THROW_CHECK_EQ(points2_1.size(), camera_idxs1.size());
                THROW_CHECK_EQ(points2_2.size(), camera_idxs2.size());
                THROW_CHECK_EQ(cams_from_rig.size(), cameras.size());
                std::optional<Rigid3d> rig2_from_rig1;
                std::optional<Rigid3d> pano2_from_pano1;
                size_t num_inliers = 0;
                std::vector<char> inlier_mask;
                if (!colmap::EstimateGeneralizedRelativePose(
                            options, points2_1, points2_2, camera_idxs1,
                            camera_idxs2, cams_from_rig, cameras,
                            &rig2_from_rig1, &pano2_from_pano1, &num_inliers,
                            &inlier_mask)) {
                    return py::none();
                }
                py::dict out;
                out["num_inliers"] = num_inliers;
                out["inlier_mask"] = InlierMaskToList(inlier_mask);
                if (rig2_from_rig1) {
                    out["rig2_from_rig1"] = *rig2_from_rig1;
                }
                if (pano2_from_pano1) {
                    out["pano2_from_pano1"] = *pano2_from_pano1;
                }
                return out;
            },
            "points2D1"_a, "points2D2"_a, "camera_idxs1"_a, "camera_idxs2"_a,
            "cams_from_rig"_a, "cameras"_a,
            "estimation_options"_a = RANSACOptions(),
            "Robustly estimate the generalized relative pose between two rig "
            "views; returns a dict with num_inliers/inlier_mask plus "
            "rig2_from_rig1 (non-panoramic case) or pano2_from_pano1 (both "
            "panoramic), or None.");

    // Fork extension (engine estimators/generalized_pose.h): structure-less
    // resection (Zheng and Wu 2013). Upstream pycolmap does not expose this
    // entry point directly.
    py::class_<colmap::StructureLessAbsolutePoseEstimationOptions>
            struct_less_options(
                    m_estimators, "StructureLessAbsolutePoseEstimationOptions",
                    "Options for structure-less absolute pose estimation.");
    struct_less_options.def(py::init<>())
            .def_readwrite("ransac_options",
                           &colmap::StructureLessAbsolutePoseEstimationOptions::
                                   ransac_options);

    m_estimators.def(
            "estimate_structure_less_absolute_pose",
            [](const colmap::StructureLessAbsolutePoseEstimationOptions&
                       options,
               const py::array_t<double,
                                 py::array::c_style | py::array::forcecast>&
                       query_points2D,
               const py::array_t<double,
                                 py::array::c_style | py::array::forcecast>&
                       world_points2D,
               const std::vector<size_t>& world_camera_idxs,
               const std::vector<Rigid3d>& world_cams_from_world,
               const std::vector<Camera>& world_cameras,
               const Camera& query_camera) -> py::object {
                auto query_points = Points2DFromNdarray(query_points2D);
                auto world_points = Points2DFromNdarray(world_points2D);
                THROW_CHECK_EQ(query_points.size(), world_points.size());
                THROW_CHECK_EQ(world_points.size(), world_camera_idxs.size());
                THROW_CHECK_EQ(world_cams_from_world.size(),
                               world_cameras.size());
                Rigid3d query_cam_from_world;
                size_t num_inliers = 0;
                std::vector<char> inlier_mask;
                if (!colmap::EstimateStructureLessAbsolutePose(
                            options, query_points, world_points,
                            world_camera_idxs, world_cams_from_world,
                            world_cameras, query_camera, &query_cam_from_world,
                            &num_inliers, &inlier_mask)) {
                    return py::none();
                }
                py::dict out;
                out["query_cam_from_world"] = query_cam_from_world;
                out["num_inliers"] = num_inliers;
                out["inlier_mask"] = InlierMaskToList(inlier_mask);
                return out;
            },
            "options"_a, "query_points2D"_a, "world_points2D"_a,
            "world_camera_idxs"_a, "world_cams_from_world"_a, "world_cameras"_a,
            "query_camera"_a,
            "Estimate the absolute pose of a query camera from structure-less "
            "2D-2D correspondences against registered world cameras; returns a "
            "dict with query_cam_from_world/num_inliers/inlier_mask or None.");

    m_estimators.attr("__docstring__") = map_shared_argument_docstrings;
}

}  // namespace estimators
}  // namespace reconstruction
}  // namespace cloudViewer
