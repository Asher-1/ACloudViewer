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
#include "estimators/bundle_adjustment_caspar.h"
#include "estimators/bundle_adjustment_ceres.h"
#include "estimators/covariance.h"
#include "estimators/generalized_pose.h"
#include "estimators/global_positioning.h"
#include "estimators/gravity_refinement.h"
#include "estimators/pose.h"
#include "estimators/rotation_averaging.h"
#include "estimators/rotation_averaging_ceres.h"
#include "estimators/rotation_averaging_l1_irls.h"
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
#include "pybind/reconstruction/enum_ctor.h"
#include "pybind/reconstruction/estimators/covariance_bindings.h"
#include "pybind/reconstruction/hash_container_casters.h"
#include "scene/camera.h"
#include "scene/pose_graph.h"
#include "scene/rig.h"
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
            .def_readwrite("random_seed", &RANSACOptions::random_seed)
            .def("check", &RANSACOptions::Check,
                 "Validate the option values (throws on invalid values).");

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
                           &AbsolutePoseRefinementOptions::refine_extra_params)
            // Upstream pycolmap parity (src/pycolmap/estimators/pose.cc):
            // pose estimation/refinement entry points attached to the
            // refinement options. The fork's estimate_relative_pose was
            // retired on the engine side and refine_relative_pose takes a
            // raw ceres::Solver::Options (no pyceres binding layer), so
            // both stay unbound and are recorded in the alignment manifest.
            .def(
                    "estimate_absolute_pose",
                    [](const AbsolutePoseRefinementOptions&,
                       const AbsolutePoseEstimationOptions& est_options,
                       const py::array_t<double, py::array::c_style |
                                                         py::array::forcecast>&
                               points2D,
                       const py::array_t<double, py::array::c_style |
                                                         py::array::forcecast>&
                               points3D,
                       Camera& camera) -> py::object {
                        auto pts2 = Points2DFromNdarray(points2D);
                        auto pts3 = Points3DFromNdarray(points3D);
                        Eigen::Vector4d qvec;
                        Eigen::Vector3d tvec;
                        size_t num_inliers = 0;
                        std::vector<char> inlier_mask;
                        if (!colmap::EstimateAbsolutePose(
                                    est_options, pts2, pts3, &qvec, &tvec,
                                    &camera, &num_inliers, &inlier_mask)) {
                            return py::none();
                        }
                        py::dict out;
                        out["cam_from_world"] =
                                Rigid3d(Eigen::Quaterniond(qvec[0], qvec[1],
                                                           qvec[2], qvec[3]),
                                        tvec);
                        out["num_inliers"] = num_inliers;
                        out["inlier_mask"] = InlierMaskToList(inlier_mask);
                        return out;
                    },
                    "estimation_options"_a, "points2D"_a, "points3D"_a,
                    "camera"_a,
                    "Estimate the absolute pose (PnP) of a camera from 2D-3D "
                    "correspondences; returns a dict with cam_from_world / "
                    "num_inliers / inlier_mask or None. The camera is "
                    "modified in-place to store the estimated focal "
                    "length.")
            .def(
                    "refine_absolute_pose",
                    [](const AbsolutePoseRefinementOptions& options,
                       const py::array_t<bool, py::array::c_style |
                                                       py::array::forcecast>&
                               inlier_mask_array,
                       const py::array_t<double, py::array::c_style |
                                                         py::array::forcecast>&
                               points2D,
                       const py::array_t<double, py::array::c_style |
                                                         py::array::forcecast>&
                               points3D,
                       Rigid3d& cam_from_world, Camera& camera) -> py::object {
                        // Match the module-level surface: numpy bool masks
                        // are accepted (the repo-wide std::vector<char>
                        // caster rejects plain lists).
                        std::vector<char> inlier_mask(inlier_mask_array.size());
                        {
                            auto buf = inlier_mask_array.unchecked<1>();
                            for (py::ssize_t i = 0; i < buf.shape(0); ++i) {
                                inlier_mask[static_cast<size_t>(i)] =
                                        buf[i] ? 1 : 0;
                            }
                        }
                        auto pts2 = Points2DFromNdarray(points2D);
                        auto pts3 = Points3DFromNdarray(points3D);
                        if (!colmap::RefineAbsolutePose(
                                    options, inlier_mask, pts2, pts3,
                                    &cam_from_world, &camera)) {
                            return py::none();
                        }
                        py::dict out;
                        out["cam_from_world"] = cam_from_world;
                        return out;
                    },
                    "inlier_mask"_a, "points2D"_a, "points3D"_a,
                    "cam_from_world"_a, "camera"_a,
                    "Refine the absolute pose of a camera from 2D-3D "
                    "correspondences; returns a dict with cam_from_world or "
                    "None.")
            .def(
                    "estimate_and_refine_absolute_pose",
                    [](const AbsolutePoseRefinementOptions& ref_options,
                       const AbsolutePoseEstimationOptions& est_options,
                       const py::array_t<double, py::array::c_style |
                                                         py::array::forcecast>&
                               points2D,
                       const py::array_t<double, py::array::c_style |
                                                         py::array::forcecast>&
                               points3D,
                       Camera& camera) -> py::object {
                        auto pts2 = Points2DFromNdarray(points2D);
                        auto pts3 = Points3DFromNdarray(points3D);
                        // Binding-level composition of the engine's
                        // EstimateAbsolutePose + RefineAbsolutePose (the
                        // fork engine has no fused entry point).
                        Eigen::Vector4d qvec;
                        Eigen::Vector3d tvec;
                        size_t num_inliers = 0;
                        std::vector<char> inlier_mask;
                        if (!colmap::EstimateAbsolutePose(
                                    est_options, pts2, pts3, &qvec, &tvec,
                                    &camera, &num_inliers, &inlier_mask)) {
                            return py::none();
                        }
                        Rigid3d cam_from_world(
                                Eigen::Quaterniond(qvec[0], qvec[1], qvec[2],
                                                   qvec[3]),
                                tvec);
                        if (!colmap::RefineAbsolutePose(
                                    ref_options, inlier_mask, pts2, pts3,
                                    &cam_from_world, &camera)) {
                            return py::none();
                        }
                        py::dict out;
                        out["cam_from_world"] = cam_from_world;
                        out["num_inliers"] = num_inliers;
                        out["inlier_mask"] = InlierMaskToList(inlier_mask);
                        return out;
                    },
                    "estimation_options"_a = AbsolutePoseEstimationOptions(),
                    "points2D"_a, "points3D"_a, "camera"_a,
                    "Estimate and refine the absolute pose of a camera from "
                    "2D-3D correspondences; returns a dict with "
                    "cam_from_world / num_inliers / inlier_mask or None.");

    py::class_<EstimateTriangulationOptions> tri_opt(
            m_estimators, "EstimateTriangulationOptions",
            "Options for multi-view triangulation.");
    tri_opt.def(py::init<>())
            .def_readwrite("min_tri_angle",
                           &EstimateTriangulationOptions::min_tri_angle)
            .def_readwrite("ransac_options",
                           &EstimateTriangulationOptions::ransac_options)
            .def(
                    "estimate_triangulation",
                    [](const EstimateTriangulationOptions& options,
                       const py::array_t<double, py::array::c_style |
                                                         py::array::forcecast>&
                               points,
                       const std::vector<Rigid3d>& cams_from_world,
                       const std::vector<Camera>& cameras) -> py::object {
                        auto pts = Points2DFromNdarray(points);
                        THROW_CHECK_EQ(cams_from_world.size(), cameras.size());
                        std::vector<const Camera*> camera_ptrs;
                        camera_ptrs.reserve(cameras.size());
                        for (const auto& camera : cameras) {
                            camera_ptrs.push_back(&camera);
                        }
                        std::vector<char> inlier_mask;
                        Eigen::Vector3d xyz;
                        if (!colmap::EstimateTriangulation(
                                    options, pts, cams_from_world, camera_ptrs,
                                    &inlier_mask, &xyz)) {
                            return py::none();
                        }
                        py::dict out;
                        out["xyz"] = xyz;
                        out["inlier_mask"] = InlierMaskToList(inlier_mask);
                        return out;
                    },
                    "points"_a, "cams_from_world"_a, "cameras"_a,
                    "Triangulate a 3D point from multiple views; returns a "
                    "dict with xyz / inlier_mask or None.");

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
            .def_readwrite("ransac", &TwoViewGeometryOptions::ransac_options)
            // Upstream pycolmap parity: the two-view geometry entry points
            // are methods on the options class. The same functionality is
            // also available as module-level free functions (bound below).
            .def(
                    "estimate_two_view_geometry",
                    [](const TwoViewGeometryOptions& options,
                       const Camera& camera1,
                       const py::array_t<double, py::array::c_style |
                                                         py::array::forcecast>&
                               points1,
                       const Camera& camera2,
                       const py::array_t<double, py::array::c_style |
                                                         py::array::forcecast>&
                               points2,
                       const py::object& matches) {
                        FeatureMatches feature_matches =
                                MatchesFromOptionalNdarray(matches);
                        auto pts1 = Points2DFromNdarray(points1);
                        auto pts2 = Points2DFromNdarray(points2);
                        if (feature_matches.empty()) {
                            THROW_CHECK_EQ(pts1.size(), pts2.size());
                            feature_matches.reserve(pts1.size());
                            for (size_t i = 0; i < pts1.size(); ++i) {
                                feature_matches.emplace_back(i, i);
                            }
                        }
                        return colmap::EstimateTwoViewGeometry(
                                camera1, pts1, camera2, pts2,
                                std::move(feature_matches), options);
                    },
                    "camera1"_a, "points1"_a, "camera2"_a, "points2"_a,
                    "matches"_a = py::none(),
                    "Estimate the two-view geometry from an image pair.")
            .def(
                    "estimate_calibrated_two_view_geometry",
                    [](const TwoViewGeometryOptions& options,
                       const Camera& camera1,
                       const py::array_t<double, py::array::c_style |
                                                         py::array::forcecast>&
                               points1,
                       const Camera& camera2,
                       const py::array_t<double, py::array::c_style |
                                                         py::array::forcecast>&
                               points2,
                       const py::object& matches) {
                        FeatureMatches feature_matches =
                                MatchesFromOptionalNdarray(matches);
                        auto pts1 = Points2DFromNdarray(points1);
                        auto pts2 = Points2DFromNdarray(points2);
                        if (feature_matches.empty()) {
                            THROW_CHECK_EQ(pts1.size(), pts2.size());
                            feature_matches.reserve(pts1.size());
                            for (size_t i = 0; i < pts1.size(); ++i) {
                                feature_matches.emplace_back(i, i);
                            }
                        }
                        return colmap::EstimateCalibratedTwoViewGeometry(
                                camera1, pts1, camera2, pts2, feature_matches,
                                options);
                    },
                    "camera1"_a, "points1"_a, "camera2"_a, "points2"_a,
                    "matches"_a = py::none(),
                    "Estimate the two-view geometry from a calibrated image "
                    "pair.")
            .def(
                    "estimate_two_view_geometry_pose",
                    [](const TwoViewGeometryOptions&, const Camera& camera1,
                       const py::array_t<double, py::array::c_style |
                                                         py::array::forcecast>&
                               points1,
                       const Camera& camera2,
                       const py::array_t<double, py::array::c_style |
                                                         py::array::forcecast>&
                               points2,
                       TwoViewGeometry& geometry) -> py::object {
                        auto pts1 = Points2DFromNdarray(points1);
                        auto pts2 = Points2DFromNdarray(points2);
                        if (!colmap::EstimateTwoViewGeometryPose(
                                    camera1, pts1, camera2, pts2, &geometry)) {
                            return py::none();
                        }
                        return py::cast(geometry);
                    },
                    "camera1"_a, "points1"_a, "camera2"_a, "points2"_a,
                    "geometry"_a,
                    "Decompose the relative pose and triangulation angle on an "
                    "estimated two-view geometry.")
            .def(
                    "compute_squared_sampson_error",
                    [](const TwoViewGeometryOptions&,
                       const py::array_t<double, py::array::c_style |
                                                         py::array::forcecast>&
                               points1,
                       const py::array_t<double, py::array::c_style |
                                                         py::array::forcecast>&
                               points2,
                       const Eigen::Matrix3d& E) {
                        auto pts1 = Points2DFromNdarray(points1);
                        auto pts2 = Points2DFromNdarray(points2);
                        THROW_CHECK_EQ(pts1.size(), pts2.size());
                        std::vector<double> residuals(pts1.size());
                        colmap::ComputeSquaredSampsonError(pts1, pts2, E,
                                                           &residuals);
                        return residuals;
                    },
                    "points1"_a, "points2"_a, "E"_a,
                    "Compute the squared Sampson error for a list of "
                    "correspondences w.r.t. the given essential/fundamental "
                    "matrix.");

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

    // Upstream pycolmap parity (src/pycolmap/estimators/alignment.cc): the
    // alignment entry points are exposed on the ImageAlignmentError class;
    // the same operations are also available as module-level free functions
    // below.
    alignment_error
            .def_static(
                    "align_reconstruction_to_locations",
                    [](const Reconstruction& src,
                       const std::vector<std::string>& tgt_image_names,
                       const std::vector<Eigen::Vector3d>& tgt_locations,
                       int min_common_images,
                       const RANSACOptions& ransac_options) -> py::object {
                        Sim3d tgt_from_src;
                        if (!colmap::AlignReconstructionToLocations(
                                    src, tgt_image_names, tgt_locations,
                                    min_common_images, ransac_options,
                                    &tgt_from_src)) {
                            return py::none();
                        }
                        return py::cast(tgt_from_src);
                    },
                    "src"_a, "tgt_image_names"_a, "tgt_image_locations"_a,
                    "min_common_images"_a, "ransac_options"_a,
                    "Align the reconstruction to target locations via "
                    "common images; returns tgt_from_src or None.")
            .def_static(
                    "align_reconstructions_via_reprojections",
                    [](const Reconstruction& src, const Reconstruction& tgt,
                       double min_inlier_observations,
                       double max_reproj_error) -> py::object {
                        Sim3d tgt_from_src;
                        if (!colmap::AlignReconstructionsViaReprojections(
                                    src, tgt, min_inlier_observations,
                                    max_reproj_error, &tgt_from_src)) {
                            return py::none();
                        }
                        return py::cast(tgt_from_src);
                    },
                    "src"_a, "tgt"_a, "min_inlier_observations"_a,
                    "max_reproj_error"_a,
                    "Align two reconstructions via common image "
                    "reprojections; returns tgt_from_src or None.")
            .def_static(
                    "align_reconstructions_via_proj_centers",
                    [](const Reconstruction& src, const Reconstruction& tgt,
                       double max_proj_center_error) -> py::object {
                        Sim3d tgt_from_src;
                        if (!colmap::AlignReconstructionsViaProjCenters(
                                    src, tgt, max_proj_center_error,
                                    &tgt_from_src)) {
                            return py::none();
                        }
                        return py::cast(tgt_from_src);
                    },
                    "src"_a, "tgt"_a, "max_proj_center_error"_a,
                    "Align two reconstructions via common projection "
                    "centers; returns tgt_from_src or None.")
            .def_static(
                    "align_reconstructions_via_points",
                    [](const Reconstruction& src, const Reconstruction& tgt,
                       size_t min_common_observations, double max_error,
                       double min_inlier_ratio) -> py::object {
                        Sim3d tgt_from_src;
                        if (!colmap::AlignReconstructionsViaPoints(
                                    src, tgt, min_common_observations,
                                    max_error, min_inlier_ratio,
                                    &tgt_from_src)) {
                            return py::none();
                        }
                        return py::cast(tgt_from_src);
                    },
                    "src"_a, "tgt"_a, "min_common_observations"_a,
                    "max_error"_a, "min_inlier_ratio"_a,
                    "Align two reconstructions via common 3D points; "
                    "returns tgt_from_src or None.")
            .def_static(
                    "align_reconstruction_to_orig_rig_scales",
                    [](Reconstruction& reconstruction,
                       const std::unordered_map<uint32_t, colmap::Rig>&
                               orig_rigs) {
                        colmap::NodeHashMap<colmap::rig_t, colmap::Rig> rigs;
                        for (const auto& [rig_id, rig] : orig_rigs) {
                            rigs.emplace(rig_id, rig);
                        }
                        return colmap::AlignReconstructionToOrigRigScales(
                                rigs, &reconstruction);
                    },
                    "reconstruction"_a, "orig_rigs"_a,
                    "Align the reconstruction rig scales to the original "
                    "rigs (dict of rig_id -> Rig); returns True on success.")
            .def_static(
                    "compare_reconstructions",
                    [](const Reconstruction& src, const Reconstruction& tgt,
                       const Sim3d& tgt_from_src) {
                        auto errors = colmap::ComputeImageAlignmentError(
                                src, tgt, tgt_from_src);
                        py::list err_list;
                        for (const auto& err : errors) {
                            err_list.append(err);
                        }
                        auto summary =
                                colmap::AlignmentErrorSummary::Compute(errors);
                        py::dict out;
                        out["image_errors"] = err_list;
                        out["summary"] = summary;
                        return out;
                    },
                    "src"_a, "tgt"_a, "tgt_from_src"_a,
                    "Compare two aligned reconstructions; returns a dict "
                    "with image_errors and summary statistics.");

    // Upstream pycolmap parity: AlignmentErrorSummary value type (returned
    // by compare_reconstructions).
    py::class_<colmap::AlignmentErrorSummary> alignment_summary(
            m_estimators, "AlignmentErrorSummary",
            "Aggregate (min/max/mean/median/p90/p99) alignment error "
            "statistics.");
    py::class_<colmap::AlignmentErrorSummary::Statistics> alignment_stats(
            alignment_summary, "Statistics",
            "Error distribution "
            "statistics.");
    alignment_stats
            .def_readwrite("min",
                           &colmap::AlignmentErrorSummary::Statistics::min)
            .def_readwrite("max",
                           &colmap::AlignmentErrorSummary::Statistics::max)
            .def_readwrite("mean",
                           &colmap::AlignmentErrorSummary::Statistics::mean)
            .def_readwrite("median",
                           &colmap::AlignmentErrorSummary::Statistics::median)
            .def_readwrite("p90",
                           &colmap::AlignmentErrorSummary::Statistics::p90)
            .def_readwrite("p99",
                           &colmap::AlignmentErrorSummary::Statistics::p99)
            .def("__repr__",
                 [](const colmap::AlignmentErrorSummary::Statistics& self) {
                     return "Statistics(min=" + std::to_string(self.min) +
                            ", max=" + std::to_string(self.max) +
                            ", mean=" + std::to_string(self.mean) + ")";
                 });
    alignment_summary
            .def_readwrite("rotation_errors_deg",
                           &colmap::AlignmentErrorSummary::rotation_errors_deg)
            .def_readwrite("proj_center_errors",
                           &colmap::AlignmentErrorSummary::proj_center_errors)
            .def_static(
                    "compute",
                    [](const std::vector<colmap::ImageAlignmentError>& errors) {
                        return colmap::AlignmentErrorSummary::Compute(errors);
                    },
                    "image_errors"_a,
                    "Compute summary statistics from per-image errors.");

    // Upstream pycolmap parity: align_reconstruction_to_orig_rig_scales.
    m_estimators.def(
            "align_reconstruction_to_orig_rig_scales",
            [](Reconstruction& reconstruction,
               const std::unordered_map<uint32_t, colmap::Rig>& orig_rigs) {
                colmap::NodeHashMap<colmap::rig_t, colmap::Rig> rigs;
                for (const auto& [rig_id, rig] : orig_rigs) {
                    rigs.emplace(rig_id, rig);
                }
                return colmap::AlignReconstructionToOrigRigScales(
                        rigs, &reconstruction);
            },
            "reconstruction"_a, "orig_rigs"_a,
            "Align the reconstruction rig scales to the original rigs "
            "(dict of rig_id -> Rig); returns True on success. The "
            "reconstruction is updated in-place.");

    // Upstream pycolmap parity (src/pycolmap/scene/rig.cc): the rig
    // configuration family and the apply_rig_config entry point.
    py::class_<colmap::RigConfig::RigCamera> rig_config_camera(
            m_estimators, "RigConfigCamera",
            "Per-sensor entry of a rig configuration.");
    rig_config_camera.def(py::init<>())
            .def_readwrite("ref_sensor",
                           &colmap::RigConfig::RigCamera::ref_sensor,
                           "bool: Whether this sensor is the reference "
                           "sensor of the rig.")
            .def_readwrite("image_prefix",
                           &colmap::RigConfig::RigCamera::image_prefix,
                           "str: The image filename prefix matched against "
                           "this sensor.")
            .def_readwrite(
                    "cam_from_rig", &colmap::RigConfig::RigCamera::cam_from_rig,
                    "Rigid3d or None: Optional known sensor-from-rig pose.")
            .def_readwrite("camera", &colmap::RigConfig::RigCamera::camera,
                           "Camera or None: Optional known camera intrinsics.")
            .def("__repr__", [](const colmap::RigConfig::RigCamera& self) {
                return "RigConfigCamera(image_prefix='" + self.image_prefix +
                       "', ref_sensor=" + (self.ref_sensor ? "True" : "False") +
                       ")";
            });

    py::class_<colmap::RigConfig> rig_config(
            m_estimators, "RigConfig",
            "Configuration of one rig (list of sensors with image prefix "
            "matching rules). See also ReadRigConfig/ApplyRigConfig.");
    rig_config.def(py::init<>())
            .def_readwrite("cameras", &colmap::RigConfig::cameras,
                           "list(RigConfigCamera, ...): The sensors of the "
                           "rig. (Upstream also exposes rig_name; the fork "
                           "config has no name member.)")
            .def("__repr__", [](const colmap::RigConfig& self) {
                return "RigConfig(num_cameras=" +
                       std::to_string(self.cameras.size()) + ")";
            });

    m_estimators.def(
            "read_rig_config",
            [](const std::string& path) { return colmap::ReadRigConfig(path); },
            "path"_a, "Read a rig configuration from a JSON file.");

    m_estimators.def(
            "apply_rig_config",
            [](const std::vector<colmap::RigConfig>& configs,
               colmap::Database& database,
               colmap::Reconstruction* reconstruction) {
                colmap::ApplyRigConfig(configs, database, reconstruction);
            },
            "configs"_a, "database"_a, "reconstruction"_a = py::none(),
            "Apply the rig configuration to the database and optionally "
            "derive camera rig extrinsics/intrinsics from the "
            "reconstruction. Existing rigs and frames are cleared.");

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

    // Upstream pycolmap parity (src/pycolmap/estimators/bundle_adjustment.cc):
    // the backend-typed summary. The fork's ceres_summary member is the raw
    // ceres::Solver::Summary, which belongs to the embedded-ceres binding
    // subproject (recorded in the alignment manifest); the common summary
    // surface is fully available here.
    py::class_<colmap::CeresBundleAdjustmentSummary,
               colmap::BundleAdjustmentSummary>(
            m_estimators, "CeresBundleAdjustmentSummary",
            "Summary of a Ceres bundle adjustment run.")
            .def(py::init<>());

    // Upstream pycolmap parity: options for pose-prior-regularized bundle
    // adjustment, split into solver-agnostic and Ceres-specific parts.
    py::enum_<colmap::CeresLossFunctionType>(
            m_estimators, "CeresLossFunctionType",
            "Robust loss function used by the Ceres bundle adjuster.")
            .value("TRIVIAL", colmap::CeresLossFunctionType::TRIVIAL)
            .value("SOFT_L1", colmap::CeresLossFunctionType::SOFT_L1)
            .value("CAUCHY", colmap::CeresLossFunctionType::CAUCHY)
            .value("HUBER", colmap::CeresLossFunctionType::HUBER);

    py::class_<colmap::CeresPosePriorBundleAdjustmentOptions,
               std::shared_ptr<colmap::CeresPosePriorBundleAdjustmentOptions>>(
            m_estimators, "CeresPosePriorBundleAdjustmentOptions",
            "Ceres-specific pose prior bundle adjustment options.")
            .def(py::init<>())
            .def_readwrite("prior_position_loss_function_type",
                           &colmap::CeresPosePriorBundleAdjustmentOptions::
                                   prior_position_loss_function_type,
                           "Loss function for prior position loss.")
            .def_readwrite("prior_position_loss_scale",
                           &colmap::CeresPosePriorBundleAdjustmentOptions::
                                   prior_position_loss_scale,
                           "Threshold on the residual for the robust loss "
                           "(chi2 for 3DOF at 95% = 7.815).")
            .def("check",
                 &colmap::CeresPosePriorBundleAdjustmentOptions::Check);

    py::class_<colmap::PosePriorBundleAdjustmentOptions> pose_prior_ba_options(
            m_estimators, "PosePriorBundleAdjustmentOptions",
            "Options for pose-prior-regularized bundle adjustment.");
    pose_prior_ba_options.def(py::init<>())
            .def_readwrite(
                    "prior_position_fallback_stddev",
                    &colmap::PosePriorBundleAdjustmentOptions::
                            prior_position_fallback_stddev,
                    "Fallback if no prior position covariance is provided.")
            .def_readwrite(
                    "alignment_ransac",
                    &colmap::PosePriorBundleAdjustmentOptions::
                            alignment_ransac_options,
                    "RANSAC options for aligning the reconstruction to the "
                    "priors before the adjustment.")
            .def_readwrite("ceres",
                           &colmap::PosePriorBundleAdjustmentOptions::ceres,
                           "Ceres-specific pose prior bundle adjustment "
                           "options.")
            .def("check", &colmap::PosePriorBundleAdjustmentOptions::Check,
                 "Validate the option values.");

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
            // Upstream pycolmap parity names for the camera-intrinsics
            // freezing family (the fork engine names them cameras).
            .def("set_constant_cam_intrinsics",
                 &colmap::BundleAdjustmentConfig::SetConstantCamera,
                 "camera_id"_a, "Upstream name of ``set_constant_camera``.")
            .def("set_variable_cam_intrinsics",
                 &colmap::BundleAdjustmentConfig::SetVariableCamera,
                 "camera_id"_a, "Upstream name of ``set_variable_camera``.")
            .def("has_constant_cam_intrinsics",
                 &colmap::BundleAdjustmentConfig::IsConstantCamera,
                 "camera_id"_a, "Upstream name of ``is_constant_camera``.")
            .def("num_constant_cam_intrinsics",
                 &colmap::BundleAdjustmentConfig::NumConstantCameras,
                 "Upstream name of ``num_constant_cameras``.")
            .def("num_constant_rig_from_world_poses",
                 &colmap::BundleAdjustmentConfig::NumConstantPoses,
                 "Upstream name of ``num_constant_poses``.")
            .def("num_constant_sensor_from_rig_poses",
                 &colmap::BundleAdjustmentConfig::NumConstantSensorFromRigPoses,
                 "The number of constant sensor-from-rig poses.")
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
    using WeightType = colmap::L1IrlsRotationAveragerOptions::WeightType;
    py::enum_<WeightType> rot_weight(
            m_estimators, "RotationWeightType",
            "Robust weight for rotation averaging IRLS.");
    rot_weight.value("GEMAN_MCCLURE", WeightType::GEMAN_MCCLURE)
            .value("HALF_NORM", WeightType::HALF_NORM);
    AddStringToEnumConstructor(rot_weight);

    py::enum_<colmap::RotationAveragingBackend> ra_backend(
            m_estimators, "RotationAveragingBackend",
            "Solver backend for rotation averaging.");
    ra_backend.value("L1_IRLS", colmap::RotationAveragingBackend::L1_IRLS)
            .value("CERES", colmap::RotationAveragingBackend::CERES);
    AddStringToEnumConstructor(ra_backend);

    py::class_<colmap::L1IrlsRotationAveragerOptions,
               std::shared_ptr<colmap::L1IrlsRotationAveragerOptions>>(
            m_estimators, "L1IrlsRotationAveragerOptions",
            "L1-IRLS-specific rotation averaging options.")
            .def(py::init<>())
            .def_readwrite("max_num_l1_iterations",
                           &colmap::L1IrlsRotationAveragerOptions::
                                   max_num_l1_iterations,
                           "Maximum number of L1 minimization iterations.")
            .def_readwrite("l1_step_convergence_threshold",
                           &colmap::L1IrlsRotationAveragerOptions::
                                   l1_step_convergence_threshold,
                           "Average step size threshold to terminate L1 "
                           "minimization.")
            .def_readwrite("max_num_irls_iterations",
                           &colmap::L1IrlsRotationAveragerOptions::
                                   max_num_irls_iterations,
                           "Number of IRLS iterations to perform.")
            .def_readwrite("irls_step_convergence_threshold",
                           &colmap::L1IrlsRotationAveragerOptions::
                                   irls_step_convergence_threshold,
                           "Average step size threshold to terminate IRLS.")
            .def_readwrite("irls_loss_parameter_sigma",
                           &colmap::L1IrlsRotationAveragerOptions::
                                   irls_loss_parameter_sigma,
                           "Scale of the Geman-McClure IRLS weight "
                           "(degrees). Unused by HALF_NORM.")
            .def_readwrite(
                    "ridge_regularization",
                    &colmap::L1IrlsRotationAveragerOptions::
                            ridge_regularization,
                    "Tikhonov ridge added to the diagonal of the normal "
                    "equations before each Cholesky factorization in the L1 "
                    "and IRLS phases. Zero disables regularization.")
            .def_readwrite("weight_type",
                           &colmap::L1IrlsRotationAveragerOptions::weight_type,
                           "Weight type for IRLS: GEMAN_MCCLURE or HALF_NORM.");

    py::class_<colmap::CeresRotationAveragerOptions,
               std::shared_ptr<colmap::CeresRotationAveragerOptions>>(
            m_estimators, "CeresRotationAveragerOptions",
            "Ceres-specific rotation averaging options.")
            .def(py::init<>())
            .def_readwrite(
                    "loss_function_type",
                    &colmap::CeresRotationAveragerOptions::loss_function_type,
                    "Loss function for relative-rotation residuals.")
            .def_readwrite(
                    "loss_function_scale",
                    &colmap::CeresRotationAveragerOptions::loss_function_scale,
                    "Loss function scale in degrees.")
            .def_readwrite("max_num_warm_start_iterations",
                           &colmap::CeresRotationAveragerOptions::
                                   max_num_warm_start_iterations,
                           "Number of solver iterations of a Huber-loss "
                           "warm-start (with the same loss scale) run by "
                           "RotationEstimator before the main solve. Set to "
                           "0 to disable.")
            .def_readwrite(
                    "solver_options",
                    &colmap::CeresRotationAveragerOptions::solver_options,
                    "pyceres.SolverOptions: Options for the Ceres solver.");

    py::class_<colmap::RotationEstimatorOptions> ra_options(
            m_estimators, "RotationEstimatorOptions",
            "Options for (GLomap-style) rotation averaging.");
    ra_options.def(py::init<>())
            .def_readwrite("random_seed",
                           &colmap::RotationEstimatorOptions::random_seed,
                           "PRNG seed. -1 for non-deterministic, >=0 for "
                           "deterministic.")
            .def_readwrite("backend",
                           &colmap::RotationEstimatorOptions::backend,
                           "Solver backend for rotation averaging: L1_IRLS "
                           "or CERES. CERES falls back to L1_IRLS when "
                           "use_gravity is set and gravity priors are given.")
            .def_readwrite("l1_irls",
                           &colmap::RotationEstimatorOptions::l1_irls,
                           "L1-IRLS-specific rotation averaging options (used "
                           "when backend == L1_IRLS and when CERES falls back "
                           "to L1_IRLS for gravity priors).")
            .def_readwrite("ceres", &colmap::RotationEstimatorOptions::ceres,
                           "Ceres-specific rotation averaging options (only "
                           "used when backend == CERES).")
            .def_readwrite("gravity_dir",
                           &colmap::RotationEstimatorOptions::gravity_dir,
                           "Gravity direction vector.")
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
            .def_readwrite("reweighting",
                           &colmap::RotationEstimatorOptions::reweighting,
                           "Reweighting scheme for the relative-rotation "
                           "constraints: UNIFORM or INLIER_MATCH_COUNT.")
            .def(
                    "run_rotation_averaging",
                    [](const colmap::RotationEstimatorOptions& options,
                       colmap::PoseGraph& pose_graph,
                       Reconstruction& reconstruction,
                       std::vector<colmap::PosePrior> pose_priors) {
                        return colmap::RunRotationAveraging(options, pose_graph,
                                                            reconstruction,
                                                            pose_priors);
                    },
                    "pose_graph"_a, "reconstruction"_a,
                    "pose_priors"_a = std::vector<colmap::PosePrior>{},
                    "Run the high-level rotation averaging solver that handles "
                    "rig expansion; returns True on success.");

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

    // Upstream pycolmap parity (src/pycolmap/estimators/
    // rotation_averaging_ceres.cc). Fork adaptations: the embedded pyceres
    // surface binds no ceres::Problem / LossFunction, so `problem` is the
    // opaque CeresProblem handle (block counts only) and residuals added from
    // Python use the trivial loss.
    py::class_<colmap::CeresRotationAverager>(
            m_estimators, "CeresRotationAverager",
            "Ceres rotation averaging problem that optimizes rotations "
            "directly in the reconstruction.")
            .def("solve", &colmap::CeresRotationAverager::Solve,
                 py::call_guard<py::gil_scoped_release>(),
                 "Optimize rotations in the reconstruction and return the "
                 "solver summary.")
            .def(
                    "add_relative_rotation_residual",
                    [](colmap::CeresRotationAverager& self,
                       const colmap::image_t image_id1,
                       const colmap::image_t image_id2,
                       const Eigen::Quaterniond& cam2_from_cam1) {
                        self.AddRelativeRotationResidual(
                                image_id1, image_id2, cam2_from_cam1, nullptr);
                    },
                    "image_id1"_a, "image_id2"_a, "cam2_from_cam1"_a,
                    "Add a trivial-loss relative-rotation residual between "
                    "images whose frame and sensor rotations are configured "
                    "in the problem.")
            .def_property_readonly(
                    "problem",
                    [](colmap::CeresRotationAverager& self) {
                        return PyCeresProblem{&self.Problem()};
                    },
                    py::keep_alive<0, 1>(),
                    "Opaque handle of the Ceres problem.")
            .def_readwrite("solver_options",
                           &colmap::CeresRotationAverager::solver_options,
                           "Ceres solver options.");

    m_estimators.def("create_default_ceres_rotation_averager",
                     &colmap::CreateDefaultCeresRotationAverager, "options"_a,
                     "pose_graph"_a, "reconstruction"_a, py::keep_alive<0, 3>(),
                     "Create a Ceres rotation averaging problem that "
                     "optimizes rotations directly in the reconstruction.");

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
            .def_readwrite("initialize_scales_from_geometry",
                           &colmap::GlobalPositionerOptions::
                                   initialize_scales_from_geometry)
            .def_readwrite("optimize_positions",
                           &colmap::GlobalPositionerOptions::optimize_positions)
            .def_readwrite("optimize_points",
                           &colmap::GlobalPositionerOptions::optimize_points)
            .def_readwrite("optimize_scales",
                           &colmap::GlobalPositionerOptions::optimize_scales)
            .def_readwrite("fix_first_scale",
                           &colmap::GlobalPositionerOptions::fix_first_scale)
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
            .def_readwrite("experimental_observation_stddev",
                           &colmap::GlobalPositionerOptions::
                                   experimental_observation_stddev)
            .def_readwrite("uncalibrated_observation_weight",
                           &colmap::GlobalPositionerOptions::
                                   uncalibrated_observation_weight)
            .def_readwrite("use_parameter_block_ordering",
                           &colmap::GlobalPositionerOptions::
                                   use_parameter_block_ordering)
            .def(
                    "run_global_positioning",
                    [](const colmap::GlobalPositionerOptions& options,
                       const colmap::PoseGraph& pose_graph,
                       Reconstruction& reconstruction) {
                        return colmap::RunGlobalPositioning(options, pose_graph,
                                                            reconstruction);
                    },
                    "pose_graph"_a, "reconstruction"_a,
                    "Solve global positioning using point-to-camera "
                    "constraints; returns True on success.");

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
                           &colmap::GravityRefinerOptions::min_num_neighbors)
            .def(
                    "run_gravity_refinement",
                    [](const colmap::GravityRefinerOptions& options,
                       const colmap::PoseGraph& pose_graph,
                       const Reconstruction& reconstruction,
                       std::vector<colmap::PosePrior>& pose_priors) {
                        colmap::RunGravityRefinement(options, pose_graph,
                                                     reconstruction,
                                                     pose_priors);
                        return pose_priors;
                    },
                    "pose_graph"_a, "reconstruction"_a, "pose_priors"_a,
                    "Refine the gravity of pose priors in place using relative "
                    "rotations from the pose graph; returns the refined pose "
                    "prior list.");

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

    // ------------------------------------------------------------------
    // Upstream pycolmap parity (top-level names; the engine declares these
    // enums nested - the bindings export them at module level exactly like
    // upstream pycolmap does).
    // ------------------------------------------------------------------

    py::enum_<colmap::RotationAveragingReweighting>(
            m_estimators, "RotationAveragingReweighting",
            "The reweighting scheme applied to the relative-rotation "
            "constraints.")
            .value("UNIFORM", colmap::RotationAveragingReweighting::UNIFORM)
            .value("INLIER_MATCH_COUNT",
                   colmap::RotationAveragingReweighting::INLIER_MATCH_COUNT)
            .export_values();

    py::enum_<colmap::TriangulationEstimator::ResidualType>(
            m_estimators, "TriangulationResidualType",
            "The residual type used during triangulation.")
            .value("ANGULAR_ERROR",
                   colmap::TriangulationEstimator::ResidualType::ANGULAR_ERROR)
            .value("REPROJECTION_ERROR",
                   colmap::TriangulationEstimator::ResidualType::
                           REPROJECTION_ERROR)
            .export_values();

    // Upstream parity (estimators/bundle_adjustment_caspar.h): the caspar
    // solver-parameter contract, shared with BundleAdjustmentOptions::caspar
    // (shared_ptr holder).
    py::class_<colmap::CasparBundleAdjustmentOptions,
               std::shared_ptr<colmap::CasparBundleAdjustmentOptions>>(
            m_estimators, "CasparBundleAdjustmentOptions")
            .def(py::init<>())
            .def_readwrite(
                    "solver_iter_max",
                    &colmap::CasparBundleAdjustmentOptions::solver_iter_max)
            .def_readwrite("pcg_iter_max",
                           &colmap::CasparBundleAdjustmentOptions::pcg_iter_max)
            .def_readwrite("diag_init",
                           &colmap::CasparBundleAdjustmentOptions::diag_init)
            .def_readwrite("diag_min",
                           &colmap::CasparBundleAdjustmentOptions::diag_min)
            .def_readwrite(
                    "diag_scaling_up",
                    &colmap::CasparBundleAdjustmentOptions::diag_scaling_up)
            .def_readwrite(
                    "diag_scaling_down",
                    &colmap::CasparBundleAdjustmentOptions::diag_scaling_down)
            .def_readwrite(
                    "diag_exit_value",
                    &colmap::CasparBundleAdjustmentOptions::diag_exit_value)
            .def_readwrite(
                    "score_exit_value",
                    &colmap::CasparBundleAdjustmentOptions::score_exit_value)
            .def_readwrite(
                    "pcg_rel_error_exit",
                    &colmap::CasparBundleAdjustmentOptions::pcg_rel_error_exit)
            .def_readwrite(
                    "pcg_rel_score_exit",
                    &colmap::CasparBundleAdjustmentOptions::pcg_rel_score_exit)
            .def_readwrite("pcg_rel_decrease_min",
                           &colmap::CasparBundleAdjustmentOptions::
                                   pcg_rel_decrease_min)
            .def_readwrite("solver_rel_decrease_min",
                           &colmap::CasparBundleAdjustmentOptions::
                                   solver_rel_decrease_min)
            .def_readwrite("gpu_index",
                           &colmap::CasparBundleAdjustmentOptions::gpu_index)
            .def_readwrite("collect_iteration_data",
                           &colmap::CasparBundleAdjustmentOptions::
                                   collect_iteration_data)
            .def("check", &colmap::CasparBundleAdjustmentOptions::Check);

    // Upstream parity (estimators/covariance.cc): internal::PoseParam
    // exposed under the upstream ExperimentalPoseParam name.
    py::class_<colmap::internal::PoseParam,
               std::shared_ptr<colmap::internal::PoseParam>>(
            m_estimators, "ExperimentalPoseParam")
            .def(py::init<>());

    py::class_<colmap::CasparBundleAdjustmentSummary,
               colmap::BundleAdjustmentSummary>(m_estimators,
                                                "CasparBundleAdjustmentSummary")
            .def(py::init<>())
            .def_readonly(
                    "iteration_count",
                    &colmap::CasparBundleAdjustmentSummary::iteration_count)
            .def_readonly(
                    "initial_score",
                    &colmap::CasparBundleAdjustmentSummary::initial_score);

    m_estimators.attr("__docstring__") = map_shared_argument_docstrings;
}

}  // namespace estimators
}  // namespace reconstruction
}  // namespace cloudViewer
