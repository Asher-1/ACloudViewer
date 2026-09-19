// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "pybind/reconstruction/estimators/covariance_bindings.h"

#include <sstream>
#include <unordered_map>

#include "estimators/bundle_adjustment.h"
#include "estimators/bundle_adjustment_ceres.h"
#include "estimators/covariance.h"
#include "pybind/docstring.h"
#include "scene/reconstruction.h"

namespace cloudViewer {
namespace reconstruction {
namespace estimators {

// Upstream pycolmap parity (src/pycolmap/estimators/covariance.cc +
// bundle_adjustment.cc): Open3D-style bindings over the fork engine types.
// The fork's CeresBundleAdjuster holds a raw Reconstruction* after Solve, so
// the binding uses the owner-wrapper pattern (py::object reference) as in the
// sfm module.
using colmap::BACovariance;
using colmap::BACovarianceOptions;
using colmap::BundleAdjustmentConfig;
using colmap::BundleAdjustmentOptions;
using colmap::CeresBundleAdjuster;
using colmap::Reconstruction;

// Owner wrapper for the Ceres bundle adjuster.
struct PyCeresBundleAdjuster {
    py::object reconstruction_ref;
    std::unique_ptr<CeresBundleAdjuster> adjuster;
};

// Opaque handle for the engine's ceres::Problem (the fork does not bind the
// ceres API surface; the handle exists so a problem can be round-tripped
// from the adjuster into estimate_ba_covariance_from_problem).
struct PyCeresProblem {
    ceres::Problem* problem = nullptr;
};

static const std::unordered_map<std::string, std::string>
        map_shared_argument_docstrings = {
                {"options", "The covariance estimation options."},
                {"reconstruction", "The reconstruction to operate on."},
};

void pybind_covariance(py::module& m) {
    py::class_<PyCeresProblem> ceres_problem(m, "CeresProblem",
                                             "Opaque engine ceres::Problem "
                                             "handle.");
    py::class_<PyCeresBundleAdjuster> ceres_adjuster(
            m, "CeresBundleAdjuster",
            "The Ceres bundle adjustment backend (create, solve, then "
            "estimate covariances on the solved problem).");
    ceres_adjuster
            .def(py::init([](const BundleAdjustmentOptions& options,
                             const BundleAdjustmentConfig& config) {
                     auto holder = new PyCeresBundleAdjuster();
                     holder->adjuster.reset(
                             new CeresBundleAdjuster(options, config));
                     return holder;
                 }),
                 "options"_a, "config"_a)
            .def(
                    "solve",
                    [](PyCeresBundleAdjuster& self,
                       py::object reconstruction_obj) {
                        self.reconstruction_ref = reconstruction_obj;
                        return self.adjuster->Solve(
                                reconstruction_obj.cast<Reconstruction*>());
                    },
                    "reconstruction"_a,
                    "Run bundle adjustment; the reconstruction must outlive "
                    "the adjuster.")
            .def_property_readonly(
                    "problem",
                    [](PyCeresBundleAdjuster& self) {
                        auto handle = new PyCeresProblem();
                        handle->problem = &self.adjuster->Problem();
                        return handle;
                    },
                    "The engine ceres::Problem handle of the last SetUp.",
                    py::return_value_policy::reference_internal);

    py::enum_<BACovarianceOptions::Params> cov_params(
            m, "BACovarianceParams",
            "Which parameters to compute the covariance for.");
    cov_params.value("POSES", BACovarianceOptions::Params::POSES)
            .value("POINTS", BACovarianceOptions::Params::POINTS)
            .value("POSES_AND_POINTS",
                   BACovarianceOptions::Params::POSES_AND_POINTS)
            .value("ALL", BACovarianceOptions::Params::ALL)
            .export_values();

    py::class_<BACovarianceOptions> cov_options(
            m, "BACovarianceOptions",
            "Options for the bundle-adjustment covariance estimation.");
    cov_options.def(py::init<>())
            .def_readwrite("params", &BACovarianceOptions::params)
            .def_readwrite("damping", &BACovarianceOptions::damping,
                           "Damping factor for the Hessian in the Schur "
                           "complement solver.");

    py::class_<BACovariance> covariance(m, "BACovariance",
                                        "Bundle-adjustment covariance "
                                        "estimates (tangent space, order "
                                        "[rotation, translation] for "
                                        "poses).");
    covariance
            .def(
                    "get_point_cov",
                    [](const BACovariance& self,
                       colmap::point3D_t point3D_id) -> py::object {
                        const auto cov = self.GetPointCov(point3D_id);
                        if (!cov) {
                            return py::none();
                        }
                        return py::cast(*cov);
                    },
                    "point3D_id"_a,
                    "Covariance for a 3D point, conditioned on all other "
                    "variables "
                    "set constant; None if not a variable in the problem.")
            .def(
                    "get_cam_cov_from_world",
                    [](const BACovariance& self,
                       colmap::image_t image_id) -> py::object {
                        const auto cov = self.GetCamCovFromWorld(image_id);
                        if (!cov) {
                            return py::none();
                        }
                        return py::cast(*cov);
                    },
                    "image_id"_a,
                    "Tangent-space camera covariance; None if not a variable "
                    "in the problem.")
            .def(
                    "get_cam_cross_cov_from_world",
                    [](const BACovariance& self, colmap::image_t image_id1,
                       colmap::image_t image_id2) -> py::object {
                        const auto cov = self.GetCamCrossCovFromWorld(
                                image_id1, image_id2);
                        if (!cov) {
                            return py::none();
                        }
                        return py::cast(*cov);
                    },
                    "image_id1"_a, "image_id2"_a,
                    "Cross covariance between two cameras; None if not "
                    "variables in the problem.")
            .def(
                    "get_cam2_cov_from_cam1",
                    [](const BACovariance& self, colmap::image_t image_id1,
                       const colmap::Rigid3d& cam1_from_world,
                       colmap::image_t image_id2,
                       const colmap::Rigid3d& cam2_from_world) -> py::object {
                        const auto cov = self.GetCam2CovFromCam1(
                                image_id1, cam1_from_world, image_id2,
                                cam2_from_world);
                        if (!cov) {
                            return py::none();
                        }
                        return py::cast(*cov);
                    },
                    "image_id1"_a, "cam1_from_world"_a, "image_id2"_a,
                    "cam2_from_world"_a,
                    "Relative pose covariance; None if some dimensions are "
                    "kept constant.");

    m.def(
            "estimate_ba_covariance",
            [](const BACovarianceOptions& options,
               const Reconstruction& reconstruction,
               PyCeresBundleAdjuster& adjuster) -> py::object {
                auto cov = colmap::EstimateBACovariance(options, reconstruction,
                                                        *adjuster.adjuster);
                if (!cov) {
                    return py::none();
                }
                return py::cast(std::move(*cov));
            },
            "options"_a, "reconstruction"_a, "bundle_adjuster"_a,
            "Compute covariances for the parameters of the solved bundle "
            "adjustment problem (Schur complement); returns BACovariance or "
            "None.");

    m.def(
            "estimate_ba_covariance_from_problem",
            [](const BACovarianceOptions& options,
               const Reconstruction& reconstruction,
               PyCeresProblem& problem) -> py::object {
                THROW_CHECK_NOTNULL(problem.problem);
                auto cov = colmap::EstimateBACovarianceFromProblem(
                        options, reconstruction, *problem.problem);
                if (!cov) {
                    return py::none();
                }
                return py::cast(std::move(*cov));
            },
            "options"_a, "reconstruction"_a, "problem"_a,
            "Compute covariances directly from a ceres problem handle.");

    m.attr("__docstring__") = map_shared_argument_docstrings;
}

}  // namespace estimators
}  // namespace reconstruction
}  // namespace cloudViewer
