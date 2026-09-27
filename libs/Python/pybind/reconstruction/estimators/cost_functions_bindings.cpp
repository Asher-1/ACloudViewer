// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "pybind/reconstruction/estimators/cost_functions_bindings.h"

#include <ceres/ceres.h>

#include <string>

#include "estimators/cost_functions/alignment.h"
#include "estimators/cost_functions/pose_prior.h"
#include "estimators/cost_functions/reprojection_error.h"
#include "estimators/cost_functions/sampson_error.h"
#include "estimators/cost_functions/utils.h"
#include "geometry/rigid3.h"

namespace cloudViewer {
namespace reconstruction {
namespace estimators {
namespace {

using namespace colmap;  // NOLINT(google-build-using-namespace)
using namespace pybind11::literals;  // NOLINT(google-build-using-namespace)

// Create is overloaded on single and per-residual stddevs, so taking its
// address needs a target signature to disambiguate.
template <class CostFunctor, typename... Args>
ceres::CostFunction* (*ScaleCost())(double, Args&&...) {
    return &ScaleWeightedCostFunctor<CostFunctor>::template Create<Args...>;
}

template <class CostFunctor, typename... Args>
ceres::CostFunction* (*ScaleVecCost())(
        const typename ScaleWeightedCostFunctor<CostFunctor>::StddevVec&,
        Args&&...) {
    return &ScaleWeightedCostFunctor<CostFunctor>::template Create<Args...>;
}

}  // namespace

void pybind_cost_functions(py::module& m) {
    py::module m_cost_functions = m.def_submodule(
            "cost_functions",
            "Ceres cost-function factories (upstream pycolmap parity). The "
            "returned objects implement the ceres::CostFunction interface "
            "exposed through reconstruction.pyceres.CostFunction.");
    m_cost_functions
            .def("ReprojErrorCost",
                 &CreateCameraCostFunction<ReprojErrorCostFunctor,
                                           const Eigen::Vector2d&>,
                 "camera_model_id"_a, "point2D"_a, "Reprojection error.")
            .def("ReprojErrorCost",
                 &CreateCovarianceWeightedCameraCostFunction<
                         ReprojErrorCostFunctor, Eigen::Matrix2d,
                         const Eigen::Vector2d&>,
                 "camera_model_id"_a, "point2D_cov"_a, "point2D"_a,
                 "Reprojection error with 2D detection noise.")
            .def("ReprojErrorCost",
                 &CreateScaleWeightedCameraCostFunction<
                         ReprojErrorCostFunctor, double,
                         const Eigen::Vector2d&>,
                 "camera_model_id"_a, "point2D_stddev"_a, "point2D"_a,
                 "Reprojection error with isotropic 2D detection noise.")
            .def("ReprojErrorCost",
                 &CreateScaleWeightedCameraCostFunction<
                         ReprojErrorCostFunctor, Eigen::Vector2d,
                         const Eigen::Vector2d&>,
                 "camera_model_id"_a, "point2D_stddev"_a, "point2D"_a,
                 "Reprojection error with per-axis 2D detection noise.")

            .def("ReprojErrorCost",
                 &CreateCameraCostFunction<ReprojErrorConstantPoseCostFunctor,
                                           const Eigen::Vector2d&,
                                           const Rigid3d&>,
                 "camera_model_id"_a, "point2D"_a, "cam_from_world"_a,
                 "Reprojection error with constant camera pose.")
            .def("ReprojErrorCost",
                 &CreateCovarianceWeightedCameraCostFunction<
                         ReprojErrorConstantPoseCostFunctor, Eigen::Matrix2d,
                         const Eigen::Vector2d&, const Rigid3d&>,
                 "camera_model_id"_a, "point2D_cov"_a, "point2D"_a,
                 "cam_from_world"_a,
                 "Reprojection error with constant camera pose and 2D "
                 "detection noise.")
            .def("ReprojErrorCost",
                 &CreateScaleWeightedCameraCostFunction<
                         ReprojErrorConstantPoseCostFunctor, double,
                         const Eigen::Vector2d&, const Rigid3d&>,
                 "camera_model_id"_a, "point2D_stddev"_a, "point2D"_a,
                 "cam_from_world"_a,
                 "Reprojection error with constant camera pose and "
                 "isotropic 2D detection noise.")
            .def("ReprojErrorCost",
                 &CreateScaleWeightedCameraCostFunction<
                         ReprojErrorConstantPoseCostFunctor,
                         Eigen::Vector2d, const Eigen::Vector2d&,
                         const Rigid3d&>,
                 "camera_model_id"_a, "point2D_stddev"_a, "point2D"_a,
                 "cam_from_world"_a,
                 "Reprojection error with constant camera pose and per-axis "
                 "2D detection noise.")

            .def("ReprojErrorCost",
                 &CreateCameraCostFunction<
                         ReprojErrorConstantPoint3DCostFunctor,
                         const Eigen::Vector2d&, const Eigen::Vector3d&>,
                 "camera_model_id"_a, "point2D"_a, "point3D"_a,
                 "Reprojection error with constant 3D point.")
            .def("ReprojErrorCost",
                 &CreateCovarianceWeightedCameraCostFunction<
                         ReprojErrorConstantPoint3DCostFunctor,
                         Eigen::Matrix2d, const Eigen::Vector2d&,
                         const Eigen::Vector3d&>,
                 "camera_model_id"_a, "point2D_cov"_a, "point2D"_a,
                 "point3D"_a,
                 "Reprojection error with constant 3D point and 2D detection "
                 "noise.")
            .def("ReprojErrorCost",
                 &CreateScaleWeightedCameraCostFunction<
                         ReprojErrorConstantPoint3DCostFunctor, double,
                         const Eigen::Vector2d&, const Eigen::Vector3d&>,
                 "camera_model_id"_a, "point2D_stddev"_a, "point2D"_a,
                 "point3D"_a,
                 "Reprojection error with constant 3D point and isotropic 2D "
                 "detection noise.")
            .def("ReprojErrorCost",
                 &CreateScaleWeightedCameraCostFunction<
                         ReprojErrorConstantPoint3DCostFunctor,
                         Eigen::Vector2d, const Eigen::Vector2d&,
                         const Eigen::Vector3d&>,
                 "camera_model_id"_a, "point2D_stddev"_a, "point2D"_a,
                 "point3D"_a,
                 "Reprojection error with constant 3D point and per-axis 2D "
                 "detection noise.")

            .def("RigReprojErrorCost",
                 &CreateCameraCostFunction<RigReprojErrorCostFunctor,
                                           const Eigen::Vector2d&>,
                 "camera_model_id"_a, "point2D"_a,
                 "Reprojection error for camera rig.")
            .def("RigReprojErrorCost",
                 &CreateCovarianceWeightedCameraCostFunction<
                         RigReprojErrorCostFunctor, Eigen::Matrix2d,
                         const Eigen::Vector2d&>,
                 "camera_model_id"_a, "point2D_cov"_a, "point2D"_a,
                 "Reprojection error for camera rig with 2D detection "
                 "noise.")
            .def("RigReprojErrorCost",
                 &CreateScaleWeightedCameraCostFunction<
                         RigReprojErrorCostFunctor, double,
                         const Eigen::Vector2d&>,
                 "camera_model_id"_a, "point2D_stddev"_a, "point2D"_a,
                 "Reprojection error for camera rig with isotropic 2D "
                 "detection noise.")
            .def("RigReprojErrorCost",
                 &CreateScaleWeightedCameraCostFunction<
                         RigReprojErrorCostFunctor, Eigen::Vector2d,
                         const Eigen::Vector2d&>,
                 "camera_model_id"_a, "point2D_stddev"_a, "point2D"_a,
                 "Reprojection error for camera rig with per-axis 2D "
                 "detection noise.")

            .def("RigReprojErrorCost",
                 &CreateCameraCostFunction<RigReprojErrorConstantRigCostFunctor,
                                           const Eigen::Vector2d&,
                                           const Rigid3d&>,
                 "camera_model_id"_a, "point2D"_a, "cam_from_rig"_a,
                 "Reprojection error for camera rig with constant "
                 "cam-from-rig pose.")
            .def("RigReprojErrorCost",
                 &CreateCovarianceWeightedCameraCostFunction<
                         RigReprojErrorConstantRigCostFunctor,
                         Eigen::Matrix2d, const Eigen::Vector2d&,
                         const Rigid3d&>,
                 "camera_model_id"_a, "point2D_cov"_a, "point2D"_a,
                 "cam_from_rig"_a,
                 "Reprojection error for camera rig with constant "
                 "cam-from-rig pose and 2D detection noise.")
            .def("RigReprojErrorCost",
                 &CreateScaleWeightedCameraCostFunction<
                         RigReprojErrorConstantRigCostFunctor, double,
                         const Eigen::Vector2d&, const Rigid3d&>,
                 "camera_model_id"_a, "point2D_stddev"_a, "point2D"_a,
                 "cam_from_rig"_a,
                 "Reprojection error for camera rig with constant "
                 "cam-from-rig pose and isotropic 2D detection noise.")
            .def("RigReprojErrorCost",
                 &CreateScaleWeightedCameraCostFunction<
                         RigReprojErrorConstantRigCostFunctor,
                         Eigen::Vector2d, const Eigen::Vector2d&,
                         const Rigid3d&>,
                 "camera_model_id"_a, "point2D_stddev"_a, "point2D"_a,
                 "cam_from_rig"_a,
                 "Reprojection error for camera rig with constant "
                 "cam-from-rig pose and per-axis 2D detection noise.")

            .def("ScaledRigReprojErrorCost",
                 &CreateCameraCostFunction<ScaledRigReprojErrorCostFunctor,
                                           const Eigen::Vector2d&, bool>,
                 "camera_model_id"_a, "point2D"_a, "use_log_scale"_a = true,
                 "Reprojection error for camera rig with a scaled "
                 "rig-from-world transform.")
            .def("ScaledRigReprojErrorCost",
                 &CreateCovarianceWeightedCameraCostFunction<
                         ScaledRigReprojErrorCostFunctor, Eigen::Matrix2d,
                         const Eigen::Vector2d&, bool>,
                 "camera_model_id"_a, "point2D_cov"_a, "point2D"_a,
                 "use_log_scale"_a = true,
                 "Reprojection error for camera rig with a scaled "
                 "rig-from-world transform and 2D detection noise.")
            .def("ScaledRigReprojErrorCost",
                 &CreateScaleWeightedCameraCostFunction<
                         ScaledRigReprojErrorCostFunctor, double,
                         const Eigen::Vector2d&, bool>,
                 "camera_model_id"_a, "point2D_stddev"_a, "point2D"_a,
                 "use_log_scale"_a = true,
                 "Reprojection error for camera rig with a scaled "
                 "rig-from-world transform and isotropic 2D detection "
                 "noise.")
            .def("ScaledRigReprojErrorCost",
                 &CreateScaleWeightedCameraCostFunction<
                         ScaledRigReprojErrorCostFunctor, Eigen::Vector2d,
                         const Eigen::Vector2d&, bool>,
                 "camera_model_id"_a, "point2D_stddev"_a, "point2D"_a,
                 "use_log_scale"_a = true,
                 "Reprojection error for camera rig with a scaled "
                 "rig-from-world transform and per-axis 2D detection "
                 "noise.")

            .def("SampsonErrorCost",
                 &SampsonErrorCostFunctor::Create<const Eigen::Vector2d&,
                                                  const Eigen::Vector2d&>,
                 "point1"_a, "point2"_a,
                 "Sampson error for two-view geometry on image-plane "
                 "points.")

            .def("AbsolutePosePriorCost",
                 &AbsolutePosePriorCostFunctor::Create<const Rigid3d&>,
                 "cam_from_world_prior"_a,
                 "6-DoF error on the absolute camera pose.")
            .def("AbsolutePosePriorCost",
                 &CovarianceWeightedCostFunctor<
                         AbsolutePosePriorCostFunctor>::Create<const Rigid3d&>,
                 "cam_cov_from_world_prior"_a, "cam_from_world_prior"_a,
                 "6-DoF error on the absolute camera pose with prior "
                 "covariance.")
            .def("AbsolutePosePriorCost",
                 ScaleVecCost<AbsolutePosePriorCostFunctor,
                              const Rigid3d&>(),
                 "cam_stddev_from_world_prior"_a, "cam_from_world_prior"_a,
                 "6-DoF error on the absolute camera pose with per-DoF "
                 "prior standard deviations. The first three are on the "
                 "rotation and the last three on the translation, so they "
                 "do not share a unit.")

            .def("AbsolutePosePositionPriorCost",
                 &AbsolutePosePositionPriorCostFunctor::Create<
                         const Eigen::Vector3d&>,
                 "position_in_world_prior"_a,
                 "3-DoF error on the absolute camera pose's position.")
            .def("AbsolutePosePositionPriorCost",
                 &CovarianceWeightedCostFunctor<
                         AbsolutePosePositionPriorCostFunctor>::
                         Create<const Eigen::Vector3d&>,
                 "position_cov_in_world_prior"_a,
                 "position_in_world_prior"_a,
                 "3-DoF error on the absolute camera pose's position with "
                 "prior covariance.")
            .def("AbsolutePosePositionPriorCost",
                 ScaleCost<AbsolutePosePositionPriorCostFunctor,
                           const Eigen::Vector3d&>(),
                 "position_stddev_in_world_prior"_a,
                 "position_in_world_prior"_a,
                 "3-DoF error on the absolute camera pose's position with "
                 "isotropic prior standard deviation.")
            .def("AbsolutePosePositionPriorCost",
                 ScaleVecCost<AbsolutePosePositionPriorCostFunctor,
                              const Eigen::Vector3d&>(),
                 "position_stddev_in_world_prior"_a,
                 "position_in_world_prior"_a,
                 "3-DoF error on the absolute camera pose's position with "
                 "per-axis prior standard deviations.")

            .def("RelativePosePriorCost",
                 &RelativePosePriorCostFunctor::Create<const Rigid3d&>,
                 "i_from_j_prior"_a,
                 "6-DoF error between two absolute camera poses based on a "
                 "prior relative pose.")
            .def("RelativePosePriorCost",
                 &CovarianceWeightedCostFunctor<
                         RelativePosePriorCostFunctor>::Create<const Rigid3d&>,
                 "i_cov_from_j_prior"_a, "i_from_j_prior"_a,
                 "6-DoF error between two absolute camera poses based on a "
                 "prior relative pose with prior covariance.")
            .def("RelativePosePriorCost",
                 ScaleVecCost<RelativePosePriorCostFunctor,
                              const Rigid3d&>(),
                 "i_stddev_from_j_prior"_a, "i_from_j_prior"_a,
                 "6-DoF error between two absolute camera poses based on a "
                 "prior relative pose with per-DoF prior standard "
                 "deviations. The first three are on the rotation and the "
                 "last three on the translation, so they do not share a "
                 "unit.")

            .def("Point3DAlignmentCost",
                 &Point3DAlignmentCostFunctor::Create<const Eigen::Vector3d&,
                                                      bool>,
                 "point_in_b_prior"_a, "use_log_scale"_a = true,
                 "Error between 3D points transformed by a 3D similarity "
                 "transform.")
            .def("Point3DAlignmentCost",
                 &CovarianceWeightedCostFunctor<
                         Point3DAlignmentCostFunctor>::
                         Create<const Eigen::Vector3d&, bool>,
                 "point_cov_in_b_prior"_a, "point_in_b_prior"_a,
                 "use_log_scale"_a = true,
                 "Error between 3D points transformed by a 3D similarity "
                 "transform with prior covariance.")
            .def("Point3DAlignmentCost",
                 ScaleCost<Point3DAlignmentCostFunctor,
                           const Eigen::Vector3d&, bool>(),
                 "point_stddev_in_b_prior"_a, "point_in_b_prior"_a,
                 "use_log_scale"_a = true,
                 "Error between 3D points transformed by a 3D similarity "
                 "transform, with an isotropic prior standard deviation.")
            .def("Point3DAlignmentCost",
                 ScaleVecCost<Point3DAlignmentCostFunctor,
                              const Eigen::Vector3d&, bool>(),
                 "point_stddev_in_b_prior"_a, "point_in_b_prior"_a,
                 "use_log_scale"_a = true,
                 "Error between 3D points transformed by a 3D similarity "
                 "transform, with per-axis prior standard deviations.");
}

}  // namespace estimators
}  // namespace reconstruction
}  // namespace cloudViewer
