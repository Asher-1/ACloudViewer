// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// SPDX-License-Identifier: MIT

#include "optim/bundle_adjustment_caspar.h"

#include <algorithm>
#include <array>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "base/camera_models.h"
#include "util/cuda.h"
#include "util/logging.h"

#ifdef CASPAR_USE_DOUBLE
#include "solver.h"
using CasparStorage = double;
#else
#include "solver.h"
using CasparStorage = float;
#endif

namespace colmap {
namespace {

struct CasparModelData {
    struct FactorData {
        std::vector<unsigned int> pose_indices;
        std::vector<unsigned int> calibration_indices;
        std::vector<unsigned int> focal_and_extra_indices;
        std::vector<unsigned int> principal_point_indices;
        std::vector<unsigned int> point_indices;
        std::vector<CasparStorage> sensor_from_rig;
        std::vector<CasparStorage> pixels;
        std::vector<CasparStorage> constant_poses;
        std::vector<CasparStorage> constant_focal_and_extra;
        std::vector<CasparStorage> constant_principal_point;
        std::vector<CasparStorage> constant_points;
    };

    // This is the upstream Caspar factor-variant order. A factor's pool is
    // selected from the four independent variable groups: pose,
    // focal-and-extra, principal point, and 3D point.
    enum FactorPool : size_t {
        kBase = 0,
        kFixedPose = 1,
        kFixedFocalAndExtra = 2,
        kFixedPrincipalPoint = 3,
        kFixedPoint = 4,
        kFixedPoseFixedFocalAndExtra = 5,
        kFixedPoseFixedPrincipalPoint = 6,
        kFixedPoseFixedPoint = 7,
        kFixedFocalAndExtraFixedPrincipalPoint = 8,
        kFixedFocalAndExtraFixedPoint = 9,
        kFixedPrincipalPointFixedPoint = 10,
        kFixedPoseFixedFocalAndExtraFixedPrincipalPoint = 11,
        kFixedPoseFixedFocalAndExtraFixedPoint = 12,
        kFixedPoseFixedPrincipalPointFixedPoint = 13,
        kFixedFocalAndExtraFixedPrincipalPointFixedPoint = 14,
        kNumFactorPools = 15,
    };

    std::vector<camera_t> camera_ids;
    std::vector<image_t> image_ids;
    std::unordered_map<camera_t, unsigned int> camera_indices;
    std::unordered_map<image_t, unsigned int> pose_indices;
    // A non-rig image remains keyed by its image id. Rig observations instead
    // share the Frame pose, exactly as the Caspar factor's sensor_from_rig
    // input expects.
    std::unordered_map<frame_t, unsigned int> frame_pose_indices;
    std::vector<frame_t> pose_frame_ids;
    std::vector<image_t> pose_image_ids;
    // Bit 0 = focal-and-extra, bit 1 = principal point. This records the
    // actual variable groups for each camera so write-back never changes a
    // group that was represented as a Caspar factor constant.
    std::unordered_map<camera_t, unsigned char> intrinsic_variable_masks;
    std::vector<CasparStorage> calibration;
    std::vector<CasparStorage> focal_and_extra;
    std::vector<CasparStorage> principal_point;
    std::vector<CasparStorage> poses;
    std::array<FactorData, kNumFactorPools> factors;
};

CasparModelData::FactorPool FactorPoolFromVariableMask(
        const bool pose_variable,
        const bool focal_and_extra_variable,
        const bool principal_point_variable,
        const bool point_variable) {
    // Index bits are pose, focal-and-extra, principal-point, point. The table
    // is the exact upstream FactorVariant lookup; the all-fixed state is not
    // useful because a BA observation must retain at least one unknown.
    static constexpr CasparModelData::FactorPool kFactorPoolTable[16] = {
            CasparModelData::kFixedPoseFixedFocalAndExtraFixedPrincipalPoint,
            CasparModelData::kFixedPoseFixedFocalAndExtraFixedPrincipalPoint,
            CasparModelData::kFixedPoseFixedFocalAndExtraFixedPoint,
            CasparModelData::kFixedPoseFixedFocalAndExtra,
            CasparModelData::kFixedPoseFixedPrincipalPointFixedPoint,
            CasparModelData::kFixedPoseFixedPrincipalPoint,
            CasparModelData::kFixedPoseFixedPoint,
            CasparModelData::kFixedPose,
            CasparModelData::kFixedFocalAndExtraFixedPrincipalPointFixedPoint,
            CasparModelData::kFixedFocalAndExtraFixedPrincipalPoint,
            CasparModelData::kFixedFocalAndExtraFixedPoint,
            CasparModelData::kFixedFocalAndExtra,
            CasparModelData::kFixedPrincipalPointFixedPoint,
            CasparModelData::kFixedPrincipalPoint,
            CasparModelData::kFixedPoint,
            CasparModelData::kBase,
    };
    const size_t index = (static_cast<size_t>(pose_variable) << 3) |
                         (static_cast<size_t>(focal_and_extra_variable) << 2) |
                         (static_cast<size_t>(principal_point_variable) << 1) |
                         static_cast<size_t>(point_variable);
    return kFactorPoolTable[index];
}

bool IsCasparModel(const int model_id) {
    return model_id == SimpleRadialCameraModel::model_id ||
           model_id == PinholeCameraModel::model_id;
}

void AppendPose(const Eigen::Vector4d& qvec,
                const Eigen::Vector3d& tvec,
                std::vector<CasparStorage>* data) {
    // COLMAP qvec is [w, x, y, z]; Caspar stores [x, y, z, w, tx, ty, tz].
    const Eigen::Vector4d normalized_qvec = NormalizeQuaternion(qvec);
    data->push_back(static_cast<CasparStorage>(normalized_qvec(1)));
    data->push_back(static_cast<CasparStorage>(normalized_qvec(2)));
    data->push_back(static_cast<CasparStorage>(normalized_qvec(3)));
    data->push_back(static_cast<CasparStorage>(normalized_qvec(0)));
    data->push_back(static_cast<CasparStorage>(tvec(0)));
    data->push_back(static_cast<CasparStorage>(tvec(1)));
    data->push_back(static_cast<CasparStorage>(tvec(2)));
}

void AppendCalibration(const Camera& camera,
                       std::vector<CasparStorage>* calibration) {
    if (camera.ModelId() == SimpleRadialCameraModel::model_id) {
        // Caspar SimpleRadialCalib is [f, k, cx, cy].
        calibration->push_back(static_cast<CasparStorage>(camera.Params(0)));
        calibration->push_back(static_cast<CasparStorage>(camera.Params(3)));
        calibration->push_back(static_cast<CasparStorage>(camera.Params(1)));
        calibration->push_back(static_cast<CasparStorage>(camera.Params(2)));
    } else {
        // Caspar PinholeCalib is [fx, fy, cx, cy].
        for (size_t i = 0; i < 4; ++i) {
            calibration->push_back(
                    static_cast<CasparStorage>(camera.Params(i)));
        }
    }
}

void WriteCalibration(Camera* camera, const CasparStorage* calibration) {
    if (camera->ModelId() == SimpleRadialCameraModel::model_id) {
        camera->Params(0) = static_cast<double>(calibration[0]);
        camera->Params(3) = static_cast<double>(calibration[1]);
        camera->Params(1) = static_cast<double>(calibration[2]);
        camera->Params(2) = static_cast<double>(calibration[3]);
    } else {
        for (size_t i = 0; i < 4; ++i) {
            camera->Params(i) = static_cast<double>(calibration[i]);
        }
    }
}

void AppendSplitIntrinsics(const Camera& camera,
                           std::vector<CasparStorage>* focal_and_extra,
                           std::vector<CasparStorage>* principal_point) {
    if (camera.ModelId() == SimpleRadialCameraModel::model_id) {
        focal_and_extra->insert(focal_and_extra->end(),
                                {static_cast<CasparStorage>(camera.Params(0)),
                                 static_cast<CasparStorage>(camera.Params(3))});
        principal_point->insert(principal_point->end(),
                                {static_cast<CasparStorage>(camera.Params(1)),
                                 static_cast<CasparStorage>(camera.Params(2))});
    } else {
        focal_and_extra->insert(focal_and_extra->end(),
                                {static_cast<CasparStorage>(camera.Params(0)),
                                 static_cast<CasparStorage>(camera.Params(1))});
        principal_point->insert(principal_point->end(),
                                {static_cast<CasparStorage>(camera.Params(2)),
                                 static_cast<CasparStorage>(camera.Params(3))});
    }
}

void WriteSplitIntrinsics(Camera* camera,
                          const CasparStorage* focal_and_extra,
                          const CasparStorage* principal_point,
                          const unsigned char variable_mask) {
    if (variable_mask & 1) {
        camera->Params(0) = static_cast<double>(focal_and_extra[0]);
        if (camera->ModelId() == SimpleRadialCameraModel::model_id) {
            camera->Params(3) = static_cast<double>(focal_and_extra[1]);
        } else {
            camera->Params(1) = static_cast<double>(focal_and_extra[1]);
        }
    }
    if (variable_mask & 2) {
        if (camera->ModelId() == SimpleRadialCameraModel::model_id) {
            camera->Params(1) = static_cast<double>(principal_point[0]);
            camera->Params(2) = static_cast<double>(principal_point[1]);
        } else {
            camera->Params(2) = static_cast<double>(principal_point[0]);
            camera->Params(3) = static_cast<double>(principal_point[1]);
        }
    }
}

void WritePose(Image* image, const CasparStorage* pose) {
    image->Qvec(0) = static_cast<double>(pose[3]);
    image->Qvec(1) = static_cast<double>(pose[0]);
    image->Qvec(2) = static_cast<double>(pose[1]);
    image->Qvec(3) = static_cast<double>(pose[2]);
    image->Tvec(0) = static_cast<double>(pose[4]);
    image->Tvec(1) = static_cast<double>(pose[5]);
    image->Tvec(2) = static_cast<double>(pose[6]);
    image->NormalizeQvec();
}

void AppendIdentitySensorFromRig(std::vector<CasparStorage>* data) {
    data->insert(data->end(), {CasparStorage(0), CasparStorage(0),
                               CasparStorage(0), CasparStorage(1),
                               CasparStorage(0), CasparStorage(0),
                               CasparStorage(0)});
}

void AppendSensorFromRig(const Reconstruction& reconstruction,
                         const Image& image,
                         const Frame* frame,
                         std::vector<CasparStorage>* data) {
    if (frame != nullptr && reconstruction.ExistsRig(frame->RigId())) {
        const Rig& rig = reconstruction.Rig(frame->RigId());
        if (rig.HasCamera(image.CameraId())) {
            AppendPose(rig.CamFromRigQvec(image.CameraId()),
                       rig.CamFromRigTvec(image.CameraId()), data);
            return;
        }
    }
    AppendIdentitySensorFromRig(data);
}

void SynchronizeFrameImages(Reconstruction* reconstruction,
                            const frame_t frame_id) {
    Frame& frame = reconstruction->Frame(frame_id);
    CHECK(frame.HasPose());
    const Rig& rig = reconstruction->Rig(frame.RigId());
    for (const image_t image_id : frame.ImageIds()) {
        Image& image = reconstruction->Image(image_id);
        CHECK(rig.HasCamera(image.CameraId()));
        Eigen::Vector4d cam_from_world_qvec;
        Eigen::Vector3d cam_from_world_tvec;
        ConcatenatePoses(rig.CamFromRigQvec(image.CameraId()),
                         rig.CamFromRigTvec(image.CameraId()),
                         frame.RigFromWorldQvec(),
                         frame.RigFromWorldTvec(),
                         &cam_from_world_qvec,
                         &cam_from_world_tvec);
        image.Qvec() = cam_from_world_qvec;
        image.Tvec() = cam_from_world_tvec;
    }
}

caspar::GraphSolver CreateSolver(const caspar::SolverParams<double>& params,
                                 const CasparModelData& simple_radial,
                                 const CasparModelData& pinhole,
                                 const size_t num_points,
                                 const int device_id) {
    const auto count = [](const CasparModelData& data,
                          const CasparModelData::FactorPool pool) {
        return data.factors[pool].pixels.size() / 2;
    };
    // Match the generated constructor's fixed ordering: four merged factors
    // for each model, then the eleven split-intrinsic factors for each model.
    return caspar::GraphSolver(
            params,
            pinhole.camera_ids.size(), pinhole.camera_ids.size(),
            pinhole.pose_frame_ids.size(), pinhole.camera_ids.size(), num_points,
            simple_radial.camera_ids.size(), simple_radial.camera_ids.size(),
            simple_radial.pose_frame_ids.size(), simple_radial.camera_ids.size(),
            count(simple_radial, CasparModelData::kBase),
            count(simple_radial, CasparModelData::kFixedPose),
            count(simple_radial, CasparModelData::kFixedPoint),
            count(simple_radial, CasparModelData::kFixedPoseFixedPoint),
            count(pinhole, CasparModelData::kBase),
            count(pinhole, CasparModelData::kFixedPose),
            count(pinhole, CasparModelData::kFixedPoint),
            count(pinhole, CasparModelData::kFixedPoseFixedPoint),
            count(simple_radial, CasparModelData::kFixedFocalAndExtra),
            count(simple_radial, CasparModelData::kFixedPrincipalPoint),
            count(simple_radial, CasparModelData::kFixedPoseFixedFocalAndExtra),
            count(simple_radial, CasparModelData::kFixedPoseFixedPrincipalPoint),
            count(simple_radial,
                  CasparModelData::kFixedFocalAndExtraFixedPrincipalPoint),
            count(simple_radial, CasparModelData::kFixedFocalAndExtraFixedPoint),
            count(simple_radial, CasparModelData::kFixedPrincipalPointFixedPoint),
            count(simple_radial,
                  CasparModelData::kFixedPoseFixedFocalAndExtraFixedPrincipalPoint),
            count(simple_radial,
                  CasparModelData::kFixedPoseFixedFocalAndExtraFixedPoint),
            count(simple_radial,
                  CasparModelData::kFixedPoseFixedPrincipalPointFixedPoint),
            count(simple_radial,
                  CasparModelData::kFixedFocalAndExtraFixedPrincipalPointFixedPoint),
            count(pinhole, CasparModelData::kFixedFocalAndExtra),
            count(pinhole, CasparModelData::kFixedPrincipalPoint),
            count(pinhole, CasparModelData::kFixedPoseFixedFocalAndExtra),
            count(pinhole, CasparModelData::kFixedPoseFixedPrincipalPoint),
            count(pinhole,
                  CasparModelData::kFixedFocalAndExtraFixedPrincipalPoint),
            count(pinhole, CasparModelData::kFixedFocalAndExtraFixedPoint),
            count(pinhole, CasparModelData::kFixedPrincipalPointFixedPoint),
            count(pinhole,
                  CasparModelData::kFixedPoseFixedFocalAndExtraFixedPrincipalPoint),
            count(pinhole,
                  CasparModelData::kFixedPoseFixedFocalAndExtraFixedPoint),
            count(pinhole,
                  CasparModelData::kFixedPoseFixedPrincipalPointFixedPoint),
            count(pinhole,
                  CasparModelData::kFixedFocalAndExtraFixedPrincipalPointFixedPoint),
            device_id);
}

bool IsRepresentable(const BundleAdjustmentOptions& options,
                     const BundleAdjustmentConfig& config,
                     Reconstruction* reconstruction) {
    // Caspar represents focal length and distortion together for
    // SimpleRadial. A partial update would have no equivalent generated
    // factor, so retain Ceres for that semantic instead of silently changing
    // the requested parameter set.
    if (!options.refine_extrinsics ||
        options.refine_focal_length != options.refine_extra_params ||
        !config.VariablePoints().empty()) {
        return false;
    }
    for (const image_t image_id : config.Images()) {
        const Image& image = reconstruction->Image(image_id);
        const Camera& camera = reconstruction->Camera(image.CameraId());
        if (config.HasConstantTvec(image_id) || !IsCasparModel(camera.ModelId())) {
            return false;
        }
    }
    return true;
}

#define CASPAR_SET_METHOD(model, suffix) \
    CASPAR_SET_METHOD_IMPL(model, suffix)
#define CASPAR_SET_METHOD_IMPL(model, suffix) solver.Set##model##suffix

// The Symforce generator names the Pinhole focal group "Focal" and the
// SimpleRadial group "FocalAndExtra". Apart from that spelling, their 15
// factor contracts are structurally identical, so keep one audited mapping.
#define DEFINE_CASPAR_FACTOR_SETTER(function_name, model, focal)                 \
    void function_name(caspar::GraphSolver& solver,                              \
                       const CasparModelData& data) {                            \
        for (size_t pool_index = 0;                                               \
             pool_index < CasparModelData::kNumFactorPools; ++pool_index) {      \
            const auto pool = static_cast<CasparModelData::FactorPool>(pool_index); \
            const CasparModelData::FactorData& d = data.factors[pool];           \
            const size_t n = d.pixels.size() / 2;                                \
            switch (pool) {                                                       \
                case CasparModelData::kBase:                                     \
                    CASPAR_SET_METHOD(model, Num)(n);                            \
                    if (n) {                                                     \
                        CASPAR_SET_METHOD(model, PoseIndicesFromHost)(d.pose_indices.data(), n); \
                        CASPAR_SET_METHOD(model, CalibIndicesFromHost)(d.calibration_indices.data(), n); \
                        CASPAR_SET_METHOD(model, PointIndicesFromHost)(d.point_indices.data(), n); \
                        CASPAR_SET_METHOD(model, SensorFromRigDataFromStackedHost)(d.sensor_from_rig.data(), 0, n); \
                        CASPAR_SET_METHOD(model, PixelDataFromStackedHost)(d.pixels.data(), 0, n); \
                    }                                                            \
                    break;                                                       \
                case CasparModelData::kFixedPose:                                \
                    CASPAR_SET_METHOD(model, FixedPoseNum)(n);                   \
                    if (n) {                                                     \
                        CASPAR_SET_METHOD(model, FixedPoseCalibIndicesFromHost)(d.calibration_indices.data(), n); \
                        CASPAR_SET_METHOD(model, FixedPosePointIndicesFromHost)(d.point_indices.data(), n); \
                        CASPAR_SET_METHOD(model, FixedPoseSensorFromRigDataFromStackedHost)(d.sensor_from_rig.data(), 0, n); \
                        CASPAR_SET_METHOD(model, FixedPosePoseDataFromStackedHost)(d.constant_poses.data(), 0, n); \
                        CASPAR_SET_METHOD(model, FixedPosePixelDataFromStackedHost)(d.pixels.data(), 0, n); \
                    }                                                            \
                    break;                                                       \
                case CasparModelData::kFixedPoint:                               \
                    CASPAR_SET_METHOD(model, FixedPointNum)(n);                  \
                    if (n) {                                                     \
                        CASPAR_SET_METHOD(model, FixedPointPoseIndicesFromHost)(d.pose_indices.data(), n); \
                        CASPAR_SET_METHOD(model, FixedPointCalibIndicesFromHost)(d.calibration_indices.data(), n); \
                        CASPAR_SET_METHOD(model, FixedPointSensorFromRigDataFromStackedHost)(d.sensor_from_rig.data(), 0, n); \
                        CASPAR_SET_METHOD(model, FixedPointPointDataFromStackedHost)(d.constant_points.data(), 0, n); \
                        CASPAR_SET_METHOD(model, FixedPointPixelDataFromStackedHost)(d.pixels.data(), 0, n); \
                    }                                                            \
                    break;                                                       \
                case CasparModelData::kFixedPoseFixedPoint:                      \
                    CASPAR_SET_METHOD(model, FixedPoseFixedPointNum)(n);         \
                    if (n) {                                                     \
                        CASPAR_SET_METHOD(model, FixedPoseFixedPointCalibIndicesFromHost)(d.calibration_indices.data(), n); \
                        CASPAR_SET_METHOD(model, FixedPoseFixedPointSensorFromRigDataFromStackedHost)(d.sensor_from_rig.data(), 0, n); \
                        CASPAR_SET_METHOD(model, FixedPoseFixedPointPoseDataFromStackedHost)(d.constant_poses.data(), 0, n); \
                        CASPAR_SET_METHOD(model, FixedPoseFixedPointPointDataFromStackedHost)(d.constant_points.data(), 0, n); \
                        CASPAR_SET_METHOD(model, FixedPoseFixedPointPixelDataFromStackedHost)(d.pixels.data(), 0, n); \
                    }                                                            \
                    break;                                                       \
                case CasparModelData::kFixedFocalAndExtra:                       \
                    CASPAR_SET_METHOD(model, SplitFixed##focal##Num)(n);         \
                    if (n) {                                                     \
                        CASPAR_SET_METHOD(model, SplitFixed##focal##PoseIndicesFromHost)(d.pose_indices.data(), n); \
                        CASPAR_SET_METHOD(model, SplitFixed##focal##PrincipalPointIndicesFromHost)(d.principal_point_indices.data(), n); \
                        CASPAR_SET_METHOD(model, SplitFixed##focal##PointIndicesFromHost)(d.point_indices.data(), n); \
                        CASPAR_SET_METHOD(model, SplitFixed##focal##SensorFromRigDataFromStackedHost)(d.sensor_from_rig.data(), 0, n); \
                        CASPAR_SET_METHOD(model, SplitFixed##focal##PixelDataFromStackedHost)(d.pixels.data(), 0, n); \
                        CASPAR_SET_METHOD(model, SplitFixed##focal##focal##DataFromStackedHost)(d.constant_focal_and_extra.data(), 0, n); \
                    }                                                            \
                    break;                                                       \
                case CasparModelData::kFixedPrincipalPoint:                      \
                    CASPAR_SET_METHOD(model, SplitFixedPrincipalPointNum)(n);    \
                    if (n) {                                                     \
                        CASPAR_SET_METHOD(model, SplitFixedPrincipalPointPoseIndicesFromHost)(d.pose_indices.data(), n); \
                        CASPAR_SET_METHOD(model, SplitFixedPrincipalPoint##focal##IndicesFromHost)(d.focal_and_extra_indices.data(), n); \
                        CASPAR_SET_METHOD(model, SplitFixedPrincipalPointPointIndicesFromHost)(d.point_indices.data(), n); \
                        CASPAR_SET_METHOD(model, SplitFixedPrincipalPointSensorFromRigDataFromStackedHost)(d.sensor_from_rig.data(), 0, n); \
                        CASPAR_SET_METHOD(model, SplitFixedPrincipalPointPixelDataFromStackedHost)(d.pixels.data(), 0, n); \
                        CASPAR_SET_METHOD(model, SplitFixedPrincipalPointPrincipalPointDataFromStackedHost)(d.constant_principal_point.data(), 0, n); \
                    }                                                            \
                    break;                                                       \
                case CasparModelData::kFixedPoseFixedFocalAndExtra:              \
                    CASPAR_SET_METHOD(model, SplitFixedPoseFixed##focal##Num)(n); \
                    if (n) {                                                     \
                        CASPAR_SET_METHOD(model, SplitFixedPoseFixed##focal##PrincipalPointIndicesFromHost)(d.principal_point_indices.data(), n); \
                        CASPAR_SET_METHOD(model, SplitFixedPoseFixed##focal##PointIndicesFromHost)(d.point_indices.data(), n); \
                        CASPAR_SET_METHOD(model, SplitFixedPoseFixed##focal##SensorFromRigDataFromStackedHost)(d.sensor_from_rig.data(), 0, n); \
                        CASPAR_SET_METHOD(model, SplitFixedPoseFixed##focal##PixelDataFromStackedHost)(d.pixels.data(), 0, n); \
                        CASPAR_SET_METHOD(model, SplitFixedPoseFixed##focal##PoseDataFromStackedHost)(d.constant_poses.data(), 0, n); \
                        CASPAR_SET_METHOD(model, SplitFixedPoseFixed##focal##focal##DataFromStackedHost)(d.constant_focal_and_extra.data(), 0, n); \
                    }                                                            \
                    break;                                                       \
                case CasparModelData::kFixedPoseFixedPrincipalPoint:             \
                    CASPAR_SET_METHOD(model, SplitFixedPoseFixedPrincipalPointNum)(n); \
                    if (n) {                                                     \
                        CASPAR_SET_METHOD(model, SplitFixedPoseFixedPrincipalPoint##focal##IndicesFromHost)(d.focal_and_extra_indices.data(), n); \
                        CASPAR_SET_METHOD(model, SplitFixedPoseFixedPrincipalPointPointIndicesFromHost)(d.point_indices.data(), n); \
                        CASPAR_SET_METHOD(model, SplitFixedPoseFixedPrincipalPointSensorFromRigDataFromStackedHost)(d.sensor_from_rig.data(), 0, n); \
                        CASPAR_SET_METHOD(model, SplitFixedPoseFixedPrincipalPointPixelDataFromStackedHost)(d.pixels.data(), 0, n); \
                        CASPAR_SET_METHOD(model, SplitFixedPoseFixedPrincipalPointPoseDataFromStackedHost)(d.constant_poses.data(), 0, n); \
                        CASPAR_SET_METHOD(model, SplitFixedPoseFixedPrincipalPointPrincipalPointDataFromStackedHost)(d.constant_principal_point.data(), 0, n); \
                    }                                                            \
                    break;                                                       \
                case CasparModelData::kFixedFocalAndExtraFixedPrincipalPoint:    \
                    CASPAR_SET_METHOD(model, SplitFixed##focal##FixedPrincipalPointNum)(n); \
                    if (n) {                                                     \
                        CASPAR_SET_METHOD(model, SplitFixed##focal##FixedPrincipalPointPoseIndicesFromHost)(d.pose_indices.data(), n); \
                        CASPAR_SET_METHOD(model, SplitFixed##focal##FixedPrincipalPointPointIndicesFromHost)(d.point_indices.data(), n); \
                        CASPAR_SET_METHOD(model, SplitFixed##focal##FixedPrincipalPointSensorFromRigDataFromStackedHost)(d.sensor_from_rig.data(), 0, n); \
                        CASPAR_SET_METHOD(model, SplitFixed##focal##FixedPrincipalPointPixelDataFromStackedHost)(d.pixels.data(), 0, n); \
                        CASPAR_SET_METHOD(model, SplitFixed##focal##FixedPrincipalPoint##focal##DataFromStackedHost)(d.constant_focal_and_extra.data(), 0, n); \
                        CASPAR_SET_METHOD(model, SplitFixed##focal##FixedPrincipalPointPrincipalPointDataFromStackedHost)(d.constant_principal_point.data(), 0, n); \
                    }                                                            \
                    break;                                                       \
                case CasparModelData::kFixedFocalAndExtraFixedPoint:             \
                    CASPAR_SET_METHOD(model, SplitFixed##focal##FixedPointNum)(n); \
                    if (n) {                                                     \
                        CASPAR_SET_METHOD(model, SplitFixed##focal##FixedPointPoseIndicesFromHost)(d.pose_indices.data(), n); \
                        CASPAR_SET_METHOD(model, SplitFixed##focal##FixedPointPrincipalPointIndicesFromHost)(d.principal_point_indices.data(), n); \
                        CASPAR_SET_METHOD(model, SplitFixed##focal##FixedPointSensorFromRigDataFromStackedHost)(d.sensor_from_rig.data(), 0, n); \
                        CASPAR_SET_METHOD(model, SplitFixed##focal##FixedPointPixelDataFromStackedHost)(d.pixels.data(), 0, n); \
                        CASPAR_SET_METHOD(model, SplitFixed##focal##FixedPoint##focal##DataFromStackedHost)(d.constant_focal_and_extra.data(), 0, n); \
                        CASPAR_SET_METHOD(model, SplitFixed##focal##FixedPointPointDataFromStackedHost)(d.constant_points.data(), 0, n); \
                    }                                                            \
                    break;                                                       \
                case CasparModelData::kFixedPrincipalPointFixedPoint:            \
                    CASPAR_SET_METHOD(model, SplitFixedPrincipalPointFixedPointNum)(n); \
                    if (n) {                                                     \
                        CASPAR_SET_METHOD(model, SplitFixedPrincipalPointFixedPointPoseIndicesFromHost)(d.pose_indices.data(), n); \
                        CASPAR_SET_METHOD(model, SplitFixedPrincipalPointFixedPoint##focal##IndicesFromHost)(d.focal_and_extra_indices.data(), n); \
                        CASPAR_SET_METHOD(model, SplitFixedPrincipalPointFixedPointSensorFromRigDataFromStackedHost)(d.sensor_from_rig.data(), 0, n); \
                        CASPAR_SET_METHOD(model, SplitFixedPrincipalPointFixedPointPixelDataFromStackedHost)(d.pixels.data(), 0, n); \
                        CASPAR_SET_METHOD(model, SplitFixedPrincipalPointFixedPointPrincipalPointDataFromStackedHost)(d.constant_principal_point.data(), 0, n); \
                        CASPAR_SET_METHOD(model, SplitFixedPrincipalPointFixedPointPointDataFromStackedHost)(d.constant_points.data(), 0, n); \
                    }                                                            \
                    break;                                                       \
                case CasparModelData::kFixedPoseFixedFocalAndExtraFixedPrincipalPoint: \
                    CASPAR_SET_METHOD(model, SplitFixedPoseFixed##focal##FixedPrincipalPointNum)(n); \
                    if (n) {                                                     \
                        CASPAR_SET_METHOD(model, SplitFixedPoseFixed##focal##FixedPrincipalPointPointIndicesFromHost)(d.point_indices.data(), n); \
                        CASPAR_SET_METHOD(model, SplitFixedPoseFixed##focal##FixedPrincipalPointSensorFromRigDataFromStackedHost)(d.sensor_from_rig.data(), 0, n); \
                        CASPAR_SET_METHOD(model, SplitFixedPoseFixed##focal##FixedPrincipalPointPixelDataFromStackedHost)(d.pixels.data(), 0, n); \
                        CASPAR_SET_METHOD(model, SplitFixedPoseFixed##focal##FixedPrincipalPointPoseDataFromStackedHost)(d.constant_poses.data(), 0, n); \
                        CASPAR_SET_METHOD(model, SplitFixedPoseFixed##focal##FixedPrincipalPoint##focal##DataFromStackedHost)(d.constant_focal_and_extra.data(), 0, n); \
                        CASPAR_SET_METHOD(model, SplitFixedPoseFixed##focal##FixedPrincipalPointPrincipalPointDataFromStackedHost)(d.constant_principal_point.data(), 0, n); \
                    }                                                            \
                    break;                                                       \
                case CasparModelData::kFixedPoseFixedFocalAndExtraFixedPoint:    \
                    CASPAR_SET_METHOD(model, SplitFixedPoseFixed##focal##FixedPointNum)(n); \
                    if (n) {                                                     \
                        CASPAR_SET_METHOD(model, SplitFixedPoseFixed##focal##FixedPointPrincipalPointIndicesFromHost)(d.principal_point_indices.data(), n); \
                        CASPAR_SET_METHOD(model, SplitFixedPoseFixed##focal##FixedPointSensorFromRigDataFromStackedHost)(d.sensor_from_rig.data(), 0, n); \
                        CASPAR_SET_METHOD(model, SplitFixedPoseFixed##focal##FixedPointPixelDataFromStackedHost)(d.pixels.data(), 0, n); \
                        CASPAR_SET_METHOD(model, SplitFixedPoseFixed##focal##FixedPointPoseDataFromStackedHost)(d.constant_poses.data(), 0, n); \
                        CASPAR_SET_METHOD(model, SplitFixedPoseFixed##focal##FixedPoint##focal##DataFromStackedHost)(d.constant_focal_and_extra.data(), 0, n); \
                        CASPAR_SET_METHOD(model, SplitFixedPoseFixed##focal##FixedPointPointDataFromStackedHost)(d.constant_points.data(), 0, n); \
                    }                                                            \
                    break;                                                       \
                case CasparModelData::kFixedPoseFixedPrincipalPointFixedPoint:   \
                    CASPAR_SET_METHOD(model, SplitFixedPoseFixedPrincipalPointFixedPointNum)(n); \
                    if (n) {                                                     \
                        CASPAR_SET_METHOD(model, SplitFixedPoseFixedPrincipalPointFixedPoint##focal##IndicesFromHost)(d.focal_and_extra_indices.data(), n); \
                        CASPAR_SET_METHOD(model, SplitFixedPoseFixedPrincipalPointFixedPointSensorFromRigDataFromStackedHost)(d.sensor_from_rig.data(), 0, n); \
                        CASPAR_SET_METHOD(model, SplitFixedPoseFixedPrincipalPointFixedPointPixelDataFromStackedHost)(d.pixels.data(), 0, n); \
                        CASPAR_SET_METHOD(model, SplitFixedPoseFixedPrincipalPointFixedPointPoseDataFromStackedHost)(d.constant_poses.data(), 0, n); \
                        CASPAR_SET_METHOD(model, SplitFixedPoseFixedPrincipalPointFixedPointPrincipalPointDataFromStackedHost)(d.constant_principal_point.data(), 0, n); \
                        CASPAR_SET_METHOD(model, SplitFixedPoseFixedPrincipalPointFixedPointPointDataFromStackedHost)(d.constant_points.data(), 0, n); \
                    }                                                            \
                    break;                                                       \
                case CasparModelData::kFixedFocalAndExtraFixedPrincipalPointFixedPoint: \
                    CASPAR_SET_METHOD(model, SplitFixed##focal##FixedPrincipalPointFixedPointNum)(n); \
                    if (n) {                                                     \
                        CASPAR_SET_METHOD(model, SplitFixed##focal##FixedPrincipalPointFixedPointPoseIndicesFromHost)(d.pose_indices.data(), n); \
                        CASPAR_SET_METHOD(model, SplitFixed##focal##FixedPrincipalPointFixedPointSensorFromRigDataFromStackedHost)(d.sensor_from_rig.data(), 0, n); \
                        CASPAR_SET_METHOD(model, SplitFixed##focal##FixedPrincipalPointFixedPointPixelDataFromStackedHost)(d.pixels.data(), 0, n); \
                        CASPAR_SET_METHOD(model, SplitFixed##focal##FixedPrincipalPointFixedPoint##focal##DataFromStackedHost)(d.constant_focal_and_extra.data(), 0, n); \
                        CASPAR_SET_METHOD(model, SplitFixed##focal##FixedPrincipalPointFixedPointPrincipalPointDataFromStackedHost)(d.constant_principal_point.data(), 0, n); \
                        CASPAR_SET_METHOD(model, SplitFixed##focal##FixedPrincipalPointFixedPointPointDataFromStackedHost)(d.constant_points.data(), 0, n); \
                    }                                                            \
                    break;                                                       \
                case CasparModelData::kNumFactorPools:                            \
                    break;                                                       \
            }                                                                    \
        }                                                                        \
    }

DEFINE_CASPAR_FACTOR_SETTER(SetSimpleRadialFactors, SimpleRadial, FocalAndExtra)
DEFINE_CASPAR_FACTOR_SETTER(SetPinholeFactors, Pinhole, Focal)

#undef DEFINE_CASPAR_FACTOR_SETTER
#undef CASPAR_SET_METHOD_IMPL
#undef CASPAR_SET_METHOD

}  // namespace

bool SolveCasparBundleAdjustment(const BundleAdjustmentOptions& options,
                                 const BundleAdjustmentConfig& config,
                                 Reconstruction* reconstruction,
                                 ceres::Solver::Summary* ceres_summary) {
    CHECK_NOTNULL(reconstruction);
    CHECK_NOTNULL(ceres_summary);
    if (!IsRepresentable(options, config, reconstruction)) {
        return false;
    }

    CasparModelData simple_radial;
    CasparModelData pinhole;
    std::unordered_map<point3D_t, unsigned int> point_indices;
    std::vector<point3D_t> point_ids;
    std::vector<CasparStorage> points;
    std::unordered_map<point3D_t, size_t> observation_counts;

    std::vector<image_t> image_ids(config.Images().begin(), config.Images().end());
    std::sort(image_ids.begin(), image_ids.end());
    for (const image_t image_id : image_ids) {
        const Image& image = reconstruction->Image(image_id);
        for (const Point2D& point2D : image.Points2D()) {
            if (point2D.HasPoint3D()) {
                ++observation_counts[point2D.Point3DId()];
            }
        }
    }
    for (const auto& entry : observation_counts) {
        if (reconstruction->Point3D(entry.first).Track().Length() != entry.second) {
            return false;
        }
    }

    std::map<int, std::vector<camera_t>> cameras_per_model;
    for (const image_t image_id : image_ids) {
        const Image& image = reconstruction->Image(image_id);
        cameras_per_model[reconstruction->Camera(image.CameraId()).ModelId()]
                .push_back(image.CameraId());
    }
    for (auto& entry : cameras_per_model) {
        auto& camera_ids = entry.second;
        std::sort(camera_ids.begin(), camera_ids.end());
        camera_ids.erase(std::unique(camera_ids.begin(), camera_ids.end()),
                         camera_ids.end());
        CasparModelData& data = entry.first == SimpleRadialCameraModel::model_id
                                        ? simple_radial
                                        : pinhole;
        data.camera_ids = camera_ids;
        for (size_t i = 0; i < camera_ids.size(); ++i) {
            data.camera_indices.emplace(camera_ids[i],
                                        static_cast<unsigned int>(i));
            const Camera& camera = reconstruction->Camera(camera_ids[i]);
            AppendCalibration(camera, &data.calibration);
            AppendSplitIntrinsics(camera, &data.focal_and_extra,
                                  &data.principal_point);
            const bool camera_constant = config.IsConstantCamera(camera_ids[i]);
            const bool focal_and_extra_variable =
                    !camera_constant && options.refine_focal_length;
            const bool principal_point_variable =
                    !camera_constant && options.refine_principal_point;
            data.intrinsic_variable_masks.emplace(
                    camera_ids[i],
                    static_cast<unsigned char>(focal_and_extra_variable ? 1 : 0) |
                            static_cast<unsigned char>(principal_point_variable ? 2
                                                                                 : 0));
        }
    }

    // A Frame can contain more than one image. Building this lookup once keeps
    // rig-aware factor assembly linear in the number of configured images.
    std::unordered_map<image_t, const Frame*> frames_by_image;
    for (const auto& frame_entry : reconstruction->Frames()) {
        const Frame& frame = frame_entry.second;
        for (const image_t frame_image_id : frame.ImageIds()) {
            frames_by_image.emplace(frame_image_id, &frame);
        }
    }

    for (const image_t image_id : image_ids) {
        Image& image = reconstruction->Image(image_id);
        const Camera& camera = reconstruction->Camera(image.CameraId());
        const auto frame_it = frames_by_image.find(image_id);
        const Frame* frame = frame_it == frames_by_image.end()
                                     ? nullptr
                                     : frame_it->second;
        CasparModelData& data = camera.ModelId() == SimpleRadialCameraModel::model_id
                                        ? simple_radial
                                        : pinhole;
        const bool pose_variable =
                !config.HasConstantPose(image_id) && options.refine_extrinsics;
        const unsigned char intrinsic_variable_mask =
                data.intrinsic_variable_masks.at(image.CameraId());
        const bool focal_and_extra_variable = intrinsic_variable_mask & 1;
        const bool principal_point_variable = intrinsic_variable_mask & 2;
        unsigned int pose_index = 0;
        if (pose_variable && frame != nullptr && frame->HasPose()) {
            const auto [it, inserted] = data.frame_pose_indices.emplace(
                    frame->FrameId(),
                    static_cast<unsigned int>(data.pose_frame_ids.size()));
            pose_index = it->second;
            if (inserted) {
                data.pose_frame_ids.push_back(frame->FrameId());
                data.pose_image_ids.push_back(kInvalidImageId);
                AppendPose(frame->RigFromWorldQvec(), frame->RigFromWorldTvec(),
                           &data.poses);
            }
        } else if (pose_variable) {
            pose_index = static_cast<unsigned int>(data.pose_frame_ids.size());
            AppendPose(image.Qvec(), image.Tvec(), &data.poses);
            data.pose_frame_ids.push_back(kInvalidFrameId);
            data.pose_image_ids.push_back(image_id);
        }
        if (pose_variable) {
            data.pose_indices.emplace(image_id, pose_index);
        }
        data.image_ids.push_back(image_id);
        for (const Point2D& point2D : image.Points2D()) {
            if (!point2D.HasPoint3D()) {
                continue;
            }
            const point3D_t point_id = point2D.Point3DId();
            const bool point_variable = !config.HasConstantPoint(point_id);
            if (!pose_variable && !focal_and_extra_variable &&
                !principal_point_variable && !point_variable) {
                // A fully constant residual cannot influence this solve and
                // Caspar intentionally has no all-fixed factor variant.
                continue;
            }
            const CasparModelData::FactorPool pool = FactorPoolFromVariableMask(
                    pose_variable, focal_and_extra_variable,
                    principal_point_variable, point_variable);
            CasparModelData::FactorData& factor = data.factors[pool];
            if (pose_variable) {
                factor.pose_indices.push_back(data.pose_indices.at(image_id));
            } else if (frame != nullptr && frame->HasPose()) {
                AppendPose(frame->RigFromWorldQvec(), frame->RigFromWorldTvec(),
                           &factor.constant_poses);
            } else {
                AppendPose(image.Qvec(), image.Tvec(), &factor.constant_poses);
            }
            const unsigned int camera_index =
                    data.camera_indices.at(image.CameraId());
            if (focal_and_extra_variable && principal_point_variable) {
                factor.calibration_indices.push_back(camera_index);
            } else {
                if (focal_and_extra_variable) {
                    factor.focal_and_extra_indices.push_back(camera_index);
                } else {
                    const size_t offset = 2 * static_cast<size_t>(camera_index);
                    factor.constant_focal_and_extra.insert(
                            factor.constant_focal_and_extra.end(),
                            data.focal_and_extra.begin() + offset,
                            data.focal_and_extra.begin() + offset + 2);
                }
                if (principal_point_variable) {
                    factor.principal_point_indices.push_back(camera_index);
                } else {
                    const size_t offset = 2 * static_cast<size_t>(camera_index);
                    factor.constant_principal_point.insert(
                            factor.constant_principal_point.end(),
                            data.principal_point.begin() + offset,
                            data.principal_point.begin() + offset + 2);
                }
            }
            if (point_variable) {
                auto point_it = point_indices.find(point_id);
                if (point_it == point_indices.end()) {
                    const unsigned int point_index =
                            static_cast<unsigned int>(point_ids.size());
                    point_it = point_indices.emplace(point_id, point_index).first;
                    point_ids.push_back(point_id);
                    const Point3D& point = reconstruction->Point3D(point_id);
                    points.insert(points.end(), {static_cast<CasparStorage>(point.X()),
                                                 static_cast<CasparStorage>(point.Y()),
                                                 static_cast<CasparStorage>(point.Z())});
                }
                factor.point_indices.push_back(point_it->second);
            } else {
                const Point3D& point = reconstruction->Point3D(point_id);
                factor.constant_points.insert(
                        factor.constant_points.end(),
                        {static_cast<CasparStorage>(point.X()),
                         static_cast<CasparStorage>(point.Y()),
                         static_cast<CasparStorage>(point.Z())});
            }
            AppendSensorFromRig(*reconstruction, image, frame,
                                &factor.sensor_from_rig);
            factor.pixels.push_back(static_cast<CasparStorage>(point2D.X()));
            factor.pixels.push_back(static_cast<CasparStorage>(point2D.Y()));
        }
    }
    const auto factor_count = [](const CasparModelData& data) {
        size_t count = 0;
        for (const CasparModelData::FactorData& factor : data.factors) {
            count += factor.pixels.size() / 2;
        }
        return count;
    };
    if (factor_count(simple_radial) + factor_count(pinhole) == 0) {
        return false;
    }

    int device_id = options.caspar_gpu_index;
    SetBestCudaDevice(device_id);
    if (device_id < 0) {
        cudaGetDevice(&device_id);
    }
    caspar::SolverParams<double> params;
    params.solver_iter_max = options.caspar_max_num_iterations;
    auto solver = CreateSolver(params, simple_radial, pinhole, point_ids.size(),
                               device_id);
    if (!point_ids.empty()) {
        solver.SetPointNodesFromStackedHost(points.data(), 0, point_ids.size());
    }
    if (!simple_radial.camera_ids.empty()) {
        solver.SetSimpleRadialCalibNodesFromStackedHost(
                simple_radial.calibration.data(), 0, simple_radial.camera_ids.size());
        solver.SetSimpleRadialFocalAndExtraNodesFromStackedHost(
                simple_radial.focal_and_extra.data(), 0,
                simple_radial.camera_ids.size());
        solver.SetSimpleRadialPrincipalPointNodesFromStackedHost(
                simple_radial.principal_point.data(), 0,
                simple_radial.camera_ids.size());
        if (!simple_radial.pose_frame_ids.empty()) {
            solver.SetSimpleRadialPoseNodesFromStackedHost(
                    simple_radial.poses.data(), 0,
                    simple_radial.pose_frame_ids.size());
        }
        SetSimpleRadialFactors(solver, simple_radial);
    }
    if (!pinhole.camera_ids.empty()) {
        solver.SetPinholeCalibNodesFromStackedHost(
                pinhole.calibration.data(), 0, pinhole.camera_ids.size());
        solver.SetPinholeFocalNodesFromStackedHost(pinhole.focal_and_extra.data(),
                                                    0,
                                                    pinhole.camera_ids.size());
        solver.SetPinholePrincipalPointNodesFromStackedHost(
                pinhole.principal_point.data(), 0, pinhole.camera_ids.size());
        if (!pinhole.pose_frame_ids.empty()) {
            solver.SetPinholePoseNodesFromStackedHost(
                    pinhole.poses.data(), 0, pinhole.pose_frame_ids.size());
        }
        SetPinholeFactors(solver, pinhole);
    }
    solver.finish_indices();
    const caspar::SolveResult result = solver.solve(
            options.solver_options.minimizer_progress_to_stdout, false);

    if (!point_ids.empty()) {
        solver.GetPointNodesToStackedHost(points.data(), 0, point_ids.size());
    }
    for (size_t i = 0; i < point_ids.size(); ++i) {
        reconstruction->Point3D(point_ids[i]).SetXYZ(
                Eigen::Vector3d(points[3 * i], points[3 * i + 1], points[3 * i + 2])
                        .cast<double>());
    }
    std::unordered_set<frame_t> updated_frame_ids;
    auto write_model = [&](CasparModelData* data, const bool is_simple_radial) {
        if (data->camera_ids.empty()) {
            return;
        }
        if (is_simple_radial) {
            solver.GetSimpleRadialCalibNodesToStackedHost(
                    data->calibration.data(), 0, data->camera_ids.size());
            solver.GetSimpleRadialFocalAndExtraNodesToStackedHost(
                    data->focal_and_extra.data(), 0, data->camera_ids.size());
            solver.GetSimpleRadialPrincipalPointNodesToStackedHost(
                    data->principal_point.data(), 0, data->camera_ids.size());
            solver.GetSimpleRadialPoseNodesToStackedHost(
                    data->poses.data(), 0, data->pose_frame_ids.size());
        } else {
            solver.GetPinholeCalibNodesToStackedHost(
                    data->calibration.data(), 0, data->camera_ids.size());
            solver.GetPinholeFocalNodesToStackedHost(data->focal_and_extra.data(),
                                                      0, data->camera_ids.size());
            solver.GetPinholePrincipalPointNodesToStackedHost(
                    data->principal_point.data(), 0, data->camera_ids.size());
            solver.GetPinholePoseNodesToStackedHost(
                    data->poses.data(), 0, data->pose_frame_ids.size());
        }
        for (size_t i = 0; i < data->camera_ids.size(); ++i) {
            Camera& camera = reconstruction->Camera(data->camera_ids[i]);
            const unsigned char variable_mask =
                    data->intrinsic_variable_masks.at(data->camera_ids[i]);
            if (variable_mask == 3) {
                WriteCalibration(&camera, data->calibration.data() + 4 * i);
            } else {
                WriteSplitIntrinsics(&camera, data->focal_and_extra.data() + 2 * i,
                                     data->principal_point.data() + 2 * i,
                                     variable_mask);
            }
        }
        for (size_t i = 0; i < data->pose_frame_ids.size(); ++i) {
            const frame_t frame_id = data->pose_frame_ids[i];
            if (frame_id != kInvalidFrameId) {
                Eigen::Vector4d qvec;
                Eigen::Vector3d tvec;
                const CasparStorage* pose = data->poses.data() + 7 * i;
                qvec << static_cast<double>(pose[3]),
                        static_cast<double>(pose[0]),
                        static_cast<double>(pose[1]),
                        static_cast<double>(pose[2]);
                tvec << static_cast<double>(pose[4]),
                        static_cast<double>(pose[5]),
                        static_cast<double>(pose[6]);
                reconstruction->Frame(frame_id).SetRigFromWorld(qvec, tvec);
                updated_frame_ids.insert(frame_id);
            } else {
                WritePose(&reconstruction->Image(data->pose_image_ids[i]),
                          data->poses.data() + 7 * i);
            }
        }
    };
    write_model(&simple_radial, true);
    write_model(&pinhole, false);
    for (const frame_t frame_id : updated_frame_ids) {
        SynchronizeFrameImages(reconstruction, frame_id);
    }

    ceres_summary->num_residuals_reduced =
            static_cast<int>(2 * (factor_count(simple_radial) +
                                  factor_count(pinhole)));
    ceres_summary->total_time_in_seconds = result.runtime;
    return true;
}

}  // namespace colmap
