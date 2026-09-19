// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// W18.5 (upstream bundle_adjustment_ceres.{h,cc} parity): the Ceres backend
// implementation of the abstract BundleAdjuster interface, split out of
// bundle_adjustment.cc; bundle_adjustment.cc keeps the solver-agnostic
// options/config surface and PrintSolverSummary.

#include "estimators/bundle_adjustment_ceres.h"

#include "estimators/alignment.h"
#include "estimators/cost_functions/manifold.h"
#include "estimators/cost_functions/pose_prior.h"
#include "estimators/cost_functions/reprojection_error.h"

#include <map>

#if defined(CASPAR_ENABLED)
#include "estimators/bundle_adjustment_caspar.h"
#endif

#include <iomanip>

#ifdef OPENMP_ENABLED
#include <omp.h>
#endif

#include "sensor/models.h"
#include "estimators/cost_functions/cost_functions.h"
#include "scene/projection.h"
#include "optim/manifold.h"
#include "util/cuda.h"
#include "util/misc.h"
#include "util/threading.h"
#include "util/timer.h"

namespace colmap {

namespace {

// Wraps the options-level check_if_stopped hook into a Ceres iteration
// callback: returning USER_ABORT stops the minimizer after the current
// iteration while keeping all accepted parameter updates.
class CancellationCallback : public ceres::IterationCallback {
public:
    explicit CancellationCallback(std::function<bool()> check_if_stopped)
        : check_if_stopped_(std::move(check_if_stopped)) {}

    ceres::CallbackReturnType operator()(
            const ceres::IterationSummary&) override {
        // Upstream Ceres 2.x renamed SOLVER_ABORT to USER_ABORT; the
        // self-built Ceres in this repo keeps the SOLVER_ABORT spelling.
        return check_if_stopped_ && check_if_stopped_()
                       ? ceres::SOLVER_ABORT
                       : ceres::SOLVER_CONTINUE;
    }

private:
    std::function<bool()> check_if_stopped_;
};

}  // namespace

////////////////////////////////////////////////////////////////////////////////
// BundleAdjuster
////////////////////////////////////////////////////////////////////////////////

CeresBundleAdjuster::CeresBundleAdjuster(
        const BundleAdjustmentOptions& options,
        const BundleAdjustmentConfig& config)
    : BundleAdjuster(options, config) {
    CHECK(options_.Check());
}
bool CeresBundleAdjuster::Solve(Reconstruction* reconstruction) {
    // W7: robustly align the reconstruction to the position priors before
    // building the problem, so the prior residuals are expressed in the
    // reconstruction's coordinate frame (upstream parity).
    if (pose_prior_options_) {
        Sim3d metric_from_orig;
        if (!AlignReconstructionToPosePriors(
                    *reconstruction,
                    pose_priors_,
                    pose_prior_options_->alignment_ransac_options,
                    pose_prior_options_->prior_position_fallback_stddev,
                    &metric_from_orig)) {
            LOG(WARNING) << "Alignment w.r.t. prior positions failed";
            return false;
        }
        reconstruction->Transform(metric_from_orig);
    }

    CHECK_NOTNULL(reconstruction);
    CHECK(!problem_) << "Cannot use the same BundleAdjuster multiple times";

    const bool use_caspar =
            options_.use_caspar ||
            options_.backend == BundleAdjustmentBackend::CASPAR;
    if (use_caspar) {
#if defined(CASPAR_ENABLED)
        if (SolveCasparBundleAdjustment(options_, config_, reconstruction,
                                        &summary_)) {
            return true;
        }
        LOG(WARNING) << "Caspar BA cannot represent this legacy problem; "
                        "using the Ceres backend for this solve";
#else
        LOG(WARNING) << "Caspar BA was requested, but this build does not "
                        "enable RECONSTRUCTION_CASPAR_ENABLED; using Ceres";
#endif
    }

    problem_.reset(new ceres::Problem());

    ceres::LossFunction* loss_function = options_.CreateLossFunction();
    SetUp(reconstruction, loss_function);

    if (problem_->NumResiduals() == 0) {
        return false;
    }

    ceres::Solver::Options solver_options =
            options_.CreateSolverOptions(config_, *problem_);

    // Cooperative cancellation: check_if_stopped is evaluated once per
    // iteration; a true return aborts the minimizer while all accepted
    // parameter updates stay applied via TearDown().
    CancellationCallback cancellation_callback(options_.check_if_stopped);
    if (options_.check_if_stopped) {
        solver_options.callbacks.push_back(&cancellation_callback);
    }

    ceres::Solve(solver_options, problem_.get(), &summary_);

    if (solver_options.minimizer_progress_to_stdout) {
        std::cout << std::endl;
    }

    if (options_.print_summary) {
        PrintHeading2("Bundle adjustment report");
        PrintSolverSummary(summary_);
    }

    TearDown(reconstruction);

    return true;
}

// Upstream parity (d3ccaf35 bundle_adjustment_ceres.cc).
namespace {

BundleAdjustmentTerminationType CeresTerminationTypeToTerminationType(
        const ceres::TerminationType ceres_type) {
    switch (ceres_type) {
        case ceres::CONVERGENCE:
            return BundleAdjustmentTerminationType::CONVERGENCE;
        case ceres::NO_CONVERGENCE:
            return BundleAdjustmentTerminationType::NO_CONVERGENCE;
        case ceres::FAILURE:
            return BundleAdjustmentTerminationType::FAILURE;
        case ceres::USER_SUCCESS:
            return BundleAdjustmentTerminationType::USER_SUCCESS;
        case ceres::USER_FAILURE:
            return BundleAdjustmentTerminationType::USER_FAILURE;
    }
    LOG(FATAL) << "Unknown Ceres termination type: " << ceres_type;
    return BundleAdjustmentTerminationType::FAILURE;
}

}  // namespace

std::shared_ptr<CeresBundleAdjustmentSummary>
CeresBundleAdjustmentSummary::Create(ceres::Solver::Summary ceres_summary) {
    auto summary = std::make_shared<CeresBundleAdjustmentSummary>();
    summary->termination_type =
            CeresTerminationTypeToTerminationType(
                ceres_summary.termination_type);
    summary->num_residuals = ceres_summary.num_residuals_reduced;
    summary->ceres_summary = std::move(ceres_summary);
    return summary;
}

std::string CeresBundleAdjustmentSummary::BriefReport() const {
    return ceres_summary.BriefReport();
}

std::shared_ptr<CeresBundleAdjustmentSummary> CeresBundleAdjuster::summary()
        const {
    return CeresBundleAdjustmentSummary::Create(summary_);
}

ceres::Problem& CeresBundleAdjuster::Problem() {
    THROW_CHECK_NOTNULL(problem_);
    return *problem_;
}

const ceres::Problem& CeresBundleAdjuster::Problem() const {
    THROW_CHECK_NOTNULL(problem_);
    return *problem_;
}

const ceres::Solver::Summary& CeresBundleAdjuster::Summary() const {
    return summary_;
}

void CeresBundleAdjuster::SetUp(Reconstruction* reconstruction,
                           ceres::LossFunction* loss_function) {
    // Warning: AddPointsToProblem assumes that AddImageToProblem is called
    // first. Do not change order of instructions!

    manifold_marked_blocks_.clear();
    // W3-2b: the shadow blocks are rebuilt for every SetUp; they are kept
    // alive after TearDown so post-solve consumers (e.g. the BA covariance
    // estimator) can still resolve the registered pose pointers.
    frame_blocks_.clear();
    sensor_blocks_.clear();

    // W3-2b step 4: forward dual-track sync. The pose parameter blocks are
    // the frame storage; make sure it holds the freshest camera poses before
    // the problem is built (the global mapper pre-syncs frames itself, but
    // image-level writers like the legacy incremental mapper may not).
    for (const image_t image_id : config_.Images()) {
        if (!reconstruction->ExistsImage(image_id)) {
            continue;
        }
        Image& image = reconstruction->Image(image_id);
        if (!image.HasFrameId() || !image.HasPose() ||
            !reconstruction->ExistsFrame(image.FrameId())) {
            continue;
        }
        const Rigid3d cam_from_world(
                Eigen::Quaterniond(image.Qvec()(0), image.Qvec()(1),
                                   image.Qvec()(2), image.Qvec()(3)),
                image.Tvec());
        reconstruction->Frame(image.FrameId())
                .SetCamFromWorld(image.CameraId(), cam_from_world);
    }
    for (const image_t image_id : config_.Images()) {
        AddImageToProblem(image_id, reconstruction, loss_function);
    }
    for (const auto point3D_id : config_.VariablePoints()) {
        AddPointToProblem(point3D_id, reconstruction, loss_function);
    }
    for (const auto point3D_id : config_.ConstantPoints()) {
        AddPointToProblem(point3D_id, reconstruction, loss_function);
    }

    ParameterizeCameras(reconstruction);
    ParameterizeRigsAndFrames(reconstruction);
    ParameterizePoints(reconstruction);

    // Upstream parity (d3ccaf35): fix the global gauge after all
    // parameter blocks have been registered. Without it, the problem has a
    // global 7-DoF null space that makes the normal equations singular for
    // noise-free problems (the non-trivial-rig GP test failure).
    switch (config_.FixedGauge()) {
        case BundleAdjustmentGauge::UNSPECIFIED:
            break;
        case BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD:
            FixGaugeWithTwoCamsFromWorld(reconstruction);
            break;
        case BundleAdjustmentGauge::THREE_POINTS:
            FixGaugeWithThreePoints(reconstruction);
            break;
    }
}

void CeresBundleAdjuster::TearDown(Reconstruction* reconstruction) {
    if (reconstruction == nullptr) {
        return;
    }
    // W3-2b step 4: the pose parameter blocks were the frame storage itself,
    // so the refined rig_from_world poses are already in place. Write the
    // refined sensor_from_rig poses back to the rigs and mirror the composed
    // camera poses into the legacy per-image buffers so both pose tracks stay
    // consistent for every consumer.
    // W3-2b step 5: write the refined single-block poses back to the
    // frames and rigs (the parameter blocks are node-based-map shadows).
    for (const auto& [key, block] : frame_blocks_) {
        if (!reconstruction->ExistsFrame(key)) {
            continue;
        }
        Frame& frame = reconstruction->Frame(key);
        if (frame.HasPose()) {
            frame.SetRigFromWorld(block.rig_from_world);
        }
    }
    for (const auto& [key, block] : sensor_blocks_) {
        if (!reconstruction->ExistsRig(key.first)) {
            continue;
        }
        Rig& rig = reconstruction->Rig(key.first);
        if (!rig.HasSensor(block.sensor_id)) {
            continue;
        }
        rig.SetSensorFromRig(block.sensor_id, block.sensor_from_rig);
    }

    for (const image_t image_id : config_.Images()) {
        if (!reconstruction->ExistsImage(image_id)) {
            continue;
        }
        Image& image = reconstruction->Image(image_id);
        if (!image.HasFrameId() ||
            !reconstruction->ExistsFrame(image.FrameId())) {
            continue;
        }
        const Frame& frame = reconstruction->Frame(image.FrameId());
        if (!frame.HasPose()) {
            continue;
        }
        Rigid3d cam_from_world = frame.RigFromWorld();
        const Rig& rig = reconstruction->Rig(frame.RigId());
        const sensor_t sensor_id =
                reconstruction->Camera(image.CameraId()).SensorId();
        if (rig.NumSensors() > 1 && rig.HasSensor(sensor_id) &&
            !rig.IsRefSensor(sensor_id) &&
            rig.HasSensorFromRig(sensor_id)) {
            cam_from_world = rig.SensorFromRig(sensor_id) * cam_from_world;
        }
        const Eigen::Quaterniond& q = cam_from_world.rotation();
        image.SetQvec(Eigen::Vector4d(q.w(), q.x(), q.y(), q.z()));
        image.SetTvec(cam_from_world.translation());
    }
    // The shadow block maps stay alive after TearDown for post-solve
    // consumers; the next SetUp clears them (see the top of SetUp).
}

CeresBundleAdjuster::SensorPoseBlock& CeresBundleAdjuster::GetOrCreateSensorBlock(
        const rig_t rig_id,
        const sensor_t sensor_id,
        const Rigid3d& sensor_from_rig) {
    const auto key = std::make_pair(rig_id, sensor_id);
    const auto it = sensor_blocks_.find(key);
    if (it != sensor_blocks_.end()) {
        return it->second;
    }
    SensorPoseBlock block;
    block.rig_id = rig_id;
    block.sensor_id = sensor_id;
    block.sensor_from_rig = sensor_from_rig;
    return sensor_blocks_.emplace(key, block).first->second;
}

const std::map<frame_t, CeresBundleAdjuster::FramePoseBlock>&
CeresBundleAdjuster::frame_blocks() const {
    return frame_blocks_;
}

CeresBundleAdjuster::FramePoseBlock& CeresBundleAdjuster::GetOrCreateFrameBlock(
        const frame_t frame_id, const Rigid3d& rig_from_world) {
    const auto it = frame_blocks_.find(frame_id);
    if (it != frame_blocks_.end()) {
        return it->second;
    }
    FramePoseBlock block;
    block.frame_id = frame_id;
    block.rig_from_world = rig_from_world;
    return frame_blocks_.emplace(frame_id, block).first->second;
}

namespace {

// Upstream parity (d3ccaf35 bundle_adjustment_ceres.cc): three non-collinear
// points fix the gauge as a degenerate-case fallback.
struct FixedGaugeWithThreePoints {
    // The number of fixed points for the Gauge.
    Eigen::Index num_fixed_points = 0;
    // The coordinates of the fixed points as columns.
    Eigen::Matrix3d fixed_points = Eigen::Matrix3d::Zero();

    bool MaybeAddFixedPoint(const Eigen::Vector3d& point) {
        if (num_fixed_points >= 3) {
            return false;
        }
        fixed_points.col(num_fixed_points) = point;
        if (fixed_points.colPivHouseholderQr().rank() > num_fixed_points) {
            ++num_fixed_points;
            return true;
        }
        fixed_points.col(num_fixed_points).setZero();
        return false;
    }
};

}  // namespace

void CeresBundleAdjuster::FixGaugeWithThreePoints(
        Reconstruction* reconstruction) {
    FixedGaugeWithThreePoints fixed_gauge;

    // First check if we already fixed enough points in the problem.
    for (const auto& [point3D_id, num_observations] :
         point3D_num_observations_) {
        const Point3D& point3D = reconstruction->Point3D(point3D_id);
        if (problem_->IsParameterBlockConstant(point3D.XYZ().data()) &&
            fixed_gauge.MaybeAddFixedPoint(point3D.XYZ()) &&
            fixed_gauge.num_fixed_points >= 3) {
            return;
        }
    }

    // Otherwise, fix sufficient points in the problem.
    for (const auto& [point3D_id, num_observations] :
         point3D_num_observations_) {
        Point3D& point3D = reconstruction->Point3D(point3D_id);
        if (!problem_->IsParameterBlockConstant(point3D.XYZ().data()) &&
            fixed_gauge.MaybeAddFixedPoint(point3D.XYZ())) {
            problem_->SetParameterBlockConstant(point3D.XYZ().data());
            if (fixed_gauge.num_fixed_points >= 3) {
                return;
            }
        }
    }

    LOG(WARNING) << "Failed to fix Gauge due to insufficient number of "
                    "fixed points: "
                 << fixed_gauge.num_fixed_points;
}

void CeresBundleAdjuster::FixGaugeWithTwoCamsFromWorld(
        Reconstruction* reconstruction) {
    // Upstream parity (d3ccaf35): no need to fix the Gauge if all frames
    // are constant.
    if (!options_.refine_rig_from_world) {
        return;
    }

    Image* image1 = nullptr;
    Image* image2 = nullptr;

    // Check if a sensor is either a reference sensor, or a non-reference
    // sensor with sensor_from_rig fixed.
    auto IsParameterizedConstSensor =
            [this](const Image& image) {
                const sensor_t sensor_id = image.CameraPtr()->SensorId();
                if (image.FramePtr()->RigPtr()->IsRefSensor(sensor_id)) {
                    return true;
                }
                const auto sb_it = sensor_blocks_.find(
                        {image.FramePtr()->RigId(), sensor_id});
                if (sb_it != sensor_blocks_.end() &&
                    problem_->HasParameterBlock(
                            sb_it->second.sensor_from_rig.params.data()) &&
                    problem_->IsParameterBlockConstant(
                            sb_it->second.sensor_from_rig.params.data())) {
                    return true;
                }
                // Cover corner case when ReprojErrorConstantPoseCostFunctor
                // is used.
                if (config_.HasConstantSensorFromRigPose(sensor_id) ||
                    !options_.refine_extrinsics ||
                    !options_.refine_sensor_from_rig) {
                    return true;
                }
                return false;
            };

    // First, search through the already fixed cameras in the problem.
    for (const image_t image_id : parameterized_image_ids_) {
        Image& image = reconstruction->Image(image_id);
        if (config_.HasConstantRigFromWorldPose(image.FrameId()) &&
            IsParameterizedConstSensor(image)) {
            if (image1 == nullptr) {
                image1 = &image;
            } else if (image1 != nullptr &&
                       image1->FrameId() != image.FrameId()) {
                // No need to fix the Gauge if two frames are already fixed.
                return;
            }
        }
    }

    // Otherwise, search through the variable cameras in the problem.
    int frame2_from_world_fixed_dim = 0;
    for (const image_t image_id : parameterized_image_ids_) {
        Image& image = reconstruction->Image(image_id);
        const auto fb_it = frame_blocks_.find(image.FrameId());
        if (fb_it == frame_blocks_.end()) {
            continue;
        }
        const Rigid3d& rig_from_world = fb_it->second.rig_from_world;
        if (image1 == nullptr && IsParameterizedConstSensor(image)) {
            image1 = &image;
        } else if (image1 != nullptr &&
                   image1->FrameId() != image.FrameId() &&
                   IsParameterizedConstSensor(image) &&
                   problem_->HasParameterBlock(rig_from_world.params.data())) {
            // Check if one of the baseline dimensions is large enough and
            // choose it as the fixed coordinate. If there is no such pair
            // of frames, then the scale is not constrained well.
            const auto fb1_it = frame_blocks_.find(image1->FrameId());
            const Eigen::Vector3d baseline =
                    (fb1_it->second.rig_from_world * Inverse(rig_from_world))
                            .translation();
            Eigen::Index max_coeff_idx = 0;
            if (baseline.cwiseAbs().maxCoeff(&max_coeff_idx) > 1e-9) {
                image2 = &image;
                frame2_from_world_fixed_dim = max_coeff_idx;
                break;
            }
        }
    }

    if (image1 == nullptr || image2 == nullptr) {
        LOG(WARNING) << "Failed to fix Gauge with two cameras. "
                        "Falling back to fixing Gauge with three points.";
        FixGaugeWithThreePoints(reconstruction);
        return;
    }

    if (!config_.HasConstantRigFromWorldPose(image1->FrameId())) {
        auto fb1_it = frame_blocks_.find(image1->FrameId());
        problem_->SetParameterBlockConstant(
                fb1_it->second.rig_from_world.params.data());
    }

    if (!config_.HasConstantRigFromWorldPose(image2->FrameId())) {
        auto fb2_it = frame_blocks_.find(image2->FrameId());
        Rigid3d& frame2_from_world = fb2_it->second.rig_from_world;
        if (options_.constant_rig_from_world_rotation) {
            SetManifold(problem_.get(),
                        frame2_from_world.params.data(),
                        CreateSubsetManifold(
                                7, {0, 1, 2, 3,
                                    4 + frame2_from_world_fixed_dim}));
        } else {
            SetManifold(problem_.get(),
                        frame2_from_world.params.data(),
                        CreateProductManifold(
                                CreateEigenQuaternionManifold(),
                                CreateSubsetManifold(
                                        3, {frame2_from_world_fixed_dim})));
        }
    }
}

void CeresBundleAdjuster::SetPosePriors(
        PosePriorBundleAdjustmentOptions options,
        std::vector<PosePrior> pose_priors) {
    pose_prior_options_ = std::move(options);
    pose_priors_ = std::move(pose_priors);
}

namespace {

ceres::LossFunction* CreatePosePriorLossFunction(
        const PosePriorBundleAdjustmentOptions& options) {
    switch (options.prior_position_loss_function_type) {
        case BundleAdjustmentOptions::LossFunctionType::TRIVIAL:
            return nullptr;
        case BundleAdjustmentOptions::LossFunctionType::SOFT_L1:
            return new ceres::SoftLOneLoss(options.prior_position_loss_scale);
        case BundleAdjustmentOptions::LossFunctionType::CAUCHY:
            return new ceres::CauchyLoss(options.prior_position_loss_scale);
        case BundleAdjustmentOptions::LossFunctionType::HUBER:
            return new ceres::HuberLoss(options.prior_position_loss_scale);
    }
    return nullptr;
}

}  // namespace

void CeresBundleAdjuster::AddImageToProblem(const image_t image_id,
                                       Reconstruction* reconstruction,
                                       ceres::LossFunction* loss_function) {
    Image& image = reconstruction->Image(image_id);
    Camera& camera = reconstruction->Camera(image.CameraId());
    double* camera_params_data = camera.ParamsData();

    // W3-2b step 5 (upstream parity, d3ccaf35): the pose parameter blocks
    // are single Rigid3d shadow blocks (node-based map, stable addresses)
    // shared by every image of a frame.
    bool frame_pose = false;
    bool compose_rig = false;
    sensor_t rig_sensor_id{};
    Rigid3d* sensor_from_rig_block = nullptr;
    Rigid3d* rig_from_world_block = nullptr;
    // Legacy frameless-cache-image path (legacy fixtures only): separate
    // qvec/tvec buffers owned by the image itself.
    bool legacy_pose = false;
    double* qvec_data = nullptr;
    double* tvec_data = nullptr;

    if (image.HasFrameId() && reconstruction->ExistsFrame(image.FrameId())) {
        Frame& frame = reconstruction->Frame(image.FrameId());
        frame_pose = true;
        FramePoseBlock& fb =
                GetOrCreateFrameBlock(image.FrameId(), frame.RigFromWorld());
        rig_from_world_block = &fb.rig_from_world;
        const Rig& rig = reconstruction->Rig(frame.RigId());
        if (options_.refine_sensor_from_rig && rig.NumSensors() > 1 &&
            rig.HasSensor(camera.SensorId()) &&
            !rig.IsRefSensor(camera.SensorId()) &&
            rig.HasSensorFromRig(camera.SensorId())) {
            compose_rig = true;
            rig_sensor_id = camera.SensorId();
            SensorPoseBlock& sb = GetOrCreateSensorBlock(
                    frame.RigId(), rig_sensor_id,
                    rig.SensorFromRig(rig_sensor_id));
            sensor_from_rig_block = &sb.sensor_from_rig;
        }
    } else {
        // CostFunction assumes unit quaternions.
        image.NormalizeQvec();
        legacy_pose = true;
        qvec_data = image.Qvec().data();
        tvec_data = image.Tvec().data();
    }

    // W7 pose-prior residual (upstream parity: single Rigid3d block functor;
    // the legacy frameless path keeps the split qvec/tvec functor).
    if (pose_prior_options_) {
        for (const PosePrior& pose_prior : pose_priors_) {
            if (pose_prior.corr_data_id.sensor_id.type ==
                        SensorType::CAMERA &&
                pose_prior.corr_data_id.id == image_id &&
                pose_prior.HasPosition()) {
                ceres::LossFunction* prior_loss =
                        CreatePosePriorLossFunction(*pose_prior_options_);
                if (frame_pose) {
                    problem_->AddResidualBlock(
                            new ceres::AutoDiffCostFunction<
                                    AbsolutePosePositionPriorCostFunctor,
                                    3,
                                    7>(
                                    new AbsolutePosePositionPriorCostFunctor(
                                            pose_prior.position)),
                            prior_loss,
                            rig_from_world_block->params.data());
                } else {
                    problem_->AddResidualBlock(
                            new ceres::AutoDiffCostFunction<
                                    AbsolutePosePositionPriorQvecTvecCostFunctor,
                                    3,
                                    4,
                                    3>(
                                    new AbsolutePosePositionPriorQvecTvecCostFunctor(
                                            pose_prior.position)),
                            prior_loss,
                            qvec_data,
                            tvec_data);
                }
                break;
            }
        }
    }

    // Upstream parity (d3ccaf35): per-image pose freezing is split into
    // rig-from-world and sensor-from-rig conditions (the fork's
    // !refine_extrinsics freezes both).
    const bool constant_rig_from_world =
            !options_.refine_extrinsics || !options_.refine_rig_from_world ||
            config_.HasConstantPose(image_id) ||
            config_.HasConstantRigFromWorldPose(image.FrameId());
    const bool constant_sensor_from_rig =
            !options_.refine_extrinsics ||
            !options_.refine_sensor_from_rig ||
            config_.HasConstantSensorFromRigPose(rig_sensor_id);

    // Add residuals to bundle adjustment problem.
    size_t num_observations = 0;
    for (const Point2D& point2D : image.Points2D()) {
        if (!point2D.HasPoint3D() ||
            config_.IsIgnoredPoint(point2D.Point3DId())) {
            continue;
        }

        Point3D& point3D = reconstruction->Point3D(point2D.Point3DId());
        assert(point3D.Track().Length() > 1);

        // Upstream parity: skip points with track length below minimum.
        if (options_.min_track_length > 0 &&
            static_cast<int>(point3D.Track().Length()) <
                    options_.min_track_length) {
            continue;
        }

        num_observations += 1;
        point3D_num_observations_[point2D.Point3DId()] += 1;

        if (legacy_pose) {
            // Legacy frameless-cache-image path: the fork's 4-block functor
            // over the image-local qvec/tvec buffers.
            ceres::CostFunction* cost_function = nullptr;
            switch (camera.ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                     \
    case CameraModel::model_id:                                            \
        cost_function = BundleAdjustmentCostFunction<CameraModel>::Create( \
                point2D.XY());                                             \
        break;

                CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
            }
            problem_->AddResidualBlock(cost_function, loss_function,
                                       qvec_data, tvec_data,
                                       point3D.XYZ().data(),
                                       camera_params_data);
            continue;
        }

        // Upstream parity (d3ccaf35): constant-pose residuals take the
        // composed camera pose by value; variable residuals carry the
        // single Rigid3d shadow blocks.
        if (compose_rig) {
            if (constant_rig_from_world && constant_sensor_from_rig) {
                const Rigid3d cam_from_world =
                        *sensor_from_rig_block * *rig_from_world_block;
                problem_->AddResidualBlock(
                        CreateCameraCostFunction<
                                ReprojErrorConstantPoseCostFunctor>(
                                camera.ModelId(), point2D.XY(), cam_from_world),
                        loss_function,
                        point3D.XYZ().data(), camera_params_data);
            } else if (!constant_rig_from_world &&
                       constant_sensor_from_rig) {
                problem_->AddResidualBlock(
                        CreateCameraCostFunction<
                                RigReprojErrorConstantRigCostFunctor>(
                                camera.ModelId(), point2D.XY(),
                                *sensor_from_rig_block),
                        loss_function,
                        point3D.XYZ().data(),
                        rig_from_world_block->params.data(),
                        camera_params_data);
            } else {
                problem_->AddResidualBlock(
                        CreateCameraCostFunction<RigReprojErrorCostFunctor>(
                                camera.ModelId(), point2D.XY()),
                        loss_function,
                        point3D.XYZ().data(),
                        sensor_from_rig_block->params.data(),
                        rig_from_world_block->params.data(),
                        camera_params_data);
            }
        } else {
            if (constant_rig_from_world) {
                problem_->AddResidualBlock(
                        CreateCameraCostFunction<
                                ReprojErrorConstantPoseCostFunctor>(
                                camera.ModelId(), point2D.XY(),
                                *rig_from_world_block),
                        loss_function,
                        point3D.XYZ().data(), camera_params_data);
            } else {
                problem_->AddResidualBlock(
                        CreateCameraCostFunction<ReprojErrorCostFunctor>(
                                camera.ModelId(), point2D.XY()),
                        loss_function,
                        point3D.XYZ().data(),
                        rig_from_world_block->params.data(),
                        camera_params_data);
            }
        }
    }

    if (num_observations > 0) {
        camera_ids_.insert(image.CameraId());
        if (frame_pose) {
            parameterized_image_ids_.insert(image_id);
        } else {
            // Legacy frameless path: keep the fork's qvec/tvec manifolds.
            if (manifold_marked_blocks_.insert(qvec_data).second) {
                SetQuaternionManifoldWxyz(problem_.get(), qvec_data);
            }
            if (config_.HasConstantTvec(image_id)) {
                if (manifold_marked_blocks_.insert(tvec_data).second) {
                    SetSubsetManifold(3, config_.ConstantTvec(image_id),
                                      problem_.get(), tvec_data);
                }
            }
        }
    }
}

void CeresBundleAdjuster::AddPointToProblem(const point3D_t point3D_id,
                                       Reconstruction* reconstruction,
                                       ceres::LossFunction* loss_function) {
    if (config_.IsIgnoredPoint(point3D_id)) {
        return;
    }
    Point3D& point3D = reconstruction->Point3D(point3D_id);

    // Upstream parity (dbb41680, Caspar wiring): points on tracks shorter
    // than min_track_length are excluded from the problem entirely.
    if (options_.min_track_length > 0 &&
        static_cast<int>(point3D.Track().Length()) <
                options_.min_track_length) {
        return;
    }

    // Is 3D point already fully contained in the problem? I.e. its entire track
    // is contained in `variable_image_ids`, `constant_image_ids`,
    // `constant_x_image_ids`.
    if (point3D_num_observations_[point3D_id] == point3D.Track().Length()) {
        return;
    }

    for (const auto& track_el : point3D.Track().Elements()) {
        // Skip observations that were already added in `FillImages`.
        if (config_.HasImage(track_el.image_id)) {
            continue;
        }

        point3D_num_observations_[point3D_id] += 1;

        Image& image = reconstruction->Image(track_el.image_id);
        Camera& camera = reconstruction->Camera(image.CameraId());
        const Point2D& point2D = image.Point2D(track_el.point2D_idx);

        // We do not want to refine the camera of images that are not
        // part of `constant_image_ids_`, `constant_image_ids_`,
        // `constant_x_image_ids_`.
        if (camera_ids_.count(image.CameraId()) == 0) {
            camera_ids_.insert(image.CameraId());
            config_.SetConstantCamera(image.CameraId());
        }

        ceres::CostFunction* cost_function = nullptr;

        if (camera.ModelId() == EquirectangularCameraModel::model_id) {
            cost_function =
                    EquirectangularBundleAdjustmentConstantPoseCostFunction::Create(
                            image.Qvec(), image.Tvec(), point2D.XY());
        } else {
            switch (camera.ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                         \
    case CameraModel::model_id:                                                \
        cost_function =                                                        \
                BundleAdjustmentConstantPoseCostFunction<CameraModel>::Create( \
                        image.Qvec(), image.Tvec(), point2D.XY());             \
        break;

            CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
        }
        }
        problem_->AddResidualBlock(cost_function, loss_function,
                                   point3D.XYZ().data(), camera.ParamsData());
    }
}

void CeresBundleAdjuster::ParameterizeCameras(Reconstruction* reconstruction) {
    const bool constant_camera = !options_.refine_focal_length &&
                                 !options_.refine_principal_point &&
                                 !options_.refine_extra_params;
    for (const camera_t camera_id : camera_ids_) {
        Camera& camera = reconstruction->Camera(camera_id);

        if (constant_camera || config_.IsConstantCamera(camera_id)) {
            problem_->SetParameterBlockConstant(camera.ParamsData());
            continue;
        } else {
            std::vector<int> const_camera_params;

            if (!options_.refine_focal_length) {
                const std::vector<size_t>& params_idxs =
                        camera.FocalLengthIdxs();
                const_camera_params.insert(const_camera_params.end(),
                                           params_idxs.begin(),
                                           params_idxs.end());
            }
            if (!options_.refine_principal_point) {
                const std::vector<size_t>& params_idxs =
                        camera.PrincipalPointIdxs();
                const_camera_params.insert(const_camera_params.end(),
                                           params_idxs.begin(),
                                           params_idxs.end());
            }
            if (!options_.refine_extra_params) {
                const std::vector<size_t>& params_idxs =
                        camera.ExtraParamsIdxs();
                const_camera_params.insert(const_camera_params.end(),
                                           params_idxs.begin(),
                                           params_idxs.end());
            }

            if (const_camera_params.size() > 0) {
                // ceres::SubsetParameterization* camera_params_parameterization
                // =
                //     new ceres::SubsetParameterization(
                //         static_cast<int>(camera.NumParams()),
                //         const_camera_params);
                // problem_->SetParameterization(camera.ParamsData(),
                //                               camera_params_parameterization);
                SetSubsetManifold(static_cast<int>(camera.NumParams()),
                                  const_camera_params, problem_.get(),
                                  camera.ParamsData());
            }
        }
    }
}

void CeresBundleAdjuster::ParameterizeRigsAndFrames(
        Reconstruction* reconstruction) {
    std::unordered_set<rig_t> parameterized_rig_ids;
    std::unordered_set<sensor_t> parameterized_sensor_ids;
    std::unordered_set<frame_t> parameterized_frame_ids;
    for (const image_t image_id : parameterized_image_ids_) {
        Image& image = reconstruction->Image(image_id);
        parameterized_rig_ids.insert(image.FramePtr()->RigId());

        // Parameterize sensor_from_rig.
        const sensor_t sensor_id = image.CameraPtr()->SensorId();
        const bool not_parameterized_before =
                parameterized_sensor_ids.insert(sensor_id).second;
        if (not_parameterized_before) {
            const auto sb_it = sensor_blocks_.find(
                    {image.FramePtr()->RigId(), sensor_id});
            if (sb_it != sensor_blocks_.end()) {
                Rigid3d& sensor_from_rig = sb_it->second.sensor_from_rig;
                // CostFunction assumes unit quaternions.
                sensor_from_rig.rotation().normalize();
                if (problem_->HasParameterBlock(
                            sensor_from_rig.params.data())) {
                    SetManifold(problem_.get(),
                                sensor_from_rig.params.data(),
                                CreateProductManifold(
                                        CreateEigenQuaternionManifold(),
                                        CreateEuclideanManifold<3>()));
                    if (!options_.refine_extrinsics ||
                        !options_.refine_sensor_from_rig ||
                        config_.HasConstantSensorFromRigPose(sensor_id)) {
                        problem_->SetParameterBlockConstant(
                                sensor_from_rig.params.data());
                    }
                }
            }
        }

        // Parameterize rig_from_world.
        if (parameterized_frame_ids.insert(image.FrameId()).second) {
            const auto fb_it = frame_blocks_.find(image.FrameId());
            if (fb_it != frame_blocks_.end()) {
                Rigid3d& rig_from_world = fb_it->second.rig_from_world;
                // CostFunction assumes unit quaternions.
                rig_from_world.rotation().normalize();
                if (problem_->HasParameterBlock(rig_from_world.params.data())) {
                    if (!options_.refine_extrinsics ||
                        !options_.refine_rig_from_world ||
                        config_.HasConstantRigFromWorldPose(image.FrameId())) {
                        problem_->SetParameterBlockConstant(
                                rig_from_world.params.data());
                    } else if (options_.constant_rig_from_world_rotation) {
                        SetManifold(problem_.get(),
                                    rig_from_world.params.data(),
                                    CreateSubsetManifold(7, {0, 1, 2, 3}));
                    } else if (config_.HasConstantTvec(image_id)) {
                        // Fork parity: freeze the selected translation
                        // dimensions within the single Rigid3d block.
                        SetManifold(problem_.get(),
                                    rig_from_world.params.data(),
                                    CreateProductManifold(
                                            CreateEigenQuaternionManifold(),
                                            CreateSubsetManifold(
                                                    3, config_.ConstantTvec(
                                                            image_id))));
                    } else {
                        SetManifold(problem_.get(),
                                    rig_from_world.params.data(),
                                    CreateProductManifold(
                                            CreateEigenQuaternionManifold(),
                                            CreateEuclideanManifold<3>()));
                    }
                }
            }
        }
    }

    // Set the rig poses as constant, if the reference sensor is not part of
    // the problem. Otherwise, the relative pose between the sensors is not
    // well constrained.
    for (const rig_t rig_id : parameterized_rig_ids) {
        Rig& rig = reconstruction->Rig(rig_id);
        if (parameterized_sensor_ids.count(rig.RefSensorId()) != 0) {
            continue;
        }
        for (auto& [key, block] : sensor_blocks_) {
            if (key.first != rig_id || block.sensor_id == rig.RefSensorId()) {
                continue;
            }
            if (problem_->HasParameterBlock(
                        block.sensor_from_rig.params.data())) {
                problem_->SetParameterBlockConstant(
                        block.sensor_from_rig.params.data());
            }
        }
    }
}

void CeresBundleAdjuster::ParameterizePoints(Reconstruction* reconstruction) {
    for (const auto elem : point3D_num_observations_) {
        Point3D& point3D = reconstruction->Point3D(elem.first);
        if (point3D.Track().Length() > elem.second) {
            problem_->SetParameterBlockConstant(point3D.XYZ().data());
        }
    }

    for (const point3D_t point3D_id : config_.ConstantPoints()) {
        Point3D& point3D = reconstruction->Point3D(point3D_id);
        problem_->SetParameterBlockConstant(point3D.XYZ().data());
    }
}

////////////////////////////////////////////////////////////////////////////////
// RigBundleAdjuster
////////////////////////////////////////////////////////////////////////////////


std::unique_ptr<BundleAdjuster> CreateDefaultBundleAdjuster(
        const BundleAdjustmentOptions& options,
        const BundleAdjustmentConfig& config) {
  return std::make_unique<CeresBundleAdjuster>(options, config);
}

}  // namespace colmap
