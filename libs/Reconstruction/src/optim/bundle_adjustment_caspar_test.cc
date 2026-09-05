// SPDX-License-Identifier: MIT

#define TEST_NAME "optim/bundle_adjustment_caspar"
#include "util/testing.h"

#include <cmath>

#include <cuda_runtime.h>

#include "base/camera_models.h"
#include "base/correspondence_graph.h"
#include "base/frame.h"
#include "base/projection.h"
#include "base/rig.h"
#include "optim/bundle_adjustment_caspar.h"

using namespace colmap;

namespace {

Reconstruction CreatePinholeParityReconstruction() {
  constexpr size_t kNumPoints = 64;
  constexpr size_t kImageWidth = 1000;
  constexpr size_t kImageHeight = 800;

  Reconstruction reconstruction;
  CorrespondenceGraph correspondence_graph;
  Camera camera;
  camera.SetCameraId(1);
  camera.InitializeWithId(PinholeCameraModel::model_id, 900.0, kImageWidth,
                          kImageHeight);
  reconstruction.AddCamera(camera);

  std::vector<point3D_t> point_ids;
  point_ids.reserve(kNumPoints);
  for (size_t i = 0; i < kNumPoints; ++i) {
    const double x = (static_cast<double>(i % 8) - 3.5) * 0.18;
    const double y = (static_cast<double>(i / 8) - 3.5) * 0.14;
    const double z = 4.5 + 0.12 * static_cast<double>(i % 5);
    point_ids.push_back(
        reconstruction.AddPoint3D(Eigen::Vector3d(x, y, z), Track()));
  }

  const std::array<Eigen::Vector3d, 2> true_tvecs = {
      Eigen::Vector3d::Zero(), Eigen::Vector3d(0.28, -0.06, 0.02)};
  for (size_t image_idx = 0; image_idx < true_tvecs.size(); ++image_idx) {
    const image_t image_id = static_cast<image_t>(image_idx + 1);
    Image image;
    image.SetImageId(image_id);
    image.SetCameraId(camera.CameraId());
    image.SetName("caspar-parity-" + std::to_string(image_id));
    image.Qvec() = ComposeIdentityQuaternion();
    image.Tvec() = true_tvecs[image_idx];

    const Eigen::Matrix3x4d projection = image.ProjectionMatrix();
    std::vector<Eigen::Vector2d> points2D;
    points2D.reserve(point_ids.size());
    for (size_t point_idx = 0; point_idx < point_ids.size(); ++point_idx) {
      const Eigen::Vector2d projected = ProjectPointToImage(
          reconstruction.Point3D(point_ids[point_idx]).XYZ(), projection,
          camera);
      const double x_noise = static_cast<double>(point_idx % 3) - 1.0;
      const double y_noise = static_cast<double>(point_idx % 5) - 2.0;
      points2D.push_back(projected + Eigen::Vector2d(0.10 * x_noise,
                                                       0.08 * y_noise));
    }
    image.SetPoints2D(points2D);
    image.SetRegistered(true);
    reconstruction.AddImage(image);
    correspondence_graph.AddImage(image_id, points2D.size());
  }

  reconstruction.SetUp(&correspondence_graph);
  for (size_t image_idx = 0; image_idx < true_tvecs.size(); ++image_idx) {
    const image_t image_id = static_cast<image_t>(image_idx + 1);
    for (size_t point_idx = 0; point_idx < point_ids.size(); ++point_idx) {
      reconstruction.AddObservation(
          point_ids[point_idx],
          TrackElement(image_id, static_cast<point2D_t>(point_idx)));
    }
  }

  reconstruction.Camera(camera.CameraId()).Params(0) *= 1.03;
  reconstruction.Camera(camera.CameraId()).Params(1) *= 0.97;
  reconstruction.Camera(camera.CameraId()).Params(2) += 4.0;
  reconstruction.Camera(camera.CameraId()).Params(3) -= 3.0;
  reconstruction.Image(1).Tvec() += Eigen::Vector3d(0.03, -0.02, 0.05);
  reconstruction.Image(2).Tvec() += Eigen::Vector3d(-0.03, 0.02, -0.04);
  for (size_t point_idx = 0; point_idx < point_ids.size(); ++point_idx) {
    reconstruction.Point3D(point_ids[point_idx]).SetXYZ(
        reconstruction.Point3D(point_ids[point_idx]).XYZ() +
        Eigen::Vector3d(0.01, -0.01, 0.02));
  }

  return reconstruction;
}

Reconstruction CreateFixedSensorFromRigReconstruction() {
  Reconstruction reconstruction = CreatePinholeParityReconstruction();

  Camera secondary_camera = reconstruction.Camera(1);
  secondary_camera.SetCameraId(2);
  reconstruction.AddCamera(secondary_camera);
  reconstruction.Image(2).SetCameraId(secondary_camera.CameraId());

  Rig rig;
  rig.SetRigId(1);
  rig.AddRefCamera(1);
  rig.AddCamera(secondary_camera.CameraId(), ComposeIdentityQuaternion(),
                Eigen::Vector3d(0.28, -0.06, 0.02));
  reconstruction.AddRig(rig);

  Frame frame;
  frame.SetFrameId(1);
  frame.SetRigId(rig.RigId());
  frame.AddImageId(1);
  frame.AddImageId(2);
  frame.SetRigFromWorld(ComposeIdentityQuaternion(),
                        Eigen::Vector3d(0.02, -0.01, 0.03));
  reconstruction.AddFrame(frame);
  return reconstruction;
}

double ComputeRmsReprojectionError(const Reconstruction& reconstruction) {
  double squared_error_sum = 0.0;
  size_t num_observations = 0;
  for (const auto& image_entry : reconstruction.Images()) {
    const Image& image = image_entry.second;
    const Camera& camera = reconstruction.Camera(image.CameraId());
    for (const Point2D& point2D : image.Points2D()) {
      if (!point2D.HasPoint3D()) {
        continue;
      }
      const double squared_error = CalculateSquaredReprojectionError(
          point2D.XY(), reconstruction.Point3D(point2D.Point3DId()).XYZ(),
          image.Qvec(), image.Tvec(), camera);
      if (!std::isfinite(squared_error)) {
        return std::numeric_limits<double>::infinity();
      }
      squared_error_sum += squared_error;
      ++num_observations;
    }
  }
  return std::sqrt(squared_error_sum / static_cast<double>(num_observations));
}

BundleAdjustmentOptions CreateParityOptions() {
  BundleAdjustmentOptions options;
  options.print_summary = false;
  options.refine_principal_point = true;
  // Ceres is intentionally CPU-only. Caspar uses its own CUDA kernels when
  // selected below, so this option must not be used to request Ceres CUDA.
  options.use_gpu = false;
  options.min_num_images_gpu_solver = 0;
  options.solver_options.num_threads = 1;
  options.solver_options.max_num_iterations = 100;
  options.caspar_max_num_iterations = 100;
  return options;
}

}  // namespace

BOOST_AUTO_TEST_CASE(TestRejectsUnportedFactorVariantsBeforeCudaExecution) {
  Reconstruction reconstruction;
  BundleAdjustmentOptions options;
  BundleAdjustmentConfig config;
  config.AddConstantPoint(1);
  ceres::Solver::Summary summary;
  BOOST_CHECK(!SolveCasparBundleAdjustment(options, config, &reconstruction,
                                           &summary));
}

BOOST_AUTO_TEST_CASE(TestPinholeCeresCasparReprojectionParity) {
  // Caspar remains a GPU backend, while the reference Ceres solve is always
  // CPU-only to keep libceres portable across hosts and CUDA toolkit versions.
  int cuda_device_count = 0;
  BOOST_REQUIRE_EQUAL(cudaGetDeviceCount(&cuda_device_count), cudaSuccess);
  BOOST_REQUIRE_GT(cuda_device_count, 0);

  const Reconstruction initial = CreatePinholeParityReconstruction();
  Reconstruction ceres_reconstruction = initial;
  Reconstruction caspar_reconstruction = initial;

  BundleAdjustmentConfig config;
  config.AddImage(1);
  config.AddImage(2);

  const BundleAdjustmentOptions ceres_options = CreateParityOptions();
  ceres::Problem options_probe;
  const ceres::Solver::Options effective_ceres_options =
      ceres_options.CreateSolverOptions(config, options_probe);
  BOOST_REQUIRE_NE(effective_ceres_options.dense_linear_algebra_library_type,
                   ceres::CUDA);
  BundleAdjuster ceres_adjuster(ceres_options, config);
  BOOST_REQUIRE(ceres_adjuster.Solve(&ceres_reconstruction));

  BundleAdjustmentOptions caspar_options = CreateParityOptions();
  caspar_options.backend = BundleAdjustmentBackend::CASPAR;
  ceres::Solver::Summary caspar_summary;
  // Call the adapter directly so a future dispatcher fallback to Ceres cannot
  // make this numerical parity test pass without executing Caspar kernels.
  BOOST_REQUIRE(SolveCasparBundleAdjustment(caspar_options, config,
                                             &caspar_reconstruction,
                                             &caspar_summary));

  const double ceres_rms = ComputeRmsReprojectionError(ceres_reconstruction);
  const double caspar_rms =
      ComputeRmsReprojectionError(caspar_reconstruction);
  BOOST_TEST_MESSAGE("Ceres CPU BA: rms=" << ceres_rms
                                                     << " time="
                                                     << ceres_adjuster.Summary()
                                                            .total_time_in_seconds
                                                     << "s; Caspar: rms="
                                                     << caspar_rms << " time="
                                                     << caspar_summary.total_time_in_seconds
                                                     << "s");
  BOOST_CHECK(std::isfinite(ceres_rms));
  BOOST_CHECK(std::isfinite(caspar_rms));
  BOOST_CHECK_LT(caspar_rms, 1.0);
  BOOST_CHECK_LE(caspar_rms, ceres_rms * 1.25 + 1e-3);
  BOOST_CHECK_GT(ceres_adjuster.Summary().total_time_in_seconds, 0.0);
  BOOST_CHECK_GT(caspar_summary.total_time_in_seconds, 0.0);
  // The fixture is intentionally small for CI. This catches runtime or graph
  // regressions without claiming a speedup from a noisy microbenchmark.
  BOOST_CHECK_LE(caspar_summary.total_time_in_seconds,
                 ceres_adjuster.Summary().total_time_in_seconds * 20.0 +
                     0.25);
}

BOOST_AUTO_TEST_CASE(TestCasparSynchronizesFixedSensorFromRigFramePose) {
  int cuda_device_count = 0;
  BOOST_REQUIRE_EQUAL(cudaGetDeviceCount(&cuda_device_count), cudaSuccess);
  BOOST_REQUIRE_GT(cuda_device_count, 0);

  Reconstruction reconstruction = CreateFixedSensorFromRigReconstruction();
  BundleAdjustmentConfig config;
  config.AddImage(1);
  config.AddImage(2);

  BundleAdjustmentOptions options = CreateParityOptions();
  options.backend = BundleAdjustmentBackend::CASPAR;
  ceres::Solver::Summary summary;
  BOOST_REQUIRE(SolveCasparBundleAdjustment(options, config, &reconstruction,
                                             &summary));
  BOOST_REQUIRE(reconstruction.Frame(1).HasPose());

  const Rig& rig = reconstruction.Rig(1);
  const Frame& frame = reconstruction.Frame(1);
  for (const image_t image_id : frame.ImageIds()) {
    const Image& image = reconstruction.Image(image_id);
    Eigen::Vector4d expected_qvec;
    Eigen::Vector3d expected_tvec;
    ConcatenatePoses(rig.CamFromRigQvec(image.CameraId()),
                     rig.CamFromRigTvec(image.CameraId()),
                     frame.RigFromWorldQvec(), frame.RigFromWorldTvec(),
                     &expected_qvec, &expected_tvec);
    BOOST_CHECK_SMALL((image.Qvec() - expected_qvec).norm(), 1e-10);
    BOOST_CHECK_SMALL((image.Tvec() - expected_tvec).norm(), 1e-10);
  }
  BOOST_CHECK_GT(summary.total_time_in_seconds, 0.0);
}

BOOST_AUTO_TEST_CASE(TestCasparMergedFixedPoseAndPointFactorVariants) {
  int cuda_device_count = 0;
  BOOST_REQUIRE_EQUAL(cudaGetDeviceCount(&cuda_device_count), cudaSuccess);
  BOOST_REQUIRE_GT(cuda_device_count, 0);

  Reconstruction reconstruction = CreatePinholeParityReconstruction();
  const Eigen::Vector4d fixed_qvec = reconstruction.Image(1).Qvec();
  const Eigen::Vector3d fixed_tvec = reconstruction.Image(1).Tvec();
  const Eigen::Vector3d fixed_point = reconstruction.Point3D(1).XYZ();

  BundleAdjustmentConfig config;
  config.AddImage(1);
  config.AddImage(2);
  config.SetConstantPose(1);
  config.AddConstantPoint(1);

  BundleAdjustmentOptions options = CreateParityOptions();
  options.backend = BundleAdjustmentBackend::CASPAR;
  ceres::Solver::Summary summary;
  BOOST_REQUIRE(SolveCasparBundleAdjustment(options, config, &reconstruction,
                                             &summary));
  BOOST_CHECK_EQUAL(summary.num_residuals_reduced, 2 * 2 * 64);
  BOOST_CHECK_SMALL((reconstruction.Image(1).Qvec() - fixed_qvec).norm(),
                    1e-14);
  BOOST_CHECK_SMALL((reconstruction.Image(1).Tvec() - fixed_tvec).norm(),
                    1e-14);
  BOOST_CHECK_SMALL((reconstruction.Point3D(1).XYZ() - fixed_point).norm(),
                    1e-14);
}

BOOST_AUTO_TEST_CASE(TestCasparSplitIntrinsicFactorVariantsCeresParity) {
  int cuda_device_count = 0;
  BOOST_REQUIRE_EQUAL(cudaGetDeviceCount(&cuda_device_count), cudaSuccess);
  BOOST_REQUIRE_GT(cuda_device_count, 0);

  // These three runs cover every generated split factor.  Each run has a
  // fixed pose (image 1), a variable pose (image 2), a fixed point (point 1),
  // and variable points; fixing focal, principal point, or both then selects
  // 4 + 4 + 3 = all 11 non-merged factor variants.
  const auto check_variant_family = [](const bool refine_focal_and_extra,
                                       const bool refine_principal_point) {
    const Reconstruction initial = CreatePinholeParityReconstruction();
    Reconstruction ceres_reconstruction = initial;
    Reconstruction caspar_reconstruction = initial;
    const auto camera_params = [](const Reconstruction& reconstruction) {
      Eigen::Vector4d params;
      for (size_t index = 0; index < 4; ++index) {
        params(index) = reconstruction.Camera(1).Params(index);
      }
      return params;
    };
    const Eigen::Vector4d initial_params = camera_params(initial);

    BundleAdjustmentConfig config;
    config.AddImage(1);
    config.AddImage(2);
    config.SetConstantPose(1);
    config.AddConstantPoint(1);

    BundleAdjustmentOptions ceres_options = CreateParityOptions();
    ceres_options.refine_focal_length = refine_focal_and_extra;
    ceres_options.refine_extra_params = refine_focal_and_extra;
    ceres_options.refine_principal_point = refine_principal_point;
    BundleAdjuster ceres_adjuster(ceres_options, config);
    BOOST_REQUIRE(ceres_adjuster.Solve(&ceres_reconstruction));

    BundleAdjustmentOptions caspar_options = ceres_options;
    caspar_options.backend = BundleAdjustmentBackend::CASPAR;
    ceres::Solver::Summary caspar_summary;
    BOOST_REQUIRE(SolveCasparBundleAdjustment(caspar_options, config,
                                               &caspar_reconstruction,
                                               &caspar_summary));

    const Eigen::Vector4d caspar_params = camera_params(caspar_reconstruction);
    const Eigen::Vector4d ceres_params = camera_params(ceres_reconstruction);
    if (!refine_focal_and_extra) {
      BOOST_CHECK_SMALL((caspar_params.head<2>() - initial_params.head<2>()).norm(),
                        1e-12);
    } else {
      BOOST_CHECK_SMALL((caspar_params.head<2>() - ceres_params.head<2>()).norm(),
                        5e-2);
    }
    if (!refine_principal_point) {
      BOOST_CHECK_SMALL((caspar_params.tail<2>() - initial_params.tail<2>()).norm(),
                        1e-12);
    } else {
      BOOST_CHECK_SMALL((caspar_params.tail<2>() - ceres_params.tail<2>()).norm(),
                        2e-2);
    }

    const double ceres_rms = ComputeRmsReprojectionError(ceres_reconstruction);
    const double caspar_rms =
        ComputeRmsReprojectionError(caspar_reconstruction);
    BOOST_TEST_MESSAGE("split intrinsics focal=" << refine_focal_and_extra
                                                   << " pp="
                                                   << refine_principal_point
                                                   << " Ceres rms=" << ceres_rms
                                                   << " Caspar rms=" << caspar_rms);
    BOOST_CHECK(std::isfinite(caspar_rms));
    BOOST_CHECK_LE(caspar_rms, ceres_rms * 1.25 + 1e-3);
    // The focal=false/principal-point=false run has one fully fixed residual
    // (fixed pose plus fixed point). Caspar has no all-fixed factor and, like
    // upstream, omits it from the active graph.
    const int expected_residuals =
        refine_focal_and_extra || refine_principal_point ? 2 * 2 * 64
                                                          : 2 * (2 * 64 - 1);
    BOOST_CHECK_EQUAL(caspar_summary.num_residuals_reduced, expected_residuals);
  };

  check_variant_family(false, true);
  check_variant_family(true, false);
  check_variant_family(false, false);
}
