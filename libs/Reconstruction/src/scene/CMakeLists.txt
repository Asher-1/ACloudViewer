# Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

set(FOLDER_NAME "scene")

# W9 (synthetic dataset generator) is ported and include-adapted but not yet
# wired into the build. Remaining mechanical adaptations, verified against the
# current compile-error list (enable COLMAP_ADD_SOURCES to iterate):
#   1. Frame/Rig pose bridge: upstream drives Rigid3d; this fork's Frame/Rig
#      store qvec/tvec. Bridge helpers in synthetic.cc's anonymous namespace:
#      Rigid3dToQvec (qvec = [w, x, y, z] from rotation().coeffs()) and
#      QvecTvecToRigid3d. Camera::SensorId -> sensor_t(camera_id);
#      Rig::SensorFromRig(id) -> SensorFromRigQvec/Tvec pair.
#   2. CameraModelId field -> int (this fork uses per-model kModelId statics;
#      SimpleRadialCameraModel::model_id already matches).
#   3. PosePrior fields/blocks are trimmed until W7 lands (pose_priors table
#      + Database::WritePosePrior/ReadAllPosePriors/UpdatePosePrior), incl.
#      the two GPS conversion helpers and SyntheticNoiseOptions prior fields.
#   4. Database::ImagePairToPairId/PairIdToImagePair are static members here:
#      add the Database:: qualifier.
#   5. Database::ExistsTwoViewGeometry -> ExistsInlierMatches (same table).
#   6. EssentialMatrixFromPose -> base/essential_matrix.h signature (R, t
#      split) via Rigid3d::rotation().toRotationMatrix()/translation();
#      FundamentalFromEssentialMatrix -> base/fundamental_matrix.h.
#      Camera::K() -> CalibrationMatrix(); Camera::IsSpherical ->
#      ModelId() == EquirectangularCameraModel::kModelId.
#   7. TwoViewGeometry::cam2_from_cam1 -> the qvec/tvec members (columns
#      already exist); Reconstruction::RegFrameIds -> iterate Frames().
#   8. synthetic_test.cc: rewrite the gtest subset as Boost.Test
#      (util/testing.h); Database::Open(...) -> Database database(":memory:").
# W9: synthetic.{h,cc} need ~10 remaining symbol adaptations against the
# frame-aware data model landed in W3-2a (EllipsoidToENU/ENUToEllipsoid GPS
# helpers vs the legacy GPSTransform API, EssentialMatrixFromPose overloads,
# ShouldSwapImagePair, DataId assignment). The numbered list is maintained in
# colmap_alignment_manifest.json (synthetic_dataset). Enable to iterate:
# COLMAP_ADD_SOURCES(
#     synthetic.h synthetic.cc
# )
#
# if (TESTS_ENABLED)
#     COLMAP_ADD_TEST(synthetic_test synthetic_test.cc)
# endif ()
