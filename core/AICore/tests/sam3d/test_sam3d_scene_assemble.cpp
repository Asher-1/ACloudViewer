// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// SAM 3D multi-object scene assembly C ABI test — fast, model-free. Covers
// the contract error paths and the documented composition math (activation,
// make_scene pose application, opacity-bound normalization) with
// independently computed expectations. Real-asset parity against the
// upstream scene-assemble flow lives in the validation runner probes.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include "aicore/sam3d_capi.h"

namespace {

// Independent activation reference (documented PLY semantics).
float ref_sigmoid(float value) { return 1.0f / (1.0f + std::exp(-value)); }

float tolerance(float expected) {
    return 1e-5f * std::max(1.0f, std::fabs(expected));
}

using SceneResult = std::unique_ptr<aicore_sam3d_scene_result,
                                    decltype(&aicore_sam3d_scene_result_free)>;

TEST(Sam3dSceneContract, HandlesNullInputsAndAccessors) {
    char err[256];
    SceneResult empty(
            aicore_sam3d_scene_assemble(nullptr, 0, 0, err, sizeof(err)),
            &aicore_sam3d_scene_result_free);
    EXPECT_EQ(empty.get(), nullptr) << "assemble(NULL, 0) rejected";
    SceneResult missing(
            aicore_sam3d_scene_assemble(nullptr, 1, 0, err, sizeof(err)),
            &aicore_sam3d_scene_result_free);
    EXPECT_EQ(missing.get(), nullptr) << "assemble(NULL, 1) rejected";
    EXPECT_EQ(aicore_sam3d_scene_result_splat_count(nullptr), 0)
            << "scene_result_splat_count(NULL) == 0";
    EXPECT_EQ(aicore_sam3d_scene_result_positions(nullptr), nullptr)
            << "scene_result_positions(NULL) == NULL";
    EXPECT_EQ(aicore_sam3d_scene_result_sh0(nullptr), nullptr)
            << "scene_result_sh0(NULL) == NULL";
    EXPECT_EQ(aicore_sam3d_scene_result_opacities(nullptr), nullptr)
            << "scene_result_opacities(NULL) == NULL";
    EXPECT_EQ(aicore_sam3d_scene_result_scales(nullptr), nullptr)
            << "scene_result_scales(NULL) == NULL";
    EXPECT_EQ(aicore_sam3d_scene_result_rotations(nullptr), nullptr)
            << "scene_result_rotations(NULL) == NULL";
    aicore_sam3d_scene_result_free(nullptr);
    EXPECT_EQ(aicore_sam3d_result_has_pose(nullptr), 0)
            << "has_pose(NULL) == 0";
    EXPECT_EQ(aicore_sam3d_result_pose(nullptr), nullptr)
            << "pose(NULL) == NULL";
    aicore_sam3d_options_set_scene_attributes(nullptr, 1);
}

class Sam3dSceneAssembly : public testing::Test {
protected:
    char err[256];
    static constexpr int64_t n = 2;
    std::vector<float> centers = {0.10f,  -0.20f, 0.30f,  //
                                  -0.10f, 0.20f,  -0.30f};
    std::vector<float> sh0 = {0.25f,  -0.50f, 0.75f,  //
                              -0.25f, 0.50f,  -0.75f};
    std::vector<float> opacity_logit = {0.3f, 5.0f};
    std::vector<float> log_scale = {std::log(0.2f), std::log(0.4f),
                                    std::log(0.6f),  //
                                    std::log(0.2f), std::log(0.4f),
                                    std::log(0.6f)};
    // Unnormalized PLY quaternion: identity direction (1,0,0,0) + small tilt.
    std::vector<float> rot_ply = {1.0f, 0.0f, 0.0f, 0.0f,  //
                                  1.0f, 0.1f, 0.0f, 0.0f};

    // Identity pose (wxyz = identity, t = 0, scale = 1).
    const float identity_pose[10] = {1.f, 0.f, 0.f, 0.f, 0.f,
                                     0.f, 0.f, 1.f, 1.f, 1.f};
    // Pure translation pose: t = (1, 2, 3).
    const float translate_pose[10] = {1.f, 0.f, 0.f, 0.f, 1.f,
                                      2.f, 3.f, 1.f, 1.f, 1.f};

    aicore_sam3d_scene_object objects[2] = {
            {n, centers.data(), sh0.data(), opacity_logit.data(),
             log_scale.data(), rot_ply.data(), identity_pose},
            {n, centers.data(), sh0.data(), opacity_logit.data(),
             log_scale.data(), rot_ply.data(), translate_pose}};
};

TEST_F(Sam3dSceneAssembly, ActivatesAttributesAndAppliesObjectPoses) {
    // Without normalization first: direct pose semantics.
    SceneResult result(
            aicore_sam3d_scene_assemble(objects, 2, 0, err, sizeof(err)),
            &aicore_sam3d_scene_result_free);
    ASSERT_TRUE(result != nullptr) << "assemble(2 objects) succeeds";
    EXPECT_EQ(aicore_sam3d_scene_result_splat_count(result.get()), 2 * n)
            << "composed splat count";
    const float* positions = aicore_sam3d_scene_result_positions(result.get());
    const float* out_sh0 = aicore_sam3d_scene_result_sh0(result.get());
    const float* opacities = aicore_sam3d_scene_result_opacities(result.get());
    const float* scales = aicore_sam3d_scene_result_scales(result.get());
    const float* rotations = aicore_sam3d_scene_result_rotations(result.get());
    ASSERT_TRUE(positions && out_sh0 && opacities && scales && rotations)
            << "all accessors non-NULL";

    // Object 0 (identity pose): activation only.
    EXPECT_NEAR(positions[0], 0.10f, tolerance(0.10f))
            << "identity keeps center x";
    EXPECT_NEAR(positions[1], -0.20f, tolerance(-0.20f))
            << "identity keeps center y";
    EXPECT_NEAR(positions[2], 0.30f, tolerance(0.30f))
            << "identity keeps center z";
    EXPECT_NEAR(out_sh0[0], 0.25f, tolerance(0.25f)) << "sh0 pass-through";
    EXPECT_NEAR(opacities[0], ref_sigmoid(0.3f), tolerance(ref_sigmoid(0.3f)))
            << "opacity sigmoid";
    EXPECT_NEAR(opacities[1], ref_sigmoid(5.0f), tolerance(ref_sigmoid(5.0f)))
            << "opacity sigmoid 2";
    EXPECT_NEAR(scales[0], 0.2f, tolerance(0.2f)) << "scale exp";
    EXPECT_NEAR(scales[5], 0.6f, tolerance(0.6f)) << "scale exp 2";
    // Rotation 0: (1,0,0,0) normalized stays identity.
    EXPECT_NEAR(rotations[0], 1.0f, tolerance(1.0f)) << "identity quaternion w";
    // Rotation 1: (1, 0.1, 0, 0) normalized.
    const float norm = std::sqrt(1.0f + 0.1f * 0.1f);
    EXPECT_NEAR(rotations[4], 1.0f / norm, tolerance(1.0f / norm))
            << "normalized quaternion w";
    EXPECT_NEAR(rotations[5], 0.1f / norm, tolerance(0.1f / norm))
            << "normalized quaternion x";

    // Object 1 (translate pose): p' = p + t.
    EXPECT_NEAR(positions[6], 0.10f + 1.0f, tolerance(0.10f + 1.0f))
            << "translate x";
    EXPECT_NEAR(positions[7], -0.20f + 2.0f, tolerance(-0.20f + 2.0f))
            << "translate y";
    EXPECT_NEAR(positions[8], 0.30f + 3.0f, tolerance(0.30f + 3.0f))
            << "translate z";
    // Scale floor: 0.2 * 1 >= 1.1 * 0.0009 * 1 keeps the value.
    EXPECT_NEAR(scales[6], 0.2f, tolerance(0.2f)) << "translate keeps scale";
}

TEST_F(Sam3dSceneAssembly, AppliesPoseScaleAndOfficialKernelFloor) {
    // ---- scale pose with the official floor ---------------------------------
    // pose scale = (10, 10, 10): activated scale 0.2 -> 2.0; a tiny scale
    // (1e-6) would floor at 1.1 * 0.0009 * 10 = 0.0099.
    const float scale_up_pose[10] = {1.f, 0.f, 0.f,  0.f,  0.f,
                                     0.f, 0.f, 10.f, 10.f, 10.f};
    std::vector<float> tiny_log_scale = {std::log(1e-6f), std::log(1e-6f),
                                         std::log(1e-6f), std::log(0.2f),
                                         std::log(0.2f),  std::log(0.2f)};
    aicore_sam3d_scene_object tiny = {n,
                                      centers.data(),
                                      sh0.data(),
                                      opacity_logit.data(),
                                      tiny_log_scale.data(),
                                      rot_ply.data(),
                                      scale_up_pose};
    SceneResult result(
            aicore_sam3d_scene_assemble(&tiny, 1, 0, err, sizeof(err)),
            &aicore_sam3d_scene_result_free);
    ASSERT_TRUE(result != nullptr) << "assemble(tiny scales) succeeds";
    const float* scales = aicore_sam3d_scene_result_scales(result.get());
    ASSERT_NE(scales, nullptr);
    EXPECT_NEAR(scales[0], 1.1f * 0.0009f * 10.f,
                tolerance(1.1f * 0.0009f * 10.f))
            << "official min-kernel scale floor";
    EXPECT_NEAR(scales[3], 0.2f * 10.f, tolerance(0.2f * 10.f))
            << "scale * pose_scale";
}

TEST_F(Sam3dSceneAssembly, NormalizesOpacityBoundsAndScales) {
    // ---- normalization
    // ------------------------------------------------------- Two splats, high
    // opacity, axis-aligned extents on x drive inv_scale.
    std::vector<float> norm_centers = {0.0f, 0.0f, 0.0f,  //
                                       2.0f, 0.0f, 0.0f};
    std::vector<float> high_opacity = {8.0f, 8.0f};  // sigmoid ~= 0.9997
    aicore_sam3d_scene_object normalized = {n,
                                            norm_centers.data(),
                                            sh0.data(),
                                            high_opacity.data(),
                                            log_scale.data(),
                                            rot_ply.data(),
                                            identity_pose};
    SceneResult result(
            aicore_sam3d_scene_assemble(&normalized, 1, 1, err, sizeof(err)),
            &aicore_sam3d_scene_result_free);
    ASSERT_TRUE(result != nullptr) << "assemble(normalize) succeeds";
    const float* positions = aicore_sam3d_scene_result_positions(result.get());
    const float* scales = aicore_sam3d_scene_result_scales(result.get());
    ASSERT_NE(positions, nullptr);
    ASSERT_NE(scales, nullptr);
    // Active bounds: x in [0, 2], y/z in [0, 0] -> inv_scale = 2.
    // center = (1/2, 0, 0); p' = p / 2 - center.
    EXPECT_NEAR(positions[0], 0.0f / 2.0f - 0.5f, tolerance(0.0f / 2.0f - 0.5f))
            << "normalize center x0";
    EXPECT_NEAR(positions[3], 2.0f / 2.0f - 0.5f, tolerance(2.0f / 2.0f - 0.5f))
            << "normalize center x1";
    EXPECT_NEAR(scales[0], 0.2f / 2.0f, tolerance(0.2f / 2.0f))
            << "normalize divides scales";
}

TEST_F(Sam3dSceneAssembly, RejectsMissingAttributeArray) {
    // Missing array is rejected with a message.
    aicore_sam3d_scene_object incomplete = {n,
                                            centers.data(),
                                            nullptr,
                                            opacity_logit.data(),
                                            log_scale.data(),
                                            rot_ply.data(),
                                            identity_pose};
    SceneResult result(
            aicore_sam3d_scene_assemble(&incomplete, 1, 0, err, sizeof(err)),
            &aicore_sam3d_scene_result_free);
    EXPECT_EQ(result.get(), nullptr) << "missing sh0 rejected";
}

}  // namespace
