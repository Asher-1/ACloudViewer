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

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "aicore/sam3d_capi.h"

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

void check_close(float actual, float expected, const char* what) {
    const float tolerance = 1e-5f * std::max(1.0f, std::fabs(expected));
    if (!(std::fabs(actual - expected) <= tolerance)) {
        std::fprintf(stderr, "FAIL: %s (got %.8f, want %.8f)\n", what, actual,
                     expected);
        ++failures;
    }
}

// Independent activation reference (documented PLY semantics).
float ref_sigmoid(float value) { return 1.0f / (1.0f + std::exp(-value)); }

}  // namespace

int main() {
    char err[256];

    // ---- contract error paths
    // ------------------------------------------------
    check(aicore_sam3d_scene_assemble(nullptr, 0, 0, err, sizeof(err)) ==
                  nullptr,
          "assemble(NULL, 0) rejected");
    check(aicore_sam3d_scene_assemble(nullptr, 1, 0, err, sizeof(err)) ==
                  nullptr,
          "assemble(NULL, 1) rejected");
    check(aicore_sam3d_scene_result_splat_count(nullptr) == 0,
          "scene_result_splat_count(NULL) == 0");
    check(aicore_sam3d_scene_result_positions(nullptr) == nullptr,
          "scene_result_positions(NULL) == NULL");
    check(aicore_sam3d_scene_result_sh0(nullptr) == nullptr,
          "scene_result_sh0(NULL) == NULL");
    check(aicore_sam3d_scene_result_opacities(nullptr) == nullptr,
          "scene_result_opacities(NULL) == NULL");
    check(aicore_sam3d_scene_result_scales(nullptr) == nullptr,
          "scene_result_scales(NULL) == NULL");
    check(aicore_sam3d_scene_result_rotations(nullptr) == nullptr,
          "scene_result_rotations(NULL) == NULL");
    aicore_sam3d_scene_result_free(nullptr);
    check(aicore_sam3d_result_has_pose(nullptr) == 0, "has_pose(NULL) == 0");
    check(aicore_sam3d_result_pose(nullptr) == nullptr, "pose(NULL) == NULL");
    aicore_sam3d_options_set_scene_attributes(nullptr, 1);

    // ---- synthetic object with a known pose ---------------------------------
    // Two splats; the identity pose keeps the activated values, a second
    // object uses a rotation-free scale+translate pose with hand-computed
    // expectations.
    const int64_t n = 2;
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

    aicore_sam3d_scene_object objects[2];
    objects[0] = {n,
                  centers.data(),
                  sh0.data(),
                  opacity_logit.data(),
                  log_scale.data(),
                  rot_ply.data(),
                  identity_pose};
    objects[1] = {n,
                  centers.data(),
                  sh0.data(),
                  opacity_logit.data(),
                  log_scale.data(),
                  rot_ply.data(),
                  translate_pose};

    // Without normalization first: direct pose semantics.
    aicore_sam3d_scene_result* result =
            aicore_sam3d_scene_assemble(objects, 2, 0, err, sizeof(err));
    check(result != nullptr, "assemble(2 objects) succeeds");
    if (result) {
        check(aicore_sam3d_scene_result_splat_count(result) == 2 * n,
              "composed splat count");
        const float* positions = aicore_sam3d_scene_result_positions(result);
        const float* out_sh0 = aicore_sam3d_scene_result_sh0(result);
        const float* opacities = aicore_sam3d_scene_result_opacities(result);
        const float* scales = aicore_sam3d_scene_result_scales(result);
        const float* rotations = aicore_sam3d_scene_result_rotations(result);
        check(positions && out_sh0 && opacities && scales && rotations,
              "all accessors non-NULL");

        // Object 0 (identity pose): activation only.
        check_close(positions[0], 0.10f, "identity keeps center x");
        check_close(positions[1], -0.20f, "identity keeps center y");
        check_close(positions[2], 0.30f, "identity keeps center z");
        check_close(out_sh0[0], 0.25f, "sh0 pass-through");
        check_close(opacities[0], ref_sigmoid(0.3f), "opacity sigmoid");
        check_close(opacities[1], ref_sigmoid(5.0f), "opacity sigmoid 2");
        check_close(scales[0], 0.2f, "scale exp");
        check_close(scales[5], 0.6f, "scale exp 2");
        // Rotation 0: (1,0,0,0) normalized stays identity.
        check_close(rotations[0], 1.0f, "identity quaternion w");
        // Rotation 1: (1, 0.1, 0, 0) normalized.
        const float norm = std::sqrt(1.0f + 0.1f * 0.1f);
        check_close(rotations[4], 1.0f / norm, "normalized quaternion w");
        check_close(rotations[5], 0.1f / norm, "normalized quaternion x");

        // Object 1 (translate pose): p' = p + t.
        check_close(positions[6], 0.10f + 1.0f, "translate x");
        check_close(positions[7], -0.20f + 2.0f, "translate y");
        check_close(positions[8], 0.30f + 3.0f, "translate z");
        // Scale floor: 0.2 * 1 >= 1.1 * 0.0009 * 1 keeps the value.
        check_close(scales[6], 0.2f, "translate keeps scale");
        aicore_sam3d_scene_result_free(result);
    }

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
    result = aicore_sam3d_scene_assemble(&tiny, 1, 0, err, sizeof(err));
    check(result != nullptr, "assemble(tiny scales) succeeds");
    if (result) {
        const float* scales = aicore_sam3d_scene_result_scales(result);
        check_close(scales[0], 1.1f * 0.0009f * 10.f,
                    "official min-kernel scale floor");
        check_close(scales[3], 0.2f * 10.f, "scale * pose_scale");
        aicore_sam3d_scene_result_free(result);
    }

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
    result = aicore_sam3d_scene_assemble(&normalized, 1, 1, err, sizeof(err));
    check(result != nullptr, "assemble(normalize) succeeds");
    if (result) {
        const float* positions = aicore_sam3d_scene_result_positions(result);
        const float* scales = aicore_sam3d_scene_result_scales(result);
        // Active bounds: x in [0, 2], y/z in [0, 0] -> inv_scale = 2.
        // center = (1/2, 0, 0); p' = p / 2 - center.
        check_close(positions[0], 0.0f / 2.0f - 0.5f, "normalize center x0");
        check_close(positions[3], 2.0f / 2.0f - 0.5f, "normalize center x1");
        check_close(scales[0], 0.2f / 2.0f, "normalize divides scales");
        aicore_sam3d_scene_result_free(result);
    }

    // Missing array is rejected with a message.
    aicore_sam3d_scene_object incomplete = {n,
                                            centers.data(),
                                            nullptr,
                                            opacity_logit.data(),
                                            log_scale.data(),
                                            rot_ply.data(),
                                            identity_pose};
    check(aicore_sam3d_scene_assemble(&incomplete, 1, 0, err, sizeof(err)) ==
                  nullptr,
          "missing sh0 rejected");

    if (failures == 0) {
        std::printf("test_sam3d_scene_assemble: all checks passed\n");
        return 0;
    }
    std::printf("test_sam3d_scene_assemble: %d failure(s)\n", failures);
    return 1;
}
