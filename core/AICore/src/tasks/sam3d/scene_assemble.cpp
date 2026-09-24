// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Ported from the upstream sam-3d-objects-ggml scene_assemble.cpp and the
// Gaussian PLY loader in gaussian_renderer.cpp. The activation, pose and
// normalization math is kept expression-for-expression identical so a
// composed scene stays bit-comparable with the upstream scene-assemble flow
// on the same per-object PLY inputs.
#include "scene_assemble.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace sam3d {
namespace {

// GS_MIN_KERNEL: the decoder representation's 3d_filter_kernel_size
// (checkpoints/hf/slat_decoder_gs.yaml).
constexpr float kMinKernel = 0.0009f;

struct Vec3 {
    float x, y, z;
};

// Same pytorch3d quaternion conventions as pose_decoder.cpp (wxyz, Hamilton
// product, unit-quaternion inverse = conjugate).
std::array<float, 4> quat_conjugate(const std::array<float, 4>& q) {
    return {q[0], -q[1], -q[2], -q[3]};
}

std::array<float, 4> quat_multiply(const std::array<float, 4>& a,
                                   const std::array<float, 4>& b) {
    const float aw = a[0], ax = a[1], ay = a[2], az = a[3];
    const float bw = b[0], bx = b[1], by = b[2], bz = b[3];
    return {
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
    };
}

// pytorch3d.transforms.quaternion_to_matrix (row-major 3x3).
std::array<std::array<float, 3>, 3> quat_to_matrix(
        const std::array<float, 4>& q) {
    const float r = q[0], i = q[1], j = q[2], k = q[3];
    const float norm2 = r * r + i * i + j * j + k * k;
    const float two_s = norm2 == 0.0f ? std::numeric_limits<float>::infinity()
                                      : 2.0f / norm2;
    return {{
            {{1.0f - two_s * (j * j + k * k), two_s * (i * j - k * r),
              two_s * (i * k + j * r)}},
            {{two_s * (i * j + k * r), 1.0f - two_s * (i * i + k * k),
              two_s * (j * k - i * r)}},
            {{two_s * (i * k - j * r), two_s * (j * k + i * r),
              1.0f - two_s * (i * i + j * j)}},
    }};
}

float sigmoid(float value) {
    if (value >= 0.0f) {
        const float exp_neg = std::exp(-value);
        return 1.0f / (1.0f + exp_neg);
    }
    const float exp_pos = std::exp(value);
    return exp_pos / (1.0f + exp_pos);
}

// Official make_scene pose semantics on the activated splat representation
// (Gaussian.get_xyz/get_rotation/get_scaling):
//   positions: p' = R^T @ diag(pose_scale) @ p + t  (scale first, then the
//              inverse rotation, then translate — the pytorch3d
//              Transform3d row-vector action, verified against the official
//              object_pointcloud numerically; the naive R @ diag(s) @ p
//              reading is wrong by exactly R vs R^T).
//   rotations: q' = standardize(normalize(inv(pose_q) (x) q))
//   scales:    s' = max(s * pose_scale, 1.1 * min_kernel * pose_scale)
void apply_scene_pose(SceneSplatSet& splats,
                      const NativeInstancePose& pose,
                      std::string& error) {
    const auto rotation = quat_to_matrix(pose.rotation_wxyz);
    const std::array<float, 4> inverse = quat_conjugate(pose.rotation_wxyz);
    const size_t count = splats.size();
    for (size_t index = 0; index < count; ++index) {
        float* position = &splats.positions[index * 3];
        const Vec3 point{position[0], position[1], position[2]};
        const Vec3 scaled{pose.scale[0] * point.x, pose.scale[1] * point.y,
                          pose.scale[2] * point.z};
        for (size_t column = 0; column < 3; ++column) {
            position[column] = rotation[0][column] * scaled.x +
                               rotation[1][column] * scaled.y +
                               rotation[2][column] * scaled.z +
                               pose.translation[column];
        }
        float* quaternion = &splats.rotations[index * 4];
        std::array<float, 4> rotated = quat_multiply(
                inverse,
                {quaternion[0], quaternion[1], quaternion[2], quaternion[3]});
        const float norm =
                std::sqrt(rotated[0] * rotated[0] + rotated[1] * rotated[1] +
                          rotated[2] * rotated[2] + rotated[3] * rotated[3]);
        if (!(norm > 0.0f) || !std::isfinite(norm)) {
            error = "scene pose produced a degenerate gaussian rotation";
            return;
        }
        for (float& value : rotated) value /= norm;
        // pytorch3d.quaternion_multiply ends with standardize_quaternion: the
        // versor is forced to a nonnegative real part.
        if (rotated[0] < 0.0f) {
            for (float& value : rotated) value = -value;
        }
        for (size_t channel = 0; channel < 4; ++channel) {
            quaternion[channel] = rotated[channel];
        }
        // Official make_scene: adjusted = scale * pose_scale, floored at
        // 1.1 * (min_kernel * pose_scale) after the per-object floor grows.
        for (size_t channel = 0; channel < 3; ++channel) {
            float* scale = &splats.scales[index * 3 + channel];
            *scale = std::max(*scale * pose.scale[channel],
                              1.1f * kMinKernel * pose.scale[channel]);
        }
    }
}

}  // namespace

bool SceneSplatSet::valid() const {
    const size_t count = size();
    return count > 0 && positions.size() == count * 3 &&
           sh0.size() == count * 3 && scales.size() == count * 3 &&
           rotations.size() == count * 4;
}

bool activate_and_pose_object(const SceneObjectInput& object,
                              SceneSplatSet& out,
                              std::string& error) {
    const size_t count = object.splat_count;
    if (count == 0) {
        error = "cannot compose an object with an empty splat set";
        return false;
    }
    if (!object.centers || !object.sh0 || !object.opacity_logit ||
        !object.log_scale || !object.rot_ply || !object.pose) {
        error = "scene object input is missing a required array or pose";
        return false;
    }
    out.positions.resize(count * 3);
    out.sh0.resize(count * 3);
    out.opacities.resize(count);
    out.scales.resize(count * 3);
    out.rotations.resize(count * 4);
    for (size_t vertex = 0; vertex < count; ++vertex) {
        // Activation mirrors the upstream load_gaussian_splat_ply exactly:
        // positions/sh0 pass through, opacity = sigmoid(logit),
        // scales = exp(log_scale), rotations = normalized quaternion.
        std::memcpy(out.positions.data() + vertex * 3,
                    object.centers + vertex * 3, 3 * sizeof(float));
        std::memcpy(out.sh0.data() + vertex * 3, object.sh0 + vertex * 3,
                    3 * sizeof(float));
        out.opacities[vertex] = sigmoid(object.opacity_logit[vertex]);
        for (size_t channel = 0; channel < 3; ++channel) {
            out.scales[vertex * 3 + channel] =
                    std::exp(object.log_scale[vertex * 3 + channel]);
        }
        float* rotation = out.rotations.data() + vertex * 4;
        std::memcpy(rotation, object.rot_ply + vertex * 4, 4 * sizeof(float));
        const float norm = std::sqrt(
                rotation[0] * rotation[0] + rotation[1] * rotation[1] +
                rotation[2] * rotation[2] + rotation[3] * rotation[3]);
        if (!(norm > 0.0f) || !std::isfinite(norm)) {
            error = "scene object contains a zero or non-finite quaternion";
            return false;
        }
        rotation[0] /= norm;
        rotation[1] /= norm;
        rotation[2] /= norm;
        rotation[3] /= norm;
    }
    apply_scene_pose(out, *object.pose, error);
    return error.empty();
}

void append_splat_set(SceneSplatSet& scene, const SceneSplatSet& object) {
    scene.positions.insert(scene.positions.end(), object.positions.begin(),
                           object.positions.end());
    scene.sh0.insert(scene.sh0.end(), object.sh0.begin(), object.sh0.end());
    scene.opacities.insert(scene.opacities.end(), object.opacities.begin(),
                           object.opacities.end());
    scene.scales.insert(scene.scales.end(), object.scales.begin(),
                        object.scales.end());
    scene.rotations.insert(scene.rotations.end(), object.rotations.begin(),
                           object.rotations.end());
}

bool normalize_scene(SceneSplatSet& scene, std::string& error) {
    const size_t count = scene.size();
    if (count == 0) {
        error = "cannot normalize an empty scene";
        return false;
    }
    // Official normalized_gaussian: the opacity > 0.9 active bounds drive a
    // uniform rescale and centering. fix_alignment=False keeps the axes.
    Vec3 lower{std::numeric_limits<float>::infinity(),
               std::numeric_limits<float>::infinity(),
               std::numeric_limits<float>::infinity()};
    Vec3 upper{-lower.x, -lower.y, -lower.z};
    size_t active = 0;
    for (size_t index = 0; index < count; ++index) {
        if (scene.opacities[index] <= 0.9f) continue;
        ++active;
        const float x = scene.positions[index * 3 + 0];
        const float y = scene.positions[index * 3 + 1];
        const float z = scene.positions[index * 3 + 2];
        lower.x = std::min(lower.x, x);
        upper.x = std::max(upper.x, x);
        lower.y = std::min(lower.y, y);
        upper.y = std::max(upper.y, y);
        lower.z = std::min(lower.z, z);
        upper.z = std::max(upper.z, z);
    }
    if (active == 0) {
        error = "scene has no gaussian with opacity > 0.9; cannot normalize";
        return false;
    }
    float inverse_scale = std::max(upper.x - lower.x, upper.y - lower.y);
    inverse_scale = std::max(inverse_scale, upper.z - lower.z);
    if (!(inverse_scale > 0.0f) || !std::isfinite(inverse_scale)) {
        error = "scene active bounds are degenerate; cannot normalize";
        return false;
    }
    const Vec3 center{(lower.x + upper.x) * 0.5f / inverse_scale,
                      (lower.y + upper.y) * 0.5f / inverse_scale,
                      (lower.z + upper.z) * 0.5f / inverse_scale};
    for (size_t index = 0; index < count; ++index) {
        for (size_t axis = 0; axis < 3; ++axis) {
            float* value = &scene.positions[index * 3 + axis];
            const float center_axis =
                    axis == 0 ? center.x : (axis == 1 ? center.y : center.z);
            *value = *value / inverse_scale - center_axis;
        }
        for (size_t channel = 0; channel < 3; ++channel) {
            scene.scales[index * 3 + channel] /= inverse_scale;
        }
    }
    return true;
}

bool compose_scene(const SceneObjectInput* objects,
                   size_t object_count,
                   bool normalize,
                   SceneSplatSet& out,
                   std::string& error) {
    if (!objects || object_count == 0) {
        error = "scene composition requires at least one object";
        return false;
    }
    out = SceneSplatSet{};
    for (size_t index = 0; index < object_count; ++index) {
        SceneSplatSet object;
        if (!activate_and_pose_object(objects[index], object, error)) {
            error = "object " + std::to_string(index) + ": " + error;
            return false;
        }
        append_splat_set(out, object);
    }
    if (normalize && !normalize_scene(out, error)) {
        return false;
    }
    return true;
}

}  // namespace sam3d
