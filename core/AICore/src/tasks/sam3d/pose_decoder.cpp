// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "pose_decoder.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>

namespace sam3d {
namespace {

constexpr std::array<float, 6> kRotation6dMean = {
        -0.06366084883674913f,   0.008438224692279752f,
        0.00017084786438302483f, 0.0007126610473540038f,
        -0.0030916726538816417f, 0.5166093753457688f,
};
constexpr std::array<float, 6> kRotation6dStd = {
        0.6656971967514863f, 0.6787012271867754f,  0.30345010594844524f,
        0.4394504420678794f, 0.39817973931717104f, 0.6176286868761914f,
};

bool all_finite(const float* values, size_t count) {
    for (size_t index = 0; index < count; ++index) {
        if (!std::isfinite(values[index])) return false;
    }
    return true;
}

float dot3(const std::array<float, 3>& left,
           const std::array<float, 3>& right) {
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
}

float norm3(const std::array<float, 3>& value) {
    return std::sqrt(dot3(value, value));
}

std::array<float, 3> normalize_like_torch(const std::array<float, 3>& value) {
    // torch.nn.functional.normalize defaults to eps=1e-12 and therefore
    // returns zero for an exactly-zero vector instead of raising an error.
    const float denominator = std::max(norm3(value), 1.0e-12f);
    return {value[0] / denominator, value[1] / denominator,
            value[2] / denominator};
}

std::array<float, 3> cross3(const std::array<float, 3>& left,
                            const std::array<float, 3>& right) {
    return {
            left[1] * right[2] - left[2] * right[1],
            left[2] * right[0] - left[0] * right[2],
            left[0] * right[1] - left[1] * right[0],
    };
}

std::array<float, 4> matrix_to_quaternion_like_pytorch3d(
        const float matrix[3][3]) {
    // This is pytorch3d.transforms.matrix_to_quaternion, including its
    // candidate selection and 0.1 denominator floor.  Reusing the same
    // branch rule prevents otherwise valid quaternion sign differences from
    // making a parity report ambiguous.
    const float m00 = matrix[0][0], m01 = matrix[0][1], m02 = matrix[0][2];
    const float m10 = matrix[1][0], m11 = matrix[1][1], m12 = matrix[1][2];
    const float m20 = matrix[2][0], m21 = matrix[2][1], m22 = matrix[2][2];
    const std::array<float, 4> q_abs = {
            std::sqrt(std::max(0.0f, 1.0f + m00 + m11 + m22)),
            std::sqrt(std::max(0.0f, 1.0f + m00 - m11 - m22)),
            std::sqrt(std::max(0.0f, 1.0f - m00 + m11 - m22)),
            std::sqrt(std::max(0.0f, 1.0f - m00 - m11 + m22)),
    };
    const std::array<std::array<float, 4>, 4> candidates = {{
            {q_abs[0] * q_abs[0], m21 - m12, m02 - m20, m10 - m01},
            {m21 - m12, q_abs[1] * q_abs[1], m10 + m01, m02 + m20},
            {m02 - m20, m10 + m01, q_abs[2] * q_abs[2], m12 + m21},
            {m10 - m01, m20 + m02, m21 + m12, q_abs[3] * q_abs[3]},
    }};
    size_t best = 0;
    for (size_t index = 1; index < q_abs.size(); ++index) {
        if (q_abs[index] > q_abs[best]) best = index;
    }
    const float denominator = 2.0f * std::max(q_abs[best], 0.1f);
    std::array<float, 4> quaternion{};
    for (size_t index = 0; index < quaternion.size(); ++index) {
        quaternion[index] = candidates[best][index] / denominator;
    }
    // pytorch3d.standardize_quaternion selects a non-negative real part.
    if (quaternion[0] < 0.0f) {
        for (float& value : quaternion) value = -value;
    }
    return quaternion;
}

void quaternion_to_matrix_like_pytorch3d(const std::array<float, 4>& quaternion,
                                         float matrix[3][3]) {
    const float r = quaternion[0], i = quaternion[1], j = quaternion[2],
                k = quaternion[3];
    const float squared_norm = r * r + i * i + j * j + k * k;
    const float two_s = squared_norm == 0.0f
                                ? std::numeric_limits<float>::infinity()
                                : 2.0f / squared_norm;
    matrix[0][0] = 1.0f - two_s * (j * j + k * k);
    matrix[0][1] = two_s * (i * j - k * r);
    matrix[0][2] = two_s * (i * k + j * r);
    matrix[1][0] = two_s * (i * j + k * r);
    matrix[1][1] = 1.0f - two_s * (i * i + k * k);
    matrix[1][2] = two_s * (j * k - i * r);
    matrix[2][0] = two_s * (i * k - j * r);
    matrix[2][1] = two_s * (j * k + i * r);
    matrix[2][2] = 1.0f - two_s * (i * i + j * j);
}

template <size_t N>
void write_array(std::ostream& stream, const std::array<float, N>& values) {
    stream << '[';
    for (size_t index = 0; index < values.size(); ++index) {
        if (index != 0) stream << ',';
        stream << values[index];
    }
    stream << ']';
}

}  // namespace

bool decode_scale_shift_invariant_pose(const float normalized_rotation_6d[6],
                                       const float log_scale[3],
                                       const float translation[3],
                                       float log_translation_scale,
                                       const float scene_scale[3],
                                       const float scene_shift[3],
                                       NativeInstancePose& output,
                                       std::string& error,
                                       int downsample_factor) {
    if (!normalized_rotation_6d || !log_scale || !translation || !scene_scale ||
        !scene_shift || !all_finite(normalized_rotation_6d, 6) ||
        !all_finite(log_scale, 3) || !all_finite(translation, 3) ||
        !std::isfinite(log_translation_scale) || !all_finite(scene_scale, 3) ||
        !all_finite(scene_shift, 3)) {
        error = "ScaleShiftInvariant pose inputs must be finite";
        return false;
    }

    std::array<float, 6> rotation_6d{};
    for (size_t index = 0; index < rotation_6d.size(); ++index) {
        rotation_6d[index] =
                normalized_rotation_6d[index] * kRotation6dStd[index] +
                kRotation6dMean[index];
    }
    const std::array<float, 3> a1 = {rotation_6d[0], rotation_6d[1],
                                     rotation_6d[2]};
    const std::array<float, 3> a2 = {rotation_6d[3], rotation_6d[4],
                                     rotation_6d[5]};
    const std::array<float, 3> b1 = normalize_like_torch(a1);
    std::array<float, 3> b2{};
    const float projection = dot3(b1, a2);
    for (size_t index = 0; index < b2.size(); ++index)
        b2[index] = a2[index] - projection * b1[index];
    b2 = normalize_like_torch(b2);
    const std::array<float, 3> b3 = cross3(b1, b2);
    const float rotation_matrix[3][3] = {
            {b1[0], b2[0], b3[0]},
            {b1[1], b2[1], b3[1]},
            {b1[2], b2[2], b3[2]},
    };

    const std::array<float, 4> decoded_rotation =
            matrix_to_quaternion_like_pytorch3d(rotation_matrix);
    float decoded_rotation_matrix[3][3]{};
    quaternion_to_matrix_like_pytorch3d(decoded_rotation,
                                        decoded_rotation_matrix);
    output.translation_scale = std::exp(log_translation_scale);
    output.scene_scale = {scene_scale[0], scene_scale[1], scene_scale[2]};
    output.scene_shift = {scene_shift[0], scene_shift[1], scene_shift[2]};
    float metric_rotation_matrix[3][3]{};
    for (size_t index = 0; index < output.scale.size(); ++index) {
        // Transform3d composes row-vector matrices in this exact order:
        // Scale(instance) * Rotate(instance) * Translate(instance) *
        // Scale(scene) * Translate(scene).  `decompose_transform` derives
        // scale from the row norms, rather than assuming scene scale is
        // isotropic (ObjectCentricSSI currently emits isotropic values).
        const float instance_scale = std::exp(log_scale[index]);
        float squared_row_norm = 0.0f;
        for (size_t column = 0; column < 3; ++column) {
            metric_rotation_matrix[index][column] =
                    instance_scale * decoded_rotation_matrix[index][column] *
                    scene_scale[column];
            squared_row_norm += metric_rotation_matrix[index][column] *
                                metric_rotation_matrix[index][column];
        }
        output.scale[index] = std::sqrt(squared_row_norm);
        output.translation[index] =
                translation[index] * scene_scale[index] + scene_shift[index];
    }
    float decomposed_rotation_matrix[3][3]{};
    for (size_t row = 0; row < 3; ++row) {
        const float denominator = output.scale[row];
        for (size_t column = 0; column < 3; ++column) {
            decomposed_rotation_matrix[row][column] =
                    denominator == 0.0f
                            ? 0.0f
                            : metric_rotation_matrix[row][column] / denominator;
        }
    }
    output.rotation_wxyz =
            matrix_to_quaternion_like_pytorch3d(decomposed_rotation_matrix);
    // Official post-decode rescale (inference_pipeline.py scale *=
    // downsample_factor): applied after the rotation decomposition so the
    // decomposed rotation stays untouched, exactly like the official order.
    if (downsample_factor > 1) {
        for (float& value : output.scale)
            value *= static_cast<float>(downsample_factor);
    }
    if (!all_finite(output.rotation_wxyz.data(), output.rotation_wxyz.size()) ||
        !all_finite(output.scale.data(), output.scale.size()) ||
        !all_finite(output.translation.data(), output.translation.size()) ||
        !std::isfinite(output.translation_scale)) {
        error = "ScaleShiftInvariant pose decode produced non-finite values";
        return false;
    }
    return true;
}

bool write_native_pose_json(const std::string& path,
                            const NativeInstancePose& pose,
                            std::string& error) {
    std::error_code filesystem_error;
    const std::filesystem::path output_path(path);
    if (!output_path.parent_path().empty()) {
        std::filesystem::create_directories(output_path.parent_path(),
                                            filesystem_error);
        if (filesystem_error) {
            error = "cannot create pose output directory: " +
                    filesystem_error.message();
            return false;
        }
    }
    std::ofstream stream(output_path, std::ios::binary | std::ios::trunc);
    if (!stream) {
        error = "cannot open pose output: " + path;
        return false;
    }
    stream << std::setprecision(std::numeric_limits<float>::max_digits10);
    // Top level follows the official pipeline pose receipt (batch-nested
    // rotation/translation/scale with the per-axis scale collapsed to its
    // mean, exactly like the pose_decoder wrapper in inference_utils.py) so
    // run_ggml.sh and run_python.sh pose receipts are interchangeable. The
    // raw native decoder fields, including the per-axis scale, remain under
    // "native" for the parity verifiers.
    const float uniform_scale =
            (pose.scale[0] + pose.scale[1] + pose.scale[2]) / 3.0f;
    stream << "{\n  \"schema\": \"sam3d.native-pose.v2\",\n"
           << "  \"rotation\": [";
    write_array(stream, pose.rotation_wxyz);
    stream << "],\n  \"translation\": [";
    write_array(stream, pose.translation);
    stream << "],\n  \"scale\": [[" << uniform_scale << ',' << uniform_scale
           << ',' << uniform_scale << "]],\n"
           << "  \"native\": {\n"
           << "    \"convention\": \"ScaleShiftInvariant\",\n"
           << "    \"rotation_wxyz\": ";
    write_array(stream, pose.rotation_wxyz);
    stream << ",\n    \"translation\": ";
    write_array(stream, pose.translation);
    stream << ",\n    \"scale\": ";
    write_array(stream, pose.scale);
    stream << ",\n    \"translation_scale\": " << pose.translation_scale
           << ",\n    \"scene_scale\": ";
    write_array(stream, pose.scene_scale);
    stream << ",\n    \"scene_shift\": ";
    write_array(stream, pose.scene_shift);
    stream << "\n  }\n}\n";
    if (!stream) {
        error = "failed while writing pose output: " + path;
        return false;
    }
    return true;
}

}  // namespace sam3d
