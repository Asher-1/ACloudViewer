// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "gaussian_renderer.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <unordered_map>

namespace sam3d {
namespace {

constexpr float kPi = 3.14159265358979323846f;

struct Vec3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}

float dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vec3 normalize(const Vec3& value) {
    const float norm = std::sqrt(dot(value, value));
    return norm > 0.0f ? Vec3{value.x / norm, value.y / norm, value.z / norm}
                       : Vec3{};
}

float radical_inverse_base_two(int value) {
    float inverse = 0.5f;
    float result = 0.0f;
    while (value > 0) {
        result += static_cast<float>(value & 1) * inverse;
        value >>= 1;
        inverse *= 0.5f;
    }
    return result;
}

float sigmoid(float value) {
    if (value >= 0.0f) {
        const float exp_neg = std::exp(-value);
        return 1.0f / (1.0f + exp_neg);
    }
    const float exp_pos = std::exp(value);
    return exp_pos / (1.0f + exp_pos);
}

bool get_property(const std::unordered_map<std::string, size_t>& properties,
                  const char* name,
                  size_t& output,
                  std::string& error) {
    const auto found = properties.find(name);
    if (found == properties.end()) {
        error = std::string("Gaussian PLY is missing property '") + name + "'";
        return false;
    }
    output = found->second;
    return true;
}

}  // namespace

bool GaussianSplatSet::valid() const {
    const size_t count = size();
    return count > 0 && positions.size() == count * 3 &&
           sh0.size() == count * 3 && scales.size() == count * 3 &&
           rotations.size() == count * 4;
}

bool load_gaussian_splat_ply(const std::string& path,
                             GaussianSplatSet& splats,
                             std::string& error) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        error = "cannot open Gaussian PLY '" + path + "'";
        return false;
    }

    std::string line;
    if (!std::getline(input, line) || line != "ply" ||
        !std::getline(input, line) ||
        line != "format binary_little_endian 1.0") {
        error = "Gaussian renderer requires a binary_little_endian PLY";
        return false;
    }

    size_t vertex_count = 0;
    bool in_vertex_element = false;
    std::vector<std::string> property_names;
    while (std::getline(input, line)) {
        if (line == "end_header") break;
        std::istringstream fields(line);
        std::string token;
        fields >> token;
        if (token == "element") {
            std::string name;
            fields >> name;
            in_vertex_element = name == "vertex";
            if (in_vertex_element) fields >> vertex_count;
        } else if (token == "property" && in_vertex_element) {
            std::string type;
            std::string name;
            fields >> type >> name;
            if (type != "float") {
                error = "Gaussian PLY vertex properties must be float";
                return false;
            }
            property_names.push_back(name);
        }
    }
    if (line != "end_header" || vertex_count == 0 || property_names.empty()) {
        error = "Gaussian PLY has no valid vertex payload";
        return false;
    }
    if (vertex_count > std::numeric_limits<size_t>::max() /
                               property_names.size() / sizeof(float)) {
        error = "Gaussian PLY vertex payload overflows host address space";
        return false;
    }

    std::unordered_map<std::string, size_t> properties;
    for (size_t index = 0; index < property_names.size(); ++index) {
        properties.emplace(property_names[index], index);
    }
    size_t x = 0, y = 0, z = 0, fdc0 = 0, fdc1 = 0, fdc2 = 0, opacity = 0;
    size_t scale0 = 0, scale1 = 0, scale2 = 0, rot0 = 0, rot1 = 0, rot2 = 0,
           rot3 = 0;
    const char* required[] = {"x",       "y",       "z",       "f_dc_0",
                              "f_dc_1",  "f_dc_2",  "opacity", "scale_0",
                              "scale_1", "scale_2", "rot_0",   "rot_1",
                              "rot_2",   "rot_3"};
    size_t* destinations[] = {&x,    &y,       &z,      &fdc0,   &fdc1,
                              &fdc2, &opacity, &scale0, &scale1, &scale2,
                              &rot0, &rot1,    &rot2,   &rot3};
    for (size_t index = 0; index < sizeof(required) / sizeof(required[0]);
         ++index) {
        if (!get_property(properties, required[index], *destinations[index],
                          error))
            return false;
    }

    std::vector<float> row(property_names.size());
    GaussianSplatSet decoded;
    decoded.positions.resize(vertex_count * 3);
    decoded.sh0.resize(vertex_count * 3);
    decoded.opacities.resize(vertex_count);
    decoded.scales.resize(vertex_count * 3);
    decoded.rotations.resize(vertex_count * 4);
    for (size_t vertex = 0; vertex < vertex_count; ++vertex) {
        if (!input.read(
                    reinterpret_cast<char*>(row.data()),
                    static_cast<std::streamsize>(row.size() * sizeof(float)))) {
            error = "Gaussian PLY ended before all vertex records";
            return false;
        }
        const float values[] = {
                row[x],    row[y],       row[z],      row[fdc0],   row[fdc1],
                row[fdc2], row[opacity], row[scale0], row[scale1], row[scale2],
                row[rot0], row[rot1],    row[rot2],   row[rot3]};
        for (size_t index = 0; index < sizeof(values) / sizeof(values[0]);
             ++index) {
            const float value = values[index];
            // Gaussian.load_ply applies torch.sigmoid to serialized opacity
            // logits. Its defined limits preserve +/-infinity as 1/0.
            if (std::isnan(value) || (!std::isfinite(value) && index != 6)) {
                error = "Gaussian PLY contains a non-finite non-opacity "
                        "property";
                return false;
            }
        }
        std::memcpy(decoded.positions.data() + vertex * 3, values,
                    3 * sizeof(float));
        std::memcpy(decoded.sh0.data() + vertex * 3, values + 3,
                    3 * sizeof(float));
        decoded.opacities[vertex] = sigmoid(values[6]);
        for (size_t channel = 0; channel < 3; ++channel) {
            decoded.scales[vertex * 3 + channel] =
                    std::exp(values[7 + channel]);
        }
        float* rotation = decoded.rotations.data() + vertex * 4;
        std::memcpy(rotation, values + 10, 4 * sizeof(float));
        const float norm = std::sqrt(
                rotation[0] * rotation[0] + rotation[1] * rotation[1] +
                rotation[2] * rotation[2] + rotation[3] * rotation[3]);
        if (!(norm > 0.0f) || !std::isfinite(norm)) {
            error = "Gaussian PLY contains a zero or non-finite quaternion";
            return false;
        }
        rotation[0] /= norm;
        rotation[1] /= norm;
        rotation[2] /= norm;
        rotation[3] /= norm;
    }
    splats = std::move(decoded);
    return true;
}

std::vector<GaussianCamera> make_gaussian_hammersley_cameras(
        int view_count,
        const GaussianRenderConfig& config,
        std::string& error) {
    if (view_count <= 0 || config.width <= 0 || config.height <= 0 ||
        !(config.radius > 0.0f) || !(config.fov_degrees > 0.0f) ||
        !(config.near_plane > 0.0f) ||
        !(config.far_plane > config.near_plane)) {
        error = "invalid Gaussian camera configuration";
        return {};
    }
    const float fov = config.fov_degrees * kPi / 180.0f;
    const float focal = 0.5f / std::tan(fov * 0.5f);
    std::vector<GaussianCamera> cameras;
    cameras.reserve(static_cast<size_t>(view_count));
    for (int index = 0; index < view_count; ++index) {
        const float u =
                static_cast<float>(index) / static_cast<float>(view_count);
        const float v = radical_inverse_base_two(index);
        const float pitch = std::acos(1.0f - 2.0f * u) - kPi * 0.5f;
        const float yaw = v * 2.0f * kPi;
        const Vec3 eye{std::sin(yaw) * std::cos(pitch) * config.radius,
                       std::cos(yaw) * std::cos(pitch) * config.radius,
                       std::sin(pitch) * config.radius};
        const Vec3 z_axis = normalize({-eye.x, -eye.y, -eye.z});
        const Vec3 x_axis = normalize(cross({0.0f, 0.0f, -1.0f}, z_axis));
        const Vec3 y_axis = normalize(cross(z_axis, x_axis));
        if (dot(x_axis, x_axis) == 0.0f || dot(y_axis, y_axis) == 0.0f) {
            error = "Hammersley camera coincides with the up axis";
            return {};
        }
        GaussianCamera camera;
        camera.extrinsics = {x_axis.x, x_axis.y, x_axis.z, -dot(x_axis, eye),
                             y_axis.x, y_axis.y, y_axis.z, -dot(y_axis, eye),
                             z_axis.x, z_axis.y, z_axis.z, -dot(z_axis, eye),
                             0.0f,     0.0f,     0.0f,     1.0f};
        camera.intrinsics = {focal, 0.0f, 0.5f, 0.0f, focal,
                             0.5f,  0.0f, 0.0f, 1.0f};
        cameras.push_back(camera);
    }
    return cameras;
}

}  // namespace sam3d
