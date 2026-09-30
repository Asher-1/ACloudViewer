// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "mesh_rasterizer.hpp"

#if defined(SAM3D_USE_NVDIFFRAST_CUDARASTER)
#include "mesh_rasterizer_cuda.hpp"
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace sam3d {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr float kEpsilon = 1.0e-7f;

struct ProjectedVertex {
    float x = 0.0f;
    float y = 0.0f;
    float inverse_depth = 0.0f;
    float depth = std::numeric_limits<float>::infinity();
};

double radical_inverse_base_two(int value) {
    double inverse = 0.5;
    double result = 0.0;
    while (value > 0) {
        result += static_cast<double>(value & 1) * inverse;
        value >>= 1;
        inverse *= 0.5;
    }
    return result;
}

bool valid_mesh(const NativeMesh& mesh, std::string& error) {
    if (mesh.positions.empty() || mesh.positions.size() % 3 != 0 ||
        mesh.indices.empty() || mesh.indices.size() % 3 != 0) {
        error = "mesh rasterizer requires non-empty triangle positions and "
                "indices";
        return false;
    }
    const size_t vertex_count = mesh.positions.size() / 3;
    for (uint32_t index : mesh.indices) {
        if (index >= vertex_count) {
            error = "mesh rasterizer received an out-of-range index";
            return false;
        }
    }
    return true;
}

bool project_vertex(const float* position,
                    const MeshRasterCamera& camera,
                    const MeshRasterConfig& config,
                    ProjectedVertex& output) {
    const float x = camera.extrinsics[0] * position[0] +
                    camera.extrinsics[1] * position[1] +
                    camera.extrinsics[2] * position[2] + camera.extrinsics[3];
    const float y = camera.extrinsics[4] * position[0] +
                    camera.extrinsics[5] * position[1] +
                    camera.extrinsics[6] * position[2] + camera.extrinsics[7];
    const float z = camera.extrinsics[8] * position[0] +
                    camera.extrinsics[9] * position[1] +
                    camera.extrinsics[10] * position[2] + camera.extrinsics[11];
    if (!(z >= config.near_plane && z <= config.far_plane) || !std::isfinite(z))
        return false;

    const float normalized_x =
            camera.intrinsics[0] * x / z + camera.intrinsics[2];
    const float normalized_y =
            camera.intrinsics[4] * y / z + camera.intrinsics[5];
    if (!std::isfinite(normalized_x) || !std::isfinite(normalized_y))
        return false;
    output.x = normalized_x * static_cast<float>(config.width);
    // nvdiffrast's raster buffer is bottom-left origin before the utility's
    // explicit vertical flip. Keep that storage convention here.
    output.y = (1.0f - normalized_y) * static_cast<float>(config.height);
    output.inverse_depth = 1.0f / z;
    output.depth = z;
    return true;
}

float edge_function(
        float ax, float ay, float bx, float by, float px, float py) {
    return (px - ax) * (by - ay) - (py - ay) * (bx - ax);
}

}  // namespace

std::vector<MeshRasterCamera> make_mesh_hammersley_cameras(
        int view_count, const MeshRasterConfig& config, std::string& error) {
    if (view_count <= 0 || config.width <= 0 || config.height <= 0 ||
        !(config.near_plane > 0.0f) ||
        !(config.far_plane > config.near_plane)) {
        error = "invalid mesh raster camera configuration";
        return {};
    }
    const float fov_radians = static_cast<float>(40.0 * kPi / 180.0);
    const float focal = 0.5f / std::tan(fov_radians * 0.5f);
    constexpr float radius = 2.0f;
    std::vector<MeshRasterCamera> cameras;
    cameras.reserve(static_cast<size_t>(view_count));
    for (int index = 0; index < view_count; ++index) {
        // sphere_hammersley_sequence() uses NumPy F64 arithmetic. Its Python
        // floats are converted once to CUDA F32 in render_utils, so preserve
        // the same double-then-F32 boundary rather than rounding each term.
        const double u =
                static_cast<double>(index) / static_cast<double>(view_count);
        const float yaw =
                static_cast<float>(radical_inverse_base_two(index) * 2.0 * kPi);
        const float pitch =
                static_cast<float>(std::acos(1.0 - 2.0 * u) - kPi * 0.5);
        const float eye_x = std::sin(yaw) * std::cos(pitch) * radius;
        const float eye_y = std::cos(yaw) * std::cos(pitch) * radius;
        const float eye_z = std::sin(pitch) * radius;
        const float z_x = -eye_x / radius;
        const float z_y = -eye_y / radius;
        const float z_z = -eye_z / radius;
        // Match utils3d.torch.extrinsics_look_at(): x = cross(-up, z), where
        // the official renderer passes up = (0, 0, 1).
        const float x_x = z_y;
        const float x_y = -z_x;
        const float x_length = std::sqrt(x_x * x_x + x_y * x_y);
        // The source implementation normalizes the floating-point result at
        // the Hammersley pole.  Reject only an exact zero so this native path
        // preserves that convention instead of introducing a different view.
        if (!(x_length > 0.0f)) {
            error = "Hammersley camera coincides with its up axis";
            return {};
        }
        const float right_x = x_x / x_length;
        const float right_y = x_y / x_length;
        float up_x = z_y * 0.0f - z_z * right_y;
        float up_y = z_z * right_x - z_x * 0.0f;
        float up_z = z_x * right_y - z_y * right_x;
        const float up_length =
                std::sqrt(up_x * up_x + up_y * up_y + up_z * up_z);
        if (!(up_length > 0.0f)) {
            error = "Hammersley camera produced a zero up vector";
            return {};
        }
        up_x /= up_length;
        up_y /= up_length;
        up_z /= up_length;

        MeshRasterCamera camera;
        camera.extrinsics = {
                right_x, right_y,
                0.0f,    -(right_x * eye_x + right_y * eye_y),
                up_x,    up_y,
                up_z,    -(up_x * eye_x + up_y * eye_y + up_z * eye_z),
                z_x,     z_y,
                z_z,     -(z_x * eye_x + z_y * eye_y + z_z * eye_z),
                0.0f,    0.0f,
                0.0f,    1.0f};
        camera.intrinsics = {focal, 0.0f, 0.5f, 0.0f, focal,
                             0.5f,  0.0f, 0.0f, 1.0f};
        // _fill_holes() is not the texture-baking camera path: it constructs
        // this OpenGL view directly and uses near=1/far=3. Preserve those
        // exact conventions instead of round-tripping through intrinsics.
        camera.view = {right_x, right_y, 0.0f,  camera.extrinsics[3],
                       -up_x,   -up_y,   -up_z, -camera.extrinsics[7],
                       -z_x,    -z_y,    -z_z,  -camera.extrinsics[11],
                       0.0f,    0.0f,    0.0f,  1.0f};
        const float visibility_focal = 2.0f * focal;
        camera.projection = {visibility_focal,
                             0.0f,
                             0.0f,
                             0.0f,
                             0.0f,
                             visibility_focal,
                             0.0f,
                             0.0f,
                             0.0f,
                             0.0f,
                             -2.0f,
                             -3.0f,
                             0.0f,
                             0.0f,
                             -1.0f,
                             0.0f};
        camera.has_view_projection = true;
        cameras.push_back(camera);
    }
    return cameras;
}

bool rasterize_mesh_uv(const NativeMesh& mesh,
                       const MeshRasterCamera& camera,
                       const MeshRasterConfig& config,
                       MeshRasterResult& result,
                       std::string& error) {
    if (!valid_mesh(mesh, error)) return false;
    if (config.width <= 0 || config.height <= 0 ||
        !(config.near_plane > 0.0f) ||
        !(config.far_plane > config.near_plane)) {
        error = "invalid mesh raster dimensions or depth range";
        return false;
    }
    const size_t vertex_count = mesh.positions.size() / 3;
    if (mesh.texcoords.size() != vertex_count * 2) {
        error = "mesh UV rasterization requires one UV pair per vertex";
        return false;
    }
    const size_t pixel_count =
            static_cast<size_t>(config.width) * config.height;
    result.uv.assign(pixel_count * 2, 0.0f);
    result.face_ids.assign(pixel_count, std::numeric_limits<uint32_t>::max());
    result.coverage.assign(pixel_count, 0);
    std::vector<float> depth(pixel_count,
                             std::numeric_limits<float>::infinity());
    const size_t face_count = mesh.indices.size() / 3;
    for (size_t face = 0; face < face_count; ++face) {
        const uint32_t* indices = mesh.indices.data() + face * 3;
        ProjectedVertex vertices[3];
        if (!project_vertex(
                    mesh.positions.data() + static_cast<size_t>(indices[0]) * 3,
                    camera, config, vertices[0]) ||
            !project_vertex(
                    mesh.positions.data() + static_cast<size_t>(indices[1]) * 3,
                    camera, config, vertices[1]) ||
            !project_vertex(
                    mesh.positions.data() + static_cast<size_t>(indices[2]) * 3,
                    camera, config, vertices[2])) {
            continue;
        }
        const float signed_area =
                edge_function(vertices[0].x, vertices[0].y, vertices[1].x,
                              vertices[1].y, vertices[2].x, vertices[2].y);
        if (std::fabs(signed_area) <= kEpsilon) continue;
        const float min_x =
                std::min({vertices[0].x, vertices[1].x, vertices[2].x});
        const float max_x =
                std::max({vertices[0].x, vertices[1].x, vertices[2].x});
        const float min_y =
                std::min({vertices[0].y, vertices[1].y, vertices[2].y});
        const float max_y =
                std::max({vertices[0].y, vertices[1].y, vertices[2].y});
        const int begin_x = std::max(0, static_cast<int>(std::floor(min_x)));
        const int end_x = std::min(config.width - 1,
                                   static_cast<int>(std::ceil(max_x)) - 1);
        const int begin_y = std::max(0, static_cast<int>(std::floor(min_y)));
        const int end_y = std::min(config.height - 1,
                                   static_cast<int>(std::ceil(max_y)) - 1);
        if (begin_x > end_x || begin_y > end_y) continue;
        for (int y = begin_y; y <= end_y; ++y) {
            for (int x = begin_x; x <= end_x; ++x) {
                const float sample_x = static_cast<float>(x) + 0.5f;
                const float sample_y = static_cast<float>(y) + 0.5f;
                const float b0 = edge_function(vertices[1].x, vertices[1].y,
                                               vertices[2].x, vertices[2].y,
                                               sample_x, sample_y) /
                                 signed_area;
                const float b1 = edge_function(vertices[2].x, vertices[2].y,
                                               vertices[0].x, vertices[0].y,
                                               sample_x, sample_y) /
                                 signed_area;
                const float b2 = 1.0f - b0 - b1;
                if (b0 < -kEpsilon || b1 < -kEpsilon || b2 < -kEpsilon)
                    continue;
                const float inverse_depth = b0 * vertices[0].inverse_depth +
                                            b1 * vertices[1].inverse_depth +
                                            b2 * vertices[2].inverse_depth;
                if (!(inverse_depth > 0.0f)) continue;
                const float sample_depth = 1.0f / inverse_depth;
                const size_t pixel = static_cast<size_t>(y) * config.width + x;
                if (sample_depth >= depth[pixel]) continue;
                const float p0 = b0 * vertices[0].inverse_depth / inverse_depth;
                const float p1 = b1 * vertices[1].inverse_depth / inverse_depth;
                const float p2 = b2 * vertices[2].inverse_depth / inverse_depth;
                result.uv[pixel * 2] =
                        p0 * mesh.texcoords[static_cast<size_t>(indices[0]) *
                                            2] +
                        p1 * mesh.texcoords[static_cast<size_t>(indices[1]) *
                                            2] +
                        p2 * mesh.texcoords[static_cast<size_t>(indices[2]) *
                                            2];
                result.uv[pixel * 2 + 1] =
                        p0 * mesh.texcoords[static_cast<size_t>(indices[0]) *
                                                    2 +
                                            1] +
                        p1 * mesh.texcoords[static_cast<size_t>(indices[1]) *
                                                    2 +
                                            1] +
                        p2 * mesh.texcoords[static_cast<size_t>(indices[2]) *
                                                    2 +
                                            1];
                result.face_ids[pixel] = static_cast<uint32_t>(face);
                result.coverage[pixel] = 1;
                depth[pixel] = sample_depth;
            }
        }
    }
    return true;
}

bool mesh_visibility_frequency(const NativeMesh& mesh,
                               int view_count,
                               const MeshRasterConfig& config,
                               std::vector<float>& frequency,
                               std::string& error) {
    if (!valid_mesh(mesh, error)) return false;
    if (view_count <= 0) {
        error = "mesh visibility requires a positive view count";
        return false;
    }
#if defined(SAM3D_USE_NVDIFFRAST_CUDARASTER)
    return mesh_visibility_hammersley_frequency_nvdiffrast_cuda(
            mesh, view_count, config, frequency, error);
#else
    const std::vector<MeshRasterCamera> cameras =
            make_mesh_hammersley_cameras(view_count, config, error);
    if (cameras.empty()) return false;
    return mesh_visibility_frequency(mesh, cameras, config, frequency, error);
#endif
}

bool mesh_visibility_frequency(const NativeMesh& mesh,
                               const std::vector<MeshRasterCamera>& cameras,
                               const MeshRasterConfig& config,
                               std::vector<float>& frequency,
                               std::string& error) {
    if (!valid_mesh(mesh, error)) return false;
    if (cameras.empty()) {
        error = "mesh visibility requires at least one camera";
        return false;
    }
    // Visibility needs only face IDs. Supply a valid temporary UV field so the
    // shared rasterizer remains the sole depth/coverage implementation.
    NativeMesh raster_mesh = mesh;
    raster_mesh.texcoords.assign((mesh.positions.size() / 3) * 2, 0.0f);
#if defined(SAM3D_USE_NVDIFFRAST_CUDARASTER)
    return mesh_visibility_frequency_nvdiffrast_cuda(mesh, cameras, config,
                                                     frequency, error);
#else
    const size_t face_count = mesh.indices.size() / 3;
    std::vector<uint32_t> hits(face_count, 0);
    std::vector<uint8_t> visible(face_count);
    for (const MeshRasterCamera& camera : cameras) {
        MeshRasterResult raster;
        if (!rasterize_mesh_uv(raster_mesh, camera, config, raster, error))
            return false;
        std::fill(visible.begin(), visible.end(), 0);
        for (uint32_t face : raster.face_ids) {
            if (face != std::numeric_limits<uint32_t>::max()) visible[face] = 1;
        }
        for (size_t face = 0; face < face_count; ++face)
            hits[face] += visible[face];
    }
    frequency.resize(face_count);
    const float reciprocal_views = 1.0f / static_cast<float>(cameras.size());
    for (size_t face = 0; face < face_count; ++face) {
        frequency[face] = static_cast<float>(hits[face]) * reciprocal_views;
    }
    return true;
#endif
}

}  // namespace sam3d
