// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "native_pbr_pipeline.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>
#include <vector>

#include "mesh_postprocess.hpp"
#include "mesh_rasterizer.hpp"
#include "mesh_rasterizer_cuda.hpp"
#include "mesh_uv.hpp"
#include "texture_baker.hpp"

namespace sam3d {

namespace {

bool valid_mesh(const NativeMesh& mesh) {
    if (mesh.positions.empty() || mesh.positions.size() % 3 != 0 ||
        mesh.indices.empty() || mesh.indices.size() % 3 != 0) {
        return false;
    }
    const size_t vertex_count = mesh.positions.size() / 3;
    return std::all_of(
            mesh.indices.begin(), mesh.indices.end(),
            [vertex_count](uint32_t index) { return index < vertex_count; });
}

void rotate_zup_mesh_to_yup(NativeMesh& mesh) {
    // postprocessing_utils.to_glb uses the row-vector transform
    // (x, y, z) -> (x, z, -y).
    for (size_t vertex = 0; vertex < mesh.positions.size(); vertex += 3) {
        const float y = mesh.positions[vertex + 1];
        mesh.positions[vertex + 1] = mesh.positions[vertex + 2];
        mesh.positions[vertex + 2] = -y;
    }
    // Normals are vectors, so apply the same orthonormal transform.  Leaving
    // them in the source Z-up frame makes the exported GLB shade as if its
    // geometry had not been rotated, which diverges from trimesh.to_glb().
    for (size_t vertex = 0; vertex < mesh.normals.size(); vertex += 3) {
        const float y = mesh.normals[vertex + 1];
        mesh.normals[vertex + 1] = mesh.normals[vertex + 2];
        mesh.normals[vertex + 2] = -y;
    }
}

}  // namespace

void compute_official_angle_weighted_vertex_normals(NativeMesh& mesh) {
    const size_t vertex_count = mesh.positions.size() / 3;
    mesh.normals.assign(vertex_count * 3, 0.0f);
    std::vector<std::array<double, 3>> accumulated(vertex_count,
                                                   {0.0, 0.0, 0.0});

    const auto position_at = [&mesh](uint32_t index) {
        const float* position = &mesh.positions[static_cast<size_t>(index) * 3];
        return std::array<double, 3>{position[0], position[1], position[2]};
    };
    const auto subtract = [](const std::array<double, 3>& left,
                             const std::array<double, 3>& right) {
        return std::array<double, 3>{left[0] - right[0], left[1] - right[1],
                                     left[2] - right[2]};
    };
    const auto cross = [](const std::array<double, 3>& left,
                          const std::array<double, 3>& right) {
        return std::array<double, 3>{
                left[1] * right[2] - left[2] * right[1],
                left[2] * right[0] - left[0] * right[2],
                left[0] * right[1] - left[1] * right[0],
        };
    };
    const auto dot = [](const std::array<double, 3>& left,
                        const std::array<double, 3>& right) {
        return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
    };
    const auto length = [&dot](const std::array<double, 3>& value) {
        return std::sqrt(dot(value, value));
    };
    const auto corner_angle = [&cross, &dot, &length](
                                      const std::array<double, 3>& left,
                                      const std::array<double, 3>& right) {
        const double left_length = length(left);
        const double right_length = length(right);
        if (!(left_length > 0.0) || !(right_length > 0.0) ||
            !std::isfinite(left_length) || !std::isfinite(right_length)) {
            return 0.0;
        }
        return std::atan2(length(cross(left, right)), dot(left, right));
    };

    for (size_t face = 0; face < mesh.indices.size(); face += 3) {
        const uint32_t i0 = mesh.indices[face + 0];
        const uint32_t i1 = mesh.indices[face + 1];
        const uint32_t i2 = mesh.indices[face + 2];
        if (i0 >= vertex_count || i1 >= vertex_count || i2 >= vertex_count) {
            continue;
        }
        const std::array<double, 3> p0 = position_at(i0);
        const std::array<double, 3> p1 = position_at(i1);
        const std::array<double, 3> p2 = position_at(i2);
        const std::array<double, 3> face_normal =
                cross(subtract(p1, p0), subtract(p2, p0));
        const double face_normal_length = length(face_normal);
        if (!(face_normal_length > 0.0) || !std::isfinite(face_normal_length)) {
            continue;
        }
        const std::array<double, 3> unit_face_normal = {
                face_normal[0] / face_normal_length,
                face_normal[1] / face_normal_length,
                face_normal[2] / face_normal_length,
        };
        const std::array<double, 3> angles = {
                corner_angle(subtract(p1, p0), subtract(p2, p0)),
                corner_angle(subtract(p2, p1), subtract(p0, p1)),
                corner_angle(subtract(p0, p2), subtract(p1, p2)),
        };
        for (const auto [index, angle] :
             {std::pair{i0, angles[0]}, std::pair{i1, angles[1]},
              std::pair{i2, angles[2]}}) {
            for (size_t component = 0; component < 3; ++component) {
                accumulated[index][component] +=
                        angle * unit_face_normal[component];
            }
        }
    }
    for (size_t vertex = 0; vertex < vertex_count; ++vertex) {
        float* normal = &mesh.normals[vertex * 3];
        const double normal_length = length(accumulated[vertex]);
        if (normal_length > 0.0 && std::isfinite(normal_length)) {
            normal[0] =
                    static_cast<float>(accumulated[vertex][0] / normal_length);
            normal[1] =
                    static_cast<float>(accumulated[vertex][1] / normal_length);
            normal[2] =
                    static_cast<float>(accumulated[vertex][2] / normal_length);
        } else {
            normal[0] = 0.0f;
            normal[1] = 0.0f;
            normal[2] = 0.0f;
        }
    }
}

bool assemble_official_pbr_cuda(const NativeMesh& source_mesh,
                                const GaussianSplatSet& splats,
                                const NativePbrPipelineConfig& config,
                                NativePbrPipelineResult& result,
                                std::string& error) {
#if !defined(SAM3D_USE_NVDIFFRAST_CUDARASTER)
    (void)source_mesh;
    (void)splats;
    (void)config;
    (void)result;
    error = "native PBR assembly requires CUDA with "
            "SAM3D_GGML_NVDIFFRAST_NONCOMMERCIAL=ON";
    return false;
#else
    if (!valid_mesh(source_mesh)) {
        error = "native PBR assembly requires a non-empty indexed Z-up mesh";
        return false;
    }
    if (!splats.valid()) {
        error = "native PBR assembly requires a valid Gaussian representation";
        return false;
    }
    if (config.visibility_views <= 0 || config.visibility_resolution <= 0 ||
        config.render_views <= 0 || config.render_resolution <= 0 ||
        config.texture_size <= 0 || config.texture_steps <= 0) {
        error = "native PBR assembly received a non-positive stage "
                "configuration";
        return false;
    }

    NativeMesh mesh = source_mesh;
    mesh.texcoords.clear();
    mesh.colors.clear();
    mesh.normals.clear();
    mesh.vertex_attributes.clear();

    if (!config.input_mesh_already_clean) {
        MeshSimplifyOptions simplify;
        simplify.target_reduction = 0.95f;
        if (!simplify_mesh_official_vtk(mesh, simplify, error)) return false;

        MeshRasterConfig visibility_raster;
        visibility_raster.width = config.visibility_resolution;
        visibility_raster.height = config.visibility_resolution;
        std::vector<float> visibility;
        if (!mesh_visibility_hammersley_frequency_nvdiffrast_cuda(
                    mesh, config.visibility_views, visibility_raster,
                    visibility, error)) {
            return false;
        }
        MeshVisibilityFilterStats filter_stats;
        if (!filter_invisible_faces_official(mesh, visibility, {},
                                             &filter_stats, error)) {
            return false;
        }
        result.pre_meshfix_faces = mesh.indices.size() / 3;

#if defined(SAM3D_USE_MESHFIX_GPL)
        // This is int(250 * sqrt(1 - 0.95)) from process_mesh().
        if (!repair_mesh_boundaries_official_meshfix(mesh, 55, true, error))
            return false;
#else
        error = "complete official mesh cleanup requires "
                "SAM3D_GGML_MESHFIX_GPL=ON "
                "because pymeshfix TMesh is GPL-3.0-or-commercial; no "
                "approximate repair is used";
        return false;
#endif
    } else {
        result.pre_meshfix_faces = mesh.indices.size() / 3;
    }

    if (!parameterize_mesh_xatlas(mesh, error)) return false;
    // Generate normals after xatlas, whose seam duplication changes adjacency.
    compute_official_angle_weighted_vertex_normals(mesh);

    GaussianRenderConfig render_config;
    render_config.width = config.render_resolution;
    render_config.height = config.render_resolution;
    std::vector<GaussianCamera> cameras = make_gaussian_hammersley_cameras(
            config.render_views, render_config, error);
    if (cameras.empty()) return false;
    std::vector<RgbaImage> observations;
    if (!render_gaussian_views_cuda(splats, cameras, render_config,
                                    observations, error)) {
        return false;
    }

    TextureBakeConfig bake_config;
    bake_config.texture_size = config.texture_size;
    bake_config.steps = config.texture_steps;
    bake_config.random_seed = config.random_seed;
    RgbaImage texture;
    if (!bake_texture_official_cuda(mesh, observations, cameras, bake_config,
                                    texture, nullptr, nullptr, error)) {
        return false;
    }
    mesh.material.base_color_texture = std::move(texture);
    std::fill_n(mesh.material.base_color, 4, 1.0f);
    mesh.material.metallic = 1.0f;
    mesh.material.roughness = 1.0f;
    // Match trimesh's GLB exporter, after baking in xatlas/OpenGL coordinates.
    // Flipping earlier would change the optimizer's observation mapping.
    for (size_t uv = 1; uv < mesh.texcoords.size(); uv += 2) {
        mesh.texcoords[uv] = 1.0f - mesh.texcoords[uv];
    }
    rotate_zup_mesh_to_yup(mesh);

    result.rendered_views = observations.size();
    result.final_faces = mesh.indices.size() / 3;
    result.mesh = std::move(mesh);
    return true;
#endif
}

}  // namespace sam3d
