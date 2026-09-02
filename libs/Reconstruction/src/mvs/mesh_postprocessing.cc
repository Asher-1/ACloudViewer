// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "mvs/mesh_postprocessing.h"

#include <AutoIO.h>
#include <ecvHObjectCaster.h>
#include <ecvMesh.h>
#include <ecvPointCloud.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <vector>

#include "meshoptimizer.h"
#include "util/logging.h"
#include "util/option_manager.h"

namespace colmap {
namespace mvs {
namespace {

using Index = unsigned int;

struct Edge {
    Index a;
    Index b;
    bool operator==(const Edge& other) const { return a == other.a && b == other.b; }
};

struct EdgeHash {
    size_t operator()(const Edge& edge) const {
        return std::hash<uint64_t>{}((uint64_t(edge.a) << 32) | edge.b);
    }
};

Edge MakeEdge(Index a, Index b) {
    if (a > b) std::swap(a, b);
    return {a, b};
}

void CompactMesh(PlyMesh* mesh, const std::vector<Index>& indices) {
    std::vector<int> remap(mesh->vertices.size(), -1);
    std::vector<PlyMeshVertex> vertices;
    std::vector<PlyMeshFace> faces;
    vertices.reserve(mesh->vertices.size());
    faces.reserve(indices.size() / 3);
    for (size_t i = 0; i + 2 < indices.size(); i += 3) {
        size_t ids[3] = {indices[i], indices[i + 1], indices[i + 2]};
        size_t compact[3];
        for (int k = 0; k < 3; ++k) {
            if (ids[k] >= mesh->vertices.size()) return;
            if (remap[ids[k]] < 0) {
                remap[ids[k]] = static_cast<int>(vertices.size());
                vertices.push_back(mesh->vertices[ids[k]]);
            }
            compact[k] = static_cast<size_t>(remap[ids[k]]);
        }
        faces.emplace_back(compact[0], compact[1], compact[2]);
    }
    mesh->vertices.swap(vertices);
    mesh->faces.swap(faces);
}

std::vector<float> Positions(const PlyMesh& mesh) {
    std::vector<float> positions;
    positions.reserve(mesh.vertices.size() * 3);
    for (const PlyMeshVertex& vertex : mesh.vertices) {
        positions.push_back(vertex.x);
        positions.push_back(vertex.y);
        positions.push_back(vertex.z);
    }
    return positions;
}

void Smooth(PlyMesh* mesh, const MeshPostProcessingOptions& options) {
    const size_t vertex_count = mesh->vertices.size();
    if (vertex_count == 0 || mesh->faces.empty() || options.smoothing_iterations <= 0)
        return;

    std::vector<std::vector<Index>> neighbors(vertex_count);
    std::unordered_map<Edge, int, EdgeHash> edge_count;
    edge_count.reserve(mesh->faces.size() * 3);
    for (const PlyMeshFace& face : mesh->faces) {
        const Index a = static_cast<Index>(face.vertex_idx1);
        const Index b = static_cast<Index>(face.vertex_idx2);
        const Index c = static_cast<Index>(face.vertex_idx3);
        const Index ids[3] = {a, b, c};
        for (int i = 0; i < 3; ++i) {
            const Index u = ids[i], v = ids[(i + 1) % 3];
            neighbors[u].push_back(v);
            neighbors[v].push_back(u);
            ++edge_count[MakeEdge(u, v)];
        }
    }
    std::vector<bool> boundary(vertex_count, false);
    for (const auto& entry : edge_count) {
        if (entry.second == 1) {
            boundary[entry.first.a] = true;
            boundary[entry.first.b] = true;
        }
    }
    for (auto& list : neighbors) {
        std::sort(list.begin(), list.end());
        list.erase(std::unique(list.begin(), list.end()), list.end());
    }

    std::vector<PlyMeshVertex> next = mesh->vertices;
    auto pass = [&](double weight) {
        next = mesh->vertices;
        for (size_t i = 0; i < vertex_count; ++i) {
            if (neighbors[i].empty() || (options.preserve_boundary && boundary[i]))
                continue;
            double mean[3] = {0, 0, 0};
            for (Index j : neighbors[i]) {
                mean[0] += mesh->vertices[j].x;
                mean[1] += mesh->vertices[j].y;
                mean[2] += mesh->vertices[j].z;
            }
            const double inv = 1.0 / neighbors[i].size();
            next[i].x = static_cast<float>(mesh->vertices[i].x +
                                           weight * (mean[0] * inv - mesh->vertices[i].x));
            next[i].y = static_cast<float>(mesh->vertices[i].y +
                                           weight * (mean[1] * inv - mesh->vertices[i].y));
            next[i].z = static_cast<float>(mesh->vertices[i].z +
                                           weight * (mean[2] * inv - mesh->vertices[i].z));
        }
        mesh->vertices.swap(next);
    };
    for (int i = 0; i < options.smoothing_iterations; ++i) {
        pass(options.smoothing_lambda);
        pass(options.smoothing_mu);
    }
}

bool ReadMesh(const std::string& path, PlyMesh* result) {
    ccMesh mesh;
    if (!mesh.CreateInternalCloud()) return false;
    cloudViewer::io::ReadTriangleMeshOptions read_options;
    read_options.print_progress = false;
    if (!cloudViewer::io::AutoReadMesh(path, mesh, read_options)) return false;
    ccPointCloud* cloud = ccHObjectCaster::ToPointCloud(mesh.getAssociatedCloud());
    if (cloud == nullptr) return false;
    result->vertices.clear();
    result->faces.clear();
    result->vertices.reserve(cloud->size());
    for (unsigned int i = 0; i < cloud->size(); ++i) {
        const CCVector3* point = cloud->getPoint(i);
        CCVector3d global(point->x, point->y, point->z);
        if (cloud->isShifted()) global = cloud->toGlobal3d(*point);
        if (cloud->hasColors()) {
            const ecvColor::Rgb& color = cloud->getPointColor(i);
            result->vertices.emplace_back(static_cast<float>(global.x),
                                          static_cast<float>(global.y),
                                          static_cast<float>(global.z), color.r,
                                          color.g, color.b);
        } else {
            result->vertices.emplace_back(static_cast<float>(global.x),
                                          static_cast<float>(global.y),
                                          static_cast<float>(global.z));
        }
    }
    result->faces.reserve(mesh.size());
    for (unsigned int i = 0; i < mesh.size(); ++i) {
        const cloudViewer::VerticesIndexes* triangle = mesh.getTriangleVertIndexes(i);
        result->faces.emplace_back(triangle->i1, triangle->i2, triangle->i3);
    }
    return true;
}

}  // namespace

bool MeshPostProcessingOptions::Check() const {
    CHECK_OPTION_GE(prune_error, 0.0);
    CHECK_OPTION_LE(prune_error, 1.0);
    CHECK_OPTION_GT(target_face_ratio, 0.0);
    CHECK_OPTION_LE(target_face_ratio, 1.0);
    CHECK_OPTION_GE(simplify_error, 0.0);
    CHECK_OPTION_LE(simplify_error, 1.0);
    CHECK_OPTION_GT(max_aspect_ratio, 0.0);
    CHECK_OPTION_GE(smoothing_iterations, 0);
    CHECK_OPTION_GT(smoothing_lambda, 0.0);
    CHECK_OPTION_LT(smoothing_lambda, 1.0);
    CHECK_OPTION_LT(smoothing_mu, 0.0);
    CHECK_OPTION_GT(smoothing_mu, -1.0);
    return true;
}

PlyMesh PostProcessMesh(const PlyMesh& input,
                        const MeshPostProcessingOptions& options,
                        MeshPostProcessingStats* stats) {
    CHECK(options.Check());
    MeshPostProcessingStats local;
    local.input_vertices = input.vertices.size();
    local.input_faces = input.faces.size();
    PlyMesh mesh = input;
    if (!options.enabled || mesh.vertices.empty() || mesh.faces.empty()) {
        local.output_vertices = mesh.vertices.size();
        local.output_faces = mesh.faces.size();
        if (stats) *stats = local;
        return mesh;
    }

    std::vector<Index> valid_indices;
    valid_indices.reserve(mesh.faces.size() * 3);
    const double min_area_sq = 1e-12;
    for (const PlyMeshFace& face : mesh.faces) {
        const size_t a = face.vertex_idx1, b = face.vertex_idx2, c = face.vertex_idx3;
        if (a >= mesh.vertices.size() || b >= mesh.vertices.size() ||
            c >= mesh.vertices.size() || a == b || b == c || a == c) {
            ++local.removed_invalid_faces;
            continue;
        }
        const auto& va = mesh.vertices[a];
        const auto& vb = mesh.vertices[b];
        const auto& vc = mesh.vertices[c];
        const double abx = vb.x - va.x, aby = vb.y - va.y, abz = vb.z - va.z;
        const double acx = vc.x - va.x, acy = vc.y - va.y, acz = vc.z - va.z;
        const double nx = aby * acz - abz * acy;
        const double ny = abz * acx - abx * acz;
        const double nz = abx * acy - aby * acx;
        const double area_sq = nx * nx + ny * ny + nz * nz;
        const double l0 = abx * abx + aby * aby + abz * abz;
        const double l1 = acx * acx + acy * acy + acz * acz;
        const double dx = vc.x - vb.x, dy = vc.y - vb.y, dz = vc.z - vb.z;
        const double l2 = dx * dx + dy * dy + dz * dz;
        const double longest = std::max({l0, l1, l2});
        const bool invalid = !std::isfinite(area_sq) || area_sq <= min_area_sq ||
                             (longest > 0 && area_sq * options.max_aspect_ratio *
                                                      options.max_aspect_ratio <
                                              longest * longest);
        if (options.remove_degenerate_faces && invalid) {
            ++local.removed_invalid_faces;
            continue;
        }
        valid_indices.push_back(static_cast<Index>(a));
        valid_indices.push_back(static_cast<Index>(b));
        valid_indices.push_back(static_cast<Index>(c));
    }
    CompactMesh(&mesh, valid_indices);

    if (options.remove_small_components && !mesh.faces.empty()) {
        std::vector<float> positions = Positions(mesh);
        std::vector<Index> indices;
        indices.reserve(mesh.faces.size() * 3);
        for (const auto& face : mesh.faces) {
            indices.push_back(static_cast<Index>(face.vertex_idx1));
            indices.push_back(static_cast<Index>(face.vertex_idx2));
            indices.push_back(static_cast<Index>(face.vertex_idx3));
        }
        std::vector<Index> pruned(indices.size());
        const size_t count = meshopt_simplifyPrune(
                pruned.data(), indices.data(), indices.size(), positions.data(),
                mesh.vertices.size(), sizeof(float) * 3,
                static_cast<float>(options.prune_error));
        pruned.resize(count);
        local.removed_small_component_faces = mesh.faces.size() - count / 3;
        CompactMesh(&mesh, pruned);
    }

    if (options.simplify && options.target_face_ratio < 1.0 && !mesh.faces.empty()) {
        std::vector<float> positions = Positions(mesh);
        std::vector<Index> indices;
        indices.reserve(mesh.faces.size() * 3);
        for (const auto& face : mesh.faces) {
            indices.push_back(static_cast<Index>(face.vertex_idx1));
            indices.push_back(static_cast<Index>(face.vertex_idx2));
            indices.push_back(static_cast<Index>(face.vertex_idx3));
        }
        const size_t target = std::max<size_t>(3, static_cast<size_t>(
                std::floor(indices.size() * options.target_face_ratio / 3.0) * 3));
        std::vector<Index> simplified(indices.size());
        const size_t count = meshopt_simplify(
                simplified.data(), indices.data(), indices.size(), positions.data(),
                mesh.vertices.size(), sizeof(float) * 3, target,
                static_cast<float>(options.simplify_error), 0, nullptr);
        simplified.resize(count - (count % 3));
        local.simplified_faces = mesh.faces.size() - simplified.size() / 3;
        CompactMesh(&mesh, simplified);
    }
    if (options.smooth) Smooth(&mesh, options);
    local.output_vertices = mesh.vertices.size();
    local.output_faces = mesh.faces.size();
    if (stats) *stats = local;
    return mesh;
}

bool PostProcessMeshFile(const std::string& input_path,
                         const std::string& output_path,
                         const MeshPostProcessingOptions& options,
                         MeshPostProcessingStats* stats) {
    PlyMesh input;
    if (!ReadMesh(input_path, &input)) return false;
    const PlyMesh output = PostProcessMesh(input, options, stats);
    if (input_path == output_path) {
        const std::string temp_path = output_path + ".meshopt.tmp.ply";
        const std::string backup_path = output_path + ".meshopt.backup.ply";
        std::error_code ec;
        std::filesystem::remove(temp_path, ec);
        WriteBinaryPlyMesh(temp_path, output);
        if (!std::filesystem::exists(temp_path, ec) || ec) {
            return false;
        }

        std::filesystem::remove(backup_path, ec);
        ec.clear();
        std::filesystem::rename(output_path, backup_path, ec);
        if (ec) {
            std::filesystem::remove(temp_path, ec);
            return false;
        }

        ec.clear();
        std::filesystem::rename(temp_path, output_path, ec);
        if (ec) {
            std::error_code restore_ec;
            std::filesystem::rename(backup_path, output_path, restore_ec);
            std::filesystem::remove(temp_path, ec);
            return false;
        }
        std::filesystem::remove(backup_path, ec);
    } else {
        WriteBinaryPlyMesh(output_path, output);
    }
    return true;
}

}  // namespace mvs
}  // namespace colmap
