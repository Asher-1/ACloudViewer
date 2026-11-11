// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// cumesh_dedup_parity — GPU bit-level consistency test for CuMesh
// remove_duplicate_faces against a CPU reference implementation.
//
// The GPU path sorts the three vertex ids of every face, orders the faces
// lexicographically with three stable radix passes, keeps the first face of
// each equal run, compacts the surviving faces and drops unreferenced
// vertices. The CPU path below reproduces those exact steps, so any
// disagreement is a difference in the CUDA 11 radix-sort rewrite or the
// rewritten in-place scan, not in the test data.
//
// This file is compiled into its own executable together with the CuMesh
// sources rather than linking libaicore_cumesh. The class methods are not part
// of the library's exported surface (only the C glue is), so linking the
// shared library would fail under hidden visibility on Linux and with no
// dllexport on Windows.

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <random>
#include <vector>

#include "cumesh.h"

struct CPUMesh {
    std::vector<float> vertices;
    std::vector<int> faces;
};

static void sort_face_vertices(int& a, int& b, int& c) {
    if (a > b) std::swap(a, b);
    if (b > c) std::swap(b, c);
    if (a > b) std::swap(a, b);
}

static CPUMesh cpu_remove_duplicate_faces(const CPUMesh& input) {
    const size_t nv = input.vertices.size() / 3;
    const size_t nf = input.faces.size() / 3;

    // Sort vertices within each face, keeping the original order for the
    // faces that survive.
    std::vector<std::pair<std::array<int, 3>, size_t>> indexed(nf);
    for (size_t i = 0; i < nf; ++i) {
        int a = input.faces[i * 3 + 0];
        int b = input.faces[i * 3 + 1];
        int c = input.faces[i * 3 + 2];
        sort_face_vertices(a, b, c);
        indexed[i] = {{a, b, c}, i};
    }

    // Stable lexicographic order: the three radix passes are stable too.
    std::stable_sort(indexed.begin(), indexed.end(),
                     [](const auto& lhs, const auto& rhs) {
                         return lhs.first < rhs.first;
                     });

    std::vector<bool> keep_original(nf, false);
    for (size_t i = 0; i < nf; ++i) {
        const bool first = i == 0 || indexed[i].first != indexed[i - 1].first;
        if (first) keep_original[indexed[i].second] = true;
    }

    std::vector<int> kept_faces;
    kept_faces.reserve(nf * 3);
    for (size_t i = 0; i < nf; ++i) {
        if (!keep_original[i]) continue;
        kept_faces.push_back(input.faces[i * 3 + 0]);
        kept_faces.push_back(input.faces[i * 3 + 1]);
        kept_faces.push_back(input.faces[i * 3 + 2]);
    }

    // Exclusive scan over the referenced flags, matching the device scan.
    std::vector<int> vertex_map(nv + 1, 0);
    for (int index : kept_faces) vertex_map[index] = 1;
    int running = 0;
    for (size_t i = 0; i < nv; ++i) {
        const int referenced = vertex_map[i];
        vertex_map[i] = running;
        running += referenced;
    }
    vertex_map[nv] = running;

    std::vector<float> kept_vertices(running * 3);
    for (size_t i = 0; i < nv; ++i) {
        if (vertex_map[i + 1] == vertex_map[i] + 1) {
            kept_vertices[vertex_map[i] * 3 + 0] = input.vertices[i * 3 + 0];
            kept_vertices[vertex_map[i] * 3 + 1] = input.vertices[i * 3 + 1];
            kept_vertices[vertex_map[i] * 3 + 2] = input.vertices[i * 3 + 2];
        }
    }
    for (int& index : kept_faces) index = vertex_map[index];

    return {kept_vertices, kept_faces};
}

static bool compare_meshes(const cumesh::HostMesh& gpu,
                           const CPUMesh& cpu,
                           const char* label) {
    if (gpu.vertices.size() != cpu.vertices.size() ||
        gpu.faces.size() != cpu.faces.size()) {
        std::fprintf(stderr,
                     "[%s] size mismatch: GPU %zu verts / %zu faces, CPU %zu "
                     "verts / %zu faces\n",
                     label, gpu.vertices.size() / 3, gpu.faces.size() / 3,
                     cpu.vertices.size() / 3, cpu.faces.size() / 3);
        return false;
    }
    for (size_t i = 0; i < gpu.vertices.size(); ++i) {
        if (gpu.vertices[i] != cpu.vertices[i]) {
            std::fprintf(stderr,
                         "[%s] vertex mismatch at %zu: GPU %.9g, CPU %.9g\n",
                         label, i, gpu.vertices[i], cpu.vertices[i]);
            return false;
        }
    }
    for (size_t i = 0; i < gpu.faces.size(); ++i) {
        if (gpu.faces[i] != cpu.faces[i]) {
            std::fprintf(stderr, "[%s] face mismatch at %zu: GPU %d, CPU %d\n",
                         label, i, gpu.faces[i], cpu.faces[i]);
            return false;
        }
    }
    return true;
}

static CPUMesh generate_test_mesh(int seed, int nv, int nf, int dup_percent) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> position(-10.0f, 10.0f);
    std::uniform_int_distribution<int> vertex(0, nv - 1);
    std::uniform_int_distribution<int> percent(0, 99);

    CPUMesh mesh;
    mesh.vertices.resize(nv * 3);
    for (float& value : mesh.vertices) value = position(rng);

    mesh.faces.reserve(nf * 3);
    std::vector<std::array<int, 3>> unique_faces;
    for (int i = 0; i < nf; ++i) {
        std::array<int, 3> face;
        if (percent(rng) < dup_percent && !unique_faces.empty()) {
            face = unique_faces[std::uniform_int_distribution<size_t>(
                    0, unique_faces.size() - 1)(rng)];
            // A duplicate is the same three vertices in another order.
            if (percent(rng) < 50) std::swap(face[0], face[1]);
            if (percent(rng) < 50) std::swap(face[1], face[2]);
        } else {
            face = {vertex(rng), vertex(rng), vertex(rng)};
            unique_faces.push_back(face);
        }
        mesh.faces.insert(mesh.faces.end(), face.begin(), face.end());
    }
    return mesh;
}

int main() {
    int device_count = 0;
    const cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess || device_count == 0) {
        std::fprintf(stderr, "No CUDA device available (%s)\n",
                     cudaGetErrorString(status));
        return 77;
    }

    const int seeds[] = {42, 1337, 9001, 314159, 271828};
    const int nvs[] = {100, 500, 1000, 2000, 5000};
    const int nfs[] = {200, 1000, 3000, 6000, 15000};
    const int dup_percents[] = {30, 50, 70, 80, 90};

    bool all_passed = true;
    for (int test = 0; test < 5; ++test) {
        char label[96];
        std::snprintf(label, sizeof(label),
                      "case %d seed %d verts %d faces %d duplicates %d%%", test,
                      seeds[test], nvs[test], nfs[test], dup_percents[test]);

        const CPUMesh input = generate_test_mesh(seeds[test], nvs[test],
                                                 nfs[test], dup_percents[test]);
        const CPUMesh cpu_result = cpu_remove_duplicate_faces(input);

        cumesh::CuMesh gpu_mesh;
        gpu_mesh.init(input.vertices.data(), static_cast<size_t>(nvs[test]),
                      input.faces.data(), static_cast<size_t>(nfs[test]));
        gpu_mesh.remove_duplicate_faces();
        const cumesh::HostMesh gpu_result = gpu_mesh.read();

        const bool passed = compare_meshes(gpu_result, cpu_result, label);
        std::printf("[%s] %s -> GPU %zu verts / %zu faces\n", label,
                    passed ? "PASS" : "FAIL", gpu_result.vertices.size() / 3,
                    gpu_result.faces.size() / 3);
        all_passed = all_passed && passed;
    }
    return all_passed ? 0 : 1;
}
