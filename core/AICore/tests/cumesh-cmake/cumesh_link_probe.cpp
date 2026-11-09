// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#if defined(TRELLIS2_HAVE_CUMESH)
#include <cstdio>
#include <cstring>

#include "cumesh_glue.h"

static bool invalid_input_contract() {
    float unused_vertex = 0;
    int unused_face = 0;
    float* vertices = &unused_vertex;
    int* faces = &unused_face;
    int nv = -1, nt = -1;
    if (cumesh_clean_mesh(nullptr, 0, nullptr, 0, &vertices, &nv, &faces,
                          &nt) ||
        vertices || faces || nv || nt) {
        return false;
    }
    int *ids = &unused_face, *map = &unused_face, *chart_faces = &unused_face;
    int *face_offsets = &unused_face, *vertex_offsets = &unused_face;
    int charts = -1;
    if (cumesh_compute_charts(nullptr, 0, nullptr, 0, 1.57f, 2, 1, 1, -0.1f, 0,
                              &ids, &map, &chart_faces, &face_offsets,
                              &vertex_offsets, &charts, &nt) ||
        ids || map || chart_faces || face_offsets || vertex_offsets || charts ||
        nt) {
        return false;
    }
    return true;
}

static bool gpu_mesh_contract() {
    // Host pointers are the bridge contract. Include an unused vertex and
    // remove a degenerate first face to exercise compaction, not just loading.
    const float vertices[] = {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 9, 9, 9};
    const int dirty_faces[] = {0, 0, 0, 0, 1, 2, 0, 2, 3};
    float* cleaned_vertices = nullptr;
    int* cleaned_faces = nullptr;
    int nv = 0, nt = 0;
    const int cleaned =
            cumesh_clean_mesh(vertices, 5, dirty_faces, 3, &cleaned_vertices,
                              &nv, &cleaned_faces, &nt);
    bool valid = cleaned && nv == 4 && nt == 2;
    if (valid) {
        for (int i = 0; i < nt * 3; ++i)
            valid = valid && cleaned_faces[i] >= 0 && cleaned_faces[i] < nv;
        for (int i = 0; i < nv * 3; ++i)
            valid = valid && cleaned_vertices[i] == vertices[i];
    }
    cumesh_free_buffer(cleaned_vertices);
    cumesh_free_buffer(cleaned_faces);
    if (!valid) {
        std::fprintf(stderr, "CuMesh GPU cleanup contract failed\n");
        return false;
    }

    const float folded_vertices[] = {0, 0, 0, 1, 0, 0, 1, 1,
                                     0, 0, 0, 1, 9, 9, 9};
    const int faces[] = {0, 1, 2, 0, 2, 3};
    int *ids = nullptr, *map = nullptr, *chart_faces = nullptr;
    int *face_offsets = nullptr, *vertex_offsets = nullptr;
    int charts = 0;
    const int clustered = cumesh_compute_charts(
            folded_vertices, 5, faces, 2, 0.01f, 2, 1, 1, -0.1f, 0, &ids, &map,
            &chart_faces, &face_offsets, &vertex_offsets, &charts, &nt);
    valid = clustered && charts == 2 && nt == 2;
    if (valid) {
        valid = face_offsets[0] == 0 && face_offsets[charts] == nt &&
                vertex_offsets[0] == 0;
        for (int i = 0; i < nt; ++i)
            valid = valid && ids[i] >= 0 && ids[i] < charts;
        for (int c = 0; c < charts; ++c) {
            const int chart_nv = vertex_offsets[c + 1] - vertex_offsets[c];
            valid = valid && chart_nv > 0 &&
                    face_offsets[c + 1] > face_offsets[c];
            for (int i = vertex_offsets[c]; i < vertex_offsets[c + 1]; ++i)
                valid = valid && map[i] >= 0 && map[i] < 4;
            for (int i = face_offsets[c] * 3; i < face_offsets[c + 1] * 3; ++i)
                valid = valid && chart_faces[i] >= vertex_offsets[c] &&
                        chart_faces[i] < vertex_offsets[c + 1];
        }
    }
    cumesh_free_buffer(ids);
    cumesh_free_buffer(map);
    cumesh_free_buffer(chart_faces);
    cumesh_free_buffer(face_offsets);
    cumesh_free_buffer(vertex_offsets);
    if (!valid) std::fprintf(stderr, "CuMesh GPU chart contract failed\n");
    return valid;
}

int main(int argc, char** argv) {
    cumesh_free_buffer(nullptr);
    if (!invalid_input_contract()) return 1;
    return argc == 2 && std::strcmp(argv[1], "--gpu") == 0
                   ? (gpu_mesh_contract() ? 0 : 1)
                   : 0;
}
#else
int main() { return 0; }
#endif
