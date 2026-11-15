// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "tasks/trellis/third_party/CuMesh/src/utils.h"

#define BLOCK_SIZE 256

namespace cumesh {

/** Host copy of a mesh: positions are [V * 3] floats, faces [F * 3] ints. */
struct HostMesh {
    std::vector<float> vertices;
    std::vector<int> faces;
};

/** Host copy of the atlas chart data produced by compute_charts. */
struct HostAtlasCharts {
    int num_charts = 0;
    std::vector<int> chart_ids;            // [F]
    std::vector<int> chart_vertex_map;     // [chart_vertex_offset[C]]
    std::vector<int> chart_faces;          // [F * 3]
    std::vector<int> chart_vertex_offset;  // [C + 1]
    std::vector<int> chart_face_offset;    // [C + 1]
};

class CuMesh {
public:
    Buffer<float3> vertices;
    Buffer<int3> faces;

    // Geometric properties
    Buffer<float> face_areas;
    Buffer<float3> face_normals;
    Buffer<float3> vertex_normals;

    // Connectivity
    Buffer<uint64_t> edges;
    Buffer<int> boundaries;
    Buffer<uint8_t> vert_is_boundary;
    Buffer<uint8_t> vert_is_manifold;
    Buffer<int> vert2edge;
    Buffer<int> vert2edge_cnt;
    Buffer<int> vert2edge_offset;
    Buffer<int> vert2bound;
    Buffer<int> vert2bound_cnt;
    Buffer<int> vert2bound_offset;
    Buffer<int> edge2face;
    Buffer<int> edge2face_cnt;
    Buffer<int> edge2face_offset;
    Buffer<int3> face2edge;
    Buffer<int> vert2face;
    Buffer<int> vert2face_cnt;
    Buffer<int> vert2face_offset;
    Buffer<int2> manifold_face_adj;
    Buffer<int2> manifold_bound_adj;
    Buffer<int> conn_comp_ids;
    Buffer<int> bound_conn_comp_ids;
    Buffer<int> loop_boundaries;
    Buffer<int> loop_boundaries_offset;
    int num_conn_comps;
    int num_bound_conn_comps;
    int num_bound_loops;

    // Cleanup
    Buffer<int> vertices_map;
    Buffer<int> faces_map;

    // Simplification
    Buffer<float> edge_collapse_costs;
    Buffer<uint64_t> propagated_costs;

    // Atlasing
    int atlas_num_charts;
    Buffer<int> atlas_chart_ids;
    Buffer<int> atlas_chart_vertex_map;
    Buffer<int3> atlas_chart_faces;
    Buffer<int> atlas_chart_faces_offset;
    Buffer<int> atlas_chart_vertex_offset;
    Buffer<float2> atlas_chart_uvs;

    Buffer<float4> atlas_chart_normal_cones;
    Buffer<uint64_t> atlas_chart_adj;
    Buffer<float> atlas_chart_adj_length;
    Buffer<float> atlas_chart_perims;
    Buffer<float> atlas_chart_areas;
    Buffer<int> atlas_chart2edge;
    Buffer<int> atlas_chart2edge_cnt;
    Buffer<int> atlas_chart2edge_offset;

    // Temporary storage
    Buffer<char> temp_storage;
    Buffer<char> cub_temp_storage;

    CuMesh();

    ~CuMesh();

    int num_vertices() const;

    int num_faces() const;

    int num_edges() const;

    int num_boundaries() const;

    int num_conneted_components() const;

    int num_boundary_conneted_components() const;

    int num_boundary_loops() const;

    void clear_cache();

    /**
     * Initialize mesh from host memory.
     *
     * @param host_vertices [num_vertices * 3] packed float positions.
     * @param num_vertices Vertex count.
     * @param host_faces [num_faces * 3] packed int indices.
     * @param num_faces Face count.
     */
    void init(const float* host_vertices,
              size_t num_vertices,
              const int* host_faces,
              size_t num_faces);

    /**
     * Copy the current mesh to host memory.
     */
    HostMesh read();

    // Geometric functions

    /**
     * Compute face areas.
     * This function refreshes:
     * - face_areas
     */
    void compute_face_areas();

    /**
     * Compute face normals.
     * This function refreshes:
     * - face_normals
     */
    void compute_face_normals();

    /**
     * Compute vertex normals.
     * This function requires:
     * - vert2face
     * - vert2face_offset
     * This function refreshes:
     * - vertex_normals
     */
    void compute_vertex_normals();

    // Connectivity functions

    /**
     * Get the vertex to face adjacency.
     * This function refreshes:
     * - vert2face
     * - vert2face_cnt
     * - vert2face_offset
     */
    void get_vertex_face_adjacency();

    /**
     * Get the edges of the mesh.
     * This function refreshes:
     * - edges
     * - edge2face_cnt
     */
    void get_edges();

    /**
     * Get the edges of the mesh.
     * This function requires:
     * - edges
     * - edge2face_cnt
     * - vert2face
     * - vert2face_offset
     * This function refreshes:
     * - edge2face
     * - edge2face_offset
     * - face2edge
     */
    void get_edge_face_adjacency();

    /**
     * Get the vertex to edge adjacency.
     * This function requires:
     * - edges
     * This function refreshes:
     * - vert2edge
     * - vert2edge_cnt
     * - vert2edge_offset
     */
    void get_vertex_edge_adjacency();

    /**
     * Get boundary information.
     * This function requires:
     * - edges
     * - edge2face_cnt
     * This function refreshes:
     * - boundaries
     * - vert_is_boundary
     */
    void get_boundary_info();

    /**
     * Get the vertex to boundary adjacency.
     * This function requires:
     * - edges
     * - boundaries
     * This function refreshes:
     * - vert2bound
     * - vert2bound_cnt
     * - vert2bound_offset
     */
    void get_vertex_boundary_adjacency();

    /**
     * Get edge is manifold information.
     * This function requires:
     * - vert2edge
     * - vert2edge_offset
     * - edge2face_cnt
     * This function refreshes:
     * - vert_is_manifold
     */
    void get_vertex_is_manifold();

    /**
     * Get the face adjacency for manifold edges.
     * This function requires:
     * - edge2face
     * - edge2face_offset
     * This function refreshes:
     * - manifold_face_adj
     */
    void get_manifold_face_adjacency();

    /**
     * Get the face adjacency for manifold boundaries.
     * This function requires:
     * - vert_is_manifold
     * - vert2bound
     * - vert2bound_offset
     * This function refreshes:
     * - manifold_bound_adj
     */
    void get_manifold_boundary_adjacency();

    /**
     * Get the connected components of the mesh.
     * This function requires:
     * - manifold_face_adj
     * This function refreshes:
     * - conn_comp_ids
     */
    void get_connected_components();

    /**
     * Get the boundary connected components of the mesh.
     * This function requires:
     * - manifold_bound_adj
     * This function refreshes:
     * - bound_conn_comp_ids
     */
    void get_boundary_connected_components();

    /**
     * Get the boundary loops of the mesh.
     * This function requires:
     * - vert2bound
     * - vert2bound_offset
     * - vert_is_boundary
     * - bound_conn_comp_ids
     * This function refreshes:
     * - loop_boundaries
     * - loop_boundaries_offset
     */
    void get_boundary_loops();

    // Cleanup functions

    /**
     * Remove faces.
     */
    void remove_faces(const bool* face_mask);
    void _remove_faces(uint8_t* face_mask);

    /**
     * Remove unreferenced vertices.
     */
    void remove_unreferenced_vertices();

    /**
     * Remove duplicate faces.
     */
    void remove_duplicate_faces();

    /**
     * Remove degenerate faces.
     */
    void remove_degenerate_faces(float abs_thresh, float rel_thresh);

    /**
     * Fill holes.
     * This function requires:
     * - loop_boundaries
     * - loop_boundaries_offset
     *
     * @param max_hole_perimeter The maximum perimeter of a hole to be filled.
     */
    void fill_holes(float max_hole_perimeter);

    /**
     * Repair Non-manifold edges by splitting edges.
     * This function requires:
     * - manifold_face_adj
     * This function refreshes:
     * - vertices
     * - faces
     * This function destroys:
     * - All connectivity information
     */
    void repair_non_manifold_edges();

    /**
     * Remove faces on non-manifold edges.
     * For each non-manifold edge (shared by >2 faces), only keep the first 2
     * faces. This repairs non-manifold edges by deleting faces instead of
     * splitting vertices. This function requires:
     * - edge2face
     * - edge2face_offset
     * - edge2face_cnt
     * This function refreshes:
     * - vertices
     * - faces
     * This function destroys:
     * - All connectivity information
     */
    void remove_non_manifold_faces();

    /**
     * Remove small connected components.
     * This function requires:
     * - conn_comp_ids
     * This function refreshes:
     * - vertices
     * - faces
     * This function destroys:
     * - All connectivity information
     *
     * @param min_area The minimum area of the connected components to be kept.
     */
    void remove_small_connected_components(float min_area);

    /**
     * Unify face orientations.
     * This function requires:
     * - manifold_face_adj
     * This function refreshes:
     * - faces
     */
    void unify_face_orientations();

    // Simplification functions

    /**
     * Run the edge collapse algorithm.
     * This function refreshes:
     * - vertices
     * - faces
     * This function destroys:
     * - All connectivity information
     *
     * @param lambda_edge_length The weight for edge length term.
     * @param lambda_skinny The weight for skinny term.
     * @param threshold The threshold for edge collapse cost.
     * @return A tuple of the number of vertices and the number of faces after
     * simplification.
     */
    std::tuple<int, int> simplify_step(float lambda_edge_length,
                                       float lambda_skinny,
                                       float threshold,
                                       bool timing = false);

    // Atlasing functions

    /**
     * Compute charts for atlasing.
     * This function requires:
     * - manifold_face_adj
     * This function refreshes:
     * - atlas_face_chart_ids
     * - atlas_chart_vertex_map
     * - atlas_chart_faces
     * - atlas_chart_faces_offset
     *
     *  @param  threshold_cone_half_angle_rad The threshold for the cone half
     * angle in radians.
     *  @param  refine_iterations             The number of refinement
     * iterations.
     *  @param  global_iterations             The number of global iterations.
     *  @param  smooth_strength               The strength of the smoothing.
     *  @param  area_penalty_weight           Coefficient for chart size
     * penalty. Cost += Area * weight. Prevents charts from becoming too large
     * if > 0, or encourages larger charts if < 0 (though usually used to
     * penalize size variance).
     *  @param  perimeter_area_ratio_weight   Coefficient for shape irregularity
     * (long-strip) penalty. Cost += (Perimeter / Area) * weight. Higher values
     * penalize long strips and encourage circular/compact shapes.
     */
    void compute_charts(float threshold_cone_half_angle_rad,
                        int refine_iterations,
                        int global_iterations,
                        float smooth_strength,
                        float area_penalty_weight,
                        float perimeter_area_ratio_weight);

    /**
     * Copy the atlas charts to host memory (see HostAtlasCharts).
     */
    HostAtlasCharts read_atlas_charts();
};

}  // namespace cumesh
