// Native inference-only implementation of the official FlexiCubes mesh path.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace sam3d {

struct FlexiCubesResult {
    std::vector<float> positions;          // xyz, vertex-major
    std::vector<float> vertex_attributes;  // RGB + learned normal map, six values per vertex
    std::vector<uint32_t> indices;         // triangle indices

    bool success() const { return !positions.empty() && !indices.empty(); }
};

// Decode the [101, sparse_cell_count] output of SLatMeshDecoder. Coordinates
// are [4, sparse_cell_count] in batch, x, y, z order. This matches
// SparseFeatures2Mesh(..., training=False), including its dense default SDF.
bool decode_flexicubes(const float* cube_features, const int32_t* coordinates,
                       int64_t sparse_cell_count, int resolution,
                       FlexiCubesResult& output, std::string& error);

}  // namespace sam3d
