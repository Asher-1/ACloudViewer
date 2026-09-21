// Shared native SLatMeshDecoder execution used by CLI and full E2E assembly.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace sam3d {

struct MeshDecoderRunOptions {
    std::string model_path;
    std::string backend = "auto";
    std::string debug_stage;
    int threads = 8;
    bool native_attention = true;
};

struct MeshDecoderRunResult {
    std::vector<float> features;            // [channels, token_count]
    std::vector<int32_t> coordinates;       // [4, token_count]
    int64_t channels = 0;
    int64_t token_count = 0;
    int resolution = 0;
    std::string backend_name;
};

// Runs SLatMeshDecoder and returns the feature support selected by
// `debug_stage` (or the final 256^3 FlexiCubes support when empty).
bool run_mesh_decoder(const float* input_features, const int32_t* input_coordinates,
                      int64_t input_token_count, const MeshDecoderRunOptions& options,
                      MeshDecoderRunResult& result, std::string& error);

}  // namespace sam3d
