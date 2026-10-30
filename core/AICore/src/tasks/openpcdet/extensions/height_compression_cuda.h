#pragma once

#include <cstdint>
#include <string>

namespace openpcdet::ext {

// Scatter sparse [N,C] features with [batch,z,y,x] coordinates directly into
// a CUDA-resident dense tensor. By default the output is HWC [H,W,C*D]; set
// output_whc for GGML's contiguous [W,H,C*D] layout. The destination is zeroed.
bool gpu_height_compress(const float * sparse_features_device,
                         const int32_t * coordinates_host,
                         int32_t active_count,
                         int32_t sparse_channels,
                         int32_t depth,
                         int32_t height,
                         int32_t width,
                         float * spatial_device,
                         std::string & error,
                         bool coordinates_are_device = false,
                         bool output_whc = false);

} // namespace openpcdet::ext
