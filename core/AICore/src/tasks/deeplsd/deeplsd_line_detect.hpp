#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "deeplsd.hpp"

namespace deeplsd {

/** AFM-guided LSD post-process (same pipeline as DeepLSD-GGML validation). */
bool DetectAfmLines(const uint8_t* gray,
                    int32_t width,
                    int32_t height,
                    int32_t row_stride,
                    const float* df_norm,
                    const float* angle_norm,
                    std::vector<LineSegment>* segments,
                    std::string* error = nullptr);

}  // namespace deeplsd
