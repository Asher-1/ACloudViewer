#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace relateanything {
struct PairInput {
    std::size_t object_count = 0;
    std::size_t feature_dim = 0;
    std::size_t feature_stride = 0;
    std::vector<float> object_features;
    std::vector<float> boxes_cxcywh;
};

struct PairOutput {
    std::size_t pair_count = 0;
    std::size_t predicate_count = 0;
    std::vector<float> predicate_logits;
    std::vector<float> pair_logits;
    std::vector<std::int32_t> subject_indices;
    std::vector<std::int32_t> object_indices;
    double milliseconds = 0.0;
};

} // namespace relateanything
