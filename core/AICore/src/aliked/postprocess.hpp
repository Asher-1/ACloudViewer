#pragma once

#include <cstdint>
#include <utility>
#include <vector>

namespace lightglue::aliked_internal {

struct DkdOptions {
  int32_t radius = 2;
  int32_t top_k = 0;
  float scores_th = 0.2f;
  int32_t n_limit = 20000;
};

struct DkdOutput {
  std::vector<float> keypoints_norm; // interleaved x,y in [-1,1]
  std::vector<float> scores;
};

DkdOutput RunDkd(const std::vector<float> &score_map, int32_t h, int32_t w,
                 const DkdOptions &options, int32_t image_width,
                 int32_t image_height);

std::vector<float>
RunSddh(const std::vector<float> &feature_map, int32_t dim, int32_t h, int32_t w,
        const std::vector<float> &keypoints_norm, int32_t kernel_size, int32_t n_pos,
        const std::vector<float> &offset_0_w, const std::vector<float> &offset_0_b,
        const std::vector<float> &offset_2_w, const std::vector<float> &offset_2_b,
        const std::vector<float> &sf_conv_w, const std::vector<float> &agg_weights);

} // namespace lightglue::aliked_internal
