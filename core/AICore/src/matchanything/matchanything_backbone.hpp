#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace matchanything {

using TensorMap = std::unordered_map<std::string, std::vector<float>>;

struct BackboneOutput {
  std::vector<float> feat_c; // NCHW @ 1/8
  std::vector<float> feat_x2; // NCHW @ 1/4 (inter_feat)
  std::vector<float> feat_x1; // NCHW @ 1/2 (inter_feat)
  int32_t hc = 0;
  int32_t wc = 0;
  int32_t cc = 0;
};

bool LoadMatchAnythingGguf(const std::string &path, TensorMap *tensors,
                           std::string *error);

/** RepVGG-A1 backbone with inter_feat outputs (MatchAnything eloftr config). */
bool RunRepVggInterBackbone(const TensorMap &weights, const std::vector<float> &input_nchw,
                            int32_t h, int32_t w, const std::string &device,
                            BackboneOutput *out, std::string *error);

} // namespace matchanything
