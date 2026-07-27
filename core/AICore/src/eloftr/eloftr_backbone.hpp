#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace eloftr {

using RepVggTensorMap = std::unordered_map<std::string, std::vector<float>>;

bool LoadRepVggGguf(const std::string &path, RepVggTensorMap *tensors, std::string *error);

// input_nchw: [1,H,W] row-major; output feat_c_nchw in ggml WHCN slice order (W,H,C).
bool RunRepVggBackbone(const RepVggTensorMap &weights, const std::vector<float> &input_nchw,
                       int32_t h, int32_t w, const std::string &device,
                       std::vector<float> *feat_c_nchw, int32_t *out_h, int32_t *out_w,
                       int32_t *out_c, std::string *error);

} // namespace eloftr
