#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace lightglue::aliked_internal {

using TensorMap = std::unordered_map<std::string, std::vector<float>>;

bool LoadAlikedTensors(const std::string &path, TensorMap *tensors,
                       int32_t *descriptor_dim, std::string *error);

inline const std::vector<float> &RequireTensor(const TensorMap &tensors,
                                               const std::string &name,
                                               std::string *error) {
  static const std::vector<float> kEmpty;
  const auto it = tensors.find(name);
  if (it == tensors.end()) {
    *error = "missing tensor: " + name;
    return kEmpty;
  }
  return it->second;
}

} // namespace lightglue::aliked_internal
