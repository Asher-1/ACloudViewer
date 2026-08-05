#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace deeplsd {

using TensorMap = std::unordered_map<std::string, std::vector<float>>;

bool LoadGguf(const std::string &path, TensorMap *tensors, std::string *error);

const std::vector<float> *FindTensor(const TensorMap &tensors, const std::string &key);

} // namespace deeplsd
