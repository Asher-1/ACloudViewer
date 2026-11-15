// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "tasks/deeplsd/gguf_loader.hpp"

#include "common/simple_gguf_io.hpp"
#include "tasks/deeplsd/deeplsd.hpp"

namespace deeplsd {

bool LoadGguf(const std::string &path, TensorMap *tensors, std::string *error) {
    if (tensors == nullptr) {
        if (error) {
            *error = "null tensors output";
        }
        return false;
    }
    aicore::common::SimpleFloatMap loaded;
    if (!aicore::common::load_simple_gguf_f32(path, &loaded, error)) {
        return false;
    }
    *tensors = std::move(loaded);
    return true;
}

const std::vector<float> *FindTensor(const TensorMap &tensors,
                                     const std::string &key) {
    const auto it = tensors.find(key);
    if (it == tensors.end()) {
        return nullptr;
    }
    return &it->second;
}

}  // namespace deeplsd
