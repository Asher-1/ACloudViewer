// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace aicore {
namespace common {

using SimpleFloatMap = std::unordered_map<std::string, std::vector<float>>;

/** Load simplified CNN GGUF (export scripts) into float tensors. */
bool load_simple_gguf_f32(const std::string& path, SimpleFloatMap* tensors,
                          std::string* error = nullptr);

/** Quantize *_weight tensors; writes simplified GGUF with dtype 0/1/2. */
bool quantize_simple_gguf_weights(const std::string& input_gguf,
                                    const std::string& output_gguf,
                                    const std::string& type_name,
                                    std::string* error = nullptr);

}  // namespace common
}  // namespace aicore
