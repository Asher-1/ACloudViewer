// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "gguf_weight_quantize.hpp"

#include "simple_gguf_io.hpp"

namespace aicore {
namespace common {

bool quantize_gguf_weights(const std::string& input_gguf,
                           const std::string& output_gguf,
                           const std::string& type_name,
                           std::string* error) {
    return quantize_simple_gguf_weights(input_gguf, output_gguf, type_name,
                                        error);
}

}  // namespace common
}  // namespace aicore
