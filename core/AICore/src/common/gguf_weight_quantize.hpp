// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <string>

namespace aicore {
namespace common {

/** Quantize CNN / conv weight tensors (*_weight, ndim>=2) to f16 or q8_0. */
bool quantize_gguf_weights(const std::string& input_gguf,
                           const std::string& output_gguf,
                           const std::string& type_name,
                           std::string* error = nullptr);

}  // namespace common
}  // namespace aicore
