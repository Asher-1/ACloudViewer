// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "model_cache.hpp"

namespace aicore {
namespace gaussian {

std::string default_model_cache_dir() {
    return aicore::gaussian_model_cache_dir();
}

}  // namespace gaussian
}  // namespace aicore
