#pragma once

#include "data_root_util.hpp"

namespace aicore {

inline std::string depth_model_cache_dir() {
    return extract_model_dir("da3_models");
}

inline std::string gaussian_model_cache_dir() {
    return extract_model_dir("freesplatter_models");
}

inline std::string lightglue_model_cache_dir() {
    return extract_model_dir("lightglue_models");
}

}  // namespace aicore
