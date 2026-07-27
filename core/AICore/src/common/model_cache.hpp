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

inline std::string eloftr_model_cache_dir() {
    return extract_model_dir("eloftr_models");
}

inline std::string deeplsd_model_cache_dir() {
    return extract_model_dir("deeplsd_models");
}

inline std::string aliked_model_cache_dir() {
    return extract_model_dir("aliked_models");
}

inline std::string matchanything_model_cache_dir() {
    return extract_model_dir("matchanything_models");
}

}  // namespace aicore
