#pragma once

#include "common/data_root_util.hpp"


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

inline std::string deeplsd_model_cache_dir() {
    return extract_model_dir("deeplsd_models");
}

inline std::string aliked_model_cache_dir() {
    // ALIKED extractor models share the same cache as LightGlue matcher models.
    return extract_model_dir("lightglue_models");
}

inline std::string facedetect_model_cache_dir() {
    return extract_model_dir("facedetect_models");
}

inline std::string rfdetr_model_cache_dir() {
    return extract_model_dir("rfdetr_models");
}

inline std::string rmbg_model_cache_dir() {
    return extract_model_dir("rmbg_models");
}

inline std::string yolo_model_cache_dir() {
    return extract_model_dir("yolo_models");
}

}  // namespace aicore
