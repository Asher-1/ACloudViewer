#pragma once

#include <string>

namespace aicore {
namespace gaussian {

// Cross-platform default directory for cached FreeSplatter GGUF models.
// Same data root as cloudViewer::data::LocateDataRoot():
//   CLOUDVIEWER_DATA_ROOT/extract/freesplatter_models
//   or ~/cloudViewer_data/extract/freesplatter_models
std::string default_model_cache_dir();

} // namespace gaussian
} // namespace aicore
