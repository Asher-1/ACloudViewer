#pragma once

#include <string>

namespace aicore {
namespace depth {

// Cross-platform default directory for cached DA3 GGUF models.
// Same data root as cloudViewer::data::LocateDataRoot():
//   CLOUDVIEWER_DATA_ROOT/extract/da3_models
//   or ~/cloudViewer_data/extract/da3_models
std::string default_model_cache_dir();

} // namespace depth
} // namespace aicore
