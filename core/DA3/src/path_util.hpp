#pragma once

#include <string>

namespace da {

// Cross-platform default directory for cached DA3 GGUF models.
// Priority: CLOUDVIEWER_DATA_ROOT/extract/da3_models
//   -> ~/cloudViewer_data/extract/da3_models (HOME / USERPROFILE)
//   -> %LOCALAPPDATA%/cloudViewer/da3_models (Windows)
//   -> temp_directory/cloudViewer_da3_models
std::string default_model_cache_dir();

}  // namespace da
