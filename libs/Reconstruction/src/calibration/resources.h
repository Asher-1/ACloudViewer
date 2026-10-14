// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <string>

namespace colmap {

// AnyCalib inference runs through the AICore anycalib task (libAICore ggml
// graph). The published GGUF models are owned by the AICore task catalog
// (aicore_anycalib_model_at / aicore_anycalib_capi.h) and resolve through
// its cross-platform cache; an empty default means "use the catalog's
// recommended variant".
inline const std::string kDefaultAnyCalibGenUri = "";

}  // namespace colmap
