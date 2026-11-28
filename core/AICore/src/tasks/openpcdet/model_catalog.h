// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <string>

namespace openpcdet {

// Per-checkpoint visualization defaults (detection score + rotated-NMS IoU)
// used to seed UI controls and the no-option inference fallback. They are
// tuned per head family for clean first-open results; the GGUF-declared
// values remain the official evaluation thresholds (read_model_info) and
// the accuracy probe pins those, so these never shift the upstream gate.
struct OpenPCDetVisDefaults {
    float score_thresh;
    float nms_thresh;
};

// Matches a catalog GGUF filename ("<stem>.<quant>.gguf") against the
// per-checkpoint table. Returns false for non-catalog files so the caller
// falls back to the global product default. Task-internal (not C ABI).
bool openpcdet_visualization_defaults(const std::string& gguf_filename,
                                      OpenPCDetVisDefaults* out);

}  // namespace openpcdet
