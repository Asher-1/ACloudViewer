// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Image-input ingest + validation. This is the UNTRUSTED surface (fuzzed): all
// bounds/finiteness checks on caller-supplied pixels live here, so the graph
// downstream can assume well-formed inputs. (The GGUF model file, by contrast,
// is trusted and not hardened.)
#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "gguf_loader.h"

namespace aicore {
namespace gaussian {

// Validate caller pixels against the model geometry. images must hold
// n_views * in_channels * height * width floats. Returns false + sets err on
// any shape/size mismatch, overflow, or non-finite value. On success, `out`
// holds a defensively-copied, finite, view-major NCHW f32 buffer.
bool ingest_images(const hparams& hp,
                   const float* images,
                   int32_t n_views,
                   int32_t height,
                   int32_t width,
                   std::vector<float>& out,
                   std::string& err);

}  // namespace gaussian
}  // namespace aicore
