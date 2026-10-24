// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <string>

namespace aicore {
namespace sam3 {

/**
 * Quantize a SAM3 / SAM2 GGUF from F32/F16 to q4_0 / q4_1 / q8_0.
 *
 * Only matmul (2D) weights whose leading dimension is block-aligned and whose
 * name is not an embedding / bias / norm parameter are quantized — the same
 * rule set the upstream sam3-ggml examples/quantize.cpp applies (which mirrors
 * the register_* macros).  All metadata (arch, hparams, tokenizer) is copied
 * through verbatim and the `sam3.ftype` KV is updated so the output loads
 * identically through aicore_sam3_load_opts / sam3_load_model.
 *
 * Supported types: "q4_0", "q4_1", "q8_0".
 *
 * @return true on success, false on failure (error logged via AICORE_LOG).
 */
bool quantize_gguf(const std::string& input_gguf,
                   const std::string& output_gguf,
                   const std::string& type_name);

}  // namespace sam3
}  // namespace aicore