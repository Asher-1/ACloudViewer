// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Shared tolerance policy for the model-free GEMM/convolution oracles
// (consumers: test_instantmesh_vae_convolution [also the UNET_CONV target],
// test_instantmesh_dino_convolution). The oracles verify task-side graph
// construction — layout, padding, type routing — against a double-precision
// reference; the GEMM arithmetic itself is a backend contract.
//
// The stock Metal kernel_mul_mm loads every operand through F16 simdgroup
// fragments (simdgroup_half8x8 in mul_mm.metal), so an F32 x F32 GEMM on
// Metal carries a per-operand ~2^-11 rounding that the CPU/CUDA/Vulkan F32
// paths do not have; F16-weight GEMMs stay exact on every backend. The
// per-backend bounds below therefore separate construction error (2e-5
// everywhere a backend computes F32 GEMMs in F32) from platform GEMM
// arithmetic (Metal F32 path, measured 2.7e-3 for the oracle shapes on an
// M2 Max — the 1e-2 bound keeps ~3.7x headroom over the deterministic
// kernel rounding).
#pragma once

#include "ggml-backend.h"
#include "ggml.h"

#include <cstring>

namespace aicore_test {

// ggml names Metal devices "MTL0" (and the family "Metal"), never a plain
// "Metal" backend string — match both, as the AICore graph builders do.
inline bool is_metal_backend(ggml_backend_t backend) {
    if (!backend) return false;
    const char* name = ggml_backend_dev_name(ggml_backend_get_device(backend));
    return name && (std::strstr(name, "Metal") || std::strstr(name, "MTL"));
}

// gemm_f32 selects the tolerance for a path whose GEMM multiplies two F32
// operands; F16-weight paths keep the construction-level bound everywhere.
inline double oracle_max_abs_tolerance(ggml_backend_t backend, bool gemm_f32) {
    if (gemm_f32 && is_metal_backend(backend)) return 1e-2;
    return 2e-5;
}

}  // namespace aicore_test
