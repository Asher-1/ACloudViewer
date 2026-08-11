// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "controllers/da3_depth_controller.h"

namespace colmap {

// DA3 inference uses AICore (ggml Metal/CUDA/OpenCL/CPU), not COLMAP CUDA.
// When AICore is linked but COLMAP PatchMatch stereo is unavailable, prefer DA3
// defaults for sparse/dense instead of COLMAP PatchMatch.
inline constexpr bool PreferDA3OverColmapPatchMatch() {
#if defined(AICore_ENABLED) && !defined(CUDA_ENABLED)
    return true;
#else
    return false;
#endif
}

inline constexpr bool DefaultDenseReconstructionEnabled() {
#if defined(CUDA_ENABLED)
    return true;
#elif defined(AICore_ENABLED)
    return true;
#else
    return false;
#endif
}

// Runtime stereo mode: COLMAP PatchMatch needs CUDA; with AICore only, route
// dense through DA3 depth + fusion so the automatic pipeline completes on
// CPU/OpenCL.
inline StereoPipelineMode EffectiveStereoPipelineMode(
        StereoPipelineMode requested) {
#if defined(AICore_ENABLED) && !defined(CUDA_ENABLED)
    if (requested == StereoPipelineMode::COLMAP_PATCH_MATCH) {
        return StereoPipelineMode::DA3_DEPTH_INFERENCE;
    }
#endif
    return requested;
}

}  // namespace colmap
