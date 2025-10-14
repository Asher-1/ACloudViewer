// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef COLMAP_SRC_MVS_GPU_MAT_PRNG_H_
#define COLMAP_SRC_MVS_GPU_MAT_PRNG_H_

#include "mvs/gpu_mat.h"

namespace colmap {
namespace mvs {

class GpuMatPRNG : public GpuMat<curandState> {
public:
    GpuMatPRNG(const int width, const int height);

private:
    void InitRandomState();
};

}  // namespace mvs
}  // namespace colmap

#endif  // COLMAP_SRC_MVS_GPU_MAT_PRNG_H_
