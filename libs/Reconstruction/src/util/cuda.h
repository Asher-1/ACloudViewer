// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef COLMAP_SRC_UTIL_CUDA_H_
#define COLMAP_SRC_UTIL_CUDA_H_

namespace colmap {

int GetNumCudaDevices();

void SetBestCudaDevice(const int gpu_index);

}  // namespace colmap

#endif  // COLMAP_SRC_UTIL_CUDA_H_
