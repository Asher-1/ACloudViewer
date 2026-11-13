#pragma once
#include <tensorview/cuda/device_ops.h>
#include <tensorview/gemm/debug.h>
#include <sam3d_spconv/cumm/common/TensorViewNVRTC.h>
namespace sam3d_spconv {
namespace cumm {
namespace common {
using TensorViewNVRTC = sam3d_spconv::cumm::common::TensorViewNVRTC;
struct TensorViewNVRTCKernel {
};
} // namespace common
} // namespace cumm
} // namespace sam3d_spconv