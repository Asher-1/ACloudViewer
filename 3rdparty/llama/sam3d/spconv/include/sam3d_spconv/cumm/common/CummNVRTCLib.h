#pragma once
#include <tensorview/cuda/nvrtc.h>
#include <tensorview/gemm/core/nvrtc_bases.h>
#include <sam3d_spconv/cumm/common/TensorView.h>
namespace sam3d_spconv {
namespace cumm {
namespace common {
using TensorView = sam3d_spconv::cumm::common::TensorView;
struct CummNVRTCLib {
};
} // namespace common
} // namespace cumm
} // namespace sam3d_spconv