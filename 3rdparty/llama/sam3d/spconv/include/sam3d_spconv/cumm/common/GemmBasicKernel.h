#pragma once
#include <tensorview/gemm/arch/memory.h>
#include <tensorview/gemm/arch/transpose.h>
#include <tensorview/gemm/arch/semaphore.h>
#include <sam3d_spconv/cumm/common/GemmBasic.h>
namespace sam3d_spconv {
namespace cumm {
namespace common {
using GemmBasic = sam3d_spconv::cumm::common::GemmBasic;
struct GemmBasicKernel {
};
} // namespace common
} // namespace cumm
} // namespace sam3d_spconv