#pragma once
#include <sam3d_spconv/cumm/common/TensorViewNVRTC.h>
namespace sam3d_spconv {
namespace sam3d {
namespace spconv {
namespace Ampere_f16f16f16f16f16tnt_m128n64k32m64n32k32A1T1688_200_C301LLL_SK {
namespace out_smem_storage {
using TensorViewNVRTC = sam3d_spconv::cumm::common::TensorViewNVRTC;
struct OutputSmemStorage {
  tv::alignedarray<tv::half_t, 1152, 16> smem;
};
} // namespace out_smem_storage
} // namespace Ampere_f16f16f16f16f16tnt_m128n64k32m64n32k32A1T1688_200_C301LLL_SK
} // namespace spconv
} // namespace sam3d
} // namespace sam3d_spconv