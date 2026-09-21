#pragma once
namespace sam3d_spconv {
namespace sam3d {
namespace spconv {
namespace Ampere_f16f16f16f16f16tnt_m128n64k32m64n32k32A1T1688_200_C301LLL_SK {
namespace gemm_smem_storage {
struct BlockMmaStorage {
  tv::alignedarray<tv::half_t, 8192, 16> smem_A;
  tv::alignedarray<tv::half_t, 4096, 16> smem_B;
};
} // namespace gemm_smem_storage
} // namespace Ampere_f16f16f16f16f16tnt_m128n64k32m64n32k32A1T1688_200_C301LLL_SK
} // namespace spconv
} // namespace sam3d
} // namespace sam3d_spconv