#include <sam3d_spconv/sam3d/spconv/ConvMainUnitTest.h>
namespace sam3d_spconv {
namespace sam3d {
namespace spconv {
using TensorView = sam3d_spconv::cumm::common::TensorView;
using GemmBasic = sam3d_spconv::cumm::common::GemmBasic;
using GemmBasicHost = sam3d_spconv::cumm::common::GemmBasicHost;
using ConvNVRTCParams = sam3d_spconv::cumm::conv::kernel::ConvNVRTCParams;
using CummNVRTCLib = sam3d_spconv::cumm::common::CummNVRTCLib;
std::vector<tv::gemm::ConvAlgoDesp> ConvMainUnitTest::get_all_conv_algo_desp()   {

  std::vector<tv::gemm::ConvAlgoDesp> desps;
  {
  tv::gemm::ConvAlgoDesp desp(3, tv::gemm::ConvOpType(0));
  desp.dtype_a = 7;
  desp.dtype_b = 7;
  desp.dtype_c = 7;
  desp.dacc = 7;
  desp.dcomp = 7;
  desp.trans_a_set(false);
  desp.trans_b_set(true);
  desp.trans_c_set(false);
  desp.tile_shape = {128, 64, 32};
  desp.warp_tile_shape = {64, 32, 32};
  desp.tensorop = {16, 8, 8};
  desp.num_stage = 2;
  desp.algo = "Ampere";
  desp.split_k_serial_set(false);
  desp.split_k_parallel_set(false);
  desp.shuffle_type = static_cast<tv::gemm::ShuffleStrideType>(0);
  desp.element_per_access_a = 8;
  desp.element_per_access_b = 8;
  desp.element_per_access_c = 8;
  desp.access_per_vector = 1;
  desp.min_arch = std::make_tuple(8, 0);
  // Conv attrs
  desp.ndim = 3;
  desp.op_type = static_cast<tv::gemm::ConvOpType>(0);
  desp.iter_algo = static_cast<tv::gemm::ConvIterAlgo>(1);
  desp.layout_i = static_cast<tv::gemm::ConvLayoutType>(1);
  desp.layout_w = static_cast<tv::gemm::ConvLayoutType>(1);
  desp.layout_o = static_cast<tv::gemm::ConvLayoutType>(1);
  desp.interleave_i = 1;
  desp.interleave_w = 1;
  desp.interleave_o = 1;
  desp.mask_sparse = true;
  desp.increment_k_first = true;
  desp.is_int8_inference = false;
  desp.dynamic_mask = false;
  TV_ASSERT_RT_ERR(desp.__repr__() == "Ampere_f16f16f16f16f16tnt_m128n64k32m64n32k32A1T1688_200_C301LLL_SK", "error", desp.__repr__(), "Ampere_f16f16f16f16f16tnt_m128n64k32m64n32k32A1T1688_200_C301LLL_SK");
  desps.push_back(desp);
  }
  return desps;
}
} // namespace spconv
} // namespace sam3d
} // namespace sam3d_spconv