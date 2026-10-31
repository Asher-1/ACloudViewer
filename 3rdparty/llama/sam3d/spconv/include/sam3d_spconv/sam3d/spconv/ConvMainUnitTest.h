#pragma once
#include <sam3d_spconv/cumm/common/TensorView.h>
#include <sam3d_spconv/cumm/common/GemmBasic.h>
#include <sam3d_spconv/cumm/common/GemmBasicHost.h>
#include <sam3d_spconv/cumm/conv/kernel/ConvNVRTCParams.h>
#include <sam3d_spconv/cumm/common/CummNVRTCLib.h>
namespace sam3d_spconv {
namespace sam3d {
namespace spconv {
using TensorView = sam3d_spconv::cumm::common::TensorView;
using GemmBasic = sam3d_spconv::cumm::common::GemmBasic;
using GemmBasicHost = sam3d_spconv::cumm::common::GemmBasicHost;
using ConvNVRTCParams = sam3d_spconv::cumm::conv::kernel::ConvNVRTCParams;
using CummNVRTCLib = sam3d_spconv::cumm::common::CummNVRTCLib;
struct ConvMainUnitTest {
  /**
   * @param params
   */
  static void matmul_split_Ampere_f16f16f16_0(tv::gemm::ConvParams params);
  /**
   * @param op_type
   * @param N
   * @param C
   * @param K
   * @param kernel_volume
   * @param in_prod
   * @param out_prod
   * @param mask_sparse
   */
  static std::array<int, 3> extract_mnk(int op_type, int N, int C, int K, int kernel_volume, int in_prod, int out_prod, bool mask_sparse);
  /**
   * @param params
   */
  static void implicit_gemm2(tv::gemm::ConvParams params);

  static std::vector<tv::gemm::ConvAlgoDesp> get_all_conv_algo_desp();
};
} // namespace spconv
} // namespace sam3d
} // namespace sam3d_spconv