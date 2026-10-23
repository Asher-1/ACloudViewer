#include <sam3d_spconv/sam3d/spconv/ConvMainUnitTest.h>
#include <sam3d_spconv/cumm/common/TensorViewKernel.h>
namespace sam3d_spconv {
namespace sam3d {
namespace spconv {
using TensorView = sam3d_spconv::cumm::common::TensorView;
using GemmBasic = sam3d_spconv::cumm::common::GemmBasic;
using GemmBasicHost = sam3d_spconv::cumm::common::GemmBasicHost;
using ConvNVRTCParams = sam3d_spconv::cumm::conv::kernel::ConvNVRTCParams;
using CummNVRTCLib = sam3d_spconv::cumm::common::CummNVRTCLib;
using TensorViewKernel = sam3d_spconv::cumm::common::TensorViewKernel;
void ConvMainUnitTest::implicit_gemm2(tv::gemm::ConvParams params)   {

  auto& algo_desp = params.conv_algo_desp;
  if (algo_desp.algo == "Ampere"&& static_cast<int>(algo_desp.shuffle_type) == 0){
    if (algo_desp.dtype_a == tv::DType(7) && algo_desp.dtype_b == tv::DType(7) && algo_desp.dtype_c == tv::DType(7)){
      return matmul_split_Ampere_f16f16f16_0(params);
    }
  }
  TV_THROW_RT_ERR("can't find any suitable algo for your parameters.",
      algo_desp.algo, algo_desp.dtype_a, algo_desp.dtype_b, algo_desp.dtype_c,
      algo_desp.__repr__());
}
} // namespace spconv
} // namespace sam3d
} // namespace sam3d_spconv