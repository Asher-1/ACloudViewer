#include <sam3d_spconv/sam3d/spconv/ConvMainUnitTest.h>
namespace sam3d_spconv {
namespace sam3d {
namespace spconv {
using TensorView = sam3d_spconv::cumm::common::TensorView;
using GemmBasic = sam3d_spconv::cumm::common::GemmBasic;
using GemmBasicHost = sam3d_spconv::cumm::common::GemmBasicHost;
using ConvNVRTCParams = sam3d_spconv::cumm::conv::kernel::ConvNVRTCParams;
using CummNVRTCLib = sam3d_spconv::cumm::common::CummNVRTCLib;
std::array<int, 3> ConvMainUnitTest::extract_mnk(int op_type, int N, int C, int K, int kernel_volume, int in_prod, int out_prod, bool mask_sparse)   {

  auto op_type_enum = static_cast<tv::gemm::ConvOpType>(op_type);
  auto res = tv::gemm::implicit_gemm_mnk(op_type_enum, N, C, K,
      kernel_volume, in_prod, out_prod, mask_sparse);
  return {res[0], res[1], res[2]};
}
} // namespace spconv
} // namespace sam3d
} // namespace sam3d_spconv