#pragma once
#include <sam3d_spconv/cumm/common/TensorViewNVRTC.h>
#include <sam3d_spconv/cumm/common/GemmBasic.h>
#include <sam3d_spconv/cumm/conv/kernel/ConvUtils.h>
#include <sam3d_spconv/cumm/gemm/utils/GemmUtilsCPU.h>
#include <sam3d_spconv/sam3d/spconv/cpAmpere_f16f16f16f16f16tnt_m128n64k32m64n32k32A1T1688_200_C301LLL_SK/cp/ConvProblem.h>
#include <sam3d_spconv/sam3d/spconv/cpAmpere_f16f16f16f16f16tnt_m128n64k32m64n32k32A1T1688_200_C301LLL_SK/itera_p/SparseParams.h>
#include <sam3d_spconv/sam3d/spconv/cpAmpere_f16f16f16f16f16tnt_m128n64k32m64n32k32A1T1688_200_C301LLL_SK/iterb_p/WeightOptParams.h>
#include <sam3d_spconv/sam3d/spconv/cpAmpere_f16f16f16f16f16tnt_m128n64k32m64n32k32A1T1688_200_C301LLL_SK/out_params_ns/OutIteratorParams.h>
#include <sam3d_spconv/sam3d/spconv/cpAmpere_f16f16f16f16f16tnt_m128n64k32m64n32k32A1T1688_200_C301LLL_SK/out_params_scalebias_ns/OutIteratorParams.h>
#include <sam3d_spconv/sam3d/spconv/cpAmpere_f16f16f16f16f16tnt_m128n64k32m64n32k32A1T1688_200_C301LLL_SK/la/TensorGeneric.h>
#include <sam3d_spconv/sam3d/spconv/cpAmpere_f16f16f16f16f16tnt_m128n64k32m64n32k32A1T1688_200_C301LLL_SK/lb/TensorGeneric.h>
#include <sam3d_spconv/sam3d/spconv/cpAmpere_f16f16f16f16f16tnt_m128n64k32m64n32k32A1T1688_200_C301LLL_SK/lc/TensorGeneric.h>
namespace sam3d_spconv {
namespace sam3d {
namespace spconv {
namespace cpAmpere_f16f16f16f16f16tnt_m128n64k32m64n32k32A1T1688_200_C301LLL_SK {
using TensorViewNVRTC = sam3d_spconv::cumm::common::TensorViewNVRTC;
using GemmBasic = sam3d_spconv::cumm::common::GemmBasic;
using ConvUtils = sam3d_spconv::cumm::conv::kernel::ConvUtils;
using GemmUtilsCPU = sam3d_spconv::cumm::gemm::utils::GemmUtilsCPU;
using ConvProblem = sam3d_spconv::sam3d::spconv::cpAmpere_f16f16f16f16f16tnt_m128n64k32m64n32k32A1T1688_200_C301LLL_SK::cp::ConvProblem;
using IterAParams = sam3d_spconv::sam3d::spconv::cpAmpere_f16f16f16f16f16tnt_m128n64k32m64n32k32A1T1688_200_C301LLL_SK::itera_p::SparseParams;
using IterBParams = sam3d_spconv::sam3d::spconv::cpAmpere_f16f16f16f16f16tnt_m128n64k32m64n32k32A1T1688_200_C301LLL_SK::iterb_p::WeightOptParams;
using OutIterParams = sam3d_spconv::sam3d::spconv::cpAmpere_f16f16f16f16f16tnt_m128n64k32m64n32k32A1T1688_200_C301LLL_SK::out_params_ns::OutIteratorParams;
using OutIterParamsScaleBias = sam3d_spconv::sam3d::spconv::cpAmpere_f16f16f16f16f16tnt_m128n64k32m64n32k32A1T1688_200_C301LLL_SK::out_params_scalebias_ns::OutIteratorParams;
using LayoutA = sam3d_spconv::sam3d::spconv::cpAmpere_f16f16f16f16f16tnt_m128n64k32m64n32k32A1T1688_200_C301LLL_SK::la::TensorGeneric;
using LayoutB = sam3d_spconv::sam3d::spconv::cpAmpere_f16f16f16f16f16tnt_m128n64k32m64n32k32A1T1688_200_C301LLL_SK::lb::TensorGeneric;
using LayoutC = sam3d_spconv::sam3d::spconv::cpAmpere_f16f16f16f16f16tnt_m128n64k32m64n32k32A1T1688_200_C301LLL_SK::lc::TensorGeneric;
struct ConvParams {
  ConvProblem problem;
  int m;
  int n;
  int k;
  int gemm_k_iterations;
  const tv::half_t* ptr_A;
  const tv::half_t* ptr_B;
  tv::half_t* ptr_C;
  const tv::half_t* ptr_D;
  const uint32_t* mask_ptr;
  uint32_t* mask_out_ptr;
  uint32_t mask_filter;
  bool reverse_mask;
  tv::half_t alpha;
  tv::half_t beta;
  tv::half_t act_alpha;
  tv::half_t act_beta;
  tv::gemm::Activation act_type;
  dim3 grid_dims;
  IterAParams itera_params_;
  IterBParams iterb_params_;
  OutIterParams out_params_;
  OutIterParams out_params_source_;
  /**
   * @param problem
   * @param A
   * @param B
   * @param C
   * @param D
   * @param mask_ptr
   * @param mask_argsort_ptr
   * @param indice_ptr
   * @param mask_out_ptr
   * @param mask_filter
   * @param reverse_mask
   * @param alpha
   * @param beta
   * @param act_alpha
   * @param act_beta
   * @param act_type
   * @param split_k_slices
   * @param d_is_bias
   */
  __host__ __device__  ConvParams(ConvProblem problem, const tv::half_t* A, const tv::half_t* B, tv::half_t* C, const tv::half_t* D, const uint32_t* mask_ptr, const int* mask_argsort_ptr, const int* indice_ptr, uint32_t* mask_out_ptr, uint32_t mask_filter, bool reverse_mask, tv::half_t alpha = tv::half_t(1), tv::half_t beta = tv::half_t(0), tv::half_t act_alpha = tv::half_t(0), tv::half_t act_beta = tv::half_t(0), tv::gemm::Activation act_type = tv::gemm::Activation::kNone, int split_k_slices = 1, bool d_is_bias = false);
};
} // namespace cpAmpere_f16f16f16f16f16tnt_m128n64k32m64n32k32A1T1688_200_C301LLL_SK
} // namespace spconv
} // namespace sam3d
} // namespace sam3d_spconv