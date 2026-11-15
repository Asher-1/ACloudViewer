#pragma once
#include <sam3d_spconv/cumm/common/TensorViewNVRTCKernel.h>
#include <sam3d_spconv/cumm/gemm/layout/RowMajor.h>
#include <sam3d_spconv/cumm/gemm/layout/ColumnMajor.h>
#include <sam3d_spconv/cumm/common/GemmBasicKernel.h>
#include <sam3d_spconv/cumm/common/GemmBasic.h>
#include <sam3d_spconv/cumm/common/GemmKernelFlags.h>
#include <sam3d_spconv/sam3d/spconv/Ampere_f16f16f16f16f16tnt_m128n64k32m64n32k32A1T1688_200_C301LLL_SK/inpitera/ForwardDgradSparseIOIterator.h>
#include <sam3d_spconv/sam3d/spconv/Ampere_f16f16f16f16f16tnt_m128n64k32m64n32k32A1T1688_200_C301LLL_SK/inpiterb/WeightIteratorDP4A.h>
#include <sam3d_spconv/sam3d/spconv/cpAmpere_f16f16f16f16f16tnt_m128n64k32m64n32k32A1T1688_200_C301LLL_SK/la/TensorGeneric.h>
#include <sam3d_spconv/sam3d/spconv/cpAmpere_f16f16f16f16f16tnt_m128n64k32m64n32k32A1T1688_200_C301LLL_SK/lb/TensorGeneric.h>
#include <sam3d_spconv/sam3d/spconv/cpAmpere_f16f16f16f16f16tnt_m128n64k32m64n32k32A1T1688_200_C301LLL_SK/lc/TensorGeneric.h>
#include <sam3d_spconv/sam3d/spconv/Ampere_f16f16f16f16f16tnt_m128n64k32m64n32k32A1T1688_200_C301LLL_SK/gemm_smem_storage/BlockMmaStorage.h>
#include <sam3d_spconv/sam3d/spconv/Ampere_f16f16f16f16f16tnt_m128n64k32m64n32k32A1T1688_200_C301LLL_SK/out_smem_storage/OutputSmemStorage.h>
#include <sam3d_spconv/sam3d/spconv/cpAmpere_f16f16f16f16f16tnt_m128n64k32m64n32k32A1T1688_200_C301LLL_SK/ConvParams.h>
#include <sam3d_spconv/sam3d/spconv/cpAmpere_f16f16f16f16f16tnt_m128n64k32m64n32k32A1T1688_200_C301LLL_SK/cp/ConvProblem.h>
#include <sam3d_spconv/sam3d/spconv/Ampere_f16f16f16f16f16tnt_m128n64k32m64n32k32A1T1688_200_C301LLL_SK/out_iter/OutIterator.h>
#include <sam3d_spconv/sam3d/spconv/Ampere_f16f16f16f16f16tnt_m128n64k32m64n32k32A1T1688_200_C301LLL_SK/out_iter_const/OutIterator.h>
#include <sam3d_spconv/sam3d/spconv/Ampere_f16f16f16f16f16tnt_m128n64k32m64n32k32A1T1688_200_C301LLL_SK/out_op/LinearCombination.h>
#include <sam3d_spconv/sam3d/spconv/Ampere_f16f16f16f16f16tnt_m128n64k32m64n32k32A1T1688_200_C301LLL_SK/mma/MmaMultiStage.h>
#include <sam3d_spconv/sam3d/spconv/Ampere_f16f16f16f16f16tnt_m128n64k32m64n32k32A1T1688_200_C301LLL_SK/mma_miterd/MaskIGemmIteratorMaskLoaderDynamic.h>
#include <sam3d_spconv/sam3d/spconv/Ampere_f16f16f16f16f16tnt_m128n64k32m64n32k32A1T1688_200_C301LLL_SK/output/Output.h>
namespace sam3d_spconv {
namespace sam3d {
namespace spconv {
namespace Ampere_f16f16f16f16f16tnt_m128n64k32m64n32k32A1T1688_200_C301LLL_SK {
using TensorViewNVRTCKernel = sam3d_spconv::cumm::common::TensorViewNVRTCKernel;
using RowMajor = sam3d_spconv::cumm::gemm::layout::RowMajor;
using ColumnMajor = sam3d_spconv::cumm::gemm::layout::ColumnMajor;
using GemmBasicKernel = sam3d_spconv::cumm::common::GemmBasicKernel;
using GemmBasic = sam3d_spconv::cumm::common::GemmBasic;
using GemmKernelFlags = sam3d_spconv::cumm::common::GemmKernelFlags;
using InputIteratorA = sam3d_spconv::sam3d::spconv::Ampere_f16f16f16f16f16tnt_m128n64k32m64n32k32A1T1688_200_C301LLL_SK::inpitera::ForwardDgradSparseIOIterator;
using InputIteratorB = sam3d_spconv::sam3d::spconv::Ampere_f16f16f16f16f16tnt_m128n64k32m64n32k32A1T1688_200_C301LLL_SK::inpiterb::WeightIteratorDP4A;
using LayoutA = sam3d_spconv::sam3d::spconv::cpAmpere_f16f16f16f16f16tnt_m128n64k32m64n32k32A1T1688_200_C301LLL_SK::la::TensorGeneric;
using LayoutB = sam3d_spconv::sam3d::spconv::cpAmpere_f16f16f16f16f16tnt_m128n64k32m64n32k32A1T1688_200_C301LLL_SK::lb::TensorGeneric;
using LayoutC = sam3d_spconv::sam3d::spconv::cpAmpere_f16f16f16f16f16tnt_m128n64k32m64n32k32A1T1688_200_C301LLL_SK::lc::TensorGeneric;
using BlockMmaStorage = sam3d_spconv::sam3d::spconv::Ampere_f16f16f16f16f16tnt_m128n64k32m64n32k32A1T1688_200_C301LLL_SK::gemm_smem_storage::BlockMmaStorage;
using OutputSmemStorage = sam3d_spconv::sam3d::spconv::Ampere_f16f16f16f16f16tnt_m128n64k32m64n32k32A1T1688_200_C301LLL_SK::out_smem_storage::OutputSmemStorage;
using ConvParams = sam3d_spconv::sam3d::spconv::cpAmpere_f16f16f16f16f16tnt_m128n64k32m64n32k32A1T1688_200_C301LLL_SK::ConvParams;
using ConvProblem = sam3d_spconv::sam3d::spconv::cpAmpere_f16f16f16f16f16tnt_m128n64k32m64n32k32A1T1688_200_C301LLL_SK::cp::ConvProblem;
using OutIter = sam3d_spconv::sam3d::spconv::Ampere_f16f16f16f16f16tnt_m128n64k32m64n32k32A1T1688_200_C301LLL_SK::out_iter::OutIterator;
using ConstOutIter = sam3d_spconv::sam3d::spconv::Ampere_f16f16f16f16f16tnt_m128n64k32m64n32k32A1T1688_200_C301LLL_SK::out_iter_const::OutIterator;
using OutputOp = sam3d_spconv::sam3d::spconv::Ampere_f16f16f16f16f16tnt_m128n64k32m64n32k32A1T1688_200_C301LLL_SK::out_op::LinearCombination;
using Mma = sam3d_spconv::sam3d::spconv::Ampere_f16f16f16f16f16tnt_m128n64k32m64n32k32A1T1688_200_C301LLL_SK::mma::MmaMultiStage;
using MaskIGemmIteratorDynamic = sam3d_spconv::sam3d::spconv::Ampere_f16f16f16f16f16tnt_m128n64k32m64n32k32A1T1688_200_C301LLL_SK::mma_miterd::MaskIGemmIteratorMaskLoaderDynamic;
using Output = sam3d_spconv::sam3d::spconv::Ampere_f16f16f16f16f16tnt_m128n64k32m64n32k32A1T1688_200_C301LLL_SK::output::Output;
/**
 * @param params
 */
__global__ void conv_kernel(ConvParams params);
struct ConvKernel {
};
} // namespace Ampere_f16f16f16f16f16tnt_m128n64k32m64n32k32A1T1688_200_C301LLL_SK
} // namespace spconv
} // namespace sam3d
} // namespace sam3d_spconv