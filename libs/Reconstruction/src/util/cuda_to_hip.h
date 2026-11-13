// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// Single point of CUDA <-> HIP source compatibility. MVS keeps CUDA spellings
// so one kernel implementation serves both CUDA and ROCm builds.
#if defined(COLMAP_HIP_ENABLED)

#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#include <cstdio>

using cudaError_t = hipError_t;
using cudaStream_t = hipStream_t;
using cudaEvent_t = hipEvent_t;
using cudaDeviceProp = hipDeviceProp_t;
using cudaExtent = hipExtent;
using cudaPos = hipPos;
using cudaPitchedPtr = hipPitchedPtr;
using cudaMemcpy3DParms = hipMemcpy3DParms;
using cudaArray = hipArray;
using cudaArray_t = hipArray_t;
using cudaChannelFormatDesc = hipChannelFormatDesc;
using cudaChannelFormatKind = hipChannelFormatKind;
using cudaResourceDesc = hipResourceDesc;
using cudaResourceType = hipResourceType;
using cudaTextureDesc = hipTextureDesc;
using cudaTextureObject_t = hipTextureObject_t;
using cudaTextureAddressMode = hipTextureAddressMode;
using cudaTextureFilterMode = hipTextureFilterMode;
using cudaTextureReadMode = hipTextureReadMode;
using curandState = hiprandState;

#define cudaSuccess hipSuccess
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaPeekAtLastError hipPeekAtLastError
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaEventCreate hipEventCreate
#define cudaEventDestroy hipEventDestroy
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaGetDevice hipGetDevice
#define cudaSetDevice hipSetDevice
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaMalloc hipMalloc
#define cudaMallocPitch hipMallocPitch
#define cudaMalloc3DArray hipMalloc3DArray
#define cudaFree hipFree
#define cudaFreeArray hipFreeArray
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpy2D hipMemcpy2D
#define cudaMemcpy3D hipMemcpy3D
#define cudaMemcpyToSymbol hipMemcpyToSymbol
#define cudaMemset hipMemset
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyHostToHost hipMemcpyHostToHost
#define make_cudaExtent make_hipExtent
#define make_cudaPos make_hipPos
#define make_cudaPitchedPtr make_hipPitchedPtr
#define cudaCreateChannelDesc hipCreateChannelDesc
#define cudaCreateTextureObject hipCreateTextureObject
#define cudaDestroyTextureObject hipDestroyTextureObject
#define cudaArrayDefault hipArrayDefault
#define cudaArrayLayered hipArrayLayered
#define cudaResourceTypeArray hipResourceTypeArray
#define cudaAddressModeWrap hipAddressModeWrap
#define cudaAddressModeClamp hipAddressModeClamp
#define cudaAddressModeMirror hipAddressModeMirror
#define cudaAddressModeBorder hipAddressModeBorder
#define cudaFilterModePoint hipFilterModePoint
#define cudaFilterModeLinear hipFilterModeLinear
#define cudaReadModeElementType hipReadModeElementType
#define cudaReadModeNormalizedFloat hipReadModeNormalizedFloat
#define curand_init hiprand_init
#define curand_uniform hiprand_uniform
#define curand_normal hiprand_normal

#elif defined(CUDA_ENABLED) || defined(__CUDACC__)

#include <cuda_runtime.h>
#include <curand_kernel.h>

#endif
