// ----------------------------------------------------------------------------
// -                        CloudViewer: www.erow.cn                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.erow.cn
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include "Gpu.h"

#include <Console.h>

#include <sstream>
#include <memory.h>

#ifdef BUILD_CUDA_MODULE
#include <cuda_runtime.h>
#endif

namespace cloudViewer {
namespace gpu {

bool gpuSupportCUDA(int minComputeCapabilityMajor,
    int minComputeCapabilityMinor,
    int minTotalDeviceMemory)
{
#ifdef BUILD_CUDA_MODULE
    int nbDevices = 0;
    cudaError_t success;
    success = cudaGetDeviceCount(&nbDevices);
    if( success != cudaSuccess )
    {
        utility::LogError("cudaGetDeviceCount failed: ", cudaGetErrorString(success));
        nbDevices = 0;
    }

    if(nbDevices > 0)
    {
        for(int i = 0; i < nbDevices; ++i)
        {
            cudaDeviceProp deviceProperties;

            if(cudaGetDeviceProperties(&deviceProperties, i) != cudaSuccess)
            {
                utility::LogError("Cannot get properties for CUDA gpu device {}", i);
                continue;
            }

            if((deviceProperties.major > minComputeCapabilityMajor ||
                (deviceProperties.major == minComputeCapabilityMajor &&
                    deviceProperties.minor >= minComputeCapabilityMinor)) &&
                deviceProperties.totalGlobalMem >= (minTotalDeviceMemory*1024*1024))
            {
                utility::LogInfo("Supported CUDA-Enabled GPU detected.");
                return true;
            }
            else
            {
                utility::LogError(
                    "CUDA-Enabled GPU detected, but the compute capabilities is not enough.\n - Device {} : {}.{}, global memory: {} MB\n - Requirements: {}.{}, global memory: {} MB\n",
                    i, deviceProperties.major, deviceProperties.minor, int(deviceProperties.totalGlobalMem / (1024*1024)),
                    minComputeCapabilityMajor, minComputeCapabilityMinor, minTotalDeviceMemory);
            }
        }
        utility::LogInfo("CUDA-Enabled GPU not supported.");
    }
    else
    {
        utility::LogInfo("Can't find CUDA-Enabled GPU.");
    }
#endif
    return false;
}

std::string gpuInformationCUDA()
{
    std::string information;
#ifdef BUILD_CUDA_MODULE
    int nbDevices = 0;
    if( cudaGetDeviceCount(&nbDevices) != cudaSuccess )
    {
        utility::LogWarning( "Could not determine number of CUDA cards in this system" );
        nbDevices = 0;
    }

    if(nbDevices > 0)
    {
        information = "CUDA-Enabled GPU.\n";
        for(int i = 0; i < nbDevices; ++i)
        {
            cudaDeviceProp deviceProperties;
            if(cudaGetDeviceProperties( &deviceProperties, i) != cudaSuccess )
            {
                utility::LogError("Cannot get properties for CUDA gpu device {}", i);
                continue;
            }

            if( cudaSetDevice( i ) != cudaSuccess )
            {
                utility::LogError("Device with number {} does not exist", i);
                continue;
            }

            std::size_t avail;
            std::size_t total;
            if(cudaMemGetInfo(&avail, &total) != cudaSuccess)
            {
                // if the card does not provide this information.
                avail = 0;
                total = 0;
                utility::LogWarning("Cannot get available memory information for CUDA gpu device {}.", i);
            }
            std::stringstream deviceSS;

            deviceSS << "Device information:" << std::endl
                << "\t- id:                      " << i << std::endl
                << "\t- name:                    " << deviceProperties.name << std::endl
                << "\t- compute capability:      " << deviceProperties.major << "." << deviceProperties.minor << std::endl
                << "\t- total device memory:     " << deviceProperties.totalGlobalMem / (1024 * 1024) << " MB " << std::endl
                << "\t- device memory available: " << avail / (1024 * 1024) << " MB " << std::endl
                << "\t- per-block shared memory: " << deviceProperties.sharedMemPerBlock << std::endl
                << "\t- warp size:               " << deviceProperties.warpSize << std::endl
                << "\t- max threads per block:   " << deviceProperties.maxThreadsPerBlock << std::endl
                << "\t- max threads per SM(X):   " << deviceProperties.maxThreadsPerMultiProcessor << std::endl
                << "\t- max block sizes:         "
                << "{" << deviceProperties.maxThreadsDim[0]
                << "," << deviceProperties.maxThreadsDim[1]
                << "," << deviceProperties.maxThreadsDim[2] << "}" << std::endl
                << "\t- max grid sizes:          "
                << "{" << deviceProperties.maxGridSize[0]
                << "," << deviceProperties.maxGridSize[1]
                << "," << deviceProperties.maxGridSize[2] << "}" << std::endl
                << "\t- max 2D array texture:    "
                << "{" << deviceProperties.maxTexture2D[0]
                << "," << deviceProperties.maxTexture2D[1] << "}" << std::endl
                << "\t- max 3D array texture:    "
                << "{" << deviceProperties.maxTexture3D[0]
                << "," << deviceProperties.maxTexture3D[1]
                << "," << deviceProperties.maxTexture3D[2] << "}" << std::endl
                << "\t- max 2D linear texture:   "
                << "{" << deviceProperties.maxTexture2DLinear[0]
                << "," << deviceProperties.maxTexture2DLinear[1]
                << "," << deviceProperties.maxTexture2DLinear[2] << "}" << std::endl
                << "\t- max 2D layered texture:  "
                << "{" << deviceProperties.maxTexture2DLayered[0]
                << "," << deviceProperties.maxTexture2DLayered[1]
                << "," << deviceProperties.maxTexture2DLayered[2] << "}" << std::endl
                << "\t- number of SM(x)s:        " << deviceProperties.multiProcessorCount << std::endl
                << "\t- registers per SM(x):     " << deviceProperties.regsPerMultiprocessor << std::endl
                << "\t- registers per block:     " << deviceProperties.regsPerBlock << std::endl
                << "\t- concurrent kernels:      " << (deviceProperties.concurrentKernels ? "yes":"no") << std::endl
                << "\t- mapping host memory:     " << (deviceProperties.canMapHostMemory ? "yes":"no") << std::endl
                << "\t- unified addressing:      " << (deviceProperties.unifiedAddressing ? "yes":"no") << std::endl
                << "\t- texture alignment:       " << deviceProperties.textureAlignment << " byte" << std::endl
                << "\t- pitch alignment:         " << deviceProperties.texturePitchAlignment << " byte" << std::endl;

            information += deviceSS.str();
        }
    }
    else
    {
        information = "No CUDA-Enabled GPU.";
    }
#else
    information = "AliceVision built without CUDA support.";
#endif
    return information;
}


} // namespace gpu
} // namespace cloudViewer
