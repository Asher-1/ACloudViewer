// Standalone framework shims for the optional nvdiffrast CUDA core.
//
// The upstream core normally receives these definitions from its PyTorch
// extension wrapper.  Keeping the adapter here lets the native runtime link
// only CUDA runtime APIs, never Python, libtorch, or cuDNN.
#ifndef SAM3D_NVDIFFRAST_STANDALONE_COMPAT_HPP
#define SAM3D_NVDIFFRAST_STANDALONE_COMPAT_HPP

#include <cuda_runtime.h>

#include <cstring>
#include <stdexcept>
#include <string>

namespace sam3d {

class NvdiffrastLogSink {
public:
    template <typename T>
    NvdiffrastLogSink& operator<<(const T&) {
        return *this;
    }
};

inline void check_nvdiffrast_cuda(cudaError_t status, const char* expression) {
    if (status == cudaSuccess) return;
    throw std::runtime_error(std::string("nvdiffrast CUDA failure in ") + expression + ": " +
                             cudaGetErrorString(status));
}

}  // namespace sam3d

#define NVDR_CHECK(condition, message)                                                \
    do {                                                                              \
        if (!(condition)) throw std::runtime_error(message);                          \
    } while (0)

#define NVDR_CHECK_CUDA_ERROR(expression) \
    ::sam3d::check_nvdiffrast_cuda((expression), #expression)

#define LOG(level) ::sam3d::NvdiffrastLogSink()

#endif  // SAM3D_NVDIFFRAST_STANDALONE_COMPAT_HPP
