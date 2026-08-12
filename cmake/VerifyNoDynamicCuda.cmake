# VerifyNoDynamicCuda.cmake
#
# Regression guard: verify that a built shared library has no DT_NEEDED on
# CUDA runtime or BLAS shared libraries (libcudart.so.*, libcublas.so.*).
# Such dependencies prevent the library from loading on machines that have
# only the NVIDIA driver installed (libcuda.so.1) but no CUDA toolkit.
#
# The checked libraries are linked statically (CUDA::cudart_static) or
# replaced by equivalent code paths (MMQ instead of cuBLAS), so the
# dynamic dependency must be absent.
#
# Usage (post-build):
#   cmake -DTARGET_FILE=<path to .so> -P VerifyNoDynamicCuda.cmake
#
# Implementation: reads the file in HEX and searches for the ASCII bytes of
# "libcudart.so" and "libcublas.so".  These strings can only appear in the
# dynamic string table as the soname of a DT_NEEDED entry.  This approach is
# portable (does not require readelf/binutils).

if(NOT DEFINED TARGET_FILE)
    message(FATAL_ERROR "VerifyNoDynamicCuda: TARGET_FILE is required")
endif()
if(NOT EXISTS "${TARGET_FILE}")
    message(FATAL_ERROR "VerifyNoDynamicCuda: target file not found: ${TARGET_FILE}")
endif()

file(READ "${TARGET_FILE}" _bin HEX)

# "libcudart.so" in ASCII = 6c 69 62 63 75 64 61 72 74 2e 73 6f
string(FIND "${_bin}" "6c69626375646172742e736f" _cudart_pos)
if(_cudart_pos GREATER -1)
    message(FATAL_ERROR
        "${TARGET_FILE} still has a dynamic dependency on libcudart.so "
        "(DT_NEEDED). Link CUDA::cudart_static on UNIX so that the library "
        "loads on machines without a CUDA toolkit installed.")
endif()

# "libcublas.so" in ASCII = 6c 69 62 63 75 62 6c 61 73 2e 73 6f
string(FIND "${_bin}" "6c69626375626c61732e736f" _cublas_pos)
if(_cublas_pos GREATER -1)
    message(FATAL_ERROR
        "${TARGET_FILE} still has a dynamic dependency on libcublas.so "
        "(DT_NEEDED). Enable GGML_CUDA_FORCE_MMQ and apply the "
        "cuda_mmq/0001-cuda-mmq-force-static.patch so that cuBLAS is not "
        "needed at runtime.")
endif()

message(STATUS
    "CUDA dep check OK - ${TARGET_FILE} has no dynamic dependency on "
    "libcudart.so or libcublas.so")