# VerifyNoDynamicCuda.cmake
#
# Regression guard: verify that a built shared library has no DT_NEEDED on
# CUDA runtime or math shared libraries (libcudart.so.*, libcublas.so.*,
# libcusolver.so.*, libcusparse.so.*).
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
# the forbidden sonames. These strings can only appear in the
# dynamic string table as the soname of a DT_NEEDED entry.  This approach is
# portable (does not require readelf/binutils).

if(NOT DEFINED TARGET_FILE)
    message(FATAL_ERROR "VerifyNoDynamicCuda: TARGET_FILE is required")
endif()
if(NOT EXISTS "${TARGET_FILE}")
    message(FATAL_ERROR "VerifyNoDynamicCuda: target file not found: ${TARGET_FILE}")
endif()

file(READ "${TARGET_FILE}" _bin HEX)

set(_forbidden_cuda_sonames
    "libcudart.so"
    "libcublas.so"
    "libcusolver.so"
    "libcusparse.so")
foreach(_soname IN LISTS _forbidden_cuda_sonames)
    string(HEX "${_soname}" _soname_hex)
    string(FIND "${_bin}" "${_soname_hex}" _soname_pos)
    if(_soname_pos GREATER -1)
        message(FATAL_ERROR
            "${TARGET_FILE} still has a dynamic dependency on ${_soname} "
            "(DT_NEEDED). Keep CUDA math libraries out of portable shared "
            "artifacts or link an equivalent static implementation.")
    endif()
endforeach()

message(STATUS
    "CUDA dep check OK - ${TARGET_FILE} has no dynamic dependency on "
    "libcudart.so, libcublas.so, libcusolver.so, or libcusparse.so")
