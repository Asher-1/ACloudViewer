include(FetchContent)

FetchContent_Declare(
    ext_pybind11
    PREFIX pybind11
    URL https://github.com/pybind/pybind11/archive/refs/tags/v3.0.0.tar.gz
    URL_HASH SHA256=453b1a3e2b266c3ae9da872411cadb6d693ac18063bd73226d96cfb7015a200c
    DOWNLOAD_DIR "${CLOUDVIEWER_THIRD_PARTY_DOWNLOAD_DIR}/pybind11"
)

FetchContent_MakeAvailable(ext_pybind11)

# Fix MSVC UTF-8 encoding issue with pybind11 source files
# pybind11 v3.0.0 uses Unicode characters (e.g., scissors "✄" in stl_bind.h line 713)
# MSVC requires /utf-8 flag to properly handle UTF-8 source files
# Apply only to C/CXX compilation, not CUDA (CUDA will handle it via CMAKE_CUDA_FLAGS)
if(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/utf-8>)
    add_compile_options($<$<COMPILE_LANGUAGE:C>:/utf-8>)
endif()