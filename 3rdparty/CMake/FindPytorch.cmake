# Find the PyTorch root and use the provided cmake module
#
# The following variables will be set:
# - Pytorch_FOUND
# - Pytorch_VERSION
# - Pytorch_ROOT
# - Pytorch_DEFINITIONS
#
# This script will call find_package( Torch ) which will define:
# - TORCH_FOUND
# - TORCH_INCLUDE_DIRS
# - TORCH_LIBRARIES
# - TORCH_CXX_FLAGS
#
# and import the target 'torch'.

# "80-real" to "8.0" and "80" to "8.0+PTX":
macro(translate_arch_string input output)
    if("${input}" MATCHES "[0-9]+-real")
        string(REGEX REPLACE "([1-9])([0-9])-real" "\\1.\\2" version "${input}")
    elseif("${input}" MATCHES "([0-9]+)")
        string(REGEX REPLACE "([1-9])([0-9])" "\\1.\\2+PTX" version "${input}")
    elseif("${input}" STREQUAL "native")
        set(version "Auto")
    else()
        message(FATAL_ERROR "Invalid architecture string: ${input}")
    endif()
    set(${output} "${version}")
endmacro()

if(NOT Pytorch_FOUND)
    # Searching for pytorch requires the python executable
    if (NOT Python3_EXECUTABLE)
        message(FATAL_ERROR "Python 3 not found in top level file")
    endif()

    message(STATUS "Getting PyTorch properties ...")

    # Set KMP_DUPLICATE_LIB_OK to avoid OpenMP initialization errors when importing torch
    # This is needed when multiple OpenMP runtimes are linked (e.g., PyTorch and system OpenMP)
    # Handle CPU-only PyTorch where torch.version.cuda returns None - convert to string 'None' to ensure list length is always 4
    set(PyTorch_FETCH_PROPERTIES
        "import os"
        "os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'"
        "import torch"
        "print(torch.__version__, end=';')"
        "print(os.path.dirname(torch.__file__), end=';')"
        "print(torch._C._GLIBCXX_USE_CXX11_ABI, end=';')"
        "cuda_ver = getattr(torch.version, 'cuda', None)"
        "print(str(cuda_ver) if cuda_ver is not None else 'None')"
    )
    execute_process(
        COMMAND ${Python3_EXECUTABLE} "-c" "${PyTorch_FETCH_PROPERTIES}"
        OUTPUT_VARIABLE PyTorch_PROPERTIES
        ERROR_VARIABLE PyTorch_PROPERTIES_ERROR
        RESULT_VARIABLE PyTorch_PROPERTIES_RESULT
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    if(NOT PyTorch_PROPERTIES_RESULT EQUAL "0")
        message(FATAL_ERROR "Failed to get PyTorch properties. Error: ${PyTorch_PROPERTIES_ERROR}\nOutput: ${PyTorch_PROPERTIES}")
    endif()

    if("${PyTorch_PROPERTIES}" STREQUAL "")
        message(FATAL_ERROR "PyTorch properties output is empty. Make sure PyTorch is installed in the Python environment.")
    endif()

    # CMake automatically treats semicolon-separated strings as lists
    # Remove any trailing semicolons and empty elements that might be created
    string(STRIP "${PyTorch_PROPERTIES}" PyTorch_PROPERTIES_STRIPPED)
    if(PyTorch_PROPERTIES_STRIPPED MATCHES ";$")
        string(REGEX REPLACE ";$" "" PyTorch_PROPERTIES_STRIPPED "${PyTorch_PROPERTIES_STRIPPED}")
    endif()
    
    # Filter out empty elements from the list
    set(PyTorch_PROPERTIES_FILTERED "")
    foreach(item IN LISTS PyTorch_PROPERTIES_STRIPPED)
        if(NOT "${item}" STREQUAL "")
            list(APPEND PyTorch_PROPERTIES_FILTERED "${item}")
        endif()
    endforeach()
    set(PyTorch_PROPERTIES "${PyTorch_PROPERTIES_FILTERED}")
    
    # Verify we have the expected number of properties
    list(LENGTH PyTorch_PROPERTIES PyTorch_PROPERTIES_COUNT)
    if(NOT PyTorch_PROPERTIES_COUNT EQUAL 4)
        message(FATAL_ERROR "Expected 4 PyTorch properties, got ${PyTorch_PROPERTIES_COUNT}. Output: ${PyTorch_PROPERTIES}")
    endif()

    list(GET PyTorch_PROPERTIES 0 Pytorch_VERSION)
    list(GET PyTorch_PROPERTIES 1 Pytorch_ROOT)
    list(GET PyTorch_PROPERTIES 2 Pytorch_CXX11_ABI)
    list(GET PyTorch_PROPERTIES 3 Pytorch_CUDA_VERSION)
    
    # Set these as cache variables so they persist across CMake invocations
    # This is especially important for Windows multi-config builds
    set(Pytorch_VERSION "${Pytorch_VERSION}" CACHE STRING "PyTorch version" FORCE)
    set(Pytorch_ROOT "${Pytorch_ROOT}" CACHE PATH "PyTorch root directory" FORCE)
    set(Pytorch_CXX11_ABI "${Pytorch_CXX11_ABI}" CACHE BOOL "PyTorch C++11 ABI" FORCE)
    set(Pytorch_CUDA_VERSION "${Pytorch_CUDA_VERSION}" CACHE STRING "PyTorch CUDA version" FORCE)

    unset(PyTorch_FETCH_PROPERTIES)
    unset(PyTorch_PROPERTIES)

    if(BUILD_CUDA_MODULE)
        # Using CUDA 12.x and Pytorch <2.4 gives the error "Unknown CUDA Architecture Name 9.0a in CUDA_SELECT_NVCC_ARCH_FLAGS".
        # As a workaround we explicitly set TORCH_CUDA_ARCH_LIST
        set(TORCH_CUDA_ARCH_LIST "")
        foreach(arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
            translate_arch_string("${arch}" ptarch)
            list(APPEND TORCH_CUDA_ARCH_LIST "${ptarch}")
        endforeach()
        message(STATUS "Using top level CMAKE_CUDA_ARCHITECTURES for TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST}")

        # fix the issues of Failed to find nvToolsExt
        message(STATUS "Pytorch_CUDA_VERSION: ${Pytorch_CUDA_VERSION}")
        if(WIN32 AND Pytorch_CUDA_VERSION VERSION_GREATER_EQUAL "12.0")
            message(STATUS "PyTorch NVTX headers workaround: Yes")
            # only do this if nvToolsExt is not defined and CUDA::nvtx3 exists
            if(NOT TARGET CUDA::nvToolsExt AND TARGET CUDA::nvtx3)
                message(STATUS "CUDA::nvToolsExt is not defined and use CUDA::nvtx3 instead!")
                add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
                # ensure that PyTorch is told to use NVTX3 headers
                target_compile_definitions(
                    CUDA::nvToolsExt INTERFACE
                    TORCH_CUDA_USE_NVTX3
                )
                target_link_libraries(CUDA::nvToolsExt INTERFACE CUDA::nvtx3)
            else()
                set(NVTX3_INCLUDE_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${Pytorch_CUDA_VERSION}/include/nvtx3")
                include_directories(${NVTX3_INCLUDE_DIR})
                message(STATUS "CUDA::nvtx3 not found, creating CUDA::nvToolsExt interface manually")
                if(NOT TARGET CUDA::nvToolsExt)
                    add_library(nvtx3_dummy INTERFACE)
                    target_include_directories(nvtx3_dummy INTERFACE "${NVTX3_INCLUDE_DIR}")
                    add_library(CUDA::nvToolsExt ALIAS nvtx3_dummy)
                endif()
            endif()
        else()
            message(STATUS "PyTorch NVTX headers workaround: No")
        endif()
    endif()

    # Use the cmake config provided by torch
    find_package(Torch REQUIRED PATHS "${Pytorch_ROOT}" NO_DEFAULT_PATH)

    if(BUILD_CUDA_MODULE)
        # Note: older versions of PyTorch have hard-coded cuda library paths, see:
        # https://github.com/pytorch/pytorch/issues/15476.
        # This issue has been addressed but we observed for the conda packages for
        # PyTorch 1.2.0 and 1.4.0 that there are still hardcoded paths in
        #  ${TORCH_ROOT}/share/cmake/Caffe2/Caffe2Targets.cmake
        # Try to fix those here
        find_package(CUDAToolkit REQUIRED)
        get_target_property( iface_link_libs torch INTERFACE_LINK_LIBRARIES )
        string( REPLACE "/usr/local/cuda" "${CUDAToolkit_LIBRARY_ROOT}" iface_link_libs "${iface_link_libs}" )
        set_target_properties( torch PROPERTIES INTERFACE_LINK_LIBRARIES "${iface_link_libs}" )
        if( TARGET torch_cuda )
            get_target_property( iface_link_libs torch_cuda INTERFACE_LINK_LIBRARIES )
            string( REPLACE "/usr/local/cuda" "${CUDAToolkit_LIBRARY_ROOT}" iface_link_libs "${iface_link_libs}" )
            set_target_properties( torch_cuda PROPERTIES INTERFACE_LINK_LIBRARIES "${iface_link_libs}" )
        endif()
        # if successful everything works :)
        # if unsuccessful CMake will complain that there are no rules to make the targets with the hardcoded paths

        # Workaround for missing c10/cuda/impl/cuda_cmake_macros.h in pip-installed PyTorch
        # This issue is more common with Python 3.13 + Ubuntu 24.04 due to incomplete wheel packaging.
        # The file is typically generated from cuda_cmake_macros.h.in during PyTorch's build.
        # NOTE: This issue has been observed specifically on Ubuntu 24.04, other platforms may not be affected.
        set(CUDA_CMAKE_MACROS_H_PATH "${Pytorch_ROOT}/include/c10/cuda/impl/cuda_cmake_macros.h")
        if(NOT EXISTS "${CUDA_CMAKE_MACROS_H_PATH}")
            message(WARNING "PyTorch cuda_cmake_macros.h not found at: ${CUDA_CMAKE_MACROS_H_PATH}")
            message(WARNING "This is a known issue with Python 3.13 + PyTorch pip packages.")
            
            # Check if we're on Ubuntu 24.04 (where this issue is specifically observed)
            if(UNIX AND NOT APPLE)
                # Try to detect Ubuntu version if UBUNTU_VERSION is not already set
                if(NOT DEFINED UBUNTU_VERSION)
                    execute_process(
                        COMMAND lsb_release -rs
                        OUTPUT_VARIABLE UBUNTU_VERSION
                        OUTPUT_STRIP_TRAILING_WHITESPACE
                        ERROR_QUIET
                    )
                endif()
                if(UBUNTU_VERSION VERSION_EQUAL "24.04")
                    message(WARNING "Detected Ubuntu 24.04 - this issue is specifically observed on this platform")
                    message(WARNING "Other platforms (Ubuntu 22.04, 20.04, etc.) may not be affected")
                endif()
            endif()
            
            message(WARNING "PyTorch version: ${Pytorch_VERSION}, Python version: ${Python3_VERSION}")
            
            # Check if PyTorch version supports Python 3.13 (>= 2.6.0 recommended)
            if(Pytorch_VERSION VERSION_LESS "2.6.0")
                message(WARNING "Consider upgrading PyTorch to >= 2.6.0 for better Python 3.13 support")
                message(WARNING "Or consider using Python 3.12 or earlier for better compatibility")
            endif()
            
            # Create the stub header in the build directory
            set(PYTORCH_STUB_INCLUDE_DIR "${CMAKE_BINARY_DIR}/pytorch_stub_include")
            set(PYTORCH_STUB_CUDA_IMPL_DIR "${PYTORCH_STUB_INCLUDE_DIR}/c10/cuda/impl")
            file(MAKE_DIRECTORY "${PYTORCH_STUB_CUDA_IMPL_DIR}")
            
            # Create a minimal stub header file
            # NOTE: This is a workaround. The actual file may contain CUDA architecture
            # definitions and other build-time configuration. If compilation succeeds but
            # runtime fails, consider upgrading PyTorch or using Python 3.12.
            file(WRITE "${PYTORCH_STUB_CUDA_IMPL_DIR}/cuda_cmake_macros.h"
                "// Auto-generated workaround header for PyTorch CUDA macros\n"
                "// This file is created as a workaround for missing cuda_cmake_macros.h\n"
                "// in pip-installed PyTorch packages, especially with Python 3.13 + Ubuntu 24.04\n"
                "//\n"
                "// NOTE: This issue has been observed specifically on Ubuntu 24.04.\n"
                "// Other platforms (Ubuntu 22.04, 20.04, Windows, macOS) may not be affected.\n"
                "//\n"
                "// Original file is generated from c10/cuda/impl/cuda_cmake_macros.h.in\n"
                "// during PyTorch's build process via CMake configure_file()\n"
                "//\n"
                "// WARNING: This is a minimal stub. For production use, consider:\n"
                "// 1. Upgrading to PyTorch >= 2.6.0 which has better Python 3.13 support\n"
                "// 2. Building PyTorch from source to get the complete header\n"
                "// 3. Using Python 3.12 or earlier for better compatibility\n"
                "// 4. Using Ubuntu 22.04 or earlier if platform flexibility allows\n"
                "\n"
                "#ifndef C10_CUDA_IMPL_CUDA_CMAKE_MACROS_H_\n"
                "#define C10_CUDA_IMPL_CUDA_CMAKE_MACROS_H_\n"
                "\n"
                "// Minimal macro definitions that may be referenced by CUDAMacros.h\n"
                "// These are typically defined during PyTorch's CMake configuration\n"
                "\n"
                "#endif // C10_CUDA_IMPL_CUDA_CMAKE_MACROS_H_\n"
            )
            # Add the stub include directory to TORCH_INCLUDE_DIRS with higher priority
            list(INSERT TORCH_INCLUDE_DIRS 0 "${PYTORCH_STUB_INCLUDE_DIR}")
            message(STATUS "Created workaround header at: ${PYTORCH_STUB_CUDA_IMPL_DIR}/cuda_cmake_macros.h")
            message(STATUS "Added workaround include directory (prepended): ${PYTORCH_STUB_INCLUDE_DIR}")
        else()
            message(STATUS "Found PyTorch cuda_cmake_macros.h at: ${CUDA_CMAKE_MACROS_H_PATH}")
        endif()

        # remove flags that nvcc does not understand
        get_target_property( iface_compile_options torch INTERFACE_COMPILE_OPTIONS )
        set_target_properties( torch PROPERTIES INTERFACE_COMPILE_OPTIONS "" )
        if (TARGET torch_cuda)
            set_target_properties( torch_cuda PROPERTIES INTERFACE_COMPILE_OPTIONS "" )
        endif()
        if (TARGET torch_cpu)
            set_target_properties( torch_cpu PROPERTIES INTERFACE_COMPILE_OPTIONS "" )
        endif()
    endif()

    # If MKL is installed in the system level (e.g. for oneAPI Toolkit),
    # caffe2::mkl and caffe2::mkldnn will be added to torch_cpu's
    # INTERFACE_LINK_LIBRARIES. However, CloudViewer already comes with MKL linkage
    # and we're not using MKLDNN.
    get_target_property(torch_cpu_INTERFACE_LINK_LIBRARIES torch_cpu
                        INTERFACE_LINK_LIBRARIES)
    list(REMOVE_ITEM torch_cpu_INTERFACE_LINK_LIBRARIES caffe2::mkl)
    list(REMOVE_ITEM torch_cpu_INTERFACE_LINK_LIBRARIES caffe2::mkldnn)
    set_target_properties(torch_cpu PROPERTIES INTERFACE_LINK_LIBRARIES
                          "${torch_cpu_INTERFACE_LINK_LIBRARIES}")
endif()

message(STATUS "PyTorch         version: ${Pytorch_VERSION}")
message(STATUS "               root dir: ${Pytorch_ROOT}")
message(STATUS "          compile flags: ${TORCH_CXX_FLAGS}")
if (UNIX AND NOT APPLE)
    message(STATUS "          use cxx11 abi: ${Pytorch_CXX11_ABI}")
endif()
foreach(idir ${TORCH_INCLUDE_DIRS})
    message(STATUS "           include dirs: ${idir}")
endforeach(idir)
foreach(lib ${TORCH_LIBRARIES})
    message(STATUS "              libraries: ${lib}")
endforeach(lib)

# Check if the c++11 ABI is compatible on Linux
if(UNIX AND NOT APPLE)
    if((Pytorch_CXX11_ABI AND (NOT GLIBCXX_USE_CXX11_ABI)) OR
       (NOT Pytorch_CXX11_ABI AND GLIBCXX_USE_CXX11_ABI))
        if(Pytorch_CXX11_ABI)
            set(NEEDED_ABI_FLAG "ON")
        else()
            set(NEEDED_ABI_FLAG "OFF")
        endif()
        message(FATAL_ERROR "PyTorch and CloudViewer ABI mismatch: ${Pytorch_CXX11_ABI} != ${GLIBCXX_USE_CXX11_ABI}.\n"
                            "Please use -DGLIBCXX_USE_CXX11_ABI=${NEEDED_ABI_FLAG} "
                            "in the cmake config command to change the CloudViewer ABI.")
    else()
        message(STATUS "PyTorch matches CloudViewer ABI: ${Pytorch_CXX11_ABI} == ${GLIBCXX_USE_CXX11_ABI}")
    endif()
endif()

message(STATUS "Pytorch_VERSION: ${Pytorch_VERSION}, CUDAToolkit_VERSION: ${CUDAToolkit_VERSION}")
if (BUILD_PYTORCH_OPS AND BUILD_CUDA_MODULE AND CUDAToolkit_VERSION
        VERSION_GREATER_EQUAL "11.0" AND Pytorch_VERSION VERSION_LESS
        "1.9")
    message(WARNING
        "--------------------------------------------------------------------------------\n"
        "                                                                                \n"
        " You are compiling PyTorch ops with CUDA 11 with PyTorch version < 1.9. This    \n"
        " configuration may have stability issues. See                                   \n"
        " https://github.com/isl-org/Open3D/issues/3324 and                              \n"
        " https://github.com/pytorch/pytorch/issues/52663 for more information on this   \n"
        " problem.                                                                       \n"
        "                                                                                \n"
        " We recommend to compile PyTorch from source with compile flags                 \n"
        "   '-Xcompiler -fno-gnu-unique'                                                 \n"
        "                                                                                \n"
        " or use the PyTorch wheels at                                                   \n"
        "   https://github.com/isl-org/open3d_downloads/releases/tag/torch1.8.2          \n"
        "                                                                                \n"
        "--------------------------------------------------------------------------------\n"
    )
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Pytorch DEFAULT_MSG Pytorch_VERSION
                                  Pytorch_ROOT)
