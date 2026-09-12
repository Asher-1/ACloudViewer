# ROCm is an SDK/toolchain dependency. Expose its imported targets behind the
# normal 3rdparty interface contract used by reconstruction dependencies.
#
# Keep discovery independent from the host distribution: ROCm may be installed
# below an explicitly configured root, a ROCM_PATH environment root, a TheRock
# SDK environment, or the conventional /opt/rocm location. This mirrors the
# upstream COLMAP HIP lookup while preserving the project-wide 3rdparty target
# convention.
if (NOT TARGET 3rdparty_rocm)
    find_program(ROCM_SDK_EXECUTABLE rocm-sdk)
    set(_rocm_default_root "/opt/rocm")
    if (DEFINED ENV{ROCM_PATH})
        set(_rocm_default_root "$ENV{ROCM_PATH}")
    elseif (ROCM_SDK_EXECUTABLE)
        execute_process(
            COMMAND "${ROCM_SDK_EXECUTABLE}" path
            OUTPUT_VARIABLE _rocm_sdk_root
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
            RESULT_VARIABLE _rocm_sdk_root_result)
        if (_rocm_sdk_root_result EQUAL 0 AND IS_DIRECTORY "${_rocm_sdk_root}")
            set(_rocm_default_root "${_rocm_sdk_root}")
        endif ()
    endif ()
    set(ROCM_PATH "${_rocm_default_root}" CACHE PATH "Path to ROCm installation")
    list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}")

    find_package(hip REQUIRED)
    find_package(hiprand REQUIRED)
    find_package(rocrand REQUIRED)

    add_library(3rdparty_rocm INTERFACE)
    target_link_libraries(3rdparty_rocm INTERFACE
        hip::host
        hip::hiprand
        roc::rocrand)
endif ()

# CMake only uses this after enable_language(HIP). A user-provided value is
# authoritative. TheRock can report the architecture set it was built for;
# use that information when available and keep a conservative fallback for
# regular ROCm installations.
if (NOT DEFINED CMAKE_HIP_ARCHITECTURES OR CMAKE_HIP_ARCHITECTURES STREQUAL "")
    set(_rocm_hip_architectures "")
    if (ROCM_SDK_EXECUTABLE)
        execute_process(
            COMMAND "${ROCM_SDK_EXECUTABLE}" targets
            OUTPUT_VARIABLE _rocm_sdk_targets
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
            RESULT_VARIABLE _rocm_sdk_targets_result)
        if (_rocm_sdk_targets_result EQUAL 0)
            string(REGEX MATCHALL "gfx[0-9a-fA-F]+" _rocm_hip_architectures
                   "${_rocm_sdk_targets}")
        endif ()
    endif ()
    if (NOT _rocm_hip_architectures)
        set(_rocm_hip_architectures "gfx90a;gfx942;gfx1100")
    endif ()
    set(CMAKE_HIP_ARCHITECTURES "${_rocm_hip_architectures}" CACHE STRING
        "AMD GPU architectures for reconstruction PatchMatch" FORCE)
endif ()
