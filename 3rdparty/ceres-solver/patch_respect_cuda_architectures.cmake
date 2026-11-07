# Ceres 2.2 unconditionally overwrites CMAKE_CUDA_ARCHITECTURES with an old
# compatibility set. Keep the parent reconstruction build's architecture list
# so the Ceres reference solver and Caspar use the same GPU code targets.
if (NOT DEFINED SOURCE_DIR)
    message(FATAL_ERROR "SOURCE_DIR is required")
endif ()

set(_ceres_cmake "${SOURCE_DIR}/CMakeLists.txt")
file(READ "${_ceres_cmake}" _ceres_source)
set(_old "set(CMAKE_CUDA_ARCHITECTURES \"50;60;70;80\")")
set(_new [=[if(NOT DEFINED CERES_CUDA_ARCHITECTURES)
  set(CERES_CUDA_ARCHITECTURES "50;60;70;80")
endif()
set(CMAKE_CUDA_ARCHITECTURES "${CERES_CUDA_ARCHITECTURES}")]=])
string(FIND "${_ceres_source}" "${_old}" _old_offset)
if (_old_offset EQUAL -1)
    message(FATAL_ERROR "Ceres CUDA architecture assignment changed upstream")
endif ()
string(REPLACE "${_old}" "${_new}" _ceres_source "${_ceres_source}")
file(WRITE "${_ceres_cmake}" "${_ceres_source}")
