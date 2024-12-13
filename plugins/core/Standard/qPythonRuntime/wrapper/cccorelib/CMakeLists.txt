cmake_minimum_required(VERSION 3.15..3.27)
project(pycvcorelib LANGUAGES CXX)

include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/Helpers.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/PythonEnvHelper.cmake")

add_subdirectory(src)

if(NOT PLUGIN_PYTHON_USE_EMBEDDED_MODULES AND BUILD_PYTHON_MODULE)
  # ensure_pybind11_cmake_module_is_in_path()
  pybind11_add_module(cccorelib ${cvcorelib_sources})
  target_link_libraries(cccorelib PRIVATE CVCoreLib)
endif()

function(embed_cvcorelib_in target)
  target_sources(${target} PRIVATE ${cvcorelib_sources})
endfunction(embed_cvcorelib_in)

# Cache variables so embed_cvcorelib_in can be used in parent projects
set(cvcorelib_sources
    ${cvcorelib_sources}
    CACHE INTERNAL ""
)
