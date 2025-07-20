cmake_minimum_required(VERSION ${CLOUDVIEWER_PLUGIN_CMAKE_MINIMUM_VERSION})

project(pycc LANGUAGES CXX)

include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/Helpers.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/PythonEnvHelper.cmake")

# MASTER_PROJECT means that this CMakeLists is the top level CMakeLists which in
# our case means that this project is not built as a part of the PythonRuntime
# but as a standalone wheel
option(DEFINE_PYCC_RUNTIME "Should the runtime functions be included" ON)

# ensure_pybind11_cmake_module_is_in_path()

add_subdirectory(src)
add_subdirectory(plugins)

if(NOT PLUGIN_PYTHON_USE_EMBEDDED_MODULES AND BUILD_PYTHON_MODULE)

  pybind11_add_module(pycc ${pycc_sources})
  target_link_libraries(pycc PRIVATE ECV_DB_LIB ECV_IO_LIB)

  target_include_directories(
    pycc PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../cccorelib/src"
  )

  if(DEFINE_PYCC_RUNTIME)
    target_link_libraries(pycc PRIVATE PythonRuntime)
    target_compile_definitions(pycc PRIVATE DEFINE_PYCC_RUNTIME)
  endif()

endif()

function(embed_pycc_in target)
  target_sources(${target} PRIVATE ${pycc_sources})
  target_include_directories(${target} PRIVATE ${cccorelib_inc})
  if(DEFINE_PYCC_RUNTIME)
    target_compile_definitions(${target} PRIVATE DEFINE_PYCC_RUNTIME)
  endif()
endfunction(embed_pycc_in)

# Cache variables so embed_pycc_in can be used in parent projects
set(pycc_sources
    ${pycc_sources}
    CACHE INTERNAL ""
)

set(cccorelib_inc
    "${CMAKE_CURRENT_SOURCE_DIR}/../cccorelib/src"
    CACHE INTERNAL ""
)
