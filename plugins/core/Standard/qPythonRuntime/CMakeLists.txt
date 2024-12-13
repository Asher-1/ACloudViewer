cmake_minimum_required(VERSION 3.15..3.27)

option(PLUGIN_PYTHON "Install Python Plugin" OFF)

if (NOT BUILD_PYTHON_MODULE)
  message(STATUS "Python plugin is disabled due to BUILD_PYTHON_MODULE=OFF!")
endif()

if(PLUGIN_PYTHON AND BUILD_PYTHON_MODULE)

  include(cmake/Helpers.cmake)
  include(cmake/CompilerWarnings.cmake)
  include(cmake/PythonEnvHelper.cmake)
  include(cmake/ACloudViewerVariables.cmake)

  project(PythonRuntime)

  addplugin(NAME ${PROJECT_NAME})

  # ensure_pybind11_cmake_module_is_in_path()

  option(PLUGIN_PYTHON_USE_EMBEDDED_MODULES
         "Should the Python wrapper libs be embedded in the plugin" ON
  )
  mark_as_advanced(PLUGIN_PYTHON_USE_EMBEDDED_MODULES)

  option(PLUGIN_PYTHON_COPY_ENV
    "Should the content of the current python site-package be copied on install" ON
  )
  mark_as_advanced(PLUGIN_PYTHON_COPY_ENV)
  
  add_subdirectory(wrapper)
  add_subdirectory(src)

  if(PLUGIN_PYTHON_USE_EMBEDDED_MODULES)
    target_compile_definitions(${PROJECT_NAME} PRIVATE USE_EMBEDDED_MODULES)
    if (WIN32) # fix duplicated defined issues in windows
      target_link_options(${PROJECT_NAME} PUBLIC "/force:multiple")
    endif ()
    embed_cvcorelib_in(${PROJECT_NAME})
    embed_pycc_in(${PROJECT_NAME})
  endif()

  target_link_libraries(${PROJECT_NAME} pybind11::embed)
  set_target_properties(${PROJECT_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION ON CXX_VISIBILITY_PRESET hidden)

  add_subdirectory(tests)
  add_subdirectory(docs)

  if(WIN32)
    # add_subdirectory(installer)
    manage_windows_install()
  elseif(UNIX AND NOT APPLE)
    if(NOT PLUGIN_PYTHON_USE_EMBEDDED_MODULES)
      installsharedlibrary(TARGET cccorelib)
      installsharedlibrary(TARGET pycc)
    else ()
      manage_linux_install()
    endif()
  endif()
endif()
