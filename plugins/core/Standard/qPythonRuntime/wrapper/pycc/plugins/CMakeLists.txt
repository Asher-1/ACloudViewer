# Depending of if pycc is being built as a stand-alone python module or a
# ACloudViewer's plugin, the path to the C++ plugins is not the same since in
# one case ACloudViewer's sources are somewere in the build dir (cmake handles
# that) or pycc's sources are within ACloudViewer's sources.
get_filename_component(
  STANDARD_PLUGINS_ROOT_DIR
  "../../../../../../core/Standard"
  ABSOLUTE
  BASE_DIR
  ${CMAKE_CURRENT_LIST_DIR}
)

# The goal of the following if chain is to set 2 variables -
# INSTALL_PLUGINS_PATH: Where we are going to install the original C++ plugin
# (Should be the same as CC when compiling in plugin mode) -
# INSTALL_PLUGINS_WRAPPERS_PATH: Where we are going to install the python
# wrapper
if(APPLE)
  set(ACloudViewer_MAC_BASE_DIR
    ${CMAKE_INSTALL_PREFIX}/${CloudViewer_INSTALL_BIN_DIR}/${MAIN_APP_NAME}/${MAIN_APP_NAME}.app
  )
  set(INSTALL_PLUGINS_PATH
  "${ACloudViewer_MAC_BASE_DIR}/Contents/cvPlugins"
  )
  set(INSTALL_PLUGINS_WRAPPERS_PATH
  "${ACloudViewer_MAC_BASE_DIR}/Contents/cvPythonPlugins"
  )
elseif(UNIX)
  set(INSTALL_PLUGINS_PATH "${CMAKE_INSTALL_PREFIX}/plugins")
  set(INSTALL_PLUGINS_WRAPPERS_PATH
      "${CMAKE_INSTALL_PREFIX}/plugins-python"
  )
elseif(WIN32)
  set(INSTALL_PLUGINS_PATH "${INSTALL_DESTINATIONS}/plugins")
  set(INSTALL_PLUGINS_WRAPPERS_PATH
      "${INSTALL_DESTINATIONS}/plugins-python"
  )
endif()

message(DEBUG "INSTALL_PLUGINS_PATH: ${INSTALL_PLUGINS_PATH}")
message(DEBUG "INSTALL_PLUGINS_WRAPPERS_PATH: ${INSTALL_PLUGINS_WRAPPERS_PATH}")

set(PYCC_PLUGIN_WRAPPER_LIST
    ""
    CACHE INTERNAL "Internal plugin wrapper list"
)

# Function to add a Python plugin that wraps a CloudComapare's C++ plugin.
#
# It handles where to install the plugin for supported platforms as well
# properly setting the RPATH on platforms that needs it.
#
# It needs two named argument: - NAME: name of the cmake target that is the
# Python wrapper - WRAPPED: name of the cmake target of the C++ plugin being
# wrapped
#
# Example: `AddPluginWrapper(NAME pym3c2 WRAPPED QM3C2_PLUGIN)`
function(AddPluginWrapper)
  # https://cmake.org/cmake/help/latest/command/cmake_parse_arguments.html
  set(options "")
  set(oneValueArgs "NAME;WRAPPED")
  set(multiValueArgs "")
  cmake_parse_arguments(
    PLUGIN_WRAPPER
    "${options}"
    "${oneValueArgs}"
    "${multiValueArgs}"
    ${ARGN}
  )

  if(NOT PLUGIN_WRAPPER_NAME)
    message(FATAL_ERROR "AddPluginWrapper missing 'NAME' parameter")
  endif()

  if(NOT PLUGIN_WRAPPER_WRAPPED)
    message(
      FATAL_ERROR
        "Plugin Wrapper named ${PLUGIN_WRAPPER_NAME} does not wrap any plugin"
    )
  endif()

  set(PYCC_PLUGIN_WRAPPER_LIST
      "${PYCC_PLUGIN_WRAPPER_LIST};${PLUGIN_WRAPPER_NAME}"
      CACHE INTERNAL "Internal plugin wrapper list"
  )

  # Install the wrapper
  install(TARGETS ${PLUGIN_WRAPPER_NAME}
          LIBRARY DESTINATION "${INSTALL_PLUGINS_WRAPPERS_PATH}"
  )

endfunction()

if(PLUGIN_STANDARD_QM3C2)
  add_subdirectory(m3c2)
endif()
