# Idempotent patch for jsoncpp ExternalProject: skip if CloudViewer changes are already present.
# Invoked as: cmake -DSOURCE_DIR=<dir> -DPATCH_FILE=<patch> -DGIT_EXECUTABLE=<git> -P apply_jsoncpp_patch.cmake

if(NOT DEFINED SOURCE_DIR OR NOT DEFINED PATCH_FILE)
  message(FATAL_ERROR "apply_jsoncpp_patch.cmake: SOURCE_DIR and PATCH_FILE must be set")
endif()

if(NOT EXISTS "${SOURCE_DIR}/CMakeLists.txt")
  message(FATAL_ERROR "apply_jsoncpp_patch.cmake: missing ${SOURCE_DIR}/CMakeLists.txt")
endif()

file(READ "${SOURCE_DIR}/CMakeLists.txt" _jsoncpp_root_cmakelists)
if(_jsoncpp_root_cmakelists MATCHES "option\\(JSONCPP_USE_CXX11_ABI")
  message(STATUS "jsoncpp: patch already applied (JSONCPP_USE_CXX11_ABI present), skipping")
  return()
endif()

if(NOT GIT_EXECUTABLE)
  find_program(GIT_EXECUTABLE NAMES git git.cmd REQUIRED)
endif()

execute_process(
  COMMAND "${GIT_EXECUTABLE}" init
  WORKING_DIRECTORY "${SOURCE_DIR}"
  RESULT_VARIABLE _git_init_rc
)
if(NOT _git_init_rc EQUAL 0)
  message(FATAL_ERROR "jsoncpp: git init failed in ${SOURCE_DIR}")
endif()

execute_process(
  COMMAND "${GIT_EXECUTABLE}" apply --ignore-space-change --ignore-whitespace "${PATCH_FILE}"
  WORKING_DIRECTORY "${SOURCE_DIR}"
  RESULT_VARIABLE _git_apply_rc
)
if(NOT _git_apply_rc EQUAL 0)
  message(FATAL_ERROR "jsoncpp: failed to apply ${PATCH_FILE}")
endif()

message(STATUS "jsoncpp: applied CloudViewer patch successfully")
