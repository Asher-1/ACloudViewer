# Root C++ CI entry point, shared by shell and PowerShell callers.
# cmake -DBUILD_DIR=<configured-build> -DCONFIG=Release -DJOBS=4 -P tests/Run.cmake
cmake_minimum_required(VERSION 3.24)
if(NOT DEFINED BUILD_DIR OR NOT EXISTS "${BUILD_DIR}/CMakeCache.txt")
    message(FATAL_ERROR "BUILD_DIR must name a configured ACloudViewer build")
endif()
# Multi-config callers pass CONFIG; single-config and fixture callers may not.
if(NOT DEFINED CONFIG OR CONFIG STREQUAL "")
    set(CONFIG Release)
endif()
# Comma-separated capability profiles form one CTest selection (no duplicate
# runs for tests that belong to more than one requested profile).
if(NOT DEFINED PROFILE)
    set(PROFILE ci)
endif()
if(NOT PROFILE MATCHES "^(ci|graphics|model|gpu|all)(,(ci|graphics|model|gpu|all))*$")
    message(FATAL_ERROR "PROFILE must contain ci, graphics, model, gpu or all")
endif()
if(DEFINED GROUP AND NOT GROUP MATCHES "^aicore-(fast|contract|model|yolo-matrix)$")
    message(FATAL_ERROR "GROUP must name an existing AICore compatibility selection")
endif()
string(REPLACE "," ";" profiles "${PROFILE}")
list(REMOVE_DUPLICATES profiles)
if(NOT DEFINED JOBS)
    set(JOBS 4)
endif()
if(NOT JOBS MATCHES "^[1-9][0-9]*$" OR JOBS GREATER 12)
    message(FATAL_ERROR "JOBS must be an integer between 1 and 12")
endif()
file(STRINGS "${BUILD_DIR}/CMakeCache.txt" enabled
    REGEX "^(BUILD_UNIT_TESTS|AICore_BUILD_TESTS|VTKENGINE_BUILD_TESTS):BOOL=(ON|TRUE|1)$")
if(NOT enabled)
    message(FATAL_ERROR "C++ tests are disabled; enable a test build before invoking the CI gate")
endif()
# Keep nested ExternalProject builds within the same ceiling. The aggregate
# CTest command is serial; only compilation uses the caller's bounded budget.
set(ENV{CMAKE_BUILD_PARALLEL_LEVEL} "${JOBS}")
set(ENV{NPROC} "${JOBS}")
set(ENV{BUILD_JOBS} "${JOBS}")
execute_process(COMMAND "${CMAKE_COMMAND}" --build "${BUILD_DIR}"
    --config "${CONFIG}" --target check-test-system --parallel 1
    RESULT_VARIABLE contract_status COMMAND_ECHO STDOUT)
if(NOT contract_status EQUAL 0)
    message(FATAL_ERROR "Unified test-system contract failed (${contract_status})")
endif()
set(build_targets)
if(DEFINED GROUP)
    list(APPEND build_targets "cpp-tests-${GROUP}")
endif()
foreach(profile IN LISTS profiles)
    if(DEFINED GROUP)
        break()
    endif()
    if(profile STREQUAL "all")
        list(APPEND build_targets cpp-tests)
    else()
        list(APPEND build_targets "cpp-tests-${profile}")
    endif()
endforeach()
execute_process(COMMAND "${CMAKE_COMMAND}" --build "${BUILD_DIR}"
    --config "${CONFIG}" --target ${build_targets} --parallel "${JOBS}"
    RESULT_VARIABLE status COMMAND_ECHO STDOUT)
if(NOT status EQUAL 0)
    message(FATAL_ERROR "Unified test prerequisite build failed (${status})")
endif()
if(DEFINED GROUP)
    set(selection "^${GROUP}$")
elseif("all" IN_LIST profiles)
    set(selection "^cloudviewer-test$")
else()
    string(JOIN "|" alternatives ${profiles})
    set(selection "^test-(${alternatives})$")
endif()
# Ordinary Qt widget contracts work without a display. Graphics owners retain
# their explicit xcb/Xvfb environment and never enter the default ci profile.
if(NOT "graphics" IN_LIST profiles AND NOT "all" IN_LIST profiles)
    set(ENV{QT_QPA_PLATFORM} offscreen)
    set(graphics_filter -LE "^graphics$")
endif()
find_program(CTEST_EXECUTABLE ctest REQUIRED)
# Some installed Vulkan drivers need an X connection even for device discovery.
# Reuse the repository's Xvfb mechanism on displayless Linux hosts; Qt contracts
# still use offscreen QPA and capability filtering still excludes graphics.
set(test_launcher)
if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux" AND "$ENV{DISPLAY}" STREQUAL "")
    find_program(TEST_XVFB_RUN xvfb-run)
    if(TEST_XVFB_RUN)
        set(test_launcher "${TEST_XVFB_RUN}" -a)
    endif()
endif()
execute_process(COMMAND ${test_launcher} "${CTEST_EXECUTABLE}" --test-dir "${BUILD_DIR}"
    -C "${CONFIG}" -L "${selection}" ${graphics_filter} --no-tests=error --output-on-failure -j1
    RESULT_VARIABLE status COMMAND_ECHO STDOUT)
if(NOT status EQUAL 0)
    message(FATAL_ERROR "Unified test gate failed (${status})")
endif()
