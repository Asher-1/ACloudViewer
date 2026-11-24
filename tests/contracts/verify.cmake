cmake_minimum_required(VERSION 3.24)
if(NOT DEFINED SOURCE_DIR OR NOT IS_DIRECTORY "${SOURCE_DIR}")
    message(FATAL_ERROR "SOURCE_DIR must name tests/contracts")
endif()
if(NOT DEFINED OUTPUT_DIR OR NOT IS_ABSOLUTE "${OUTPUT_DIR}")
    message(FATAL_ERROR "OUTPUT_DIR must be an absolute fixture build directory")
endif()
find_program(CTEST_EXECUTABLE ctest REQUIRED)

# Never recursively delete a caller-selected path. Existing fixture builds are
# reused by CMake; each case has its own deterministic binary directory.
file(REAL_PATH "${SOURCE_DIR}" source_real)
file(REAL_PATH "${OUTPUT_DIR}" output_real)
cmake_path(IS_PREFIX source_real "${output_real}" NORMALIZE inside_source)
cmake_path(IS_PREFIX output_real "${source_real}" NORMALIZE contains_source)
get_filename_component(repository "${SOURCE_DIR}/../.." ABSOLUTE)
cmake_path(IS_PREFIX output_real "${repository}" NORMALIZE contains_repository)
get_filename_component(output_parent "${output_real}" DIRECTORY)
if(inside_source OR contains_source OR contains_repository OR
        output_parent STREQUAL output_real)
    message(FATAL_ERROR "OUTPUT_DIR must be a dedicated fixture directory outside its source tree")
endif()
# Exercise rejection before any fixture files can be created at protected paths.
cmake_path(GET output_real ROOT_PATH filesystem_root)
foreach(protected_path "${filesystem_root}" "${repository}" "${source_real}")
    execute_process(COMMAND "${CMAKE_COMMAND}" -DSOURCE_DIR=${SOURCE_DIR}
        -DOUTPUT_DIR=${protected_path} -P "${CMAKE_CURRENT_LIST_FILE}"
        RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
    if(result EQUAL 0 OR NOT "${output}\n${error}" MATCHES "OUTPUT_DIR must be a dedicated")
        message(FATAL_ERROR "Fixture accepted a protected output path: ${protected_path}")
    endif()
endforeach()
file(MAKE_DIRECTORY "${OUTPUT_DIR}")

function(configure_case case_name expected_success expected_text)
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -S "${SOURCE_DIR}" -B "${OUTPUT_DIR}/${case_name}"
            -DCASE=${case_name}
        RESULT_VARIABLE result
        OUTPUT_VARIABLE stdout
        ERROR_VARIABLE stderr)
    set(output "${stdout}\n${stderr}")
    if(expected_success AND NOT result EQUAL 0)
        message(FATAL_ERROR "${case_name} fixture configure failed:\n${output}")
    elseif(NOT expected_success AND result EQUAL 0)
        message(FATAL_ERROR "${case_name} fixture unexpectedly configured successfully")
    endif()
    if(expected_text AND NOT output MATCHES "${expected_text}")
        message(FATAL_ERROR "${case_name} fixture output lacked '${expected_text}':\n${output}")
    endif()
endfunction()

function(assert_selected build_dir test_name label expected_count)
    execute_process(COMMAND "${CTEST_EXECUTABLE}" --test-dir "${build_dir}"
        -N -R "^${test_name}$" -L "^${label}$"
        RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
    if(NOT result EQUAL 0 OR NOT output MATCHES "Total Tests: ${expected_count}([^0-9]|$)")
        message(FATAL_ERROR "${test_name} label ${label}: expected ${expected_count} selected test(s):\n${output}\n${error}")
    endif()
endfunction()

configure_case(valid TRUE "")
set(valid_dir "${OUTPUT_DIR}/valid")
execute_process(COMMAND "${CMAKE_COMMAND}" --build "${valid_dir}" --target cpp-tests
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "valid fixture build failed:\n${output}\n${error}")
endif()
foreach(check
        "fixture-direct|cpp|1"
        "fixture-direct|aicore-model|1"
        "fixture-direct|aicore-contract|1"
        "fixture-direct|aicore-fast|0"
        "fixture-direct|cpp-ci|0"
        "fixture-wrapper|cpp-ci|1"
        "fixture-wrapper|aicore-contract|1"
        "fixture-wrapper|aicore-fast|1"
        "fixture-forward|cpp-gpu|1"
        "fixture-forward|cpp-ci|0"
        "fixture-forward|aicore-fast|0"
        "fixture-script|aicore-contract|1"
        "fixture-script|aicore-fast|0"
        "fixture-script|cpp|0"
        "fixture-cvplugin|cpp-ci|1"
        "fixture-cvplugin|aicore-fast|0"
        "fixture-network|cpp|1"
        "fixture-network|cpp-ci|0"
        "fixture-network|aicore-fast|0"
        "fixture-graphics|cpp|1"
        "fixture-graphics|cpp-graphics|1"
        "fixture-graphics|cpp-ci|0"
        "fixture-graphics|test-ci|0"
        "fixture-graphics|test-graphics|1"
        "fixture-shared-graphics|test-ci|0"
        "fixture-extra-script|test-ci|1"
        "fixture-extra-script|cpp|0"
        "fixture-app-registration|test-ci|1"
        "fixture-app-registration|suite-app|1")
    string(REPLACE "|" ";" fields "${check}")
    list(GET fields 0 test_name)
    list(GET fields 1 label)
    list(GET fields 2 expected_count)
    assert_selected("${valid_dir}" "${test_name}" "${label}" "${expected_count}")
endforeach()
execute_process(COMMAND "${CTEST_EXECUTABLE}" --test-dir "${valid_dir}"
    --output-on-failure RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "valid fixture tests failed:\n${output}\n${error}")
endif()

configure_case(model-disabled TRUE "")
assert_selected("${OUTPUT_DIR}/model-disabled" fixture-direct aicore-model 1)
assert_selected("${OUTPUT_DIR}/model-disabled" fixture-direct aicore-contract 0)
configure_case(gtest TRUE "")
configure_case(nongtest FALSE "must link GoogleTest")
configure_case(duplicate FALSE "Duplicate test 'fixture-direct'")
configure_case(missing FALSE "references missing build target")
configure_case(bypass FALSE "bypasses")
configure_case(without-app-registration TRUE "")
assert_selected("${OUTPUT_DIR}/without-app-registration" fixture-app-registration test-ci 0)
configure_case(graphics-only TRUE "")
assert_selected("${OUTPUT_DIR}/graphics-only" fixture-graphics test-ci 0)
assert_selected("${OUTPUT_DIR}/graphics-only" fixture-graphics test-graphics 1)
configure_case(script-only TRUE "")
execute_process(COMMAND "${CMAKE_COMMAND}" --build "${OUTPUT_DIR}/script-only"
    --target check-cpp RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "script-only CI gate failed:\n${output}\n${error}")
endif()
configure_case(empty TRUE "")
execute_process(COMMAND "${CTEST_EXECUTABLE}" --test-dir "${OUTPUT_DIR}/empty"
    -L "^cpp-ci$" --no-tests=error
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(result EQUAL 0)
    message(FATAL_ERROR "empty C++ CI selection unexpectedly succeeded")
endif()

execute_process(COMMAND "${CMAKE_COMMAND}" -S "${SOURCE_DIR}/run_vtk_only"
    -B "${OUTPUT_DIR}/run-vtk-only"
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "VTK-only fixture configure failed:\n${output}\n${error}")
endif()
foreach(valid_jobs 1 12)
    execute_process(COMMAND "${CMAKE_COMMAND}"
        -DBUILD_DIR=${OUTPUT_DIR}/run-vtk-only -DJOBS=${valid_jobs}
        -P "${SOURCE_DIR}/../Run.cmake"
        RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "Run.cmake rejected VTK-only/JOBS=${valid_jobs}:\n${output}\n${error}")
    endif()
endforeach()
foreach(profile ci graphics model gpu all ci,gpu)
    execute_process(COMMAND "${CMAKE_COMMAND}" -DBUILD_DIR=${valid_dir}
        -DJOBS=1 -DPROFILE=${profile} -P "${SOURCE_DIR}/../Run.cmake"
        RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "profile ${profile} failed:\n${output}\n${error}")
    endif()
endforeach()
foreach(group aicore-fast aicore-contract aicore-model aicore-yolo-matrix)
    execute_process(COMMAND "${CMAKE_COMMAND}" -DBUILD_DIR=${valid_dir}
        -DJOBS=1 -DGROUP=${group} -P "${SOURCE_DIR}/../Run.cmake"
        RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "compatibility group ${group} failed:\n${output}\n${error}")
    endif()
endforeach()
# Public build targets must forward to the same policy as Run.cmake.
foreach(check check-cpp check-cpp-graphics check-cpp-model check-cpp-gpu check-cpp-all)
    execute_process(COMMAND "${CMAKE_COMMAND}" --build "${valid_dir}" --target ${check}
        RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "forwarding target ${check} failed:\n${output}\n${error}")
    endif()
endforeach()
execute_process(COMMAND "${CMAKE_COMMAND}" -DBUILD_DIR=${OUTPUT_DIR}/run-vtk-only
    -DPROFILE=unknown -P "${SOURCE_DIR}/../Run.cmake"
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(result EQUAL 0 OR NOT "${output}\n${error}" MATCHES "PROFILE must")
    message(FATAL_ERROR "Run.cmake accepted an unknown profile")
endif()
foreach(invalid_jobs 0 13 2.5)
    execute_process(COMMAND "${CMAKE_COMMAND}"
        -DBUILD_DIR=${OUTPUT_DIR}/run-vtk-only -DJOBS=${invalid_jobs}
        -P "${SOURCE_DIR}/../Run.cmake"
        RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
    if(result EQUAL 0 OR NOT "${output}\n${error}" MATCHES "JOBS must be an integer between 1 and 12")
        message(FATAL_ERROR "Run.cmake did not reject JOBS=${invalid_jobs}:\n${output}\n${error}")
    endif()
endforeach()
if(APPLE)
    get_filename_component(repository "${SOURCE_DIR}/../.." ABSOLUTE)
    set(loader_build "${OUTPUT_DIR}/macos-loader")
    execute_process(COMMAND "${CMAKE_COMMAND}"
        -S "${repository}/core/tests/runtime-environment" -B "${loader_build}"
        -DCMAKE_BUILD_TYPE=Release RESULT_VARIABLE result
        OUTPUT_VARIABLE output ERROR_VARIABLE error)
    if(result EQUAL 0)
        execute_process(COMMAND "${CMAKE_COMMAND}" --build "${loader_build}"
            --parallel 1 RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
    endif()
    if(result EQUAL 0)
        execute_process(COMMAND "${CTEST_EXECUTABLE}" --test-dir "${loader_build}"
            --output-on-failure RESULT_VARIABLE result
            OUTPUT_VARIABLE output ERROR_VARIABLE error)
    endif()
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "macOS loader contract failed:\n${output}\n${error}")
    endif()
endif()
message(STATUS "Test infrastructure contract fixtures passed")
