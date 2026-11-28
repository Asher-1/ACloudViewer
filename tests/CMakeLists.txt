# Refuse first-party registrations that bypass the common contract. Native
# third-party/standalone fixture projects retain their own add_test semantics.
function(cloudviewer_audit_test_registration directory)
    get_property(source DIRECTORY "${directory}" PROPERTY SOURCE_DIR)
    file(RELATIVE_PATH relative "${CMAKE_SOURCE_DIR}" "${source}")
    if(relative MATCHES "^(core|libs|app|plugins/core)(/|$)" AND
       NOT relative MATCHES "(^|/)(3rdparty|thirdparty|third_party|extern|SIBR|QDarkStyleSheet)(/|$)")
        get_property(tests DIRECTORY "${directory}" PROPERTY TESTS)
        foreach(name IN LISTS tests)
            string(SHA256 key "${name}")
            get_property(owner GLOBAL PROPERTY "CV_TEST_${key}_SOURCE")
            if(NOT owner)
                message(FATAL_ERROR "First-party test '${name}' in ${relative} bypasses cloudviewer_add_test")
            endif()
        endforeach()
    endif()
    get_property(children DIRECTORY "${directory}" PROPERTY SUBDIRECTORIES)
    foreach(child IN LISTS children)
        cloudviewer_audit_test_registration("${child}")
    endforeach()
endfunction()
cloudviewer_audit_test_registration("${CMAKE_SOURCE_DIR}")

# Follow actual CMake link dependencies, including interface wrappers and aliases.
# External process probes declare integration explicitly. Native unit-test
# executables must resolve GoogleTest; linkage does not prove assertion coverage.
function(cloudviewer_target_uses_gtest target output)
    set(queue "${target}")
    set(visited)
    while(queue)
        list(POP_FRONT queue current)
        if(current IN_LIST visited OR NOT TARGET "${current}")
            continue()
        endif()
        list(APPEND visited "${current}")
        if(current MATCHES "^(GTest::)?(gtest|gtest_main|gmock|gmock_main|GTest|Main)$")
            set(${output} TRUE PARENT_SCOPE)
            return()
        endif()
        foreach(property LINK_LIBRARIES INTERFACE_LINK_LIBRARIES)
            get_target_property(links "${current}" ${property})
            if(links AND NOT links MATCHES "NOTFOUND$")
                string(REGEX MATCHALL "[A-Za-z0-9_.+-]+(::[A-Za-z0-9_.+-]+)*" dependencies "${links}")
                list(APPEND queue ${dependencies})
            endif()
        endforeach()
    endwhile()
    set(${output} FALSE PARENT_SCOPE)
endfunction()

# All component registrations have completed before this directory is added.
# One source of truth drives both the build closure and CTest selection.
get_property(native_tests GLOBAL PROPERTY CV_CPP_TEST_NAMES)
get_property(registered_tests GLOBAL PROPERTY CV_TEST_NAMES)
set(catalog "name\towner\ttargets\tlabels\n")
set(all_targets)
foreach(name IN LISTS registered_tests)
    string(SHA256 key "${name}")
    get_property(owner GLOBAL PROPERTY "CV_TEST_${key}_OWNER")
    get_property(targets GLOBAL PROPERTY "CV_TEST_${key}_TARGETS")
    get_property(labels GLOBAL PROPERTY "CV_TEST_${key}_LABELS")
    foreach(target IN LISTS targets)
        if(NOT TARGET "${target}")
            message(FATAL_ERROR "Test '${name}' references missing build target '${target}'")
        endif()
        if(NOT target IN_LIST validated_native_targets)
            get_target_property(type "${target}" TYPE)
            get_target_property(kind "${target}" CLOUDVIEWER_TEST_KIND)
            if(kind AND NOT kind STREQUAL "integration")
                message(FATAL_ERROR "Unsupported CLOUDVIEWER_TEST_KIND '${kind}' on '${target}'")
            endif()
            if(type STREQUAL "EXECUTABLE" AND NOT kind STREQUAL "integration")
                cloudviewer_target_uses_gtest("${target}" uses_gtest)
                if(NOT uses_gtest)
                    message(FATAL_ERROR "Native unit-test target '${target}' must link GoogleTest; external process probes must declare CLOUDVIEWER_TEST_KIND=integration")
                endif()
            endif()
            list(APPEND validated_native_targets "${target}")
        endif()
    endforeach()
    string(APPEND catalog "${name}\t${owner}\t${targets}\t${labels}\n")
    list(APPEND all_targets ${targets})
    foreach(profile ci graphics model gpu)
        if("test-${profile}" IN_LIST labels)
            list(APPEND profile_${profile}_targets ${targets})
            list(APPEND profile_${profile}_tests "${name}")
        endif()
    endforeach()
endforeach()
file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/tests/catalog-$<CONFIG>.tsv" CONTENT "${catalog}")

# CMake's contract is not a C++ unit test. Keep its configure/build matrix
# independently runnable and invoke it from the unified root CI entry point.
if(NOT CLOUDVIEWER_TEST_CONTRACT_FIXTURE)
    cloudviewer_add_test(NAME test_system_contract
        COMMAND "${CMAKE_COMMAND}"
            -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/contracts
            -DOUTPUT_DIR=${CMAKE_BINARY_DIR}/tests/contracts-fixture
            -P ${CMAKE_CURRENT_SOURCE_DIR}/contracts/verify.cmake
        LABELS test-system)
    add_custom_target(check-test-system
        COMMAND ${CMAKE_CTEST_COMMAND} --test-dir "${CMAKE_BINARY_DIR}"
            -C "$<CONFIG>" -R "^test_system_contract$" --no-tests=error
            --output-on-failure
        USES_TERMINAL VERBATIM)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/AICoreTiers.cmake")

add_custom_target(cpp-tests)
if(all_targets)
    list(REMOVE_DUPLICATES all_targets)
    add_dependencies(cpp-tests ${all_targets})
endif()
foreach(profile ci graphics model gpu)
    if(NOT profile_${profile}_tests)
        continue()
    endif()
    add_custom_target(cpp-tests-${profile})
    if(profile_${profile}_targets)
        list(REMOVE_DUPLICATES profile_${profile}_targets)
        add_dependencies(cpp-tests-${profile} ${profile_${profile}_targets})
    endif()
    if(profile STREQUAL "ci")
        set(check_target check-cpp)
    else()
        set(check_target check-cpp-${profile})
    endif()
    add_custom_target(${check_target}
        COMMAND "${CMAKE_COMMAND}" -DBUILD_DIR=${CMAKE_BINARY_DIR}
            -DCONFIG=$<CONFIG> -DPROFILE=${profile}
            -P "${CMAKE_CURRENT_LIST_DIR}/Run.cmake"
        USES_TERMINAL VERBATIM)
endforeach()
if(registered_tests)
    add_custom_target(check-cpp-all
        COMMAND "${CMAKE_COMMAND}" -DBUILD_DIR=${CMAKE_BINARY_DIR}
            -DCONFIG=$<CONFIG> -DPROFILE=all
            -P "${CMAKE_CURRENT_LIST_DIR}/Run.cmake"
        USES_TERMINAL VERBATIM)
endif()
list(LENGTH native_tests count)
message(STATUS "Unified C++ test catalog: ${count} registered native test commands")
